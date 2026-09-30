//! Qwen3.8 on the GPU: forty-eight gated delta-net blocks and sixteen attention blocks.
//!
//! ```text
//! every block:   prev = X
//!                Cur  = rms_norm(X, attn_norm)
//!                O    = gated_delta(Cur)   OR   gated_attention(Cur)   <- 48 / 16
//!                X    = prev + O
//!                Cur  = rms_norm(X, post_attention_norm)
//!                X    = X + ffn_down(silu(ffn_gate @ Cur) * (ffn_up @ Cur))
//! tail:          logits = output @ rms_norm(X_last, output_norm)
//! ```
//!
//! `workflow_cpu` is the oracle, and the two paths now run the SAME two recurrent ops in
//! the same order -- `causal_conv` over the whole batch, then `delta_net` over the whole
//! batch. That is deliberate: an oracle whose shape differs from the path it checks is a
//! second design, and a disagreement between them says nothing about which is wrong.
//!
//! THE TWO GATES. An attention block ends `cur * sigmoid(gate)` where the gate is the
//! second half of each packed `[query | gate]` head; a delta block ends
//! `rms_norm(core, ssm_norm) * silu(z)`. Two gates, two activations, one architecture --
//! and `set_activation` bakes ONE of them into every epilogue for the whole process, which
//! is why the attention gate is its own op (`mul_strided_sigmoid`) and not an epilogue.
//!
//! WHAT IS NOT HERE YET: no mega-kernel entry (task #165), so decode dispatches per
//! operator; and the delta rule is the scalar recurrence, which is the chunked scan's
//! reference (task #164) rather than its competitor.

use imparo_backend::{BufId, ConvForm};

use crate::ModelPlan;
use crate::gpu_support::{
    BufferRequirement, Placement, be, fuse_epilogue_enabled, gprobe,
    gpu_probe_last_row, gpu_probe_layer, half_activation_mirror_requirements,
    kv_dequant_scratch_requirements, kvq_mask_on, layer_skip_log, scores_needed,
    should_fuse_epilogue, tail_align, tail_split, trace_rows,
};
use crate::kv::{KvType, effective_workflow_kv_route, had_nrot, ring_mask};
use crate::qwen35::Qwen35;
use crate::qwen35::workflow_cpu::{MixerW, delta_shape};

/// QWEN3.8'S PRIVATE BUFFER SLOTS.
///
/// `MIX` is the mixer's wide projection, and the two block kinds put different things in
/// it -- an attention block's packed `[query | gate]` heads (2 * n_head * head_dim), a
/// delta block's packed `[Q | K | V]` (the conv's channel count). One buffer, sized for
/// the wider, because the two never run in the same layer.
pub const MIX: BufId = BufId::Model0;
/// The convolved `[Q | K | V]`, which the delta rule reads while the conv's own input is
/// still live in `MIX` -- so it is a second buffer, not a rewrite of the first.
pub const CONV: BufId = BufId::Model1;
/// The delta block's `z` gate, and after the gated norm the mixer output the out
/// projection reads.
pub const Z: BufId = BufId::Model2;
/// The raw alpha and beta projections, one value per value head per token. The delta op
/// applies `softplus`/`dt_bias` and `sigmoid` itself, so a caller cannot get the order
/// wrong.
pub const ALPHA: BufId = BufId::Model3;
pub const BETA: BufId = BufId::Model4;

/// Admits the device-owned quantized weight layouts this backend offers, for the same
/// spans LFM2 offers: the feed-forward projections and the output head.
///
/// # Errors
/// When the backend refuses the admission.
pub fn prepare_device(wf: &mut Qwen35) -> Result<(), String> {
    if !be().quantized_weight_cache_enabled() {
        return Ok(());
    }
    let n_in = wf.plan.config.n_ff;
    let n_out = wf.plan.config.n_embd;
    let cache = be().quantized_weight_cache_plan();
    let mut spans = Vec::new();
    let mut push = |tensor: &imparo_gguf::weights::Tensor, width: u32, rows: u32| {
        let kind = imparo_gguf::weights::weight_kind(tensor.ggml_type)
            .expect("validated at load") as u32;
        spans.push(imparo_backend::QuantizedWeightPrepack {
            offset: tensor.offset as u64,
            n_in: width,
            n_out: rows,
            kind,
            reserved: 0,
        });
    };
    if cache.include_down {
        for layer in &wf.w.layers {
            if cache.include_full_ffn {
                push(&layer.ffn_gate, n_out, n_in);
                push(&layer.ffn_up, n_out, n_in);
            }
            push(&layer.ffn_down, n_in, n_out);
        }
    }
    if cache.include_head {
        // NOT tied: `output.weight` is its own tensor here, unlike LFM2's head.
        push(&wf.w.output, n_out, wf.plan.config.vocab_size);
    }
    be().prepare_quantized_weight_cache(&spans)
        .map(|_| ())
        .map_err(|rc| format!("backend quantized-weight admission failed rc={rc}"))
}

/// Every activation buffer this model needs for a batch of `b`.
///
/// The mixer group and the feed-forward group alias, because a block runs one and then the
/// other. Within the mixer group nothing aliases: `MIX` is live while `CONV` is being
/// written (the conv reads its input and the state advance reads it again), and the
/// attention gate is read out of `MIX` after attention has run.
///
/// The recurrent state is NOT here. It is per conversation and constant in context, so it
/// is allocated once like KV rather than resized per batch.
///
/// A FREE FUNCTION OVER THE PLAN, so it is testable without a 15 GB file on disk.
#[must_use]
pub fn buffer_requirements(
    plan: &ModelPlan,
    b: usize,
    capacity: usize,
) -> Vec<BufferRequirement> {
    let c = &plan.config;
    let n_embd = c.n_embd as usize;
    let n_ff = c.n_ff as usize;
    let head_max = plan
        .layers
        .iter()
        .map(|l| l.attention.head_dim() as usize)
        .max()
        .unwrap_or(0);
    let q_width = c.n_heads as usize * head_max;
    // The delta block's widths, from the plan alone: `s_elems = v_heads * value * key`
    // and `r_elems = (taps - 1) * qkv_width`. The conv's channel count is not stored, so
    // it is the largest r_elems over the layers divided by (taps - 1) -- which the plan
    // does not carry either, so instead take the widest projection the mixer can hold:
    // the packed `[Q | K | V]` is what the conv runs over and `r_elems` is a multiple of
    // it, so `r_elems` itself is an upper bound and one that is never far off (taps 4
    // gives 3x). Sizing MIX and CONV by that bound costs a few MB and cannot be short.
    let recur_r = plan
        .layers
        .iter()
        .map(|l| match l.attention {
            crate::Attention::Recurrent { r_elems, .. } => r_elems as usize,
            _ => 0,
        })
        .max()
        .unwrap_or(0);
    let v_width = plan
        .layers
        .iter()
        .map(|l| match l.attention {
            crate::Attention::Recurrent {
                s_elems,
                key_dim,
                value_dim,
                ..
            } if key_dim > 0 && value_dim > 0 => (s_elems / key_dim) as usize,
            _ => 0,
        })
        .max()
        .unwrap_or(0);
    let v_heads = if v_width > 0 {
        plan.layers
            .iter()
            .map(|l| match l.attention {
                crate::Attention::Recurrent { value_dim, .. } if value_dim > 0 => {
                    v_width / value_dim as usize
                }
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    } else {
        0
    };
    let mix_width = (2 * q_width).max(recur_r);
    let f = |n: usize| (n * 4) as u64;
    let need = |id: BufId, bytes: u64, placement: Placement| BufferRequirement {
        id,
        bytes,
        placement,
    };
    // Must match ATTN_MAX_SLICES in the Metal source.
    const MAX_SPLITS: usize = 256;
    let mut requirements = vec![
        need(BufId::X, f(b * n_embd), Placement::Dedicated),
        need(BufId::Cur, f(b * n_embd), Placement::Dedicated),
        need(BufId::O, f(b * n_embd), Placement::Dedicated),
        need(BufId::Q, f(b * q_width), Placement::Group(0)),
        need(
            BufId::K,
            f(b * c.n_kv_heads as usize * head_max),
            Placement::Group(0),
        ),
        need(
            BufId::V,
            f(b * c.n_kv_heads as usize * head_max),
            Placement::Group(0),
        ),
        // ATTN does not share, for the reason the other models record at their own lists:
        // the register-tiled GEMM reads whole 8-row tiles and cannot mask, so it reads
        // past the token count, and what those rows hold changes the logits. It is also
        // the delta rule's output, which is exactly as wide (v_heads * value_dim).
        need(
            BufId::Attn,
            f(b * q_width.max(v_width)),
            Placement::Dedicated,
        ),
        need(MIX, f(b * mix_width), Placement::Group(0)),
        need(CONV, f(b * recur_r.max(1)), Placement::Group(0)),
        need(Z, f(b * v_width.max(1)), Placement::Group(0)),
        need(ALPHA, f(b * v_heads.max(1)), Placement::Group(0)),
        need(BETA, f(b * v_heads.max(1)), Placement::Group(0)),
        need(BufId::G, f(b * n_ff), Placement::Group(1)),
        need(BufId::U, f(b * n_ff), Placement::Group(1)),
        need(
            BufId::Logits,
            f(c.vocab_size as usize),
            Placement::Dedicated,
        ),
        need(BufId::Tmp, f(b * n_embd), Placement::Dedicated),
        need(BufId::Tokens, (b * 4) as u64, Placement::Dedicated),
        need(BufId::Pick, 8, Placement::Dedicated),
        need(
            BufId::AttnPart,
            f(c.n_heads as usize * MAX_SPLITS * (head_max + 2)),
            Placement::Dedicated,
        ),
    ];
    let diagnostic = kvq_mask_on();
    requirements.extend(kv_dequant_scratch_requirements(
        plan,
        capacity,
        KvType::k() != KvType::F16 || diagnostic,
        KvType::v() != KvType::F16 || diagnostic,
    ));
    // The half-precision activation mirrors, on the same rule LFM2 states: they alias U,
    // which the fused feed-forward epilogue leaves untouched, and declaring them is what
    // enables the whole family.
    if fuse_epilogue_enabled() {
        requirements.extend(half_activation_mirror_requirements(b, n_ff, BufId::U));
    }
    requirements
}

/// Qwen3.8's layer graph on the device.
///
/// `argmax` true means: do not copy the logits back, write the greedy pick's index into
/// `out[0]` as raw bits.
///
/// # Errors
/// When a dispatch fails, when the plan and the resolved weights disagree about a block
/// kind, or when this backend does not serve the gated delta-net mixer.
#[allow(clippy::too_many_lines)]
pub fn batch(
    wf: &mut Qwen35,
    tokens: &[u32],
    start_pos: usize,
    out: &mut Vec<f32>,
    argmax: bool,
) -> Result<(), String> {
    // ONE QUESTION, ASKED ONCE. Three of this architecture's ops have no default body
    // that writes anything, and a backend without them would leave the previous layer's
    // values in the buffers -- a plausible answer, and a wrong one.
    if !be().supports_gated_delta() {
        return Err(format!(
            "qwen35: the {} backend does not serve the gated delta-net mixer",
            be().device_tag()
        ));
    }
    let c = wf.plan.config.clone();
    let kv_quant_route = effective_workflow_kv_route(
        &wf.plan,
        KvType::k(),
        KvType::v(),
        be().kv_quantization_route(),
        be().kv_quantization_route_override(),
    )?;
    let n_embd = c.n_embd;
    let n_head = c.n_heads;
    let n_kv = c.n_kv_heads;
    let n_ff = c.n_ff;
    let eps = c.norm_eps;
    let mut b = u32::try_from(tokens.len()).map_err(|_| "batch too large")?;
    let mut sp = u32::try_from(start_pos).map_err(|_| "start_pos too large")?;
    let recur = wf.plan.recurrent_layout();
    let act = wf.plan.layers[0].ffn.activation().epilogue();
    wf.gpu_fit_batch(tokens.len())?;
    let decode = b == 1;
    let replayed = if decode {
        be().decode_prepare(tokens[0], sp, argmax)
            .map_err(|rc| format!("qwen35 GPU decode prepare failed rc={rc}"))?
    } else {
        false
    };
    if replayed {
        be().end()
            .map_err(|rc| format!("qwen35 GPU forward failed rc={rc}"))?;
        if argmax {
            out.resize(1, 0.0);
            be().read(BufId::Tmp, 0, out);
        } else {
            out.resize(c.vocab_size as usize, 0.0);
            be().read(BufId::Logits, 0, out);
        }
        return Ok(());
    }

    let wkind = |t: &imparo_gguf::weights::Tensor| {
        imparo_gguf::weights::weight_kind(t.ggml_type).expect("validated at load")
            as u32
    };

    be().begin_forward(decode);
    let pipe = wf.state.pipe;
    // ---- embeddings. No scale: qwen35 sets scale_by_sqrt_embd false. ----------------
    if !pipe.is_some_and(|p| p.token_on_device) {
        be().write_u32(BufId::Tokens, 0, tokens);
    }
    // The head is its OWN tensor here, so the embedding table and the lm head are two
    // different weights -- unlike LFM2, where one tensor serves both.
    let embd_kind = wkind(&wf.w.token_embd);
    let embd_off = wf.w.token_embd.offset as u64;
    let gathered = be().gather_rows(
        embd_kind,
        embd_off,
        n_embd,
        c.vocab_size,
        1.0,
        BufId::X,
        0,
        BufId::Tokens,
        b,
    );
    if !gathered {
        for (t, &tok) in tokens.iter().enumerate() {
            be().row(
                embd_kind,
                embd_off,
                n_embd,
                tok,
                1.0,
                BufId::X,
                t as u32 * n_embd,
            );
        }
    }
    // The probed row: the chunk's first, or its last under IMPARO_GPU_PROBE_ROW=last.
    let mut prow: u64 = if gpu_probe_last_row() {
        u64::from(b - 1)
    } else {
        0
    };
    gprobe(
        "inp_embd",
        BufId::X,
        prow * u64::from(n_embd),
        n_embd as usize,
    );
    // THE RECURRENT PLANES for this batch (task #165). A decode step reads one plane and
    // writes the next, so the plane it read stays intact as its rollback point and a failed
    // step is undone by an index, not by a 149.6 MiB copy before every step. Prefill is
    // IN PLACE: nothing rolls a chunk back to a per-token boundary.
    let recur_elems = wf.plan.recurrent_elems();
    let (plane_in, plane_out) = if b == 1 {
        let (i, o) = wf.state.recur_decode_planes();
        (i * recur_elems, o * recur_elems)
    } else {
        let p = wf.state.recur_plane * recur_elems;
        (p, p)
    };

    let n_layers_total = wf.plan.layers.len();
    // THE CUT. Every layer of this architecture writes state (the full-attention layers
    // their K/V, the others -- the last one included -- their recurrent state), so the
    // last state-writing operator is the last layer's delta rule, and nothing after it
    // feeds a cache: a chunk whose logits are not wanted stops there, and on the final
    // chunk only the last row's logits are consumed, so that layer's remaining work
    // (its attention when it is a full-attention layer, the out projection, the
    // residual and the FFN) runs on the chunk's last 64-or-more rows. The tail starts
    // on the 16-row query-tile grid, the measured constraint of the attention path
    // (rows 66 and 385 were nondeterministic, 32/64/352/400/448 pin EXACT), and keeps
    // 64 rows or more so the GEMMs take the wide route. Worth about one layer of a
    // chunk's GEMM work; the design and its measurement are in
    // docs/prefill-ends-at-the-last-state-writing-layer.md. Not taken under 128 rows,
    // so the move never overlaps itself.
    let tail = if wf.state.logits_wanted && b >= 128 {
        tail_split(b, tail_align())
    } else {
        None
    };
    let flush_every = crate::gpu_support::flush_layers_bounded(b, n_layers_total);
    // Set by the PREVIOUS layer's residual when it also produced this layer's operator
    // norm (F2's cross-layer half, below). Cleared every iteration, so a layer that took
    // the mega tail's early `continue` leaves the next one doing its own norm.
    let mut attn_norm_fused = false;
    for (li, (&layer, &(r_off, s_off, _, _))) in
        wf.plan.layers.clone().iter().zip(recur.iter()).enumerate()
    {
        let lw = &wf.w.layers[li];
        let last_layer = li + 1 == n_layers_total;
        if !attn_norm_fused {
            be().rms_norm_from(
                BufId::Cur,
                BufId::X,
                lw.attn_norm.offset,
                n_embd,
                eps,
                b,
                n_embd,
                0,
            );
        }
        attn_norm_fused = false;
        if li == gpu_probe_layer() {
            gprobe(
                "attn_norm",
                BufId::Cur,
                prow * u64::from(n_embd),
                n_embd as usize,
            );
        }

        match (&lw.mixer, layer.attention) {
            (
                MixerW::Attention {
                    q_norm,
                    k_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                },
                crate::Attention::Full {
                    head_dim: hd,
                    rope_base,
                    rope_dim,
                },
            ) => {
                let (qw, kw) = (n_head * hd, n_kv * hd);
                // The cache basis this block must reproduce (task #156): the Hadamard
                // widths the quantized modes rotate Q/K and V by (0 = plain).
                let qk_hadamard_nrot = if KvType::k() == KvType::F16 {
                    0
                } else {
                    had_nrot("IMPARO_HAD_K", kv_quant_route.key, hd)?
                };
                let v_hadamard_nrot = if KvType::v() == KvType::F16 {
                    0
                } else {
                    had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?
                };
                // ONE projection, 24 heads of [query(256) | gate(256)]. Reading it as two
                // 6144-wide halves puts every head after the first on the wrong side --
                // a plausible model that is wrong everywhere -- so the query is lifted out
                // per head and the gate is read in place at the end of the block.
                be().matmat(
                    wkind(wq),
                    wq.offset as u64,
                    n_embd,
                    2 * qw,
                    BufId::Cur,
                    MIX,
                    b,
                );
                be().copy_strided(BufId::Q, MIX, hd, 0, 2 * hd, b * n_head);
                be().matmat(
                    wkind(wk),
                    wk.offset as u64,
                    n_embd,
                    kw,
                    BufId::Cur,
                    BufId::K,
                    b,
                );
                be().matmat(
                    wkind(wv),
                    wv.offset as u64,
                    n_embd,
                    kw,
                    BufId::Cur,
                    BufId::V,
                    b,
                );
                be().head_norm_rope_hadamard(
                    BufId::Q,
                    q_norm.offset,
                    hd,
                    eps,
                    n_head,
                    sp,
                    b,
                    rope_dim,
                    rope_base,
                    None,
                    qk_hadamard_nrot,
                );
                be().head_norm_rope_hadamard(
                    BufId::K,
                    k_norm.offset,
                    hd,
                    eps,
                    n_kv,
                    sp,
                    b,
                    rope_dim,
                    rope_base,
                    None,
                    qk_hadamard_nrot,
                );
                if v_hadamard_nrot != 0 {
                    be().hadamard(BufId::V, b * kw, v_hadamard_nrot);
                }
                if li == gpu_probe_layer() {
                    gprobe("Qcur_pos", BufId::Q, prow * u64::from(qw), qw as usize);
                }
                let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
                be().kv_store(BufId::K, li as u32, kw, sp, b, false, ring);
                be().kv_store(BufId::V, li as u32, kw, sp, b, true, ring);
                if last_layer {
                    // The chunk's last state write is behind us: the attention below and
                    // everything after it feed nothing but this chunk's own logits.
                    if !wf.state.logits_wanted {
                        if layer_skip_log() {
                            eprintln!(
                                "[imparo] chunk at sp={sp}: cut after layer {li}'s K/V store"
                            );
                        }
                        break;
                    }
                    if let Some((r0, bt)) = tail {
                        // Only the last row's logits are consumed: narrow to the tail
                        // rows before this layer's attention. Live per-row inputs here:
                        // the residual, the roped Q, and the packed projection (the
                        // output gate reads it); K/V of every row are already stored.
                        be().copy_range(
                            BufId::X,
                            0,
                            BufId::X,
                            r0 * n_embd,
                            bt * n_embd,
                        );
                        be().copy_range(BufId::Q, 0, BufId::Q, r0 * qw, bt * qw);
                        be().copy_range(MIX, 0, MIX, r0 * 2 * qw, bt * 2 * qw);
                        sp += r0;
                        b = bt;
                        if gpu_probe_last_row() {
                            prow = u64::from(b - 1);
                        }
                        if layer_skip_log() {
                            eprintln!(
                                "[imparo]   tail inside layer {li}: rows {r0}.. as a {b}-row batch at sp={sp}"
                            );
                        }
                    }
                }
                be().attention(
                    li as u32,
                    hd,
                    n_head,
                    n_kv,
                    kw,
                    sp,
                    1.0 / (hd as f32).sqrt(),
                    // No sliding window: a file that set one would plan Attention::Window
                    // and this arm would not match it.
                    0,
                    b,
                    scores_needed(sp, b, 0),
                    ring,
                );
                if v_hadamard_nrot != 0 {
                    be().hadamard(BufId::Attn, b * qw, v_hadamard_nrot);
                }
                // THE OUTPUT GATE, before o_proj: `cur = cur * sigmoid(gate)`, the gate
                // being the second half of each packed head. Plain sigmoid, NOT the SiLU
                // the delta block's z gate uses.
                be().mul_strided_sigmoid(
                    BufId::Attn,
                    MIX,
                    hd,
                    hd,
                    2 * hd,
                    hd,
                    b * n_head,
                );
                if li == gpu_probe_layer() {
                    gprobe(
                        "attention_gated",
                        BufId::Attn,
                        prow * u64::from(qw),
                        qw as usize,
                    );
                }
                be().matmat(
                    wkind(wo),
                    wo.offset as u64,
                    qw,
                    n_embd,
                    BufId::Attn,
                    BufId::O,
                    b,
                );
            }
            (
                MixerW::GatedDelta {
                    qkv,
                    gate: gate_w,
                    conv,
                    a,
                    dt_bias,
                    alpha,
                    beta,
                    ssm_norm,
                    out: out_w,
                },
                crate::Attention::Recurrent { .. },
            ) => {
                let (sh, taps) = delta_shape(layer.attention, conv.w.len())?;
                let qkv_width =
                    u32::try_from(sh.qkv_width()).map_err(|_| "qkv width")?;
                let v_width = u32::try_from(sh.v_heads * sh.value_dim)
                    .map_err(|_| "value width")?;
                let (kd, vd) = (sh.key_dim as u32, sh.value_dim as u32);
                let vh = u32::try_from(sh.v_heads).map_err(|_| "value heads")?;
                let taps = u32::try_from(taps).map_err(|_| "conv taps")?;

                be().matmat(
                    wkind(qkv),
                    qkv.offset as u64,
                    n_embd,
                    qkv_width,
                    BufId::Cur,
                    MIX,
                    b,
                );
                be().matmat(
                    wkind(gate_w),
                    gate_w.offset as u64,
                    n_embd,
                    v_width,
                    BufId::Cur,
                    Z,
                    b,
                );
                // MEASUREMENT LEVER, not a route. These two are n_out = v_heads = 48 on a
                // 64-row tile: one row block, and each reads the whole activation tile to
                // produce 48 columns. IMPARO_SKIP_AB=1 drops both so two unprofiled runs
                // can be differenced; the logits are wrong with it set, deliberately.
                if !skip_alpha_beta() {
                    be().matmat(
                        wkind(alpha),
                        alpha.offset as u64,
                        n_embd,
                        vh,
                        BufId::Cur,
                        ALPHA,
                        b,
                    );
                    be().matmat(
                        wkind(beta),
                        beta.offset as u64,
                        n_embd,
                        vh,
                        BufId::Cur,
                        BETA,
                        b,
                    );
                }
                if li == gpu_probe_layer() {
                    gprobe(
                        "qkv_proj",
                        MIX,
                        prow * u64::from(qkv_width),
                        qkv_width.min(n_embd) as usize,
                    );
                    gprobe("alpha", ALPHA, prow * u64::from(vh), vh as usize);
                    gprobe("beta", BETA, prow * u64::from(vh), vh as usize);
                }
                // A checkpoint boundary inside this batch: write the conv state as of that
                // position aside BEFORE the advance overwrites the pre-batch history. The
                // delta rule below writes its matrix aside the same way, in passing.
                if let Some(k) = wf.state.recur_snap {
                    be().causal_conv_snapshot(
                        ConvForm::PlainSilu,
                        MIX,
                        BufId::Recur,
                        r_off + plane_in,
                        BufId::RecurSnap,
                        r_off,
                        qkv_width,
                        taps,
                        k,
                    );
                }
                be().causal_conv(
                    ConvForm::PlainSilu,
                    MIX,
                    conv.offset,
                    BufId::Recur,
                    r_off + plane_in,
                    r_off + plane_out,
                    CONV,
                    qkv_width,
                    taps,
                    b,
                );
                if li == gpu_probe_layer() {
                    gprobe(
                        "conv_out",
                        CONV,
                        prow * u64::from(qkv_width),
                        qkv_width.min(n_embd) as usize,
                    );
                }
                // F1: the gated-RMS epilogue rides inside the rule when the backend
                // reproduces its bits (see Backend::delta_net_fuses_epilogue). The probe
                // layer opts out because `gprobe("delta_core")` reads Attn, which the
                // fused form never writes -- a probe reading a buffer nobody wrote is a
                // plausible number and a wrong one.
                let fuse_epi = (li != gpu_probe_layer()
                    || std::env::var_os("IMPARO_DELTA_EPI_PROBE").is_some())
                    && be().delta_net_fuses_epilogue();
                if !be().delta_net(&imparo_backend::DeltaNet {
                    qkv: CONV,
                    alpha: ALPHA,
                    beta: BETA,
                    a_off: a.offset,
                    dt_bias_off: dt_bias.offset,
                    state: BufId::Recur,
                    state_off: s_off + plane_in,
                    state_out_off: s_off + plane_out,
                    out: BufId::Attn,
                    epilogue: fuse_epi.then_some(imparo_backend::DeltaEpilogue {
                        norm_w_off: ssm_norm.offset,
                        gate: Z,
                    }),
                    // The matrix as of the boundary goes to the snapshot plane's copy of
                    // this layer's region, beside the conv history written above.
                    snap: wf.state.recur_snap.map(|k| imparo_backend::DeltaSnapshot {
                        buf: BufId::RecurSnap,
                        off: s_off,
                        row: k,
                    }),
                    k_heads: u32::try_from(sh.k_heads).map_err(|_| "key heads")?,
                    v_heads: vh,
                    key_dim: kd,
                    value_dim: vd,
                    n_tok: b,
                    eps,
                }) {
                    return Err(format!(
                        "qwen35 layer {li}: the backend refused a {vd}x{kd} delta rule"
                    ));
                }
                if li == gpu_probe_layer() {
                    gprobe(
                        "delta_core",
                        BufId::Attn,
                        prow * u64::from(v_width),
                        v_width.min(n_embd) as usize,
                    );
                }
                if last_layer && !wf.state.logits_wanted {
                    // The chunk's last state write is behind us: the gated norm, the out
                    // projection, the residual and the FFN below would feed nothing.
                    if layer_skip_log() {
                        eprintln!(
                            "[imparo] chunk at sp={sp}: cut after layer {li}'s delta rule"
                        );
                    }
                    break;
                }
                // Gated RMS norm, per VALUE head: the norm weight is one head wide and
                // shared by all of them, which is exactly `rms_norm`'s row geometry.
                // `act_mul` then leaves silu(z) * core in Z, which the out projection
                // reads -- the model's activation is SiLU, so the epilogue is the gate.
                if !fuse_epi {
                    be().rms_norm(BufId::Attn, ssm_norm.offset, vd, eps, b * vh, vd, 0);
                    be().act_mul(Z, BufId::Attn, b * v_width);
                }
                if li == gpu_probe_layer() {
                    gprobe(
                        "delta_gated",
                        Z,
                        prow * u64::from(v_width),
                        v_width.min(n_embd) as usize,
                    );
                }
                if let Some((r0, bt)) = tail.filter(|_| last_layer) {
                    // Only the last row's logits are consumed: narrow to the tail rows.
                    // Live per-row inputs here are the residual and the gated core the
                    // out projection reads (both epilogue forms leave it in Z).
                    be().copy_range(BufId::X, 0, BufId::X, r0 * n_embd, bt * n_embd);
                    be().copy_range(Z, 0, Z, r0 * v_width, bt * v_width);
                    sp += r0;
                    b = bt;
                    if gpu_probe_last_row() {
                        prow = u64::from(b - 1);
                    }
                    if layer_skip_log() {
                        eprintln!(
                            "[imparo]   tail inside layer {li}: rows {r0}.. as a {b}-row batch at sp={sp}"
                        );
                    }
                }
                be().matmat(
                    wkind(out_w),
                    out_w.offset as u64,
                    v_width,
                    n_embd,
                    Z,
                    BufId::O,
                    b,
                );
            }
            (m, at) => {
                return Err(format!(
                    "qwen35 layer {li}: plan says {at:?} but the weights resolved as {}",
                    match m {
                        MixerW::Attention { .. } => "full attention",
                        MixerW::GatedDelta { .. } => "gated delta-net",
                    }
                ));
            }
        }
        // ---- the residual and the SwiGLU tail, identical on both block kinds --------
        // THE MEGA ENTRY (task #165): the residual, the FFN norm, the gated pair, the down
        // projection and the second residual as ONE persistent dispatch. `Mixer::None` is
        // the tail alone -- the mixer above ran on the dispatch path and left its output in
        // O. Refused (false, nothing written) on any backend that does not serve it, on the
        // probe layer, and at any batch but one token.
        let tail_in_block = b == 1
            && li != gpu_probe_layer()
            && be().mega_layer(&imparo_backend::MegaEntry {
                layer: imparo_backend::MegaLayer::Qwen35(
                    imparo_backend::Qwen35MegaLayer {
                        mixer: imparo_backend::Qwen35MegaMixer::None,
                        gate_kind: wkind(&lw.ffn_gate),
                        gate_off: lw.ffn_gate.offset as u64,
                        up_kind: wkind(&lw.ffn_up),
                        up_off: lw.ffn_up.offset as u64,
                        down_kind: wkind(&lw.ffn_down),
                        down_off: lw.ffn_down.offset as u64,
                        ffn_norm_off: lw.ffn_norm.offset,
                        n_embd,
                        n_ff,
                        eps,
                        x: BufId::X,
                        add: BufId::O,
                        g: BufId::G,
                        u: BufId::U,
                        // Free here: the mixer's projections consumed it above and the next
                        // layer writes it fresh.
                        cur: BufId::Cur,
                    },
                ),
                n_tok: b,
            });
        if tail_in_block {
            if flush_every > 0 && (li + 1) % flush_every == 0 && li + 1 < n_layers_total
            {
                be().flush();
            }
            continue;
        }
        // F2, half of it: the mixer's residual add and the FFN's input norm are one
        // dispatch. `add_rms_norm` is the general facility LFM2 already ships, and its
        // contract is that the bits match the two-step form; returning false promises
        // nothing was written, so the fallback below is the old pair unchanged. The probe
        // layer keeps the pair because gprobe reads X between the two.
        //
        // AT PREFILL ONLY, and for the reason the activation epilogue below gives: the
        // fused kernel is one threadgroup PER ROW, so at one token it does the residual
        // add of 5120 floats in a single threadgroup where the standalone `add` had the
        // whole grid. Measured, rotated NEW/OLD/OLD/NEW, Qwen3.8-27B UD-Q4_K_S:
        //   prefill 512 tokens   4978.7 / 4961.7 fused   against 4992.1 / 4987.8   +0.40%
        //   decode  1 token       129.59 /  127.47 fused  against  129.19 /  126.31  -0.6%
        // Both rounds agree on both signs, so it is routed by row count and not averaged.
        let mixer_ffn_norm_fused = b > 1
            && li != gpu_probe_layer()
            && be().add_rms_norm(
                BufId::Cur,
                BufId::X,
                BufId::O,
                lw.ffn_norm.offset,
                n_embd,
                eps,
                b,
                n_embd,
                0,
            );
        if !mixer_ffn_norm_fused {
            be().add(BufId::X, BufId::O, b * n_embd);
        }
        if li == gpu_probe_layer() {
            gprobe(
                "mixer_out",
                BufId::X,
                prow * u64::from(n_embd),
                n_embd as usize,
            );
        }

        if !mixer_ffn_norm_fused {
            be().rms_norm_from(
                BufId::Cur,
                BufId::X,
                lw.ffn_norm.offset,
                n_embd,
                eps,
                b,
                n_embd,
                0,
            );
        }
        let fused_ffn = li != gpu_probe_layer()
            && be().ffn_gated_down(
                wkind(&lw.ffn_gate),
                lw.ffn_gate.offset as u64,
                wkind(&lw.ffn_up),
                lw.ffn_up.offset as u64,
                wkind(&lw.ffn_down),
                lw.ffn_down.offset as u64,
                n_embd,
                n_ff,
                n_embd,
                BufId::Cur,
                BufId::G,
                BufId::O,
                b,
            );
        if !fused_ffn {
            let fused_pair = be().matmat_gated(
                wkind(&lw.ffn_gate),
                lw.ffn_gate.offset as u64,
                wkind(&lw.ffn_up),
                lw.ffn_up.offset as u64,
                n_embd,
                n_ff,
                BufId::Cur,
                BufId::G,
                BufId::U,
                b,
            );
            if !fused_pair {
                be().matmat(
                    wkind(&lw.ffn_gate),
                    lw.ffn_gate.offset as u64,
                    n_embd,
                    n_ff,
                    BufId::Cur,
                    BufId::G,
                    b,
                );
                // Fuse at prefill only: at one token the epilogue is a read-modify-write
                // per output row inside the GEMV, where the standalone pass is wide and
                // vectorised.
                let fused = should_fuse_epilogue(b, be().supports_epilogue(act));
                if fused {
                    be().set_epilogue(act);
                }
                be().matmat(
                    wkind(&lw.ffn_up),
                    lw.ffn_up.offset as u64,
                    n_embd,
                    n_ff,
                    BufId::Cur,
                    if fused { BufId::G } else { BufId::U },
                    b,
                );
                if fused {
                    be().set_epilogue(imparo_backend::Epilogue::None);
                } else {
                    be().act_mul(BufId::G, BufId::U, b * n_ff);
                }
            }
            be().matmat(
                wkind(&lw.ffn_down),
                lw.ffn_down.offset as u64,
                n_ff,
                n_embd,
                BufId::G,
                BufId::O,
                b,
            );
        }
        // F2's cross-layer half: this layer's FFN residual also produces the NEXT layer's
        // operator norm, so the pair at the layer boundary is one dispatch too. `Cur` is
        // free here for the same reason it is free above -- the projections consumed it
        // and nothing between the two layers touches it, a flush included.
        //
        // Prefill only, same rule and same reason as the in-layer half: one threadgroup
        // per row is the wrong grid for a 5120-float add at one token.
        let next_attn_fused = b > 1
            && li + 1 < n_layers_total
            && li != gpu_probe_layer()
            && be().add_rms_norm(
                BufId::Cur,
                BufId::X,
                BufId::O,
                wf.w.layers[li + 1].attn_norm.offset,
                n_embd,
                eps,
                b,
                n_embd,
                0,
            );
        if !next_attn_fused {
            be().add(BufId::X, BufId::O, b * n_embd);
        }
        attn_norm_fused = next_attn_fused;
        if li == gpu_probe_layer() {
            gprobe("l_out", BufId::X, prow * u64::from(n_embd), n_embd as usize);
            gprobe(
                "l_out_last",
                BufId::X,
                u64::from(b - 1) * u64::from(n_embd),
                n_embd as usize,
            );
        }
        if flush_every > 0 && (li + 1) % flush_every == 0 && li + 1 < n_layers_total {
            be().flush();
        }
    }
    be().mega_program_end();

    // ---- last token only: the final norm and the lm head, same command buffer -------
    if !wf.state.logits_wanted {
        be().end()
            .map_err(|rc| format!("qwen35 forward failed rc={rc}"))?;
        crate::gpu_support::prefill_region_ended();
        return Ok(());
    }
    be().rms_norm(
        BufId::X,
        wf.w.output_norm.offset,
        n_embd,
        eps,
        1,
        n_embd,
        (b - 1) * n_embd,
    );
    be().matmat_from(
        wkind(&wf.w.output),
        wf.w.output.offset as u64,
        n_embd,
        c.vocab_size,
        BufId::X,
        BufId::Logits,
        1,
        b - 1,
    );
    if let Some(cap) = wf.plan.output.logit_softcap {
        be().softcap(BufId::Logits, cap, c.vocab_size);
    }
    if let Some(p) = pipe {
        be().argmax_feed(
            BufId::Logits,
            BufId::Tokens,
            BufId::Pick,
            p.pick_slot,
            c.vocab_size,
        );
        return be()
            .end_async()
            .map_err(|rc| format!("qwen35 pipelined step failed rc={rc}"));
    }
    if argmax {
        be().argmax(BufId::Logits, BufId::Tmp, c.vocab_size);
    }
    be().end()
        .map_err(|rc| format!("qwen35 forward failed rc={rc}"))?;
    crate::gpu_support::prefill_region_ended();

    if argmax {
        out.resize(1, 0.0);
        be().read(BufId::Tmp, 0, out);
    } else {
        out.resize(c.vocab_size as usize, 0.0);
        be().read(BufId::Logits, 0, out);
    }
    Ok(())
}

/// One co-batched decode step (docs/continuous-batching.md): row r decodes `rows[r].token` in
/// slot `rows[r].slot` at `rows[r].pos`. The operations are the one-row dispatch path's decode
/// (`batch` at one token, mega route off), in its order and with its arithmetic choices: the
/// projections, norms and FFN take all rows at once on the step's route
/// (`WorkflowState::row_route`, `Backend::set_decode_rows`), so on the exact route every row
/// gets its lone decode's bits. Each operation that touches one conversation's own state -- Q
/// and K normed and roped at the row's position, the KV store, attention over the row's cache,
/// the convolution and the delta rule on the row's slot -- is handed every row with its
/// position and slot.
///
/// # Errors
/// For a quantized cache, a backend without decode rows or a per-row operation, or a device
/// failure.
pub fn rows(
    wf: &mut Qwen35,
    rows: &[crate::DecodeRow],
    logits: Option<&mut Vec<f32>>,
    picks: &mut Vec<u32>,
) -> Result<(), String> {
    if !be().supports_gated_delta() {
        return Err(format!(
            "qwen35: the {} backend does not serve the gated delta-net mixer",
            be().device_tag()
        ));
    }
    if KvType::k() != KvType::F16 || KvType::v() != KvType::F16 {
        return Err("co-batched decode reads an f16 cache only".into());
    }
    if !be().supports_argmax_rows() {
        return Err("co-batched decode needs a per-row argmax".into());
    }
    let b = u32::try_from(rows.len()).map_err(|_| "too many co-batched rows")?;
    let most = be().decode_rows_max(wf.state.row_route);
    if rows.len() > most {
        return Err(format!(
            "{} co-batched rows; the {:?} route serves at most {most}",
            rows.len(),
            wf.state.row_route
        ));
    }
    // Every row's logits are live: the lm head writes one row per conversation.
    wf.state.output_demand = crate::OutputDemand::AllTokens;
    let fit = wf.gpu_fit_batch(rows.len());
    wf.state.output_demand = crate::OutputDemand::LastToken;
    fit?;
    if !be().set_decode_rows(Some(wf.state.row_route)) {
        return Err("the backend has no decode rows".into());
    }
    let encoded = encode_rows(wf, rows, b);
    be().set_decode_rows(None);
    encoded?;
    let mut got = vec![0.0_f32; rows.len()];
    be().read(BufId::Tmp, 0, &mut got);
    picks.clear();
    picks.extend(got.iter().map(|v| v.to_bits()));
    if let Some(out) = logits {
        out.resize(rows.len() * wf.plan.config.vocab_size as usize, 0.0);
        be().read(BufId::Logits, 0, out);
    }
    Ok(())
}

/// The graph of [`rows`], ending with the per-row argmax in `BufId::Tmp`.
#[allow(clippy::too_many_lines)]
fn encode_rows(wf: &Qwen35, rows: &[crate::DecodeRow], b: u32) -> Result<(), String> {
    let c = &wf.plan.config;
    let n_embd = c.n_embd;
    let n_head = c.n_heads;
    let n_kv = c.n_kv_heads;
    let n_ff = c.n_ff;
    let eps = c.norm_eps;
    let wkind = |t: &imparo_gguf::weights::Tensor| {
        imparo_gguf::weights::weight_kind(t.ggml_type).expect("validated at load")
            as u32
    };
    let tokens: Vec<u32> = rows.iter().map(|r| r.token).collect();
    let pos: Vec<u32> = rows.iter().map(|r| r.pos).collect();
    let slot_rows: Vec<imparo_backend::SlotRow> = rows
        .iter()
        .map(|r| imparo_backend::SlotRow {
            slot: r.slot,
            pos: r.pos,
        })
        .collect();
    let recur = wf.plan.recurrent_layout();
    let recur_elems = wf.plan.recurrent_elems();
    let state_rows = |off: u32| -> Vec<imparo_backend::SlotStateRow> {
        rows.iter()
            .map(|r| imparo_backend::SlotStateRow {
                slot: r.slot,
                state_off: off + r.plane_in * recur_elems,
                state_out_off: off + r.plane_out * recur_elems,
            })
            .collect()
    };

    be().begin_forward(false);
    be().write_u32(BufId::Tokens, 0, &tokens);
    let embd_kind = wkind(&wf.w.token_embd);
    let embd_off = wf.w.token_embd.offset as u64;
    if !be().gather_rows(
        embd_kind,
        embd_off,
        n_embd,
        c.vocab_size,
        1.0,
        BufId::X,
        0,
        BufId::Tokens,
        b,
    ) {
        for (t, &tok) in tokens.iter().enumerate() {
            be().row(
                embd_kind,
                embd_off,
                n_embd,
                tok,
                1.0,
                BufId::X,
                t as u32 * n_embd,
            );
        }
    }
    trace_rows("embd", 0, BufId::X, b, n_embd);
    let n_layers = wf.plan.layers.len();
    // The decode seat: every row is a decode row.
    let flush_every = crate::gpu_support::flush_layers_bounded(1, n_layers);
    for (li, (&layer, &(r_off, s_off, _, _))) in
        wf.plan.layers.iter().zip(recur.iter()).enumerate()
    {
        let lw = &wf.w.layers[li];
        be().rms_norm_from(
            BufId::Cur,
            BufId::X,
            lw.attn_norm.offset,
            n_embd,
            eps,
            b,
            n_embd,
            0,
        );
        match (&lw.mixer, layer.attention) {
            (
                MixerW::Attention {
                    q_norm,
                    k_norm,
                    wq,
                    wk,
                    wv,
                    wo,
                },
                crate::Attention::Full {
                    head_dim: hd,
                    rope_base,
                    rope_dim,
                },
            ) => {
                let (qw, kw) = (n_head * hd, n_kv * hd);
                be().matmat(
                    wkind(wq),
                    wq.offset as u64,
                    n_embd,
                    2 * qw,
                    BufId::Cur,
                    MIX,
                    b,
                );
                be().copy_strided(BufId::Q, MIX, hd, 0, 2 * hd, b * n_head);
                be().matmat(
                    wkind(wk),
                    wk.offset as u64,
                    n_embd,
                    kw,
                    BufId::Cur,
                    BufId::K,
                    b,
                );
                be().matmat(
                    wkind(wv),
                    wv.offset as u64,
                    n_embd,
                    kw,
                    BufId::Cur,
                    BufId::V,
                    b,
                );
                let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
                let max_scores: Vec<u32> =
                    pos.iter().map(|&p| scores_needed(p, 1, 0)).collect();
                let served = be().head_norm_rope_at(
                    BufId::Q,
                    q_norm.offset,
                    hd,
                    eps,
                    n_head,
                    &pos,
                    rope_dim,
                    rope_base,
                    None,
                ) && be().head_norm_rope_at(
                    BufId::K,
                    k_norm.offset,
                    hd,
                    eps,
                    n_kv,
                    &pos,
                    rope_dim,
                    rope_base,
                    None,
                ) && be().kv_store_slot_rows(
                    BufId::K,
                    li as u32,
                    kw,
                    &slot_rows,
                    false,
                    ring,
                ) && be().kv_store_slot_rows(
                    BufId::V,
                    li as u32,
                    kw,
                    &slot_rows,
                    true,
                    ring,
                ) && be().attention_slot_rows(
                    li as u32,
                    hd,
                    n_head,
                    n_kv,
                    kw,
                    1.0 / (hd as f32).sqrt(),
                    0,
                    &slot_rows,
                    &max_scores,
                    ring,
                );
                if !served {
                    return Err(format!(
                        "co-batched attention not served at layer {li}"
                    ));
                }
                // The output gate, before o_proj: plain sigmoid of each packed head's
                // second half.
                be().mul_strided_sigmoid(
                    BufId::Attn,
                    MIX,
                    hd,
                    hd,
                    2 * hd,
                    hd,
                    b * n_head,
                );
                be().matmat(
                    wkind(wo),
                    wo.offset as u64,
                    qw,
                    n_embd,
                    BufId::Attn,
                    BufId::O,
                    b,
                );
            }
            (
                MixerW::GatedDelta {
                    qkv,
                    gate: gate_w,
                    conv,
                    a,
                    dt_bias,
                    alpha,
                    beta,
                    ssm_norm,
                    out: out_w,
                },
                crate::Attention::Recurrent { .. },
            ) => {
                let (sh, taps) = delta_shape(layer.attention, conv.w.len())?;
                let qkv_width =
                    u32::try_from(sh.qkv_width()).map_err(|_| "qkv width")?;
                let v_width = u32::try_from(sh.v_heads * sh.value_dim)
                    .map_err(|_| "value width")?;
                let (kd, vd) = (sh.key_dim as u32, sh.value_dim as u32);
                let vh = u32::try_from(sh.v_heads).map_err(|_| "value heads")?;
                let taps = u32::try_from(taps).map_err(|_| "conv taps")?;
                be().matmat(
                    wkind(qkv),
                    qkv.offset as u64,
                    n_embd,
                    qkv_width,
                    BufId::Cur,
                    MIX,
                    b,
                );
                be().matmat(
                    wkind(gate_w),
                    gate_w.offset as u64,
                    n_embd,
                    v_width,
                    BufId::Cur,
                    Z,
                    b,
                );
                if !skip_alpha_beta() {
                    be().matmat(
                        wkind(alpha),
                        alpha.offset as u64,
                        n_embd,
                        vh,
                        BufId::Cur,
                        ALPHA,
                        b,
                    );
                    be().matmat(
                        wkind(beta),
                        beta.offset as u64,
                        n_embd,
                        vh,
                        BufId::Cur,
                        BETA,
                        b,
                    );
                }
                if !be().causal_conv_slot_rows(
                    ConvForm::PlainSilu,
                    MIX,
                    conv.offset,
                    BufId::Recur,
                    &state_rows(r_off),
                    CONV,
                    qkv_width,
                    taps,
                ) {
                    return Err(format!(
                        "co-batched convolution not served at layer {li}"
                    ));
                }
                let fuse_epi = be().delta_net_fuses_epilogue();
                if !be().delta_net_slot_rows(
                    &imparo_backend::DeltaNet {
                        qkv: CONV,
                        alpha: ALPHA,
                        beta: BETA,
                        a_off: a.offset,
                        dt_bias_off: dt_bias.offset,
                        state: BufId::Recur,
                        state_off: 0,
                        state_out_off: 0,
                        out: BufId::Attn,
                        epilogue: fuse_epi.then_some(imparo_backend::DeltaEpilogue {
                            norm_w_off: ssm_norm.offset,
                            gate: Z,
                        }),
                        snap: None,
                        k_heads: u32::try_from(sh.k_heads).map_err(|_| "key heads")?,
                        v_heads: vh,
                        key_dim: kd,
                        value_dim: vd,
                        n_tok: b,
                        eps,
                    },
                    &state_rows(s_off),
                ) {
                    return Err(format!(
                        "co-batched delta rule not served at layer {li}"
                    ));
                }
                if !fuse_epi {
                    be().rms_norm(BufId::Attn, ssm_norm.offset, vd, eps, b * vh, vd, 0);
                    be().act_mul(Z, BufId::Attn, b * v_width);
                }
                be().matmat(
                    wkind(out_w),
                    out_w.offset as u64,
                    v_width,
                    n_embd,
                    Z,
                    BufId::O,
                    b,
                );
            }
            (_, at) => {
                return Err(format!(
                    "qwen35 layer {li}: plan says {at:?} but the weights resolved otherwise"
                ));
            }
        }
        trace_rows("mix", li, BufId::O, b, n_embd);
        // The one-row decode's tail on the dispatch path: the residual, the FFN norm, gate
        // and up as two projections multiplied by the activation, down, the residual.
        be().add(BufId::X, BufId::O, b * n_embd);
        be().rms_norm_from(
            BufId::Cur,
            BufId::X,
            lw.ffn_norm.offset,
            n_embd,
            eps,
            b,
            n_embd,
            0,
        );
        let fused_ffn = be().ffn_gated_down(
            wkind(&lw.ffn_gate),
            lw.ffn_gate.offset as u64,
            wkind(&lw.ffn_up),
            lw.ffn_up.offset as u64,
            wkind(&lw.ffn_down),
            lw.ffn_down.offset as u64,
            n_embd,
            n_ff,
            n_embd,
            BufId::Cur,
            BufId::G,
            BufId::O,
            b,
        );
        if !fused_ffn {
            let fused_pair = be().matmat_gated(
                wkind(&lw.ffn_gate),
                lw.ffn_gate.offset as u64,
                wkind(&lw.ffn_up),
                lw.ffn_up.offset as u64,
                n_embd,
                n_ff,
                BufId::Cur,
                BufId::G,
                BufId::U,
                b,
            );
            if !fused_pair {
                // No activation epilogue on the up projection: a one-row decode never fuses it.
                be().matmat(
                    wkind(&lw.ffn_gate),
                    lw.ffn_gate.offset as u64,
                    n_embd,
                    n_ff,
                    BufId::Cur,
                    BufId::G,
                    b,
                );
                be().matmat(
                    wkind(&lw.ffn_up),
                    lw.ffn_up.offset as u64,
                    n_embd,
                    n_ff,
                    BufId::Cur,
                    BufId::U,
                    b,
                );
                be().act_mul(BufId::G, BufId::U, b * n_ff);
            }
            be().matmat(
                wkind(&lw.ffn_down),
                lw.ffn_down.offset as u64,
                n_ff,
                n_embd,
                BufId::G,
                BufId::O,
                b,
            );
        }
        be().add(BufId::X, BufId::O, b * n_embd);
        trace_rows("out", li, BufId::X, b, n_embd);
        if flush_every > 0 && (li + 1) % flush_every == 0 && li + 1 < n_layers {
            be().flush();
        }
    }
    be().rms_norm(BufId::X, wf.w.output_norm.offset, n_embd, eps, b, n_embd, 0);
    be().matmat_from(
        wkind(&wf.w.output),
        wf.w.output.offset as u64,
        n_embd,
        c.vocab_size,
        BufId::X,
        BufId::Logits,
        b,
        0,
    );
    if let Some(cap) = wf.plan.output.logit_softcap {
        be().softcap(BufId::Logits, cap, b * c.vocab_size);
    }
    // A row whose step ends on a checkpoint boundary keeps its state there: its slot's
    // snapshot twin takes a copy of the plane the step wrote, as the one-row decode's
    // boundary snapshot writes it aside.
    if rows.iter().any(|r| r.snap) {
        for r in rows.iter().filter(|r| r.snap) {
            if !be().select_slot(r.slot) {
                return Err(format!(
                    "co-batched snapshot: slot {} not selected",
                    r.slot
                ));
            }
            be().copy_range(
                BufId::RecurSnap,
                0,
                BufId::Recur,
                r.plane_out * recur_elems,
                recur_elems,
            );
        }
        if !be().select_slot(wf.state.slot) {
            return Err(format!(
                "co-batched snapshot: slot {} not selected",
                wf.state.slot
            ));
        }
    }
    be().argmax_rows(BufId::Logits, BufId::Tmp, c.vocab_size, b);
    be().end()
        .map_err(|rc| format!("qwen35 co-batched step failed rc={rc}"))
}

/// IMPARO_SKIP_AB=1 drops the alpha and beta projections. A skip lever for the
/// difference-two-runs method (docs: `prof_log` says to use skip levers rather than the
/// profiler for per-category time), read once. Wrong logits are the point of it.
fn skip_alpha_beta() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_SKIP_AB").is_ok_and(|v| v == "1"))
}
