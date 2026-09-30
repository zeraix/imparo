//! Qwen3's layer graph on the device.
//!
//! ```text
//! every block:   prev = X
//!                Cur  = rms_norm(X, attn_norm)
//!                Q,K,V = wq @ Cur, wk @ Cur, wv @ Cur
//!                Q    = rope(rms_norm_per_head(Q, attn_q_norm))
//!                K    = rope(rms_norm_per_head(K, attn_k_norm))     -> KV cache
//!                X    = prev + wo @ attention(Q, cache)
//!                Cur  = rms_norm(X, ffn_norm)
//!                X    = X + ffn_down(silu(ffn_gate @ Cur) * ffn_up @ Cur)
//! tail:          logits = head @ rms_norm(X_last, output_norm)
//! ```
//!
//! No recurrent state, no convolution, no sliding window, no per-layer embeddings, no
//! output scale, no softcap, no mega program. Every block takes the same path, so there
//! is no arm here that the plan chooses between -- what the other architectures spend
//! their length on is exactly what this one does not have.
//!
//! The fusions are ASKED FOR AND CHECKED, never assumed: each of `kv_head_postprocess_store`,
//! `add_rms_norm`, `ffn_gated_down` and `matmat_gated` returns whether the backend served
//! it, and the unfused sequence that follows is the same arithmetic. A backend that serves
//! none of them still runs this model.

use imparo_backend::BufId;

use crate::gpu_support::{
    BufferRequirement, Placement, be, gprobe, gpu_probe_layer,
    half_activation_mirror_requirements, kv_dequant_scratch_requirements, kvq_mask_on,
    scores_needed,
};
use crate::kv::{KvType, effective_workflow_kv_route, had_nrot, ring_mask};
use crate::qwen3::Qwen3;
use crate::{Attention, ModelPlan};

/// The buffers one forward of this model needs, as a function of the plan and the batch.
///
/// A free function over the plan: it reads nothing else, so it is testable without the
/// file on disk.
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
        need(
            BufId::Q,
            f(b * c.n_heads as usize * head_max),
            Placement::Group(0),
        ),
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
        // ATTN DOES NOT SHARE: the register-tiled GEMM reads whole 8-row tiles and cannot
        // mask, so it reads past the token count, and what those rows hold changes the
        // logits.
        need(
            BufId::Attn,
            f(b * c.n_heads as usize * head_max),
            Placement::Dedicated,
        ),
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
    // The half-precision activation mirrors, declared only when the epilogue fusion that
    // reads them is on: declaring them is what enables the family, so not declaring them
    // is the single-point way to keep the fusion off rather than corrupt.
    if crate::gpu_support::fuse_epilogue_enabled() {
        requirements.extend(half_activation_mirror_requirements(b, n_ff, BufId::U));
    }
    requirements
}

/// Nothing to prepare beyond what the backend does for every model.
///
/// The quantized weight prepack LFM2 asks for here is a speed facility, not a
/// correctness one; this model has not been measured with it, so it does not claim it.
///
/// # Errors
/// Never; the signature is the architecture table's.
pub fn prepare_device(_wf: &mut Qwen3) -> Result<(), String> {
    Ok(())
}

/// Runs ONE chunk of Qwen3's forward on the device.
///
/// `argmax` true means: do not copy the logits back, write the greedy pick's index into
/// `out[0]` as raw bits.
///
/// # Errors
/// When the output demand is one this workflow does not serve, or a dispatch fails.
#[allow(clippy::too_many_lines)]
pub fn batch(
    wf: &mut Qwen3,
    tokens: &[u32],
    start_pos: usize,
    out: &mut Vec<f32>,
    argmax: bool,
) -> Result<(), String> {
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
    let b = u32::try_from(tokens.len()).map_err(|_| "batch too large")?;
    let sp = u32::try_from(start_pos).map_err(|_| "start_pos too large")?;

    // THE DRAFTER READS THESE. A paired DSpark drafter takes the target's hidden state
    // at its tap layers; a forward that publishes none fails at the pairing with "no
    // forward was captured", which is what this model did before the record() below.
    let layer_outputs = wf
        .state
        .layer_outputs
        .as_ref()
        .is_some_and(crate::layer_outputs::LayerOutputCapture::active);
    let row_argmax = wf.state.output_demand == crate::OutputDemand::RowArgmax;
    let all_logits = wf.state.output_demand.requires_all_positions();
    if all_logits && argmax {
        return Err("all-position logits cannot request scalar argmax".into());
    }
    if row_argmax && !be().supports_argmax_rows() {
        return Err("row argmax is not served by this backend".into());
    }
    // A PREFIX VERIFICATION IS JUST THIS BATCH. The caller snapshots the recurrent state
    // after each prefix so a partial accept can roll back to it; this architecture keeps
    // none, so there is nothing to snapshot and the all-logits forward below already
    // answers every prefix. Refusing it instead sent the engine down the sequential
    // verifier, which verifies one token per forward -- and then the drafter's feature
    // bookkeeping, which asks for the rows of ONE forward, could not be satisfied.
    if wf.state.verification_prefix_tokens != 0
        && (wf.state.verification_prefix_tokens != tokens.len() || !all_logits)
    {
        return Err(
            "verification prefixes require the complete all-logits batch".into(),
        );
    }
    if wf.state.output_demand == crate::OutputDemand::GreedyVerification {
        return Err("qwen3 serves no device greedy verification".into());
    }
    // A ROW-LAYOUT BATCH IS A TREE. Each row carries its own position and its own 64-bit
    // visibility mask in BufId::RowLayout, so siblings do not see each other; the rows are
    // stored contiguously and the mask, not the store, decides what each row attends to.
    // It keeps every row and reads an f16 cache.
    let row_layout = wf.state.row_layout;
    if row_layout != 0
        && (row_layout != b
            || !all_logits
            || KvType::k() != KvType::F16
            || KvType::v() != KvType::F16)
    {
        return Err("a row-layout batch keeps every row and reads an f16 cache".into());
    }

    // Activation aliases and scratch ranges are planned for the live batch width.
    wf.gpu_fit_batch(tokens.len())?;
    let decode = b == 1;
    let decode_projection_preparation =
        decode && be().use_decode_projection_preparation();
    let prefill_projection_preparation =
        !decode && be().use_prefill_projection_preparation();
    let replayed = if decode {
        // A captured forward must run its layers rather than replay a graph that
        // retains the observer's buffers.
        be().decode_prepare(tokens[0], sp, argmax && !layer_outputs)
            .map_err(|rc| format!("qwen3 GPU decode prepare failed rc={rc}"))?
    } else {
        false
    };
    if replayed {
        be().end()
            .map_err(|rc| format!("qwen3 GPU forward failed rc={rc}"))?;
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

    // A pipelined step: the token is already in Tokens[0], written by the previous step's
    // argmax_feed on the device.
    let pipe = wf.state.pipe;
    if !pipe.is_some_and(|p| p.token_on_device) {
        be().write_u32(BufId::Tokens, 0, tokens);
    }
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
    gprobe("inp_embd", BufId::X, 0, n_embd as usize);

    let n_layers_total = wf.plan.layers.len();
    let flush_every = crate::gpu_support::flush_layers_bounded(b, n_layers_total);
    let act = wf.plan.layers[0].ffn.activation().epilogue();

    // `operator_norm_ready` is the previous layer's fused residual-and-norm having
    // already written Cur for this layer. It is a fact about what ran, not a setting.
    let mut operator_norm_ready = false;

    for (li, layer) in wf.plan.layers.iter().enumerate() {
        let lw = &wf.w.layers[li];
        let probe_here = li == gpu_probe_layer();
        let Attention::Full {
            head_dim,
            rope_base,
            rope_dim,
        } = layer.attention
        else {
            return Err(format!(
                "qwen3 layer {li}: the plan says {:?}, but every Qwen3 block attends",
                layer.attention
            ));
        };
        let hd = head_dim;
        let (qw, kw) = (n_head * hd, n_kv * hd);

        if !operator_norm_ready {
            if decode_projection_preparation || prefill_projection_preparation {
                be().rms_norm_projection(
                    BufId::Cur,
                    BufId::X,
                    lw.attn_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                );
            } else {
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
        }
        operator_norm_ready = false;
        if probe_here {
            gprobe("attn_norm", BufId::Cur, 0, n_embd as usize);
        }

        be().matmat(
            wkind(&lw.wq),
            lw.wq.offset as u64,
            n_embd,
            qw,
            BufId::Cur,
            BufId::Q,
            b,
        );
        be().matmat(
            wkind(&lw.wk),
            lw.wk.offset as u64,
            n_embd,
            kw,
            BufId::Cur,
            BufId::K,
            b,
        );
        be().matmat(
            wkind(&lw.wv),
            lw.wv.offset as u64,
            n_embd,
            kw,
            BufId::Cur,
            BufId::V,
            b,
        );

        let qk_hadamard_nrot = if KvType::k() == KvType::F16 {
            0
        } else {
            had_nrot("IMPARO_HAD_QK", kv_quant_route.key, hd)?
        };
        let v_hadamard_nrot = if KvType::v() == KvType::F16 {
            0
        } else {
            had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?
        };
        // PER-HEAD NORM THEN ROTATE, Q AND K BOTH -- the step a reader of the tensor list
        // would not know is there, because the norms are one vector each.
        if row_layout != 0 {
            // Each row ropes at ITS OWN layout position. The cache is f16 here, so there
            // is no rotation to apply and K is done beside Q.
            for (buf, norm, heads) in [
                (BufId::Q, lw.q_norm.offset, n_head),
                (BufId::K, lw.k_norm.offset, n_kv),
            ] {
                if !be().head_norm_rope_rows(
                    buf, norm, hd, eps, heads, b, rope_dim, rope_base, None,
                ) {
                    return Err(format!(
                        "row-layout head norm and rope not served at layer {li}"
                    ));
                }
            }
        } else {
            be().head_norm_rope_hadamard(
                BufId::Q,
                lw.q_norm.offset,
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
        }

        let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
        let direct_kv = row_layout == 0
            && !probe_here
            && KvType::k() == KvType::Q8_0
            && KvType::v() == KvType::Q8_0
            && be().kv_head_postprocess_store(
                BufId::K,
                BufId::V,
                lw.k_norm.offset,
                false,
                hd,
                eps,
                n_kv,
                sp,
                b,
                rope_dim,
                rope_base,
                None,
                qk_hadamard_nrot,
                v_hadamard_nrot,
                li as u32,
                ring,
            );
        if !direct_kv {
            if row_layout == 0 {
                be().head_norm_rope_hadamard(
                    BufId::K,
                    lw.k_norm.offset,
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
            }
            be().kv_store(BufId::K, li as u32, kw, sp, b, false, ring);
            be().kv_store(BufId::V, li as u32, kw, sp, b, true, ring);
        }
        if probe_here {
            gprobe("Qcur_pos", BufId::Q, 0, qw as usize);
        }

        if row_layout != 0 {
            if ring != 0
                || !be().attention_rows(
                    li as u32,
                    hd,
                    n_head,
                    n_kv,
                    kw,
                    sp,
                    0,
                    1.0 / (hd as f32).sqrt(),
                    b,
                    crate::verification::tree_float_q(),
                )
            {
                return Err(format!("row-layout attention not served at layer {li}"));
            }
        } else {
            be().attention(
                li as u32,
                hd,
                n_head,
                n_kv,
                kw,
                sp,
                1.0 / (hd as f32).sqrt(),
                // Qwen3 carries no sliding window; a file that set one would plan
                // Attention::Window and this arm would not match it.
                0,
                b,
                scores_needed(sp, b, 0),
                ring,
            );
        }
        if v_hadamard_nrot != 0 {
            be().hadamard(BufId::Attn, b * qw, v_hadamard_nrot);
        }
        if probe_here {
            gprobe("attention_full", BufId::Attn, 0, (b * qw) as usize);
        }
        be().matmat(
            wkind(&lw.wo),
            lw.wo.offset as u64,
            qw,
            n_embd,
            BufId::Attn,
            BufId::O,
            b,
        );

        // Residual and the FFN's pre-norm, fused when the backend serves it. The fused
        // form adds into X itself; the unfused pair below is the same arithmetic.
        let ffn_norm_fused = be().add_rms_norm(
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
        if !ffn_norm_fused {
            be().add(BufId::X, BufId::O, b * n_embd);
            if decode_projection_preparation || prefill_projection_preparation {
                be().rms_norm_projection(
                    BufId::Cur,
                    BufId::X,
                    lw.ffn_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                );
            } else {
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
        }
        if probe_here {
            gprobe("ffn_norm", BufId::Cur, 0, n_embd as usize);
        }

        let (gate_kind, gate_off) = (wkind(&lw.ffn_gate), lw.ffn_gate.offset as u64);
        let (up_kind, up_off) = (wkind(&lw.ffn_up), lw.ffn_up.offset as u64);
        let (down_kind, down_off) = (wkind(&lw.ffn_down), lw.ffn_down.offset as u64);
        let whole_ffn = be().ffn_gated_down(
            gate_kind,
            gate_off,
            up_kind,
            up_off,
            down_kind,
            down_off,
            n_embd,
            n_ff,
            n_embd,
            BufId::Cur,
            BufId::G,
            BufId::O,
            b,
        );
        if !whole_ffn {
            let paired = be().matmat_gated(
                gate_kind,
                gate_off,
                up_kind,
                up_off,
                n_embd,
                n_ff,
                BufId::Cur,
                BufId::G,
                BufId::U,
                b,
            );
            if !paired {
                be().matmat(gate_kind, gate_off, n_embd, n_ff, BufId::Cur, BufId::G, b);
                let fused_epilogue = crate::gpu_support::should_fuse_epilogue(
                    b,
                    be().supports_epilogue(act),
                );
                if fused_epilogue {
                    be().set_epilogue(act);
                }
                be().matmat(
                    up_kind,
                    up_off,
                    n_embd,
                    n_ff,
                    BufId::Cur,
                    if fused_epilogue { BufId::G } else { BufId::U },
                    b,
                );
                if fused_epilogue {
                    be().set_epilogue(imparo_backend::Epilogue::None);
                } else {
                    be().act_mul(BufId::G, BufId::U, b * n_ff);
                }
            }
            be().matmat(down_kind, down_off, n_ff, n_embd, BufId::G, BufId::O, b);
        }

        // The last layer's residual has no next norm to fuse into.
        if li + 1 < n_layers_total {
            let next_norm = wf.w.layers[li + 1].attn_norm.offset;
            operator_norm_ready = be().add_rms_norm(
                BufId::Cur,
                BufId::X,
                BufId::O,
                next_norm,
                n_embd,
                eps,
                b,
                n_embd,
                0,
            );
        }
        if !operator_norm_ready {
            be().add(BufId::X, BufId::O, b * n_embd);
        }
        if let Some(capture) = &wf.state.layer_outputs {
            capture.record(li as u32, sp, b, BufId::X)?;
        }
        if probe_here {
            gprobe("l_out", BufId::X, 0, n_embd as usize);
        }
        if flush_every != 0 && (li + 1) % flush_every == 0 && li + 1 < n_layers_total {
            be().flush();
        }
    }

    if !wf.state.output_demand.wants_logits() {
        be().end()
            .map_err(|rc| format!("qwen3 forward failed rc={rc}"))?;
        crate::gpu_support::prefill_region_ended();
        return Ok(());
    }
    let output_rows = if all_logits { b } else { 1 };
    let output_start = if all_logits { 0 } else { b - 1 };
    let output_words = output_rows
        .checked_mul(c.vocab_size)
        .ok_or_else(|| "all-position logits count exceeds u32".to_string())?;
    let (lm_head_src, lm_head_src_row) = if prefill_projection_preparation {
        be().rms_norm_projection(
            BufId::Cur,
            BufId::X,
            wf.w.output_norm.offset,
            n_embd,
            eps,
            output_rows,
            n_embd,
            output_start * n_embd,
        );
        (BufId::Cur, 0)
    } else if decode_projection_preparation {
        be().rms_norm_projection(
            BufId::X,
            BufId::X,
            wf.w.output_norm.offset,
            n_embd,
            eps,
            1,
            n_embd,
            0,
        );
        (BufId::X, 0)
    } else {
        be().rms_norm(
            BufId::X,
            wf.w.output_norm.offset,
            n_embd,
            eps,
            output_rows,
            n_embd,
            output_start * n_embd,
        );
        (BufId::X, output_start)
    };
    be().matmat_from(
        wkind(&wf.w.head),
        wf.w.head.offset as u64,
        n_embd,
        c.vocab_size,
        lm_head_src,
        BufId::Logits,
        output_rows,
        lm_head_src_row,
    );
    if let Some(cap) = wf.plan.output.logit_softcap {
        be().softcap(BufId::Logits, cap, output_words);
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
            .map_err(|rc| format!("qwen3 pipelined step failed rc={rc}"));
    }
    if row_argmax {
        be().argmax_rows(BufId::Logits, BufId::Tmp, c.vocab_size, b);
    } else if argmax {
        be().argmax(BufId::Logits, BufId::Tmp, c.vocab_size);
    }
    be().end()
        .map_err(|rc| format!("qwen3 forward failed rc={rc}"))?;
    crate::gpu_support::prefill_region_ended();

    if row_argmax {
        out.resize(b as usize, 0.0);
        be().read(BufId::Tmp, 0, out);
    } else if argmax {
        out.resize(1, 0.0);
        be().read(BufId::Tmp, 0, out);
    } else {
        out.resize(output_words as usize, 0.0);
        be().read(BufId::Logits, 0, out);
    }
    Ok(())
}
