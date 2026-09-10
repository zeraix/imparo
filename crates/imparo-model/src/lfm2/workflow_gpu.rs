//! LFM2 on the GPU: what the model contributes, and nothing that is not its own.
//!
//! WHY THIS EXISTS BEFORE THE FORWARD. The point of a second model is to say which of the
//! first model's code was machinery and which was gemma4. Guessing that from one example is
//! how a "shared" helper ends up with one architecture's assumptions baked in; writing the
//! second model's declarations is how it gets settled. Everything here is either a fact
//! about LFM2 or a trait method it must supply -- the pool geometry, the capture engine,
//! the arena placement and the probes are all reached, not re-written.
//!
//! The forward itself needs kernels that do not exist yet (ShortConv, and a SiLU epilogue),
//! so it is not here. What IS here compiles, allocates and joins the pool.

use imparo_backend::BufId;

use crate::ModelPlan;
use crate::gpu_support::{
    BufferRequirement, Placement, be, gprobe, gpu_probe_layer,
    half_activation_mirror_requirements, kv_dequant_scratch_requirements, kvq_mask_on,
    scores_needed, tail_align, tail_split, tail_split_min_rows,
};
use crate::kv::{KvType, effective_workflow_kv_route, had_nrot, ring_mask};
use crate::lfm2::Lfm2;
use crate::lfm2::workflow_cpu::MixerW;

/// LFM2'S PRIVATE BUFFER SLOTS.
///
/// `bcx` is the ShortConv input projection: n_embd -> 3 * n_embd, carrying b, c and x
/// concatenated. gemma4 has no such buffer, and before the shared slots were generic there
/// was no name for it -- BufId's model-private range was three entries called Gate, Back
/// and PerLayer, which are gemma4's per-layer-embedding buffers.
pub const BCX: BufId = BufId::Model0;

/// Admit optional device-owned layouts for LFM2 Down projections. The model
/// supplies resolved offsets and shapes only; each backend decides whether the
/// format is useful and unsupported backends preserve their established path.
pub fn prepare_device(wf: &mut Lfm2) -> Result<(), String> {
    if !be().quantized_weight_cache_enabled() {
        return Ok(());
    }
    let n_in = wf.plan.config.n_ff;
    let n_out = wf.plan.config.n_embd;
    let cache = be().quantized_weight_cache_plan();
    let mut spans = Vec::with_capacity(
        wf.w.layers.len() * if cache.include_full_ffn { 3 } else { 1 }
            + usize::from(cache.include_head),
    );
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
        push(&wf.w.token_embd, n_out, wf.plan.config.vocab_size);
    }
    be().prepare_quantized_weight_cache(&spans)
        .map(|_| ())
        .map_err(|rc| format!("backend quantized-weight admission failed rc={rc}"))
}

/// Every activation buffer this model needs for a batch of `b`.
///
/// WHAT DIFFERS FROM gemma4, which is the whole reason to write this now:
///
///   - `bcx` at 3 * n_embd exists here and nowhere else;
///   - there is no Gate, Back or PerLayer -- those are gemma4's per-layer embeddings;
///   - Q is n_head * 64 and K/V are n_kv * 64, so the attention group is a quarter the
///     width gemma4's is, while the FFN group is wider (n_ff 10752 against 10240);
///   - the ShortConv state is NOT here. It is per-conversation and constant in context,
///     so it is allocated once like KV rather than resized per batch -- see the state
///     kinds table in docs/unified-kv-pool.md.
///
/// The attention and feed-forward groups alias for the same reason they do in gemma4: a
/// block runs one and then the other. `bcx` joins the ATTENTION group because a
/// ShortConv block and an attention block never both run in one layer -- the projection
/// is live exactly where Q/K/V would be.
///
/// A FREE FUNCTION OVER THE PLAN, not a method: it reads nothing else, and that is what
/// makes it testable without a 2.8 GB file on disk. A buffer list nothing can check is a
/// buffer list nobody has checked.
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
        // ATTN does not share, for the reason gemma4 records at its own list: the
        // register-tiled GEMM reads whole 8-row tiles and cannot mask, so it reads past
        // the token count, and what those rows hold changes the logits.
        need(
            BufId::Attn,
            f(b * c.n_heads as usize * head_max),
            Placement::Dedicated,
        ),
        need(BCX, f(b * 3 * n_embd), Placement::Group(0)),
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
    // The half-precision activation mirrors. n_ff is the widest activation staged
    // through them (the down projection reads it), and U is dead during prefill because
    // the fused epilogue writes G.
    //
    // THE FUSION IS THE PRECONDITION, so the mirrors are declared only when it is on.
    // Unfused, the up projection writes U while reading its own input's mirror out of
    // U's pages, and the logits come back NaN -- verified, not theorised. Declaring the
    // buffers is what enables the whole family (every producer and reader gates on
    // `bufs[B_XH] != nil`), so not declaring them is the complete, single-point way to
    // keep IMPARO_FUSE_EPILOGUE=0 a legal configuration rather than a corrupt one.
    if fuse_epilogue_enabled() {
        requirements.extend(half_activation_mirror_requirements(b, n_ff, BufId::U));
    }
    requirements
}

/// LFM2's layer graph on the device.
///
/// ```text
/// every block:   prev = X
///                Cur  = rms_norm(X, operator_norm)
///                O    = shortconv(Cur)   OR   attention(Cur)
///                X    = prev + O
///                Cur  = rms_norm(X, ffn_norm)
///                X    = X + ffn_down(silu(ffn_gate @ Cur) * ffn_up @ Cur)
/// tail:          logits = token_embd @ rms_norm(X_last, token_embd_norm)
/// ```
///
/// No post-norms, no per-layer embeddings, no output scale, no V norm and no softcap --
/// gemma4 has all five and LFM2 none, which is most of why this is a third the length.
///
/// `argmax` true means: do not copy the logits back, write the greedy pick's index into
/// `out[0]` as raw bits. `Workflow` decodes that convention, once.
///
/// # Errors
/// When a dispatch fails.
#[allow(clippy::too_many_lines)]
pub fn batch(
    wf: &mut Lfm2,
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
    let recur = wf.plan.recurrent_layout();
    let act = wf.plan.layers[0].ffn.activation().epilogue();
    // Activation aliases and scratch ranges are planned for the live batch width.
    // Recurrent state and KV are allocated separately and survive this resize.
    // Without the fit, a resumed LFM2 forward kept the initial 512-token arena for
    // narrower chunks, violating the same per-batch contract Gemma4 uses.
    wf.gpu_fit_batch(tokens.len())?;
    let decode = b == 1;
    let decode_projection_preparation =
        decode && be().use_decode_projection_preparation();
    let prefill_projection_preparation =
        !decode && be().use_prefill_projection_preparation();
    let projection_preparation =
        decode_projection_preparation || prefill_projection_preparation;
    let replayed = if decode {
        be().decode_prepare(tokens[0], sp, argmax)
            .map_err(|rc| format!("LFM2 GPU decode prepare failed rc={rc}"))?
    } else {
        false
    };
    if replayed {
        be().end()
            .map_err(|rc| format!("LFM2 GPU forward failed rc={rc}"))?;
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

    // CUDA may retain and replay a stable single-token graph. Other backends inherit
    // `begin_forward`'s conservative `begin()` default, so Metal keeps its established
    // execution path.
    be().begin_forward(decode);

    // A pipelined step (docs/decode-turnaround.md): the token is already in Tokens[0],
    // written by the previous step's argmax_feed on the device.
    let pipe = wf.state.pipe;
    // ---- embeddings. No scale: LFM2 sets scale_by_sqrt_embd false. ------------------
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
    // THE RECURRENT PLANES for this batch (task #165). A decode step reads one plane and
    // writes the next, so the plane it read stays intact and a discarded or failed step is
    // undone by an index rather than by a pre-step copy of the whole history. Prefill is
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
    let flush_every = crate::gpu_support::flush_layers_bounded(b == 1, n_layers_total);
    let tail = if wf.state.logits_wanted && b >= 128 {
        tail_split(b, tail_align())
    } else {
        None
    };
    let row_local_tail = if wf.state.logits_wanted && b > 1 {
        tail_split_min_rows(b, 1, be().row_local_prefill_tail_rows())
    } else {
        None
    };
    let mut b = b;
    let mut sp = sp;
    let mut operator_norm_ready = false;
    for (li, (&layer, &(r_off, _, _, _))) in
        wf.plan.layers.clone().iter().zip(recur.iter()).enumerate()
    {
        let lw = &wf.w.layers[li];
        // One LFM2 layer, or its tail alone, as one persistent dispatch (Metal, decode).
        // With a mixer the block forms the operator norm from the residual itself and runs
        // the mixer and its state advance; every mode ends with the tail. Refused (false,
        // nothing written) elsewhere and on the probe layer.
        let b_tok = b;
        let mega_entry = |mixer: imparo_backend::Lfm2MegaMixer| -> bool {
            b_tok == 1
                && li != gpu_probe_layer()
                && be().mega_layer(&imparo_backend::MegaEntry {
                    layer: imparo_backend::MegaLayer::Lfm2(
                        imparo_backend::Lfm2MegaLayer {
                            mixer,
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
                        },
                    ),
                    n_tok: b_tok,
                })
        };
        // The whole layer in the mega-kernel, tried before the dispatch path's operator norm
        // (the kernel forms its own). Not with a rotated (quantised) cache, and not on a
        // final layer whose mixer output and FFN are dead. A plan/weights mismatch is
        // reported by the dispatch match below.
        let layer_in_block = match (&lw.mixer, layer.attention) {
            (
                MixerW::ShortConv {
                    conv,
                    in_proj,
                    out_proj,
                },
                crate::Attention::Recurrent { .. },
            ) => {
                let kern = u32::try_from(conv.w.len() / n_embd as usize)
                    .map_err(|_| "conv kernel width")?;
                // A checkpoint armed inside a one-token batch sits at that token: the
                // kernel writes the advanced history to the snapshot too.
                let snap = wf.state.recur_snap.map(|_| (BufId::RecurSnap, r_off));
                (wf.state.logits_wanted || li + 1 != n_layers_total)
                    && mega_entry(imparo_backend::Lfm2MegaMixer::ShortConv {
                        op_norm_off: lw.op_norm.offset,
                        in_kind: wkind(in_proj),
                        in_off: in_proj.offset as u64,
                        conv_off: conv.offset,
                        out_kind: wkind(out_proj),
                        out_off: out_proj.offset as u64,
                        kernel: kern,
                        bcx: BCX,
                        state: BufId::Recur,
                        state_off: r_off + plane_in,
                        state_out_off: r_off + plane_out,
                        snap,
                    })
            }
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
                // The cache basis the block must reproduce (task #156): the Hadamard widths
                // the quantized modes rotate Q/K and V by (0 = plain), resolved as below.
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
                (wf.state.logits_wanted || li + 1 != n_layers_total)
                    && mega_entry(imparo_backend::Lfm2MegaMixer::Attention {
                        op_norm_off: lw.op_norm.offset,
                        wq_kind: wkind(wq),
                        wq_off: wq.offset as u64,
                        wk_kind: wkind(wk),
                        wk_off: wk.offset as u64,
                        wv_kind: wkind(wv),
                        wv_off: wv.offset as u64,
                        wo_kind: wkind(wo),
                        wo_off: wo.offset as u64,
                        q_norm_off: q_norm.offset,
                        k_norm_off: k_norm.offset,
                        head_dim: hd,
                        n_heads: n_head,
                        n_kv,
                        kv_width: n_kv * hd,
                        kv_layer: li as u32,
                        start_pos: sp,
                        window: 0,
                        ring: ring_mask(layer.attention, wf.state.kv_ring_batch),
                        rope_dim,
                        rope_base,
                        q_scale: 1.0 / (hd as f32).sqrt(),
                        q: BufId::Q,
                        k: BufId::K,
                        v: BufId::V,
                        attn: BufId::Attn,
                        had_k: qk_hadamard_nrot,
                        had_v: v_hadamard_nrot,
                    })
            }
            _ => false,
        };

        // ---- operator norm, then whichever mixer this block is ---------------------
        if !layer_in_block && !operator_norm_ready {
            if projection_preparation && li != gpu_probe_layer() {
                be().rms_norm_projection(
                    BufId::Cur,
                    BufId::X,
                    lw.op_norm.offset,
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
                    lw.op_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                );
            }
        }
        if li == gpu_probe_layer() {
            gprobe("operator_norm", BufId::Cur, 0, n_embd as usize);
            gprobe("operator_norm_full", BufId::Cur, 0, (b * n_embd) as usize);
            gprobe(
                "operator_norm_last",
                BufId::Cur,
                u64::from(b - 1) * u64::from(n_embd),
                n_embd as usize,
            );
        }

        match (&lw.mixer, layer.attention) {
            _ if layer_in_block => {}
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
                let qw = n_head * hd;
                let kw = n_kv * hd;
                // Per-head norm then rope, on Q and K. No frequency factors: LFM2
                // carries no rope_freqs tensor, so every layer ropes plainly.
                let qk_hadamard_nrot = if KvType::k() == KvType::F16 {
                    0
                } else {
                    had_nrot("IMPARO_HAD_K", kv_quant_route.key, hd)?
                };
                be().matmat(
                    wkind(wq),
                    wq.offset as u64,
                    n_embd,
                    qw,
                    BufId::Cur,
                    BufId::Q,
                    b,
                );
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
                if li == gpu_probe_layer() {
                    // Preserve the individually observable operations on the requested
                    // probe layer. Other layers use the backend semantic fusion hook;
                    // its default implementation is this exact three-operation sequence.
                    be().rms_norm(BufId::Q, q_norm.offset, hd, eps, b * n_head, hd, 0);
                    be().rms_norm(BufId::K, k_norm.offset, hd, eps, b * n_kv, hd, 0);
                    be().rope(BufId::Q, rope_dim, rope_base, hd, n_head, sp, b, None);
                    be().rope(BufId::K, rope_dim, rope_base, hd, n_kv, sp, b, None);
                    if qk_hadamard_nrot != 0 {
                        be().hadamard(BufId::Q, b * qw, qk_hadamard_nrot);
                        be().hadamard(BufId::K, b * kw, qk_hadamard_nrot);
                    }
                } else {
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
                }
                // Match the common llama attention graph: rotate Q/K/V before the
                // Q scale, store the rotated K/V bytes, and invert V after attention.
                if KvType::v() != KvType::F16 {
                    be().hadamard(
                        BufId::V,
                        b * kw,
                        had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?,
                    );
                }
                // Read-only post-Hadamard/pre-quant witnesses. `gprobe` is inert
                // unless IMPARO_GPU_PROBE is set and never modifies the buffer.
                if li == gpu_probe_layer() {
                    gprobe(
                        "Kcur_post_hadamard_last",
                        BufId::K,
                        u64::from(b - 1) * u64::from(kw),
                        kw as usize,
                    );
                    gprobe(
                        "Vcur_post_hadamard_last",
                        BufId::V,
                        u64::from(b - 1) * u64::from(kw),
                        kw as usize,
                    );
                }
                // The score scale is the attention op's (Backend::attention applies
                // 1/sqrt(d) to Q inside its entry); gemma4's is folded into its q_norm
                // weights in the file and it passes 1.0. The Q probes below therefore see
                // the UNSCALED projection.
                if li == gpu_probe_layer() {
                    gprobe("Qcur_pos", BufId::Q, 0, qw as usize);
                    gprobe(
                        "Qcur_pos_last",
                        BufId::Q,
                        u64::from(b - 1) * u64::from(qw),
                        qw as usize,
                    );
                }
                let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
                be().kv_store(BufId::K, li as u32, kw, sp, b, false, ring);
                be().kv_store(BufId::V, li as u32, kw, sp, b, true, ring);
                // The final layer's KV stores are this non-final chunk's last
                // externally visible state writes. Its remaining attention and FFN
                // feed only logits which the chunk loop discards.
                if li + 1 == n_layers_total && !wf.state.logits_wanted {
                    break;
                }
                if li + 1 == n_layers_total {
                    if let Some((r0, bt)) = tail {
                        be().copy_range(
                            BufId::X,
                            0,
                            BufId::X,
                            r0 * n_embd,
                            bt * n_embd,
                        );
                        be().copy_range(
                            BufId::Q,
                            0,
                            BufId::Q,
                            r0 * n_head * hd,
                            bt * n_head * hd,
                        );
                        sp += r0;
                        b = bt;
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
                    // LFM2.5 carries no sliding window. A file that set one would plan
                    // Attention::Window, and this arm would not match it.
                    0,
                    b,
                    scores_needed(sp, b, 0),
                    ring,
                );
                if KvType::v() != KvType::F16 {
                    be().hadamard(
                        BufId::Attn,
                        b * qw,
                        had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?,
                    );
                }
                if li == gpu_probe_layer() {
                    gprobe("attention_full", BufId::Attn, 0, (b * qw) as usize);
                    gprobe(
                        "attention_last",
                        BufId::Attn,
                        u64::from(b - 1) * u64::from(qw),
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
                MixerW::ShortConv {
                    conv,
                    in_proj,
                    out_proj,
                },
                crate::Attention::Recurrent { .. },
            ) => {
                let kern = u32::try_from(conv.w.len() / n_embd as usize)
                    .map_err(|_| "conv kernel width")?;
                be().matmat(
                    wkind(in_proj),
                    in_proj.offset as u64,
                    n_embd,
                    3 * n_embd,
                    BufId::Cur,
                    BCX,
                    b,
                );
                if li == gpu_probe_layer() {
                    gprobe("conv.in_proj", BCX, 0, (3 * n_embd) as usize);
                    gprobe("conv.in_proj_full", BCX, 0, (b * 3 * n_embd) as usize);
                    gprobe(
                        "conv.in_proj_last",
                        BCX,
                        u64::from(b - 1) * u64::from(3 * n_embd),
                        (3 * n_embd) as usize,
                    );
                }
                // A checkpoint boundary inside this batch: write the state as of that
                // position aside BEFORE the advance overwrites the pre-batch history,
                // which the answer needs whenever the boundary is nearer than `kern - 1`
                // tokens in. One dispatch of `n_embd` threads; the batch runs on.
                if let Some(k) = wf.state.recur_snap {
                    be().causal_conv_snapshot(
                        imparo_backend::ConvForm::GatedBcx,
                        BCX,
                        BufId::Recur,
                        r_off + plane_in,
                        BufId::RecurSnap,
                        r_off,
                        n_embd,
                        kern,
                        k,
                    );
                }
                // ATTN is the scratch: on a recurrent layer nothing attends, and it is
                // sized n_head * head_dim = 2048, exactly the model width.
                be().causal_conv(
                    imparo_backend::ConvForm::GatedBcx,
                    BCX,
                    conv.offset,
                    BufId::Recur,
                    r_off + plane_in,
                    r_off + plane_out,
                    BufId::Attn,
                    n_embd,
                    kern,
                    b,
                );
                // ShortConv advances the recurrent state. On a non-final Prefill
                // chunk, the last layer's out projection and FFN are dead once that
                // state write is complete.
                if li + 1 == n_layers_total && !wf.state.logits_wanted {
                    break;
                }
                if li + 1 == n_layers_total {
                    if let Some((r0, bt)) = row_local_tail {
                        be().copy_range(
                            BufId::X,
                            0,
                            BufId::X,
                            r0 * n_embd,
                            bt * n_embd,
                        );
                        be().copy_range(
                            BufId::Attn,
                            0,
                            BufId::Attn,
                            r0 * n_embd,
                            bt * n_embd,
                        );
                        be().copy_range(
                            BufId::Xh,
                            0,
                            BufId::Xh,
                            r0 * n_embd / 2,
                            bt * n_embd / 2,
                        );
                        b = bt;
                    }
                }
                if li == gpu_probe_layer() {
                    gprobe("conv.conv", BufId::Attn, 0, n_embd as usize);
                    gprobe("conv.gated_full", BufId::Attn, 0, (b * n_embd) as usize);
                    gprobe("conv.conv_full", BufId::Attn, 0, (b * n_embd) as usize);
                    gprobe(
                        "conv.conv_last",
                        BufId::Attn,
                        u64::from(b - 1) * u64::from(n_embd),
                        n_embd as usize,
                    );
                }
                be().matmat(
                    wkind(out_proj),
                    out_proj.offset as u64,
                    n_embd,
                    n_embd,
                    BufId::Attn,
                    BufId::O,
                    b,
                );
            }
            // The plan and the resolved weights come from the same file, so a mismatch
            // means the plan builder and `prepare` disagree -- a defect, not a bad file.
            (m, a) => {
                return Err(format!(
                    "lfm2 layer {li}: plan says {a:?} but the weights resolved as {}",
                    match m {
                        MixerW::Attention { .. } => "attention",
                        MixerW::ShortConv { .. } => "a short convolution",
                    }
                ));
            }
        }
        if li == gpu_probe_layer() {
            gprobe("mixer_proj_full", BufId::O, 0, (b * n_embd) as usize);
            gprobe(
                "mixer_proj_last",
                BufId::O,
                u64::from(b - 1) * u64::from(n_embd),
                n_embd as usize,
            );
        }
        // The tail as one persistent dispatch (Metal, decode): residual add + ffn norm, the
        // gated FFN, residual add, and the next layer's operator norm. When it ran (with the
        // mixer above, or alone here), every step below is already done.
        let mega_tail_done =
            layer_in_block || mega_entry(imparo_backend::Lfm2MegaMixer::None);
        // Ordinary residual+norm fusion does not require a private projection layout.
        // Each backend admits the operation itself; false preserves the split path.
        let mixer_ffn_norm_fused = mega_tail_done
            || li != gpu_probe_layer()
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
            gprobe("mixer_out", BufId::X, 0, n_embd as usize);
            gprobe(
                "mixer_out_last",
                BufId::X,
                u64::from(b - 1) * u64::from(n_embd),
                n_embd as usize,
            );
        }

        // ---- SwiGLU feed-forward, identical on both block kinds ---------------------
        if !mixer_ffn_norm_fused {
            if projection_preparation && li != gpu_probe_layer() {
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
        if li == gpu_probe_layer() {
            gprobe(
                "ffn_norm_last",
                BufId::Cur,
                u64::from(b - 1) * u64::from(n_embd),
                n_embd as usize,
            );
        }
        // A backend may own the complete transaction and keep the gated activation
        // private. Probes retain the materialized boundary, and every backend that
        // does not implement this capability returns false without writing buffers.
        let fused_ffn = mega_tail_done
            || li != gpu_probe_layer()
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
            // THE GATED PAIR first: one dispatch computes act(gate @ Cur) * (up @ Cur) into G
            // (imparo.metal, "THE GATED PAIR"). The backend takes it only where it has that
            // kernel -- on Metal the prefill half-activation GEMM -- and `false` promises it
            // wrote nothing, so the two projections below then run exactly as before.
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
                if li == gpu_probe_layer() {
                    gprobe(
                        "ffn_gate_last",
                        BufId::G,
                        u64::from(b - 1) * u64::from(n_ff),
                        n_ff as usize,
                    );
                }
                // Fuse at prefill only: at one token the epilogue is a read-modify-write per
                // output row inside the GEMV, where the standalone pass is wide and vectorised.
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
                if !fused && li == gpu_probe_layer() {
                    gprobe(
                        "ffn_up_last",
                        BufId::U,
                        u64::from(b - 1) * u64::from(n_ff),
                        n_ff as usize,
                    );
                }
                if fused {
                    be().set_epilogue(imparo_backend::Epilogue::None);
                } else {
                    be().act_mul(BufId::G, BufId::U, b * n_ff);
                }
            }
            if li == gpu_probe_layer() {
                gprobe(
                    "ffn_swiglu_last",
                    BufId::G,
                    u64::from(b - 1) * u64::from(n_ff),
                    n_ff as usize,
                );
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
            if li == gpu_probe_layer() {
                gprobe(
                    "ffn_down_last",
                    BufId::O,
                    u64::from(b - 1) * u64::from(n_embd),
                    n_embd as usize,
                );
            }
        }
        // A block tail leaves the next layer's operator norm to that layer (its block forms
        // its own; the dispatch path norms at its top).
        let next_operator_norm_fused = if mega_tail_done {
            false
        } else if li + 1 < n_layers_total
            && li != gpu_probe_layer()
            && li + 1 != gpu_probe_layer()
        {
            be().add_rms_norm(
                BufId::Cur,
                BufId::X,
                BufId::O,
                wf.w.layers[li + 1].op_norm.offset,
                n_embd,
                eps,
                b,
                n_embd,
                0,
            )
        } else {
            false
        };
        if !mega_tail_done && !next_operator_norm_fused {
            be().add(BufId::X, BufId::O, b * n_embd);
        }
        operator_norm_ready = next_operator_norm_fused;
        if li == gpu_probe_layer() {
            gprobe("l_out", BufId::X, 0, n_embd as usize);
            gprobe("l_out_full", BufId::X, 0, (b * n_embd) as usize);
            // The LAST token too: the logits come from it, and an error that only
            // affects later tokens is invisible at token 0 -- which is exactly how this
            // graph looked correct at n=1 and wrong at n=4.
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
    // The mega program (task #153): the layers recorded into one run are encoded here, before
    // anything after the loop reads their output.
    be().mega_program_end();

    // ---- last token only: the final norm and the tied lm_head, same command buffer --
    if !wf.state.logits_wanted {
        return be()
            .end()
            .map_err(|rc| format!("lfm2 forward failed rc={rc}"));
    }
    let prefill_final_projection_preparation = prefill_projection_preparation;
    let (lm_head_src, lm_head_src_row) = if prefill_final_projection_preparation {
        be().rms_norm_projection(
            BufId::Cur,
            BufId::X,
            wf.w.output_norm.offset,
            n_embd,
            eps,
            1,
            n_embd,
            (b - 1) * n_embd,
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
            1,
            n_embd,
            (b - 1) * n_embd,
        );
        (BufId::X, b - 1)
    };
    be().matmat_from(
        embd_kind,
        embd_off,
        n_embd,
        c.vocab_size,
        lm_head_src,
        BufId::Logits,
        1,
        lm_head_src_row,
    );
    if let Some(cap) = wf.plan.output.logit_softcap {
        be().softcap(BufId::Logits, cap, c.vocab_size);
    }
    if let Some(p) = pipe {
        // The pick feeds the next step's gather and the host's slot; the region is
        // committed without waiting so the next step can be encoded behind it.
        be().argmax_feed(
            BufId::Logits,
            BufId::Tokens,
            BufId::Pick,
            p.pick_slot,
            c.vocab_size,
        );
        return be()
            .end_async()
            .map_err(|rc| format!("lfm2 pipelined step failed rc={rc}"));
    }
    if argmax {
        be().argmax(BufId::Logits, BufId::Tmp, c.vocab_size);
    }
    be().end()
        .map_err(|rc| format!("lfm2 forward failed rc={rc}"))?;

    if argmax {
        out.resize(1, 0.0);
        be().read(BufId::Tmp, 0, out);
    } else {
        out.resize(c.vocab_size as usize, 0.0);
        be().read(BufId::Logits, 0, out);
    }
    Ok(())
}

fn should_fuse_epilogue(n_tok: u32, backend_supports_activation: bool) -> bool {
    if !fuse_epilogue_enabled() {
        return false;
    }
    n_tok > 1 && backend_supports_activation
}

/// IMPARO_FUSE_EPILOGUE=0 issues the two projections and a separate `act_mul` instead,
/// which is what llama.cpp does (two `mul_mat`s and a GLU op).
///
/// The two are not the trade they look like. Fusing moves FEWER bytes -- the up
/// projection reads the gate output and writes once, where the split path writes the up
/// output, then reads both and writes again -- but on the Metal Q8 GEMM it forces the
/// MASKED write-back for every one of those dispatches, because the predicate-free route
/// requires `epilogue == 0`. That path spills the accumulators through threadgroup
/// memory, barriers, and finishes with a SCALAR per-element read-modify-write where the
/// unfused route ends in a vector `simdgroup_store`. Which side wins is a measurement,
/// and it had never been made on this model.
fn fuse_epilogue_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_FUSE_EPILOGUE").map_or(true, |v| v != "0"))
}

#[cfg(test)]
mod tests {
    use super::{BCX, buffer_requirements};
    use crate::gpu_support::Placement;
    use crate::{
        Activation, Attention, EmbedPlan, Ffn, KvSource, LayerPlan, ModelConfig,
        ModelPlan, OutputPlan,
    };
    use imparo_backend::BufId;

    /// LFM2.5-2.6B's real geometry, small enough to write down: 30 blocks, 8 attention at
    /// [2,5,9,13,17,21,24,27] and 22 recurrent, n_embd 2048, head_dim 64, l_cache 3.
    fn plan() -> ModelPlan {
        let attention_at = [2u32, 5, 9, 13, 17, 21, 24, 27];
        let layers = (0..30u32)
            .map(|index| LayerPlan {
                index,
                attention: if attention_at.contains(&index) {
                    Attention::Full {
                        head_dim: 64,
                        rope_base: 1e7,
                        rope_dim: 64,
                    }
                } else {
                    Attention::Recurrent {
                        r_elems: 2048 * (3 - 1),
                        s_elems: 0,
                        // LFM2's short conv carries no delta-rule state, so it has no
                        // key/value geometry: the recurrent dims are the delta net's.
                        key_dim: 0,
                        value_dim: 0,
                    }
                },
                ffn: Ffn::Dense {
                    activation: Activation::Silu,
                    hidden: 10752,
                },
                kv_source: KvSource::Own,
            })
            .collect();
        ModelPlan {
            config: ModelConfig {
                architecture: "lfm2".into(),
                n_layers: 30,
                n_embd: 2048,
                n_ff: 10752,
                n_heads: 32,
                n_kv_heads: 8,
                context_length: 128_000,
                vocab_size: 128_000,
                norm_eps: 1e-5,
            },
            embed: EmbedPlan {
                scale_by_sqrt_embd: false,
                per_layer_dim: None,
                per_layer_row_bytes: None,
            },
            kv_storage_basis: crate::KvStorageBasisPolicy::BackendOverrideOrCanonical,
            weight_residency: crate::WeightResidencyPlan::default(),
            layers,
            output: OutputPlan {
                final_norm: true,
                logit_softcap: None,
                tied_embeddings: true,
            },
        }
    }

    fn find(
        reqs: &[super::BufferRequirement],
        id: BufId,
    ) -> Option<super::BufferRequirement> {
        reqs.iter().find(|r| r.id as u32 == id as u32).copied()
    }

    /// The ShortConv projection is THREE times the model width, and it is the buffer that
    /// had no name in the shared enum before the model-private slots were made generic.
    #[test]
    fn the_shortconv_projection_is_three_widths_and_has_a_slot() {
        let b = 128;
        let reqs = buffer_requirements(&plan(), b, 8192);
        let bcx = find(&reqs, BCX).expect("no bcx buffer");
        assert_eq!(bcx.bytes, (b * 3 * 2048 * 4) as u64);
        // Live exactly where Q/K/V would be: a block does one or the other, never both.
        assert_eq!(bcx.placement, Placement::Group(0));
    }

    /// The half-activation mirrors must be declared, or every gate on that path reads a
    /// nil buffer and the model silently converts f32 to half inline in every
    /// threadgroup -- correct, slower, and invisible. LFM2 shipped exactly that way.
    #[test]
    fn the_half_activation_mirrors_are_declared() {
        // 455, the short leg's real batch, NOT a multiple of the token tile. At 128 this
        // test passed whatever the padding did, which is why it did not catch the
        // conversion pass writing 537 KB past the mirror.
        let b = 455;
        let reqs = buffer_requirements(&plan(), b, 8192);
        let padded = b.next_multiple_of(imparo_backend::MAX_GEMM_TOKEN_TILE);
        assert_eq!(
            padded, 512,
            "455 tokens must round up to a whole token tile"
        );
        for (slot, id) in [BufId::Xh, BufId::Xh2].into_iter().enumerate() {
            let m = find(&reqs, id).expect("no half-activation mirror");
            // n_ff, the widest activation staged through it: the down projection reads it,
            // and the rows are padded so a GEMM may walk whole token tiles off the end.
            assert_eq!(m.bytes, (padded * 10752 * 2) as u64);
            // Inside U's pages, which are dead at prefill because the up epilogue is fused.
            assert_eq!(
                m.placement,
                Placement::Within {
                    host: BufId::U,
                    slot: slot as u8,
                }
            );
        }
    }

    /// gemma4's per-layer-embedding buffers must NOT appear. They shared the model-private
    /// slots this model now uses, so asking for them would silently collide with bcx.
    #[test]
    fn none_of_gemma4s_private_buffers_are_requested() {
        let reqs = buffer_requirements(&plan(), 128, 8192);
        for id in [BufId::Model1, BufId::Model2] {
            assert!(find(&reqs, id).is_none(), "{id:?} is not lfm2's");
        }
    }

    /// Attention is sized from head_dim 64, not from the model width: Q is 32 heads x 64
    /// and K/V are 8 x 64, so the whole attention group is far narrower than gemma4's.
    #[test]
    fn attention_buffers_follow_the_head_dim_not_the_width() {
        let b = 128;
        let reqs = buffer_requirements(&plan(), b, 8192);
        assert_eq!(
            find(&reqs, BufId::Q).unwrap().bytes,
            (b * 32 * 64 * 4) as u64
        );
        assert_eq!(
            find(&reqs, BufId::K).unwrap().bytes,
            (b * 8 * 64 * 4) as u64
        );
        assert_eq!(
            find(&reqs, BufId::V).unwrap().bytes,
            (b * 8 * 64 * 4) as u64
        );
        // ATTN is dedicated for the reason gemma4 measured: the register-tiled GEMM reads
        // past the token count and what those rows hold moves the logits.
        assert_eq!(
            find(&reqs, BufId::Attn).unwrap().placement,
            Placement::Dedicated
        );
    }

    /// Every id requested must be inside the table the backends size themselves to.
    #[test]
    fn every_requested_buffer_is_addressable() {
        for r in buffer_requirements(&plan(), 128, 8192) {
            assert!(
                (r.id as usize) < BufId::COUNT,
                "{:?} is outside BufId::COUNT",
                r.id
            );
        }
    }

    #[test]
    fn lfm2_fuses_only_when_the_backend_supports_silu() {
        assert!(!super::should_fuse_epilogue(1, true));
        assert!(!super::should_fuse_epilogue(128, false));
        assert!(super::should_fuse_epilogue(128, true));
    }

    #[test]
    fn quantized_kv_scratch_is_a_shared_capacity_contract() {
        let plan = plan();
        let both = crate::gpu_support::kv_dequant_scratch_requirements(
            &plan, 8192, true, true,
        );
        let expected = 8192_u64 * 8 * 64 * 2;
        assert_eq!(find(&both, BufId::Kdq).unwrap().bytes, expected);
        assert_eq!(find(&both, BufId::Vdq).unwrap().bytes, expected);
        assert_eq!(
            crate::gpu_support::kv_dequant_scratch_requirements(
                &plan, 8192, false, false,
            )
            .len(),
            0
        );
    }
}
