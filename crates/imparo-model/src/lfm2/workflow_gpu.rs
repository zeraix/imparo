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
    BufferRequirement, Placement, be, fuse_epilogue_enabled, gprobe, gpu_probe_layer,
    half_activation_mirror_requirements, kv_dequant_scratch_requirements, kvq_mask_on,
    layer_skip_log, scores_needed, should_fuse_epilogue, tail_align, tail_split,
    tail_split_min_rows, trace_rows,
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

#[doc(hidden)]
pub use crate::window_history::enable_lfm_full_history_control;

/// Admit optional device-owned layouts for LFM2 Down projections. The model
/// supplies resolved offsets and shapes only; each backend decides whether the
/// format is useful and unsupported backends preserve their established path.
pub fn prepare_device(wf: &mut Lfm2) -> Result<(), String> {
    // Explicit target preparation only: neither a process-wide native default
    // nor a verification guard may change the independently owned DSpark draft.
    if crate::lfm_retained_domain() != 0
        || std::env::var("IMPARO_LAB_BATCH_INVARIANT_Q8_V1").as_deref() == Ok("1")
    {
        if std::env::var("IMPARO_LAB_VERIFY_M1_QUANT").as_deref() == Ok("1") {
            return Err(
                "BatchInvariantQ8V1 conflicts with the M1-quant diagnostic".into()
            );
        }
        let admitted = be().prepare_batch_invariant_q8_v1().map_err(|rc| {
            format!("target BatchInvariantQ8V1 preparation failed rc={rc}")
        })?;
        if !admitted {
            return Err(
                "target BatchInvariantQ8V1 is unsupported by this backend".into()
            );
        }
    }
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

/// Rows needed before a suffix of finite-history mixers. Intermediate prefix
/// outputs inside the retained suffix may be dead; every observable state and
/// final row must lie beyond the accumulated history. An interior checkpoint
/// expands the same contiguous suffix backwards rather than joining disjoint
/// windows or treating unrelated rows as adjacent tokens.
fn finite_history_tail(
    plan: &ModelPlan,
    batch: u32,
    logits: bool,
    checkpoint: Option<u32>,
    minimum: u32,
    align: u32,
) -> Option<(usize, u32, u32)> {
    if batch < 128 || plan.config.n_embd == 0 {
        return None;
    }
    let last_attention = plan
        .layers
        .iter()
        .rposition(|l| !matches!(l.attention, crate::Attention::Recurrent { .. }))?;
    if last_attention + 1 == plan.layers.len() {
        return None;
    }
    let mut history = 0_u32;
    for layer in &plan.layers[last_attention + 1..] {
        let crate::Attention::Recurrent {
            r_elems,
            s_elems: 0,
            key_dim: 0,
            value_dim: 0,
        } = layer.attention
        else {
            return None;
        };
        if r_elems == 0 || r_elems % plan.config.n_embd != 0 {
            return None;
        }
        history = history.checked_add(r_elems / plan.config.n_embd)?;
    }
    let mut needed = history.checked_add(u32::from(logits))?.max(minimum);
    if let Some(k) = checkpoint {
        if k == 0 || k > batch {
            return None;
        }
        needed = needed.max(batch.checked_sub(k)?.checked_add(history)?);
    }
    let (start, rows) = tail_split_min_rows(batch, align, needed)?;
    // Backend copy_range is not a memmove. No zero-benefit or overlapping move.
    (start >= rows && start > 0).then_some((last_attention, start, rows))
}

#[cfg(feature = "cuda-speculative")]
fn tree_graph_eligible(wf: &Lfm2, rows: u32, argmax: bool) -> bool {
    // Match backend::active's static-CUDA composition, including its CPU opt-out.
    // A feature flag or host_forward alone does not establish backend identity.
    if rows != 16
        || argmax
        || wf.state.output_demand != crate::OutputDemand::RowArgmax
        || wf.state.host_forward
        || !crate::backend::gpu_requested_from_env()
        || std::env::var("IMPARO_BACKEND").is_ok_and(|v| v == "cpu")
        || wf.state.pipe.is_some()
        || !wf.state.queued.is_empty()
        || wf.state.recur_planes != 1
        || wf.state.recur_plane != 0
        || wf.state.recur_plane_next != 0
        || wf.state.recur_snap.is_some()
        || wf.state.verification_prefix_tokens != 0
        || std::env::var_os("IMPARO_GPU_PROBE").is_some()
        || gpu_probe_layer() != usize::MAX
        || std::env::var_os("IMPARO_CUDA_PROFILE_FORWARD").is_some()
    {
        return false;
    }
    // Native replay republishes only DSpark's host feature metadata. An arbitrary
    // observer cannot be skipped, even if a DSpark session also happens to exist.
    wf.state.layer_outputs.as_ref().is_some_and(|capture| {
        capture.attached()
            && capture.active()
            && !capture.layers.is_empty()
            && capture.layers.windows(2).all(|pair| pair[0] < pair[1])
            && capture
                .layers
                .last()
                .is_some_and(|&layer| (layer as usize) < wf.plan.layers.len())
            && capture.capture.is_native_dspark()
    })
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
    let full_history_control =
        crate::window_history::lfm_full_history_control_enabled();
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
    let layer_outputs = wf
        .state
        .layer_outputs
        .as_ref()
        .is_some_and(crate::layer_outputs::LayerOutputCapture::active);
    // This configured target Prefill policy fixes the crop boundary independently
    // of whether a feature subscriber is active. Otherwise attaching a drafter
    // changes the target's attention/FFN shapes before verification even begins.
    let post_layer_tail = b > 1
        && (crate::lfm_retained_domain() != 0
            || std::env::var("IMPARO_LAB_CAPTURE_AWARE_PREFILL_TAIL").as_deref()
                == Ok("1"));
    let row_argmax = wf.state.output_demand == crate::OutputDemand::RowArgmax;
    let greedy_return =
        wf.state.output_demand == crate::OutputDemand::GreedyVerification;
    let all_logits = wf.state.output_demand.requires_all_positions();
    if greedy_return
        && (!be().supports_greedy_verification()
            || wf.state.verification_prefix_tokens != tokens.len()
            || argmax)
    {
        return Err(
            "device greedy verification requires admitted complete prefix state".into(),
        );
    }
    if row_argmax && !be().supports_argmax_rows() {
        return Err("row argmax is not served by this backend".into());
    }
    // A ROW-LAYOUT BATCH (a tree): positions, visibility and conv ancestors come from
    // BufId::RowLayout. It keeps every row, reads an f16 cache, and writes each row's
    // recurrent state aside instead of advancing the live one.
    let row_layout = wf.state.row_layout;
    if row_layout != 0
        && (row_layout != b
            || !all_logits
            || KvType::k() != KvType::F16
            || KvType::v() != KvType::F16
            || wf.state.verification_prefix_tokens != 0
            || wf.state.recur_snap.is_some())
    {
        return Err(
            "a row-layout batch keeps every row, reads an f16 cache and takes no other snapshot"
                .into(),
        );
    }
    let last_logits = wf.state.output_demand == crate::OutputDemand::LastToken;
    if all_logits && argmax {
        return Err("all-position logits cannot request scalar argmax".into());
    }
    let verification_prefix_tokens = wf.state.verification_prefix_tokens;
    let verification_recurrent_elems = if verification_prefix_tokens != 0 {
        if verification_prefix_tokens != tokens.len() || !all_logits {
            return Err(
                "verification prefix snapshots require the complete all-logits batch"
                    .into(),
            );
        }
        let elems = wf.plan.recurrent_elems() as u64;
        // Snapshot offsets use u32 float elements. Check before encoding any work.
        elems
            .checked_mul(u64::from(b) + 1)
            .filter(|&elems| u32::try_from(elems).is_ok())
            .ok_or_else(|| {
                "verification prefix snapshot range exceeds u32".to_string()
            })?;
        elems
    } else {
        0
    };
    let sp = u32::try_from(start_pos).map_err(|_| "start_pos too large")?;
    let recur = wf.plan.recurrent_layout();
    // A tree forward keeps every node's input to each convolution window in RowInputs, where the
    // commit reads them back.
    let row_inputs = if row_layout != 0 {
        Some(crate::verification::RowInputLayout::of(
            &wf.plan,
            &super::plan::conv_windows(&wf.plan),
        )?)
    } else {
        None
    };
    let act = wf.plan.layers[0].ffn.activation().epilogue();
    // Activation aliases and scratch ranges are planned for the live batch width.
    // Recurrent state and KV are allocated separately and survive this resize.
    // Without the fit, a resumed LFM2 forward kept the initial 512-token arena for
    // narrower chunks, violating the same per-batch contract Gemma4 uses.
    wf.gpu_fit_batch(tokens.len())?;
    let decode = b == 1;
    // A replayable single-plane step keeps checkpoint writes outside its graph.
    // Capturing an armed/unarmed step must not freeze that request's snapshot flag.
    let external_decode_snapshot =
        decode && wf.state.recur_planes == 1 && wf.state.pipe.is_none();
    let decode_projection_preparation =
        decode && be().use_decode_projection_preparation();
    let prefill_projection_preparation =
        !decode && be().use_prefill_projection_preparation();
    let projection_preparation =
        decode_projection_preparation || prefill_projection_preparation;
    let replayed = if decode {
        // Active observers need each layer callback and fresh host capture metadata.
        // CUDA's prepare argmax bit also gates both graph replay and new capture;
        // the actual output argmax below remains unchanged. Keep decode preparation
        // and arithmetic while preventing graphs from retaining observer-owned buffers.
        be().decode_prepare(tokens[0], sp, argmax && !layer_outputs)
            .map_err(|rc| format!("LFM2 GPU decode prepare failed rc={rc}"))?
    } else {
        false
    };
    if replayed {
        be().end()
            .map_err(|rc| format!("LFM2 GPU forward failed rc={rc}"))?;
        finish_decode_snapshot(wf, external_decode_snapshot)?;
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
    // The native tree verifier reads this only after a successful forward.
    // Ordinary is the path actually entered below; graph eligibility alone never
    // labels a replay. General row-layout verification leaves its result Unknown.
    #[cfg(feature = "cuda-speculative")]
    if row_argmax && row_layout == 0 && b > 1 {
        wf.state.tree_submission = crate::speculative::TreeSubmission::Ordinary;
    }
    #[cfg(feature = "cuda-speculative")]
    let (tree_graph_capture, tree_graph_flush) = if tree_graph_eligible(wf, b, argmax) {
        // Preserve bounded-flush bookkeeping on both ordinary and replay paths.
        // Static CUDA's flush is a no-op; a nonzero layer seat is still eligible.
        let flush = crate::gpu_support::flush_layers_bounded(b, wf.plan.layers.len());
        match unsafe { imparo_cuda::tree::graph_prepare()? } {
            imparo_cuda::tree::TreeGraphSubmission::Ordinary => (None, Some(flush)),
            imparo_cuda::tree::TreeGraphSubmission::Capture(capture) => {
                wf.state.tree_submission = crate::speculative::TreeSubmission::Capture;
                (Some(capture), Some(flush))
            }
            imparo_cuda::tree::TreeGraphSubmission::Replayed => {
                wf.state.tree_submission = crate::speculative::TreeSubmission::Replayed;
                be().end()
                    .map_err(|rc| format!("lfm2 tree graph replay failed rc={rc}"))?;
                crate::gpu_support::prefill_region_ended();
                out.resize(16, 0.0);
                be().read(BufId::Tmp, 0, out);
                return Ok(());
            }
        }
    } else {
        (None, None)
    };
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
    #[cfg(feature = "cuda-speculative")]
    let flush_every = tree_graph_flush
        .unwrap_or_else(|| crate::gpu_support::flush_layers_bounded(b, n_layers_total));
    #[cfg(not(feature = "cuda-speculative"))]
    let flush_every = crate::gpu_support::flush_layers_bounded(b, n_layers_total);
    let tail = if last_logits && !layer_outputs && !post_layer_tail && b >= 128 {
        tail_split(b, tail_align())
    } else {
        None
    };
    let mut row_local_tail =
        if last_logits && !layer_outputs && !post_layer_tail && b > 1 {
            tail_split_min_rows(b, 1, be().row_local_prefill_tail_rows(b))
        } else {
            None
        };
    let proposed_history_tail = if !full_history_control
        && !all_logits
        && (!layer_outputs || post_layer_tail)
        && gpu_probe_layer() == usize::MAX
    {
        be().finite_history_prefill_tail_rows().and_then(|minimum| {
            finite_history_tail(
                &wf.plan,
                b,
                wf.state.output_demand.wants_logits(),
                wf.state.recur_snap,
                minimum,
                tail_align(),
            )
        })
    } else {
        None
    };
    // Dense feature subscribers need the full cut-layer residual. The selected
    // target policy uses that same boundary when no subscriber is active.
    let deferred_history_tail = if post_layer_tail {
        proposed_history_tail.filter(|&(li, _, _)| {
            !wf.state
                .layer_outputs
                .as_ref()
                .is_some_and(|c| c.observes_after(li as u32))
        })
    } else {
        None
    };
    let history_tail = if layer_outputs || post_layer_tail {
        None
    } else {
        proposed_history_tail
    };
    let mut recurrent_snapshot = if external_decode_snapshot {
        None
    } else {
        wf.state.recur_snap
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
                && !layer_outputs
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
                let snap = recurrent_snapshot.map(|_| (BufId::RecurSnap, r_off));
                (wf.state.output_demand.wants_logits() || li + 1 != n_layers_total)
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
                (wf.state.output_demand.wants_logits() || li + 1 != n_layers_total)
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
                if row_layout != 0 {
                    // Each row ropes at its own layout position. The cache is f16 (checked
                    // above), so there is no rotation; K is done here with Q.
                    for (buf, norm, heads) in [
                        (BufId::Q, q_norm.offset, n_head),
                        (BufId::K, k_norm.offset, n_kv),
                    ] {
                        if !be().head_norm_rope_rows(
                            buf, norm, hd, eps, heads, b, rope_dim, rope_base, None,
                        ) {
                            return Err(format!(
                                "row-layout head norm and rope not served at layer {li}"
                            ));
                        }
                    }
                } else if li == gpu_probe_layer() {
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
                }
                let v_hadamard_nrot = if KvType::v() == KvType::F16 {
                    0
                } else {
                    had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?
                };
                let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
                let direct_kv = li != gpu_probe_layer()
                    && KvType::k() == KvType::Q8_0
                    && KvType::v() == KvType::Q8_0
                    && be().kv_head_postprocess_store(
                        BufId::K,
                        BufId::V,
                        k_norm.offset,
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
                    if li != gpu_probe_layer() && row_layout == 0 {
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
                    if v_hadamard_nrot != 0 {
                        be().hadamard(BufId::V, b * kw, v_hadamard_nrot);
                    }
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
                if !direct_kv {
                    be().kv_store(BufId::K, li as u32, kw, sp, b, false, ring);
                    be().kv_store(BufId::V, li as u32, kw, sp, b, true, ring);
                }
                if let Some((cut_layer, r0, bt)) = history_tail {
                    if li == cut_layer {
                        // All attention KV rows are already durable. The later
                        // finite-history states observe only the proven suffix.
                        be().copy_range(
                            BufId::X,
                            0,
                            BufId::X,
                            r0 * n_embd,
                            bt * n_embd,
                        );
                        be().copy_range(BufId::Q, 0, BufId::Q, r0 * qw, bt * qw);
                        recurrent_snapshot = recurrent_snapshot.map(|k| k - r0);
                        if wf.state.output_demand.wants_logits() {
                            row_local_tail = tail_split_min_rows(
                                bt,
                                1,
                                be().row_local_prefill_tail_rows(b),
                            )
                            .filter(|&(start, rows)| start >= rows && start > 0);
                        }
                        if layer_skip_log() {
                            eprintln!(
                                "[imparo] finite-history tail after KV layer {li}: rows {r0}..{b}, retained={bt}, logits_wanted={}, snapshot={recurrent_snapshot:?}",
                                wf.state.output_demand.wants_logits()
                            );
                        }
                        sp += r0;
                        b = bt;
                    }
                }
                // The final layer's KV stores are this non-final chunk's last
                // externally visible state writes. Its remaining attention and FFN
                // feed only logits which the chunk loop discards.
                if li + 1 == n_layers_total && !wf.state.output_demand.wants_logits() {
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
                        return Err(format!(
                            "row-layout attention not served at layer {li}"
                        ));
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
                        // LFM2.5 carries no sliding window. A file that set one would plan
                        // Attention::Window, and this arm would not match it.
                        0,
                        b,
                        scores_needed(sp, b, 0),
                        ring,
                    );
                }
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
                let projected_shortconv = plane_in == plane_out
                    && row_layout == 0
                    && recurrent_snapshot.is_none()
                    && verification_prefix_tokens == 0
                    && std::env::var_os("IMPARO_GPU_PROBE").is_none()
                    && be().matmat_shortconv(
                        wkind(in_proj),
                        in_proj.offset as u64,
                        conv.offset,
                        BufId::Cur,
                        BCX,
                        BufId::Recur,
                        r_off + plane_in,
                        BufId::Attn,
                        n_embd,
                        kern,
                        b,
                    );
                if !projected_shortconv {
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
                    }
                    if row_layout != 0 {
                        // Each row reads its own path and the live state is only read. Each
                        // row's input to the window goes aside to RowInputs for the commit.
                        let inputs_off = row_inputs
                            .as_ref()
                            .and_then(|layout| layout.input_off(li))
                            .ok_or_else(|| {
                                format!("layer {li} has no row-input window")
                            })?;
                        let row_elems =
                            row_inputs.as_ref().map_or(0, |layout| layout.row_elems);
                        if !be().causal_conv_row_inputs(
                            imparo_backend::ConvForm::GatedBcx,
                            BCX,
                            BufId::RowInputs,
                            inputs_off,
                            row_elems,
                            n_embd,
                            b,
                        ) || !be().causal_conv_rows(
                            imparo_backend::ConvForm::GatedBcx,
                            BCX,
                            conv.offset,
                            BufId::Recur,
                            r_off + plane_in,
                            BufId::Attn,
                            n_embd,
                            kern,
                            b,
                        ) {
                            return Err(format!(
                                "row-layout short convolution not served at layer {li}"
                            ));
                        }
                    } else {
                        if let Some(k) = recurrent_snapshot {
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
                        if verification_prefix_tokens != 0 {
                            for k in 1..b {
                                let snap_off = ((u64::from(k) + 1)
                                    * verification_recurrent_elems
                                    + u64::from(r_off))
                                    as u32;
                                be().causal_conv_snapshot(
                                    imparo_backend::ConvForm::GatedBcx,
                                    BCX,
                                    BufId::Recur,
                                    r_off + plane_in,
                                    BufId::RecurSnap,
                                    snap_off,
                                    n_embd,
                                    kern,
                                    k,
                                );
                            }
                        }
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
                    }
                }
                // ShortConv advances the recurrent state. On a non-final Prefill
                // chunk, the last layer's out projection and FFN are dead once that
                // state write is complete.
                if li + 1 == n_layers_total && !wf.state.output_demand.wants_logits() {
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
        if let Some(capture) = &wf.state.layer_outputs {
            capture.record(li as u32, sp, b, BufId::X)?;
        }
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
        if let Some((cut_layer, r0, retained)) = deferred_history_tail {
            if li == cut_layer {
                // record() above has published every required dense row. Only
                // suffix X remains live; recompute the next norm at the new width
                // instead of reusing its full-width prepared projection metadata.
                be().copy_range(BufId::X, 0, BufId::X, r0 * n_embd, retained * n_embd);
                recurrent_snapshot = recurrent_snapshot.map(|k| k - r0);
                sp += r0;
                b = retained;
                operator_norm_ready = false;
                if layer_skip_log() {
                    eprintln!(
                        "[imparo] post-layer finite-history tail after layer {li}: retained={retained} start={sp}"
                    );
                }
            }
        }
        if flush_every > 0 && (li + 1) % flush_every == 0 && li + 1 < n_layers_total {
            be().flush();
        }
    }
    // The mega program (task #153): the layers recorded into one run are encoded here, before
    // anything after the loop reads their output.
    be().mega_program_end();

    // Final norm/head cover every requested position in the same command buffer.
    if !wf.state.output_demand.wants_logits() {
        be().end()
            .map_err(|rc| format!("lfm2 forward failed rc={rc}"))?;
        crate::gpu_support::prefill_region_ended();
        return finish_decode_snapshot(wf, external_decode_snapshot);
    }
    let output_rows = if all_logits { b } else { 1 };
    let output_start = if all_logits { 0 } else { b - 1 };
    let output_words = output_rows
        .checked_mul(c.vocab_size)
        .ok_or_else(|| "all-position logits count exceeds u32".to_string())?;
    let prefill_final_projection_preparation = prefill_projection_preparation;
    let (lm_head_src, lm_head_src_row) = if prefill_final_projection_preparation {
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
        embd_kind,
        embd_off,
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
    if row_argmax {
        be().argmax_rows(BufId::Logits, BufId::Tmp, c.vocab_size, b);
    }
    #[cfg(feature = "cuda-speculative")]
    if let Some(capture) = tree_graph_capture {
        capture.finish()?;
    }
    if greedy_return {
        be().verify_greedy_and_restore(
            BufId::Logits,
            BufId::Tokens,
            BufId::Tmp,
            BufId::Recur,
            BufId::RecurSnap,
            c.vocab_size,
            b,
            wf.plan.recurrent_elems(),
        )
        .map_err(|rc| format!("lfm2 device greedy verification rc={rc}"))?;
    } else if argmax {
        be().argmax(BufId::Logits, BufId::Tmp, c.vocab_size);
    }
    be().end()
        .map_err(|rc| format!("lfm2 forward failed rc={rc}"))?;
    crate::gpu_support::prefill_region_ended();
    finish_decode_snapshot(wf, external_decode_snapshot)?;

    if row_argmax {
        out.resize(b as usize, 0.0);
        be().read(BufId::Tmp, 0, out);
    } else if greedy_return {
        out.resize(3, 0.0);
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

/// One co-batched decode step (docs/continuous-batching.md): row r decodes `rows[r].token` in
/// slot `rows[r].slot` at `rows[r].pos`. The operations are the one-row dispatch path's, in its
/// order. The projections, norms and FFN take all rows at once, on the step's route
/// (`WorkflowState::row_route`, `Backend::set_decode_rows`): on the exact route every row gets
/// the one-row decode's arithmetic, so its logits are bit-equal to its lone step on the dispatch
/// path; on the fast route the backend picks the kernel for the row count. Each operation that
/// touches one conversation's own state -- head norm and rope at the row's position, the KV
/// store, attention over the row's cache, the convolution step on the row's slot -- is handed
/// every row with its position and slot.
///
/// # Errors
/// For unsupported KV codecs, a backend without decode rows or a per-row operation, or a
/// device failure.
pub fn rows(
    wf: &mut Lfm2,
    rows: &[crate::DecodeRow],
    logits: Option<&mut Vec<f32>>,
    picks: &mut Vec<u32>,
) -> Result<(), String> {
    let codecs = (KvType::k().ggml_id(), KvType::v().ggml_id());
    if !<crate::lfm2::Lfm2Arch as crate::Architecture>::DEVICE_ROWS_KV_CODECS
        .contains(&codecs)
        || !be().supports_decode_rows_kv(codecs.0, codecs.1)
    {
        return Err("co-batched LFM2 decode does not support these KV codecs".into());
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
    crate::gpu_support::submit_decode_rows(be(), wf.state.row_route, || {
        encode_rows(wf, rows, b)
    })?;
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
fn encode_rows(wf: &Lfm2, rows: &[crate::DecodeRow], b: u32) -> Result<(), String> {
    let c = &wf.plan.config;
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
    let mut operator_norm_ready = false;
    for (li, (layer, &(r_off, _, _, _))) in
        wf.plan.layers.iter().zip(recur.iter()).enumerate()
    {
        let lw = &wf.w.layers[li];
        if !operator_norm_ready {
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
        trace_rows("opnorm", li, BufId::Cur, b, n_embd);
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
                let qw = n_head * hd;
                let kw = n_kv * hd;
                let hk = if KvType::k() == KvType::F16 {
                    0
                } else {
                    had_nrot("IMPARO_HAD_K", kv_quant_route.key, hd)?
                };
                let hv = if KvType::v() == KvType::F16 {
                    0
                } else {
                    had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?
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
                let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
                let max_scores: Vec<u32> =
                    pos.iter().map(|&p| scores_needed(p, 1, 0)).collect();
                let heads_served = be().head_norm_rope_hadamard_at(
                    BufId::Q,
                    q_norm.offset,
                    hd,
                    eps,
                    n_head,
                    &pos,
                    rope_dim,
                    rope_base,
                    None,
                    hk,
                ) && be().head_norm_rope_hadamard_at(
                    BufId::K,
                    k_norm.offset,
                    hd,
                    eps,
                    n_kv,
                    &pos,
                    rope_dim,
                    rope_base,
                    None,
                    hk,
                );
                if !heads_served {
                    return Err(format!("co-batched LFM2 head transform not served at layer {li}"));
                }
                // LFM2 rotates V without the V RMS normalization used by Gemma4.
                if hv != 0 {
                    be().hadamard(BufId::V, b * kw, hv);
                }
                let served = be().kv_store_slot_rows(
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
                // Return the weighted values to the ordinary output basis.
                if hv != 0 {
                    be().hadamard(BufId::Attn, b * qw, hv);
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
                trace_rows("in_proj", li, BCX, b, 3 * n_embd);
                let conv_rows: Vec<imparo_backend::SlotStateRow> = rows
                    .iter()
                    .map(|r| imparo_backend::SlotStateRow {
                        slot: r.slot,
                        state_off: r_off + r.plane_in * recur_elems,
                        state_out_off: r_off + r.plane_out * recur_elems,
                    })
                    .collect();
                if !be().causal_conv_slot_rows(
                    imparo_backend::ConvForm::GatedBcx,
                    BCX,
                    conv.offset,
                    BufId::Recur,
                    &conv_rows,
                    BufId::Attn,
                    n_embd,
                    kern,
                ) {
                    return Err(format!(
                        "co-batched convolution not served at layer {li}"
                    ));
                }
                trace_rows("conv", li, BufId::Attn, b, n_embd);
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
            (_, a) => {
                return Err(format!(
                    "lfm2 layer {li}: plan says {a:?} but the weights resolved otherwise"
                ));
            }
        }
        trace_rows("mix", li, BufId::O, b, n_embd);
        if !be().add_rms_norm(
            BufId::Cur,
            BufId::X,
            BufId::O,
            lw.ffn_norm.offset,
            n_embd,
            eps,
            b,
            n_embd,
            0,
        ) {
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
        }
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
        operator_norm_ready = li + 1 < n_layers
            && be().add_rms_norm(
                BufId::Cur,
                BufId::X,
                BufId::O,
                wf.w.layers[li + 1].op_norm.offset,
                n_embd,
                eps,
                b,
                n_embd,
                0,
            );
        if !operator_norm_ready {
            be().add(BufId::X, BufId::O, b * n_embd);
        }
        trace_rows("ffn", li, BufId::O, b, n_embd);
        trace_rows("out", li, BufId::X, b, n_embd);
        if flush_every > 0 && (li + 1) % flush_every == 0 && li + 1 < n_layers {
            be().flush();
        }
    }
    be().rms_norm(BufId::X, wf.w.output_norm.offset, n_embd, eps, b, n_embd, 0);
    be().matmat_from(
        embd_kind,
        embd_off,
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
    be().argmax_rows(BufId::Logits, BufId::Tmp, c.vocab_size, b);
    Ok(())
}

/// Preserve the boundary state only after all recurrent layers have advanced.
/// Both a newly captured graph and a replay finish before this copy; no snapshot
/// node can then overwrite the saved boundary during later unarmed replays.
fn finish_decode_snapshot(wf: &Lfm2, external: bool) -> Result<(), String> {
    if !external || wf.state.recur_snap != Some(1) {
        return Ok(());
    }
    let n = wf.plan.recurrent_elems();
    let (_, plane_out) = wf.state.recur_decode_planes();
    let offset = n
        .checked_mul(plane_out)
        .ok_or("LFM2 snapshot plane overflow")?;
    be().copy_range_after_forward(BufId::RecurSnap, 0, BufId::Recur, offset, n)
        .map_err(|rc| format!("LFM2 decode snapshot failed rc={rc}"))
}

#[cfg(test)]
mod tests {
    use super::{BCX, buffer_requirements, finite_history_tail};
    use crate::gpu_support::Placement;
    use crate::{
        Activation, Attention, EmbedPlan, Ffn, KvSource, LayerPlan, ModelConfig,
        ModelPlan, OutputPlan,
    };
    use imparo_backend::BufId;

    #[test]
    fn finite_history_tail_covers_checkpoint_and_rejects_overlapping_moves() {
        let p = plan();
        assert_eq!(
            finite_history_tail(&p, 512, false, None, 64, 16),
            Some((27, 448, 64))
        );
        assert_eq!(
            finite_history_tail(&p, 512, true, Some(450), 64, 16),
            Some((27, 432, 80))
        );
        assert_eq!(finite_history_tail(&p, 512, true, Some(256), 64, 16), None);
        assert_eq!(finite_history_tail(&p, 512, true, Some(1), 64, 16), None);
        assert_eq!(finite_history_tail(&p, 512, true, Some(513), 64, 16), None);
        assert_eq!(
            finite_history_tail(&p, 129, true, None, 16, 16),
            Some((27, 112, 17))
        );
        assert_eq!(finite_history_tail(&p, 65, true, None, 16, 16), None);
    }

    #[test]
    fn finite_history_tail_does_not_cross_a_later_attention_or_unknown_state() {
        let mut p = plan();
        p.layers[29].attention = p.layers[27].attention;
        assert_eq!(finite_history_tail(&p, 512, true, None, 64, 16), None);
        p = plan();
        p.layers[29].attention = Attention::Recurrent {
            r_elems: 4096,
            s_elems: 1,
            key_dim: 1,
            value_dim: 1,
        };
        assert_eq!(finite_history_tail(&p, 512, true, None, 64, 16), None);
        p.layers[29].attention = Attention::Recurrent {
            r_elems: 4097,
            s_elems: 0,
            key_dim: 0,
            value_dim: 0,
        };
        assert_eq!(finite_history_tail(&p, 512, true, None, 64, 16), None);
    }

    /// LFM2.5-2.6B's real geometry, small enough to write down: 30 blocks, 8 attention at
    /// [2,5,9,13,17,21,24,27] and 22 recurrent, n_embd 2048, head_dim 64, l_cache 3.
    #[test]
    fn a_paired_drafters_caches_follow_the_targets() {
        let target = plan();
        let paired = ModelPlan {
            drafter: Some(crate::DrafterPlan {
                layers: 5,
                kv_heads: 8,
                head_dim: 64,
                taps: vec![2, 9, 17, 21, 27],
            }),
            ..target.clone()
        };
        let (positions, capacity, ring_batch) = (1000, 4096, 512);
        let bytes = crate::kv::kv_bytes_for(&paired, positions, capacity, ring_batch);
        let layout = crate::kv::kv_layout_for(&paired, positions, capacity, ring_batch);
        assert_eq!(paired.kv_layer_count(), 35);
        assert_eq!((bytes.len(), layout.len()), (35, 35));
        // The target's entries do not move.
        assert_eq!(
            bytes[..30],
            crate::kv::kv_bytes_for(&target, positions, capacity, ring_batch)[..]
        );
        // Each drafter cache is sized as a target full-attention layer of the same width
        // (8 KV heads x 64 in both).
        let attention = target
            .layers
            .iter()
            .find(|l| l.attention.is_attention())
            .expect("LFM2 has attention layers")
            .index as usize;
        for index in 30..35 {
            assert_eq!(bytes[index], bytes[attention]);
            assert_eq!(layout[index].layer, index as u32);
            assert_eq!(layout[index].logical_slots, layout[attention].logical_slots);
            assert_eq!(layout[index].k_stride, layout[attention].k_stride);
            assert_eq!(layout[index].v_stride, layout[attention].v_stride);
        }
        assert_eq!(
            crate::kv::drafter_kv_bytes_per_token(&paired),
            5 * (layout[attention].k_stride + layout[attention].v_stride)
        );
        assert_eq!(crate::kv::drafter_kv_bytes_per_token(&target), 0);
        assert_eq!(paired.draft_feature_bytes(512), 512 * 5 * 2048 * 4);
        assert_eq!(target.draft_feature_bytes(512), 0);
    }

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
            decode_interleave: true,
            mega_decode: true,
            drafter: None,
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
        assert!(!crate::gpu_support::should_fuse_epilogue(1, true));
        assert!(!crate::gpu_support::should_fuse_epilogue(128, false));
        assert!(crate::gpu_support::should_fuse_epilogue(128, true));
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

    /// The pool's device tier from the fit's KV tier, on LFM2's geometry (8 pooled layers,
    /// 1024-byte rows: one block is 1 MiB across the pooled layers, K and V).
    #[test]
    fn the_pool_tier_follows_the_kv_tier_and_never_drops_below_one_context() {
        use crate::kv::pool_capacity_blocks;
        let plan = plan();
        let (ctx, batch) = (8192, 512);
        let one_context = ctx / imparo_kv::page_cells();
        let block: u64 = 8 * 2 * 64 * 1024;
        let reserve = crate::placement::kv_reserve_bytes(&plan, ctx, batch);
        // No tier: one conversation at the configured context.
        assert_eq!(
            pool_capacity_blocks(&plan, ctx, batch, None, None),
            one_context
        );
        // A tier of exactly the reserve holds one context; ten blocks' worth more adds ten.
        assert_eq!(
            pool_capacity_blocks(&plan, ctx, batch, Some(reserve), None),
            one_context
        );
        assert_eq!(
            pool_capacity_blocks(&plan, ctx, batch, Some(reserve + 10 * block), None),
            one_context + 10
        );
        // A tier smaller than the reserve still holds one context: the fit guarantees it.
        assert_eq!(
            pool_capacity_blocks(&plan, ctx, batch, Some(block), None),
            one_context
        );
        // One layer side may not pass the largest view: 100 blocks of 64 KiB rows.
        let (small_ctx, view) = (4096, 100 * 64 * 1024);
        assert_eq!(
            pool_capacity_blocks(&plan, small_ctx, batch, Some(1 << 40), Some(view)),
            100
        );
    }
}
