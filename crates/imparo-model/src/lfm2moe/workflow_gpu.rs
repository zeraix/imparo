//! LFM2-MoE's layer graph on the device.
//!
//! ```text
//! every block:   prev = X
//!                Cur  = rms_norm(X, operator_norm)
//!                O    = shortconv(Cur)   OR   attention(Cur)
//!                X    = prev + O
//!                Cur  = rms_norm(X, ffn_norm)
//!                X    = X + ffn(Cur)          dense on the leading blocks, routed after
//! tail:          logits = head @ rms_norm(X_last, token_embd_norm)
//! ```
//!
//! The tower is LFM2's -- the same short convolution on most blocks and the same attention
//! on the rest -- and the ROUTED FEED-FORWARD is what this file has and `lfm2/workflow_gpu.rs`
//! does not. It is written out here rather than added as an arm there because the dense model
//! is what the shipping speculation work measures, and it does not move to make room.
//!
//! A ROUTED LAYER ON THE DEVICE, in the order it runs:
//!
//! ```text
//!   router matmul      Cur -> Scores            one row of n_expert per token
//!   moe_gate           Scores -> Probs, Sel     Sel = Probs + the file's bias
//!   top_k_rows         Sel -> Topk              the k picks, best first
//!   moe_plan           Topk, Probs -> Perm, Wgt, Seg, Inv
//!   moe_grouped x2     Cur[Perm] -> G, U        gate and up, each expert's own weights
//!   act_mul            G = silu(G) * U
//!   moe_grouped        G -> U                   down, back to n_embd; U is free by now
//!   moe_combine        U, Wgt, Inv -> O         each token's k rows, in slot order
//! ```
//!
//! THE PICK READS `Sel` AND THE WEIGHTS READ `Probs`. The file's `exp_probs_b` selects and
//! does not weigh; one buffer for both would run the right experts with the wrong weights on
//! a biased file, and the text would still read fluently.
//!
//! What this file does NOT have, and the dense LFM2 does: the mega program, the
//! finite-history prefill tails, and the co-batched decode path. Each is a speed facility
//! measured on that model; none is needed for a forward to be right.

use imparo_backend::BufId;

use crate::gpu_support::{
    BufferRequirement, Placement, be, conv_windows, fuse_epilogue_enabled, gprobe,
    gpu_probe_layer, half_activation_mirror_requirements,
    kv_dequant_scratch_requirements, kvq_mask_on, scores_needed, should_fuse_epilogue,
};
use crate::kv::{KvType, effective_workflow_kv_route, had_nrot, ring_mask};
use crate::lfm2moe::Lfm2Moe;
use crate::lfm2moe::workflow_cpu::{FfnW, MixerW};
use crate::{Attention, Ffn, ModelPlan};

/// THE SHORT CONVOLUTION'S PROJECTION: n_embd -> 3 * n_embd, carrying b, c and x
/// concatenated. LFM2's own file names the same slot; the two models never share a forward,
/// so each names its own.
pub const BCX: BufId = BufId::Model0;
/// The router's outputs, one row of `n_expert` per token.
const SCORES: BufId = BufId::Model1;
/// Those scores as probabilities -- THE WEIGHTS COME FROM HERE.
const PROBS: BufId = BufId::Model2;
/// The probabilities plus the file's per-expert bias -- THE PICK COMES FROM HERE.
const SEL: BufId = BufId::Model3;
/// `top_k_rows` over `SEL`: the picked expert ids, then its values and working space.
const TOPK: BufId = BufId::Model4;
/// Work row -> the token it belongs to.
const PERM: BufId = BufId::Model5;
/// Work row -> its routing weight.
const WGT: BufId = BufId::Model6;
/// Expert -> its half-open range of work rows, `n_expert + 1` entries.
const SEG: BufId = BufId::Model7;
/// (token, slot) -> work row; what the combine reads back.
const INV: BufId = BufId::Model8;

/// The widest routed layer: experts used, one expert's hidden width, and the expert count.
///
/// A free function over the plan, so the buffer list can be checked without the file.
fn routed_shape(plan: &ModelPlan) -> Option<(u32, u32, u32)> {
    plan.layers
        .iter()
        .filter_map(|l| match l.ffn {
            Ffn::Moe {
                expert_hidden,
                experts,
                experts_used,
                ..
            } => Some((experts_used, expert_hidden, experts)),
            Ffn::Dense { .. } => None,
        })
        .reduce(|a, b| (a.0.max(b.0), a.1.max(b.1), a.2.max(b.2)))
}

/// The buffers one forward of this model needs, as a function of the plan and the batch.
///
/// A free function over the plan: it reads nothing else, so it is testable without the
/// file on disk.
#[allow(clippy::too_many_lines)]
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
    let (used, hidden, experts) = routed_shape(plan).unwrap_or((0, 0, 0));
    // WORK ROWS, not tokens: a routed layer runs each token through `k` experts, so its
    // hidden activations are `b * k` rows wide. The dense blocks' `n_ff` shares the same
    // two buffers and is the wider of the two on this file (7168 against 4 * 1792).
    let rows = b * used as usize;
    let g_words = (b * n_ff).max(rows * hidden as usize);
    // The down projection writes back into U, which the gated multiply has just finished
    // reading -- so U carries the widest of the three.
    let u_words = g_words.max(rows * n_embd);
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
        // ATTN does not share: the register-tiled GEMM reads whole 8-row tiles and cannot
        // mask, so it reads past the token count, and what those rows hold changes the
        // logits.
        need(
            BufId::Attn,
            f(b * c.n_heads as usize * head_max),
            Placement::Dedicated,
        ),
        need(BCX, f(b * 3 * n_embd), Placement::Group(0)),
        need(BufId::G, f(g_words), Placement::Group(1)),
        need(BufId::U, f(u_words), Placement::Group(1)),
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
    if used != 0 {
        // The routing scratch joins the feed-forward group: it is live exactly where the
        // dense blocks' G and U are, and dead everywhere else.
        let e = experts as usize;
        let k = used as usize;
        requirements.extend([
            need(SCORES, f(b * e), Placement::Group(1)),
            need(PROBS, f(b * e), Placement::Group(1)),
            need(SEL, f(b * e), Placement::Group(1)),
            need(
                TOPK,
                f(be().top_k_rows_len(experts, b as u32, used) as usize),
                Placement::Group(1),
            ),
            need(PERM, f(b * k), Placement::Group(1)),
            need(WGT, f(b * k), Placement::Group(1)),
            // n_expert+1 segment offsets, then the active count and the compacted
            // list of experts that hold a work row (moe_plan writes the tail).
            need(SEG, f(2 * e + 2), Placement::Group(1)),
            need(INV, f(b * k), Placement::Group(1)),
        ]);
    }
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
    if fuse_epilogue_enabled() {
        requirements.extend(half_activation_mirror_requirements(
            b,
            g_words / b.max(1),
            BufId::U,
        ));
    }
    requirements
}

/// Nothing to prepare beyond what the backend does for every model.
///
/// # Errors
/// Never; the signature is the architecture table's.
pub fn prepare_device(_wf: &mut Lfm2Moe) -> Result<(), String> {
    Ok(())
}

/// One routed feed-forward, `Cur` in and `O` out.
///
/// `rows` is `b * experts_used`: every token runs `k` experts, so the hidden activations
/// carry one row per (token, pick).
/// `next_norm`: the next block's operator norm (weight offset, eps) when the caller will add
/// this layer's output to the residual and normalise it. Then the combine is folded into that
/// norm (`moe_combine_add_rms_norm`) when the backend serves it, and the return is `true`:
/// X holds the new residual and Cur its normalised row. `false`: O holds the output, as before.
#[allow(clippy::too_many_arguments)]
fn routed_ffn(
    w: &FfnW,
    ffn: Ffn,
    n_embd: u32,
    b: u32,
    wkind: &impl Fn(&imparo_gguf::weights::Tensor) -> u32,
    li: usize,
    prefill_chunk: bool,
    next_norm: Option<(u64, f32)>,
) -> Result<bool, String> {
    let (
        FfnW::Moe {
            router,
            bias,
            gate,
            up,
            down,
        },
        Ffn::Moe {
            expert_hidden,
            experts,
            experts_used,
            gating,
            normalise_weights,
            weights_scale,
            shared_hidden,
            ..
        },
    ) = (w, ffn)
    else {
        return Err(format!(
            "lfm2moe layer {li}: a routed layer resolved dense weights"
        ));
    };
    if shared_hidden != 0 {
        return Err(format!(
            "lfm2moe layer {li}: an always-on shared expert is not served on the device"
        ));
    }
    if !be().supports_moe() || !be().supports_top_k_rows(experts_used) {
        return Err(format!(
            "lfm2moe layer {li}: this backend serves no routed feed-forward"
        ));
    }
    let rows = b
        .checked_mul(experts_used)
        .ok_or_else(|| format!("lfm2moe layer {li}: work rows exceed u32"))?;
    // One expert's slice of the stacked `[n_embd, n_ff_exp, n_expert]` tensor. Derived from
    // the tensor's own byte count rather than from the shape, so a file whose stack is not
    // divisible by its expert count is refused here instead of read at wrong addresses.
    let stride = |t: &imparo_gguf::weights::Tensor| -> Result<u64, String> {
        let e = experts as usize;
        if e == 0 || t.bytes % e != 0 {
            return Err(format!(
                "lfm2moe layer {li}: {experts} experts do not divide a {}-byte stack",
                t.bytes
            ));
        }
        Ok((t.bytes / e) as u64)
    };

    be().matmat(
        wkind(router),
        router.offset as u64,
        n_embd,
        experts,
        BufId::Cur,
        SCORES,
        b,
    );
    if li == gpu_probe_layer() {
        gprobe("moe.router", SCORES, 0, experts as usize);
    }
    // THE WHOLE ROUTE IN ONE DISPATCH where the backend serves it. The pick reads what the
    // gating wrote and the plan reads what the pick wrote, so the three were already a
    // chain: folding them costs no overlap and saves two launches a routed layer.
    let routed = be().moe_route(
        SCORES,
        PROBS,
        SEL,
        TOPK,
        PERM,
        WGT,
        SEG,
        INV,
        bias.offset,
        b,
        experts,
        experts_used,
        gating,
        normalise_weights,
        weights_scale,
    );
    if !routed {
        if !be().moe_gate(SCORES, PROBS, SEL, bias.offset, b, experts, gating) {
            return Err(format!("lfm2moe layer {li}: the gating was refused"));
        }
        if !be().top_k_rows(SEL, TOPK, experts, b, experts_used) {
            return Err(format!("lfm2moe layer {li}: the expert pick was refused"));
        }
        if !be().moe_plan(
            TOPK,
            PROBS,
            PERM,
            WGT,
            SEG,
            INV,
            b,
            experts,
            experts_used,
            normalise_weights,
            weights_scale,
        ) {
            return Err(format!("lfm2moe layer {li}: the work-row plan was refused"));
        }
    }
    if li == gpu_probe_layer() {
        // The probabilities of the first token, then its picks and their weights. WGT and
        // PERM are indexed by WORK ROW, and the work rows are sorted by expert -- so rows
        // 0.. are not token 0's picks, and printing them as if they were showed four
        // weights that summed to 0.94 and meant nothing. `INV` is the map that answers it.
        gprobe("moe.probs", PROBS, 0, experts as usize);
        crate::gpu_support::gprobe_u32("moe.picks", TOPK, 0, experts_used as usize);
        crate::gpu_support::gprobe_u32(
            "moe.token0_rows",
            INV,
            0,
            experts_used as usize,
        );
        crate::gpu_support::gprobe_u32("moe.segments", SEG, 0, experts as usize + 1);
    }

    let grouped = |t: &imparo_gguf::weights::Tensor,
                   n_in: u32,
                   n_out: u32,
                   src: BufId,
                   dst: BufId,
                   src_work_rows: bool|
     -> Result<(), String> {
        if be().moe_grouped(
            wkind(t),
            t.offset as u64,
            stride(t)?,
            src,
            dst,
            PERM,
            SEG,
            n_in,
            n_out,
            experts,
            b,
            rows,
            src_work_rows,
            prefill_chunk,
        ) {
            Ok(())
        } else {
            Err(format!("lfm2moe layer {li}: a routed matmul was refused"))
        }
    };
    // The gate and up read the TOKEN's normalised hidden state; the down reads the work
    // rows those two just wrote.
    //
    // ONE DISPATCH FOR THE THREE when the backend serves it: gate, up and the activation
    // between them. Both projections read the same hidden state and neither feeds the
    // other, so this merges two independent grids and does not make up wait for gate --
    // which is the trade a fused epilogue on the up projection alone would make.
    let paired = wkind(gate) == wkind(up)
        && stride(gate)? == stride(up)?
        && be().moe_grouped_pair(
            wkind(gate),
            gate.offset as u64,
            up.offset as u64,
            stride(gate)?,
            BufId::Cur,
            BufId::G,
            PERM,
            SEG,
            n_embd,
            expert_hidden,
            experts,
            b,
            rows,
            prefill_chunk,
        );
    if !paired {
        grouped(gate, n_embd, expert_hidden, BufId::Cur, BufId::G, false)?;
        grouped(up, n_embd, expert_hidden, BufId::Cur, BufId::U, false)?;
        // NOT the fused epilogue: that form writes through the up projection's own output
        // rows, and a routed matmul's rows belong to whichever expert claimed them. The
        // standalone multiply is the same arithmetic over the same rows.
        be().act_mul(BufId::G, BufId::U, rows * expert_hidden);
    }
    if li == gpu_probe_layer() {
        gprobe("moe.swiglu", BufId::G, 0, expert_hidden as usize);
    }
    // THE DOWN PROJECTION WRITES INTO U. The multiply above consumed it, and its rows are
    // the widest thing this layer stages, so the buffer list sizes U for whichever of the
    // two is larger.
    grouped(down, expert_hidden, n_embd, BufId::G, BufId::U, true)?;
    if let Some((norm, eps)) = next_norm {
        if be().moe_combine_add_rms_norm(
            BufId::U,
            WGT,
            INV,
            experts_used,
            BufId::Cur,
            BufId::X,
            norm,
            n_embd,
            eps,
            b,
        ) {
            return Ok(true);
        }
    }
    if !be().moe_combine(BufId::U, WGT, INV, BufId::O, n_embd, experts_used, b) {
        return Err(format!("lfm2moe layer {li}: the combine was refused"));
    }
    Ok(false)
}

/// Runs ONE chunk of LFM2-MoE's forward on the device.
///
/// `argmax` true means: do not copy the logits back, write the greedy pick's index into
/// `out[0]` as raw bits.
///
/// # Errors
/// When the output demand is one this workflow does not serve, or a dispatch fails.
#[allow(clippy::too_many_lines)]
pub fn batch(
    wf: &mut Lfm2Moe,
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
    let eps = c.norm_eps;
    let b = u32::try_from(tokens.len()).map_err(|_| "batch too large")?;
    let sp = u32::try_from(start_pos).map_err(|_| "start_pos too large")?;

    // The drafter reads these: a paired DSpark drafter takes the target's hidden state at
    // its tap layers, and a forward that publishes none fails at the pairing.
    let layer_outputs = wf
        .state
        .layer_outputs
        .as_ref()
        .is_some_and(crate::layer_outputs::LayerOutputCapture::active);
    let row_argmax = wf.state.output_demand == crate::OutputDemand::RowArgmax;
    let all_logits = wf.state.output_demand.requires_all_positions();
    // A PROMPT CHUNK, as the routed matmuls need to know: its positions must compute the same
    // at every chunk width (a resumed prompt is chunked unlike a cold one). A forward that
    // wants every position's output is a verify batch, which keeps decode-time arithmetic.
    let prefill_chunk = !all_logits;
    if all_logits && argmax {
        return Err("all-position logits cannot request scalar argmax".into());
    }
    if row_argmax && !be().supports_argmax_rows() {
        return Err("row argmax is not served by this backend".into());
    }
    if wf.state.output_demand == crate::OutputDemand::GreedyVerification {
        return Err("lfm2moe serves no device greedy verification".into());
    }
    // A PREFIX VERIFICATION carries the recurrent state aside at every prefix boundary, so a
    // partial accept rolls back to the state that prefix left rather than replaying it.
    let verification_prefix_tokens = wf.state.verification_prefix_tokens;
    let verification_recurrent_elems = if verification_prefix_tokens != 0 {
        if verification_prefix_tokens != tokens.len() || !all_logits {
            return Err(
                "verification prefix snapshots require the complete all-logits batch"
                    .into(),
            );
        }
        let elems = wf.plan.recurrent_elems() as u64;
        elems
            .checked_mul(u64::from(b) + 1)
            .filter(|&e| u32::try_from(e).is_ok())
            .ok_or_else(|| {
                "verification prefix snapshot range exceeds u32".to_string()
            })?;
        elems
    } else {
        0
    };
    // A ROW-LAYOUT BATCH IS A TREE: each row carries its own position and visibility mask in
    // BufId::RowLayout, and each row's input to every convolution window goes aside to
    // RowInputs so a commit can rebuild the accepted path's windows.
    let row_layout = wf.state.row_layout;
    if row_layout != 0
        && (row_layout != b
            || !all_logits
            || KvType::k() != KvType::F16
            || KvType::v() != KvType::F16
            || verification_prefix_tokens != 0
            || wf.state.recur_snap.is_some())
    {
        return Err(
            "a row-layout batch keeps every row, reads an f16 cache and takes no other snapshot"
                .into(),
        );
    }
    let row_inputs = if row_layout != 0 {
        Some(crate::verification::RowInputLayout::of(
            &wf.plan,
            &conv_windows(&wf.plan),
        )?)
    } else {
        None
    };

    // Activation aliases and scratch ranges are planned for the live batch width. Recurrent
    // state and KV are allocated separately and survive this resize.
    wf.gpu_fit_batch(tokens.len())?;
    let decode = b == 1;
    let decode_projection_preparation =
        decode && be().use_decode_projection_preparation();
    let prefill_projection_preparation =
        !decode && be().use_prefill_projection_preparation();
    let replayed = if decode {
        be().decode_prepare(tokens[0], sp, argmax && !layer_outputs)
            .map_err(|rc| format!("lfm2moe GPU decode prepare failed rc={rc}"))?
    } else {
        false
    };
    if replayed {
        be().end()
            .map_err(|rc| format!("lfm2moe GPU forward failed rc={rc}"))?;
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
    if !pipe.is_some_and(|p| p.token_on_device) {
        be().write_u32(BufId::Tokens, 0, tokens);
    }
    // No embedding scale: LFM2 sets scale_by_sqrt_embd false and this file inherits it.
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

    // THE RECURRENT PLANES for this batch: a decode step reads one plane and writes the
    // next, so the plane it read stays intact and a discarded step is undone by an index.
    // Prefill is in place.
    let recur = wf.plan.recurrent_layout();
    let recur_elems = wf.plan.recurrent_elems();
    let (plane_in, plane_out) = if decode {
        let (i, o) = wf.state.recur_decode_planes();
        (i * recur_elems, o * recur_elems)
    } else {
        let p = wf.state.recur_plane * recur_elems;
        (p, p)
    };
    let recurrent_snapshot = wf.state.recur_snap;

    let n_layers_total = wf.plan.layers.len();
    let flush_every = crate::gpu_support::flush_layers_bounded(b, n_layers_total);
    let act = wf.plan.layers[0].ffn.activation().epilogue();
    let mut operator_norm_ready = false;

    for (li, (layer, &(r_off, _, _, _))) in
        wf.plan.layers.clone().iter().zip(recur.iter()).enumerate()
    {
        let lw = &wf.w.layers[li];
        let probe_here = li == gpu_probe_layer();
        // The next layer's operator norm, formed by the entry from the finished X into Cur so
        // that layer skips its norm dispatch. Not into the probe layer (its probe reads the
        // dispatch path's norm), and not past the last layer (the output norm is the head's).
        let next_norm_off = if li + 1 < n_layers_total && li + 1 != gpu_probe_layer() {
            wf.w.layers[li + 1].op_norm.offset
        } else {
            imparo_backend::NO_WEIGHT
        };
        let next_norm_folded = next_norm_off != imparo_backend::NO_WEIGHT;
        // A ROUTED LAYER'S TAIL AS ONE PERSISTENT DISPATCH (Metal, decode): the residual and
        // the FFN norm, the router, the route, the k experts' gate/up and down rows, the
        // combine and the residual again -- seven dependent dispatches on the path below. When
        // it ran, X holds the layer's output. Refused (false, nothing encoded) on the dense
        // leading layers, off decode, with a drafter capturing, and on the probe layer.
        let routed_entry = || -> bool {
            if !decode || layer_outputs || probe_here {
                return false;
            }
            let (
                FfnW::Moe {
                    router,
                    bias,
                    gate,
                    up,
                    down,
                },
                Ffn::Moe {
                    expert_hidden,
                    experts,
                    experts_used,
                    gating,
                    normalise_weights,
                    weights_scale,
                    shared_hidden: 0,
                    ..
                },
            ) = (&lw.ffn, layer.ffn)
            else {
                return false;
            };
            // One expert's slice of each stack, from the stack's own byte count (the rule
            // `routed_ffn` refuses a non-dividing file by).
            let per = |t: &imparo_gguf::weights::Tensor| {
                (experts != 0 && t.bytes % experts as usize == 0)
                    .then(|| (t.bytes / experts as usize) as u64)
            };
            let (Some(gate_stride), Some(up_stride), Some(down_stride)) =
                (per(gate), per(up), per(down))
            else {
                return false;
            };
            be().mega_layer(&imparo_backend::MegaEntry {
                layer: imparo_backend::MegaLayer::Lfm2Moe(
                    imparo_backend::Lfm2MoeMegaLayer {
                        router_kind: wkind(router),
                        router_off: router.offset as u64,
                        bias_off: bias.offset,
                        gate_kind: wkind(gate),
                        gate_off: gate.offset as u64,
                        gate_stride,
                        up_kind: wkind(up),
                        up_off: up.offset as u64,
                        up_stride,
                        down_kind: wkind(down),
                        down_off: down.offset as u64,
                        down_stride,
                        ffn_norm_off: lw.ffn_norm.offset,
                        n_embd,
                        n_ff: expert_hidden,
                        n_expert: experts,
                        k: experts_used,
                        gating,
                        normalise: normalise_weights,
                        weights_scale,
                        eps,
                        x: BufId::X,
                        add: BufId::O,
                        g: BufId::G,
                        scores: SCORES,
                        next_norm_off,
                        cur: BufId::Cur,
                    },
                ),
                n_tok: b,
            })
        };
        if !operator_norm_ready {
            if decode_projection_preparation || prefill_projection_preparation {
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
        operator_norm_ready = false;
        if probe_here {
            gprobe("operator_norm", BufId::Cur, 0, n_embd as usize);
        }

        // THE MIXER'S OUTPUT ADDED STRAIGHT INTO THE RESIDUAL at prefill: the output projection
        // stores X + W . a (`matmat_resid`) instead of writing O for the FFN norm to add back,
        // and that norm then reads X alone (`rms_norm_resid`). Not at decode (the routed mega
        // entry reads O) and not on a probed layer (probes read O).
        let fold_mixer = !decode && !probe_here;
        let mixer_folded;
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
                Attention::Full {
                    head_dim: hd,
                    rope_base,
                    rope_dim,
                },
            ) => {
                let (qw, kw) = (n_head * hd, n_kv * hd);
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
                // Per-head norm then rope, on Q and K. No frequency factors: this file
                // carries no rope_freqs tensor, so every layer ropes plainly.
                if row_layout != 0 {
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
                let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
                let direct_kv = row_layout == 0
                    && !probe_here
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
                    if row_layout == 0 {
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
                        // No sliding window: a file that set one would plan
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
                mixer_folded = fold_mixer
                    && be().matmat_resid(
                        wkind(wo),
                        wo.offset as u64,
                        qw,
                        n_embd,
                        BufId::Attn,
                        BufId::X,
                        b,
                    );
                if !mixer_folded {
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
            }
            (
                MixerW::ShortConv {
                    conv,
                    in_proj,
                    out_proj,
                },
                Attention::Recurrent { .. },
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
                if probe_here {
                    gprobe("conv.in_proj", BCX, 0, (3 * n_embd) as usize);
                }
                if row_layout != 0 {
                    // Each row reads its own path and the live state is only read; each
                    // row's input to the window goes aside for the commit.
                    let inputs_off = row_inputs
                        .as_ref()
                        .and_then(|layout| layout.input_off(li))
                        .ok_or_else(|| format!("layer {li} has no row-input window"))?;
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
                    // A PREFIX VERIFICATION takes the state aside at every boundary, so
                    // a partial accept rolls back to what that prefix left rather than
                    // replaying it.
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
                if probe_here {
                    gprobe("conv.conv", BufId::Attn, 0, n_embd as usize);
                }
                mixer_folded = fold_mixer
                    && be().matmat_resid(
                        wkind(out_proj),
                        out_proj.offset as u64,
                        n_embd,
                        n_embd,
                        BufId::Attn,
                        BufId::X,
                        b,
                    );
                if !mixer_folded {
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
            }
            // The plan and the resolved weights come from the same file, so a mismatch is a
            // defect in this crate, not a bad file.
            (m, a) => {
                return Err(format!(
                    "lfm2moe layer {li}: plan says {a:?} but the weights resolved as {}",
                    match m {
                        MixerW::Attention { .. } => "attention",
                        MixerW::ShortConv { .. } => "a short convolution",
                    }
                ));
            }
        }

        // The routed tail: the mixer ran on the dispatch path and left its output in O.
        let mega_tail_done = routed_entry();
        if mega_tail_done {
            operator_norm_ready = next_norm_folded;
            if let Some(capture) = &wf.state.layer_outputs {
                capture.record(li as u32, sp, b, BufId::X)?;
            }
            if flush_every != 0
                && (li + 1) % flush_every == 0
                && li + 1 < n_layers_total
            {
                be().flush();
            }
            continue;
        }

        // Residual and the feed-forward's pre-norm, fused when the backend serves it. A folded
        // mixer already added into X, and its norm must be the pre-add one's floats.
        let ffn_norm_fused = if mixer_folded {
            if !be().rms_norm_resid(BufId::Cur, BufId::X, lw.ffn_norm.offset, n_embd, eps, b) {
                return Err(format!(
                    "lfm2moe layer {li}: the mixer added into X but its norm was refused"
                ));
            }
            true
        } else {
            be().add_rms_norm(
                BufId::Cur,
                BufId::X,
                BufId::O,
                lw.ffn_norm.offset,
                n_embd,
                eps,
                b,
                n_embd,
                0,
            )
        };
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

        // Set when a routed layer folded its combine into the next block's norm.
        let mut ffn_folded_norm = false;
        match (&lw.ffn, layer.ffn) {
            (FfnW::Dense { gate, up, down }, Ffn::Dense { hidden, .. }) => {
                let (gate_kind, gate_off) = (wkind(gate), gate.offset as u64);
                let (up_kind, up_off) = (wkind(up), up.offset as u64);
                let (down_kind, down_off) = (wkind(down), down.offset as u64);
                let whole_ffn = be().ffn_gated_down(
                    gate_kind,
                    gate_off,
                    up_kind,
                    up_off,
                    down_kind,
                    down_off,
                    n_embd,
                    hidden,
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
                        hidden,
                        BufId::Cur,
                        BufId::G,
                        BufId::U,
                        b,
                    );
                    if !paired {
                        be().matmat(
                            gate_kind,
                            gate_off,
                            n_embd,
                            hidden,
                            BufId::Cur,
                            BufId::G,
                            b,
                        );
                        let fused =
                            should_fuse_epilogue(b, be().supports_epilogue(act));
                        if fused {
                            be().set_epilogue(act);
                        }
                        be().matmat(
                            up_kind,
                            up_off,
                            n_embd,
                            hidden,
                            BufId::Cur,
                            if fused { BufId::G } else { BufId::U },
                            b,
                        );
                        if fused {
                            be().set_epilogue(imparo_backend::Epilogue::None);
                        } else {
                            be().act_mul(BufId::G, BufId::U, b * hidden);
                        }
                    }
                    be().matmat(
                        down_kind,
                        down_off,
                        hidden,
                        n_embd,
                        BufId::G,
                        BufId::O,
                        b,
                    );
                }
            }
            (w @ FfnW::Moe { .. }, ffn @ Ffn::Moe { .. }) => {
                let next_norm = (li + 1 < n_layers_total)
                    .then(|| (wf.w.layers[li + 1].op_norm.offset, eps));
                ffn_folded_norm = routed_ffn(w, ffn, n_embd, b, &wkind, li, prefill_chunk, next_norm)?;
            }
            (w, f) => {
                return Err(format!(
                    "lfm2moe layer {li}: plan says {f:?} but the weights resolved as {}",
                    match w {
                        FfnW::Dense { .. } => "a dense feed-forward",
                        FfnW::Moe { .. } => "routed experts",
                    }
                ));
            }
        }

        // The last layer's residual has no next norm to fuse into. A routed layer that folded
        // its combine into that norm has already done both.
        if ffn_folded_norm {
            operator_norm_ready = true;
        } else if li + 1 < n_layers_total {
            let next_norm = wf.w.layers[li + 1].op_norm.offset;
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

    // A mega run recorded in the program form is encoded here, before the head reads X (a
    // no-op when nothing is pending; the per-layer form, this family's default, records none).
    be().mega_program_end();

    if !wf.state.output_demand.wants_logits() {
        be().end()
            .map_err(|rc| format!("lfm2moe forward failed rc={rc}"))?;
        crate::gpu_support::prefill_region_ended();
        return Ok(());
    }
    let output_rows = if all_logits { b } else { 1 };
    let output_start = if all_logits { 0 } else { b - 1 };
    let output_words = output_rows
        .checked_mul(c.vocab_size)
        .ok_or_else(|| "all-position logits count exceeds u32".to_string())?;
    let (head_src, head_src_row) = if prefill_projection_preparation {
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
        head_src,
        BufId::Logits,
        output_rows,
        head_src_row,
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
            .map_err(|rc| format!("lfm2moe pipelined step failed rc={rc}"));
    }
    if row_argmax {
        be().argmax_rows(BufId::Logits, BufId::Tmp, c.vocab_size, b);
    } else if argmax {
        be().argmax(BufId::Logits, BufId::Tmp, c.vocab_size);
    }
    be().end()
        .map_err(|rc| format!("lfm2moe forward failed rc={rc}"))?;
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
