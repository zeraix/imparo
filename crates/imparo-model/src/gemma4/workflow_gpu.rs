//! Gemma4 forward with activations resident on the GPU.
//!
//! The whole batch is encoded into one command buffer and synchronised once, instead of
//! one command buffer plus two memcpys per matmul. The CPU reference in `gemma4.rs`
//! stays the oracle this is checked against.

// No OS gate: this is pure Backend-trait code (zero backend-crate references). It
// compiles on every target; whether it RUNS is a runtime question routing answers via
// backend::active(). Backend construction is the only conditional (backend.rs).
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use imparo_backend::BufId;

#[cfg(feature = "cuda-gate0-capture")]
use crate::gate0_capture::{CaptureOp, CaptureRun};
use crate::gpu_support::{layer_skip_log, tail_align, tail_split};
use crate::{Attention, KvSource, ModelPlan};

// GEMMA4'S PRIVATE BUFFER SLOTS. These were named Gate, Back and PerLayer in the shared
// BufId, which put one architecture's per-layer-embedding machinery in every backend's
// vocabulary -- and left LFM2 with no name for its ShortConv projection. The slots are
// generic now and a model says what its own are.
const GATE: BufId = BufId::Model0;
const BACK: BufId = BufId::Model1;
const PER_LAYER: BufId = BufId::Model2;
/// Host-staged rows: this batch's rows of the per-layer token-embedding table, in
/// token order, gathered by the host (`Backend::stage_rows`).
const PLE_ROWS: BufId = BufId::Model3;
// Shared device-path machinery; see the charter note in crate::gpu_support.
use crate::gpu_support::{
    BufferRequirement, Placement, be, gprobe, gpu_probe_layer,
    half_activation_mirror_requirements, kv_dequant_scratch_requirements, kvq_mask_on,
    probe_first, scores_needed,
};

/// KV cache storage types (GGML type ids: 1 = f16, 2 = q4_0, 8 = q8_0). User config via
/// --cache-type-k/-v (environment IMPARO_CTK/IMPARO_CTV); f16 is the default and its
/// path is byte-identical to before this feature existed.
pub(crate) use crate::kv::KvType;

use crate::gemma4::Gemma4;
// The pool geometry is shared: see the charter note in crate::kv.
use crate::kv::{effective_workflow_kv_route, had_nrot, ring_mask};

/// Prepare the complete Gemma4 FFN/PLE Q4 hot set at model admission when the active
/// backend has selected a persistent transformed-weight route. Offsets and shapes come
/// from the resolved model rather than CUDA constants; Metal and safe-off CUDA return
/// before allocating this descriptor list.
pub fn prepare_device(wf: &mut Gemma4) -> Result<(), String> {
    if !be().quantized_weight_cache_enabled() {
        return Ok(());
    }
    let n_embd = wf.plan.config.n_embd;
    let n_ff = wf.plan.config.n_ff;
    let ple = wf.plan.embed.per_layer_dim.unwrap_or(0);
    let mut spans = Vec::with_capacity(wf.w.layers.len() * if ple > 0 { 5 } else { 3 });
    let mut push = |tensor: &imparo_gguf::weights::Tensor, n_in: u32, n_out: u32| {
        let kind = imparo_gguf::weights::weight_kind(tensor.ggml_type)
            .expect("validated at load") as u32;
        spans.push(imparo_backend::QuantizedWeightPrepack {
            offset: tensor.offset as u64,
            n_in,
            n_out,
            kind,
            reserved: 0,
        });
    };
    for layer in &wf.w.layers {
        push(&layer.ffn_gate, n_embd, n_ff);
        push(&layer.ffn_up, n_embd, n_ff);
        push(&layer.ffn_down, n_ff, n_embd);
        if let (Some(gate), Some(proj)) = (&layer.inp_gate, &layer.proj) {
            push(gate, n_embd, ple);
            push(proj, ple, n_embd);
        }
    }
    be().prepare_quantized_weight_cache(&spans)
        .map(|_| ())
        .map_err(|rc| format!("backend quantized-weight admission failed rc={rc}"))
}

// gemma4's layer graph, and nothing else. Everything that used to sit beside it --
// allocating activations, sizing the rings, growing the cache, the footprint probes --
// read the plan and the shared state and nothing about gemma4, so it is `Workflow<A>`'s
// now and lives in `gpu_support`.
/// Runs the batch on the GPU.
///
/// `argmax` true means: do not copy the logits back at all, write the greedy pick's
/// index into `out[0]` as raw bits. `Workflow`'s `device_argmax` decodes that, once
/// -- one decode step saves a 1 MiB read plus a 262144-element host scan.
///
/// # Errors
///
/// Returns an error when a dispatch fails.
#[allow(clippy::too_many_lines)]
pub fn batch(
    wf: &mut Gemma4,
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
    let n_layers = c.n_layers as usize;
    let ple = wf.plan.embed.per_layer_dim.unwrap_or(0);
    let b = u32::try_from(tokens.len()).map_err(|_| "batch too large".to_string())?;
    let sp = u32::try_from(start_pos).map_err(|_| "position too large".to_string())?;
    let width = ple * c.n_layers; // per-layer embedding row width
    probe_first("gpu batch entry");
    wf.gpu_fit_batch(tokens.len())?;
    let mw = &wf.w;
    let decode = b == 1;
    #[cfg(feature = "cuda-gate0-capture")]
    let mut gate0_capture = CaptureRun::from_environment(
        &wf.plan,
        &wf.weights,
        tokens,
        start_pos,
        &format!("{kv_quant_route:?}"),
        be(),
    )?;
    #[cfg(feature = "cuda-gate0-capture")]
    let gate0_capture_active = gate0_capture.is_active();
    #[cfg(not(feature = "cuda-gate0-capture"))]
    let gate0_capture_active = false;
    // A replayed graph hides every intermediate boundary. Explicit capture therefore
    // takes the ordinary encode path; the feature/env-off path is the original branch.
    let replayed = if gate0_capture_active {
        false
    } else if decode {
        be().decode_prepare(tokens[0], sp, argmax)
            .map_err(|rc| format!("GPU decode prepare failed rc={rc}"))?
    } else {
        be().prefill_prepare(tokens, sp, argmax)
            .map_err(|rc| format!("GPU prefill prepare failed rc={rc}"))?
    };
    if replayed {
        be().end()
            .map_err(|rc| format!("GPU forward failed rc={rc}"))?;
        probe_first("gpu after submit");
        if argmax {
            out.resize(1, 0.0);
            be().read(BufId::Tmp, 0, out);
        } else {
            out.resize(c.vocab_size as usize, 0.0);
            be().read(BufId::Logits, 0, out);
        }
        if std::env::var("IMPARO_KV_SCAN").as_deref() == Ok("1")
            && KvType::k() == KvType::F16
            && KvType::v() == KvType::F16
        {
            wf.kv_scan(sp as usize + b as usize);
        }
        return Ok(());
    }
    // A backend may use a decode-specific submission lifecycle; the default
    // remains `begin()`, so CPU and Metal retain their established behavior.
    be().begin_forward(decode);

    // embedding, scaled by sqrt(n_embd)
    //
    // ONE DISPATCH FOR THE BATCH. This was a `row` per token, so a 512-token batch
    // issued 512 dispatches of n_embd/32 threads each. gather_rows takes the index
    // vector instead. The per-token loop remains the fallback: the backend refuses a
    // width it cannot vectorise, and a backend without the op returns false.
    let embd_scale = (f64::from(n_embd)).sqrt() as f32;
    let wkind = imparo_gguf::weights::weight_kind(mw.token_embd.ggml_type)
        .expect("validated at load") as u32;
    // A pipelined step (docs/decode-turnaround.md): the token is already in Tokens[0],
    // written by the previous step's argmax_feed on the device. A model with a
    // host-staged table never runs pipelined (`decode_pipelined`), so the staging
    // below always has the token.
    let pipe = wf.state.pipe;
    if !pipe.is_some_and(|p| p.token_on_device) {
        be().write_u32(BufId::Tokens, 0, tokens);
    }
    // The PLE table is a row-gathered tensor; when it is host-staged the host copies this batch's
    // rows into PLE_ROWS next to the token write, so the device never maps the table.
    // false = this backend reads the table itself (CPU-style), and the PLE step below
    // gathers on the device as before.
    let ple_staged = match (wf.plan.embed.per_layer_row_bytes, mw.per_layer_token_embd)
    {
        (Some(row_bytes), Some(pt)) => {
            be().stage_rows(pt.offset as u64, row_bytes as u32, tokens, PLE_ROWS)
        }
        _ => false,
    };
    let gathered = be().gather_rows(
        wkind,
        mw.token_embd.offset as u64,
        n_embd,
        c.vocab_size,
        embd_scale,
        BufId::X,
        0,
        BufId::Tokens,
        b,
    );
    if !gathered {
        for (t, &tok) in tokens.iter().enumerate() {
            be().row(
                wkind,
                mw.token_embd.offset as u64,
                n_embd,
                tok,
                embd_scale,
                BufId::X,
                (t as u32) * n_embd,
            );
        }
    }

    gprobe("embed", BufId::X, 0, n_embd as usize);
    gprobe(
        "embed_last",
        BufId::X,
        (b as u64 - 1) * n_embd as u64,
        n_embd as usize,
    );

    if ple > 0 {
        if let (Some(pm), Some(pt)) = (mw.per_layer_model_proj, mw.per_layer_token_embd)
        {
            be().matmat(
                imparo_gguf::weights::weight_kind(pm.ggml_type)
                    .expect("validated at load") as u32,
                pm.offset as u64,
                n_embd,
                width,
                BufId::X,
                PER_LAYER,
                b,
            );
            be().scale(PER_LAYER, 1.0 / embd_scale, b * width);
            be().rms_norm(
                PER_LAYER,
                mw.per_layer_proj_norm.offset,
                ple,
                eps,
                b * c.n_layers,
                ple,
                0,
            );
            let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
            if ple_staged {
                be().ple_gather_combine_staged(
                    PER_LAYER,
                    PLE_ROWS,
                    width,
                    (f64::from(ple)).sqrt() as f32,
                    inv_sqrt2,
                    b,
                );
            } else {
                be().ple_gather_combine(
                    PER_LAYER,
                    BufId::Tokens,
                    pt.offset as u64,
                    width,
                    (f64::from(ple)).sqrt() as f32,
                    inv_sqrt2,
                    b,
                );
            }
            gprobe("ple_combined", PER_LAYER, 0, ple as usize);
        }
    }

    // E4B may keep its very large token and PLE tables in the streamed tier. A
    // reusable Prefill graph must therefore start only after those request-specific
    // rows have been gathered. Otherwise capture freezes the first prompt's staged
    // rows and silently replays stale embeddings for a later prompt of the same
    // shape. Backends without split-Prefill replay inherit a conservative no-op.
    if !decode
        && be()
            .prefill_body_prepare(tokens, sp, argmax)
            .map_err(|rc| format!("GPU prefill body prepare failed rc={rc}"))?
    {
        be().end()
            .map_err(|rc| format!("GPU forward failed rc={rc}"))?;
        probe_first("gpu after body submit");
        if argmax {
            out.resize(1, 0.0);
            be().read(BufId::Tmp, 0, out);
        } else {
            out.resize(c.vocab_size as usize, 0.0);
            be().read(BufId::Logits, 0, out);
        }
        if std::env::var("IMPARO_KV_SCAN").as_deref() == Ok("1")
            && KvType::k() == KvType::F16
            && KvType::v() == KvType::F16
        {
            wf.kv_scan(sp as usize + b as usize);
        }
        return Ok(());
    }

    // Split the token's command buffer every N layers so the GPU starts on the first
    // layers while the CPU is still encoding the rest.
    //
    // N trades decode speed against memory, because the driver's footprint grows with
    // the number of live command buffers:
    //
    // Measured with one autorelease pool per forward pass (without it every command
    // buffer stays alive for the whole request and the footprint grows with N):
    //
    //   N=0  (1 cb/token)   39.5 tok/s   210.1 MiB
    //   N=2  (21 cb/token)  39.9         218.9    all 21 are live at once
    //   N=7  (6 cb/token)   39.9         210.1    <- the speed of N=2 at the memory of N=0
    //
    // Separate values for decode and prefill. A decode token encodes ~1100 dispatches
    // for ~25 ms of GPU work, so the encode is a real share of the step; a prefill
    // chunk of 512 tokens does far more GPU work per dispatch, where extra command
    // buffers may only add commits. Prefill defaults to 0 but is swept by imparo-tune
    // rather than assumed. Layers, not dispatches, so a flush never lands mid-layer.
    let flush_every =
        crate::gpu_support::flush_layers_bounded(b == 1, wf.plan.layers.len());
    // Decode ramps the first chunks geometrically (1, 2, 4, ... capped at flush_every)
    // instead of a fixed modulus. The GPU is idle until the FIRST commit, so the first
    // chunk's encode is the one that can never be hidden -- at a cadence of 7 that was
    // ~7 layers of encode before any kernel ran, every token. The steady-state chunk
    // stays flush_every, so the knob keeps its meaning: layers per command buffer once
    // the pipeline is full. Prefill keeps the plain modulus: its GPU work per layer is
    // hundreds of times the encode cost, so the ramp has nothing to hide there.
    let ramp = b == 1 && flush_every > 0;
    let mut chunk = 1_usize;
    let mut next_flush = usize::from(ramp);
    // A prefill chunk whose logits are not wanted ends at the last KV-writing layer.
    // The layers after it (E4B: 24..41, `KvSource::SharedWith`) store no KV, and nothing
    // downstream of this chunk reads their activations, so their q/o/FFN/PLE work is
    // dead. mlx-lm gets the same skip from lazy evaluation (its prefill loop evaluates
    // only the cache states); here it is explicit. The final chunk runs every layer for
    // every row, so the pinned logits and the decode hashes cannot move (#127).
    // The last layer that writes KV. Everything after its kv_store -- its own attention,
    // o_proj, FFN and PLE projections, and every later (shared-KV) layer -- is dead work
    // for rows whose logits nobody reads (#127, refined to the operator in #130).
    let last_own = wf
        .plan
        .layers
        .iter()
        .rposition(|l| l.kv_source == KvSource::Own);
    let n_run = if wf.state.logits_wanted {
        n_layers
    } else {
        last_own.map_or(n_layers, |l| l + 1)
    };
    // On the final chunk only the last row's logits are consumed, so from the first
    // shared-KV layer on, the chunk narrows to its last TAIL_ROWS rows (#128): the
    // residual and the per-layer inputs of those rows move to the front, the start
    // position advances, and the tail layers run as a TAIL_ROWS-row batch. 64 rows keep
    // the same GEMM route and st_gemm shape as a 512-row chunk (the token tile ends at 47
    // and the shape choice pads 64 and 512 the same way) and the same attention kernel,
    // so each row's arithmetic is the one it had in the wide chunk. Not taken when the
    // chunk is under 2 x TAIL_ROWS, so the move never overlaps itself.
    const TAIL_ROWS: u32 = 64;
    let tail = if wf.state.logits_wanted && b >= 2 * TAIL_ROWS && last_own.is_some() {
        tail_split(b, tail_align())
    } else {
        None
    };
    let (b_chunk, sp_chunk) = (b, sp);
    let mut b = b;
    let mut sp = sp;
    if layer_skip_log() {
        eprintln!(
            "[imparo] prefill chunk b={b}: running {n_run} of {n_layers} layers (logits_wanted={}, tail={tail:?})",
            wf.state.logits_wanted
        );
    }
    // Set by a layer's PLE tail when it already wrote the NEXT layer's normalised input
    // (or the final norm) in the same dispatch; that norm's own dispatch is then skipped.
    let mut input_norm_ready = false;
    let mut final_norm_ready = false;
    for li in 0..n_run {
        let layer = wf.plan.layers[li];
        let lw = &mw.layers[li];
        let hd = layer.attention.head_dim();
        // Only the full-attention layers scale rope, which is what the CPU path does.
        let is_full = matches!(layer.attention, Attention::Full { .. });
        let rope_freqs = if is_full {
            wf.w.rope_freqs.as_deref()
        } else {
            None
        };
        let (rope_base, rope_dim, window) = match layer.attention {
            Attention::Full {
                rope_base,
                rope_dim,
                ..
            } => (rope_base, rope_dim, 0),
            Attention::Window {
                rope_base,
                rope_dim,
                window,
                ..
            } => (rope_base, rope_dim, window),
            // gemma4 emits Full and Window only; a conv block here means the plan
            // builder produced something this workflow was never written for.
            Attention::Recurrent { .. } => {
                unreachable!("gemma4 layer {li} planned as a recurrent block")
            }
        };
        // The weight-type -> kernel table index, READ FROM THE TENSOR and validated
        // at load (task #17), so a mixed-quant file dispatches each projection on its
        // own kernel.
        //
        // It was called `q4`, which reads as "use the q4 kernel" -- the opposite of
        // what it does. Only F32 and Q4_0 have kernels today, so every call happens to
        // return Q4_0 and the name looked right; it would have stopped being right the
        // moment a third type landed, silently, at the point of use.
        let wkind = |t: &imparo_gguf::weights::Tensor| {
            imparo_gguf::weights::weight_kind(t.ggml_type).expect("validated at load")
                as u32
        };

        #[cfg(feature = "cuda-gate0-capture")]
        let gate0_needs_down_input = gate0_capture.wants(li, CaptureOp::FfnDownInput);
        #[cfg(not(feature = "cuda-gate0-capture"))]
        let gate0_needs_down_input = false;
        // MEGA BLOCK (#141): at decode a backend may run the layer from the FFN input to its end
        // -- FFN, post-FFN norm+add, the per-layer block and its tail (which also produces the
        // next layer's input norm) -- as one dispatch (stage 2a), and with `mega_front_wanted`
        // also the o_proj projection and the sandwich norm in front of it (stage 2, first
        // step). On success the dispatch sequence is skipped and the tail's outputs are
        // recorded exactly as the fused tail records them.
        // The next layer's input norm is the block's to form at the whole-layer level (its
        // first phase norms the residual itself), so the tail leaves it alone there: normed on
        // threadgroup 0 after every other threadgroup had left, it was the dispatch's last
        // microseconds for nothing. The final norm is still the tail's (the lm-head reads it).
        let mega_next = if b == 1 {
            if li + 1 < n_run {
                if be().mega_qkv_wanted() {
                    None
                } else {
                    Some((mw.layers[li + 1].attn_norm.offset, BufId::Cur))
                }
            } else if li + 1 == n_layers && wf.state.logits_wanted {
                Some((mw.output_norm.offset, BufId::X))
            } else {
                None
            }
        } else {
            None
        };
        // Bisect switch: IMPARO_MEGA_SKIP_LAST=1 leaves the last layer on the dispatch path.
        // Read once per process: an env lookup per layer per token is host time on the
        // decode path.
        static MEGA_SKIP_LAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mega_skip_last = *MEGA_SKIP_LAST.get_or_init(|| {
            std::env::var("IMPARO_MEGA_SKIP_LAST").is_ok_and(|v| v == "1")
        });
        let mega_eligible = b == 1
            && ple > 0
            && li != gpu_probe_layer()
            && !gate0_needs_down_input
            && !(mega_skip_last && li + 1 >= n_run);
        // THE LAYER'S MEGA ENTRY (task #158 step 2c): the tail's operands, which are the same
        // whatever the entry starts at; each level below offers this same tail with its own
        // FRONT. `None` here = this layer has no per-layer gate or projection, so no entry.
        let mega_tail = match (lw.inp_gate, lw.proj) {
            (Some(ig), Some(pj)) => Some(imparo_backend::Gemma4MegaLayer {
                gate_kind: wkind(&lw.ffn_gate),
                gate_off: lw.ffn_gate.offset as u64,
                up_kind: wkind(&lw.ffn_up),
                up_off: lw.ffn_up.offset as u64,
                down_kind: wkind(&lw.ffn_down),
                down_off: lw.ffn_down.offset as u64,
                post_ffw_norm_off: lw.post_ffw_norm.offset,
                pg_kind: wkind(&ig),
                pg_off: ig.offset as u64,
                pp_kind: wkind(&pj),
                pp_off: pj.offset as u64,
                post_norm_off: lw.post_norm.offset,
                next_norm_off: mega_next
                    .map_or(imparo_backend::NO_WEIGHT, |(w2, _)| w2),
                out_scale: lw.out_scale,
                n_embd,
                n_ff,
                ple,
                per_layer_off: (li as u32) * ple,
                eps,
                src: BufId::Cur,
                x: BufId::X,
                add: BufId::O,
                g: BufId::G,
                u: BufId::U,
                gate: GATE,
                per_layer: PER_LAYER,
                back: BACK,
                next_out: mega_next.map_or(BufId::X, |(_, out)| out),
                front: None,
            }),
            _ => None,
        };
        // WHOLE LAYER (level 5): the block starts at the residual -- input norm, q/k/v rows,
        // head norm + rope, the cache write, attention, o_proj, the sandwich, FFN, PLE, tails.
        let mega_kv_layer = match layer.kv_source {
            KvSource::Own => li as u32,
            KvSource::SharedWith(src) => src,
        };
        // The cache basis the block must reproduce: the Hadamard widths the quantized
        // modes rotate Q/K and V by (the same resolution the dispatch path makes below).
        let mega_had_k = if KvType::k() == KvType::F16 {
            0
        } else {
            had_nrot("IMPARO_HAD_K", kv_quant_route.key, hd)?
        };
        let mega_had_v = if KvType::v() == KvType::F16 {
            0
        } else {
            had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?
        };
        let mega_qkv_done = mega_eligible
            && be().mega_qkv_wanted()
            && mega_tail.is_some_and(|t| {
                be().mega_layer(&imparo_backend::MegaEntry {
                    layer: imparo_backend::MegaLayer::Gemma4(
                        imparo_backend::Gemma4MegaLayer {
                            front: Some(imparo_backend::MegaFront {
                                wo_kind: wkind(&lw.wo),
                                wo_off: lw.wo.offset as u64,
                                attn: BufId::Attn,
                                attn_in: n_head * hd,
                                post_attn_norm_off: lw.post_attention_norm.offset,
                                ffn_norm_off: lw.ffn_norm.offset,
                                head_dim: hd,
                                attention: Some(imparo_backend::MegaAttn {
                                    kv_layer: mega_kv_layer,
                                    n_heads: n_head,
                                    n_kv,
                                    kv_width: n_kv
                                        * wf.plan.layers[mega_kv_layer as usize]
                                            .attention
                                            .head_dim(),
                                    start_pos: sp,
                                    window,
                                    ring: ring_mask(
                                        wf.plan.layers[mega_kv_layer as usize]
                                            .attention,
                                        wf.state.kv_ring_batch,
                                    ),
                                    q: BufId::Q,
                                    had_k: mega_had_k,
                                    had_v: mega_had_v,
                                }),
                                qkv: Some(imparo_backend::MegaQkv {
                                    wq_kind: wkind(&lw.wq),
                                    wq_off: lw.wq.offset as u64,
                                    wk: lw.wk.map(|w| (wkind(&w), w.offset as u64)),
                                    wv: lw.wv.map(|w| (wkind(&w), w.offset as u64)),
                                    q_norm_off: lw.attn_q_norm.offset,
                                    k_norm_off: lw.attn_k_norm.offset,
                                    in_norm_off: lw.attn_norm.offset,
                                    rope_dim,
                                    rope_base,
                                    freqs: rope_freqs,
                                    k: BufId::K,
                                    v: BufId::V,
                                    layer: li as u32,
                                }),
                            }),
                            ..t
                        },
                    ),
                    n_tok: b,
                })
            });
        if !mega_qkv_done {
            if !input_norm_ready {
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
            }
            input_norm_ready = false;

            if li == gpu_probe_layer() {
                gprobe("attn_norm", BufId::Cur, 0, n_embd as usize);
                gprobe(
                    "attn_norm_last",
                    BufId::Cur,
                    (b as u64 - 1) * n_embd as u64,
                    n_embd as usize,
                );
            }
            be().matmat(
                wkind(&lw.wq),
                lw.wq.offset as u64,
                n_embd,
                n_head * hd,
                BufId::Cur,
                BufId::Q,
                b,
            );
            if li == gpu_probe_layer() {
                gprobe("Qcur", BufId::Q, 0, (n_head * hd) as usize);
            }
            if li == gpu_probe_layer() {
                gprobe("Qcur_full", BufId::Q, 0, (b * n_head * hd) as usize);
            }
            // Quantized-K defense, llama.cpp's own (attn_rot_k): rotate Q and K by an
            // orthonormal blockwise Hadamard before the cache quantization. Scores are
            // invariant -- (Hq)-dot-(Hk) = q-dot-k -- and the rotation spreads each
            // 32-value block's energy so the shared 4-bit scale stops starving the
            // small values. f16 caches skip all of it: bytes untouched.
            let q_hadamard_nrot = if KvType::k() == KvType::F16 {
                0
            } else {
                had_nrot("IMPARO_HAD_K", kv_quant_route.key, hd)?
            };
            if li == gpu_probe_layer() {
                // Preserve the individually observable operations on the requested probe
                // layer; normal execution uses the backend's semantic fusion hook.
                be().rms_norm(
                    BufId::Q,
                    lw.attn_q_norm.offset,
                    hd,
                    eps,
                    b * n_head,
                    hd,
                    0,
                );
                be().rope(BufId::Q, rope_dim, rope_base, hd, n_head, sp, b, rope_freqs);
                if q_hadamard_nrot != 0 {
                    be().hadamard(BufId::Q, b * n_head * hd, q_hadamard_nrot);
                }
            } else {
                be().head_norm_rope_hadamard(
                    BufId::Q,
                    lw.attn_q_norm.offset,
                    hd,
                    eps,
                    n_head,
                    sp,
                    b,
                    rope_dim,
                    rope_base,
                    rope_freqs,
                    q_hadamard_nrot,
                );
            }

            let kv_width = n_kv * hd;
            if let (Some(wk), Some(wv)) = (lw.wk, lw.wv) {
                be().matmat_pair(
                    wkind(&wk),
                    wk.offset as u64,
                    BufId::K,
                    wkind(&wv),
                    wv.offset as u64,
                    BufId::V,
                    n_embd,
                    kv_width,
                    BufId::Cur,
                    b,
                );
                if li == gpu_probe_layer() {
                    gprobe("Kcur_raw", BufId::K, 0, (b * kv_width) as usize);
                    gprobe("Vcur_raw", BufId::V, 0, (b * kv_width) as usize);
                }
                // The quantized-KV rotation (see the Q site). K rotates after rope --
                // the cache stores the final value; V after its norm. The attention
                // output is rotated back right after the attention call below.
                let k_hadamard_nrot = if KvType::k() == KvType::F16 {
                    0
                } else {
                    had_nrot("IMPARO_HAD_K", kv_quant_route.key, hd)?
                };
                let v_hadamard_nrot = if KvType::v() == KvType::F16 {
                    0
                } else {
                    had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?
                };
                if li == gpu_probe_layer() {
                    be().rms_norm(
                        BufId::K,
                        lw.attn_k_norm.offset,
                        hd,
                        eps,
                        b * n_kv,
                        hd,
                        0,
                    );
                    be().rms_norm(
                        BufId::V,
                        imparo_backend::NO_WEIGHT,
                        hd,
                        eps,
                        b * n_kv,
                        hd,
                        0,
                    );
                    be().rope(
                        BufId::K,
                        rope_dim,
                        rope_base,
                        hd,
                        n_kv,
                        sp,
                        b,
                        rope_freqs,
                    );
                    if k_hadamard_nrot != 0 {
                        be().hadamard(BufId::K, b * kv_width, k_hadamard_nrot);
                    }
                    if v_hadamard_nrot != 0 {
                        be().hadamard(BufId::V, b * kv_width, v_hadamard_nrot);
                    }
                } else {
                    be().kv_head_postprocess(
                        BufId::K,
                        BufId::V,
                        lw.attn_k_norm.offset,
                        hd,
                        eps,
                        n_kv,
                        sp,
                        b,
                        rope_dim,
                        rope_base,
                        rope_freqs,
                        k_hadamard_nrot,
                        v_hadamard_nrot,
                    );
                }
                let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
                if li == gpu_probe_layer() {
                    gprobe("Kcur_full", BufId::K, 0, (b * kv_width) as usize);
                    gprobe("Vcur_full", BufId::V, 0, (b * kv_width) as usize);
                }
                be().kv_store(BufId::K, li as u32, kv_width, sp, b, false, ring);
                be().kv_store(BufId::V, li as u32, kv_width, sp, b, true, ring);
            }
        }
        // The layers below read the KV source too (attention, its fallback call).
        let kv_layer = mega_kv_layer;
        if Some(li) == last_own {
            // The chunk's last state write is behind us (#130).
            if !wf.state.logits_wanted {
                break;
            }
            if let Some((r0, bt)) = tail {
                // Only the last row's logits are consumed: narrow to the tail rows before
                // this layer's attention. Live per-row inputs here: the residual, the
                // per-layer inputs, and Q (K/V of every row are already stored).
                be().copy_range(BufId::X, 0, BufId::X, r0 * n_embd, bt * n_embd);
                if ple > 0 {
                    be().copy_range(PER_LAYER, 0, PER_LAYER, r0 * width, bt * width);
                }
                be().copy_range(
                    BufId::Q,
                    0,
                    BufId::Q,
                    r0 * n_head * hd,
                    bt * n_head * hd,
                );
                sp += r0;
                b = bt;
                if layer_skip_log() {
                    eprintln!(
                        "[imparo]   tail inside layer {li}: rows {r0}.. as a {b}-row batch at sp={sp}"
                    );
                }
            }
        }
        if li == gpu_probe_layer() {
            gprobe("Qcur_pos", BufId::Q, 0, (n_head * hd) as usize);
            gprobe(
                "Qcur_pos_last",
                BufId::Q,
                (b as u64 - 1) * (n_head * hd) as u64,
                (n_head * hd) as usize,
            );
        }
        let kvw = n_kv * wf.plan.layers[kv_layer as usize].attention.head_dim();
        let kv_ring = ring_mask(
            wf.plan.layers[kv_layer as usize].attention,
            wf.state.kv_ring_batch,
        );
        // A quantized cache needs a dequant pass before the prefill attention reads
        // it. That belongs to the backend's `attention`, which knows the cache type
        // and the block table -- a model asking to attend should not also have to
        // know how a quantized cache is read. It used to live here, and LFM2 never
        // grew a copy.
        // With the attention phase (level 4) the block starts at Q: the call replaces the
        // attention dispatch, o_proj, the sandwich and everything after it in this layer.
        let mega_attn_done = mega_qkv_done
            || mega_eligible
                && be().mega_attn_wanted()
                && mega_tail.is_some_and(|t| {
                    be().mega_layer(&imparo_backend::MegaEntry {
                        layer: imparo_backend::MegaLayer::Gemma4(
                            imparo_backend::Gemma4MegaLayer {
                                front: Some(imparo_backend::MegaFront {
                                    wo_kind: wkind(&lw.wo),
                                    wo_off: lw.wo.offset as u64,
                                    attn: BufId::Attn,
                                    attn_in: n_head * hd,
                                    post_attn_norm_off: lw.post_attention_norm.offset,
                                    ffn_norm_off: lw.ffn_norm.offset,
                                    head_dim: hd,
                                    attention: Some(imparo_backend::MegaAttn {
                                        kv_layer,
                                        n_heads: n_head,
                                        n_kv,
                                        kv_width: kvw,
                                        start_pos: sp,
                                        window,
                                        ring: kv_ring,
                                        q: BufId::Q,
                                        had_k: mega_had_k,
                                        had_v: mega_had_v,
                                    }),
                                    qkv: None,
                                }),
                                ..t
                            },
                        ),
                        n_tok: b,
                    })
                });
        if !mega_attn_done {
            be().attention(
                kv_layer,
                hd,
                n_head,
                n_kv,
                kvw,
                sp,
                1.0,
                window,
                b,
                scores_needed(sp, b, window),
                kv_ring,
            );
            if li == gpu_probe_layer() {
                gprobe("attn_rot_basis", BufId::Attn, 0, (n_head * hd) as usize);
            }
            // Rotate the attention output back to model space (H is symmetric and
            // orthonormal, so the same kernel inverts). Writing ATTN also kills the
            // half-A mirror marker through haz(), so the wo projection correctly falls
            // back to the cvt pass in quantized modes.
            if KvType::v() != KvType::F16 {
                be().hadamard(
                    BufId::Attn,
                    b * n_head * hd,
                    had_nrot("IMPARO_HAD_V", kv_quant_route.value, hd)?,
                );
            }
            if li == gpu_probe_layer() {
                gprobe("attn", BufId::Attn, 0, (n_head * hd) as usize);
                if std::env::var("IMPARO_GPU_PROBE_FULL").is_ok() {
                    gprobe("attn_full", BufId::Attn, 0, (b * n_head * hd) as usize);
                }
                // per-token checksums: which query tokens diverge under a probe diff
                if std::env::var("IMPARO_GPU_PROBE_TOKENS").is_ok() {
                    for t in 0..b {
                        gprobe(
                            &format!("attn_t{t}"),
                            BufId::Attn,
                            (t as u64) * (n_head * hd) as u64,
                            (n_head * hd) as usize,
                        );
                    }
                }
            }
            if li == gpu_probe_layer() {
                gprobe(
                    "attn_last",
                    BufId::Attn,
                    (b as u64 - 1) * (n_head * hd) as u64,
                    (n_head * hd) as usize,
                );
            }
        }
        let mega_front_done = mega_attn_done
            || mega_eligible
                && be().mega_front_wanted()
                && mega_tail.is_some_and(|t| {
                    be().mega_layer(&imparo_backend::MegaEntry {
                        layer: imparo_backend::MegaLayer::Gemma4(
                            imparo_backend::Gemma4MegaLayer {
                                front: Some(imparo_backend::MegaFront {
                                    wo_kind: wkind(&lw.wo),
                                    wo_off: lw.wo.offset as u64,
                                    attn: BufId::Attn,
                                    attn_in: n_head * hd,
                                    post_attn_norm_off: lw.post_attention_norm.offset,
                                    ffn_norm_off: lw.ffn_norm.offset,
                                    head_dim: hd,
                                    attention: None,
                                    qkv: None,
                                }),
                                ..t
                            },
                        ),
                        n_tok: b,
                    })
                });
        if !mega_front_done {
            be().matmat(
                wkind(&lw.wo),
                lw.wo.offset as u64,
                n_head * hd,
                n_embd,
                BufId::Attn,
                BufId::O,
                b,
            );
            if li == gpu_probe_layer() {
                gprobe("wo_out", BufId::O, 0, n_embd as usize);
                if std::env::var("IMPARO_GPU_PROBE_FULL").is_ok() {
                    gprobe("wo_out_full", BufId::O, 0, (b * n_embd) as usize);
                }
                gprobe(
                    "wo_out_last",
                    BufId::O,
                    (b as u64 - 1) * n_embd as u64,
                    n_embd as usize,
                );
            }
            // A backend may fuse this adjacent residual and FFN-normalization boundary
            // while still materializing O for the later FFN residual. The default trait
            // returns false, so Metal and every other backend retain the exact sequence.
            let fused_dual_norm = be().rms_norm_add_dual_projection(
                BufId::O,
                BufId::X,
                lw.post_attention_norm.offset,
                BufId::O,
                lw.ffn_norm.offset,
                BufId::Cur,
                n_embd,
                eps,
                b,
                n_embd,
                0,
                1.0,
            );
            // CUDA mirrors llama.cpp's native RMSNorm*weight+residual association here.
            // Backends that do not expose that optional, model-agnostic fusion retain the
            // established two-dispatch path. In particular this leaves Metal unchanged.
            let fused_attn_residual = fused_dual_norm
                || be().rms_norm_add(
                    BufId::O,
                    BufId::O,
                    lw.post_attention_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                    BufId::X,
                    1.0,
                );
            if !fused_attn_residual {
                be().rms_norm(
                    BufId::O,
                    lw.post_attention_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                );
            }
            if li == gpu_probe_layer() && !fused_attn_residual {
                gprobe(
                    "attn_post_norm_last",
                    BufId::O,
                    (b as u64 - 1) * n_embd as u64,
                    n_embd as usize,
                );
            }
            if !fused_attn_residual {
                be().add(BufId::O, BufId::X, b * n_embd); // O = attn_out
            }
            if li == gpu_probe_layer() {
                gprobe("attn_out", BufId::O, 0, n_embd as usize);
                if std::env::var("IMPARO_GPU_PROBE_FULL").is_ok() {
                    gprobe("attn_out_full", BufId::O, 0, (b * n_embd) as usize);
                }
            }
            if li == gpu_probe_layer() {
                gprobe(
                    "attn_out_last",
                    BufId::O,
                    (b as u64 - 1) * n_embd as u64,
                    n_embd as usize,
                );
            }

            if !fused_dual_norm {
                be().rms_norm_projection(
                    BufId::Cur,
                    BufId::O,
                    lw.ffn_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                );
            }
            if li == gpu_probe_layer() {
                gprobe(
                    "ffn_norm_last",
                    BufId::Cur,
                    (b as u64 - 1) * n_embd as u64,
                    n_embd as usize,
                );
            }
        }
        #[cfg(feature = "cuda-gate0-capture")]
        gate0_capture.capture_tensor(
            be(),
            decode,
            li,
            CaptureOp::FfnNormInput,
            BufId::Cur,
            0,
            b as usize,
            n_embd as usize,
        )?;
        // The block without its front (level 2), from the FFN input the sandwich norm produced.
        let mega_done = mega_front_done
            || mega_eligible
                && mega_tail.is_some_and(|t| {
                    be().mega_layer(&imparo_backend::MegaEntry {
                        layer: imparo_backend::MegaLayer::Gemma4(
                            imparo_backend::Gemma4MegaLayer { front: None, ..t },
                        ),
                        n_tok: b,
                    })
                });
        if mega_done {
            match mega_next {
                Some((_, BufId::Cur)) => input_norm_ready = true,
                Some(_) => final_norm_ready = true,
                None => {}
            }
        }
        let mut layer_scaled = mega_done;
        if !mega_done {
            // A backend may own the complete gate/up/down sidecar route, but probes
            // and explicit Gate0 capture retain the materialized G/U boundaries.
            let fused_ffn = !gate0_needs_down_input
                && li != gpu_probe_layer()
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
                    BufId::X,
                    b,
                );
            if !fused_ffn {
                let fused_gated = be().matmat_gated(
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
                if !fused_gated {
                    be().matmat(
                        wkind(&lw.ffn_gate),
                        lw.ffn_gate.offset as u64,
                        n_embd,
                        n_ff,
                        BufId::Cur,
                        BufId::G,
                        b,
                    );
                    // Prefill folds activation*up into the up projection write-back. Decode
                    // first offers the pair to the backend's state-selected gated primitive.
                    let fused_prefill = b > 1;
                    if fused_prefill {
                        be().set_epilogue(layer.ffn.activation().epilogue());
                    }
                    be().matmat(
                        wkind(&lw.ffn_up),
                        lw.ffn_up.offset as u64,
                        n_embd,
                        n_ff,
                        BufId::Cur,
                        if fused_prefill { BufId::G } else { BufId::U },
                        b,
                    );
                    if fused_prefill {
                        be().set_epilogue(imparo_backend::Epilogue::None);
                    }
                    if li == gpu_probe_layer() {
                        gprobe("gate_raw_full", BufId::G, 0, (b * n_ff) as usize);
                        gprobe("gate_at_2045", BufId::G, 2045, 8);
                        gprobe("up_at_2045", BufId::U, 2045, 8);
                    }
                    if !fused_prefill {
                        be().act_mul(BufId::G, BufId::U, b * n_ff);
                    }
                }
                if li == gpu_probe_layer() {
                    gprobe("ffn_geglu", BufId::G, 0, 8);
                    gprobe("ffn_G_full", BufId::G, 0, (b * n_ff) as usize);
                    gprobe("ffn_U_full", BufId::U, 0, (b * n_ff) as usize);
                    // The FFN output as the down projection reads it (the half mirror),
                    // sampled at tile corners so a tile or lane mapping fault shows up as
                    // a position rather than only as an aggregate checksum.
                    let spots: Vec<(u32, u32)> =
                        std::env::var("IMPARO_GPU_PROBE_FFN").ok().map_or_else(
                            || vec![(0, 0), (0, 8), (1, 0), (8, 0)],
                            |v| {
                                v.split(';')
                                    .filter_map(|p| {
                                        let (t, c) = p.split_once(',')?;
                                        Some((
                                            t.trim().parse().ok()?,
                                            c.trim().parse().ok()?,
                                        ))
                                    })
                                    .collect()
                            },
                        );
                    gprobe("ffn_mirror_full", BufId::Xh2, 0, ((b * n_ff) / 2) as usize);
                    for &(t, c) in &spots {
                        if t < b && c < n_ff {
                            crate::gpu_support::gprobe_half(
                                &format!("ffn_mirror t{t} c{c}"),
                                BufId::Xh2,
                                u64::from(t) * u64::from(n_ff) + u64::from(c),
                                8,
                            );
                        }
                    }
                }
                #[cfg(feature = "cuda-gate0-capture")]
                gate0_capture.capture_tensor(
                    be(),
                    decode,
                    li,
                    CaptureOp::FfnDownInput,
                    BufId::G,
                    0,
                    b as usize,
                    n_ff as usize,
                )?;
                be().matmat(
                    wkind(&lw.ffn_down),
                    lw.ffn_down.offset as u64,
                    n_ff,
                    n_embd,
                    BufId::G,
                    BufId::X,
                    b,
                );
                if li == gpu_probe_layer() {
                    gprobe("ffn_down_out", BufId::X, 0, (b * n_embd) as usize);
                }
            }
            let projection_follows =
                ple > 0 && lw.inp_gate.is_some() && lw.proj.is_some();
            let prepared_ffn_residual = projection_follows
                && be().rms_norm_add_projection(
                    BufId::X,
                    BufId::X,
                    lw.post_ffw_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                    BufId::O,
                );
            let fused_ffn_residual = prepared_ffn_residual
                || be().rms_norm_add(
                    BufId::X,
                    BufId::X,
                    lw.post_ffw_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                    BufId::O,
                    1.0,
                );
            if !fused_ffn_residual {
                be().rms_norm(
                    BufId::X,
                    lw.post_ffw_norm.offset,
                    n_embd,
                    eps,
                    b,
                    n_embd,
                    0,
                );
            }
            if li == gpu_probe_layer() && !fused_ffn_residual {
                gprobe(
                    "ffn_mlp_last",
                    BufId::X,
                    (b as u64 - 1) * n_embd as u64,
                    n_embd as usize,
                );
            }
            if !fused_ffn_residual {
                be().add(BufId::X, BufId::O, b * n_embd); // X = pe_in
            }
            if li == gpu_probe_layer() {
                gprobe(
                    "pe_in_last",
                    BufId::X,
                    (b as u64 - 1) * n_embd as u64,
                    n_embd as usize,
                );
            }
            if li == gpu_probe_layer() {
                gprobe("pe_in", BufId::X, 0, n_embd as usize);
                if std::env::var("IMPARO_GPU_PROBE_FULL").is_ok() {
                    gprobe("pe_in_full", BufId::X, 0, (b * n_embd) as usize);
                }
            }

            if ple > 0 {
                if let (Some(ig), Some(pj)) = (lw.inp_gate, lw.proj) {
                    if li == gpu_probe_layer() {
                        // Keep this diagnostic boundary observable. Normal execution uses
                        // the backend semantic hook and hides the intermediate scheduling.
                        be().matmat(
                            wkind(&ig),
                            ig.offset as u64,
                            n_embd,
                            ple,
                            BufId::X,
                            GATE,
                            b,
                        );
                        be().act(GATE, b * ple);
                        be().mul_strided(
                            GATE,
                            PER_LAYER,
                            ple,
                            (li as u32) * ple,
                            width,
                            ple,
                            b,
                        );
                        be().matmat(
                            wkind(&pj),
                            pj.offset as u64,
                            ple,
                            n_embd,
                            GATE,
                            BACK,
                            b,
                        );
                    } else {
                        be().ple_project(
                            wkind(&ig),
                            ig.offset as u64,
                            wkind(&pj),
                            pj.offset as u64,
                            n_embd,
                            ple,
                            BufId::X,
                            GATE,
                            PER_LAYER,
                            (li as u32) * ple,
                            width,
                            BACK,
                            b,
                        );
                    }
                    if li == gpu_probe_layer() {
                        gprobe("ple_gate", GATE, 0, 8);
                    }
                    // THE PLE TAIL AND THE NEXT INPUT NORM IN ONE DISPATCH, at one row. The
                    // residual after the PLE feeds nothing but the next layer's input norm (or
                    // the final norm), and a norm dispatch at one row is ~6 us of latency for
                    // ~0.4 us of work; the dual form writes X and the normalised row together.
                    // Bit-identical to the two dispatches: same kernel for the first half, the
                    // same reduction shape for the second. A backend without the fusion, or a
                    // layer whose output aliases its residual, takes the old path below.
                    let next_norm = if b == 1 {
                        if li + 1 < n_run {
                            Some((mw.layers[li + 1].attn_norm.offset, BufId::Cur))
                        } else if li + 1 == n_layers && wf.state.logits_wanted {
                            Some((mw.output_norm.offset, BufId::X))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    let fused_tail = match next_norm {
                        Some((w2, out)) => be().rms_norm_add_dual_projection(
                            BACK,
                            BufId::X,
                            lw.post_norm.offset,
                            BufId::X,
                            w2,
                            out,
                            n_embd,
                            eps,
                            b,
                            n_embd,
                            0,
                            lw.out_scale,
                        ),
                        None => false,
                    };
                    if fused_tail {
                        match next_norm {
                            Some((_, BufId::Cur)) => input_norm_ready = true,
                            Some(_) => final_norm_ready = true,
                            None => {}
                        }
                    }
                    let fused_ple_residual = fused_tail
                        || be().rms_norm_add(
                            BufId::X,
                            BACK,
                            lw.post_norm.offset,
                            n_embd,
                            eps,
                            b,
                            n_embd,
                            0,
                            BufId::X,
                            lw.out_scale,
                        );
                    if !fused_ple_residual {
                        be().rms_norm(
                            BACK,
                            lw.post_norm.offset,
                            n_embd,
                            eps,
                            b,
                            n_embd,
                            0,
                        );
                        if li == gpu_probe_layer() {
                            gprobe("ple_back", BACK, 0, n_embd as usize);
                        }
                        // Fused X = (X + BACK) * out_scale: identical arithmetic to the
                        // add-then-scale pair, one pass over X instead of two.
                        be().add_scale(BufId::X, BACK, lw.out_scale, b * n_embd);
                    }
                    layer_scaled = true;
                }
            }
        }
        if !layer_scaled {
            be().scale(BufId::X, lw.out_scale, b * n_embd);
        }
        if li == gpu_probe_layer() {
            gprobe("l_out0", BufId::X, 0, n_embd as usize);
            if std::env::var("IMPARO_GPU_PROBE_FULL").is_ok() {
                gprobe("l_out_full", BufId::X, 0, (b * n_embd) as usize);
            }
        }
        if li == gpu_probe_layer() {
            gprobe(
                "l_out_last",
                BufId::X,
                (b as u64 - 1) * n_embd as u64,
                n_embd as usize,
            );
        }
        // Hand the GPU a batch of layers to run while the CPU encodes the next ones.
        // A decode token encodes ~1100 dispatches; done as one command buffer the GPU
        // waits for all of that encoding before it starts anything.
        if ramp {
            if li + 1 == next_flush && li + 1 < n_run {
                be().flush();
                chunk = (chunk * 2).min(flush_every);
                next_flush += chunk;
            }
        } else if flush_every > 0 && (li + 1) % flush_every == 0 && li + 1 < n_run {
            be().flush();
        }
    }
    // The mega program (task #153): the layers recorded into one run are encoded here, before
    // anything after the loop reads their output.
    be().mega_program_end();
    gprobe("final_x", BufId::X, 0, n_embd as usize);
    gprobe(
        "final_x_last",
        BufId::X,
        (b as u64 - 1) * n_embd as u64,
        n_embd as usize,
    );
    // last token only: final norm and the tied lm_head, in the SAME command buffer.
    // Splitting it cost a sync plus a round trip of the hidden state per token.
    // Skipped for a non-final prefill chunk (WorkflowState::logits_wanted): nothing reads
    // those logits; the command buffer still ends so the chunk's writes land.
    let logits_wanted = wf.state.logits_wanted;
    if logits_wanted {
        // final_norm_ready: the last layer's tail already wrote the normed row into X
        // (dual form of the sandwich boundary); only the norm dispatch is skipped.
        if !final_norm_ready {
            be().rms_norm(
                BufId::X,
                mw.output_norm.offset,
                n_embd,
                eps,
                1,
                n_embd,
                (b - 1) * n_embd,
            );
        }
        be().matmat_from(
            imparo_gguf::weights::weight_kind(mw.token_embd.ggml_type)
                .expect("validated at load") as u32,
            mw.token_embd.offset as u64,
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
                .map_err(|rc| format!("GPU pipelined step failed rc={rc}"));
        }
        if argmax {
            // The pick runs where the logits already are; only the index crosses back.
            // TMP is free here -- nothing reads it after the layer loop.
            be().argmax(BufId::Logits, BufId::Tmp, c.vocab_size);
        }
    }
    probe_first("gpu after writes");
    be().end()
        .map_err(|rc| format!("GPU forward failed rc={rc}"))?;
    probe_first("gpu after submit");

    if !logits_wanted {
        #[cfg(feature = "cuda-gate0-capture")]
        gate0_capture.finalize()?;
        return Ok(());
    }
    if argmax {
        out.resize(1, 0.0);
        be().read(BufId::Tmp, 0, out); // one u32 in a float's clothing; see caller
    } else {
        out.resize(c.vocab_size as usize, 0.0);
        be().read(BufId::Logits, 0, out);
    }
    #[cfg(feature = "cuda-gate0-capture")]
    gate0_capture.finalize()?;
    // Diagnostic (IMPARO_KV_SCAN=1, f16 cache only): per-layer K/V value structure,
    // read from the cache AFTER the command buffer completed. Prints per layer the
    // row rms, the max |value|, and the largest per-dim column maxima -- the shape
    // that decides how much a shared 32-value quantization scale costs.
    if std::env::var("IMPARO_KV_SCAN").as_deref() == Ok("1")
        && KvType::k() == KvType::F16
        && KvType::v() == KvType::F16
    {
        // The chunk's own extent, not the tail's: the KV written by layers 0..first_shared
        // covers every row of the chunk.
        wf.kv_scan(sp_chunk as usize + b_chunk as usize);
    }
    Ok(())
}

/// Every activation buffer this model needs for a batch of `b`, and which of them may
/// share bytes.
///
/// Q/K/V/ATTN and G/U SHARE memory: a layer runs attention and then the feed-forward,
/// and neither set is live while the other runs. Allocated separately they were the sum
/// of both, which is why the footprint grew ~160 MiB per doubling of ubatch against
/// llama.cpp's 25-50 -- its graph allocator reuses by liveness and this did not.
///
///   attention     Q + K + V + ATTN   40960 bytes a token
///   feed-forward  G + U              81920
///   shared        81920, saving 40960 a token: 21 MiB at ubatch 512, 84 at 2048
pub fn buffer_requirements(
    plan: &ModelPlan,
    b: usize,
    capacity: usize,
) -> Vec<BufferRequirement> {
    let c = &plan.config;
    let n_embd = c.n_embd as usize;
    let n_ff = c.n_ff as usize;
    let ple = plan.embed.per_layer_dim.unwrap_or(0) as usize;
    let n_layers = c.n_layers as usize;
    let head_max = plan
        .layers
        .iter()
        .map(|l| l.attention.head_dim() as usize)
        .max()
        .unwrap_or(0);
    let f = |n: usize| (n * 4) as u64;
    let qsz = f(b * c.n_heads as usize * head_max);
    let ksz = f(b * c.n_kv_heads as usize * head_max);

    // Q, K and V share; ATTN does NOT, and that is not an oversight.
    //
    // `matmat(ATTN -> O)` is the register-tiled GEMM, which reads activations in whole
    // 8-row tiles and cannot mask, so it READS rows past the token count -- the reason
    // `b` is rounded up to a GEMM tile. Those reads were harmless while the padding
    // held stale attention values; with the feed-forward's activations there instead
    // they move the logits. Measured: sharing Q, K or V alone reproduces the baseline
    // exactly, sharing ATTN does not.
    //
    // So the note that rows past the token count "do not affect results" holds for
    // small leftovers, not for arbitrary ones. IMPARO_ARENA_MASK re-runs that test: a
    // bit clear here moves that buffer out of the arena entirely.
    let mask: u32 = std::env::var("IMPARO_ARENA_MASK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x7);
    let shared = |bit: u32| {
        if mask & bit != 0 {
            Placement::Group(0)
        } else {
            Placement::Dedicated
        }
    };
    let need = |id: BufId, bytes: u64, placement: Placement| BufferRequirement {
        id,
        bytes,
        placement,
    };

    let mut reqs = vec![
        need(BufId::X, f(b * n_embd), Placement::Dedicated),
        need(BufId::Cur, f(b * n_embd), Placement::Dedicated),
        need(BufId::O, f(b * n_embd), Placement::Dedicated),
        need(BufId::Q, qsz, shared(1)),
        need(BufId::K, ksz, shared(2)),
        need(BufId::V, ksz, shared(4)),
        need(BufId::Attn, qsz, shared(8)),
        need(BufId::G, f(b * n_ff), Placement::Group(1)),
        need(BufId::U, f(b * n_ff), Placement::Group(1)),
        need(GATE, f(b * ple.max(1)), Placement::Dedicated),
        need(BACK, f(b * n_embd), Placement::Dedicated),
        need(PER_LAYER, f(b * ple * n_layers), Placement::Dedicated),
        need(
            BufId::Logits,
            f(c.vocab_size as usize),
            Placement::Dedicated,
        ),
        need(BufId::Tmp, f(b * n_embd), Placement::Dedicated),
        need(BufId::Tokens, (b * 4) as u64, Placement::Dedicated),
        need(BufId::Pick, 8, Placement::Dedicated),
    ];
    if let Some(row_bytes) = plan.embed.per_layer_row_bytes {
        reqs.push(need(PLE_ROWS, b as u64 * row_bytes, Placement::Dedicated));
    }

    let diagnostic = kvq_mask_on();
    reqs.extend(kv_dequant_scratch_requirements(
        plan,
        capacity,
        KvType::k() != KvType::F16 || diagnostic,
        KvType::v() != KvType::F16 || diagnostic,
    ));
    // The half-precision activation mirrors. n_ff is the widest activation staged
    // through them (the down projection reads it), and U is dead during prefill because
    // the fused epilogue writes G.
    reqs.extend(half_activation_mirror_requirements(b, n_ff, BufId::U));
    // Must match ATTN_MAX_SLICES in the Metal source.
    const MAX_SPLITS: usize = 256;
    reqs.push(need(
        BufId::AttnPart,
        f(c.n_heads as usize * MAX_SPLITS * (head_max + 2)),
        Placement::Dedicated,
    ));
    reqs
}
