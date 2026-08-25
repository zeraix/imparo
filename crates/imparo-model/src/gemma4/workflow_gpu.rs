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

use crate::{Attention, KvSource, ModelPlan};

// GEMMA4'S PRIVATE BUFFER SLOTS. These were named Gate, Back and PerLayer in the shared
// BufId, which put one architecture's per-layer-embedding machinery in every backend's
// vocabulary -- and left LFM2 with no name for its ShortConv projection. The slots are
// generic now and a model says what its own are.
const GATE: BufId = BufId::Model0;
const BACK: BufId = BufId::Model1;
const PER_LAYER: BufId = BufId::Model2;
// Shared device-path machinery; see the charter note in crate::gpu_support.
use crate::gpu_support::{
    BufferRequirement, Placement, be, gprobe, gpu_probe_layer, kvq_mask_on,
    probe_first, scores_needed,
};

/// Slots a KV-owning layer starts with, before any conversation needs more.
///
/// RE-EXPORTED from crate::kv, not redeclared: a second copy of a shared constant is a
/// second chance to disagree with it, which is what happened to NO_WEIGHT.
pub(crate) use crate::kv::KV_FIRST_SLOTS;

/// KV cache storage types (GGML type ids: 1 = f16, 2 = q4_0, 8 = q8_0). User config via
/// --cache-type-k/-v (environment IMPARO_CTK/IMPARO_CTV); f16 is the default and its
/// path is byte-identical to before this feature existed.
pub(crate) use crate::kv::KvType;

use crate::gemma4::Gemma4;
// The pool geometry is shared: see the charter note in crate::kv.
use crate::kv::ring_mask;

/// Rotation width per side: IMPARO_HAD_K / IMPARO_HAD_V = "0" (off) | a power of two |
/// "hd" (full head). Defaults: K "hd" (llama.cpp's attn_rot_k choice; 64 recovers only
/// half the win) and V "128" (drift profile across n=128/900/1500/2000: V=128 max
/// 0.34/0.52/0.63/0.48 -- the flattest worst-case; the fork's V=64 reads 0.29/0.22/
/// 0.89/0.43 here, better typical but a worse tail).
fn had_nrot(var: &str, dflt: &str, hd: u32) -> u32 {
    let v = std::env::var(var).unwrap_or_else(|_| dflt.to_string());
    match v.as_str() {
        "0" => 0,
        "hd" => hd,
        s => s.parse().unwrap_or(64),
    }
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
    be().begin();

    // embedding, scaled by sqrt(n_embd)
    //
    // ONE DISPATCH FOR THE BATCH. This was a `row` per token, so a 512-token batch
    // issued 512 dispatches of n_embd/32 threads each. gather_rows takes the index
    // vector instead. The per-token loop remains the fallback: the backend refuses a
    // width it cannot vectorise, and a backend without the op returns false.
    let embd_scale = (f64::from(n_embd)).sqrt() as f32;
    let wkind = imparo_gguf::weights::weight_kind(mw.token_embd.ggml_type)
        .expect("validated at load") as u32;
    be().write_u32(BufId::Tokens, 0, tokens);
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
            be().write_u32(BufId::Tokens, 0, tokens);
            let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
            be().ple_gather_combine(
                PER_LAYER,
                BufId::Tokens,
                pt.offset as u64,
                width,
                (f64::from(ple)).sqrt() as f32,
                inv_sqrt2,
                b,
            );
            gprobe("ple_combined", PER_LAYER, 0, ple as usize);
        }
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
    let flush_every = be().flush_layers(b == 1) as usize;
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
    for li in 0..n_layers {
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

        if li == gpu_probe_layer() {
            gprobe("attn_norm", BufId::Cur, 0, n_embd as usize);
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
        be().rms_norm(BufId::Q, lw.attn_q_norm.offset, hd, eps, b * n_head, hd, 0);
        be().rope(BufId::Q, rope_dim, rope_base, hd, n_head, sp, b, rope_freqs);
        // Quantized-K defense, llama.cpp's own (attn_rot_k): rotate Q and K by an
        // orthonormal blockwise Hadamard before the cache quantization. Scores are
        // invariant -- (Hq)-dot-(Hk) = q-dot-k -- and the rotation spreads each
        // 32-value block's energy so the shared 4-bit scale stops starving the
        // small values. f16 caches skip all of it: bytes untouched.
        if KvType::k() != KvType::F16 {
            be().hadamard(
                BufId::Q,
                b * n_head * hd,
                had_nrot("IMPARO_HAD_K", "hd", hd),
            );
        }

        let kv_layer = match layer.kv_source {
            KvSource::Own => li as u32,
            KvSource::SharedWith(src) => src,
        };
        let kv_width = n_kv * hd;
        if let (Some(wk), Some(wv)) = (lw.wk, lw.wv) {
            be().matmat(
                wkind(&wk),
                wk.offset as u64,
                n_embd,
                kv_width,
                BufId::Cur,
                BufId::K,
                b,
            );
            be().matmat(
                wkind(&wv),
                wv.offset as u64,
                n_embd,
                kv_width,
                BufId::Cur,
                BufId::V,
                b,
            );
            be().rms_norm(BufId::K, lw.attn_k_norm.offset, hd, eps, b * n_kv, hd, 0);
            be().rms_norm(
                BufId::V,
                imparo_backend::NO_WEIGHT,
                hd,
                eps,
                b * n_kv,
                hd,
                0,
            );
            be().rope(BufId::K, rope_dim, rope_base, hd, n_kv, sp, b, rope_freqs);
            // The quantized-KV rotation (see the Q site). K rotates after rope --
            // the cache stores the final value; V after its norm. The attention
            // output is rotated back right after the attention call below.
            if KvType::k() != KvType::F16 {
                be().hadamard(
                    BufId::K,
                    b * kv_width,
                    had_nrot("IMPARO_HAD_K", "hd", hd),
                );
            }
            if KvType::v() != KvType::F16 {
                be().hadamard(
                    BufId::V,
                    b * kv_width,
                    had_nrot("IMPARO_HAD_V", "128", hd),
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
        if li == gpu_probe_layer() {
            gprobe("Qcur_pos", BufId::Q, 0, (n_head * hd) as usize);
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
        be().attention(
            kv_layer,
            hd,
            n_head,
            n_kv,
            kvw,
            sp,
            window,
            b,
            scores_needed(sp, b, window),
            kv_ring,
        );
        // Rotate the attention output back to model space (H is symmetric and
        // orthonormal, so the same kernel inverts). Writing ATTN also kills the
        // half-A mirror marker through haz(), so the wo projection correctly falls
        // back to the cvt pass in quantized modes.
        if KvType::v() != KvType::F16 {
            be().hadamard(
                BufId::Attn,
                b * n_head * hd,
                had_nrot("IMPARO_HAD_V", "128", hd),
            );
        }
        if li == gpu_probe_layer() {
            gprobe("attn", BufId::Attn, 0, (n_head * hd) as usize);
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
        }
        // Norm and residual add stay SEPARATE.
        //
        // Fusing them into one dispatch was implemented and measured twice: decode
        // 37.8 -> 37.0, prefill 504.8 -> 505.1. The norm runs one threadgroup per row,
        // so at decode a single threadgroup does the add that `add` spreads over many;
        // at prefill there are rows to spare but the pair is only 0.4% of the time, so
        // there is nothing to win. `imparo_metal_rms_norm_add` is kept for a model
        // where these are a larger share.
        be().rms_norm(
            BufId::O,
            lw.post_attention_norm.offset,
            n_embd,
            eps,
            b,
            n_embd,
            0,
        );
        be().add(BufId::O, BufId::X, b * n_embd); // O = attn_out
        if li == gpu_probe_layer() {
            gprobe("attn_out", BufId::O, 0, n_embd as usize);
        }
        if li == gpu_probe_layer() {
            gprobe(
                "attn_out_last",
                BufId::O,
                (b as u64 - 1) * n_embd as u64,
                n_embd as usize,
            );
        }

        be().rms_norm_from(
            BufId::Cur,
            BufId::O,
            lw.ffn_norm.offset,
            n_embd,
            eps,
            b,
            n_embd,
            0,
        );
        be().matmat(
            wkind(&lw.ffn_gate),
            lw.ffn_gate.offset as u64,
            n_embd,
            n_ff,
            BufId::Cur,
            BufId::G,
            b,
        );
        // The up projection writes G = gelu(G) * up, folding in the gated activation
        // and removing the separate gelu_mul pass. Only the batched prefill kernel has
        // the epilogue; decode keeps the two-step form.
        // PREFILL ONLY. Both kernels carry the epilogue, but at decode it measured
        // 38.2-38.6 -> 37.9-38.2: one token means a read-modify-write per output row
        // inside the GEMV, where the standalone gelu_mul is a wide vectorised pass.
        let fused = b > 1;
        // WHETHER to fuse, not WHICH activation: which one is a function constant the
        // pipelines were specialised with, chosen from the plan at bring-up. A workflow
        // that hardcoded GELU here gave a SwiGLU model plausible wrong numbers.
        if fused {
            be().set_epilogue(layer.ffn.activation().epilogue());
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
        }
        if li == gpu_probe_layer() {
            gprobe("gate_raw_full", BufId::G, 0, (b * n_ff) as usize);
            gprobe("gate_at_2045", BufId::G, 2045, 8);
            gprobe("up_at_2045", BufId::U, 2045, 8);
        }
        if !fused {
            be().act_mul(BufId::G, BufId::U, b * n_ff);
        }
        if li == gpu_probe_layer() {
            gprobe("ffn_geglu", BufId::G, 0, 8);
            gprobe("ffn_G_full", BufId::G, 0, (b * n_ff) as usize);
            gprobe("ffn_U_full", BufId::U, 0, (b * n_ff) as usize);
        }
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
        be().rms_norm(BufId::X, lw.post_ffw_norm.offset, n_embd, eps, b, n_embd, 0);
        be().add(BufId::X, BufId::O, b * n_embd); // X = pe_in
        let mut layer_scaled = false;
        if li == gpu_probe_layer() {
            gprobe("pe_in", BufId::X, 0, n_embd as usize);
        }

        if ple > 0 {
            if let (Some(ig), Some(pj)) = (lw.inp_gate, lw.proj) {
                be().matmat(
                    wkind(&ig),
                    ig.offset as u64,
                    n_embd,
                    ple,
                    BufId::X,
                    GATE,
                    b,
                );
                // Applies the process activation, which for gemma4 is GELU -- the
                // only model with a per-layer-embedding gate, and GELU throughout.
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
                be().matmat(wkind(&pj), pj.offset as u64, ple, n_embd, GATE, BACK, b);
                if li == gpu_probe_layer() {
                    gprobe("ple_gate", GATE, 0, 8);
                }
                be().rms_norm(BACK, lw.post_norm.offset, n_embd, eps, b, n_embd, 0);
                if li == gpu_probe_layer() {
                    gprobe("ple_back", BACK, 0, n_embd as usize);
                }
                // Fused X = (X + BACK) * out_scale: identical arithmetic to the
                // add-then-scale pair, one pass over X instead of two.
                be().add_scale(BufId::X, BACK, lw.out_scale, b * n_embd);
                layer_scaled = true;
            }
        }
        if !layer_scaled {
            be().scale(BufId::X, lw.out_scale, b * n_embd);
        }
        if li == gpu_probe_layer() {
            gprobe("l_out0", BufId::X, 0, n_embd as usize);
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
            if li + 1 == next_flush && li + 1 < n_layers {
                be().flush();
                chunk = (chunk * 2).min(flush_every);
                next_flush += chunk;
            }
        } else if flush_every > 0 && (li + 1) % flush_every == 0 && li + 1 < n_layers {
            be().flush();
        }
    }
    gprobe("final_x", BufId::X, 0, n_embd as usize);
    gprobe(
        "final_x_last",
        BufId::X,
        (b as u64 - 1) * n_embd as u64,
        n_embd as usize,
    );
    // last token only: final norm and the tied lm_head, in the SAME command buffer.
    // Splitting it cost a sync plus a round trip of the hidden state per token.
    be().rms_norm(
        BufId::X,
        mw.output_norm.offset,
        n_embd,
        eps,
        1,
        n_embd,
        (b - 1) * n_embd,
    );
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
    if argmax {
        // The pick runs where the logits already are; only the index crosses back.
        // TMP is free here -- nothing reads it after the layer loop.
        be().argmax(BufId::Logits, BufId::Tmp, c.vocab_size);
    }
    probe_first("gpu after writes");
    be().end()
        .map_err(|rc| format!("metal forward failed rc={rc}"))?;
    probe_first("gpu after submit");

    if argmax {
        out.resize(1, 0.0);
        be().read(BufId::Tmp, 0, out); // one u32 in a float's clothing; see caller
    } else {
        out.resize(c.vocab_size as usize, 0.0);
        be().read(BufId::Logits, 0, out);
    }
    // Diagnostic (IMPARO_KV_SCAN=1, f16 cache only): per-layer K/V value structure,
    // read from the cache AFTER the command buffer completed. Prints per layer the
    // row rms, the max |value|, and the largest per-dim column maxima -- the shape
    // that decides how much a shared 32-value quantization scale costs.
    if std::env::var("IMPARO_KV_SCAN").as_deref() == Ok("1")
        && KvType::k() == KvType::F16
        && KvType::v() == KvType::F16
    {
        wf.kv_scan(sp as usize + b as usize);
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
    ];

    // Quantized KV: the half scratch one layer's cache is dequantised into before its
    // prefill attention. Sized for the largest layer at full capacity; f16 mode
    // allocates nothing and takes none of these paths.
    if KvType::k() != KvType::F16 || KvType::v() != KvType::F16 || kvq_mask_on() {
        let scratch =
            (capacity.max(KV_FIRST_SLOTS) * c.n_kv_heads as usize * head_max * 2)
                as u64;
        if KvType::k() != KvType::F16 || kvq_mask_on() {
            reqs.push(need(BufId::Kdq, scratch, Placement::Dedicated));
        }
        if KvType::v() != KvType::F16 || kvq_mask_on() {
            reqs.push(need(BufId::Vdq, scratch, Placement::Dedicated));
        }
    }
    // XH and XH2: the half-precision mirrors IMPARO_HALF_A stages activations through.
    // They ALIAS U's pages and cost zero memory -- U is dead during prefill, because the
    // fused epilogue writes G, and the mirrors are used only at prefill.
    //
    // Two slots rather than one: the fused up epilogue writes G's mirror while the SAME
    // dispatch is still reading CUR's mirror out of XH, so the two cannot share bytes.
    // Slot 1 is skipped when it would not fit in U's region, and the backend then keeps
    // converting G with the cvt pass -- slower, and correct.
    reqs.push(need(
        BufId::Xh,
        (b * n_ff * 2) as u64,
        Placement::Within {
            host: BufId::U,
            slot: 0,
        },
    ));
    reqs.push(need(
        BufId::Xh2,
        (b * n_ff * 2) as u64,
        Placement::Within {
            host: BufId::U,
            slot: 1,
        },
    ));
    // Must match ATTN_MAX_SLICES in the Metal source.
    const MAX_SPLITS: usize = 256;
    reqs.push(need(
        BufId::AttnPart,
        f(c.n_heads as usize * MAX_SPLITS * (head_max + 2)),
        Placement::Dedicated,
    ));
    reqs
}
