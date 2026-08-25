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

use crate::gpu_support::{BufferRequirement, Placement, be, gpu_probe_layer, gprobe,
    scores_needed};
use crate::kv::ring_mask;
use crate::lfm2::Lfm2;
use crate::lfm2::workflow_cpu::MixerW;
use crate::ModelPlan;

/// LFM2'S PRIVATE BUFFER SLOTS.
///
/// `bcx` is the ShortConv input projection: n_embd -> 3 * n_embd, carrying b, c and x
/// concatenated. gemma4 has no such buffer, and before the shared slots were generic there
/// was no name for it -- BufId's model-private range was three entries called Gate, Back
/// and PerLayer, which are gemma4's per-layer-embedding buffers.
pub const BCX: BufId = BufId::Model0;

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
    _capacity: usize,
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
    vec![
        need(BufId::X, f(b * n_embd), Placement::Dedicated),
        need(BufId::Cur, f(b * n_embd), Placement::Dedicated),
        need(BufId::O, f(b * n_embd), Placement::Dedicated),
        need(BufId::Q, f(b * c.n_heads as usize * head_max), Placement::Group(0)),
        need(BufId::K, f(b * c.n_kv_heads as usize * head_max), Placement::Group(0)),
        need(BufId::V, f(b * c.n_kv_heads as usize * head_max), Placement::Group(0)),
        // ATTN does not share, for the reason gemma4 records at its own list: the
        // register-tiled GEMM reads whole 8-row tiles and cannot mask, so it reads past
        // the token count, and what those rows hold changes the logits.
        need(BufId::Attn, f(b * c.n_heads as usize * head_max), Placement::Dedicated),
        need(BCX, f(b * 3 * n_embd), Placement::Group(0)),
        need(BufId::G, f(b * n_ff), Placement::Group(1)),
        need(BufId::U, f(b * n_ff), Placement::Group(1)),
        need(BufId::Logits, f(c.vocab_size as usize), Placement::Dedicated),
        need(BufId::Tmp, f(b * n_embd), Placement::Dedicated),
        need(BufId::Tokens, (b * 4) as u64, Placement::Dedicated),
        need(
            BufId::AttnPart,
            f(c.n_heads as usize * MAX_SPLITS * (head_max + 2)),
            Placement::Dedicated,
        ),
    ]
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
    let n_embd = c.n_embd;
    let n_head = c.n_heads;
    let n_kv = c.n_kv_heads;
    let n_ff = c.n_ff;
    let eps = c.norm_eps;
    let b = u32::try_from(tokens.len()).map_err(|_| "batch too large")?;
    let sp = u32::try_from(start_pos).map_err(|_| "start_pos too large")?;
    let recur = wf.plan.recurrent_layout();
    let act = wf.plan.layers[0].ffn.activation().epilogue();

    let wkind = |t: &imparo_gguf::weights::Tensor| {
        imparo_gguf::weights::weight_kind(t.ggml_type).expect("validated at load") as u32
    };

    be().begin();

    // ---- embeddings. No scale: LFM2 sets scale_by_sqrt_embd false. ------------------
    be().write_u32(BufId::Tokens, 0, tokens);
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
            be().row(embd_kind, embd_off, n_embd, tok, 1.0, BufId::X, t as u32 * n_embd);
        }
    }
    gprobe("inp_embd", BufId::X, 0, n_embd as usize);

    let flush_every = be().flush_layers(b == 1) as usize;
    for (li, (&layer, &(r_off, _, _, _))) in wf
        .plan
        .layers
        .clone()
        .iter()
        .zip(recur.iter())
        .enumerate()
    {
        let lw = &wf.w.layers[li];

        // ---- operator norm, then whichever mixer this block is ---------------------
        be().rms_norm_from(
            BufId::Cur, BufId::X, lw.op_norm.offset, n_embd, eps, b, n_embd, 0,
        );
        if li == gpu_probe_layer() {
            gprobe("operator_norm", BufId::Cur, 0, n_embd as usize);
        }

        match (&lw.mixer, layer.attention) {
            (
                MixerW::Attention { q_norm, k_norm, wq, wk, wv, wo },
                crate::Attention::Full { head_dim: hd, rope_base, rope_dim },
            ) => {
                let qw = n_head * hd;
                let kw = n_kv * hd;
                be().matmat(wkind(wq), wq.offset as u64, n_embd, qw, BufId::Cur, BufId::Q, b);
                be().matmat(wkind(wk), wk.offset as u64, n_embd, kw, BufId::Cur, BufId::K, b);
                be().matmat(wkind(wv), wv.offset as u64, n_embd, kw, BufId::Cur, BufId::V, b);
                // Per-head norm then rope, on Q and K. No frequency factors: LFM2
                // carries no rope_freqs tensor, so every layer ropes plainly.
                be().rms_norm(BufId::Q, q_norm.offset, hd, eps, b * n_head, hd, 0);
                be().rms_norm(BufId::K, k_norm.offset, hd, eps, b * n_kv, hd, 0);
                be().rope(BufId::Q, rope_dim, rope_base, hd, n_head, sp, b, None);
                be().rope(BufId::K, rope_dim, rope_base, hd, n_kv, sp, b, None);
                // softmax(q.k / sqrt(d)) == softmax((q / sqrt(d)).k), so the scale is
                // applied to Q once here. `Backend::attention` takes no scale because no
                // model needed one: gemma4's is already folded into its q_norm weights
                // in the FILE. Invisible at one token -- a softmax over a single element
                // is 1 whatever the scale -- which is exactly how this read correct at
                // n=1 and wrong at n=4.
                be().scale(BufId::Q, 1.0 / (hd as f32).sqrt(), b * qw);
                if li == gpu_probe_layer() {
                    gprobe("Qcur_pos", BufId::Q, 0, qw as usize);
                }
                let ring = ring_mask(layer.attention, wf.state.kv_ring_batch);
                be().kv_store(BufId::K, li as u32, kw, sp, b, false, ring);
                be().kv_store(BufId::V, li as u32, kw, sp, b, true, ring);
                be().attention(
                    li as u32, hd, n_head, n_kv, kw, sp,
                    // LFM2.5 carries no sliding window. A file that set one would plan
                    // Attention::Window, and this arm would not match it.
                    0,
                    b,
                    scores_needed(sp, b, 0),
                    ring,
                );
                be().matmat(wkind(wo), wo.offset as u64, qw, n_embd, BufId::Attn, BufId::O, b);
            }
            (
                MixerW::ShortConv { conv, in_proj, out_proj },
                crate::Attention::Recurrent { .. },
            ) => {
                let kern = u32::try_from(conv.w.len() / n_embd as usize)
                    .map_err(|_| "conv kernel width")?;
                be().matmat(
                    wkind(in_proj), in_proj.offset as u64, n_embd, 3 * n_embd,
                    BufId::Cur, BCX, b,
                );
                if li == gpu_probe_layer() {
                    gprobe("conv.in_proj", BCX, 0, (3 * n_embd) as usize);
                }
                // A checkpoint boundary inside this batch: write the state as of that
                // position aside BEFORE the advance overwrites the pre-batch history,
                // which the answer needs whenever the boundary is nearer than `kern - 1`
                // tokens in. One dispatch of `n_embd` threads; the batch runs on.
                if let Some(k) = wf.state.recur_snap {
                    be().shortconv_snapshot(
                        BCX, BufId::Recur, r_off, BufId::RecurSnap, r_off,
                        n_embd, kern, k,
                    );
                }
                // ATTN is the scratch: on a recurrent layer nothing attends, and it is
                // sized n_head * head_dim = 2048, exactly the model width.
                be().shortconv(
                    BCX, conv.offset, BufId::Recur, r_off, BufId::Attn,
                    n_embd, kern, b,
                );
                if li == gpu_probe_layer() {
                    gprobe("conv.conv", BufId::Attn, 0, n_embd as usize);
                }
                be().matmat(
                    wkind(out_proj), out_proj.offset as u64, n_embd, n_embd,
                    BufId::Attn, BufId::O, b,
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
        be().add(BufId::X, BufId::O, b * n_embd);
        if li == gpu_probe_layer() {
            gprobe("mixer_out", BufId::X, 0, n_embd as usize);
        }

        // ---- SwiGLU feed-forward, identical on both block kinds ---------------------
        be().rms_norm_from(
            BufId::Cur, BufId::X, lw.ffn_norm.offset, n_embd, eps, b, n_embd, 0,
        );
        be().matmat(
            wkind(&lw.ffn_gate), lw.ffn_gate.offset as u64, n_embd, n_ff,
            BufId::Cur, BufId::G, b,
        );
        // Fuse at prefill only: at one token the epilogue is a read-modify-write per
        // output row inside the GEMV, where the standalone pass is wide and vectorised.
        let fused = b > 1;
        if fused {
            be().set_epilogue(act);
        }
        be().matmat(
            wkind(&lw.ffn_up), lw.ffn_up.offset as u64, n_embd, n_ff,
            BufId::Cur, if fused { BufId::G } else { BufId::U }, b,
        );
        if fused {
            be().set_epilogue(imparo_backend::Epilogue::None);
        } else {
            be().act_mul(BufId::G, BufId::U, b * n_ff);
        }
        be().matmat(
            wkind(&lw.ffn_down), lw.ffn_down.offset as u64, n_ff, n_embd,
            BufId::G, BufId::O, b,
        );
        be().add(BufId::X, BufId::O, b * n_embd);
        if li == gpu_probe_layer() {
            gprobe("l_out", BufId::X, 0, n_embd as usize);
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
        if flush_every > 0 && (li + 1) % flush_every == 0 {
            be().flush();
        }
    }

    // ---- last token only: the final norm and the tied lm_head, same command buffer --
    be().rms_norm(
        BufId::X, wf.w.output_norm.offset, n_embd, eps, 1, n_embd, (b - 1) * n_embd,
    );
    be().matmat_from(
        embd_kind, embd_off, n_embd, c.vocab_size, BufId::X, BufId::Logits, 1, b - 1,
    );
    if let Some(cap) = wf.plan.output.logit_softcap {
        be().softcap(BufId::Logits, cap, c.vocab_size);
    }
    if argmax {
        be().argmax(BufId::Logits, BufId::Tmp, c.vocab_size);
    }
    be().end().map_err(|rc| format!("lfm2 forward failed rc={rc}"))?;

    if argmax {
        out.resize(1, 0.0);
        be().read(BufId::Tmp, 0, out);
    } else {
        out.resize(c.vocab_size as usize, 0.0);
        be().read(BufId::Logits, 0, out);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BCX, buffer_requirements};
    use crate::{
        Activation, Attention, EmbedPlan, Ffn, KvSource, LayerPlan, ModelConfig, ModelPlan,
        OutputPlan,
    };
    use crate::gpu_support::Placement;
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
            },
            layers,
            output: OutputPlan {
                final_norm: true,
                logit_softcap: None,
                tied_embeddings: true,
            },
        }
    }

    fn find(reqs: &[super::BufferRequirement], id: BufId) -> Option<super::BufferRequirement> {
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
        assert_eq!(find(&reqs, BufId::Q).unwrap().bytes, (b * 32 * 64 * 4) as u64);
        assert_eq!(find(&reqs, BufId::K).unwrap().bytes, (b * 8 * 64 * 4) as u64);
        assert_eq!(find(&reqs, BufId::V).unwrap().bytes, (b * 8 * 64 * 4) as u64);
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
}
