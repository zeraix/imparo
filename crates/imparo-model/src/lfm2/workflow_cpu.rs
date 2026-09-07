//! LFM2 forward pass, CPU reference.
//!
//! Correctness oracle for the LFM2 kernels, the same role gemma4's has. It is short
//! because the parts that are not LFM2 -- the cache, the tensor resolver, causal
//! attention, RoPE over heads, the lm_head tail -- come from [`crate::cpu_support`].
//! What is left is the two things LFM2 actually is:
//!
//! ```text
//! every block:   prev = x
//!                cur  = rms_norm(x, operator_norm)
//!                cur  = shortconv(cur)   OR   attention(cur)     <- 22 of 30 / 8 of 30
//!                x    = prev + cur
//!                x    = x + ffn_down(silu(ffn_gate @ rms_norm(x, ffn_norm)) * ffn_up @ ..)
//! tail:          logits = token_embd @ rms_norm(x, token_embd_norm)
//! ```
//!
//! No post-norms, no per-layer embeddings, no output scale -- gemma4 has all three and
//! LFM2 has none of them, which is why they stayed at gemma4's call sites.

use crate::cpu_support::{
    AttnShape, NormW, RopeShape, Tensors, attend, embed_tokens, norm_and_rope, probe,
    probe_layer,
};
use crate::lfm2::Lfm2;
use crate::{Attention, ModelPlan};
use imparo_cpu::ops;
use imparo_gguf::weights::{Tensor, Weights};

/// A block's mixer weights.
///
/// An enum, not a struct of Options: the two arms share NO tensor at all -- an attention
/// block has no convolution kernel and a recurrent block has no Q. Written flat this
/// would be nine fields of which four are always absent, and every read of one would
/// need a comment saying which layers it is valid for.
pub enum MixerW {
    Attention {
        q_norm: NormW,
        k_norm: NormW,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
    },
    ShortConv {
        /// Channel-major with the tap fastest: element (k, ch) is at `ch * kernel + k`,
        /// which is what the GGUF dims `(l_cache, n_embd)` mean.
        ///
        /// A `NormW` because it is the same two-halves fact a norm is: the host path
        /// reads the values, the device path binds the offset, and neither means
        /// anything without knowing they name one tensor. It was a bare `Vec<f32>` and
        /// the device forward had nothing to bind.
        conv: NormW,
        in_proj: Tensor,
        out_proj: Tensor,
    },
}

/// Per-layer weights resolved once.
///
/// TWO norms, against gemma4's seven. Both are pre-norms; LFM2 adds nothing after a
/// residual. The file spells the first `attn_norm` even on a recurrent block, where the
/// reference calls it `operator_norm` -- it norms whatever the block's mixer is.
pub struct LayerW {
    pub(crate) op_norm: NormW,
    pub(crate) ffn_norm: NormW,
    pub(crate) mixer: MixerW,
    pub(crate) ffn_gate: Tensor,
    pub(crate) ffn_up: Tensor,
    pub(crate) ffn_down: Tensor,
}

pub struct ModelW {
    pub(crate) token_embd: Tensor,
    /// Spelled `token_embd_norm` in the file, not `output_norm`: the export name is
    /// wrong and the reference carries the same fix-up.
    pub(crate) output_norm: NormW,
    pub(crate) layers: Vec<LayerW>,
}

/// Resolves LFM2's weights out of one file.
///
/// # Errors
/// When a required tensor is absent, or carries a quant no kernel can read.
pub fn prepare(weights: &Weights, plan: &ModelPlan) -> Result<ModelW, String> {
    let f = Tensors::new(weights);
    let mut layers = Vec::with_capacity(plan.layers.len());
    for l in &plan.layers {
        let i = l.index;
        // The plan decides which mixer a layer is; this reads the tensors that arm
        // needs and NOTHING from the other. A file whose tensors disagree with its
        // head_count_kv array fails here, by name.
        let mixer = if l.attention.is_attention() {
            MixerW::Attention {
                q_norm: f.norm(&format!("blk.{i}.attn_q_norm.weight"))?,
                k_norm: f.norm(&format!("blk.{i}.attn_k_norm.weight"))?,
                wq: f.matmul(&format!("blk.{i}.attn_q.weight"))?,
                wk: f.matmul(&format!("blk.{i}.attn_k.weight"))?,
                wv: f.matmul(&format!("blk.{i}.attn_v.weight"))?,
                wo: f.matmul(&format!("blk.{i}.attn_output.weight"))?,
            }
        } else {
            MixerW::ShortConv {
                conv: f.norm(&format!("blk.{i}.shortconv.conv.weight"))?,
                in_proj: f.matmul(&format!("blk.{i}.shortconv.in_proj.weight"))?,
                out_proj: f.matmul(&format!("blk.{i}.shortconv.out_proj.weight"))?,
            }
        };
        layers.push(LayerW {
            op_norm: f.norm(&format!("blk.{i}.attn_norm.weight"))?,
            ffn_norm: f.norm(&format!("blk.{i}.ffn_norm.weight"))?,
            mixer,
            ffn_gate: f.matmul(&format!("blk.{i}.ffn_gate.weight"))?,
            ffn_up: f.matmul(&format!("blk.{i}.ffn_up.weight"))?,
            ffn_down: f.matmul(&format!("blk.{i}.ffn_down.weight"))?,
        });
    }
    Ok(ModelW {
        token_embd: f.matmul("token_embd.weight")?,
        output_norm: f.norm("token_embd_norm.weight")?,
        layers,
    })
}

#[allow(clippy::too_many_lines)] // one block, read top to bottom; splitting it hides the order
/// Runs ONE chunk of LFM2's forward on the host, leaving the last token's logits in
/// `out`. `start_pos` is the chunk's ABSOLUTE start, which is what the convolution state
/// depends on: it holds the tail of the previous chunk.
///
/// # Errors
/// When the plan and the resolved weights disagree about a layer's block kind.
pub fn batch(
    wf: &mut Lfm2,
    tokens: &[u32],
    start_pos: usize,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let c = wf.plan.config.clone();
    let n_embd = c.n_embd as usize;
    let n_head = c.n_heads as usize;
    let n_kv = c.n_kv_heads as usize;
    let eps = c.norm_eps;
    let b = tokens.len();

    let weights = &wf.weights;
    let mw = &wf.w;
    let kv = &mut wf.state.kv;
    let conv = &mut wf.state.recurrent;
    let plan_layers = &wf.plan.layers;
    let softcap = wf.plan.output.logit_softcap;
    let pl = probe_layer();

    let mut x = vec![0.0_f32; b * n_embd];
    let mut cur = vec![0.0_f32; b * n_embd];
    let mut mix = vec![0.0_f32; b * n_embd];
    embed_tokens(weights, &mw.token_embd, tokens, n_embd, None, &mut x);
    probe(pl.is_some(), "inp_embd", -1, &x[..n_embd]);

    let hd_max = plan_layers
        .iter()
        .map(|l| l.attention.head_dim() as usize)
        .max()
        .unwrap_or(0);
    let n_ff = c.n_ff as usize;
    let mut q = vec![0.0_f32; b * n_head * hd_max];
    let mut attn = vec![0.0_f32; b * n_head * hd_max];
    let mut kbuf = vec![0.0_f32; b * n_kv * hd_max];
    let mut vbuf = vec![0.0_f32; b * n_kv * hd_max];
    let mut bcx = vec![0.0_f32; b * 3 * n_embd];
    let mut gbuf = vec![0.0_f32; b * n_ff];
    let mut upbuf = vec![0.0_f32; b * n_ff];

    for (li, layer) in plan_layers.iter().enumerate() {
        let lw = &mw.layers[li];
        let p = pl == Some(li);

        // ---- operator norm, then whichever mixer this block is -------------------
        cur.copy_from_slice(&x);
        for t in 0..b {
            ops::rms_norm(
                &mut cur[t * n_embd..(t + 1) * n_embd],
                lw.op_norm.as_slice(),
                eps,
            );
        }
        probe(p, "operator_norm", li as i32, &cur[..n_embd]);

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
                    head_dim,
                    rope_base,
                    rope_dim,
                },
            ) => {
                let hd = head_dim as usize;
                let qw = n_head * hd;
                let kw = n_kv * hd;
                ops::mul_mat_batch(weights, wq, &cur, b, &mut q[..b * qw]);
                ops::mul_mat_batch(weights, wk, &cur, b, &mut kbuf[..b * kw]);
                ops::mul_mat_batch(weights, wv, &cur, b, &mut vbuf[..b * kw]);
                let rope = RopeShape {
                    b,
                    heads: n_head,
                    head_dim: hd,
                    start_pos,
                    rope_dim: rope_dim as usize,
                    rope_base,
                    eps,
                };
                // No frequency factors: LFM2 carries no rope_freqs tensor, so every
                // attention layer ropes plainly at base 1e7.
                norm_and_rope(&mut q[..b * qw], q_norm.as_slice(), None, &rope);
                norm_and_rope(
                    &mut kbuf[..b * kw],
                    k_norm.as_slice(),
                    None,
                    &RopeShape {
                        heads: n_kv,
                        ..rope
                    },
                );
                probe(p, "Qcur_pos", li as i32, &q[..qw]);
                kv.append(li, start_pos, b, &kbuf[..b * kw], &vbuf[..b * kw]);
                attend(
                    &q[..b * qw],
                    &mut attn[..b * qw],
                    &kv.k[li],
                    &kv.v[li],
                    kv.width[li],
                    &AttnShape {
                        b,
                        start_pos,
                        n_head,
                        n_kv,
                        head_dim: hd,
                        // LFM2.5 has no sliding window. A file that sets one plans
                        // Attention::Window, and this arm would not match it.
                        window: None,
                        scale: 1.0 / (hd as f32).sqrt(),
                    },
                );
                ops::mul_mat_batch(weights, wo, &attn[..b * qw], b, &mut mix);
            }
            (
                MixerW::ShortConv {
                    conv: conv_w,
                    in_proj,
                    out_proj,
                },
                Attention::Recurrent { .. },
            ) => {
                let kernel = conv_w.w.len() / n_embd;
                ops::mul_mat_batch(weights, in_proj, &cur, b, &mut bcx);
                probe(p, "conv.in_proj", li as i32, &bcx[..3 * n_embd]);
                // `mix` holds c * conv(b * x) here; out_proj turns it back into the
                // residual stream below, exactly as wo does for an attention block.
                ops::shortconv(
                    &bcx,
                    &conv_w.w,
                    &mut conv.r[li],
                    &mut mix[..b * n_embd],
                    n_embd,
                    kernel,
                    b,
                );
                probe(p, "conv.conv", li as i32, &mix[..n_embd]);
                let y = mix.clone();
                ops::mul_mat_batch(weights, out_proj, &y, b, &mut mix);
            }
            // The plan and the resolved weights are built from the same file, so a
            // mismatch here means the plan builder and `prepare` disagree about what
            // block kind layer `li` is -- a defect, not a bad file.
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

        for i in 0..b * n_embd {
            x[i] += mix[i];
        }
        probe(p, "mixer_out", li as i32, &x[..n_embd]);

        // ---- SwiGLU feed-forward, identical on both block kinds -------------------
        cur.copy_from_slice(&x);
        for t in 0..b {
            ops::rms_norm(
                &mut cur[t * n_embd..(t + 1) * n_embd],
                lw.ffn_norm.as_slice(),
                eps,
            );
        }
        ops::mul_mat_batch(weights, &lw.ffn_gate, &cur, b, &mut gbuf);
        ops::mul_mat_batch(weights, &lw.ffn_up, &cur, b, &mut upbuf);
        // SiLU on the GATE, then multiply by up: llama.cpp's LLM_FFN_SILU with
        // LLM_FFN_PAR. Applying it to `up` instead is a silent 1-line wrong answer.
        ops::silu(&mut gbuf);
        ops::mul_into(&mut gbuf, &upbuf);
        ops::mul_mat_batch(weights, &lw.ffn_down, &gbuf, b, &mut mix);
        for i in 0..b * n_embd {
            x[i] += mix[i];
        }
        probe(p, "l_out", li as i32, &x[..n_embd]);
        probe(p, "l_out_last", li as i32, &x[(b - 1) * n_embd..b * n_embd]);
    }

    let last = (b - 1) * n_embd;
    let mut h = x[last..last + n_embd].to_vec();
    *out = crate::cpu_support::logits(
        weights,
        &mw.token_embd,
        &mut h,
        mw.output_norm.as_slice(),
        eps,
        softcap,
    );
    Ok(())
}
