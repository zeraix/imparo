//! Gemma4 forward pass, CPU reference.
//!
//! Correctness oracle, not a fast path. Every backend is checked against this.
//! Instrumentation is gated on IMPARO_LOG and prints per-stage shapes and timings.

use crate::gemma4::Gemma4;
use crate::{Attention, KvSource, ModelPlan};

use crate::cpu_support::{
    AttnShape, NormW, RopeShape, Tensors, attend, embed_tokens, norm_and_rope, probe,
    probe_layer,
};
use imparo_cpu::ops;
use imparo_gguf::weights::{Tensor, Weights};

/// Per-layer weights resolved once.
///
/// SEVEN norms, which is what makes gemma4 unusual: a pre-norm and a post-norm on each of
/// attention and the FFN, a norm on each of Q and K, and one more on the per-layer-
/// embedding branch. Every one carries a device offset as well as its values, which is
/// why `NormW` exists -- written longhand this struct had fourteen fields for seven facts.
pub struct LayerW {
    pub(crate) attn_norm: NormW,
    pub(crate) attn_q_norm: NormW,
    pub(crate) attn_k_norm: NormW,
    pub(crate) post_attention_norm: NormW,
    pub(crate) ffn_norm: NormW,
    pub(crate) post_ffw_norm: NormW,
    pub(crate) post_norm: NormW,
    pub(crate) out_scale: f32,
    pub(crate) wq: Tensor,
    pub(crate) wk: Option<Tensor>,
    pub(crate) wv: Option<Tensor>,
    pub(crate) wo: Tensor,
    pub(crate) ffn_gate: Tensor,
    pub(crate) ffn_up: Tensor,
    pub(crate) ffn_down: Tensor,
    pub(crate) inp_gate: Option<Tensor>,
    pub(crate) proj: Option<Tensor>,
}

pub struct ModelW {
    pub(crate) token_embd: Tensor,
    pub(crate) output_norm: NormW,
    pub(crate) per_layer_model_proj: Option<Tensor>,
    pub(crate) per_layer_token_embd: Option<Tensor>,
    pub(crate) per_layer_proj_norm: NormW,
    pub(crate) rope_freqs: Option<Vec<f32>>,
    pub(crate) layers: Vec<LayerW>,
}

/// Resolves gemma4's weights out of one file.
///
/// # Errors
/// When a required tensor is absent, or carries a quant no kernel can read.
pub fn prepare(weights: &Weights, plan: &ModelPlan) -> Result<ModelW, String> {
    let f = Tensors::new(weights);
    let mut layers = Vec::with_capacity(plan.layers.len());
    for l in &plan.layers {
        let i = l.index;
        let owns = l.kv_source == KvSource::Own;
        layers.push(LayerW {
            attn_norm: f.norm(&format!("blk.{i}.attn_norm.weight"))?,
            attn_q_norm: f.norm(&format!("blk.{i}.attn_q_norm.weight"))?,
            // A sharing layer projects no K, so it has no K norm either. Required
            // when it owns KV, absent otherwise -- not optional-in-general.
            attn_k_norm: if owns {
                f.norm(&format!("blk.{i}.attn_k_norm.weight"))?
            } else {
                NormW::absent()
            },
            post_attention_norm: f
                .norm(&format!("blk.{i}.post_attention_norm.weight"))?,
            ffn_norm: f.norm(&format!("blk.{i}.ffn_norm.weight"))?,
            post_ffw_norm: f.norm(&format!("blk.{i}.post_ffw_norm.weight"))?,
            post_norm: f.norm(&format!("blk.{i}.post_norm.weight"))?,
            out_scale: f.scalar(&format!("blk.{i}.layer_output_scale.weight"), 1.0),
            wq: f.matmul(&format!("blk.{i}.attn_q.weight"))?,
            wk: if owns {
                Some(f.matmul(&format!("blk.{i}.attn_k.weight"))?)
            } else {
                None
            },
            wv: if owns {
                Some(f.matmul(&format!("blk.{i}.attn_v.weight"))?)
            } else {
                None
            },
            wo: f.matmul(&format!("blk.{i}.attn_output.weight"))?,
            ffn_gate: f.matmul(&format!("blk.{i}.ffn_gate.weight"))?,
            ffn_up: f.matmul(&format!("blk.{i}.ffn_up.weight"))?,
            ffn_down: f.matmul(&format!("blk.{i}.ffn_down.weight"))?,
            inp_gate: f.matmul_opt(&format!("blk.{i}.inp_gate.weight"))?,
            proj: f.matmul_opt(&format!("blk.{i}.proj.weight"))?,
        });
    }
    Ok(ModelW {
        token_embd: f.matmul("token_embd.weight")?,
        output_norm: f.norm("output_norm.weight")?,
        per_layer_model_proj: f.matmul_opt("per_layer_model_proj.weight")?,
        per_layer_token_embd: f.get("per_layer_token_embd.weight").ok(),
        per_layer_proj_norm: f.norm_opt("per_layer_proj_norm.weight"),
        rope_freqs: f.norm("rope_freqs.weight").ok().map(|n| n.w),
        layers,
    })
}

/// Runs ONE chunk of gemma4's forward on the host, leaving the last token's logits in
/// `out`. `start_pos` is the chunk's ABSOLUTE start.
///
/// # Errors
/// When a shape disagrees with the plan.
pub fn batch(
    wf: &mut Gemma4,
    tokens: &[u32],
    start_pos: usize,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let c = wf.plan.config.clone();
    let n_embd = c.n_embd as usize;
    let n_head = c.n_heads as usize;
    let n_kv = c.n_kv_heads as usize;
    let n_ff = c.n_ff as usize;
    let eps = c.norm_eps;
    let n_layers = c.n_layers as usize;
    let ple = wf.plan.embed.per_layer_dim.unwrap_or(0) as usize;
    let b = tokens.len();

    let weights = &wf.weights;
    let mw = &wf.w;
    let kv = &mut wf.state.kv;
    let plan_layers = &wf.plan.layers;
    let softcap = wf.plan.output.logit_softcap;
    let pl = probe_layer();

    // token-major scratch
    let mut x = vec![0.0_f32; b * n_embd];
    let mut cur = vec![0.0_f32; b * n_embd];
    let mut per_layer = vec![0.0_f32; b * ple * n_layers];

    embed_tokens(
        weights,
        &mw.token_embd,
        tokens,
        n_embd,
        Some((n_embd as f32).sqrt()),
        &mut x,
    );
    probe(pl.is_some(), "inp_scaled", -1, &x[..n_embd]);

    if ple > 0 {
        if let (Some(pm), Some(pt)) = (mw.per_layer_model_proj, mw.per_layer_token_embd)
        {
            let width = ple * n_layers;
            let mut proj = vec![0.0_f32; b * width];
            ops::mul_mat_batch(weights, &pm, &x, b, &mut proj);
            ops::scale(&mut proj, 1.0 / (n_embd as f32).sqrt());
            let mut emb = vec![0.0_f32; width];
            for t in 0..b {
                for l in 0..n_layers {
                    let o = t * width + l * ple;
                    ops::rms_norm(
                        &mut proj[o..o + ple],
                        mw.per_layer_proj_norm.as_slice(),
                        eps,
                    );
                }
                ops::row(weights, &pt, tokens[t] as usize, &mut emb);
                ops::scale(&mut emb, (ple as f32).sqrt());
                for i in 0..width {
                    per_layer[t * width + i] =
                        (proj[t * width + i] + emb[i]) / 2.0_f32.sqrt();
                }
            }
        }
    }

    let mut q = vec![0.0_f32; b * n_head * 512];
    let mut attn = vec![0.0_f32; b * n_head * 512];
    let mut kbuf = vec![0.0_f32; b * n_kv * 512];
    let mut vbuf = vec![0.0_f32; b * n_kv * 512];
    let mut o = vec![0.0_f32; b * n_embd];
    let mut gbuf = vec![0.0_f32; b * n_ff];
    let mut upbuf = vec![0.0_f32; b * n_ff];
    let mut gate = vec![0.0_f32; b * ple.max(1)];
    let mut back = vec![0.0_f32; b * n_embd];

    #[allow(clippy::needless_range_loop)] // li also indexes mw.layers and the KV
    for li in 0..n_layers {
        let layer = plan_layers[li];
        let lw = &mw.layers[li];
        let p = pl == Some(li);
        let hd = layer.attention.head_dim() as usize;
        let (rope_base, rope_dim, window, is_full) = match layer.attention {
            Attention::Full {
                rope_base,
                rope_dim,
                ..
            } => (rope_base, rope_dim as usize, None, true),
            Attention::Window {
                rope_base,
                rope_dim,
                window,
                ..
            } => (rope_base, rope_dim as usize, Some(window as usize), false),
            // gemma4 emits Full and Window only; a conv block here means the plan
            // builder produced something this workflow was never written for.
            Attention::Recurrent { .. } => {
                unreachable!("gemma4 layer {li} planned as a recurrent block")
            }
        };
        let freq = if is_full {
            mw.rope_freqs.as_deref()
        } else {
            None
        };

        cur.copy_from_slice(&x);
        for t in 0..b {
            ops::rms_norm(
                &mut cur[t * n_embd..(t + 1) * n_embd],
                lw.attn_norm.as_slice(),
                eps,
            );
        }
        probe(p, "attn_norm", li as i32, &cur[..n_embd]);
        probe(
            p,
            "attn_norm_last",
            li as i32,
            &cur[(b - 1) * n_embd..b * n_embd],
        );

        let qw = n_head * hd;
        ops::mul_mat_batch(weights, &lw.wq, &cur, b, &mut q[..b * qw]);
        probe(p, "Qcur", li as i32, &q[..qw]);
        let rope = RopeShape {
            b,
            heads: n_head,
            head_dim: hd,
            start_pos,
            rope_dim,
            rope_base,
            eps,
        };
        norm_and_rope(&mut q[..b * qw], lw.attn_q_norm.as_slice(), freq, &rope);
        probe(p, "Qcur_pos", li as i32, &q[..qw]);

        let kv_layer = match layer.kv_source {
            KvSource::Own => li,
            KvSource::SharedWith(src) => src as usize,
        };
        if let (Some(wk), Some(wv)) = (lw.wk, lw.wv) {
            let kw = n_kv * hd;
            ops::mul_mat_batch(weights, &wk, &cur, b, &mut kbuf[..b * kw]);
            ops::mul_mat_batch(weights, &wv, &cur, b, &mut vbuf[..b * kw]);
            let krope = RopeShape {
                heads: n_kv,
                ..rope
            };
            norm_and_rope(&mut kbuf[..b * kw], lw.attn_k_norm.as_slice(), freq, &krope);
            // V is RMS-normed with NO weight vector, which is gemma4's own and stays
            // here: pushing it into the shared helper would mean a flag per model.
            for t in 0..b {
                for h in 0..n_kv {
                    let off = t * kw + h * hd;
                    ops::rms_norm(&mut vbuf[off..off + hd], None, eps);
                }
            }
            kv.append(li, start_pos, b, &kbuf[..b * kw], &vbuf[..b * kw]);
        }

        // Scale 1.0: gemma4 folds 1/sqrt(head_dim) into the Q norm weights, so
        // applying it again here would halve every score.
        attend(
            &q[..b * qw],
            &mut attn[..b * qw],
            &kv.k[kv_layer],
            &kv.v[kv_layer],
            kv.width[kv_layer],
            &AttnShape {
                b,
                start_pos,
                n_head,
                n_kv,
                head_dim: hd,
                window,
                scale: 1.0,
            },
        );

        ops::mul_mat_batch(weights, &lw.wo, &attn[..b * qw], b, &mut o);
        for t in 0..b {
            let r = t * n_embd..(t + 1) * n_embd;
            ops::rms_norm(&mut o[r.clone()], lw.post_attention_norm.as_slice(), eps);
            for i in r {
                o[i] += x[i];
            }
        }
        probe(p, "attn_out", li as i32, &o[..n_embd]);
        probe(
            p,
            "attn_out_last",
            li as i32,
            &o[(b - 1) * n_embd..b * n_embd],
        );
        let attn_out = o.clone();

        cur.copy_from_slice(&attn_out);
        for t in 0..b {
            ops::rms_norm(
                &mut cur[t * n_embd..(t + 1) * n_embd],
                lw.ffn_norm.as_slice(),
                eps,
            );
        }
        ops::mul_mat_batch(weights, &lw.ffn_gate, &cur, b, &mut gbuf);
        ops::mul_mat_batch(weights, &lw.ffn_up, &cur, b, &mut upbuf);
        ops::gelu(&mut gbuf);
        ops::mul_into(&mut gbuf, &upbuf);
        ops::mul_mat_batch(weights, &lw.ffn_down, &gbuf, b, &mut x);
        for t in 0..b {
            let r = t * n_embd..(t + 1) * n_embd;
            ops::rms_norm(&mut x[r.clone()], lw.post_ffw_norm.as_slice(), eps);
            for i in r {
                x[i] += attn_out[i];
            }
        }
        probe(p, "pe_in", li as i32, &x[..n_embd]);

        if ple > 0 {
            if let (Some(ig), Some(pj)) = (lw.inp_gate, lw.proj) {
                ops::mul_mat_batch(weights, &ig, &x, b, &mut gate[..b * ple]);
                ops::gelu(&mut gate[..b * ple]);
                let width = ple * n_layers;
                for t in 0..b {
                    let g = t * ple..(t + 1) * ple;
                    let src = t * width + li * ple;
                    for (k, i) in g.enumerate() {
                        gate[i] *= per_layer[src + k];
                    }
                }
                ops::mul_mat_batch(weights, &pj, &gate[..b * ple], b, &mut back);
                for t in 0..b {
                    let r = t * n_embd..(t + 1) * n_embd;
                    ops::rms_norm(&mut back[r.clone()], lw.post_norm.as_slice(), eps);
                    for i in r {
                        x[i] += back[i];
                    }
                }
                probe(p, "per_layer_embd_out", li as i32, &back[..n_embd]);
            }
        }
        ops::scale(&mut x, lw.out_scale);
        probe(p, "l_out", li as i32, &x[..n_embd]);
        // The LAST token as well: the logits are taken from it, and an error that only
        // affects later tokens is invisible at token 0.
        probe(p, "l_out_last", li as i32, &x[(b - 1) * n_embd..b * n_embd]);
    }

    // only the LAST token needs logits
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
