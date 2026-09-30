//! Qwen3 forward pass, CPU reference.
//!
//! The correctness oracle for the Qwen3 kernels. One block, repeated:
//!
//! ```text
//! every block:   prev = x
//!                cur  = rms_norm(x, attn_norm)
//!                q,k,v = wq @ cur, wk @ cur, wv @ cur
//!                q    = rope(rms_norm_per_head(q, attn_q_norm))
//!                k    = rope(rms_norm_per_head(k, attn_k_norm))
//!                x    = prev + wo @ attention(q, k, v)
//!                x    = x + ffn_down(silu(ffn_gate @ n) * (ffn_up @ n)),
//!                       n = rms_norm(x, ffn_norm)
//! tail:          logits = head @ rms_norm(x, output_norm)
//! ```
//!
//! THE PER-HEAD NORM IS THE STEP THAT LOOKS OPTIONAL AND IS NOT. `attn_q_norm` and
//! `attn_k_norm` are one head-wide vector each, applied to EVERY head before the rotation
//! -- Qwen3's own graph (llama.cpp `src/models/qwen3.cpp`) does the same, and a forward
//! that skips them still writes fluent text with wrong logits, which no output check that
//! reads text would catch.
//!
//! Two smaller facts, both read from the file rather than assumed:
//!
//!   - The head dim comes from the plan (`attention.key_length`), so a sibling whose
//!     embedding width is not `heads * head_dim` runs unchanged here.
//!   - The head is tied on files that carry no `output.weight`; `prepare` resolves which
//!     tensor the tail multiplies, and the forward has one spelling either way.

use crate::cpu_support::{
    AttnShape, NormW, RopeShape, Tensors, attend, embed_tokens, norm_and_rope, probe,
    probe_layer,
};
use crate::qwen3::Qwen3;
use crate::{Attention, ModelPlan};
use imparo_cpu::ops;
use imparo_gguf::weights::{Tensor, Weights};

pub struct LayerW {
    pub(crate) attn_norm: NormW,
    pub(crate) ffn_norm: NormW,
    pub(crate) q_norm: NormW,
    pub(crate) k_norm: NormW,
    pub(crate) wq: Tensor,
    pub(crate) wk: Tensor,
    pub(crate) wv: Tensor,
    pub(crate) wo: Tensor,
    pub(crate) ffn_gate: Tensor,
    pub(crate) ffn_up: Tensor,
    pub(crate) ffn_down: Tensor,
}

pub struct ModelW {
    pub(crate) token_embd: Tensor,
    /// What the tail multiplies: `output.weight` when the file carries one, and the
    /// embedding table when it does not. Resolved once, so neither path re-decides.
    pub(crate) head: Tensor,
    pub(crate) output_norm: NormW,
    pub(crate) layers: Vec<LayerW>,
}

/// Resolves Qwen3's weights out of one file.
///
/// # Errors
/// When a required tensor is absent, or carries a quant no kernel can read.
pub fn prepare(weights: &Weights, plan: &ModelPlan) -> Result<ModelW, String> {
    let f = Tensors::new(weights);
    let mut layers = Vec::with_capacity(plan.layers.len());
    for l in &plan.layers {
        let i = l.index;
        layers.push(LayerW {
            attn_norm: f.norm(&format!("blk.{i}.attn_norm.weight"))?,
            ffn_norm: f.norm(&format!("blk.{i}.ffn_norm.weight"))?,
            q_norm: f.norm(&format!("blk.{i}.attn_q_norm.weight"))?,
            k_norm: f.norm(&format!("blk.{i}.attn_k_norm.weight"))?,
            wq: f.matmul(&format!("blk.{i}.attn_q.weight"))?,
            wk: f.matmul(&format!("blk.{i}.attn_k.weight"))?,
            wv: f.matmul(&format!("blk.{i}.attn_v.weight"))?,
            wo: f.matmul(&format!("blk.{i}.attn_output.weight"))?,
            ffn_gate: f.matmul(&format!("blk.{i}.ffn_gate.weight"))?,
            ffn_up: f.matmul(&format!("blk.{i}.ffn_up.weight"))?,
            ffn_down: f.matmul(&format!("blk.{i}.ffn_down.weight"))?,
        });
    }
    let token_embd = f.matmul("token_embd.weight")?;
    let head = if plan.output.tied_embeddings {
        f.matmul("token_embd.weight")?
    } else {
        f.matmul("output.weight")?
    };
    Ok(ModelW {
        token_embd,
        head,
        output_norm: f.norm("output_norm.weight")?,
        layers,
    })
}

/// Runs ONE chunk of Qwen3's forward on the host, leaving the last token's logits in
/// `out`. `start_pos` is the chunk's ABSOLUTE start, which is what the KV cache and the
/// rotation are indexed by.
///
/// # Errors
/// When a layer's plan entry is not full attention, which this architecture's plan
/// builder never produces.
pub fn batch(
    wf: &mut Qwen3,
    tokens: &[u32],
    start_pos: usize,
    out: &mut Vec<f32>,
) -> Result<(), String> {
    let c = wf.plan.config.clone();
    let (n_embd, n_head, n_kv) =
        (c.n_embd as usize, c.n_heads as usize, c.n_kv_heads as usize);
    let (n_ff, eps, b) = (c.n_ff as usize, c.norm_eps, tokens.len());

    let weights = &wf.weights;
    let mw = &wf.w;
    let kv = &mut wf.state.kv;
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
    let mut q = vec![0.0_f32; b * n_head * hd_max];
    let mut attn = vec![0.0_f32; b * n_head * hd_max];
    let mut kbuf = vec![0.0_f32; b * n_kv * hd_max];
    let mut vbuf = vec![0.0_f32; b * n_kv * hd_max];
    let mut gbuf = vec![0.0_f32; b * n_ff];
    let mut upbuf = vec![0.0_f32; b * n_ff];

    for (li, layer) in plan_layers.iter().enumerate() {
        let lw = &mw.layers[li];
        let p = pl == Some(li);
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
        let hd = head_dim as usize;
        let (qw, kw) = (n_head * hd, n_kv * hd);

        cur.copy_from_slice(&x);
        for t in 0..b {
            ops::rms_norm(
                &mut cur[t * n_embd..(t + 1) * n_embd],
                lw.attn_norm.as_slice(),
                eps,
            );
        }
        probe(p, "attn_norm", li as i32, &cur[..n_embd]);

        ops::mul_mat_batch(weights, &lw.wq, &cur, b, &mut q[..b * qw]);
        ops::mul_mat_batch(weights, &lw.wk, &cur, b, &mut kbuf[..b * kw]);
        ops::mul_mat_batch(weights, &lw.wv, &cur, b, &mut vbuf[..b * kw]);

        // NORM THEN ROTATE, PER HEAD, Q AND K BOTH. One `norm_and_rope` call does the two
        // in the order the device path fuses them; V is neither normalised nor rotated.
        let rope = RopeShape {
            b,
            heads: n_head,
            head_dim: hd,
            start_pos,
            rope_dim: rope_dim as usize,
            rope_base,
            eps,
        };
        norm_and_rope(&mut q[..b * qw], lw.q_norm.as_slice(), None, &rope);
        norm_and_rope(
            &mut kbuf[..b * kw],
            lw.k_norm.as_slice(),
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
                window: None,
                scale: 1.0 / (hd as f32).sqrt(),
            },
        );
        ops::mul_mat_batch(weights, &lw.wo, &attn[..b * qw], b, &mut mix);
        for i in 0..b * n_embd {
            x[i] += mix[i];
        }
        probe(p, "attn_out", li as i32, &x[..n_embd]);

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
        ops::silu(&mut gbuf);
        ops::mul_into(&mut gbuf, &upbuf);
        ops::mul_mat_batch(weights, &lw.ffn_down, &gbuf, b, &mut mix);
        for i in 0..b * n_embd {
            x[i] += mix[i];
        }
        probe(p, "l_out", li as i32, &x[..n_embd]);
    }

    let last = (b - 1) * n_embd;
    let mut h = x[last..last + n_embd].to_vec();
    *out = crate::cpu_support::logits(
        weights,
        &mw.head,
        &mut h,
        mw.output_norm.as_slice(),
        eps,
        softcap,
    );
    Ok(())
}
