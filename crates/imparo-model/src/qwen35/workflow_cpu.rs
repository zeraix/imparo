//! Qwen3.8 forward pass, CPU reference.
//!
//! The correctness oracle for the Qwen3.8 kernels. Sixteen of its sixty-four blocks are
//! ordinary GQA with a per-head sigmoid gate on the output; the other forty-eight are
//! GATED DELTA-NET, which is the part this engine has not had before:
//!
//! ```text
//! every block:   prev = x
//!                cur  = rms_norm(x, attn_norm)
//!                cur  = gated_delta(cur)  OR  gated_attention(cur)   <- 48 / 16
//!                x    = prev + cur
//!                x    = x + ffn_down(silu(ffn_gate @ n) * (ffn_up @ n)),
//!                       n = rms_norm(x, post_attention_norm)
//! tail:          logits = output @ rms_norm(x, output_norm)
//! ```
//!
//! EVERY STEP BELOW WAS READ from llama.cpp upstream -- `src/models/qwen35.cpp` and
//! `src/models/delta-net-base.cpp` -- not from the model card, and the four that would
//! have been easy to get wrong are marked at their call sites. The delta rule, per value
//! head `h`, with the key head `h % k_heads` (see `KEY HEAD MAPPING` below):
//!
//! ```text
//!   S      *= exp(g[h])                        decay, one scalar per value head
//!   sk[j]   = SUM_i S[j][i] * khat[i]          what the state already remembers of k
//!   d[j]    = (v[j] - sk[j]) * beta[h]         the delta rule's correction
//!   S[j][i]+= khat[i] * d[j]                   rank-one update
//!   o[j]    = SUM_i S[j][i] * qhat[i]          read it back with the query
//! ```
//!
//! `S` is `[value_head][value_coord][key_coord]` with the key coordinate contiguous, which
//! is the order the read and both writes above walk it in.

use crate::cpu_support::{
    AttnShape, NormW, RopeShape, Tensors, attend, embed_tokens, norm_and_rope, probe,
    probe_layer,
};
use crate::qwen35::Qwen35;
use crate::{Attention, ModelPlan};
use imparo_cpu::ops;
use imparo_gguf::weights::{Tensor, Weights};

/// A block's mixer weights.
///
/// An enum, not a struct of Options: the two arms share NO tensor. An attention block has
/// no convolution and no recurrent state; a delta-net block has no K or V projection of
/// its own -- its Q, K and V come out of ONE `attn_qkv` tensor.
pub enum MixerW {
    /// Full attention. `wq` is 24 heads of `[query(256) | gate(256)]`, so it is twice the
    /// width of an ordinary Q projection and must be split PER HEAD.
    Attention {
        q_norm: NormW,
        k_norm: NormW,
        wq: Tensor,
        wk: Tensor,
        wv: Tensor,
        wo: Tensor,
    },
    /// Gated delta-net.
    GatedDelta {
        /// 5120 -> 10240, packed `[Q | K | V]` = `[2048 | 2048 | 6144]`.
        qkv: Tensor,
        /// 5120 -> 6144: the `z` gate, the same width as V.
        gate: Tensor,
        /// Channel-major with the TAP fastest: element (tap, channel) is at
        /// `channel * 4 + tap`, which is what the GGUF dims `(4, 10240)` mean.
        conv: NormW,
        /// `-exp(A_log)`, one per value head. Stored already negated -- read out of the
        /// file it is every value in [-0.34, -0.004] -- so `a * softplus(dt)` is the
        /// NEGATIVE log decay and `exp` of it lands in (0, 1).
        a: NormW,
        dt_bias: NormW,
        alpha: Tensor,
        beta: Tensor,
        /// One head's width (128), shared by every value head.
        ssm_norm: NormW,
        out: Tensor,
    },
}

pub struct LayerW {
    pub(crate) attn_norm: NormW,
    /// The FFN's PRE-norm, despite the name. Qwen spells it `post_attention_norm`
    /// because it sits after the attention residual; there is no `ffn_norm` in the file.
    pub(crate) ffn_norm: NormW,
    pub(crate) mixer: MixerW,
    pub(crate) ffn_gate: Tensor,
    pub(crate) ffn_up: Tensor,
    pub(crate) ffn_down: Tensor,
}

pub struct ModelW {
    pub(crate) token_embd: Tensor,
    /// Its own tensor: `output.weight` is present, so the head is NOT tied to the
    /// embedding table the way LFM2's is.
    pub(crate) output: Tensor,
    pub(crate) output_norm: NormW,
    pub(crate) layers: Vec<LayerW>,
}

/// The delta-net's geometry and its convolution's tap count, derived from the layer's own
/// plan entry and the length of its conv weight -- nothing here is written down.
///
/// ```text
///   conv_w.len() = taps * qkv_width           the tensor is (taps, qkv_width)
///   r_elems      = (taps - 1) * qkv_width     the plan's rolling-state count
///   => qkv_width = conv_w.len() - r_elems     one subtraction, no assumed tap count
///   => taps      = conv_w.len() / qkv_width
/// ```
///
/// `v_heads` then falls out of the matrix count, and `k_heads` out of what is left of the
/// packed width once Q and V are accounted for. An earlier version read `taps` by dividing
/// by a literal 4, which is this file's tap count and not a fact about the architecture.
///
/// # Errors
/// When the layer is not recurrent, or the counts do not divide.
pub(crate) fn delta_shape(
    attention: crate::Attention,
    conv_len: usize,
) -> Result<(ops::DeltaShape, usize), String> {
    let crate::Attention::Recurrent {
        r_elems,
        s_elems,
        key_dim,
        value_dim,
    } = attention
    else {
        return Err("qwen35: delta_shape asked about a layer that attends".into());
    };
    let (kd, vd) = (key_dim as usize, value_dim as usize);
    let qkv_width = conv_len
        .checked_sub(r_elems as usize)
        .filter(|w| *w > 0 && conv_len % w == 0)
        .ok_or_else(|| {
            format!("qwen35: conv weight is {conv_len} and r_elems {r_elems}, which is not a conv")
        })?;
    let taps = conv_len / qkv_width;
    if kd == 0 || vd == 0 || s_elems as usize % (kd * vd) != 0 {
        return Err(format!(
            "qwen35: a {vd}x{kd} matrix does not divide s_elems {s_elems}"
        ));
    }
    let v_heads = s_elems as usize / (kd * vd);
    let k_heads = qkv_width
        .checked_sub(v_heads * vd)
        .filter(|rest| kd > 0 && rest % (2 * kd) == 0)
        .map(|rest| rest / (2 * kd))
        .ok_or_else(|| {
            format!(
                "qwen35: packed width {qkv_width} leaves no whole number of {kd}-wide \
                 Q and K heads after {v_heads} value heads of {vd}"
            )
        })?;
    Ok((
        ops::DeltaShape {
            k_heads,
            v_heads,
            key_dim: kd,
            value_dim: vd,
        },
        taps,
    ))
}

/// Resolves Qwen3.8's weights out of one file.
///
/// # Errors
/// When a required tensor is absent, or carries a quant no kernel can read.
pub fn prepare(weights: &Weights, plan: &ModelPlan) -> Result<ModelW, String> {
    super::weight_basis::ensure_workflow_basis(weights)?;
    let f = Tensors::new(weights);
    let mut layers = Vec::with_capacity(plan.layers.len());
    for l in &plan.layers {
        let i = l.index;
        // The plan decides which mixer a layer is, and it decided by looking at the
        // tensors; this reads the arm's tensors and NOTHING from the other.
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
            MixerW::GatedDelta {
                qkv: f.matmul(&format!("blk.{i}.attn_qkv.weight"))?,
                gate: f.matmul(&format!("blk.{i}.attn_gate.weight"))?,
                conv: f.norm(&format!("blk.{i}.ssm_conv1d.weight"))?,
                a: f.norm(&format!("blk.{i}.ssm_a"))?,
                dt_bias: f.norm(&format!("blk.{i}.ssm_dt.bias"))?,
                alpha: f.matmul(&format!("blk.{i}.ssm_alpha.weight"))?,
                beta: f.matmul(&format!("blk.{i}.ssm_beta.weight"))?,
                ssm_norm: f.norm(&format!("blk.{i}.ssm_norm.weight"))?,
                out: f.matmul(&format!("blk.{i}.ssm_out.weight"))?,
            }
        };
        layers.push(LayerW {
            attn_norm: f.norm(&format!("blk.{i}.attn_norm.weight"))?,
            ffn_norm: f.norm(&format!("blk.{i}.post_attention_norm.weight"))?,
            mixer,
            ffn_gate: f.matmul(&format!("blk.{i}.ffn_gate.weight"))?,
            ffn_up: f.matmul(&format!("blk.{i}.ffn_up.weight"))?,
            ffn_down: f.matmul(&format!("blk.{i}.ffn_down.weight"))?,
        });
    }
    Ok(ModelW {
        token_embd: f.matmul("token_embd.weight")?,
        output: f.matmul("output.weight")?,
        output_norm: f.norm("output_norm.weight")?,
        layers,
    })
}

/// Splits a projection packed as `[query_head | gate_head]` PER HEAD.
///
/// Qwen interleaves by head. Treating the 12288-wide tensor as two 6144 halves puts every
/// head after the first on the wrong side, which reads as a plausible model that is wrong
/// everywhere -- so this is its own function with its own name.
fn split_query_gate(
    packed: &[f32],
    heads: usize,
    head_dim: usize,
    q: &mut [f32],
    g: &mut [f32],
) {
    for h in 0..heads {
        let (src, dst) = (h * head_dim * 2, h * head_dim);
        q[dst..dst + head_dim].copy_from_slice(&packed[src..src + head_dim]);
        g[dst..dst + head_dim]
            .copy_from_slice(&packed[src + head_dim..src + head_dim * 2]);
    }
}

#[allow(clippy::too_many_lines)] // one block, read top to bottom
/// Runs ONE chunk of Qwen3.8's forward on the host, leaving the last token's logits in
/// `out`. `start_pos` is the chunk's ABSOLUTE start: the conv history and the recurrent
/// matrix both carry the previous chunk's tail.
///
/// # Errors
/// When the plan and the resolved weights disagree about a layer's block kind.
pub fn batch(
    wf: &mut Qwen35,
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
    let rec = &mut wf.state.recurrent;
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
    let mut qg = vec![0.0_f32; b * n_head * hd_max * 2];
    let mut q = vec![0.0_f32; b * n_head * hd_max];
    let mut gate = vec![0.0_f32; b * n_head * hd_max];
    let mut attn = vec![0.0_f32; b * n_head * hd_max];
    let mut kbuf = vec![0.0_f32; b * n_kv * hd_max];
    let mut vbuf = vec![0.0_f32; b * n_kv * hd_max];
    let mut gbuf = vec![0.0_f32; b * n_ff];
    let mut upbuf = vec![0.0_f32; b * n_ff];

    for (li, layer) in plan_layers.iter().enumerate() {
        let lw = &mw.layers[li];
        let p = pl == Some(li);

        cur.copy_from_slice(&x);
        for t in 0..b {
            ops::rms_norm(
                &mut cur[t * n_embd..(t + 1) * n_embd],
                lw.attn_norm.as_slice(),
                eps,
            );
        }
        probe(p, "attn_norm", li as i32, &cur[..n_embd]);

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
                let (qw, kw) = (n_head * hd, n_kv * hd);
                ops::mul_mat_batch(weights, wq, &cur, b, &mut qg[..b * qw * 2]);
                for t in 0..b {
                    split_query_gate(
                        &qg[t * qw * 2..(t + 1) * qw * 2],
                        n_head,
                        hd,
                        &mut q[t * qw..(t + 1) * qw],
                        &mut gate[t * qw..(t + 1) * qw],
                    );
                }
                ops::mul_mat_batch(weights, wk, &cur, b, &mut kbuf[..b * kw]);
                ops::mul_mat_batch(weights, wv, &cur, b, &mut vbuf[..b * kw]);
                let rope = RopeShape {
                    b,
                    heads: n_head,
                    head_dim: hd,
                    start_pos,
                    // PARTIAL rope: 64 of a 256-wide head. The multimodal section split
                    // [11,11,10,0] collapses to plain NEOX here because a text tower's
                    // positions are (p,p,p,0) in all three sections.
                    rope_dim: rope_dim as usize,
                    rope_base,
                    eps,
                };
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
                        window: None,
                        scale: 1.0 / (hd as f32).sqrt(),
                    },
                );
                // THE OUTPUT GATE, before o_proj: `cur = cur * sigmoid(gate)`
                // (qwen35.cpp). Plain sigmoid here, NOT the SiLU the delta-net's z gate
                // uses -- two gates, two activations, in one architecture.
                for (a, g) in attn[..b * qw].iter_mut().zip(&gate[..b * qw]) {
                    *a *= ops::sigmoid_f32(*g);
                }
                ops::mul_mat_batch(weights, wo, &attn[..b * qw], b, &mut mix);
            }
            (
                MixerW::GatedDelta {
                    qkv,
                    gate: gate_w,
                    conv,
                    a,
                    dt_bias,
                    alpha,
                    beta,
                    ssm_norm,
                    out: out_w,
                },
                Attention::Recurrent { .. },
            ) => {
                let (sh, taps) = delta_shape(layer.attention, conv.w.len())?;
                let (vd, vh) = (sh.value_dim, sh.v_heads);
                let (qkv_width, vwidth) = (sh.qkv_width(), vh * vd);

                let mut packed = vec![0.0_f32; b * qkv_width];
                let mut conved = vec![0.0_f32; b * qkv_width];
                let mut z = vec![0.0_f32; b * vwidth];
                let mut alpha_v = vec![0.0_f32; b * vh];
                let mut beta_v = vec![0.0_f32; b * vh];
                let mut core = vec![0.0_f32; b * vwidth];
                ops::mul_mat_batch(weights, qkv, &cur, b, &mut packed);
                ops::mul_mat_batch(weights, gate_w, &cur, b, &mut z);
                ops::mul_mat_batch(weights, alpha, &cur, b, &mut alpha_v);
                ops::mul_mat_batch(weights, beta, &cur, b, &mut beta_v);
                probe(p, "qkv_proj", li as i32, &packed[..qkv_width.min(n_embd)]);

                // THE SAME TWO OPS THE DEVICE DISPATCHES, in the same order: the whole
                // batch through the convolution, then the whole batch through the delta
                // rule. The conv does not read the recurrence, so running it for every
                // token first is the same answer as interleaving them per token -- and it
                // is the shape the device path has, which is what makes this an oracle
                // for that path rather than a second design.
                let (hist, state) = (&mut rec.r[li], &mut rec.s[li]);
                ops::causal_conv(
                    ops::ConvForm::PlainSilu,
                    &packed,
                    &conv.w,
                    hist,
                    &mut conved,
                    qkv_width,
                    taps,
                    b,
                );
                let mut log_decay = vec![0.0_f32; b * vh];
                let mut bt = vec![0.0_f32; b * vh];
                for t in 0..b {
                    for h in 0..vh {
                        // `a` is already -exp(A_log), so this is the NEGATIVE log decay.
                        log_decay[t * vh + h] = a.w[h]
                            * ops::softplus_f32(alpha_v[t * vh + h] + dt_bias.w[h]);
                        bt[t * vh + h] = ops::sigmoid_f32(beta_v[t * vh + h]);
                    }
                }
                ops::delta_net(&conved, &log_decay, &bt, state, &mut core, sh, b, eps);
                probe(p, "delta_core", li as i32, &core[..vwidth.min(n_embd)]);

                // Gated RMS norm: rms_norm(core, ssm_norm) * silu(z), per VALUE head.
                // The norm weight is one head wide and shared by all of them.
                for t in 0..b {
                    let row = t * vwidth;
                    for h in 0..vh {
                        let head = &mut core[row + h * vd..row + (h + 1) * vd];
                        ops::rms_norm(head, ssm_norm.as_slice(), eps);
                        for (value, &zg) in
                            head.iter_mut().zip(&z[row + h * vd..row + (h + 1) * vd])
                        {
                            *value *= zg * ops::sigmoid_f32(zg);
                        }
                    }
                }
                ops::mul_mat_batch(weights, out_w, &core, b, &mut mix);
            }
            (m, at) => {
                return Err(format!(
                    "qwen35 layer {li}: plan says {at:?} but the weights resolved as {}",
                    match m {
                        MixerW::Attention { .. } => "full attention",
                        MixerW::GatedDelta { .. } => "gated delta-net",
                    }
                ));
            }
        }

        for i in 0..b * n_embd {
            x[i] += mix[i];
        }
        probe(p, "mixer_out", li as i32, &x[..n_embd]);

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
        &mw.output,
        &mut h,
        mw.output_norm.as_slice(),
        eps,
        softcap,
    );
    Ok(())
}
