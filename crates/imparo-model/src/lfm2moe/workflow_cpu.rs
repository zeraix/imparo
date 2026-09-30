//! LFM2-MoE forward pass, CPU reference.
//!
//! The correctness oracle. Every block is LFM2's -- operator norm, then a short
//! convolution or grouped-query attention, residual, feed-forward norm, residual -- and
//! the only new arm is the routed feed-forward:
//!
//! ```text
//!   logits          = router @ cur                      [experts] per token
//!   probs           = softmax(logits)  OR  sigmoid(logits)
//!   selection       = probs + exp_probs_b               THE BIAS SELECTS, and only selects
//!   picked          = top_k(selection, experts_used)
//!   w               = probs[picked]                     UNBIASED probs are the weights
//!   w              /= sum(w)                            when the file asks for it
//!   w              *= weights_scale
//!   y               = SUM over picked e of w_e * down_e( silu(gate_e @ cur) * (up_e @ cur) )
//! ```
//!
//! THE BIAS IS THE STEP THAT LOOKS LIKE BOOKKEEPING AND IS NOT. `exp_probs_b` is added
//! before the pick and NOT to the weights -- llama.cpp says so in a comment at the line
//! that does it ("leave probs unbiased as it's later used to get expert weights"). Adding
//! it to both, or to neither, gives a model that writes fluent text from the wrong
//! experts, which no test that reads output would catch.
//!
//! The three expert tensors are ONE stacked tensor each, `[n_embd, n_ff_exp, n_expert]`.
//! A per-expert 2-D view is a slice of it at `offset + e * bytes_per_expert`, which is
//! what `expert_view` builds -- the same bytes the device path will address with a
//! stride, so the two cannot disagree about which expert ran.

use crate::cpu_support::{
    AttnShape, NormW, RopeShape, Tensors, attend, embed_tokens, norm_and_rope, probe,
    probe_layer,
};
use crate::lfm2moe::Lfm2Moe;
use crate::{Attention, ExpertGating, Ffn, ModelPlan};
use imparo_cpu::ops;
use imparo_gguf::weights::{Tensor, Weights};

/// A block's mixer weights: the two arms share no tensor.
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
        conv: NormW,
        in_proj: Tensor,
        out_proj: Tensor,
    },
}

/// A block's feed-forward weights, in the two shapes the plan distinguishes.
pub enum FfnW {
    Dense {
        gate: Tensor,
        up: Tensor,
        down: Tensor,
    },
    Moe {
        router: Tensor,
        /// `[n_expert]`, absent when the file carries no `exp_probs_b`.
        bias: NormW,
        /// Each is the stacked `[n_embd, n_ff_exp, n_expert]` tensor.
        gate: Tensor,
        up: Tensor,
        down: Tensor,
    },
}

pub struct LayerW {
    pub(crate) op_norm: NormW,
    pub(crate) ffn_norm: NormW,
    pub(crate) mixer: MixerW,
    pub(crate) ffn: FfnW,
}

pub struct ModelW {
    pub(crate) token_embd: Tensor,
    /// Spelled `token_embd_norm`, like LFM2's.
    pub(crate) output_norm: NormW,
    pub(crate) head: Tensor,
    pub(crate) layers: Vec<LayerW>,
}

/// One expert's 2-D matrix inside a stacked `[n_in, n_out, n_expert]` tensor.
///
/// # Errors
/// When the stack does not divide into whole experts, which would make every view after
/// the first read from the middle of a quantisation block.
pub(crate) fn expert_view(
    stacked: &Tensor,
    experts: usize,
    e: usize,
) -> Result<Tensor, String> {
    if experts == 0 || stacked.bytes % experts != 0 || e >= experts {
        return Err(format!(
            "expert {e} of {experts} does not divide a {}-byte stack",
            stacked.bytes
        ));
    }
    let per = stacked.bytes / experts;
    Ok(Tensor {
        offset: stacked.offset + e * per,
        bytes: per,
        ggml_type: stacked.ggml_type,
        ne: [stacked.ne[0], stacked.ne[1], 1, 1],
        n_dims: 2,
    })
}

/// Resolves LFM2-MoE's weights out of one file.
///
/// # Errors
/// When a required tensor is absent, or carries a quant no kernel can read.
pub fn prepare(weights: &Weights, plan: &ModelPlan) -> Result<ModelW, String> {
    let f = Tensors::new(weights);
    let mut layers = Vec::with_capacity(plan.layers.len());
    for l in &plan.layers {
        let i = l.index;
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
        // The plan decided dense or routed by reading the tensors; this reads that arm's
        // tensors and nothing from the other.
        let ffn = match l.ffn {
            Ffn::Dense { .. } => FfnW::Dense {
                gate: f.matmul(&format!("blk.{i}.ffn_gate.weight"))?,
                up: f.matmul(&format!("blk.{i}.ffn_up.weight"))?,
                down: f.matmul(&format!("blk.{i}.ffn_down.weight"))?,
            },
            Ffn::Moe { expert_bias, .. } => FfnW::Moe {
                router: f.matmul(&format!("blk.{i}.ffn_gate_inp.weight"))?,
                bias: if expert_bias {
                    f.norm(&format!("blk.{i}.exp_probs_b.bias"))?
                } else {
                    NormW::absent()
                },
                gate: f.matmul(&format!("blk.{i}.ffn_gate_exps.weight"))?,
                up: f.matmul(&format!("blk.{i}.ffn_up_exps.weight"))?,
                down: f.matmul(&format!("blk.{i}.ffn_down_exps.weight"))?,
            },
        };
        layers.push(LayerW {
            op_norm: f.norm(&format!("blk.{i}.attn_norm.weight"))?,
            ffn_norm: f.norm(&format!("blk.{i}.ffn_norm.weight"))?,
            mixer,
            ffn,
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
        output_norm: f.norm("token_embd_norm.weight")?,
        head,
        layers,
    })
}

/// One token through a routed feed-forward, adding its result into `out`.
///
/// Scratch buffers are the caller's: a routed layer runs this once per token and
/// allocating inside would be allocating per token per layer.
#[allow(clippy::too_many_arguments)]
fn moe_token(
    weights: &Weights,
    w: &FfnW,
    ffn: Ffn,
    probe_here: bool,
    cur: &[f32],
    out: &mut [f32],
    scores: &mut Vec<f32>,
    gbuf: &mut Vec<f32>,
    ubuf: &mut Vec<f32>,
    ybuf: &mut Vec<f32>,
) -> Result<(), String> {
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
            ..
        },
    ) = (w, ffn)
    else {
        return Err("lfm2moe: a routed layer resolved dense weights".into());
    };
    let (experts, used, hidden) = (
        experts as usize,
        experts_used as usize,
        expert_hidden as usize,
    );
    scores.clear();
    scores.resize(experts, 0.0);
    ops::mul_mat(weights, router, cur, scores);
    crate::cpu_support::probe(probe_here, "moe.router", -1, scores);

    match gating {
        ExpertGating::Softmax => ops::softmax(scores),
        ExpertGating::Sigmoid => {
            for s in scores.iter_mut() {
                *s = ops::sigmoid_f32(*s);
            }
        }
    }
    // THE BIAS SELECTS AND DOES NOT WEIGH. Picking reads probs + bias; the weights below
    // read `scores`, which is still the unbiased probability.
    let mut picked: Vec<(usize, f32)> = (0..experts)
        .map(|e| {
            let selection = scores[e] + bias.as_slice().map_or(0.0, |b| b[e]);
            (e, selection)
        })
        .collect();
    // Ties go to the lower expert index, which is what a stable sort by score gives.
    picked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    picked.truncate(used);

    crate::cpu_support::probe(probe_here, "moe.probs", -1, scores);
    if probe_here {
        let ids: Vec<f32> = picked.iter().map(|&(e, _)| e as f32).collect();
        crate::cpu_support::probe(true, "moe.picks", -1, &ids);
    }
    let mut weight: Vec<f32> = picked.iter().map(|&(e, _)| scores[e]).collect();
    if normalise_weights {
        // Clamped the way the reference clamps it: to the smallest normal f16, so a row
        // of zeros divides by that rather than by zero. Written as f32 because that is
        // what divides here.
        const MIN_NORMAL_F16: f32 = 6.103_515_6e-5;
        let sum = weight.iter().sum::<f32>().max(MIN_NORMAL_F16);
        for v in &mut weight {
            *v /= sum;
        }
    }
    // The plan already turned the reference's two "no scale" values into a multiplier of
    // one, so this multiplies unconditionally -- by 1.0, which is exact.
    for v in &mut weight {
        *v *= weights_scale;
    }

    gbuf.clear();
    gbuf.resize(hidden, 0.0);
    ubuf.clear();
    ubuf.resize(hidden, 0.0);
    ybuf.clear();
    ybuf.resize(out.len(), 0.0);
    crate::cpu_support::probe(probe_here, "moe.weights", -1, &weight);
    for (&(e, _), &wgt) in picked.iter().zip(weight.iter()) {
        let ge = expert_view(gate, experts, e)?;
        let ue = expert_view(up, experts, e)?;
        let de = expert_view(down, experts, e)?;
        ops::mul_mat(weights, &ge, cur, gbuf);
        ops::mul_mat(weights, &ue, cur, ubuf);
        ops::silu(gbuf);
        ops::mul_into(gbuf, ubuf);
        ops::mul_mat(weights, &de, gbuf, ybuf);
        for (o, &y) in out.iter_mut().zip(ybuf.iter()) {
            *o += wgt * y;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)] // one block, read top to bottom
/// Runs ONE chunk of LFM2-MoE's forward on the host, leaving the last token's logits in
/// `out`. `start_pos` is the chunk's ABSOLUTE start, which the convolution state depends
/// on: it holds the tail of the previous chunk.
///
/// # Errors
/// When the plan and the resolved weights disagree about a layer's block or FFN kind.
pub fn batch(
    wf: &mut Lfm2Moe,
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
    // The routed arm's per-token scratch, allocated once for the whole chunk.
    let (mut scores, mut eg, mut eu, mut ey) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for (li, layer) in plan_layers.iter().enumerate() {
        let lw = &mw.layers[li];
        let p = pl == Some(li);

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
                let (qw, kw) = (n_head * hd, n_kv * hd);
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
                ops::causal_conv(
                    ops::ConvForm::GatedBcx,
                    &bcx,
                    &conv_w.w,
                    &mut conv.r[li],
                    &mut mix[..b * n_embd],
                    n_embd,
                    kernel,
                    b,
                );
                let y = mix.clone();
                ops::mul_mat_batch(weights, out_proj, &y, b, &mut mix);
            }
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
        match &lw.ffn {
            FfnW::Dense { gate, up, down } => {
                ops::mul_mat_batch(weights, gate, &cur, b, &mut gbuf);
                ops::mul_mat_batch(weights, up, &cur, b, &mut upbuf);
                ops::silu(&mut gbuf);
                ops::mul_into(&mut gbuf, &upbuf);
                ops::mul_mat_batch(weights, down, &gbuf, b, &mut mix);
            }
            routed @ FfnW::Moe { .. } => {
                // Routing is PER TOKEN: two tokens of one chunk run different experts,
                // so this cannot be a batched matmul the way the dense arm is.
                mix[..b * n_embd].fill(0.0);
                for t in 0..b {
                    let (lo, hi) = (t * n_embd, (t + 1) * n_embd);
                    moe_token(
                        weights,
                        routed,
                        layer.ffn,
                        p && t == 0,
                        &cur[lo..hi],
                        &mut mix[lo..hi],
                        &mut scores,
                        &mut eg,
                        &mut eu,
                        &mut ey,
                    )?;
                }
            }
        }
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
