//! What every model's CPU reference does the same way.
//!
//! The CPU path is the correctness oracle: a backend is believed when it agrees with this.
//! That makes a second copy of it worse than a second copy of anything else -- two oracles
//! that drift disagree with each other, and nothing in the build can tell you which one is
//! right.
//!
//! WHAT LIVES HERE is what a second model would otherwise retype: the cache arrays, the
//! tensor resolver, the causal attention core, RoPE over heads, and the lm_head tail.
//! WHAT DOES NOT is anything one architecture happens to do -- gemma4 norms V with no
//! weight vector and scales each layer's output; LFM2 does neither. Those stay at their
//! call sites. A shared helper that grows a flag per model is how the sharing stops paying
//! for itself.

use imparo_cpu::ops;
use imparo_gguf::weights::{Tensor, Weights, supported_weight_types};

use crate::{KvSource, ModelPlan};

// ---------------------------------------------------------------------------------------
// instrumentation
// ---------------------------------------------------------------------------------------

/// Which layer to print per-stage tensors for, or None.
#[must_use]
pub fn probe_layer() -> Option<usize> {
    std::env::var("IMPARO_PROBE_LAYER")
        .ok()
        .and_then(|v| v.parse().ok())
}

/// Prints the same shape of information `llama-eval-callback` does, so the two can be
/// diffed stage by stage. Gated: the whole block, not just the print.
pub fn probe(enabled: bool, name: &str, li: i32, x: &[f32]) {
    if !enabled {
        return;
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
    let sum: f32 = x.iter().sum();
    let head: Vec<String> = x.iter().take(5).map(|v| format!("{v:.5}")).collect();
    eprintln!(
        "[probe] {name:<22} il={li:<3} n={:<7} rms={rms:.5} sum={sum:.4} [{}]",
        x.len(),
        head.join(", ")
    );
}

// ---------------------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------------------

/// A norm vector, resolved once.
///
/// The two halves are ONE fact about ONE tensor, and writing them as two fields is how a
/// model ends up with fourteen: the CPU path reads `w`, the device path binds `offset`,
/// and neither is meaningful without knowing they name the same tensor.
///
/// Owned rather than looked up per use: the norms are 2.6 MB across all of gemma4's 42
/// layers, and resolving them per layer per token cost ~420 allocations and 4 MB of memcpy
/// before any arithmetic happened.
pub struct NormW {
    /// The values, empty when the tensor is absent.
    pub w: Vec<f32>,
    /// Byte offset into the weight mapping, `u64::MAX` when absent. The device reads the
    /// tensor in place from here.
    pub offset: u64,
}

impl NormW {
    /// A norm this layer does not have.
    #[must_use]
    pub fn absent() -> Self {
        Self {
            w: Vec::new(),
            offset: u64::MAX,
        }
    }

    /// The weight for `ops::rms_norm`, or None when the tensor is absent -- which is the
    /// same None that means "normalise without a weight vector", and that is correct: an
    /// absent norm weight IS no weight.
    #[must_use]
    pub fn as_slice(&self) -> Option<&[f32]> {
        if self.w.is_empty() {
            None
        } else {
            Some(&self.w)
        }
    }
}

/// Resolves tensors out of one file, with the checks that must happen at LOAD.
///
/// The point of the type is the `matmul` path: an unsupported quant is rejected BY NAME
/// here and can never reach a kernel built for another layout. `weights.get(..).ok()`
/// would drop such a tensor silently, which is the misread this exists to prevent.
pub struct Tensors<'a> {
    pub weights: &'a Weights,
}

impl<'a> Tensors<'a> {
    #[must_use]
    pub fn new(weights: &'a Weights) -> Self {
        Self { weights }
    }

    /// Any tensor, by name.
    ///
    /// # Errors
    /// When the file has no such tensor.
    pub fn get(&self, name: &str) -> Result<Tensor, String> {
        self.weights
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing tensor {name}"))
    }

    /// A required F32 vector plus its offset.
    ///
    /// # Errors
    /// When the file has no such tensor.
    pub fn norm(&self, name: &str) -> Result<NormW, String> {
        let t = self.get(name)?;
        Ok(NormW {
            w: self.weights.f32s(&t).to_vec(),
            offset: t.offset as u64,
        })
    }

    /// An F32 vector that may be absent.
    #[must_use]
    pub fn norm_opt(&self, name: &str) -> NormW {
        self.norm(name).unwrap_or_else(|_| NormW::absent())
    }

    /// A tensor destined for a MATMUL kernel, validated against the weight-type -> kernel
    /// table.
    ///
    /// # Errors
    /// When the tensor is absent, or carries a quant no kernel can read.
    pub fn matmul(&self, name: &str) -> Result<Tensor, String> {
        let x = self.get(name)?;
        if imparo_gguf::weights::weight_kind(x.ggml_type).is_none() {
            return Err(format!(
                "tensor '{name}': unsupported weight type (ggml type id {}) for a \
                 matmul; supported today: {}. Refusing to load \
                 rather than silently misread the blocks.",
                x.ggml_type,
                supported_weight_types()
            ));
        }
        Ok(x)
    }

    /// An OPTIONAL matmul tensor: absent is fine, present-but-unsupported is an error.
    ///
    /// # Errors
    /// When the tensor is present and carries a quant no kernel can read.
    pub fn matmul_opt(&self, name: &str) -> Result<Option<Tensor>, String> {
        if self.weights.get(name).is_none() {
            return Ok(None);
        }
        self.matmul(name).map(Some)
    }

    /// The first element of an F32 tensor, or `default` when it is absent. Some files
    /// carry a per-layer scalar as a length-1 tensor.
    #[must_use]
    pub fn scalar(&self, name: &str, default: f32) -> f32 {
        self.get(name)
            .ok()
            .and_then(|t| self.weights.f32s(&t).first().copied())
            .unwrap_or(default)
    }
}

// ---------------------------------------------------------------------------------------
// the cache
// ---------------------------------------------------------------------------------------

/// The CPU backend's KV arrays.
///
/// The counters that describe the DEVICE cache live in [`crate::kv::KvRuntime`]; they were
/// in here, which is two different things in one struct.
pub struct KvCache {
    /// One entry per layer, indexed by layer number. A layer that owns no KV holds an
    /// empty vector.
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    /// Elements per position, per layer: `n_kv_heads * head_dim`, or 0 when the layer owns
    /// no KV (it shares another layer's, or it does not attend at all).
    pub width: Vec<usize>,
}

impl KvCache {
    /// Sized from the plan.
    ///
    /// `allocate` false gives the widths and no storage: the GPU path keeps its own
    /// half-precision ring buffers and never reads these, so allocating them cost 223 MiB
    /// of pure waste -- more than half the engine's footprint, and invisible until the
    /// startup stages were measured one at a time.
    #[must_use]
    pub fn new(plan: &ModelPlan, capacity: usize, allocate: bool) -> Self {
        let n_layers = plan.config.n_layers as usize;
        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);
        let mut width = vec![0_usize; n_layers];
        for layer in &plan.layers {
            // A recurrent block owns no KV at all: head_dim() is 0 for it, and its state
            // is a separate per-conversation allocation (docs/unified-kv-pool.md).
            let owns =
                layer.kv_source == KvSource::Own && layer.attention.is_attention();
            let w = if owns {
                plan.config.n_kv_heads as usize * layer.attention.head_dim() as usize
            } else {
                0
            };
            width[layer.index as usize] = w;
            let slots = if allocate { capacity } else { 0 };
            k.push(vec![0.0; w * slots]);
            v.push(vec![0.0; w * slots]);
        }
        Self { k, v, width }
    }

    /// Writes this batch's K and V rows at `start_pos ..`.
    ///
    /// # Panics
    /// When the cache is too short for the positions written -- which is the caller's
    /// capacity check having been skipped.
    pub fn append(
        &mut self,
        li: usize,
        start_pos: usize,
        b: usize,
        kbuf: &[f32],
        vbuf: &[f32],
    ) {
        let w = self.width[li];
        // A slice index panic here says "range end 512 out of range for slice of length
        // 0", which names neither the cause nor the layer. The cause is always the same:
        // the host forward ran on weights that had been handed to a device, so this cache
        // was deliberately not allocated.
        assert!(
            !self.k[li].is_empty(),
            "layer {li} has no host KV cache -- the host forward ran on weights that \
             were handed to a device"
        );
        for t in 0..b {
            let pos = start_pos + t;
            self.k[li][pos * w..(pos + 1) * w]
                .copy_from_slice(&kbuf[t * w..(t + 1) * w]);
            self.v[li][pos * w..(pos + 1) * w]
                .copy_from_slice(&vbuf[t * w..(t + 1) * w]);
        }
    }
}

/// Per-conversation recurrent state, one buffer per recurrent layer.
///
/// NOT a KV-pool tenant. The pool exists for state that GROWS with context; recurrent
/// state is constant in it, so "pooling them adds bookkeeping for no saving"
/// (docs/unified-kv-pool.md). It is allocated with the conversation and advanced in place.
///
/// Two element counts because that is what the family needs: LFM2 uses `r` alone (its
/// convolution history), Mamba and RWKV use both. The plan carries the counts; how a model
/// derives them is the plan builder's business and never reaches here.
pub struct RecurrentState {
    /// Indexed by layer number. A layer that is not recurrent holds an empty vector.
    pub r: Vec<Vec<f32>>,
    pub s: Vec<Vec<f32>>,
}

impl RecurrentState {
    /// Sized from the plan; zero is the state of a conversation that has seen nothing.
    ///
    /// `allocate` false gives empty vectors: the device path keeps this state in one
    /// buffer of its own and never reads these, the same rule [`KvCache::new`] follows.
    #[must_use]
    pub fn new(plan: &ModelPlan, allocate: bool) -> Self {
        let n = plan.config.n_layers as usize;
        let mut r = vec![Vec::new(); n];
        let mut s = vec![Vec::new(); n];
        if allocate {
            for l in &plan.layers {
                if let crate::Attention::Recurrent { r_elems, s_elems, .. } = l.attention {
                    r[l.index as usize] = vec![0.0; r_elems as usize];
                    s[l.index as usize] = vec![0.0; s_elems as usize];
                }
            }
        }
        Self { r, s }
    }

    /// Back to a fresh conversation. A forward that resumed on a dirty state would give
    /// the right shape and the wrong numbers, which is the failure that reads as success.
    pub fn reset(&mut self) {
        for b in self.r.iter_mut().chain(self.s.iter_mut()) {
            b.fill(0.0);
        }
    }

    /// Bytes one conversation holds, for the footprint report.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.r
            .iter()
            .chain(self.s.iter())
            .map(Vec::len)
            .sum::<usize>()
            * 4
    }
}

// ---------------------------------------------------------------------------------------
// forward primitives
// ---------------------------------------------------------------------------------------

/// Gathers one embedding row per token into a token-major buffer, optionally scaled.
///
/// `scale` is gemma4's `sqrt(n_embd)`; LFM2 passes None. It is an Option rather than a
/// 1.0 default so that "this model does not scale" reads differently from "this model
/// scales by one".
pub fn embed_tokens(
    weights: &Weights,
    embd: &Tensor,
    tokens: &[u32],
    n_embd: usize,
    scale: Option<f32>,
    out: &mut [f32],
) {
    for (t, &token) in tokens.iter().enumerate() {
        let r = &mut out[t * n_embd..(t + 1) * n_embd];
        ops::row(weights, embd, token as usize, r);
        if let Some(k) = scale {
            ops::scale(r, k);
        }
    }
}

/// Geometry for [`norm_and_rope`].
pub struct RopeShape {
    pub b: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// Absolute position of token 0 of this batch.
    pub start_pos: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    pub eps: f32,
}

/// Per-head RMS norm then NeoX RoPE over a token-major `[b, heads, head_dim]` buffer.
///
/// `norm` is None for a model with no per-head norm. `freq` divides theta per pair, which
/// is how gemma4's full-attention layers get proportional rope and its windowed ones do
/// not.
pub fn norm_and_rope(
    buf: &mut [f32],
    norm: Option<&[f32]>,
    freq: Option<&[f32]>,
    s: &RopeShape,
) {
    let row = s.heads * s.head_dim;
    for t in 0..s.b {
        for h in 0..s.heads {
            let off = t * row + h * s.head_dim;
            let x = &mut buf[off..off + s.head_dim];
            ops::rms_norm(x, norm, s.eps);
            ops::rope_neox(x, (s.start_pos + t) as u32, s.rope_dim, s.rope_base, freq);
        }
    }
}

/// Geometry for [`attend`].
pub struct AttnShape {
    pub b: usize,
    /// Absolute position of token 0 of this batch. Causality within the batch falls out
    /// of the position bound; there is no separate mask.
    pub start_pos: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// Sliding window, or None for full attention.
    pub window: Option<usize>,
    /// Multiplies the scores before the softmax. gemma4 folds its `1/sqrt(head_dim)` into
    /// the Q norm and passes 1.0; LFM2 passes `1/sqrt(head_dim)` here.
    pub scale: f32,
}

/// Causal attention of `q` against one layer's cache, into `out`.
///
/// Both `q` and `out` are token-major `[b, n_head, head_dim]`; `k` and `v` are
/// position-major `[positions, kv_width]` with `kv_width = n_kv * head_dim`. Grouped-query
/// mapping is `h / (n_head / n_kv)`, the contiguous grouping every GQA file uses.
///
/// # Panics
/// When `n_kv` does not divide `n_head`.
pub fn attend(
    q: &[f32],
    out: &mut [f32],
    k: &[f32],
    v: &[f32],
    kv_width: usize,
    s: &AttnShape,
) {
    assert!(
        s.n_kv > 0 && s.n_head % s.n_kv == 0,
        "n_kv {} must divide n_head {}",
        s.n_kv,
        s.n_head
    );
    let hd = s.head_dim;
    let qw = s.n_head * hd;
    let per_kv = s.n_head / s.n_kv;
    // One buffer, not one per head per token: the longest run is the last token's.
    let mut scores = vec![0.0_f32; s.start_pos + s.b];
    for t in 0..s.b {
        let pos = s.start_pos + t;
        let lo = s.window.map_or(0, |w| (pos + 1).saturating_sub(w));
        let n = pos + 1 - lo;
        for h in 0..s.n_head {
            let kvh = h / per_kv;
            let qoff = t * qw + h * hd;
            let sc = &mut scores[..n];
            for (si, pp) in (lo..=pos).enumerate() {
                let base = pp * kv_width + kvh * hd;
                sc[si] = ops::dot(&q[qoff..qoff + hd], &k[base..base + hd]) * s.scale;
            }
            ops::softmax(sc);
            let o = &mut out[qoff..qoff + hd];
            o.fill(0.0);
            for (si, pp) in (lo..=pos).enumerate() {
                let base = pp * kv_width + kvh * hd;
                let w = sc[si];
                for (oo, vv) in o.iter_mut().zip(&v[base..base + hd]) {
                    *oo += w * vv;
                }
            }
        }
    }
}

/// Final norm, lm_head and optional softcap, for ONE hidden vector.
///
/// Only the last token of a batch needs logits; running the head for every token is the
/// single most expensive thing a CPU reference can get wrong.
pub fn logits(
    weights: &Weights,
    head: &Tensor,
    h: &mut [f32],
    norm: Option<&[f32]>,
    eps: f32,
    softcap: Option<f32>,
) -> Vec<f32> {
    ops::rms_norm(h, norm, eps);
    let mut out = vec![0.0_f32; head.ne1()];
    ops::mul_mat(weights, head, h, &mut out);
    if let Some(cap) = softcap {
        ops::softcap(&mut out, cap);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{AttnShape, NormW, attend};

    /// An absent norm is the same None `ops::rms_norm` takes for "no weight vector".
    #[test]
    fn an_absent_norm_reads_as_no_weight() {
        assert!(NormW::absent().as_slice().is_none());
        let present = NormW {
            w: vec![1.0, 2.0],
            offset: 0,
        };
        assert_eq!(present.as_slice(), Some(&[1.0_f32, 2.0][..]));
    }

    /// One head, two positions, identity V: the output is the softmax weights themselves,
    /// so the causal bound and the scale are both visible in the numbers.
    #[test]
    fn attention_is_causal_and_scaled() {
        let hd = 2;
        let s = AttnShape {
            b: 2,
            start_pos: 0,
            n_head: 1,
            n_kv: 1,
            head_dim: hd,
            window: None,
            scale: 1.0,
        };
        // q[t] = [1, 0] for both tokens; k[0] = [0, 1] (dot 0), k[1] = [1, 0] (dot 1).
        let q = vec![1.0, 0.0, 1.0, 0.0];
        let k = vec![0.0, 1.0, 1.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];
        attend(&q, &mut out, &k, &v, hd, &s);
        // Token 0 sees only position 0, so it is v[0] exactly.
        assert_eq!(&out[..2], &[1.0, 0.0]);
        // Token 1 sees both; scores 0 and 1 -> softmax weights that must sum to 1.
        let sum = out[2] + out[3];
        assert!((sum - 1.0).abs() < 1e-6, "weights sum to {sum}");
        assert!(
            out[3] > out[2],
            "position 1 scores higher and must weigh more"
        );
    }

    /// The window bounds the LOW end, so a window of 1 leaves each token attending to
    /// itself alone.
    #[test]
    fn a_window_of_one_leaves_each_token_alone() {
        let hd = 2;
        let s = AttnShape {
            b: 2,
            start_pos: 0,
            n_head: 1,
            n_kv: 1,
            head_dim: hd,
            window: Some(1),
            scale: 1.0,
        };
        let q = vec![1.0, 0.0, 1.0, 0.0];
        let k = vec![0.0, 1.0, 1.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];
        attend(&q, &mut out, &k, &v, hd, &s);
        assert_eq!(&out[..2], &[1.0, 0.0]);
        assert_eq!(&out[2..], &[0.0, 1.0]);
    }

    /// Grouped-query mapping is contiguous: heads 0,1 share KV head 0 and 2,3 share 1.
    #[test]
    fn grouped_query_heads_map_contiguously() {
        let hd = 1;
        let s = AttnShape {
            b: 1,
            start_pos: 0,
            n_head: 4,
            n_kv: 2,
            head_dim: hd,
            window: None,
            scale: 1.0,
        };
        let q = vec![1.0, 1.0, 1.0, 1.0];
        // one position, kv_width 2: KV head 0 holds 10, KV head 1 holds 20.
        let k = vec![1.0, 1.0];
        let v = vec![10.0, 20.0];
        let mut out = vec![0.0; 4];
        attend(&q, &mut out, &k, &v, 2, &s);
        assert_eq!(out, vec![10.0, 10.0, 20.0, 20.0]);
    }
}
