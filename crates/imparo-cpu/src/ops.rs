//! Model-agnostic kernels. Nothing here knows what a gemma is.

pub use imparo_backend::ConvForm;

/// x * rsqrt(mean(x^2) + eps), then elementwise weight when present.
pub fn rms_norm(x: &mut [f32], weight: Option<&[f32]>, eps: f32) {
    let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    match weight {
        Some(w) => {
            debug_assert_eq!(w.len(), x.len());
            for (v, wv) in x.iter_mut().zip(w) {
                *v = *v * scale * *wv;
            }
        }
        None => {
            for v in x.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// Exact tanh-based GELU, matching ggml's `ggml_gelu` approximation.
pub fn gelu(x: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const COEF: f32 = 0.044_715;
    for v in x.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + (SQRT_2_OVER_PI * (t + COEF * t * t * t)).tanh());
    }
}

/// SiLU (swish): x * sigmoid(x). The gate half of a SwiGLU feed-forward.
/// Logistic sigmoid, in the branch-free-per-sign form that avoids `exp` overflowing for
/// large negative `x`. `1/(1+exp(-x))` alone returns inf/inf = NaN there.
#[must_use]
pub fn sigmoid_f32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `ln(1 + exp(x))`, saturating to `x` where the two are equal in f32.
///
/// The cutoff is not a tolerance: above ~20, `ln(1+exp(x))` and `x` differ by less than an
/// f32 ulp, while `exp(x)` is still six orders of magnitude from overflowing -- so the
/// branch buys accuracy nothing and costs nothing. It is here because a gated delta-net's
/// decay is `A * softplus(dt)`, and a saturating softplus is what keeps that finite.
#[must_use]
pub fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Scales `x` to unit L2 norm, dividing by `max(norm, eps)` rather than `norm + eps`.
///
/// WHICH GUARD IT IS MATTERS. `max` leaves an ordinary vector exactly normalised and only
/// engages on a near-zero one; `+ eps` shrinks every vector slightly, which is a different
/// function and reads as a small systematic error rather than a defect. The accumulation
/// mirrors ggml: each square rounds in f32, the sum accumulates in double, one f32 sqrt.
pub fn l2_normalize_max_eps(x: &mut [f32], eps: f32) {
    let sum = x.iter().map(|v| f64::from(*v * *v)).sum::<f64>();
    let norm = (sum as f32).sqrt();
    let inv = 1.0 / norm.max(eps);
    for value in x {
        *value *= inv;
    }
}

pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let t = *v;
        *v = t / (1.0 + (-t).exp());
    }
}

/// A causal depthwise convolution over per-conversation history, the CPU reference.
///
/// The convolution is the same in every architecture that has one; `form` says what the
/// taps run over and what the epilogue does, and that is the only difference between
/// LFM2's gated short convolution and Qwen3.8's delta-net convolution:
///
/// ```text
/// GatedBcx    src row = [b | c | x]     value = b * x    out[t] = c[t] * sum
/// PlainSilu   src row = [x]             value = x        out[t] = silu(sum)
/// ```
///
/// In both:
///
/// ```text
/// seq     = state ++ value              causal: the state is PREPENDED
/// sum[t]  = sum_k conv_w[ch][k] * seq[t + k]
/// state'  = the last (kernel - 1) values of seq
/// ```
///
/// `conv_w` is channel-major with the tap fastest, which is what dims `(kernel, width)`
/// mean: element (k, ch) sits at `ch * kernel + k`. Tap 0 multiplies the OLDEST value.
///
/// `state` holds `kernel - 1` past RAW values per channel, oldest first, and is advanced
/// in place. RAW, not activated: storing the epilogue's output would decay the history
/// through the activation, a slow drift that still runs. A fresh conversation starts at
/// zero.
///
/// # Panics
/// Panics when any slice length disagrees with `form`, `width`, `kernel` and `n_tok`.
pub fn causal_conv(
    form: ConvForm,
    src: &[f32],
    conv_w: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    width: usize,
    kernel: usize,
    n_tok: usize,
) {
    assert!(
        kernel >= 2,
        "a causal conv needs at least 2 taps to carry state"
    );
    let history = kernel - 1;
    let stride = form.src_stride(width as u32) as usize;
    assert_eq!(src.len(), n_tok * stride, "src is n_tok x form stride");
    assert_eq!(conv_w.len(), width * kernel, "conv_w is width x kernel");
    assert_eq!(state.len(), history * width, "state is (kernel-1) x width");
    assert_eq!(out.len(), n_tok * width, "out is n_tok x width");

    // seq[e] for e in [0, history + n_tok): the state, then this batch's values.
    let value_at = |e: usize, ch: usize, state: &[f32]| -> f32 {
        if e < history {
            state[e * width + ch]
        } else {
            let row = (e - history) * stride;
            match form {
                ConvForm::GatedBcx => src[row + ch] * src[row + 2 * width + ch],
                ConvForm::PlainSilu => src[row + ch],
            }
        }
    };

    for ch in 0..width {
        let w = &conv_w[ch * kernel..(ch + 1) * kernel];
        for t in 0..n_tok {
            let mut acc = 0.0_f32;
            for (k, &wk) in w.iter().enumerate() {
                acc += wk * value_at(t + k, ch, state);
            }
            out[t * width + ch] = match form {
                ConvForm::GatedBcx => src[t * stride + width + ch] * acc,
                ConvForm::PlainSilu => acc / (1.0 + (-acc).exp()),
            };
        }
    }

    // The new state is the tail of seq, and it must be computed from the OLD state -- so
    // every output above is finished before anything here writes. Descending would read a
    // slot this loop has already overwritten.
    let mut next = vec![0.0_f32; history * width];
    for s in 0..history {
        for ch in 0..width {
            next[s * width + ch] = value_at(n_tok + s, ch, state);
        }
    }
    state.copy_from_slice(&next);
}

/// The geometry of a gated delta-net mixer, derived once from the widths a file carries.
#[derive(Clone, Copy, Debug)]
pub struct DeltaShape {
    /// Q/K head count. A value head reads key head `h % k_heads` -- see `delta_net`.
    pub k_heads: usize,
    pub v_heads: usize,
    /// The Q/K head width, which is also the state's key coordinate.
    pub key_dim: usize,
    /// The V head width, which is also the state's value coordinate.
    pub value_dim: usize,
}

impl DeltaShape {
    /// Elements one token occupies in the packed `[Q | K | V]` projection.
    #[must_use]
    pub fn qkv_width(&self) -> usize {
        2 * self.k_heads * self.key_dim + self.v_heads * self.value_dim
    }
    /// Elements the recurrent matrix occupies.
    #[must_use]
    pub fn state_elems(&self) -> usize {
        self.v_heads * self.value_dim * self.key_dim
    }
}

/// The gated delta rule over a batch, advancing `state` in place -- the CPU reference.
///
/// `qkv` is the CONVOLVED projection, `shape.qkv_width()` per token, packed `[Q | K | V]`.
/// `log_decay` and `beta` are one value per value head per token, ALREADY reduced: the
/// caller has applied `a * softplus(alpha + dt_bias)` and `sigmoid` respectively.
///
/// Per value head `h`, reading key head `h % k_heads`:
///
/// ```text
///   S      *= exp(g[h])
///   sk[j]   = SUM_i S[j][i] * khat[i]      what the state already remembers of k
///   d[j]    = (v[j] - sk[j]) * beta[h]     the delta rule's correction
///   S[j][i]+= khat[i] * d[j]               rank-one update
///   o[j]    = SUM_i S[j][i] * qhat[i]      read it back with the query
/// ```
///
/// `qhat` and `khat` are L2-normalised per head; `qhat` is then scaled by
/// `1 / sqrt(key_dim)` BEFORE the state product, which is where the reference scales and
/// therefore where the rounding happens.
///
/// KEY HEAD MAPPING. The reference widens Q and K from `k_heads` to `v_heads` with
/// `ggml_repeat_4d`, and ggml's repeat TILES -- 16 heads over 48 give [0..15, 0..15,
/// 0..15]. So value head `h` reads key head `h % k_heads`, NOT `h / group`. A grouped-
/// query attention in the same model uses `h / group`, because there the widening is a
/// strided view. Both mappings are right for their own tensor, and writing one of them
/// twice is a wrong answer that still runs.
///
/// # Panics
/// Panics when any slice length disagrees with `shape` and `n_tok`.
#[allow(clippy::too_many_arguments)]
pub fn delta_net(
    qkv: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    shape: DeltaShape,
    n_tok: usize,
    eps: f32,
) {
    let (kd, vd) = (shape.key_dim, shape.value_dim);
    let (kh, vh) = (shape.k_heads, shape.v_heads);
    let qkv_width = shape.qkv_width();
    let v_width = vh * vd;
    assert_eq!(qkv.len(), n_tok * qkv_width, "qkv is n_tok x qkv_width");
    assert_eq!(log_decay.len(), n_tok * vh, "log_decay is n_tok x v_heads");
    assert_eq!(beta.len(), n_tok * vh, "beta is n_tok x v_heads");
    assert_eq!(state.len(), shape.state_elems(), "state is v_heads x value_dim x key_dim");
    assert_eq!(out.len(), n_tok * v_width, "out is n_tok x v_heads x value_dim");

    let qscale = 1.0 / (kd as f32).sqrt();
    let mut qn = vec![0.0_f32; kh * kd];
    let mut kn = vec![0.0_f32; kh * kd];
    let mut d = vec![0.0_f32; vd];
    for t in 0..n_tok {
        let row = &qkv[t * qkv_width..(t + 1) * qkv_width];
        qn.copy_from_slice(&row[..kh * kd]);
        kn.copy_from_slice(&row[kh * kd..2 * kh * kd]);
        let v = &row[2 * kh * kd..];
        // L2 over the KEY heads, before the tiling below reads them. `max(norm, eps)`,
        // not `norm + eps`: the second shrinks every vector slightly.
        for head in qn.chunks_exact_mut(kd) {
            l2_normalize_max_eps(head, eps);
        }
        for head in kn.chunks_exact_mut(kd) {
            l2_normalize_max_eps(head, eps);
        }
        for value in &mut qn {
            *value *= qscale;
        }
        for h in 0..vh {
            let kb = (h % kh) * kd;
            let vb = h * vd;
            let sb = h * vd * kd;
            let decay = log_decay[t * vh + h].exp();
            let bt = beta[t * vh + h];
            for cell in &mut state[sb..sb + vd * kd] {
                *cell *= decay;
            }
            for (j, dj) in d.iter_mut().enumerate() {
                let r = sb + j * kd;
                let mut remembered = 0.0_f32;
                for i in 0..kd {
                    remembered += state[r + i] * kn[kb + i];
                }
                *dj = (v[vb + j] - remembered) * bt;
            }
            for (j, &dj) in d.iter().enumerate() {
                let r = sb + j * kd;
                for i in 0..kd {
                    state[r + i] += kn[kb + i] * dj;
                }
            }
            for (j, o) in out[t * v_width + vb..t * v_width + vb + vd].iter_mut().enumerate() {
                let r = sb + j * kd;
                let mut projected = 0.0_f32;
                for i in 0..kd {
                    projected += state[r + i] * qn[kb + i];
                }
                *o = projected;
            }
        }
    }
}

/// The chunk width `delta_net_chunked` takes when a caller has no reason to pick another.
/// 64 is `CS` in our llama.cpp fork's scalar-decay `build_delta_net_chunking`, and it is
/// the width a GPU threadgroup will hold: the C x C matrices below are the working set.
pub const DELTA_CHUNK: usize = 64;

/// The gated delta rule again, a CHUNK of tokens at a time -- the same answer as
/// `delta_net` with a serial depth of `n_tok / chunk` steps instead of `n_tok`.
///
/// `delta_net` is the right shape for decode, where `n_tok` is 1 and there is nothing to
/// amortise. At prefill its depth IS the token count: 512 tokens are 512 dependent steps
/// per head, each one a rank-one update too small to fill a machine. This form unrolls the
/// recurrence inside a chunk, which leaves matrix products everywhere except the carry from
/// one chunk to the next. With `gc[t] = SUM_{u<=t} g[u]` taken inside the chunk:
///
/// ```text
///   S_t = exp(gc[t]) S0 + SUM_{u<=t} exp(gc[t]-gc[u]) d_u k_u^T
/// ```
///
/// Substituting that into `d_t = beta_t (v_t - S'_t k_t)` leaves a unit lower triangular
/// system in `d`, and that system is the whole trick:
///
/// ```text
///   PARALLEL over (value head, chunk)          C tokens, dk = key_dim, dv = value_dim
///     D   = exp(gc_t - gc_u), u <= t                         [C,C]
///     A   = (K_b K^T) (*) D, STRICTLY lower                  [C,C]   K_b = k (*) beta
///     Kq  = (Q K^T)   (*) D, lower INCLUDING the diagonal    [C,C]
///     T   = (I + A)^-1                                       [C,C]   unit lower triangular
///     Vb' = T @ (v (*) beta)                                 [C,dv]
///     Kcd = T @ (K_b (*) exp(gc))                            [C,dk]
///   SEQUENTIAL over chunks -- the only part that carries
///     d = Vb' - Kcd @ S                                      [C,dv]
///     O = (Q (*) exp(gc)) @ S^T + Kq @ d                      [C,dv]
///     S = exp(gc[C-1]) S + d^T @ (K (*) exp(gc[C-1]-gc))     [dv,dk]
/// ```
///
/// `Kq` keeps its diagonal because `o_t` reads the state AFTER token t's own update, which
/// is what `delta_net` does; `A` drops it because `d_t` is what token t adds.
///
/// Two things this trades. It computes about 1.8x the arithmetic (the C x C matrices and
/// the triangular inverse are new work), and it is NOT bit-identical to `delta_net`: the
/// same quantities are summed in a different order and pass through the inverse. It is the
/// same function of the same inputs, checked to a tolerance in this crate's tests.
///
/// Every exponent here is `<= 0` -- `log_decay` is a log of a decay and never positive, so
/// `gc` descends and every difference taken above is negative-or-zero. Nothing overflows.
///
/// # Panics
/// Panics when `chunk` is zero, or any slice length disagrees with `shape` and `n_tok`.
#[allow(clippy::too_many_arguments)]
pub fn delta_net_chunked(
    qkv: &[f32],
    log_decay: &[f32],
    beta: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    shape: DeltaShape,
    n_tok: usize,
    eps: f32,
    chunk: usize,
) {
    let (kd, vd) = (shape.key_dim, shape.value_dim);
    let (kh, vh) = (shape.k_heads, shape.v_heads);
    let qkv_width = shape.qkv_width();
    let v_width = vh * vd;
    let k_width = kh * kd;
    assert!(chunk > 0, "chunk is at least one token");
    assert_eq!(qkv.len(), n_tok * qkv_width, "qkv is n_tok x qkv_width");
    assert_eq!(log_decay.len(), n_tok * vh, "log_decay is n_tok x v_heads");
    assert_eq!(beta.len(), n_tok * vh, "beta is n_tok x v_heads");
    assert_eq!(state.len(), shape.state_elems(), "state is v_heads x value_dim x key_dim");
    assert_eq!(out.len(), n_tok * v_width, "out is n_tok x v_heads x value_dim");
    if n_tok == 0 {
        return;
    }

    // Normalise once for the whole batch, per token and per KEY head, and scale q where
    // `delta_net` scales it -- same operation in the same place, so the same rounding.
    let qscale = 1.0 / (kd as f32).sqrt();
    let mut qn = vec![0.0_f32; n_tok * k_width];
    let mut kn = vec![0.0_f32; n_tok * k_width];
    for t in 0..n_tok {
        let row = &qkv[t * qkv_width..(t + 1) * qkv_width];
        let q = &mut qn[t * k_width..(t + 1) * k_width];
        q.copy_from_slice(&row[..k_width]);
        for head in q.chunks_exact_mut(kd) {
            l2_normalize_max_eps(head, eps);
            for value in head.iter_mut() {
                *value *= qscale;
            }
        }
        let k = &mut kn[t * k_width..(t + 1) * k_width];
        k.copy_from_slice(&row[k_width..2 * k_width]);
        for head in k.chunks_exact_mut(kd) {
            l2_normalize_max_eps(head, eps);
        }
    }

    let cmax = chunk.min(n_tok);
    let mut gc = vec![0.0_f32; cmax];
    let mut eg = vec![0.0_f32; cmax];
    let mut lower = vec![0.0_f32; cmax * cmax]; // A, strictly lower
    let mut tri = vec![0.0_f32; cmax * cmax]; // T = (I + A)^-1
    let mut kq = vec![0.0_f32; cmax * cmax];
    let mut vbt = vec![0.0_f32; cmax * vd];
    let mut kcd = vec![0.0_f32; cmax * kd];
    let mut delta = vec![0.0_f32; cmax * vd];

    for h in 0..vh {
        let kb = (h % kh) * kd;
        let vb = h * vd;
        let sb = h * vd * kd;
        let mut t0 = 0;
        while t0 < n_tok {
            let c = chunk.min(n_tok - t0);
            let mut acc = 0.0_f32;
            for t in 0..c {
                acc += log_decay[(t0 + t) * vh + h];
                gc[t] = acc;
                eg[t] = acc.exp();
            }
            let last = gc[c - 1];

            for t in 0..c {
                let bt = beta[(t0 + t) * vh + h];
                let qt = &qn[(t0 + t) * k_width + kb..][..kd];
                let kt = &kn[(t0 + t) * k_width + kb..][..kd];
                for u in 0..c {
                    if u > t {
                        kq[t * c + u] = 0.0;
                        lower[t * c + u] = 0.0;
                        continue;
                    }
                    let ku = &kn[(t0 + u) * k_width + kb..][..kd];
                    let decay = (gc[t] - gc[u]).exp();
                    kq[t * c + u] = decay * dot_f32(qt, ku);
                    // A is STRICTLY lower: token t's own delta is what the row solves for.
                    let a = if u == t { 0.0 } else { bt * decay * dot_f32(kt, ku) };
                    lower[t * c + u] = a;
                }
            }

            // T = (I + A)^-1 for a unit lower triangular A: T[t][t] = 1 and
            // T[t][u] = -SUM_{w=u..t-1} A[t][w] T[w][u]. This is `ggml_solve_tri` in the
            // reference graph; the GPU form wants the inverse itself, not a solve, because
            // it multiplies two right-hand sides by it.
            for t in 0..c {
                for u in 0..t {
                    let mut sum = 0.0_f32;
                    for w in u..t {
                        sum += lower[t * c + w] * tri[w * c + u];
                    }
                    tri[t * c + u] = -sum;
                }
                tri[t * c + t] = 1.0;
                for u in t + 1..c {
                    tri[t * c + u] = 0.0;
                }
            }

            for t in 0..c {
                for j in 0..vd {
                    let mut sum = 0.0_f32;
                    for u in 0..=t {
                        let bu = beta[(t0 + u) * vh + h];
                        let v = qkv[(t0 + u) * qkv_width + 2 * k_width + vb + j];
                        sum += tri[t * c + u] * bu * v;
                    }
                    vbt[t * vd + j] = sum;
                }
                for i in 0..kd {
                    let mut sum = 0.0_f32;
                    for u in 0..=t {
                        let bu = beta[(t0 + u) * vh + h];
                        let ki = kn[(t0 + u) * k_width + kb + i];
                        sum += tri[t * c + u] * bu * eg[u] * ki;
                    }
                    kcd[t * kd + i] = sum;
                }
            }

            // d = Vb' - Kcd @ S, against the state as it stands at the chunk's START.
            for t in 0..c {
                for j in 0..vd {
                    let r = sb + j * kd;
                    let mut sum = 0.0_f32;
                    for i in 0..kd {
                        sum += kcd[t * kd + i] * state[r + i];
                    }
                    delta[t * vd + j] = vbt[t * vd + j] - sum;
                }
            }

            // O = (Q (*) exp(gc)) @ S^T + Kq @ d, still against the chunk-start state.
            for t in 0..c {
                let qt = &qn[(t0 + t) * k_width + kb..][..kd];
                for j in 0..vd {
                    let r = sb + j * kd;
                    let mut sum = 0.0_f32;
                    for i in 0..kd {
                        sum += eg[t] * qt[i] * state[r + i];
                    }
                    for u in 0..=t {
                        sum += kq[t * c + u] * delta[u * vd + j];
                    }
                    out[(t0 + t) * v_width + vb + j] = sum;
                }
            }

            // The carry: everything the chunk added, decayed to its last position.
            for j in 0..vd {
                let r = sb + j * kd;
                for i in 0..kd {
                    let mut sum = state[r + i] * last.exp();
                    for u in 0..c {
                        let ki = kn[(t0 + u) * k_width + kb + i];
                        sum += delta[u * vd + j] * (last - gc[u]).exp() * ki;
                    }
                    state[r + i] = sum;
                }
            }
            t0 += c;
        }
    }
}

/// NEOX rope: dimension i pairs with i + n_rot/2, not with i+1.
///
/// `freq_factors`, when present, DIVIDES theta per pair -- that is how gemma4's
/// full-attention layers get proportional rope while its windowed layers do not.
pub fn rope_neox(
    head: &mut [f32],
    position: u32,
    n_rot: usize,
    base: f32,
    freq_factors: Option<&[f32]>,
) {
    let half = n_rot / 2;
    for i in 0..half {
        let inv = base.powf(-2.0 * (i as f32) / (n_rot as f32));
        let ff = freq_factors.map_or(1.0, |f| f[i]);
        let theta = (f64::from(position) * f64::from(inv / ff)) as f32;
        let (sin, cos) = theta.sin_cos();
        let x0 = head[i];
        let x1 = head[i + half];
        head[i] = x0 * cos - x1 * sin;
        head[i + half] = x0 * sin + x1 * cos;
    }
}

/// In-place softmax with the usual max subtraction.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

pub fn add_into(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

pub fn mul_into(dst: &mut [f32], src: &[f32]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d *= *s;
    }
}

pub fn scale(x: &mut [f32], k: f32) {
    for v in x.iter_mut() {
        *v *= k;
    }
}

/// logit softcapping: cap * tanh(x / cap)
pub fn softcap(x: &mut [f32], cap: f32) {
    let inv = 1.0 / cap;
    for v in x.iter_mut() {
        *v = cap * (*v * inv).tanh();
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (mut s0, mut s1, mut s2, mut s3) = (0.0_f32, 0.0, 0.0, 0.0);
    let chunks = a.len() / 4;
    for c in 0..chunks {
        let i = c * 4;
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in chunks * 4..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Greedy pick over logits: the smallest index attaining the maximum. The CPU
/// fallback for sampling; the GPU `imparo_argmax`/`k_argmax` kernels implement the
/// SAME smallest-index-at-max rule, and `Gemma4::forward_next` relies on the agreement.
///
/// Two passes, not one: tracking the index inside the scan defeats the vectoriser
/// (0.405 ms vs 0.085 at vocab 262144). NaN never wins; all-NaN returns 0.
#[must_use]
// Exact bitwise re-find of the fold's maximum, not a tolerance question (float_cmp).
#[allow(clippy::float_cmp)]
pub fn argmax_f32(logits: &[f32]) -> u32 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    logits.iter().position(|&v| v == m).unwrap_or(0) as u32
}

use crate::quants;
use imparo_gguf::weights::{
    GGML_F32, GGML_Q4_0, GGML_Q4_0_TM, GGML_Q8_0, GGML_Q8_0_TM, Q4_0_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES, QK4_0, QK8_0, TM_RULES, Tensor, Weights, f16_to_f32,
    q8_0_tm_payload_offset, q8_0_tm_scale_offset,
};

/// Worker threads for row-parallel matmuls. IMPARO_THREADS overrides.
#[must_use]
pub fn threads() -> usize {
    std::env::var("IMPARO_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
        })
}

/// y[j] = sum_i W[j, i] * x[i], for a weight whose ne[0] is the input width.
///
/// Rows are independent, so they split across threads with no synchronisation beyond
/// the scope join. Threads are created per call; at these matrix sizes the spawn cost
/// is small against the work, and a pool would add state this reference does not need.
///
/// # Panics
///
/// Panics on an unsupported quantisation or a length mismatch.
pub fn mul_mat(weights: &Weights, w: &Tensor, x: &[f32], y: &mut [f32]) {
    let n_in = w.ne0();
    let n_out = w.ne1();
    assert_eq!(x.len(), n_in, "mul_mat input width");
    assert_eq!(y.len(), n_out, "mul_mat output width");
    let threads = threads();
    if threads <= 1 || n_out < 64 {
        mul_mat_range(weights, w, x, y, 0);
        return;
    }
    let chunk = n_out.div_ceil(threads);
    std::thread::scope(|s| {
        for (t, part) in y.chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            s.spawn(move || mul_mat_range(weights, w, x, part, base));
        }
    });
}

/// Batched: `x` is `n_tokens` inputs of width ne0, `y` is `n_tokens` outputs of ne1,
/// both token-major.
///
/// This is where prefill speed comes from. Per-token `mul_mat` re-reads -- and for
/// Q4_0 re-dequantises -- every weight row for every token, so a 300-token prompt
/// walks 4.2 GB of weights 300 times. Here a row is read and dequantised ONCE and
/// dotted against all tokens, so weight traffic is amortised over the batch.
///
/// # Panics
///
/// Panics on an unsupported quantisation or a length mismatch.
pub fn mul_mat_batch(
    weights: &Weights,
    w: &Tensor,
    x: &[f32],
    n_tokens: usize,
    y: &mut [f32],
) {
    let n_in = w.ne0();
    let n_out = w.ne1();
    assert_eq!(x.len(), n_in * n_tokens, "mul_mat_batch input");
    assert_eq!(y.len(), n_out * n_tokens, "mul_mat_batch output");
    if n_tokens == 1 {
        mul_mat(weights, w, x, y);
        return;
    }
    let threads = threads().min(n_out.max(1));
    let chunk = n_out.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        // split the OUTPUT ROWS across threads; each thread writes a strided set of
        // columns across all tokens, so no two threads touch the same element
        let y_ptr = YPtr(y.as_mut_ptr());
        for t in 0..threads {
            let base = t * chunk;
            if base >= n_out {
                break;
            }
            let rows = chunk.min(n_out - base);
            let yp = y_ptr;
            s.spawn(move || {
                let mut row_buf = vec![0.0_f32; n_in];
                for r in base..base + rows {
                    row_f32(weights, w, r, &mut row_buf);
                    for t2 in 0..n_tokens {
                        let v = dot_f32(&row_buf, &x[t2 * n_in..(t2 + 1) * n_in]);
                        // SAFETY: each (t2, r) element is written by exactly one thread.
                        unsafe { yp.write_at(t2 * n_out + r, v) };
                    }
                }
            });
        }
    });
}

/// Materialises one weight row as f32, dequantising when needed.
fn row_f32(weights: &Weights, w: &Tensor, r: usize, out: &mut [f32]) {
    row_bytes(weights.raw(w), w.ggml_type, w.ne0(), r, out);
}

/// One weight row, dequantised, from a tensor's RAW BYTES.
///
/// The same body `row_f32` uses. Split out because the `Backend` seam addresses
/// weights by (kind, byte offset) rather than by `Tensor` -- and one dequant
/// implementation is the point: two would be two chances to disagree about a
/// block layout, and the disagreement reads as a slightly wrong answer.
fn row_bytes(data: &[u8], kind: u32, n_in: usize, r: usize, out: &mut [f32]) {
    match kind {
        GGML_F32 => {
            let lo = r * n_in * 4;
            for (o, c) in out.iter_mut().zip(data[lo..lo + n_in * 4].chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        // Tile-major kinds first: their bytes are not a row slice, so the row's address
        // comes from the TmRule, not from `row_bytes_len`. Q8_0 and Q4_0 keep hand-written
        // readers because their dots are the CPU's fast path; every other TM kind goes
        // through the generic gather, which is the same row codec at the rule's addresses.
        GGML_Q8_0_TM => dequant_q8_0_tm_row(data, n_in, r, out),
        GGML_Q4_0_TM => dequant_q4_0_tm_row(data, n_in, r, out),
        other if imparo_gguf::weights::tm_rule_to(other).is_some() => {
            quants::tm_row(data, other, n_in, r, out);
        }
        // Every row-major block type is one entry in `quants::row_codec`: the row is a
        // whole number of blocks at a derived offset, and the codec is the format.
        other => {
            let codec = quants::row_codec(other)
                .unwrap_or_else(|| panic!("row_bytes: unsupported ggml type {other}"));
            let rb = row_bytes_len(other, n_in);
            codec(&data[r * rb..(r + 1) * rb], out);
        }
    }
}

/// Bytes one row of `kind` occupies -- what a caller slicing a weight blob needs.
///
/// Derived from `imparo_gguf::tensor_layout`, the ONE table that says how many elements a
/// block holds and how many bytes it takes. It used to repeat the Q4_0 / Q8_0 constants
/// here, which is the same duplication `TmRule` exists to prevent: a second copy of a
/// layout is a second chance to disagree with it. A tile-major kind keeps its row-major
/// size by construction, so it needs no case of its own.
///
/// # Panics
/// On a type the layout table does not know, or a row width that is not a whole number of
/// blocks -- both are load-time errors that must not reach a kernel as a wrong slice.
#[must_use]
pub fn row_bytes_len(kind: u32, n_in: usize) -> usize {
    let l = imparo_gguf::tensor_layout(kind)
        .unwrap_or_else(|_| panic!("row_bytes_len: unsupported ggml type {kind}"));
    let (elems, bytes) = (l.block_elements as usize, l.block_bytes as usize);
    assert_eq!(
        n_in % elems,
        0,
        "{}: row width {n_in} is not a whole number of {elems}-element blocks",
        l.name
    );
    n_in / elems * bytes
}

/// `y[t, o] = dot(W[o], x[t])` over a weight blob addressed by BYTES.
///
/// The `Backend` seam's matmul: same math and same dequant as `mul_mat_batch`,
/// reached without a `Tensor`. `blob` is the weight matrix's own bytes,
/// `n_out` rows of `row_bytes_len(kind, n_in)` each.
///
/// # Panics
/// When the blob is shorter than the shape says, or the kind is unsupported.
pub fn mul_mat_bytes(
    blob: &[u8],
    kind: u32,
    n_in: usize,
    n_out: usize,
    x: &[f32],
    n_tokens: usize,
    y: &mut [f32],
) {
    let rb = row_bytes_len(kind, n_in);
    assert!(
        blob.len() >= n_out * rb,
        "weight blob shorter than its shape"
    );
    assert_eq!(x.len(), n_in * n_tokens, "mul_mat_bytes input");
    assert_eq!(y.len(), n_out * n_tokens, "mul_mat_bytes output");
    let threads = threads().min(n_out.max(1));
    let chunk = n_out.div_ceil(threads.max(1));
    std::thread::scope(|s| {
        let y_ptr = YPtr(y.as_mut_ptr());
        for t in 0..threads {
            let base = t * chunk;
            if base >= n_out {
                break;
            }
            let rows = chunk.min(n_out - base);
            let yp = y_ptr;
            s.spawn(move || {
                let mut row_buf = vec![0.0_f32; n_in];
                for r in base..base + rows {
                    row_bytes(blob, kind, n_in, r, &mut row_buf);
                    for t2 in 0..n_tokens {
                        let v = dot_f32(&row_buf, &x[t2 * n_in..(t2 + 1) * n_in]);
                        // SAFETY: each (t2, r) element is written by exactly one thread.
                        unsafe { yp.write_at(t2 * n_out + r, v) };
                    }
                }
            });
        }
    });
}

/// One embedding row from a weight blob addressed by BYTES; see `mul_mat_bytes`.
///
/// # Panics
/// When `out` is not `width` long, or the kind is unsupported.
pub fn row_from_bytes(
    blob: &[u8],
    kind: u32,
    width: usize,
    index: usize,
    out: &mut [f32],
) {
    assert_eq!(out.len(), width, "row width");
    row_bytes(blob, kind, width, index, out);
}

/// Computes output rows `[base, base + y.len())` into `y`.
fn mul_mat_range(weights: &Weights, w: &Tensor, x: &[f32], y: &mut [f32], base: usize) {
    let n_in = w.ne0();
    match w.ggml_type {
        GGML_F32 => {
            let data = weights.f32s(w);
            for (j, out) in y.iter_mut().enumerate() {
                let r = base + j;
                *out = dot_f32(&data[r * n_in..(r + 1) * n_in], x);
            }
        }
        GGML_Q4_0 => {
            let data = weights.raw(w);
            assert_eq!(n_in % QK4_0, 0, "Q4_0 row not a multiple of 32");
            let bpr = n_in / QK4_0;
            for (j, out) in y.iter_mut().enumerate() {
                let r = base + j;
                *out = dot_q4_0_row(
                    &data[r * bpr * Q4_0_BLOCK_BYTES..(r + 1) * bpr * Q4_0_BLOCK_BYTES],
                    x,
                );
            }
        }
        GGML_Q8_0 => {
            let data = weights.raw(w);
            assert_eq!(n_in % QK8_0, 0, "Q8_0 row not a multiple of 32");
            let bpr = n_in / QK8_0;
            for (j, out) in y.iter_mut().enumerate() {
                let r = base + j;
                *out = dot_q8_0_row(
                    &data[r * bpr * Q8_0_BLOCK_BYTES..(r + 1) * bpr * Q8_0_BLOCK_BYTES],
                    x,
                );
            }
        }
        GGML_Q8_0_TM => {
            let data = weights.raw(w);
            for (j, out) in y.iter_mut().enumerate() {
                *out = dot_q8_0_tm_row(data, n_in, base + j, x);
            }
        }
        GGML_Q4_0_TM => {
            let data = weights.raw(w);
            for (j, out) in y.iter_mut().enumerate() {
                *out = dot_q4_0_tm_row(data, n_in, base + j, x);
            }
        }
        // Dequantise the row, then dot. The arms above are OPTIMISATIONS (a fused dot for
        // the kinds a model actually runs on the CPU); this is the route every other
        // format takes, and it is correct for all of them. The CPU backend is the ORACLE,
        // not a fast path: one correct block interior beats a hand-written dot per quant.
        // If a format ever becomes hot here, it earns a dot_* like Q4_0's.
        _ => {
            let data = weights.raw(w);
            let mut row = vec![0.0_f32; n_in];
            for (j, out) in y.iter_mut().enumerate() {
                row_bytes(data, w.ggml_type, n_in, base + j, &mut row);
                *out = dot_f32(&row, x);
            }
        }
    }
}

/// Copies one row of a 2-D tensor (an embedding lookup) into `out`.
///
/// # Panics
///
/// Panics on an unsupported quantisation or a length mismatch.
pub fn row(weights: &Weights, t: &Tensor, index: usize, out: &mut [f32]) {
    let width = t.ne0();
    assert_eq!(out.len(), width, "row width");
    match t.ggml_type {
        GGML_F32 => {
            out.copy_from_slice(&weights.f32s(t)[index * width..(index + 1) * width]);
        }
        // Through row_bytes on purpose: it IS the dequant implementation, and a second
        // copy here would be a second chance to disagree with a block interior.
        _ => row_bytes(weights.raw(t), t.ggml_type, width, index, out),
    }
}

/// Raw output pointer shared across scope threads that write disjoint elements.
#[derive(Clone, Copy)]
struct YPtr(*mut f32);
// SAFETY: threads write disjoint (token, row) elements; no aliasing writes occur.
unsafe impl Send for YPtr {}
unsafe impl Sync for YPtr {}

impl YPtr {
    /// Writes one element.
    ///
    /// Taking `self` by value matters: with 2021 disjoint capture a closure that touches
    /// `yp.0` captures the raw pointer field, which is not `Send`, so the wrapper's
    /// `Send` impl never applies. A method on `self` captures the whole struct.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds and written by exactly one thread.
    unsafe fn write_at(self, index: usize, value: f32) {
        unsafe { *self.0.add(index) = value };
    }
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // four accumulators: the compiler can keep these in registers and the partial sums
    // are independent, which is what lets this vectorise at all.
    let (mut s0, mut s1, mut s2, mut s3) = (0.0_f32, 0.0, 0.0, 0.0);
    let chunks = a.len() / 4;
    for c in 0..chunks {
        let i = c * 4;
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in chunks * 4..a.len() {
        s += a[i] * b[i];
    }
    s
}

fn dot_q4_0_row(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (b, chunk) in row.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        let base = b * QK4_0;
        let mut block = 0.0_f32;
        for i in 0..QK4_0 / 2 {
            let byte = chunk[2 + i];
            block += (f32::from(byte & 0x0F) - 8.0) * x[base + i];
            block += (f32::from(byte >> 4) - 8.0) * x[base + i + QK4_0 / 2];
        }
        acc += block * d;
    }
    acc
}

/// Q8_0_TM: the same values as Q8_0 read from the tile-major layout (`data` is the WHOLE
/// tensor, `n_in` its row width, `r` the row). The row count comes from the blob's length,
/// which is why no caller signature changed. Same accumulation order as the row-major
/// readers, so the oracle's numbers are the GGUF's numbers.
fn q8_0_tm_rows(data: &[u8], n_in: usize) -> usize {
    assert_eq!(n_in % QK8_0, 0, "Q8_0_TM row not a multiple of 32");
    data.len() / (n_in / QK8_0 * Q8_0_BLOCK_BYTES)
}

fn dequant_q8_0_tm_row(data: &[u8], n_in: usize, r: usize, out: &mut [f32]) {
    let blocks = n_in / QK8_0;
    let n_out = q8_0_tm_rows(data, n_in);
    for b in 0..blocks {
        let p = q8_0_tm_payload_offset(r, b, blocks);
        let s = q8_0_tm_scale_offset(r, b, blocks, n_out);
        let d = f16_to_f32(u16::from_le_bytes([data[s], data[s + 1]]));
        let base = b * QK8_0;
        for i in 0..QK8_0 {
            out[base + i] = f32::from(data[p + i] as i8) * d;
        }
    }
}

fn dot_q8_0_tm_row(data: &[u8], n_in: usize, r: usize, x: &[f32]) -> f32 {
    let blocks = n_in / QK8_0;
    let n_out = q8_0_tm_rows(data, n_in);
    let mut acc = 0.0_f32;
    for b in 0..blocks {
        let p = q8_0_tm_payload_offset(r, b, blocks);
        let s = q8_0_tm_scale_offset(r, b, blocks, n_out);
        let d = f16_to_f32(u16::from_le_bytes([data[s], data[s + 1]]));
        let base = b * QK8_0;
        let mut block = 0.0_f32;
        for i in 0..QK8_0 {
            block += f32::from(data[p + i] as i8) * x[base + i];
        }
        acc += block * d;
    }
    acc
}

/// Q4_0_TM: the same values as Q4_0 read from the tile-major layout (16-byte payload per
/// block, scales after the payload); `data` is the WHOLE tensor. Same nibble order and the
/// same accumulation order as `dot_q4_0_row` / `dequant_q4_0_row`.
fn q4_0_tm_rows(data: &[u8], n_in: usize) -> usize {
    assert_eq!(n_in % QK4_0, 0, "Q4_0_TM row not a multiple of 32");
    data.len() / (n_in / QK4_0 * Q4_0_BLOCK_BYTES)
}

fn dequant_q4_0_tm_row(data: &[u8], n_in: usize, r: usize, out: &mut [f32]) {
    let rule = &TM_RULES[1];
    let blocks = n_in / QK4_0;
    let n_out = q4_0_tm_rows(data, n_in);
    for b in 0..blocks {
        let p = rule.payload_offset(r, b, blocks);
        let s = rule.scale_offset(r, b, blocks, n_out);
        let d = f16_to_f32(u16::from_le_bytes([data[s], data[s + 1]]));
        let base = b * QK4_0;
        for i in 0..QK4_0 / 2 {
            let byte = data[p + i];
            out[base + i] = (f32::from(byte & 0x0F) - 8.0) * d;
            out[base + i + QK4_0 / 2] = (f32::from(byte >> 4) - 8.0) * d;
        }
    }
}

fn dot_q4_0_tm_row(data: &[u8], n_in: usize, r: usize, x: &[f32]) -> f32 {
    let rule = &TM_RULES[1];
    let blocks = n_in / QK4_0;
    let n_out = q4_0_tm_rows(data, n_in);
    let mut acc = 0.0_f32;
    for b in 0..blocks {
        let p = rule.payload_offset(r, b, blocks);
        let s = rule.scale_offset(r, b, blocks, n_out);
        let d = f16_to_f32(u16::from_le_bytes([data[s], data[s + 1]]));
        let base = b * QK4_0;
        let mut block = 0.0_f32;
        for i in 0..QK4_0 / 2 {
            let byte = data[p + i];
            block += (f32::from(byte & 0x0F) - 8.0) * x[base + i];
            block += (f32::from(byte >> 4) - 8.0) * x[base + i + QK4_0 / 2];
        }
        acc += block * d;
    }
    acc
}

/// Block-local sum then one multiply by the scale -- the SAME accumulation order as
/// `dot_q4_0_row`, so the two quants differ in layout and nothing else.
fn dot_q8_0_row(row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for (b, chunk) in row.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        let base = b * QK8_0;
        let mut block = 0.0_f32;
        for i in 0..QK8_0 {
            block += f32::from(chunk[2 + i] as i8) * x[base + i];
        }
        acc += block * d;
    }
    acc
}

#[cfg(test)]
// Every operand and scale below is an exact binary fraction, so equality is the right
// assertion: these tests guard the byte layout and the accumulation order, not a
// tolerance.
#[allow(clippy::float_cmp)]
mod tests {
    use super::{ConvForm, Q8_0_BLOCK_BYTES, QK8_0, causal_conv, dot_q8_0_row};

    /// The Q8_0 row codec, through the one table every reader goes through.
    fn dequant_q8_0_row(row: &[u8], out: &mut [f32]) {
        crate::quants::row_codec(imparo_gguf::weights::GGML_Q8_0).unwrap()(row, out);
    }
    use imparo_gguf::weights::f32_to_f16;

    /// One Q8_0 block: f16 scale in bytes 0..2, then 32 signed values.
    fn block(scale: f32, values: &[i8; QK8_0]) -> [u8; Q8_0_BLOCK_BYTES] {
        let mut out = [0u8; Q8_0_BLOCK_BYTES];
        out[..2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        for (i, &v) in values.iter().enumerate() {
            out[2 + i] = v as u8;
        }
        out
    }

    /// Element i is byte 2+i and it is SIGNED. A Q4_0-shaped reader would take the low
    /// nibble of byte 2+i instead, so -1 (0xFF) would read as 15 - 8 = 7.
    #[test]
    fn a_q8_0_byte_is_the_value_and_it_is_signed() {
        let mut v = [0i8; QK8_0];
        v[0] = 127;
        v[1] = -1;
        v[2] = -128;
        v[31] = 3;
        let row = block(0.5, &v);
        let mut out = [0.0f32; QK8_0];
        dequant_q8_0_row(&row, &mut out);
        assert_eq!(out[0], 63.5);
        assert_eq!(out[1], -0.5);
        assert_eq!(out[2], -64.0);
        assert_eq!(out[31], 1.5);
        assert_eq!(out[3], 0.0);
    }

    /// A batch processed in ONE call must equal the same tokens fed one at a time.
    ///
    /// This is the property the whole design rests on: the state carries a conversation
    /// across batch boundaries, so prefill-then-decode has to agree with decode-only. It
    /// also catches the tap ordering, because feeding one token at a time makes the state
    /// the only path by which history reaches an output.
    #[test]
    fn a_batch_equals_the_same_tokens_one_at_a_time() {
        let (width, kernel, n_tok) = (4usize, 3usize, 5usize);
        let bcx: Vec<f32> = (0..n_tok * 3 * width)
            .map(|i| ((i % 7) as f32) - 3.0)
            .collect();
        let conv_w: Vec<f32> = (0..width * kernel)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.25)
            .collect();

        let mut s_batch = vec![0.0f32; (kernel - 1) * width];
        let mut out_batch = vec![0.0f32; n_tok * width];
        causal_conv(
            ConvForm::GatedBcx,
            &bcx,
            &conv_w,
            &mut s_batch,
            &mut out_batch,
            width,
            kernel,
            n_tok,
        );

        let mut s_one = vec![0.0f32; (kernel - 1) * width];
        let mut out_one = vec![0.0f32; n_tok * width];
        for t in 0..n_tok {
            let row = &bcx[t * 3 * width..(t + 1) * 3 * width];
            let mut o = vec![0.0f32; width];
            causal_conv(ConvForm::GatedBcx, row, &conv_w, &mut s_one, &mut o, width, kernel, 1);
            out_one[t * width..(t + 1) * width].copy_from_slice(&o);
        }
        assert_eq!(out_batch, out_one, "batched and stepwise outputs differ");
        assert_eq!(s_batch, s_one, "batched and stepwise states differ");
    }

    /// Tap 0 multiplies the OLDEST value. With a single tap live, the output is that tap
    /// times a known element of the window, so a reversed kernel shows up immediately.
    #[test]
    fn tap_zero_is_the_oldest_value() {
        let (width, kernel) = (1usize, 3usize);
        // b = x = 1 for every token, so b*x is 1 and the conv sees a clean window.
        let bcx: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 2 tokens x (b,c,x)
        let mut state = vec![5.0, 7.0]; // oldest first: seq = [5, 7, 1, 1]
        let mut out = vec![0.0f32; 2];

        // Only tap 0 is non-zero: out[t] = c * seq[t + 0].
        let conv_w = vec![1.0, 0.0, 0.0];
        causal_conv(ConvForm::GatedBcx, &bcx, &conv_w, &mut state, &mut out, width, kernel, 2);
        assert_eq!(
            out,
            vec![5.0, 7.0],
            "tap 0 must select the oldest window value"
        );
        // The new state is the last two of seq = [5,7,1,1] -> [1, 1].
        assert_eq!(state, vec![1.0, 1.0]);

        // Only the LAST tap: out[t] = c * seq[t + 2], the current token.
        let mut state2 = vec![5.0, 7.0];
        let mut out2 = vec![0.0f32; 2];
        let conv_w2 = vec![0.0, 0.0, 1.0];
        causal_conv(ConvForm::GatedBcx, &bcx, &conv_w2, &mut state2, &mut out2, width, kernel, 2);
        assert_eq!(
            out2,
            vec![1.0, 1.0],
            "the last tap must select the current value"
        );
    }

    /// c gates the OUTPUT, b and x form the convolved signal. Zeroing c must zero the
    /// output while still advancing the state.
    #[test]
    fn c_gates_the_output_and_the_state_still_advances() {
        let (width, kernel, n_tok) = (2usize, 3usize, 2usize);
        let mut bcx = vec![0.0f32; n_tok * 3 * width];
        for t in 0..n_tok {
            for ch in 0..width {
                bcx[t * 3 * width + ch] = 2.0; // b
                bcx[t * 3 * width + width + ch] = 0.0; // c
                bcx[t * 3 * width + 2 * width + ch] = 3.0; // x
            }
        }
        let conv_w = vec![1.0f32; width * kernel];
        let mut state = vec![0.0f32; (kernel - 1) * width];
        let mut out = vec![9.0f32; n_tok * width];
        causal_conv(ConvForm::GatedBcx, &bcx, &conv_w, &mut state, &mut out, width, kernel, n_tok);
        assert!(out.iter().all(|&v| v == 0.0), "c = 0 must zero the output");
        // b*x = 6 flowed into the state regardless.
        assert!(state.iter().all(|&v| v == 6.0), "state must still advance");
    }

    /// The dot must equal the dequantised dot exactly: same block-local sum, same single
    /// multiply by the scale. Two blocks, so a wrong per-block scale shows up.
    #[test]
    fn the_dot_agrees_with_dequantise_then_dot() {
        let mut a = [0i8; QK8_0];
        let mut b = [0i8; QK8_0];
        for i in 0..QK8_0 {
            a[i] = (i as i8) - 16;
            b[i] = 8 - (i as i8);
        }
        let mut row = Vec::new();
        row.extend_from_slice(&block(0.25, &a));
        row.extend_from_slice(&block(2.0, &b));
        let x: Vec<f32> = (0..2 * QK8_0).map(|i| ((i % 5) as f32) - 2.0).collect();

        let mut deq = vec![0.0f32; 2 * QK8_0];
        dequant_q8_0_row(&row, &mut deq);
        let mut expect = 0.0f32;
        for blk in 0..2 {
            let mut acc = 0.0f32;
            for i in 0..QK8_0 {
                let k = blk * QK8_0 + i;
                acc += deq[k] * x[k];
            }
            expect += acc;
        }
        // dot_q8_0_row sums the raw bytes first and scales once, which is the same value
        // here because each block's scale is a power of two.
        assert_eq!(dot_q8_0_row(&row, &x), expect);
    }
}
