//! Model-agnostic kernels. Nothing here knows what a gemma is.

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
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let t = *v;
        *v = t / (1.0 + (-t).exp());
    }
}

/// LFM2's gated short convolution, the CPU reference.
///
/// `bcx` is the input projection, TOKEN-MAJOR with three chunks of `width` per token:
/// b at offset 0, c at `width`, x at `2 * width` -- the order the reference's three chunk
/// views take (llama.cpp/src/models/lfm2.cpp).
///
/// ```text
/// bx      = b * x                          elementwise, per token per channel
/// seq     = state ++ bx                    causal: the state is PREPENDED
/// out[t]  = c[t] * sum_k conv_w[ch][k] * seq[t + k]
/// state'  = the last (kernel - 1) values of seq
/// ```
///
/// `conv_w` is channel-major with the tap fastest, which is what dims `(l_cache, width)`
/// mean: element (k, ch) sits at `ch * kernel + k`. Tap 0 multiplies the OLDEST value.
///
/// `state` holds `kernel - 1` past values per channel, oldest first, and is advanced in
/// place. A fresh conversation starts it at zero.
///
/// # Panics
/// Panics when any slice length disagrees with `width`, `kernel` and `n_tok`.
pub fn shortconv(
    bcx: &[f32],
    conv_w: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    width: usize,
    kernel: usize,
    n_tok: usize,
) {
    assert!(
        kernel >= 2,
        "a short conv needs at least 2 taps to carry state"
    );
    let history = kernel - 1;
    assert_eq!(bcx.len(), n_tok * 3 * width, "bcx is n_tok x 3 x width");
    assert_eq!(conv_w.len(), width * kernel, "conv_w is width x kernel");
    assert_eq!(state.len(), history * width, "state is (kernel-1) x width");
    assert_eq!(out.len(), n_tok * width, "out is n_tok x width");

    // seq[e] for e in [0, history + n_tok): the state, then this batch's b*x.
    let bx_at = |e: usize, ch: usize, state: &[f32]| -> f32 {
        if e < history {
            state[e * width + ch]
        } else {
            let t = e - history;
            let row = t * 3 * width;
            bcx[row + ch] * bcx[row + 2 * width + ch]
        }
    };

    for ch in 0..width {
        let w = &conv_w[ch * kernel..(ch + 1) * kernel];
        for t in 0..n_tok {
            let mut acc = 0.0_f32;
            for (k, &wk) in w.iter().enumerate() {
                acc += wk * bx_at(t + k, ch, state);
            }
            out[t * width + ch] = bcx[t * 3 * width + width + ch] * acc;
        }
    }

    // The new state is the tail of seq, and it must be computed from the OLD state -- so
    // every output above is finished before anything here writes. Descending would read a
    // slot this loop has already overwritten.
    let mut next = vec![0.0_f32; history * width];
    for s in 0..history {
        for ch in 0..width {
            next[s * width + ch] = bx_at(n_tok + s, ch, state);
        }
    }
    state.copy_from_slice(&next);
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
        GGML_Q4_0 => {
            assert_eq!(n_in % QK4_0, 0, "Q4_0 row not a multiple of 32");
            let bpr = n_in / QK4_0;
            dequant_q4_0_row(
                &data[r * bpr * Q4_0_BLOCK_BYTES..(r + 1) * bpr * Q4_0_BLOCK_BYTES],
                out,
            );
        }
        GGML_Q8_0 => {
            assert_eq!(n_in % QK8_0, 0, "Q8_0 row not a multiple of 32");
            let bpr = n_in / QK8_0;
            dequant_q8_0_row(
                &data[r * bpr * Q8_0_BLOCK_BYTES..(r + 1) * bpr * Q8_0_BLOCK_BYTES],
                out,
            );
        }
        GGML_Q8_0_TM => dequant_q8_0_tm_row(data, n_in, r, out),
        GGML_Q4_0_TM => dequant_q4_0_tm_row(data, n_in, r, out),
        other => panic!("row_bytes: unsupported ggml type {other}"),
    }
}

/// Bytes one row of `kind` occupies -- what a caller slicing a weight blob needs.
///
/// # Panics
/// On an unsupported quantisation.
#[must_use]
pub fn row_bytes_len(kind: u32, n_in: usize) -> usize {
    match kind {
        GGML_F32 => n_in * 4,
        GGML_Q4_0 | GGML_Q4_0_TM => n_in / QK4_0 * Q4_0_BLOCK_BYTES,
        GGML_Q8_0 | GGML_Q8_0_TM => n_in / QK8_0 * Q8_0_BLOCK_BYTES,
        other => panic!("row_bytes_len: unsupported ggml type {other}"),
    }
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
        other => panic!("mul_mat: unsupported ggml type {other}"),
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
        GGML_Q4_0 => {
            let data = weights.raw(t);
            let blocks = width / QK4_0;
            let row = &data[index * blocks * Q4_0_BLOCK_BYTES
                ..(index + 1) * blocks * Q4_0_BLOCK_BYTES];
            dequant_q4_0_row(row, out);
        }
        GGML_Q8_0 => {
            let data = weights.raw(t);
            assert_eq!(width % QK8_0, 0, "Q8_0 row not a multiple of 32");
            let blocks = width / QK8_0;
            let row = &data[index * blocks * Q8_0_BLOCK_BYTES
                ..(index + 1) * blocks * Q8_0_BLOCK_BYTES];
            dequant_q8_0_row(row, out);
        }
        GGML_Q8_0_TM => dequant_q8_0_tm_row(weights.raw(t), width, index, out),
        GGML_Q4_0_TM => dequant_q4_0_tm_row(weights.raw(t), width, index, out),
        other => panic!("row: unsupported ggml type {other}"),
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

/// One Q4_0 block is an f16 scale then 16 bytes; the LOW nibble of byte i is element i and
/// the HIGH nibble is element i+16. Getting that pairing wrong produces plausible garbage.
fn dequant_q4_0_row(row: &[u8], out: &mut [f32]) {
    for (b, chunk) in row.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        let base = b * QK4_0;
        for i in 0..QK4_0 / 2 {
            let byte = chunk[2 + i];
            out[base + i] = (f32::from(byte & 0x0F) - 8.0) * d;
            out[base + i + QK4_0 / 2] = (f32::from(byte >> 4) - 8.0) * d;
        }
    }
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

/// One Q8_0 block is an f16 scale followed by 32 SIGNED bytes, element i at byte 2+i.
/// No nibble pairing and no -8 bias -- reusing the Q4_0 reader here would read 16
/// plausible values instead of 32 correct ones.
fn dequant_q8_0_row(row: &[u8], out: &mut [f32]) {
    for (b, chunk) in row.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        let base = b * QK8_0;
        for i in 0..QK8_0 {
            out[base + i] = f32::from(chunk[2 + i] as i8) * d;
        }
    }
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
    use super::{Q8_0_BLOCK_BYTES, QK8_0, dequant_q8_0_row, dot_q8_0_row, shortconv};
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
        shortconv(
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
            shortconv(row, &conv_w, &mut s_one, &mut o, width, kernel, 1);
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
        shortconv(&bcx, &conv_w, &mut state, &mut out, width, kernel, 2);
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
        shortconv(&bcx, &conv_w2, &mut state2, &mut out2, width, kernel, 2);
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
        shortconv(&bcx, &conv_w, &mut state, &mut out, width, kernel, n_tok);
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
