//! `ops::delta_net_chunked` computes the same function as `ops::delta_net`.
//!
//! `delta_net` is the definition: one token at a time, the state updated in place. The
//! chunked form unrolls that recurrence over C tokens so all but the carry between chunks
//! becomes matrix products -- the shape a prefill GPU kernel wants, where the serial form's
//! depth is the token count. Proving the transcription here means a later mismatch on the
//! GPU is a kernel, not the arithmetic.
//!
//! The two are NOT bit-identical -- the same quantities are summed in a different order and
//! pass through a triangular inverse -- so these compare to a tolerance. Every case reports
//! its own worst error, and the bound is well above what is measured (see `agree`).

use imparo_cpu::ops::{DELTA_CHUNK, DeltaShape, delta_net, delta_net_chunked};

/// A deterministic stream. Not a good generator; it only has to be the same every run and
/// not correlated with the block structure the chunked form imposes.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005);
        self.0 = self.0.wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0 // (-1, 1)
    }
}

/// Inputs shaped like the model's: q/k/v of order 1, a decay in (0, 1) so `log_decay` is
/// negative, and beta in (0, 1) as a sigmoid produces it.
type Inputs = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
fn inputs(shape: DeltaShape, n_tok: usize, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let heads = n_tok * shape.v_heads;
    let qkv = (0..n_tok * shape.qkv_width()).map(|_| rng.next()).collect();
    let log_decay = (0..heads)
        .map(|_| -(0.02 + 0.6 * (rng.next() + 1.0)))
        .collect();
    let beta = (0..heads)
        .map(|_| 1.0 / (1.0 + (-rng.next()).exp()))
        .collect();
    let state = (0..shape.state_elems()).map(|_| 0.3 * rng.next()).collect();
    (qkv, log_decay, beta, state)
}

/// Runs both forms on the same inputs and returns (worst output error, worst state error),
/// each relative to the reference's own largest magnitude in that array.
fn agree(
    shape: DeltaShape,
    n_tok: usize,
    chunk: usize,
    seed: u64,
    eps: f32,
) -> (f32, f32) {
    let (qkv, log_decay, beta, state0) = inputs(shape, n_tok, seed);
    let v_width = shape.v_heads * shape.value_dim;

    let mut ref_state = state0.clone();
    let mut ref_out = vec![0.0_f32; n_tok * v_width];
    delta_net(
        &qkv,
        &log_decay,
        &beta,
        &mut ref_state,
        &mut ref_out,
        shape,
        n_tok,
        eps,
    );

    let mut got_state = state0;
    let mut got_out = vec![0.0_f32; n_tok * v_width];
    let (s, o) = (&mut got_state, &mut got_out);
    delta_net_chunked(&qkv, &log_decay, &beta, s, o, shape, n_tok, eps, chunk);

    let worst = |a: &[f32], b: &[f32]| -> f32 {
        let scale = a.iter().fold(0.0_f32, |m, v| m.max(v.abs())).max(1e-6);
        a.iter()
            .zip(b)
            .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
            / scale
    };
    (worst(&ref_out, &got_out), worst(&ref_state, &got_state))
}

/// The shape that exercises the key-head tiling: value head h reads key head `h % k_heads`,
/// so with 4 value heads over 2 key heads the mapping is [0, 1, 0, 1] and a `h / group`
/// reading would give [0, 0, 1, 1] -- a different answer that still runs.
fn shape() -> DeltaShape {
    DeltaShape {
        k_heads: 2,
        v_heads: 4,
        key_dim: 8,
        value_dim: 8,
    }
}

const EPS: f32 = 1e-6;
/// Well above every error measured below (worst seen: see the test output).
const BOUND: f32 = 1e-4;

#[test]
fn chunked_matches_the_serial_rule_on_a_ragged_batch() {
    // 13 tokens over a chunk of 4: three full chunks and a tail of one.
    let (out, state) = agree(shape(), 13, 4, 0x51ed, EPS);
    println!("ragged 13/4: out {out:.3e} state {state:.3e}");
    assert!(out < BOUND && state < BOUND, "out {out:e} state {state:e}");
}

#[test]
fn one_chunk_holds_the_whole_batch() {
    let (out, state) = agree(shape(), 16, 64, 0x2f10, EPS);
    println!("single chunk 16/64: out {out:.3e} state {state:.3e}");
    assert!(out < BOUND && state < BOUND, "out {out:e} state {state:e}");
}

/// A chunk of one is the serial recurrence written as 1x1 matrices: T = [1], Kq = [1].
/// It is the degenerate case, and it is where a sign or an index slip shows up alone.
#[test]
fn a_chunk_of_one_token_is_the_recurrence() {
    let (out, state) = agree(shape(), 9, 1, 0x77c3, EPS);
    println!("chunk 1, 9 tokens: out {out:.3e} state {state:.3e}");
    assert!(out < BOUND && state < BOUND, "out {out:e} state {state:e}");
}

/// qwen35's own widths at the chunk the kernel will carry -- 6 of its 48 value heads, which
/// is enough to cross the key-head tiling twice and keeps the test under a second.
#[test]
fn the_model_widths_at_the_default_chunk() {
    let shape = DeltaShape {
        k_heads: 2,
        v_heads: 6,
        key_dim: 128,
        value_dim: 128,
    };
    let (out, state) = agree(shape, 200, DELTA_CHUNK, 0x9a41, EPS);
    println!("qwen35 widths 200/{DELTA_CHUNK}: out {out:.3e} state {state:.3e}");
    assert!(out < BOUND && state < BOUND, "out {out:e} state {state:e}");
}

/// The carry is the only part that is still sequential, so a batch split across two calls
/// must land where one call of the whole batch lands. This one compares the chunked form
/// against ITSELF -- it says nothing about whether the arithmetic matches `delta_net`, which
/// is what the tests above are for.
#[test]
fn the_state_carries_across_calls() {
    let shape = shape();
    let n_tok = 12;
    let split = 5;
    let (qkv, log_decay, beta, state0) = inputs(shape, n_tok, 0x1234);
    let v_width = shape.v_heads * shape.value_dim;
    let qw = shape.qkv_width();
    let vh = shape.v_heads;

    let mut whole_state = state0.clone();
    let mut whole_out = vec![0.0_f32; n_tok * v_width];
    let (s, o) = (&mut whole_state, &mut whole_out);
    delta_net_chunked(&qkv, &log_decay, &beta, s, o, shape, n_tok, EPS, 4);

    let mut part_state = state0;
    let mut part_out = vec![0.0_f32; n_tok * v_width];
    let (head, tail) = part_out.split_at_mut(split * v_width);
    delta_net_chunked(
        &qkv[..split * qw],
        &log_decay[..split * vh],
        &beta[..split * vh],
        &mut part_state,
        head,
        shape,
        split,
        EPS,
        4,
    );
    delta_net_chunked(
        &qkv[split * qw..],
        &log_decay[split * vh..],
        &beta[split * vh..],
        &mut part_state,
        tail,
        shape,
        n_tok - split,
        EPS,
        4,
    );

    let worst = |a: &[f32], b: &[f32]| -> f32 {
        let scale = a.iter().fold(0.0_f32, |m, v| m.max(v.abs())).max(1e-6);
        a.iter()
            .zip(b)
            .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
            / scale
    };
    let (out, state) = (
        worst(&whole_out, &part_out),
        worst(&whole_state, &part_state),
    );
    println!("split 5+7 vs 12: out {out:.3e} state {state:.3e}");
    assert!(out < BOUND && state < BOUND, "out {out:e} state {state:e}");
}
