//! The four kernels of a routed feed-forward, each checked on its own.
//!
//! WHY SEPARATELY. A routed layer is a gate, a counting sort, a matmul whose weight is chosen
//! per row, and a weighted sum. Run together, a wrong permutation and a wrong address produce
//! the same symptom -- plausible numbers -- and neither one is visible. Each kernel here is
//! checked against something already trusted:
//!
//! ```text
//!   imparo_moe_gate      against softmax / logistic in f32 on the host
//!   imparo_moe_plan      against the same counting sort written out in Rust
//!   imparo_moe_grouped   against THE ONE-ROW GEMV of the expert's own slice -- the kernel
//!                        the decode path already gates, pointed at the same weights
//!   imparo_moe_combine   against the weighted sum in f32 on the host
//! ```
//!
//! The weights are the brick fixture's Q4_K blocks (`crates/imparo-cpu/tests/data/quant_rows.bin`)
//! placed by a hash so every row differs, ROW-MAJOR: an expert stack is 3-D and the tile-major
//! transform refuses a 3-D tensor by name, so the routed matmul reads the row-major layout and
//! this fixture is laid out the way the loader would leave it.
#![cfg(target_os = "macos")]

use imparo_backend::{Backend as _, BufId};
use imparo_metal::MetalBackend;

const FIXTURE: &[u8] = include_bytes!("../../imparo-cpu/tests/data/quant_rows.bin");
const PAGE: usize = 16384;
const ALIGN: usize = 256;
/// Q4_K: ggml type 12, and the wire kind of its row-major form.
const Q4_K_GGML: u32 = 12;
const Q4_K_KIND: u32 = 5;

/// The shape under test. `n_in` is a multiple of the 256-element super-block, `n_out` is the
/// expert's hidden width, and the token count is not a multiple of the eight work rows one
/// pass of the grouped matmul holds -- so the guarded tail runs.
const N_IN: usize = 512;
const N_OUT: usize = 64;
const EXPERTS: usize = 8;
const USED: usize = 3;
const TOKENS: usize = 11;

const SCORES: BufId = BufId::Model1;
const PROBS: BufId = BufId::Model2;
const SEL: BufId = BufId::Model3;
const TOPK: BufId = BufId::Model4;
const PERM: BufId = BufId::Model5;
const WGT: BufId = BufId::Model6;
const SEG: BufId = BufId::Model7;
const INV: BufId = BufId::Model8;

fn u32_at(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize
}

/// Every fixture block of `ggml`, concatenated.
fn fixture_blocks(ggml: u32) -> Vec<u8> {
    assert_eq!(&FIXTURE[0..4], b"IQFX", "fixture magic");
    let cases = u32_at(FIXTURE, 4);
    let mut off = 8;
    let mut out = Vec::new();
    for _ in 0..cases {
        let (kind, n_elems, n_bytes) = (
            u32_at(FIXTURE, off),
            u32_at(FIXTURE, off + 4),
            u32_at(FIXTURE, off + 8),
        );
        off += 12;
        if kind as u32 == ggml {
            out.extend_from_slice(&FIXTURE[off..off + n_bytes]);
        }
        off += n_bytes + n_elems * 4;
    }
    assert!(
        !out.is_empty(),
        "the fixture has no case of ggml type {ggml}"
    );
    out
}

/// A deterministic hash of three indices, for data that is not exactly representable.
fn mix(a: usize, b: usize, c: usize) -> u32 {
    let mut h = (a as u32).wrapping_mul(0x9E37_79B1)
        ^ (b as u32).wrapping_mul(0x85EB_CA77)
        ^ (c as u32).wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^ (h >> 12)
}

fn unit(a: usize, b: usize, c: usize) -> f32 {
    (mix(a, b, c) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// The stacked expert weight, row-major: `EXPERTS * N_OUT` rows of `N_IN`, expert `e` owning
/// rows `e * N_OUT ..`, which is exactly what `expert_stride` steps over.
fn stacked() -> Vec<u8> {
    let layout = imparo_gguf::tensor_layout(Q4_K_GGML).expect("Q4_K has a layout");
    let (be_, bb) = (layout.block_elements as usize, layout.block_bytes as usize);
    let pool = fixture_blocks(Q4_K_GGML);
    let n_pool = pool.len() / bb;
    let blocks = N_IN / be_;
    let rows = EXPERTS * N_OUT;
    let mut out = vec![0_u8; rows * blocks * bb];
    for r in 0..rows {
        for b in 0..blocks {
            let pick = mix(r, b, 7) as usize % n_pool;
            out[(r * blocks + b) * bb..][..bb]
                .copy_from_slice(&pool[pick * bb..][..bb]);
        }
    }
    out
}

fn expert_stride() -> u64 {
    let layout = imparo_gguf::tensor_layout(Q4_K_GGML).expect("Q4_K has a layout");
    (N_OUT * (N_IN / layout.block_elements as usize) * layout.block_bytes as usize)
        as u64
}

/// The host's own top-k over a row: larger first, the smaller index first among equals --
/// `top_k_rows`' rule.
fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

#[test]
#[ignore = "requires a physical Metal device; run explicitly with --ignored"]
fn the_routed_feed_forward_kernels_each_agree_with_what_they_replace() {
    // WHICH ROUTED MATMUL THIS RUN PINS. Two kernels serve `moe_grouped`, and they do not
    // agree to the same place, so the bound is not one number:
    //
    //   matrix unit (SHIPS, IMPARO_MOE_MMA_MIN unset)  an 8x8 fragment order of its own
    //   scalar      (IMPARO_MOE_MMA_MIN large)         the one-row GEMV's own sub-block order
    //
    // The scalar kernel sums each sub-block exactly as the one-row GEMV does, so it agrees
    // to 8.151e-7 and is held to 2e-5. The matrix unit accumulates through simdgroup
    // fragments -- the same property the dense rows matmul has, where a row is explicitly
    // NOT the one-row kernel's order -- and lands at 5.1e-5, so it is held to 2e-4.
    //
    // The environment is read ONCE per process by the backend, so one run pins one kernel;
    // the arm is printed below and the gate runs both.
    let mma = std::env::var("IMPARO_MOE_MMA_MIN")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .is_none_or(|v| v <= TOKENS as u32);
    let (arm, grouped_tol) = if mma {
        ("matrix unit", 2e-4_f32)
    } else {
        ("scalar", 2e-5_f32)
    };
    // ---- the mapping: the expert stack, then the selection bias ------------------------
    let w = stacked();
    let bias: Vec<f32> = (0..EXPERTS).map(|e| unit(e, 3, 9) * 0.25).collect();
    // NOT AT ZERO: the backend reads offset 0 as "this weight is absent", as every other
    // entry in that file does, and a real tensor never starts there -- the GGUF header is
    // before it. The stack starts one alignment in, the way the loader would leave it.
    let w_off = ALIGN;
    let bias_off = (w_off + w.len()).div_ceil(ALIGN) * ALIGN;
    let len = (bias_off + bias.len() * 4).div_ceil(PAGE) * PAGE;
    let layout = std::alloc::Layout::from_size_align(len, PAGE).unwrap();
    // Leaked on purpose: Metal wraps the mapping without copying and it must outlive the run.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    let bytes = unsafe { std::slice::from_raw_parts_mut(base, len) };
    bytes[w_off..w_off + w.len()].copy_from_slice(&w);
    for (i, v) in bias.iter().enumerate() {
        bytes[bias_off + i * 4..][..4].copy_from_slice(&v.to_le_bytes());
    }
    imparo_metal::set_weight_kind_types(&imparo_gguf::weights::wire_kind_types());
    unsafe { imparo_metal::init(base.cast_const(), len as u64) }.unwrap();

    let be = MetalBackend;
    let f = |n: usize| (n * 4) as u64;
    be.alloc(BufId::X, f(TOKENS * N_IN)).unwrap();
    be.alloc(BufId::O, f(TOKENS * USED * N_OUT)).unwrap();
    be.alloc(BufId::G, f(TOKENS * N_IN)).unwrap();
    be.alloc(SCORES, f(TOKENS * EXPERTS)).unwrap();
    be.alloc(PROBS, f(TOKENS * EXPERTS)).unwrap();
    be.alloc(SEL, f(TOKENS * EXPERTS)).unwrap();
    be.alloc(
        TOPK,
        f(be.top_k_rows_len(EXPERTS as u32, TOKENS as u32, USED as u32) as usize),
    )
    .unwrap();
    be.alloc(PERM, f(TOKENS * USED)).unwrap();
    be.alloc(WGT, f(TOKENS * USED)).unwrap();
    // n_expert + 1 segment offsets, then the active count, then up to n_expert active ids:
    // moe_plan writes that list into seg's tail so the routed grid can span ACTIVE experts,
    // and the host refuses a seg that cannot hold it.
    be.alloc(SEG, f(2 * EXPERTS + 2)).unwrap();
    be.alloc(INV, f(TOKENS * USED)).unwrap();

    let x: Vec<f32> = (0..TOKENS)
        .flat_map(|t| (0..N_IN).map(move |i| unit(t, i, 2)))
        .collect();
    be.write(BufId::X, 0, &x);
    let scores: Vec<f32> = (0..TOKENS)
        .flat_map(|t| (0..EXPERTS).map(move |e| unit(t, e, 5) * 4.0))
        .collect();
    be.write(SCORES, 0, &scores);

    let mut failures: Vec<String> = Vec::new();

    // ---- 1. the gate --------------------------------------------------------------------
    for (gating, name) in [
        (imparo_backend::ExpertGating::Softmax, "softmax"),
        (imparo_backend::ExpertGating::Sigmoid, "sigmoid"),
    ] {
        for with_bias in [false, true] {
            be.begin();
            assert!(
                be.moe_gate(
                    SCORES,
                    PROBS,
                    SEL,
                    if with_bias { bias_off as u64 } else { u64::MAX },
                    TOKENS as u32,
                    EXPERTS as u32,
                    gating,
                ),
                "moe_gate refused ({name}, bias {with_bias})"
            );
            be.end().unwrap();
            let mut probs = vec![0.0_f32; TOKENS * EXPERTS];
            let mut sel = vec![0.0_f32; TOKENS * EXPERTS];
            be.read(PROBS, 0, &mut probs);
            be.read(SEL, 0, &mut sel);
            for t in 0..TOKENS {
                let row = &scores[t * EXPERTS..][..EXPERTS];
                let want: Vec<f32> = match gating {
                    imparo_backend::ExpertGating::Softmax => {
                        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let ex: Vec<f32> = row.iter().map(|s| (s - m).exp()).collect();
                        let sum: f32 = ex.iter().sum();
                        ex.iter().map(|v| v / sum).collect()
                    }
                    imparo_backend::ExpertGating::Sigmoid => {
                        row.iter().map(|s| 1.0 / (1.0 + (-s).exp())).collect()
                    }
                };
                for e in 0..EXPERTS {
                    let got = probs[t * EXPERTS + e];
                    if (got - want[e]).abs() > 2e-6 {
                        failures.push(format!(
                            "gate {name} bias={with_bias}: token {t} expert {e} probability \
                             {got} against {}",
                            want[e]
                        ));
                    }
                    let want_sel = want[e] + if with_bias { bias[e] } else { 0.0 };
                    let got_sel = sel[t * EXPERTS + e];
                    if (got_sel - want_sel).abs() > 2e-6 {
                        failures.push(format!(
                            "gate {name} bias={with_bias}: token {t} expert {e} selection \
                             score {got_sel} against {want_sel}"
                        ));
                    }
                }
            }
        }
    }

    // The rest of the run uses the sigmoid gate with the bias, which is LFM2.5-8B-A1B's.
    be.begin();
    assert!(be.moe_gate(
        SCORES,
        PROBS,
        SEL,
        bias_off as u64,
        TOKENS as u32,
        EXPERTS as u32,
        imparo_backend::ExpertGating::Sigmoid,
    ));
    assert!(
        be.top_k_rows(SEL, TOPK, EXPERTS as u32, TOKENS as u32, USED as u32),
        "top_k_rows refused"
    );
    be.end().unwrap();
    let mut probs = vec![0.0_f32; TOKENS * EXPERTS];
    let mut sel = vec![0.0_f32; TOKENS * EXPERTS];
    be.read(PROBS, 0, &mut probs);
    be.read(SEL, 0, &mut sel);
    let mut picked_bits = [0.0_f32; TOKENS * USED];
    be.read(TOPK, 0, &mut picked_bits);
    let picked: Vec<usize> = picked_bits.iter().map(|v| v.to_bits() as usize).collect();
    for t in 0..TOKENS {
        let want = top_k(&sel[t * EXPERTS..][..EXPERTS], USED);
        let got = &picked[t * USED..][..USED];
        if got != want.as_slice() {
            failures.push(format!("pick: token {t} chose {got:?} against {want:?}"));
        }
    }

    // ---- 2. the plan --------------------------------------------------------------------
    const SCALE: f32 = 1.5;
    be.begin();
    assert!(
        be.moe_plan(
            TOPK,
            PROBS,
            PERM,
            WGT,
            SEG,
            INV,
            TOKENS as u32,
            EXPERTS as u32,
            USED as u32,
            true,
            SCALE,
        ),
        "moe_plan refused"
    );
    be.end().unwrap();
    let rows = TOKENS * USED;
    let mut perm_bits = vec![0.0_f32; rows];
    let mut wgt = vec![0.0_f32; rows];
    let mut seg_bits = vec![0.0_f32; EXPERTS + 1];
    let mut inv_bits = vec![0.0_f32; rows];
    be.read(PERM, 0, &mut perm_bits);
    be.read(WGT, 0, &mut wgt);
    be.read(SEG, 0, &mut seg_bits);
    be.read(INV, 0, &mut inv_bits);
    let perm: Vec<usize> = perm_bits.iter().map(|v| v.to_bits() as usize).collect();
    let seg: Vec<usize> = seg_bits.iter().map(|v| v.to_bits() as usize).collect();
    let inv: Vec<usize> = inv_bits.iter().map(|v| v.to_bits() as usize).collect();

    // The same counting sort, written out. The kernel's positions come from threadgroup
    // atomics, so WITHIN an expert the order of its tokens is not promised -- what is
    // promised is that each expert's segment holds exactly its tokens, once each, and that
    // `inv` points every (token, slot) at the row carrying that token.
    let mut want_counts = [0_usize; EXPERTS];
    for t in 0..TOKENS {
        for j in 0..USED {
            want_counts[picked[t * USED + j]] += 1;
        }
    }
    let mut run = 0;
    for e in 0..EXPERTS {
        if seg[e] != run {
            failures.push(format!(
                "plan: expert {e} starts at {} against {run}",
                seg[e]
            ));
        }
        run += want_counts[e];
    }
    if seg[EXPERTS] != run {
        failures.push(format!(
            "plan: the last bound is {} against {run}",
            seg[EXPERTS]
        ));
    }
    let mut seen = vec![0_usize; rows];
    for t in 0..TOKENS {
        // The picked weights, renormalised over the picks, then scaled.
        let sum: f32 = (0..USED)
            .map(|j| probs[t * EXPERTS + picked[t * USED + j]])
            .sum();
        for j in 0..USED {
            let e = picked[t * USED + j];
            let row = inv[t * USED + j];
            if row >= rows {
                failures.push(format!("plan: token {t} slot {j} points at row {row}"));
                continue;
            }
            seen[row] += 1;
            if row < seg[e] || row >= seg[e + 1] {
                failures.push(format!(
                    "plan: token {t} slot {j} (expert {e}) landed at row {row}, outside \
                     {}..{}",
                    seg[e],
                    seg[e + 1]
                ));
            }
            if perm[row] != t {
                failures.push(format!(
                    "plan: row {row} says token {} against {t}",
                    perm[row]
                ));
            }
            let want = probs[t * EXPERTS + e] / sum * SCALE;
            if (wgt[row] - want).abs() > 2e-6 {
                failures.push(format!(
                    "plan: row {row} weight {} against {want}",
                    wgt[row]
                ));
            }
        }
    }
    if seen.iter().any(|&n| n != 1) {
        failures.push(format!(
            "plan: rows claimed {seen:?}, each must be claimed once"
        ));
    }

    // ---- 3. the routed matmul -----------------------------------------------------------
    // Its work rows, against the one-row GEMV of the same expert's slice on the same token.
    be.begin();
    assert!(
        be.moe_grouped(
            Q4_K_KIND,
            w_off as u64,
            expert_stride(),
            BufId::X,
            BufId::O,
            PERM,
            SEG,
            N_IN as u32,
            N_OUT as u32,
            EXPERTS as u32,
            TOKENS as u32,
            rows as u32,
            false,
            false,
        ),
        "moe_grouped refused"
    );
    be.end().unwrap();
    let mut got = vec![0.0_f32; rows * N_OUT];
    be.read(BufId::O, 0, &mut got);
    let mut worst = 0.0_f32;
    for e in 0..EXPERTS {
        for row in seg[e]..seg[e + 1] {
            let t = perm[row];
            // The trusted route: one row, this expert's own offset, reading token t.
            be.begin();
            imparo_metal::matmat_from(
                Q4_K_KIND,
                w_off as u64 + e as u64 * expert_stride(),
                N_IN as u32,
                N_OUT as u32,
                BufId::X as u32,
                BufId::G as u32,
                1,
                t as u32,
            );
            be.end().unwrap();
            let mut want = vec![0.0_f32; N_OUT];
            be.read(BufId::G, 0, &mut want);
            for i in 0..N_OUT {
                let d = (got[row * N_OUT + i] - want[i]).abs();
                let scale = want[i].abs().max(1.0);
                worst = worst.max(d / scale);
                if d / scale > grouped_tol {
                    failures.push(format!(
                        "grouped: row {row} (token {t}, expert {e}) output {i} is {} against \
                         the one-row GEMV's {}",
                        got[row * N_OUT + i],
                        want[i]
                    ));
                }
            }
        }
    }

    // ---- 4. the combine -----------------------------------------------------------------
    be.alloc(BufId::Cur, f(TOKENS * N_OUT)).unwrap();
    be.begin();
    assert!(
        be.moe_combine(
            BufId::O,
            WGT,
            INV,
            BufId::Cur,
            N_OUT as u32,
            USED as u32,
            TOKENS as u32,
        ),
        "moe_combine refused"
    );
    be.end().unwrap();
    let mut combined = vec![0.0_f32; TOKENS * N_OUT];
    be.read(BufId::Cur, 0, &mut combined);
    let mut worst_combine = 0.0_f32;
    for t in 0..TOKENS {
        for i in 0..N_OUT {
            // Slot order, the kernel's own.
            let mut want = 0.0_f32;
            let mut magnitude = 0.0_f32;
            for j in 0..USED {
                let row = inv[t * USED + j];
                want += wgt[row] * got[row * N_OUT + i];
                magnitude += (wgt[row] * got[row * N_OUT + i]).abs();
            }
            // AGAINST THE TERMS, NOT THE RESULT. The terms cancel -- three values of
            // 1e4 can land at 20 -- so a rounding of the sum is a rounding of the
            // LARGEST term, and measuring it against the small result reads as a huge
            // relative error for what is one bit of one multiply. The kernel also
            // contracts its multiply-add into an fma where the host cannot.
            let d = (combined[t * N_OUT + i] - want).abs();
            worst_combine = worst_combine.max(d / magnitude.max(1.0));
            if d / magnitude.max(1.0) > 1e-6 {
                failures.push(format!(
                    "combine: token {t} output {i} is {} against {want} (terms {magnitude})",
                    combined[t * N_OUT + i]
                ));
            }
        }
    }

    // ---- the same run twice, bit for bit ------------------------------------------------
    let once: Vec<u32> = combined.iter().map(|v| v.to_bits()).collect();
    for _ in 0..2 {
        be.begin();
        assert!(be.moe_grouped(
            Q4_K_KIND,
            w_off as u64,
            expert_stride(),
            BufId::X,
            BufId::O,
            PERM,
            SEG,
            N_IN as u32,
            N_OUT as u32,
            EXPERTS as u32,
            TOKENS as u32,
            rows as u32,
            false,
            false,
        ));
        assert!(be.moe_combine(
            BufId::O,
            WGT,
            INV,
            BufId::Cur,
            N_OUT as u32,
            USED as u32,
            TOKENS as u32,
        ));
        be.end().unwrap();
        let mut again = vec![0.0_f32; TOKENS * N_OUT];
        be.read(BufId::Cur, 0, &mut again);
        let repeat_bits: Vec<u32> = again.iter().map(|v| v.to_bits()).collect();
        if repeat_bits != once {
            let i = repeat_bits
                .iter()
                .zip(&once)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            failures.push(format!(
                "determinism: element {i} came back {:08x} against {:08x}",
                repeat_bits[i], once[i]
            ));
        }
    }

    eprintln!(
        "moe: {TOKENS} tokens, {EXPERTS} experts, {USED} used, {rows} work rows; the \
         {arm} routed matmul, held to {grouped_tol:.0e}; worst relative difference from \
         the one-row GEMV {worst:.3e}, from the combine's own terms {worst_combine:.3e}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
