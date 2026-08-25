//! Isolates the GPU matmul: how many GB/s does the kernel actually sustain?
//!
//! Six tuning axes were flat at ~63 GB/s end to end. This answers whether the kernel is the
//! limit or whether the surrounding per-forward work is, which decides where to optimise.

// The METAL device benchmark (category-2 device profile: peak TFLOPS, bandwidth, the
// per-shape GEMV geometry) -- named for what it is. Metal-only, whole-file macOS gate.
// A CUDA device bench appears later as a parallel imparo-cudabench; this stays Metal.
// NOT a whole-file #![cfg]: a bin whose contents are all cfg'd out has no `main` and
// fails to COMPILE on other platforms, killing their workspace builds.
#![cfg_attr(not(target_os = "macos"), allow(unused))]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("imparo-metalbench profiles the Metal device and only runs on macOS");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod bench {

    use std::path::PathBuf;

    use imparo_backend::BufId;
    use imparo_metal as m;
    use imparo_model::weights::Weights;

    /// Largest absolute difference between two equal-length slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len(), "compared slices differ in length");
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from((x - y).abs()))
            .fold(0.0, f64::max)
    }

    pub fn main() -> Result<(), Box<dyn std::error::Error>> {
        let path = PathBuf::from(
            std::env::args()
                .nth(1)
                .ok_or("usage: imparo-metalbench MODEL.gguf")?,
        );
        let _iters: usize = std::env::args()
            .nth(2)
            .and_then(|v| v.parse().ok())
            .unwrap_or(200);
        unsafe { std::env::set_var("IMPARO_GPU", "1") };
        // BEFORE enable_gpu, which is where the pipelines are built: only the selected
        // Q8 shapes exist otherwise, and the check has to cover every shape the tuner
        // can pick. Setting it afterwards would leave nine of the ten nil.
        if std::env::var("IMPARO_Q8_CHECK").is_ok() {
            m::set_q8_all(1);
        }
        let mut w = Weights::open(&path)?;
        // The plan is what says which activation the epilogue kernels are specialised
        // with, so the bench profiles the pipelines this model would actually run.
        let plan = imparo_model::build_plan(&imparo_gguf::read(&path)?, &path)?;
        imparo_model::backend::enable_gpu(&mut w, &plan)?;
        if !w.gpu_enabled() {
            return Err("metal not enabled".into());
        }

        // ---- SHORTCONV CORRECTNESS, GPU AGAINST THE CPU REFERENCE ---------------------
        //
        // IMPARO_SHORTCONV_CHECK=1 runs the kernel against `imparo_cpu::ops::shortconv`
        // on the model's OWN convolution weights, at the token counts that reach each
        // case. Before any timing, and before any workflow uses it: a kernel that has not
        // been shown to compute the right thing is not something to build a forward on.
        //
        //   n_tok = 1     decode. n_tok < history, so the new state is a SHIFT of the old
        //                 one -- the case where a thread that wrote before reading would
        //                 clobber a slot it still needs.
        //   n_tok = 2     n_tok == history exactly.
        //   n_tok = 5     the ordinary case, new state entirely from this batch.
        //
        // Each runs TWICE in sequence on the same state, because the second call is the
        // one that proves the state carried: a kernel that ignored the state entirely
        // would pass a single-call check at n_tok >= history.
        if std::env::var("IMPARO_SHORTCONV_CHECK").is_ok() {
            let be = imparo_model::backend::active().ok_or("no backend")?;
            let name = "blk.0.shortconv.conv.weight";
            let Some(cw_t) = w.get(name).copied() else {
                return Err(format!("{name} absent: this check needs an LFM2 file").into());
            };
            let cw = w.f32s(&cw_t).to_vec();
            let width = cw_t.ne1();
            let kern = cw_t.ne0();
            let history = kern - 1;
            println!("shortconv check: width={width} kernel={kern} from {name}");

            let mut worst = 0.0_f64;
            let mut fails = 0usize;
            for &n_tok in &[1usize, 2, 5] {
                // Deterministic inputs, and NOT symmetric across the three chunks: b, c
                // and x must be distinguishable or swapping two of them would pass.
                let bcx: Vec<f32> = (0..n_tok * 3 * width)
                    .map(|i| ((i % 37) as f32 - 18.0) / 23.0)
                    .collect();
                let seed: Vec<f32> = (0..history * width)
                    .map(|i| ((i % 11) as f32 - 5.0) / 17.0)
                    .collect();
                let mut host_state = seed.clone();
                let mut dev_state = seed;
                be.alloc(BufId::Model0, (bcx.len() * 4) as u64)
                    .map_err(|rc| format!("alloc bcx rc={rc}"))?;
                be.alloc(BufId::Recur, (dev_state.len() * 4) as u64)
                    .map_err(|rc| format!("alloc state rc={rc}"))?;
                be.alloc(BufId::O, (n_tok * width * 4) as u64)
                    .map_err(|rc| format!("alloc out rc={rc}"))?;
                be.write(BufId::Recur, 0, &dev_state);

                let mut host_out = vec![0.0_f32; n_tok * width];
                let mut dev_out = vec![0.0_f32; n_tok * width];
                for call in 0..2 {
                    imparo_model::ops::shortconv(
                        &bcx, &cw, &mut host_state, &mut host_out, width, kern, n_tok,
                    );
                    be.begin();
                    be.write(BufId::Model0, 0, &bcx);
                    be.shortconv(
                        BufId::Model0,
                        cw_t.offset as u64,
                        BufId::Recur,
                        0,
                        BufId::O,
                        width as u32,
                        kern as u32,
                        n_tok as u32,
                    );
                    // `end`, NOT `flush`. flush commits and returns so the CPU can keep
                    // encoding; only end waits. Reading after flush gave an output of
                    // exactly zeros and a state exactly as written -- which reads as "the
                    // kernel does nothing" and is really "the harness read too early".
                    be.end().map_err(|rc| format!("shortconv end rc={rc}"))?;
                    be.read(BufId::O, 0, &mut dev_out);
                    be.read(BufId::Recur, 0, &mut dev_state);

                    let d_out = max_abs_diff(&host_out, &dev_out);
                    let d_state = max_abs_diff(&host_state, &dev_state);
                    let d = d_out.max(d_state);
                    worst = worst.max(d);
                    // 1e-5: both sides sum `kernel` products in the same order, so the
                    // only spread is fused-multiply-add on one side and not the other.
                    let ok = d < 1e-5;
                    if !ok {
                        fails += 1;
                    }
                    println!(
                        "  n_tok={n_tok} call={call} out={d_out:.3e} state={d_state:.3e} {}",
                        if ok { "OK" } else { "FAIL" }
                    );
                    if !ok && std::env::var("IMPARO_SHORTCONV_DUMP").is_ok() {
                        println!("    out host {:?}", &host_out[..4.min(host_out.len())]);
                        println!("    out dev  {:?}", &dev_out[..4.min(dev_out.len())]);
                        println!("    st  host {:?}", &host_state[..4.min(host_state.len())]);
                        println!("    st  dev  {:?}", &dev_state[..4.min(dev_state.len())]);
                        println!("    cw       {:?}", &cw[..6.min(cw.len())]);
                        println!("    bcx      {:?}", &bcx[..4.min(bcx.len())]);
                    }
                }
            }
            // ---- THE BOUNDARY SNAPSHOT ------------------------------------------
            //
            // The reference is not a second kernel, it is the DEFINITION: the state at
            // position k is what the CPU reference leaves after running the first k
            // tokens from the same seed. The snapshot must produce exactly that from a
            // batch that runs past k and never stops there.
            //
            //   k=1  k < history: part of the answer is still the PRE-batch state, so
            //        this is the case that fails if the advance is dispatched first.
            //   k=2  k == history exactly.
            //   k=5  k == the whole batch: the same value the advance leaves.
            {
                let n_tok = 5usize;
                let bcx: Vec<f32> = (0..n_tok * 3 * width)
                    .map(|i| ((i % 37) as f32 - 18.0) / 23.0)
                    .collect();
                let seed: Vec<f32> = (0..history * width)
                    .map(|i| ((i % 11) as f32 - 5.0) / 17.0)
                    .collect();
                be.alloc(BufId::Model0, (bcx.len() * 4) as u64)
                    .map_err(|rc| format!("alloc bcx rc={rc}"))?;
                be.alloc(BufId::Recur, (seed.len() * 4) as u64)
                    .map_err(|rc| format!("alloc state rc={rc}"))?;
                be.alloc(BufId::RecurSnap, (seed.len() * 4) as u64)
                    .map_err(|rc| format!("alloc snapshot rc={rc}"))?;
                be.alloc(BufId::O, (n_tok * width * 4) as u64)
                    .map_err(|rc| format!("alloc out rc={rc}"))?;
                for k in [1usize, 2, 5] {
                    let mut ref_state = seed.clone();
                    let mut ref_out = vec![0.0_f32; k * width];
                    imparo_model::ops::shortconv(
                        &bcx[..k * 3 * width], &cw, &mut ref_state, &mut ref_out,
                        width, kern, k,
                    );
                    be.write(BufId::Recur, 0, &seed);
                    be.begin();
                    be.write(BufId::Model0, 0, &bcx);
                    be.shortconv_snapshot(
                        BufId::Model0, BufId::Recur, 0, BufId::RecurSnap, 0,
                        width as u32, kern as u32, k as u32,
                    );
                    be.shortconv(
                        BufId::Model0, cw_t.offset as u64, BufId::Recur, 0, BufId::O,
                        width as u32, kern as u32, n_tok as u32,
                    );
                    be.end().map_err(|rc| format!("snapshot end rc={rc}"))?;
                    let mut snap = vec![0.0_f32; seed.len()];
                    be.read(BufId::RecurSnap, 0, &mut snap);
                    let d = max_abs_diff(&ref_state, &snap);
                    worst = worst.max(d);
                    let ok = d < 1e-5;
                    if !ok {
                        fails += 1;
                    }
                    println!(
                        "  snapshot n_tok={n_tok} at k={k} state={d:.3e} {}",
                        if ok { "OK" } else { "FAIL" }
                    );
                }
            }
            println!(
                "shortconv check: {} worst={worst:.3e}",
                if fails == 0 { "ALL OK" } else { "FAILURES" }
            );
            if fails > 0 {
                return Err("shortconv disagrees with the CPU reference".into());
            }
        }

        // ---- Q8_0 CORRECTNESS, GPU AGAINST THE CPU REFERENCE --------------------------
        //
        // IMPARO_Q8_CHECK=1 compares every Q8_0 route against imparo-cpu's mul_mat_batch
        // on the model's own tensors. This runs BEFORE any timing: a Q8 number measured
        // from a kernel that has not been shown to compute the right thing means nothing.
        //
        // The token counts are chosen to reach each route and each edge:
        //   1    the decode GEMV
        //   4    the narrow token tile (<= q8_gemv_max_tok, default 8)
        //   32   the wide GEMM, exact token grid
        //   33   the wide GEMM with a partial token tile (the masked write-back)
        //   64   two full token tiles
        if std::env::var("IMPARO_Q8_CHECK").is_ok() {
            // gemma4's names are here too, so the SAME check runs on a Q4_0 model as a
            // control: the prefill GEMM stages both operands as half on either quant, so
            // a Q8-only measurement cannot tell staging error from a Q8 defect.
            let names = [
                "blk.0.shortconv.out_proj.weight",
                "blk.0.ffn_gate.weight",
                "blk.0.ffn_down.weight",
                "blk.0.attn_output.weight",
            ];
            let mut checked = 0usize;
            let mut fails = 0usize;
            let mut worst = 0.0_f64;
            for name in names {
                let Some(t) = w.get(name).copied() else { continue };
                let Some(kind) = imparo_gguf::weights::weight_kind(t.ggml_type) else {
                    println!("SKIP {name}: ggml type {} has no kernel", t.ggml_type);
                    continue;
                };
                let wire = kind as u32;
                if wire == 0 {
                    continue; // F32 has no quant staging to check
                }
                let n_in = t.ne0() as u32;
                let n_out = t.ne1() as u32;
                // EVERY SELECTABLE PIPELINE, not just the default. Q8_DECODE_ROWS and
                // Q8_TOKEN_TILE are function constants, so each value is a separately
                // compiled kernel; a shape the tuner can pick and nobody checked is a
                // wrong answer waiting for a config change.
                let combos: Vec<(u32, u32, u32, u32, u32)> = if wire == 2 {
                    let mut v = Vec::new();
                    for rows in [1u32, 2, 4] {
                        for sgs in [1u32, 4, 32] {
                            v.push((1u32, rows, sgs, 0, 0));
                        }
                    }
                    for tile in [1u32, 2, 4, 8] {
                        for sgs in [1u32, 8] {
                            v.push((4u32, tile, sgs, 0, 0));
                        }
                    }
                    for shape in 0..m::q8_gemm_shapes() {
                        for n_tok in [32u32, 33, 64] {
                            v.push((n_tok, 0, 0, shape, 1));
                        }
                    }
                    v
                } else {
                    vec![(1, 0, 0, 0, 0), (4, 0, 0, 0, 0), (32, 0, 0, 0, 0),
                         (33, 0, 0, 0, 0), (64, 0, 0, 0, 0)]
                };
                for (n_tok, cfa, cfb, shape, is_gemm) in combos {
                    if wire == 2 {
                        if is_gemm == 1 {
                            m::set_q8_gemm_shape(shape);
                            m::set_q8_gemm_large_shape(shape);
                        } else if n_tok == 1 {
                            m::set_q8_decode_rows(cfa);
                            m::set_q8_decode_sgs(cfb);
                        } else {
                            m::set_q8_token_tile(cfa);
                            m::set_q8_batch_sgs(cfb);
                        }
                    }
                    // Deterministic, spread over a couple of octaves so a dropped block
                    // or a wrong scale cannot cancel.
                    let x: Vec<f32> = (0..(n_in as usize) * (n_tok as usize))
                        .map(|i| {
                            let k = (i % 97) as f32;
                            (k - 48.0) / 64.0
                        })
                        .collect();
                    let mut want = vec![0.0f32; (n_out as usize) * (n_tok as usize)];
                    imparo_cpu::ops::mul_mat_batch(&w, &t, &x, n_tok as usize, &mut want);

                    m::alloc(m::buf::X, x.len() as u64 * 4).map_err(|e| format!("alloc x {e}"))?;
                    m::alloc(m::buf::LOGITS, want.len() as u64 * 4)
                        .map_err(|e| format!("alloc y {e}"))?;
                    m::write(m::buf::X, 0, &x);
                    m::begin();
                    m::matmat(
                        wire,
                        t.offset as u64,
                        n_in,
                        n_out,
                        m::buf::X,
                        m::buf::LOGITS,
                        n_tok,
                    );
                    m::end().map_err(|e| format!("end {e}"))?;
                    let mut got = vec![0.0f32; want.len()];
                    m::read(m::buf::LOGITS, 0, &mut got);

                    // Relative to the row magnitude, not absolute: n_in = 10752 sums a
                    // lot of terms, so the accumulation order alone moves the last bits.
                    let mut num = 0.0_f64;
                    let mut den = 0.0_f64;
                    let mut max_abs = 0.0_f64;
                    for (a, b) in want.iter().zip(got.iter()) {
                        num += f64::from((a - b).abs());
                        den += f64::from(a.abs());
                        max_abs = max_abs.max(f64::from((a - b).abs()));
                    }
                    let rel = if den > 0.0 { num / den } else { num };
                    worst = worst.max(rel);
                    checked += 1;
                    let route = if wire != 2 {
                        if n_tok > 1 { "gemm" } else { "gemv" }
                    } else if n_tok == 1 {
                        "gemv"
                    } else if n_tok <= m::q8_gemv_max_tok() {
                        "tile"
                    } else {
                        "gemm"
                    };
                    let cfg = if wire != 2 {
                        String::new()
                    } else if route == "gemm" {
                        format!("shape={shape}")
                    } else if route == "gemv" {
                        format!("rows={} sgs={}", m::q8_decode_rows(), m::q8_decode_sgs())
                    } else {
                        format!("tile={} sgs={}", m::q8_token_tile(), m::q8_batch_sgs())
                    };
                    let bad = if route == "gemm" { rel >= 1e-3 } else { rel >= 1e-5 };
                    if bad {
                        fails += 1;
                    }
                    println!(
                        "Q{} {name:<34} {n_in:>6}->{n_out:<6} n_tok={n_tok:<3} {route:<4} \
                         {cfg:<18} rel={rel:.3e} max_abs={max_abs:.3e}{}",
                        if wire == 2 { 8 } else { 4 },
                        if bad { "  <-- FAIL" } else { "" }
                    );
                }
            }
            if checked == 0 {
                return Err("quant check found no quantised tensor in this model".into());
            }
            // TWO TOLERANCES, because the routes do not carry the same arithmetic.
            //
            //   gemv / narrow tile   f32 throughout            1e-5
            //   prefill GEMM         both operands staged f16   1e-3
            //
            // f16 carries 11 mantissa bits, so its relative epsilon is ~4.9e-4 and a sum
            // of thousands of such terms lands near 2e-4. That is the DESIGN -- it is
            // what llama.cpp's mul_mm does and what this engine's Q4_0 prefill already
            // did -- so a single tight gate would have called the staging a defect.
            println!(
                "ALL quant check {checked} comparisons, {fails} over tolerance, worst rel \
                 {worst:.3e} -> {}",
                if fails == 0 { "PASS" } else { "FAIL" }
            );
            return Ok(());
        }

        // Which lanes-per-row value each REAL decode shape wants. One global value is set
        // today; the probe suggests short-output matmuls want a different one from lm_head.
        if std::env::var("IMPARO_LANES").is_ok() {
            let mut shapes: Vec<(String, u64, u32, u32)> = Vec::new();
            for name in [
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ] {
                if let Some(t) = w.get(&format!("blk.0.{name}.weight")) {
                    shapes.push((
                        name.to_string(),
                        t.offset as u64,
                        t.ne0() as u32,
                        t.ne1() as u32,
                    ));
                }
            }
            if let Some(t) = w.get("token_embd.weight") {
                shapes.push((
                    "lm_head".into(),
                    t.offset as u64,
                    t.ne0() as u32,
                    t.ne1() as u32,
                ));
            }
            m::alloc(m::buf::CUR, 262_144 * 4).map_err(|e| format!("alloc {e}"))?;
            m::alloc(m::buf::LOGITS, 262_144 * 4).map_err(|e| format!("alloc {e}"))?;
            m::write(m::buf::CUR, 0, &vec![0.01_f32; 262_144]);
            for (name, off, n_in, n_out) in &shapes {
                let mib = f64::from(*n_in) * f64::from(*n_out) * 18.0
                    / 32.0
                    / (1u64 << 20) as f64;
                let mut row =
                    format!("{name:<12} {n_in:>6}->{n_out:<7} {mib:>6.1} MiB");
                for lanes in [4u32, 8, 16, 32] {
                    m::set_lanes(lanes);
                    let run = || {
                        m::begin();
                        for _ in 0..20 {
                            m::matmat(
                                1,
                                *off,
                                *n_in,
                                *n_out,
                                m::buf::CUR,
                                m::buf::LOGITS,
                                1,
                            );
                        }
                        let _ = m::end();
                    };
                    run(); // warm
                    let mut v = [0.0f64; 5];
                    for x in &mut v {
                        let t = std::time::Instant::now();
                        run();
                        *x = mib
                            / (1u32 << 10) as f64
                            / (t.elapsed().as_secs_f64() / 20.0);
                    }
                    v.sort_by(f64::total_cmp);
                    row.push_str(&format!("  l{lanes}={:>6.1}", v[2]));
                }
                println!("{row} GB/s");
            }
            return Ok(());
        }

        // What a pure streaming read sustains, with no dequantise and no math. Every
        // memory-bound kernel is judged against THIS, not against the 150 GB/s on the
        // spec sheet -- and not against an assertion that whatever we measured is the top.
        if std::env::var("IMPARO_BW").is_ok() {
            let bytes = 256u64 << 20;
            let mut best = (0.0_f64, 0u32, 0u32);
            for tpg in [256u32, 512, 1024] {
                for mul in [4u32, 8, 16, 32, 64] {
                    let tgs = 18 * mul;
                    let gbs = m::bw_read(bytes, 4, tgs, tpg);
                    println!("bw_read tpg={tpg:<5} tgs={tgs:<5} {gbs:>7.1} GB/s");
                    if gbs > best.0 {
                        best = (gbs, tgs, tpg);
                    }
                }
            }
            println!(
                "BW peak {:.1} GB/s at tgs={} tpg={}",
                best.0, best.1, best.2
            );

            // Same launch geometry as the decode GEMV, arithmetic removed.
            for (n_in, n_out, label) in [
                (2560u32, 2048u32, "qkv-ish  2560->2048"),
                (2560, 10240, "ffn_gate 2560->10240"),
                (10240, 2560, "ffn_down 10240->2560"),
                (2560, 262_144, "lm_head  2560->262144"),
            ] {
                for lanes in [8u32, 16, 32] {
                    let mut row = format!("{label:<22} lanes={lanes:<3}");
                    for split in [1u32, 2, 4, 8] {
                        // Median of three: single samples on this probe move by ~20%.
                        let mut v = [0.0f64; 3];
                        for x in &mut v {
                            *x = m::gemv_probe(n_in, n_out, lanes, 4, 0, 16, split);
                        }
                        v.sort_by(f64::total_cmp);
                        row.push_str(&format!("  s{split}={:>6.1}", v[1]));
                    }
                    println!("{row}");
                }
            }
            return Ok(());
        }

        // Cold-read bandwidth: cycle through DIFFERENT layers so nothing is served from
        // cache. Repeating one 14 MB matmul measured 131 GB/s, which was the system cache,
        // not DRAM -- in a real forward each layer's weights are touched exactly once.
        m::alloc(m::buf::CUR, 4 * 262_144 * 4).map_err(|e| format!("alloc {e}"))?;
        m::alloc(m::buf::LOGITS, 4 * 262_144 * 4).map_err(|e| format!("alloc {e}"))?;
        let x = vec![0.01_f32; 262_144];
        m::write(m::buf::CUR, 0, &x);

        for (label, suffix) in [
            ("ffn_gate 2560->10240", "ffn_gate"),
            ("ffn_down 10240->2560", "ffn_down"),
        ] {
            let mut tensors = Vec::new();
            for li in 0..42 {
                if let Some(t) = w.get(&format!("blk.{li}.{suffix}.weight")) {
                    tensors.push(*t);
                }
            }
            let bytes: f64 = tensors.iter().map(|t| t.bytes as f64).sum();
            // one pass over every layer, so 2.4 GB of distinct weights
            let t0 = std::time::Instant::now();
            m::begin();
            for t in &tensors {
                m::matmat(
                    1,
                    t.offset as u64,
                    t.ne0() as u32,
                    t.ne1() as u32,
                    m::buf::CUR,
                    m::buf::LOGITS,
                    1,
                );
            }
            m::end().map_err(|e| format!("end {e}"))?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "{label:<24} {:>7.1} MiB over {} layers  {ms:>7.2} ms  {:>6.1} GB/s COLD",
                bytes / (1 << 20) as f64,
                tensors.len(),
                bytes / (ms / 1e3) / 1e9
            );
        }

        // What the matrix units can actually do, before asking what the GEMM achieves.
        {
            println!("-- simdgroup matrix ceiling (operands in registers) --");
            println!(
                "  threadgroup memory vs throughput, 8 acc, 6 loads per 8 multiplies:"
            );
            for smem in [2048u32, 8192, 16896, 32768] {
                let t = m::mma_loaded_smem(280, 8, 4096, smem);
                println!("    {smem:5} bytes   {t:5.2} TFLOPS");
            }
            println!("  A operands from device at row stride:");
            // stride 8 is what a TILED activation layout would give: an 8x8 tile in one
            // contiguous 64-float run, so one transaction instead of eight.
            for stride in [8u32, 64, 2560, 10240] {
                let t = m::mma_device_a(280, 8, 4096, stride);
                println!("    stride {stride:5}   {t:5.2} TFLOPS");
            }
            for (tgs, sgs) in [(72u32, 8u32), (144, 8), (288, 4), (576, 2)] {
                let t = m::mma_peak(tgs, sgs, 4096);
                let l = m::mma_loaded(tgs, sgs, 4096);
                println!(
                    "  {tgs:4} threadgroups x {sgs} simdgroups   registers {t:5.2} \
                      TFLOPS   reloaded {l:5.2} TFLOPS"
                );
            }
        }

        // Prefill GEMM throughput per SHAPE.
        //
        // Prefill's aggregate came out at 4.79 TFLOPS against a ~6.45 peak, but that averages
        // wide FFN matmuls with narrow q/k/v ones. A narrow n_out launches few threadgroups and
        // may simply not fill the GPU, which is a different problem from a slow kernel -- and
        // the two want opposite fixes. Cycling layers keeps every read cold, as above.
        {
            let n_tok: u32 = std::env::var("IMPARO_BENCH_TOK")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(440);
            m::alloc(m::buf::X, (n_tok as u64) * 10_240 * 4)
                .map_err(|e| format!("alloc {e}"))?;
            m::alloc(m::buf::O, (n_tok as u64) * 10_240 * 4)
                .map_err(|e| format!("alloc {e}"))?;
            let xin = vec![0.01_f32; (n_tok as usize) * 10_240];
            m::write(m::buf::X, 0, &xin);
            // Routing is by batch width now (multi-token -> GEMM family, one token -> GEMV);
            // there is no prefill-kernel flag to set or clear.
            println!("-- prefill GEMM, n_tok={n_tok} --");
            for (label, suffix) in [
                ("ffn_gate  2560->10240", "ffn_gate"),
                ("ffn_down 10240->2560", "ffn_down"),
                ("attn_q    2560->2048", "attn_q"),
                ("attn_k    2560-> 512", "attn_k"),
                ("attn_output       ->", "attn_output"),
            ] {
                let mut tensors = Vec::new();
                for li in 0..42 {
                    if let Some(t) = w.get(&format!("blk.{li}.{suffix}.weight")) {
                        tensors.push(*t);
                    }
                }
                if tensors.is_empty() {
                    continue;
                }
                let flops: f64 = tensors
                    .iter()
                    .map(|t| {
                        2.0 * f64::from(t.ne0() as u32)
                            * f64::from(t.ne1() as u32)
                            * f64::from(n_tok)
                    })
                    .sum();
                let t0 = std::time::Instant::now();
                m::begin();
                for t in &tensors {
                    m::matmat(
                        1,
                        t.offset as u64,
                        t.ne0() as u32,
                        t.ne1() as u32,
                        m::buf::X,
                        m::buf::O,
                        n_tok,
                    );
                }
                m::end().map_err(|e| format!("end {e}"))?;
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let tg = (tensors[0].ne1() as u32).div_ceil(64) * n_tok.div_ceil(64);
                println!(
                    "{label:<22} {} layers {ms:>8.2} ms {:>6.2} TFLOPS  {tg:>5} threadgroups/mm",
                    tensors.len(),
                    flops / (ms / 1e3) / 1e12
                );
            }
        }

        // the tied lm_head, read once per token
        {
            let t = *w.get("token_embd.weight").ok_or("missing token_embd")?;
            let t0 = std::time::Instant::now();
            m::begin();
            m::matmat(
                1,
                t.offset as u64,
                t.ne0() as u32,
                t.ne1() as u32,
                m::buf::CUR,
                m::buf::LOGITS,
                1,
            );
            m::end().map_err(|e| format!("end {e}"))?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!(
                "{:<24} {:>7.1} MiB single             {ms:>7.2} ms  {:>6.1} GB/s COLD",
                "lm_head 2560->262144",
                t.bytes as f64 / (1 << 20) as f64,
                t.bytes as f64 / (ms / 1e3) / 1e9
            );
        }

        // Cost of a TINY dispatch. A decode step issues ~900 of these (norms, rope, adds,
        // copies, scales) over 2560-float rows; if each costs microseconds, they dominate a
        // forward whose matmuls could finish in ~20 ms.
        let n_small = 2000;
        m::alloc(m::buf::X, 2560 * 4).map_err(|e| format!("alloc {e}"))?;
        m::begin();
        for _ in 0..64 {
            m::scale(m::buf::X, 1.0, 2560);
        }
        m::end().map_err(|e| format!("end {e}"))?;
        let t0 = std::time::Instant::now();
        m::begin();
        for _ in 0..n_small {
            m::scale(m::buf::X, 1.0, 2560);
        }
        m::end().map_err(|e| format!("end {e}"))?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / f64::from(n_small);
        println!("tiny dispatch (2560 floats): {us:.2} us each");

        let t1 = std::time::Instant::now();
        m::begin();
        for _ in 0..n_small {
            m::rms_norm(m::buf::X, m::NO_WEIGHT, 2560, 1e-6, 1, 2560, 0);
        }
        m::end().map_err(|e| format!("end {e}"))?;
        let us2 = t1.elapsed().as_secs_f64() * 1e6 / f64::from(n_small);
        println!("rms_norm  (2560 floats): {us2:.2} us each");
        // every op type, at the shapes a batch-1 decode step actually uses
        m::alloc(m::buf::Q, 8 * 512 * 4).map_err(|e| format!("alloc {e}"))?;
        m::alloc(m::buf::K, 2 * 512 * 4).map_err(|e| format!("alloc {e}"))?;
        m::alloc(m::buf::ATTN, 8 * 512 * 4).map_err(|e| format!("alloc {e}"))?;
        m::alloc(m::buf::G, 10240 * 4).map_err(|e| format!("alloc {e}"))?;
        m::alloc(m::buf::U, 10240 * 4).map_err(|e| format!("alloc {e}"))?;
        m::alloc(m::buf::O, 2560 * 4).map_err(|e| format!("alloc {e}"))?;
        m::alloc_kv(&[2 * 512 * 2048 * 4]).map_err(|e| format!("kv {e}"))?;

        let time = |label: &str, count_per_token: f64, f: &dyn Fn()| {
            m::begin();
            for _ in 0..64 {
                f();
            }
            let _ = m::end();
            let t = std::time::Instant::now();
            m::begin();
            for _ in 0..500 {
                f();
            }
            let _ = m::end();
            let us = t.elapsed().as_secs_f64() * 1e6 / 500.0;
            println!(
                "{label:<28}{us:>9.2} us x{count_per_token:>6.0}/token = {:>7.2} ms",
                us * count_per_token / 1e3
            );
            us * count_per_token / 1e3
        };
        let mut ms = 0.0;
        ms += time("rms_norm w=2560 rows=1", 210.0, &|| {
            m::rms_norm(m::buf::X, m::NO_WEIGHT, 2560, 1e-6, 1, 2560, 0);
        });
        ms += time("rms_norm w=256  rows=8", 126.0, &|| {
            m::rms_norm(m::buf::Q, m::NO_WEIGHT, 256, 1e-6, 8, 256, 0);
        });
        ms += time("rope  heads=8 dim=256", 42.0, &|| {
            m::rope(m::buf::Q, 256, 10000.0, 256, 8, 0, 1, None);
        });
        ms += time("attention h=8 pos=200", 42.0, &|| {
            m::attention(0, 256, 8, 2, 1024, 200, 512, 1, 201, 511);
        });
        // Decode attention against KV length. If it is bandwidth bound the time should track
        // the KV bytes read; if it is dispatch/occupancy bound it should be nearly flat.
        if std::env::var("IMPARO_ATTN").is_ok() {
            if let Ok(v) = std::env::var("IMPARO_ATTN_MIN_TGS") {
                if let Ok(n) = v.parse::<u32>() {
                    m::set_attn_min_tgs(n);
                }
            }
            // 8 query heads over 2 KV heads: one threadgroup per query head means each KV
            // entry is read by 4 threadgroups. Same n_kv, fewer query heads, tells us whether
            // those repeat reads cost DRAM traffic (time flat) or issue/L2 (time scales).
            for pos in [512u32, 1024] {
                for nh in [2u32, 4, 8] {
                    let us = time(
                        &format!("attn pos={pos} qheads={nh} kv=2"),
                        42.0,
                        &|| {
                            m::attention(
                                0,
                                256,
                                nh,
                                2,
                                2048,
                                pos,
                                1024,
                                1,
                                pos + 1,
                                2047,
                            );
                        },
                    );
                    let _ = us;
                }
            }
        }
        ms += time("kv_store w=1024", 48.0, &|| {
            m::kv_store(m::buf::K, 0, 1024, 0, 1, false, 511);
        });
        ms += time("add 2560", 126.0, &|| m::add(m::buf::X, m::buf::O, 2560));
        ms += time("copy 2560", 84.0, &|| m::copy(m::buf::CUR, m::buf::X, 2560));
        ms += time("scale 2560", 42.0, &|| m::scale(m::buf::X, 1.0, 2560));
        ms += time("act_mul 10240", 42.0, &|| {
            m::act_mul(m::buf::G, m::buf::U, 10240);
        });
        ms += time("act 256", 42.0, &|| m::act(m::buf::GATE, 256));
        ms += time("mul_strided 256", 42.0, &|| {
            m::mul_strided(m::buf::GATE, m::buf::PER_LAYER, 256, 0, 10752, 256, 1);
        });
        println!(
            "ALL non-matmul total {ms:.2} ms/token (matmuls at 131 GB/s would be ~19.6 ms)"
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    bench::main()
}
