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

    /// The tree delta probe's geometry, IMPARO_BENCH_DELTA_GEOM="k_heads,v_heads,key_dim,value_dim";
    /// Qwen3.8-27B's recurrent layer when unset.
    fn bench_delta_geom() -> Result<(usize, usize, usize, usize), String> {
        let spec = std::env::var("IMPARO_BENCH_DELTA_GEOM")
            .unwrap_or_else(|_| "16,48,128,128".into());
        let dims: Vec<usize> = spec
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        match dims[..] {
            [kh, vh, kd, vd]
                if kh > 0 && vh > 0 && kd > 0 && vd > 0 && vh % kh == 0 =>
            {
                Ok((kh, vh, kd, vd))
            }
            _ => Err(format!(
                "IMPARO_BENCH_DELTA_GEOM={spec}: want k_heads,v_heads,key_dim,value_dim"
            )),
        }
    }

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
        // THE TREE DELTA PROBE'S HEAD WIDTHS, before enable_gpu builds the library: the delta
        // kernels are compiled for them, and a plan without a delta layer sets none.
        if std::env::var_os("IMPARO_BENCH_DELTA_TREE").is_some() {
            let (_, _, kd, vd) = bench_delta_geom()?;
            m::set_recurrent_dims(u32::try_from(kd)?, u32::try_from(vd)?);
        }
        let mut w = Weights::open(&path)?;
        // The plan is what says which activation the epilogue kernels are specialised
        // with, so the bench profiles the pipelines this model would actually run.
        let plan = imparo_model::build_plan(&imparo_gguf::read(&path)?, &path)?;
        imparo_model::backend::enable_gpu(
            &mut w,
            &plan,
            plan.config.context_length as usize,
        )?;
        if !w.gpu_enabled() {
            return Err("metal not enabled".into());
        }

        // ---- ATTENTION PROBE: one prefill attention dispatch per geometry ----------------
        //
        // IMPARO_BENCH_ATTN="n_tok,pos;n_tok,pos" times ONE dispatch of this model's prefill
        // attention (its attention layers' head dim, full attention, causal over pos + n_tok
        // keys) on synthetic f16 K/V, GPU-timed, min of five after a warm-up. It is how this
        // engine's kernel is put beside another engine's on the SAME geometry with neither
        // graph around it: llama.cpp's `test-backend-ops perf -o FLASH_ATTN_EXT` with
        // TBO_CAUSAL_MASK=1 at (hsk = hd, nh = kv heads, nr23 = {GQA, 1}, kv = pos + n_tok,
        // nb = n_tok) is the other side (docs/evidence/bracket/2026-09-01-attention-
        // isolation.md). The route follows the engine's own selection, so IMPARO_ATTN_FA=1
        // and the tuned config apply and the time is the kernel the engine would run. Light
        // by construction: one dispatch per sample, no sweep.
        if let Ok(spec) = std::env::var("IMPARO_BENCH_ATTN") {
            let c = &plan.config;
            let pick = |want_full: bool| {
                plan.layers.iter().find_map(|l| match l.attention {
                    imparo_model::Attention::Full { head_dim, .. } if want_full => {
                        Some(head_dim)
                    }
                    imparo_model::Attention::Window { head_dim, .. } if !want_full => {
                        Some(head_dim)
                    }
                    _ => None,
                })
            };
            // IMPARO_BENCH_ATTN_WINDOW=1 times the WINDOWED layers' head dim instead (gemma4
            // E4B: 35 layers at hd 256 over at most 512 keys -- the fixed per-token attention
            // cost at every context length); default is the full-attention head dim.
            let want_window =
                std::env::var("IMPARO_BENCH_ATTN_WINDOW").is_ok_and(|v| v == "1");
            let hd = pick(!want_window)
                .or_else(|| pick(want_window))
                .ok_or("model has no attention layer")?;
            let geoms: Vec<(u32, u32)> = spec
                .split(';')
                .filter_map(|p| {
                    let (a, b) = p.split_once(',')?;
                    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
                })
                .collect();
            let span = geoms
                .iter()
                .map(|&(n, p)| n + p)
                .max()
                .ok_or("IMPARO_BENCH_ATTN: no geometry (want \"n_tok,pos;...\")")?;
            // One KV layer (index 0), f16 rows of kv_width, with the tuner's allowance over
            // the span so the ring the engine allocates is reproduced.
            let kv_width = c.n_kv_heads * hd;
            let slots = u64::from(span + span / 4);
            let kv_bytes = u64::from(kv_width) * 2 * slots;
            // COLD K/V (IMPARO_BENCH_KV_LAYERS=N): N layers' caches, the probe cycling through
            // them one per dispatch, so consecutive dispatches never find their K/V in the
            // cache -- the engine's case, where a gigabyte of weights streams between two
            // attention layers. One layer (the default) re-reads a warm cache.
            let kv_layers: u32 = std::env::var("IMPARO_BENCH_KV_LAYERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1)
                .max(1);
            let layer_bytes: Vec<u64> = (0..kv_layers).map(|_| kv_bytes).collect();
            m::alloc_kv(&layer_bytes).map_err(|rc| format!("alloc_kv rc={rc}"))?;
            // WRITE THE CACHE BEFORE READING IT. A buffer that was allocated and never
            // touched is demand-zero pages; the kernel would time page faults, not loads,
            // and the more span it sweeps the more of them. Real half values in every row,
            // as the tuner stages its synthetic K/V (IMPARO_BENCH_KV_ZERO=1 keeps the
            // untouched buffer, for measuring that difference itself).
            if std::env::var_os("IMPARO_BENCH_KV_ZERO").is_none() {
                let mut row =
                    vec![0_u8; usize::try_from(kv_bytes).expect("kv bytes fit usize")];
                for (i, pair) in row.chunks_exact_mut(2).enumerate() {
                    // f16 in [-0.5, 0.5): a 10-bit mantissa pattern keyed on the element index
                    let v = ((i as u32 * 2_654_435_761_u32) >> 22) as u16;
                    let bits = 0x3000u16 | (v & 0x03ff) | ((i as u16 & 1) << 15);
                    pair.copy_from_slice(&bits.to_le_bytes());
                }
                for l in 0..kv_layers {
                    m::write_kv_bytes(l, false, 0, &row);
                    m::write_kv_bytes(l, true, 0, &row);
                }
            }
            // SCATTERED PAGES (IMPARO_BENCH_KV_SCATTER=1): the pool places a conversation's
            // 64-cell pages wherever it has room, so the engine's attention walks a permuted
            // table where the bench's default walks the identity. A fixed pseudo-random
            // permutation of the layer's pages puts the placement cost on the probe; the
            // difference against the identity run is what the pool's placement costs the
            // kernel. Same set of pages, so every read stays inside the allocation.
            if std::env::var_os("IMPARO_BENCH_KV_SCATTER").is_some() {
                let npages = u32::try_from(slots / 64).expect("page count fits u32");
                let mut perm: Vec<u32> = (0..npages).collect();
                let mut x: u32 = 0x9e37_79b9;
                for i in (1..perm.len()).rev() {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5; // xorshift32, deterministic
                    let j = (x as usize) % (i + 1);
                    perm.swap(i, j);
                }
                m::set_kv_page_table(0, &perm);
                println!("-- pages scattered: {npages} pages, a fixed permutation --");
            }
            let max_tok = geoms.iter().map(|&(n, _)| n).max().unwrap_or(1);
            let act = u64::from(max_tok) * u64::from(c.n_heads) * u64::from(hd) * 4;
            m::alloc(m::buf::Q, act).map_err(|rc| format!("alloc Q rc={rc}"))?;
            m::alloc(m::buf::ATTN, act).map_err(|rc| format!("alloc ATTN rc={rc}"))?;
            println!(
                "-- attention probe: hd={hd} heads={}/{} f16 KV, one dispatch, min of 5 --",
                c.n_heads, c.n_kv_heads
            );
            // WARM THE CLOCK FIRST. The GPU idles down between processes, and a probe taken
            // on the way back up reads high and swings: the 1024-key geometry read 2.6 ms in
            // one run and 1.15 in the next with nothing changed. ~300 ms of the deepest
            // geometry's own dispatches before the first measurement, as the tuner warms up.
            if let Some(&(wn, wp)) = geoms.iter().max_by_key(|&&(n, p)| n + p) {
                let t0 = std::time::Instant::now();
                while t0.elapsed().as_millis() < 300 {
                    m::begin();
                    m::attention(
                        0,
                        hd,
                        c.n_heads,
                        c.n_kv_heads,
                        kv_width,
                        wp,
                        0,
                        wn,
                        (wp + wn).clamp(1, 8192),
                        0,
                    );
                    m::end().map_err(|rc| format!("attention rc={rc}"))?;
                }
            }
            // IMPARO_BENCH_ATTN_ROWS=1 also times the row-layout entry, on a chain layout: row t
            // sees rows 0..=t at position pos + t, the visibility the causal entry synthesises,
            // so the two times differ only by the masked kernel's own work (plan step 2, A2).
            // Each arm is warmed, then the arms alternate over two rounds.
            let rows_env = std::env::var("IMPARO_BENCH_ATTN_ROWS").unwrap_or_default();
            let rows_arm = rows_env == "1" || rows_env == "float";
            // IMPARO_BENCH_ATTN_ROWS=float adds a third arm: the row-layout entry with Q kept in
            // float (function constant 26), on the same chain layout.
            let float_arm = rows_env == "float";
            if rows_arm {
                if !m::supports_row_layout(hd) {
                    return Err(format!(
                        "IMPARO_BENCH_ATTN_ROWS: no row-layout entry at head dim {hd}"
                    )
                    .into());
                }
                m::alloc(
                    BufId::RowLayout as u32,
                    (imparo_backend::ROW_LAYOUT_MAX_ROWS
                        * imparo_backend::ROW_LAYOUT_WORDS
                        * 4) as u64,
                )
                .map_err(|rc| format!("alloc row layout rc={rc}"))?;
            }
            for &(n_tok, pos) in &geoms {
                let keys = pos + n_tok;
                let rows_here = rows_arm && (2..=64).contains(&n_tok);
                if rows_here {
                    let width = imparo_backend::ROW_LAYOUT_WORDS;
                    let mut words = vec![0_u32; n_tok as usize * width];
                    for t in 0..n_tok as usize {
                        let row = &mut words[t * width..(t + 1) * width];
                        let seen = if t >= 63 {
                            u64::MAX
                        } else {
                            (1_u64 << (t + 1)) - 1
                        };
                        row[0] = pos + t as u32;
                        row[1] = t as u32;
                        row[2] = seen as u32;
                        row[3] = (seen >> 32) as u32;
                        for back in 1..=imparo_backend::ROW_LAYOUT_ANCESTORS.min(t) {
                            row[3 + back] = (t - back) as u32;
                        }
                    }
                    m::write_u32(BufId::RowLayout as u32, 0, &words);
                }
                let next_layer = std::cell::Cell::new(0u32);
                // Arm 0 the causal entry, 1 the row-layout entry, 2 the row-layout entry with float Q.
                let dispatch = |arm: u8| -> bool {
                    let l = next_layer.get();
                    next_layer.set((l + 1) % kv_layers);
                    if arm > 0 {
                        return m::attention_rows(
                            l,
                            hd,
                            c.n_heads,
                            c.n_kv_heads,
                            kv_width,
                            pos,
                            0,
                            n_tok,
                            1.0,
                            BufId::RowLayout as u32,
                            arm == 2,
                        );
                    }
                    m::attention(
                        l,
                        hd,
                        c.n_heads,
                        c.n_kv_heads,
                        kv_width,
                        pos,
                        0,
                        n_tok,
                        keys.clamp(1, 8192),
                        0,
                    );
                    true
                };
                // SEVERAL DISPATCHES PER COMMAND BUFFER, time divided by their count, as the
                // tuner's time_us does. A command buffer carries a fixed GPU-side cost that a
                // single ~1 ms dispatch cannot hide: one dispatch per buffer read 1.7-1.95 ms
                // for a kernel the tuner times at 1.0, and matched it within 3% at 20 ms.
                // Two phases, because the clock ramps over the first buffers and any ONE
                // early sample over-sizes nothing: five single-dispatch buffers give a min,
                // that min sizes the rep count to ~4 ms of work, and five buffers of that
                // many dispatches give the number reported.
                // Both the min and the mean are printed: llama.cpp's `test-backend-ops perf`
                // reports the MEAN over >= 1 s of runs, so a comparison against it reads the
                // mean column; the min is the kernel's own cost with the machine's phase out.
                // Five buffers gave a mean 5-31% over the min (one slow buffer in five moves
                // it); the mean is taken over >= 300 ms of GPU time instead, at least five
                // buffers, each still <= ~4 ms of dispatches so nothing here can stall the
                // screen.
                // IMPARO_BENCH_ATTN_REPS=N pins the dispatches per buffer (1 = one dispatch
                // per command buffer: the LATENCY of a dispatch standing alone, the way a
                // decode step runs it, plus the buffer's fixed cost, which an A/B cancels).
                let forced_reps: Option<usize> =
                    std::env::var("IMPARO_BENCH_ATTN_REPS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .filter(|&r: &usize| r >= 1);
                let sample = |arm: u8,
                              reps: usize,
                              floor_us: f64|
                 -> Result<(f64, f64, usize), String> {
                    let reps = forced_reps.unwrap_or(reps);
                    let (mut best, mut sum, mut n) = (f64::INFINITY, 0.0, 0usize);
                    while n < 5 || sum < floor_us {
                        m::begin();
                        for _ in 0..reps {
                            if !dispatch(arm) {
                                let _ = m::end();
                                return Err(
                                    "the row-layout attention entry refused".into()
                                );
                            }
                        }
                        m::end().map_err(|rc| format!("attention rc={rc}"))?;
                        let us = m::last_gpu_us();
                        best = best.min(us / reps as f64);
                        sum += us;
                        n += 1;
                    }
                    Ok((best, sum / (n * reps) as f64, n * reps))
                };
                let arms: &[u8] = match (rows_here, float_arm) {
                    (true, true) => &[0, 1, 2],
                    (true, false) => &[0, 1],
                    _ => &[0],
                };
                for &arm in arms {
                    sample(arm, 1, 0.0)?;
                }
                let single = sample(0, 1, 0.0)?;
                let reps = ((4000.0 / single.0.max(1.0)) as usize).clamp(1, 64);
                let rounds = if rows_here { 2 } else { 1 };
                for round in 1..=rounds {
                    for &arm in arms {
                        let (best, mean, n) = sample(arm, reps, 300_000.0)?;
                        let entry = match (rows_arm, arm) {
                            (false, _) => String::new(),
                            (true, 2) => format!("entry=rows_float_q round={round}  "),
                            (true, 1) => format!("entry=rows round={round}  "),
                            (true, _) => format!("entry=causal round={round}  "),
                        };
                        println!(
                            "attn probe  {entry}n_tok={n_tok} pos={pos} keys={keys}  min {best:.1} us  mean {mean:.1} us  ({n} dispatches, {reps} per buffer)"
                        );
                    }
                }
            }
            return Ok(());
        }

        // ---- GATED DELTA RULE OVER A DRAFT TREE: CHECKED, THEN TIMED BESIDE THE SERIAL RULE --
        //
        // IMPARO_BENCH_DELTA_TREE="16,64" checks `imparo_delta_net_tree` two ways -- against
        // `imparo_cpu::ops::delta_net` run along each node's own path from the same state (the
        // definition), and bit for bit against `imparo_delta_net` run the same way (the kernel
        // it mirrors) -- then times it beside `imparo_delta_net` (implementation plan, step 6).
        // Synthetic rows at IMPARO_BENCH_DELTA_GEOM (Qwen3.8-27B's recurrent layer when unset),
        // so any model file serves: the head widths were set before init, and two of the file's
        // F32 tensors stand in for a and dt_bias. A tree is laid depth-first as a verify lays it
        // -- a chain for the drafter's first choice at every depth, the other nodes branching off
        // it -- and each size also runs as a plain chain while that fits the kernel's depth.
        // IMPARO_BENCH_DELTA_TREE_TIME=0 stops after the checks.
        if let Ok(spec) = std::env::var("IMPARO_BENCH_DELTA_TREE") {
            use imparo_backend::{ROW_LAYOUT_MAX_ROWS, ROW_LAYOUT_WORDS};
            let (kh, vh, kd, vd) = bench_delta_geom()?;
            if !m::supports_gated_delta() {
                return Err(
                    "IMPARO_BENCH_DELTA_TREE: no delta pipeline was built".into()
                );
            }
            let sizes: Vec<usize> = spec
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .filter(|&n| n > 0)
                .collect();
            let max_n =
                sizes.iter().copied().max().ok_or(
                    "IMPARO_BENCH_DELTA_TREE: want node counts, e.g. \"16,64\"",
                )?;
            if max_n > ROW_LAYOUT_MAX_ROWS {
                return Err(format!(
                    "IMPARO_BENCH_DELTA_TREE: {max_n} nodes; a row layout holds {ROW_LAYOUT_MAX_ROWS}"
                )
                .into());
            }
            let shape = imparo_model::ops::DeltaShape {
                k_heads: kh,
                v_heads: vh,
                key_dim: kd,
                value_dim: vd,
            };
            let (qkv_width, state_elems, v_width) =
                (shape.qkv_width(), shape.state_elems(), vh * vd);
            let (khu, vhu, kdu, vdu) = (kh as u32, vh as u32, kd as u32, vd as u32);
            let depth_limit = m::delta_tree_depth() as usize;
            let eps = 1e-6_f32;
            println!(
                "-- delta tree probe: k_heads={kh} v_heads={vh} key_dim={kd} value_dim={vd} depth<{depth_limit} --"
            );

            // Stand-ins for a (whose sign decides whether a head forgets) and dt_bias: the first
            // F32 tensor with a negative value among its first v_heads, and the first F32 tensor.
            let pick = |want_negative: bool| {
                w.tensors
                    .iter()
                    .find(|(_, t)| {
                        let elems: u64 = t.ne[..t.n_dims as usize].iter().product();
                        t.ggml_type == 0
                            && elems >= vh as u64
                            && (!want_negative
                                || w.f32s(t)[..vh].iter().any(|&x| x < 0.0))
                    })
                    .map(|(_, t)| *t)
            };
            let (Some(a_t), Some(dt_t)) = (pick(true), pick(false)) else {
                return Err(
                    "IMPARO_BENCH_DELTA_TREE: no F32 tensors in the file to stand in for a and dt_bias"
                        .into(),
                );
            };
            let wa: Vec<f32> = w.f32s(&a_t)[..vh].to_vec();
            let wdt: Vec<f32> = w.f32s(&dt_t)[..vh].to_vec();
            let (a_off, dt_off) = (a_t.offset as u64, dt_t.offset as u64);

            // Deterministic rows, xorshift64 into [-1, 1). A head whose stand-in a is positive
            // would grow its state, so its alpha sits far below zero (decay 1); the others draw
            // alpha from [-3, 1). The CPU rule takes the gates reduced, as the kernels reduce them.
            let mut rng = 0x2545_f491_4f6c_dd1d_u64;
            let mut uniform = move || -> f32 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 40) as f32 / (1_u64 << 24) as f32 * 2.0 - 1.0
            };
            let qkv: Vec<f32> = (0..max_n * qkv_width).map(|_| uniform()).collect();
            let alpha: Vec<f32> = (0..max_n * vh)
                .map(|i| {
                    if wa[i % vh] > 0.0 {
                        -12.0
                    } else {
                        2.0 * uniform() - 1.0
                    }
                })
                .collect();
            let beta_raw: Vec<f32> = (0..max_n * vh).map(|_| 3.0 * uniform()).collect();
            let s0: Vec<f32> = (0..state_elems).map(|_| 0.3 * uniform()).collect();
            let softplus = |x: f32| if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
            let sigmoid = |x: f32| {
                if x >= 0.0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                }
            };
            let log_decay: Vec<f32> = alpha
                .iter()
                .enumerate()
                .map(|(i, &x)| wa[i % vh] * softplus(x + wdt[i % vh]))
                .collect();
            let beta: Vec<f32> = beta_raw.iter().map(|&x| sigmoid(x)).collect();
            for (id, elems, what) in [
                (BufId::Model0, max_n * qkv_width, "qkv"),
                (BufId::Model1, max_n * vh, "alpha"),
                (BufId::Model2, max_n * vh, "beta"),
                (BufId::Model3, max_n * qkv_width, "path qkv"),
                (BufId::Model4, max_n * vh, "path alpha"),
                (BufId::Model5, max_n * vh, "path beta"),
                (BufId::Recur, 2 * state_elems, "state"),
                (BufId::O, max_n * v_width, "tree out"),
                (BufId::Attn, max_n * v_width, "serial out"),
                (BufId::RowLayout, max_n * ROW_LAYOUT_WORDS, "row layout"),
            ] {
                m::alloc(id as u32, (elems * 4) as u64)
                    .map_err(|rc| format!("alloc {what} rc={rc}"))?;
            }
            m::write(BufId::Model0 as u32, 0, &qkv);
            m::write(BufId::Model1 as u32, 0, &alpha);
            m::write(BufId::Model2 as u32, 0, &beta_raw);
            m::write(BufId::Recur as u32, 0, &s0);

            // Parents of an `n`-node tree laid depth-first, the drafter's first choice first: a
            // chain of `chain` nodes, then the other nodes hung off it, mostly from chain nodes
            // spread over the depths, every third below the node made just before it while that
            // stays shallower than `limit`. Children are visited in the order they were made.
            let draft_tree = |n: usize, chain: usize, limit: usize| -> Vec<i32> {
                let chain = chain.clamp(1, n);
                let mut parent: Vec<usize> = vec![usize::MAX];
                let mut depth: Vec<usize> = vec![0];
                for i in 1..n {
                    let p = if i < chain {
                        i - 1
                    } else {
                        let k = i - chain;
                        if k % 3 == 2 && depth[i - 1] + 1 < limit {
                            i - 1
                        } else {
                            (k * 5) % (chain - 1).max(1)
                        }
                    };
                    parent.push(p);
                    depth.push(depth[p] + 1);
                }
                let mut children = vec![Vec::new(); n];
                for (i, &p) in parent.iter().enumerate().skip(1) {
                    children[p].push(i);
                }
                let mut order = Vec::with_capacity(n);
                let mut stack = vec![0_usize];
                while let Some(v) = stack.pop() {
                    order.push(v);
                    stack.extend(children[v].iter().rev().copied());
                }
                let mut row_of = vec![0_usize; n];
                for (row, &v) in order.iter().enumerate() {
                    row_of[v] = row;
                }
                order
                    .iter()
                    .map(|&v| if v == 0 { -1 } else { row_of[parent[v]] as i32 })
                    .collect()
            };
            let mut cases: Vec<(String, Vec<u32>)> = Vec::new();
            for &n in &sizes {
                let mut shapes: Vec<(&str, Vec<i32>)> = Vec::new();
                if n <= depth_limit {
                    shapes.push(("chain", (0..n).map(|i| i as i32 - 1).collect()));
                }
                shapes.push((
                    "tree",
                    draft_tree(n, (n / 2).clamp(1, depth_limit), depth_limit),
                ));
                for (kind, parents) in shapes {
                    let words = imparo_model::tree_row_layout(0, &parents)?;
                    let deepest = words
                        .chunks_exact(ROW_LAYOUT_WORDS)
                        .map(|r| r[1])
                        .max()
                        .unwrap_or(0);
                    cases.push((format!("{kind} n={n} depth={deepest}"), words));
                }
            }
            let tree = |words: &[u32]| {
                m::delta_net_tree(
                    BufId::Model0 as u32,
                    BufId::Model1 as u32,
                    BufId::Model2 as u32,
                    a_off,
                    dt_off,
                    BufId::Recur as u32,
                    0,
                    BufId::O as u32,
                    BufId::RowLayout as u32,
                    words,
                    khu,
                    vhu,
                    kdu,
                    vdu,
                    eps,
                )
            };
            // The serial kernel reads plane 0 and writes plane 1, so every dispatch starts from S0.
            let serial = |qkv_buf: u32, alpha_buf: u32, beta_buf: u32, n_tok: u32| {
                m::delta_net(
                    qkv_buf,
                    alpha_buf,
                    beta_buf,
                    a_off,
                    dt_off,
                    BufId::Recur as u32,
                    0,
                    state_elems as u32,
                    BufId::Attn as u32,
                    khu,
                    vhu,
                    kdu,
                    vdu,
                    n_tok,
                    eps,
                    m::NO_EPILOGUE,
                    0,
                    m::NO_SNAP,
                    0,
                    0,
                )
            };

            let mut all_pass = true;
            for (name, words) in &cases {
                let n = words.len() / ROW_LAYOUT_WORDS;
                m::write_u32(BufId::RowLayout as u32, 0, words);
                m::begin();
                let accepted = tree(words);
                m::end().map_err(|rc| format!("{name}: rc={rc}"))?;
                if !accepted {
                    return Err(format!(
                        "{name}: the tree delta kernel refused the layout"
                    )
                    .into());
                }
                let mut got = vec![0.0_f32; n * v_width];
                m::read(BufId::O as u32, 0, &mut got);
                let cpu_start = std::time::Instant::now();
                let mut want = vec![0.0_f32; n * v_width];
                let mut serial_rows = vec![0.0_f32; n * v_width];
                for t in 0..n {
                    let mut path = vec![t];
                    let mut at = t;
                    while words[at * ROW_LAYOUT_WORDS + 1] > 0 {
                        at = words[at * ROW_LAYOUT_WORDS + 4] as usize;
                        path.push(at);
                    }
                    path.reverse();
                    let len = path.len();
                    let gather = |src: &[f32], width: usize| -> Vec<f32> {
                        path.iter()
                            .flat_map(|&r| {
                                src[r * width..(r + 1) * width].iter().copied()
                            })
                            .collect()
                    };
                    let rows = gather(&qkv, qkv_width);
                    // The definition: the CPU rule along the node's path from S0.
                    let mut state = s0.clone();
                    let mut out = vec![0.0_f32; len * v_width];
                    imparo_model::ops::delta_net(
                        &rows,
                        &gather(&log_decay, vh),
                        &gather(&beta, vh),
                        &mut state,
                        &mut out,
                        shape,
                        len,
                        eps,
                    );
                    want[t * v_width..(t + 1) * v_width]
                        .copy_from_slice(&out[(len - 1) * v_width..]);
                    // The kernel this one mirrors, along the same path from the same state.
                    m::write(BufId::Model3 as u32, 0, &rows);
                    m::write(BufId::Model4 as u32, 0, &gather(&alpha, vh));
                    m::write(BufId::Model5 as u32, 0, &gather(&beta_raw, vh));
                    m::begin();
                    let ran = serial(
                        BufId::Model3 as u32,
                        BufId::Model4 as u32,
                        BufId::Model5 as u32,
                        len as u32,
                    );
                    m::end().map_err(|rc| format!("{name}: serial rc={rc}"))?;
                    if !ran {
                        return Err(
                            format!("{name}: the serial delta kernel refused").into()
                        );
                    }
                    let mut path_out = vec![0.0_f32; len * v_width];
                    m::read(BufId::Attn as u32, 0, &mut path_out);
                    serial_rows[t * v_width..(t + 1) * v_width]
                        .copy_from_slice(&path_out[(len - 1) * v_width..]);
                }
                let cpu_ms = cpu_start.elapsed().as_secs_f64() * 1e3;
                let (mut worst_rel, mut worst_abs, mut largest, mut worst_at) =
                    (0.0_f64, 0.0_f64, 0.0_f64, 0_usize);
                for (i, (&g, &c)) in got.iter().zip(&want).enumerate() {
                    let (g, c) = (f64::from(g), f64::from(c));
                    let abs = (g - c).abs();
                    let rel = if g.is_finite() {
                        abs / c.abs().max(1e-3)
                    } else {
                        f64::INFINITY
                    };
                    if rel > worst_rel {
                        worst_rel = rel;
                        worst_at = i;
                    }
                    worst_abs = worst_abs.max(abs);
                    largest = largest.max(c.abs());
                }
                let differ = got
                    .iter()
                    .zip(&serial_rows)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let serial_worst = max_abs_diff(&got, &serial_rows);
                // The CPU bound is the prototype's, not a gate: the CPU rule sums in its own order.
                let pass = worst_rel < 1e-3;
                all_pass &= pass;
                println!(
                    "delta tree check  {name}  cpu rule: worst_rel={worst_rel:.2e} (node {} value {}) worst_abs={worst_abs:.2e} max|o|={largest:.2e}  serial kernel: {differ}/{} values differ, largest {serial_worst:.2e}  {cpu_ms:.0} ms  {}",
                    worst_at / v_width,
                    worst_at % v_width,
                    got.len(),
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            if let Some((_, words)) = cases.last()
                && words.len() >= 2 * ROW_LAYOUT_WORDS
            {
                let mut jump = words.clone();
                jump[ROW_LAYOUT_WORDS + 1] = 2;
                m::begin();
                let refused = !tree(&jump);
                m::end().map_err(|rc| format!("depth jump: rc={rc}"))?;
                println!(
                    "delta tree check  a row two levels below the row before it is refused: {refused}"
                );
                all_pass &= refused;
            }
            if !all_pass {
                return Err("delta tree check FAILED".into());
            }
            if std::env::var("IMPARO_BENCH_DELTA_TREE_TIME").as_deref() == Ok("0") {
                return Ok(());
            }

            // Arms: the tree kernel on every case, the serial kernel over the same rows at each
            // size, and the serial kernel at 1 and 4 rows, what a commit replays.
            let mut arms: Vec<(String, Option<&[u32]>, u32)> = cases
                .iter()
                .map(|(name, words)| {
                    (format!("tree kernel, {name}"), Some(words.as_slice()), 0)
                })
                .collect();
            for &n in sizes.iter().chain(&[1, 4]) {
                arms.push((format!("serial kernel, n={n}"), None, n as u32));
            }
            let dispatch = |words: Option<&[u32]>, n_tok: u32| -> bool {
                match words {
                    Some(words) => tree(words),
                    None => serial(
                        BufId::Model0 as u32,
                        BufId::Model1 as u32,
                        BufId::Model2 as u32,
                        n_tok,
                    ),
                }
            };
            // Several dispatches per command buffer, each buffer about 4 ms, min and mean over
            // >= 300 ms of GPU time: the attention probe's instrument.
            let sample = |arm: &(String, Option<&[u32]>, u32),
                          reps: usize,
                          floor_us: f64|
             -> Result<(f64, f64, usize), String> {
                if let Some(words) = arm.1 {
                    m::write_u32(BufId::RowLayout as u32, 0, words);
                }
                let (mut best, mut sum, mut n) = (f64::INFINITY, 0.0, 0_usize);
                while n < 5 || sum < floor_us {
                    m::begin();
                    for _ in 0..reps {
                        if !dispatch(arm.1, arm.2) {
                            let _ = m::end();
                            return Err(format!("{}: refused", arm.0));
                        }
                    }
                    m::end().map_err(|rc| format!("{}: rc={rc}", arm.0))?;
                    let us = m::last_gpu_us();
                    best = best.min(us / reps as f64);
                    sum += us;
                    n += 1;
                }
                Ok((best, sum / (n * reps) as f64, n * reps))
            };
            if let Some(heaviest) = arms.iter().rev().find(|a| a.1.is_some()) {
                let warm = std::time::Instant::now();
                while warm.elapsed().as_millis() < 300 {
                    sample(heaviest, 1, 0.0)?;
                }
            }
            let mut reps = Vec::with_capacity(arms.len());
            for arm in &arms {
                let single = sample(arm, 1, 0.0)?;
                reps.push(((4000.0 / single.0.max(1.0)) as usize).clamp(1, 64));
            }
            for round in 1..=2 {
                for (arm, &r) in arms.iter().zip(&reps) {
                    let (best, mean, n) = sample(arm, r, 300_000.0)?;
                    println!(
                        "delta probe  {:<34} round={round}  min {best:.1} us  mean {mean:.1} us  ({n} dispatches, {r} per buffer)",
                        arm.0
                    );
                }
            }
            return Ok(());
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
                return Err(
                    format!("{name} absent: this check needs an LFM2 file").into()
                );
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
                    imparo_model::ops::causal_conv(
                        imparo_backend::ConvForm::GatedBcx,
                        &bcx,
                        &cw,
                        &mut host_state,
                        &mut host_out,
                        width,
                        kern,
                        n_tok,
                    );
                    be.begin();
                    be.write(BufId::Model0, 0, &bcx);
                    be.causal_conv(
                        imparo_backend::ConvForm::GatedBcx,
                        BufId::Model0,
                        cw_t.offset as u64,
                        BufId::Recur,
                        0,
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
                        println!(
                            "    out host {:?}",
                            &host_out[..4.min(host_out.len())]
                        );
                        println!("    out dev  {:?}", &dev_out[..4.min(dev_out.len())]);
                        println!(
                            "    st  host {:?}",
                            &host_state[..4.min(host_state.len())]
                        );
                        println!(
                            "    st  dev  {:?}",
                            &dev_state[..4.min(dev_state.len())]
                        );
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
                    imparo_model::ops::causal_conv(
                        imparo_backend::ConvForm::GatedBcx,
                        &bcx[..k * 3 * width],
                        &cw,
                        &mut ref_state,
                        &mut ref_out,
                        width,
                        kern,
                        k,
                    );
                    be.write(BufId::Recur, 0, &seed);
                    be.begin();
                    be.write(BufId::Model0, 0, &bcx);
                    be.causal_conv_snapshot(
                        imparo_backend::ConvForm::GatedBcx,
                        BufId::Model0,
                        BufId::Recur,
                        0,
                        BufId::RecurSnap,
                        0,
                        width as u32,
                        kern as u32,
                        k as u32,
                    );
                    be.causal_conv(
                        imparo_backend::ConvForm::GatedBcx,
                        BufId::Model0,
                        cw_t.offset as u64,
                        BufId::Recur,
                        0,
                        0,
                        BufId::O,
                        width as u32,
                        kern as u32,
                        n_tok as u32,
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
        //   4    the narrow token tile (the check raises the one-row GEMV boundary to 8
        //        for itself; the engine never dispatches this kernel by default)
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
            // The Q8 token tile sits above the engine's one-row GEMV boundary
            // (g_gemv_max_tok in the bridge). Raised to 8 for the Q8 tensors so the kernel
            // is exercised; the Q4 control keeps the engine's boundary; put back below.
            let saved_gemv_max = m::gemv_max_tok_current();
            let mut worst = 0.0_f64;
            for name in names {
                let Some(t) = w.get(name).copied() else {
                    continue;
                };
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
                    for shape in 0..m::st_gemm_shapes() {
                        for n_tok in [32u32, 33, 64] {
                            v.push((n_tok, 0, 0, shape, 1));
                        }
                    }
                    v
                } else {
                    vec![
                        (1, 0, 0, 0, 0),
                        (4, 0, 0, 0, 0),
                        (32, 0, 0, 0, 0),
                        (33, 0, 0, 0, 0),
                        (64, 0, 0, 0, 0),
                    ]
                };
                for (n_tok, cfa, cfb, shape, is_gemm) in combos {
                    m::set_gemv_max_tok(if wire == 2 { 8 } else { saved_gemv_max });
                    if wire == 2 {
                        if is_gemm == 1 {
                            // PINNED, not seated: the engine derives the token tile
                            // from the seat per dispatch, so seating a shape here would
                            // check whichever tile the rule widened to rather than the
                            // one this row names. A pin is exact at every width.
                            m::set_st_gemm_shape_pin(shape);
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
                    imparo_cpu::ops::mul_mat_batch(
                        &w,
                        &t,
                        &x,
                        n_tok as usize,
                        &mut want,
                    );

                    m::alloc(m::buf::X, x.len() as u64 * 4)
                        .map_err(|e| format!("alloc x {e}"))?;
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
                    } else if n_tok <= m::gemv_max_tok_current() {
                        "tile"
                    } else {
                        "gemm"
                    };
                    let cfg = if wire != 2 {
                        String::new()
                    } else if route == "gemm" {
                        format!("shape={shape}")
                    } else if route == "gemv" {
                        format!(
                            "rows={} sgs={}",
                            m::q8_decode_rows(),
                            m::q8_decode_sgs()
                        )
                    } else {
                        format!("tile={} sgs={}", m::q8_token_tile(), m::q8_batch_sgs())
                    };
                    let bad = if route == "gemm" {
                        rel >= 1e-3
                    } else {
                        rel >= 1e-5
                    };
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
            m::set_gemv_max_tok(saved_gemv_max);
            if checked == 0 {
                return Err(
                    "quant check found no quantised tensor in this model".into()
                );
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
            // The ladder starts BELOW one threadgroup per core on purpose. A dispatch whose
            // threadgroup count is fixed by the model -- the delta rule launches one per
            // value head, 48 -- cannot choose a fat grid, and "the ceiling is 136" was
            // measured at 72 and above. A narrow dispatch that is starved reads as a slow
            // kernel, so the ceiling has to be known AT THE SHAPE the kernel runs.
            for tpg in [128u32, 256, 512, 1024] {
                for tgs in
                    [18u32, 36, 48, 54, 72, 144, 288, 576, 1152, 2304, 4608, 9216]
                {
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
            // 6144 = st_gemm shape 3's allocation, 8448 = rt_gemm's gated 64x64 and
            // 12288 = st_gemm at K chunk 64: the residency points the GEMM shapes sit at.
            for smem in [2048u32, 6144, 8192, 8448, 12288, 16896, 32768] {
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
        // IMPARO_BENCH_GEMV=1 times this model's DECODE GEMV (n_tok = 1) per projection
        // shape, on the model's own weight kind -- the row-major Q8_0 file and the
        // tile-major Q8_0_TM file therefore measure their own kernels on identical shapes.
        // One dispatch per layer's tensor of a shape, several per command buffer, min and
        // mean over >= 300 ms of GPU time, after a 300 ms clock warm-up: the same
        // instrument as the attention probe, because the tuner's decode workload floor
        // (5-10%) cannot resolve the 1-3% the bracket showed and a bracket costs 7 minutes.
        // IMPARO_BENCH_GEMV_ROWS=N (2..8) times a co-batched step's N rows instead, on the fast
        // route: the kernel each format's rows take there (a decode-rows GEMV or the GEMM).
        if std::env::var("IMPARO_BENCH_GEMV").is_ok() {
            let rows: u32 = std::env::var("IMPARO_BENCH_GEMV_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|r| (1..=8).contains(r))
                .unwrap_or(1);
            if rows > 1 {
                m::set_decode_rows(Some(imparo_backend::RowRoute::Fast));
            }
            // IMPARO_BENCH_BLK_ROWS_GEMV_MAX=N and IMPARO_BENCH_BLK_ROWS_MMA_MAX=N set the block
            // formats' crossings for this run, so their decode-rows GEMV and their matrix-unit
            // rows kernel can be timed on either side of the compiled defaults.
            if let Some(n) = std::env::var("IMPARO_BENCH_BLK_ROWS_GEMV_MAX")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
            {
                m::set_blk_rows_gemv_max(n);
            }
            if let Some(n) = std::env::var("IMPARO_BENCH_BLK_ROWS_MMA_MAX")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
            {
                m::set_blk_rows_mma_max(n);
            }
            let mut groups: std::collections::BTreeMap<
                String,
                Vec<(u32, u64, u32, u32)>,
            > = std::collections::BTreeMap::new();
            let (mut max_in, mut max_out) = (0u64, 0u64);
            for (name, t) in &w.tensors {
                if t.n_dims != 2 || name == "token_embd.weight" {
                    continue;
                }
                // A block past the plan's layers (Qwen3.8-27B's MTP head, blk.64) is unread:
                // the fit gives it no segment, so a dispatch on it has no weights to bind.
                let block = name
                    .strip_prefix("blk.")
                    .and_then(|r| r.split('.').next())
                    .and_then(|b| b.parse::<usize>().ok());
                if block.is_some_and(|b| b >= plan.layers.len()) {
                    continue;
                }
                let Some(kind) = imparo_gguf::weights::weight_kind(t.ggml_type) else {
                    continue;
                };
                let wire = kind as u32;
                // Every quantized 2-D projection: Q4_0 / Q8_0 / Q8_0_TM take their own
                // GEMVs, the block quants (Q4_K, IQ4_XS, ...) the tile-major brick GEMV.
                // F32 (norm vectors are 1-D anyway) and the embedding are skipped.
                if wire == 0 {
                    continue;
                }
                // Group by ROLE AND KIND: a UD mix gives the same projection a different
                // quant in different layers, and the rate per byte is the format's.
                let suffix = format!(
                    "{}@{kind:?}",
                    name.splitn(3, '.').nth(2).unwrap_or(name.as_str())
                );
                groups.entry(suffix).or_default().push((
                    wire,
                    t.offset as u64,
                    t.ne0() as u32,
                    t.ne1() as u32,
                ));
                max_in = max_in.max(t.ne0() as u64);
                max_out = max_out.max(t.ne1() as u64);
            }
            if groups.is_empty() {
                return Err(
                    "IMPARO_BENCH_GEMV: no quantized 2-D projection in this model"
                        .into(),
                );
            }
            // IMPARO_BENCH_ROWS_SEATS=1: time BOTH rows kernels on every (kind, shape) this model
            // carries and print the tune file's seat lines. This is the measurement the tuner
            // records -- per tensor, never a step total, which is what a per-format sweep could
            // only read.
            if std::env::var("IMPARO_BENCH_ROWS_SEATS").is_ok_and(|v| v == "1") {
                let flat: Vec<(u32, u64, u32, u32)> =
                    groups.values().flatten().copied().collect();
                let seats = m::measure_blk_rows_kernels(&flat, rows)?;
                println!("-- rows kernel seats at {rows} rows: kind.n_in.n_out --");
                // TWO TOTALS, and only the second one is a reason to do anything. Summing every
                // triple where the GEMV wins prices this against a build that never had the
                // per-format table; what a seat actually BUYS is the difference from what ships,
                // and the shipped choice is the engine's own answer rather than a list retyped
                // here.
                let (mut gemv_wins, mut saved_us, mut over_shipped) =
                    (0_usize, 0.0_f64, 0.0_f64);
                for s in &seats {
                    let seat = s.seat(rows);
                    let shipped = if m::blk_rows_gemv_max_for(s.wkind) >= rows {
                        s.gemv_us
                    } else {
                        s.mma_us
                    };
                    over_shipped +=
                        (shipped - s.gemv_us.min(s.mma_us)) * f64::from(s.tensors);
                    if seat != 1 {
                        gemv_wins += 1;
                        saved_us += (s.mma_us - s.gemv_us) * f64::from(s.tensors);
                    }
                    println!(
                        "blk_rows.{}.{}.{}={seat}   x{:<3} gemv {:>8.1} us  mma {:>8.1} us  {}",
                        s.wkind,
                        s.n_in,
                        s.n_out,
                        s.tensors,
                        s.gemv_us,
                        s.mma_us,
                        if seat == 1 { "matrix unit" } else { "GEMV" }
                    );
                }
                println!(
                    "-- {} triples, the GEMV wins {gemv_wins}. Worth {:.3} ms a step OVER WHAT \
                     SHIPS (the per-format crossing); {:.3} ms against the matrix unit \
                     everywhere, which is NOT the number to act on -- it counts tensors the \
                     per-format table already routes correctly --",
                    seats.len(),
                    over_shipped / 1000.0,
                    saved_us / 1000.0
                );
                return Ok(());
            }
            // Room for a padded 64-row tile when rows > 1: the GEMM takes a step's rows only where
            // both operands hold one, as the forward's activation buffers do.
            let alloc_rows = if rows > 1 { u64::from(rows).max(64) } else { 1 };
            m::alloc(m::buf::X, alloc_rows * max_in * 4)
                .map_err(|e| format!("alloc {e}"))?;
            m::alloc(m::buf::O, alloc_rows * max_out * 4)
                .map_err(|e| format!("alloc {e}"))?;
            let xin: Vec<f32> = (0..alloc_rows * max_in)
                .map(|i| 0.01 + (i % 7) as f32 * 1e-3)
                .collect();
            m::write(m::buf::X, 0, &xin);
            let first = groups.values().next().unwrap().clone();
            let t0 = std::time::Instant::now();
            while t0.elapsed().as_millis() < 300 {
                m::begin();
                for &(wire, off, n_in, n_out) in &first {
                    m::matmat(wire, off, n_in, n_out, m::buf::X, m::buf::O, rows);
                }
                m::end().map_err(|rc| format!("gemv rc={rc}"))?;
            }
            println!("-- gemv probe: n_tok={rows}, min and mean per dispatch --");
            for (suffix, tensors) in &groups {
                let sample =
                    |reps: usize, floor_us: f64| -> Result<(f64, f64, usize), String> {
                        let (mut best, mut sum, mut n) = (f64::INFINITY, 0.0, 0usize);
                        let per_buf = reps * tensors.len();
                        while n < 5 || sum < floor_us {
                            m::begin();
                            for _ in 0..reps {
                                for &(wire, off, n_in, n_out) in tensors {
                                    m::matmat(
                                        wire,
                                        off,
                                        n_in,
                                        n_out,
                                        m::buf::X,
                                        m::buf::O,
                                        rows,
                                    );
                                }
                            }
                            m::end().map_err(|rc| format!("gemv rc={rc}"))?;
                            let us = m::last_gpu_us();
                            best = best.min(us / per_buf as f64);
                            sum += us;
                            n += 1;
                        }
                        Ok((best, sum / (n * per_buf) as f64, n * per_buf))
                    };
                let routes_before = m::matmul_routes();
                let single = sample(1, 0.0)?;
                let reps = ((4000.0 / (single.0 * tensors.len() as f64).max(1.0))
                    as usize)
                    .clamp(1, 64);
                let (best, mean, n) = sample(reps, 300_000.0)?;
                // The kernel this group's dispatches ran: the matmul routes that grew.
                let routes_after = m::matmul_routes();
                let mut kernels: Vec<&str> = routes_after
                    .iter()
                    .zip(&routes_before)
                    .filter(|((_, a), (_, b))| a > b)
                    .map(|((name, _), _)| *name)
                    .collect();
                if kernels.is_empty() {
                    kernels.push("none");
                }
                let kernel = kernels.join("+");
                let flag = if kernel.contains("_fallback") {
                    "FALLBACK "
                } else {
                    ""
                };
                let (wire, _, n_in, n_out) = tensors[0];
                // Bytes per dispatch from the format's block size, so the row carries
                // the rate the kernel sustained against the 136 GB/s ordinary-grid wall.
                let bytes = w
                    .tensors
                    .iter()
                    .find(|(nm, tt)| {
                        tt.n_dims == 2
                            && tt.offset as u64 == tensors[0].1
                            && nm.as_str() != "token_embd.weight"
                    })
                    .map_or(0.0, |(_, tt)| tt.bytes as f64);
                println!(
                    "{flag}gemv probe  {suffix:<36} kind={wire:<2} {n_in:>5}->{n_out:<6} x{:<2} min {best:>7.1} us  mean {mean:>7.1} us  {:>6.1} GB/s at min  ({n} dispatches)  kernel={kernel}",
                    tensors.len(),
                    bytes / best / 1e3
                );
            }
            return Ok(());
        }
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
