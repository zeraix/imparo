//! Dev tool: run the CPU reference forward pass over explicit token ids and print top logits.
//!
//! Token ids rather than text so this milestone does not depend on the tokenizer. The output
//! is meant to be compared against llama.cpp for the same ids.

use std::path::PathBuf;

use imparo_model::build_plan;
// No `use` for Model or KvPoolMember: the tool holds a Box<dyn Model>, and a trait
// object dispatches its own methods -- including the supertrait's -- without them.
use imparo_model::weights::Weights;

fn top10_string(logits: &[f32], out: &mut String) {
    use std::fmt::Write as _;
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in idx.iter().take(10) {
        let _ = write!(out, " {i}:{:.6}", logits[i]);
    }
}

fn print_top10(logits: &[f32]) {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    print!("top10");
    for &i in idx.iter().take(10) {
        print!(" {i}:{:.6}", logits[i]);
    }
    println!();
}

/// Test-only full-distribution witness. The environment variable is deliberately
/// opt-in so normal CLI output and hot-path behavior do not change. The file is a
/// headerless sequence of IEEE-754 f32 values in vocabulary order and little-endian
/// byte order, which lets a cross-process gate compare the actual logits byte-for-byte.
fn dump_logits_to_path(
    logits: &[f32],
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity(std::mem::size_of_val(logits));
    for value in logits {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, raw)
}

fn dump_logits_for_env(logits: &[f32], variable: &str) -> std::io::Result<()> {
    let Some(path) = std::env::var_os(variable) else {
        return Ok(());
    };
    dump_logits_to_path(logits, path)
}

fn dump_logits_if_requested(logits: &[f32]) -> std::io::Result<()> {
    dump_logits_for_env(logits, "IMPARO_LOGITS_DUMP")
}

fn phase_a1_prefill_wall_line(
    enabled: bool,
    rep: usize,
    token_count: usize,
    start_pos: usize,
    split_mode: &str,
    split_at: Option<usize>,
    split_parts: Option<&[usize]>,
    elapsed: std::time::Duration,
) -> Result<Option<String>, serde_json::Error> {
    if !enabled {
        return Ok(None);
    }
    serde_json::to_string(&serde_json::json!({
        "schema": 1,
        "phase": "A1",
        "event": "prefill_wall",
        "lab_only": true,
        "production_authority": false,
        "rep": rep,
        "token_count": token_count,
        "start_pos": start_pos,
        "split_state": {
            "mode": split_mode,
            "split_at": split_at,
            "parts": split_parts,
        },
        "prefill_wall_ms": elapsed.as_secs_f64() * 1e3,
    }))
    .map(Some)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .ok_or("usage: imparo-forward MODEL.gguf TOKEN...")?;
    if first == "--correctness-template" {
        let _config = PathBuf::from(
            args.next()
                .ok_or("--correctness-template needs CONFIG and MODEL")?,
        );
        let _model_path = PathBuf::from(
            args.next()
                .ok_or("--correctness-template needs CONFIG and MODEL")?,
        );
        let kv = if let Some(flag) = args.next() {
            if flag != "--kv" {
                return Err("--correctness-template optional argument is --kv".into());
            }
            let value = args
                .next()
                .ok_or("--correctness-template --kv needs a value")?;
            if args.next().is_some() {
                return Err(
                    "--correctness-template accepts only CONFIG MODEL [--kv TYPE]"
                        .into(),
                );
            }
            value
        } else {
            "q4_0".to_string()
        };
        if kv != "q4_0" && kv != "q8_0" {
            return Err("--correctness-template --kv supports q4_0 or q8_0".into());
        }
        #[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
        {
            unsafe { std::env::set_var("IMPARO_HOST_CONFIG", &_config) };
            let document = imparo_gguf::read(&_model_path)?;
            let plan = build_plan(&document, &_model_path)?;
            let weights = Weights::open_with(&document, &_model_path)?;
            let kv_layout_sha256 =
                imparo_model::kv::effective_kv_byte_layout_profile(&plan, &kv, &kv)?
                    .sha256_identity();
            let receipt = imparo_cuda::correctness_receipt_template(
                &_config,
                weights.byte_len(),
                weights.full_file_sha256(),
                *plan.sha256_identity().as_bytes(),
                kv_layout_sha256,
                &kv,
                &kv,
            )?;
            println!("{}", serde_json::to_string(&receipt)?);
            return Ok(());
        }
        #[cfg(not(any(feature = "cuda", feature = "cuda-dynamic")))]
        return Err("--correctness-template requires a CUDA-enabled build".into());
    }
    // --repeat N: run the forward N times in ONE process, printing stats+top10 per rep.
    // The det gate uses this to sample determinism without paying the ~2s Metal init per
    // sample; it still runs multiple PROCESSES too, since the visibility race was
    // observed across fresh processes. Any cross-rep state leak shows up as a
    // determinism failure, which is exactly what the gate exists to catch.
    let mut first = first;
    let mut repeat = 1_usize;
    let mut decode_n = 0_usize;
    let mut graph_decode_n = 0_usize;
    let mut graph_decode_paged = false;
    if first == "--repeat" {
        repeat = args.next().ok_or("--repeat needs a count")?.parse()?;
        first = args.next().ok_or("--repeat N MODEL.gguf TOKEN...")?;
    }
    // Dev-only same-shape Graph witness. Each repetition replaces the final token
    // with BASE+rep while retaining one model/backend process. Combined with
    // IMPARO_LOGITS_DUMP_DIR this proves that replay consumes fresh request data,
    // not merely that one repeated prompt is deterministic.
    let mut vary_last: Option<u32> = None;
    if first == "--vary-last" {
        vary_last = Some(
            args.next()
                .ok_or("--vary-last needs a base token")?
                .parse()?,
        );
        first = args.next().ok_or("--vary-last BASE MODEL.gguf TOKEN...")?;
    }
    // --split K: forward tokens[..K] at 0, then tokens[K..] at K -- the KV-pool
    // continuation instrument. The harness compares the opt-in raw final-logit dump;
    // top10 remains diagnostic output only.
    let mut split: Option<usize> = None;
    // --decode N: after the prefill, run N greedy single-token steps and print the
    // token trail plus an FNV hash of every step's logits. With --repeat this is
    // the engine-level decode-determinism gate (task #26's reproducer).
    if first == "--decode" {
        decode_n = args.next().ok_or("--decode needs a count")?.parse()?;
        first = args.next().ok_or("--decode N MODEL.gguf TOKEN...")?;
    }
    // --decode-pipe N: the same N greedy steps through the PIPELINED decode API
    // (queue_step / wait_step: step k+1 is encoded and committed before step k's pick is
    // read; docs/decode-turnaround.md). The host never sees the logits, so the output is
    // the token trail and the wall time per step; the trail must equal --decode's.
    let mut decode_pipe = false;
    if first == "--decode-pipe" {
        decode_n = args.next().ok_or("--decode-pipe needs a count")?.parse()?;
        decode_pipe = true;
        first = args.next().ok_or("--decode-pipe N MODEL.gguf TOKEN...")?;
    }
    // Real CUDA-Graph probe: unlike --decode, this uses `forward_next` so device
    // argmax is active and the native backend may capture/replay. The paged form
    // installs a non-identity table before prefill, so captured attention nodes use
    // the dedicated paged symbols rather than merely validating identity graphs.
    if first == "--graph-decode" || first == "--paged-graph-decode" {
        graph_decode_paged = first == "--paged-graph-decode";
        graph_decode_n = args.next().ok_or("--graph-decode needs a count")?.parse()?;
        first = args.next().ok_or("--graph-decode N MODEL.gguf TOKEN...")?;
    }
    // --dbatch B: feed B rows per decode forward instead of 1. NOT a correctness path
    // -- the rows carry the same token at consecutive positions, so the logits are
    // meaningless -- it exists to price the SHAPE. A co-batched decode (two
    // conversations in one forward) reads the weights once for B rows, and what that
    // is worth is exactly `B x t(1) - t(B)`.
    let mut dbatch = 1_usize;
    if first == "--dbatch" {
        dbatch = args.next().ok_or("--dbatch needs a count")?.parse()?;
        first = args.next().ok_or("--dbatch B MODEL.gguf TOKEN...")?;
    }
    if first == "--split" {
        split = Some(args.next().ok_or("--split needs a position")?.parse()?);
        first = args.next().ok_or("--split K MODEL.gguf TOKEN...")?;
    }
    // --spill DIR / --restore DIR: the disk-tier instrument. --spill runs the full
    // forward, captures the KV state at the unit boundary and commits it to the
    // store; --restore (a FRESH process: the durability proof) probes the store
    // with the request's own hashes, restores, and forwards only the tail.
    let mut spill_dir: Option<PathBuf> = None;
    let mut restore_dir: Option<PathBuf> = None;
    if first == "--spill" {
        spill_dir = Some(PathBuf::from(args.next().ok_or("--spill needs a dir")?));
        first = args.next().ok_or("--spill DIR MODEL.gguf TOKEN...")?;
    }
    if first == "--restore" {
        restore_dir = Some(PathBuf::from(args.next().ok_or("--restore needs a dir")?));
        first = args.next().ok_or("--restore DIR MODEL.gguf TOKEN...")?;
    }
    // --scramble-kv: forward twice in one process -- identity placement, then
    // pair-swapped block tables -- and compare top10 internally. Proves physical
    // KV placement is invisible to the output (the isolation invariant).
    let mut scramble = false;
    if first == "--scramble-kv" {
        scramble = true;
        first = args.next().ok_or("--scramble-kv MODEL.gguf TOKEN...")?;
    }
    // Dev-only exact-128 mask sweep: one model load, one mask per repeat.
    // This accelerates numerical-route search without changing backend ABI or
    // production policy; ordinary runs never set the environment variable.
    let exact128_mask_sweep: Option<Vec<String>> =
        std::env::var("IMPARO_EXACT128_MASK_SWEEP")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(str::to_owned)
                    .collect()
            });
    if let Some(masks) = &exact128_mask_sweep {
        if masks.is_empty() {
            return Err("IMPARO_EXACT128_MASK_SWEEP contains no masks".into());
        }
        repeat = masks.len();
    }
    let path = PathBuf::from(first);
    // Token ids as args, or -t FILE (whitespace-separated) -- long contexts
    // exceed the OS argv limit around 90k tokens.
    let mut rest: Vec<String> = args.collect();
    let tokens: Vec<u32> = if rest.first().map(String::as_str) == Some("-t") {
        let tf = rest.get(1).ok_or("-t needs a file")?;
        std::fs::read_to_string(tf)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?
    } else {
        rest.drain(..)
            .map(|a| a.parse::<u32>())
            .collect::<Result<_, _>>()?
    };
    if tokens.is_empty() {
        return Err("give at least one token id".into());
    }
    // Phase-A1-only denominator instrument. Exact value "1" is required; when
    // absent (the normal CLI), this changes no output, route, math, or model state.
    let phase_a1_prefill_wall =
        std::env::var("IMPARO_CUDA_PHASE_A1_PREFILL_WALL").as_deref() == Ok("1");

    let t0 = std::time::Instant::now();
    let document = imparo_gguf::read(&path)?;
    let plan = build_plan(&document, &path)?;
    let mut weights = Weights::open(&path)?;
    // IMPARO_KV_CAP: force the KV capacity (test instrument). The server sizes
    // capacity from -c, not the prompt; this reproduces that shape here. Known before
    // the GPU is enabled because the weight placement reserves the KV for it.
    let capacity = std::env::var("IMPARO_KV_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| tokens.len().max(8));
    imparo_model::backend::enable_gpu(
        &mut weights,
        &plan,
        capacity,
        imparo_model::prefill_batch(),
    )?;
    let tensor_count = weights.tensors.len();
    let layer_count = plan.config.n_layers;
    let vocab_size = plan.config.vocab_size;

    let mut model = imparo_model::load(weights, plan, capacity)?;
    // BOTH questions, because they are different ones: a backend is active whenever one
    // is compiled in (and IMPARO_BACKEND=cpu makes the CPU backend the active one), while
    // `has_device_workflow` says whether THIS architecture has a device forward to make
    // ready. Asking only the first sent a CPU-only architecture into a refusal.
    if model.has_device_workflow() && imparo_model::backend::active().is_some() {
        model.ensure_gpu_ready()?;
    }
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "load  ms={load_ms:.1} tensors={tensor_count} layers={layer_count} vocab={vocab_size}"
    );
    if graph_decode_paged {
        model.kv_prepare_pool()?;
        model.kv_set_scrambled_tables();
    }

    let logits_dump_dir = std::env::var_os("IMPARO_LOGITS_DUMP_DIR").map(PathBuf::from);
    if let Some(dir) = &logits_dump_dir {
        std::fs::create_dir_all(dir)?;
    }
    let mut logits = Vec::new();
    for rep in 0..repeat {
        let mut toks_rep = tokens.clone();
        if let Some(base) = vary_last {
            let replacement = base
                .checked_add(u32::try_from(rep)?)
                .ok_or("--vary-last token overflow")?;
            *toks_rep.last_mut().expect("non-empty tokens checked above") = replacement;
        }
        let t1 = std::time::Instant::now();
        let mut wall_token_count = toks_rep.len();
        let mut wall_start_pos = 0_usize;
        let mut wall_split_mode = "single";
        let mut wall_split_at = None;
        let mut wall_split_parts: Option<Vec<usize>> = None;
        if let Some(dir) = &restore_dir {
            // Restart-durability path: nothing of this conversation has run in this
            // process. The request's tokens are the whole key.
            let digest = imparo_model::kv::model_digest(&path)?;
            let root = imparo_model::kv::config_root(
                model.plan(),
                &digest,
                &imparo_model::backend::active().map_or_else(
                    || "none".to_string(),
                    imparo_backend::Backend::device_tag,
                ),
            );
            let store = imparo_kv::Store::open(dir, &root).map_err(|e| e.clone())?;
            let state =
                store.read_whole(&root, &model.kv_state_geometry(), &toks_rep)?;
            let boundary = state.boundary;
            model.kv_restore(&state)?;
            println!("restored boundary={boundary}");
            wall_token_count = toks_rep.len() - boundary;
            wall_start_pos = boundary;
            wall_split_mode = "restore-tail";
            wall_split_at = Some(boundary);
            if phase_a1_prefill_wall {
                wall_split_parts = Some(vec![wall_token_count]);
            }
            logits = model.forward(&toks_rep[boundary..], boundary)?;
        } else {
            logits = match split {
                Some(k) if k > 0 && k < toks_rep.len() => {
                    wall_split_mode = "explicit-split";
                    wall_split_at = Some(k);
                    if phase_a1_prefill_wall {
                        wall_split_parts = Some(vec![k, toks_rep.len() - k]);
                    }
                    let _ = model.forward(&toks_rep[..k], 0)?;
                    model.forward(&toks_rep[k..], k)?
                }
                // --spill: forward to the BOUNDARY, checkpoint there, then forward
                // the tail. A recurrent state is one buffer holding "now", so a
                // checkpoint is only valid at the position the device is at -- which is
                // why a server pauses at its branch point rather than spilling after the
                // fact. Position-indexed KV would not care; this makes both correct.
                _ if spill_dir.is_some() => {
                    let b = toks_rep.len() / imparo_kv::grid_tokens()
                        * imparo_kv::grid_tokens();
                    wall_split_mode = "spill-boundary";
                    wall_split_at = Some(b);
                    if phase_a1_prefill_wall {
                        wall_split_parts = Some(if b == toks_rep.len() {
                            vec![b]
                        } else {
                            vec![b, toks_rep.len() - b]
                        });
                    }
                    if b > 0 {
                        model.forward(&toks_rep[..b], 0)?;
                    }
                    spill_now(
                        model.as_mut(),
                        spill_dir.as_ref().expect("checked"),
                        &path,
                        &toks_rep,
                    )?;
                    model.forward(&toks_rep[b..], b)?
                }
                _ => {
                    let splits = std::env::var("IMPARO_PREFILL_SPLITS")
                        .ok()
                        .map(|raw| {
                            raw.split(',')
                                .map(str::parse::<usize>)
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()?;
                    let chunk = std::env::var("IMPARO_PREFILL_CHUNK")
                        .ok()
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|&value| value > 0);
                    if let Some(splits) = splits {
                        if splits.iter().sum::<usize>() != toks_rep.len()
                            || splits.contains(&0)
                        {
                            return Err("IMPARO_PREFILL_SPLITS must be positive and sum to the prompt length".into());
                        }
                        wall_split_mode = "env-splits";
                        if phase_a1_prefill_wall {
                            wall_split_parts = Some(splits.clone());
                        }
                        model.ensure_gpu_ready()?;
                        model.kv_fit(toks_rep.len())?;
                        let mut last = Vec::new();
                        let mut start = 0;
                        for count in splits {
                            last = model
                                .forward(&toks_rep[start..start + count], start)?;
                            start += count;
                        }
                        last
                    } else if let Some(chunk) = chunk {
                        wall_split_mode = "env-chunk";
                        if phase_a1_prefill_wall {
                            wall_split_parts = Some(
                                toks_rep.chunks(chunk).map(<[u32]>::len).collect(),
                            );
                        }
                        let mut last = Vec::new();
                        for (index, part) in toks_rep.chunks(chunk).enumerate() {
                            last = model.forward(part, index * chunk)?;
                        }
                        last
                    } else {
                        model.forward(&toks_rep, 0)?
                    }
                }
            };
        }
        let prefill_wall = t1.elapsed();
        imparo_model::host::prof_log("prefill", prefill_wall.as_secs_f64() * 1e3);
        if let Some(line) = phase_a1_prefill_wall_line(
            phase_a1_prefill_wall,
            rep,
            wall_token_count,
            wall_start_pos,
            wall_split_mode,
            wall_split_at,
            wall_split_parts.as_deref(),
            prefill_wall,
        )? {
            println!("{line}");
        }
        if scramble && rep == 0 {
            // Both legs run in this process, so each needs its own machine-readable
            // full-distribution witness for the fail-closed KV placement gate.
            dump_logits_for_env(&logits, "IMPARO_SCRAMBLE_IDENTITY_DUMP")?;

            let mut a = String::new();
            top10_string(&logits, &mut a);
            if std::env::var("IMPARO_SCRAMBLE_DUMP").is_ok() {
                let geom = model
                    .kv_state_geometry()
                    .into_iter()
                    .find(|g| g.layer == 5)
                    .expect("layer 5 geometry");
                for pos in [0usize, 63, 64, 127] {
                    let mut row = vec![0u8; geom.k_stride];
                    imparo_model::backend::active().unwrap().read_kv_bytes(
                        5,
                        false,
                        (pos * geom.k_stride) as u64,
                        &mut row,
                    );
                    let head: Vec<String> =
                        row[..8].iter().map(|b| format!("{b:02x}")).collect();
                    println!("k5-identity pos={pos} head={}", head.join(""));
                }
            }
            model.kv_set_scrambled_tables();
            // With --split K, the scrambled run RESUMES at K instead of running cold:
            // paged placement AND a resume, which is the cell neither half covered.
            // Identity+cold vs identity+resume is the split gate; identity+cold vs
            // paged+cold is this gate's own row; paged+RESUME is where a quantized
            // cache was found to diverge.
            let l2 = if let Some(k) = split {
                model.forward(&toks_rep[..k], 0)?;
                model.forward(&toks_rep[k..], k)?
            } else {
                model.forward(&toks_rep, 0)?
            };
            let mut b = String::new();
            top10_string(&l2, &mut b);
            dump_logits_for_env(&l2, "IMPARO_SCRAMBLE_PAGED_DUMP")?;
            println!("identity {a}");
            println!("scrambled{b}");
            println!(
                "scramble-kv: {}",
                if a == b { "BYTE-EQUAL" } else { "MISMATCH" }
            );
            // Store-side verification: position p's K row must sit at the
            // pair-swapped physical slot. Layer 5 is the first full-attention layer.
            if std::env::var("IMPARO_SCRAMBLE_DUMP").is_ok() {
                let geom = model
                    .kv_state_geometry()
                    .into_iter()
                    .find(|g| g.layer == 5)
                    .expect("layer 5 geometry");
                for pos in [0usize, 63, 64, 127] {
                    let phys = ((pos / 64) ^ 1) * 64 + pos % 64; // pair-swap map
                    let mut row = vec![0u8; geom.k_stride];
                    imparo_model::backend::active().unwrap().read_kv_bytes(
                        5,
                        false,
                        (phys * geom.k_stride) as u64,
                        &mut row,
                    );
                    let head: Vec<String> =
                        row[..8].iter().map(|b| format!("{b:02x}")).collect();
                    println!("k5 pos={pos} phys={phys} head={}", head.join(""));
                }
            }
        }
        if graph_decode_n > 0 {
            let mut next = logits
                .iter()
                .enumerate()
                .max_by(|left, right| {
                    left.1
                        .partial_cmp(right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index as u32)
                .ok_or("graph decode prefill produced no logits")?;
            let base = toks_rep.len();
            let mut trail = vec![next];
            let mut step_us = Vec::with_capacity(graph_decode_n);
            for step in 0..graph_decode_n {
                let step_start = std::time::Instant::now();
                next = model.forward_next(next, base + step)?;
                step_us.push(step_start.elapsed().as_secs_f64() * 1e6);
                trail.push(next);
                println!(
                    "graph-dstep placement={} i={step} token={next}",
                    if graph_decode_paged {
                        "paged"
                    } else {
                        "identity"
                    }
                );
            }
            // Step 0 warms the route and step 1 captures the graph. Report only the
            // steady replay suffix so model load and graph construction cannot be
            // mistaken for per-token Decode latency.
            let warmup_steps = step_us.len().min(2);
            let mut steady = step_us[warmup_steps..].to_vec();
            steady.sort_by(f64::total_cmp);
            if !steady.is_empty() {
                let median_us = if steady.len() % 2 == 0 {
                    (steady[steady.len() / 2 - 1] + steady[steady.len() / 2]) * 0.5
                } else {
                    steady[steady.len() / 2]
                };
                let mean_us = steady.iter().sum::<f64>() / steady.len() as f64;
                println!(
                    concat!(
                        "graph-decode-timing warmup_steps={} samples={} ",
                        "median_us={:.3} mean_us={:.3} min_us={:.3} max_us={:.3}"
                    ),
                    warmup_steps,
                    steady.len(),
                    median_us,
                    mean_us,
                    steady[0],
                    steady[steady.len() - 1]
                );
            }
            println!(
                "graph-decode placement={} steps={graph_decode_n} trail={trail:?}",
                if graph_decode_paged {
                    "paged"
                } else {
                    "identity"
                }
            );
        }
        if decode_n > 0 {
            // FNV-1a 64-bit: standard offset basis and prime. Order-sensitive byte
            // fingerprint -- two runs must produce identical hashes at every step.
            const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
            const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
            let fnv = |bytes: &mut dyn Iterator<Item = u8>| {
                bytes.fold(FNV_OFFSET, |h, b| {
                    (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
                })
            };
            let argmax = |v: &[f32]| {
                let mut best = 0usize;
                for (i, &x) in v.iter().enumerate() {
                    if x > v[best] {
                        best = i;
                    }
                }
                best as u32
            };
            let base = toks_rep.len();
            let print_decode_top10 = std::env::var("IMPARO_DECODE_TOP10").is_ok();
            if print_decode_top10 {
                let mut row = String::new();
                top10_string(&logits, &mut row);
                println!("decode-top10{row}");
            }
            let mut next = argmax(&logits);
            let mut trail: Vec<u32> = vec![next];
            let mut lg = Vec::new();
            let mut th = FNV_OFFSET;
            let t_dec = std::time::Instant::now();
            if decode_pipe {
                if !model.decode_pipelined() {
                    return Err(
                        "--decode-pipe: decode is not pipelined on this backend/model"
                            .into(),
                    );
                }
                model.queue_step(Some(next), base)?;
                for i in 0..decode_n {
                    if i + 1 < decode_n {
                        model.queue_step(None, base + i + 1)?;
                    }
                    next = model.wait_step()?;
                    trail.push(next);
                }
                let dt_ms = t_dec.elapsed().as_secs_f64() * 1e3;
                println!(
                    "decode-pipe rep={rep} steps={decode_n} ms_per_forward={:.3} pos0={base}",
                    dt_ms / decode_n as f64
                );
                println!("decode-pipe rep={rep} steps={decode_n} trail={trail:?}");
                continue;
            }
            let mut step = vec![next; dbatch];
            for i in 0..decode_n {
                // Refilled every step: the greedy trail is what the determinism gate
                // reads, and a `step` built once would forward the FIRST token forever
                // while still printing a plausible hash per step.
                step.fill(next);
                model.forward_into(&step, base + i * dbatch, &mut lg)?;
                if std::env::var("IMPARO_KV_PROBE_STEP")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .is_some_and(|wanted| wanted == i)
                {
                    let layer = std::env::var("IMPARO_KV_PROBE_LAYER")
                        .ok()
                        .and_then(|value| value.parse::<u32>().ok())
                        .unwrap_or(0);
                    let geom = model
                        .kv_state_geometry()
                        .into_iter()
                        .find(|geom| geom.layer == layer)
                        .ok_or_else(|| {
                            format!("missing KV geometry for layer {layer}")
                        })?;
                    let slots = match geom.kind {
                        imparo_kv::StateKind::Window { ring, .. } => ring,
                        imparo_kv::StateKind::Full => base + (i + 1) * dbatch,
                    };
                    let is_v = std::env::var("IMPARO_KV_PROBE_SIDE")
                        .is_ok_and(|side| side.eq_ignore_ascii_case("v"));
                    let stride = if is_v { geom.v_stride } else { geom.k_stride };
                    let mut bytes = vec![0_u8; slots * stride];
                    imparo_model::backend::active()
                        .unwrap()
                        .read_kv_bytes(layer, is_v, 0, &mut bytes);
                    let path = std::env::var("IMPARO_KV_PROBE_DUMP")
                        .map_err(|_| "IMPARO_KV_PROBE_DUMP is required")?;
                    std::fs::write(&path, bytes)?;
                    eprintln!(
                        "[gpu] KV probe layer={layer} V={is_v} step={i} path={path}"
                    );
                }
                if print_decode_top10 {
                    let mut row = String::new();
                    top10_string(&lg, &mut row);
                    println!("decode-top10{row}");
                }
                let h = fnv(&mut lg.iter().flat_map(|v| v.to_le_bytes()));
                if dbatch == 1 {
                    println!("dstep rep={rep} i={i} h={h:016x}");
                }
                // IMPARO_DECODE_DUMP_DIR: the decode step's logits as raw f32 LE, one file per
                // step, for a bit-level A/B between two routes (the prefill dump above cannot
                // see a decode-only route).
                if let Some(dir) = std::env::var_os("IMPARO_DECODE_DUMP_DIR") {
                    let dir = PathBuf::from(dir);
                    std::fs::create_dir_all(&dir)?;
                    dump_logits_to_path(
                        &lg,
                        dir.join(format!("dstep-{rep:03}-{i:03}.raw")),
                    )?;
                }
                th = fnv(&mut h.to_le_bytes().into_iter().chain(th.to_le_bytes()));
                next = argmax(&lg);
                trail.push(next);
            }
            let dt_ms = t_dec.elapsed().as_secs_f64() * 1e3;
            // Plain wall clock, always printed: IMPARO_PROF=1 costs three orders of
            // magnitude here (see host::prof_log), so it can never answer "what does
            // this shape cost".
            println!(
                "decode rep={rep} steps={decode_n} rows={dbatch} \
                 ms_per_forward={:.3} pos0={base}",
                dt_ms / decode_n as f64
            );
            if dbatch == 1 {
                println!(
                    "decode rep={rep} steps={decode_n} stephash={th:016x} trail={trail:?}"
                );
            }
            imparo_model::host::prof_log("decode", dt_ms);
        }
        if let Some(dir) = &logits_dump_dir {
            dump_logits_to_path(&logits, dir.join(format!("rep-{rep:03}.raw")))?;
        }
        if rep + 1 < repeat {
            print_top10(&logits);
        }
    }

    dump_logits_if_requested(&logits)?;
    let finite = logits.iter().filter(|v| v.is_finite()).count();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    // NO TIMING IS PRINTED by default, deliberately. Nothing parses one -- not the gates, not
    // bracket.py, which is the only sanctioned speed comparison and drives the server.
    // The `fwd ms=` line that used to be here existed solely to be grepped by ad-hoc
    // benchmarking, and a full engine forward is the wrong instrument for kernel speed:
    // it is ~32 s at 16k, pins the GPU, and answers questions a seconds-long micro-bench
    // answers better. Speed work goes through the tuner's micro-benches (#39, #44).
    // The exact-value Phase-A1 lab opt-in above is the sole exception and remains
    // explicitly non-production-authoritative.
    println!("logits={} finite={finite}", logits.len());
    println!("stats min={min:.4} max={max:.4} mean={mean:.4}");

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    print!("top10");
    // Six decimals, not four: this output is diffed against llama.cpp's logprobs, and at
    // four the print resolution was the same order as the difference being measured.
    for &i in idx.iter().take(10) {
        print!(" {i}:{:.6}", logits[i]);
    }
    println!();

    if finite != logits.len() {
        return Err("non-finite logits".into());
    }
    Ok(())
}

/// Writes the current state to the store, AT the position the device is at.
///
/// Split out of `main` because the order matters: the caller forwards to the boundary,
/// calls this, and only then forwards the tail. A recurrent state is one buffer holding
/// "now", so a checkpoint taken after the whole forward has absorbed the tail -- and
/// reprocessing that tail on restore would apply it twice.
fn spill_now(
    model: &mut dyn imparo_model::Model,
    dir: &std::path::Path,
    path: &std::path::Path,
    tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let state = model.kv_spill().ok_or("spill: nothing to spill")?;
    let digest = imparo_model::kv::model_digest(path)?;
    let root = imparo_model::kv::config_root(
        model.plan(),
        &digest,
        &imparo_model::backend::active()
            .map_or_else(|| "none".to_string(), imparo_backend::Backend::device_tag),
    );
    // One call: the extents, the anchor link and the manifest come from the store,
    // which is also what the server's non-pool commit uses. This was a copy of that.
    let store = imparo_kv::Store::open(dir, &root).map_err(|e| e.clone())?;
    let staged = store.stage_whole(&root, tokens, &state)?;
    store.commit("cli", &staged.manifest(false), &staged.at)?;
    Ok(())
}

#[cfg(test)]
mod phase_a1_prefill_wall_tests {
    use super::phase_a1_prefill_wall_line;

    #[test]
    fn absent_opt_in_produces_no_line() {
        let line = phase_a1_prefill_wall_line(
            false,
            0,
            512,
            0,
            "single",
            None,
            None,
            std::time::Duration::from_millis(7),
        )
        .unwrap();
        assert!(line.is_none());
    }

    #[test]
    fn opt_in_line_is_strict_lab_only_json() {
        let line = phase_a1_prefill_wall_line(
            true,
            3,
            384,
            128,
            "restore-tail",
            Some(128),
            Some(&[384]),
            std::time::Duration::from_micros(12_345),
        )
        .unwrap()
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["schema"], 1);
        assert_eq!(value["phase"], "A1");
        assert_eq!(value["event"], "prefill_wall");
        assert_eq!(value["lab_only"], true);
        assert_eq!(value["production_authority"], false);
        assert_eq!(value["rep"], 3);
        assert_eq!(value["token_count"], 384);
        assert_eq!(value["start_pos"], 128);
        assert_eq!(value["split_state"]["mode"], "restore-tail");
        assert_eq!(value["split_state"]["split_at"], 128);
        assert_eq!(value["split_state"]["parts"], serde_json::json!([384]));
        assert_eq!(value["prefill_wall_ms"], 12.345);
        assert!(!line.contains('\n'));
    }
}
