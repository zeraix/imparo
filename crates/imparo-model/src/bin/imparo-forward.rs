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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let first = args
        .next()
        .ok_or("usage: imparo-forward MODEL.gguf TOKEN...")?;
    // --repeat N: run the forward N times in ONE process, printing stats+top10 per rep.
    // The det gate uses this to sample determinism without paying the ~2s Metal init per
    // sample; it still runs multiple PROCESSES too, since the visibility race was
    // observed across fresh processes. Any cross-rep state leak shows up as a
    // determinism failure, which is exactly what the gate exists to catch.
    let mut first = first;
    let mut repeat = 1_usize;
    let mut decode_n = 0_usize;
    if first == "--repeat" {
        repeat = args.next().ok_or("--repeat needs a count")?.parse()?;
        first = args.next().ok_or("--repeat N MODEL.gguf TOKEN...")?;
    }
    // --split K: forward tokens[..K] at 0, then tokens[K..] at K -- the KV-pool
    // continuation instrument. Byte-equal top10 vs the unsplit run proves a resumed
    // tail reproduces the cold prefill's logits exactly.
    let mut split: Option<usize> = None;
    // --decode N: after the prefill, run N greedy single-token steps and print the
    // token trail plus an FNV hash of every step's logits. With --repeat this is
    // the engine-level decode-determinism gate (task #26's reproducer).
    if first == "--decode" {
        decode_n = args.next().ok_or("--decode needs a count")?.parse()?;
        first = args.next().ok_or("--decode N MODEL.gguf TOKEN...")?;
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

    let t0 = std::time::Instant::now();
    let document = imparo_gguf::read(&path)?;
    let plan = build_plan(&document, &path)?;
    let mut weights = Weights::open(&path)?;
    imparo_model::backend::enable_gpu(&mut weights, &plan)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "load  ms={load_ms:.1} tensors={} layers={} vocab={}",
        weights.tensors.len(),
        plan.config.n_layers,
        plan.config.vocab_size
    );

    // IMPARO_KV_CAP: force the KV capacity (test instrument). The server sizes
    // capacity from -c, not the prompt; this reproduces that shape here.
    let capacity = std::env::var("IMPARO_KV_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| tokens.len().max(8));
    let mut model = imparo_model::load(weights, plan, capacity)?;

    let mut logits = Vec::new();
    for rep in 0..repeat {
        let toks_rep = tokens.clone();
        let t1 = std::time::Instant::now();
        if let Some(dir) = &restore_dir {
            // Restart-durability path: nothing of this conversation has run in this
            // process. The request's tokens are the whole key.
            let digest = imparo_model::kv::model_digest(&path)?;
            let root = imparo_model::kv::config_root(model.plan(), &digest);
            let hashes = imparo_kv::unit_hashes(&root, &toks_rep);
            let store = imparo_kv::Store::open(dir, &root).map_err(|e| e.clone())?;
            let (mpath, manifest) = store
                .best_match(&hashes)
                .ok_or("restore: no committed prefix matches these tokens")?;
            let units: Vec<Vec<u8>> = manifest
                .hashes
                .iter()
                .map(|h| store.get_unit(h).ok_or("restore: unit missing"))
                .collect::<Result<_, _>>()?;
            let ck = store
                .get_checkpoint(&manifest.ckpts.last().ok_or("restore: no checkpoint")?.blob)
                .or_else(|| store.checkpoint_for(&mpath))
                .ok_or("restore: no checkpoint")?;
            let state = imparo_kv::state::state_from_blobs(&units, &ck)?;
            let boundary = state.boundary;
            model.kv_restore(&state)?;
            println!("restored boundary={boundary} units={}", units.len());
            logits = model.forward(&toks_rep[boundary..], boundary)?;
        } else {
            logits = match split {
                Some(k) if k > 0 && k < toks_rep.len() => {
                    let _ = model.forward(&toks_rep[..k], 0)?;
                    model.forward(&toks_rep[k..], k)?
                }
                // --spill: forward to the BOUNDARY, checkpoint there, then forward
                // the tail. A recurrent state is one buffer holding "now", so a
                // checkpoint is only valid at the position the device is at -- which is
                // why a server pauses at its branch point rather than spilling after the
                // fact. Position-indexed KV would not care; this makes both correct.
                _ if spill_dir.is_some() => {
                    let b = toks_rep.len() / imparo_kv::UNIT_TOKENS
                        * imparo_kv::UNIT_TOKENS;
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
                _ => model.forward(&toks_rep, 0)?,
            };
        }
        imparo_model::host::prof_log("prefill", t1.elapsed().as_secs_f64() * 1e3);
        if scramble && rep == 0 {
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
            let mut next = argmax(&logits);
            let mut trail: Vec<u32> = vec![next];
            let mut lg = Vec::new();
            let mut th = FNV_OFFSET;
            let t_dec = std::time::Instant::now();
            let mut step = vec![next; dbatch];
            for i in 0..decode_n {
                // Refilled every step: the greedy trail is what the determinism gate
                // reads, and a `step` built once would forward the FIRST token forever
                // while still printing a plausible hash per step.
                step.fill(next);
                model.forward_into(&step, base + i * dbatch, &mut lg)?;
                let h = fnv(&mut lg.iter().flat_map(|v| v.to_le_bytes()));
                if dbatch == 1 {
                    println!("dstep rep={rep} i={i} h={h:016x}");
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
        if rep + 1 < repeat {
            print_top10(&logits);
        }
    }

    let finite = logits.iter().filter(|v| v.is_finite()).count();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    // NO TIMING IS PRINTED, deliberately. Nothing parses one -- not the gates, not
    // bracket.py, which is the only sanctioned speed comparison and drives the server.
    // The `fwd ms=` line that used to be here existed solely to be grepped by ad-hoc
    // benchmarking, and a full engine forward is the wrong instrument for kernel speed:
    // it is ~32 s at 16k, pins the GPU, and answers questions a seconds-long micro-bench
    // answers better. Speed work goes through the tuner's micro-benches (#39, #44).
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
    let root = imparo_model::kv::config_root(model.plan(), &digest);
    let hashes = imparo_kv::unit_hashes(&root, &tokens[..state.boundary]);
    let store = imparo_kv::Store::open(dir, &root).map_err(|e| e.clone())?;
    let blobs = imparo_kv::state::unit_blobs(&state);
    for (h, b) in hashes.iter().zip(&blobs) {
        store.put_unit(h, b)?;
    }
    let addr = store.put_checkpoint(&imparo_kv::state::checkpoint_blob(&state))?;
    store.commit(
        "cli",
        &imparo_kv::Manifest {
            boundary: state.boundary as u64,
            hashes,
            tail: Vec::new(),
            keyless: false,
            ckpts: vec![imparo_kv::store::Ckpt {
                boundary: state.boundary as u64,
                from: 0,
                blob: addr,
                tail: Vec::new(),
            }],
        },
    )?;
    println!("spilled boundary={} units={}", state.boundary, blobs.len());
    Ok(())
}
