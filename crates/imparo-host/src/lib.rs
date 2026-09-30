//! Backend-agnostic host infrastructure: the host fingerprint, the tuned-config store,
//! and the search-space version. Lifted out of imparo-metal (2026-08-20, user rule):
//! a file that needs win+mac branches is in the wrong crate -- the Metal backend is
//! macOS-only by definition, while this store serves every backend (CUDA-on-Windows
//! reuses it unchanged).
//!
//! APPLYING a stored configuration is backend-specific and stays with each backend
//! (imparo-metal's `apply_host_config`); this crate only identifies the host, locates
//! the file, and parses it. The fingerprint currently carries no per-backend device
//! component; when the #16 backend trait lands, backends contribute one through it.

//! Stored host configuration for shape/dispatch values.
//!
//! Per `docs/identity-and-configuration.md`: these values depend on the hardware, not on
//! the model, so they are measured once by `imparo-tune` and loaded here. The fingerprint
//! is deliberately COARSE -- if it is wrong the cost is a re-benchmark, never a wrong
//! answer, which is what separates it from the correctness key.

pub mod correctness;
pub mod receipted_config;

use std::path::PathBuf;
use std::process::Command;

/// Bumped when the set of knobs changes, so an engine update invalidates a stored file
/// without needing the hardware to change.
///
/// THE definition. imparo-tune reads it from here rather than keeping its own: the tuner
/// writes the file and this module reads it, so a version that differs between them puts
/// the file somewhere nobody looks. That is exactly what happened -- the tuner wrote
/// `...space-v3.txt` while the server went on loading `...space-v2.txt`.
///
/// v21 (Metal): REMOVED gemv_max_tok, q8_gemv_max_tok, q8_batch_sgs, q8_token_tile. The
/// GEMV/GEMM boundary is one row in every weight family: the GEMV's k-order is not the
/// GEMM's, so a boundary above one row made a prompt's K/V bytes depend on its chunk
/// width (Qwen3.8-27B, batch 64 against 512 at n=2000), which the KV pool's identity
/// contract forbids. The token tile the Q8 shape knobs ranked is unreachable by default.
///
/// v11: ADDED gemv_max_tok -- batches of 2..N tokens take the GEMV again. The v9
/// forfeit is repaid: the multi-token GEMV's non-determinism was root-caused as the
/// engine's own arena aliasing (overlapping ranges in distinct MTLBuffers, invisible
/// to per-resource hazard tracking) and fixed at the resource level, so the boundary
/// went back to being the performance crossover the old scan measured (~16 vs the
/// wide tile; vs nb8 it is measured per machine by the new crossing scan).
///
/// v10: ADDED nb8_max and nb8_shape -- the narrow-N GEMM boundary (batches of
/// 2..nb8_max tokens take the 64x8 tile) and its simdgroup variant. Both are
/// measured by the repurposed micro scan UNDER SHIPPING ROUTING; the old scan
/// timed the multi-token GEMV at min_tok=1, i.e. the task-#5 racy path, and its
/// own doc recorded that the crossing never reproduced.
///
/// v9: REMOVED prefill_min_tok. The multi-token GEMV it gated is non-deterministic for
/// remainder batches (task #5: same prompt, different logits run to run; GEMM-routed the
/// same prompts are bit-stable across 14 runs). Every n_tok > 1 batch now takes the
/// batched GEMM; the measured crossover (~16) is forfeited -- ~10 ms per prompt whose
/// remainder falls in 2..15 -- until the race is diagnosed. The env override remains for
/// A/B; nothing persists it, mirroring the v7 prefill_kernel removal.
///
/// v8: flush_layers MOVED MEANING at decode. A fixed geometric ramp (chunks of 1, 2, 4, ...
/// capped at flush_layers) now precedes the steady state, so the GPU starts after one
/// layer's encode instead of a full chunk's; the knob now names the steady-state chunk
/// size. A value tuned under the old plain-modulus semantics answers a different question,
/// so stored v7 files must not be applied. (Same release: the compiled default for lanes
/// moved 8 -> 16, measured 37.4 -> 38.3 decode tok/s on M3 Pro -- a default change only,
/// lanes keeps its meaning and its stage-1 sweep.)
///
/// v7: REMOVED prefill_kernel. It was a boolean -- batched GEMM or decode GEMV for a batch
/// -- and once the gate became `n_tok >= prefill_min_tok` the crossover is the whole
/// decision. The only thing the boolean could still say is "never batch, whatever the
/// width", which measures 6.8x slower on prefill and can never be the answer. It stays
/// reachable as IMPARO_PREFILL_KERNEL=0 for A/B work; nothing persists it.
///
/// v6: added attn_blk, the register-blocking depth in the attention matrix phases. The
/// ceiling is the register file, a device property, and how much is needed depends on QT and
/// head_dim, model properties -- so neither alone decides it. 4 and 2 tie on an M3 Pro and
/// 8 spills there, costing 12%; a Mac with a larger register file may want 8.
///
/// v5: added attn_min_tgs, how many threadgroups it takes to fill this GPU. It sets the
/// decode split count and is a pure MACHINE property -- 72 suited an 18-core M3 Pro and a
/// 40-core Mac wants roughly twice that -- but it had a setter and an env override and was
/// never swept.
///
/// v4: added prefill_min_tok, the width at which the batched prefill kernel starts beating
/// the decode GEMV. It was a fixed `n_tok > 1` gate, which is the wrong end of a curve that
/// crosses around 10 tokens on an M3 Pro.
///
/// v3: dropped norm_threads and norm_threads_decode (`imparo_metal_rms_norm` derives its
/// thread count from the row width and never read them); added attn_threads_prefill,
/// rt_shape, flush_layers, flush_layers_prefill and nr0, which until then were hand-written
/// into a file the tuner never produced.
/// Host+backend identity a stored tuning is valid for. `device` is the backend's
/// device_tag ("metal", "cuda"), `space` its OWN search-space version
/// (BackendKnobs::space_version): one backend's space bump never invalidates
/// another's stored config.
#[must_use]
pub fn fingerprint_for(device: &str, space: u32, kv: &str) -> String {
    format!("{}|dev={device}|space=v{space}|kv={kv}", fingerprint())
}

/// Canonical cache-type component of every host-config key and correctness receipt.
/// Equal K/V types retain the historic short spelling; mixed types use one comma.
/// Backends and tools must call this helper instead of choosing their own separator.
#[must_use]
pub fn canonical_kv_tag(k: &str, v: &str) -> String {
    if k == v {
        k.to_owned()
    } else {
        format!("{k},{v}")
    }
}

/// Host identity a stored tuning is valid for: CPU brand, core count, memory.
///
/// THE OS VERSION IS DELIBERATELY NOT HERE, and it used to be. What a tuning measures is
/// the hardware; the OS string was standing in for the Metal toolchain, and it is a bad
/// proxy in both directions -- a security patch moves it when codegen did not, and a
/// toolchain update need not move it at all. macOS 15.7.7 -> 15.7.9 discarded both the
/// tune config and the measured device profile on this machine, silently, and the engine
/// ran on compiled defaults until someone measured it.
///
/// The asymmetry decides it: a key that is too LOOSE costs a re-benchmark, while a key
/// that is too STRICT costs the whole tuning with one log line to say so. The OS version
/// is recorded inside the file instead (see `toolchain`), where a mismatch warns and
/// keeps the values.
///
/// Per-OS collection; every branch degrades to empty strings rather than failing.
#[must_use]
pub fn fingerprint() -> String {
    let (cpu, cores, mem, os) = host_facts();
    let _ = &os;
    format!("{cpu}|cores={cores}|mem={mem}")
}

fn host_facts() -> (String, String, String, String) {
    #[cfg(target_os = "macos")]
    let (cpu, cores, mem, os) = {
        let sysctl = |k: &str| -> String {
            Command::new("sysctl")
                .args(["-n", k])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };
        let os = Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        (
            sysctl("machdep.cpu.brand_string"),
            sysctl("hw.ncpu"),
            sysctl("hw.memsize"),
            os,
        )
    };
    #[cfg(target_os = "windows")]
    let (cpu, cores, mem, os) = {
        let env = |k: &str| std::env::var(k).unwrap_or_default();
        // Total memory via GlobalMemoryStatusEx would need winapi; `wmic` is deprecated.
        // The OS build from `cmd /c ver` plus CPU identity and core count is enough to
        // tell machines apart, which is all the fingerprint is for.
        let os = Command::new("cmd")
            .args(["/c", "ver"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        (
            env("PROCESSOR_IDENTIFIER"),
            env("NUMBER_OF_PROCESSORS"),
            String::new(),
            os,
        )
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let (cpu, cores, mem, os) = {
        let read = |p: &str| std::fs::read_to_string(p).unwrap_or_default();
        let cpu = read("/proc/cpuinfo")
            .lines()
            .find_map(|l| {
                l.strip_prefix("model name").map(|r| {
                    r.trim_start_matches(|c: char| c.is_ascii_whitespace() || c == ':')
                        .to_string()
                })
            })
            .unwrap_or_default();
        let mem = read("/proc/meminfo")
            .lines()
            .find_map(|l| l.strip_prefix("MemTotal:").map(|r| r.trim().to_string()))
            .unwrap_or_default();
        let cores = std::thread::available_parallelism()
            .map(|n| n.get().to_string())
            .unwrap_or_default();
        let os = Command::new("uname")
            .arg("-r")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        (cpu, cores, mem, os)
    };
    (cpu, cores, mem, os)
}

/// The toolchain the stored numbers were produced under -- the OS version, which is the
/// available proxy for the shader compiler that turns our Metal source into GPU code.
///
/// ADVISORY, NOT PART OF ANY KEY. A mismatch means "these numbers were measured under a
/// different compiler, they may have drifted", which is a reason to warn and suggest a
/// re-tune, not a reason to throw them away. The value that is genuinely a compiler
/// property is the register spill cliff, and re-measuring that is one command:
/// `imparo-tune MODEL --discover-only`.
#[must_use]
pub fn toolchain() -> String {
    let (_cpu, _cores, _mem, os) = host_facts();
    os
}

#[must_use]
pub fn slug(fp: &str) -> String {
    fp.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .to_lowercase()
}

/// One shared root for tuner writers and runtime readers.
///
/// Windows commonly has no `HOME`; `USERPROFILE` is the required fallback there.
#[must_use]
pub fn config_dir() -> PathBuf {
    config_dir_from(std::env::var_os("HOME"), std::env::var_os("USERPROFILE"))
}

fn config_dir_from(
    home: Option<std::ffi::OsString>,
    user_profile: Option<std::ffi::OsString>,
) -> PathBuf {
    home.or(user_profile)
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join(".imparo")
}

/// THE MODEL IS PART OF THE TUNE FILE'S KEY.
///
/// A tuning is measured on one model's shapes, and the reader has always refused a file
/// whose `model_bytes=` line names another model. But until 2026-09-01 the file's NAME
/// carried only host, backend, space and cache type, so two models tuned on one host took
/// turns overwriting the same file, and the loser ran on compiled defaults with one log
/// line saying so: every E4B run from Aug 31 to Sep 1 was untuned because LFM2 had been
/// tuned last. The byte count is the key (it is what the reader checks and the tuner has
/// before it opens the model); the model's name stays inside the file for people.
#[must_use]
pub fn path_for(fp: &str, model_bytes: u64) -> PathBuf {
    config_dir().join(format!("tune-{}-model-{model_bytes}.txt", slug(fp)))
}

/// The pre-2026-09-01 name: one file per host, backend, space and cache type, whichever
/// model was tuned last. Read as a fallback so a tuning measured before the model joined
/// the key is not lost; never written.
#[must_use]
pub fn legacy_path_for(fp: &str) -> PathBuf {
    config_dir().join(format!("tune-{}.txt", slug(fp)))
}

/// THE DEVICE PROFILE IS KEYED BY THE HOST AND THE BACKEND, AND BY NOTHING ELSE.
///
/// What a GPU's register file, cache and DRAM do is a property of the machine. It was
/// stored in the tune file, which is additionally keyed by the knob-space version, the
/// KV cache type and the model -- so a measurement that was still true got discarded by
/// three things that cannot change it:
///
///   add a knob        -> space v12 -> v13 -> file not found
///   tune f16, run q4  -> kv tag differs        -> file not found
///   tune E4B, run LFM -> model_bytes differs   -> mismatch
///
/// and every shape derived from the measurement silently reverted to its compiled
/// fallback, which is the literal that deriving it was meant to replace.
#[must_use]
pub fn device_fingerprint_for(device: &str) -> String {
    format!("{}|dev={device}", fingerprint())
}

#[must_use]
pub fn device_path_for(device: &str) -> PathBuf {
    config_dir().join(format!(
        "device-{}.txt",
        slug(&device_fingerprint_for(device))
    ))
}

/// Reads this host's measured device profile, or `None` when there is none for this
/// host. A file written on another machine is REFUSED, not adapted: these numbers are
/// the ground truth other values are derived from, so a wrong one is worse than absent.
#[must_use]
pub fn read_device_profile(device: &str) -> Option<Vec<(String, u64)>> {
    let want = device_fingerprint_for(device);
    let body = std::fs::read_to_string(device_path_for(device)).ok()?;
    let mut fp = String::new();
    let mut stored_toolchain = String::new();
    let mut out = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        if k == "fingerprint" {
            fp = v.to_string();
        } else if k == "toolchain" {
            stored_toolchain = v.to_string();
        } else if k.starts_with("device_") {
            if let Ok(n) = v.parse::<u64>() {
                out.push((k.to_string(), n));
            }
        }
    }
    if fp != want {
        eprintln!(
            "[imparo] device profile at {} was measured on another host; ignored",
            device_path_for(device).display()
        );
        return None;
    }
    // The spill cliff IS a compiler property, so this is the profile most exposed to a
    // toolchain change -- and still not a reason to discard a measurement and silently
    // ship a compiled fallback in its place. Warn; `--discover-only` re-measures in
    // seconds.
    let now = toolchain();
    if !stored_toolchain.is_empty() && stored_toolchain != now {
        eprintln!(
            "[imparo] device profile was measured under {stored_toolchain}, running \
             {now}; values kept -- re-run `imparo-tune MODEL --discover-only` to refresh"
        );
    }
    (!out.is_empty()).then_some(out)
}

/// Writes this host's measured device profile. The tuner calls this once per machine;
/// `discover` measures it in seconds and a process start cannot afford to at all.
///
/// # Errors
/// Propagates any filesystem error from creating the directory or writing the file.
pub fn write_device_profile(
    device: &str,
    values: &[(&str, u64)],
) -> std::io::Result<()> {
    let mut body = String::new();
    {
        use std::fmt::Write as _;
        let _ = writeln!(
            body,
            "# imparo device profile -- MEASURED properties of this machine's {device} GPU."
        );
        let _ = writeln!(
            body,
            "# Not a tuning: no knob, model or cache type is part of this file's key,"
        );
        let _ = writeln!(
            body,
            "# because none of them changes what the hardware does."
        );
        let _ = writeln!(body, "fingerprint={}", device_fingerprint_for(device));
        let _ = writeln!(body, "toolchain={}", toolchain());
    }
    for (k, v) in values {
        use std::fmt::Write as _;
        let _ = writeln!(body, "{k}={v}");
    }
    let path = device_path_for(device);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    receipted_config::write_atomic(&path, body.as_bytes())
}

/// THE VERIFY-COST TABLE'S OWN FILE, for the same reason the device profile has one.
///
/// A speculative verify's cost by row count is MEASURED BY THE ENGINE, from the rounds it
/// runs (the tuner never loads a model's tensors, so it cannot run a verify at all). It is
/// a property of this machine, this model and the kernels those rows dispatch -- so it
/// belongs beside the device profile, not inside the tune file:
///
///   tune file    what we CHOSE to do        written by imparo-tune
///   this file    what the machine DID       written by the engine, as it serves
///
/// Keeping them apart means a serving engine never rewrites the tuner's file, two engines
/// on one host cannot lose each other's knobs, and a retune does not have to carry a
/// measurement it did not make.
///
/// THE KEY IS NOT ENOUGH ON ITS OWN, so the file also carries the row CLASSES it was
/// measured under. A retune that moves the GEMM's tile seat changes where the cost steps;
/// the reader rebuilds the classes from the backend and refuses a file whose boundaries
/// differ, which catches that without a second fingerprint to keep in step.
#[must_use]
pub fn verify_cost_path_for(fp: &str, model_bytes: u64) -> PathBuf {
    config_dir().join(format!("verify-{}-model-{model_bytes}.txt", slug(fp)))
}

/// WHERE DESIGN 11.5B'S DICTIONARY LIVES. Keyed like the cost table -- the device's config
/// key and the model's bytes -- because a table learned against one drafter, one target and
/// one tokenizer means nothing to another. Another model starts empty rather than wrong.
///
/// It holds fragments of the user's text, so it is local only and one `rm` removes it: the
/// same privacy class as the KV disk tier.
#[must_use]
pub fn dspark_dictionary_path_for(fp: &str, model_bytes: u64) -> PathBuf {
    config_dir().join(format!("dspark-dict-{}-model-{model_bytes}.txt", slug(fp)))
}

/// The dictionary's rows, `(context hash, token, count)`. A row that does not parse is
/// SKIPPED rather than failing the load: a truncated or hand-edited file costs the counts it
/// lost, and nothing else.
///
/// Read from the newest older search space when this one has no file (`space_candidates`).
#[must_use]
pub fn read_dspark_dictionary(fp: &str, model_bytes: u64) -> Vec<(u64, u32, u32)> {
    let Some((key, body)) = space_candidates(fp).into_iter().find_map(|key| {
        let body =
            std::fs::read_to_string(dspark_dictionary_path_for(&key, model_bytes))
                .ok()?;
        Some((key, body))
    }) else {
        return Vec::new();
    };
    if key != fp {
        eprintln!(
            "[imparo] dspark dictionary carried forward from {}",
            dspark_dictionary_path_for(&key, model_bytes).display()
        );
    }
    body.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((
                it.next()?.parse().ok()?,
                it.next()?.parse().ok()?,
                it.next()?.parse().ok()?,
            ))
        })
        .collect()
}

/// Written through a temporary file and a rename, like the KV disk layout, so a crash mid-save
/// leaves the previous dictionary intact rather than half of two.
///
/// # Errors
/// When the directory cannot be created or the file cannot be written.
pub fn write_dspark_dictionary(
    fp: &str,
    model_bytes: u64,
    rows: &[(u64, u32, u32)],
) -> std::io::Result<()> {
    let mut body =
        String::from("# imparo dspark dictionary: context-hash token count\n");
    for (k, t, c) in rows {
        body.push_str(&format!("{k} {t} {c}\n"));
    }
    let path = dspark_dictionary_path_for(fp, model_bytes);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    receipted_config::write_atomic(&path, body.as_bytes())
}

/// One class of the stored table: the row range, its measured microseconds, and how many
/// samples stood behind it.
pub type VerifyCostRow = (usize, usize, f64, u64);

/// One class's LEARNED COST LAW: the row range, the basis weights, the learner's accumulated
/// squared gradients (its learning rate), how many rounds trained it, the context span it was
/// trained over, the scale it normalises by, and its floor.
///
/// This is the half of the table that cannot be learned inside one request. The context only
/// moves a few percent across a single answer, so the SLOPE is not identifiable from one --
/// it can only come from rounds at different depths, which means it can only come through
/// this file. Storing the law is not a convenience here; it is what makes the law learnable.
pub type VerifyCostLawRow = (usize, usize, [f64; 5], [f64; 5], u64, f64, f64, f64, f64);

/// One class law's SAMPLE BANK, `(lo, hi, [(context, ms)])`: the newest samples of each context
/// band, which the refit at a request end trains on. Stored because one request measures one
/// context band, and a refit that saw only that band would replace a law fitted over many.
pub type VerifyCostBankRow = (usize, usize, Vec<(u32, f32)>);

/// One context band's LONG-RUN RATE, `(band, tokens, microseconds, rounds)`: the decayed sums
/// behind the tokens per microsecond this device delivers there.
pub type RateRow = (usize, f64, f64, u64);

/// THE ACCEPTANCE MODEL'S LEARNER STATE (design 6.6): everything it has learned, so a request
/// boundary changes nothing about it. The server attaches a fresh provider for every request, so
/// what is not stored here is relearned from the prior every request.
///
/// `top_k` is the key: the model's features are shares among the drafter's top K, so weights
/// learned under another K mean something else. Per coefficient, `prior`, FTRL's accumulated
/// gradient `z` and squared-gradient sum `n`; the weights are recomputed from those three.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptRow {
    pub top_k: u32,
    /// Whether the start was fitted on measured rounds (it prices from the first round) or derived.
    pub fitted: bool,
    /// Labelled parents behind the state.
    pub seen: u64,
    /// The guard's decayed log-loss sums: the model, today's estimate.
    pub guard: [f64; 2],
    pub prior: Vec<f64>,
    pub z: Vec<f64>,
    pub n: Vec<f64>,
    /// Labelled nodes behind each coefficient.
    pub labels: Vec<u64>,
}

/// A stored table: the anchors the estimator falls back to, and the laws it prices with.
#[derive(Clone, Debug, Default)]
pub struct VerifyCostTable {
    pub steps: Vec<VerifyCostRow>,
    pub laws: Vec<VerifyCostLawRow>,
    /// Each law's sample bank.
    pub banks: Vec<VerifyCostBankRow>,
    /// THE VALUE SIDE's learned scale, `(num, den, seen)`: how much the drafter's predicted
    /// marginal is actually worth. It is stored here for the same reason the laws are --
    /// only a round that runs WIDE reveals it, so one short request rarely collects enough.
    pub gain: ([f64; 3], [f64; 2], u64),
    /// The same scale for the n-gram's and agreed nodes' marginal (design 6.6's g_n).
    pub gain_n: ([f64; 3], [f64; 2], u64),
    /// The long-run rate by context band. Stored because the server's provider lives for one
    /// request, and a rate that starts cold every request never leaves its warm-up.
    pub rate: Vec<RateRow>,
    /// The acceptance model's learner state, when the model ran.
    pub accept: Option<AcceptRow>,
}

/// Reads this host's stored verify-cost table for one model, or `None` when there is none
/// for this host and model. The caller checks the row ranges against the backend's classes.
/// Read from the newest older search space when this one has no file (`space_candidates`).
#[must_use]
pub fn read_verify_cost(fp: &str, model_bytes: u64) -> Option<VerifyCostTable> {
    space_candidates(fp).into_iter().find_map(|key| {
        let path = verify_cost_path_for(&key, model_bytes);
        let table = parse_verify_cost(&std::fs::read_to_string(&path).ok()?, &key)?;
        if key != fp {
            eprintln!(
                "[imparo] verify-cost table carried forward from {}",
                path.display()
            );
        }
        Some(table)
    })
}

/// A key, then the same key at every OLDER search space, newest first.
///
/// The engine's learned files -- the verify-cost table and the DSpark dictionary -- are keyed
/// by the tune file's fingerprint, and the fingerprint names the search space. So a space
/// bump started both empty: adding one knob threw away every round the engine had learned
/// from. Neither file depends on the space. The cost table carries the widths it learned, and
/// the engine seeds its costs from it only when this build's ladder holds them; the dictionary
/// holds the user's text. So when this space has no file, the newest older
/// space's is read, as the tune file is (`read_for_device`), and the engine writes its next
/// save under the current key.
fn space_candidates(fp: &str) -> Vec<String> {
    const TAG: &str = "|space=v";
    let mut keys = vec![fp.to_owned()];
    let Some(at) = fp.find(TAG) else {
        return keys;
    };
    let rest = &fp[at + TAG.len()..];
    let end = rest.find('|').unwrap_or(rest.len());
    let Ok(space) = rest[..end].parse::<u32>() else {
        return keys;
    };
    let (head, tail) = (&fp[..at], &rest[end..]);
    keys.extend((1..space).rev().map(|s| format!("{head}{TAG}{s}{tail}")));
    keys
}

/// The stored table's text, as rows. Separate from the read so the round trip against
/// `verify_cost_body` is a test and not a file: the writer emitted a row the reader silently
/// dropped for as long as the two were only ever exercised through the disk.
#[must_use]
pub fn parse_verify_cost(body: &str, fp: &str) -> Option<VerifyCostTable> {
    let mut stored_fp = String::new();
    let mut rows = Vec::new();
    let mut laws: Vec<VerifyCostLawRow> = Vec::new();
    let mut banks: Vec<VerifyCostBankRow> = Vec::new();
    let mut gain = ([0.0_f64; 3], [0.0_f64; 2], 0_u64);
    let mut gain_n = ([0.0_f64; 3], [0.0_f64; 2], 0_u64);
    let mut rate: Vec<RateRow> = Vec::new();
    let mut accept: Option<AcceptRow> = None;
    for line in body.lines() {
        let line = line.trim();
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "fingerprint" => stored_fp = v.trim().to_string(),
            // Five fitted scalars and a count. A line of any other width is from a
            // build whose gain had a different shape; it is DROPPED, not reinterpreted,
            // and the reader falls back to the prior (no correction).
            "gain" | "gain_n" => {
                let f: Vec<&str> = v.trim().split(',').collect();
                if let [x0, x1, x2, y0, y1, seen] = f[..] {
                    let n: Vec<f64> = [x0, x1, x2, y0, y1]
                        .iter()
                        .filter_map(|s| s.parse::<f64>().ok())
                        .filter(|v| v.is_finite())
                        .collect();
                    if let (5, Ok(c)) = (n.len(), seen.parse::<u64>()) {
                        let g = ([n[0], n[1], n[2]], [n[3], n[4]], c);
                        if k.trim() == "gain" {
                            gain = g;
                        } else {
                            gain_n = g;
                        }
                    }
                }
            }
            "rate" => {
                let f: Vec<&str> = v.trim().split(',').collect();
                if let [band, t, us, n] = f[..] {
                    if let (Ok(band), Ok(t), Ok(us), Ok(n)) = (
                        band.parse::<usize>(),
                        t.parse::<f64>(),
                        us.parse::<f64>(),
                        n.parse::<u64>(),
                    ) {
                        if t.is_finite() && us.is_finite() && t >= 0.0 && us > 0.0 {
                            rate.push((band, t, us, n));
                        }
                    }
                }
            }
            // top_k, fitted, seen, the guard's two sums, then prior, z, n and labels, one per
            // coefficient. A row that does not parse is dropped whole: half a learner is not a
            // learner.
            "accept" => {
                let f: Vec<&str> = v.trim().split(',').collect();
                if f.len() > 5 && (f.len() - 5) % 4 == 0 {
                    let c = (f.len() - 5) / 4;
                    let nums: Option<Vec<f64>> = f[3..5 + 3 * c]
                        .iter()
                        .map(|x| x.parse::<f64>().ok().filter(|v| v.is_finite()))
                        .collect();
                    let labels: Option<Vec<u64>> = f[5 + 3 * c..]
                        .iter()
                        .map(|x| x.parse::<u64>().ok())
                        .collect();
                    if let (Ok(top_k), Ok(fitted), Ok(seen), Some(nums), Some(labels)) = (
                        f[0].parse::<u32>(),
                        f[1].parse::<u8>(),
                        f[2].parse::<u64>(),
                        nums,
                        labels,
                    ) {
                        let n = nums[2 + 2 * c..].to_vec();
                        if fitted <= 1 && n.iter().all(|x| *x >= 0.0) {
                            accept = Some(AcceptRow {
                                top_k,
                                fitted: fitted == 1,
                                seen,
                                guard: [nums[0], nums[1]],
                                prior: nums[2..2 + c].to_vec(),
                                z: nums[2 + c..2 + 2 * c].to_vec(),
                                n,
                                labels,
                            });
                        }
                    }
                }
            }
            "step" => {
                let f: Vec<&str> = v.trim().split(',').collect();
                if let [lo, hi, us, seen] = f[..] {
                    if let (Ok(lo), Ok(hi), Ok(us), Ok(seen)) = (
                        lo.parse::<usize>(),
                        hi.parse::<usize>(),
                        us.parse::<f64>(),
                        seen.parse::<u64>(),
                    ) {
                        // `(0, 0, ..)` is the round's FIXED cost -- the table's clock --
                        // which belongs to no class and which the writer emits. Dropping
                        // it here made the reader disagree with the writer, so a restart
                        // silently lost the clock's epoch and re-established it from
                        // whatever context the new run happened to start at.
                        let class = lo >= 2 && hi >= lo;
                        let clock = lo == 0 && hi == 0;
                        if (class || clock) && us.is_finite() && us > 0.0 {
                            rows.push((lo, hi, us, seen));
                        }
                    }
                }
            }
            // lo, hi, then `context:ms` pairs separated by `;`. A pair that does not parse is
            // skipped; a row with no range is dropped.
            "bank" => {
                let mut f = v.trim().splitn(3, ',');
                if let (Some(lo), Some(hi), Some(pairs)) = (f.next(), f.next(), f.next()) {
                    if let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                        let samples: Vec<(u32, f32)> = pairs
                            .split(';')
                            .filter_map(|p| p.split_once(':'))
                            .filter_map(|(c, v)| Some((c.parse::<u32>().ok()?, v.parse::<f32>().ok()?)))
                            .filter(|&(c, v)| c > 0 && v.is_finite() && v > 0.0)
                            .collect();
                        if !samples.is_empty() {
                            banks.push((lo, hi, samples));
                        }
                    }
                }
            }
            "law" => {
                let f: Vec<&str> = v.trim().split(',').collect();
                // lo, hi, five weights, five squared gradients, trained rounds, lo_ctx, hi_ctx,
                // scale, floor. A line with no squared gradients was written before they were
                // stored; it reads as zeros, the rate of a law that has seen nothing.
                if let [lo, hi, w @ .., seen, lo_ctx, hi_ctx, scale, floor] = &f[..] {
                    if w.len() != 5 && w.len() != 10 {
                        continue;
                    }
                    let nums: Option<Vec<f64>> =
                        w.iter().map(|x| x.parse::<f64>().ok()).collect();
                    if let (Ok(lo), Ok(hi), Some(nums), Ok(seen)) = (
                        lo.parse::<usize>(),
                        hi.parse::<usize>(),
                        nums,
                        seen.parse::<u64>(),
                    ) {
                        if let (Ok(a), Ok(b), Ok(sc), Ok(fl)) = (
                            lo_ctx.parse::<f64>(),
                            hi_ctx.parse::<f64>(),
                            scale.parse::<f64>(),
                            floor.parse::<f64>(),
                        ) {
                            let (mut wv, mut nv) = ([0.0; 5], [0.0; 5]);
                            wv.copy_from_slice(&nums[..5]);
                            if nums.len() == 10 {
                                nv.copy_from_slice(&nums[5..]);
                            }
                            if wv.iter().chain(&nv).all(|x| x.is_finite())
                                && sc > 0.0
                                && fl > 0.0
                            {
                                laws.push((lo, hi, wv, nv, seen, a, b, sc, fl));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // A table measured on another host or for another model is REFUSED, not adapted: these
    // are wall times from one machine's kernels.
    if stored_fp != fp || rows.is_empty() {
        return None;
    }
    Some(VerifyCostTable {
        steps: rows,
        laws,
        banks,
        gain,
        gain_n,
        rate,
        accept,
    })
}

/// Writes this host's verify-cost table. `ctx` is advisory -- the context the rows were
/// measured at, recorded so a reader can see what state produced them, never part of the
/// key (the survey found the cost moves 7-12% across a 19x context change, which the
/// engine's own running estimate absorbs).
pub fn write_verify_cost(
    fp: &str,
    model_bytes: u64,
    ctx: usize,
    table: &VerifyCostTable,
) -> std::io::Result<()> {
    let body = verify_cost_body(fp, model_bytes, ctx, table);
    let path = verify_cost_path_for(fp, model_bytes);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    receipted_config::write_atomic(&path, body.as_bytes())
}

/// The file's text. `parse_verify_cost` must read back every row this writes.
#[must_use]
pub fn verify_cost_body(
    fp: &str,
    model_bytes: u64,
    ctx: usize,
    table: &VerifyCostTable,
) -> String {
    use std::fmt::Write as _;
    let mut body = String::new();
    let _ = writeln!(
        body,
        "# imparo verify-cost table -- what a speculative verify of n rows MEASURED here."
    );
    let _ = writeln!(
        body,
        "# Written by the engine from the rounds it ran, not by imparo-tune: the tuner"
    );
    let _ = writeln!(
        body,
        "# never loads a model's tensors and so cannot run a verify."
    );
    let _ = writeln!(
        body,
        "# A reader must check `step` ranges against its own ladder of widths and discard"
    );
    let _ = writeln!(
        body,
        "# the file when one does not fit (a build with a different widest verify)."
    );
    let _ = writeln!(body, "fingerprint={fp}");
    let _ = writeln!(body, "model_bytes={model_bytes}");
    let _ = writeln!(body, "toolchain={}", toolchain());
    let _ = writeln!(body, "measured_at_context={ctx}");
    for (lo, hi, us, seen) in &table.steps {
        let _ = writeln!(body, "step={lo},{hi},{us:.1},{seen}");
    }
    for (name, (gxx, gxy, gs)) in [("gain", table.gain), ("gain_n", table.gain_n)] {
        if gs > 0 {
            let g: Vec<String> = gxx
                .iter()
                .chain(gxy.iter())
                .map(|v| format!("{v:.6}"))
                .collect();
            let _ = writeln!(body, "{name}={},{gs}", g.join(","));
        }
    }
    for (lo, hi, w, n, seen, lo_ctx, hi_ctx, scale, floor) in &table.laws {
        let ws: Vec<String> = w
            .iter()
            .map(|x| format!("{x:.6}"))
            .chain(n.iter().map(|x| format!("{x:.4}")))
            .collect();
        let _ = writeln!(
            body,
            "law={lo},{hi},{},{seen},{lo_ctx:.0},{hi_ctx:.0},{scale:.4},{floor:.6}",
            ws.join(",")
        );
    }
    for (lo, hi, samples) in &table.banks {
        let pairs: Vec<String> = samples.iter().map(|(c, v)| format!("{c}:{v}")).collect();
        let _ = writeln!(body, "bank={lo},{hi},{}", pairs.join(";"));
    }
    // The learners' state is written EXACTLY (`{:e}` round-trips an f64): a request boundary must
    // not move what they have learned, and a rounded sum would move it every request.
    for (band, t, us, n) in &table.rate {
        let _ = writeln!(body, "rate={band},{t:e},{us:e},{n}");
    }
    if let Some(a) = &table.accept {
        let nums: Vec<String> = a
            .guard
            .iter()
            .chain(&a.prior)
            .chain(&a.z)
            .chain(&a.n)
            .map(|x| format!("{x:e}"))
            .chain(a.labels.iter().map(ToString::to_string))
            .collect();
        let _ = writeln!(
            body,
            "accept={},{},{},{}",
            a.top_k,
            u8::from(a.fitted),
            a.seen,
            nums.join(",")
        );
    }
    body
}

/// Resolve an explicit candidate/config path or this host's canonical default.
/// `IMPARO_HOST_CONFIG` is shared by the receipt sealer and runtime loader.
#[must_use]
pub fn selected_config_path(fp: &str, model_bytes: u64) -> PathBuf {
    std::env::var_os("IMPARO_HOST_CONFIG")
        .filter(|value| !value.is_empty())
        .map_or_else(|| path_for(fp, model_bytes), PathBuf::from)
}

/// Every value a stored file can carry, parsed but NOT applied.
///
/// GENERIC on purpose: knob names are a backend's private business (metal's nb8_max,
/// cuda's gemm_stages), so the store neither knows nor validates them -- it hands
/// `(name, value)` pairs to the backend, whose registry is the one enumeration.
/// This struct once hardcoded metal's field list, and every knob added to the engine
/// had to be added here, in the tuner's Candidate, in apply_host_config and in the
/// stored file writer; `lanes` and later `gemv_max_tok` were each lost in one of the
/// five copies.
///
/// `batch` stays typed: it is not a backend knob (the model layer chunks prefill) and
/// the engine reads it before any backend exists.
///
/// Split from apply because the tuner has to READ the configuration a host is already
/// running -- it measures its own result against that -- while applying it would change
/// the thing being measured.
#[derive(Clone, Debug, Default)]
pub struct Stored {
    /// Backend knob values, in file order. Interpretation belongs to the backend.
    pub knobs: Vec<(String, u32)>,
    /// PER-TENSOR SEATS: `(wire kind, n_in, n_out, value)` from `blk_rows.K.IN.OUT=V` lines.
    ///
    /// A knob is one number for the whole engine; these are one number per TENSOR, because
    /// which kernel wins is a property of the tensor and not of its weight format -- on
    /// Qwen3.8-27B UD-Q4_K_S the scalar rows GEMV beats the matrix unit on six of Q4_K's seven
    /// shapes and loses on the seventh. They cannot ride in `knobs`, whose keys are matched
    /// against the backend's knob registry by name.
    ///
    /// Interpretation belongs to the backend, as with `knobs`: this parses the shape and keeps
    /// file order, nothing more.
    pub seats: Vec<(u32, u32, u32, u32)>,
    pub batch: Option<usize>,
    /// The file the knobs were read from (the exact space's file, or an older space's
    /// carried forward).
    pub path: PathBuf,
}

/// Reads this host's stored configuration for one backend without applying any of it.
/// `device` = (device_tag, backend space version).
///
/// `None` when there is no file, or it was written for another host, search space or
/// model. `quiet` suppresses the explanation, for a caller that is only asking whether
/// one exists rather than about to run on it.
#[must_use]
/// THE KV CACHE TYPE IS PART OF THE KEY, and it has to be.
///
/// A tuning is only valid for the cache it was measured on. Quantized caches take
/// DIFFERENT KERNELS -- the QT-8 prefill attention path exists only for them -- and those
/// pipelines are built at init from the configured type, so a process serving q4 never
/// runs the kernels an f16 tuning ranked. Before this, one file per host was applied to
/// every cache type, which is a measurement from one kernel steering another.
pub fn read_for_device(
    model_bytes: u64,
    quiet: bool,
    device: (&str, u32, &str),
) -> Option<Stored> {
    let fp = fingerprint_for(device.0, device.1, device.2);
    let path = path_for(&fp, model_bytes);
    // THE OVERRIDE FIRST, AND LOUDLY. `IMPARO_HOST_CONFIG` names one file to run; it is what
    // a written-file A/B and the receipt sealer use. Until 2026-09-02 only
    // `selected_config_path` (the untuned check) read it and this loader went straight to
    // the canonical lookup, so an "A/B of the written file" ran the stored file on both
    // arms and could only ever tie (review #116, #120). A set-but-unusable override is
    // reported and NOT silently replaced by the lookup: that is the same trap again.
    if let Some(over) = std::env::var_os("IMPARO_HOST_CONFIG").filter(|v| !v.is_empty())
    {
        // Logged as the ABSOLUTE path: a relative override in a log line cannot be checked
        // against the file that actually ran once the working directory is gone.
        let over = PathBuf::from(over);
        let over = std::fs::canonicalize(&over).unwrap_or(over);
        match std::fs::read_to_string(&over) {
            Ok(body) => {
                if let Some(mut c) = parse_stored(&body, &fp, &over, model_bytes, quiet)
                {
                    c.path.clone_from(&over);
                    if !quiet {
                        eprintln!(
                            "[imparo] host config loaded from IMPARO_HOST_CONFIG ({})",
                            over.display()
                        );
                    }
                    return Some(c);
                }
                eprintln!(
                    "[imparo] IMPARO_HOST_CONFIG={} does not parse for this device/space/model; \
                     running compiled defaults, NOT the stored file",
                    over.display()
                );
                return None;
            }
            Err(e) => {
                eprintln!(
                    "[imparo] IMPARO_HOST_CONFIG={} unreadable ({e}); running compiled \
                     defaults, NOT the stored file",
                    over.display()
                );
                return None;
            }
        }
    }
    // CANDIDATES, NEWEST SPACE FIRST, THE MODEL-KEYED NAME BEFORE THE LEGACY ONE.
    //
    // Carry forward across a space bump: the search-space version is part of the
    // filename, so adding one knob used to make every measured value on this host vanish
    // at once -- the engine ran compiled defaults (st_gemm_large_shape 7 -> 3 was a
    // measured -4% on LFM2) and nothing but one log line said so. A knob that was
    // measured is still measured after another knob joins the space. So the newest OLDER
    // space's file for the same host, cache type and model is read instead; its knobs
    // apply through the registry as before (a key the build no longer has is reported and
    // ignored there), and the knobs added since keep their compiled defaults until
    // imparo-tune seats them.
    //
    // The legacy (un-keyed) name is content-checked like any other candidate: it may hold
    // another model's tuning, which is a miss, not an error, and the search goes on.
    for space in (1..=device.1).rev() {
        let space_fp = fingerprint_for(device.0, space, device.2);
        for candidate in [path_for(&space_fp, model_bytes), legacy_path_for(&space_fp)]
        {
            let Ok(body) = std::fs::read_to_string(&candidate) else {
                continue;
            };
            let Some(mut c) =
                parse_stored(&body, &space_fp, &candidate, model_bytes, true)
            else {
                continue;
            };
            c.path.clone_from(&candidate);
            if !quiet {
                // Re-run the parse loudly for its advisory lines (toolchain drift).
                let _ = parse_stored(&body, &space_fp, &candidate, model_bytes, false);
                if space != device.1 {
                    eprintln!(
                        "[imparo] host config carried forward from search space v{space} \
                         ({}); knobs added since keep their compiled defaults -- run \
                         imparo-tune to seat them and write the v{} file",
                        candidate.display(),
                        device.1
                    );
                }
            }
            return Some(c);
        }
    }
    // Say so. A missing file once returned silently, so an untuned engine looked exactly
    // like a tuned one in a benchmark.
    if !quiet {
        eprintln!(
            "[imparo] no host config at {} (nor an older space's, nor a pre-model-key file \
             measured on this model); using defaults (run imparo-tune)",
            path.display()
        );
    }
    None
}

/// One stored file: the knob lines it carries, if it was written for this host, this
/// search space (`expected_fp`) and this model.
fn parse_stored(
    body: &str,
    expected_fp: &str,
    path: &std::path::Path,
    model_bytes: u64,
    quiet: bool,
) -> Option<Stored> {
    let mut c = Stored::default();
    let mut stored_fp = String::new();
    let mut stored_model: Option<u64> = None;
    let mut stored_toolchain = String::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let (k, v) = (k.trim(), v.trim());
        match k {
            "fingerprint" => stored_fp = v.to_string(),
            "model" => {}
            "model_bytes" => stored_model = v.parse().ok(),
            "batch" => c.batch = v.parse().ok(),
            // ADVISORY: what the numbers were measured under, not part of the key.
            "toolchain" => stored_toolchain = v.to_string(),
            _ if k.starts_with("measured_") => {}
            // Legacy: device measurements used to live here. They are a property of the
            // machine, not of a knob space, a cache type or a model, so they moved to
            // their own file -- see `read_device_profile`. Old files still carry the
            // lines; ignore them rather than reading a copy that may be older.
            _ if k.starts_with("device_") => {}
            // A PER-TENSOR SEAT: `blk_rows.<wire kind>.<n_in>.<n_out>=<value>`. Four numbers,
            // all of which must parse -- a line with a missing or unparsable field is DROPPED,
            // like a knob's, because a seat guessed from half a key would steer a tensor the
            // tuner never measured.
            _ if k.starts_with("blk_rows.") => {
                let mut f = k["blk_rows.".len()..].split('.');
                if let (
                    Some(Ok(kind)),
                    Some(Ok(n_in)),
                    Some(Ok(n_out)),
                    None,
                    Ok(val),
                ) = (
                    f.next().map(str::parse::<u32>),
                    f.next().map(str::parse::<u32>),
                    f.next().map(str::parse::<u32>),
                    f.next(),
                    v.parse::<u32>(),
                ) {
                    c.seats.push((kind, n_in, n_out, val));
                }
            }
            _ => {
                // Every remaining key is a backend knob. An unparsable value is DROPPED
                // (the backend keeps its compiled default), never guessed.
                if let Ok(n) = v.parse::<u32>() {
                    c.knobs.push((k.to_string(), n));
                }
            }
        }
    }
    // TOOLCHAIN DRIFT WARNS, IT DOES NOT DISCARD. These numbers were measured under a
    // different shader compiler, which may have changed codegen -- or may not have. The
    // one value that really is a compiler property is the register spill cliff, and
    // re-measuring it is `imparo-tune MODEL --discover-only`, seconds of work.
    let now = toolchain();
    if !stored_toolchain.is_empty() && stored_toolchain != now && !quiet {
        eprintln!(
            "[imparo] host config was tuned under {stored_toolchain}, running {now}; \
             values kept -- re-run imparo-tune if you suspect drift"
        );
    }
    if stored_model.is_some_and(|m| m != model_bytes) {
        if !quiet {
            eprintln!(
                "[imparo] host config at {} was measured on a different model \
                       ({} bytes, this one is {model_bytes}); using defaults (run \
                       imparo-tune for this model)",
                path.display(),
                stored_model.unwrap_or(0)
            );
        }
        return None;
    }
    if stored_fp != expected_fp {
        if !quiet {
            eprintln!(
                "[imparo] host config at {} is for a different host or search \
                       space; using defaults (run imparo-tune)",
                path.display()
            );
        }
        return None;
    }
    Some(c)
}

// ---- OS memory accounting (moved from imparo-runtime/imparo-model, 2026-08-20) ----
// Pure-OS: no backend, no model. Per-OS branches live HERE because OS accounting is
// this crate's job (the cfg-placement rule).

/// This process's physical footprint -- the number the OS charges it. macOS
/// phys_footprint (RUSAGE_INFO_V0), Windows PrivateUsage, Linux VmRSS.
#[must_use]
pub fn footprint_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        #[derive(Default)]
        struct RUsageInfoV0 {
            uuid: [u8; 16],
            user_time: u64,
            system_time: u64,
            pkg_idle_wkups: u64,
            interrupt_wkups: u64,
            pageins: u64,
            wired_size: u64,
            resident_size: u64,
            phys_footprint: u64,
            proc_start_abstime: u64,
            proc_exit_abstime: u64,
        }
        unsafe extern "C" {
            fn proc_pid_rusage(
                pid: i32,
                flavor: i32,
                buffer: *mut core::ffi::c_void,
            ) -> i32;
        }
        let mut info = RUsageInfoV0::default();
        // SAFETY: the struct matches RUSAGE_INFO_V0 exactly and outlives the call.
        let rc = unsafe {
            proc_pid_rusage(
                std::process::id() as i32,
                0,
                std::ptr::addr_of_mut!(info).cast(),
            )
        };
        if rc == 0 { info.phys_footprint } else { 0 }
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Linux VmRSS; Windows returns 0 for now (a psapi query is the port).
        if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
            for line in s.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Some(kb) = rest.split_whitespace().next() {
                        return kb.parse::<u64>().unwrap_or(0) * 1024;
                    }
                }
            }
        }
        0
    }
}

#[cfg(target_os = "macos")]
#[link(name = "objc")]
unsafe extern "C" {
    fn objc_autoreleasePoolPush() -> *mut core::ffi::c_void;
    fn objc_autoreleasePoolPop(pool: *mut core::ffi::c_void);
}

/// An autorelease pool around one unit of work on a thread that outlives it. Metal hands
/// out objects autoreleased, and each keeps alive what it references until its pool drains;
/// a thread that never exits never drains its own, so a long-lived worker opens one scope
/// per unit of work, as a run loop does. The pool drains when the scope drops, on the
/// thread that opened it. No-op off macOS.
pub struct AutoreleaseScope {
    #[cfg(target_os = "macos")]
    pool: *mut core::ffi::c_void,
}

impl AutoreleaseScope {
    #[must_use = "the pool drains when the scope drops"]
    pub fn open() -> Self {
        Self {
            // SAFETY: a libobjc entry point with no precondition.
            #[cfg(target_os = "macos")]
            pool: unsafe { objc_autoreleasePoolPush() },
        }
    }
}

impl Drop for AutoreleaseScope {
    fn drop(&mut self) {
        // SAFETY: `pool` came from objc_autoreleasePoolPush on this thread (the raw pointer
        // keeps the scope on it), and every pool opened inside it was closed first.
        #[cfg(target_os = "macos")]
        unsafe {
            objc_autoreleasePoolPop(self.pool);
        }
    }
}

/// Hand the allocator's free pages back to the OS. Freeing is not returning: libmalloc
/// keeps dropped pages on its free lists and the OS goes on charging them. Safe at a
/// request boundary, pointless in a hot loop. No-op off macOS.
pub fn release_free_heap() {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn malloc_default_zone() -> *mut core::ffi::c_void;
            fn malloc_zone_pressure_relief(
                zone: *mut core::ffi::c_void,
                goal: usize,
            ) -> usize;
        }
        // SAFETY: libmalloc entry points; the default zone is valid and pressure relief
        // only releases memory the allocator already considers free.
        unsafe {
            let zone = malloc_default_zone();
            if !zone.is_null() {
                malloc_zone_pressure_relief(zone, 0);
            }
        }
    }
}

#[cfg(test)]
mod config_identity_tests {

    /// PER-TENSOR SEAT LINES parse into `seats` and not into `knobs`, a malformed one is dropped
    /// rather than half-read, and an ordinary knob line is untouched by the new arm.
    ///
    /// The prefix matters: a seat carries four numbers (wire kind, n_in, n_out, value) where a
    /// knob carries one, and `knobs` keys are matched against the backend's registry by name --
    /// a seat landing there would be reported as an unknown knob and dropped.
    #[test]
    fn seat_lines_parse_apart_from_knobs_and_a_malformed_one_is_dropped() {
        let body = "\
fingerprint=FP
model_bytes=99
streamk=1
blk_rows.21.5120.10240=2
blk_rows.26.17408.5120=1
blk_rows.21.5120=2
blk_rows.21.5120.10240.7=2
blk_rows.21.5120.wide=2
blk_rows.21.5120.10240=x
";
        let c = super::parse_stored(body, "FP", std::path::Path::new("t"), 99, true)
            .expect("the fingerprint and model match, so this file is for us");
        assert_eq!(c.knobs, vec![("streamk".into(), 1)]);
        assert_eq!(c.seats, vec![(21, 5120, 10240, 2), (26, 17408, 5120, 1)]);
    }

    /// The learned files are read at the key's own space first, then each older space,
    /// newest first; everything else in the key stays as it was.
    #[test]
    fn a_learned_file_is_looked_up_at_every_older_space_newest_first() {
        let fp = super::fingerprint_for("metal", 3, "f16");
        let keys = super::space_candidates(&fp);
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0], fp);
        assert_eq!(keys[1], super::fingerprint_for("metal", 2, "f16"));
        assert_eq!(keys[2], super::fingerprint_for("metal", 1, "f16"));
        // A key without a space has nothing older to try.
        assert_eq!(
            super::space_candidates("Test Host|dev=metal"),
            vec!["Test Host|dev=metal"]
        );
    }

    /// EVERY ROW THE WRITER EMITS MUST COME BACK, the learned laws included.
    ///
    /// The reader used to filter on `lo >= 2`, which silently dropped the `(0, 0, ..)` clock
    /// row -- the round's fixed cost -- so a restart lost the table's epoch and rebuilt it
    /// from whatever context the new run began at. Nothing caught it because the two halves
    /// were only ever exercised through a file.
    #[test]
    fn the_verify_cost_file_round_trips_every_row_including_the_clock_and_the_laws() {
        let fp = "Test Host|cores=1|mem=1|dev=metal|space=v21|kv=f16";
        let table = super::VerifyCostTable {
            steps: vec![
                (2, 8, 30_988.0, 27),
                (9, 16, 35_561.4, 56),
                (17, 24, 49_880.2, 3),
                (0, 0, 10_940.5, 108), // the clock
            ],
            laws: vec![
                (
                    2,
                    8,
                    [0.778, 0.030, 0.027, -0.026, 0.028],
                    [38.2, 12.25, 0.0, 400.0, 3.5],
                    312,
                    443.0,
                    8444.0,
                    31.8,
                    0.97,
                ),
                (
                    0,
                    0,
                    [0.9, 0.02, 0.0, 0.0, 0.01],
                    [400.0, 400.0, 0.0, 0.0, 17.0],
                    400,
                    443.0,
                    8444.0,
                    10.9,
                    0.99,
                ),
            ],
            banks: vec![(2, 8, vec![(443, 25.03), (1596, 26.9), (8444, 35.5)])],
            // the value model's scale rides the same round trip
            gain: ([2.431, -0.118, 0.409], [2.150, -0.061], 41),
            gain_n: ([0.52, 0.018, 0.001], [0.61, 0.004], 9),
            rate: vec![(10, 311.25, 1.25e7, 90), (14, 0.1 + 0.2, 3.0e6 / 7.0, 12)],
            accept: Some(super::AcceptRow {
                top_k: 8,
                fitted: false,
                seen: 1234,
                guard: [51.2, 1.0 / 3.0],
                prior: vec![1.05, 0.93, 0.42, 0.98, 0.89, 0.72, -0.228_220_000_000_001],
                z: vec![0.0, -12.5, 60.000_000_000_1, 0.0, 1e-300, -7.25, 3.5],
                n: vec![0.0, 400.0, 399.999_999, 0.0, 17.5, 2.0 / 3.0, 400.0],
                labels: vec![0, 90_210, 7, 0, 12, 1, u64::from(u32::MAX) + 3],
            }),
        };
        let body = super::verify_cost_body(fp, 2_874_779_456, 8444, &table);
        let back =
            super::parse_verify_cost(&body, fp).expect("the writer's own text parses");
        assert_eq!(back.steps, table.steps, "an anchor was dropped");
        assert_eq!(back.banks, table.banks, "a sample bank was dropped");
        assert_eq!(back.laws.len(), table.laws.len(), "a law was dropped");
        for (got, want) in back.laws.iter().zip(&table.laws) {
            assert_eq!((got.0, got.1, got.4), (want.0, want.1, want.4));
            for (a, b) in got.2.iter().zip(&want.2) {
                assert!((a - b).abs() < 1e-5, "weight {a} != {b}");
            }
            // the learner's rate: a restored law must learn as slowly as the one that was stored
            for (a, b) in got.3.iter().zip(&want.3) {
                assert!((a - b).abs() < 1e-3, "squared gradient {a} != {b}");
            }
            assert!(
                (got.7 - want.7).abs() < 1e-3,
                "scale {} != {}",
                got.7,
                want.7
            );
        }

        // A law line written before the squared gradients were stored reads with zeros.
        let old_line = body.replace(
            "law=2,8,0.778000,0.030000,0.027000,-0.026000,0.028000,38.2000,12.2500,0.0000,400.0000,3.5000,",
            "law=2,8,0.778000,0.030000,0.027000,-0.026000,0.028000,",
        );
        assert_ne!(old_line, body, "the test's old-format line was not built");
        let old = super::parse_verify_cost(&old_line, fp).expect("an old law line parses");
        let law = old.laws.iter().find(|l| l.0 == 2).expect("the old law is kept");
        assert!(law.3.iter().all(|&x| x == 0.0), "an old law reads with zero squared gradients");

        // The value model's fit rides the same round trip, and nothing checked it before.
        for (a, b) in back
            .gain
            .0
            .iter()
            .chain(back.gain.1.iter())
            .zip(table.gain.0.iter().chain(table.gain.1.iter()))
        {
            assert!((a - b).abs() < 1e-5, "gain {a} != {b}");
        }
        assert_eq!(back.gain.2, table.gain.2, "the gain's count was dropped");
        assert_eq!(
            back.gain_n.2, table.gain_n.2,
            "the n-gram gain's count was dropped"
        );
        assert!(
            (back.gain_n.1[0] - table.gain_n.1[0]).abs() < 1e-5,
            "the n-gram gain did not round-trip"
        );

        // The learners' state comes back BIT FOR BIT: a stored sum that rounds moves the learner
        // at every request boundary.
        assert_eq!(
            back.rate, table.rate,
            "the long-run rate did not round-trip exactly"
        );
        assert_eq!(
            back.accept, table.accept,
            "the acceptance model did not round-trip exactly"
        );

        // A file from another host is refused whole, not adapted: these are wall times.
        assert!(
            super::parse_verify_cost(&body, "Another Host|cores=1|mem=1").is_none()
        );
    }

    /// A gain written by a build that fitted a different shape must be DROPPED. Reading its
    /// numbers into today's fit would seed the learner with a quantity it never measured --
    /// silently, because every field still parses as a float.
    #[test]
    fn a_gain_of_the_wrong_width_is_dropped() {
        let fp = "Test Host|cores=1|mem=1";
        let body =
            format!("fingerprint={fp}\nstep=2,8,30000.0,9\ngain=2.431,1.702,41\n");
        let back = super::parse_verify_cost(&body, fp).expect("the step still reads");
        assert_eq!(
            back.gain,
            ([0.0; 3], [0.0; 2], 0),
            "a stale gain was adopted"
        );
    }

    /// A row that is neither a class (`lo >= 2`) nor the clock (`0,0`) is not something this
    /// writer emits, so reading one means the file was edited or written by another version.
    #[test]
    fn a_row_that_is_neither_a_class_nor_the_clock_is_dropped() {
        let fp = "Test Host|cores=1|mem=1";
        let body = format!("fingerprint={fp}\nstep=1,1,500.0,3\nstep=2,8,30000.0,9\n");
        let back = super::parse_verify_cost(&body, fp).expect("one good row");
        assert_eq!(back.steps, vec![(2, 8, 30_000.0, 9)]);
        assert!(back.laws.is_empty());
    }

    use super::*;

    #[test]
    fn canonical_kv_tags_have_one_unambiguous_spelling() {
        assert_eq!(canonical_kv_tag("q4_0", "q4_0"), "q4_0");
        assert_eq!(canonical_kv_tag("q4_0", "f16"), "q4_0,f16");
        assert_ne!(canonical_kv_tag("q4_0", "f16"), "q4_0-f16");
    }

    #[cfg(windows)]
    #[test]
    fn windows_without_home_uses_userprofile() {
        let profile = std::env::temp_dir()
            .join(format!("imparo-userprofile-test-{}", std::process::id()));
        assert_eq!(
            config_dir_from(None, Some(profile.clone().into_os_string())),
            profile.join(".imparo")
        );
    }
}
