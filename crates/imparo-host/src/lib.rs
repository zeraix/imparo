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
