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

/// Host identity a stored tuning is valid for: CPU brand, core count, memory, OS
/// version, and the search-space version. Per-OS collection; every branch degrades to
/// empty strings rather than failing, because a wrong fingerprint only costs a
/// re-benchmark, never a wrong answer.
#[must_use]
pub fn fingerprint() -> String {
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
                l.strip_prefix("model name")
                    .map(|r| r.trim_start_matches([' ', ':']).to_string())
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
    format!("{cpu}|cores={cores}|mem={mem}|os={os}")
}

#[must_use]
pub fn slug(fp: &str) -> String {
    fp.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .to_lowercase()
}

#[must_use]
pub fn path_for(fp: &str) -> PathBuf {
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
    let dir = home
        .map_or_else(|_| PathBuf::from("."), PathBuf::from)
        .join(".imparo");
    dir.join(format!("tune-{}.txt", slug(fp)))
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
    /// MEASURED device ground truth, `device_`-prefixed in the file.
    ///
    /// Cached because measuring it costs seconds -- a spill-cliff sweep and a cache-knee
    /// sweep -- which the tuner can afford once and an engine start cannot afford at all.
    /// The engine needs these before it compiles its kernel library, because shape values
    /// are DERIVED from them and injected as preprocessor defines. Without the cache a
    /// derivation would have to fall back to a literal, which is the thing deriving it
    /// was for.
    ///
    /// Keyed by the same host fingerprint as the knobs, so a profile measured on another
    /// machine is never read here.
    pub device: Vec<(String, u64)>,
    pub batch: Option<usize>,
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
    let path = path_for(&fp);
    let Ok(body) = std::fs::read_to_string(&path) else {
        // Say so. A missing file once returned silently, so an untuned engine looked
        // exactly like a tuned one in a benchmark. The search-space version is part of
        // the FILENAME, so bumping it lands here rather than in the mismatch branch.
        if !quiet {
            eprintln!(
                "[imparo] no host config at {}; using defaults (run imparo-tune)",
                path.display()
            );
        }
        return None;
    };
    let mut c = Stored::default();
    let mut stored_fp = String::new();
    let mut stored_model: Option<u64> = None;
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
            _ if k.starts_with("measured_") => {}
            _ if k.starts_with("device_") => {
                if let Ok(n) = v.parse::<u64>() {
                    c.device.push((k.to_string(), n));
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
    if stored_fp != fp {
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
