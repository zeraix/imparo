//! Execution-host glue that is NOT pure OS and NOT pure model: the thin wrappers the
//! server and tuner call. Each is now backend-agnostic -- it asks the active Backend
//! (imparo-backend trait) or imparo-host, with zero `cfg(target_os)`. (The pure-OS
//! pieces moved to imparo-host; argmax moved to imparo-cpu::ops; per the rule that a
//! file needing per-backend branches is sitting below the trait it should use.)

use crate::backend::active;

/// Log the GPU submission profile for one phase, if `IMPARO_PROF=1`. No-op when no
/// backend is active or the device reports no timings.
///
/// NOT A TIMING INSTRUMENT. `IMPARO_PROF=1` turns on dispatch-boundary counter sampling,
/// and the sampling costs far more than anything it measures: a 512-token prefill chunk
/// that normally takes tens of milliseconds was logged at 65 994 ms with it on. That is a
/// three-orders-of-magnitude observer effect, not the couple of percent a phase probe
/// costs, so any number read from a profiled run describes the profiler.
///
/// Use it for SHAPE and CALL COUNTS -- which categories ran, how many dispatches, how many
/// command buffers. For per-category TIME, difference two unprofiled runs using the skip
/// levers instead: `IMPARO_SKIP_ATTN=2` drops the full-attention layers, `IMPARO_SKIP_CAT`
/// drops a category, and neither costs anything to have compiled in.
pub fn prof_log(phase: &str, wall_ms: f64) {
    if !std::env::var("IMPARO_PROF").is_ok_and(|v| v == "1") {
        return;
    }
    let Some(be) = active() else { return };
    let p = be.prof_stats();
    eprintln!(
        "[prof] {phase} wall={wall_ms:.0}ms gpu_busy={:.0}ms ({:.0}% of wall) \
               submit_wall={:.0}ms cbs={} dispatches={} barriers={} gpu_per_cb={:.0}us",
        p.gpu_s * 1e3,
        100.0 * p.gpu_s * 1e3 / wall_ms.max(1e-9),
        p.wall_s * 1e3,
        p.cbs,
        p.dispatches,
        p.barriers,
        1e6 * p.gpu_s / (p.cbs.max(1) as f64)
    );
    let total: f64 = p.categories.iter().map(|(_, t, _)| *t).sum();
    if total <= 0.0 {
        let mut c = p.categories;
        c.sort_by_key(|(_, _, calls)| std::cmp::Reverse(*calls));
        for (name, _, calls) in &c {
            eprintln!("[prof]   {name:<15} calls={calls}");
        }
        return;
    }
    let mut cats = p.categories;
    cats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (name, t, calls) in &cats {
        eprintln!(
            "[prof]   {name:<15} {:5.1}%  calls={calls:<6} gpu_ms~{:7.1}",
            100.0 * t / total.max(1e-9),
            wall_ms * (t / total.max(1e-9))
        );
    }
}

/// Configure the KV cache storage types before the engine is built (server CLI path).
/// Returns false when a name is not one of f16 | q4_0 | q8_0. No-op backend side when
/// no GPU backend is active.
#[must_use]
pub fn configure_kv_types(k: &str, v: &str) -> bool {
    use crate::kv::{KV_CFG_K, KV_CFG_V, KvType};
    let (Some(tk), Some(tv)) = (KvType::parse(k), KvType::parse(v)) else {
        return false;
    };
    let _ = KV_CFG_K.set(tk);
    let _ = KV_CFG_V.set(tv);
    if let Some(be) = active() {
        be.set_kv_types(tk.ggml_id(), tv.ggml_id());
    }
    true
}

/// Log the footprint at a named startup stage when `IMPARO_LOG` is set: OS footprint
/// (imparo-host) plus the active backend's device allocation (trait).
pub fn log_footprint(stage: &str) {
    if !std::env::var("IMPARO_LOG").is_ok_and(|v| v != "0" && !v.is_empty()) {
        return;
    }
    let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
    let gpu = active().map_or(0, imparo_backend::Backend::allocated_bytes);
    eprintln!(
        "[imparo] footprint after {stage:<18} {:.1} MiB  gpu={:.1} MiB",
        mib(imparo_host::footprint_bytes()),
        mib(gpu)
    );
}

/// Ask the allocator to return freed pages to the OS (imparo-host); safe at a request
/// boundary.
pub fn release_free_memory() {
    imparo_host::release_free_heap();
}

/// Install SIGTERM/SIGINT handlers that write one byte to a self-pipe, and
/// return the read end. The handler does nothing but `write(2)` -- the only
/// async-signal-safe thing worth doing -- so the server can run its shutdown
/// work (spilling resident KV) on a normal thread that blocks on this stream.
/// Unix only; on other targets returns None and a hard kill skips the spill.
#[cfg(unix)]
pub fn shutdown_pipe() -> Option<std::os::unix::net::UnixStream> {
    use std::os::fd::AsRawFd;
    use std::sync::atomic::{AtomicI32, Ordering};
    static WRITE_FD: AtomicI32 = AtomicI32::new(-1);
    extern "C" fn on_signal(_sig: libc::c_int) {
        let fd = WRITE_FD.load(Ordering::Relaxed);
        if fd >= 0 {
            unsafe {
                let _ = libc::write(fd, b"s".as_ptr().cast(), 1);
            }
        }
    }
    let (rx, tx) = std::os::unix::net::UnixStream::pair().ok()?;
    // The write end must outlive the process's signal handling: leak it.
    let tx = Box::leak(Box::new(tx));
    WRITE_FD.store(tx.as_raw_fd(), Ordering::Relaxed);
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = on_signal as *const libc::c_void as usize;
        libc::sigemptyset(&raw mut sa.sa_mask);
        libc::sigaction(libc::SIGTERM, &raw const sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGINT, &raw const sa, std::ptr::null_mut());
    }
    Some(rx)
}

#[cfg(not(unix))]
pub fn shutdown_pipe() -> Option<std::convert::Infallible> {
    None
}
