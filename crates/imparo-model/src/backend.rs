//! Backend selection -- the composition seam. Which GPU backend is compiled is a
//! feature/target choice (imparo-metal on macOS, imparo-cuda behind `cuda`); WHETHER
//! one is active at runtime is what routing asks, via `active()`. This is the one
//! place the cfg that names a backend lives (the design doc's rule: cfg(target_os)
//! only in imparo-host, whole-crate backend gates, and here at the composition root).
//!
//! When the app grows to construct backends explicitly (multiple runtime-selectable
//! devices, a drafter's second instance), this becomes a constructor the binaries
//! call and thread as `&dyn Backend`; today one process uses one backend, so a
//! compiled-in selector is enough and keeps the workflow free of any backend cfg.
//!
//! ONE MODEL PER PROCESS, and what it would take to lift that. The Metal backend keeps a
//! single `Context`: one device, one queue, one weight buffer, one activation buffer
//! array, one pipeline set. Three things are chosen once and baked into that set --
//!
//!   KV cache type      function constants 3 and 4
//!   register tile      which pipelines get built at all
//!   FFN activation     function constant 11
//!
//! -- so a second model is not a matter of loading more weights: it needs its own
//! pipelines. The change is making `Context` a per-model object the workflow holds,
//! rather than a global. Until then a second `init_weights` REFUSES with rc=4, because
//! the alternative is the first model silently running with the second's activation.

use imparo_backend::Backend;

/// The active GPU backend, or None when none is compiled in (e.g. Windows without the
/// `cuda` feature) -- in which case the forward runs on the CPU reference path.
#[must_use]
pub fn active() -> Option<&'static dyn Backend> {
    // IMPARO_BACKEND=cpu runs the SAME op graph on the host. Not a fallback and not a
    // debug mode: everything built on this trait -- the KV pool, block tables,
    // checkpoints, the disk tier -- is reachable from it, which is the whole reason a
    // host implementation exists next to the reference forward.
    if std::env::var("IMPARO_BACKEND").is_ok_and(|v| v == "cpu") {
        static BE: imparo_cpu::CpuBackend = imparo_cpu::CpuBackend;
        return Some(&BE);
    }
    #[cfg(feature = "cuda")]
    {
        static BE: imparo_cuda::CudaBackend = imparo_cuda::CudaBackend;
        return Some(&BE);
    }
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    {
        static BE: imparo_metal::MetalBackend = imparo_metal::MetalBackend;
        Some(&BE)
    }
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    {
        // No GPU backend compiled in: the host backend, so this platform gets the
        // pool and the tier rather than reprocessing every prompt.
        static BE: imparo_cpu::CpuBackend = imparo_cpu::CpuBackend;
        Some(&BE)
    }
}

/// Composition-root weight init: when `IMPARO_GPU` is set AND a GPU backend is
/// compiled in, hand the weight mapping to it and
/// mark the Weights so the workflow routes to the GPU. Returns Ok with no backend
/// touched when the env is unset or no backend is active (CPU path).
///
/// # Errors
/// Propagates the backend's init failure -- refusing to fall back silently, because a
/// broken shader that quietly runs on CPU reads as a catastrophic regression.
pub fn enable_gpu(
    weights: &mut imparo_gguf::weights::Weights,
    plan: &crate::ModelPlan,
) -> Result<(), String> {
    if !std::env::var("IMPARO_GPU").is_ok_and(|v| v != "0" && !v.is_empty()) {
        return Ok(());
    }
    let Some(be) = active() else { return Ok(()) }; // no GPU backend compiled in
    // BEFORE init_weights: the activation SPECIALISES the epilogue kernels at pipeline
    // build, and setting it afterwards is a silent no-op that would leave GELU.
    be.set_activation(model_activation(plan)?.epilogue());
    // SAFETY: base_ptr/byte_len describe the live weight mmap, which outlives the process.
    match unsafe { be.init_weights(weights.base_ptr(), weights.byte_len()) } {
        Ok(()) => {
            weights.mark_gpu();
            eprintln!(
                "[imparo] {}: weights shared, GPU path enabled",
                be.device_tag()
            );
            Ok(())
        }
        Err(rc) => Err(format!(
            "backend init_weights failed rc={rc} (rc=2 = the shader library did not \
             compile, and the message above says which kernel; rc=4 = a model is \
             already loaded, and this build runs one model per process). IMPARO_GPU was \
             set, so refusing to fall back to CPU -- unset it to run on CPU \
             deliberately."
        )),
    }
}

/// The one activation this model's feed-forward uses.
///
/// One per process, because the epilogue is specialised at pipeline build. Every model
/// here uses a single activation throughout; a file that mixed them per layer is
/// REJECTED rather than run with whichever layer came first, because the wrong
/// activation produces plausible numbers and not a crash.
///
/// # Errors
/// When the plan's layers disagree.
fn model_activation(plan: &crate::ModelPlan) -> Result<crate::Activation, String> {
    let mut seen: Option<crate::Activation> = None;
    for l in &plan.layers {
        let a = l.ffn.activation();
        match seen {
            None => seen = Some(a),
            Some(first) if first != a => {
                return Err(format!(
                    "layer {} uses {a:?} but layer 0 uses {first:?}; the fused epilogue \
                     is specialised once per process and cannot mix them",
                    l.index
                ));
            }
            Some(_) => {}
        }
    }
    Ok(seen.unwrap_or(crate::Activation::Gelu))
}
