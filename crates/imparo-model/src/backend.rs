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

use imparo_backend::{Backend, StreamedWeightSpan};

/// The backend, with its KV page declared to `imparo-kv`.
///
/// `imparo-kv` cannot ask for it: that crate is below this one and the backend registry
/// lives here, so the value is pushed down at the one point every path goes through.
/// ONE declaration, set once -- `page_cells()` and `grid_tokens()` both read it, the
/// grid being the page while a resident unit is one block, so there is no second state
/// to keep in step.
///
/// Idempotent, and a disagreement is a hard stop rather than a silent re-init: stored
/// prefixes are `H(root || tokens[0..p])` at multiples of it, so two pages in one
/// process means a store that nothing can match. Every backend declares 64 today, so
/// this can only fire after someone changes a page without meaning to.
fn with_page(be: &'static dyn Backend) -> &'static dyn Backend {
    if let Err(e) = imparo_kv::set_page_cells(be.pool_caps().page_cells as usize) {
        panic!("[imparo] kv page: {e}");
    }
    be
}

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
        return Some(with_page(&BE));
    }
    #[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
    {
        static BE: imparo_cuda::CudaBackend = imparo_cuda::CudaBackend;
        return Some(with_page(&BE));
    }
    #[cfg(all(
        target_os = "macos",
        not(any(feature = "cuda", feature = "cuda-dynamic"))
    ))]
    {
        static BE: imparo_metal::MetalBackend = imparo_metal::MetalBackend;
        Some(with_page(&BE))
    }
    #[cfg(all(
        not(target_os = "macos"),
        not(any(feature = "cuda", feature = "cuda-dynamic"))
    ))]
    {
        // No GPU backend compiled in: the host backend, so this platform gets the
        // pool and the tier rather than reprocessing every prompt.
        static BE: imparo_cpu::CpuBackend = imparo_cpu::CpuBackend;
        Some(with_page(&BE))
    }
}

#[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
fn install_cuda_correctness_identity_if_needed(
    weights: &imparo_gguf::weights::Weights,
    plan: &crate::ModelPlan,
) -> Result<(), String> {
    use imparo_backend::BackendKnobs as _;
    let gate_mode = std::env::var("IMPARO_CORRECTNESS_GATE").as_deref() == Ok("1");
    if std::env::var_os("IMPARO_NO_HOSTCONFIG").is_some() && !gate_mode {
        return Ok(());
    }
    let k = crate::kv::KvType::k().as_str();
    let v = crate::kv::KvType::v().as_str();
    let kv_layout_sha256 =
        crate::kv::effective_kv_byte_layout_profile(plan, k, v)?.sha256_identity();
    let kv = imparo_host::canonical_kv_tag(k, v);
    let fp = imparo_host::fingerprint_for(
        "cuda",
        imparo_cuda::CudaBackend.space_version(),
        &kv,
    );
    if !gate_mode
        && !imparo_host::selected_config_path(&fp, weights.byte_len()).is_file()
    {
        return Ok(());
    }
    imparo_cuda::install_correctness_identity(
        weights.full_file_sha256(),
        *plan.sha256_identity().as_bytes(),
        kv_layout_sha256,
    )
}
fn normalize_streamed_spans(
    model_bytes: u64,
    mut spans: Vec<StreamedWeightSpan>,
) -> Result<Vec<StreamedWeightSpan>, String> {
    spans.sort_unstable_by_key(|span| span.offset);
    let mut previous_end = 0_u64;
    for span in &spans {
        let end = span
            .offset
            .checked_add(span.bytes)
            .ok_or_else(|| "streamed weight span overflows u64".to_string())?;
        if span.bytes == 0 {
            return Err("streamed weight span is empty".to_string());
        }
        if span.offset < previous_end {
            return Err("streamed weight spans overlap".to_string());
        }
        if end > model_bytes {
            return Err(format!(
                "streamed weight span {}..{} exceeds model mapping {}",
                span.offset, end, model_bytes
            ));
        }
        previous_end = end;
    }
    Ok(spans)
}

/// Exact composition predicate used by both weight activation and KV identity.
/// Kept pure so route selection can be tested without mutating process-global env.
pub(crate) fn gpu_requested(value: Option<&str>) -> bool {
    // UNSET MEANS GPU (user decision 2026-09-02). The Triton merge made GPU opt-in, and a
    // bare `imparo-forward MODEL TOKENS...` then died in the GPU arena path with a nil
    // device ("metal arena rc=3") instead of running anywhere. The explicit refusals
    // stay: IMPARO_GPU=0 (or empty) runs on CPU deliberately, and a GPU init failure
    // still errors rather than falling back silently.
    value.is_none_or(|value| value != "0" && !value.is_empty())
}

pub(crate) fn gpu_requested_from_env() -> bool {
    let value = std::env::var("IMPARO_GPU").ok();
    gpu_requested(value.as_deref())
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
    capacity: usize,
    max_batch: usize,
) -> Result<(), String> {
    if !gpu_requested_from_env() {
        return Ok(());
    }
    let Some(be) = active() else { return Ok(()) }; // no GPU backend compiled in
    #[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
    install_cuda_correctness_identity_if_needed(weights, plan)?;
    // BEFORE init_weights: the activation SPECIALISES the epilogue kernels at pipeline
    // build, and setting it afterwards is a silent no-op that would leave GELU.
    be.set_activation(model_activation(plan)?.epilogue());
    // BEFORE init_weights for the same reason: the attention kernels are specialised per
    // head dim when the library is compiled, and this model's dims are what should be
    // compiled. Taken from the PLAN's layers rather than a model-level field, because a
    // network can carry several geometries -- gemma4's windowed layers are head_dim 256
    // and its full layers 512 -- which is what the LayerPlan comment says a model-level
    // field cannot express.
    let mut hds: Vec<u32> = plan
        .layers
        .iter()
        .filter(|l| l.attention.is_attention())
        .map(|l| l.attention.head_dim())
        .collect();
    hds.sort_unstable();
    hds.dedup();
    be.set_attention_head_dims(&hds);
    // The K/V row width goes with each dim: kv heads x head dim. Every K/V tile load in the
    // prefill attention kernels carries that stride, and compiled as a constant it is an
    // immediate in the address arithmetic instead of a uniform read per tile.
    let kvws: Vec<u32> = hds.iter().map(|hd| plan.config.n_kv_heads * hd).collect();
    be.set_attention_kv_widths(&kvws);
    // WHERE THE WEIGHTS LIVE (docs/memory-tiers-and-fit.md): the common runtime decides,
    // once, from the plan, the configured context, the prefill batch and the backend's
    // budget; the backend implements the tiers. The decision is printed unconditionally --
    // a model with layers in the slow tier must say so, and a model that fits must be seen
    // to (the no-regression guard reads this line).
    // Every weight type in the file must have a kernel on this backend; a tile-major kind
    // without readers here (a converted file meant for another backend) is refused by name.
    for (name, t) in &weights.tensors {
        if !be.serves_weight_type(t.ggml_type) {
            return Err(format!(
                "{name} has ggml type {} ({}), which the {} backend has no kernels for",
                t.ggml_type,
                imparo_gguf::tensor_layout(t.ggml_type)
                    .map(|l| l.name)
                    .unwrap_or("unknown"),
                be.device_tag()
            ));
        }
    }
    let placement = crate::placement::plan_placement(
        &weights.tensors,
        plan,
        capacity,
        max_batch,
        be.fast_tier_budget(),
    );
    eprintln!(
        "[imparo] {}",
        crate::placement::describe(&placement, capacity, max_batch)
    );
    // IMPARO_PLACEMENT_LOG=1: every segment and every tensor, for a "weight offset is in
    // no segment" abort -- the answer is which tensor sits at that offset.
    if std::env::var("IMPARO_PLACEMENT_LOG").is_ok_and(|v| v == "1") {
        for s in &placement.segments {
            eprintln!(
                "[placement] seg off={} bytes={} {:?}",
                s.offset, s.bytes, s.tier
            );
        }
        for (name, t) in &weights.tensors {
            eprintln!(
                "[placement] tensor {name} off={} bytes={}",
                t.offset, t.bytes
            );
        }
    }
    // The spans outside the fast tier, validated against the mapping the way the old
    // streamed-tensor list was (sorted, disjoint, in bounds).
    normalize_streamed_spans(weights.byte_len(), placement.slow_spans())?;
    // SAFETY: base_ptr/byte_len describe the live weight mmap, which outlives the process;
    // every segment was built from that mapping's own tensor table.
    match unsafe {
        be.init_weights_with_placement(
            weights.base_ptr(),
            weights.byte_len(),
            &placement,
        )
    } {
        Ok(()) => {
            weights.mark_gpu();
            eprintln!(
                "[imparo] {}: weights shared, GPU path enabled",
                be.device_tag()
            );
            load_time_repack(weights, &placement, be)?;
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

/// The tile-major rule the load-time repack WILL apply to a tensor on this backend, or
/// None: the transform is off (`IMPARO_LOAD_REPACK=0`), the tensor has no rule or is a
/// row-gathered role, the rule's kind has no readers anywhere, or the active backend does
/// not serve it. The tuner asks this so it ranks knobs on the kinds the engine actually
/// dispatches after load, not the file's (an original Q8_0 file runs the tile-major
/// kernels once transformed).
#[must_use]
pub fn load_time_rule(
    name: &str,
    ggml_type: u32,
    dims: &[u64],
) -> Option<&'static imparo_gguf::weights::TmRule> {
    if std::env::var("IMPARO_LOAD_REPACK").is_ok_and(|v| v == "0") {
        return None;
    }
    let rule = imparo_gguf::weights::tm_applies(name, ggml_type, dims).ok()?;
    if rule.readers.is_empty() {
        return None;
    }
    let be = active()?;
    be.serves_weight_type(rule.to).then_some(rule)
}

/// The ggml type a tensor has AFTER load on this backend (the rule's kind when the
/// transform applies, the file's otherwise).
#[must_use]
pub fn runtime_weight_type(name: &str, ggml_type: u32, dims: &[u64]) -> u32 {
    load_time_rule(name, ggml_type, dims).map_or(ggml_type, |r| r.to)
}

/// The startup repack (docs/memory-tiers-and-fit.md section 7): every fast-tier tensor
/// with a tile-major rule is handed to the backend, which may rewrite it into a private
/// copy; the tensors it transformed change type here so dispatch routes to the tile-major
/// kernels. The file is never written. IMPARO_LOAD_REPACK=0 turns it off (the A/B);
/// IMPARO_LOAD_REPACK_VERIFY=1 reads every transformed tensor back and checks it byte for
/// byte against the rule -- the GPU kernel is pinned to the one rule table this way.
fn load_time_repack(
    weights: &mut imparo_gguf::weights::Weights,
    placement: &imparo_backend::WeightPlacement,
    be: &'static dyn imparo_backend::Backend,
) -> Result<(), String> {
    if std::env::var("IMPARO_LOAD_REPACK").is_ok_and(|v| v == "0") {
        eprintln!("[imparo] repack at load: off (IMPARO_LOAD_REPACK=0)");
        return Ok(());
    }
    let in_fast = |off: u64| {
        placement.segments.iter().any(|s| {
            matches!(s.tier, imparo_backend::WeightTier::Fast)
                && off >= s.offset
                && off < s.offset + s.bytes
        })
    };
    let mut names: Vec<String> = Vec::new();
    let mut jobs: Vec<imparo_backend::WeightTransform> = Vec::new();
    let mut skipped_slow = 0_usize;
    for (name, t) in &weights.tensors {
        // The same decision the tuner makes (`load_time_rule`): a rule with readers on
        // this backend, for a tensor that is not a row-gathered role.
        let Some(rule) = load_time_rule(name, t.ggml_type, &t.ne[..t.n_dims as usize])
        else {
            continue;
        };
        if !in_fast(t.offset as u64) {
            skipped_slow += 1;
            continue;
        }
        names.push(name.clone());
        jobs.push(imparo_backend::WeightTransform {
            offset: t.offset as u64,
            bytes: t.bytes as u64,
            from_type: rule.from,
            to_type: rule.to,
            n_in: t.ne[0] as u32,
            n_out: t.ne[1] as u32,
        });
    }
    if jobs.is_empty() {
        return Ok(());
    }
    let t0 = std::time::Instant::now();
    let applied = be.transform_weights(&jobs)?;
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    let verify = std::env::var("IMPARO_LOAD_REPACK_VERIFY").is_ok_and(|v| v == "1");
    let mut n_applied = 0_usize;
    let mut bytes_applied = 0_u64;
    let mut kinds: std::collections::BTreeMap<&'static str, usize> = Default::default();
    for (i, job) in jobs.iter().enumerate() {
        if !applied[i] {
            continue;
        }
        let rule = imparo_gguf::weights::tm_rule_for(job.from_type)
            .expect("job came from a rule");
        if verify {
            let t = weights.tensors[&names[i]];
            let src = weights.raw(&t).to_vec();
            let expected = rule.convert(&src, job.n_in as usize, job.n_out as usize);
            let mut actual = vec![0_u8; job.bytes as usize];
            if !be.read_weight_bytes(job.offset, &mut actual) {
                return Err(format!("repack verify: cannot read back {}", names[i]));
            }
            if actual != expected {
                let first = actual
                    .iter()
                    .zip(&expected)
                    .position(|(a, e)| a != e)
                    .unwrap_or(0);
                return Err(format!(
                    "repack verify: {} differs from the rule at byte {first} of {}",
                    names[i], job.bytes
                ));
            }
        }
        if let Some(t) = weights.tensors.get_mut(&names[i]) {
            t.ggml_type = job.to_type;
        }
        n_applied += 1;
        bytes_applied += job.bytes;
        *kinds.entry(rule.to_name).or_default() += 1;
    }
    let kinds_s: Vec<String> =
        kinds.iter().map(|(k, n)| format!("{n} -> {k}")).collect();
    eprintln!(
        "[imparo] repack at load: {n_applied} of {} fast-tier tensors transformed ({} MiB; {}) in {ms:.0} ms{}{}",
        jobs.len(),
        bytes_applied >> 20,
        if kinds_s.is_empty() {
            "none".to_string()
        } else {
            kinds_s.join(", ")
        },
        if skipped_slow > 0 {
            format!("; {skipped_slow} outside the fast tier, read in the file's layout")
        } else {
            String::new()
        },
        if verify {
            "; verified byte-for-byte against the rule"
        } else {
            ""
        },
    );
    Ok(())
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
pub fn model_activation(plan: &crate::ModelPlan) -> Result<crate::Activation, String> {
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
#[cfg(test)]
mod tests {
    use super::{StreamedWeightSpan, gpu_requested, normalize_streamed_spans};

    #[test]
    fn gpu_request_predicate_matches_enable_gpu_contract() {
        assert!(gpu_requested(None)); // unset = GPU when a backend is compiled in
        assert!(!gpu_requested(Some("")));
        assert!(!gpu_requested(Some("0")));
        assert!(gpu_requested(Some("1")));
        assert!(gpu_requested(Some("yes")));
    }

    #[test]
    fn streamed_spans_are_sorted() {
        let spans = normalize_streamed_spans(
            100,
            vec![
                StreamedWeightSpan {
                    offset: 40,
                    bytes: 10,
                },
                StreamedWeightSpan {
                    offset: 5,
                    bytes: 7,
                },
            ],
        )
        .unwrap();
        assert_eq!(spans[0].offset, 5);
        assert_eq!(spans[1].offset, 40);
    }

    #[test]
    fn streamed_spans_reject_empty_overflow_bounds_and_overlap() {
        assert!(
            normalize_streamed_spans(
                100,
                vec![StreamedWeightSpan {
                    offset: 1,
                    bytes: 0
                }],
            )
            .is_err()
        );
        assert!(
            normalize_streamed_spans(
                u64::MAX,
                vec![StreamedWeightSpan {
                    offset: u64::MAX,
                    bytes: 1
                }],
            )
            .is_err()
        );
        assert!(
            normalize_streamed_spans(
                100,
                vec![StreamedWeightSpan {
                    offset: 90,
                    bytes: 11
                }],
            )
            .is_err()
        );
        assert!(
            normalize_streamed_spans(
                100,
                vec![
                    StreamedWeightSpan {
                        offset: 10,
                        bytes: 20
                    },
                    StreamedWeightSpan {
                        offset: 29,
                        bytes: 1
                    },
                ],
            )
            .is_err()
        );
    }

    #[test]
    fn empty_residency_plan_is_valid() {
        assert!(
            normalize_streamed_spans(100, Vec::new())
                .unwrap()
                .is_empty()
        );
    }
}
