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
    // ONE MEANING, TWO SPELLINGS. `IMPARO_GPU=0` is the documented way to say "run on
    // the CPU deliberately" (see `gpu_requested`), but only `enable_gpu` used to read it:
    // `active()` still handed back the device backend, so a CPU run reached the GPU
    // arena path and died on an uninitialised device ("metal arena rc=3") instead of
    // running anywhere. Both spellings select the host backend here.
    if std::env::var("IMPARO_BACKEND").is_ok_and(|v| v == "cpu")
        || !gpu_requested_from_env()
    {
        static BE: imparo_cpu::CpuBackend = imparo_cpu::CpuBackend;
        return Some(with_page(&BE));
    }
    #[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
    {
        static BE: imparo_cuda::CudaBackend = imparo_cuda::CudaBackend;
        Some(with_page(&BE))
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
        && !imparo_host::selected_config_path(&fp, weights.model_bytes()).is_file()
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
) -> Result<(), String> {
    enable_gpu_with_appended(weights, plan, capacity, &[])
}

/// [`enable_gpu`] for a mapping that holds more than the model: `appended` is the
/// (offset, bytes) of each tensor past the model's own -- a paired drafter's -- which the
/// weight placement must cover before any kernel reads one.
///
/// # Errors
/// As [`enable_gpu`].
pub fn enable_gpu_with_appended(
    weights: &mut imparo_gguf::weights::Weights,
    plan: &crate::ModelPlan,
    capacity: usize,
    appended: &[(u64, u64)],
) -> Result<(), String> {
    if !gpu_requested_from_env() {
        return Ok(());
    }
    let Some(be) = active() else { return Ok(()) }; // no GPU backend compiled in
    // THE CHUNK IS DECIDED BEFORE ANYTHING IS SIZED FROM IT. Placement reserves KV state and
    // activations for one chunk, so the tuned config's chunk for this model is read here,
    // before placement; `IMPARO_BATCH` set by the user still wins. It was applied inside the
    // backend's init, after placement had reserved for the compiled 512 -- harmless only
    // while every stored chunk was 512.
    if std::env::var("IMPARO_BATCH").is_err() {
        if let Some(batch) = be.stored_prefill_batch(weights.model_bytes()) {
            unsafe { std::env::set_var("IMPARO_BATCH", batch.to_string()) };
        }
    }
    let max_batch = crate::prefill_batch();
    #[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
    install_cuda_correctness_identity_if_needed(weights, plan)?;
    // Basis metadata changes model mathematics. Resolve and validate it before any
    // device allocation, then register it on the initialized backend before execution.
    let input_transforms = if plan.config.architecture == "qwen35" {
        let document =
            imparo_gguf::read(weights.source_path()).map_err(|e| e.to_string())?;
        let transforms = crate::qwen35::weight_basis::from_document(
            &document,
            weights.source_path(),
        )?;
        for transform in &transforms {
            if !weights.tensors.values().any(|tensor| {
                tensor.offset as u64 == transform.weight_offset
                    && tensor.ne[0] == u64::from(transform.width)
            }) {
                return Err(
                    "Prism transform no longer names the loaded weight mapping".into(),
                );
            }
        }
        transforms
    } else {
        Vec::new()
    };
    // BEFORE init_weights: the activation SPECIALISES the epilogue kernels at pipeline
    // build, and setting it afterwards is a silent no-op that would leave GELU.
    be.set_activation(model_activation(plan)?.epilogue());
    // BEFORE init_weights for the same reason: the attention kernels are specialised per
    // head dim when the library is compiled, and this model's dims are what should be
    // compiled. Taken from the PLAN's layers rather than a model-level field, because a
    // network can carry several geometries -- gemma4's windowed layers are head_dim 256
    // and its full layers 512 -- which is what the LayerPlan comment says a model-level
    // field cannot express.
    // Each dim goes with its K/V row width (see below), from the target's layers and from a
    // paired drafter's, whose attention runs on the same library.
    let mut dims: Vec<(u32, u32)> = plan
        .layers
        .iter()
        .filter(|l| l.attention.is_attention())
        .map(|l| {
            let hd = l.attention.head_dim();
            (hd, plan.config.n_kv_heads * hd)
        })
        .chain(plan.drafter.iter().map(|d| (d.head_dim, d.kv_width())))
        .collect();
    dims.sort_unstable();
    dims.dedup();
    if let Some(pair) = dims.windows(2).find(|p| p[0].0 == p[1].0) {
        return Err(format!(
            "head dim {} comes with two K/V row widths ({} and {}); the attention library \
             compiles one width per head dim",
            pair[0].0, pair[0].1, pair[1].1
        ));
    }
    let hds: Vec<u32> = dims.iter().map(|d| d.0).collect();
    // WHICH GGML TYPE EACH WIRE KIND IS. A backend switches kernels on the wire kind and
    // decodes by the ggml type; handing over the pairs keeps the mapping stated once, in
    // imparo-gguf, instead of copied into every backend.
    be.set_weight_kind_types(&imparo_gguf::weights::wire_kind_types());
    be.set_attention_head_dims(&hds);
    // The recurrent MATRIX's coordinates, same timing and same reason: the delta kernel
    // holds one of its rows per lane in registers, and a register array's size must be a
    // constant expression. A plan with no such layer sets nothing, so a probe that set the
    // dims itself (imparo-metalbench) keeps them; unset, no delta pipeline is built.
    if let Some((rec_k, rec_v)) = plan.recurrent_dims()? {
        be.set_recurrent_dims(rec_k, rec_v);
    }
    // The K/V row width goes with each dim: the kv heads of the plan that owns the dim x the
    // head dim. Every K/V tile load in the
    // prefill attention kernels carries that stride, and compiled as a constant it is an
    // immediate in the address arithmetic instead of a uniform read per tile.
    let kvws: Vec<u32> = dims.iter().map(|d| d.1).collect();
    be.set_attention_kv_widths(&kvws);
    // WHERE THE WEIGHTS LIVE (docs/memory-tiers-and-fit.md): the common runtime decides,
    // once, from the plan, the configured context, the prefill batch and the backend's
    // budget; the backend implements the tiers. The decision is printed unconditionally --
    // a model with layers in the slow tier must say so, and a model that fits must be seen
    // to (the no-regression guard reads this line).
    // Every weight type in the file must have a kernel on this backend; a tile-major kind
    // without readers here (a converted file meant for another backend) is refused by name.
    for (name, t) in &weights.tensors {
        // The RUNTIME type, not the file's: a load-time repack changes what the backend is
        // handed, so asking about the file's type refused a k-quant the transform was about
        // to turn into a layout this backend reads. `runtime_weight_type` is the same
        // derivation the transform itself uses, so the two cannot disagree.
        let dims: Vec<u64> = t.ne[..t.n_dims as usize].to_vec();
        let runtime = crate::backend::runtime_weight_type(name, t.ggml_type, &dims);
        if !be.serves_weight_type(runtime) {
            let named =
                |k: u32| imparo_gguf::tensor_layout(k).map_or("unknown", |l| l.name);
            let becomes = if runtime == t.ggml_type {
                String::new()
            } else {
                format!(" (repacked to {} ({runtime}))", named(runtime))
            };
            return Err(format!(
                "{name} has ggml type {} ({}){becomes}, which the {} backend has no \
                 kernels for",
                t.ggml_type,
                named(t.ggml_type),
                be.device_tag()
            ));
        }
    }
    let (spread, max_batch, reserve_scratch) = load_time_spread_schedule(
        be.device_tag().starts_with("cuda"), weights.model_bytes(), max_batch,
    )?;
    let placement = crate::placement::plan_placement_schedule(
        &weights.tensors, appended, plan, capacity, max_batch, be.fast_tier_budget(), spread,
        reserve_scratch,
    );
    if spread { eprintln!("[imparo] load-time weight_transfer_policy=2 placement=spread-by-layer-bytes"); }
    eprintln!(
        "[imparo] {}",
        crate::placement::describe(&placement, capacity, max_batch)
    );
    // Only a backend whose committed KV follows the blocks written can hold a device tier
    // larger than one conversation without paying for it before it is used.
    if be.kv_commits_on_demand() {
        if let Some(tier) = crate::placement::kv_tier_bytes(&placement) {
            crate::placement::set_kv_tier(tier);
            eprintln!(
                "[imparo] kv tier: {:.0} MiB (the fast tier less its weights, activations, \
scratch and margin)",
                tier as f64 / (1u64 << 20) as f64
            );
        }
    }
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
    // Before the mapping is handed over: the repack reads converted tensors from the FILE,
    // not through the mapping, so those bytes never enter the page cache.
    be.set_weight_path(weights.source_path());
    be.set_model_bytes(weights.model_bytes());
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
            be.register_weight_input_transforms(&input_transforms)?;
            #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
            imparo_cuda::knobs::prepare_e4b_retained_decode_policy(
                u32::try_from(capacity)
                    .map_err(|_| "E4B context capacity exceeds u32")?,
                plan.config.n_embd,
                plan.config.n_ff,
                plan.config.n_layers,
                plan.config.n_heads,
                plan.config.n_kv_heads,
            )
            .map_err(|rc| {
                format!("retained E4B policy domain admission failed rc={rc}")
            })?;
            weights.mark_gpu();
            eprintln!(
                "[imparo] {}: weights shared, GPU path enabled",
                be.device_tag()
            );
            #[cfg(feature = "cuda-speculative")]
            if LOAD_TIME_Q8_REPACK_MODE.load(std::sync::atomic::Ordering::Relaxed) == 5 {
                let registry = cuda_knob_registry();
                let knob = registry.iter().find(|k| k.name == "mmq_q8_canonical_gate_up_pair")
                    .ok_or("missing load-time Q8 layout selector")?;
                (knob.apply)(5);
            }
            load_time_repack(weights, &placement, be)?;
            #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
            if imparo_cuda::knobs::lfm_retained_execution_enabled() {
                // Inspect actual post-repack tensor layouts, not a model filename.
                let down_kind = if capacity == 1024 { 3 } else { 2 };
                for layer in 0..plan.config.n_layers {
                    for (role, kind, width, rows) in [
                        ("gate", 3, 2048, 10752),
                        ("up", 3, 2048, 10752),
                        ("down", down_kind, 10752, 2048),
                    ] {
                        let name = format!("blk.{layer}.ffn_{role}.weight");
                        let tensor = weights
                            .tensors
                            .get(&name)
                            .ok_or_else(|| format!("retained LFM missing {name}"))?;
                        if tensor.n_dims != 2
                            || tensor.ne[0] != width
                            || tensor.ne[1] != rows
                            || imparo_gguf::weights::weight_kind(tensor.ggml_type)
                                .map(|k| k as u32)
                                != Some(kind)
                        {
                            return Err(format!(
                                "retained LFM tensor layout mismatch: {name}"
                            ));
                        }
                    }
                }
                imparo_cuda::knobs::prepare_lfm_retained_policy(
                    u32::try_from(capacity).map_err(|_| "LFM capacity exceeds u32")?,
                    u32::try_from(max_batch).map_err(|_| "LFM batch exceeds u32")?,
                    plan.config.n_embd,
                    plan.config.n_ff,
                    plan.config.n_layers,
                    plan.config.n_heads,
                    plan.config.n_kv_heads,
                    down_kind,
                )
                .map_err(|rc| {
                    format!("retained LFM domain admission failed rc={rc}")
                })?;
            }
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

/// A rule as the `Backend` seam's data. THE one place `TmRule` crosses into a backend:
/// the scale spans first, then the payload spans in source order, so a repack can move
/// them without knowing which format it is moving.
///
/// # Panics
/// When a rule needs more spans than the seam carries. A format the seam cannot describe
/// must not be repacked at some other layout by accident.
#[must_use]
pub fn block_layout(
    rule: &imparo_gguf::weights::TmRule,
) -> imparo_backend::WeightBlockLayout {
    let pay = rule.payload_spans();
    let n = rule.scale_spans.len() + pay.len();
    assert!(
        n <= imparo_backend::MAX_BLOCK_SPANS,
        "{}: {n} spans exceeds the backend seam's {}",
        rule.to_name,
        imparo_backend::MAX_BLOCK_SPANS
    );
    let mut l = imparo_backend::WeightBlockLayout {
        block_elems: rule.block_elems as u32,
        block_bytes: rule.block_bytes() as u32,
        unit_rows: rule.unit_rows as u32,
        n_spans: n as u32,
        n_scale_spans: rule.scale_spans.len() as u32,
        ..Default::default()
    };
    for (i, &(off, len)) in rule.scale_spans.iter().chain(pay.iter()).enumerate() {
        l.span_off[i] = off as u32;
        l.span_len[i] = len as u32;
    }
    l
}

pub fn load_time_rule(
    name: &str,
    ggml_type: u32,
    dims: &[u64],
) -> Option<&'static imparo_gguf::weights::TmRule> {
    if std::env::var("IMPARO_LOAD_REPACK").is_ok_and(|v| v == "0") {
        return None;
    }
    let rule = imparo_gguf::weights::tm_applies(name, ggml_type, dims).ok()?;
    if !rule.has_readers_for(dims) {
        return None;
    }
    let be = active()?;
    if !be.supports_load_time_repack() || !be.serves_weight_type(rule.to) { return None; }
    if be.q8_split_gate_up_repack() {
        if ggml_type != 8 || dims.len() != 2 || dims[0] % 256 != 0 || dims[1] % 128 != 0
            || !(name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_up.weight")) {
            return None;
        }
        return Some(&imparo_gguf::weights::CUDA_Q8_SPLIT_RULE);
    }
    Some(rule)
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
    let mut selected_rules = Vec::new();
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
        selected_rules.push(rule);
        jobs.push(imparo_backend::WeightTransform {
            offset: t.offset as u64,
            bytes: t.bytes as u64,
            from_type: rule.from,
            to_type: rule.to,
            n_in: t.ne[0] as u32,
            // EVERY ROW OF THE STACK, not one slice's. An expert tensor is
            // `[n_in, n_out, n_expert]` -- n_expert matrices end to end -- and the
            // transform permutes bytes inside a unit of `unit_rows`, so a stack whose
            // slice rows divide the unit converts as one tall matrix. Passing ne[1] here
            // converted expert 0 and left the other 31 row-major; the verify named it
            // exactly, "differs from the rule at byte 2064384 of 66060288", which is where
            // expert 1 begins.
            n_out: (t.ne[1] * t.ne.get(2).copied().unwrap_or(1)) as u32,
            layout: block_layout(rule),
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
    let mut kinds: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    for (i, job) in jobs.iter().enumerate() {
        if !applied[i] {
            continue;
        }
        let rule = selected_rules[i];
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

/// Static CUDA knob registry used when loading an explicit draft profile.
#[cfg(feature = "cuda-speculative")]
pub fn cuda_knob_registry() -> &'static [imparo_backend::KnobDecl] {
    use imparo_backend::BackendKnobs as _;
    imparo_cuda::CudaBackend.knob_registry()
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

#[cfg(feature = "cuda-speculative")]
static LOAD_TIME_TRANSFER_POLICY: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(u32::MAX);

#[cfg(feature = "cuda-speculative")]
static LOAD_TIME_Q8_REPACK_MODE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

#[cfg(feature = "cuda-speculative")]
fn lab_knob_values() -> Result<std::collections::BTreeMap<String, u32>, String> {
    // One existing registry/loader for every laboratory model. Keep the old
    // DSpark environment name as an alias; conflicting sources are an error.
    let generic = std::env::var_os("IMPARO_LAB_KNOBS");
    let legacy = std::env::var_os("IMPARO_DSPARK_KNOBS");
    if generic.is_some() && legacy.is_some() {
        return Err(
            "IMPARO_LAB_KNOBS and IMPARO_DSPARK_KNOBS are mutually exclusive".into(),
        );
    }
    let Some(path) = generic.or(legacy) else {
        return Ok(std::collections::BTreeMap::new());
    };
    let values: std::collections::BTreeMap<String, u32> =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    Ok(values)
}

fn load_time_spread_schedule(
    cuda: bool, model_bytes: u64, max_batch: usize,
) -> Result<(bool, usize, u64), String> {
    #[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
    if cuda {
        let (value, stored_batch) = imparo_cuda::prepare_host_config(model_bytes)?;
        // Use the same validated snapshot and explicit-override precedence as
        // native initialization. Fit must not reserve the default batch while
        // the activation allocator later uses the tuned batch.
        let max_batch = if std::env::var("IMPARO_BATCH").is_err() {
            stored_batch.unwrap_or(max_batch)
        } else { max_batch };
        #[cfg(feature = "cuda-speculative")]
        let (value, router_override, down_override, gateup_override, down_mmvq_override, gateup_mmvq_override) = {
            let values = lab_knob_values()?;
            LOAD_TIME_Q8_REPACK_MODE.store(values.get("mmq_q8_canonical_gate_up_pair").copied().unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
            (values.get("weight_transfer_policy").copied().unwrap_or(value),
                values.get("moe_router_f32").copied(), values.get("moe_down_mmq").copied(),
                values.get("moe_gateup_mmq").copied(), values.get("moe_down_mmvq").copied(),
                values.get("moe_gateup_mmvq").copied())
        };
        #[cfg(not(feature = "cuda-speculative"))]
        let (router_override, down_override, gateup_override, down_mmvq_override, gateup_mmvq_override) = (None, None, None, None, None);
        if value > 2 { return Err("invalid load-time weight transfer policy".into()); }
        let reserve_scratch = imparo_cuda::prepare_moe_router_scratch(router_override, down_override, gateup_override, down_mmvq_override, gateup_mmvq_override)?;
        #[cfg(feature = "cuda-speculative")]
        LOAD_TIME_TRANSFER_POLICY.store(value, std::sync::atomic::Ordering::Relaxed);
        return Ok((value == 2, max_batch, reserve_scratch));
    }
    let _ = (cuda, model_bytes);
    Ok((false, max_batch, 0))
}

/// Explicit laboratory overrides shared by server and correctness CLI.
/// With neither legacy nor generic lab variable set, normal receipt loading is unchanged.
#[cfg(feature = "cuda-speculative")]
pub fn apply_lab_knobs_from_env() -> Result<(), String> {
    let values = lab_knob_values()?;
    imparo_cuda::validate_moe_router_scratch(
        values.get("moe_router_f32").copied(), values.get("moe_down_mmq").copied(),
        values.get("moe_gateup_mmq").copied(),
        values.get("moe_down_mmvq").copied(),
        values.get("moe_gateup_mmvq").copied(),
    )?;
    let layout = LOAD_TIME_Q8_REPACK_MODE.load(std::sync::atomic::Ordering::Relaxed);
    let requested_layout = values.get("mmq_q8_canonical_gate_up_pair").copied().unwrap_or(layout);
    if (layout == 5 || requested_layout == 5) && layout != requested_layout {
        return Err("Q8 layout selector changed after load-time preparation".into());
    }
    let planned = LOAD_TIME_TRANSFER_POLICY.load(std::sync::atomic::Ordering::Relaxed);
    let requested = values.get("weight_transfer_policy").copied().unwrap_or(planned);
    if (planned == 2 || requested == 2) && planned != requested {
        return Err("weight transfer policy changed after load-time placement".into());
    }
    let registry = cuda_knob_registry();
    for (name, value) in values {
        let knob = registry
            .iter()
            .find(|k| k.name == name)
            .ok_or_else(|| format!("unknown knob {name}"))?;
        (knob.apply)(value);
        if (knob.current)() != value {
            return Err(format!("knob {name} rejected {value}"));
        }
        eprintln!("[imparo] lab existing knob {name}={value}");
    }
    Ok(())
}

#[cfg(not(feature = "cuda-speculative"))]
pub fn apply_lab_knobs_from_env() -> Result<(), String> {
    if std::env::var_os("IMPARO_LAB_KNOBS").is_some()
        || std::env::var_os("IMPARO_DSPARK_KNOBS").is_some() {
        return Err("CUDA laboratory knob overrides require cuda-speculative".into());
    }
    Ok(())
}
