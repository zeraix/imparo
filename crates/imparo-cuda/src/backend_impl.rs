//! `Backend` for CUDA: FFI onto native/imparo_cuda.cu's extern "C" surface.
//! WRITTEN UNVERIFIED on a Mac (task #19) -- a CUDA host compiles and gates it.
//!
//! Per-op status (implemented-unverified means: real code, contracts mirrored from
//! the Metal backend, never run under nvcc here):
//!
//! | op                          | status                  |
//! |-----------------------------|-------------------------|
//! | buffers/arena/kv alloc+grow | implemented, unverified |
//! | matmat q4_0 (GEMV + GEMM + fused epilogue) | implemented, unverified; UNTUNED |
//! | matmat_gated               | decode Q4_0/GELU fusion; explicit fallback otherwise |
//! | matmat f32                  | implemented, unverified |
//! | rms_norm (+from)            | implemented, unverified |
//! | rms_norm_add               | quarantined: opt-in diagnostic; safe split fallback by default |
//! | rope (neox, freqs)          | implemented, unverified (freqs upload TODO)     |
//! | hadamard (FWHT)             | implemented, unverified |
//! | kv_store f16/q4_0/q8_0      | implemented, unverified |
//! | attention D512 / D256       | tiled D512; D256 conservative fallback, tiled quarantined |
//! | elementwise + softcap       | implemented, unverified |
//! | argmax                      | implemented, unverified |
//! | row (embedding)             | implemented, unverified |
//! | gather_rows                 | branch kernel wired; Q4_0, offset-zero shapes only |
//! | matmat_from                 | implemented via matmat's src_row |
//! | ple_gather_combine          | SM86 whole-model LFM2 Q8 agreement verified |
//! | shortconv (+snapshot)       | SM86 split, decode and restart lifecycle verified |
//! | SiLU act / act_mul          | standalone CUDA path; fused SiLU unsupported and routed safely |
//! | block-table state/ABI      | store, dequant and all attention K/V reads mapped; paged_reads=true |
//! | KV regions / advise         | region 0; lifecycle-checked fixed-offset ownership |
//! | KV tier movement            | transactional Device <-> pinned Host; durable Disk handoff in server |
//! | tuner measurement methods   | explicit zero: native probes land in Step 7 |
//! | device tag / allocation     | native driver query, wired through `CudaContext` |
//! | set_epilogue / flush_layers | implemented (flush policy = compiled default) |
//!
//! The half-A mirror (XH/XH2) and the KVQ diagnostics are Metal-side host logic and
//! do not exist here; a CUDA port decides its own activation-precision staging.

use crate::ffi::*;
use core::mem::size_of;
use imparo_backend::{
    Backend, BatchGeometry, BufId, Epilogue, HadamardWidth, HostTierProfile,
    KvByteCodec, KvByteCodecRoute, KvHostHandle, KvLayout, KvQuantizationRoute,
    KvTransferSpan, QuantizedWeightCachePlan, StreamedWeightSpan, WeightKindWire,
    Workload,
};
use std::path::Path;
use std::sync::{
    OnceLock,
    atomic::{AtomicU32, Ordering},
};

static ACTIVATION: AtomicU32 = AtomicU32::new(Epilogue::Gelu as u32);
static KV_K: AtomicU32 = AtomicU32::new(1);
static KV_V: AtomicU32 = AtomicU32::new(1);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CorrectnessModelIdentity {
    model_sha256: [u8; 32],
    model_plan_sha256: [u8; 32],
    kv_layout_sha256: [u8; 32],
}

static CORRECTNESS_MODEL_IDENTITY: OnceLock<CorrectnessModelIdentity> = OnceLock::new();
// Captured immediately after native SM discovery and before any stored config is
// applied. A future same-process reload must start here; otherwise a rejected second
// config could leave numerical knobs from an earlier admitted receipt live.
static CUDA_SAFE_DEFAULTS: OnceLock<Vec<(&'static str, u32)>> = OnceLock::new();

fn capture_cuda_safe_defaults() {
    use imparo_backend::BackendKnobs as _;
    let _ = CUDA_SAFE_DEFAULTS.set(
        CudaBackend
            .knob_registry()
            .iter()
            .map(|declaration| (declaration.name, (declaration.current)()))
            .collect(),
    );
}

fn reset_cuda_safe_defaults() {
    use imparo_backend::BackendKnobs as _;
    let Some(defaults) = CUDA_SAFE_DEFAULTS.get() else {
        return;
    };
    let registry = CudaBackend.knob_registry();
    for &(name, value) in defaults {
        let declaration = registry
            .iter()
            .find(|declaration| declaration.name == name)
            .expect("captured CUDA default must remain in the same registry version");
        (declaration.apply)(value);
    }
}

/// Bind any candidate host config to the complete mapped model and semantic plan.
/// Reinstalling the same one-model-per-process identity is harmless; a different model
/// is rejected before native initialization.
pub fn install_correctness_identity(
    model_sha256: [u8; 32],
    model_plan_sha256: [u8; 32],
    kv_layout_sha256: [u8; 32],
) -> Result<(), String> {
    let identity = CorrectnessModelIdentity {
        model_sha256,
        model_plan_sha256,
        kv_layout_sha256,
    };
    if let Some(installed) = CORRECTNESS_MODEL_IDENTITY.get() {
        return if *installed == identity {
            Ok(())
        } else {
            Err(
                "a different CUDA correctness model identity is already installed"
                    .into(),
            )
        };
    }
    CORRECTNESS_MODEL_IDENTITY
        .set(identity)
        .map_err(|_| "CUDA correctness model identity raced during install".into())
}

fn correctness_math_mode() -> crate::correctness::CudaMathMode {
    match crate::CUDA_MATH_MODE {
        "precise" => crate::correctness::CudaMathMode::Precise,
        _ => crate::correctness::CudaMathMode::Fast,
    }
}

fn kv_name(kind: u32) -> &'static str {
    match kind {
        2 => "q4_0",
        8 => "q8_0",
        _ => "f16",
    }
}

fn configured_kv_tag() -> String {
    let k = KV_K.load(Ordering::Relaxed);
    let v = KV_V.load(Ordering::Relaxed);
    imparo_host::canonical_kv_tag(kv_name(k), kv_name(v))
}

fn kv_code(value: Option<&str>) -> u32 {
    match value {
        Some("q4_0") => 2,
        Some("q8_0") => 8,
        _ => 1,
    }
}

fn prefill_dequant_slots(start_pos: u32, n_tok: u32, ring: u32) -> u32 {
    let initialized = start_pos.saturating_add(n_tok);
    if ring == 0 {
        initialized
    } else {
        initialized.min(ring.saturating_add(1))
    }
}

/// Probe/dev binaries configure KV storage through the environment. Production
/// callers normally invoke `Backend::set_kv_types` before initialization, so only
/// override the native state when at least one environment variable is present.
/// This must happen before host-config lookup and native allocation: Rust sizes the
/// KV buffers from the same variables, and a stale f16 native default would write past
/// a q4/q8 allocation.
fn apply_env_kv_types() {
    let k = std::env::var("IMPARO_CTK").ok();
    let v = std::env::var("IMPARO_CTV").ok();
    if k.is_none() && v.is_none() {
        return;
    }
    let k = kv_code(k.as_deref());
    let v = kv_code(v.as_deref());
    KV_K.store(k, Ordering::Relaxed);
    KV_V.store(v, Ordering::Relaxed);
    crate::context::CudaContext::get().set_kv_types(k, v);
}

/// Uploads the weight blob; call once before any op. Applies this host's stored
/// tuned configuration first (unless IMPARO_NO_HOSTCONFIG -- the tuner measures
/// from compiled defaults), mirroring the Metal backend's init.
pub fn init(weights: &[u8]) -> Result<(), i32> {
    init_with_streamed(weights, &[])
}

fn init_with_streamed(
    weights: &[u8],
    streamed: &[StreamedWeightSpan],
) -> Result<(), i32> {
    apply_env_kv_types();
    let capacity = unsafe { imparo_cuda_buf_count() } as usize;
    if let Err(message) = imparo_backend::check_buf_table("cuda", capacity) {
        eprintln!("[imparo] {message}");
        return Err(1);
    }
    // Native init discovers the live SM and installs architecture defaults. A stored
    // config is applied only afterwards so it cannot be overwritten by that discovery.
    unsafe {
        crate::context::CudaContext::get().init_weights(
            weights.as_ptr(),
            weights.len() as u64,
            streamed,
        )
    }?;
    capture_cuda_safe_defaults();
    if std::env::var("IMPARO_NO_HOSTCONFIG").is_err() {
        if let Some(batch) = apply_host_config(weights.len() as u64) {
            if std::env::var("IMPARO_BATCH").is_err() {
                unsafe { std::env::set_var("IMPARO_BATCH", batch.to_string()) };
            }
        }
    }
    Ok(())
}

/// Registry-driven config apply -- the CUDA analogue of imparo-metal's: every stored
/// key goes through the registry's declared hook, so a knob the tuner writes cannot
/// be silently dropped here. UNVERIFIED on real hardware, like the rest of this crate.
#[must_use]
pub fn apply_host_config(model_bytes: u64) -> Option<usize> {
    use imparo_backend::BackendKnobs as _;

    // Reset before every lookup and before every possible early return. Startup and a
    // future hot reload therefore share the same fail-closed behavior: missing/torn/
    // stale receipts cannot retain a numerical route admitted by an older snapshot.
    reset_cuda_safe_defaults();

    let space = CudaBackend.space_version();
    let kv = configured_kv_tag();
    let fp = imparo_host::fingerprint_for("cuda", space, &kv);
    let config_path = imparo_host::selected_config_path(&fp, model_bytes);
    if !config_path.is_file() {
        return None;
    }
    let model = *CORRECTNESS_MODEL_IDENTITY.get().or_else(|| {
        eprintln!(
            "[imparo] CUDA tuned config rejected: complete model identity unavailable; \
             using safe defaults"
        );
        None
    })?;
    let runtime = crate::runtime_identity().map_err(|error| {
        eprintln!(
            "[imparo] CUDA tuned config rejected: runtime identity unavailable ({error}); \
             using safe defaults"
        );
    }).ok()?;
    let correctness = crate::correctness::CudaCorrectnessIdentity {
        runtime: &runtime,
        model_sha256: model.model_sha256,
        model_plan_sha256: model.model_plan_sha256,
        kv_layout_sha256: model.kv_layout_sha256,
        kv_k: kv_name(KV_K.load(Ordering::Relaxed)),
        kv_v: kv_name(KV_V.load(Ordering::Relaxed)),
        math_mode: correctness_math_mode(),
    };

    let gate_mode = std::env::var("IMPARO_CORRECTNESS_GATE").as_deref() == Ok("1");
    let stored = if gate_mode {
        let result = imparo_host::receipted_config::inspect_unreceipted_for_gate_at(
            &config_path,
            model_bytes,
            ("cuda", space, &kv),
            |candidate| {
                crate::correctness::expected_correctness(candidate, &correctness)
            },
        );
        let (stored, expected) = match result {
            Ok(value) => value,
            Err(error) => {
                eprintln!(
                    "[imparo] isolated CUDA gate candidate rejected ({error}); \
                     using safe defaults"
                );
                return None;
            }
        };
        if expected.routes.len() != 1 || expected.routes[0].operation != "model.forward"
        {
            eprintln!(
                "[imparo] isolated CUDA gate route is incomplete; using safe defaults"
            );
            return None;
        }
        eprintln!(
            "[imparo] WARNING: applying unreceipted CUDA config only because \
             IMPARO_CORRECTNESS_GATE=1"
        );
        stored
    } else {
        let admitted = imparo_host::receipted_config::read_for_device_receipted(
            model_bytes,
            false,
            ("cuda", space, &kv),
            |candidate| {
                crate::correctness::expected_correctness(candidate, &correctness)
            },
        )?;
        let (stored, routes) = admitted.into_parts();
        if routes.len() != 1
            || routes
                .iter()
                .next()
                .is_none_or(|route| route.operation != "model.forward")
        {
            eprintln!(
                "[imparo] CUDA receipt lacks model.forward authority; using safe defaults"
            );
            return None;
        }
        stored
    };

    let registry = CudaBackend.knob_registry();
    if stored.knobs.iter().any(|(name, _)| {
        !registry
            .iter()
            .any(|declaration| declaration.name == name.as_str())
    }) {
        eprintln!(
            "[imparo] CUDA config changed after policy validation; using safe defaults"
        );
        return None;
    }
    for (name, value) in &stored.knobs {
        let declaration = registry
            .iter()
            .find(|declaration| declaration.name == name.as_str())
            .expect("all names validated before applying any knob");
        (declaration.apply)(*value);
    }
    eprintln!("[imparo] host config loaded from {}", config_path.display());
    stored.batch
}

/// Build an unsigned, fail-closed receipt template for one explicit gate candidate.
pub fn correctness_receipt_template(
    config_path: &Path,
    model_bytes: u64,
    model_sha256: [u8; 32],
    model_plan_sha256: [u8; 32],
    kv_layout_sha256: [u8; 32],
    kv_k: &str,
    kv_v: &str,
) -> Result<imparo_host::correctness::CorrectnessReceipt, String> {
    use imparo_backend::BackendKnobs as _;
    if std::env::var("IMPARO_CORRECTNESS_GATE").as_deref() != Ok("1") {
        return Err("--correctness-template requires IMPARO_CORRECTNESS_GATE=1".into());
    }
    let runtime = crate::runtime_identity()?;
    let identity = crate::correctness::CudaCorrectnessIdentity {
        runtime: &runtime,
        model_sha256,
        model_plan_sha256,
        kv_layout_sha256,
        kv_k,
        kv_v,
        math_mode: correctness_math_mode(),
    };
    let space = CudaBackend.space_version();
    let kv = imparo_host::canonical_kv_tag(kv_k, kv_v);
    let (_, expected) = imparo_host::receipted_config::inspect_unreceipted_for_gate_at(
        config_path,
        model_bytes,
        ("cuda", space, &kv),
        |candidate| crate::correctness::expected_correctness(candidate, &identity),
    )?;
    Ok(crate::correctness::receipt_skeleton(&expected))
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBackend;

/// Enable the static CUDA kernel-lab timing surface before backend initialization.
/// Dynamic release plugins deliberately retain the no-op shim until a future ABI owns
/// event timing; this function never mutates production selector state.
pub fn prepare_tuner_lab() {
    unsafe { imparo_cuda_enable_tuner_lab() }
}

#[inline]
fn b(id: BufId) -> u32 {
    id as u32
}

fn ffn_sidecar_requested(n_tok: u32) -> bool {
    if !(9..=512).contains(&n_tok) {
        return false;
    }
    let lab_requested =
        std::env::var("IMPARO_CUDA_PREFILL_FFN_SIDECAR_ONLY_LAB").as_deref() == Ok("1");
    let tuned_min_tokens = crate::knobs::ffn_sidecar_min_tokens();
    let exact128 = n_tok == 128
        && (crate::knobs::prefill_exact128_sm86_route_enabled()
            || crate::knobs::prefill_exact128_fast_transaction_enabled());
    lab_requested || exact128 || (tuned_min_tokens != 0 && n_tok >= tuned_min_tokens)
}

fn split_prefill_graph_requested(count: u32) -> bool {
    let exact128_sidecar_graph = count == 128
        && (crate::knobs::prefill_exact128_sm86_route_enabled()
            || crate::knobs::prefill_exact128_fast_transaction_enabled())
        && (crate::knobs::prefill_exact128_graph_enabled()
            || std::env::var_os("IMPARO_CUDA_PREFILL_GRAPH_LAB").is_some());
    let sidecar_graph_lab = count == 128
        && ffn_sidecar_requested(count)
        && std::env::var("IMPARO_CUDA_PREFILL_SIDECAR_GRAPH_LAB").as_deref() == Ok("1");
    exact128_sidecar_graph || sidecar_graph_lab
}

#[allow(unused_variables)]
impl Backend for CudaBackend {
    fn last_gpu_us(&self) -> f64 {
        unsafe { imparo_cuda_last_gpu_us() }
    }
    fn tuner_route_evidence(&self, workload: Workload) -> u32 {
        match workload {
            Workload::PrefillFfnExact128 => unsafe {
                imparo_cuda_exact128_route_hits_lab()
            },
            _ => 0,
        }
    }
    fn device_profile(&self) -> imparo_backend::DeviceProfile {
        let mut wire = DeviceProfileWire {
            struct_bytes: size_of::<DeviceProfileWire>() as u32,
            ..DeviceProfileWire::default()
        };
        let rc = unsafe {
            imparo_cuda_device_profile(
                &raw mut wire,
                size_of::<DeviceProfileWire>() as u32,
            )
        };
        if rc != 0
            || wire.struct_bytes != size_of::<DeviceProfileWire>() as u32
            || wire.max_threads == 0
            || wire.threadgroup_bytes == 0
        {
            return imparo_backend::DeviceProfile::default();
        }
        imparo_backend::DeviceProfile {
            threadgroup_bytes: wire.threadgroup_bytes,
            max_threads: wire.max_threads,
            ..imparo_backend::DeviceProfile::default()
        }
    }
    fn spill_rate(&self, idx: u32, tgs: u32, tpg: u32, iters: u32) -> f64 {
        unsafe { imparo_cuda_probe(1, u64::from(idx), tgs, tpg, iters, 0) }
    }
    fn bw_read(&self, bytes: u64, reps: u32, tgs: u32, tpg: u32) -> f64 {
        unsafe { imparo_cuda_probe(2, bytes, reps, tgs, tpg, 0) }
    }
    fn scoremix_rate(
        &self,
        tgs: u32,
        sgs: u32,
        iters: u32,
        stride: u32,
        kspan: u32,
    ) -> f64 {
        unsafe { imparo_cuda_probe(3, u64::from(tgs), sgs, iters, stride, kspan) }
    }
    fn commit_overhead(&self, n: u32) -> f64 {
        unsafe { imparo_cuda_probe(4, u64::from(n), 0, 0, 0, 0) }
    }
    fn sync_overhead(&self, n: u32) -> f64 {
        unsafe { imparo_cuda_probe(5, u64::from(n), 0, 0, 0, 0) }
    }
    fn encode_cost(&self, n: u32) -> f64 {
        unsafe { imparo_cuda_probe(6, u64::from(n), 0, 0, 0, 0) }
    }
    fn kv_tag(&self) -> String {
        configured_kv_tag()
    }
    fn kv_quantization_route(&self) -> KvQuantizationRoute {
        // Pinned llama 4695f001 uses the full head for K and a fixed 64-value
        // block for V. Raw-cache A/B on SM86 makes layer-0 V 16,299/16,304
        // Q4 blocks byte-identical (versus 0/16,304 for the shared V=128 route).
        // Keep this CUDA-local: Metal retains Backend's established V=128 default.
        KvQuantizationRoute {
            key: HadamardWidth::FullHead,
            value: HadamardWidth::Fixed(64),
        }
    }
    fn kv_quantization_route_override(&self) -> Option<KvQuantizationRoute> {
        Some(self.kv_quantization_route())
    }
    fn kv_byte_codec_route(&self) -> KvByteCodecRoute {
        KvByteCodecRoute {
            q8_0: KvByteCodec::Q8_0RoundAwayV1,
            ..KvByteCodecRoute::default()
        }
    }
    fn begin(&self) {
        unsafe { imparo_cuda_begin() }
    }
    fn begin_forward(&self, decode: bool) {
        unsafe { imparo_cuda_begin_forward(u32::from(decode)) }
    }
    fn decode_prepare(
        &self,
        token: u32,
        start_pos: u32,
        argmax: bool,
    ) -> Result<bool, i32> {
        match unsafe { imparo_cuda_decode_prepare(token, start_pos, u32::from(argmax)) }
        {
            0 => Ok(false),
            1 => Ok(true),
            rc if rc < 0 => Err(-rc),
            rc => Err(rc),
        }
    }
    fn prefill_prepare(
        &self,
        tokens: &[u32],
        start_pos: u32,
        argmax: bool,
    ) -> Result<bool, i32> {
        let count = u32::try_from(tokens.len()).map_err(|_| 3)?;
        // Generic/diagnostic sidecars remain outside Graph. The receipted exact-128
        // route may enter through its versioned safe-off knob or the explicit lab
        // override. Native admission still requires a complete model hot set and two
        // ordinary warm forwards before capture. No other production shape is widened.
        if ffn_sidecar_requested(count) && !split_prefill_graph_requested(count) {
            return Ok(false);
        }
        match unsafe {
            imparo_cuda_prefill_prepare(
                tokens.as_ptr(),
                count,
                start_pos,
                u32::from(argmax),
            )
        } {
            0 => Ok(false),
            1 => Ok(true),
            rc if rc < 0 => Err(-rc),
            rc => Err(rc),
        }
    }
    fn prefill_body_prepare(
        &self,
        tokens: &[u32],
        start_pos: u32,
        argmax: bool,
    ) -> Result<bool, i32> {
        let count = u32::try_from(tokens.len()).map_err(|_| 3)?;
        if !split_prefill_graph_requested(count) {
            return Ok(false);
        }
        match unsafe {
            imparo_cuda_prefill_prepare(
                tokens.as_ptr(),
                count,
                start_pos,
                u32::from(argmax),
            )
        } {
            0 => Ok(false),
            1 => Ok(true),
            rc if rc < 0 => Err(-rc),
            rc => Err(rc),
        }
    }
    fn set_batch_geometry(&self, geometry: BatchGeometry) -> Result<(), i32> {
        match unsafe {
            imparo_cuda_set_batch_geometry(
                geometry.absolute_start,
                geometry.active_tokens,
                geometry.phase as u32,
            )
        } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn flush(&self) {
        unsafe { imparo_cuda_flush() }
    }
    fn end(&self) -> Result<(), i32> {
        match unsafe { imparo_cuda_end() } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn set_tuner_mode(&self, enabled: bool) -> Result<(), i32> {
        match unsafe { imparo_cuda_set_tuner_mode(u32::from(enabled)) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn tuner_requires_device_timing(&self) -> bool {
        true
    }
    fn validate_tuner_profile(
        &self,
        p: &imparo_backend::DeviceProfile,
    ) -> Result<(), String> {
        let established = p.threadgroup_bytes > 0
            && p.max_threads > 0
            && p.cache_knee_bytes > 0
            && p.dram_read_mbs > 0
            && p.max_accumulators > 0
            && p.attn_score_ceiling_gflops > 0
            && p.fill_threads_membound > 0
            && p.commit_overhead_ns > 0
            && p.encode_cost_ns > 0
            && p.fill_threadgroups > 0;
        if !established {
            return Err(
                "CUDA Step-1 discovery did not establish every required device/profile metric; tuning result will not be written"
                    .into(),
            );
        }
        let sync = self.sync_overhead(4);
        if !sync.is_finite() || sync <= 0.0 {
            return Err(
                "CUDA Step-1 sync-overhead probe failed; tuning result will not be written"
                    .into(),
            );
        }
        Ok(())
    }
    fn reset_tuner_dispatch_proof(&self) {
        unsafe { imparo_cuda_dispatch_proof_reset() }
    }
    fn set_tuner_dispatch_expectation(&self, names: &[&str]) -> Result<(), String> {
        let mut mask = 0_u64;
        for name in names {
            let slot = crate::knobs::slot_for_name(name).ok_or_else(|| {
                format!("CUDA dispatch expectation names unknown knob {name}")
            })?;
            if slot >= 64 {
                return Err(format!(
                    "CUDA knob {name} has unrepresentable slot {slot}"
                ));
            }
            mask |= 1_u64 << slot;
        }
        let rc = unsafe { imparo_cuda_dispatch_expectation(mask) };
        if rc == 0 {
            Ok(())
        } else {
            Err(format!(
                "CUDA dispatch expectation rejected (rc={rc}, mask={mask:#018x})"
            ))
        }
    }
    fn validate_tuner_dispatch_proof(&self) -> Result<(), String> {
        let mut proof = DispatchProofWire {
            struct_bytes: size_of::<DispatchProofWire>() as u32,
            ..DispatchProofWire::default()
        };
        let rc = unsafe {
            imparo_cuda_dispatch_proof(
                &raw mut proof,
                size_of::<DispatchProofWire>() as u32,
            )
        };
        if rc != 0
            || proof.struct_bytes != size_of::<DispatchProofWire>() as u32
            || proof.dispatches == 0
            || proof.observed_choice_epoch != proof.choice_epoch
            || proof.observed_knob_mask & proof.expected_knob_mask
                != proof.expected_knob_mask
        {
            return Err(format!(
                "CUDA dispatch proof missing/stale (rc={rc}, family={}, dispatches={}, choice_epoch={}, observed_epoch={}, expected_knobs={:#018x}, observed_knobs={:#018x})",
                proof.family,
                proof.dispatches,
                proof.choice_epoch,
                proof.observed_choice_epoch,
                proof.expected_knob_mask,
                proof.observed_knob_mask,
            ));
        }
        Ok(())
    }
    fn flush_layers(&self, decode: bool) -> u32 {
        // Stream-queued kernels need no cb-length policy; a registry knob later.
        let _ = decode;
        7
    }
    fn row_local_prefill_tail_rows(&self) -> u32 {
        crate::knobs::row_local_prefill_tail_rows()
    }
    fn alloc(&self, id: BufId, bytes: u64) -> Result<(), i32> {
        match unsafe { imparo_cuda_alloc(b(id), bytes) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn arena(&self, bytes: u64) -> Result<(), i32> {
        match unsafe { imparo_cuda_arena(bytes) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn place(&self, id: BufId, offset: u64, bytes: u64) -> Result<(), i32> {
        match unsafe { imparo_cuda_place(b(id), offset, bytes) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn page_round(&self, n: u64) -> u64 {
        unsafe { imparo_cuda_page_round(n) }
    }
    fn alloc_kv(&self, bytes: &[u64]) -> Result<(), i32> {
        match unsafe { imparo_cuda_alloc_kv(bytes.len() as u32, bytes.as_ptr()) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn grow_kv(&self, bytes: &[u64]) -> Result<(), i32> {
        match unsafe { imparo_cuda_grow_kv(bytes.len() as u32, bytes.as_ptr()) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn alloc_kv_layout(&self, bytes: &[u64], layouts: &[KvLayout]) -> Result<(), i32> {
        if bytes.len() != layouts.len() {
            return Err(2);
        }
        let count = u32::try_from(bytes.len()).map_err(|_| 2)?;
        match unsafe {
            imparo_cuda_alloc_kv_layout(count, bytes.as_ptr(), layouts.as_ptr(), count)
        } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn grow_kv_layout(&self, bytes: &[u64], layouts: &[KvLayout]) -> Result<(), i32> {
        if bytes.len() != layouts.len() {
            return Err(2);
        }
        let count = u32::try_from(bytes.len()).map_err(|_| 2)?;
        match unsafe {
            imparo_cuda_grow_kv_layout(count, bytes.as_ptr(), layouts.as_ptr(), count)
        } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn write(&self, id: BufId, off: u64, src: &[f32]) {
        unsafe { imparo_cuda_write(b(id), off, src.as_ptr(), src.len() as u64) }
    }
    fn write_u32(&self, id: BufId, off: u64, src: &[u32]) {
        unsafe { imparo_cuda_write_u32(b(id), off, src.as_ptr(), src.len() as u64) }
    }
    fn read(&self, id: BufId, off: u64, dst: &mut [f32]) {
        unsafe { imparo_cuda_read(b(id), off, dst.as_mut_ptr(), dst.len() as u64) }
    }
    fn read_kv_bytes(&self, layer: u32, is_v: bool, off: u64, dst: &mut [u8]) {
        let rc = unsafe {
            imparo_cuda_read_kv(
                layer,
                u32::from(is_v),
                off,
                dst.as_mut_ptr(),
                dst.len() as u64,
            )
        };
        assert_eq!(
            rc,
            0,
            "CUDA bounded KV read failed: layer={layer} is_v={is_v} off={off} len={}",
            dst.len()
        );
    }
    fn set_kv_page_table(&self, layer: u32, entries: &[u32]) {
        let count = u32::try_from(entries.len())
            .expect("CUDA KV page table length exceeds u32");
        let pointer = if entries.is_empty() {
            core::ptr::null()
        } else {
            entries.as_ptr()
        };
        let rc = unsafe { imparo_cuda_set_kv_pages(layer, pointer, count) };
        assert_eq!(
            rc, 0,
            "CUDA KV page-table install failed: layer={layer} entries={count}"
        );
    }
    fn write_kv_bytes(&self, layer: u32, is_v: bool, off: u64, src: &[u8]) {
        let rc = unsafe {
            imparo_cuda_write_kv(
                layer,
                u32::from(is_v),
                off,
                src.as_ptr(),
                src.len() as u64,
            )
        };
        assert_eq!(
            rc,
            0,
            "CUDA bounded KV write failed: layer={layer} is_v={is_v} off={off} len={}",
            src.len()
        );
    }
    fn set_kv_region(&self, layer: u32, k_off: u64, v_off: u64) {
        assert!(
            k_off == 0 && v_off == 0,
            "cuda Step 8 is required before selecting nonzero KV regions (layer {layer})"
        );
    }
    fn kv_advise_free(&self, layer: u32, is_v: bool, off: u64, len: u64) {
        let rc =
            unsafe { imparo_cuda_kv_advise_free(layer, u32::from(is_v), off, len) };
        assert_eq!(
            rc, 0,
            "CUDA KV ownership release failed: layer={layer} is_v={is_v} off={off} len={len}"
        );
    }
    fn kv_advise_reuse(&self, layer: u32, is_v: bool, off: u64, len: u64) {
        let rc =
            unsafe { imparo_cuda_kv_advise_reuse(layer, u32::from(is_v), off, len) };
        assert_eq!(
            rc, 0,
            "CUDA KV ownership reuse failed: layer={layer} is_v={is_v} off={off} len={len}"
        );
    }
    fn kv_host_alloc(&self, bytes: u64) -> Result<KvHostHandle, i32> {
        let mut handle = 0_u64;
        match unsafe { imparo_cuda_host_alloc(bytes, &raw mut handle) } {
            0 if handle != 0 => Ok(KvHostHandle(handle)),
            0 => Err(2),
            rc => Err(rc),
        }
    }
    fn kv_host_free(&self, handle: KvHostHandle) -> Result<(), i32> {
        match unsafe { imparo_cuda_host_free(handle.0) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn kv_demote(&self, spans: &[KvTransferSpan]) -> Result<(), i32> {
        let count = u32::try_from(spans.len()).map_err(|_| 2)?;
        let pointer = if spans.is_empty() {
            core::ptr::null()
        } else {
            spans.as_ptr()
        };
        match unsafe { imparo_cuda_kv_demote(pointer, count) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn kv_promote(&self, spans: &[KvTransferSpan]) -> Result<(), i32> {
        let count = u32::try_from(spans.len()).map_err(|_| 2)?;
        let pointer = if spans.is_empty() {
            core::ptr::null()
        } else {
            spans.as_ptr()
        };
        match unsafe { imparo_cuda_kv_promote(pointer, count) } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn kv_host_read(
        &self,
        handle: KvHostHandle,
        off: u64,
        dst: &mut [u8],
    ) -> Result<(), i32> {
        match unsafe {
            imparo_cuda_host_read(handle.0, off, dst.as_mut_ptr(), dst.len() as u64)
        } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn kv_host_write(
        &self,
        handle: KvHostHandle,
        off: u64,
        src: &[u8],
    ) -> Result<(), i32> {
        match unsafe {
            imparo_cuda_host_write(handle.0, off, src.as_ptr(), src.len() as u64)
        } {
            0 => Ok(()),
            rc => Err(rc),
        }
    }
    fn kv_host_allocated_bytes(&self) -> u64 {
        let mut bytes = 0_u64;
        if unsafe { imparo_cuda_host_allocated_bytes(&raw mut bytes) } == 0 {
            bytes
        } else {
            0
        }
    }
    fn kv_host_profile(&self) -> Option<HostTierProfile> {
        let mut wire = HostProfileWire {
            struct_bytes: size_of::<HostProfileWire>() as u32,
            ..HostProfileWire::default()
        };
        let rc = unsafe {
            imparo_cuda_host_profile(&raw mut wire, size_of::<HostProfileWire>() as u32)
        };
        if rc != 0
            || wire.struct_bytes != size_of::<HostProfileWire>() as u32
            || wire.reserved != 0
            || wire.available_host_bytes == 0
            || wire.pinned_h2d_bytes_per_second == 0
            || wire.pinned_d2h_bytes_per_second == 0
        {
            return None;
        }
        Some(HostTierProfile {
            available_host_bytes: wire.available_host_bytes,
            pinned_h2d_bytes_per_second: wire.pinned_h2d_bytes_per_second,
            pinned_d2h_bytes_per_second: wire.pinned_d2h_bytes_per_second,
        })
    }
    fn matmat(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        n_tok: u32,
    ) {
        unsafe {
            imparo_cuda_matmat(wkind, w_off, n_in, n_out, b(src), b(dst), n_tok, 0)
        }
    }
    fn matmat_from(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        n_tok: u32,
        src_row: u32,
    ) {
        unsafe {
            imparo_cuda_matmat(
                wkind,
                w_off,
                n_in,
                n_out,
                b(src),
                b(dst),
                n_tok,
                src_row,
            )
        }
    }
    fn matmat_gated(
        &self,
        gate_kind: WeightKindWire,
        gate_off: u64,
        up_kind: WeightKindWire,
        up_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        tmp: BufId,
        n_tok: u32,
    ) -> bool {
        let activation = ACTIVATION.load(Ordering::Relaxed);
        let q4_gelu_prefill = n_tok > 1
            && [
                "IMPARO_CUDA_PREFILL_PAIR_V1",
                "IMPARO_CUDA_PREFILL_INTERLEAVED_R2_LAB",
                "IMPARO_CUDA_PREFILL_ALIGNED_PACK_LAB",
                "IMPARO_CUDA_PREFILL_Q8_READY_PACK_LAB",
                "IMPARO_CUDA_PREFILL_Q8_READY_BATCHED_LAB",
            ]
            .iter()
            .any(|name| std::env::var_os(name).is_some());
        let q4_gelu = gate_kind == 1
            && up_kind == 1
            && (n_tok == 1 || q4_gelu_prefill)
            && activation == Epilogue::Gelu as u32;
        let q8_tm_silu = gate_kind == 3
            && up_kind == 3
            && n_tok > 8
            && activation == Epilogue::Silu as u32
            && crate::knobs::q8_tm_silu_pair_enabled();
        let q8_tm_decode_silu = gate_kind == 3
            && up_kind == 3
            && n_tok == 1
            && activation == Epilogue::Silu as u32
            && std::env::var_os("IMPARO_GPU_PROBE").is_none()
            && crate::knobs::q8_tm_decode_silu_pair_enabled();
        if std::env::var_os("IMPARO_CUDA_NO_MATMAT_GATED").is_some()
            || (!q4_gelu && !q8_tm_silu && !q8_tm_decode_silu)
        {
            return false;
        }
        unsafe {
            imparo_cuda_matmat_gated(
                gate_kind,
                gate_off,
                up_kind,
                up_off,
                n_in,
                n_out,
                b(src),
                b(dst),
                b(tmp),
                n_tok,
                if q8_tm_silu || q8_tm_decode_silu {
                    Epilogue::Silu as u32
                } else {
                    0
                },
            )
        }
        true
    }
    fn ffn_gated_down(
        &self,
        gate_kind: WeightKindWire,
        gate_off: u64,
        up_kind: WeightKindWire,
        up_off: u64,
        down_kind: WeightKindWire,
        down_off: u64,
        n_in: u32,
        n_mid: u32,
        n_out: u32,
        src: BufId,
        gated_tmp: BufId,
        dst: BufId,
        n_tok: u32,
    ) -> bool {
        let activation = ACTIVATION.load(Ordering::Relaxed);
        let q4_gelu = ffn_sidecar_requested(n_tok)
            && gate_kind == 1
            && up_kind == 1
            && down_kind == 1
            && n_tok > 8
            && n_tok <= 512
            && activation == Epilogue::Gelu as u32;
        let q8_tm_min_tokens = crate::knobs::q8_tm_silu_private_down_min_tokens();
        let q8_tm_silu = q8_tm_min_tokens != 0
            && n_tok >= q8_tm_min_tokens
            && gate_kind == 3
            && up_kind == 3
            && down_kind == 3
            && n_tok > 8
            && activation == Epilogue::Silu as u32
            && crate::knobs::q8_tm_silu_pair_enabled();
        if !q4_gelu && !q8_tm_silu {
            return false;
        }
        unsafe {
            imparo_cuda_ffn_gated_down(
                gate_kind,
                gate_off,
                up_kind,
                up_off,
                down_kind,
                down_off,
                n_in,
                n_mid,
                n_out,
                b(src),
                b(gated_tmp),
                b(dst),
                n_tok,
            ) != 0
        }
    }
    fn matmat_pair(
        &self,
        first_kind: WeightKindWire,
        first_off: u64,
        first_dst: BufId,
        second_kind: WeightKindWire,
        second_off: u64,
        second_dst: BufId,
        n_in: u32,
        n_out: u32,
        src: BufId,
        n_tok: u32,
    ) {
        unsafe {
            imparo_cuda_matmat_pair(
                first_kind,
                first_off,
                b(first_dst),
                second_kind,
                second_off,
                b(second_dst),
                n_in,
                n_out,
                b(src),
                n_tok,
            );
        }
    }
    fn ple_project(
        &self,
        gate_kind: WeightKindWire,
        gate_off: u64,
        proj_kind: WeightKindWire,
        proj_off: u64,
        n_embd: u32,
        ple_width: u32,
        src: BufId,
        gate: BufId,
        per_layer: BufId,
        per_layer_off: u32,
        per_layer_stride: u32,
        back: BufId,
        n_tok: u32,
    ) {
        if ACTIVATION.load(Ordering::Relaxed) != Epilogue::Gelu as u32 {
            self.matmat(gate_kind, gate_off, n_embd, ple_width, src, gate, n_tok);
            self.act(gate, n_tok * ple_width);
            self.mul_strided(
                gate,
                per_layer,
                ple_width,
                per_layer_off,
                per_layer_stride,
                ple_width,
                n_tok,
            );
            self.matmat(proj_kind, proj_off, ple_width, n_embd, gate, back, n_tok);
            return;
        }
        unsafe {
            imparo_cuda_ple_project(
                gate_kind,
                gate_off,
                proj_kind,
                proj_off,
                n_embd,
                ple_width,
                b(src),
                b(gate),
                b(per_layer),
                per_layer_off,
                per_layer_stride,
                b(back),
                n_tok,
            );
        }
    }
    fn set_epilogue(&self, epi: imparo_backend::Epilogue) {
        unsafe { imparo_cuda_set_epilogue(epi as u32) }
    }
    fn supports_epilogue(&self, epi: Epilogue) -> bool {
        matches!(epi, Epilogue::None | Epilogue::Gelu)
    }
    fn row(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        width: u32,
        index: u32,
        scale: f32,
        dst: BufId,
        dst_off: u32,
    ) {
        unsafe { imparo_cuda_row(wkind, w_off, width, index, scale, b(dst), dst_off) }
    }
    fn gather_rows(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        width: u32,
        table_rows: u32,
        scale: f32,
        dst: BufId,
        dst_off: u32,
        idx: BufId,
        n_rows: u32,
    ) -> bool {
        if !matches!(wkind, 1 | 2) || dst_off != 0 || width % 32 != 0 || n_rows == 0 {
            return false;
        }
        unsafe {
            imparo_cuda_rows(
                wkind,
                w_off,
                width,
                table_rows,
                b(idx),
                scale,
                b(dst),
                n_rows,
            )
        }
        true
    }
    fn rms_norm(
        &self,
        buf: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    ) {
        // NO_WEIGHT sentinel (u64::MAX) means normalize without a weight vector.
        let has_w = u32::from(w_off != u64::MAX);
        unsafe {
            imparo_cuda_rms_norm(
                b(buf),
                b(buf),
                w_off,
                width,
                eps,
                n_row,
                row_stride,
                base_off,
                has_w,
            )
        }
    }
    fn rms_norm_from(
        &self,
        buf: BufId,
        src: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    ) {
        let has_w = u32::from(w_off != u64::MAX);
        unsafe {
            imparo_cuda_rms_norm(
                b(buf),
                b(src),
                w_off,
                width,
                eps,
                n_row,
                row_stride,
                base_off,
                has_w,
            )
        }
    }
    fn use_decode_projection_preparation(&self) -> bool {
        crate::knobs::decode_graph_q8_producer_reuse_enabled()
    }
    fn use_prefill_projection_preparation(&self) -> bool {
        crate::knobs::prefill_projection_q8_d4_enabled()
    }
    fn rms_norm_projection(
        &self,
        buf: BufId,
        src: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    ) {
        unsafe {
            imparo_cuda_rms_norm_project(
                b(buf),
                b(src),
                w_off,
                width,
                eps,
                n_row,
                row_stride,
                base_off,
            );
        }
    }
    fn head_norm_rope_hadamard(
        &self,
        buf: BufId,
        w_off: u64,
        head_dim: u32,
        eps: f32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        rope_dim: u32,
        rope_base: f32,
        freqs: Option<&[f32]>,
        hadamard_nrot: u32,
    ) {
        unsafe {
            imparo_cuda_head_norm_rope_hadamard(
                b(buf),
                w_off,
                head_dim,
                eps,
                n_heads,
                start_pos,
                n_tok,
                rope_dim,
                rope_base,
                freqs.map_or(std::ptr::null(), <[f32]>::as_ptr),
                hadamard_nrot,
            );
        }
    }
    fn kv_head_postprocess(
        &self,
        k: BufId,
        v: BufId,
        k_norm_off: u64,
        head_dim: u32,
        eps: f32,
        n_kv: u32,
        start_pos: u32,
        n_tok: u32,
        rope_dim: u32,
        rope_base: f32,
        freqs: Option<&[f32]>,
        k_hadamard_nrot: u32,
        v_hadamard_nrot: u32,
    ) {
        unsafe {
            imparo_cuda_kv_head_postprocess(
                b(k),
                b(v),
                k_norm_off,
                head_dim,
                eps,
                n_kv,
                start_pos,
                n_tok,
                rope_dim,
                rope_base,
                freqs.map_or(std::ptr::null(), <[f32]>::as_ptr),
                k_hadamard_nrot,
                v_hadamard_nrot,
            );
        }
    }
    fn rms_norm_add(
        &self,
        dst: BufId,
        src: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
        add: BufId,
        output_scale: f32,
    ) -> bool {
        if !crate::knobs::rms_norm_add_enabled()
            || width == 0
            || n_row == 0
            || w_off == u64::MAX
            || row_stride != width
            || base_off != 0
            || (output_scale.to_bits() != 1.0_f32.to_bits()
                && std::env::var_os("IMPARO_CUDA_NO_RMS_ADD_SCALE").is_some())
        {
            return false;
        }
        unsafe {
            imparo_cuda_rms_norm_add(
                b(dst),
                b(src),
                w_off,
                width,
                eps,
                n_row,
                b(add),
                output_scale,
            )
        }
        true
    }
    fn rms_norm_add_projection(
        &self,
        dst: BufId,
        src: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
        add: BufId,
    ) -> bool {
        if std::env::var_os("IMPARO_CUDA_RMS_ADD_Q8_LAB").is_none()
            || !crate::knobs::rms_norm_add_enabled()
            || dst != src
            || width == 0
            || width % 128 != 0
            || n_row <= 8
            || w_off == u64::MAX
            || row_stride != width
            || base_off != 0
        {
            return false;
        }
        unsafe {
            imparo_cuda_rms_norm_add_project(b(dst), w_off, width, eps, n_row, b(add))
        }
        true
    }
    fn add_rms_norm(
        &self,
        dst: BufId,
        resid: BufId,
        other: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    ) -> bool {
        if !crate::knobs::rms_norm_add_enabled()
            || crate::knobs::prefill_projection_q8_d4_mode() != 2
            || std::env::var_os("IMPARO_GPU_PROBE").is_some()
            || dst == resid
            || dst == other
            || resid == other
            || width == 0
            || width % 128 != 0
            || n_row <= 8
            || row_stride != width
            || base_off != 0
            || w_off == u64::MAX
        {
            return false;
        }
        unsafe {
            imparo_cuda_add_rms_norm_project(
                b(dst),
                b(resid),
                b(other),
                w_off,
                width,
                eps,
                n_row,
            ) != 0
        }
    }
    fn rms_norm_add_dual_projection(
        &self,
        src: BufId,
        residual: BufId,
        first_w_off: u64,
        mid: BufId,
        second_w_off: u64,
        out: BufId,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
        output_scale: f32,
    ) -> bool {
        if std::env::var_os("IMPARO_CUDA_PREFILL_DUAL_RMS_Q8_READY_LAB").is_none()
            || output_scale.to_bits() != 1.0_f32.to_bits()
            || std::env::var_os("IMPARO_GPU_PROBE").is_some()
            || !crate::knobs::rms_norm_add_enabled()
            || src != mid
            || out == mid
            || out == residual
            || width != 2560
            || n_row <= 8
            || n_row > 512
            || first_w_off == u64::MAX
            || second_w_off == u64::MAX
            || row_stride != width
            || base_off != 0
        {
            return false;
        }
        unsafe {
            imparo_cuda_rms_norm_add_dual_project(
                b(src),
                b(residual),
                first_w_off,
                b(mid),
                second_w_off,
                b(out),
                width,
                eps,
                n_row,
            ) != 0
        }
    }
    fn rope(
        &self,
        buf: BufId,
        n_rot: u32,
        base: f32,
        head_dim: u32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        freqs: Option<&[f32]>,
    ) {
        // TODO(cuda dev): freqs live on the host; upload once at init like the .mm
        // keeps them in a buffer. Passing the host pointer is a placeholder that a
        // real port replaces with a device-resident copy.
        let p = freqs.map_or(core::ptr::null(), <[f32]>::as_ptr);
        unsafe {
            imparo_cuda_rope(
                b(buf),
                n_rot,
                base,
                head_dim,
                n_heads,
                start_pos,
                n_tok,
                p,
            )
        }
    }
    fn hadamard(&self, buf: BufId, n: u32, nrot: u32) {
        unsafe { imparo_cuda_hadamard(b(buf), n, nrot) }
    }
    fn kv_store(
        &self,
        src: BufId,
        layer: u32,
        width: u32,
        start_pos: u32,
        n_tok: u32,
        is_v: bool,
        ring: u32,
    ) {
        unsafe {
            imparo_cuda_kv_store(
                b(src),
                layer,
                width,
                start_pos,
                n_tok,
                u32::from(is_v),
                ring,
            )
        }
    }
    fn attention(
        &self,
        kv_layer: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv: u32,
        kv_width: u32,
        start_pos: u32,
        scale: f32,
        window: u32,
        n_tok: u32,
        max_scores: u32,
        ring: u32,
    ) {
        // The op applies its scale: Q is scaled in place before the kernel reads it (the
        // dispatch the workflow used to issue). The kernel path below still receives the
        // value as its convert-then-scale metadata, exactly as before: Q arrives scaled,
        // and the half route reconstructs the oracle's order from the number. Unchanged
        // numerics; compile-verified only (no CUDA hardware here, #61).
        if scale.to_bits() != 1.0_f32.to_bits() {
            self.scale(BufId::Q, scale, n_tok * n_heads * head_dim);
        }
        let applied_q_scale = scale;
        let _ = max_scores; // Metal's split-KV sizing hint; unused by the CUDA shape
        // Decode dequantizes the addressed row inside the native attention path. Prefill
        // reads a half-precision layer scratch instead, so populate that scratch here.
        // v2 deliberately removed `kv_dequant` from the shared Backend trait: cache
        // representation is a backend-owned detail of attention, not model workflow.
        if n_tok > 1 {
            let slots = prefill_dequant_slots(start_pos, n_tok, ring);
            unsafe {
                imparo_cuda_kv_dequant(
                    kv_layer,
                    kv_width,
                    slots,
                    0,
                    b(BufId::Kdq),
                    ring,
                );
                imparo_cuda_kv_dequant(
                    kv_layer,
                    kv_width,
                    slots,
                    1,
                    b(BufId::Vdq),
                    ring,
                );
            }
        }
        unsafe {
            imparo_cuda_attention(
                kv_layer,
                head_dim,
                n_heads,
                n_kv,
                kv_width,
                start_pos,
                applied_q_scale,
                window,
                n_tok,
                ring,
                b(BufId::Q),
                b(BufId::Attn),
                b(BufId::Kdq),
                b(BufId::Vdq),
            )
        }
    }
    fn set_activation(&self, act: Epilogue) {
        ACTIVATION.store(act as u32, Ordering::Relaxed);
    }
    fn act(&self, a: BufId, n: u32) {
        match ACTIVATION.load(Ordering::Relaxed) {
            x if x == Epilogue::None as u32 => {}
            x if x == Epilogue::Gelu as u32 => unsafe { imparo_cuda_gelu(b(a), n) },
            x if x == Epilogue::Silu as u32 => unsafe { imparo_cuda_silu(b(a), n) },
            _ => unreachable!("Epilogue has a closed repr(u32) wire set"),
        }
    }
    fn act_mul(&self, a: BufId, bb: BufId, n: u32) {
        match ACTIVATION.load(Ordering::Relaxed) {
            x if x == Epilogue::Gelu as u32 => unsafe {
                imparo_cuda_gelu_mul(b(a), b(bb), n)
            },
            x if x == Epilogue::Silu as u32 => unsafe {
                imparo_cuda_silu_mul(b(a), b(bb), n)
            },
            _ => unreachable!("act_mul requires GELU or SiLU activation"),
        }
    }
    fn add(&self, a: BufId, bb: BufId, n: u32) {
        unsafe { imparo_cuda_add(b(a), b(bb), n) }
    }
    fn add_scale(&self, a: BufId, bb: BufId, k: f32, n: u32) {
        unsafe { imparo_cuda_add_scale(b(a), b(bb), k, n) }
    }
    fn scale(&self, a: BufId, k: f32, n: u32) {
        unsafe { imparo_cuda_scale(b(a), k, n) }
    }
    fn copy(&self, dst: BufId, src: BufId, n: u32) {
        unsafe { imparo_cuda_copy(b(dst), b(src), n) }
    }
    fn copy_range(&self, dst: BufId, dst_off: u32, src: BufId, src_off: u32, n: u32) {
        unsafe { imparo_cuda_copy_range(b(dst), dst_off, b(src), src_off, n) }
    }
    fn mul_strided(
        &self,
        a: BufId,
        bb: BufId,
        n: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_tok: u32,
    ) {
        unsafe {
            imparo_cuda_mul_strided(b(a), b(bb), n, b_off, b_stride, a_stride, n_tok)
        }
    }
    fn softcap(&self, a: BufId, cap: f32, n: u32) {
        unsafe { imparo_cuda_softcap(b(a), cap, n) }
    }
    fn argmax(&self, src: BufId, dst: BufId, n: u32) {
        unsafe { imparo_cuda_argmax(b(src), b(dst), n) }
    }
    fn ple_gather_combine(
        &self,
        proj: BufId,
        tokens_buf: BufId,
        w_offset: u64,
        width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    ) {
        unsafe {
            imparo_cuda_ple_gather_combine(
                b(proj),
                b(tokens_buf),
                w_offset,
                width,
                emb_scale,
                comb_scale,
                n_tok,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn shortconv(
        &self,
        bcx: BufId,
        w_off: u64,
        state: BufId,
        state_off: u32,
        out: BufId,
        width: u32,
        kernel: u32,
        n_tok: u32,
    ) {
        unsafe {
            imparo_cuda_shortconv(
                b(bcx),
                w_off,
                b(state),
                state_off,
                b(out),
                width,
                kernel,
                n_tok,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn shortconv_snapshot(
        &self,
        bcx: BufId,
        state: BufId,
        state_off: u32,
        snap: BufId,
        snap_off: u32,
        width: u32,
        kernel: u32,
        n_tok: u32,
    ) {
        unsafe {
            imparo_cuda_shortconv_snapshot(
                b(bcx),
                b(state),
                state_off,
                b(snap),
                snap_off,
                width,
                kernel,
                n_tok,
            )
        }
    }
    unsafe fn init_weights(&self, base: *const u8, len: u64) -> Result<(), i32> {
        // CUDA uploads the blob to device memory.
        let bytes = unsafe { core::slice::from_raw_parts(base, len as usize) };
        init(bytes)
    }
    unsafe fn init_weights_with_residency(
        &self,
        base: *const u8,
        len: u64,
        streamed: &[StreamedWeightSpan],
    ) -> Result<(), i32> {
        let bytes = unsafe { core::slice::from_raw_parts(base, len as usize) };
        init_with_streamed(bytes, streamed)
    }
    /// The fast tier is the VRAM (docs/memory-tiers-and-fit.md section 2: the whole card,
    /// the reserve is computed by the common runtime); 0 / unknown reports None.
    fn fast_tier_budget(&self) -> Option<u64> {
        let total = crate::context::CudaContext::get().memory_info().total;
        (total > 0).then_some(total)
    }
    fn quantized_weight_cache_enabled(&self) -> bool {
        crate::knobs::prefill_exact128_fast_transaction_enabled()
            || crate::knobs::decode_shadow_cache_enabled()
            || crate::knobs::prefill_down_q4_layer_mask() != 0
            || std::env::var_os("IMPARO_LAB_LFM2_Q8_TM_DOWN_Q4_SHADOW").is_some()
    }
    fn quantized_weight_cache_plan(&self) -> QuantizedWeightCachePlan {
        let lab_mode = std::env::var("IMPARO_LAB_LFM2_Q8_TM_DOWN_Q4_SHADOW").ok();
        let tuned_full_ffn = crate::knobs::decode_ffn_q5_layer_mask() != 0;
        QuantizedWeightCachePlan {
            include_down: lab_mode.as_deref() != Some("head")
                || crate::knobs::prefill_down_q4_layer_mask() != 0,
            include_full_ffn: tuned_full_ffn
                || lab_mode.as_deref() == Some("mixed-ffn"),
            include_head: matches!(
                lab_mode.as_deref(),
                Some("head") | Some("down-head") | Some("q5")
            ) && std::env::var_os("IMPARO_LAB_LFM2_Q5_DOWN_ONLY")
                .is_none(),
        }
    }
    fn prepare_quantized_weight_cache(
        &self,
        weights: &[imparo_backend::QuantizedWeightPrepack],
    ) -> Result<bool, i32> {
        let count = u32::try_from(weights.len()).map_err(|_| 3)?;
        let rc = unsafe {
            imparo_cuda_prepare_quantized_weight_cache(weights.as_ptr(), count)
        };
        match rc {
            1 => Ok(true),
            0 => Ok(false),
            error => Err(error),
        }
    }
    fn set_kv_types(&self, k: u32, v: u32) {
        KV_K.store(k, Ordering::Relaxed);
        KV_V.store(v, Ordering::Relaxed);
        crate::context::CudaContext::get().set_kv_types(k, v);
    }
    fn prof_stats(&self) -> imparo_backend::ProfStats {
        imparo_backend::ProfStats::default()
    }
    fn prof_enable(&self, on: bool) {
        let _ = on;
    }
    fn allocated_bytes(&self) -> u64 {
        crate::context::CudaContext::get().memory_info().allocated
    }
    fn pool_caps(&self) -> imparo_backend::PoolCaps {
        use imparo_backend::Tier;
        imparo_backend::PoolCaps {
            // CUDA's block table maps one 64-cell page. The current byte-identity
            // gates establish 64 as a conservative cut quantum for every selected
            // CUDA attention route; a finer declaration requires its own gate.
            page_cells: 64,
            finest_cut_tokens: 64,
            paged_reads: true,
            shared_address: false,
            // Explicit-address CUDA uses the complete ordered ladder. The server
            // still enables it only after conservative Host/disk auto-fit succeeds.
            tiers: &[Tier::Device, Tier::Host, Tier::Disk],
        }
    }
    fn device_tag(&self) -> String {
        crate::context::CudaContext::get().device_tag()
    }
}

#[cfg(test)]
mod tests {
    use super::{CudaBackend, kv_code, prefill_dequant_slots};
    use imparo_backend::{Backend, Epilogue, PoolAddressing, Tier};

    #[test]
    fn cuda_fuses_only_the_epilogues_its_native_kernels_implement() {
        assert!(CudaBackend.supports_epilogue(Epilogue::None));
        assert!(CudaBackend.supports_epilogue(Epilogue::Gelu));
        assert!(!CudaBackend.supports_epilogue(Epilogue::Silu));
    }

    #[test]
    fn kv_environment_names_match_native_wire_values() {
        assert_eq!(kv_code(None), 1);
        assert_eq!(kv_code(Some("f16")), 1);
        assert_eq!(kv_code(Some("q4_0")), 2);
        assert_eq!(kv_code(Some("q8_0")), 8);
        assert_eq!(kv_code(Some("unknown")), 1);
    }

    #[test]
    fn cuda_advertises_only_the_complete_explicit_tier_contract() {
        let caps = CudaBackend.pool_caps();
        assert_eq!(caps.page_cells, 64);
        assert_eq!(caps.finest_cut_tokens, 64);
        assert!(caps.paged_reads);
        assert!(!caps.shared_address);
        assert_eq!(caps.tiers, &[Tier::Device, Tier::Host, Tier::Disk]);
        assert_eq!(
            caps.validate_for_pool(64),
            Ok(PoolAddressing::ExplicitHostTransfers)
        );
    }

    #[test]
    fn prefill_dequant_covers_initialized_rows_without_crossing_a_ring() {
        assert_eq!(prefill_dequant_slots(512, 512, 0), 1024);
        assert_eq!(prefill_dequant_slots(512, 512, 1023), 1024);
        assert_eq!(prefill_dequant_slots(2048, 1, 1023), 1024);
        assert_eq!(prefill_dequant_slots(u32::MAX - 3, 8, 0), u32::MAX);
    }

    #[test]
    fn paging_store_graph_contract_appends_the_stable_table_argument() {
        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("constexpr uint32_t kKvStoreGraphArgCount = 8;"));
        assert!(native.contains("constexpr uint32_t kKvStoreGraphStartArg = 3;"));
        assert!(native.contains("constexpr uint32_t kKvStoreGraphPageTableArg = 7;"));
        assert!(native.contains("kD512IdentityPartialGraphArgCount = 13;"));
        assert!(native.contains("kD512IdentityControlledGraphArgCount = 14;"));
        assert!(native.contains("kD512PartialGraphArgCount = 14;"));
        assert!(native.contains("kD512ControlledGraphArgCount = 15;"));
        assert!(native.contains("partial_f16_controlled_paged"));
        assert!(native.contains("partial_f16_paged"));
        assert!(native.contains("scores_paged<256, 2>"));
        assert!(native.contains("values_combine_paged<256, 2, 2>"));
        assert!(native.contains("decode_control_arg(), page_table"));

        let setter = native
            .split_once("extern \"C\" int imparo_cuda_set_kv_pages")
            .expect("set_kv_pages definition")
            .1
            .split_once("static int ensure_kv_stage")
            .expect("set_kv_pages body")
            .0;
        let capture_guard = setter
            .find("if (g.graph_capturing) return CUDA_RC_INVALID;")
            .expect("capture guard");
        let prepare = setter
            .find("prepare_page_update")
            .expect("page update preparation");
        let class_flip = setter
            .find("if (mapped_before != mapped_after) destroy_decode_graph();")
            .expect("identity/paged specialization transition");
        let commit = setter
            .find("commit_page_update")
            .expect("page update commit");
        assert!(capture_guard < prepare);
        assert!(commit < class_flip);
        assert_eq!(setter.matches("destroy_decode_graph").count(), 1);

        let replace = native
            .split_once("static int replace_kv_arena")
            .expect("replace_kv_arena definition")
            .1;
        let guard = replace
            .find("if (g.graph_capturing) return CUDA_RC_INVALID;")
            .expect("capture guard");
        let build = replace.find("build_page_tables").expect("page-table build");
        assert!(
            guard < build,
            "capture guard must precede allocation planning"
        );
    }

    #[test]
    fn sidecar_prefill_graph_lab_remains_exact_shape_and_fail_closed() {
        let rust = include_str!("backend_impl.rs");
        assert!(rust.contains("let sidecar_graph_lab = count == 128"));
        assert!(rust.contains("IMPARO_CUDA_PREFILL_SIDECAR_GRAPH_LAB"));

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains(
            "return token_count == 128 && value && std::strcmp(value, \"1\") == 0;"
        ));
        assert!(
            native
                .contains("g.ffn_sidecar_model_ready && g.prefill_warm_forwards >= 2")
        );
        assert!(rust.contains("fn prefill_body_prepare("));
        let workflow = include_str!("../../imparo-model/src/gemma4/workflow_gpu.rs");
        assert!(workflow.contains(".prefill_body_prepare(tokens, sp, argmax)"));
        assert!(workflow.contains(
            "Prefill graph must therefore start only after those request-specific"
        ));
        assert!(native.contains("static int prefill_body_capture_or_replay("));
        assert!(native.contains("boundary=post-embedding"));
        assert!(native.contains("prefill body prefix sync"));
        assert!(native.contains("if (g.forward_start)"));
        assert!(native.contains("g.prefill_graph_blocked = true;"));
    }
}
