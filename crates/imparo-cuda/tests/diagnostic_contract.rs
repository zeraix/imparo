// Diagnostic instrumentation shipped in a release must be observational. These source
// contracts deliberately sit beside the existing LFM2 native contracts: they catch an
// accidental reintroduction of the one-off file dump and input injection controls without
// requiring a GPU or granting tests filesystem paths supplied through environment variables.

const NATIVE: &str = include_str!("../native/imparo_cuda.cu");
const GPU_SUPPORT: &str = include_str!("../../imparo-model/src/gpu_support.rs");
const GEMMA_WORKFLOW: &str =
    include_str!("../../imparo-model/src/gemma4/workflow_gpu.rs");
const LFM2_WORKFLOW: &str = include_str!("../../imparo-model/src/lfm2/workflow_gpu.rs");

#[test]
fn model_gpu_diagnostics_cannot_dump_or_override_activations() {
    for forbidden in [
        "IMPARO_GPU_PROBE_DUMP",
        "IMPARO_GPU_PROBE_DUMP_NAME",
        "IMPARO_GPU_PROBE_DUMP_OCCURRENCE",
        "IMPARO_GPU_PROBE_OVERRIDE",
        "IMPARO_GPU_PROBE_OVERRIDE_NAME",
        "IMPARO_GPU_PROBE_OVERRIDE_OCCURRENCE",
        "LAYER_PROBE_MAGIC",
        "write_layer_probe",
        "override_layer_probe",
    ] {
        assert!(
            !GPU_SUPPORT.contains(forbidden),
            "model GPU support reintroduced raw activation diagnostic {forbidden}"
        );
        assert!(
            !GEMMA_WORKFLOW.contains(forbidden),
            "Gemma4 workflow reintroduced activation override {forbidden}"
        );
        assert!(
            !LFM2_WORKFLOW.contains(forbidden),
            "LFM2 workflow reintroduced activation override {forbidden}"
        );
    }

    // The read-only, opt-in numerical witness remains available.
    assert!(GPU_SUPPORT.contains("pub(crate) fn gprobe("));
    assert!(GPU_SUPPORT.contains("be().read(buf, off, &mut v)"));
}

#[test]
fn native_cuda_diagnostics_cannot_dump_or_override_raw_buffers() {
    for forbidden in [
        "IMPARO_CUDA_Q8_PROBE",
        "IMPARO_CUDA_KV_ROW_DUMP_",
        "IMPARO_CUDA_KV_OVERRIDE_",
        "IMPARO_CUDA_KV_DUMP_",
        "kv_override_slot_count",
        "override_cache",
        "dump_cache",
    ] {
        assert!(
            !NATIVE.contains(forbidden),
            "native CUDA reintroduced raw-buffer diagnostic {forbidden}"
        );
    }

    // Stable observation-only profiling and route receipts are intentionally retained.
    for retained in [
        "IMPARO_CUDA_PROFILE_MATMUL",
        "IMPARO_CUDA_PROFILE_OPS",
        "IMPARO_CUDA_PROFILE_ATTN_STAGES",
        "IMPARO_CUDA_MMQ_TRACE",
        "trace_d256_attention_route",
    ] {
        assert!(
            NATIVE.contains(retained),
            "removed retained diagnostic {retained}"
        );
    }
}
