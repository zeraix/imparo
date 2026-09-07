const HEADER: &str = include_str!("../native/sm86/rms_norm_add_dual_q8_ready.cuh");
const NATIVE: &str = include_str!("../native/imparo_cuda.cu");
const BACKEND: &str = include_str!("../src/backend_impl.rs");
const HARNESS: &str =
    include_str!("../native/tests/sm86_dual_rms_q8_ready_contract.cu");

#[test]
fn reused_reduction_slots_have_a_post_read_barrier() {
    let reduction = HEADER
        .find("value = lane < Warps ? shared[lane] : 0.0f;")
        .expect("second-stage shared reduction read");
    let tail = &HEADER[reduction..];
    let warp_finish = tail
        .find("value = warp_sum_xor(value);")
        .expect("warp-zero reduction completion");
    let reuse_barrier = tail
        .find("__syncthreads();")
        .expect("post-read reuse barrier");
    let result = tail.find("return value;").expect("reduction return");
    assert!(warp_finish < reuse_barrier && reuse_barrier < result);
    assert!(HEADER.contains("invokes this reduction twice with the same shared slots"));
}
#[test]
fn dual_rms_route_remains_lab_only_and_commits_after_launch() {
    assert!(BACKEND.contains(
        "std::env::var_os(\"IMPARO_CUDA_PREFILL_DUAL_RMS_Q8_READY_LAB\").is_none()",
    ));
    assert!(BACKEND.contains("std::env::var_os(\"IMPARO_GPU_PROBE\").is_some()"));
    let entry = NATIVE
        .find("extern \"C\" uint32_t imparo_cuda_rms_norm_add_dual_project")
        .expect("native dual RMS entry");
    let body = &NATIVE[entry..];
    assert!(body.contains("g.sm_version != 86 || g.graph_capturing"));
    assert!(body.contains("|| g.prefill_capture_active"));
    assert!(body.contains("std::getenv(\"IMPARO_GPU_PROBE\") != nullptr"));
    let launch = body
        .find("imparo_sm86_dual_rms_q8_ready::launch(")
        .expect("candidate launch");
    let commit = body.find("mark_buf_written(mid);").expect("dense commit");
    let ownership = body
        .find("own_q8_cache(out, width, n_row, 0, Q8_LAYOUT_MMA_READY);")
        .expect("Q8 ownership commit");
    let trace = body
        .find("IMPARO_CUDA_PREFILL_DUAL_RMS_TRACE")
        .expect("post-commit diagnostic trace");
    assert!(launch < commit && commit < ownership && ownership < trace);
}

#[test]
fn native_contract_covers_dense_q8_boundaries_and_fallbacks() {
    assert!(HARNESS.contains("for (const uint32_t tokens : {9u, 128u, 449u, 512u})"));
    for field in [
        "mid_byte_equal",
        "norm_byte_equal",
        "quant_byte_equal",
        "scale_byte_equal",
        "fallback_zero_tokens",
        "fallback_wrong_sm",
    ] {
        assert!(HARNESS.contains(field), "missing contract field {field}");
    }
    assert!(
        HARNESS
            .contains("return mid_equal && norm_equal && quant_equal && scale_equal;")
    );
}
