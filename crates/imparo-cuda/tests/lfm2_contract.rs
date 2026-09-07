const NATIVE: &str = include_str!("../native/imparo_cuda.cu");
const CUDA_KNOBS: &str = include_str!("../src/knobs.rs");
const HEADER: &str = include_str!("../native/lfm2_ops.cuh");
const Q8_MMVQ: &str = include_str!("../native/sm80/mmvq_q8_q8_1.cuh");
const Q8_MMQ: &str = include_str!("../native/sm80/mmq_q8_q8_1.cuh");
const Q4_MMQ: &str = include_str!("../native/sm80/mmq_q4_q8_1.cuh");
const Q8_REPLAY_PLAN: &str = include_str!("../native/sm80/mmq_q8_replay_plan.h");
const SMALL_FA: &str =
    include_str!("../native/sm80/attention_prefill_d512_small_f16.cuh");
const D64_WIDE_FA: &str =
    include_str!("../native/sm80/attention_prefill_d64_wide_f16.cuh");
const D64_MMA_PLAN: &str =
    include_str!("../native/sm80/attention_prefill_mma_d64_plan.h");
const D64_MMA_FA: &str =
    include_str!("../native/sm80/attention_prefill_mma_d64_f16.cuh");
const D64_MMA_SM86: &str =
    include_str!("../native/sm86/attention_prefill_mma_d64_profile.h");
const D64_Q4_VEC: &str = include_str!("../native/sm80/attention_decode_vec_d64_q4.cuh");
const D64_Q8_VEC: &str = include_str!("../native/sm80/attention_decode_vec_d64_q8.cuh");
const D256_Q4_VEC: &str = include_str!("../native/sm80/attention_decode_vec_d256.cuh");
const D512_FLASH: &str = include_str!("../native/sm80/attention_flash_d512_f16.cuh");
const CUDA_BACKEND: &str = include_str!("../src/backend_impl.rs");
const LFM2_WORKFLOW: &str = include_str!("../../imparo-model/src/lfm2/workflow_gpu.rs");

fn body_after<'a>(source: &'a str, marker: &str) -> &'a str {
    source
        .split_once(marker)
        .unwrap_or_else(|| panic!("missing native marker: {marker}"))
        .1
}

#[test]
fn d64_lfm2_does_not_sweep_the_d256_d512_value_tile_knob() {
    let declaration = body_after(CUDA_KNOBS, "\"attn_value_tiles\",")
        .split_once("),")
        .expect("attn_value_tiles declaration must terminate")
        .0;
    assert!(declaration.contains("Wl::AttentionDecode"));
    assert!(
        declaration.contains("applies = |m| m.head_dim != 64"),
        "the D64 specialized attention route must not claim the D256/D512 value-tile knob"
    );
}

#[test]
fn q8_lfm2_does_not_sweep_q4_mmvq_row_grouping_knobs() {
    for name in ["narrow_gemv_rows_per_cta", "batch_mmvq_rows_per_cta"] {
        let marker = format!("\"{name}\",");
        let declaration = body_after(CUDA_KNOBS, &marker)
            .lines()
            .take(12)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            declaration.contains("applies = |m| m.weight_kinds & (1 << 1) != 0"),
            "{name} must only tune the Q4 MMVQ route"
        );
    }
}

#[test]
fn virtual_r64_mmq_is_default_on_and_shape_scoped() {
    assert!(
        Q4_MMQ.contains("std::getenv(\"IMPARO_CUDA_NO_MMQ_VIRTUAL_R64\") == nullptr")
    );
    assert!(Q4_MMQ.contains("n_in == 2560 && n_out == 10240"));
    assert!(Q4_MMQ.contains("virtual_token_base % 512 == 0"));
    assert!(Q4_MMQ.contains("LaunchRoute::VirtualNoSeamRows64"));
    assert!(Q4_MMQ.contains("q4_q8_1_full_tile_r64<true>"));
    assert!(Q4_MMQ.contains("q4_q8_1_full_tile_r64<false>"));
}

#[test]
fn shortconv_validates_shape_buffers_aliases_and_weights_before_launch() {
    let body = body_after(NATIVE, "extern \"C\" void imparo_cuda_shortconv(");
    let shape = body.find("checked_shortconv_layout").unwrap();
    let buffers = body.find("buffer_slice(bcx").unwrap();
    let aliases = body.find("byte_ranges_overlap").unwrap();
    let weights = body.find("weight_slice(w_off").unwrap();
    let output = body.find("shortconv_output_kernel<<<").unwrap();
    let advance = body.find("shortconv_state_kernel<<<").unwrap();
    assert!(shape < buffers && buffers < aliases && aliases < weights);
    assert!(weights < output && output < advance);
    assert!(NATIVE.contains("id >= B_COUNT || !g.bufs[id]"));
    assert!(NATIVE.contains("bytes > g.sizes[id] - byte_offset"));
}

#[test]
fn state_commit_has_one_owner_per_channel_and_snapshot_reuses_it() {
    let state = body_after(HEADER, "__global__ void shortconv_state_kernel");
    let state = state.split_once("__global__ void silu_kernel").unwrap().0;
    assert!(state.contains("const uint32_t channel"));
    assert!(state.contains("shortconv_state_channel"));
    let helper = body_after(HEADER, "shortconv_state_channel(");
    let helper = helper
        .split_once("inline void shortconv_state_reference")
        .unwrap()
        .0;
    assert!(helper.contains("for (uint32_t slot = 0; slot < history; ++slot)"));

    let snapshot =
        body_after(NATIVE, "extern \"C\" void imparo_cuda_shortconv_snapshot(");
    let snapshot = snapshot
        .split_once("extern \"C\" void imparo_cuda_silu(")
        .unwrap()
        .0;
    assert!(snapshot.contains("shortconv_state_kernel<<<"));
    assert!(!snapshot.contains("shortconv_output_kernel<<<"));
}

#[test]
fn unsupported_fused_silu_fails_before_epilogue_state_changes() {
    let setter = body_after(NATIVE, "extern \"C\" void imparo_cuda_set_epilogue");
    let setter = setter
        .split_once("extern \"C\" void imparo_cuda_set_knob")
        .unwrap()
        .0;
    assert!(
        setter.find("if (on > 1)").unwrap() < setter.find("g.epilogue = on").unwrap()
    );
    assert!(setter.contains("set_pending(CUDA_RC_INVALID"));
}

#[test]
fn abi24_rows_dispatches_each_weight_layout_and_binds_table_extent() {
    assert!(NATIVE.contains("imparo_cuda_row(uint32_t wkind"));
    assert!(NATIVE.contains("imparo_cuda_rows(uint32_t wkind"));
    let rows = body_after(NATIVE, "extern \"C\" void imparo_cuda_rows(");
    let rows = rows
        .split_once("extern \"C\" void imparo_cuda_ple_gather_combine")
        .unwrap()
        .0;
    let token_gate = rows.find("tokens[t] >= table_rows").unwrap();
    let resident_lookup = rows
        .find("resident_weight_range(w_off, table_bytes)")
        .unwrap();
    let resident_launch = rows.find("k_rows_q8_0<<<").unwrap();
    let staged_launch = rows.find("k_rows_q8_0_packed<<<").unwrap();
    assert!(token_gate < resident_lookup);
    assert!(resident_lookup < resident_launch && resident_launch < staged_launch);
    assert!(rows.contains("uint64_t(table_rows) > UINT64_MAX / row_bytes"));
    assert!(rows.contains("uint64_t(tokens[t]) > (UINT64_MAX - w_off) / row_bytes"));
}

#[test]
fn q8_matmat_uses_its_34_byte_layout_and_bounded_weight_slices() {
    let matmat = body_after(NATIVE, "extern \"C\" void imparo_cuda_matmat(");
    let matmat = matmat
        .split_once("extern \"C\" void imparo_cuda_matmat_gated")
        .unwrap()
        .0;
    let full_range = matmat
        .find("resident_weight_range(w_off, tensor_bytes)")
        .unwrap();
    let cache_bound = matmat.find("g.weight_cache_limit").unwrap();
    let slice = matmat.find("weight_slice(w_off +").unwrap();
    let q8_launch = matmat.find("k_gemm_q8_0_f32<<<").unwrap();
    assert!(full_range < cache_bound && cache_bound < slice && slice < q8_launch);
    assert!(matmat.contains("(wkind == 1 ? 18 : 34)"));
    assert!(matmat.contains("(wkind == 2 || wkind == 3) && g.epilogue != 0"));
    assert!(NATIVE.contains("float(int8_t(blk[2 + i]))"));
}

#[test]
fn q8_short_batches_use_q8_1_mmvq_and_keep_an_explicit_f32_fallback() {
    let matmat = body_after(NATIVE, "extern \"C\" void imparo_cuda_matmat(");
    let matmat = matmat
        .split_once("extern \"C\" void imparo_cuda_matmat_gated")
        .unwrap()
        .0;
    assert!(matmat.contains("const bool q8_weight = wkind == 2 || wkind == 3"));
    assert!(matmat.contains("const bool q8_mmvq = q8_weight && g.sm_version >= 80"));
    assert!(matmat.contains("std::getenv(\"IMPARO_CUDA_Q8_F32\") != nullptr"));
    assert!(matmat.contains("(wkind == 1 && !force_f32_gemv) || q8_mmvq"));
    assert!(matmat.contains("small_mmvq || mmvq_batched || q8_mmvq"));
    assert!(matmat.contains("q8_mmvq") && matmat.contains("!q8_force_f32"));
    assert!(matmat.contains("q8_mmq") && matmat.contains("!q8_force_f32"));
    assert!(matmat.contains("imparo_sm80_mmvq::quantize_q8_1"));
    assert!(matmat.contains("imparo_sm80_q8_mmvq::launch("));
    assert!(matmat.contains("k_gemm_q8_0_f32<<<"));
    assert!(matmat.contains("n_tok <= imparo_sm80_mmvq::kMaxTokens"));

    assert!(Q8_MMVQ.contains("sumi = __dp4a"));
    assert!(Q8_MMVQ.contains("weight_scale * activation_scale * float(sumi)"));
    assert!(Q8_MMVQ.contains("IMPARO_Q8_MMVQ_CASE(1, 4, 1)"));
    assert!(Q8_MMVQ.contains("IMPARO_Q8_MMVQ_CASE(4, 4, 2)"));
    assert!(Q8_MMVQ.contains("IMPARO_Q8_MMVQ_CASE(8, 2, 2)"));
}

#[test]
fn q8_tile_major_has_complete_decode_prefill_and_paged_fallback_routes() {
    let matmat = body_after(NATIVE, "extern \"C\" void imparo_cuda_matmat(");
    let matmat = matmat
        .split_once("extern \"C\" void imparo_cuda_matmat_gated")
        .unwrap()
        .0;

    assert!(matmat.contains("wkind != 2 && wkind != 3"));
    assert!(matmat.contains("Q8_0_TM output rows must be tile-aligned"));
    assert!(matmat.contains("row_capacity -= row_capacity % 8"));
    assert!(matmat.contains("Q8_0_TM cache cannot hold one 8-row weight tile"));
    assert!(matmat.contains("? q8_tm_weight_slice("));
    let tm_slice = body_after(NATIVE, "int q8_tm_weight_slice(");
    let tm_slice = tm_slice
        .split_once("struct QuantizedWeightPrepackWire")
        .unwrap()
        .0;
    assert!(tm_slice.contains("const uint64_t payload_total"));
    assert!(tm_slice.contains("cache + payload_bytes"));
    assert!(tm_slice.contains("tensor_off + payload_total"));
    assert_eq!(tm_slice.matches("cudaMemcpyAsync(").count(), 2);

    assert!(matmat.contains("imparo_sm80_q8_mmvq::launch_tile_major"));
    assert!(matmat.contains("imparo_sm80_q8_mmq::launch<true>"));
    assert!(matmat.contains("launch_aligned_whole_k<true>"));
    assert!(matmat.contains("k_gemm_q8_0_tm_f32<<<"));

    assert!(NATIVE.contains("w + uint64_t(n_out) * n_in"));
    assert!(NATIVE.contains("unit * 256 + (row & 7) * 32"));
    assert!(NATIVE.contains("scales[unit * 8 + (row & 7)]"));
    assert!(Q8_MMVQ.contains("template <bool TileMajor>"));
    assert!(Q8_MMVQ.contains("__global__ void q8_0_tm_q8_1_single"));
    assert!(Q8_MMVQ.contains("const uint64_t first_unit"));
    assert!(Q8_MMVQ.contains("if (n_tok == 1)"));
    assert!(Q8_MMQ.contains(
        "const uint32_t qblock = TileMajor\n                ? linear / kRows"
    ));
    assert!(Q8_MMQ.contains("reinterpret_cast<const uint4 *>(weight_values)"));

    assert!(Q8_MMVQ.contains("launch_layout<true>"));
    assert!(Q8_MMQ.contains(concat!(
        "template <uint32_t Tokens, bool AlignedWholeK = false, ",
        "bool TileMajor = false,\n          uint32_t Epilogue = 0>"
    )));
    assert!(Q8_MMQ.contains("q8_0_q8_1_mma<Tokens, false, TileMajor>"));
    assert!(Q8_MMQ.contains("q8_0_q8_1_mma<Tokens, true, TileMajor, Epilogue>"));
    assert!(Q8_MMQ.contains("if constexpr (Epilogue >= 2 && Epilogue <= 4)"));
    assert!(Q8_MMQ.contains("if constexpr (Epilogue == 3 || Epilogue == 4)"));
    assert!(Q8_MMQ.contains("BlockQ8_1Mmq * __restrict__ output_q8"));
    assert!(Q8_MMQ.contains("out->d[block_in_group] = d"));
    assert!(
        !CUDA_BACKEND.contains("Q8_0_TM (imparo-repack) weights have no CUDA kernel")
    );
    assert!(CUDA_KNOBS.contains("m.weight_kinds & ((1 << 2) | (1 << 3)) != 0"));
}

#[test]
fn d64_wide_prefill_is_bounded_and_retains_the_small_chain_fallback() {
    assert!(SMALL_FA.contains("HeadDim == 64 || HeadDim == 256 || HeadDim == 512"));
    assert!(NATIVE.contains("scores_paged<64, CacheType>"));
    assert!(NATIVE.contains("scores<64, CacheType>"));
    assert!(NATIVE.contains("values_combine_paged<64, CacheType, 1>"));
    assert!(NATIVE.contains("values_combine<64, CacheType, 1>"));
    assert!(NATIVE.contains("head_dim == 64 || head_dim == 256 || head_dim == 512"));
    assert!(NATIVE.contains("head_dim != 64 || n_tok >= 3"));
    assert!(NATIVE.contains("IMPARO_CUDA_ATTN_D64_STREAM_PART_CAP"));
    assert!(NATIVE.contains("IMPARO_CUDA_ATTN_D64_WORKSPACE_MIB"));
    let wide_route = NATIVE.find("const bool d64_f16_wide =").unwrap();
    let wide_launch = NATIVE.find("&& launch_attention_d64_wide(").unwrap();
    let chain_route = NATIVE.find("const bool d64_f16_chain =").unwrap();
    assert!(wide_route < wide_launch && wide_launch < chain_route);
    let wide = body_after(NATIVE, "static bool launch_attention_d64_wide(");
    let wide = wide
        .split_once("static bool launch_attention_small_chain(")
        .unwrap()
        .0;
    assert!(wide.contains("while (!ensure_attention_scratch"));
    assert!(wide.contains("&& chunk_heads > 1"));
    assert!(wide.contains("return false;"));
    assert!(NATIVE.contains("IMPARO_CUDA_NO_ATTN_D64_WIDE"));
    assert!(NATIVE.contains("const bool d64_f16_chain = g.sm_version >= 80"));
    assert!(NATIVE.contains("n_tok > imparo_sm80_d512_small::kQueryTokens"));
    assert!(NATIVE.contains("n_tok <= imparo_sm80_prefill::kWideQueryTokens"));
    assert!(NATIVE.contains("IMPARO_CUDA_NO_ATTN_D64_CHAIN"));

    assert!(D64_WIDE_FA.contains("const float * q"));
    assert!(D64_WIDE_FA.contains("float qk_scale"));
    assert!(D64_WIDE_FA.contains("probability_tile[kKeyTiles][kWarps][16 * 16]"));
    assert!(D64_WIDE_FA.contains("for (int32_t block = int32_t(last) - 1"));
}

#[test]
fn d64_sm86_mma_prefill_owns_whole_k_and_fails_back_explicitly() {
    assert!(D64_MMA_PLAN.contains("kQueryTokens = 16"));
    assert!(D64_MMA_PLAN.contains("kKeysPerUpdate = 64"));
    assert!(D64_MMA_PLAN.contains("query_tiles * n_kv"));
    assert!(D64_MMA_SM86.contains("sm_version == 86"));
    assert!(D64_MMA_SM86.contains("kLaunchOccupancy = 2"));

    assert!(D64_MMA_FA.contains("__launch_bounds__(kThreads, 2)"));
    assert!(D64_MMA_FA.contains(
        "for (uint32_t key_update = 0; key_update < key_updates; ++key_update)"
    ));
    assert!(D64_MMA_FA.contains("previous_scale[l % 2]"));
    assert!(D64_MMA_FA.contains("kSoftmaxFtzThreshold = -20.0f"));
    assert!(D64_MMA_FA.contains("float denominator[2] = {row_sum[0], row_sum[1]}"));
    assert!(D64_MMA_FA.contains("denominator[owned_column] += __shfl_xor_sync"));
    assert!(!D64_MMA_FA.contains("sum_add[owned_column] += __shfl_xor_sync"));
    assert!(D64_MMA_FA.contains("score[tile].x[2 * l], score[tile].x[2 * l + 1]"));
    assert!(!D64_MMA_FA.contains("probability_tile"));
    assert!(D64_MMA_FA.contains("qr[0] * inverse_q_scale"));
    assert!(D64_MMA_FA.contains("q_scale_h2"));
    assert!(!D64_MMA_FA.contains("numerators"));
    assert!(!D64_MMA_FA.contains("combine"));

    let mma = NATIVE.find("const bool d64_mma_prefill =").unwrap();
    let wide = NATIVE.find("const bool d64_f16_wide =").unwrap();
    assert!(
        mma < wide,
        "whole-K route must precede the Stream-K fallback"
    );
    assert!(NATIVE.contains("imparo_sm86_d64_mma_profile::applies_to(g.sm_version)"));
    assert!(NATIVE.contains("const bool d64_mma_prefill = tuner_knob(37) != 0"));
    assert!(NATIVE.contains("IMPARO_CUDA_NO_ATTN_D64_MMA_PREFILL"));
    assert!(NATIVE.contains("IMPARO_CUDA_ATTN_D64_MMA_Q_AS_IS"));
    assert!(NATIVE.contains("launch_attention_d64_mma_prefill("));
    assert!(NATIVE.contains("constexpr float qk_scale = 1.0f"));
}

#[test]
fn q8_replay_is_preflighted_once_and_oom_falls_back_before_mmq_enqueue() {
    assert!(Q8_REPLAY_PLAN.contains("canonical_whole_k_plan("));
    assert!(Q8_REPLAY_PLAN.contains("plan.physical_grid = plan.logical_tiles"));
    assert!(Q8_REPLAY_PLAN.contains("plan.replay = false"));
    assert!(Q8_REPLAY_PLAN.contains("case PlanStatus::Overflow:"));
    assert!(
        Q8_REPLAY_PLAN
            .contains("case PlanStatus::GridLimit: return ExecutionRoute::Reject")
    );
    assert!(Q8_MMQ.contains("enum class LaunchResult"));
    assert!(Q8_MMQ.contains("LaunchResult::NotSupported"));
    assert!(Q8_MMQ.contains("LaunchResult::Error"));
    assert!(Q8_MMQ.contains("slice_tiles("));
    assert!(Q8_MMQ.contains("dim3(add_grid_x, token_tiles)"));
    assert!(NATIVE.contains("n_kv == 0 || n_heads == 0 || n_heads % n_kv != 0"));
    assert!(Q8_REPLAY_PLAN.contains("enum class ExecutionRoute"));
    assert!(Q8_REPLAY_PLAN.contains("workspace_bytes_per_row"));
    assert!(Q8_REPLAY_PLAN.contains("slice_workspace_bytes("));
    assert!(
        Q8_REPLAY_PLAN.contains("plan.prefix_phases = raw_prefix_phases < seam_slots")
    );
    assert!(Q8_REPLAY_PLAN.contains("n_in % 128 != 0"));
    assert!(Q8_REPLAY_PLAN.contains("PlanStatus::NotApplicable"));

    assert!(Q8_MMQ.contains("#include \"mmq_q8_replay_plan.h\""));
    assert!(Q8_MMQ.contains("if (tid == 0)"));
    assert!(Q8_MMQ.contains("segment_for_phase("));
    assert!(Q8_MMQ.contains("__syncthreads();"));
    assert!(!Q8_MMQ.contains("__device__ __forceinline__ bool previous_stream_seam"));
    assert!(!Q8_MMQ.contains("inline uint32_t select_tile_tokens"));
    assert!(Q8_MMQ.contains("const imparo_sm80_q8_replay::ReplayPlan & plan"));

    let matmat = body_after(NATIVE, "extern \"C\" void imparo_cuda_matmat(");
    let plan = matmat.find("imparo_sm80_q8_replay::make_plan(").unwrap();
    let canonical = matmat
        .find("q8_mmq_plan = imparo_sm80_q8_replay::canonical_whole_k_plan(")
        .unwrap();
    let scratch = matmat.find("ensure_q8_scratch(bytes)").unwrap();
    let loop_start = matmat.find("for (uint32_t row_base = 0;").unwrap();
    assert!(canonical < plan && plan < scratch && scratch < loop_start);
    assert!(matmat.contains("std::min(rows_per_slice, n_out)"));
    assert!(matmat.contains("IMPARO_CUDA_Q8_REPLAY_FORCE_OOM"));
    assert!(matmat.contains("IMPARO_CUDA_Q8_SCRATCH_FORCE_OOM"));
    assert!(matmat.contains("q8_mmq = false;"));
    assert!(matmat.contains("k_gemm_q8_0_f32<<<"));

    let grow = body_after(NATIVE, "int ensure_q8_scratch(uint64_t bytes)");
    let grow = grow.split_once("bool ensure_q8_scratch_next").unwrap().0;
    let allocate = grow.find("alloc_raw(&next").unwrap();
    let publish = grow.find("g.q8_scratch = next").unwrap();
    let release = grow.find("cudaFree(previous)").unwrap();
    assert!(allocate < publish && publish < release);
}

#[test]
fn lfm2_resizes_only_activation_state_for_each_live_batch() {
    let fit = LFM2_WORKFLOW
        .find("wf.gpu_fit_batch(tokens.len())?")
        .unwrap();
    let prepare = LFM2_WORKFLOW.find("be().decode_prepare(").unwrap();
    let begin = LFM2_WORKFLOW.find("be().begin_forward(decode);").unwrap();
    let shortconv = LFM2_WORKFLOW.find("be().shortconv(").unwrap();
    assert!(fit < prepare && prepare < begin && begin < shortconv);
}

#[test]
fn lfm2_uses_the_shared_backend_decode_graph_lifecycle() {
    assert!(LFM2_WORKFLOW.contains("let decode = b == 1;"));
    assert!(LFM2_WORKFLOW.contains("if replayed {"));
    assert!(LFM2_WORKFLOW.contains("LFM2 GPU decode prepare failed"));
    assert!(LFM2_WORKFLOW.contains("LFM2 GPU forward failed"));
    assert!(LFM2_WORKFLOW.contains("be().begin_forward(decode);"));
}

#[test]
fn lfm2_quantized_kv_rotation_matches_the_upstream_attention_order() {
    let rope = LFM2_WORKFLOW
        .find("be().rope(BufId::K")
        .expect("LFM2 must rope K before its cache transform");
    let q_rotate = LFM2_WORKFLOW
        .find("be().hadamard(BufId::Q")
        .expect("quantized LFM2 must rotate Q");
    let k_rotate = LFM2_WORKFLOW
        .find("be().hadamard(BufId::K")
        .expect("quantized LFM2 must rotate K");
    let value_tail = &LFM2_WORKFLOW[k_rotate..];
    let v_rotate = k_rotate
        + value_tail
            .find("BufId::V,")
            .expect("quantized LFM2 must rotate V");
    let store = LFM2_WORKFLOW
        .find("be().kv_store(BufId::K")
        .expect("LFM2 must store rotated K");
    let attention = LFM2_WORKFLOW
        .find("be().attention(")
        .expect("LFM2 must run attention");
    let inverse = attention
        + LFM2_WORKFLOW[attention..]
            .find("BufId::Attn,")
            .expect("quantized LFM2 must invert the V basis");
    let output = inverse
        + LFM2_WORKFLOW[inverse..]
            .find("wkind(wo)")
            .expect("LFM2 must project the restored attention basis");
    let workflow_attention = &LFM2_WORKFLOW[attention..];
    assert!(
        workflow_attention.contains("1.0 / (hd as f32).sqrt()"),
        "LFM2 must pass its attention scale to the backend-owned attention op"
    );
    let cuda_attention = body_after(CUDA_BACKEND, "fn attention(");
    let scale_apply = cuda_attention
        .find("self.scale(BufId::Q, scale")
        .expect("CUDA attention must apply the workflow-provided scale");
    let native_attention = cuda_attention
        .find("imparo_cuda_attention(")
        .expect("CUDA attention must reach the native dispatch");
    assert!(scale_apply < native_attention);
    assert!(
        rope < q_rotate
            && q_rotate < k_rotate
            && k_rotate < v_rotate
            && v_rotate < store
            && store < attention
            && attention < inverse
            && inverse < output
    );
    assert!(LFM2_WORKFLOW.contains("if KvType::k() != KvType::F16"));
    assert!(LFM2_WORKFLOW.contains("if KvType::v() != KvType::F16"));
}
#[test]
fn lfm2_graph_stages_token_without_weakening_the_ple_contract() {
    let stage = body_after(NATIVE, "int stage_decode_inputs(uint32_t token)");
    let stage = stage
        .split_once("int upload_staged_decode_inputs()")
        .unwrap()
        .0;
    assert!(stage.contains("if (!g.decode_ple_desc_valid) return 0;"));
    assert!(
        stage.contains(
            "if (!g.ple_stage_host || staged_bytes > g.ple_stage_host_bytes)"
        )
    );
    assert!(
        stage
            .find("*static_cast<uint32_t *>(g.decode_token_stage_host) = token;")
            .unwrap()
            < stage
                .find("if (!g.decode_ple_desc_valid) return 0;")
                .unwrap()
    );

    let upload = body_after(NATIVE, "int upload_staged_decode_inputs()");
    let upload = upload.split_once("} // namespace").unwrap().0;
    assert!(upload.contains("const uint64_t ple_bytes = g.decode_ple_desc_valid"));
    assert!(upload.contains("if (ple_bytes && (!g.ple_stage_host || !g.weight_cache"));
    assert!(upload.contains("if (ple_bytes && cudaMemcpyAsync("));
    assert!(!NATIVE.contains(
        "g.decode_row_desc_valid && g.decode_row_bytes == 0\n        && g.decode_ple_desc_valid"
    ));
}

#[test]
fn d64_graph_route_is_controlled_classified_and_fail_closed() {
    assert!(NATIVE.contains("__global__ void k_attention_d64_controlled("));
    assert!(NATIVE.contains("if (control) start_pos = *control;"));
    assert!(NATIVE.contains("kD64ControlledGraphArgCount = 18"));
    assert!(NATIVE.contains("kD64ControlledPageTableArg = 16"));
    assert!(NATIVE.contains("kD64ControlledControlArg = 17"));
    assert!(
        NATIVE.contains(
            "graph_kernel_is(params.func, (void *)k_attention_d64_controlled)"
        )
    );
    assert!(NATIVE.contains("const bool d64_graph_q4 = n_kv != 0 && direct_q4_decode"));
    assert!(NATIVE.contains("if (g.graph_capturing && d64_graph_q4)"));
    assert!(NATIVE.contains("++g.graph_expected_dynamic_nodes;"));
    assert!(NATIVE.contains(
        "decode_graph_candidate()\n        && !d64_graph_q4 && !d64_graph_q8"
    ));
    let old = NATIVE.find("__global__ void k_attention(").unwrap();
    let controlled = NATIVE
        .find("__global__ void k_attention_d64_controlled(")
        .unwrap();
    assert!(
        old < controlled,
        "the established non-graph symbol remains independent"
    );
}

#[test]
fn d64_attention_keeps_the_oracle_f32_value_reduction_by_default() {
    let controlled = body_after(NATIVE, "if (g.graph_capturing && d64_graph_q4) {")
        .split_once("const bool d256_tiled4 =")
        .unwrap()
        .0;
    assert!(controlled.contains("IMPARO_CUDA_ATTN_D64_HALF"));

    let fallback = body_after(NATIVE, "const bool d64_f32 = head_dim == 64");
    assert!(fallback.contains("IMPARO_CUDA_ATTN_D64_HALF"));
    assert!(fallback.contains("std::getenv(\"IMPARO_CUDA_ATTN_F32\")"));
    assert!(fallback.contains("k_attention<<<"));
}

#[test]
fn d64_q4_decode_owns_a_separate_quant_vector_contract_and_fallback() {
    assert!(D64_Q4_VEC.contains("__launch_bounds__(kThreads, 1) void partial_q4("));
    assert!(D64_Q4_VEC.contains("qk_scale * raw.x"));
    assert!(D64_Q4_VEC.contains("roundf(value.x / d)"));
    assert!(D64_Q4_VEC.contains("if (warp == 0) {"));
    assert!(!D64_Q4_VEC.contains("if (warp == 0 && lane < 16)"));
    assert!(
        D64_Q4_VEC.contains(
            "const uint32_t safe_logical = attended ? logical : logical_base"
        )
    );
    assert!(!D64_Q4_VEC.contains("if (attended) {"));
    assert!(D64_Q4_VEC.contains("const int sumi = __dp4a"));
    assert!(D64_Q4_VEC.contains("d4 * (float(sumi) * ds.x - ds.y)"));
    assert!(D64_Q4_VEC.contains("const float2 value0 = q4_float2"));
    assert!(D64_Q4_VEC.contains("imparo_cuda_kv::physical_row("));
    assert!(D64_Q4_VEC.contains("const uint32_t * decode_control"));
    assert!(D64_Q4_VEC.contains("const uint32_t * page_table"));
    assert!(D64_Q4_VEC.contains("part + stripe_round * parts < stripes"));
    assert!(D64_Q4_VEC.contains("lane < kWarps ? warp_denominators[lane] : 0.0f"));
    assert!(D64_Q4_VEC.contains("const float combined_denominator = warp_sum32("));
    assert!(
        !D64_Q4_VEC.contains("for (uint32_t source = 0; source < kWarps; ++source)")
    );
    assert!(D64_Q4_VEC.contains("numerator += scale * src[i]"));

    assert!(NATIVE.contains("#include \"sm80/attention_decode_vec_d64_q4.cuh\""));
    assert!(
        NATIVE.contains("imparo_sm86_d64_q4_vec_profile::applies_to(g.sm_version)")
    );
    assert!(NATIVE.contains("std::getenv(\"IMPARO_CUDA_ATTN_D64_VEC_Q4\") != nullptr"));
    assert!(NATIVE.contains("IMPARO_CUDA_NO_ATTN_D64_VEC_Q4"));
    assert!(NATIVE.contains("IMPARO_CUDA_ATTN_D64_VEC_PART_CAP"));
    assert!(NATIVE.contains("imparo_sm80_d64_q4_vec::partial_q4"));
    assert!(NATIVE.contains("imparo_sm80_d64_q4_vec::combine_q4"));
    assert!(NATIVE.contains("kD64Q4VecGraphArgCount = 14"));
    assert!(NATIVE.contains("kD64Q4VecPageTableArg = 13"));
    assert!(NATIVE.contains(
        "graph_kernel_is(\n                params.func, (void *)imparo_sm80_d64_q4_vec::partial_q4)"
    ));
    let vector = NATIVE.find("const bool d64_vec_q4 =").unwrap();
    let controlled = NATIVE
        .find("if (g.graph_capturing && d64_graph_q4) {")
        .unwrap();
    assert!(
        vector < controlled,
        "the controlled scalar route remains the fallback"
    );
}
#[test]
fn d64_q8_decode_is_a_receipted_vector_route_with_graph_replay() {
    assert!(D64_Q8_VEC.contains("void partial_q8("));
    assert!(D64_Q8_VEC.contains("const int sumi = __dp4a("));
    assert!(D64_Q8_VEC.contains("float(sumi) * d8 * q_ds[block_index].x"));
    assert!(D64_Q8_VEC.contains("const float2 value0 = q8_float2"));
    assert!(D64_Q8_VEC.contains("const uint32_t * decode_control"));
    assert!(D64_Q8_VEC.contains("const uint32_t * page_table"));
    assert!(NATIVE.contains("g.knobs[41] = 0;"));
    assert!(NATIVE.contains("const bool d64_q8_vec_requested = tuner_knob(41) != 0"));
    assert!(NATIVE.contains("g.kv_type_k == 8 && g.kv_type_v == 8"));
    assert!(NATIVE.contains("imparo_sm80_d64_q8_vec::partial_q8"));
    assert!(NATIVE.contains("imparo_sm80_d64_q4_vec::combine_q4"));
    assert!(NATIVE.contains("const bool d64_graph_q8 = n_kv != 0"));
    assert!(NATIVE.contains("(!g.graph_capturing || d64_graph_q8)"));
    let q8_launch = body_after(NATIVE, "const bool d64_vec_q8 =")
        .split_once("if (g.graph_capturing && d64_graph_q4)")
        .expect("D64 Q8 launch boundary")
        .0;
    let graph_accounting = q8_launch.find("if (g.graph_capturing) {").unwrap();
    let optional_trace = q8_launch
        .find("if (std::getenv(\"IMPARO_CUDA_ATTN_D64_VEC_TRACE\"))")
        .unwrap();
    assert!(graph_accounting < optional_trace);

    assert!(NATIVE.contains("params.func, (void *)imparo_sm80_d64_q8_vec::partial_q8"));
    assert!(NATIVE.contains("&& !d64_graph_q4 && !d64_graph_q8"));

    let declaration = body_after(CUDA_KNOBS, "\"attn_d64_q8_vec\",")
        .split_once("),")
        .expect("D64 Q8 vector declaration must terminate")
        .0;
    assert!(declaration.contains("SLOT_ATTN_D64_Q8_VEC"));
    assert!(declaration.contains("Wl::AttentionDecodeDeep"));
    assert!(declaration.contains("bit_affecting = true"));
    assert!(declaration.contains("m.deep_head_dim == 64"));
    assert!(declaration.contains("m.n_head == 4 * m.n_kv"));
}

#[test]
fn d256_gqa4_decode_shares_kv_without_changing_the_numeric_or_fallback_contract() {
    for marker in [
        "void partial_q4_gqa4(",
        "void combine_q4_gqa4_final(",
        "constexpr uint32_t kGqaChildStride = kHeadDim + 2 + 32",
        "dst[kHeadDim + 2 + lane] = running_sum",
        "__fmul_rn(scales[0], base[i])",
        "__fadd_rn(numerator, scaled)",
    ] {
        assert!(
            D256_Q4_VEC.contains(marker),
            "D256 GQA4 numerical contract lost marker {marker}"
        );
    }
    assert!(D256_Q4_VEC.contains("void partial_q4("));
    assert!(D256_Q4_VEC.contains("void combine_q4("));

    for marker in [
        "g.sm_version == 86",
        "n_heads / n_kv == imparo_sm80_d256_vec::kGqaHeads",
        "window == imparo_sm80_d256_vec::kWindowSpan",
        "g.kv_type_k == 2 && g.kv_type_v == 2",
        "IMPARO_CUDA_NO_ATTN_D256_GQA4",
        "IMPARO_CUDA_ATTN_D256_GQA4_COMPARE",
        "imparo_sm80_d256_vec::partial_q4_gqa4",
        "imparo_sm80_d256_vec::combine_q4_gqa4_final",
    ] {
        assert!(
            NATIVE.contains(marker),
            "native selector lost marker {marker}"
        );
    }
    assert!(
        NATIVE.contains("params.func, (void *)imparo_sm80_d256_vec::partial_q4_gqa4")
    );
}

#[test]
fn d512_fused_prefill_reuses_stable_cell_bounds_and_fails_closed() {
    for marker in [
        "uint32_t canonical_parts, uint32_t virtual_stream_blocks",
        "stable_virtual_stream_segment_bounds(",
        "canonical_segment_bounds(",
        "stream_segment_bounds(",
        "for (uint32_t slot_rev = segment_slots; slot_rev > 0; --slot_rev)",
        "__fmul_rn(scale0, part_sum[qcol])",
        "__fmul_rn(scale1, part_sum[kColumns + qcol])",
    ] {
        assert!(
            D512_FLASH.contains(marker),
            "D512 fused numerical contract lost marker {marker}"
        );
    }

    for marker in [
        "g.sm_version == 86 && stable_cell_d512 && ring == 0",
        "canonical_parts > 0 && virtual_stream_blocks == 0",
        "segment_slots <= imparo_sm80_prefill::kCanonicalD512Parts",
        "IMPARO_CUDA_ENABLE_ATTN_D512_FUSED",
        "IMPARO_CUDA_NO_ATTN_D512_FUSED",
        "canonical_parts, virtual_stream_blocks",
    ] {
        assert!(
            NATIVE.contains(marker),
            "D512 fused selector lost marker {marker}"
        );
    }
    assert!(
        NATIVE.contains("fused_d512_stable\n                || std::getenv("),
        "only the receipt-backed StableCell route may be default-on"
    );
}

#[test]
fn q8_large_batches_use_an_independent_full_precision_d4_contract() {
    let matmat = body_after(NATIVE, "extern \"C\" void imparo_cuda_matmat(");
    let matmat = matmat
        .split_once("extern \"C\" void imparo_cuda_matmat_gated")
        .unwrap()
        .0;
    assert!(
        matmat.contains("bool q8_mmq = q8_weight && g.sm_version >= 80 && n_tok > 8")
    );
    assert!(NATIVE.contains("Q8_LAYOUT_MMQ_D4 = 3"));
    assert!(matmat.contains("q8_mmq ? Q8_LAYOUT_MMQ_D4"));
    assert!(matmat.contains("n_in, n_tok, src_row, q8_mmq"));

    let quantizer = body_after(NATIVE, "__global__ void k_quantize_q8_1_mmq(");
    let quantizer = quantizer
        .split_once("__device__ __forceinline__ float dot_q4_0_q8_1_half")
        .unwrap()
        .0;
    assert!(quantizer.contains("bool full_precision_scale"));
    assert!(quantizer.contains(
        "full_precision_scale\n            ? d : __half2float(__float2half(d))"
    ));

    assert!(matmat.contains("imparo_sm80_q8_mmq::launch("));
    assert!(Q8_MMQ.contains("Q8_0 x Q8_1 MMQ for batches above the MMVQ boundary"));
    assert!(Q8_MMQ.contains("kActivationStride == 144"));
    assert!(Q8_MMQ.contains("mma_m16n8k32"));
    assert!(Q8_MMQ.contains("bool AlignedWholeK = false, bool TileMajor = false"));
    assert!(Q8_MMQ.contains("if constexpr (!AlignedWholeK)"));
    assert!(Q8_MMQ.contains("q8_0_q8_1_mma<Tokens, true, TileMajor, Epilogue>"));
    assert!(Q8_MMQ.contains("inline LaunchResult launch_aligned_whole_k("));
    assert!(NATIVE.contains("g.knobs[42] = 0;"));
    assert!(NATIVE.contains("const bool aligned_whole_k_requested = tensor_resident"));
    assert!(NATIVE.contains("tuner_knob(42) != 0"));
    assert!(NATIVE.contains("launch_aligned_whole_k("));
    assert!(NATIVE.contains(
        "launch_result\n                    == imparo_sm80_q8_mmq::LaunchResult::NotSupported"
    ));
}
