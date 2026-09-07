const BACKEND: &str = include_str!("../../imparo-backend/src/lib.rs");
const FFI: &str = include_str!("../src/ffi.rs");
const CUDA_BACKEND: &str = include_str!("../src/backend_impl.rs");
const METAL: &str = include_str!("../../imparo-metal/src/backend_impl.rs");
const WORKFLOW: &str = include_str!("../../imparo-model/src/gemma4/workflow_gpu.rs");
const LFM2_WORKFLOW: &str = include_str!("../../imparo-model/src/lfm2/workflow_gpu.rs");
const NATIVE: &str = include_str!("../native/imparo_cuda.cu");
const PLE_GATE: &str = include_str!("../native/sm86/mmq_q4_q8_ple_gate.cuh");
const READY: &str = include_str!("../native/sm86/q8_ready_batched_r2.cuh");
const TUNER_MICRO: &str = include_str!("../../imparo-tune/src/micro.rs");
const TUNER_MAIN: &str = include_str!("../../imparo-tune/src/main.rs");
const TUNER_KNOBS: &str = include_str!("../../imparo-tune/src/knobs.rs");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

#[test]
fn shortconv_fused_route_records_tuner_dispatch_proof() {
    let shortconv =
        braced_item_after(NATIVE, "extern \"C\" void imparo_cuda_shortconv(");
    let selector = shortconv
        .find("const uint32_t fused_selector = tuner_knob(51);")
        .expect("ShortConv selector read");
    let proof = shortconv
        .find("mark_tuner_dispatch(4,")
        .expect("ShortConv tuner dispatch proof");
    let candidate = shortconv
        .find("if (fused_selector != 0 && g.sm_version == 86 && n_tok == 1)")
        .expect("ShortConv candidate branch");
    assert!(selector < proof && proof < candidate);
    assert!(shortconv.contains("if (fused_selector != 0 && g.tuner_mode)"));
    assert!(shortconv.contains("shortconv fused candidate did not dispatch"));
}

#[test]
fn decode_projection_preparation_is_cuda_policy_not_a_cross_backend_default() {
    let method = compact(braced_item_after(
        BACKEND,
        "fn use_decode_projection_preparation(",
    ));
    assert!(method.contains(")->bool{false}"));
    assert!(!METAL.contains("fn use_decode_projection_preparation("));
    assert!(
        CUDA_BACKEND
            .contains("crate::knobs::decode_graph_q8_producer_reuse_enabled()",)
    );
}

#[test]
fn lfm2_decode_prepares_each_projection_boundary_without_touching_prefill() {
    let workflow = compact(LFM2_WORKFLOW);
    assert!(workflow.contains(
        "letdecode_projection_preparation=decode&&be().use_decode_projection_preparation();",
    ));
    assert_eq!(
        LFM2_WORKFLOW.matches("be().rms_norm_projection(").count(),
        3,
        "operator, FFN and final projection boundaries must be explicit",
    );
    assert_eq!(
        workflow
            .matches("ifdecode_projection_preparation&&li!=gpu_probe_layer()")
            .count(),
        2,
        "layer probes must retain their established materialized path",
    );
    assert!(workflow.contains(
        "ifdecode_projection_preparation{be().rms_norm_projection(BufId::X,BufId::X,",
    ));
}

#[test]
fn capture_local_q8_reuse_requires_current_decode_generation() {
    let cache_match = compact(braced_item_after(NATIVE, "bool q8_cache_matches("));
    assert!(cache_match.contains("g.graph_capturing"));
    assert!(cache_match.contains("g.knobs[52]==0"));
    assert!(cache_match.contains("!g.forward_decode"));
    assert!(cache_match.contains("g.prefill_capture_active"));
    assert!(
        cache_match
            .contains("g.q8_owner_capture_generation!=g.graph_capture_generation",)
    );

    let owner = compact(braced_item_after(NATIVE, "void own_q8_cache("));
    assert!(owner.contains(
        "g.graph_capturing&&g.forward_decode&&!g.prefill_capture_active?g.graph_capture_generation:0",
    ));
    let invalidate = compact(braced_item_after(NATIVE, "void invalidate_q8_cache("));
    assert!(invalidate.contains("g.q8_owner_capture_generation=0;"));
}

#[test]
fn capture_and_policy_changes_fail_closed_for_q8_ownership() {
    let begin = compact(braced_item_after(NATIVE, "static void begin_forward("));
    let invalidate = begin
        .find("invalidate_q8_cache();")
        .expect("capture must invalidate any eager Q8 owner");
    let capture = begin
        .find("cudaStreamBeginCapture(")
        .expect("Decode capture start");
    let generation = begin
        .find("++g.graph_capture_generation;")
        .expect("successful capture generation bump");
    assert!(invalidate < capture && capture < generation);

    let end = compact(braced_item_after(
        NATIVE,
        "extern \"C\" int imparo_cuda_end(",
    ));
    assert!(end.contains(
        "cudaStreamEndCapture(g.stream,&captured);g.graph_capturing=false;invalidate_q8_cache();",
    ));
    let set_knob = compact(braced_item_after(
        NATIVE,
        "extern \"C\" void imparo_cuda_set_knob(",
    ));
    assert!(set_knob.contains("destroy_decode_graph();"));
    assert!(set_knob.contains("invalidate_q8_cache();"));
    assert!(NATIVE.contains("cudaGraphGetNodes("));
    assert!(NATIVE.contains("dynamic=%zu/%u total=%zu total_ok=%u"));
}

fn braced_item_after<'a>(source: &'a str, marker: &str) -> &'a str {
    let start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing source marker {marker:?}"));
    let tail = &source[start..];
    let open = tail
        .find('{')
        .unwrap_or_else(|| panic!("marker {marker:?} has no body"));
    let mut depth = 0_u32;
    for (offset, byte) in tail.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &tail[..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unclosed body after marker {marker:?}");
}

fn braced_item_containing<'a>(source: &'a str, marker: &str) -> &'a str {
    let marker_start = source
        .find(marker)
        .unwrap_or_else(|| panic!("missing source marker {marker:?}"));
    let item_start = source[..marker_start]
        .rfind("KnobDecl {")
        .unwrap_or_else(|| panic!("marker {marker:?} is not inside a KnobDecl"));
    braced_item_after(&source[item_start..], "KnobDecl {")
}

#[test]
fn backend_default_is_false_and_metal_does_not_override_it() {
    let method = compact(braced_item_after(BACKEND, "fn ffn_gated_down("));
    assert!(
        method.contains(")->bool{false}"),
        "Backend default must remain a side-effect-free false capability"
    );
    assert!(BACKEND.contains("Returning `false` promises"));
    assert!(BACKEND.contains("that no public buffer has been written"));
    assert!(
        !METAL.contains("fn ffn_gated_down("),
        "Metal must inherit the conservative default until it owns this transaction"
    );
}

#[test]
fn dynamic_surface_resolves_the_versioned_prefill_cache_transaction() {
    let fields_start = FFI.find("macro_rules! fields").unwrap();
    let fields_end =
        FFI[fields_start..].find("fields!(declare_api);").unwrap() + fields_start;
    assert!(
        FFI[fields_start..fields_end].contains("imparo_cuda_ffn_gated_down("),
        "ABI 26 must resolve the full FFN transaction"
    );
    assert!(
        FFI[fields_start..fields_end]
            .contains("imparo_cuda_prepare_quantized_weight_cache("),
        "ABI 26 must resolve admission-time quantized-weight preparation"
    );
    assert!(
        !FFI[fields_start..fields_end].contains("imparo_cuda_enable_tuner_lab"),
        "the static tuner switch must not expand the production plugin ABI"
    );
}

#[test]
fn gemma_workflow_guards_the_entire_legacy_sequence_and_probe_path() {
    let workflow = compact(WORKFLOW);
    assert!(
        workflow.contains(
            "letfused_ffn=!gate0_needs_down_input&&li!=gpu_probe_layer()&&be().ffn_gated_down("
        )
    );
    assert!(
        workflow.contains(
            "n_embd,n_ff,n_embd,BufId::Cur,BufId::G,BufId::X,b,);if!fused_ffn{"
        )
    );
    assert_eq!(workflow.matches("be().ffn_gated_down(").count(), 1);

    let fallback = compact(braced_item_after(WORKFLOW, "if !fused_ffn {"));
    let gated = fallback.find("be().matmat_gated(").unwrap();
    let gate = fallback.find("be().matmat(wkind(&lw.ffn_gate),").unwrap();
    let up = fallback.find("be().matmat(wkind(&lw.ffn_up),").unwrap();
    let act = fallback
        .find("be().act_mul(BufId::G,BufId::U,b*n_ff)")
        .unwrap();
    let down_call = "be().matmat(wkind(&lw.ffn_down),lw.ffn_down.offsetasu64,n_ff,n_embd,\
         BufId::G,BufId::X,b,)";
    let down = fallback.find(down_call).unwrap();
    assert!(gated < gate && gate < up && up < act && act < down);

    for probe in [
        "gprobe(\"gate_raw_full\"",
        "gprobe(\"ffn_geglu\"",
        "gprobe(\"ffn_down_out\"",
    ] {
        assert!(fallback.contains(probe), "probe escaped fallback: {probe}");
    }
    assert_eq!(workflow.matches(down_call).count(), 1);
}

#[test]
fn lfm2_workflow_uses_the_complete_transaction_without_weakening_probes() {
    let workflow = compact(LFM2_WORKFLOW);
    assert!(
        workflow.contains("letfused_ffn=li!=gpu_probe_layer()&&be().ffn_gated_down(")
    );
    assert!(
        workflow.contains(
            "n_embd,n_ff,n_embd,BufId::Cur,BufId::G,BufId::O,b,);if!fused_ffn{"
        )
    );
    assert_eq!(workflow.matches("be().ffn_gated_down(").count(), 1);

    let fallback = compact(braced_item_after(LFM2_WORKFLOW, "if !fused_ffn {"));
    assert!(fallback.contains("be().matmat_gated("));
    assert!(fallback.contains(
        "be().matmat(wkind(&lw.ffn_down),lw.ffn_down.offsetasu64,n_ff,n_embd,"
    ));
    assert!(fallback.contains("gprobe(\"ffn_swiglu_last\""));
    assert!(fallback.contains("gprobe(\"ffn_down_last\""));
    assert!(!fallback.contains("be().add(BufId::X,BufId::O,b*n_embd)"));
    let fallback_start = workflow.find("if!fused_ffn{").unwrap();
    let residual_after_fallback = workflow[fallback_start..]
        .find("be().add(BufId::X,BufId::O,b*n_embd)")
        .unwrap();
    assert!(residual_after_fallback >= fallback.len());

    let transaction = compact(NATIVE);
    assert!(transaction.contains("try_q8_tm_silu_private_down("));
    assert!(transaction.contains("g.epilogue=4"));
    assert!(transaction.contains("launch_aligned_whole_k<true,4>"));
}

#[test]
fn lfm2_attention_uses_semantic_qk_fusion_without_weakening_probes() {
    let workflow = compact(LFM2_WORKFLOW);
    assert!(
        workflow.contains(
            "letqk_hadamard_nrot=ifKvType::k()==KvType::F16{0}else{had_nrot("
        )
    );
    assert_eq!(workflow.matches("be().head_norm_rope_hadamard(").count(), 2);
    assert!(workflow.contains(
        "be().head_norm_rope_hadamard(BufId::Q,q_norm.offset,hd,eps,n_head,sp,b,"
    ));
    assert!(workflow.contains(
        "be().head_norm_rope_hadamard(BufId::K,k_norm.offset,hd,eps,n_kv,sp,b,"
    ));

    let probe = compact(braced_item_after(
        LFM2_WORKFLOW,
        "if li == gpu_probe_layer() {\n                    // Preserve the individually observable operations",
    ));
    for operation in [
        "be().rms_norm(BufId::Q",
        "be().rms_norm(BufId::K",
        "be().rope(BufId::Q",
        "be().rope(BufId::K",
        "be().hadamard(BufId::Q",
        "be().hadamard(BufId::K",
    ] {
        assert!(probe.contains(operation), "probe path lost {operation}");
    }
    assert!(!probe.contains("be().head_norm_rope_hadamard("));
}

#[test]
fn failure_injection_preserves_the_transaction_commit_boundary() {
    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    for flag in [
        "IMPARO_CUDA_PREFILL_FFN_SIDECAR_FAIL_PRECOMMIT_LAB",
        "IMPARO_CUDA_PREFILL_FFN_SIDECAR_FAIL_POSTCOMMIT_LAB",
    ] {
        assert!(transaction.contains(flag), "missing injection flag {flag}");
    }
    assert!(
        transaction.contains("std::strcmp(fail_precommit_env,\"1\")==0"),
        "pre-commit injection must require the exact laboratory value"
    );
    assert!(
        transaction.contains("std::strcmp(fail_postcommit_env,\"1\")==0"),
        "post-commit injection must require the exact laboratory value"
    );

    let producer_done = transaction.find("if(!gate_up_launched){").unwrap();
    let precommit_return = transaction.find("if(fail_precommit){").unwrap();
    let public_commit = transaction
        .find("boolpublic_output_committed=false;")
        .unwrap();
    assert!(producer_done < precommit_return && precommit_return < public_commit);
    let precommit_body = compact(braced_item_after(
        &transaction[precommit_return..],
        "if(fail_precommit)",
    ));
    assert!(precommit_body.contains("return0;"));
    assert!(precommit_body.contains("reject_exact128_graph_capture("));

    let postcommit_injection = transaction
        .find("if(down_launched&&fail_postcommit){public_output_committed=true;down_launched=false;}")
        .unwrap();
    let fatal = transaction
        .find("if(!down_launched){cudaGetLastError();if(public_output_committed){")
        .unwrap();
    let fallback_return = transaction[fatal..].find("return0;").unwrap() + fatal;
    assert!(postcommit_injection < fatal && fatal < fallback_return);
    let fatal_body = compact(braced_item_after(
        &transaction[fatal..],
        "if(public_output_committed)",
    ));
    assert!(fatal_body.contains("mark_buf_written(dst);"));
    assert!(fatal_body.contains("set_pending(CUDA_RC_ERROR,"));
    assert!(fatal_body.contains("return1;"));
}

#[test]
fn sidecar_is_prefill_only_and_validates_logical_grid_before_packing() {
    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    assert!(transaction.contains("g.batch_geometry_phase==0"));
    assert!(
        transaction
            .contains("if(!logical_tiles64||logical_tiles64>UINT32_MAX)return0;")
    );
    assert!(
        !transaction.contains("logical_tiles64<uint32_t(g.sm_count)"),
        "Stream-K-capable sub-SM logical grids must be decided by measured selector evidence"
    );
    let reachability = transaction.find("constuint64_tlogical_tiles64=").unwrap();
    let first_pack = transaction
        .find("ensure_aligned_packed_q4_lab(gate,")
        .unwrap();
    assert!(
        reachability < first_pack,
        "logical-tile rejection must happen before allocating packed sidecars"
    );
}

#[test]
fn model_sidecar_fit_and_commit_are_all_or_none() {
    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    let model_fit = transaction
        .find("layer_bytes*layers>g.packed_q4_budget")
        .unwrap();
    let first_pack = transaction
        .find("ensure_aligned_packed_q4_lab(gate,")
        .unwrap();
    assert!(model_fit < first_pack);
    let complete_pack = transaction
        .find("if(!kernel_lab&&!g.ffn_sidecar_model_ready){++g.ffn_sidecar_model_pack_calls;return0;}")
        .unwrap();
    let first_public_output = transaction
        .find("boolpublic_output_committed=false;")
        .unwrap();
    assert!(first_pack < complete_pack && complete_pack < first_public_output);
    let incomplete_pack = transaction
        .find("if(!packed_gate||!packed_up||!packed_down){")
        .unwrap();
    let incomplete_pack_body = compact(braced_item_after(
        &transaction[incomplete_pack..],
        "if(!packed_gate||!packed_up||!packed_down)",
    ));
    assert!(incomplete_pack_body.contains("g.ffn_sidecar_model_pack_failed=true;"));
    assert!(transaction.contains("++g.ffn_sidecar_model_commits;"));

    let end = compact(braced_item_after(
        NATIVE,
        "extern \"C\" int imparo_cuda_end(",
    ));
    assert!(end.contains(
        "!g.ffn_sidecar_model_ready&&g.ffn_sidecar_model_calls&&g.ffn_sidecar_model_calls==g.kv_layout.layers&&g.ffn_sidecar_model_pack_calls==g.kv_layout.layers&&!g.ffn_sidecar_model_pack_failed&&launch==cudaSuccess&&sync==cudaSuccess&&!g.pending_error"
    ));
    assert!(end.contains("g.ffn_sidecar_model_ready=true;"));
    assert!(end.contains(
        "g.ffn_sidecar_model_calls!=g.ffn_sidecar_model_commits||g.ffn_sidecar_model_commits!=g.kv_layout.layers"
    ));
    assert!(end.contains("sidecar-onlyFFNmodeladmissionwasnotall-or-none"));
}

#[test]
fn optional_sidecar_yields_to_activation_kv_and_decode_allocations() {
    let reclaim = compact(braced_item_after(
        NATIVE,
        "bool release_packed_q4_for_priority_allocation() {",
    ));
    for required in [
        "destroy_decode_graph();",
        "destroy_prefill_graph();",
        "cudaStreamSynchronize(g.stream)",
        "cudaFree(span.packed)",
        "g.packed_q4_budget_initialized=false;",
        "g.ffn_sidecar_model_ready=false;",
    ] {
        assert!(
            reclaim.contains(required),
            "missing reclaim invariant {required}"
        );
    }

    let retry = compact(braced_item_after(
        NATIVE,
        "int alloc_raw_with_packed_q4_reclaim(",
    ));
    assert!(retry.contains("g.packed_q4_force_reclaim_active&&g.packed_q4_bytes"));
    assert!(retry.contains("if(rc!=CUDA_RC_OOM)returnrc;"));
    assert!(retry.contains("release_packed_q4_for_priority_allocation()"));
    assert_eq!(retry.matches("alloc_raw(out,bytes,what)").count(), 2);

    let alloc = compact(braced_item_after(
        NATIVE,
        "extern \"C\" int imparo_cuda_alloc(",
    ));
    let arena = compact(braced_item_after(
        NATIVE,
        "extern \"C\" int imparo_cuda_arena(",
    ));
    let kv = compact(braced_item_after(NATIVE, "static int replace_kv_arena("));
    assert!(alloc.contains("alloc_raw_with_packed_q4_reclaim("));
    assert!(arena.contains("alloc_raw_with_packed_q4_reclaim("));
    assert_eq!(kv.matches("alloc_raw_with_packed_q4_reclaim(").count(), 2);

    let q8 = compact(braced_item_after(NATIVE, "int ensure_q8_scratch("));
    let attention =
        compact(braced_item_after(NATIVE, "bool ensure_attention_scratch("));
    assert!(q8.contains("rc==CUDA_RC_OOM&&g.forward_decode"));
    assert!(
        attention.contains("allocation==cudaErrorMemoryAllocation&&g.forward_decode")
    );

    let begin = compact(braced_item_after(NATIVE, "static void begin_forward("));
    assert!(begin.contains("IMPARO_CUDA_Q4_PACKED_SIDECAR_FORCE_RECLAIM_LAB"));
    assert!(begin.contains("std::strcmp(force_reclaim_env,\"1\")==0"));
    assert!(begin.contains("g.packed_q4_force_reclaim_exercised=true;"));
    assert!(begin.contains("g.packed_q4_force_reclaim_active=true;"));
    assert!(begin.contains("g.packed_q4_force_reclaim_active=false;"));
}

#[test]
fn prefill_graph_admits_only_the_warmed_receipted_exact128_sidecar() {
    let helper = compact(braced_item_after(
        CUDA_BACKEND,
        "fn ffn_sidecar_requested(n_tok: u32)",
    ));
    assert!(helper.contains("if!(9..=512).contains(&n_tok){returnfalse;}"));
    assert!(helper.contains(
        "lab_requested||exact128||(tuned_min_tokens!=0&&n_tok>=tuned_min_tokens)"
    ));

    let graph_request = compact(braced_item_after(
        CUDA_BACKEND,
        "fn split_prefill_graph_requested(count: u32)",
    ));
    assert!(graph_request.contains("prefill_exact128_sm86_route_enabled()"));
    assert!(graph_request.contains("prefill_exact128_fast_transaction_enabled()"));
    assert!(graph_request.contains("prefill_exact128_graph_enabled()"));
    let prepare = compact(braced_item_after(CUDA_BACKEND, "fn prefill_prepare("));
    assert!(prepare.contains(
        "ifffn_sidecar_requested(count)&&!split_prefill_graph_requested(count){returnOk(false);}"
    ));

    let candidate = compact(braced_item_after(NATIVE, "bool prefill_graph_candidate("));
    assert!(candidate.contains("token_count>1"));
    assert!(candidate.contains("prefill_graph_enabled_for(token_count)"));
    assert!(!candidate.contains("&&argmax"));
    let policy = compact(braced_item_after(NATIVE, "bool prefill_graph_enabled_for("));
    assert!(policy.contains("g.knobs[40]==1"));
    assert!(policy.contains("tuned_exact128_sm86_route(token_count)"));
    assert!(policy.contains("tuned_exact128_fast_transaction(token_count)"));
    assert!(policy.contains("prefill_graph_lab_override()"));
    let graph_decl = compact(braced_item_containing(
        include_str!("../src/knobs.rs"),
        "name: \"prefill_exact128_graph\"",
    ));
    assert!(graph_decl.contains("category:KnobCategory::EndToEnd"));
    assert!(graph_decl.contains("sweep:Sw::External"));

    let tuner = compact(TUNER_MICRO);
    assert!(
        tuner
            .contains("SweepKind::Derived|SweepKind::External|SweepKind::Values=>None")
    );
    assert!(tuner.contains(
        "SweepKind::External=>{letvalue=(d.current)();ifverbose{println!(\"{}={value}(externalwhole-engineadmission;microtunerpreservesincumbent)\",d.name);}value}"
    ));
    assert!(
        !tuner.contains("SweepKind::External=>sweep(d)"),
        "end-to-end policies must never be ranked by the micro sweep"
    );

    let ready = compact(braced_item_after(
        NATIVE,
        "bool prefill_exact128_sidecar_capture_ready(",
    ));
    assert!(ready.contains("g.graph_capturing&&g.prefill_capture_active"));
    assert!(ready.contains("tuned_exact128_fast_transaction(n_tok)"));
    assert!(ready.contains("g.ffn_sidecar_model_ready&&g.prefill_warm_forwards>=2"));

    let native_prepare = compact(braced_item_after(
        NATIVE,
        "extern \"C\" int imparo_cuda_prefill_prepare(",
    ));
    assert!(
        native_prepare
            .contains("constuint32_trequired_warm_forwards=exact128_sidecar?2u:1u;")
    );
    assert!(native_prepare.contains("tuned_exact128_fast_transaction(count)"));
    assert!(native_prepare.contains("(!exact128_sidecar||g.ffn_sidecar_model_ready)"));
    assert!(compact(NATIVE).contains("[cuda-prefill-graph]action=replay"));

    let ffn = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    assert!(ffn.contains("||exact128_graph_capture)"));
    assert!(ffn.contains("exact128FFNGraphincompletepacked-Q4hotset"));
    assert!(ffn.contains("exact128FFNGraphpreflight"));
    assert!(ffn.contains("g.graph_capture_compatible=false;"));

    let ple = compact(braced_item_after(
        NATIVE,
        "extern \"C\" void imparo_cuda_ple_project(",
    ));
    assert!(ple.contains("exact128PLEGraphprojectionroute"));
    assert!(ple.contains("exact128PLEGraphreadyrouteunavailable"));
    assert!(NATIVE.contains("[cuda-prefill-graph] action=capture"));
    assert!(NATIVE.contains("[cuda-prefill-graph] action=discard"));
    assert!(
        NATIVE
            .matches("if (g.graph_capturing && !g.prefill_capture_active)")
            .count()
            >= 2
    );
}

#[test]
fn ple_q8_ready_treats_true_as_success() {
    let ple = compact(braced_item_after(
        NATIVE,
        "extern \"C\" void imparo_cuda_ple_project(",
    ));
    assert!(ple.contains(
        "ensure_q8_scratch_next(quant_bytes+scale_bytes)&&(direct_projection||ensure_attention_scratch(fixup_bytes))"
    ));
    assert!(!ple.contains("ensure_q8_scratch_next(quant_bytes+scale_bytes)==0"));
    assert!(ple.contains("launch_q8_ready_direct("));
    assert!(ple.contains("&public_output_committed"));
    assert!(ple.contains("if(public_output_committed)"));
    assert!(ple.contains("SM86fusedPLEprojectionfailedaftercommit"));
}

#[test]
fn ple_short_direct_k_is_explicit_exact_128_and_keeps_long_route_separate() {
    let ple = compact(braced_item_after(
        NATIVE,
        "extern \"C\" void imparo_cuda_ple_project(",
    ));
    assert!(ple.contains("IMPARO_CUDA_PLE_FUSED_SHORT_DIRECT_K_LAB"));
    assert!(ple.contains(
        "(ple_fused_sm86&&n_tok>448&&n_tok<=512)||(ple_fused_short_direct_k_sm86&&n_tok==128)||tuned_exact128_fast_transaction(n_tok)||tuned_exact128_sm86_route(n_tok)"
    ));
    assert!(ple.contains(
        "constboolple_ready_route=tuned_exact128_sm86_route(n_tok)||tuned_exact128_fast_transaction(n_tok)||std::getenv(\"IMPARO_CUDA_PLE_FUSED_Q8_READY_LAB\")!=nullptr"
    ));

    let kernel = compact(PLE_GATE);
    assert!(kernel.contains("template<boolReadyOutput,boolDirectK=false>"));
    assert!(kernel.contains("ifconstexpr(DirectK)"));
    assert!(kernel.contains("n_tok,0,kBlocks"));
    assert!(kernel.contains("constbooldirect_k=n_tok==128"));
    assert!(kernel.contains(
        "constuint32_tgrid=(kOutput/kRows)*((n_tok+kTileTokens-1)/kTileTokens)"
    ));
}

#[test]
fn ffn_down_full_k_128_is_exact_shape_and_keeps_449_selector_separate() {
    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    assert!(transaction.contains("IMPARO_CUDA_PREFILL_FFN_DOWN_FULL_K_128_LAB"));
    assert!(transaction.contains("IMPARO_CUDA_PREFILL_FFN_DOWN_FULL_K_449_LAB"));
    assert!(transaction.contains(
        "constchar*full_k_env=n_tok==128?full_k_128_env:(n_tok==449?full_k_449_env:nullptr)"
    ));
    assert!(transaction.contains(
        "constboolfull_k_shape=tuned_exact128||(n_tok==128&&full_k_128_env)||(n_tok==449&&full_k_449_env)"
    ));
    assert!(transaction.contains("if(full_k_requested){"));
    assert!(transaction.contains("Ready::launch_q8_ready_direct("));
    assert!(
        transaction
            .contains("(!full_k_requested&&!ensure_attention_scratch(fixup_bytes))")
    );
}

#[test]
fn ffn_down_r96_lab_is_default_off_tail_guarded_and_keeps_r128_production() {
    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    let ready = compact(READY);

    assert!(transaction.contains(
        "Ready::DirectRowsfull_k_rows=tuned_exact128?Ready::DirectRows::Rows128:Ready::DirectRows::Environment"
    ));
    assert!(ready.contains(
        "{DirectRows::Rows96,\"r96\",\"ffn_sidecar_down_full_k_r96\",kReadyBalancedRows,kReadyBalancedWarps,kReadyBalancedSharedBytes}"
    ));
    assert!(ready.contains("Rows96=96"));
    assert!(ready.contains("kReadyBalancedRows=96"));
    assert!(ready.contains("kReadyBalancedWarps=6"));
    assert!(ready.contains("q8_ready_full_tile_direct_r96"));
    assert!(
        ready.contains(
            "compute_segment<true,kReadyBalancedRows,kReadyBalancedWarps,true>"
        )
    );
    assert!(ready.contains("local_row<active_rows"));
    assert!(ready.contains("local_row>=active_rows"));
    assert!(ready.contains("(n_out+tile_rows-1)/tile_rows"));
    assert!(ready.contains("configure_q8_ready_direct_r96"));
}

#[test]
fn ffn_down_r96_w24_partitions_token_groups_without_changing_production() {
    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    let ready = compact(READY);

    assert!(ready.contains(
        "{DirectRows::Rows96Warp24,\"r96w24\",\"ffn_sidecar_down_full_k_r96_w24\",kReadyBalancedRows,kReadyPartitionedWarps,kReadyBalancedSharedBytes}"
    ));
    assert!(transaction.contains(
        "Ready::DirectRowsfull_k_rows=tuned_exact128?Ready::DirectRows::Rows128:Ready::DirectRows::Environment"
    ));
    assert!(ready.contains("Rows96Warp24=0x6018"));
    assert!(ready.contains("kReadyPartitionedWarps=24"));
    assert!(ready.contains("__launch_bounds__(768,1)"));
    assert!(ready.contains("q8_ready_full_tile_direct_r96_w24"));
    assert!(ready.contains("floatpartial[16]"));
    assert!(ready.contains(
        "compute_segment<true,kReadyBalancedRows,kReadyPartitionedWarps,true,true>"
    ));
    assert!(ready.contains("constuint32_trow_pair=warp_pair%3"));
    assert!(ready.contains("constuint32_ttoken_group=warp_pair/3"));
    assert!(ready.contains("((token_fragment*2+row_fragment)*4)+item"));
    assert!(
        ready.contains("dim3(32,kReadyPartitionedWarps),kReadyBalancedSharedBytes")
    );
}

#[test]
fn ffn_down_r80_is_exact_divisor_default_off_and_keeps_r128_production() {
    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    let ready = compact(READY);

    assert!(ready.contains(
        "{DirectRows::Rows80,\"r80\",\"ffn_sidecar_down_full_k_r80\",kReadyR80Rows,kReadyR80Warps,kReadyR80SharedBytes}"
    ));
    assert!(transaction.contains(
        "Ready::DirectRowsfull_k_rows=tuned_exact128?Ready::DirectRows::Rows128:Ready::DirectRows::Environment"
    ));
    assert!(ready.contains("Rows80=80"));
    assert!(ready.contains("kReadyR80Rows=80"));
    assert!(ready.contains("kReadyR80Warps=10"));
    assert!(ready.contains("__launch_bounds__(320,1)"));
    assert!(ready.contains("q8_ready_full_tile_direct_r80"));
    assert!(ready.contains("floatpartial[32]"));
    assert!(ready.contains("compute_segment<true,kReadyR80Rows,kReadyR80Warps,true>"));
    assert!(ready.contains("local_row>=active_rows"));
    assert!(ready.contains("dim3(32,kReadyR80Warps),kReadyR80SharedBytes"));
    assert!(
        ready.contains("constuint64_tgrid=uint64_t((n_out+tile_rows-1)/tile_rows)")
    );
}

#[test]
fn direct_down_schedule_metadata_has_one_authority_and_no_tuner_promotion() {
    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    let ready = compact(READY);

    assert!(ready.contains("structDirectScheduleDescriptor"));
    assert!(ready.contains("kDirectSchedules[]"));
    assert!(ready.contains("parse_direct_schedule"));
    assert!(ready.contains("direct_schedule_profile_label"));
    assert!(ready.contains("configure_q8_ready_direct_schedule"));
    assert!(transaction.contains("Ready::parse_direct_schedule(full_k_env)"));
    assert!(
        transaction.contains("Ready::configure_q8_ready_direct_schedule(full_k_rows)")
    );
    assert!(transaction.contains("Ready::direct_schedule_profile_label(full_k_rows)"));
    assert!(!transaction.contains("std::strcmp(full_k_env"));
    assert!(transaction.contains(
        "Ready::DirectRowsfull_k_rows=tuned_exact128?Ready::DirectRows::Rows128:Ready::DirectRows::Environment"
    ));
}

#[test]
fn exact128_receipted_selector_is_one_atomic_safe_off_route() {
    let defaults = compact(braced_item_after(
        NATIVE,
        "void initialize_arch_knob_defaults(int sm_version)",
    ));
    assert!(defaults.contains("g.knobs[39]=0;"));
    let selector = compact(braced_item_after(
        NATIVE,
        "static bool tuned_exact128_sm86_route(uint32_t n_tok)",
    ));
    assert!(
        selector.contains(
            "returnn_tok==128&&((g.knobs[39]==1||g.knobs[39]==2)||atomic_lab);"
        )
    );
    let token64 = compact(braced_item_after(
        NATIVE,
        "static bool tuned_exact128_token64(uint32_t n_tok)",
    ));
    assert!(token64.contains("returnn_tok==128&&g.knobs[39]==2;"));
    let fast_transaction = compact(braced_item_after(
        NATIVE,
        "static bool tuned_exact128_fast_transaction(uint32_t n_tok)",
    ));
    assert!(fast_transaction.contains(
        "returnn_tok==128&&g.sm_version==86&&(g.knobs[39]==4||g.knobs[39]==5)&&g.ffn_sidecar_model_ready;"
    ));

    let transaction = compact(braced_item_after(
        NATIVE,
        "extern \"C\" uint32_t imparo_cuda_ffn_gated_down(",
    ));
    let ple = compact(braced_item_after(
        NATIVE,
        "extern \"C\" void imparo_cuda_ple_project(",
    ));
    assert!(transaction.contains(
        "tuned_ffn_sidecar_requested(n_tok)||tuned_exact128_sm86_route(n_tok)||tuned_exact128_fast_transaction(n_tok)"
    ));
    assert!(
        transaction
            .contains("constbooltuned_exact128=tuned_exact128_sm86_route(n_tok)")
    );
    for bit in [
        "EXACT128_FFN_ADMITTED",
        "EXACT128_FFN_DOWN",
        "EXACT128_FFN_COMMITTED",
    ] {
        assert!(transaction.contains(bit), "missing FFN evidence bit {bit}");
    }
    for bit in ["EXACT128_PLE_GATE", "EXACT128_PLE_DIRECT"] {
        assert!(ple.contains(bit), "missing PLE evidence bit {bit}");
    }
    assert!(NATIVE.contains("g.tune_exact128_route_hits = 0;"));
    for counter in [
        "g.tune_exact128_attn_commits = 0;",
        "g.tune_exact128_ffn_commits = 0;",
        "g.tune_exact128_ple_commits = 0;",
        "g.tune_exact128_token64_commits = 0;",
    ] {
        assert!(NATIVE.contains(counter), "missing counter reset {counter}");
    }
    assert!(
        transaction.contains("exact128_record_commit(g.tune_exact128_ffn_commits);")
    );
    assert!(ple.contains("exact128_record_commit(g.tune_exact128_ple_commits);"));
    assert!(
        NATIVE.contains("return g.tuner_lab ? exact128_packed_route_evidence() : 0u;")
    );
}

#[test]
fn exact128_atomic_route_owns_d256_batch32_attention() {
    let attention = compact(braced_item_after(
        NATIVE,
        "extern \"C\" void imparo_cuda_attention(",
    ));
    assert!(
        attention.contains("constboolexact128_d256_batch32=head_dim==256&&n_tok==128")
    );
    assert!(attention.contains("g.sm_version==86&&tuned_exact128_sm86_route(n_tok)"));
    assert!(
        attention.contains("(tuner_knob(29)||d256_f32_fa_lab)&&!exact128_d256_batch32")
    );
    assert!(
        attention.contains(
            "exact128_d256_batch32||std::getenv(\"IMPARO_CUDA_ATTN_BATCH32\")"
        )
    );
    assert!(attention.contains("constuint32_thalf_v_accum=exact128_d256_batch32?0u"));
    assert!(attention.contains("constuint32_tllama_reduce=exact128_d256_batch32?0u"));
    assert!(
        attention.contains("exact128_d256_batch32?\"batch32-exact128\":\"batch32\"")
    );
    assert!(NATIVE.contains("template <bool ExactF32>"));
    assert!(
        NATIVE
            .contains("const bool use_half_v_accum = !ExactF32 && half_v_accum != 0;")
    );
    assert!(attention.contains("IMPARO_LAUNCH_BATCH32(true,0u,0u);"));
    assert!(attention.contains("g.tune_exact128_route_hits|=EXACT128_ATTN;"));
    assert!(
        attention.contains("exact128_record_commit(g.tune_exact128_attn_commits);")
    );
}

#[test]
fn exact128_attention_scores_use_four_private_warp_tiles() {
    let kernel = compact(braced_item_after(
        NATIVE,
        "__global__ void k_attention_batch32_f16(",
    ));
    assert!(kernel.contains("constexpruint32_tscore_warps=ExactF32?4u:1u;"));
    assert!(kernel.contains("for(uint32_tj0=0;j0<score_batch;j0+=score_warps)"));
    assert!(kernel.contains("constuint32_tj=j0+warp;"));
    assert!(kernel.contains("constexpruint32_texact_q_tile_count=ExactF32?16u:1u;"));
    assert!(kernel.contains(
        "exact_q[tile][cell]=cell<16u?__hmul(__float2half(qr[tile*16u+cell]),qk_scale_h)"
    ));
    assert!(
        kernel.contains("if(cell>=16u)mma_k[e/(16u*16u)][cell]=__float2half(0.0f);")
    );
    assert!(kernel.contains("mma_k[warp][lane]"));
    assert!(kernel.contains("__shared__volatileuint32_texact_k_rounds;"));
    assert!(kernel.contains("constuint32_tk_rounds=ExactF32?exact_k_rounds:0u;"));
    assert!(kernel.contains("for(uint32_tk_round=0;k_round<k_rounds;++k_round)"));
    assert!(kernel.contains("wmma::load_matrix_sync(a,exact_q[k0/16u],16);"));
    assert!(kernel.contains("physical_rows[threadIdx.x]="));
    assert!(kernel.matches("ps=physical_rows[j];").count() >= 2);
    assert!(kernel.contains("scores[j]=mma_c[warp][0];"));

    let attention = compact(braced_item_after(
        NATIVE,
        "extern \"C\" void imparo_cuda_attention(",
    ));
    assert!(attention.contains("EXACT_F32?128u:threads"));
}

#[test]
fn exact128_tuner_measures_model_weighted_attention_ple_ffn_transaction() {
    assert!(TUNER_KNOBS.contains("ple_tensors: Some((\"inp_gate\", \"proj\"))"));
    assert!(TUNER_MAIN.contains("ple_transaction: None"));
    assert!(TUNER_MAIN.contains("exact128_local_q: None"));
    assert!(TUNER_MAIN.contains("let ple_transaction = mbench.ple_tensors.and_then"));
    assert!(TUNER_MAIN.contains("let exact128_local_q = lookup_tensor(\"attn_q\")"));

    let transaction = compact(braced_item_after(
        TUNER_MICRO,
        "let run_exact128_transaction = |candidate_must_run: bool|",
    ));
    assert!(
        transaction.contains(
            "for(layer,&(head_dim,window))ins.attn_layers.iter().enumerate()"
        )
    );
    assert!(transaction.contains("ifhead_dim==256"));
    assert!(transaction.contains(
        "layer.checked_mul(gate_out).expect(\"per-layerembeddingoffsetfitsu32\")"
    ));
    assert!(!transaction.contains("find_map("));
    assert!(!transaction.contains("or_else("));
    let local_q = transaction
        .find("b.matmat(q_kind,q_off,q_in,q_out,BufId::Cur,BufId::X,128);")
        .expect("exact route must execute majority local-Q");
    let attention = transaction
        .find("b.attention(")
        .expect("exact route must execute D256 attention");
    let ffn = transaction
        .find("run_ffn_transaction(128,candidate_must_run);")
        .expect("exact route must execute FFN");
    let ple = transaction
        .find("b.ple_project(")
        .expect("exact route must execute PLE");
    assert!(
        local_q < attention && attention < ffn && ffn < ple,
        "tuner transaction must match workflow order"
    );
    assert!(transaction.contains("BufId::Model2"));
    assert!(transaction.contains("ple.per_layer_stride"));
    assert!(compact(TUNER_MICRO).contains(
        "letexact128_facts_complete=usize::try_from(s.n_layers).ok()==Some(s.attn_layers.len())"
    ));
    assert!(compact(TUNER_MICRO).contains("if!exact128_facts_complete"));
    assert!(compact(TUNER_MICRO).contains(
        "letexact128_base_evidence=exact128_route_evidence(exact128_d256_layers,s.attn_layers.len(),0)"
    ));
    assert!(compact(TUNER_MICRO).contains(
        "b.tuner_route_evidence(Workload::PrefillFfnExact128)==exact128_expected_evidence()"
    ));
    assert!(compact(TUNER_MICRO).contains(
        "letper_buf=reps_per_buffer(reps,one,restore_each_rep||evidence.is_some())"
    ));
    assert!(TUNER_MICRO.contains("validate_tuner_dispatch_proof"));
    assert!(TUNER_MICRO.contains("EXACT128_ATTN_COUNT_SHIFT"));
    assert!(TUNER_MICRO.contains("EXACT128_FFN_COUNT_SHIFT"));
    assert!(TUNER_MICRO.contains("EXACT128_PLE_COUNT_SHIFT"));
    assert!(TUNER_MICRO.contains("EXACT128_TOKEN64_COUNT_SHIFT"));
    assert!(NATIVE.contains("static uint32_t exact128_packed_route_evidence()"));
    assert!(NATIVE.contains("tune_exact128_token64_commits"));

    let sweep = compact(braced_item_after(TUNER_MICRO, "let sweep = |d: &KnobDecl|"));
    assert!(sweep.contains("probe_exact128_candidate()"));
    assert!(sweep.contains("exact128_expected_evidence()"));
    assert!(sweep.contains("time_us_checked("));

    let dynamic_stub = compact(braced_item_after(
        FFI,
        "pub(crate) unsafe fn imparo_cuda_exact128_route_hits_lab()",
    ));
    assert!(dynamic_stub.contains("->u32{0}"));
}

#[test]
fn rejected_reload_restores_architecture_safe_defaults_before_lookup() {
    let init = compact(braced_item_after(CUDA_BACKEND, "fn init_with_streamed("));
    assert!(init.contains("capture_cuda_safe_defaults();"));

    let apply = compact(braced_item_after(
        CUDA_BACKEND,
        "pub fn apply_host_config(model_bytes: u64)",
    ));
    let reset = apply
        .find("reset_cuda_safe_defaults();")
        .expect("config apply must restore defaults");
    let lookup = apply
        .find("selected_config_path(&fp,model_bytes)")
        .expect("config apply must perform lookup");
    assert!(reset < lookup, "every early return must occur after reset");
}

#[test]
fn attention_decode_knob_screen_uses_an_attention_decode_dispatch() {
    let screen = compact(braced_item_after(
        TUNER_MICRO,
        "let screen_probe_attn_decode = ||",
    ));
    assert!(screen.contains("attn(1,512)();"));
    assert!(screen.contains("validate_tuner_dispatch_proof()"));

    let selection = compact(braced_item_after(
        TUNER_MICRO,
        "let probe: &dyn Fn() -> f64 = match d.workload",
    ));
    assert!(selection.contains("Workload::AttentionDecode=>&screen_probe_attn_decode"));
}
