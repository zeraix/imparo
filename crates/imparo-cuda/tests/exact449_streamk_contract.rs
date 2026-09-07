const NATIVE: &str = include_str!("../native/imparo_cuda.cu");
const HEADER: &str = include_str!("../native/sm86/q8_ready_batched_r2.cuh");
const EXPORTS: &str = include_str!("../native/imparo_cuda.def");

fn compact(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
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

fn stream_boundary(index: u32) -> u32 {
    const TOTAL_WORK: u64 = 80 * 320;
    const GRID: u64 = 60;
    const BLOCKS: u64 = 320;
    let mut boundary = u64::from(index) * TOTAL_WORK / GRID;
    boundary -= (boundary % BLOCKS) % 8;
    u32::try_from(boundary).unwrap()
}

#[test]
fn fixed_three_phase_plan_exactly_matches_the_admitted_grid60_boundaries() {
    for worker in 0_u32..60 {
        let group = worker / 3;
        let phase = worker % 3;
        let base_tile = 4 * group;
        let (start_tile, start_k, stop_tile, stop_k) = match phase {
            0 => (base_tile, 0, base_tile + 1, 104),
            1 => (base_tile + 1, 104, base_tile + 2, 208),
            2 => (base_tile + 2, 208, base_tile + 4, 0),
            _ => unreachable!(),
        };
        assert_eq!(stream_boundary(worker), start_tile * 320 + start_k);
        assert_eq!(stream_boundary(worker + 1), stop_tile * 320 + stop_k);
    }
    assert_eq!(stream_boundary(60), 80 * 320);

    let kernel = compact(braced_item_after(
        HEADER,
        "__global__ void q8_ready_physical_stream_exact449_grid60(",
    ));
    for expression in [
        "worker_group=worker/3",
        "phase=worker-worker_group*3",
        "first_tile_token=phase*kTokens",
        "first_segment_begin=phase*104",
        "second_tile_token=first_tile_token+kTokens",
        "phase==2?kExact449Blocks:(phase+1)*104",
        "for(uint32_tpass=0;pass<2;++pass)",
    ] {
        assert!(
            kernel.contains(expression),
            "missing periodic-plan expression {expression}"
        );
    }
    assert!(!kernel.contains("stream_boundary("));
    assert!(!kernel.contains("total_work"));
}

#[test]
fn selector_is_exact_value_shape_and_grid_gated_with_precommit_fallback() {
    let native = compact(NATIVE);
    assert!(
        native.contains(
            "std::getenv(\"IMPARO_CUDA_PREFILL_FFN_DOWN_STREAMK_AOT_449_LAB\")"
        )
    );
    assert!(native.contains("exact449_env&&std::strcmp(exact449_env,\"1\")==0"));
    for guard in [
        "n_tok==Ready::kExact449NTok",
        "n_mid==Ready::kExact449NIn",
        "n_out==Ready::kExact449NOut",
        "stream_grid==Ready::kExact449PhysicalGrid",
        "output_layout.token_tiles==Ready::kExact449TokenTiles",
    ] {
        assert!(
            native.contains(guard),
            "missing exact admission guard {guard}"
        );
    }
    assert!(native.contains("if(!down_launched&&!public_output_committed)"));
    assert!(native.contains("launch_q8_ready_physical_stream("));
    assert!(
        !EXPORTS.contains("exact449"),
        "the laboratory specialization must not expand the DLL ABI"
    );
}

#[test]
fn exact_launcher_commits_after_main_and_never_before_fixup_failure() {
    let launcher = compact(braced_item_after(
        HEADER,
        "inline bool launch_q8_ready_physical_stream_exact449_grid60(",
    ));
    let reset = launcher
        .find("*public_output_committed=false")
        .expect("missing prelaunch commit reset");
    let main = launcher
        .find("q8_ready_physical_stream_exact449_grid60<<<")
        .expect("missing exact main launch");
    let launch_check = launcher[main..]
        .find("cudaPeekAtLastError()!=cudaSuccess")
        .map(|offset| main + offset)
        .expect("missing main launch check");
    let commit = launcher
        .find("*public_output_committed=true")
        .expect("missing public-output commit point");
    let fixup = launcher
        .find("q8_ready_stream_fixup_exact449_grid60<<<")
        .expect("missing compact fixup launch");
    assert!(
        reset < main && main < launch_check && launch_check < commit && commit < fixup
    );
    assert!(launcher.contains("token_tiles!=kExact449TokenTiles"));
    assert!(launcher.contains("stream_grid!=kExact449PhysicalGrid"));
    assert!(launcher.contains("dim3(kExact449FixupSeams,4)"));
}
