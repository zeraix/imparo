//! THE REGISTRY IS A PROTOCOL, and this is the only thing that checks it holds.
//!
//! Every tuned value reaches the engine the same way: the tuner searches knob VECTORS,
//! writes the winner to a host config, and `apply_host_config` pushes each stored key
//! back through `(decl.apply)` at startup. Each step assumes, silently, that
//!
//! ```text
//!     (decl.apply)(v)   then   (decl.current)() == v
//! ```
//!
//! When that is false the tuner measures configuration A while the engine runs B, and
//! every number it produced is attributed to the wrong state. That is not hypothetical:
//! a tuned `rt_shape` was dropped exactly this way and nothing said so.
//!
//! PRE-INIT is the moment that matters, because knobs are applied BEFORE the Metal
//! library is compiled -- function constants are baked in at library-compile time, so a
//! value that does not stick before init never reaches a kernel. Checking it after init
//! would check a different object.
//!
//! Two properties, because the registry holds two kinds of coordinate:
//!
//! ```text
//!   settable     apply(v) then current() == v          the round trip
//!   report-only  apply(v) must NOT move current()      pt_512x / pt_256x, apply is |_| {}
//! ```
//!
//! The second is the same defect from the other side: a stored config key naming a
//! derived coordinate must not corrupt it.
#![cfg(target_os = "macos")]

use imparo_backend::BackendKnobs as _;
use imparo_metal::MetalBackend;

/// One legal value per SETTABLE registry coordinate, chosen from its declared `values`
/// where it has them. The point is coverage, not the value: a coordinate absent here is
/// a coordinate nothing checks, so `probe_covers_every_registry_coordinate` fails when
/// the registry grows and this list does not.
const PROBE: &[(&str, u32)] = &[
    ("sgs", 4),
    ("lanes", 8),
    ("nb8_shape", 1),
    ("nb8_max", 16),
    ("q8_decode_sgs", 2),
    ("q8_tm_decode_sgs", 4),
    ("q8_decode_rows", 4),
    ("st_gemm_shape", 3),
    ("st_gemm_narrow_max", 16),
    ("q8_full_tiles", 1),
    ("q8_design", 0),
    ("attn_threads", 256),
    ("attn_threads_prefill", 128),
    ("rt_shape", 2),
    ("nr0", 2),
    ("attn_min_tgs", 72),
    ("attn_qcomb_mask", 1),
    ("attn_qcomb_pt", 256),
    ("qcomb_blk", 4),
    ("attn_qcomb_nsg", 8),
    ("attn_fa_nsg", 4),
    ("attn_fd_chunk", 256),
    ("attn_vec_max_keys", 1024),
    ("attn_stream_hq", 2),
    ("attn_stream_slices", 16),
    ("attn_stream_min_pos", 1024),
    ("flush_layers", 4),
    ("flush_layers_prefill", 7),
    ("mega_tgs", 1), // portable even when init discovers fewer than 32 GPU cores
    ("mega_nsg", 16),
    ("mega_tgs_large", 1), // the second pipeline slot's seat (task #203)
    // Where a co-batched step's rows leave the GEMV for the rows matmul, per format, and
    // the block GEMV's own widest. Values inside each declared `legal`: the two crossings
    // take 1..=8, the two matmul ceilings <= 24, and the block one <= 8.
    ("q4_rows_gemv_max", 4),
    ("q4_rows_mma_max", 16),
    ("q8_rows_gemv_max", 4),
    ("q8_tm_rows_mma_max", 16),
    ("blk_rows_mma_max", 8),
];

/// Coordinates whose `apply` is deliberately a no-op: their value comes from `derive`
/// and a config key naming them is a REPORT, not a request.
const REPORT_ONLY: &[&str] = &["pt_512x", "pt_256x"];

/// THE REGISTRY IS ONE PROCESS-WIDE OBJECT, and cargo runs a binary's tests on several
/// threads. Every test below that calls `apply` takes this first, so no test reads a
/// coordinate another test is in the middle of moving. Without it the vector test
/// intermittently needed a second pass -- a race in the test, not a gap in the registry.
fn knobs_serially() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn declaration(name: &str) -> &'static imparo_backend::KnobDecl {
    MetalBackend
        .knob_registry()
        .iter()
        .find(|d| d.name == name)
        .unwrap_or_else(|| panic!("missing Metal registry knob {name}"))
}

#[test]
fn probe_covers_every_registry_coordinate() {
    let mut covered: std::collections::BTreeSet<&str> =
        PROBE.iter().map(|(n, _)| *n).collect();
    covered.extend(REPORT_ONLY.iter().copied());
    let declared: std::collections::BTreeSet<&str> = MetalBackend
        .knob_registry()
        .iter()
        .map(|d| d.name)
        .collect();
    assert_eq!(covered, declared, "probe and registry disagree");
    assert_eq!(
        PROBE.len() + REPORT_ONLY.len(),
        MetalBackend.knob_registry().len(),
        "each coordinate must appear exactly once"
    );
}

/// APPLYING A WHOLE VECTOR IS A CONVERGENCE, NOT A PASS. Knobs are not independent --
/// one can refuse a value until another is set -- so this iterates until the registry
/// agrees or reports which coordinates never did. `apply_host_config` makes a SINGLE
/// pass in file order, so anything needing more than one round here is a live gap there.
#[test]
fn a_settable_vector_round_trips_before_init() {
    let _serial = knobs_serially();
    for pass in 0..PROBE.len().max(1) {
        for &(name, v) in PROBE {
            (declaration(name).apply)(v);
        }
        if PROBE
            .iter()
            .all(|&(name, v)| (declaration(name).current)() == v)
        {
            assert_eq!(
                pass,
                0,
                "the registry needed {} passes to converge",
                pass + 1
            );
            return;
        }
    }
    let bad: Vec<String> = PROBE
        .iter()
        .filter_map(|&(name, v)| {
            let got = (declaration(name).current)();
            (got != v).then(|| format!("{name} requested={v} actual={got}"))
        })
        .collect();
    panic!("registry vector did not converge: {}", bad.join(", "));
}

/// #74: a stored value the setter refuses must be REPORTED, not silently replaced. `lanes`
/// accepts 4 / 8 / 16 / 32 and ignores anything else, so applying 5 leaves the previous
/// value in place and the readback must name it; a legal value reads back clean.
#[test]
fn the_readback_names_a_knob_whose_setter_refused_the_stored_value() {
    let _serial = knobs_serially();
    let reg = MetalBackend.knob_registry();
    let d = declaration("lanes");
    let before = (d.current)();
    (d.apply)(5);
    let missed = imparo_metal::verify_applied(reg, [("lanes", 5u32)].into_iter());
    assert_eq!(missed, vec![("lanes".to_string(), 5, before)]);
    (d.apply)(before);
    let clean = imparo_metal::verify_applied(reg, [("lanes", before)].into_iter());
    assert!(
        clean.is_empty(),
        "a value that took must read back clean: {clean:?}"
    );
}

#[test]
fn a_report_only_coordinate_is_not_moved_by_applying_to_it() {
    let _serial = knobs_serially();
    for name in REPORT_ONLY {
        let d = declaration(name);
        let before = (d.current)();
        (d.apply)(before.wrapping_add(1));
        assert_eq!(
            (d.current)(),
            before,
            "{name} is derived; applying to it must not move it"
        );
    }
}

/// THE PROPERTY THE `rt_shape` DEFECT BROKE: a tuple accepted before init must survive
/// init unchanged. Everything above runs without a device and so cannot see this -- init
/// is exactly the step that was silently rewriting an accepted value.
///
/// Gated because it selects a real adapter and compiles real pipelines. It belongs in the
/// same released device gate as det_gate / kv_gates, not in `cargo test`, and it is
/// listed there rather than left present-but-never-run.
#[test]
#[ignore = "compiles real Metal pipelines; run only in the explicitly released GPU gate"]
fn an_accepted_tuple_survives_successful_init_exactly() {
    let _serial = knobs_serially();
    imparo_metal::set_q8_all(1);
    for &(name, v) in PROBE {
        (declaration(name).apply)(v);
    }
    let registry = MetalBackend.knob_registry();
    let accepted: Vec<(&str, u32)> =
        registry.iter().map(|d| (d.name, (d.current)())).collect();

    const PAGE: usize = 16 * 1024;
    let layout = std::alloc::Layout::from_size_align(PAGE, PAGE).expect("page layout");
    let weights = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!weights.is_null());
    // Metal keeps this mapping for the process lifetime: leak the one-page fixture rather
    // than invalidating the context that just initialised on top of it.
    unsafe { imparo_metal::init(weights.cast_const(), PAGE as u64) }
        .expect("the accepted registry tuple must initialise");

    for (d, (name, v)) in registry.iter().zip(accepted) {
        assert_eq!(
            (d.current)(),
            v,
            "successful init silently changed accepted knob {name}"
        );
    }
    // A stored seat above one per core is applied as written: the engine does not cap
    // the grid at the core count. The tuner measures what a pipeline admits and writes
    // a value under it (docs/megakernel-decode.md, constraint 2). The engine's only
    // bound is a pipeline's width budget; this test builds no mega pipeline, so the
    // readback is exactly the request. No experimental env pin is set by this test.
    assert_eq!(
        imparo_metal::mega_threads_limit(),
        0,
        "the test declares no head dim, so no mega pipeline and no width budget"
    );
    let two_per_core = imparo_metal::gpu_cores().max(1).saturating_mul(2);
    imparo_metal::set_mega_tgs(two_per_core);
    assert_eq!(imparo_metal::mega_tgs_current(), two_per_core);
    imparo_metal::set_mega_tgs(1);
}

/// A SWEPT KNOB MUST HAVE SOMETHING TO SWEEP.
///
/// The sweep resolves a knob's ladder as "candidates if declared, else values". Declare
/// neither and the ladder is empty: the knob is skipped, no winner is recorded, and
/// nothing anywhere says so. `attn_qcomb_mask` shipped in exactly that state -- its
/// comment described a `2^slots` derivation and the hook was never written, so `values`
/// stayed `&[]` and the knob could not be chosen.
///
/// Pre-init on purpose. The device hooks return their documented fallbacks before a GPU
/// exists (slot count 0 gives one mask; a zero tile limit gives the compiled default), so
/// this runs anywhere -- and a hook that returns nothing pre-init would be a hook that
/// hands the sweep an empty ladder on a machine whose probe has not run.
#[test]
fn every_swept_knob_resolves_to_a_non_empty_ladder() {
    let facts = imparo_backend::ModelFacts {
        mega_seat: imparo_backend::MegaSeat::None,
        n_embd: 2048,
        n_ff: 8192,
        n_head: 32,
        n_kv: 8,
        head_dim: 64,
        deep_head_dim: 64,
        n_experts: 0,
        experts_used: 0,
        windowed_layers: 0,
        n_layers: 32,
        layer_dispatches: 0,
        weight_kinds: 0b111,
    };
    let profile = imparo_backend::DeviceProfile {
        max_threads: 1024,
        threadgroup_bytes: 32768,
        ..Default::default()
    };
    let empty: Vec<&str> = MetalBackend
        .knob_registry()
        .iter()
        .filter(|d| matches!(d.sweep, imparo_backend::SweepKind::Values))
        .filter(|d| match d.candidates {
            Some(f) => f(&facts, &profile).is_empty(),
            None => d.values.is_empty(),
        })
        .map(|d| d.name)
        .collect();
    assert!(
        empty.is_empty(),
        "these knobs are swept over nothing: {empty:?}"
    );
}
