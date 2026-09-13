//! What the Metal knob registry PROMISES, checked without a GPU.
//!
//! These are declaration-level invariants, so they need no device: a knob whose declared
//! candidate the setter refuses, or whose `applies` gate is inverted, is wrong before any
//! measurement happens. The tuner would still produce numbers -- it would sweep two
//! candidates that both left the same value in place and report the winner as a tie.
#![cfg(target_os = "macos")]

use imparo_backend::{BackendKnobs as _, KnobDecl, ModelFacts, SweepKind};
use imparo_metal::MetalBackend;

/// Q8_0 = 2 on the wire, so its presence is bit 2 of the mask.
const Q8_BIT: u64 = 1 << 2;
const Q4_BIT: u64 = 1 << 1;
/// Q8_0_TM (tile-major, imparo-repack) = 3 on the wire.
const TM_BIT: u64 = 1 << 3;

/// The grid knobs are ranked on the persistent kernel itself (task #203): one decode step
/// of mega entries at a short span, cross-checked at depth. Never on `DecodeMix`, whose
/// independent matmuls reach no grid barrier and once called them INERT; and no longer
/// `External`, which a plain tune never ranks at all.
#[test]
fn persistent_grid_is_ranked_on_the_mega_decode_step() {
    for name in ["mega_tgs", "mega_nsg", "mega_tgs_large"] {
        let d = MetalBackend
            .knob_registry()
            .iter()
            .find(|d| d.name == name)
            .unwrap();
        assert_eq!(d.category, imparo_backend::KnobCategory::Benched, "{name}");
        assert_eq!(d.sweep, SweepKind::Values, "{name}");
        assert_eq!(
            d.workload,
            imparo_backend::Workload::MegaDecodeStep,
            "{name}"
        );
        assert_eq!(
            d.cross_check,
            Some(imparo_backend::Workload::MegaDecodeStepDeep),
            "{name}"
        );
        assert!(
            d.bit_affecting,
            "grid geometry can change attention reduction order"
        );
    }
}

/// The grid knobs apply where the seat GOVERNS A DISPATCH (task #203), and that is the
/// `mega_seat` fact -- what form the model's decode runs its mega entries in -- not the
/// weight kinds it happens to carry.
#[test]
fn persistent_grid_applies_where_the_seat_governs_a_dispatch() {
    use imparo_backend::MegaSeat;
    let reg = MetalBackend.knob_registry();
    let applies_of = |name: &str| {
        reg.iter()
            .find(|d| d.name == name)
            .unwrap()
            .applies
            .expect("persistent grid must declare its applicability")
    };
    let (tgs, nsg, deep) = (
        applies_of("mega_tgs"),
        applies_of("mega_nsg"),
        applies_of("mega_tgs_large"),
    );
    let with = |seat: MegaSeat| {
        let mut f = facts(Q4_BIT);
        f.mega_seat = seat;
        f
    };
    // Per-layer entries at the seat (gemma4): the grid and the width both govern.
    assert!(tgs(&with(MegaSeat::Grid)) && nsg(&with(MegaSeat::Grid)));
    // The per-token program (LFM2's default): one threadgroup per core, so only the width
    // floor applies -- the grid knob would rank identical candidates.
    assert!(!tgs(&with(MegaSeat::OnePerCore)) && nsg(&with(MegaSeat::OnePerCore)));
    // A decode that dispatches no entry (qwen35 today): neither.
    assert!(!tgs(&with(MegaSeat::None)) && !nsg(&with(MegaSeat::None)));
    // The second slot's knob needs a compiled second pipeline; a test process has none.
    assert!(!deep(&with(MegaSeat::Grid)));
    // Weight kinds no longer decide -- the seat fact does.
    for mask in [0, 1, Q8_BIT, TM_BIT, Q4_BIT | TM_BIT] {
        let mut f = facts(mask);
        f.mega_seat = MegaSeat::Grid;
        assert!(tgs(&f), "mask {mask:#x} must not gate the grid knob");
    }
}

#[test]
fn persistent_width_ladder_keeps_intermediate_legal_occupancy_points() {
    let d = MetalBackend
        .knob_registry()
        .iter()
        .find(|d| d.name == "mega_nsg")
        .unwrap();
    let candidates = d.candidates.unwrap();
    let mut device = imparo_backend::DeviceProfile {
        max_threads: 1024,
        ..Default::default()
    };
    assert_eq!(candidates(&facts(TM_BIT), &device), vec![8, 16, 24, 32]);
    device.max_threads = 512;
    assert_eq!(candidates(&facts(TM_BIT), &device), vec![8, 16]);
    device.max_threads = 256;
    assert_eq!(candidates(&facts(TM_BIT), &device), vec![8]);
}

fn facts(weight_kinds: u64) -> ModelFacts {
    // gemma4 E4B's shape, which is the model this registry was tuned against; only
    // `weight_kinds` is under test here.
    ModelFacts {
        mega_seat: imparo_backend::MegaSeat::None,
        n_embd: 2560,
        n_ff: 10240,
        n_head: 8,
        n_kv: 2,
        head_dim: 256,
        deep_head_dim: 512,
        n_experts: 0,
        n_layers: 42,
        layer_dispatches: 20,
        weight_kinds,
    }
}

fn q8_knobs() -> Vec<&'static KnobDecl> {
    MetalBackend
        .knob_registry()
        .iter()
        .filter(|d| d.name.starts_with("q8_"))
        .collect()
}

/// The whole Q8 family is gated on the model HAVING Q8_0 projections. Swept on a Q4_0
/// model, every one of them would rank candidates against a kernel the workload never
/// dispatches -- the same defect as declaring a prefill tile knob on a decode workload.
#[test]
fn the_q8_family_applies_only_to_a_model_with_q8_weights() {
    let knobs = q8_knobs();
    assert!(!knobs.is_empty(), "no q8_ knobs in the registry");
    for d in &knobs {
        let applies = d.applies.expect("a q8 knob must declare `applies`");
        if d.name.starts_with("q8_tm_") {
            // The tile-major sub-family governs kernels only a Q8_0_TM tensor dispatches:
            // it must not be swept on a plain Q8_0 model (the workload would rank the
            // row-major kernel), and must be on a converted one.
            assert!(
                applies(&facts(TM_BIT)),
                "{} should apply to a Q8_0_TM (converted) model",
                d.name
            );
            assert!(
                !applies(&facts(Q8_BIT)),
                "{} must NOT be swept on a row-major Q8_0 model",
                d.name
            );
            continue;
        }
        assert!(
            applies(&facts(Q8_BIT)),
            "{} should apply to a Q8_0 model",
            d.name
        );
        assert!(
            applies(&facts(Q8_BIT | Q4_BIT)),
            "{} should apply to a mixed model that contains Q8_0",
            d.name
        );
        assert!(
            !applies(&facts(Q4_BIT)),
            "{} must NOT be swept on a Q4_0-only model",
            d.name
        );
    }
}

#[test]
fn the_q4_family_applies_only_to_a_model_with_q4_weights() {
    // sgs / lanes / nr0 select Q4 kernels, nb8_* / rt_shape the register-tiled route's
    // (Q4_0 and every block-quant format); the Q8 route dispatches none of them, so on a
    // Q8-only model they must be skipped, not swept and recorded.
    for d in MetalBackend.knob_registry() {
        if !matches!(
            d.name,
            "sgs" | "lanes" | "nr0" | "nb8_shape" | "nb8_max" | "rt_shape"
        ) {
            continue;
        }
        let applies = d.applies.expect("a q4 knob must declare `applies`");
        assert!(
            applies(&facts(Q4_BIT)),
            "{} must apply to a Q4 model",
            d.name
        );
        assert!(
            applies(&facts(Q4_BIT | Q8_BIT)),
            "{} must apply to a mixed model",
            d.name
        );
        assert!(
            !applies(&facts(Q8_BIT)),
            "{} must not apply to a Q8-only model",
            d.name
        );
    }
}

/// Every declared candidate must survive `apply` then `current`.
///
/// The setters clamp, because the registry applies knobs before the device exists and a
/// value with no compiled pipeline must not be selectable. That makes a disagreement
/// between `values` and the clamp SILENT: the sweep measures two candidates, the setter
/// keeps the old value for one of them, and the two timings differ only by noise. Reading
/// the value back is what catches it.
#[test]
fn every_declared_candidate_round_trips_through_its_setter() {
    for d in MetalBackend.knob_registry() {
        if !matches!(d.sweep, SweepKind::Values) || d.values.is_empty() {
            continue;
        }
        let saved = (d.current)();
        for &v in d.values {
            (d.apply)(v);
            let got = (d.current)();
            assert_eq!(
                got, v,
                "{}: candidate {v} did not stick (read back {got}) -- `values` and the \
                 setter's clamp disagree",
                d.name
            );
        }
        (d.apply)(saved);
        assert_eq!(
            (d.current)(),
            saved,
            "{}: failed to restore {saved}",
            d.name
        );
    }
}

/// A `Crossing` knob drives its own ladder, so `values` must be empty -- the tuner reads
/// the ladder and would otherwise also sweep a candidate list nothing produced.
#[test]
fn a_boundary_knob_carries_no_candidate_list() {
    for d in MetalBackend.knob_registry() {
        if let SweepKind::Crossing { ladder, hi, lo } = d.sweep {
            assert!(
                d.values.is_empty(),
                "{}: a Crossing knob must not also carry `values`",
                d.name
            );
            assert!(!ladder.is_empty(), "{}: empty ladder", d.name);
            assert_ne!(
                hi, lo,
                "{}: hi and lo must select different kernels",
                d.name
            );
        }
    }
}

/// `after` names a knob that has to be settled first, and the tuner drives the registry
/// in declaration order -- so the named knob must appear EARLIER. Nothing enforced this
/// until the dependency was written down.
#[test]
fn every_after_dependency_is_declared_earlier() {
    let reg = MetalBackend.knob_registry();
    for (i, d) in reg.iter().enumerate() {
        for dep in d.after {
            let at = reg.iter().position(|o| o.name == *dep);
            let at =
                at.unwrap_or_else(|| panic!("{}: `after` names unknown {dep}", d.name));
            assert!(
                at < i,
                "{}: depends on {dep}, which is declared later ({at} >= {i})",
                d.name
            );
        }
    }
}
