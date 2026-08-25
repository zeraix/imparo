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
const Q8_BIT: u32 = 1 << 2;
const Q4_BIT: u32 = 1 << 1;

fn facts(weight_kinds: u32) -> ModelFacts {
    // gemma4 E4B's shape, which is the model this registry was tuned against; only
    // `weight_kinds` is under test here.
    ModelFacts {
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
        assert!(applies(&facts(Q8_BIT)), "{} should apply to a Q8_0 model", d.name);
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
        assert_eq!((d.current)(), saved, "{}: failed to restore {saved}", d.name);
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
            assert_ne!(hi, lo, "{}: hi and lo must select different kernels", d.name);
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
            let at = at.unwrap_or_else(|| panic!("{}: `after` names unknown {dep}", d.name));
            assert!(
                at < i,
                "{}: depends on {dep}, which is declared later ({at} >= {i})",
                d.name
            );
        }
    }
}
