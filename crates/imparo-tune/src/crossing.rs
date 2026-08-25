//! WHERE ONE KERNEL OVERTAKES ANOTHER, from what each one COSTS rather than from their
//! ratio.
//!
//! A boundary knob asks: past which span does the alternative kernel win? Two earlier
//! answers to that were wrong in instructive ways.
//!
//! FIRST -- threshold the ratio on a ladder, with five patches: start only on a real win,
//! continue through a tie, must be bracketed both sides, never the last rung, three scans
//! with a median. Each patch fixed a real failure; all five rescue one bad choice, reducing
//! a continuous measurement to a boolean per rung. One rung read 0.98, 1.00 and 1.01 across
//! runs and the answer oscillated between adjacent rungs.
//!
//! SECOND -- fit log(ratio) against log(span). Better, and still wrong: THE RATIO OF TWO
//! AFFINE FUNCTIONS IS NOT A POWER LAW. The model could not fit, and the estimator honestly
//! kept answering "interval too wide to act on" rather than inventing a crossing.
//!
//! WHAT THE MEASUREMENT SHOWS once each side is printed per position instead of as a ratio:
//!
//!     span    stream us/pos   score us/pos
//!     2048    0.3936          0.4061
//!     8192    0.2823          0.3596
//!     32768   0.2399          0.3614      <- score-tile FLAT, streaming FALLING
//!
//! Score-tile is flat because it is bandwidth-bound: it moves bytes proportional to the
//! span and nothing else. Streaming falls because it carries a FIXED cost -- a combine over
//! its slices, paid whatever the span -- which amortizes as the span grows.
//!
//! So each kernel is AFFINE in span, and that is the model:
//!
//!     t(n) = fixed + marginal * n
//!
//!     stream   336 us + 0.2296/pos    large combine, but reads shared across query heads
//!     score     98 us + 0.3584/pos    small combine, reads KV once per QUERY head
//!
//! Both halves are physical, which is what makes this more than curve fitting. `marginal`
//! is bytes-per-position over bandwidth, divided by how many query heads share a read --
//! streaming's rate being 64% of score-tile's IS its 2-way head sharing. `fixed` is the
//! combine over `slices` partials plus dispatch overhead.
//!
//! The crossing is then arithmetic, not a search:
//!
//!     n* = (fixed_hi - fixed_lo) / (marginal_lo - marginal_hi)
//!
//! Two lines also cost less to find than one ratio: three points per side determine them,
//! and every further point tightens both slopes instead of casting a vote on a threshold.

/// One measurement: what each kernel cost at this span, in microseconds.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub span: u32,
    /// The kernel the boundary switches TO above the crossing.
    pub hi_us: f64,
    /// The kernel used below it.
    pub lo_us: f64,
    /// This rung's OWN repeat scatter, as a fraction of its time. Measured from the legs
    /// that produced `hi_us`/`lo_us`, never a typed tolerance: a difference smaller than
    /// the scatter of the legs it came from is not a difference.
    pub noise: f64,
}

/// An affine cost model for one kernel: `fixed + marginal * span`.
#[derive(Clone, Copy, Debug)]
struct Line {
    fixed: f64,
    marginal: f64,
    /// Largest residual as a fraction of the fitted value -- how well affine describes it.
    worst_rel: f64,
}

/// A fitted cost line, for reporting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineOut {
    pub fixed: f64,
    pub marginal: f64,
}

/// What the samples support. Every arm but `Crossing` keeps the caller's default and says
/// why, in terms the caller can print.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// The span where the two cost lines meet.
    Crossing { span: f64, hi: LineOut, lo: LineOut },
    /// The marginal rates are too close to place a crossing: near-parallel lines put their
    /// intersection anywhere. Also the physical statement that neither kernel has a real
    /// per-position advantage -- the derived form of "no crossing on this ladder".
    NearParallel { hi_marginal: f64, lo_marginal: f64 },
    /// A kernel is not affine in span, so the model does not describe it and nothing
    /// derived from it is worth reporting. Something SWITCHES rather than scaling.
    NotAffine { which: &'static str, worst_rel: f64 },
    /// The lines meet outside the measured spans, so the answer would be extrapolation.
    OutsideRange { span: f64 },
    /// Fewer than three samples: a line through two points has no residual to judge.
    TooFew,
}

/// A kernel deviating from affine by more than this is not described by the model. Set well
/// above the few percent these benches repeat to, so it fires on structure, not scatter.
const AFFINE_LIMIT: f64 = 0.12;

/// The marginal rates must differ by at least this fraction of the larger. Below it the
/// intersection is not locatable.
const MIN_MARGINAL_GAP: f64 = 0.05;

/// Least squares of `t = fixed + marginal * span`.
fn fit(spans: &[f64], times: &[f64]) -> Line {
    let n = spans.len() as f64;
    let xbar = spans.iter().sum::<f64>() / n;
    let ybar = times.iter().sum::<f64>() / n;
    let sxx: f64 = spans.iter().map(|x| (x - xbar) * (x - xbar)).sum();
    let sxy: f64 = spans
        .iter()
        .zip(times)
        .map(|(x, y)| (x - xbar) * (y - ybar))
        .sum();
    let marginal = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let fixed = ybar - marginal * xbar;
    let worst_rel = spans
        .iter()
        .zip(times)
        .map(|(x, y)| {
            let pred = fixed + marginal * x;
            if pred.abs() > 1e-9 {
                ((y - pred) / pred).abs()
            } else {
                0.0
            }
        })
        .fold(0.0_f64, f64::max);
    Line {
        fixed,
        marginal,
        worst_rel,
    }
}

/// The threshold a `n_pos >= T` knob should hold, taken from the SIGN of each rung's
/// measured difference rather than from where two fitted lines meet.
///
/// WHY NOT THE FIT, ON THIS WORKLOAD. The rungs are timed inside a whole decode step, so a
/// common ~18 ms of weight streaming sits under both sides. The note above says common work
/// cancels in the difference. That is true of its MEAN and false of its NOISE, and the noise
/// is what decides here: at the shallow rungs the entire per-position signal is about 26 us
/// on an 18344 us measurement, 0.14%. The fitted crossing is
///
///     n* = (fixed_hi - fixed_lo) / (marginal_lo - marginal_hi)
///
/// a ratio of two differences that are each well under 1% of that baseline. Measured on one
/// machine, one afternoon, three cache types:
///
///     f16     hi = 18274 us + 0.2293/pos, lo = 18187 us + 0.2841/pos  ->  1587
///     q8_0    hi = 0 us + 0.0000/pos, lo = 0 us + 0.0000/pos          ->  degenerate
///     q4_0    the lines meet at -5388                                 ->  declined
///
/// Three answers from one physical situation, because the intercepts carry 18 ms of work
/// that has nothing to do with either kernel. The SIGN of a rung's difference survives what
/// the magnitude of an intercept cannot, and a `>= T` knob only ever asks a yes/no question
/// per span anyway.
///
/// `T` is the first rung from which `hi` wins and keeps winning. A win below a later loss is
/// scatter, not a crossing -- the knob cannot express "stream here, stop, stream again".
#[must_use]
pub fn threshold_from_signs(samples: &[Sample]) -> Option<u32> {
    if samples.len() < 2 {
        return None;
    }
    // A rung is a win only if `hi` beat `lo` by more than that rung's OWN repeat scatter.
    let won = |s: &Sample| (s.lo_us - s.hi_us) / s.lo_us.max(1e-9) > s.noise;
    // T is the first rung of the longest all-wins SUFFIX, and ties do not count as wins.
    //
    // The weaker rule -- "no rung above here lost" -- reads a run of ties as permission to
    // switch, and on the f16 ladder that opens at 128: the three shallow rungs measure
    // -0.08%, -0.23% and -0.32% against a 0.4% scatter, so nothing there is evidence either
    // way. Absence of evidence is not a reason to change kernel. Switch where the switch was
    // MEASURED to help, and everywhere above it.
    (0..samples.len())
        .find(|&i| samples[i..].iter().all(won))
        .map(|i| samples[i].span)
}

/// Estimate the span at which the two kernels cost the same.
#[must_use]
pub fn estimate(samples: &[Sample]) -> Verdict {
    if samples.len() < 3 {
        return Verdict::TooFew;
    }
    // DIRECT EVIDENCE FIRST. If one side wins at every span measured, the lines do not
    // cross inside the ladder and no fit may say otherwise. Least squares is pulled by the
    // widely-spaced deep rungs, and on f16 it produced a crossing at ~11000 from samples
    // where streaming was ahead or level at ALL SIX rungs:
    //
    //     span    4096   8192   12288  16384  24576  32768
    //     stream  19197  20393  21256  22104  23467  25683
    //     score   19401  20392  21773  22379  25212  29802
    //
    // The fitted score-tile line read 17350 us at 4096 against a measured 19401 -- a 12%
    // miss, right on AFFINE_LIMIT -- and that error is what manufactured the crossing.
    // The knob is `n_pos >= threshold`, so when hi wins throughout the honest answer is
    // the smallest span measured: below it, nothing was measured and nothing is claimed.
    let eps = 1.0 + f64::EPSILON;
    if samples.iter().all(|s| s.hi_us <= s.lo_us * eps) {
        return Verdict::Crossing {
            span: f64::from(samples[0].span),
            // No fit was used, and saying so beats printing invented coefficients.
            hi: LineOut { fixed: 0.0, marginal: 0.0 },
            lo: LineOut { fixed: 0.0, marginal: 0.0 },
        };
    }
    if samples.iter().all(|s| s.lo_us <= s.hi_us * eps) {
        return Verdict::OutsideRange {
            span: f64::from(samples[samples.len() - 1].span),
        };
    }
    let spans: Vec<f64> = samples.iter().map(|s| f64::from(s.span)).collect();
    let his: Vec<f64> = samples.iter().map(|s| s.hi_us).collect();
    let los: Vec<f64> = samples.iter().map(|s| s.lo_us).collect();
    let hi = fit(&spans, &his);
    let lo = fit(&spans, &los);

    // Does the model describe each kernel at all? Ask before deriving anything from it.
    if hi.worst_rel > AFFINE_LIMIT {
        return Verdict::NotAffine {
            which: "hi",
            worst_rel: hi.worst_rel,
        };
    }
    if lo.worst_rel > AFFINE_LIMIT {
        return Verdict::NotAffine {
            which: "lo",
            worst_rel: lo.worst_rel,
        };
    }

    let gap = lo.marginal - hi.marginal;
    let scale = lo.marginal.abs().max(hi.marginal.abs()).max(1e-12);
    if gap.abs() / scale < MIN_MARGINAL_GAP {
        return Verdict::NearParallel {
            hi_marginal: hi.marginal,
            lo_marginal: lo.marginal,
        };
    }

    let span = (hi.fixed - lo.fixed) / gap;
    let out = |l: &Line| LineOut {
        fixed: l.fixed,
        marginal: l.marginal,
    };
    let lowest = spans.iter().copied().fold(f64::MAX, f64::min);
    let highest = spans.iter().copied().fold(0.0_f64, f64::max);
    if !span.is_finite() || span < lowest || span > highest {
        return Verdict::OutsideRange { span };
    }
    Verdict::Crossing {
        span,
        hi: out(&hi),
        lo: out(&lo),
    }
}

/// The value to store, given an estimate and what is stored now.
///
/// SNAP puts the estimate on the granularity the engine actually switches at -- precision
/// past that changes nothing. HYSTERESIS then keeps the stored value unless the estimate
/// moved more than half a step, because a stable estimate is not the same as a stable
/// config: a 2% shift between runs must not rewrite one.
#[must_use]
pub fn settle(estimate: f64, current: u32, granularity: f64) -> u32 {
    let snapped = snap(estimate, granularity);
    let cur = f64::from(current);
    if cur > 0.0 && estimate > 0.0 {
        let steps = (estimate / cur).ln().abs() / granularity.ln();
        if steps < 0.5 {
            return current;
        }
    }
    snapped
}

/// Snap to the nearest rung OF THE LADDER THAT WAS MEASURED, on a log scale, with half-a-
/// rung hysteresis against the current value.
///
/// `settle` with a fixed granularity of 2 snaps to powers of two -- 4096, 8192, 16384 --
/// which is NOT the ladder. On this model the crossing lands at 11400-12800, and the
/// geometric midpoint of 8192 and 16384 is 11585, so a 6% run-to-run wobble in a stable
/// estimate flipped the answer between 8192 and 16384, a 2x swing, while skipping the
/// 12288 rung the estimate actually points at:
///
///     crossing 12773 -> 16384      crossing 11786 -> 16384      crossing 11435 -> 8192
///
/// Snapping to the measured rungs gives 12288 for all three. The knob is a `>= threshold`
/// test against spans the ladder actually visited, so a rung is the only value the
/// measurement supports anyway.
pub fn settle_to_ladder(estimate: f64, current: u32, ladder: &[u32]) -> u32 {
    if ladder.is_empty() || estimate <= 0.0 {
        return current;
    }
    let logd = |a: f64, b: f64| (a.max(1.0).ln() - b.max(1.0).ln()).abs();
    let nearest = ladder
        .iter()
        .copied()
        .min_by(|&a, &b| {
            logd(f64::from(a), estimate)
                .total_cmp(&logd(f64::from(b), estimate))
        })
        .unwrap_or(current);
    // Hysteresis: hold the current value unless the estimate is more than half a rung
    // away from it, measured in the ladder's own log spacing.
    if let Some(i) = ladder.iter().position(|&r| r == current) {
        let step = if ladder.len() > 1 {
            let a = f64::from(ladder[ladder.len() - 1]).ln();
            let b = f64::from(ladder[0]).ln();
            #[allow(clippy::cast_precision_loss)]
            {
                (a - b) / (ladder.len() - 1) as f64
            }
        } else {
            f64::MAX
        };
        if logd(f64::from(ladder[i]), estimate) < step * 0.5 {
            return current;
        }
    }
    nearest
}

/// Round to the nearest multiple of `granularity` on a LOG scale, since a span ladder is
/// geometric: halfway between 8192 and 16384 is 11585, not 12288.
fn snap(v: f64, granularity: f64) -> u32 {
    let g = granularity.max(1.000_001).ln();
    let stepped = (v.max(1.0).ln() / g).round() * g;
    let out = stepped.exp().round();
    if out < 1.0 {
        1
    } else if out > f64::from(u32::MAX) {
        u32::MAX
    } else {
        out as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(span: u32, hi_us: f64, lo_us: f64) -> Sample {
        // Noiseless by default: the fit tests are about the model, not the scatter band.
        Sample { span, hi_us, lo_us, noise: 0.0 }
    }

    fn sn(span: u32, hi_us: f64, lo_us: f64, noise: f64) -> Sample {
        Sample { span, hi_us, lo_us, noise }
    }

    /// The real f16 ladder: score-tile ahead at the shallow rungs, streaming ahead from
    /// 8192. The fit answered 1587 on these same numbers.
    #[test]
    fn signs_pick_the_rung_where_streaming_starts_winning() {
        let samples = [
            sn(128, 18344.1, 18329.6, 0.004),
            sn(512, 18406.8, 18364.9, 0.004),
            sn(2048, 18687.0, 18628.1, 0.004),
            sn(8192, 20138.9, 20489.7, 0.004),
            sn(16384, 22043.8, 22871.0, 0.004),
        ];
        assert_eq!(threshold_from_signs(&samples), Some(8192));
    }

    /// A win smaller than its own rung's scatter is not a win. Same f16 numbers, but with
    /// legs that disagreed by 2%, only the 16384 rung clears its band.
    #[test]
    fn a_win_inside_its_own_scatter_does_not_count() {
        let samples = [
            sn(128, 18344.1, 18329.6, 0.02),
            sn(512, 18406.8, 18364.9, 0.02),
            sn(2048, 18687.0, 18628.1, 0.02),
            sn(8192, 20138.9, 20489.7, 0.02),
            sn(16384, 22043.8, 22871.0, 0.02),
        ];
        assert_eq!(threshold_from_signs(&samples), Some(16384));
    }

    /// Streaming ahead at every rung: the threshold is the shallowest span measured, and
    /// nothing is claimed below it.
    #[test]
    fn winning_everywhere_opens_at_the_lowest_rung() {
        let samples = [
            sn(128, 18372.5, 18467.8, 0.002),
            sn(512, 18375.8, 18458.1, 0.002),
            sn(2048, 18514.3, 18706.2, 0.002),
        ];
        assert_eq!(threshold_from_signs(&samples), Some(128));
    }

    /// Streaming never wins: there is no threshold to report, and the caller keeps its
    /// default rather than being handed the top rung.
    #[test]
    fn no_win_anywhere_is_no_threshold() {
        let samples = [
            sn(128, 18400.0, 18300.0, 0.001),
            sn(512, 18500.0, 18400.0, 0.001),
            sn(2048, 18700.0, 18600.0, 0.001),
        ];
        assert_eq!(threshold_from_signs(&samples), None);
    }

    /// Two clean lines meet exactly where the arithmetic says.
    #[test]
    fn two_lines_meet_where_they_should() {
        // hi = 300 + 0.2n, lo = 100 + 0.3n -> cross at (300-100)/(0.3-0.2) = 2000
        let samples: Vec<Sample> = [1000u32, 2000, 4000, 8000]
            .iter()
            .map(|&n| {
                let f = f64::from(n);
                s(n, 300.0 + 0.2 * f, 100.0 + 0.3 * f)
            })
            .collect();
        match estimate(&samples) {
            Verdict::Crossing { span, hi, lo } => {
                assert!((span - 2000.0).abs() < 1.0, "span {span}");
                assert!((hi.marginal - 0.2).abs() < 1e-6);
                assert!((lo.fixed - 100.0).abs() < 1e-6);
            }
            v => panic!("expected a crossing, got {v:?}"),
        }
    }

    /// THE REAL MEASUREMENT, on which `hi` is ahead at every rung. No fit is consulted:
    /// direct evidence decides, and the answer is the shallowest span measured, because
    /// below it nothing was measured and nothing is claimed.
    ///
    /// This test used to assert the recovered coefficients (streaming carrying the larger
    /// fixed combine and the lower per-position rate). Those assertions moved to
    /// `two_lines_meet_where_they_should`, which uses a fixture that actually crosses inside
    /// its ladder -- the only case where a fit is reached at all.
    #[test]
    fn one_side_ahead_everywhere_needs_no_fit() {
        let samples = vec![
            s(2048, 806.1, 831.7),
            s(4096, 1381.7, 1621.6),
            s(8192, 2312.8, 2946.2),
            s(16384, 4221.2, 5889.9),
            s(24576, 6208.2, 8491.8),
            s(32768, 7860.1, 11841.4),
        ];
        match estimate(&samples) {
            Verdict::Crossing { span, .. } => assert!((span - 2048.0).abs() < 1.0, "span {span}"),
            v => panic!("expected a crossing at the lowest rung, got {v:?}"),
        }
    }

    /// Parallel lines never cross, and "they meet at infinity" is not an answer.
    ///
    /// The fixture must actually change sign across the ladder, or the direct-evidence guard
    /// answers first and the near-parallel check is never reached. The old fixture had `lo`
    /// ahead at every rung, which is a different statement entirely.
    #[test]
    fn near_parallel_locates_nothing() {
        let samples: Vec<Sample> = [1000u32, 2000, 4000, 8000]
            .iter()
            .map(|&n| {
                let f = f64::from(n);
                // lo ahead at 1000 (585 vs 600), hi ahead at 8000 (2700 vs 2720): the sign
                // turns over, but the rates differ by 1.6% so the meeting point is anywhere.
                s(n, 300.0 + 0.300 * f, 280.0 + 0.305 * f)
            })
            .collect();
        assert!(matches!(estimate(&samples), Verdict::NearParallel { .. }));
    }

    /// A kernel that JUMPS is not affine, and a line through it would report a confident
    /// crossing that means nothing.
    #[test]
    fn a_jump_is_not_affine() {
        let samples = vec![
            s(1000, 300.0, 400.0),
            s(2000, 500.0, 700.0),
            s(4000, 900.0, 1300.0),
            s(8000, 4000.0, 2500.0), // the hi side steps
        ];
        match estimate(&samples) {
            Verdict::NotAffine { which, .. } => assert_eq!(which, "hi"),
            v => panic!("expected NotAffine, got {v:?}"),
        }
    }

    /// A crossing outside the measured spans is extrapolation, not measurement.
    ///
    /// `lo` leads at every rung and the lines meet at 32000, past the top of the ladder.
    /// The old fixture had `hi` leading everywhere, which is the opposite case and now
    /// reports a threshold at the lowest rung instead -- see
    /// `one_side_ahead_everywhere_needs_no_fit`.
    #[test]
    fn outside_the_range_is_not_reported() {
        let samples: Vec<Sample> = [4000u32, 8000, 16000]
            .iter()
            .map(|&n| {
                let f = f64::from(n);
                s(n, 1700.0 + 0.25 * f, 100.0 + 0.30 * f)
            })
            .collect();
        assert!(matches!(estimate(&samples), Verdict::OutsideRange { .. }));
    }

    #[test]
    fn two_points_is_not_enough() {
        assert_eq!(
            estimate(&[s(4000, 1.0, 2.0), s(8000, 2.0, 3.0)]),
            Verdict::TooFew
        );
    }

    /// Hysteresis: a small move keeps the stored value, a real one takes the new estimate.
    #[test]
    fn small_moves_do_not_rewrite_the_config() {
        assert_eq!(settle(16700.0, 16384, 2.0), 16384);
        assert_eq!(settle(32768.0, 16384, 2.0), 32768);
    }

    /// Snapping is geometric, because the ladder is.
    #[test]
    fn snapping_is_geometric() {
        assert_eq!(snap(11000.0, 2.0), 8192);
        assert_eq!(snap(12000.0, 2.0), 16384);
    }
}
