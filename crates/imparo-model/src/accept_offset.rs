//! THE PER-REQUEST OFFSET ON THE ACCEPTANCE ESTIMATE (design 11.4).
//!
//! # What this corrects, and why a per-request term can
//!
//! `q` is the drafter's estimate that a node is accepted given its parent was. The confidence
//! head that produces it was trained offline and is frozen, so its LEVEL can be wrong for the
//! text being served right now. Measured on 80 held-out public prompts at a pinned 16 rows,
//! which holds the policy constant so this is the drafter alone:
//!
//! ```text
//!   across prompts   2.33 .. 8.83 accepted tokens per round      sd 1.530
//!   within a prompt  |run1 - run2| over two near-replicates      mean 0.090
//!   ratio                                                        17x
//! ```
//!
//! Acceptance is a property OF THE PROMPT and is orthogonal to context: one suite runs 8.83 at
//! 1024 and 2.69 at 16384 while another runs 2.93 at 8192 and 8.53 at 16384. That is why width
//! chosen per CONTEXT BAND is worth 0.18% against the best constant while a per-PROMPT oracle
//! is worth 1.48% -- and a term that resets per request is the one that can read it.
//!
//! # Why an offset in logit space, and not a scale on q
//!
//! ```text
//!   Correct: q' = sigmoid(logit(q) + b)   q=0 stays 0, q=1 stays 1, order preserved
//!   Wrong:   q' = clamp(k * q)            a scale pushes q past 1 and has to be clipped,
//!                                         which destroys exactly the confident nodes
//! ```
//!
//! Correcting `q` AT THE SOURCE fixes `S` at every width at once, because `S` is built from it.
//! A scale on the marginal `S(hi) - S(lo)` fixes it only between two widths -- that is a
//! different object with a different lifetime (`cost_model::MarginGain`, design 11.8).
//!
//! # The update is a likelihood step, not a heuristic
//!
//! Every verify hands over labels for free. A node is labelled exactly when its PARENT is on
//! the accepted path (design 11.3): the target judged that position in its real context. Its
//! label is whether the node itself was accepted. With `q'` as the predicted probability, the
//! gradient of the log-loss with respect to `b` is exactly `q' - y`, so:
//!
//! ```text
//!   per labelled node   g = q' - y            y = 1 accepted, 0 rejected
//!   step                b -= ALPHA * g / (BETA + sqrt(sum of g^2))
//! ```
//!
//! That is one scalar of AdaGrad, which is why there is no learning rate to guess: the rate
//! falls as the evidence accumulates, so `b` settles inside a request instead of drifting.
//!
//! THIS IS THE RANK-0 CASE OF REFITTING THE CONFIDENCE HEAD (design 11.6). The head is
//! `sigmoid(w . x + b0)` with `x` already in registers when the tree is built and `y` handed
//! over by the verify, so the same machinery widens from this one scalar to the whole vector
//! if the measurement asks. Nothing here writes into the drafter's weights.

/// The step size, MEASURED rather than assumed (`show_how_fast_the_offset_travels`). A request
/// gives roughly a dozen labelled nodes per round, so n = 120 is about ten rounds. The two
/// columns are the whole trade -- how fast a real correction arrives, against how far a
/// CALIBRATED head is walked away from zero by the same rate:
///
/// ```text
///   alpha   reaches |b| ~ 1.2 at      drift on a calibrated head: n=120 / settled
///   0.05    n = 600  (~50 rounds)     +0.054 / +0.026      too slow: a short answer never
///                                                          moves at all
///   0.10    n = 300  (~25 rounds)     +0.091 / +0.017
///   0.15    n = 120  (~10 rounds)     +0.116 / +0.009      <- shipped
///   0.25    n =  60  (~5 rounds)      +0.145 / +0.001
///   0.40    n <  60                   +0.160 / +0.012
/// ```
///
/// Drift shrinks with n at every rate -- AdaGrad's falling rate does that on its own -- so the
/// only real cost of speed is the early transient, and +0.116 in logit units moves a q of 0.70
/// to 0.72. Speed is worth more than that, because a request is often a few hundred tokens.
const ALPHA: f64 = 0.15;

/// The rate's stabiliser, so the first sample of a request cannot take a full step on its own.
const BETA: f64 = 1.0;

/// How far `b` may travel, in logit units. A bound is not optional: the labels are censored
/// (a node under a rejected parent never gets one), so a long run of rejections can push in
/// one direction without the evidence that would pull it back.
const BAND: f64 = 2.0;

/// Design 11.4's offset: one scalar, fitted online from acceptance labels, reset per request.
#[derive(Clone, Copy, Debug, Default)]
pub struct AcceptOffset {
    /// The offset itself, in logit units. 0 is "the head is right as trained".
    b: f64,
    /// AdaGrad's accumulated squared gradient.
    g2: f64,
    /// Labelled nodes folded in, for the probe line.
    seen: u64,
}

impl AcceptOffset {
    /// The corrected estimate for one node.
    #[must_use]
    pub fn apply(self, q: f64) -> f64 {
        shift(q, self.b)
    }

    /// The offset in logit units, for the probe and the tree builder.
    #[must_use]
    pub fn value(self) -> f64 {
        self.b
    }

    /// Labelled nodes folded in since the last reset.
    #[must_use]
    pub fn seen(self) -> u64 {
        self.seen
    }

    /// One labelled node: the estimate the tree was built with, and what the target did.
    ///
    /// The caller passes the RAW `q`, not the shifted one -- the prediction whose error this
    /// measures is `q'`, and recomputing it here keeps the two from drifting apart if the
    /// caller ever applies the offset at a different place in the round.
    pub fn observe(&mut self, q: f64, accepted: bool) {
        if !(0.0..=1.0).contains(&q) {
            return;
        }
        let g = shift(q, self.b) - f64::from(u8::from(accepted));
        if !g.is_finite() {
            return;
        }
        self.g2 += g * g;
        let step = ALPHA * g / (BETA + self.g2.sqrt());
        let next = self.b - step;
        if next.is_finite() {
            self.b = next.clamp(-BAND, BAND);
        }
        self.seen += 1;
    }

    /// A new request: the offset is a property of the text being served, so it does not
    /// outlive it. Design 11.2's level 1.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// `observe` at an arbitrary rate. Instrumentation only: the shipped path has ONE rate, so a
/// run that is measured and a run that is verified cannot differ by a parameter.
#[cfg(test)]
fn step_at(o: &mut AcceptOffset, q: f64, accepted: bool, alpha: f64) {
    let g = shift(q, o.b) - f64::from(u8::from(accepted));
    o.g2 += g * g;
    o.b = (o.b - alpha * g / (BETA + o.g2.sqrt())).clamp(-BAND, BAND);
    o.seen += 1;
}

/// `sigmoid(logit(q) + b)`, written so the endpoints do not go through an infinite logit:
/// `q e^b / (q e^b + 1 - q)`. At `q = 0` or `q = 1` it returns them unchanged, which is what
/// keeps an offset from inventing probability where the drafter had none.
pub(crate) fn shift(q: f64, b: f64) -> f64 {
    if b == 0.0 {
        return q;
    }
    let w = q * b.exp();
    let d = w + (1.0 - q);
    if d > 0.0 && w.is_finite() {
        (w / d).clamp(0.0, 1.0)
    } else {
        q
    }
}

#[cfg(test)]
mod tests {
    use super::{AcceptOffset, BAND, shift};

    /// The endpoints are fixed points: an offset re-ranks what the drafter proposed, it does
    /// not create a candidate out of a zero or doubt a certainty into one.
    #[test]
    fn the_shift_keeps_probabilities_and_fixes_the_endpoints() {
        for &b in &[-3.0, -0.4, 0.0, 0.4, 3.0] {
            assert!((shift(0.0, b) - 0.0).abs() < 1e-12, "q=0 moved at b={b}");
            assert!((shift(1.0, b) - 1.0).abs() < 1e-12, "q=1 moved at b={b}");
            for &q in &[0.01, 0.25, 0.5, 0.75, 0.99] {
                let s = shift(q, b);
                assert!((0.0..=1.0).contains(&s), "q={q} b={b} left [0,1]: {s}");
            }
        }
        // And it is monotone in q, so the per-node ordering the drafter gave is preserved.
        for &b in &[-2.0, 2.0] {
            let mut last = -1.0;
            for i in 0..=100 {
                let s = shift(f64::from(i) / 100.0, b);
                assert!(s >= last, "not monotone at b={b}");
                last = s;
            }
        }
    }

    /// An untrained offset is the identity. Anything else would mean turning the feature on
    /// changes the trees before it has learned a single thing.
    #[test]
    fn an_untrained_offset_is_the_identity() {
        let o = AcceptOffset::default();
        for &q in &[0.0, 0.1, 0.5, 0.9, 1.0] {
            assert!((o.apply(q) - q).abs() < 1e-12);
        }
        assert!((o.value() - 0.0).abs() < 1e-12);
    }

    /// INSTRUMENTATION, gates nothing: how far the offset has travelled after n labelled
    /// nodes, against the offset the data actually implies. A request gives roughly a dozen
    /// labels per round, so n = 120 is about ten rounds and n = 1200 about a hundred.
    #[test]
    #[ignore = "instrumentation: prints the offset's trajectory, gates nothing"]
    fn show_how_fast_the_offset_travels() {
        // A head that says 0.8 where the true rate is 0.3 implies b = logit(.3) - logit(.8).
        let want = (0.3f64 / 0.7).ln() - (0.8f64 / 0.2).ln();
        println!("target b = {want:.3} (q 0.8 -> 0.3)");
        for alpha in [0.05, 0.10, 0.15, 0.25, 0.40] {
            let mut o = AcceptOffset::default();
            let mut row = String::new();
            for i in 0..1200 {
                super::step_at(&mut o, 0.8, i % 10 < 3, alpha);
                if matches!(i + 1, 60 | 120 | 300 | 600 | 1200) {
                    row.push_str(&format!(
                        "  n={:<5}b={:+.3} q'={:.3}",
                        i + 1,
                        o.value(),
                        o.apply(0.8)
                    ));
                }
            }
            println!("alpha {alpha:.2}{row}");
        }
        // The other side of the same choice: a CALIBRATED head must not be walked away from
        // zero by the same rate. Deterministic 7-in-10 and a pseudo-random 7-in-10, because a
        // regular pattern can cancel where real labels would not.
        println!(
            "drift on a calibrated head (q 0.7, true rate 0.7), |b| after n labels:"
        );
        for alpha in [0.05, 0.10, 0.15, 0.25, 0.40] {
            let (mut reg, mut rnd) = (AcceptOffset::default(), AcceptOffset::default());
            let mut lcg: u64 = 0x2545_F491_4F6C_DD1D;
            let mut row = String::new();
            for i in 0..1200 {
                lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = ((lcg >> 33) % 1000) as f64 / 1000.0;
                super::step_at(&mut reg, 0.7, i % 10 < 7, alpha);
                super::step_at(&mut rnd, 0.7, u < 0.7, alpha);
                if matches!(i + 1, 120 | 600 | 1200) {
                    row.push_str(&format!(
                        "  n={:<5}regular {:+.3} random {:+.3}",
                        i + 1,
                        reg.value(),
                        rnd.value()
                    ));
                }
            }
            println!("alpha {alpha:.2}{row}");
        }
    }

    /// The direction: a head that is systematically OVER-confident (says 0.8, accepted 30% of
    /// the time) must be pulled DOWN, and vice versa.
    #[test]
    fn the_offset_follows_the_error_in_both_directions() {
        let mut over = AcceptOffset::default();
        for i in 0..400 {
            over.observe(0.8, i % 10 < 3);
        }
        assert!(
            over.value() < -0.5,
            "over-confident not pulled down: {}",
            over.value()
        );
        assert!(
            over.apply(0.8) < 0.6,
            "corrected q still high: {}",
            over.apply(0.8)
        );

        let mut under = AcceptOffset::default();
        for i in 0..400 {
            under.observe(0.3, i % 10 < 8);
        }
        assert!(
            under.value() > 0.5,
            "under-confident not pulled up: {}",
            under.value()
        );
        assert!(
            under.apply(0.3) > 0.4,
            "corrected q still low: {}",
            under.apply(0.3)
        );
    }

    /// A calibrated head must be left alone: feeding it labels that match its own rate should
    /// not walk the offset away from zero.
    #[test]
    fn a_calibrated_head_stays_near_zero() {
        let mut o = AcceptOffset::default();
        for i in 0..1000 {
            o.observe(0.7, i % 10 < 7);
        }
        assert!(
            o.value().abs() < 0.2,
            "calibrated head drifted to {}",
            o.value()
        );
    }

    /// The bound holds under the censoring it exists for: an unbroken run of rejections.
    #[test]
    fn a_run_of_one_label_cannot_leave_the_band() {
        let mut o = AcceptOffset::default();
        for _ in 0..100_000 {
            o.observe(0.9, false);
        }
        assert!(o.value() >= -BAND, "left the band: {}", o.value());
        let mut up = AcceptOffset::default();
        for _ in 0..100_000 {
            up.observe(0.1, true);
        }
        assert!(up.value() <= BAND, "left the band: {}", up.value());
    }

    /// Reset is what makes this level 1: the next request starts from the trained head.
    #[test]
    fn reset_returns_the_identity() {
        let mut o = AcceptOffset::default();
        for _ in 0..200 {
            o.observe(0.9, false);
        }
        assert!(o.value() < -0.1 && o.seen() == 200);
        o.reset();
        assert!((o.value() - 0.0).abs() < 1e-12 && o.seen() == 0);
        assert!((o.apply(0.42) - 0.42).abs() < 1e-12);
    }

    /// Junk in, nothing out -- a q outside [0,1] is a defect upstream, not a sample.
    #[test]
    fn a_q_that_is_not_a_probability_is_not_a_sample() {
        let mut o = AcceptOffset::default();
        o.observe(1.5, true);
        o.observe(-0.2, false);
        o.observe(f64::NAN, true);
        assert!(o.seen() == 0 && (o.value() - 0.0).abs() < 1e-12);
    }
}
