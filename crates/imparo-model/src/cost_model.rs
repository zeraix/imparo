//! ONLINE ELASTIC-NET QUANTILE REGRESSION: what a round costs, as a function of the state
//! it runs in.
//!
//! # Why a regression at all
//!
//! The verify cost is not a number, it is a FUNCTION of the context. Measured on LFM2 over
//! a 19x context range, every row class and the drafter move together and move differently:
//!
//! ```text
//!    n    a (ms)   b (us per context token)     T_n(c) = a + b*c, worst residual
//!    8    28.745      0.2657                    0.01%
//!   16    32.258      0.3916                    0.09%
//!   32    44.129      0.7808                    0.21%
//!   64    76.970      1.1442                    0.06%
//!    D     8.936      0.2287                    0.17%   (the drafter)
//! ```
//!
//! `a` is the weight pass -- that is where the GEMM tile's steps live. `b` is the KV pass per
//! token of context, and it grows 4x from 8 rows to 64. A cost table that holds ONE number per
//! class cannot express that, and the error is systematic, not noise:
//!
//! ```text
//!   Correct: T_k(c) = a_k + b_k*c         seven classes, seven DIFFERENT slopes
//!   Wrong:   T_k(c) = m_k * s             seven classes, ONE shared multiplier
//! ```
//!
//! A single multiplicative scale splits the difference and is wrong for every class at once.
//! Worse, it LOOKS like machine drift: a scalar that rises as the context grows is exactly
//! what a drift estimator reports, so the context effect gets charged to the wrong term.
//!
//! # Why this estimator and not least squares
//!
//! THE LOSS IS ONE-SIDED. A round can be slower than the machine's capability -- a browser on
//! the GPU, a cold pipeline, a thermal drop -- and never faster. Squared loss fits the MEAN of
//! a right-skewed distribution, so every burst of interference pushes the model up and it
//! never comes back. The pinball loss at a low quantile fits the LOWER ENVELOPE instead:
//!
//! ```text
//!   rho_tau(u) = u * (tau - 1[u < 0])        tau small => the model tracks the fast rounds
//! ```
//!
//! THE BASIS IS REDUNDANT ON PURPOSE. Affine in the context fits THIS model on THIS machine
//! to 0.25%; it is not a law. Another kernel regime -- a flash-decoding slice boundary, a
//! paged-KV page count, a different attention route past some key count -- can put curvature
//! in it. So the model is handed `[1, u, u^2, sqrt(u), ln(1+u)]` and the regulariser decides
//! which terms earn their place, rather than the author asserting the shape.
//!
//! WHICH IS WHY THE PENALTY IS ELASTIC NET AND NOT LASSO. Over any short run of rounds the
//! context barely moves -- a 256-token answer spans 3% of an 8444-token context -- and those
//! basis terms are then almost perfectly collinear. Pure L1 picks one of a collinear group
//! arbitrarily and flips between them as samples arrive, which makes the price jump for no
//! physical reason. The L2 term shares the weight across the group and holds it still. L1
//! still does its job: a class that has only ever run at one context gets ZERO slope, so it
//! prices as a constant, which is the honest answer when the data cannot see a slope.
//!
//! # The solver
//!
//! FTRL-Proximal (McMahan et al.): the standard online learner for an L1 + L2 objective.
//! Per round it is O(d) flops and O(d) state -- no matrix, no inversion, nothing that grows
//! with the number of rounds. The per-coordinate adaptive rate is what lets an intercept of
//! ~30 and a feature of ~1 be trained by the same update.

/// Terms in the basis: `[1, u, u^2, sqrt(u), ln(1+u)]`.
pub const DIM: usize = 5;

/// The context in units of 8192 tokens, so every basis term is O(1) at the depths served and
/// the L1 threshold means the same thing in each coordinate.
const CTX_UNIT: f64 = 8192.0;

/// The quantile the model tracks. Low, because the noise is one-sided -- but not zero: a hard
/// minimum cannot rise when the machine really does get slower, and one freakishly fast round
/// would pin it forever.
const TAU: f64 = 0.15;

/// Samples held back before the law starts, and whose MINIMUM sets its scale.
///
/// THE FIRST ROUND OF ANYTHING IS USUALLY THE COLD ONE -- a pipeline not yet built, a weight
/// not yet resident. Scaling the whole law by it put the estimate 3x high and left the
/// learner crawling back down at its step size. The minimum of a few discards the cold one
/// without being told which it was.
const WARM: usize = 3;

/// THE FORGETTING FLOOR. FTRL's per-coordinate rate decays as `1/sqrt(n)`, which is right
/// for a stationary law and wrong for a process that runs for days: after enough rounds the
/// learner freezes and can no longer follow a real change. Capping the accumulator floors
/// the rate at `ALPHA / (BETA + sqrt(N_CAP))`, so the law stays permanently able to move.
/// This is the online equivalent of a recursive least-squares forgetting factor.
const N_CAP: f64 = 400.0;

/// FTRL's learning rate and its stabiliser.
///
/// THE TARGET IS NORMALISED BY THE COST'S OWN SCALE, so these are dimensionless and mean the
/// same thing for every class and every model. In absolute milliseconds they would not: the
/// classes measured here span 29 to 87 ms, so one step size is 3x coarser at one end than the
/// other -- and it showed, as the widest class wandering to -23% while the narrow ones fitted
/// to 1%. A model whose verify costs 500 ms would have been hopeless.
const ALPHA: f64 = 0.05;
const BETA: f64 = 1.0;

/// L1: a coefficient worth less than this FRACTION of the cost's own scale is not worth
/// carrying, and is set to exactly zero. L2: shares weight across collinear basis terms
/// instead of letting L1 pick one of them arbitrarily and flip between them.
const L1: f64 = 0.002;
const L2: f64 = 0.01;

/// Samples the bank keeps per CONTEXT BAND (a context's bit length), newest first out.
const BANK: usize = 16;

/// Passes the refit makes over the retained window. THE REFIT IS THE SAME LEARNER RUN TO
/// CONVERGENCE: one pass under-shoots a coefficient that many passes over the same samples
/// recover, and reusing the update means no second solver and no second failure mode.
const EPOCHS: usize = 24;

/// Two contexts within this factor of each other are one span: a slope read between them is
/// noise (`has_span`), and a refit window reaching within it of a law's trained span covers it.
const SPAN: f64 = 1.5;

/// How far from the class's own measured floor a prediction may stray. THE MODEL IS A GUESS
/// ABOUT A CONTEXT IT MAY NEVER HAVE SEEN; the budget must never be handed a negative time or
/// an absurd one because the basis extrapolated badly.
const BAND: (f64, f64) = (0.5, 3.0);

/// How far a marginal gain estimate may be trusted. S is a sum of the drafter's own
/// probabilities and its SCALE is wrong by a factor that changes with context -- measured
/// 0.59x at ctx 1596 and 1.13x at 8444, opposite directions. Outside this band the fit is
/// reading noise, not a scale.
const GAIN_BAND: (f64, f64) = (0.25, 4.0);

/// Where the gain's context term is centred, in `CTX_UNIT`s. Centring makes the fit's two
/// columns nearly orthogonal over the contexts a session actually visits; uncentred, `p` and
/// `p*u` point almost the same way and the slope reads noise instead of context.
const GAIN_PIVOT: f64 = 0.5;

/// How hard the gain is shrunk toward "no correction", as a FRACTION of each column's own
/// evidence. It has to be a fraction: the slope's column is `p*v` and carries several times
/// less squared mass than the level's `p`, so ONE absolute weight shrinks the slope hard
/// while barely touching the level -- measured, a fit whose data said 1.69 held 1.56.
const GAIN_SHRINK: f64 = 0.02;

/// The prior's weight when there is no evidence at all. It is what makes an empty fit solve
/// to exactly `(1, 0)`, damps a single wild round, and keeps a stream that has only ever
/// seen ONE context solvable instead of singular.
const GAIN_FLOOR: f64 = 0.05;

/// THE MARGINAL'S SCALE, learned from what the round already reveals -- as a LINE IN CONTEXT.
///
/// The budget compares widths by `1 + S(n)`, and S is calibrated well enough as a per-node
/// probability -- but the quantity the choice actually turns on is the MARGINAL, `S(hi) -
/// S(lo)`, and that is mis-scaled. It correlates with the realised marginal at r ~ 0.35, so
/// the signal is there; only the factor is wrong.
///
/// ONE NUMBER IS NOT ENOUGH, and that was measured: the factor is 1.69 at a 1596-token
/// context and 0.885 at 8444 -- opposite sides of 1. A pooled scalar averages the two to
/// ~1.0, which IS the identity, so the correction does nothing at either end. A trained
/// session read exactly that: g = 1.032 over 888 wide rounds. So the gain gets the same
/// treatment the prices get -- a function of context, fitted online:
///
/// ```text
///   actual marginal  ~  predicted marginal * g(ctx)
///   g(u) = a + b*(u - GAIN_PIVOT),   u = ctx / CTX_UNIT
/// ```
///
/// `a` and `b` are weighted least squares on the two columns `[p, p*(u - GAIN_PIVOT)]`: a
/// 2x2 normal system accumulated in five decayed scalars and solved where it is read. The
/// system is ridge-pulled toward `(a, b) = (1, 0)`, so an unfitted gain IS the identity and
/// there is no warm-up count to tune -- the prior simply stops mattering as evidence arrives.
///
/// The factor is observable at no cost. A round that ran at `hi` rows reports the accepted
/// walk, and because every prefix of the best-first order is itself a tree, that one walk
/// gives `accept(lo)` and `accept(hi)` together.
///
/// IT IS CENSORED. Only a round that RAN at `hi` reveals the numerator, so a policy that
/// never runs wide never learns, and a policy that never learns never runs wide. The budget
/// breaks that by starting on the WIDEST width it is offered; this type records what arrives.
#[derive(Clone, Copy, Debug, Default)]
pub struct MarginGain {
    /// The decayed `X'X`, lower triangle: `[p*p, p*p*v, p*p*v*v]`.
    xx: [f64; 3],
    /// The decayed `X'y`: `[p*y, p*v*y]`.
    xy: [f64; 2],
    seen: u64,
}

impl MarginGain {
    /// `(a, b)` from the ridge-pulled normal equations, or the identity when they are
    /// singular. The prior sits on the diagonal AND on the right-hand side, which is what
    /// makes an empty fit solve to exactly `(1, 0)` instead of to zero.
    fn fit(&self) -> (f64, f64) {
        let [m00, m01, m11] = self.xx;
        let (p0, p1) = (
            m00.max(0.0) * GAIN_SHRINK + GAIN_FLOOR,
            m11.max(0.0) * GAIN_SHRINK + GAIN_FLOOR,
        );
        let (a00, a01, a11) = (m00 + p0, m01, m11 + p1);
        // The prior's MEAN is (1, 0), so only the level's row takes a right-hand term.
        let (b0, b1) = (self.xy[0] + p0, self.xy[1]);
        let det = a00 * a11 - a01 * a01;
        if !det.is_finite() || det.abs() < 1e-12 {
            return (1.0, 0.0);
        }
        let a = (b0 * a11 - b1 * a01) / det;
        let b = (a00 * b1 - a01 * b0) / det;
        if a.is_finite() && b.is_finite() {
            (a, b)
        } else {
            (1.0, 0.0)
        }
    }

    /// The factor to apply to a predicted marginal AT THIS CONTEXT.
    #[must_use]
    pub fn at(&self, ctx: usize) -> f64 {
        let (a, b) = self.fit();
        let g = a + b * (ctx_u(ctx) - GAIN_PIVOT);
        if g.is_finite() {
            g.clamp(GAIN_BAND.0, GAIN_BAND.1)
        } else {
            1.0
        }
    }

    /// One observation: a round ran wide at `ctx`, and these are the two marginals it showed.
    ///
    /// A round whose predicted marginal is ~0 says nothing about a SCALE and is dropped -- it
    /// would weight the fit with a row that carries no information about the slope either.
    pub fn observe(&mut self, ctx: usize, predicted: f64, actual: f64) {
        if !predicted.is_finite() || !actual.is_finite() || predicted <= 1e-3 {
            return;
        }
        let v = ctx_u(ctx) - GAIN_PIVOT;
        let (x0, x1) = (predicted, predicted * v);
        // A gentle decay keeps the fit following a change instead of averaging over a whole
        // session; 1/256 is ~a request's worth of rounds.
        const KEEP: f64 = 1.0 - 1.0 / 256.0;
        for s in &mut self.xx {
            *s *= KEEP;
        }
        for s in &mut self.xy {
            *s *= KEEP;
        }
        self.xx[0] += x0 * x0;
        self.xx[1] += x0 * x1;
        self.xx[2] += x1 * x1;
        self.xy[0] += x0 * actual;
        self.xy[1] += x1 * actual;
        self.seen += 1;
    }

    #[must_use]
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// For the store.
    #[must_use]
    pub fn state(&self) -> ([f64; 3], [f64; 2], u64) {
        (self.xx, self.xy, self.seen)
    }

    /// From the store. This is what makes the gain arrive TRAINED: the SLOPE needs rounds at
    /// more than one context, which one request rarely spans.
    pub fn seed(&mut self, xx: [f64; 3], xy: [f64; 2], seen: u64) {
        if xx.iter().chain(xy.iter()).all(|v| v.is_finite()) {
            self.xx = xx;
            self.xy = xy;
            self.seen = seen;
        }
    }
}

/// The context in the basis's own unit.
fn ctx_u(ctx: usize) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let u = ctx as f64 / CTX_UNIT;
    u
}

/// Rounds a context band must hold before its rate prices a round: until then the budget keeps
/// the per-round ratio (design 6.6, theory item 1, the cold start).
const RATE_WARM: u64 = 8;

/// How fast the rate forgets, in rounds of its own band: a round this many rounds old weighs
/// half of the newest.
const RATE_HALF_LIFE: f64 = 64.0;

/// Context bands: a context's bit length, so 128..255 is one band and 16384..32767 another.
const RATE_BANDS: usize = usize::BITS as usize + 1;

/// RHO, THE PROCESS'S LONG-RUN RATE: the tokens per microsecond it actually delivers, per
/// context band (design 6.6, theory item 1).
///
/// The width that maximises tokens per unit time ACROSS rounds is the one that maximises
/// `(1 + S(n)) - rho (D + T(n))` in every round, where rho is that long-run rate (Dinkelbach
/// 1967). The per-round ratio `(1 + S) / (D + T)` is the same rule with rho replaced by the
/// round's own rate, which is wrong exactly when rounds are uneven: a good round asks too much
/// of a wider tree and a poor one too little.
///
/// ```text
///   evidence   a round's tokens (1 + accepted) and its microseconds (the drafter, the verify
///              and the commit), in the band of the context it ran at
///   rho        decayed tokens / decayed microseconds of that band
/// ```
///
/// BY BAND because throughput falls with context and a request can start at any length: a rate
/// measured at 128 tokens of context would make every round at 16k look poor. KEPT FOR THE
/// PROCESS, not the request, because it prices time, and a microsecond costs the same whichever
/// request spends it.
#[derive(Clone, Debug)]
pub struct LongRunRate {
    /// Per band: decayed tokens, decayed microseconds, rounds seen.
    bands: [(f64, f64, u64); RATE_BANDS],
}

impl Default for LongRunRate {
    fn default() -> Self {
        Self {
            bands: [(0.0, 0.0, 0); RATE_BANDS],
        }
    }
}

impl LongRunRate {
    fn band(ctx: usize) -> usize {
        (usize::BITS - ctx.leading_zeros()) as usize
    }

    /// One round at `ctx`: the `tokens` it delivered in `us` microseconds.
    pub fn observe(&mut self, ctx: usize, tokens: usize, us: f64) {
        if tokens == 0 || !us.is_finite() || us <= 0.0 {
            return;
        }
        let keep = 0.5_f64.powf(1.0 / RATE_HALF_LIFE);
        let (t, s, n) = &mut self.bands[Self::band(ctx)];
        let tokens = tokens as f64;
        *t = *t * keep + tokens;
        *s = *s * keep + us;
        *n = n.saturating_add(1);
    }

    /// Tokens per microsecond at `ctx`, or `None` until its band holds `RATE_WARM` rounds.
    #[must_use]
    pub fn at(&self, ctx: usize) -> Option<f64> {
        let (t, s, n) = self.bands[Self::band(ctx)];
        (n >= RATE_WARM && s > 0.0).then(|| t / s)
    }

    /// Rounds folded into the band `ctx` falls in, for the probe line.
    #[must_use]
    pub fn seen(&self, ctx: usize) -> u64 {
        self.bands[Self::band(ctx)].2
    }

    /// The bands that hold rounds, `(band, tokens, microseconds, rounds)`, for the store.
    #[must_use]
    pub fn rows(&self) -> Vec<(usize, f64, f64, u64)> {
        self.bands
            .iter()
            .enumerate()
            .filter(|(_, b)| b.2 > 0)
            .map(|(i, &(t, s, n))| (i, t, s, n))
            .collect()
    }

    /// From the store. A band past the table or a row that cannot be a rate is skipped.
    pub fn seed(&mut self, rows: &[(usize, f64, f64, u64)]) {
        for &(band, t, s, n) in rows {
            if band < RATE_BANDS
                && t.is_finite()
                && s.is_finite()
                && t >= 0.0
                && s > 0.0
            {
                self.bands[band] = (t, s, n);
            }
        }
    }
}

/// A context's band: its bit length, so 256..511 is one band and 8192..16383 another.
fn band_of(ctx: usize) -> usize {
    (usize::BITS - ctx.leading_zeros()) as usize
}

/// The basis at a context.
#[must_use]
pub fn features(ctx: usize) -> [f64; DIM] {
    let u = ctx_u(ctx);
    [1.0, u, u * u, u.sqrt(), u.ln_1p()]
}

/// One cost's law: milliseconds as a function of the context it ran at.
#[derive(Clone, Debug)]
pub struct CostLaw {
    /// FTRL's accumulated per-coordinate gradient, offset for the proximal step.
    z: [f64; DIM],
    /// FTRL's accumulated squared gradient -- the per-coordinate adaptive rate.
    n: [f64; DIM],
    /// Rounds folded in.
    seen: u64,
    /// The smallest and largest context this law has ever been trained at. A slope fitted
    /// across a span the data never covered is fitted to noise, so `spread` gates it.
    lo_ctx: f64,
    hi_ctx: f64,
    /// The cost's own scale, from the minimum of the first `WARM` observations: the learner
    /// works on `ms / scale`, so its rates and penalties are dimensionless.
    scale: f64,
    /// The held-back samples, until there are `WARM` of them. None are discarded -- once the
    /// scale is known they are all folded in.
    warm: Vec<(usize, f64)>,
    /// THE SAMPLE BANK: the newest `BANK` observations of every context band, in arrival
    /// order. The hot path only pushes here; the refit reads it, and the store keeps it, so the
    /// slow fit sees every band the law has ever been measured in -- not just the last request's.
    ///
    /// ```text
    ///   Correct: bank of 443 / 1596 / 8444 samples -> interleaved refit -> 25.7 / 26.7 / 35.1 ms
    ///            (measured low quantiles 25.2 / 26.7 / 35.3; Qwen3-4B, 8 rows)
    ///   Wrong:   the online pass alone, contexts arriving one request at a time
    ///            -> 28.9 / 29.8 / 35.0: every sample moves the intercept toward the current
    ///            block, and only the contrasts between blocks teach the slope
    /// ```
    bank: Vec<(u32, f32)>,
    /// The smallest NORMALISED value ever observed, which anchors the band.
    floor: f64,
}

impl Default for CostLaw {
    fn default() -> Self {
        Self::new()
    }
}

impl CostLaw {
    #[must_use]
    pub fn new() -> Self {
        Self {
            z: [0.0; DIM],
            n: [0.0; DIM],
            seen: 0,
            lo_ctx: f64::INFINITY,
            hi_ctx: 0.0,
            scale: 1.0,
            warm: Vec::new(),
            bank: Vec::new(),
            floor: f64::INFINITY,
        }
    }

    /// FTRL's weights, derived from `z` and `n` rather than stored. The L1 threshold is applied
    /// here, and a coordinate reads exactly zero only while `|z| <= L1`. That lasts for a
    /// rarely visited coordinate, NOT for a context term: it is updated every round, so its `|z|`
    /// grows like the gradient's scale times sqrt(rounds) and passes L1 on the first fold.
    ///
    /// AND THE CONTEXT TERMS CANNOT BE NEGATIVE -- which is what actually holds an unsupported
    /// slope at zero, on its negative side. More context is more KV to read; no kernel here gets
    /// cheaper as the cache grows, so a negative slope is not a cheap regime, it is noise. Measured: a class that ran its three probe rounds inside a
    /// TEN-TOKEN span fitted coefficients like `-0.0051`, which is a slope read off nothing.
    /// Every basis term after the intercept is non-decreasing in the context, so clamping
    /// them at zero makes the law non-decreasing -- a projection onto the feasible set, which
    /// is what projected gradient descent does and which `z` recovers from as evidence
    /// arrives.
    #[must_use]
    pub fn weights(&self) -> [f64; DIM] {
        let mut w = [0.0; DIM];
        for (i, ((out, &z), &n)) in w.iter_mut().zip(&self.z).zip(&self.n).enumerate() {
            if z.abs() <= L1 {
                continue; // L1 kills it outright -- not shrunk, zero
            }
            let v = -(z - z.signum() * L1) / ((BETA + n.sqrt()) / ALPHA + L2);
            *out = if i == 0 { v } else { v.max(0.0) };
        }
        w
    }

    /// The raw model output in milliseconds, before the band.
    fn raw(&self, ctx: usize) -> f64 {
        let (w, x) = (self.weights(), features(ctx));
        w.iter().zip(&x).map(|(a, b)| a * b).sum()
    }

    /// THE PRICE at a context, in milliseconds, or `None` until the law has been trained.
    ///
    /// The band is a guard, not a model: a basis evaluated far outside the contexts it was
    /// trained on can return anything, and the budget must never divide by it.
    #[must_use]
    pub fn ms(&self, ctx: usize) -> Option<f64> {
        if self.seen == 0 || !self.floor.is_finite() {
            return None;
        }
        let y = self
            .raw(ctx)
            .clamp(self.floor * BAND.0, self.floor * BAND.1);
        Some(y * self.scale)
    }

    /// Whether this law has seen contexts far enough apart for a SLOPE to mean anything.
    /// Under it the context terms are noise: the clamp in `weights` zeroes their negative
    /// side, L1 does not zero them. Nothing gates the fit on this; only tests read it.
    #[must_use]
    pub fn has_span(&self) -> bool {
        self.hi_ctx > self.lo_ctx * SPAN && self.seen >= 8
    }

    #[must_use]
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Folds in one round: this cost took `ms` milliseconds at context `ctx`.
    ///
    /// The subgradient of the pinball loss with respect to the prediction is
    /// `1[t < y_hat] - tau`, so a round FASTER than the model pulls it down hard (weight
    /// `1 - tau`) and a slower one pushes up gently (weight `tau`). That asymmetry is the
    /// whole point: interference only ever adds.
    pub fn observe(&mut self, ctx: usize, ms: f64) {
        if !ms.is_finite() || ms <= 0.0 {
            return;
        }
        // THE WARM-UP. The first `WARM` samples are held back, their minimum becomes the
        // scale, and then every one of them is folded in. Warm-starting from a single
        // sample is what put a cold first round into the scale.
        if self.seen == 0 && self.warm.len() < WARM {
            self.warm.push((ctx, ms));
            if self.warm.len() < WARM {
                return;
            }
            self.scale = self
                .warm
                .iter()
                .map(|&(_, v)| v)
                .fold(f64::INFINITY, f64::min);
            self.seed_intercept(1.0);
            let held = std::mem::take(&mut self.warm);
            for (c, v) in held {
                self.fold(c, v);
            }
            return;
        }
        self.fold(ctx, ms);
    }

    /// One trained step, once the scale is known.
    fn fold(&mut self, ctx: usize, ms: f64) {
        let band = band_of(ctx);
        if self.bank.iter().filter(|&&(c, _)| band_of(c as usize) == band).count() >= BANK {
            if let Some(i) = self.bank.iter().position(|&(c, _)| band_of(c as usize) == band) {
                self.bank.remove(i);
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        self.bank.push((ctx as u32, ms as f32));
        #[allow(clippy::cast_precision_loss)]
        let c = ctx as f64;
        let y = ms / self.scale;
        let x = features(ctx);
        // The pinball subgradient with respect to the PREDICTION is `1[t < y_hat] - tau`,
        // and with respect to the weights it is that times the feature. A round faster than
        // the model pulls it down with weight `1 - tau`; a slower one pushes up with `tau`.
        let pull = if y < self.raw(ctx) { 1.0 - TAU } else { -TAU };
        let w = self.weights();
        for (((z, n), &xi), &wi) in self.z.iter_mut().zip(&mut self.n).zip(&x).zip(&w) {
            let g = pull * xi;
            let grown = (*n + g * g).min(N_CAP);
            *z += g - (grown.sqrt() - n.sqrt()) / ALPHA * wi;
            *n = grown;
        }
        self.seen += 1;
        self.lo_ctx = self.lo_ctx.min(c);
        self.hi_ctx = self.hi_ctx.max(c);
        self.floor = self.floor.min(y);
    }

    /// THE BATCH REFIT, for a caller that can spend milliseconds off the hot path.
    ///
    /// A candidate law is trained from scratch over the retained window, for `EPOCHS` passes
    /// instead of one, and both laws are scored on samples the candidate never saw. The
    /// candidate is adopted only if it wins. Returns whether it did.
    ///
    /// WHETHER IT HELPS IS NOT ESTABLISHED. A joint fit over a window is the better
    /// estimator in principle, and the held-out gate means adopting one can never make the
    /// table worse -- but on clean data with enough samples the online pass already lands
    /// within 4% of the truth and this is no closer. It is here because the machinery is
    /// sound and cheap, not because a measurement asked for it.
    pub fn refit(&mut self) -> bool {
        let (train, held) = self.split_held();
        if train.len() < WARM + WARM || held.is_empty() {
            return false;
        }
        // THE BANK MUST COVER THE CONTEXTS THE LAW WAS TRAINED ON. The held-out test scores
        // both laws only at the bank's contexts, so a bank from one band -- a law restored from
        // a store written before the bank existed -- would replace a law fitted over many with
        // one that is flat everywhere else.
        //
        //   Correct: bank spans the law's contexts -> the held-out test can judge the law
        //   Wrong:   samples at 8460..8584 only -> a flat fit wins there -> the 8-row law trained
        //            over 443..8584 prices 8 rows at 35.1 ms at a 1596-token context, where they
        //            take 26.9 (Qwen3-4B, learned widths: the chooser then held 7 rows)
        let lo = self.bank.iter().map(|&(c, _)| f64::from(c)).fold(f64::INFINITY, f64::min);
        let hi = self.bank.iter().map(|&(c, _)| f64::from(c)).fold(0.0, f64::max);
        if lo > self.lo_ctx * SPAN || hi * SPAN < self.hi_ctx {
            return false;
        }
        let mut cand = Self::new();
        for _ in 0..EPOCHS {
            for &(c, v) in &train {
                cand.observe(c as usize, f64::from(v));
            }
        }
        if cand.score(&held) < self.score(&held) {
            // The candidate's own counts describe the replay, not the evidence: the law keeps
            // what it had seen, the span it was trained over and its bank.
            cand.bank = std::mem::take(&mut self.bank);
            cand.seen = self.seen;
            cand.lo_ctx = self.lo_ctx;
            cand.hi_ctx = self.hi_ctx;
            *self = cand;
            return true;
        }
        false
    }

    /// The bank split for a refit: the newest quarter of every band with four samples or more
    /// held out, and the rest interleaved across bands, so no band arrives as a block.
    fn split_held(&self) -> (Vec<(u32, f32)>, Vec<(u32, f32)>) {
        let mut bands: Vec<(usize, Vec<(u32, f32)>)> = Vec::new();
        for &x in &self.bank {
            let b = band_of(x.0 as usize);
            match bands.iter_mut().find(|(k, _)| *k == b) {
                Some((_, v)) => v.push(x),
                None => bands.push((b, vec![x])),
            }
        }
        let mut held = Vec::new();
        for (_, v) in &mut bands {
            let k = v.len() / 4;
            held.extend(v.drain(v.len() - k..));
        }
        let mut train = Vec::new();
        let longest = bands.iter().map(|(_, v)| v.len()).max().unwrap_or(0);
        for i in 0..longest {
            for (_, v) in &bands {
                if let Some(&x) = v.get(i) {
                    train.push(x);
                }
            }
        }
        (train, held)
    }

    /// The bank, for the store.
    #[must_use]
    pub fn bank(&self) -> &[(u32, f32)] {
        &self.bank
    }

    /// Restores a stored bank after `seed`. A sample that cannot be a time is skipped, and a
    /// band past its share keeps its newest.
    pub fn seed_bank(&mut self, samples: &[(u32, f32)]) {
        self.bank.clear();
        for &(c, v) in samples {
            if c > 0 && v.is_finite() && v > 0.0 {
                let band = band_of(c as usize);
                if self.bank.iter().filter(|&&(x, _)| band_of(x as usize) == band).count() >= BANK {
                    if let Some(i) = self.bank.iter().position(|&(x, _)| band_of(x as usize) == band) {
                        self.bank.remove(i);
                    }
                }
                self.bank.push((c, v));
            }
        }
    }

    /// Mean pinball loss on held-out samples, in the law's own normalised units so the
    /// comparison means the same thing whatever the cost's scale.
    fn score(&self, at: &[(u32, f32)]) -> f64 {
        if at.is_empty() {
            return f64::INFINITY;
        }
        let mut sum = 0.0;
        for &(c, v) in at {
            let Some(p) = self.ms(c as usize) else {
                return f64::INFINITY;
            };
            let u = (f64::from(v) - p) / self.scale;
            sum += u * if u >= 0.0 { TAU } else { TAU - 1.0 };
        }
        #[allow(clippy::cast_precision_loss)]
        {
            sum / at.len() as f64
        }
    }

    /// Puts the intercept at `y` and every other coordinate at zero, by inverting FTRL's
    /// weight rule at `n = 0`. The first round already knows the answer to within noise;
    /// learning ~1.0 from zero would waste tens of rounds at this step size.
    fn seed_intercept(&mut self, y: f64) {
        let k = BETA / ALPHA + L2;
        self.z = [0.0; DIM];
        self.z[0] = -y * k - L1;
    }

    /// Restores a law stored between runs: the weights, the accumulated squared gradients,
    /// what it has seen, and the floor.
    ///
    /// THE LEARNING RATE IS PART OF THE LAW. FTRL's per-coordinate rate falls as `n` grows, so a
    /// law trained on hundreds of rounds moves little on one more. Restored with `n` at zero it
    /// learns at the full first-round rate again, and every request restores it:
    ///
    /// ```text
    ///   Correct: restore w and n -> one cold first round moves the 8-row law 0.5%
    ///   Wrong:   restore w only  -> the same round moves it 3%, two warm rounds 7% back
    ///            (LFM2.5-2.6B at 8,444 tokens: 8 rows dropped out of the offer, and the chooser
    ///            held 6 rows for 25 rounds)
    /// ```
    ///
    /// `z` is recovered from `w` and `n` by inverting `weights`; a store written before `n` was
    /// kept passes zeros, which is the rate of a law that has seen nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn seed(
        &mut self,
        w: &[f64; DIM],
        n: &[f64; DIM],
        seen: u64,
        lo_ctx: f64,
        hi_ctx: f64,
        scale: f64,
        floor: f64,
    ) {
        for ((z, &wi), (slot, &ni)) in self.z.iter_mut().zip(w).zip(self.n.iter_mut().zip(n)) {
            let ni = ni.clamp(0.0, N_CAP);
            *slot = ni;
            let k = (BETA + ni.sqrt()) / ALPHA + L2;
            *z = if wi == 0.0 {
                0.0
            } else {
                -wi * k - wi.signum() * L1
            };
        }
        self.seen = seen;
        self.lo_ctx = lo_ctx;
        self.hi_ctx = hi_ctx;
        self.scale = scale;
        self.warm.clear();
        self.bank.clear();
        self.floor = floor;
    }

    /// What a store needs: the weights, the accumulated squared gradients, the sample count,
    /// the trained span, the scale, the floor.
    #[must_use]
    pub fn state(&self) -> ([f64; DIM], [f64; DIM], u64, f64, f64, f64, f64) {
        (
            self.weights(),
            self.n,
            self.seen,
            self.lo_ctx,
            self.hi_ctx,
            self.scale,
            self.floor,
        )
    }
}

#[cfg(test)]
mod tests {
    /// The rate a stored and seeded learner reports is the rate it was stored with, in every band,
    /// and it goes on learning from there exactly.
    #[test]
    fn a_seeded_long_run_rate_continues_exactly() {
        let mut a = super::LongRunRate::default();
        for k in 0..40_usize {
            a.observe(100 + 700 * k, 1 + k % 5, 30_000.0 + 97.0 * k as f64);
        }
        let mut b = super::LongRunRate::default();
        b.seed(&a.rows());
        for ctx in [100_usize, 1000, 5000, 20_000, 30_000] {
            assert_eq!(a.at(ctx).map(f64::to_bits), b.at(ctx).map(f64::to_bits));
            assert_eq!(a.seen(ctx), b.seen(ctx));
        }
        a.observe(5000, 3, 41_000.0);
        b.observe(5000, 3, 41_000.0);
        assert_eq!(a.at(5000).map(f64::to_bits), b.at(5000).map(f64::to_bits));
    }

    use super::*;

    /// The 78-run survey on LFM2: verify milliseconds by row class and context, and the
    /// drafter beside them. Every number here was measured; the test is whether an online
    /// learner fed these rounds one at a time recovers the law they follow.
    const SURVEY: [(usize, [f64; 3]); 4] = [
        (8, [28.859, 29.173, 30.988]),
        (16, [32.407, 32.912, 35.561]),
        (32, [44.555, 45.281, 50.735]),
        (64, [77.434, 78.846, 86.624]),
    ];
    const CTXS: [usize; 3] = [443, 1596, 8444];

    /// A deterministic pseudo-random stream, so a failure is reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            {
                (self.0 >> 11) as f64 / (1u64 << 53) as f64
            }
        }
    }

    /// THE MEASUREMENT THE ENGINE ACTUALLY GETS: the true cost, plus one-sided interference.
    /// A round can be slower than the machine's capability and never faster.
    fn noisy(rng: &mut Rng, truth: f64) -> f64 {
        let r = rng.next();
        if r < 0.15 {
            truth * (1.0 + 0.6 * rng.next()) // a browser on the GPU, a cold pipeline
        } else {
            truth * (1.0 + 0.02 * rng.next())
        }
    }

    /// What the learner COSTS, since it runs inside the decode loop on the host CPU.
    #[test]
    #[ignore = "instrumentation: prints ns per call, gates nothing"]
    fn show_what_a_round_costs() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = Rng(5);
        let mut law = CostLaw::new();
        for _ in 0..500 {
            for &c in &CTXS {
                law.observe(c, noisy(&mut rng, 35.0));
            }
        }
        const N: u32 = 200_000;
        let at = Instant::now();
        for i in 0..N {
            black_box(law.ms(black_box(8444 + i as usize % 17)));
        }
        let predict = at.elapsed().as_secs_f64() * 1e9 / f64::from(N);

        let at = Instant::now();
        for i in 0..N {
            law.observe(black_box(8444 + i as usize % 17), black_box(35.0));
        }
        let train = at.elapsed().as_secs_f64() * 1e9 / f64::from(N);

        // What ONE ROUND asks of it: a price per envelope width, plus the clock, plus two
        // observations (the class that ran, and the round's fixed cost).
        let per_round = 7.0 * predict + 2.0 * train;
        println!(
            "predict {predict:.1} ns   train {train:.1} ns   per round ~{per_round:.0} ns              = {:.5}% of a 40 ms round",
            per_round / 40e6 * 100.0
        );
    }

    /// What the SLOW timescale costs, since it runs at a request boundary.
    #[test]
    #[ignore = "instrumentation: prints ms per refit, gates nothing"]
    fn show_what_a_refit_costs() {
        use std::hint::black_box;
        use std::time::Instant;
        let mut rng = Rng(11);
        let mut law = CostLaw::new();
        for _ in 0..60 {
            for &c in &CTXS {
                law.observe(c, noisy(&mut rng, 35.0));
            }
        }
        const N: u32 = 200;
        let at = Instant::now();
        for _ in 0..N {
            let mut l = law.clone();
            black_box(l.refit());
        }
        let one = at.elapsed().as_secs_f64() * 1e3 / f64::from(N);
        println!(
            "refit {one:.3} ms per law, {:.2} ms for a ten-law table -- at a request boundary",
            one * 10.0
        );
    }

    /// DOES THE BATCH REFIT EARN ITS PLACE? Instrumentation, not a gate.
    ///
    /// The gate inside `refit` scores on the ring's own last quarter, so all it can tell you
    /// is that the candidate is better THERE. What matters is whether the adopted law prices
    /// better at contexts it has not seen, because that is what the budget asks of it every
    /// round.
    ///
    /// So: feed ONE stream of rounds to three learners and score all three against the TRUTH
    /// at fresh contexts across the corpus's range.
    ///
    /// ```text
    ///   online   one FTRL pass, refit never called
    ///   shipped  refit() at each request boundary -- candidate trained on the ring's first
    ///            96, gated on the last 32, adopted as it is
    ///   all128   same selection, but the law that SHIPS is then retrained on all 128
    /// ```
    ///
    /// The third arm is the standard practice the shipped one skips: select on the held-out
    /// slice, then refit on everything. The 32 newest samples -- the ones that carry any
    /// drift -- are currently thrown away by the very law that was adopted for handling them.
    #[test]
    #[ignore = "instrumentation: prints the refit's value, gates nothing"]
    fn show_whether_the_refit_earns_its_place() {
        // The survey's class-8 law, in the units the learner sees: ms, and ms per token.
        const A: f64 = 28.745;
        const B: f64 = 0.000_265_7;
        const PROBE: [usize; 8] = [128, 512, 1024, 2048, 4096, 8192, 12288, 16384];
        #[allow(clippy::cast_precision_loss)]
        fn truth(c: usize) -> f64 {
            A + B * c as f64
        }
        // Worst relative error over the range, which is what a budget decision turns on --
        // an average would hide the one context where the law is wrong enough to flip a pick.
        fn worst(law: &CostLaw) -> f64 {
            let mut w = 0.0_f64;
            for &c in &PROBE {
                let p = law.ms(c).unwrap_or(f64::NAN);
                w = w.max(((p - truth(c)) / truth(c)).abs());
            }
            w * 100.0
        }
        /// Select on the held-out slice, then retrain the law that ships on ALL of it.
        fn refit_on_all(law: &mut CostLaw) -> bool {
            let (train, held) = law.split_held();
            if train.len() < WARM + WARM || held.is_empty() {
                return false;
            }
            let mut cand = CostLaw::new();
            for _ in 0..EPOCHS {
                for &(c, v) in &train {
                    cand.observe(c as usize, f64::from(v));
                }
            }
            if cand.score(&held) >= law.score(&held) {
                return false;
            }
            let all: Vec<(u32, f32)> = train.iter().chain(&held).copied().collect();
            let mut full = CostLaw::new();
            for _ in 0..EPOCHS {
                for &(c, v) in &all {
                    full.observe(c as usize, f64::from(v));
                }
            }
            full.bank = law.bank.clone();
            *law = full;
            true
        }

        println!(
            "{:>7} {:>9} {:>9} {:>9}   {:>7} {:>7}",
            "rounds", "online%", "shipped%", "all128%", "adopt_s", "adopt_a"
        );
        for &n in &[24_usize, 48, 96, 192, 384, 768] {
            let (mut on, mut sh, mut al) =
                (CostLaw::new(), CostLaw::new(), CostLaw::new());
            let mut rng = Rng(7);
            let (mut as_, mut aa) = (0, 0);
            for i in 0..n {
                // A request's worth of rounds at ONE context, then the context moves. That is
                // what a session does, and why the slope is not identifiable inside a request.
                let c = PROBE[(i / 8) % PROBE.len()];
                let ms = noisy(&mut rng, truth(c));
                on.observe(c, ms);
                sh.observe(c, ms);
                al.observe(c, ms);
                if i % 8 == 7 {
                    as_ += usize::from(sh.refit());
                    aa += usize::from(refit_on_all(&mut al));
                }
            }
            println!(
                "{n:>7} {:>9.2} {:>9.2} {:>9.2}   {as_:>7} {aa:>7}",
                worst(&on),
                worst(&sh),
                worst(&al)
            );
        }
    }

    /// WHY IS THE LEARNED SLOPE UNDER-FITTED? Instrumentation, not a gate.
    ///
    /// A live class-8 law read 0.120 us per context token against the survey's 0.2657 -- 45%
    /// of it -- so the law prices 6.1% low at 16384. Three mechanisms could do that, and this
    /// separates them instead of picking one:
    ///
    /// ```text
    ///   L1 shrinkage     |z| - L1 in the numerator biases every kept coefficient toward 0
    ///   the >= 0 clamp   every non-intercept coefficient is clamped non-negative, so a fit
    ///                    that wants a negative u^2 against a positive u cannot have it
    ///   leverage         if a class runs mostly at ONE end of the range, the slope rests on
    ///                    a few high-leverage points -- which is what censoring produces
    /// ```
    ///
    /// Arms. `relaxed` keeps L1's SELECTION and drops its SHRINKAGE (relaxed lasso, adapted
    /// to FTRL's closed form: `-z` instead of `-(z - sgn(z)L1)`); `free` additionally lets the
    /// non-intercept coefficients go negative.
    #[test]
    #[ignore = "instrumentation: prints why the slope under-fits, gates nothing"]
    fn show_why_the_slope_under_fits() {
        const A: f64 = 28.745;
        const B: f64 = 0.000_265_7;
        #[allow(clippy::cast_precision_loss)]
        fn truth(c: usize) -> f64 {
            A + B * c as f64
        }
        /// The shipped weight rule with the two corrections switchable.
        fn weights_as(law: &CostLaw, shrink: bool, clamp: bool) -> [f64; DIM] {
            let mut w = [0.0; DIM];
            for (i, ((out, &z), &n)) in w.iter_mut().zip(&law.z).zip(&law.n).enumerate()
            {
                if z.abs() <= L1 {
                    continue;
                }
                let num = if shrink { z - z.signum() * L1 } else { z };
                let v = -num / ((BETA + n.sqrt()) / ALPHA + L2);
                *out = if i == 0 || !clamp { v } else { v.max(0.0) };
            }
            w
        }
        fn slope_at(law: &CostLaw, w: &[f64; DIM]) -> f64 {
            let at = |c: usize| -> f64 {
                let f = features(c);
                w.iter().zip(f).map(|(a, b)| a * b).sum::<f64>() * law.scale
            };
            (at(16384) - at(128)) / (16384.0 - 128.0) * 1000.0
        }

        // Two context streams over the same range: one that visits every band equally, one
        // skewed to the long end the way a policy that prefers narrow-at-depth would make it.
        const BANDS: [usize; 8] = [128, 512, 1024, 2048, 4096, 8192, 12288, 16384];
        for (name, skew) in [("uniform", false), ("long-skewed", true)] {
            let mut law = CostLaw::new();
            let mut rng = Rng(3);
            for i in 0..800 {
                let c = if skew && i % 8 != 0 {
                    BANDS[5 + (i % 3)] // 8192 / 12288 / 16384 seven rounds in eight
                } else {
                    BANDS[(i / 8) % BANDS.len()]
                };
                law.observe(c, noisy(&mut rng, truth(c)));
            }
            println!("\n  {name}: {} observations", law.seen);
            for (arm, shrink, clamp) in [
                ("shipped", true, true),
                ("relaxed", false, true),
                ("free", false, false),
            ] {
                let w = weights_as(&law, shrink, clamp);
                let s = slope_at(&law, &w);
                println!(
                    "    {arm:>8}: slope {s:.4} us/tok  ({:.0}% of the survey's 0.2657)  w={:?}",
                    s / 0.2657 * 100.0,
                    w.map(|x| (x * 1e3).round() / 1e3)
                );
            }
        }
    }

    /// Instrumentation, not a gate: prints what the learner actually fitted.
    #[test]
    #[ignore = "instrumentation: prints the fit, gates nothing"]
    fn show_the_fitted_weights() {
        let mut rng = Rng(12345);
        println!(
            "{:>4} {:>8} {:>8} {:>8} {:>8} {:>8}   errors at 443/1596/8444",
            "n", "1", "u", "u^2", "sqrt", "ln1p"
        );
        for (rows, at) in SURVEY {
            let mut law = CostLaw::new();
            for _ in 0..300 {
                for (i, &c) in CTXS.iter().enumerate() {
                    law.observe(c, noisy(&mut rng, at[i]));
                }
            }
            let w = law.weights();
            let e: Vec<String> = CTXS
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    format!("{:+.1}%", (law.ms(c).unwrap() - at[i]) / at[i] * 100.0)
                })
                .collect();
            println!(
                "{rows:4} {:8.3} {:8.3} {:8.3} {:8.3} {:8.3}   {}",
                w[0],
                w[1],
                w[2],
                w[3],
                w[4],
                e.join(" ")
            );
        }
    }

    #[test]
    fn it_learns_the_surveys_context_law_from_rounds_it_is_fed_one_at_a_time() {
        let mut rng = Rng(12345);
        for (rows, at) in SURVEY {
            let mut law = CostLaw::new();
            // Rounds arrive interleaved across contexts, the way a server sees requests.
            for _ in 0..300 {
                for (i, &c) in CTXS.iter().enumerate() {
                    law.observe(c, noisy(&mut rng, at[i]));
                }
            }
            assert!(law.has_span(), "n={rows}: three contexts is a span");
            for (i, &c) in CTXS.iter().enumerate() {
                let got = law.ms(c).expect("trained");
                let err = (got - at[i]).abs() / at[i] * 100.0;
                assert!(
                    err < 3.0,
                    "n={rows} ctx={c}: predicted {got:.2} against {:.2} ({err:.1}%)",
                    at[i]
                );
            }
            // AND IT MUST RISE WITH CONTEXT. That is the whole point: a wider class pays
            // more per context token, so the budget's comparison has to move with depth.
            let (near, far) = (law.ms(443).unwrap(), law.ms(8444).unwrap());
            assert!(
                far > near,
                "n={rows}: {far:.2} at 8444 is not above {near:.2} at 443"
            );
        }
    }

    /// A CLASS THAT HAS ONLY EVER RUN AT ONE CONTEXT CANNOT SEE A SLOPE, and the honest
    /// answer is a constant. This is L1 doing its job: with nothing to separate the basis
    /// terms, every coefficient but the intercept is driven to EXACTLY zero, not to a small
    /// number fitted to noise that then extrapolates wildly.
    #[test]
    fn one_context_gives_exactly_zero_slope_and_a_flat_price() {
        let mut rng = Rng(999);
        let mut law = CostLaw::new();
        for _ in 0..400 {
            law.observe(8444, noisy(&mut rng, 30.988));
        }
        assert!(!law.has_span(), "one context is not a span");
        let near = law.ms(443).expect("trained");
        let far = law.ms(8444).expect("trained");
        assert!(
            (near - far).abs() < 0.5,
            "a law with one context extrapolated: {near:.2} at 443 vs {far:.2} at 8444"
        );
    }

    /// THE REFIT IS GATED ON HELD-OUT SAMPLES, so it can only ever help.
    ///
    /// IT IS NOT ESTABLISHED THAT IT HELPS. The live engine's class-8 law implied a +2.25%
    /// rise from 443 to 8444 where the survey measured +7.4%, which looked like a 3x
    /// under-fit -- but that was an UNCONTROLLED comparison: a law's own output against a
    /// survey taken in different runs, with the engine's residual and drift terms left out
    /// of one side. On clean data with enough samples the online pass lands within 4% of
    /// the truth and the refit is no closer. So this tests the MECHANISM -- a candidate is
    /// trained over the retained window, scored on samples it never saw, and adopted only
    /// if it wins -- and claims nothing about magnitude that has not been measured.
    #[test]
    fn a_batch_refit_is_adopted_only_when_it_wins_on_held_out_samples() {
        let truth = |c: usize| {
            #[allow(clippy::cast_precision_loss)]
            {
                28.745 + 0.000_265_7 * c as f64
            }
        };
        let mut rng = Rng(808);
        let mut law = CostLaw::new();
        for _ in 0..60 {
            for &c in &CTXS {
                law.observe(c, noisy(&mut rng, truth(c)));
            }
        }
        let held = law.split_held().1;
        let before = law.score(&held);
        let took = law.refit();
        let after = law.score(&held);
        if took {
            assert!(after <= before, "adopted a worse law: {after} vs {before}");
        } else {
            assert!(
                (after - before).abs() < 1e-12,
                "refused, so nothing may have moved"
            );
        }
        // Either way the law still prices sanely at both ends of its training range.
        for &c in &CTXS {
            let got = law.ms(c).expect("trained");
            assert!(
                (got - truth(c)).abs() / truth(c) < 0.08,
                "ctx={c}: {got:.2} against {:.2}",
                truth(c)
            );
        }
    }

    /// COST CANNOT FALL WITH CONTEXT, and a class whose whole training span is ten tokens
    /// wide has seen nothing that could say otherwise. Before the clamp, six of this
    /// engine's nine classes carried negative context terms fitted from their three probe
    /// rounds -- a slope read off noise.
    #[test]
    fn a_law_trained_on_a_sliver_of_context_never_slopes_down() {
        let mut rng = Rng(2024);
        let mut law = CostLaw::new();
        // Three probe rounds inside a ten-token span, which is what a class the budget
        // never chooses actually gets.
        for i in 0..3 {
            law.observe(470 + i * 5, noisy(&mut rng, 49.88));
        }
        assert!(!law.has_span());
        for (i, &w) in law.weights().iter().enumerate().skip(1) {
            assert!(
                w >= 0.0,
                "context term {i} is {w}, which prices depth as a discount"
            );
        }
        let (near, far) = (law.ms(443).unwrap(), law.ms(16_000).unwrap());
        assert!(
            far >= near,
            "the law falls with context: {near:.2} -> {far:.2}"
        );
    }

    /// ONE-SIDED NOISE MUST NOT INFLATE THE ESTIMATE. A mean-fitting loss would climb with
    /// every burst of interference and never come back down; the pinball loss at a low
    /// quantile tracks the fast rounds, which is what the machine can actually do.
    #[test]
    fn interference_pushes_the_mean_up_and_the_model_stays_down() {
        let mut rng = Rng(4242);
        let mut law = CostLaw::new();
        let truth = 35.561;
        let mut sum = 0.0;
        for _ in 0..400 {
            let s = noisy(&mut rng, truth);
            sum += s;
            law.observe(8444, s);
        }
        let mean = sum / 400.0;
        let got = law.ms(8444).expect("trained");
        assert!(
            mean > truth * 1.03,
            "the test's own noise is not one-sided: {mean:.2}"
        );
        assert!(
            got < mean,
            "the model tracked the mean ({got:.2} vs mean {mean:.2}, truth {truth:.2})"
        );
        assert!(
            (got - truth).abs() / truth < 0.08,
            "predicted {got:.2} vs {truth:.2}"
        );
    }

    /// The band is a guard on extrapolation: a basis evaluated far outside its training range
    /// can return anything, and the budget divides by this number.
    #[test]
    fn a_wild_extrapolation_is_clamped_to_the_band() {
        let mut rng = Rng(7);
        let mut law = CostLaw::new();
        for _ in 0..200 {
            for &c in &CTXS {
                law.observe(c, noisy(&mut rng, 30.0));
            }
        }
        let far = law.ms(4_000_000).expect("trained");
        assert!(far.is_finite() && far > 0.0, "the guard let {far} through");
        assert!(
            far <= law.floor * BAND.1 * law.scale + 1e-9,
            "{far} is outside the band"
        );
    }

    /// A law survives a restart: the weights are what the store holds.
    #[test]
    fn a_stored_law_seeds_a_restart() {
        let mut rng = Rng(31337);
        let mut law = CostLaw::new();
        for _ in 0..200 {
            for (i, &c) in CTXS.iter().enumerate() {
                law.observe(c, noisy(&mut rng, SURVEY[1].1[i]));
            }
        }
        let (w, n, seen, lo, hi, scale, floor) = law.state();
        let mut back = CostLaw::new();
        back.seed(&w, &n, seen, lo, hi, scale, floor);
        for &c in &CTXS {
            let (a, b) = (law.ms(c).unwrap(), back.ms(c).unwrap());
            assert!((a - b).abs() < 1e-6, "ctx={c}: {a} != {b} after a restart");
        }
        assert!(back.has_span());
    }

    /// A RESTORED LAW LEARNS AT THE RATE IT WAS STORED WITH. The same cold round -- 10% slow --
    /// must move the restored law exactly as it moves the one that was stored; restored with its
    /// squared gradients at zero, it moved several times as far (`seed`).
    #[test]
    fn a_restored_law_learns_at_the_rate_it_was_stored_with() {
        let mut rng = Rng(4242);
        let mut law = CostLaw::new();
        for _ in 0..60 {
            for (i, &c) in CTXS.iter().enumerate() {
                law.observe(c, noisy(&mut rng, SURVEY[1].1[i]));
            }
        }
        let (w, n, seen, lo, hi, scale, floor) = law.state();
        let mut back = CostLaw::new();
        back.seed(&w, &n, seen, lo, hi, scale, floor);
        let mut forgetful = CostLaw::new();
        forgetful.seed(&w, &[0.0; super::DIM], seen, lo, hi, scale, floor);
        let c = CTXS[CTXS.len() - 1];
        let before = law.ms(c).unwrap();
        for l in [&mut law, &mut back, &mut forgetful] {
            l.observe(c, before * 1.1);
        }
        let (kept, restored, reset) = (law.ms(c).unwrap(), back.ms(c).unwrap(), forgetful.ms(c).unwrap());
        assert!((kept - restored).abs() < 1e-6, "restored {restored} against kept {kept}");
        assert!(
            (reset - before).abs() > 2.0 * (kept - before).abs(),
            "a reset rate should move further: {reset} vs {kept} from {before}"
        );
    }

    /// THE REASON THE GAIN IS A LINE AND NOT A NUMBER.
    ///
    /// The two contexts a trained session actually visited want opposite corrections -- 1.69
    /// at 1596 tokens and 0.885 at 8444. One scalar cannot hold both: pooled, it lands at
    /// ~1.0, which is the identity, and a session that fitted one measured exactly that
    /// (g = 1.032 over 888 wide rounds). The line must recover BOTH ends from the same
    /// interleaved stream.
    #[test]
    fn the_gain_holds_two_contexts_that_want_opposite_corrections() {
        let mut g = super::MarginGain::default();
        assert!(
            (g.at(1596) - 1.0).abs() < 1e-9 && (g.at(8444) - 1.0).abs() < 1e-9,
            "an unfitted gain must be the identity at every context"
        );
        // Interleaved, as rounds arrive -- not one context then the other.
        for i in 0..200 {
            let p = 0.4 + 0.002 * f64::from(i % 7);
            g.observe(1596, p, p * 1.69);
            g.observe(8444, p, p * 0.885);
        }
        assert!(
            (g.at(1596) - 1.69).abs() < 0.06,
            "near end {} is not 1.69",
            g.at(1596)
        );
        assert!(
            (g.at(8444) - 0.885).abs() < 0.06,
            "far end {} is not 0.885",
            g.at(8444)
        );
        // And it survives the store, which is where it has to arrive from.
        let (xx, xy, seen) = g.state();
        let mut back = super::MarginGain::default();
        back.seed(xx, xy, seen);
        assert!((back.at(1596) - g.at(1596)).abs() < 1e-9);
        assert!((back.at(8444) - g.at(8444)).abs() < 1e-9);
    }

    /// The band bounds the CORRECTION, so a context far outside the fitted span cannot turn
    /// the slope into an absurd multiplier -- the same rule `BAND` applies to the prices.
    #[test]
    fn the_gain_is_bounded_outside_the_span_it_was_fitted_on() {
        let mut g = super::MarginGain::default();
        for _ in 0..400 {
            g.observe(1024, 0.5, 0.5 * 2.5);
            g.observe(2048, 0.5, 0.5 * 1.2);
        }
        let far = g.at(131_072);
        assert!(
            (super::GAIN_BAND.0..=super::GAIN_BAND.1).contains(&far),
            "extrapolated to {far}"
        );
    }

    /// A band prices nothing until it holds `RATE_WARM` rounds, then reports tokens over
    /// microseconds -- and a round at another band's context does not count toward it.
    #[test]
    fn the_rate_is_cold_until_its_band_has_rounds_then_reads_tokens_per_microsecond() {
        let mut r = super::LongRunRate::default();
        for _ in 0..super::RATE_WARM - 1 {
            r.observe(1500, 4, 40_000.0);
            r.observe(9000, 2, 50_000.0);
        }
        assert_eq!(r.at(1500), None);
        assert_eq!(r.seen(1500), super::RATE_WARM - 1);
        r.observe(1100, 4, 40_000.0); // 1024..2047 is the same band as 1500
        let rate = r.at(1500).expect("warm");
        assert!((rate - 1e-4).abs() < 1e-12, "{rate}");
        assert_eq!(r.at(9000), None, "8192..16383 has its own rounds");
        assert_eq!(r.at(512), None, "an empty band stays cold");
    }

    /// The half-life is in rounds: after `RATE_HALF_LIFE` rounds at a new rate, the old rounds
    /// weigh what those new ones weigh together, so the reading sits between the two.
    #[test]
    fn the_rate_forgets_a_workload_with_a_half_life_in_rounds() {
        let mut r = super::LongRunRate::default();
        for _ in 0..4096 {
            r.observe(2000, 8, 40_000.0); // 2e-4 tokens per microsecond
        }
        let half = super::RATE_HALF_LIFE.round() as usize;
        for _ in 0..half {
            r.observe(2000, 2, 40_000.0); // 5e-5
        }
        // Equal weight on both, over the same microseconds per round: (8 + 2) / 2 per 40 ms.
        let rate = r.at(2000).expect("warm");
        assert!((rate - 1.25e-4).abs() < 2e-6, "{rate}");
        for _ in 0..16 * half {
            r.observe(2000, 2, 40_000.0);
        }
        let rate = r.at(2000).expect("warm");
        assert!((rate - 5e-5).abs() < 1e-6, "{rate}");
    }

    /// A law RESTORED without a bank (a store written before it existed) and then fed one band
    /// cannot be replaced by a refit over that band: the held-out test cannot see what a narrow
    /// fit breaks elsewhere.
    #[test]
    fn a_refit_from_one_band_keeps_the_law_that_spans_the_contexts() {
        // 8 rows on Qwen3-4B: 25.3 ms at 443 tokens, 26.9 at 1596, 35.5 at 8444.
        let truth = |c: usize| 24.7 + 0.001_28 * c as f64;
        let mut law = CostLaw::new();
        for _ in 0..40 {
            for c in [443, 1596, 8444] {
                law.observe(c, truth(c));
            }
        }
        let (w, n, seen, lo, hi, scale, floor) = law.state();
        let mut restored = CostLaw::new();
        restored.seed(&w, &n, seen, lo, hi, scale, floor);
        for i in 0..64 {
            restored.observe(8460 + i, truth(8460 + i));
        }
        assert!(!restored.refit(), "a bank at 8460..8523 replaced a law trained from 443");
    }

    /// Contexts that arrive one block at a time -- one request per context, restored between
    /// requests -- are fitted by the refit over the bank, where the online pass alone drags
    /// the intercept toward the newest block.
    #[test]
    fn the_bank_refit_fits_contexts_that_arrived_in_blocks() {
        let truth = |c: usize| 24.7 + 0.001_21 * c as f64;
        let mut law = CostLaw::new();
        for &c in &[443, 1596, 8444, 443, 1596, 8444] {
            for i in 0..50 {
                law.observe(c + i % 7, truth(c + i % 7));
            }
            law.refit();
            let (w, n, seen, lo, hi, scale, floor) = law.state();
            let bank = law.bank().to_vec();
            let mut next = CostLaw::new();
            next.seed(&w, &n, seen, lo, hi, scale, floor);
            next.seed_bank(&bank);
            law = next;
        }
        for c in [443, 1596, 8444] {
            let got = law.ms(c).expect("trained");
            assert!((got - truth(c)).abs() / truth(c) < 0.03, "ctx={c}: {got:.2} against {:.2}", truth(c));
        }
    }

    /// Replays a real sample file -- `context ms` lines, `RESTORE` between processes -- through
    /// three laws: one that never restarts, one restored between processes the way the engine
    /// did before the bank, and one refit over its bank at each request end and restored with
    /// it. `PROBE_SAMPLES=path cargo test ... show_how_laws_fit_replayed_samples -- --ignored`.
    #[test]
    #[ignore = "instrumentation: prints the three laws' prices, gates nothing"]
    fn show_how_laws_fit_replayed_samples() {
        let path = std::env::var("PROBE_SAMPLES").expect("PROBE_SAMPLES");
        let text = std::fs::read_to_string(path).expect("read");
        let mut one = CostLaw::new();
        let mut restored = CostLaw::new();
        let mut banked = CostLaw::new();
        for line in text.lines() {
            if line == "RESTORE" {
                if banked.seen() > 0 {
                    banked.refit();
                    let (w, n, seen, lo, hi, scale, floor) = banked.state();
                    let bank = banked.bank().to_vec();
                    let mut next = CostLaw::new();
                    next.seed(&w, &n, seen, lo, hi, scale, floor);
                    next.seed_bank(&bank);
                    banked = next;
                }
                if restored.seen() > 0 {
                    let (w, n, seen, lo, hi, scale, floor) = restored.state();
                    let mut next = CostLaw::new();
                    next.seed(&w, &n, seen, lo, hi, scale, floor);
                    restored = next;
                }
                continue;
            }
            let mut it = line.split_whitespace();
            let c: usize = it.next().unwrap().parse().unwrap();
            let ms: f64 = it.next().unwrap().parse().unwrap();
            one.observe(c, ms);
            restored.observe(c, ms);
            banked.observe(c, ms);
        }
        banked.refit();
        for c in [443, 1596, 8444] {
            println!("ctx {c}: banked refit per process {:.2}", banked.ms(c).unwrap());
        }
        for c in [443, 1596, 8444] {
            println!("ctx {c}: one process {:.2}  restored per process {:.2}", one.ms(c).unwrap(), restored.ms(c).unwrap());
        }
        // a batch fit over the same samples, the bands interleaved (the last 16 of each band)
        let mut bands: std::collections::BTreeMap<u32, Vec<(usize, f64)>> = std::collections::BTreeMap::new();
        for line in text.lines().filter(|l| *l != "RESTORE") {
            let mut it = line.split_whitespace();
            let c: usize = it.next().unwrap().parse().unwrap();
            let ms: f64 = it.next().unwrap().parse().unwrap();
            bands.entry(usize::BITS - c.leading_zeros()).or_default().push((c, ms));
        }
        let kept: Vec<Vec<(usize, f64)>> = bands.values().map(|v| v[v.len().saturating_sub(16)..].to_vec()).collect();
        let mut inter = Vec::new();
        for i in 0..16 { for v in &kept { if let Some(&x) = v.get(i) { inter.push(x); } } }
        let mut batch = CostLaw::new();
        for _ in 0..EPOCHS { for &(c, ms) in &inter { batch.observe(c, ms); } }
        for (band, v) in &bands {
            let mut ms: Vec<f64> = v.iter().map(|x| x.1).collect(); ms.sort_by(f64::total_cmp);
            println!("band {band}: n={} p15 {:.2} median {:.2}", ms.len(), ms[ms.len() * 15 / 100], ms[ms.len() / 2]);
        }
        for c in [443, 1596, 8444] {
            println!("ctx {c}: batch interleaved {:.2}", batch.ms(c).unwrap());
        }
        println!("one w={:?}", one.weights());
        println!("restored w={:?}", restored.weights());
    }
}
