//! What a verify of n rows costs, measured from the rounds the engine actually runs.
//!
//! DESIGN. A speculative round chooses how many tree rows to verify, and design 6.3 picks
//! that count by `argmax (1 + S(n)) / T(n)`: expected accepted tokens over measured time.
//! `S` comes from the drafter's own probabilities; `T` is this table.
//!
//! WHY THE ENGINE MEASURES IT AND NOT THE TUNER. `imparo-tune` never loads a model's
//! tensors -- it dispatches synthetic micro workloads against arena buffers -- so it cannot
//! run a verify at all, and giving it a model load plus a prefill would turn a 34-second
//! tune into minutes. The engine already runs verifies; each one is a measurement of the
//! row count it ran.
//!
//! WHY THAT DOES NOT COST SERVING CYCLES. The design rejected online cost learning because
//! oMLX's controller spends up to 15% of its cycles on probes. Those probes do no useful
//! work. Here the probe IS the verify: it produces this round's tokens whatever row count
//! it runs at, so the only cost is having run at a row count that was not optimal, for the
//! handful of rounds it takes to fill the table once.
//!
//! WHAT MAKES IT A HANDFUL. A matrix kernel pads to its tile, so T is flat over a range of
//! row counts and steps between ranges, and within a flat range S grows while T does not --
//! design 6.3's own observation -- which means only the LARGEST row count of a range is ever
//! worth running. The table calls such a range a STEP (or class) and prices it at its top.
//!
//! Measured (LFM2, 30-layer verify, 443-token context, one engine run per row count): rows
//! 2/4/6/8 read 29.3 / 29.5 / 29.7 / 28.9 ms, rows 10/12/16 read 33.0 / 33.2 / 32.4, rows
//! 20/24/32 read 45.3 / 45.5 / 44.6. Flat inside a range to under 1 ms and not rising with n.
//!
//! THE STEPS ARE LEARNED, NOT DECLARED. The table starts from a ladder of widths (2, 4, 8, ...
//! up to the widest verify) and splits an interval only where a width inside it could be
//! chosen (`refine`), so it needs nothing from the backend but the rounds it serves.
//!
//! Rejected: the backend declaring its tile classes per row count. Only the 8-bit kernel ever
//! declared them, so every 4-bit and routed target borrowed that kernel's partition. Measured
//! against it (dspark-paper/runs/ab_classes7: 9 cells x 2 repeats, store learned once per model;
//! f55a73de is the last commit with both arms), learned over declared ms/token:
//!
//! ```text
//!   LFM2.5-2.6B Q8     -0.8%     LFM2.5-8B-A1B      -4.1%
//!   Qwen3-4B Q4_K_M    +0.15%    Qwen3-8B Q4_K_M    +0.15%
//! ```
//!
//! The Qwen3 targets tie because 8 rows is the best width there at every context, and the
//! declared partition happened to hold 8. They lost 2.3-3.4% until each law kept a bank of
//! samples per context band (`CostLaw::bank`): the online law had mispriced 8 rows at short
//! context, and the chooser held 6 or 7 rows instead.

//! THE ESTIMATOR. Inside a class the cost is a constant plus noise, and the noise is
//! ONE-SIDED: a browser taking the GPU, a cold pipeline, a thermal drop all make a round
//! slower, and nothing makes a verify faster than the machine can run it. So the estimate
//! of a class is the MINIMUM of its recent samples, not their mean -- a mean carries one
//! interference spike forever, a minimum never sees it. This is the rule the bracket and
//! the tuner already work by.
//!
//! The minimum is over a WINDOW, because the cost really does drift: it rises with the
//! context the verify attends over (survey, 443 -> 8444 tokens: +7.6% at 8 rows, +12.6% at
//! 32). A window lets an estimate rise again when the machine gets slower; an all-time
//! minimum could only ever fall.
//!
//! AND THE STALE-CHEAP TRAP. Only the row count the budget CHOOSES runs every round, so
//! every other class's minimum was taken earlier -- at a shorter context, on a quieter
//! machine. Left alone, the alternatives always look cheaper than the incumbent, which is
//! exactly the error that makes a budget switch when it should not. So each class records
//! the table's SCALE when its minimum was taken, and a class that has stopped running is
//! read through the ratio. Nothing is re-run to learn the scale.
//!
//! THE SCALE COMES FROM ONE CLOCK, and the clock is the round's FIXED cost -- the drafter's
//! forward and the tree build, the one thing every round pays whatever it verifies. Its
//! window is therefore always the last `WINDOW` rounds, and its base is the minimum at the
//! table's epoch and never moves, so the scale is a ratio against ONE instant.
//!
//! ```text
//!   Correct: one scale = "the machine is now s times the table's epoch",
//!            measured by the only quantity that runs every round.
//!   Wrong:   every class writes a new GLOBAL scale from its OWN epoch, so each write
//!            re-prices every other class through an epoch that is not theirs -- and
//!            through that class's window-minimum noise.
//! ```
//!
//! The wrong form was built and measured: at an 8444-token context the whole table swung
//! 11% up and down several times per run (T(8) 28866 <-> 32094 against a true 30988, T(16)
//! 31953 <-> 35527 against 35561), because one class re-anchored, overwrote the scale, and
//! the next class overwrote it back. It is a multiplicative random walk, not drift, and it
//! pushed the budget onto the wider tree in 42 of 63 rounds.

/// Samples kept per class. The chosen class re-measures every round, so this is also how
/// often the scale is refreshed: often enough to follow a growing context, long enough that
/// one slow round does not become the estimate.
const WINDOW: usize = 8;

/// Samples before a class has an estimate at all. The first is usually the cold one, and
/// the minimum discards it without being told to.
const PROBES: u64 = 3;

/// A clock reading outside this is not drift, it is a broken measurement, and a broken
/// measurement must not be allowed to rescale the whole table.
const SCALE_STEP: (f64, f64) = (0.5, 2.0);

/// LEARNED CLASSES: the widths the table starts from before it has measured anything. Powers
/// of two up to the widest verify; every other width is found by splitting an interval
/// (`refine`).
const LADDER_START: usize = 2;

/// Rounds the chooser must hold one width before the table splits an interval next to it. A
/// width it is still moving between says nothing yet about which interval matters.
const REFINE_HOLD: u64 = 8;

/// A step this large, relative to its bottom, is steep: a kernel's tile step rather than a
/// routed target's gradual rise. The walk away from the held width stops past one, and one right
/// next to the held width is split beside it first (`refine`).
const STEEP: f64 = 0.25;

/// At most this many learned widths. Each costs `PROBES` rounds to price, so the bound is a
/// bound on exploration.
const MAX_STEPS: usize = 24;

/// Whether the context law prices at all. Off means the anchor alone: a constant per class.
fn law_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_COST_LAW").as_deref() != Ok("0"))
}

/// One row count worth timing, and what it has measured.
#[derive(Clone, Debug)]
pub struct Step {
    /// The smallest row count in this class.
    pub lo: usize,
    /// The largest -- the only count in the class worth running, and what `probe` returns.
    pub hi: usize,
    /// The last `WINDOW` samples, microseconds.
    window: Vec<f64>,
    /// Ring cursor into `window`.
    next: usize,
    /// Samples reported at this class, ever.
    pub seen: u64,
    /// The minimum this class contributed, and the table scale it was taken at. `None`
    /// until the class has `PROBES` samples.
    anchor: Option<(f64, f64)>,
    /// Samples since the anchor was taken. At `WINDOW` the window holds nothing older, so
    /// a fresh minimum is comparable with the anchored one and the scale can move.
    since: usize,
    /// THIS CLASS'S COST AS A FUNCTION OF THE CONTEXT (`cost_model`). The anchor above is a
    /// single number and cannot express a slope; the classes measured here have SEVEN
    /// different slopes, so one number per class is the wrong shape and the error it makes
    /// is systematic, not noise.
    law: crate::cost_model::CostLaw,
    /// The table's scale when this class last measured anything. TWO TIMESCALES: the law is
    /// the CONTEXT law, a property of the kernels that changes on a retune; the clock is the
    /// machine's DRIFT right now. A class that is running has already absorbed current drift
    /// into its law, so this equals the scale and the ratio is 1. A class that stopped is
    /// carried by how far the clock has moved since -- which is the only part of its price
    /// a law it never re-trained cannot know.
    at_scale: f64,
    /// THE RESIDUAL: the last `WINDOW` ratios of what a round ACTUALLY took over what this
    /// step's law said it would, and the ring cursor into them. The law is the slow shape;
    /// the residual is the fast level. A step that is running tracks its own level here
    /// within a window, which is what a rigid law cannot do and must not try to.
    ratios: Vec<f64>,
    ratio_next: usize,
    /// Samples that came back below this step's top while it was still being probed -- the
    /// round asked for the top and the tree had fewer nodes -- and whether that has happened
    /// `PROBES` times, which makes the top a width no round can build.
    short: u64,
    unreachable: bool,
}

impl Step {
    fn min(&self) -> Option<f64> {
        self.window.iter().copied().reduce(f64::min)
    }

    /// Folds a sample in and re-anchors at the table's current `scale` once a full window
    /// has been measured since the last anchor.
    ///
    /// A STEP NEVER WRITES THE SCALE. It has one epoch -- the instant its own anchor was
    /// taken -- and a scale computed from that epoch is meaningless to every other step.
    /// The table's clock is the fixed cost, which runs every round.
    fn observe(&mut self, us: f64, scale: f64, at: usize) {
        // The residual is taken BEFORE this sample trains the law, so it measures what the
        // law did not already know.
        if let Some(p) = self.law.ms(at).filter(|p| *p > 0.0) {
            let r = us / (p * 1e3);
            if self.ratios.len() < WINDOW {
                self.ratios.push(r);
            } else {
                self.ratios[self.ratio_next] = r;
            }
            self.ratio_next = (self.ratio_next + 1) % WINDOW;
        }
        self.law.observe(at, us / 1e3);
        self.at_scale = scale;
        if self.window.len() < WINDOW {
            self.window.push(us);
        } else {
            self.window[self.next] = us;
        }
        self.next = (self.next + 1) % WINDOW;
        self.seen += 1;
        self.since += 1;
        let Some(fresh) = self.min() else { return };
        match self.anchor {
            // A FULL WINDOW of samples since the anchor: this minimum stands on its own,
            // so it replaces the anchored one at the scale the clock reads now. Before
            // that the window still straddles the anchor, where a fresh minimum can only
            // have fallen -- which is noise, not drift -- so the anchor is left where it is.
            Some(_) if self.since >= WINDOW => {
                self.anchor = Some((fresh, scale));
                self.since = 0;
            }
            // The first estimate: it anchors at the scale the table is at now.
            None if self.seen >= PROBES => {
                self.anchor = Some((fresh, scale));
                self.since = 0;
            }
            // Anchored but mid-window, or not yet at PROBES samples: nothing to move.
            Some(_) | None => {}
        }
    }

    fn blank(lo: usize, hi: usize) -> Self {
        Self {
            lo,
            hi,
            window: Vec::with_capacity(WINDOW),
            next: 0,
            seen: 0,
            anchor: None,
            since: 0,
            law: crate::cost_model::CostLaw::new(),
            at_scale: 1.0,
            ratios: Vec::with_capacity(WINDOW),
            ratio_next: 0,
            short: 0,
            unreachable: false,
        }
    }
    /// THE PRICE AT A CONTEXT, microseconds.
    ///
    /// The law answers when it has been trained. Under that -- the first rounds of a class
    /// -- the anchored minimum carried by the clock stands in, which is what this table did
    /// before the law existed.
    fn us(&self, scale: f64, at: usize) -> Option<f64> {
        // THE CONTROL ARM. `IMPARO_DSPARK_COST_LAW=0` prices from the anchor alone, which
        // is a CONSTANT per class -- what this table did before it had a context law. One
        // binary, so the arm under test and the arm it is measured against are the same
        // code everywhere else.
        if !law_on() {
            return self.anchor.map(|(m, was)| m * scale / was);
        }
        // THE THREE TERMS, each on its own timescale:
        //   law(ctx)          the context shape          slow, a kernel property
        //   resid             this step's own level      fast, its last WINDOW rounds
        //   scale / at_scale  the machine since it ran   zero while it is running
        self.law
            .ms(at)
            .map(|ms| ms * 1e3 * self.resid() * scale / self.at_scale)
            .or_else(|| self.anchor.map(|(m, was)| m * scale / was))
    }

    /// The fast level: a MINIMUM over the window, for the same reason every estimate here
    /// is one -- interference only ever adds -- and a window, not all time, so it can come
    /// back down when the machine does.
    fn resid(&self) -> f64 {
        self.ratios
            .iter()
            .copied()
            .reduce(f64::min)
            .map_or(1.0, |m| m.clamp(SCALE_STEP.0, SCALE_STEP.1))
    }
}

/// One stored law: the range, the basis weights, FTRL's accumulated squared gradients (its
/// learning rate), the rounds behind it, the context span it was trained over, the scale it
/// normalises by, and its floor.
pub type LawRow = (
    usize,
    usize,
    [f64; crate::cost_model::DIM],
    [f64; crate::cost_model::DIM],
    u64,
    f64,
    f64,
    f64,
    f64,
);

/// The measured verify time by row count.
#[derive(Clone, Debug)]
pub struct VerifyCost {
    steps: Vec<Step>,
    /// What a round costs whatever row count it verifies: the drafter's forward and the
    /// tree build. THE TABLE'S CLOCK: it is the one thing every round pays, so its window
    /// is always the last `WINDOW` rounds and it always has a current reading.
    fixed: Step,
    /// The clock's minimum at the table's epoch. It never moves, so the scale is a ratio
    /// against one instant and cannot compound.
    base: Option<f64>,
    /// How much slower the machine is now than its own law says it should be. Only the
    /// clock writes it, and only as a residual, so context is never charged to drift.
    scale: f64,
    /// THE CONTEXT THE TABLE IS PRICING FOR, set once per round by the engine. Cost is a
    /// function of this, not a constant, and it changes every round.
    at: usize,
    /// The scale of the VALUE side's marginal, learned the same way the prices are. It sits
    /// here because it is stored with them and for the same reason: it needs rounds at more
    /// than one width, which one short request does not give.
    gain: crate::cost_model::MarginGain,
    /// The same scale for the n-gram's and agreed nodes' marginal (design 6.6's g_n).
    gain_n: crate::cost_model::MarginGain,
    /// The width the chooser held last round, and for how many rounds in a row.
    incumbent: usize,
    held: u64,
}

impl VerifyCost {
    /// A table that LEARNS WHERE THE COST STEPS: one step per width of a ladder (2, 4, 8, ...
    /// up to `max_rows`), each covering the widths down to the one below it. `refine` splits an
    /// interval next to the width in use when a width inside it could be chosen, so the widths
    /// the chooser can pick become finer exactly where the choice is made. `None` when
    /// `max_rows` leaves no width to choose.
    #[must_use]
    pub fn new(max_rows: usize) -> Option<Self> {
        if max_rows < LADDER_START {
            return None;
        }
        let mut steps = Vec::new();
        let (mut lo, mut hi) = (LADDER_START, LADDER_START);
        while hi < max_rows {
            steps.push(Step::blank(lo, hi));
            lo = hi + 1;
            hi *= 2;
        }
        steps.push(Step::blank(lo, max_rows));
        Some(Self {
            steps,
            fixed: Step::blank(0, 0),
            base: None,
            scale: 1.0,
            at: 0,
            gain: crate::cost_model::MarginGain::default(),
            gain_n: crate::cost_model::MarginGain::default(),
            incumbent: 0,
            held: 0,
        })
    }

    /// SPLITS AN INTERVAL ONLY WHERE A WIDTH INSIDE IT COULD BE CHOSEN. Called once per round,
    /// after the chooser, with the width it holds and `could_beat(rows, us)`: whether `rows`
    /// verified in `us` would clear the chooser's own switching bar against the held width.
    /// Returns the width it added, which `probe` then runs until it has a price.
    ///
    /// OPTIMISM UNDER UNCERTAINTY, with the shape the kernels give. Between two measured widths
    /// `a < b` the time of any width inside is at least `T(a)`, and its expected tokens are at
    /// most those of `b - 1`, which the round already knows. So the best any width inside could
    /// do is `b - 1` rows at `T(a)`: if even that cannot beat the held width, no width inside can
    /// ever be chosen and none is measured. The table explores only where the answer could
    /// change the choice, which is what makes it cheap to learn once per model.
    ///
    /// ```text
    ///   LFM2.5 Q8, held 16:  17..31 add almost no tokens, cost >= T(16)  -> never probed
    ///   routed target, held 8: 12 rows could buy more than they cost     -> probed, halved
    /// ```
    ///
    /// WHERE THE SPLIT GOES. An interval next to the held width whose top costs more than
    /// `STEEP` above its bottom is split beside the held width: a kernel step is usually sharp,
    /// and one probe shows whether it sits at the boundary. Every other interval is halved.
    pub fn refine(
        &mut self,
        incumbent: usize,
        could_beat: impl Fn(usize, f64) -> bool,
    ) -> Option<usize> {
        if incumbent == self.incumbent {
            self.held += 1;
        } else {
            self.incumbent = incumbent;
            self.held = 0;
        }
        if self.held < REFINE_HOLD || self.steps.len() >= MAX_STEPS || self.probe().is_some() {
            return None;
        }
        let j = self.steps.iter().position(|s| s.hi == incumbent)?;
        let price: Vec<Option<f64>> =
            self.steps.iter().map(|s| s.us(self.scale, self.at)).collect();
        // The intervals nearest the held width first: a split there can change the next choice.
        let mut order: Vec<usize> = (0..self.steps.len().saturating_sub(1)).collect();
        order.sort_by_key(|&k| if k >= j { k - j } else { j - k - 1 });
        let (k, m) = order.into_iter().find_map(|k| {
            let (bottom, top) = (self.steps[k].hi, self.steps[k + 1].hi);
            let (pb, pt) = (price[k]?, price[k + 1]?);
            if top - bottom < 2 || !could_beat(top - 1, pb) {
                return None;
            }
            let steep = pt > pb * (1.0 + STEEP);
            let m = match (k == j, k + 1 == j) {
                (true, _) if steep => bottom + 1,
                (_, true) if steep => top - 1,
                _ => bottom.midpoint(top),
            };
            Some((k + 1, m))
        })?;
        Self::split(&mut self.steps, k, m);
        // The new width is probed next; the chooser is held again before the next split.
        self.held = 0;
        Some(m)
    }

    /// Splits step `b` at width `m`: a new, unmeasured step `lo..=m` below the old one, which
    /// keeps its samples and law because they were measured at its top.
    fn split(steps: &mut Vec<Step>, b: usize, m: usize) {
        let lo = steps[b].lo;
        steps[b].lo = m + 1;
        steps.insert(b, Step::blank(lo, m));
    }

    /// Folds in what a verify of `rows` rows took.
    ///
    /// A STEP IS PRICED AT ITS TOP ONLY. Between two measured widths nothing says a narrower
    /// width costs what the top does, and a narrower sample would price the top below what it
    /// costs; so a row count that is not a step's top is not recorded. A probe that could not
    /// build its width is `unreached`'s business, told when the tree is built.
    pub fn observe(&mut self, rows: usize, elapsed: std::time::Duration) {
        let (us, scale, at) = (elapsed.as_secs_f64() * 1e6, self.scale, self.at);
        if let Some(step) = self.steps.iter_mut().find(|s| s.hi == rows) {
            step.observe(us, scale, at);
        }
    }

    /// A PROBE THAT ASKED FOR MORE ROWS THAN THE ROUND COULD BUILD: `asked` is the probed width,
    /// `built` the tree the round actually built. Told when the tree is built, because only the
    /// caller knows a round was a probe -- the verify's width alone cannot say which step a
    /// short tree was meant for.
    ///
    /// ```text
    ///   Correct: probe 64 -> tree of 57 -> unreached(64, 57) -> the widest step ends at 57
    ///   Wrong:   probe 64 -> verify of 57 -> observe(57) lands in 33..57 -> 58..64 still unmeasured
    ///            -> every later round probes 64 again (a restarted table re-split at 57 did this:
    ///            Qwen3-4B 3x slower, every round a 57-row verify)
    /// ```
    ///
    /// THE WIDEST STEP LEARNS ITS REACH: 7 positions of 8 candidates offer 57 rows, not 64, so its
    /// top becomes the width built -- or, if a narrower step already ends at or above it, the
    /// step goes. ANY OTHER STEP a tree keeps falling short of is a width no round builds; after
    /// `PROBES` such rounds `probe` stops asking for it, or every round would become one.
    pub fn unreached(&mut self, asked: usize, built: usize) {
        if built >= asked {
            return;
        }
        let last = self.steps.len().saturating_sub(1);
        let Some(i) = self.steps.iter().position(|s| s.hi == asked) else {
            return;
        };
        if self.steps[i].seen >= PROBES {
            return;
        }
        if i == last && i > 0 {
            if built >= self.steps[i].lo {
                self.steps[i].hi = built;
            } else {
                self.steps.pop();
            }
            return;
        }
        let step = &mut self.steps[i];
        step.short += 1;
        step.unreachable = step.short >= PROBES;
    }

    /// THE CONTEXT THE NEXT ROUND RUNS AT. The engine sets it once per round, before it asks
    /// for a price and before it reports what a round took: cost is a function of this, and
    /// it changes every round.
    pub fn set_context(&mut self, at: usize) {
        self.at = at;
    }

    /// The context the table is pricing for.
    #[must_use]
    pub fn context(&self) -> usize {
        self.at
    }

    /// Folds in the round's cost that does NOT depend on the row count -- the drafter's
    /// forward, the tree build, the append.
    ///
    /// DESIGN 6.3 LEFT THIS OUT, on the grounds that the drafter's time "is already spent
    /// when n is chosen". That is a sunk-cost argument, and the objective is not one
    /// round's ratio: it is tokens per second across rounds, and the drafter is paid once
    /// PER ROUND. A narrower tree makes rounds shorter but more numerous, and each new
    /// round pays it again. Measured on LFM2 with the fixed cost left out, the budget chose
    /// 8 rows in 50 of 64 rounds at a 1596-token context and read +2.2% ms/token against a
    /// fixed 16.
    pub fn observe_fixed(&mut self, elapsed: std::time::Duration) {
        let us = elapsed.as_secs_f64() * 1e6;
        // THE DRIFT IS A RESIDUAL, measured BEFORE this sample trains the law: what the
        // round actually took, over what its law said it would take at this context.
        //
        //   Correct: drift = observed / law(ctx)   context is explained, so it never
        //                                          reaches the drift term
        //   Wrong:   drift = observed / a constant a table with no context term charges
        //                                          context growth to the machine
        //
        // The second form is what this table did, and at an 8444-token context it read the
        // drift at 1.11 when the machine had not drifted at all.
        self.fixed.observe(us, self.scale, self.at);
        // The clock's residual IS the machine's drift: what the one thing every round pays
        // took, over what its own law expected. Context is explained by the law, so it can
        // never reach this term.
        self.scale = self.fixed.resid();
        if self.base.is_none() && self.fixed.seen >= PROBES {
            self.base = self.fixed.min();
        }
    }

    /// The round's n-independent cost, or `None` until it has samples.
    ///
    /// The clock reads itself: its window is the last `WINDOW` rounds, so its own minimum
    /// IS the current cost. Carrying it through the scale would divide it by its own base.
    ///
    /// Before this run has a single round -- a table seeded from a store -- the stored base
    /// stands in. It is one round old at most: the first `observe_fixed` replaces it.
    #[must_use]
    pub fn fixed_us(&self) -> Option<f64> {
        // The clock reads its own law at this round's context, and never through the scale
        // it is itself the source of.
        if !law_on() {
            return self.fixed.min().or(self.base);
        }
        self.fixed
            .law
            .ms(self.at)
            .map(|ms| ms * 1e3 * self.fixed.resid())
            .or_else(|| self.fixed.min())
            .or(self.base)
    }

    /// The cost of verifying `rows` rows, made monotone: the cheapest measured class that
    /// holds at least `rows`.
    ///
    /// Monotone because a wider verify is sometimes CHEAPER -- on LFM2, 40 rows on the
    /// 64-token tile read 79.5 ms against 48 rows on the seat's two walks at 67.8 -- and
    /// asking for 40 rows should then be priced at what sending 48 costs, since sending 48
    /// is what a caller would do.
    ///
    /// `None` when nothing at or above `rows` has been measured yet.
    #[must_use]
    pub fn cost_us(&self, rows: usize) -> Option<f64> {
        self.steps
            .iter()
            .filter(|s| s.hi >= rows)
            .filter_map(|s| s.us(self.scale, self.at))
            .reduce(f64::min)
    }

    /// The next row count to run to fill the table, or `None` once every class AND the
    /// round's fixed cost have their samples. Classes fill in order, so the cheap widths
    /// are priced first.
    ///
    /// A SEEDED TABLE STILL PROBES for the fixed cost: it is not stored with a class, and
    /// without it the budget would spend its first rounds on design 6.3's ratio, which is
    /// the one this table exists to correct. Any row count measures it -- it does not
    /// depend on n -- so the narrowest, which is the cheapest round.
    #[must_use]
    pub fn probe(&self) -> Option<usize> {
        if let Some(s) = self.steps.iter().find(|s| s.seen < PROBES && !s.unreachable) {
            return Some(s.hi);
        }
        // A seeded table carries the clock's BASE, which is an estimate of the fixed cost
        // until this run has measured its own -- so it needs no probe round for it.
        if self.fixed.seen < PROBES && self.base.is_none() {
            return self.steps.first().map(|s| s.hi);
        }
        None
    }

    /// Seeds the classes from a stored table, so a restarted engine does not re-probe.
    ///
    /// The stored widths are split into this build's ladder first, and every stored range
    /// must then be one of its steps, entry for entry. A file that names any other range --
    /// one written by a build with a different widest verify -- is refused whole rather than
    /// mixed. Returns whether it took.
    ///
    /// A seeded class starts with ONE sample and the scale at 1. The stored number is what
    /// the machine did last time, at whatever context that run reached; it is a start, not a
    /// claim, and the window and scale correct it from the first rounds this run measures.
    pub fn seed(&mut self, rows: &[(usize, usize, f64, u64)]) -> bool {
        let classes: Vec<(usize, usize, f64, u64)> = rows
            .iter()
            .copied()
            .filter(|(lo, hi, _, _)| !(*lo == 0 && *hi == 0))
            .collect();
        // THE WIDTHS THE STORED RUN HAD SPLIT ARE RESTORED FIRST: every stored top that is not
        // yet a step is split in, on a copy, so the check below sees the same steps the stored
        // run had -- and a refused file still changes nothing.
        let mut steps = self.steps.clone();
        let ladder: Vec<usize> = steps.iter().map(|s| s.hi).collect();
        let mut tops: Vec<usize> = classes.iter().map(|c| c.1).collect();
        tops.sort_unstable();
        tops.dedup();
        for &h in &tops {
            if let Some(b) = steps.iter().position(|s| s.lo <= h && h < s.hi) {
                Self::split(&mut steps, b, h);
            }
        }
        // THE STORED REACH. A split always leaves the step above it stored with its samples,
        // and the ladder is probed from the narrow end, so a widest stored top that is not a
        // ladder width can only be the reach the widest step learned (`unreached`): nothing
        // above it is built, and nothing above it is kept.
        if let Some(&widest) = tops.last() {
            if !ladder.contains(&widest) {
                steps.retain(|s| s.lo <= widest);
            }
        }
        // EVERY STORED CLASS MUST BE ONE OF THIS BUILD'S, checked before anything is taken, so a
        // refused file changes nothing. A class the file does not hold is probed as on a first
        // run: a file written mid-probe carries only the classes measured so far.
        if classes.is_empty()
            || classes.iter().any(|(lo, hi, us, _)| {
                *us <= 0.0 || !steps.iter().any(|s| s.lo == *lo && s.hi == *hi)
            })
        {
            return false;
        }
        self.steps = steps;
        // A `(0, 0, ..)` row is the round's fixed cost, which belongs to no class.
        if let Some((_, _, us, _)) =
            rows.iter().find(|(lo, hi, _, _)| *lo == 0 && *hi == 0)
        {
            // THE STORED CLOCK READING IS A BASE, NOT A SAMPLE. Its window is "the last
            // WINDOW rounds", and a number from a previous process is not one of them --
            // seeded into the window it would hold the clock at the old run's reading for
            // a whole window and then STEP when it rolled out, re-pricing every class that
            // had re-anchored in the meantime. So the window starts empty and the stored
            // value becomes the epoch the scale is measured against: a class stored at the
            // old run's price is carried by how far the clock has moved since the store.
            self.fixed = Step::blank(0, 0);
            self.base = Some(*us);
        }
        for (lo, hi, us, seen) in classes {
            let Some(step) = self.steps.iter_mut().find(|s| s.lo == lo && s.hi == hi)
            else {
                continue;
            };
            step.window.clear();
            step.window.push(us);
            step.next = 1 % WINDOW;
            // THE STORED COUNT, so a class stored short of `PROBES` finishes its probe instead
            // of starting it again, and a probed class is not probed twice. A class short of it
            // has no estimate yet: its stored sample waits in the window, and the probe's
            // minimum over the window becomes its first anchor as on a first run.
            step.seen = seen.max(1);
            step.anchor = (seen >= PROBES).then_some((us, 1.0));
            step.since = 0;
        }
        self.scale = 1.0;
        true
    }

    /// The table as rows to store: every class that has an estimate, at the current scale, with
    /// the samples behind it -- including a probe that is not finished.
    ///
    /// A HALF-PROBED TABLE IS STORED, because the server's provider lives for one request: a
    /// cold table needs `PROBES` rounds per class, more rounds than a short answer runs, and a
    /// probe that is dropped at every request end never finishes. A restart seeded from it
    /// resumes the probe at the classes still short, and the budget chooses nothing until every
    /// class has its samples (`probe`), so a half table never makes it choose among the classes
    /// that happened to fill first.
    #[must_use]
    pub fn rows(&self) -> Vec<(usize, usize, f64, u64)> {
        let mut out: Vec<(usize, usize, f64, u64)> = self
            .steps
            .iter()
            .filter(|s| s.seen > 0)
            .filter_map(|s| {
                // A class still in its probe has no price yet; its lowest sample is stored.
                s.us(self.scale, self.at)
                    .or_else(|| s.min())
                    .map(|us| (s.lo, s.hi, us, s.seen))
            })
            .collect();
        if let Some(us) = self.fixed_us() {
            out.push((0, 0, us, self.fixed.seen));
        }
        out
    }

    /// THE SLOW TIMESCALE. Every law refits over its retained window and is kept only if it
    /// beats the incumbent on samples it never saw. Returns how many were adopted.
    ///
    /// CALLED AT A REQUEST BOUNDARY, not inside a round: it is ~0.18 ms per law, which is a
    /// whole decode round, and nothing is waiting when a request ends. The fast path stays
    /// 399 ns and its weights are read by the very next round; this is the other half.
    pub fn refit(&mut self) -> usize {
        self.steps
            .iter_mut()
            .chain(std::iter::once(&mut self.fixed))
            .map(|s| usize::from(s.law.refit()))
            .sum()
    }

    /// The learned scale for a predicted marginal, or 1.0 until there is enough to say.
    #[must_use]
    pub fn gain(&self) -> f64 {
        self.gain.at(self.at)
    }

    /// One round's evidence about that scale: what the value model predicted the extra rows
    /// would buy, and what they actually bought.
    pub fn observe_gain(&mut self, ctx: usize, predicted: f64, actual: f64) {
        self.gain.observe(ctx, predicted, actual);
    }

    /// `(num, den, seen)` for the store.
    #[must_use]
    pub fn gain_state(&self) -> ([f64; 3], [f64; 2], u64) {
        self.gain.state()
    }

    /// From the store, so the gain arrives trained.
    pub fn seed_gain(&mut self, xx: [f64; 3], xy: [f64; 2], seen: u64) {
        self.gain.seed(xx, xy, seen);
    }

    /// The n-gram's and agreed nodes' marginal scale (design 6.6's g_n), 1.0 until evidenced.
    #[must_use]
    pub fn gain_n(&self) -> f64 {
        self.gain_n.at(self.at)
    }

    /// One wide round's evidence about g_n.
    pub fn observe_gain_n(&mut self, ctx: usize, predicted: f64, actual: f64) {
        self.gain_n.observe(ctx, predicted, actual);
    }

    /// g_n's state, for the store.
    #[must_use]
    pub fn gain_n_state(&self) -> ([f64; 3], [f64; 2], u64) {
        self.gain_n.state()
    }

    /// g_n from the store.
    pub fn seed_gain_n(&mut self, xx: [f64; 3], xy: [f64; 2], seen: u64) {
        self.gain_n.seed(xx, xy, seen);
    }

    /// THE LEARNED LAWS, to store: `(lo, hi, weights, squared gradients, trained rounds,
    /// lo_ctx, hi_ctx, scale, floor)`, the clock as `(0, 0, ..)`.
    ///
    /// This is the half of the table that CANNOT be learned inside one request. A single
    /// answer moves the context by a few percent, so the slope is not identifiable from it;
    /// it can only come from rounds at different depths, which means it can only come
    /// through the store. Persisting the law is what makes the law learnable at all.
    #[must_use]
    pub fn laws(&self) -> Vec<LawRow> {
        let one = |s: &Step| {
            let (w, n, seen, lo_ctx, hi_ctx, scale, floor) = s.law.state();
            (seen > 0 && floor.is_finite())
                .then_some((s.lo, s.hi, w, n, seen, lo_ctx, hi_ctx, scale, floor))
        };
        self.steps
            .iter()
            .chain(std::iter::once(&self.fixed))
            .filter_map(one)
            .collect()
    }

    /// Every law's sample bank, `(lo, hi, samples)`, for the store: what the refit at the next
    /// request end trains on, so the slow fit keeps every context band it has been measured in.
    #[must_use]
    pub fn banks(&self) -> Vec<(usize, usize, Vec<(u32, f32)>)> {
        self.steps
            .iter()
            .chain(std::iter::once(&self.fixed))
            .filter(|s| !s.law.bank().is_empty())
            .map(|s| (s.lo, s.hi, s.law.bank().to_vec()))
            .collect()
    }

    /// Restores the banks, after `seed_laws`. A bank whose range is not one of THIS build's
    /// classes is dropped, as its law is.
    pub fn seed_banks(&mut self, banks: &[(usize, usize, Vec<(u32, f32)>)]) {
        for (lo, hi, samples) in banks {
            let step = if *lo == 0 && *hi == 0 {
                Some(&mut self.fixed)
            } else {
                self.steps.iter_mut().find(|s| s.lo == *lo && s.hi == *hi)
            };
            if let Some(step) = step {
                step.law.seed_bank(samples);
            }
        }
    }

    /// Restores the laws a previous run learned. A law whose range is not one of THIS
    /// build's classes is dropped: the same rule the anchors follow, for the same reason.
    pub fn seed_laws(&mut self, laws: &[LawRow]) {
        for &(lo, hi, w, n, seen, lo_ctx, hi_ctx, scale, floor) in laws {
            let step = if lo == 0 && hi == 0 {
                Some(&mut self.fixed)
            } else {
                self.steps.iter_mut().find(|s| s.lo == lo && s.hi == hi)
            };
            if let Some(step) = step {
                step.law.seed(&w, &n, seen, lo_ctx, hi_ctx, scale, floor);
                step.at_scale = self.scale;
            }
        }
    }

    /// THE ROW COUNTS A CHOOSER SHOULD ACTUALLY BE OFFERED: each measured class's top and
    /// its OWN cost, with every class a WIDER one beats on price dropped.
    ///
    /// `cost_us` is monotone -- it prices "at least n rows" at the cheapest class that
    /// holds them -- and that is the right PRICE but the wrong OFFER. On LFM2, 40 rows
    /// cost 79.5 ms while 48 rows cost 67.8, so `cost_us(40)` is 67.8; a chooser handed
    /// `(40, 67.8)` would build a 40-row tree and pay 79.5. There is never a reason to
    /// send 40 rows when 48 cost less and accept at least as much, so 40 is not offered
    /// at all.
    ///
    /// Ascending, and empty until something is measured.
    #[must_use]
    pub fn envelope(&self) -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        let mut cheapest = f64::INFINITY;
        for s in self.steps.iter().rev() {
            if let Some(us) = s.us(self.scale, self.at) {
                if us <= cheapest {
                    cheapest = us;
                    out.push((s.hi, us));
                }
            }
        }
        out.reverse();
        out
    }

    /// The row counts worth choosing between: the top of each class.
    #[must_use]
    pub fn candidates(&self) -> Vec<usize> {
        self.steps.iter().map(|s| s.hi).collect()
    }

    #[must_use]
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// How much slower the machine is now than when the table's anchors were taken.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// One entry per class, for the round clock: `lo..hi=ms(samples)`.
    #[must_use]
    pub fn report(&self) -> String {
        let mut out = self
            .steps
            .iter()
            .map(|s| {
                let us = s
                    .us(self.scale, self.at)
                    .map_or_else(|| "-".to_string(), |v| format!("{:.3}", v / 1e3));
                format!("{}..{}={us}({})", s.lo, s.hi, s.seen)
            })
            .collect::<Vec<_>>()
            .join(" ");
        if (self.scale - 1.0).abs() > 0.001 {
            out.push_str(&format!(" scale={:.3}", self.scale));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{PROBES, REFINE_HOLD, VerifyCost, WINDOW};
    use std::time::Duration;

    /// A fresh table over the ladder 2, 4, 8, ... `max`.
    fn table(max: usize) -> VerifyCost {
        VerifyCost::new(max).expect("a ladder")
    }

    /// THE PRICE IS A FITTED QUANTILE NOW, NOT A MEASURED MINIMUM. `cost_model`'s law
    /// answers at the round's context, so a price lands NEAR what was measured rather than
    /// exactly on it -- and it has to, because a law that reproduces one context exactly
    /// cannot also carry the slope to another. These tests therefore assert a band; what is
    /// still exact is the SHAPE: which class answers, the monotone rule, the envelope.
    #[track_caller]
    fn near(got: Option<f64>, want: f64) {
        let got = got.expect("a measured class has a price");
        assert!(
            (got - want).abs() / want < 0.06,
            "priced {got:.0} against {want:.0} ({:+.1}%)",
            (got - want) / want * 100.0
        );
    }

    fn feed(t: &mut VerifyCost, rows: usize, ms: u64, times: usize) {
        for _ in 0..times {
            t.observe(rows, Duration::from_millis(ms));
        }
    }

    /// Rounds of the table's clock -- the fixed cost every round pays.
    fn tick(t: &mut VerifyCost, ms: f64, times: usize) {
        for _ in 0..times {
            t.observe_fixed(Duration::from_secs_f64(ms / 1e3));
        }
    }

    #[test]
    fn the_table_starts_from_the_ladder_and_probes_it_narrow_first() {
        let mut t = table(64);
        assert_eq!(t.candidates(), vec![2, 4, 8, 16, 32, 64]);
        assert_eq!(t.cost_us(2), None);
        assert_eq!(t.probe(), Some(2));
        feed(&mut t, 2, 12, PROBES as usize);
        feed(&mut t, 4, 13, PROBES as usize);
        feed(&mut t, 8, 29, PROBES as usize);
        assert_eq!(t.probe(), Some(16));
        // Every row count up to a measured top is priced by the cheapest step that holds it.
        near(t.cost_us(3), 13_000.0);
        for n in 5..=8 {
            near(t.cost_us(n), 29_000.0);
        }
        assert_eq!(t.cost_us(9), None);
        assert!(VerifyCost::new(1).is_none());
    }

    #[test]
    fn the_table_is_monotone_because_wider_is_sometimes_cheaper() {
        let mut t = table(32);
        feed(&mut t, 16, 79, PROBES as usize);
        feed(&mut t, 32, 67, PROBES as usize);
        // Asking for 16 rows is priced at what sending 32 costs, because that is what a
        // caller would do -- the measured inversion of the 64-token tile against two walks.
        near(t.cost_us(16), 67_000.0);
        near(t.cost_us(32), 67_000.0);
    }

    /// The reason the estimate is a minimum: a mean would carry this spike forever.
    #[test]
    fn one_slow_round_does_not_become_the_estimate() {
        let mut t = table(32);
        t.observe(8, Duration::from_millis(95)); // cold pipeline
        t.observe(8, Duration::from_millis(29));
        t.observe(8, Duration::from_millis(88)); // a browser took the GPU
        near(t.cost_us(8), 29_000.0);
        // A mean of the three would read 70.7 ms, which is not what this width costs.
    }

    /// A SLOWER MACHINE IS THE CLOCK'S JOB, NOT THE LAW'S -- the two timescales, tested.
    ///
    /// ```text
    ///   slow: the context LAW    kernels, tiles, geometry     changes on a retune
    ///   fast: the DRIFT          thermals, other GPU clients  changes in seconds
    /// ```
    ///
    /// The law is deliberately rigid -- it is a property of the kernels and must not chase
    /// every thermal wobble -- so when the machine slows the CLOCK sees it within a window
    /// and carries every class, including ones that have not run since. Before this split
    /// one scalar carried both, and at an 8444-token context it read 1.11 with no drift at
    /// all: it was charging context growth to the machine.
    #[test]
    fn a_slower_machine_is_carried_by_the_clock_not_relearned_by_the_law() {
        let mut t = table(32);
        tick(&mut t, 10.0, WINDOW);
        feed(&mut t, 8, 29, WINDOW);
        near(t.cost_us(8), 29_000.0);

        // Everything slows by 38%: the drafter and the verify alike.
        tick(&mut t, 13.8, 2 * WINDOW);
        feed(&mut t, 8, 40, WINDOW);
        assert!(
            t.scale() > 1.2,
            "the clock missed the slowdown: {}",
            t.scale()
        );
        near(t.cost_us(8), 40_000.0);
    }

    /// THE STALE-CHEAP TRAP. A class measured early, at a short context, must not keep
    /// looking cheap once the machine has slowed under the class that keeps running.
    #[test]
    fn a_class_that_stopped_running_is_carried_by_the_clock() {
        let mut t = table(32);
        // The clock -- the round's fixed cost -- runs every round, classes or no classes.
        tick(&mut t, 10.0, PROBES as usize);
        feed(&mut t, 8, 30, PROBES as usize);
        feed(&mut t, 32, 45, PROBES as usize);
        near(t.cost_us(2), 30_000.0);
        assert!((t.scale() - 1.0).abs() < 0.02, "scale {}", t.scale());
        // The machine gets 50% slower. The clock is what sees it.
        tick(&mut t, 15.0, WINDOW);
        // The drift is a RESIDUAL against the clock's own law, so it lands near 1.5 rather
        // than exactly on it: the law absorbs some of the change as it trains, the drift
        // carries the rest, and between them the price is right.
        let want = 15.0 / 10.0;
        assert!(
            (t.scale() - want).abs() / want < 0.05,
            "scale {} against {want}",
            t.scale()
        );
        // Neither class ran again, and BOTH are carried by the clock's ratio. Their laws
        // are still where they were trained -- a law cannot see drift it never measured --
        // so the clock's ratio is what moves them, which is exactly what it is for.
        near(t.cost_us(2), 30_000.0 * want);
        near(t.cost_us(25), 45_000.0 * want);
        // And the clock reads ITSELF, never through its own scale.
        near(t.fixed_us(), 15_000.0);
    }

    /// THE DEFECT THE CLOCK REPLACED. Two classes each re-anchoring wrote a new GLOBAL
    /// scale from their OWN epoch, so the table swung by the difference between them, over
    /// and over. Here class 8 is steady and class 32 is not; with the old rule class 8's
    /// price moved every time class 32 re-anchored. It must not move at all.
    #[test]
    fn a_running_class_is_not_repriced_by_another_class() {
        let mut t = table(32);
        tick(&mut t, 10.0, PROBES as usize);
        feed(&mut t, 8, 30, PROBES as usize);
        feed(&mut t, 32, 45, PROBES as usize);
        let steady = t.cost_us(2).expect("measured");
        for _ in 0..4 {
            // 32 swings hard; the clock is steady, and 8 keeps measuring what it always did.
            feed(&mut t, 32, 90, WINDOW);
            tick(&mut t, 10.0, WINDOW);
            feed(&mut t, 8, 30, WINDOW);
            feed(&mut t, 32, 45, WINDOW);
            tick(&mut t, 10.0, WINDOW);
            feed(&mut t, 8, 30, WINDOW);
            assert!((t.scale() - 1.0).abs() < 0.02, "scale {}", t.scale());
            let now = t.cost_us(2).expect("measured");
            assert!(
                (now - steady).abs() / steady < 0.02,
                "the 8-row class moved from {steady:.0} to {now:.0} while only 32 swung"
            );
        }
    }

    /// A RESTART KEEPS THE LAW, which is the only way the law can exist at all: one answer
    /// moves the context by a few percent, so the slope is not identifiable inside a single
    /// request. It comes from rounds at different depths, and those only meet in the store.
    #[test]
    fn a_stored_law_survives_a_restart_and_still_rises_with_context() {
        let mut t = table(32);
        // Rounds at three depths, the way a server sees requests -- and a cost that grows
        // with the context, which is what the law is for.
        // Every class needs samples before the table is complete enough to store.
        for _ in 0..40 {
            for (ctx, base) in [(443usize, 29u64), (1596, 30), (8444, 35)] {
                t.set_context(ctx);
                t.observe_fixed(Duration::from_millis(9));
                for (rows, add) in [(8u64, 0u64), (16, 4), (32, 16)] {
                    t.observe(rows as usize, Duration::from_millis(base + add));
                }
            }
        }
        let (rows, laws) = (t.rows(), t.laws());
        assert!(!laws.is_empty(), "nothing to store");

        let mut back = table(32);
        assert!(back.seed(&rows));
        back.seed_laws(&laws);
        // The restored law prices at the round's own context, and it RISES with it.
        back.set_context(443);
        let near_ctx = back.cost_us(8).expect("restored");
        back.set_context(8444);
        let far_ctx = back.cost_us(8).expect("restored");
        assert!(
            far_ctx > near_ctx * 1.05,
            "the restored law is flat: {near_ctx:.0} at 443 vs {far_ctx:.0} at 8444"
        );
    }

    /// One broken measurement must not rescale the whole table.
    #[test]
    fn the_scale_moves_by_bounded_steps() {
        let mut t = table(32);
        tick(&mut t, 10.0, WINDOW);
        tick(&mut t, 1_000.0, WINDOW); // 100x rounds: a stall, not drift
        assert!((t.scale() - 2.0).abs() < 1e-9, "scale {}", t.scale());
    }

    /// A restart starts where the last run finished.
    #[test]
    fn a_stored_table_seeds_a_restart() {
        let mut t = table(32);
        for (rows, ms) in [(2, 28), (4, 28), (8, 29), (16, 33), (32, 45)] {
            feed(&mut t, rows, ms, PROBES as usize);
        }
        // A table with no fixed cost is not complete: the budget would run on design 6.3's
        // ratio, which is the thing this table exists to correct. It is stored all the same,
        // so a restart does not measure the five widths again.
        assert_eq!(t.probe(), Some(2), "the fixed cost still needs its samples");
        assert_eq!(
            t.rows().len(),
            5,
            "the measured classes are stored before the clock"
        );
        for _ in 0..PROBES {
            t.observe_fixed(Duration::from_millis(9));
        }
        let stored = t.rows();
        assert_eq!(stored.len(), 6, "five classes and the round's fixed cost");

        let mut fresh = table(32);
        assert!(
            fresh.rows().is_empty(),
            "a table with no samples stores nothing"
        );
        assert!(fresh.seed(&stored));
        assert_eq!(fresh.probe(), None, "a seeded table needs no probe rounds");
        near(fresh.cost_us(2), 28_000.0);
        near(fresh.cost_us(24), 45_000.0);
        near(fresh.fixed_us(), 9_000.0);
    }

    /// A probe cut off by the end of a request RESUMES after a restart: the classes it finished
    /// are not probed again, and the class it was inside finishes its own count.
    #[test]
    fn a_half_probed_table_resumes_its_probe_after_a_restart() {
        let mut t = table(32);
        feed(&mut t, 2, 29, PROBES as usize);
        feed(&mut t, 4, 30, 1);
        assert_eq!(t.probe(), Some(4));
        let stored = t.rows();
        assert_eq!(
            stored.len(),
            2,
            "the finished class and the one in progress"
        );

        let mut fresh = table(32);
        assert!(fresh.seed(&stored));
        let mut probed = Vec::new();
        while let Some(rows) = fresh.probe() {
            probed.push(rows);
            match rows {
                // the clock's own probe runs at the narrowest width: its rounds time the fixed cost
                2 => fresh.observe_fixed(Duration::from_millis(9)),
                w => feed(&mut fresh, w, 30 + w as u64, 1),
            }
            assert!(probed.len() < 30, "the probe did not end: {probed:?}");
        }
        let mut want = vec![4; PROBES as usize - 1];
        for w in [8, 16, 32] {
            want.extend(vec![w; PROBES as usize]);
        }
        want.extend(vec![2; PROBES as usize]);
        assert_eq!(
            probed, want,
            "the 2-row class was probed again, or 4 started over"
        );
        near(fresh.cost_us(2), 29_000.0);
    }

    /// A class a wider one beats on price is not offered at all: sending its rows would
    /// pay ITS cost, not the cheaper one the monotone price quotes.
    #[test]
    fn the_envelope_drops_a_class_a_wider_one_beats() {
        let mut t = table(64);
        feed(&mut t, 8, 29, PROBES as usize);
        feed(&mut t, 16, 33, PROBES as usize);
        feed(&mut t, 32, 79, PROBES as usize); // the 64-token tile's inversion
        feed(&mut t, 64, 67, PROBES as usize);
        // The SHAPE is exact -- which classes survive -- and the prices are the law's.
        let offered: Vec<usize> = t.envelope().iter().map(|&(n, _)| n).collect();
        assert_eq!(
            offered,
            vec![8, 16, 64],
            "32 rows at 79 ms is beaten by 64 rows at 67"
        );
        for (n, want) in [(8, 29_000.0), (16, 33_000.0), (64, 67_000.0)] {
            near(
                t.envelope()
                    .iter()
                    .find(|&&(k, _)| k == n)
                    .map(|&(_, us)| us),
                want,
            );
        }
        // The monotone PRICE still quotes the cheaper class, which is what it is for.
        near(t.cost_us(24), 67_000.0);
    }

    /// A table with every ladder width priced from `ms(width)` and its clock ticked, so `probe`
    /// has nothing left to ask for.
    fn learned_full(ms: impl Fn(usize) -> u64) -> VerifyCost {
        let mut t = table(64);
        for w in t.candidates() {
            feed(&mut t, w, ms(w), PROBES as usize);
        }
        tick(&mut t, 9.0, PROBES as usize);
        assert_eq!(t.probe(), None);
        t
    }

    /// A value model for the gate: `tokens(rows)` expected tokens, a fixed 10 ms per round, and
    /// the long-run rate `rho` tokens per ms. `could_beat` is the chooser's bar: the value must
    /// clear the held width's by 4% of its tokens.
    fn gate<F: Fn(usize) -> f64>(
        t: &VerifyCost,
        held: usize,
        tokens: F,
    ) -> impl Fn(usize, f64) -> bool + use<F> {
        let rho = 0.15;
        let held_ms = t.cost_us(held).expect("held width priced") / 1e3;
        let held_v = tokens(held) - rho * (10.0 + held_ms);
        let bar = held_v + 0.04 * tokens(held);
        move |rows, us| tokens(rows) - rho * (10.0 + us / 1e3) > bar
    }

    /// Holds `width` for `rounds` rounds under the value model `tokens` and returns the last split.
    fn hold(
        t: &mut VerifyCost,
        width: usize,
        rounds: u64,
        tokens: &dyn Fn(usize) -> f64,
    ) -> Option<usize> {
        (0..rounds)
            .filter_map(|_| {
                let g = gate(t, width, tokens);
                t.refine(width, g)
            })
            .last()
    }

    /// Dense: a tree's tokens saturate, 0.5 per row to 8 rows, 0.02 per row after.
    fn saturating(rows: usize) -> f64 {
        1.0 + 0.5 * rows.min(8) as f64 + 0.02 * rows.saturating_sub(8) as f64
    }

    /// Routed: every row keeps adding tokens.
    fn linear(rows: usize) -> f64 {
        1.0 + 0.5 * rows as f64
    }

    #[test]
    fn a_step_no_width_inside_could_win_is_never_probed() {
        // The 4-bit shape: flat to 8 rows, then a 50% step -- and past 8 rows the tree's tokens
        // barely grow, so no width in 8..16 can pay for more time than 8 rows take.
        let mut t = learned_full(|w| if w <= 8 { 26 } else { 40 });
        assert_eq!(hold(&mut t, 8, 4 * REFINE_HOLD, &saturating), None);
        assert_eq!(t.candidates(), vec![2, 4, 8, 16, 32, 64]);
        // The 8-bit shape held at 16: 31 rows at 16's price would pay, so the gate asks once --
        // beside the held width, because the step is steep -- and 17 costing the whole step
        // settles the interval.
        let mut t = learned_full(|w| if w <= 16 { 25 } else { 45 });
        assert_eq!(hold(&mut t, 16, REFINE_HOLD + 1, &saturating), Some(17));
        feed(&mut t, 17, 45, PROBES as usize);
        assert_eq!(hold(&mut t, 16, 4 * REFINE_HOLD, &saturating), None);
    }

    #[test]
    fn a_step_a_width_could_win_is_split_beside_the_held_width() {
        // Every row adds tokens, and 16 rows cost 50% more than 8: a width inside might pay.
        let mut t = learned_full(|w| if w <= 8 { 26 } else { 40 });
        assert_eq!(hold(&mut t, 8, REFINE_HOLD, &linear), None);
        assert_eq!(hold(&mut t, 8, 2, &linear), Some(9));
        assert_eq!(t.probe(), Some(9));
        // Probing blocks the next split.
        assert_eq!(hold(&mut t, 8, 3 * REFINE_HOLD, &linear), None);
    }

    #[test]
    fn a_gradual_rise_is_mapped_by_halving() {
        // A routed target: every row adds about the same time and the same tokens.
        let ms = |w: usize| 10 + w as u64;
        let mut t = learned_full(ms);
        let mut added = Vec::new();
        for _ in 0..3 {
            let m = hold(&mut t, 8, REFINE_HOLD + 1, &linear).expect("a split");
            feed(&mut t, m, ms(m), PROBES as usize);
            added.push(m);
        }
        // Beside the held width first, then the middle of what is left.
        assert_eq!(added, vec![9, 12, 10]);
    }

    #[test]
    fn nothing_is_split_while_the_chooser_moves() {
        let mut t = learned_full(|w| 10 + w as u64);
        for r in 0..(4 * REFINE_HOLD) {
            let w = if r % 2 == 0 { 8 } else { 16 };
            let g = gate(&t, w, linear);
            assert_eq!(t.refine(w, g), None);
        }
    }

    #[test]
    fn a_step_is_priced_by_its_top_only() {
        let mut t = table(64);
        // 10 rows is inside the 9..16 step; its time says nothing certain about 16 rows.
        feed(&mut t, 10, 20, PROBES as usize);
        assert!(t.steps().iter().all(|s| s.seen == 0));
        feed(&mut t, 16, 30, 1);
        assert_eq!(t.steps().iter().find(|s| s.hi == 16).map(|s| s.seen), Some(1));
    }

    #[test]
    fn the_widest_step_learns_how_wide_a_tree_can_be() {
        let mut t = table(64);
        for w in [2, 4, 8, 16, 32] {
            feed(&mut t, w, 10 + w as u64, PROBES as usize);
        }
        tick(&mut t, 9.0, PROBES as usize);
        // Asked for 64; a drafter of 7 positions and 8 candidates builds 57.
        assert_eq!(t.probe(), Some(64));
        t.unreached(64, 57);
        feed(&mut t, 57, 70, 1);
        assert_eq!(t.candidates(), vec![2, 4, 8, 16, 32, 57]);
        assert_eq!(t.probe(), Some(57));
        feed(&mut t, 57, 70, PROBES as usize - 1);
        assert_eq!(t.probe(), None);
        near(t.cost_us(57), 70_000.0);
    }

    #[test]
    fn a_middle_top_no_tree_reaches_stops_being_probed() {
        let mut t = table(64);
        for w in [2, 4, 8] {
            feed(&mut t, w, 10 + w as u64, PROBES as usize);
        }
        // Asked for 16, the tree held 12, every time: 16 is marked and skipped.
        for _ in 0..PROBES {
            assert_eq!(t.probe(), Some(16));
            t.unreached(16, 12);
            feed(&mut t, 12, 30, 1);
        }
        assert_eq!(t.probe(), Some(32));
    }

    /// The failure a restart made: the stored table ended at 57 and the rebuilt ladder at 64, and
    /// the seed left a 58..64 step no tree reaches, probed every round. The stored reach is kept.
    #[test]
    fn a_restarted_table_keeps_the_reach_it_learned() {
        let mut t = table(64);
        for w in [2, 4, 8, 16, 32] {
            feed(&mut t, w, 10 + w as u64, PROBES as usize);
        }
        tick(&mut t, 9.0, PROBES as usize);
        t.unreached(64, 57);
        feed(&mut t, 57, 70, PROBES as usize);
        assert_eq!(t.probe(), None);
        let rows = t.rows();
        let mut restarted = table(64);
        assert!(restarted.seed(&rows));
        assert_eq!(restarted.candidates(), vec![2, 4, 8, 16, 32, 57]);
        assert_eq!(restarted.probe(), None);
        // A table stored mid-probe, its widest stored top a ladder width, keeps the rest to probe.
        let mut early = table(64);
        feed(&mut early, 2, 12, PROBES as usize);
        let mut again = table(64);
        assert!(again.seed(&early.rows()));
        assert_eq!(again.candidates(), vec![2, 4, 8, 16, 32, 64]);
    }

    #[test]
    fn a_stored_table_restores_its_splits() {
        let mut t = learned_full(|w| if w <= 8 { 26 } else { 40 });
        assert_eq!(hold(&mut t, 8, REFINE_HOLD + 1, &linear), Some(9));
        feed(&mut t, 9, 40, PROBES as usize);
        let rows = t.rows();
        let mut restarted = table(64);
        assert!(restarted.seed(&rows));
        assert_eq!(restarted.candidates(), vec![2, 4, 8, 9, 16, 32, 64]);
        // A row the ladder cannot hold is refused, and the refusal changes nothing.
        let mut other = table(64);
        assert!(!other.seed(&[(5, 70, 1000.0, 3)]));
        assert_eq!(other.candidates(), vec![2, 4, 8, 16, 32, 64]);
        // So is a file from a build whose widest verify was wider: its widest step does not fit.
        let mut narrower = table(48);
        assert!(!narrower.seed(&rows));
        assert_eq!(narrower.candidates(), vec![2, 4, 8, 16, 32, 48]);
    }
}

