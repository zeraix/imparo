//! DESIGN 6.6'S ACCEPTANCE MODEL: how likely each of the drafter's candidates is to be the target's
//! pick at its parent -- the value side's twin of the cost model.
//!
//! # The form
//!
//! A parent's children exclude each other: at most one is the target's pick. So the candidates of one
//! parent and a NONE outcome share one softmax, and their probabilities cannot sum past 1:
//!
//! ```text
//!   P(c_j | parent accepted) = exp(s_j) / (1 + sum_k exp(s_k))      P(NONE) = 1 / (1 + sum_k exp(s_k))
//!   pick      s = a_p + b_p logit(head) + c_p logit(pick share)                            + t + b
//!   sibling   s = a_s + b_s log r        + c_s logit(head)       + d_s logit(pick share)   + t + b
//!   n-gram    s = logit(the source's precision) + w_new[bucket]                            + t + b
//!   agreed    a pick or sibling the n-gram also proposed:
//!             its drafter score + max(0, logit(the source's precision)) + w_agreed[kind, bucket]
//!   t         depth coefficient x (depth - 1) + context coefficient x context / 8192, every node
//! ```
//!
//! `head` is the drafter's confidence head at the position, a share is a candidate's softmax share
//! among the K, and `r` a sibling's share of the siblings' total. `b` is the request's intercept.
//!
//! # The n-gram's kinds (design 6.6)
//!
//! A bucket is `source (request index | stored table) x follower count (1 | 2 | 3-4 | 5+) x n-gram
//! tokens directly above the node on its path (0 | 1-2 | 3+)`. Every bucket coefficient starts at 0, so
//! each kind starts at design 6.6's prior, the source's top-1 precision `p` measured on the committed
//! stream:
//!
//! ```text
//!   n-gram only   logit(p): the only estimate that exists for a token the drafter did not offer
//!   agreed        the drafter's score + max(0, logit(p)). Agreement multiplies the odds by
//!                 P(agree | the node is the pick) / P(agree | it is not) = p / ((1 - p) c), where c is
//!                 how often a wrong follower lands on this very candidate; c <= 1, so logit(p) is the
//!                 ratio's lower bound. The floor keeps a weak source from counting against a token it
//!                 agrees with.
//! ```
//!
//! MEASURED before the floor and the rate below were set (one conversation's two turns, LFM2.5):
//! agreed picks realised 78 of 78 where a start at the drafter's own estimate predicted 0.85-0.95,
//! and no bucket weight moved past 0.005 in a request at the drafter coefficients' rate.
//!
//! # Why both of the drafter's signals
//!
//! Measured on 80 held-out prompts: where the head and the pick's share disagree, the SHARE is right
//! (head 0.34 / share 0.94: the pick was accepted 0.75 of the time; head 0.70 / share 0.43: 0.40). A
//! prior derived from the head alone was worse than today's estimate; this form, fitted by maximum
//! likelihood, cut the per-node log loss 14% on prompts it never saw (evidence:
//! 2026-09-17-acceptance-prior-discard-review.md).
//!
//! # How it learns
//!
//! ```text
//!   the prior     the seven coefficients, fitted offline on measured rounds, or derived when none exist
//!   per round     every candidate of every accepted parent is labelled, placed or not: FTRL-Proximal on
//!                 the deviation from the prior, a coordinate leaving the prior only when its
//!                 standardized score passes GATE; the request's intercept by AdaGrad
//!   the guard     before a round's labels update anything, this model and today's estimate both score
//!                 them; the model prices the tree only while its decayed log loss is the lower
//! ```

/// A probability is clipped to `[HEAD_EPS, 1 - HEAD_EPS]` before its logit, which bounds every logit
/// feature at `LOGIT_CLIP`. Past it the head says nothing the clip loses.
const HEAD_EPS: f64 = 1e-7;

/// A sibling's log share of the siblings' total floors here: a share of about 1e-13 of that total.
const LOG_R_FLOOR: f64 = -30.0;

/// The drafter's coefficients: `[a_p, b_p, c_p, a_s, b_s, c_s, d_s]`.
pub const DRAFTER_COEFS: usize = 7;

/// Every node's depth and context terms.
const DEPTH: usize = 7;
const CONTEXT: usize = 8;

/// An n-gram bucket: source (2) x follower count (4) x run above (3).
pub const BUCKETS: usize = 24;

/// Where the n-gram-only buckets start, and the agreed ones: a pick's, then a sibling's.
const NGRAM: usize = 9;
const AGREED: usize = NGRAM + BUCKETS;

/// N-gram evidence slots: source (2) x evidence (4), `NgramTerms::slot`.
pub const SLOTS: usize = 8;

/// Where the contradiction buckets start: one per evidence slot, a pick's, then a sibling's.
const CONTRA: usize = AGREED + 2 * BUCKETS;

/// Every coefficient: the drafter's, depth, context, the n-gram-only buckets, the agreed buckets, the
/// contradiction buckets.
pub const COEFS: usize = CONTRA + 2 * SLOTS;

/// The context term's unit, in tokens: the cost laws' (`cost_model`).
const CONTEXT_UNIT: f64 = 8192.0;

/// The most terms one node carries: a sibling's four, depth, context, and an agreed or a contradiction
/// bucket (never both: the n-gram proposes one token at a parent).
const MAX_TERMS: usize = 7;

/// FTRL-Proximal's rate and stabiliser, as the cost laws use them (`cost_model`).
const ALPHA: f64 = 0.05;
const BETA: f64 = 1.0;

/// The n-gram buckets' rate. A bucket sees 1-40 labels a request where the drafter's coefficients see
/// hundreds, and starts from a cruder prior; at `ALPHA` a bucket with 30 labels that all disagree with
/// it by 0.13 moves about 0.1 logit. Unmeasured choice; the gate is what judges it.
const NGRAM_ALPHA: f64 = 0.5;

/// A coefficient's FTRL rate.
fn alpha(coefficient: usize) -> f64 {
    if coefficient >= NGRAM {
        NGRAM_ALPHA
    } else {
        ALPHA
    }
}

/// L2 on the deviation from the prior.
const L2: f64 = 0.01;

/// A coefficient leaves its prior only once `|z| > GATE sqrt(n)`: its accumulated gradient is GATE
/// standard deviations from zero. Unmeasured choice; the gate is what judges it.
const GATE: f64 = 3.0;

/// The squared-gradient sum is capped, which floors the step so the model can still follow a
/// workload that changes after thousands of labels (design 6.6, theory item 4).
const N_CAP: f64 = 400.0;

/// The request intercept's AdaGrad rate, band and squared-gradient cap. The rate and band are the
/// offset's (`accept_offset`); the cap keeps the step from freezing inside a long request.
const B_ALPHA: f64 = 0.15;
const B_BAND: f64 = 2.0;
const B_G2_CAP: f64 = 16.0;

/// The guard forgets a labelled parent's log loss with this half-life, in parents.
const GUARD_HALF_LIFE: f64 = 256.0;

fn logit_clip() -> f64 {
    ((1.0 - HEAD_EPS) / HEAD_EPS).ln()
}

/// A probability's logit, clipped like the head's.
fn logit(p: f64) -> f64 {
    let p = p.clamp(HEAD_EPS, 1.0 - HEAD_EPS);
    (p / (1.0 - p)).ln()
}

/// What a candidate is to this model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The drafter's pick at its position.
    Pick,
    /// Another of the drafter's K at the same position.
    Sibling,
    /// A token only the n-gram proposed.
    Ngram,
}

/// Which n-gram table answered a lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The request's own index (level 1).
    Request,
    /// The stored table, read when the request's index holds nothing for the context (level 2).
    Table,
}

/// What the n-gram said about a node: which table, how often the follower was seen after the
/// context, how long a copy match it came from, and how many n-gram tokens sit directly above the
/// node on its path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NgramTerms {
    pub source: Source,
    pub count: u32,
    /// The copy match's length in tokens (`NgramIndex::longest`), 0 when the 3-token index answered.
    pub matched: u32,
    pub run: u32,
}

/// A copy match this long or longer is its own evidence class: a copy in progress, as good as
/// certain. Shorter ones share a class with each other.
pub const LONG_MATCH: u32 = 16;

impl NgramTerms {
    /// `source x evidence x run (0 | 1-2 | 3+)`, in `0..BUCKETS`. The request's evidence is a copy
    /// match's length when it has one (`MATCH_KEY`-15 | 16+) and otherwise the 3-token index's
    /// count (1 | 2+); the stored table never copies, so its evidence is its count (1 | 2 | 3-4 |
    /// 5+).
    #[must_use]
    pub fn bucket(&self) -> usize {
        (self.source_index() * 4 + self.evidence()) * 3 + self.run_index()
    }

    /// Which precision slot this node's estimate reads: `source * 4 + evidence`, in `0..8`.
    #[must_use]
    pub fn slot(&self) -> usize {
        self.source_index() * 4 + self.evidence()
    }

    fn source_index(&self) -> usize {
        match self.source {
            Source::Request => 0,
            Source::Table => 1,
        }
    }

    fn evidence(&self) -> usize {
        match self.source {
            Source::Request if self.matched >= LONG_MATCH => 3,
            Source::Request if self.matched > 0 => 2,
            Source::Request => usize::from(self.count > 1),
            Source::Table => match self.count {
                0 | 1 => 0,
                2 => 1,
                3 | 4 => 2,
                _ => 3,
            },
        }
    }

    fn run_index(&self) -> usize {
        match self.run {
            0 => 0,
            1 | 2 => 1,
            _ => 2,
        }
    }
}

/// One candidate's inputs.
#[derive(Clone, Copy, Debug)]
pub struct Features {
    pub kind: Kind,
    pub logit_head: f64,
    pub logit_share_pick: f64,
    /// `log r` for a sibling; 0 otherwise.
    pub log_r: f64,
    /// The node's depth below the anchor, minus one: 0 for a child of the anchor.
    pub depth: u32,
    /// The round's context, in tokens.
    pub context: usize,
    /// An n-gram node's bucket, or the bucket of the n-gram that agreed with a drafter node.
    pub ngram: Option<NgramTerms>,
    /// For a drafter node: the n-gram proposed ANOTHER token at its parent, with this evidence.
    pub contra: Option<NgramTerms>,
    /// What the n-gram adds before any bucket has learned: `logit(p)` for an n-gram-only node,
    /// `max(0, logit(p))` for an agreed one, `-max(0, logit(p))` for a contradicted one, `p` the
    /// evidence slot's measured top-1 precision; 0 otherwise.
    pub base: f64,
}

impl Features {
    /// A token only the n-gram proposed, at `depth` below the anchor (minus one), in a round at
    /// `context`, from a source whose measured top-1 precision is `precision`.
    #[must_use]
    pub fn ngram(
        terms: NgramTerms,
        precision: f64,
        depth: u32,
        context: usize,
    ) -> Self {
        Self {
            kind: Kind::Ngram,
            logit_head: 0.0,
            logit_share_pick: 0.0,
            log_r: 0.0,
            depth,
            context,
            ngram: Some(terms),
            contra: None,
            base: logit(precision),
        }
    }

    /// This drafter candidate with the n-gram agreeing: its bucket, and the floored start.
    #[must_use]
    pub fn agreed(self, terms: NgramTerms, precision: f64) -> Self {
        Self {
            ngram: Some(terms),
            contra: None,
            base: logit(precision).max(0.0),
            ..self
        }
    }

    /// THIS DRAFTER CANDIDATE, CONTRADICTED: the n-gram proposed another token at its parent. Agreement
    /// is evidence FOR a node, worth `logit(p)` in the independence approximation; a different
    /// proposal is the same evidence against it, so the start is its negative, floored the same way --
    /// a proposal right less than half the time says nothing against the drafter. MEASURED before
    /// adding it (copy chains, two real-use turns): a pick whose parent held another n-gram proposal
    /// was accepted 0.539 of the time against 0.678 without one, and the model predicted 0.598. An
    /// agreed node stays agreed.
    #[must_use]
    pub fn contradicted(self, terms: NgramTerms, precision: f64) -> Self {
        if self.ngram.is_some() || self.kind == Kind::Ngram {
            return self;
        }
        Self {
            contra: Some(terms),
            base: -logit(precision).max(0.0),
            ..self
        }
    }

    /// One drafter position: its confidence head and its K biased values, the pick's first and the
    /// largest. The pick's share keeps its precision because its logit is taken from the siblings'
    /// total -- `-ln(sum of exp(v_j - v_pick))`, by log-sum-exp -- never from 1 minus a share near 1.
    /// Depth and context are 0 here; the caller that knows them sets them.
    #[must_use]
    pub fn position(confidence: f32, values: &[f32]) -> Vec<Self> {
        let clip = logit_clip();
        let Some((&top, rest)) = values.split_first() else {
            return Vec::new();
        };
        let top = f64::from(top);
        let rel: Vec<f64> = rest.iter().map(|&v| f64::from(v) - top).collect();
        let m = rel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let ln_tail = if m.is_finite() {
            m + rel.iter().map(|r| (r - m).exp()).sum::<f64>().ln()
        } else {
            f64::NEG_INFINITY
        };
        let logit_share_pick = (-ln_tail).clamp(-clip, clip);
        let h = f64::from(confidence).clamp(HEAD_EPS, 1.0 - HEAD_EPS);
        let logit_head = (h / (1.0 - h)).ln();
        let drafted = |kind, log_r| Self {
            kind,
            logit_head,
            logit_share_pick,
            log_r,
            depth: 0,
            context: 0,
            ngram: None,
            contra: None,
            base: 0.0,
        };
        let mut out = Vec::with_capacity(values.len());
        out.push(drafted(Kind::Pick, 0.0));
        out.extend(
            rel.iter()
                .map(|&r| drafted(Kind::Sibling, (r - ln_tail).max(LOG_R_FLOOR))),
        );
        out
    }

    /// This candidate's nonzero terms against the coefficient layout, `(coefficient, value)`, and
    /// how many there are. Sparse: a node touches at most `MAX_TERMS` of the `COEFS` coefficients.
    fn terms(&self) -> ([(usize, f64); MAX_TERMS], usize) {
        let mut t = [(0_usize, 0.0_f64); MAX_TERMS];
        let mut n = 0;
        let mut put = |i: usize, x: f64| {
            t[n] = (i, x);
            n += 1;
        };
        match self.kind {
            Kind::Pick => {
                put(0, 1.0);
                put(1, self.logit_head);
                put(2, self.logit_share_pick);
            }
            Kind::Sibling => {
                put(3, 1.0);
                put(4, self.log_r);
                put(5, self.logit_head);
                put(6, self.logit_share_pick);
            }
            Kind::Ngram => {}
        }
        if let Some(terms) = self.ngram {
            match self.kind {
                Kind::Ngram => put(NGRAM + terms.bucket(), 1.0),
                Kind::Pick => put(AGREED + terms.bucket(), 1.0),
                Kind::Sibling => put(AGREED + BUCKETS + terms.bucket(), 1.0),
            }
        } else if let Some(terms) = self.contra {
            match self.kind {
                Kind::Pick => put(CONTRA + terms.slot(), 1.0),
                Kind::Sibling => put(CONTRA + SLOTS + terms.slot(), 1.0),
                Kind::Ngram => {}
            }
        }
        // THE DEPTH TERM IS THE DRAFTER'S. Its slope is learned from drafter nodes, whose
        // acceptance decays with depth; a copy's does not -- a chain 20 tokens into a copied
        // passage is as likely as one 2 tokens in -- and its own run bucket carries where it sits.
        if self.kind != Kind::Ngram {
            put(DEPTH, f64::from(self.depth));
        }
        #[allow(clippy::cast_precision_loss)]
        put(CONTEXT, self.context as f64 / CONTEXT_UNIT);
        (t, n)
    }

    /// The coefficient of this node's n-gram bucket, when it has one: its own, its agreement's, or its
    /// contradiction's.
    #[must_use]
    pub fn bucket_coefficient(&self) -> Option<usize> {
        self.ngram
            .map(|terms| match self.kind {
                Kind::Ngram => NGRAM + terms.bucket(),
                Kind::Pick => AGREED + terms.bucket(),
                Kind::Sibling => AGREED + BUCKETS + terms.bucket(),
            })
            .or_else(|| {
                self.contra.and_then(|terms| match self.kind {
                    Kind::Pick => Some(CONTRA + terms.slot()),
                    Kind::Sibling => Some(CONTRA + SLOTS + terms.slot()),
                    Kind::Ngram => None,
                })
            })
    }
}

/// FTRL-Proximal's weight from its state: the prior, moved by the accumulated gradient `z` only
/// once `|z|` passes the standardized gate `GATE sqrt(n)`.
fn weight(prior: f64, z: f64, n: f64, alpha: f64) -> f64 {
    let gate = GATE * n.sqrt();
    if z.abs() <= gate {
        prior
    } else {
        prior - (z - z.signum() * gate) / ((BETA + n.sqrt()) / alpha + L2)
    }
}

/// DESIGN 11.2'S LEVEL 2 of this model: everything it learned that outlives a request. The
/// request's intercept and its report sums are level 1 and are not in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Level2 {
    pub fitted: bool,
    pub seen: u64,
    pub guard: (f64, f64),
    pub prior: [f64; COEFS],
    pub z: [f64; COEFS],
    pub n: [f64; COEFS],
    /// Labelled nodes that carried each coefficient's term: what the exploration row reads to find
    /// the n-gram bucket with the least evidence.
    pub labels: [u64; COEFS],
}

/// Per-node binary log loss, both estimates scored the same way.
fn node_loss(q: f64, accepted: bool) -> f64 {
    let q = q.clamp(1e-12, 1.0 - 1e-12);
    if accepted { -q.ln() } else { -(1.0 - q).ln() }
}

/// One n-gram bucket's request so far: labelled nodes, the model's probabilities summed over them,
/// and how many were the target's pick. Predicted against realised, per bucket.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BucketReport {
    pub labels: u64,
    pub predicted: f64,
    pub accepted: u64,
}

/// The model: level-2 coefficients learned across requests, a level-1 intercept, and the guard.
#[derive(Clone, Debug)]
pub struct AcceptModel {
    prior: [f64; COEFS],
    /// Whether the prior was fitted on measured rounds. A fitted prior prices from the first round;
    /// a derived one only after it has beaten today's estimate on labels.
    fitted: bool,
    w: [f64; COEFS],
    z: [f64; COEFS],
    n: [f64; COEFS],
    labels: [u64; COEFS],
    b: f64,
    b_g2: f64,
    /// Decayed per-node log loss over labelled parents: this model, today's estimate.
    guard: (f64, f64),
    /// This request's undecayed sums: this model, today's estimate, labelled nodes.
    request: (f64, f64, u64),
    /// This request's per-bucket calibration, indexed by coefficient; only bucket coefficients fill.
    buckets: [BucketReport; COEFS],
    /// Labelled parents folded in since the prior was set.
    seen: u64,
}

impl AcceptModel {
    /// A model whose drafter coefficients were fitted on measured rounds. Depth, context and every
    /// n-gram bucket start at 0.
    #[must_use]
    pub fn fitted(drafter: [f64; DRAFTER_COEFS]) -> Self {
        Self::start(drafter, true)
    }

    /// A model starting from the derived coefficients (design 6.6): the head's logit for the pick,
    /// `log r` for a sibling, intercepts from the drafter's coverage `kappa` of the top-K. It prices
    /// nothing until the guard says it has beaten today's estimate.
    #[must_use]
    pub fn derived(kappa: f64) -> Self {
        let kappa = kappa.clamp(1e-3, 1.0 - 1e-3);
        Self::start(
            [
                -(1.0 - kappa).ln(),
                1.0,
                0.0,
                (kappa / (1.0 - kappa)).ln(),
                1.0,
                0.0,
                0.0,
            ],
            false,
        )
    }

    fn start(drafter: [f64; DRAFTER_COEFS], fitted: bool) -> Self {
        let mut prior = [0.0; COEFS];
        prior[..DRAFTER_COEFS].copy_from_slice(&drafter);
        Self {
            prior,
            fitted,
            w: prior,
            z: [0.0; COEFS],
            n: [0.0; COEFS],
            labels: [0; COEFS],
            b: 0.0,
            b_g2: 0.0,
            guard: (0.0, 0.0),
            request: (0.0, 0.0, 0),
            buckets: [BucketReport::default(); COEFS],
            seen: 0,
        }
    }

    /// Everything this model learned that outlives the request, for the store.
    #[must_use]
    pub fn level2(&self) -> Level2 {
        Level2 {
            fitted: self.fitted,
            seen: self.seen,
            guard: self.guard,
            prior: self.prior,
            z: self.z,
            n: self.n,
            labels: self.labels,
        }
    }

    /// The model a stored `level2` describes, at the start of a request. Learning continues
    /// exactly where it stopped: the weights are recomputed from the prior, `z` and `n` by the
    /// same rule the learner uses.
    #[must_use]
    pub fn resume(state: &Level2) -> Self {
        let mut w = state.prior;
        for (i, wi) in w.iter_mut().enumerate() {
            *wi = weight(state.prior[i], state.z[i], state.n[i], alpha(i));
        }
        Self {
            prior: state.prior,
            fitted: state.fitted,
            w,
            z: state.z,
            n: state.n,
            labels: state.labels,
            b: 0.0,
            b_g2: 0.0,
            guard: state.guard,
            request: (0.0, 0.0, 0),
            buckets: [BucketReport::default(); COEFS],
            seen: state.seen,
        }
    }

    /// Whether this model prices the tree now.
    #[must_use]
    pub fn prices(&self) -> bool {
        if self.fitted {
            self.guard.0 <= self.guard.1
        } else {
            self.guard.0 < self.guard.1
        }
    }

    /// The current coefficients and the labelled parents behind them, for the store.
    #[must_use]
    pub fn state(&self) -> ([f64; COEFS], u64) {
        (self.w, self.seen)
    }

    /// Labelled nodes behind a coefficient.
    #[must_use]
    pub fn labels(&self, coefficient: usize) -> u64 {
        self.labels.get(coefficient).copied().unwrap_or(0)
    }

    fn score(&self, f: &Features) -> f64 {
        let (terms, n) = f.terms();
        terms[..n].iter().map(|&(i, x)| self.w[i] * x).sum::<f64>() + f.base + self.b
    }

    /// The probability of each candidate of ONE parent.
    #[must_use]
    pub fn probabilities(&self, group: &[Features]) -> Vec<f64> {
        let s: Vec<f64> = group.iter().map(|f| self.score(f)).collect();
        let m = s.iter().copied().fold(0.0_f64, f64::max);
        let lse = m + ((-m).exp() + s.iter().map(|x| (x - m).exp()).sum::<f64>()).ln();
        s.iter().map(|x| (x - lse).exp()).collect()
    }

    /// One labelled parent: its candidates, today's estimate for each (after today's offset), and
    /// which candidate was the target's pick (`None`: none of them). Scores both estimates for the
    /// guard BEFORE learning from the label.
    pub fn observe(&mut self, group: &[Features], today: &[f64], pick: Option<usize>) {
        if group.is_empty() {
            return;
        }
        let p = self.probabilities(group);
        let (mut lm, mut lt) = (0.0, 0.0);
        for (j, (&pm, &pt)) in p.iter().zip(today).enumerate() {
            lm += node_loss(pm, pick == Some(j));
            lt += node_loss(pt, pick == Some(j));
        }
        let keep = 0.5_f64.powf(1.0 / GUARD_HALF_LIFE);
        self.guard = (self.guard.0 * keep + lm, self.guard.1 * keep + lt);
        self.request.0 += lm;
        self.request.1 += lt;
        self.request.2 += group.len() as u64;
        self.seen += 1;

        // The multinomial's gradient on each score is P - y; NONE has no parameters. A coefficient
        // no candidate touches has a zero gradient, and FTRL's step on a zero gradient changes
        // nothing, so only the touched ones are stepped.
        let mut g = [0.0_f64; COEFS];
        let mut touched = [false; COEFS];
        let mut gb = 0.0;
        for (j, (f, &pj)) in group.iter().zip(&p).enumerate() {
            let y = pick == Some(j);
            let d = pj - f64::from(u8::from(y));
            gb += d;
            let (terms, n) = f.terms();
            for &(i, x) in &terms[..n] {
                g[i] += d * x;
                touched[i] = true;
                if x != 0.0 {
                    self.labels[i] += 1;
                }
            }
            if let Some(c) = f.bucket_coefficient() {
                let r = &mut self.buckets[c];
                r.labels += 1;
                r.predicted += pj;
                r.accepted += u64::from(y);
            }
        }
        for i in 0..COEFS {
            if touched[i] {
                self.ftrl(i, g[i]);
            }
        }
        if gb.is_finite() {
            self.b_g2 = (self.b_g2 + gb * gb).min(B_G2_CAP);
            self.b = (self.b - B_ALPHA * gb / (1.0 + self.b_g2.sqrt()))
                .clamp(-B_BAND, B_BAND);
        }
    }

    /// FTRL-Proximal on the deviation from the prior, with the standardized-score gate.
    fn ftrl(&mut self, i: usize, g: f64) {
        if !g.is_finite() {
            return;
        }
        let d = self.w[i] - self.prior[i];
        let n_new = (self.n[i] + g * g).min(N_CAP);
        let sigma = (n_new.sqrt() - self.n[i].sqrt()) / alpha(i);
        self.z[i] += g - sigma * d;
        self.n[i] = n_new;
        self.w[i] = weight(self.prior[i], self.z[i], self.n[i], alpha(i));
    }

    /// A new request: its intercept and its sums start again; the coefficients carry on.
    pub fn reset_request(&mut self) {
        self.b = 0.0;
        self.b_g2 = 0.0;
        self.request = (0.0, 0.0, 0);
        self.buckets = [BucketReport::default(); COEFS];
    }

    /// The request's log loss per labelled node for this model and today's estimate, and the node
    /// count, for the report line.
    #[must_use]
    pub fn request_losses(&self) -> (f64, f64, u64) {
        let n = self.request.2.max(1) as f64;
        (self.request.0 / n, self.request.1 / n, self.request.2)
    }

    /// This request's calibration per n-gram bucket that saw a label: `(name, report, weight)`,
    /// the name `new` / `agreed-pick` / `agreed-sib`, source, count bucket and run bucket.
    #[must_use]
    pub fn bucket_reports(&self) -> Vec<(String, BucketReport, f64)> {
        let mut out = Vec::new();
        for (c, r) in self.buckets.iter().enumerate() {
            if r.labels == 0 || c < NGRAM {
                continue;
            }
            if c >= CONTRA {
                const SLOT_NAMES: [&str; SLOTS] = [
                    "req/n1",
                    "req/n2+",
                    "req/copy4-15",
                    "req/copy16+",
                    "table/n1",
                    "table/n2",
                    "table/n3-4",
                    "table/n5+",
                ];
                let (kind, slot) = if c - CONTRA < SLOTS {
                    ("contra-pick", c - CONTRA)
                } else {
                    ("contra-sib", c - CONTRA - SLOTS)
                };
                out.push((format!("{kind}/{}", SLOT_NAMES[slot]), *r, self.w[c]));
                continue;
            }
            let (kind, b) = match c - NGRAM {
                x if x < BUCKETS => ("new", x),
                x if x < 2 * BUCKETS => ("agreed-pick", x - BUCKETS),
                x => ("agreed-sib", x - 2 * BUCKETS),
            };
            let (source, evidence) = if b / 12 == 0 {
                ("req", ["n1", "n2+", "copy4-15", "copy16+"][(b / 3) % 4])
            } else {
                ("table", ["n1", "n2", "n3-4", "n5+"][(b / 3) % 4])
            };
            let run = ["0", "1-2", "3+"][b % 3];
            out.push((format!("{kind}/{source}/{evidence}/r{run}"), *r, self.w[c]));
        }
        out
    }

    #[must_use]
    pub fn intercept(&self) -> f64 {
        self.b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The features follow the definitions, and the pick's share keeps its precision when the
    /// siblings are far below the pick (a share of 1 - 1e-20 would round to 1 as a probability).
    #[test]
    fn features_follow_the_definitions_and_keep_a_confident_pick_s_share() {
        let f = Features::position(0.8, &[2.0, 1.0, 0.0]);
        let tail = (-1.0_f64).exp() + (-2.0_f64).exp();
        assert_eq!(f.len(), 3);
        assert_eq!(f[0].kind, Kind::Pick);
        assert!((f[0].logit_head - (0.8_f64 / 0.2).ln()).abs() < 1e-6);
        assert!((f[0].logit_share_pick + tail.ln()).abs() < 1e-12);
        assert!((f[1].log_r - ((-1.0_f64).exp() / tail).ln()).abs() < 1e-12);
        assert!((f[2].log_r - ((-2.0_f64).exp() / tail).ln()).abs() < 1e-12);
        // Siblings 46 logits below: the pick's share is 1 - 1e-20 but its logit is still 46,
        // clipped to the bound, not the 36.7 that 1 - share rounding to 0 would never reach.
        let far = Features::position(0.9, &[50.0, 4.0]);
        assert!((far[0].logit_share_pick - logit_clip()).abs() < 1e-9);
        assert!(
            far[1].log_r.abs() < 1e-12,
            "a lone sibling is all of the siblings' total"
        );
    }

    /// A parent's probabilities and NONE sum to exactly 1, whatever the coefficients.
    #[test]
    fn a_parent_s_probabilities_and_none_sum_to_one() {
        let m = AcceptModel::fitted([1.05, 0.93, 0.42, 0.98, 0.89, 0.72, -0.23]);
        let g = Features::position(0.7, &[3.0, 2.5, 1.0, -4.0]);
        let p = m.probabilities(&g);
        let s: f64 = p.iter().sum();
        assert!(s < 1.0 && s > 0.0, "{s}");
        // NONE by its own formula.
        let scores: Vec<f64> = g.iter().map(|f| m.score(f)).collect();
        let none = 1.0 / (1.0 + scores.iter().map(|x| x.exp()).sum::<f64>());
        assert!((s + none - 1.0).abs() < 1e-12);
        // An n-gram follower joins the same softmax: it takes probability from the drafter's
        // candidates, and the group still sums below 1.
        let mut mixed = g.clone();
        let terms = NgramTerms {
            source: Source::Request,
            count: 3,
            matched: 0,
            run: 0,
        };
        mixed.push(Features::ngram(terms, 0.7, 0, 0));
        let q = m.probabilities(&mixed);
        assert!(q.iter().sum::<f64>() < 1.0);
        assert!(q[0] < p[0], "the follower took nothing from the pick");
    }

    /// A copy match is its own evidence: its length picks the class, and the count only speaks
    /// when there is no match. Every bucket and slot stays in range.
    #[test]
    fn a_copy_match_s_length_is_its_evidence() {
        let at = |source, count, matched, run| NgramTerms {
            source,
            count,
            matched,
            run,
        };
        let short = at(Source::Request, 1, 0, 0);
        let repeated = at(Source::Request, 5, 0, 0);
        let copy = at(Source::Request, 1, 4, 0);
        let long_copy = at(Source::Request, 9, LONG_MATCH, 0);
        let slots: Vec<usize> = [short, repeated, copy, long_copy]
            .iter()
            .map(NgramTerms::slot)
            .collect();
        assert_eq!(slots, vec![0, 1, 2, 3]);
        assert_eq!(at(Source::Table, 5, 0, 3).slot(), 7);
        assert_eq!(at(Source::Table, 5, 0, 3).bucket(), BUCKETS - 1);
        assert_eq!(copy.bucket(), 6);
        // n-gram-only nodes carry no depth term; drafter nodes do.
        let deep = Features::ngram(long_copy, 0.9, 30, 0);
        let (terms, n) = deep.terms();
        assert!(!terms[..n].iter().any(|&(c, _)| c == DEPTH));
    }

    /// A contradicted drafter node carries its slot's contradiction bucket and the negative floored
    /// start; an agreed node cannot also be contradicted; a sibling's terms still fit.
    #[test]
    fn a_contradicted_drafter_node_starts_below_itself() {
        let copy = NgramTerms {
            source: Source::Request,
            count: 0,
            matched: LONG_MATCH,
            run: 0,
        };
        let g = Features::position(0.7, &[3.0, 2.5, 1.0]);
        let pick = g[0].contradicted(copy, 0.9);
        assert!(
            (pick.base + (0.9_f64 / 0.1).ln()).abs() < 1e-12,
            "{}",
            pick.base
        );
        assert_eq!(pick.bucket_coefficient(), Some(CONTRA + copy.slot()));
        let sib = g[1].contradicted(copy, 0.3);
        assert!(
            sib.base.abs() < 1e-12,
            "a proposal right under half the time says nothing"
        );
        assert_eq!(sib.bucket_coefficient(), Some(CONTRA + SLOTS + copy.slot()));
        let (_, n) = sib.terms();
        assert!(n <= MAX_TERMS);
        let agreed = g[0].agreed(copy, 0.9).contradicted(copy, 0.9);
        assert!(agreed.contra.is_none() && agreed.base > 0.0);
        let m = AcceptModel::fitted([1.05, 0.93, 0.42, 0.98, 0.89, 0.72, -0.23]);
        let before = m.probabilities(&g)[0];
        let after = m.probabilities(&[pick, g[1], g[2]])[0];
        assert!(after < before, "{after} >= {before}");
    }

    /// The derived prior IS the coverage-corrected estimate: P(pick) = head, P(sibling j) =
    /// (1 - head) kappa r_j, P(NONE) = (1 - head)(1 - kappa).
    #[test]
    fn the_derived_prior_is_the_coverage_corrected_estimate() {
        let kappa = 0.616;
        let m = AcceptModel::derived(kappa);
        let (head, values) = (0.8_f32, [1.0_f32, 0.2, -0.7]);
        let g = Features::position(head, &values);
        let p = m.probabilities(&g);
        let rel: Vec<f64> = values[1..].iter().map(|&v| f64::from(v) - 1.0).collect();
        let tail: f64 = rel.iter().map(|r| r.exp()).sum();
        let h = f64::from(head);
        assert!((p[0] - h).abs() < 1e-6, "{}", p[0]);
        for (j, r) in rel.iter().enumerate() {
            let want = (1.0 - h) * kappa * r.exp() / tail;
            assert!((p[j + 1] - want).abs() < 1e-6, "{} vs {want}", p[j + 1]);
        }
        assert!(
            !m.prices(),
            "a derived start does not price before it has beaten today"
        );
    }

    /// Labels drawn from known coefficients move a model that starts elsewhere toward them, and the
    /// guard then prefers it to an estimate that ignores the pick's share.
    #[test]
    fn labels_from_known_coefficients_pull_the_model_and_win_the_guard() {
        let truth = AcceptModel::fitted([0.94, 0.86, 0.47, 0.80, 0.90, 0.62, -0.17]);
        let mut m = AcceptModel::derived(0.6);
        let mut state = 12345_u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1_u64 << 53) as f64)
        };
        for _ in 0..20_000 {
            let head = (0.05 + 0.9 * rnd()) as f32;
            let top = 4.0 * rnd();
            let values: Vec<f32> = (0..8)
                .map(|k| {
                    if k == 0 {
                        top as f32
                    } else {
                        (top - 8.0 * rnd()) as f32
                    }
                })
                .collect();
            let g = Features::position(head, &values);
            let p = truth.probabilities(&g);
            let u = rnd();
            let mut acc = 0.0;
            let mut pick = None;
            for (j, pj) in p.iter().enumerate() {
                acc += pj;
                if u < acc {
                    pick = Some(j);
                    break;
                }
            }
            // "Today": the head for the pick, the share for a sibling.
            let total: f64 = values.iter().map(|&v| (f64::from(v) - top).exp()).sum();
            let mut today = vec![f64::from(head)];
            today.extend(
                values[1..]
                    .iter()
                    .map(|&v| (f64::from(v) - top).exp() / total),
            );
            m.observe(&g, &today, pick);
        }
        assert!(m.prices(), "guard {:?}", m.guard);
        let (w, seen) = m.state();
        assert_eq!(seen, 20_000);
        // The pick's share coefficient left its derived prior of 0 toward the truth's 0.47.
        assert!(w[2] > 0.1, "c_p stayed at {}", w[2]);
    }

    /// A request boundary changes nothing about what the model learned: one model that learns two
    /// requests' labels in one process, and one stored after the first and resumed for the second,
    /// end bit for bit equal.
    #[test]
    fn a_stored_and_resumed_model_learns_on_exactly_as_one_that_never_stopped() {
        let mut state = 99_u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1_u64 << 53) as f64)
        };
        let mut groups = Vec::new();
        for _ in 0..3000 {
            let head = (0.05 + 0.9 * rnd()) as f32;
            let top = 4.0 * rnd();
            let values: Vec<f32> = (0..8)
                .map(|k| {
                    if k == 0 {
                        top as f32
                    } else {
                        (top - 8.0 * rnd()) as f32
                    }
                })
                .collect();
            let g = Features::position(head, &values);
            let pick = match (rnd() * 10.0) as usize {
                k if k < 8 => Some(k),
                _ => None,
            };
            let today: Vec<f64> = (0..8).map(|k| 0.9 / f64::from(k + 1)).collect();
            groups.push((g, today, pick));
        }
        let mut straight = AcceptModel::derived(0.5);
        let mut stopped = AcceptModel::derived(0.5);
        for (g, today, pick) in &groups[..1700] {
            straight.observe(g, today, *pick);
            stopped.observe(g, today, *pick);
        }
        // The request ends: level 1 resets in the process, and the store keeps level 2.
        straight.reset_request();
        let mut resumed = AcceptModel::resume(&stopped.level2());
        for (g, today, pick) in &groups[1700..] {
            straight.observe(g, today, *pick);
            resumed.observe(g, today, *pick);
        }
        assert_eq!(straight.level2(), resumed.level2());
        assert_eq!(straight.w.map(f64::to_bits), resumed.w.map(f64::to_bits));
        assert_eq!(straight.b.to_bits(), resumed.b.to_bits());
        assert_eq!(straight.request_losses(), resumed.request_losses());
        assert!(
            straight.w.map(f64::to_bits) != straight.prior.map(f64::to_bits),
            "the test must move a weight off its prior"
        );
    }

    /// An unevidenced coordinate stays exactly at its prior: gradients that average to zero never
    /// pass the standardized gate.
    #[test]
    fn noise_alone_leaves_a_coefficient_exactly_at_its_prior() {
        let prior = [1.0, 0.9, 0.4, 1.0, 0.9, 0.7, -0.2];
        let mut m = AcceptModel::fitted(prior);
        for k in 0..4000 {
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            m.ftrl(2, 0.3 * sign);
        }
        assert_eq!(m.state().0[2].to_bits(), prior[2].to_bits());
    }
}
