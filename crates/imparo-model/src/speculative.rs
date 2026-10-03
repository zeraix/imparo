//! Model-independent greedy continuation; default single-token generation is unchanged.
use crate::Model;

/// Serialized draft provider paired with the target. On error, terminate the
/// continuation: the target may have committed before an auxiliary-state failure.
#[derive(Clone, Debug)]
pub struct DraftTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
    /// Row -> the rank best-first gave it, and the anchor -1. THE COUNTERFACTUAL: because
    /// every prefix of the best-first order is itself a tree, a walk through this says what
    /// would have been accepted at any narrower width, from the one round that ran.
    pub rank: Vec<i32>,
    /// `s[k]` = expected accepted tokens from the first k nodes, so `s[n - 1]` is S(n) for
    /// every width the budget could have chosen, not only the one it did.
    pub s: Vec<f64>,
    /// Row -> the candidate it holds, the anchor -1: what a learner reads to label every
    /// candidate of an accepted parent, placed or not.
    pub node: Vec<i32>,
    /// `s_n[k]`: the part of `s[k]` that n-gram and agreed nodes carry. The width decision scales
    /// it by the n-gram's own gain (design 6.6).
    pub s_n: Vec<f64>,
    /// An exploration round: its last row went to the least-labelled n-gram bucket, so the order
    /// is not best-first there and the round must not teach the width's gain.
    pub explored: bool,
    /// Row -> the source that proposed it; `None` for the anchor and for every row of a tree
    /// built by hand.
    pub source: Vec<Option<NodeSource>>,
}
impl DraftTree {
    /// A tree built by hand -- an instrument's fixed shape, not the budget's. It has no
    /// best-first plan behind it, so there are no ranks and no S to report; the value model
    /// learns nothing from these rounds and must not pretend otherwise.
    #[must_use]
    pub fn plain(tokens: Vec<u32>, parents: Vec<i32>) -> Self {
        let rank = vec![-1_i32; tokens.len()];
        let node = vec![-1_i32; tokens.len()];
        let source = vec![None; tokens.len()];
        Self {
            tokens,
            parents,
            rank,
            s: vec![0.0],
            node,
            s_n: vec![0.0],
            explored: false,
            source,
        }
    }
}

/// Draft tokens the target verified, and the ones it accepted; the anchor is never counted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub verified: usize,
    pub accepted: usize,
}

/// A continuation's draft tokens by the source that proposed each (design 6.6). A chain round's
/// tokens are all the drafter's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftTally {
    pub drafter: Tally,
    pub ngram: Tally,
    pub agreed: Tally,
}
impl DraftTally {
    fn of(&mut self, source: NodeSource) -> &mut Tally {
        match source {
            NodeSource::Drafter => &mut self.drafter,
            NodeSource::Ngram => &mut self.ngram,
            NodeSource::Agreed => &mut self.agreed,
        }
    }
    /// Every source together: the response's `draft_verified` and `draft_accepted`.
    #[must_use]
    pub fn total(&self) -> Tally {
        Tally {
            verified: self.drafter.verified
                + self.ngram.verified
                + self.agreed.verified,
            accepted: self.drafter.accepted
                + self.ngram.accepted
                + self.agreed.accepted,
        }
    }
    /// A tree round: every row past the anchor was verified, and the path past the anchor
    /// accepted. A row no source claims is the drafter's: only the budget's trees hold n-gram rows,
    /// and they mark every row.
    fn tree(&mut self, tree: &DraftTree, path: &[i32]) {
        let source = |row: usize| {
            tree.source
                .get(row)
                .copied()
                .flatten()
                .unwrap_or(NodeSource::Drafter)
        };
        for row in 1..tree.tokens.len() {
            self.of(source(row)).verified += 1;
        }
        for row in path.iter().skip(1).filter_map(|&r| usize::try_from(r).ok()) {
            self.of(source(row)).accepted += 1;
        }
    }
    /// A chain round: `verified` drafted tokens past the anchor, the first `accepted` taken.
    fn chain(&mut self, verified: usize, accepted: usize) {
        self.drafter.verified += verified;
        self.drafter.accepted += accepted;
    }
}

/// How this verification actually submitted its target work, not whether a
/// graph was eligible. Unknown keeps uninstrumented adapters out of cost classes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TreeSubmission {
    #[default]
    Unknown,
    Ordinary,
    Capture,
    Replayed,
}

/// One successful speculative transaction, including its accepted history commit.
/// Target KV/recurrent commit is already inside `verify`; `commit` measures only
/// the additional drafter/history work. Tail rounds are censored for rate learning.
#[derive(Clone, Copy, Debug)]
pub struct TreeRoundTiming {
    pub context: usize,
    pub rows: usize,
    /// Accepted input rows including the anchor, unlike `DraftTally`.
    pub accepted: usize,
    pub fixed: std::time::Duration,
    pub verify: std::time::Duration,
    pub commit: std::time::Duration,
    pub total: std::time::Duration,
    pub submission: TreeSubmission,
    pub continuing: bool,
}

#[derive(Clone, Debug)]
pub struct TreeVerification {
    pub path: Vec<i32>,
    pub next_token: u32,
    pub submission: TreeSubmission,
}

/// A node the tree builder may place: its token, the candidate it hangs from (-1 for the anchor)
/// and `q`, the estimated probability that it is accepted when its parent is (design 6.2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeCandidate {
    pub token: u32,
    pub parent: i32,
    pub q: f64,
}

/// Which source proposed a candidate (design 6.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeSource {
    /// Only the drafter.
    Drafter,
    /// Only the n-gram.
    Ngram,
    /// The drafter, and the n-gram's follower was the same token.
    Agreed,
}

/// How a round's tree may branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shape {
    /// Any node may hold several children: design 6.2's tree.
    #[default]
    Tree,
    /// One child per node: best-first becomes a single path, the highest-Q child at each step,
    /// verified through the same rows as a tree. Everything else -- candidates, pricing, the
    /// budget -- is the tree's, so Tree against Chain measures what branching itself is worth.
    Chain,
}

/// What a round knows about its candidates beyond each one's q, and the shape its tree may take.
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeRound<'a> {
    /// Candidate -> its source; empty when every candidate is the drafter's.
    pub of: &'a [NodeSource],
    /// On an exploration round, candidate -> the labels behind its n-gram bucket, `u64::MAX` for a
    /// candidate that is not an n-gram-only node. `None` on every other round.
    pub explore: Option<&'a [u64]>,
    pub shape: Shape,
}

impl TreeRound<'_> {
    fn is_ngram(&self, candidate: usize) -> bool {
        self.of
            .get(candidate)
            .is_some_and(|s| *s != NodeSource::Drafter)
    }
}

/// How wide a round's tree is: an instrument, or design 6.3's budget.
///
/// Every variant carries `offset`, design 11.4's per-request correction to `q` in logit units
/// (`accept_offset::AcceptOffset`). It belongs here and not only on the budget because it
/// corrects the ESTIMATE, not the policy: a fixed width built from a corrected `q` is still a
/// fixed width, and an A/B of the offset must therefore switch the OFFSET, not the width.
#[derive(Clone, Copy, Debug)]
pub enum TreeBudget<'a> {
    /// A fixed node count, the anchor included. An instrument, not the budget.
    Fixed { nodes: usize, offset: f64 },
    /// Design 6.3's budget.
    Choose(Choice<'a>),
}

/// What design 6.3's budget reads in one round.
#[derive(Clone, Copy, Debug)]
pub struct Choice<'a> {
    /// The cost table's envelope, `(rows, microseconds)` ascending.
    pub widths: &'a [(usize, f64)],
    /// What the round costs whatever width it runs.
    pub fixed_us: f64,
    /// The learned scale of the drafter nodes' marginal (`cost_model::MarginGain`), 1.0 when
    /// untrained.
    pub gain: f64,
    /// The same scale for the marginal the n-gram's and agreed nodes carry.
    pub gain_n: f64,
    /// The two scales for a width BELOW `base_rows`, `(g_d, g_n)`: what the rows it gives up are
    /// worth. Fitted on every round's revealed prefixes, not on the wide rounds `gain` comes
    /// from (`VerifyCost::gain_below`).
    pub gain_below: (f64, f64),
    /// The width this provider is currently sitting on, or 0 before the first round. The
    /// budget keeps it unless an alternative clears it by `switch_margin`.
    pub hold: usize,
    /// Design 11.4's per-request offset on `q`, in logit units. 0 is the frozen head.
    pub offset: f64,
    /// The process's measured tokens per microsecond at this round's context
    /// (`cost_model::LongRunRate`), or `None` to price a round's time at its own ratio.
    pub rho: Option<f64>,
    /// THE WIDTH THE MARGINAL IS MEASURED FROM, or 0 for the narrowest offered width. The gains
    /// are learned as realised over predicted tokens above this width, so it must not move when
    /// the cost table learns a narrower width: a baseline of 2 rows and one of 8 fit different
    /// gains, and a gain fitted above 2 undervalues what rows above 8 buy.
    pub base_rows: usize,
}

impl TreeBudget<'_> {
    /// The tree this budget asks for. `stops` are the tokens that end the answer: a node
    /// carrying one can never be accepted, so it never takes a row.
    ///
    /// # Errors
    /// As `best_first_tree` and `budgeted_tree`.
    pub fn tree(
        &self,
        anchor: u32,
        candidates: &[TreeCandidate],
        stops: &[u32],
        round: TreeRound<'_>,
    ) -> Result<DraftTree, String> {
        match self {
            Self::Fixed { nodes, offset } => {
                fixed_tree(anchor, candidates, *nodes, *offset, stops, round)
            }
            Self::Choose(choice) => {
                budgeted_tree(anchor, candidates, stops, *choice, round)
            }
        }
    }
}

/// The best-first selection, shared by the fixed-width builder and the budget.
struct Plan {
    /// Candidate `i`'s children at index `i`, the anchor's at index `candidates.len()`.
    children: Vec<Vec<usize>>,
    /// Q: the product of q along a node's path, so `big_q[i]` is the probability that this
    /// node is accepted -- its whole ancestor chain and itself.
    big_q: Vec<f64>,
    /// Candidates in the order best-first took them. Taking the first `k` gives a tree of
    /// `k + 1` rows, and every prefix is itself a valid tree: Q never grows along a path,
    /// so a node is always taken after its parent.
    order: Vec<usize>,
}

fn plan(
    candidates: &[TreeCandidate],
    max_nodes: usize,
    offset: f64,
    stops: &[u32],
    shape: Shape,
) -> Result<Plan, String> {
    if max_nodes == 0 {
        return Err("a tree holds its anchor, so at least one node".into());
    }
    let n = candidates.len();
    let mut children = vec![Vec::new(); n + 1];
    let mut big_q = vec![0.0_f64; n];
    for (i, c) in candidates.iter().enumerate() {
        if !(0.0..=1.0).contains(&c.q) {
            return Err(format!(
                "tree candidate {i}: q {} is not a probability",
                c.q
            ));
        }
        // Design 11.4's correction lands HERE, before Q is formed, so one offset fixes S at
        // every width at once. It is monotone in q, so a node's own ordering among its
        // siblings is untouched; what moves is how depth trades against breadth, because Q is
        // a PRODUCT of corrected factors.
        let q = crate::accept_offset::shift(c.q, offset);
        let parent = match usize::try_from(c.parent) {
            Ok(p) if p < i => {
                big_q[i] = big_q[p] * q;
                p
            }
            Err(_) if c.parent == -1 => {
                big_q[i] = q;
                n
            }
            _ => {
                return Err(format!(
                    "tree candidate {i}: parent {} is not an earlier candidate",
                    c.parent
                ));
            }
        };
        // A STOP-TOKEN NODE IS WORTH 0, EXACTLY. The walk ends at a stop pick before it looks
        // for a child (`verification::tree_accepted_path`), so this node is never entered and
        // neither is anything below it. Left linked, it would take a row and count in S. Not
        // linked, it never reaches the frontier, and its children's Q is 0 through the
        // product.
        if stops.contains(&c.token) {
            big_q[i] = 0.0;
            continue;
        }
        children[parent].push(i);
    }
    let mut order = Vec::with_capacity(max_nodes.saturating_sub(1));
    let mut frontier = children[n].clone();
    while order.len() + 1 < max_nodes {
        let Some(at) = (0..frontier.len()).max_by(|&a, &b| {
            big_q[frontier[a]]
                .total_cmp(&big_q[frontier[b]])
                .then(frontier[b].cmp(&frontier[a]))
        }) else {
            break;
        };
        let i = frontier.swap_remove(at);
        order.push(i);
        // A chain keeps only the node it took: its siblings leave the frontier with it.
        if shape == Shape::Chain {
            frontier.clear();
        }
        frontier.extend_from_slice(&children[i]);
    }
    Ok(Plan {
        children,
        big_q,
        order,
    })
}

/// THE VALUE PROBE, off unless `IMPARO_DSPARK_SCORE=1`.
///
/// The budget's value model is S(n), a sum of Q. Whether Q is calibrated is a question about
/// what the target ACCEPTED, and the engine already knows that -- so the probe writes the two
/// halves and the arithmetic happens offline:
///
/// ```text
///   dspark plan   Q in best-first order, and each ROW's best-first rank
///   dspark path   the round's fixed cost, its row count, the accepted rows
/// ```
///
/// Because every prefix of the best-first order is itself a tree, one round run at N rows
/// says what WOULD have been accepted at every n <= N: the accepted walk cut at the first
/// node whose rank is >= n - 1. One wide run therefore gives accept(n) for all n, exactly.
pub(crate) fn score_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_SCORE").as_deref() == Ok("1"))
}

/// THE CHOOSER PROBE, off unless `IMPARO_DSPARK_TRACE=1`: one line a round with what the width
/// chooser saw -- every offered width with the cost model's verify estimate and the expected
/// accepted nodes the plan gives it -- and the node count it kept. With `IMPARO_DSPARK_ROUND=1`
/// (the measured verify) and `IMPARO_DSPARK_SCORE=1` (the accepted path) it is enough to replay
/// the choice offline.
/// `IMPARO_DSPARK_CHAIN_MAX=N`: a chain verify keeps at most N of the block's draft tokens.
/// Unset, the chain is the whole block. A tree is not affected.
fn chain_max() -> Option<usize> {
    static MAX: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("IMPARO_DSPARK_CHAIN_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
    })
}

/// `IMPARO_DSPARK_CHAIN_ROUTE=block` verifies a chain round on the causal block verify, the route it
/// took before chains were laid out as trees; unset or `tree`, a chain goes through the tree verify
/// whenever the target admits one. One binary, so the two routes can be compared.
fn chain_route_tree() -> bool {
    static TREE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TREE.get_or_init(|| std::env::var("IMPARO_DSPARK_CHAIN_ROUTE").as_deref() != Ok("block"))
}

fn trace_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_TRACE").as_deref() == Ok("1"))
}

/// The chooser half of the trace, inside the caller's gate.
fn probe_choice(s: &[f64], s_n: &[f64], order_len: usize, choice: &Choice<'_>, keep: usize) {
    let widths: Vec<String> = choice
        .widths
        .iter()
        .map(|&(rows, us)| {
            let k = rows.saturating_sub(1).min(order_len);
            format!(
                "{rows}:{us:.1}:{:.4}:{:.4}",
                s.get(k).copied().unwrap_or(0.0),
                s_n.get(k).copied().unwrap_or(0.0)
            )
        })
        .collect();
    eprintln!(
        "dspark choose kept={keep} hold={} fixed_us={:.1} gain={:.4} gain_n={:.4} gain_below={:.4} gain_n_below={:.4} rho={} order={order_len} widths={}",
        choice.hold,
        choice.fixed_us,
        choice.gain,
        choice.gain_n,
        choice.gain_below.0,
        choice.gain_below.1,
        choice
            .rho
            .map_or_else(|| "-".to_string(), |r| format!("{:.3}", r * 1e6)),
        widths.join(",")
    );
}

/// THE ROUND SPLIT PROBE, off unless `IMPARO_DSPARK_ROUND=1`: one line a round with the time
/// in the drafter's forward, the tree build, the verify and the commit, so a round's cost is
/// attributed to its stages instead of read as one number.
fn round_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_ROUND").as_deref() == Ok("1"))
}

/// The plan half of the probe. The whole block is inside the caller's gate.
fn probe_plan(p: &Plan, keep: usize, ranks: &[i32]) {
    let q: Vec<String> = p
        .order
        .iter()
        .take(keep)
        .map(|&i| format!("{:.6}", p.big_q[i]))
        .collect();
    let r: Vec<String> = ranks.iter().map(ToString::to_string).collect();
    eprintln!(
        "dspark plan keep={keep} q={} rank={}",
        q.join(","),
        r.join(",")
    );
}

/// The tree over the first `keep` of the plan's order: rows depth-first, the most likely
/// child first (design 6.5), so an accepted prefix of the drafter's chain needs no KV copy.
fn emit(
    anchor: u32,
    candidates: &[TreeCandidate],
    p: &Plan,
    keep: usize,
    round: TreeRound<'_>,
) -> Result<DraftTree, String> {
    let n = candidates.len();
    let mut chosen = vec![false; n];
    for &i in p.order.iter().take(keep) {
        chosen[i] = true;
    }
    let order = |list: &[usize]| {
        let mut kept: Vec<usize> =
            list.iter().copied().filter(|&c| chosen[c]).collect();
        kept.sort_by(|&a, &b| p.big_q[b].total_cmp(&p.big_q[a]).then(a.cmp(&b)));
        kept
    };
    // Row -> the position best-first gave that node. EVERY tree carries it now, not only a
    // probed one: it is what lets a later commit read accept(m) for every m <= the width
    // that ran, which is the only uncensored evidence the value model ever gets. It costs
    // one inverse permutation over the chosen nodes.
    let of_candidate: Vec<i32> = {
        let mut inv = vec![-1_i32; n];
        for (rank, &i) in p.order.iter().take(keep).enumerate() {
            inv[i] = i32::try_from(rank).map_err(|_| "tree rank overflow")?;
        }
        inv
    };
    // S over the WHOLE best-first order, not just the part kept: the learner needs S at the
    // widths it did not run as well as the one it did.
    let (svals, s_n) = prefix_sums(p, round);
    let mut rank = vec![-1_i32]; // the anchor is in every tree
    let mut tokens = vec![anchor];
    let mut parents = vec![-1_i32];
    let mut node = vec![-1_i32];
    let mut source = vec![None];
    // (candidate, its parent's row), the most likely child on top.
    let mut stack: Vec<(usize, i32)> = order(&p.children[n])
        .into_iter()
        .rev()
        .map(|c| (c, 0))
        .collect();
    while let Some((c, parent_row)) = stack.pop() {
        let row = i32::try_from(tokens.len()).map_err(|_| "tree row overflow")?;
        tokens.push(candidates[c].token);
        parents.push(parent_row);
        rank.push(of_candidate[c]);
        node.push(i32::try_from(c).map_err(|_| "tree candidate index overflow")?);
        source.push(Some(
            round.of.get(c).copied().unwrap_or(NodeSource::Drafter),
        ));
        stack.extend(order(&p.children[c]).into_iter().rev().map(|k| (k, row)));
    }
    Ok(DraftTree {
        tokens,
        parents,
        rank,
        s: svals,
        node,
        s_n,
        explored: false,
        source,
    })
}

/// S over the whole best-first order, and the part of it the n-gram's and agreed nodes carry.
fn prefix_sums(p: &Plan, round: TreeRound<'_>) -> (Vec<f64>, Vec<f64>) {
    let mut s = Vec::with_capacity(p.order.len() + 1);
    let mut s_n = Vec::with_capacity(p.order.len() + 1);
    s.push(0.0_f64);
    s_n.push(0.0_f64);
    for &i in &p.order {
        let (last, last_n) = (*s.last().unwrap_or(&0.0), *s_n.last().unwrap_or(&0.0));
        s.push(last + p.big_q[i]);
        s_n.push(last_n + if round.is_ngram(i) { p.big_q[i] } else { 0.0 });
    }
    (s, s_n)
}

/// DESIGN 6.6'S EXPLORATION ROW: the last row of the chosen width goes to the frontier's n-gram node
/// in the bucket with the fewest labels, the highest Q among those. A frontier node hangs from a
/// chosen node (or the anchor) and is not chosen itself; the removed node is the last one taken,
/// which best-first guarantees is a leaf, and a node hanging from it is not a candidate for its
/// place. In a chain the node must hang where the removed one did, or the path would branch. Returns
/// whether a node was swapped in.
///
/// Why only n-gram nodes: every candidate of an accepted parent is labelled whether or not it is
/// placed, so the drafter's kinds need no row to learn. An n-gram chain below a node that is never
/// placed never has an accepted parent, so its buckets would stay unlabelled; placing one frontier
/// node a round in sixteen is what lets its children be judged. No random draw: near-tie text stays
/// reproducible.
fn explore(
    p: &mut Plan,
    keep: usize,
    candidates: &[TreeCandidate],
    labels: &[u64],
    shape: Shape,
) -> bool {
    if keep == 0 {
        return false;
    }
    let last = p.order[keep - 1];
    let mut chosen = vec![false; candidates.len()];
    for &i in &p.order[..keep - 1] {
        chosen[i] = true;
    }
    let mut best: Option<(u64, f64, usize)> = None;
    for (i, c) in candidates.iter().enumerate() {
        let n = labels.get(i).copied().unwrap_or(u64::MAX);
        if n == u64::MAX || chosen[i] || i == last || p.big_q[i] <= 0.0 {
            continue;
        }
        let hangs = match (shape, usize::try_from(c.parent)) {
            (Shape::Chain, _) => c.parent == candidates[last].parent,
            (Shape::Tree, Ok(parent)) => chosen[parent],
            (Shape::Tree, Err(_)) => c.parent == -1,
        };
        // The walk only reaches a node through its parent's children, which the plan left unlinked
        // for a stop token: such a node has Q 0 and is already excluded above.
        if !hangs {
            continue;
        }
        let better =
            best.is_none_or(|(bn, bq, _)| n < bn || (n == bn && p.big_q[i] > bq));
        if better {
            best = Some((n, p.big_q[i], i));
        }
    }
    let Some((_, _, pick)) = best else {
        return false;
    };
    match p.order.iter().position(|&i| i == pick) {
        Some(j) => p.order.swap(keep - 1, j),
        None => p.order[keep - 1] = pick,
    }
    true
}

/// Design 6.2's tree over `candidates`: best-first on Q, the product of q along a node's path,
/// until the tree holds `max_nodes` rows with the anchor; then rows depth-first, the most likely
/// child first (design 6.5). Q never grows along a path, so every chosen node hangs from a node
/// chosen before it. On equal Q the earlier candidate goes first.
///
/// A FIXED WIDTH IS AN INSTRUMENT, not the budget: `budgeted_tree` is design 6.3.
///
/// A candidate carrying a token in `stops` never takes a row, and neither does anything below it.
///
/// # Errors
/// When `max_nodes` is 0, a candidate's parent is not an earlier candidate, or a q is not a
/// probability.
pub fn best_first_tree(
    anchor: u32,
    candidates: &[TreeCandidate],
    max_nodes: usize,
    offset: f64,
    stops: &[u32],
) -> Result<DraftTree, String> {
    fixed_tree(
        anchor,
        candidates,
        max_nodes,
        offset,
        stops,
        TreeRound::default(),
    )
}

fn fixed_tree(
    anchor: u32,
    candidates: &[TreeCandidate],
    max_nodes: usize,
    offset: f64,
    stops: &[u32],
    round: TreeRound<'_>,
) -> Result<DraftTree, String> {
    let mut p = plan(candidates, max_nodes, offset, stops, round.shape)?;
    let keep = p.order.len();
    let explored = round
        .explore
        .is_some_and(|labels| explore(&mut p, keep, candidates, labels, round.shape));
    emit_probed(anchor, candidates, &p, keep, round, explored)
}

/// `emit`, and the probe's plan line when it is on.
fn emit_probed(
    anchor: u32,
    candidates: &[TreeCandidate],
    p: &Plan,
    keep: usize,
    round: TreeRound<'_>,
    explored: bool,
) -> Result<DraftTree, String> {
    let mut tree = emit(anchor, candidates, p, keep, round)?;
    tree.explored = explored;
    if score_probe() {
        probe_plan(p, keep, &tree.rank);
        probe_kids(candidates, p);
    }
    Ok(tree)
}

/// THE STRUCTURE PROBE. The whole block is inside the caller's gate.
///
/// A parent's children are mutually exclusive -- at most one child's token is the target's
/// argmax -- so their true acceptance probabilities sum to at most 1. `raw` sums the estimates
/// as their sources give them; `used` sums them after the offset, as the plan priced them.
///
/// `s` is S over the WHOLE best-first order, not only the kept part: an offline pass needs the
/// value of every width the budget could have chosen, not just the one it did.
fn probe_kids(candidates: &[TreeCandidate], p: &Plan) {
    let n = candidates.len();
    let (mut parents, mut counts, mut raw, mut used) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for parent in std::iter::once(n).chain(0..n) {
        let kids = &p.children[parent];
        if kids.is_empty() {
            continue;
        }
        // The anchor always commits, so its children's Q is already their conditional q.
        let q_parent = if parent == n { 1.0 } else { p.big_q[parent] };
        let raw_sum: f64 = kids.iter().map(|&c| candidates[c].q).sum();
        let used_sum: f64 = if q_parent > 0.0 {
            kids.iter().map(|&c| p.big_q[c] / q_parent).sum()
        } else {
            f64::NAN
        };
        parents.push(if parent == n {
            "a".to_string()
        } else {
            parent.to_string()
        });
        counts.push(kids.len().to_string());
        raw.push(format!("{raw_sum:.4}"));
        used.push(format!("{used_sum:.4}"));
    }
    let mut acc = 0.0_f64;
    let s: Vec<String> = std::iter::once("0".to_string())
        .chain(p.order.iter().map(|&i| {
            acc += p.big_q[i];
            format!("{acc:.5}")
        }))
        .collect();
    eprintln!(
        "dspark kids parents={} n={} raw={} used={}",
        parents.join(","),
        counts.join(","),
        raw.join(","),
        used.join(",")
    );
    eprintln!("dspark order s={}", s.join(","));
    // Every candidate, placed or not: token, parent (-1 = the anchor), q as its source gave it,
    // q after the offset, and its position in the best-first order (-1 = beyond the widest
    // width). With the path's rows and the target's pick at each accepted parent, this is what
    // the drafter's coverage and a reliability table by kind and depth are computed from.
    let mut ord = vec![-1_i64; n];
    for (r, &i) in p.order.iter().enumerate() {
        ord[i] = i64::try_from(r).unwrap_or(i64::MAX);
    }
    let used_q = |i: usize| -> String {
        let parent_q =
            usize::try_from(candidates[i].parent).map_or(1.0, |pi| p.big_q[pi]);
        if parent_q > 0.0 {
            format!("{:.4}", p.big_q[i] / parent_q)
        } else {
            "nan".to_string()
        }
    };
    let join = |it: &mut dyn Iterator<Item = String>| it.collect::<Vec<_>>().join(",");
    // `raw` at full precision: a fit of the acceptance model reads a sibling's share and the
    // pick's share through logarithms, and four decimals print most siblings as 0.
    eprintln!(
        "dspark cands tok={} par={} raw={} used={} ord={}",
        join(&mut candidates.iter().map(|c| c.token.to_string())),
        join(&mut candidates.iter().map(|c| c.parent.to_string())),
        join(&mut candidates.iter().map(|c| format!("{:e}", c.q))),
        join(&mut (0..n).map(used_q)),
        join(&mut ord.iter().map(ToString::to_string))
    );
}

/// THE ACCEPTANCE LABELS ONE VERIFY PRODUCES (design 11.3), as `(q, accepted)` pairs.
///
/// A node is labelled exactly when its PARENT is on the accepted path -- then the target saw
/// that position in its real context and either took this node or took something else:
///
/// ```text
///   parent accepted, node accepted     y = 1
///   parent accepted, node not          y = 0    a sibling won, or the path stopped here
///   parent REJECTED                    no label -- the target never judged this position
/// ```
///
/// `q` is the candidate's own, read through the row's `node`: the estimate as its source gave
/// it. The plan's S cannot supply it, because S is built from q AFTER the offset, and an offset
/// learner fed that q applies its own correction twice:
///
/// ```text
///   Correct: q -> the tree shifts it by b -> label (q, y) -> the learner shifts q by b, compares
///   Wrong:   q -> the tree shifts it by b -> label (q', y) -> the learner shifts q' by b again
/// ```
///
/// A row with no candidate behind it (`node` -1, a hand-built tree) contributes nothing.
///
/// This is where every level-1 and level-3 fit of design 11 gets its supervision, so it is one
/// function rather than a rule re-implemented per learner.
#[must_use]
pub fn acceptance_labels(
    parents: &[i32],
    node: &[i32],
    candidates: &[TreeCandidate],
    path: &[i32],
) -> Vec<(f64, bool)> {
    let mut on_path = vec![false; parents.len()];
    for &row in path {
        if let Ok(r) = usize::try_from(row) {
            if let Some(slot) = on_path.get_mut(r) {
                *slot = true;
            }
        }
    }
    let mut out = Vec::new();
    for row in 1..parents.len() {
        let Ok(parent) = usize::try_from(parents[row]) else {
            continue;
        };
        if !on_path.get(parent).copied().unwrap_or(false) {
            continue;
        }
        let Some(c) = node
            .get(row)
            .and_then(|&i| usize::try_from(i).ok())
            .and_then(|i| candidates.get(i))
        else {
            continue;
        };
        out.push((c.q, on_path[row]));
    }
    out
}

/// DESIGN 6.3'S BUDGET: the tree whose row count maximises `(1 + S(n)) / T(n)`.
///
/// ```text
///   S(n)  expected accepted tokens from the first n rows -- the sum of Q over the nodes
///         best-first would take, because a node joins the accepted path exactly when its
///         whole chain is accepted, which is what Q is
///   T(n)  the MEASURED time to verify and commit n rows (verify_cost.rs)
///   +1    the anchor, which always commits
/// ```
///
/// `widths` is `(rows, microseconds)` ascending, from the cost table's envelope. Only those
/// row counts are worth considering: T is flat inside a class while S grows, so the best n
/// always sits at a class's top, and a class a wider one beats on price is not offered.
///
/// `fixed_us` is what the round costs WHATEVER row count it verifies -- the drafter's
/// forward, the tree build, the append. Design 6.3 left it out ("the drafter's time is
/// already spent when n is chosen"), which is a sunk-cost argument: the objective is tokens
/// per second across rounds, and a narrower tree makes rounds shorter but more numerous,
/// each paying the drafter again. Measured on LFM2 without it, the budget took 8 rows in 50
/// of 64 rounds at a 1596-token context and read +2.2% ms/token against a fixed 16.
///
/// Ties go to the FEWER rows. Equal tokens per millisecond at a narrower tree is the same
/// throughput at lower latency, and it leaves the drafter's candidates for the next round.
///
/// THE PRICE OF A ROUND'S TIME. With `choice.rho` set, the width maximises
/// `(1 + S(n)) - rho (D + T(n))` instead (design 6.6, theory item 1). Both rules aim at tokens
/// per unit time across rounds; they differ in what a microsecond is worth:
///
/// ```text
///   ratio      widen iff  dS / dT > (1 + S(lo)) / (D + T(lo))   this round's own rate
///   long run   widen iff  dS / dT > rho                          the process's measured rate
/// ```
///
/// A good round demands too much of a wider tree under the ratio, and a poor round too little.
/// If rounds are alike the two agree; they part where rounds are uneven.
///
/// # Errors
/// As `best_first_tree`, or when `widths` is empty.
pub fn budgeted_tree(
    anchor: u32,
    candidates: &[TreeCandidate],
    stops: &[u32],
    choice: Choice<'_>,
    round: TreeRound<'_>,
) -> Result<DraftTree, String> {
    if choice.widths.is_empty() {
        return Err("the budget has no measured row count to choose from".into());
    }
    let max = choice
        .widths
        .iter()
        .map(|&(rows, _)| rows)
        .max()
        .unwrap_or(1);
    let mut p = plan(candidates, max, choice.offset, stops, round.shape)?;
    // S over the order best-first took: s[k] is the expected accepted tokens from k nodes, and
    // s_n[k] the part of it the n-gram's and agreed nodes carry.
    let (s, s_n) = prefix_sums(&p, round);
    let keep = choose_keep(&s, &s_n, p.order.len(), &choice, choice.rho);
    if trace_probe() {
        probe_choice(&s, &s_n, p.order.len(), &choice, keep);
    }
    if score_probe() {
        // Both rules on the same S, so an offline pass can count the rounds they part on
        // without re-deriving either.
        let other = if choice.rho.is_some() {
            choose_keep(&s, &s_n, p.order.len(), &choice, None)
        } else {
            keep
        };
        eprintln!(
            "dspark rule rho_tok_s={} kept={keep} ratio_kept={other}",
            choice
                .rho
                .map_or_else(|| "-".to_string(), |r| format!("{:.3}", r * 1e6)),
        );
        probe_ride(candidates, stops, &choice, round, keep)?;
    }
    let explored = round
        .explore
        .is_some_and(|labels| explore(&mut p, keep, candidates, labels, round.shape));
    emit_probed(anchor, candidates, &p, keep, round, explored)
}

/// FREE RIDE OR PAID RIDE (design 6.6), inside the caller's gate: the width this round kept, and the
/// width the same budget keeps with every n-gram-only node priced at 0. Equal widths with n-gram
/// rows in the tree are a free ride; a wider width is a paid one.
fn probe_ride(
    candidates: &[TreeCandidate],
    stops: &[u32],
    choice: &Choice<'_>,
    round: TreeRound<'_>,
    keep: usize,
) -> Result<(), String> {
    if !round.of.contains(&NodeSource::Ngram) {
        return Ok(());
    }
    let max = choice
        .widths
        .iter()
        .map(|&(rows, _)| rows)
        .max()
        .unwrap_or(1);
    let drafter_only: Vec<TreeCandidate> = candidates
        .iter()
        .zip(round.of)
        .map(|(c, s)| TreeCandidate {
            q: if *s == NodeSource::Ngram { 0.0 } else { c.q },
            ..*c
        })
        .collect();
    let p = plan(&drafter_only, max, choice.offset, stops, round.shape)?;
    let (s, _) = prefix_sums(&p, TreeRound::default());
    let zeros = vec![0.0; s.len()];
    let alone = choose_keep(&s, &zeros, p.order.len(), choice, choice.rho);
    eprintln!("dspark ride kept={keep} drafter_only={alone}");
    Ok(())
}

/// WHAT A WIDTH IS WORTH THIS ROUND: its expected tokens and its value against its time, under
/// the ratio rule (`rho` = `None`) or the long-run rule. One definition, read by the chooser and
/// by the cost table's split gate (`VerifyCost::refine`), so a width is explored only when it
/// could win the same comparison the chooser makes.
pub(crate) struct Valuer<'a> {
    s: &'a [f64],
    s_n: &'a [f64],
    order_len: usize,
    base_s: f64,
    base_n: f64,
    base_keep: usize,
    gain: f64,
    gain_n: f64,
    gain_below: (f64, f64),
    fixed_us: f64,
    rho: Option<f64>,
}

impl<'a> Valuer<'a> {
    pub(crate) fn new(
        s: &'a [f64],
        s_n: &'a [f64],
        order_len: usize,
        choice: &Choice<'_>,
        rho: Option<f64>,
    ) -> Self {
        // THE BASELINE WIDTH. Everything above it is a MARGINAL, and the marginal is the part
        // whose scale is learned: S is a decent per-node probability but a poor estimate of what
        // the extra rows buy (measured 0.59x at ctx 1596, 1.13x at 8444 -- opposite directions,
        // which is why no single constant corrects it).
        let base_rows = match choice.base_rows {
            0 => choice.widths.first().map_or(0, |&(rows, _)| rows),
            n => n,
        };
        let base_keep = base_rows.saturating_sub(1).min(order_len);
        Self {
            s,
            s_n,
            order_len,
            base_s: s.get(base_keep).copied().unwrap_or(0.0),
            base_n: s_n.get(base_keep).copied().unwrap_or(0.0),
            base_keep,
            gain: choice.gain,
            gain_n: choice.gain_n,
            gain_below: choice.gain_below,
            fixed_us: choice.fixed_us,
            rho,
        }
    }

    /// Expected tokens a round commits when it keeps `keep` nodes: each source's marginal by its
    /// own gain, design 6.6's `g_d dS_d(n) + g_n dS_n(n)`. Below the reference the marginal is
    /// negative -- the rows the width gives up -- and is scaled by the gains fitted on those rows.
    pub(crate) fn tokens(&self, keep: usize) -> f64 {
        let raw = self.s.get(keep).copied().unwrap_or(0.0);
        let raw_n = self.s_n.get(keep).copied().unwrap_or(0.0);
        let (d_n, d_all) = (raw_n - self.base_n, raw - self.base_s);
        let (g_d, g_n) = if keep < self.base_keep {
            self.gain_below
        } else {
            (self.gain, self.gain_n)
        };
        1.0 + self.base_s + g_d * (d_all - d_n) + g_n * d_n
    }

    pub(crate) fn value(&self, keep: usize, us: f64) -> f64 {
        let time = self.fixed_us.max(0.0) + us;
        self.rho
            .map_or_else(|| self.tokens(keep) / time, |r| self.tokens(keep) - r * time)
    }

    /// The node count a verify of `rows` rows keeps.
    pub(crate) fn keep(&self, rows: usize) -> usize {
        rows.saturating_sub(1).min(self.order_len)
    }

    /// Whether `rows` at `us` would clear the chooser's switching bar against `held` at
    /// `held_us`: the same margin the hysteresis applies, so the split gate asks exactly the
    /// question the chooser will.
    pub(crate) fn could_beat(&self, rows: usize, us: f64, held: usize, held_us: f64) -> bool {
        let (k, hk) = (self.keep(rows), self.keep(held));
        let (v, hv) = (self.value(k, us), self.value(hk, held_us));
        let bar = self
            .rho
            .map_or_else(|| hv * (1.0 + switch_margin()), |_| hv + switch_margin() * self.tokens(hk));
        v > bar
    }
}

/// The node count design 6.3's budget keeps, under the ratio rule (`rho` = `None`) or the
/// long-run rule.
fn choose_keep(
    s: &[f64],
    s_n: &[f64],
    order_len: usize,
    choice: &Choice<'_>,
    rho: Option<f64>,
) -> usize {
    let valuer = Valuer::new(s, s_n, order_len, choice, rho);
    let tokens = |keep: usize| valuer.tokens(keep);
    let value = |keep: usize, us: f64| valuer.value(keep, us);

    let mut best: Option<(f64, usize, usize)> = None;
    let mut incumbent: Option<(f64, usize)> = None;
    for &(rows, us) in choice.widths {
        if us <= 0.0 || rows < 2 {
            continue;
        }
        let keep = (rows - 1).min(order_len);
        let v = value(keep, us);
        if rows == choice.hold {
            incumbent = Some((v, keep));
        }
        // Strictly better wins; on a tie the earlier (narrower) width keeps the seat.
        if best.is_none_or(|(b, _, _)| v > b) {
            best = Some((v, keep, rows));
        }
    }

    // HYSTERESIS. Near the bar the per-round crossings are noise, and a policy that flips on
    // every crossing is a coin toss around the optimum -- which loses to simply staying put.
    // Measured: argmax lands +1.70% / -0.95% against the best fixed width at two contexts,
    // and the same rule with a margin lands -0.40% / -0.42%. So an alternative must BEAT the
    // incumbent by a margin, not merely tie it, and the incumbent's own score is the floor.
    //
    // The long-run value is a DIFFERENCE, so the same margin is stated on its scale: the
    // incumbent's expected tokens times the margin, which is what the ratio's relative margin
    // is when rho equals the incumbent's own ratio.
    match (best, incumbent) {
        (Some((bv, bk, brows)), Some((iv, ik))) if brows != choice.hold => {
            let bar = rho.map_or_else(
                || iv * (1.0 + switch_margin()),
                |_| iv + switch_margin() * tokens(ik),
            );
            if bv > bar { bk } else { ik }
        }
        (Some((_, bk, _)), _) => bk,
        (None, _) => order_len,
    }
}

/// How much better an alternative width must be before the budget leaves the one it is on.
///
/// 0.04 is measured: below it the policy still flips on noise (+0.40..+0.94% against the best
/// fixed width), above ~0.05 it never leaves its incumbent at all and simply BECOMES that
/// fixed width. `IMPARO_DSPARK_SWITCH` overrides it for an A/B.
fn switch_margin() -> f64 {
    static M: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("IMPARO_DSPARK_SWITCH")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| (0.0..=1.0).contains(v))
            .unwrap_or(0.04)
    })
}
pub trait DraftProvider {
    /// A tree over the drafted `chain`, or `None` to verify the chain. `stops` end the answer:
    /// a node carrying one can never be accepted, so a tree must not spend a row on it.
    fn tree_proposal(
        &mut self,
        _start: usize,
        _anchor: u32,
        _chain: &[u32],
        _stops: &[u32],
    ) -> Result<Option<DraftTree>, String> {
        Ok(None)
    }
    /// Commit a verified tree's accepted path. `next` is the target's pick after the path's last
    /// node -- with the path's own tokens, the target's pick at every accepted parent.
    fn commit_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        _path: &[i32],
        _next: u32,
    ) -> Result<(), String> {
        self.commit(start, inputs)
    }
    /// Whether the history takes `n` rows at `start` that no later round of this request drafts
    /// from: the reply's last round, and the one-token steps after drafting stopped. A provider
    /// whose history outlives the request takes them, so the next turn resumes where this
    /// stream ends. The default declines, and the cursor then stops capturing with the last
    /// round.
    fn keeps_rows(&self, _start: usize, _n: usize) -> bool {
        false
    }
    /// Puts a verified tree's accepted path into the history WITHOUT learning from the round:
    /// the commit for a round no later round drafts from (`keeps_rows`).
    fn keep_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        _path: &[i32],
    ) -> Result<(), String> {
        self.commit(start, inputs)
    }
    /// What a verify of `rows` rows COST, reported by the cursor that ran it.
    ///
    /// A provider that chooses its own row count needs the price of each count it could
    /// choose, and only the cursor is on both sides of the call. The time is wall time
    /// around `verify_greedy_tree`, which is where this round's target forward is.
    ///
    /// Not called for a chain block: a chain has no row count to choose.
    fn observe_verify(&mut self, _rows: usize, _elapsed: std::time::Duration) {}

    /// What the round cost APART from the verify: the drafter's forward and the tree build.
    /// It does not depend on the row count, and a chooser needs it -- the round pays it
    /// again every time, so a narrower tree does not save all of what it looks like it saves.
    fn observe_round_fixed(&mut self, _elapsed: std::time::Duration) {}

    /// A complete successful tree round, reported once after its history commit
    /// or tail keep. Failed verification/commit, abstention and chain fallback do
    /// not report a sample. Existing verify/fixed observers retain their ordering.
    fn observe_tree_round(&mut self, _tree: &DraftTree, _timing: &TreeRoundTiming) {}

    /// Opt-in observation of the existing chain, without changing its proposal,
    /// verification or commit. Unknown submission modes cannot teach a price.
    fn observes_chain_round(&self) -> bool { false }
    fn observe_chain_round(&mut self, _timing: &TreeRoundTiming) {}

    /// Target inputs per block, including its unconsumed anchor.
    fn block_size(&self) -> usize;
    /// Smallest remaining output budget worth a full physical verification.
    /// The default keeps the established fixed-block tail fallback.
    fn minimum_remaining(&self) -> usize {
        self.block_size()
    }
    fn initialize(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// Reuse only an adapter-proven committed prefix; ordinary providers start cold.
    fn initialize_at(&mut self, start: usize) -> Result<(), String> {
        if start != 0 {
            return Err("draft provider has no cached history".into());
        }
        self.initialize()
    }
    /// Idempotent observation switch. Never resets committed draft history.
    fn set_capture(&mut self, _enabled: bool) -> Result<(), String> {
        Ok(())
    }
    /// A provider can decline boundaries unsupported by its target feature contract.
    fn can_draft(&self, _start: usize, _remaining: usize) -> bool {
        true
    }
    /// Temporary boundary: captured single-token commits can reach a legal draft
    /// within remaining output budget and provider capacity. Default declines it.
    fn can_bridge(&self, _start: usize, _remaining: usize) -> bool {
        false
    }
    /// Exactly block_size-1 successor candidates; excludes the anchor. `target` is the model
    /// the provider is paired with: a drafter that keeps its caches in the target's grows them
    /// through it.
    fn draft(
        &mut self,
        target: &mut dyn Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String>;
    /// None declines only this round. Leave the target's committed state unchanged (a drafter
    /// may grow the cache and write above its filled rows) and keep provider capture/committed
    /// history ready for the ordinary M1 commit.
    /// Errors remain fatal; existing providers retain their fixed-block behavior.
    fn try_draft(
        &mut self,
        target: &mut dyn Model,
        start: usize,
        anchor: u32,
    ) -> Result<Option<Vec<u32>>, String> {
        self.draft(target, start, anchor).map(Some)
    }
    /// Commit target features for only these accepted inputs, excluding the bonus.
    fn commit(&mut self, _start: usize, _inputs: &[u32]) -> Result<(), String> {
        Ok(())
    }
    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Called only after the target has successfully verified and committed a tree.
/// Keeping the observation beside the fallible history commit prevents rejected
/// or partially committed rounds from becoming cost samples.
fn commit_observed_tree<D: DraftProvider + ?Sized>(
    provider: &mut D,
    tree: &DraftTree,
    path: &[i32],
    inputs: &[u32],
    next: u32,
    started: std::time::Instant,
    mut timing: TreeRoundTiming,
) -> Result<(), String> {
    if !timing.continuing && !provider.keeps_rows(timing.context, inputs.len()) {
        return Ok(());
    }
    let commit_at = std::time::Instant::now();
    if timing.continuing {
        provider.commit_tree(timing.context, inputs, path, next)?;
    } else {
        provider.keep_tree(timing.context, inputs, path)?;
    }
    timing.commit = commit_at.elapsed();
    timing.total = started.elapsed();
    provider.observe_tree_round(tree, &timing);
    Ok(())
}

/// Keep the chain's existing commit/tail behavior; observe only after success.
/// The same full-round clock as tree includes drafting and target state commit.
fn commit_observed_chain<D: DraftProvider + ?Sized>(
    provider: &mut D,
    inputs: &[u32],
    started: std::time::Instant,
    mut timing: TreeRoundTiming,
) -> Result<(), String> {
    if !timing.continuing && !provider.keeps_rows(timing.context, inputs.len()) {
        return Ok(());
    }
    let commit_at = std::time::Instant::now();
    provider.commit(timing.context, inputs)?;
    timing.commit = commit_at.elapsed();
    timing.total = started.elapsed();
    provider.observe_chain_round(&timing);
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    Limit,
    StopToken(u32),
}
#[derive(Debug)]
pub struct Continuation {
    pub tokens: Vec<u32>,
    pub consumed: usize,
    pub next_token: u32,
    pub reason: StopReason,
    pub draft_calls: usize,
    pub verified_blocks: usize,
    pub sequential_steps: usize,
    pub drafted: DraftTally,
}
/// Incremental version of the same greedy block algorithm. Execution can lead
/// delivery: consumed is actual target state; emitted is returned token count.
/// An error poisons the cursor because a target commit cannot be undone by hiding it.
#[allow(clippy::struct_excessive_bools)]
pub struct GreedyCursor {
    pub consumed: usize,
    pub next_token: u32,
    pub reason: StopReason,
    pub draft_calls: usize,
    pub verified_blocks: usize,
    pub sequential_steps: usize,
    /// Draft tokens verified and accepted, by the source that proposed each.
    pub drafted: DraftTally,
    pub emitted: usize,
    /// Diagnostic only: index is accepted input count, including the anchor.
    pub verified_consumed_histogram: Option<Vec<usize>>,
    start: usize,
    pos: usize,
    limit: usize,
    generated: usize,
    block: usize,
    minimum_remaining: usize,
    vocab: u32,
    stops: Vec<u32>,
    pending: std::collections::VecDeque<u32>,
    speculative: bool,
    started: bool,
    /// Whether the provider captures the target's rows: from the first refill while rounds
    /// draft, and after drafting stops for as long as the provider keeps the rows.
    capturing: bool,
    terminal: bool,
    capture_closed: bool,
    poison: Option<String>,
}
impl GreedyCursor {
    pub fn new<D: DraftProvider + ?Sized>(
        target: &dyn Model,
        provider: &D,
        start: usize,
        first: u32,
        limit: usize,
        stops: &[u32],
    ) -> Result<Self, String> {
        let block = provider.block_size();
        if block < 2 {
            return Err("draft block must include anchor and a proposal".into());
        }
        if start != target.kv_runtime().filled || start > target.kv_runtime().capacity {
            return Err("continuation start differs from target state".into());
        }
        let minimum_remaining = provider.minimum_remaining();
        if minimum_remaining == 0 || minimum_remaining > block {
            return Err("invalid draft minimum budget".into());
        }
        let vocab = target.plan().config.vocab_size;
        if first >= vocab {
            return Err("anchor exceeds vocabulary".into());
        }
        let mut pending = std::collections::VecDeque::new();
        let mut reason = StopReason::Limit;
        let mut terminal = limit == 0;
        if !terminal {
            if stops.contains(&first) {
                reason = StopReason::StopToken(first);
                terminal = true;
            } else {
                let end = start
                    .checked_add(limit - 1)
                    .ok_or("continuation position overflow")?;
                if end > target.kv_runtime().capacity {
                    return Err("output budget exceeds target capacity".into());
                }
                pending.push_back(first);
                terminal = limit == 1;
            }
        }
        let verified_consumed_histogram =
            if std::env::var("IMPARO_LAB_DRAFT_ACCEPTANCE_TRACE").as_deref() == Ok("1")
            {
                Some(vec![
                    0;
                    block
                        .checked_add(1)
                        .ok_or("draft histogram size overflow")?
                ])
            } else {
                None
            };
        Ok(Self {
            consumed: 0,
            next_token: first,
            reason,
            draft_calls: 0,
            verified_blocks: 0,
            sequential_steps: 0,
            drafted: DraftTally::default(),
            emitted: 0,
            verified_consumed_histogram,
            start,
            pos: start,
            limit,
            generated: pending.len(),
            block,
            minimum_remaining,
            vocab,
            stops: stops.to_vec(),
            pending,
            speculative: true,
            started: false,
            capturing: false,
            terminal,
            capture_closed: false,
            poison: None,
        })
    }
    pub fn is_finished(&self) -> bool {
        self.poison.is_none() && self.terminal && self.pending.is_empty()
    }
    /// Every token the last round verified has been handed out: the target's cache holds all
    /// of them but the last, which is the next round's anchor. A plain decode can take over
    /// here, feeding that token.
    #[must_use]
    pub fn between_rounds(&self) -> bool {
        self.pending.is_empty()
    }
    /// Pending output is drained without a forward, draft or history update.
    pub fn next<D: DraftProvider + ?Sized>(
        &mut self,
        target: &mut dyn Model,
        provider: &mut D,
    ) -> Result<Option<u32>, String> {
        if let Some(e) = &self.poison {
            return Err(e.clone());
        }
        match self.next_inner(target, provider) {
            Ok(t) => Ok(t),
            Err(e) => {
                self.pending.clear();
                self.terminal = true;
                let e = match provider.set_capture(false) {
                    Ok(()) => {
                        self.capture_closed = true;
                        self.capturing = false;
                        e
                    }
                    Err(c) => format!("{e}; disable draft capture: {c}"),
                };
                self.poison = Some(e.clone());
                Err(e)
            }
        }
    }
    fn next_inner<D: DraftProvider + ?Sized>(
        &mut self,
        target: &mut dyn Model,
        provider: &mut D,
    ) -> Result<Option<u32>, String> {
        if self.pending.is_empty() && !self.terminal {
            self.refill(target, provider)?;
        }
        if self.terminal && !self.capture_closed {
            provider.set_capture(false)?;
            self.capture_closed = true;
            self.capturing = false;
        }
        let token = self.pending.pop_front();
        if token.is_some() {
            self.emitted += 1;
        }
        Ok(token)
    }
    fn refill<D: DraftProvider + ?Sized>(
        &mut self,
        target: &mut dyn Model,
        provider: &mut D,
    ) -> Result<(), String> {
        // A ROUND'S FORWARDS ARE FEW ROWS -- the drafter's block, the verify (chain or tree), the
        // drafter's append -- so they encode on the backend's fast rows route, which reads each
        // weight once for all of them. Left to the prefill route, a 9-row drafter block read
        // LFM2.5-8B-A1B's Q6_K lm head once per row and a 10-row verify decoded a padded tile:
        // drafter 21.5 -> 11.0 ms, verify 44.3 -> 26.9 ms a round.
        // Keep this projection hint separate from independent-slot co-batching:
        // CUDA's co-batch owner excludes verification graphs and tree contexts.
        let be = crate::backend::active().filter(|be| be.set_speculative_rows(true));
        let round = self.refill_round(target, provider);
        if let Some(be) = be {
            be.set_speculative_rows(false);
        }
        round
    }
    fn refill_round<D: DraftProvider + ?Sized>(
        &mut self,
        target: &mut dyn Model,
        provider: &mut D,
    ) -> Result<(), String> {
        if target.kv_runtime().filled != self.pos {
            return Err("continuation target position changed outside cursor".into());
        }
        if provider.block_size() != self.block {
            return Err("draft block size changed during continuation".into());
        }
        if !self.started {
            provider.set_capture(true)?;
            self.started = true;
            self.capturing = true;
        }
        let remaining = self.limit - self.generated;
        let anchor = self.next_token;
        let can_run = self.speculative
            && remaining >= self.minimum_remaining
            && provider.can_draft(self.pos, remaining);
        if self.speculative
            && !can_run
            && (remaining < self.minimum_remaining
                || !provider.can_bridge(self.pos, remaining))
        {
            // Drafting ends here; capture goes on only while the provider keeps the steps' rows.
            if !provider.keeps_rows(self.pos, 1) {
                provider.set_capture(false)?;
                self.capturing = false;
            }
            self.speculative = false;
        }
        let mut input = Vec::new();
        let mut proposed_tree = None;
        // THE ROUND'S FIXED COST, from the drafter's forward through the tree build: every
        // round pays it whatever row count it then verifies, so the budget divides by it.
        let fixed_at = std::time::Instant::now();
        let mut probe_draft = std::time::Duration::ZERO;
        if can_run {
            let proposal = provider.try_draft(target, self.pos, anchor)?;
            if round_probe() {
                probe_draft = fixed_at.elapsed();
                // Under IMPARO_PROF=1, the drafter's kernel classes alone (the log resets them).
                crate::host::prof_log("round draft", probe_draft.as_secs_f64() * 1e3);
            }
            self.draft_calls += 1;
            if let Some(proposal) = proposal {
                if proposal.len() != self.block - 1
                    || proposal.iter().any(|&x| x >= self.vocab)
                {
                    return Err("invalid draft proposal shape/token".into());
                }
                proposed_tree =
                    provider.tree_proposal(self.pos, anchor, &proposal, &self.stops)?;
                if let Some(tree) = proposed_tree.as_ref() {
                    if !target.prepare_greedy_tree(tree, self.pos)? {
                        proposed_tree = None;
                    }
                }
                // A chain proposal ends before its first stop token. Verifying the stop token
                // as an input would let the verify accept it and emit tokens past the end of the
                // answer; with the cut, the verify's pick at the last kept row still ends the
                // answer when the target stops there. Speculation stays on.
                let mut proposal = proposal;
                let cut = if proposed_tree.is_none() {
                    proposal.iter().position(|x| self.stops.contains(x))
                } else {
                    None
                };
                if let Some(cut) = cut {
                    proposal.truncate(cut);
                }
                // A chain of at most N draft tokens: llama.cpp's DSpark configuration
                // (`--spec-draft-n-max`, 3 by default), so the two engines can be compared
                // on the same algorithm.
                if proposed_tree.is_none() {
                    if let Some(max) = chain_max() {
                        proposal.truncate(max);
                    }
                }
                if proposed_tree.is_some() || !proposal.is_empty() {
                    input.push(anchor);
                    input.extend(proposal);
                }
            } else if self.verified_consumed_histogram.is_some() {
                eprintln!("[imparo] draft-abstain start={}", self.pos);
            }
        }
        let mut round_fixed = std::time::Duration::ZERO;
        if can_run {
            round_fixed = fixed_at.elapsed();
            provider.observe_round_fixed(round_fixed);
        }
        if self.speculative && can_run && !input.is_empty() {
            let old = self.pos;
            let tree = proposed_tree;
            // A proposal cut at a stop token verifies fewer rows than a block. A TREE's path is
            // bounded by the tree, not by the drafter's block: an n-gram chain that reads a copied
            // passage on past the block is accepted as far as the target agrees with it. Bounding it
            // by `input` (the anchor and the block) capped every round at the block, whatever the
            // tree held.
            let commit_limit = remaining.min(
                tree.as_ref()
                    .map_or(input.len(), |t| t.tokens.len().max(input.len())),
            );
            let mut selected_path = None;
            let probe_rows = tree.as_ref().map_or(input.len(), |t| t.tokens.len());
            let probe_verify_at = round_probe().then(std::time::Instant::now);
            let mut selected_tree = None;
            let mut chain_timing = None;
            let (v, accepted_inputs) = if let Some(tree) = tree {
                // The clock spans the whole verify because that is what the provider's
                // choice buys or spends: the rows go in, the accepted path comes back, and
                // reading the path is what makes the GPU work land inside this call.
                let at = std::time::Instant::now();
                let result =
                    target.verify_greedy_tree(&tree, old, commit_limit, &self.stops)?;
                let verify_elapsed = at.elapsed();
                provider.observe_verify(tree.tokens.len(), verify_elapsed);
                self.drafted.tree(&tree, &result.path);
                if score_probe() {
                    // The stop set once, so an offline pass can count stop-token nodes.
                    static STOPS_SAID: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !STOPS_SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        let stops: Vec<String> =
                            self.stops.iter().map(ToString::to_string).collect();
                        eprintln!("dspark stops={}", stops.join(","));
                    }
                    let path: Vec<String> =
                        result.path.iter().map(ToString::to_string).collect();
                    // `next` is the target's pick at the path's last node: with it, every
                    // accepted parent's pick is known, placed candidate or not.
                    eprintln!(
                        "dspark path fixed_us={:.1} rows={} path={} next={}",
                        round_fixed.as_secs_f64() * 1e6,
                        tree.tokens.len(),
                        path.join(","),
                        result.next_token
                    );
                }
                let inputs: Vec<u32> = result
                    .path
                    .iter()
                    .map(|&i| tree.tokens[i as usize])
                    .collect();
                let v = crate::GreedyVerification {
                    consumed: inputs.len(),
                    next_token: result.next_token,
                };
                selected_tree = Some((tree, result.submission, verify_elapsed));
                selected_path = Some(result.path);
                (v, inputs)
            } else {
                let at = provider.observes_chain_round().then(|| {
                    target.state_mut().tree_submission = TreeSubmission::Unknown;
                    std::time::Instant::now()
                });
                // A CHAIN IS THE TREE WITH ONE CHILD PER NODE, and the tree verify runs the row
                // layout's attention (packed heads, key split) where the block verify runs the
                // causal batch's. Measured before this route (Qwen3-4B, 8,444 keys): an 8-row
                // chain verified in 74.7 ms against 35.0 ms for an 8-row tree. A chain laid out
                // as a tree computes the chain's rows bit for bit, so only the time moves; the
                // round still commits and learns as a chain (no path handed to the provider).
                let chain_tree = (input.len() >= 2 && chain_route_tree())
                    .then(|| {
                        let parents = (0..input.len())
                            .map(|i| i32::try_from(i).map(|p| p - 1))
                            .collect::<Result<Vec<i32>, _>>()
                            .map_err(|_| "chain row overflow".to_string())?;
                        Ok::<_, String>(DraftTree::plain(input.clone(), parents))
                    })
                    .transpose()?;
                let chain_tree = match chain_tree {
                    Some(t) if target.prepare_greedy_tree(&t, old)? => Some(t),
                    _ => None,
                };
                let mut tree_submission = None;
                let v = if let Some(t) = chain_tree {
                    let result = target.verify_greedy_tree(&t, old, commit_limit, &self.stops)?;
                    // The accepted path of a chain is its first rows in order.
                    if result.path.iter().enumerate().any(|(i, &r)| usize::try_from(r) != Ok(i)) {
                        return Err("chain verified as a tree returned a non-prefix path".into());
                    }
                    tree_submission = Some(result.submission);
                    crate::GreedyVerification {
                        consumed: result.path.len(),
                        next_token: result.next_token,
                    }
                } else if commit_limit < input.len() {
                    target.verify_greedy_block_limited(&input, old, commit_limit)?
                } else {
                    target.verify_greedy_block(&input, old)?
                };
                chain_timing = at.map(|at| {
                    (at.elapsed(), tree_submission.unwrap_or(target.state().tree_submission))
                });
                let inputs = input
                    .get(..v.consumed)
                    .ok_or("invalid accepted range")?
                    .to_vec();
                self.drafted
                    .chain(input.len() - 1, v.consumed.saturating_sub(1));
                (v, inputs)
            };
            let probe_verify = probe_verify_at.map(|t| t.elapsed());
            if let Some(t) = probe_verify {
                crate::host::prof_log("round verify", t.as_secs_f64() * 1e3);
            }
            if v.consumed == 0
                || v.consumed > commit_limit
                || v.next_token >= self.vocab
            {
                return Err("invalid verification result".into());
            }
            let pos = old.checked_add(v.consumed).ok_or("verification overflow")?;
            if target.kv_runtime().filled != pos {
                return Err("verification committed position mismatch".into());
            }
            if let Some(histogram) = self.verified_consumed_histogram.as_mut() {
                // Sized for a chain round; a tree round's path can run past the block.
                if histogram.len() <= v.consumed {
                    histogram.resize(v.consumed + 1, 0);
                }
                histogram[v.consumed] += 1;
                eprintln!(
                    "[imparo] draft-acceptance start={old} consumed={}",
                    v.consumed
                );
            }
            self.verified_blocks += 1;
            self.pos = pos;
            self.consumed = pos - self.start;
            self.next_token = v.next_token;
            self.pending
                .extend(accepted_inputs[1..v.consumed].iter().copied());
            self.generated += v.consumed - 1;
            if self.stops.contains(&v.next_token) {
                self.reason = StopReason::StopToken(v.next_token);
                self.terminal = true;
            } else {
                self.pending.push_back(v.next_token);
                self.generated += 1;
                self.terminal = self.generated == self.limit;
            }
            // A round that a later round drafts from is committed and learned from. A round no
            // later round drafts from -- the reply's last, or the last before drafting stops --
            // goes into the history unlearned when the provider keeps it, so the history ends
            // where the stream ends and the next turn can resume at the stream's grid point.
            let probe_commit_at = round_probe().then(std::time::Instant::now);
            let continuing = self.drafts_again(provider, pos);
            if let (Some((tree, submission, verify_elapsed)), Some(path)) =
                (selected_tree, selected_path)
            {
                commit_observed_tree(
                    provider,
                    &tree,
                    &path,
                    &accepted_inputs,
                    v.next_token,
                    fixed_at,
                    TreeRoundTiming {
                        context: old,
                        rows: tree.tokens.len(),
                        accepted: v.consumed,
                        fixed: round_fixed,
                        verify: verify_elapsed,
                        commit: std::time::Duration::ZERO,
                        total: std::time::Duration::ZERO,
                        submission,
                        continuing,
                    },
                )?;
            } else if let Some((verify, submission)) = chain_timing {
                commit_observed_chain(provider, &accepted_inputs, fixed_at, TreeRoundTiming {
                    context: old, rows: input.len(), accepted: v.consumed,
                    fixed: round_fixed, verify, submission, continuing,
                    commit: std::time::Duration::ZERO,
                    total: std::time::Duration::ZERO,
                })?;
            } else if continuing || provider.keeps_rows(old, v.consumed) {
                provider.commit(old, &accepted_inputs)?;
            }
            if let (Some(verify), Some(commit_at)) = (probe_verify, probe_commit_at) {
                eprintln!(
                    "dspark round start={old} rows={probe_rows} consumed={} draft_us={:.0} \
                     tree_us={:.0} verify_us={:.0} commit_us={:.0}",
                    v.consumed,
                    probe_draft.as_secs_f64() * 1e6,
                    round_fixed.saturating_sub(probe_draft).as_secs_f64() * 1e6,
                    verify.as_secs_f64() * 1e6,
                    commit_at.elapsed().as_secs_f64() * 1e6
                );
            }
        } else {
            let old = self.pos;
            let next = target.forward_next(anchor, old)?;
            let pos = old.checked_add(1).ok_or("decode overflow")?;
            if next >= self.vocab || target.kv_runtime().filled != pos {
                return Err("invalid sequential continuation result".into());
            }
            self.pos = pos;
            self.consumed = pos - self.start;
            self.next_token = next;
            self.sequential_steps += 1;
            if self.stops.contains(&next) {
                self.reason = StopReason::StopToken(next);
                self.terminal = true;
            } else {
                self.pending.push_back(next);
                self.generated += 1;
                self.terminal = self.generated == self.limit;
            }
            if self.capturing {
                if (self.speculative && self.drafts_again(provider, pos))
                    || provider.keeps_rows(old, 1)
                {
                    provider.commit(old, &[anchor])?;
                } else if !self.speculative {
                    // The provider stopped keeping rows, and a later row cannot follow a
                    // missing one: the rest of the reply decodes without capture.
                    provider.set_capture(false)?;
                    self.capturing = false;
                }
            }
        }
        Ok(())
    }
    /// Whether a round drafts after `pos`: the rows up to `pos` must then be in the history,
    /// because that round's drafter reads them.
    fn drafts_again<D: DraftProvider + ?Sized>(
        &self,
        provider: &D,
        pos: usize,
    ) -> bool {
        let remaining = self.limit - self.generated;
        !self.terminal
            && remaining >= self.minimum_remaining
            && (provider.can_draft(pos, remaining)
                || provider.can_bridge(pos, remaining))
    }
}
/// Collect the same cursor; callers initialize and commit prompt history first.
pub fn continue_greedy<D: DraftProvider + ?Sized>(
    target: &mut dyn Model,
    provider: &mut D,
    start: usize,
    first: u32,
    limit: usize,
    stops: &[u32],
) -> Result<Continuation, String> {
    let mut cursor =
        match GreedyCursor::new(&*target, provider, start, first, limit, stops) {
            Ok(c) => c,
            Err(e) => {
                return match provider.set_capture(false) {
                    Ok(()) => Err(e),
                    Err(c) => Err(format!("{e}; disable draft capture: {c}")),
                };
            }
        };
    let mut tokens = Vec::new();
    while let Some(t) = cursor.next(target, provider)? {
        tokens.push(t);
    }
    Ok(Continuation {
        tokens,
        consumed: cursor.consumed,
        next_token: cursor.next_token,
        reason: cursor.reason,
        draft_calls: cursor.draft_calls,
        verified_blocks: cursor.verified_blocks,
        sequential_steps: cursor.sequential_steps,
        drafted: cursor.drafted,
    })
}

/// The Gemma4 MTP drafter a pairing manifest names, admitted for `target`: the manifest must name this
/// target, and each file must be the one paired (its recorded stamp, or its SHA-256 when the
/// stamp moved or was not recorded; `imparo_gguf::pairing::verify`). Returns the manifest and
/// the drafter's path; the caller maps the two files with `Weights::open_with_appended`.
///
/// # Errors
/// When the manifest cannot be read or names another target, or a file is not the one paired.
#[cfg(feature = "cuda-speculative")]
pub(crate) fn admit_pairing(
    manifest: &std::path::Path,
    target: &std::path::Path,
) -> Result<(serde_json::Value, std::path::PathBuf), String> {
    use imparo_gguf::pairing::{Checked, verify};
    let v: serde_json::Value = serde_json::from_slice(
        &std::fs::read(manifest).map_err(|e| format!("{}: {e}", manifest.display()))?,
    )
    .map_err(|e| format!("{}: {e}", manifest.display()))?;
    let text = |k: &str| {
        v[k].as_str()
            .ok_or_else(|| format!("pairing manifest has no {k}"))
    };
    let n = |k: &str| {
        v[k].as_u64()
            .ok_or_else(|| format!("pairing manifest has no {k}"))
    };
    let canonical = |p: &std::path::Path| {
        p.canonicalize()
            .map_err(|e| format!("{}: {e}", p.display()))
    };
    if canonical(std::path::Path::new(text("target_path")?))? != canonical(target)? {
        return Err(format!(
            "the pairing names {} as its target, not {}",
            text("target_path")?,
            target.display()
        ));
    }
    let draft = std::path::PathBuf::from(text("draft_path")?);
    let t0 = std::time::Instant::now();
    let mut hashed = Vec::new();
    for (key, name, path) in [
        ("target", "target", target),
        ("draft", "drafter", draft.as_path()),
    ] {
        let how = verify(
            path,
            n(&format!("{key}_bytes"))?,
            text(&format!("{key}_sha256"))?,
            v[format!("{key}_stamp")].as_str(),
        )?;
        if how == Checked::Hashed {
            hashed.push(name);
        }
    }
    if hashed.is_empty() {
        eprintln!(
            "[imparo] draft pairing: target and drafter unchanged since pairing (file stamps)"
        );
    } else {
        eprintln!(
            "[imparo] draft pairing: {} hashed in {:.0} ms (no stamp, or a moved one)",
            hashed.join(" and "),
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
    Ok((v, draft))
}

/// Where the paired drafter's file sits in `weights`, checked to be `draft`'s: the offsets a
/// drafter reads its tensors at.
///
/// # Errors
/// When `weights` maps no drafter, or another file than `draft`.
#[cfg(feature = "speculative")]
pub(crate) fn drafter_offset(
    weights: &crate::weights::Weights,
    draft: &std::path::Path,
) -> Result<u64, String> {
    match weights.appended() {
        Some(a) if a.path == draft => Ok(a.offset),
        Some(a) => Err(format!(
            "the weight mapping holds {} after the target, the pairing names {}",
            a.path.display(),
            draft.display()
        )),
        None => Err(
            "the weight mapping holds no drafter: map the pair with Weights::open_with_appended"
                .into(),
        ),
    }
}

/// Source selection belongs to model adapters, not the token scheduling algorithm.
#[cfg(feature = "speculative")]
pub enum DraftSpec {
    Dspark(crate::dspark::Pairing),
    #[cfg(feature = "cuda-speculative")]
    GemmaMtp(crate::gemma4_mtp::Pairing),
}
#[cfg(feature = "speculative")]
pub type DraftRun<'a> =
    dyn FnMut(&mut dyn Model, &mut dyn DraftProvider) -> Result<(), String> + 'a;
#[cfg(feature = "speculative")]
impl DraftSpec {
    pub fn can_start(&self, start: usize, limit: usize, capacity: usize) -> bool {
        match self {
            Self::Dspark(pair) => pair.can_start(start, limit, capacity),
            #[cfg(feature = "cuda-speculative")]
            Self::GemmaMtp(pair) => pair.can_start(start, limit, capacity),
        }
    }

    /// The drafter's tensor spans in the paired mapping, for the weight placement. The
    /// Gemma4 MTP provider owns its buffers; both drafters declare weight residency here.
    ///
    /// # Errors
    /// When the drafter file does not read against this target's mapping.
    pub fn appended_spans(
        &self,
        weights: &crate::weights::Weights,
        target_layer_count: u32,
    ) -> Result<Vec<(u64, u64)>, String> {
        match self {
            Self::Dspark(pair) => Ok(pair
                .descriptor(weights, target_layer_count)?
                .appended_spans()),
            #[cfg(feature = "cuda-speculative")]
            Self::GemmaMtp(pair) => pair.appended_spans(weights),
        }
    }

    /// What the model's plan must carry for this drafter before the device is enabled: its
    /// caches, attention dims and feature rows are sized from the plan. `None` for a drafter that
    /// places its own buffers.
    ///
    /// # Errors
    /// When the drafter file does not read against this target's mapping.
    pub fn drafter_plan(
        &self,
        weights: &crate::weights::Weights,
        target_layer_count: u32,
    ) -> Result<Option<crate::DrafterPlan>, String> {
        match self {
            Self::Dspark(pair) => {
                let descriptor = pair.descriptor(weights, target_layer_count)?;
                // Pairing::run_cached selects the native CUDA session in this build.
                // It owns separate KV/features; adding them to the target plan would
                // allocate them twice and alter its persisted state geometry.
                if cfg!(feature = "cuda-speculative") {
                    Ok(None)
                } else {
                    Ok(Some(descriptor.drafter_plan()))
                }
            }
            #[cfg(feature = "cuda-speculative")]
            Self::GemmaMtp(_) => Ok(None),
        }
    }

    pub fn can_resume(&self, prompt: &[u32], start: usize) -> bool {
        match self {
            Self::Dspark(pair) => pair.can_resume(prompt, start),
            #[cfg(feature = "cuda-speculative")]
            Self::GemmaMtp(pair) => pair.can_resume(prompt, start),
        }
    }
    /// Whether a request whose prefill starts at `start` can draft. On this build a DSpark history
    /// that does not cover the restore point is rebuilt from it (design 5.5), so every request
    /// drafts. The CUDA path resumes only its own parked history: the request must continue the
    /// conversation that parked it (`same_conversation`) and agree with that history below
    /// `start`.
    pub fn can_draft_from(
        &self,
        prompt: &[u32],
        start: usize,
        same_conversation: bool,
    ) -> bool {
        #[cfg(not(feature = "cuda-speculative"))]
        {
            let _ = (prompt, start, same_conversation);
            match self {
                Self::Dspark(_) => true,
            }
        }
        #[cfg(feature = "cuda-speculative")]
        {
            start == 0 || (same_conversation && self.can_resume(prompt, start))
        }
    }
    pub(crate) fn run_cached<A: crate::Architecture>(
        &self,
        target: &mut crate::Workflow<A>,
        prompt: &[u32],
        start: usize,
        run: &mut DraftRun<'_>,
    ) -> Result<bool, String> {
        match self {
            Self::Dspark(pair) => pair.run_cached(target, prompt, start, run),
            #[cfg(feature = "cuda-speculative")]
            Self::GemmaMtp(pair) => pair.run_cached(target, prompt, start, run),
        }
    }
    pub(crate) fn run<A: crate::Architecture>(
        &self,
        target: &mut crate::Workflow<A>,
        run: &mut DraftRun<'_>,
    ) -> Result<(), String> {
        match self {
            Self::Dspark(pair) => pair.run(target, run),
            #[cfg(feature = "cuda-speculative")]
            Self::GemmaMtp(pair) => pair.run(target, run),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TreeCandidate, acceptance_labels, best_first_tree, budgeted_tree};

    #[derive(Default)]
    struct RoundObserver {
        keep_tail: bool,
        fail: bool,
        events: Vec<&'static str>,
        timings: Vec<super::TreeRoundTiming>,
    }

    impl super::DraftProvider for RoundObserver {
        fn block_size(&self) -> usize {
            3
        }
        fn draft(
            &mut self,
            _target: &mut dyn crate::Model,
            _start: usize,
            _anchor: u32,
        ) -> Result<Vec<u32>, String> {
            unreachable!("the fixture starts after successful target verification")
        }
        fn keeps_rows(&self, _start: usize, _n: usize) -> bool {
            self.keep_tail
        }
        fn commit(&mut self, start: usize, inputs: &[u32]) -> Result<(), String> {
            assert_eq!((start, inputs), (512, &[10, 11][..]));
            self.events.push("chain-commit");
            if self.fail { Err("history commit failed".into()) } else { Ok(()) }
        }
        fn observe_chain_round(&mut self, timing: &super::TreeRoundTiming) {
            self.events.push("chain-observe");
            self.timings.push(*timing);
        }
        fn commit_tree(
            &mut self,
            start: usize,
            inputs: &[u32],
            path: &[i32],
            next: u32,
        ) -> Result<(), String> {
            assert_eq!((start, inputs, path, next), (512, &[10, 11][..], &[0, 1][..], 99));
            self.events.push("commit");
            if self.fail { Err("history commit failed".into()) } else { Ok(()) }
        }
        fn keep_tree(
            &mut self,
            start: usize,
            inputs: &[u32],
            path: &[i32],
        ) -> Result<(), String> {
            assert_eq!((start, inputs, path), (512, &[10, 11][..], &[0, 1][..]));
            self.events.push("keep");
            if self.fail { Err("history keep failed".into()) } else { Ok(()) }
        }
        fn observe_tree_round(
            &mut self,
            tree: &super::DraftTree,
            timing: &super::TreeRoundTiming,
        ) {
            assert_eq!(tree.tokens, [10, 11, 12]);
            self.events.push("observe");
            self.timings.push(*timing);
        }
    }

    fn observe_round(
        provider: &mut RoundObserver,
        continuing: bool,
    ) -> Result<(), String> {
        use std::time::{Duration, Instant};
        super::commit_observed_tree(
            provider,
            &super::DraftTree::plain(vec![10, 11, 12], vec![-1, 0, 1]),
            &[0, 1],
            &[10, 11],
            99,
            Instant::now() - Duration::from_millis(3),
            super::TreeRoundTiming {
                context: 512,
                rows: 3,
                accepted: 2,
                fixed: Duration::from_millis(1),
                verify: Duration::from_millis(2),
                commit: Duration::ZERO,
                total: Duration::ZERO,
                submission: super::TreeSubmission::Replayed,
                continuing,
            },
        )
    }

    #[test]
    fn tree_round_is_observed_once_after_successful_history_commit() {
        let mut provider = RoundObserver::default();
        observe_round(&mut provider, true).unwrap();
        assert_eq!(provider.events, ["commit", "observe"]);
        assert_eq!(provider.timings.len(), 1);
        let timing = provider.timings[0];
        assert_eq!((timing.context, timing.rows, timing.accepted), (512, 3, 2));
        assert_eq!(timing.submission, super::TreeSubmission::Replayed);
        assert!(timing.continuing);
        assert!(timing.total >= timing.fixed + timing.verify + timing.commit);
    }

    #[test]
    fn tree_round_kept_tail_is_censored_and_unkept_tail_has_no_sample() {
        let mut provider = RoundObserver { keep_tail: true, ..RoundObserver::default() };
        observe_round(&mut provider, false).unwrap();
        assert_eq!(provider.events, ["keep", "observe"]);
        assert_eq!(provider.timings.len(), 1);
        assert!(!provider.timings[0].continuing);

        let mut provider = RoundObserver::default();
        observe_round(&mut provider, false).unwrap();
        assert!(provider.events.is_empty());
        assert!(provider.timings.is_empty());
    }

    #[test]
    fn tree_round_failed_history_commit_or_keep_has_no_sample() {
        for continuing in [true, false] {
            let mut provider = RoundObserver {
                keep_tail: true,
                fail: true,
                ..RoundObserver::default()
            };
            assert!(observe_round(&mut provider, continuing).is_err());
            assert_eq!(provider.events, [if continuing { "commit" } else { "keep" }]);
            assert!(provider.timings.is_empty());
        }
    }

    #[test]
    fn chain_cost_observation_preserves_commit_and_censors_tail_or_failure() {
        use std::time::{Duration, Instant};
        for (continuing, keep_tail, fail, commits, observes) in [
            (true, false, false, true, true),
            (false, true, false, true, true),
            (false, false, false, false, false),
            (true, false, true, true, false),
            (false, true, true, true, false),
        ] {
            let mut provider = RoundObserver { keep_tail, fail, ..RoundObserver::default() };
            let result = super::commit_observed_chain(&mut provider, &[10, 11],
                Instant::now() - Duration::from_millis(3), super::TreeRoundTiming {
                    context: 512, rows: 3, accepted: 2, continuing,
                    fixed: Duration::from_millis(1), verify: Duration::from_millis(2),
                    commit: Duration::ZERO, total: Duration::ZERO,
                    submission: super::TreeSubmission::Replayed,
                });
            assert_eq!(result.is_err(), fail && commits);
            assert_eq!(provider.events.contains(&"chain-commit"), commits);
            assert_eq!(provider.events.contains(&"chain-observe"), observes);
            assert_eq!(provider.timings.len(), usize::from(observes));
            for timing in provider.timings {
                assert_eq!(timing.continuing, continuing);
                assert_eq!((timing.rows, timing.accepted), (3, 2));
                assert!(timing.total >= timing.fixed + timing.verify + timing.commit);
            }
        }
    }

    /// `budgeted_tree` at anchor 5 with no stop tokens, no offset and an untrained gain; the
    /// per-round ratio unless `rho` is given.
    fn budget(
        cands: &[TreeCandidate],
        widths: &[(usize, f64)],
        fixed_us: f64,
        hold: usize,
        rho: Option<f64>,
    ) -> Result<super::DraftTree, String> {
        let choice = super::Choice {
            widths,
            fixed_us,
            gain: 1.0,
            gain_n: 1.0,
            gain_below: (1.0, 1.0),
            hold,
            offset: 0.0,
            rho,
            base_rows: 0,
        };
        budgeted_tree(5, cands, &[], choice, super::TreeRound::default())
    }

    fn c(token: u32, parent: i32, q: f64) -> TreeCandidate {
        TreeCandidate { token, parent, q }
    }

    /// A CHAIN keeps one child per node: best-first takes the likeliest child and its siblings leave
    /// the frontier, so the rows are a path -- and where an n-gram node beats the drafter's pick, the
    /// path follows the n-gram.
    #[test]
    fn a_chain_shape_keeps_one_child_per_node() {
        use super::{Shape, TreeRound};
        // a (Q .9) -> c (.72), d (.27); b (.5) under the anchor.
        let cands = [c(10, -1, 0.9), c(11, -1, 0.5), c(12, 0, 0.8), c(13, 0, 0.3)];
        let chain = TreeRound {
            shape: Shape::Chain,
            ..TreeRound::default()
        };
        let tree =
            super::fixed_tree(5, &cands, 4, 0.0, &[], TreeRound::default()).unwrap();
        assert_eq!(
            tree.tokens,
            vec![5, 10, 12, 11],
            "the tree branches at the anchor"
        );
        let path = super::fixed_tree(5, &cands, 4, 0.0, &[], chain).unwrap();
        assert_eq!(
            path.tokens,
            vec![5, 10, 12],
            "the chain ends where its last node has no child"
        );
        assert_eq!(path.parents, vec![-1, 0, 1]);
        // The drafter's pick p (.4) and an n-gram follower n (.6) under the anchor, n's own child
        // m (.9): the path goes n -> m.
        let cands = [c(20, -1, 0.4), c(21, -1, 0.6), c(22, 1, 0.9), c(23, 0, 0.9)];
        let path = super::fixed_tree(5, &cands, 3, 0.0, &[], chain).unwrap();
        assert_eq!(path.tokens, vec![5, 21, 22]);
    }

    /// THE DRAFT TALLY: a tree round counts every row past the anchor as verified and the path
    /// past the anchor as accepted, each under the source that proposed it; a chain round is all
    /// the drafter's, and so is a tree built by hand.
    #[test]
    fn the_draft_tally_counts_rows_and_the_accepted_path_by_source() {
        use super::{DraftTally, NodeSource, Tally, TreeRound};
        // The drafter's pick a (Q .9), its child c (.8) that the n-gram agreed with, and an n-gram
        // node b (.5) under the anchor.
        let cands = [c(10, -1, 0.9), c(11, -1, 0.5), c(12, 0, 0.8)];
        let of = [NodeSource::Drafter, NodeSource::Ngram, NodeSource::Agreed];
        let round = TreeRound {
            of: &of,
            ..TreeRound::default()
        };
        let tree = super::fixed_tree(5, &cands, 4, 0.0, &[], round).unwrap();
        assert_eq!(tree.tokens, vec![5, 10, 12, 11]);
        assert_eq!(
            tree.source,
            vec![
                None,
                Some(NodeSource::Drafter),
                Some(NodeSource::Agreed),
                Some(NodeSource::Ngram)
            ]
        );
        let mut t = DraftTally::default();
        t.tree(&tree, &[0, 1, 2]); // anchor -> a -> c
        let one = |accepted| Tally {
            verified: 1,
            accepted,
        };
        assert_eq!((t.drafter, t.agreed, t.ngram), (one(1), one(1), one(0)));
        t.chain(9, 4);
        assert_eq!(
            t.drafter,
            Tally {
                verified: 10,
                accepted: 5
            }
        );
        assert_eq!(
            t.total(),
            Tally {
                verified: 12,
                accepted: 6
            }
        );
        let mut by_hand = DraftTally::default();
        by_hand.tree(
            &super::DraftTree::plain(vec![5, 10, 11], vec![-1, 0, 1]),
            &[0, 1],
        );
        assert_eq!(
            by_hand.drafter,
            Tally {
                verified: 2,
                accepted: 1
            }
        );
        assert_eq!(by_hand.total(), by_hand.drafter);
    }

    /// A width below the reference prices the rows it gives up with the gains fitted on those rows.
    #[test]
    fn a_width_below_the_reference_is_priced_by_the_gain_fitted_below_it() {
        // S over the best-first order, keep k -> s[k]; the reference is 8 rows (keep 7).
        let s = [0.0, 0.9, 1.6, 2.2, 2.7, 3.1, 3.4, 3.6, 3.75, 3.85];
        let none = [0.0; 10];
        let widths = [(4, 100.0), (8, 100.0), (10, 100.0)];
        let choice = |below: f64| super::Choice {
            widths: &widths,
            fixed_us: 0.0,
            gain: 0.25,
            gain_n: 1.0,
            gain_below: (below, 1.0),
            hold: 8,
            offset: 0.0,
            rho: None,
            base_rows: 8,
        };
        let (c, low) = (choice(1.0), choice(0.25));
        let v = super::Valuer::new(&s, &none, s.len() - 1, &c, None);
        // 4 rows give up s[7] - s[3] = 1.4 nodes, at the scale fitted below the reference ...
        assert!((v.tokens(3) - (1.0 + 3.6 - 1.4)).abs() < 1e-9);
        // ... and 10 rows add s[9] - s[7] = 0.25 at the scale fitted above it.
        assert!((v.tokens(9) - (1.0 + 3.6 + 0.25 * 0.25)).abs() < 1e-9);
        // The single scale this replaces discounted what 4 rows give up as well, so a narrow
        // width looked almost as productive as the reference.
        let w = super::Valuer::new(&s, &none, s.len() - 1, &low, None);
        assert!((w.tokens(3) - (1.0 + 3.6 - 0.25 * 1.4)).abs() < 1e-9);
        assert!(w.tokens(3) > v.tokens(3));
    }

    /// THE EXPLORATION ROW: the last row of the width goes to the frontier's n-gram node in the
    /// bucket with the fewest labels, even when a better-priced n-gram node is on the frontier; a
    /// round with no such node keeps its tree.
    #[test]
    fn the_exploration_row_takes_the_least_labelled_frontier_node() {
        use super::{NodeSource, TreeRound};
        // a (drafter, Q .9) -> c (drafter, Q .72); b (n-gram, under the anchor, Q .5); d (n-gram,
        // under a, Q .27). Three rows keep a and c.
        let cands = [c(10, -1, 0.9), c(11, -1, 0.5), c(12, 0, 0.8), c(13, 0, 0.3)];
        let of = [
            NodeSource::Drafter,
            NodeSource::Ngram,
            NodeSource::Drafter,
            NodeSource::Ngram,
        ];
        let plain =
            super::fixed_tree(5, &cands, 3, 0.0, &[], TreeRound::default()).unwrap();
        assert_eq!(plain.tokens, vec![5, 10, 12]);
        assert!(!plain.explored);
        // b is better priced, d has fewer labels: d takes c's row.
        let labels = [u64::MAX, 7, u64::MAX, 2];
        let sources = TreeRound {
            of: &of,
            explore: Some(&labels),
            ..TreeRound::default()
        };
        let t = super::fixed_tree(5, &cands, 3, 0.0, &[], sources).unwrap();
        assert!(t.explored);
        assert_eq!(t.tokens, vec![5, 10, 13]);
        assert_eq!(t.parents, vec![-1, 0, 1]);
        // No n-gram node on the frontier: the tree is the plain one.
        let none = [u64::MAX; 4];
        let sources = TreeRound {
            of: &of,
            explore: Some(&none),
            ..TreeRound::default()
        };
        let t = super::fixed_tree(5, &cands, 3, 0.0, &[], sources).unwrap();
        assert!(!t.explored);
        assert_eq!(t.tokens, plain.tokens);
    }

    /// Sorted `(q, y)` pairs, so a test states WHAT was labelled without depending on the
    /// row order the emitter happened to produce.
    fn labels(
        t: &super::DraftTree,
        cands: &[TreeCandidate],
        path: &[i32],
    ) -> Vec<(String, bool)> {
        let mut v: Vec<(String, bool)> =
            acceptance_labels(&t.parents, &t.node, cands, path)
                .into_iter()
                .map(|(q, y)| (format!("{q:.4}"), y))
                .collect();
        v.sort();
        v
    }

    /// The censoring rule, which is the whole point: a node under a REJECTED parent carries no
    /// label, because the target never judged that position.
    #[test]
    fn only_a_node_whose_parent_was_accepted_is_labelled() {
        let chain = [c(10, -1, 0.9), c(11, 0, 0.8), c(12, 1, 0.7)];
        let t = best_first_tree(5, &chain, 64, 0.0, &[]).unwrap();
        assert_eq!(t.parents, vec![-1, 0, 1, 2]);

        // The target took the anchor and row 1, then stopped.
        //   row 1  parent 0 accepted, itself accepted    -> (0.9, true)
        //   row 2  parent 1 accepted, itself rejected    -> (0.8, false)
        //   row 3  parent 2 REJECTED                     -> no label
        assert_eq!(
            labels(&t, &chain, &[0, 1]),
            vec![("0.8000".into(), false), ("0.9000".into(), true)]
        );

        // Nothing accepted past the anchor: only the anchor's own child is judged.
        assert_eq!(
            labels(&t, &chain, &[0]),
            vec![("0.9000".into(), true)]
                .into_iter()
                .map(|(q, _): (String, bool)| (q, false))
                .collect::<Vec<_>>()
        );

        // The whole chain accepted: every node is judged, and the last one's child does not
        // exist, so there are exactly three labels and all are true.
        assert_eq!(
            labels(&t, &chain, &[0, 1, 2, 3]),
            vec![
                ("0.7000".into(), true),
                ("0.8000".into(), true),
                ("0.9000".into(), true)
            ]
        );
    }

    /// A sibling that lost is a REJECTION at a judged position -- the label the offset most
    /// needs, and the one a chain-only scheme never sees.
    #[test]
    fn the_losing_sibling_of_an_accepted_node_is_labelled_rejected() {
        //   anchor's children: 10 (q 0.9) and 20 (q 0.3); 10's child 11 (q 0.5)
        let cands = [c(10, -1, 0.9), c(20, -1, 0.3), c(11, 0, 0.5)];
        let t = best_first_tree(5, &cands, 4, 0.0, &[]).unwrap();
        // The target took 10, so 20 lost at a position it was judged at, and 11 was judged too.
        let row_of = |tok: u32| {
            i32::try_from(t.tokens.iter().position(|&x| x == tok).unwrap()).unwrap()
        };
        let got = labels(&t, &cands, &[0, row_of(10)]);
        assert_eq!(
            got,
            vec![
                ("0.3000".into(), false), // the sibling that lost
                ("0.5000".into(), false), // the accepted node's child, not taken
                ("0.9000".into(), true),  // the accepted node
            ],
            "labels were {got:?}"
        );
    }

    /// A node deep in the tree comes back with its own conditional, not its path product.
    #[test]
    fn a_deep_node_reports_its_own_q_not_its_path_product() {
        let chain = [c(10, -1, 0.5), c(11, 0, 0.4), c(12, 1, 0.25)];
        let t = best_first_tree(5, &chain, 64, 0.0, &[]).unwrap();
        // Q along the path is 0.5, 0.2, 0.05 -- the labels must be 0.5, 0.4, 0.25.
        assert_eq!(
            labels(&t, &chain, &[0, 1, 2, 3]),
            vec![
                ("0.2500".into(), true),
                ("0.4000".into(), true),
                ("0.5000".into(), true)
            ]
        );
    }

    /// Under an offset a label still carries the candidate's own q, which is what
    /// `AcceptOffset::observe` shifts. The tree was built from the shifted q (0.7311, 0.6444 at
    /// b = 1); a label carrying those would be shifted a second time.
    #[test]
    fn a_label_carries_the_candidate_s_own_q_under_an_offset() {
        let chain = [c(10, -1, 0.5), c(11, 0, 0.4)];
        let t = best_first_tree(5, &chain, 64, 1.0, &[]).unwrap();
        assert!((crate::accept_offset::shift(0.5, 1.0) - 0.7311).abs() < 1e-4);
        assert_eq!(
            labels(&t, &chain, &[0, 1, 2]),
            vec![("0.4000".into(), true), ("0.5000".into(), true)]
        );
    }

    /// A tree built by hand has no candidates behind its rows (`node` all -1), so it must yield
    /// NO labels rather than garbage -- the same rule `DraftTree::plain` documents.
    #[test]
    fn a_plain_tree_teaches_nothing() {
        let t = super::DraftTree::plain(vec![5, 10, 11], vec![-1, 0, 1]);
        let cands = [c(10, -1, 0.9), c(11, 0, 0.8)];
        assert!(acceptance_labels(&t.parents, &t.node, &cands, &[0, 1]).is_empty());
    }

    #[test]
    fn a_chain_alone_comes_back_in_order() {
        let chain = [c(10, -1, 0.9), c(11, 0, 0.8), c(12, 1, 0.7)];
        let t = best_first_tree(5, &chain, 64, 0.0, &[]).unwrap();
        assert_eq!(t.tokens, vec![5, 10, 11, 12]);
        assert_eq!(t.parents, vec![-1, 0, 1, 2]);
    }

    #[test]
    fn a_likely_sibling_takes_a_row_before_a_deep_chain_node() {
        // Q: 10 is 0.9, 11 is 0.45, 12 is 0.225; the sibling 20 is 0.3.
        let cands = [c(10, -1, 0.9), c(20, -1, 0.3), c(11, 0, 0.5), c(12, 2, 0.5)];
        let t = best_first_tree(5, &cands, 4, 0.0, &[]).unwrap();
        assert_eq!(t.tokens, vec![5, 10, 11, 20]);
        assert_eq!(t.parents, vec![-1, 0, 1, 0]);
    }

    #[test]
    fn equal_q_keeps_the_earlier_candidate_first() {
        let cands = [c(10, -1, 0.5), c(20, -1, 0.5), c(30, -1, 0.5)];
        let t = best_first_tree(5, &cands, 3, 0.0, &[]).unwrap();
        assert_eq!(t.tokens, vec![5, 10, 20]);
        assert_eq!(t.parents, vec![-1, 0, 0]);
    }

    #[test]
    fn rows_are_depth_first_with_the_likely_child_first() {
        // Q: a 0.6, b 0.4, a1 0.3, a2 0.15, b1 0.36.
        let cands = [
            c(1, -1, 0.6),
            c(2, -1, 0.4),
            c(3, 0, 0.5),
            c(4, 0, 0.25),
            c(5, 1, 0.9),
        ];
        let t = best_first_tree(0, &cands, 64, 0.0, &[]).unwrap();
        assert_eq!(t.tokens, vec![0, 1, 3, 4, 2, 5]);
        assert_eq!(t.parents, vec![-1, 0, 1, 1, 0, 4]);
    }

    #[test]
    fn a_parent_after_its_child_a_q_past_one_or_no_room_is_refused() {
        assert!(best_first_tree(0, &[c(1, 0, 0.5)], 4, 0.0, &[]).is_err());
        assert!(best_first_tree(0, &[c(1, -1, 1.5)], 4, 0.0, &[]).is_err());
        assert!(best_first_tree(0, &[c(1, -1, 0.5)], 0, 0.0, &[]).is_err());
    }

    /// THE BUDGET IS A RATIO, not a row count: when the extra nodes are unlikely, the
    /// narrow class carries more expected tokens per millisecond even though it verifies
    /// fewer rows.
    #[test]
    fn unlikely_extra_nodes_do_not_earn_their_rows() {
        let mut cands = vec![c(10, -1, 0.9), c(11, 0, 0.9), c(12, 1, 0.9)];
        for k in 0..20u32 {
            cands.push(c(100 + k, -1, 0.01));
        }
        // 4 rows at 30 ms:  (1 + 2.439) / 30 = 0.1146 tokens/ms
        // 32 rows at 45 ms: (1 + 2.639) / 45 = 0.0809
        let t = budget(&cands, &[(4, 30_000.0), (32, 45_000.0)], 0.0, 0, None).unwrap();
        assert_eq!(t.tokens.len(), 4);
    }

    /// And when they ARE likely it takes the wide class, at a cost 50% higher.
    #[test]
    fn likely_extra_nodes_buy_the_wider_class() {
        let cands: Vec<TreeCandidate> = (0..31)
            .map(|k| c(10 + k, if k == 0 { -1 } else { k as i32 - 1 }, 0.95))
            .collect();
        // 4 rows:  (1 + 2.71) / 30  = 0.124 tokens/ms
        // 32 rows: (1 + 15.1) / 45  = 0.358
        let t = budget(&cands, &[(4, 30_000.0), (32, 45_000.0)], 0.0, 0, None).unwrap();
        assert_eq!(t.tokens.len(), 32);
    }

    /// THE ROUND'S FIXED COST CHANGES THE ANSWER, and leaving it out is why design 6.3's
    /// ratio picks trees that are too narrow: the drafter is paid once per round, so a
    /// narrower tree makes rounds shorter but more numerous and does not save what it looks
    /// like it saves.
    #[test]
    fn the_fixed_cost_moves_the_choice_to_the_wider_tree() {
        // A chain of 8 at q=0.8: S(3) = 1.952, S(7) = 3.329.
        let cands: Vec<TreeCandidate> = (0..8)
            .map(|k| c(10 + k, if k == 0 { -1 } else { k as i32 - 1 }, 0.8))
            .collect();
        let widths = [(4, 29_000.0), (8, 45_000.0)];
        // Verify time alone: 2.952/29 = 0.1018 beats 4.329/45 = 0.0962 -- the narrow tree.
        let narrow = budget(&cands, &widths, 0.0, 0, None).unwrap();
        assert_eq!(narrow.tokens.len(), 4);
        // With a 30 ms round cost: 2.952/59 = 0.0500 against 4.329/75 = 0.0577 -- the wide.
        let wide = budget(&cands, &widths, 30_000.0, 0, None).unwrap();
        assert_eq!(wide.tokens.len(), 8);
    }

    /// Equal tokens per millisecond at a narrower tree is the same throughput at lower
    /// latency, and it leaves the drafter's candidates for the next round.
    #[test]
    fn a_tie_keeps_the_narrower_tree() {
        let cands = [c(10, -1, 1.0)];
        let t = budget(&cands, &[(2, 10_000.0), (4, 10_000.0)], 0.0, 0, None).unwrap();
        assert_eq!(t.tokens.len(), 2);
    }

    /// The budget's tree at n rows IS the fixed-width tree at n rows: best-first takes
    /// nodes in one order and every prefix of it is a tree, so choosing a width can never
    /// produce a shape the width alone would not have.
    #[test]
    fn the_budget_and_a_fixed_width_agree_on_the_same_row_count() {
        let cands = [
            c(10, -1, 0.9),
            c(20, -1, 0.3),
            c(11, 0, 0.5),
            c(12, 2, 0.5),
            c(21, 1, 0.9),
        ];
        for rows in 2..=6 {
            let fixed = best_first_tree(5, &cands, rows, 0.0, &[]).unwrap();
            let budget = budget(&cands, &[(rows, 1_000.0)], 0.0, 0, None).unwrap();
            assert_eq!(fixed.tokens, budget.tokens, "rows={rows}");
            assert_eq!(fixed.parents, budget.parents, "rows={rows}");
        }
    }

    /// A backend that declared no classes leaves nothing to choose from, and the round
    /// must not guess a width.
    #[test]
    fn no_measured_width_is_an_error_not_a_guess() {
        assert!(budget(&[c(10, -1, 0.9)], &[], 0.0, 0, None).is_err());
    }

    /// A STOP-TOKEN NODE CAN NEVER BE ACCEPTED -- the walk ends at a stop pick before it looks
    /// for a child -- so it must not take a row, count in S, or let its subtree in.
    #[test]
    fn a_stop_token_node_and_everything_below_it_take_no_row() {
        // The anchor's children: 10 (q 0.9) and the stop token 99 (q 0.8); 99's child 30.
        let cands = [c(10, -1, 0.9), c(99, -1, 0.8), c(30, 1, 0.9), c(11, 0, 0.5)];
        let t = best_first_tree(5, &cands, 64, 0.0, &[99]).unwrap();
        assert_eq!(t.tokens, vec![5, 10, 11]);
        // S counts only what can be accepted: 0.9 + 0.45.
        assert!((t.s[t.s.len() - 1] - 1.35).abs() < 1e-12, "{:?}", t.s);
        // Without the stop set the same candidates place all four.
        let all = best_first_tree(5, &cands, 64, 0.0, &[]).unwrap();
        assert_eq!(all.tokens.len(), 5);
    }

    /// THE LONG-RUN RULE PRICES TIME AT THE PROCESS'S RATE, not the round's own: a round whose
    /// own ratio is high refuses rows that a slower process should buy.
    #[test]
    fn the_long_run_rule_buys_rows_the_rounds_own_ratio_refuses() {
        // A chain at q 0.9, 0.5, 0.5: S(1) = 0.9, S(3) = 0.9 + 0.45 + 0.225 = 1.575.
        let cands = [c(10, -1, 0.9), c(11, 0, 0.5), c(12, 1, 0.5)];
        let widths = [(2, 10_000.0), (4, 20_000.0)];
        // ratio            1.9 / 10 ms = 0.190   against  2.575 / 20 ms = 0.129  -> 2 rows
        assert_eq!(
            budget(&cands, &widths, 0.0, 0, None).unwrap().tokens.len(),
            2
        );
        // rho 0.05 per ms  1.9 - 0.5 = 1.400     against  2.575 - 1.0 = 1.575    -> 4 rows
        let slow = budget(&cands, &widths, 0.0, 0, Some(5e-5)).unwrap();
        assert_eq!(slow.tokens.len(), 4);
        // rho 0.10 per ms  1.9 - 1.0 = 0.900     against  2.575 - 2.0 = 0.575    -> 2 rows
        let fast = budget(&cands, &widths, 0.0, 0, Some(1e-4)).unwrap();
        assert_eq!(fast.tokens.len(), 2);
    }

    /// The hysteresis on the long-run scale: an alternative must beat the incumbent by the
    /// margin times the incumbent's expected tokens, 0.04 x 1.9 = 0.076 here.
    #[test]
    fn the_long_run_rule_keeps_its_incumbent_inside_the_margin() {
        let cands = [c(10, -1, 0.9), c(11, 0, 0.5), c(12, 1, 0.5)];
        let widths = [(2, 10_000.0), (4, 20_000.0)];
        // rho 0.065 per ms: 4 rows lead by 1.275 - 1.250 = 0.025, inside the margin.
        let free = budget(&cands, &widths, 0.0, 0, Some(6.5e-5)).unwrap();
        assert_eq!(free.tokens.len(), 4, "no incumbent: the better width");
        let held = budget(&cands, &widths, 0.0, 2, Some(6.5e-5)).unwrap();
        assert_eq!(held.tokens.len(), 2, "sitting on 2 rows: stays");
        // rho 0.05 per ms: the lead is 0.175, clear of the margin.
        let leaves = budget(&cands, &widths, 0.0, 2, Some(5e-5)).unwrap();
        assert_eq!(leaves.tokens.len(), 4);
    }
}
