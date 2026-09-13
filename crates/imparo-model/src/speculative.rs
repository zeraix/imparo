//! Model-independent greedy continuation; default single-token generation is unchanged.
use crate::Model;

/// Serialized draft provider paired with the target. On error, terminate the
/// continuation: the target may have committed before an auxiliary-state failure.
#[derive(Clone, Debug)]
pub struct DraftTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<i32>,
}
#[derive(Clone, Debug)]
pub struct TreeVerification {
    pub path: Vec<i32>,
    pub next_token: u32,
}
pub trait DraftProvider {
    fn tree_proposal(
        &self,
        _start: usize,
        _anchor: u32,
        _chain: &[u32],
    ) -> Result<Option<DraftTree>, String> {
        Ok(None)
    }
    fn commit_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        _path: &[i32],
    ) -> Result<(), String> {
        self.commit(start, inputs)
    }
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
    /// Exactly block_size-1 successor candidates; excludes the anchor.
    fn draft(&mut self, start: usize, anchor: u32) -> Result<Vec<u32>, String>;
    /// None declines only this round. Leave target state unchanged and keep
    /// provider capture/committed history ready for the ordinary M1 commit.
    /// Errors remain fatal; existing providers retain their fixed-block behavior.
    fn try_draft(
        &mut self,
        start: usize,
        anchor: u32,
    ) -> Result<Option<Vec<u32>>, String> {
        self.draft(start, anchor).map(Some)
    }
    /// Commit target features for only these accepted inputs, excluding the bonus.
    fn commit(&mut self, _start: usize, _inputs: &[u32]) -> Result<(), String> {
        Ok(())
    }
    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }
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
    terminal: bool,
    capture_closed: bool,
    poison: Option<String>,
}
impl GreedyCursor {
    pub fn new<M: Model + ?Sized, D: DraftProvider + ?Sized>(
        target: &M,
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
            terminal,
            capture_closed: false,
            poison: None,
        })
    }
    pub fn is_finished(&self) -> bool {
        self.poison.is_none() && self.terminal && self.pending.is_empty()
    }
    /// Pending output is drained without a forward, draft or history update.
    pub fn next<M: Model + ?Sized, D: DraftProvider + ?Sized>(
        &mut self,
        target: &mut M,
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
                        e
                    }
                    Err(c) => format!("{e}; disable draft capture: {c}"),
                };
                self.poison = Some(e.clone());
                Err(e)
            }
        }
    }
    fn next_inner<M: Model + ?Sized, D: DraftProvider + ?Sized>(
        &mut self,
        target: &mut M,
        provider: &mut D,
    ) -> Result<Option<u32>, String> {
        if self.pending.is_empty() && !self.terminal {
            self.refill(target, provider)?;
        }
        if self.terminal && !self.capture_closed {
            provider.set_capture(false)?;
            self.capture_closed = true;
        }
        let token = self.pending.pop_front();
        if token.is_some() {
            self.emitted += 1;
        }
        Ok(token)
    }
    fn refill<M: Model + ?Sized, D: DraftProvider + ?Sized>(
        &mut self,
        target: &mut M,
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
            provider.set_capture(false)?;
            self.speculative = false;
        }
        let mut input = Vec::new();
        let mut proposed_tree = None;
        if can_run {
            let proposal = provider.try_draft(self.pos, anchor)?;
            self.draft_calls += 1;
            if let Some(proposal) = proposal {
                if proposal.len() != self.block - 1
                    || proposal.iter().any(|&x| x >= self.vocab)
                {
                    return Err("invalid draft proposal shape/token".into());
                }
                proposed_tree = provider.tree_proposal(self.pos, anchor, &proposal)?;
                if let Some(tree) = proposed_tree.as_ref() {
                    if !target.prepare_greedy_tree(tree, self.pos)? {
                        proposed_tree = None;
                    }
                }
                if proposed_tree.is_none()
                    && proposal.iter().any(|x| self.stops.contains(x))
                {
                    provider.set_capture(false)?;
                    self.speculative = false;
                } else {
                    input.push(anchor);
                    input.extend(proposal);
                }
            } else if self.verified_consumed_histogram.is_some() {
                eprintln!("[imparo] draft-abstain start={}", self.pos);
            }
        }
        if self.speculative && can_run && !input.is_empty() {
            let old = self.pos;
            let commit_limit = remaining.min(self.block);
            let tree = proposed_tree;
            let mut selected_path = None;
            let (v, accepted_inputs) = if let Some(tree) = tree {
                let result =
                    target.verify_greedy_tree(&tree, old, commit_limit, &self.stops)?;
                let inputs: Vec<u32> = result
                    .path
                    .iter()
                    .map(|&i| tree.tokens[i as usize])
                    .collect();
                let v = crate::GreedyVerification {
                    consumed: inputs.len(),
                    next_token: result.next_token,
                };
                selected_path = Some(result.path);
                (v, inputs)
            } else {
                let v = if commit_limit < self.block {
                    target.verify_greedy_block_limited(&input, old, commit_limit)?
                } else {
                    target.verify_greedy_block(&input, old)?
                };
                let inputs = input
                    .get(..v.consumed)
                    .ok_or("invalid accepted range")?
                    .to_vec();
                (v, inputs)
            };
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
                return Ok(());
            }
            self.pending.push_back(v.next_token);
            self.generated += 1;
            self.terminal = self.generated == self.limit;
            let remaining = self.limit - self.generated;
            if !self.terminal
                && remaining >= self.minimum_remaining
                && (provider.can_draft(pos, remaining)
                    || provider.can_bridge(pos, remaining))
            {
                if let Some(path) = selected_path {
                    provider.commit_tree(old, &accepted_inputs, &path)?;
                } else {
                    provider.commit(old, &accepted_inputs)?;
                }
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
            if self.speculative && !self.terminal {
                let remaining = self.limit - self.generated;
                if remaining >= self.minimum_remaining
                    && (provider.can_draft(pos, remaining)
                        || provider.can_bridge(pos, remaining))
                {
                    provider.commit(old, &[anchor])?;
                }
            }
        }
        Ok(())
    }
}
/// Collect the same cursor; callers initialize and commit prompt history first.
pub fn continue_greedy<M: Model + ?Sized, D: DraftProvider + ?Sized>(
    target: &mut M,
    provider: &mut D,
    start: usize,
    first: u32,
    limit: usize,
    stops: &[u32],
) -> Result<Continuation, String> {
    let mut cursor =
        match GreedyCursor::new(target, provider, start, first, limit, stops) {
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
    })
}

/// Source selection belongs to model adapters, not the token scheduling algorithm.
#[cfg(feature = "cuda-speculative")]
pub enum DraftSpec {
    Dspark(crate::dspark::Pairing),
    GemmaMtp(crate::gemma4_mtp::Pairing),
}
#[cfg(feature = "cuda-speculative")]
pub type DraftRun<'a> =
    dyn FnMut(&mut dyn Model, &mut dyn DraftProvider) -> Result<(), String> + 'a;
#[cfg(feature = "cuda-speculative")]
impl DraftSpec {
    pub fn can_start(&self, start: usize, limit: usize, capacity: usize) -> bool {
        match self {
            Self::Dspark(pair) => pair.can_start(start, limit, capacity),
            Self::GemmaMtp(pair) => pair.can_start(start, limit, capacity),
        }
    }

    pub fn can_resume(&self, prompt: &[u32], start: usize) -> bool {
        match self {
            Self::Dspark(pair) => pair.can_resume(prompt, start),
            Self::GemmaMtp(pair) => pair.can_resume(prompt, start),
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
            Self::GemmaMtp(pair) => pair.run(target, run),
        }
    }
}
