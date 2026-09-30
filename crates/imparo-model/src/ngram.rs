//! THE REQUEST'S N-GRAM INDEX (design 11.4, level 1) -- and the shape level 2's dictionary reuses.
//!
//! # What it is for
//!
//! The drafter proposes from what it learned offline. It cannot know that THIS request keeps
//! re-emitting the same identifier, the same tool output, the same file path. An index over the
//! request's own text can, and it costs one rolling hash per committed token.
//!
//! ```text
//! every committed token   keyed by the K tokens before it -> the token that followed
//! at tree-build time      the last K tokens of a position's context -> its continuations
//! ```
//!
//! # Why the key is three tokens and not two
//!
//! MEASURED by our fork, and it is a precision question, not a recall one: a 2-token key doubled
//! the rounds with a match (14% to 26%) and bought one more accepted token. A match that is wrong
//! costs a verify row and teaches the acceptance estimate nothing; a match that is right extends a
//! path past the drafter's block. So the default is 3 and widening is a measurement, not a guess.
//!
//! # Lifetime
//!
//! Level 1: it lives for the request and resets with the drafter's context. Level 2's dictionary
//! (11.5B) is the same structure with a longer life and a disk form -- one hash, one eviction rule
//! and one persistence path serve both, which is why the key spaces are built here rather than
//! inside the per-request path.

/// Tokens of context a key holds. Three, from the fork's measurement above.
pub const KEY_TOKENS: usize = 3;

/// Continuations kept per context. A context with more than this many distinct followers is not
/// predictive enough to be worth more rows.
const FOLLOWERS: usize = 4;

/// Tokens a COPY MATCH must share with an earlier stretch of the request before it proposes the
/// token that followed there. MEASURED on full replies (a replay over a traced run): matches of
/// 1-3 tokens added about 1% of the copy rule's gain, and below this length the 3-token index
/// answers instead.
pub const MATCH_KEY: usize = 4;

/// The longest match measured backward. A match this long is a copy in progress; measuring
/// further only costs compares.
pub const MATCH_MAX: usize = 64;

/// Earlier occurrences kept per `MATCH_KEY` context, latest last. A copy follows the most recent
/// long match; older occurrences of a common 4-token context are rarely the one being copied.
const OCCURRENCES: usize = 8;

/// Contexts held before the index stops growing. A request is bounded, so this is a guard against
/// a pathological prompt rather than a working eviction policy; level 2's dictionary needs the
/// decayed-count rule instead.
const MAX_CONTEXTS: usize = 1 << 16;

/// FNV-1a over the key's token ids. Order matters -- `(a, b, c)` and `(c, b, a)` are different
/// contexts, and a sum or an xor would collide them.
fn key_of(tokens: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// What followed one context, most frequent first.
#[derive(Clone, Debug, Default)]
struct Followers {
    /// `(token, count)`, kept sorted by count descending then token ascending, so a tie is
    /// resolved the same way every run and the index cannot make a round nondeterministic.
    seen: Vec<(u32, u32)>,
}

impl Followers {
    fn observe(&mut self, token: u32) {
        if let Some(slot) = self.seen.iter_mut().find(|(t, _)| *t == token) {
            slot.1 = slot.1.saturating_add(1);
        } else if self.seen.len() < FOLLOWERS {
            self.seen.push((token, 1));
        } else {
            // Full and the token is new: it displaces the weakest only by beating it, so a
            // one-off cannot evict something the request has repeated.
            return;
        }
        self.seen.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    }
}

/// Design 11.4's per-request index.
#[derive(Debug)]
pub struct NgramIndex {
    map: std::collections::HashMap<u64, Followers>,
    /// The committed tokens, so a key can be formed from the tail without the caller keeping one.
    tail: Vec<u32>,
    /// Tokens folded in since the last reset, for the probe line.
    seen: u64,
    /// Contexts held before a new one is refused.
    limit: usize,
    /// THE COPY MATCH'S INDEX: for each `MATCH_KEY`-token context of `tail`, the positions in
    /// `tail` of the tokens that followed it, latest last, at most `OCCURRENCES`. Keyed by hash;
    /// a match is confirmed token by token, so a collision costs a compare, never a wrong match.
    followed_at: std::collections::HashMap<u64, Vec<u32>>,
}

impl Default for NgramIndex {
    fn default() -> Self {
        Self::with_limit(MAX_CONTEXTS)
    }
}

impl NgramIndex {
    /// An index that holds up to `limit` contexts. The request's index uses `MAX_CONTEXTS`; the
    /// stored table (11.5B) needs its own, larger bound, or its decay-at-save eviction could never
    /// run -- one bound for both was measured to leave the table full at load and the request's own
    /// text unable to add a context.
    #[must_use]
    pub fn with_limit(limit: usize) -> Self {
        Self {
            map: std::collections::HashMap::new(),
            tail: Vec::new(),
            seen: 0,
            limit,
            followed_at: std::collections::HashMap::new(),
        }
    }

    /// One committed token. The caller passes them in commit order; the index keys each one on the
    /// `KEY_TOKENS` that preceded it, so the first `KEY_TOKENS` tokens of a request only build
    /// context and teach nothing.
    pub fn observe(&mut self, token: u32) {
        let at = self.tail.len();
        if at >= KEY_TOKENS {
            let k = key_of(&self.tail[at - KEY_TOKENS..]);
            self.count_key(k, token);
        }
        if at >= MATCH_KEY {
            if let Ok(position) = u32::try_from(at) {
                let seen = self
                    .followed_at
                    .entry(key_of(&self.tail[at - MATCH_KEY..]))
                    .or_default();
                if seen.len() == OCCURRENCES {
                    seen.remove(0);
                }
                seen.push(position);
            }
        }
        self.tail.push(token);
        self.seen += 1;
    }

    /// THE COPY MATCH: the longest earlier stretch of this request's committed text that `context`
    /// ends with, at least `MATCH_KEY` and at most `MATCH_MAX` tokens, as `(length, position)`:
    /// `tail()[position]` is the token that followed it. The longest wins, and the latest among
    /// equals. `context` may run past the committed text (a path and a chain's own tokens); only
    /// its suffix is compared.
    #[must_use]
    pub fn longest(&self, context: &[u32]) -> Option<(usize, usize)> {
        if context.len() < MATCH_KEY {
            return None;
        }
        let seen = self
            .followed_at
            .get(&key_of(&context[context.len() - MATCH_KEY..]))?;
        let mut best: Option<(usize, usize)> = None;
        for &position in seen.iter().rev() {
            let p = position as usize;
            let limit = MATCH_MAX.min(context.len()).min(p);
            let len = (0..limit)
                .take_while(|&i| context[context.len() - 1 - i] == self.tail[p - 1 - i])
                .count();
            if len >= MATCH_KEY && best.is_none_or(|(b, _)| len > b) {
                best = Some((len, p));
            }
        }
        best
    }

    /// One follower of `context` counted WITHOUT extending this index's own tail: the stored table
    /// learns the request's text keyed by the request's context, which lives in the request's
    /// index. A context shorter than a key teaches nothing.
    pub fn count(&mut self, context: &[u32], token: u32) {
        if context.len() >= KEY_TOKENS {
            self.count_key(key_of(&context[context.len() - KEY_TOKENS..]), token);
            self.seen += 1;
        }
    }

    fn count_key(&mut self, k: u64, token: u32) {
        if self.map.len() < self.limit {
            self.map.entry(k).or_default().observe(token);
        } else if let Some(f) = self.map.get_mut(&k) {
            // Full: an EXISTING context still learns, so a repeated identifier keeps sharpening.
            f.observe(token);
        }
    }

    /// Every committed token of a prompt or a continuation, in order.
    pub fn observe_all(&mut self, tokens: &[u32]) {
        for &t in tokens {
            self.observe(t);
        }
    }

    /// What this request has seen follow `context`, most frequent first. `context` is the tokens
    /// BEFORE the position being predicted; only its last `KEY_TOKENS` are used, and a shorter one
    /// matches nothing.
    #[must_use]
    pub fn follows(&self, context: &[u32]) -> &[(u32, u32)] {
        if context.len() < KEY_TOKENS {
            return &[];
        }
        self.map
            .get(&key_of(&context[context.len() - KEY_TOKENS..]))
            .map_or(&[], |f| f.seen.as_slice())
    }

    /// The tokens committed so far, which is the context a caller extends.
    #[must_use]
    pub fn tail(&self) -> &[u32] {
        &self.tail
    }

    /// Contexts held, and tokens folded in.
    #[must_use]
    pub fn size(&self) -> (usize, u64) {
        (self.map.len(), self.seen)
    }

    /// A new request. Level 1 ends here (design 11.2): the counts AND the context go.
    pub fn reset(&mut self) {
        self.map.clear();
        self.reset_request();
    }

    /// A new request for a LEVEL 2 table: the context goes, the counts stay. This is the whole
    /// difference between 11.4's index and 11.5B's dictionary -- one structure, two lifetimes.
    /// The context must still go, or the first key of a new request would be formed from the
    /// last tokens of the previous one.
    pub fn reset_request(&mut self) {
        self.tail.clear();
        self.followed_at.clear();
        self.seen = 0;
    }

    // ---- LEVEL 2 (design 11.5B): the same structure, kept across requests --------------
    //
    // A dictionary is this index with a longer life and a disk form. One hash, one eviction
    // rule and one persistence path serve both, which is why these live here rather than in a
    // second module that would drift from the one the tests cover.

    /// Every `(context hash, token, count)` it holds, for a snapshot. The KEY IS THE HASH, not
    /// the tokens: a stored file is only valid for the tokenizer that produced it, which is
    /// part of what the file's key records.
    pub fn rows(&self) -> impl Iterator<Item = (u64, u32, u32)> + '_ {
        self.map
            .iter()
            .flat_map(|(&k, f)| f.seen.iter().map(move |&(t, c)| (k, t, c)))
    }

    /// One stored row back. Counts ADD, so a snapshot can be replayed over a live index.
    pub fn seed(&mut self, key: u64, token: u32, count: u32) {
        if count == 0 || self.map.len() >= self.limit && !self.map.contains_key(&key) {
            return;
        }
        let f = self.map.entry(key).or_default();
        if let Some(slot) = f.seen.iter_mut().find(|(t, _)| *t == token) {
            slot.1 = slot.1.saturating_add(count);
        } else if f.seen.len() < FOLLOWERS {
            f.seen.push((token, count));
        } else if let Some(weakest) = f.seen.iter_mut().min_by_key(|(_, c)| *c) {
            // A stored row that BEATS the weakest replaces it; equal does not, so replay is
            // stable and a tie cannot flip between loads.
            if count > weakest.1 {
                *weakest = (token, count);
            }
        }
        f.seen.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    }

    /// DECAY, which is what makes the dictionary's eviction a policy rather than a cap: every
    /// count is halved and anything that reaches zero is dropped. Run once per snapshot, so a
    /// context that stops appearing fades over a few sessions instead of being held forever by
    /// one old burst.
    pub fn decay(&mut self) {
        self.map.retain(|_, f| {
            f.seen.retain_mut(|(_, c)| {
                *c /= 2;
                *c > 0
            });
            !f.seen.is_empty()
        });
    }

    /// Contexts held, so a caller can report what it loaded.
    #[must_use]
    pub fn contexts(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{FOLLOWERS, KEY_TOKENS, MATCH_KEY, MATCH_MAX, NgramIndex, key_of};

    #[test]
    fn a_key_is_ordered_and_three_tokens_wide() {
        assert_ne!(key_of(&[1, 2, 3]), key_of(&[3, 2, 1]), "order must matter");
        assert_ne!(key_of(&[1, 2, 3]), key_of(&[1, 2, 4]));
        assert_eq!(key_of(&[7, 8, 9]), key_of(&[7, 8, 9]));
        assert_eq!(KEY_TOKENS, 3);
    }

    /// The first tokens of a request build context and teach nothing -- a match before there is a
    /// full key would be a match on a prefix, which is a different (and weaker) predictor.
    #[test]
    fn nothing_is_learned_until_a_full_key_exists() {
        let mut ix = NgramIndex::default();
        ix.observe_all(&[10, 11, 12]);
        assert_eq!(
            ix.size().0,
            0,
            "three tokens are one key and no follower yet"
        );
        ix.observe(13);
        assert_eq!(ix.follows(&[10, 11, 12]), &[(13, 1)]);
    }

    /// The point of the thing: a repeated continuation becomes the first candidate.
    #[test]
    fn a_repeated_continuation_sorts_first() {
        let mut ix = NgramIndex::default();
        // "a b c" is followed by 99 three times and by 50 once.
        for t in [99, 50, 99, 99] {
            ix.observe_all(&[1, 2, 3]);
            ix.observe(t);
        }
        let f = ix.follows(&[1, 2, 3]);
        assert_eq!(f[0], (99, 3), "got {f:?}");
        assert!(f.contains(&(50, 1)));
    }

    /// A context the request has not produced must return nothing rather than a near match.
    #[test]
    fn an_unseen_context_matches_nothing() {
        let mut ix = NgramIndex::default();
        ix.observe_all(&[1, 2, 3, 4]);
        assert!(ix.follows(&[9, 9, 9]).is_empty());
        assert!(
            ix.follows(&[2, 3]).is_empty(),
            "a short context is not a key"
        );
    }

    /// Ties break the same way every run, or the index could make a round nondeterministic.
    #[test]
    fn equal_counts_break_on_the_token_id() {
        let mut ix = NgramIndex::default();
        for t in [77, 22] {
            ix.observe_all(&[5, 5, 5]);
            ix.observe(t);
        }
        assert_eq!(ix.follows(&[5, 5, 5])[0], (22, 1));
    }

    /// A one-off cannot displace something the request has repeated.
    #[test]
    fn a_full_context_keeps_what_it_has_seen_most() {
        let mut ix = NgramIndex::default();
        for i in 0..FOLLOWERS as u32 {
            for _ in 0..3 {
                ix.observe_all(&[1, 2, 3]);
                ix.observe(100 + i);
            }
        }
        ix.observe_all(&[1, 2, 3]);
        ix.observe(999); // new, and the context is full
        let f = ix.follows(&[1, 2, 3]);
        assert_eq!(f.len(), FOLLOWERS);
        assert!(
            !f.iter().any(|&(t, _)| t == 999),
            "a one-off evicted a repeat: {f:?}"
        );
    }

    /// Reset is what makes this level 1.
    #[test]
    fn reset_empties_the_index_and_the_context() {
        let mut ix = NgramIndex::default();
        ix.observe_all(&[1, 2, 3, 4, 5]);
        assert!(ix.size().0 > 0 && !ix.tail().is_empty());
        ix.reset();
        assert_eq!(ix.size(), (0, 0));
        assert!(ix.tail().is_empty());
        assert!(ix.follows(&[1, 2, 3]).is_empty());
    }

    /// A snapshot round trip must hold every row, or the dictionary silently loses what it
    /// learned each time the process exits.
    #[test]
    fn a_snapshot_round_trips_every_row() {
        let mut ix = NgramIndex::default();
        for t in [99_u32, 50, 99, 99, 12] {
            ix.observe_all(&[1, 2, 3]);
            ix.observe(t);
        }
        let rows: Vec<(u64, u32, u32)> = ix.rows().collect();
        assert!(!rows.is_empty());
        let mut back = NgramIndex::default();
        for (k, t, c) in rows {
            back.seed(k, t, c);
        }
        assert_eq!(back.contexts(), ix.contexts());
        // The live index needs a context to query with; the stored one is keyed by hash.
        ix.observe_all(&[1, 2, 3]);
        back.observe_all(&[1, 2, 3]);
        assert_eq!(back.follows(&[1, 2, 3]), ix.follows(&[1, 2, 3]));
    }

    /// Decay is the eviction POLICY: a context that stops appearing fades, and one that keeps
    /// appearing survives. A cap alone would hold an old burst forever.
    #[test]
    fn decay_halves_counts_and_drops_what_reaches_zero() {
        let mut ix = NgramIndex::default();
        for _ in 0..4 {
            ix.observe_all(&[1, 2, 3]);
            ix.observe(99);
        }
        ix.observe_all(&[4, 5, 6]);
        ix.observe(88); // seen once
        ix.observe_all(&[1, 2, 3]);
        ix.observe_all(&[4, 5, 6]);

        ix.decay();
        assert_eq!(ix.follows(&[1, 2, 3])[0], (99, 2), "4 -> 2");
        assert!(ix.follows(&[4, 5, 6]).is_empty(), "1 -> 0 is dropped");
        ix.decay();
        assert_eq!(ix.follows(&[1, 2, 3])[0], (99, 1));
        ix.decay();
        assert!(ix.follows(&[1, 2, 3]).is_empty(), "it fades out entirely");
    }

    /// Seeding ADDS, so replaying a snapshot over a live index accumulates rather than
    /// replacing -- that is what lets one session's text join what earlier sessions learned.
    #[test]
    fn seeding_adds_to_what_is_already_there() {
        let mut ix = NgramIndex::default();
        ix.observe_all(&[1, 2, 3]);
        ix.observe(99);
        ix.observe_all(&[1, 2, 3]);
        // The key must be NAMED, not taken from `rows()`: by now the index holds four
        // contexts and a HashMap's order is unspecified, so `rows().next()` is arbitrary.
        ix.seed(key_of(&[1, 2, 3]), 99, 10);
        assert_eq!(ix.follows(&[1, 2, 3])[0], (99, 11));
    }

    /// A copy match reports its full length and the position after it, and a shorter one does
    /// not propose at all.
    #[test]
    fn a_copy_match_measures_the_whole_shared_stretch() {
        let mut ix = NgramIndex::default();
        let passage = [10_u32, 11, 12, 13, 14, 15, 16, 17];
        ix.observe_all(&passage);
        ix.observe_all(&[90, 91]);
        // "12 13 14 15 16" again: five tokens shared, and 17 followed them at position 7.
        let context = [99_u32, 12, 13, 14, 15, 16];
        assert_eq!(ix.longest(&context), Some((5, 7)));
        assert_eq!(ix.tail()[7], 17);
        // Three shared tokens are below the key: no copy match.
        assert_eq!(ix.longest(&[99, 14, 15, 16]), None);
        const { assert!(MATCH_KEY > 3 && MATCH_MAX >= 16) };
    }

    /// Between two earlier occurrences the longer shared stretch wins, and between equals the
    /// later one: a copy follows what was written most recently.
    #[test]
    fn the_longest_match_wins_and_the_latest_breaks_a_tie() {
        let mut ix = NgramIndex::default();
        ix.observe_all(&[1, 2, 3, 4, 5, 6, 50]); // "1 2 3 4 5 6" -> 50 at position 6
        ix.observe_all(&[8, 3, 4, 5, 6, 60]); //    "3 4 5 6"     -> 60 at position 12
        // The context shares six tokens with the first occurrence and four with the second.
        assert_eq!(ix.longest(&[1, 2, 3, 4, 5, 6]), Some((6, 6)));
        // Four shared with both: the later one.
        ix.observe_all(&[9, 3, 4, 5, 6, 70]); //    "3 4 5 6"     -> 70 at position 18
        assert_eq!(ix.longest(&[7, 3, 4, 5, 6]), Some((4, 18)));
    }

    /// A reset forgets where contexts were followed along with the text itself.
    #[test]
    fn a_reset_forgets_the_copy_matches() {
        let mut ix = NgramIndex::default();
        ix.observe_all(&[1, 2, 3, 4, 5, 1, 2, 3, 4]);
        assert!(ix.longest(&[1, 2, 3, 4]).is_some());
        ix.reset();
        assert_eq!(ix.longest(&[1, 2, 3, 4]), None);
    }

    /// The real shape it is for: text that repeats itself predicts its own continuation.
    #[test]
    fn a_repeated_phrase_predicts_its_own_continuation() {
        let phrase = [41_u32, 42, 43, 44, 45, 46];
        let mut ix = NgramIndex::default();
        ix.observe_all(&phrase);
        ix.observe_all(&[7, 8, 9]);
        ix.observe_all(&phrase);
        // Having seen "41 42 43" twice, the index knows 44 comes next.
        assert_eq!(ix.follows(&[41, 42, 43])[0].0, 44);
        assert_eq!(ix.follows(&[43, 44, 45])[0].0, 46);
    }
}
