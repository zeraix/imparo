//! The backend seam (task #16). llama.cpp's LAYERING without ggml's graph engine:
//! a trait shaped by the REAL op surface the gemma4 workflow uses -- derived from
//! gemma4_metal.rs's call list, not idealized -- plus named buffer slots each backend
//! maps to its own allocations. Model workflows are hand-scheduled functions calling
//! this trait directly; there is no graph IR, no scheduler, no fusion pass (user
//! decision, 2026-08-20: the direct style beat the fork's graph machinery on host
//! overhead, and simplicity is the point of the handoff).
//!
//! Registry/buffer-slot ideas are borrowed from ggml-backend.h as ideas only.
//!
//! Tuned per-backend geometry (the set_* knob family) is deliberately NOT on the
//! trait: knobs are a backend's private business, applied from the shared
//! imparo-host store by the backend's own apply_host_config.

pub mod numerical;

/// Submission at the boundary after request-dependent inputs are materialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VerificationSubmission {
    Eager,
    Capture,
    Replay,
}

/// Named activation/scratch slots. Each backend maps a slot to its own allocation;
/// the ids are the cross-backend vocabulary the workflows speak.
///
/// The discriminants are today's wire values (imparo-metal's `buf` module), so the
/// Metal implementation is a pass-through.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum BufId {
    X = 0,
    Cur = 1,
    Q = 2,
    K = 3,
    V = 4,
    Attn = 5,
    O = 6,
    G = 7,
    U = 8,
    /// MODEL-PRIVATE SLOTS. A buffer only one architecture has does not belong in a
    /// shared enum under that architecture's name: three of these were Gate, Back and
    /// PerLayer, which exist solely for gemma4's per-layer embeddings, while LFM2 had no
    /// way to name its ShortConv projection at all. A model gives its own slots names
    /// (`const GATE: BufId = BufId::Model0;`) and the backend only ever sees an index.
    Model0 = 9,
    Model1 = 10,
    Model2 = 11,
    Logits = 12,
    Tokens = 13,
    Tmp = 14,
    AttnPart = 15,
    Xh = 16,
    Kdq = 17,
    Vdq = 18,
    Xh2 = 19,
    // Appended rather than interleaved: every index above is a wire value the backends
    // already use. Five more because a linear-attention block needs more scratch than a
    // transformer one -- Qwen3.5's gated delta rule carries a conv state, a state matrix
    // and its own intermediates.
    Model3 = 20,
    Model4 = 21,
    Model5 = 22,
    Model6 = 23,
    Model7 = 24,
    /// Per-conversation RECURRENT state: a short convolution's history, a linear
    /// attention's state matrix. Shared rather than model-private because every
    /// recurrent architecture needs exactly one of these and they all need it for the
    /// same reason -- and because it is the one activation-side buffer that must SURVIVE
    /// a batch-width change, which the machinery has to know without asking the model.
    ///
    /// NOT a KV-pool tenant: the pool exists for state that grows with context, and this
    /// is `n_embd * (l_cache - 1)` floats however long the conversation runs.
    Recur = 25,
    /// Where a recurrent snapshot is written when the boundary it describes falls INSIDE
    /// a batch. Same shape as `Recur` and shared for the same reason.
    ///
    /// Why a second buffer rather than stopping the batch on the boundary: chunks are cut
    /// on the prefill grid and pool units are 256 tokens, so a boundary routinely lands
    /// mid-chunk. Cutting the chunk to stand on it measured +29 ms on an 800-token
    /// prefill. The state at that boundary does not need standing on -- it is the tail of
    /// values the batch already computed -- so it is written aside by one small dispatch
    /// while the batch runs to its natural end.
    RecurSnap = 26,
    /// PIPELINED DECODE (docs/decode-turnaround.md): the greedy pick of each queued step,
    /// two u32 slots. Step N writes slot N % 2 and step N+1's gather reads `Tokens[0]`,
    /// which the same kernel wrote; the host reads slot N % 2 after step N retires and
    /// before it queues step N+2, the next writer of that slot.
    Pick = 27,
    /// A batch whose rows are not one causal chain -- a tree verify -- described once per
    /// batch in `ROW_LAYOUT_WORDS` u32 words per row. Only the `_rows` entries read it.
    RowLayout = 28,
    /// A tree verify's input to every convolution window, one row per node: what a commit
    /// rebuilds the accepted path's windows from, instead of a window kept per node.
    RowInputs = 29,
    /// A paired drafter's input: the target's tapped layer outputs, the taps side by side in
    /// one row per forward row (row r, tap k at `r * taps * n_embd + k * n_embd`). Written as
    /// each tapped layer finishes; the drafter reads it after the target's forward.
    DraftFeatures = 30,
    /// A drafter's head logits, one row per drafted position (block x vocab).
    DraftLogits = 31,
    /// The drafted position being biased: one vocabulary row copied out of `DraftLogits`.
    DraftColumn = 32,
    /// The Markov head's bias for the column, `W2 . W1[previous token]`.
    DraftBias = 33,
    /// The Markov head's rank rows `W1[previous token]`, one per drafted position, then the
    /// drafted ids as u32.
    DraftRank = 34,
    /// A drafter's candidates: `top_k_rows` over `DraftLogits` once the chain has written its
    /// biased columns there, ids and values, then the entry's working space.
    DraftTop = 35,
    /// A drafter's confidence head: its weights and bias as the drafter uploaded them, then one
    /// output per drafted position and the logistic entry's working space.
    DraftConfidence = 36,
    /// A ninth model-private slot. A ROUTED feed-forward needs nine pieces of scratch where a
    /// dense one needs none: the router's scores, the probabilities, the biased scores the
    /// pick reads, the picked ids, and the four arrays of the work-row sort -- beside the
    /// convolution projection LFM2's recurrent blocks already hold.
    Model8 = 37,
}

impl BufId {
    /// Entries a backend's buffer table must hold.
    ///
    /// THIS is the definition; a backend's own size is a CAPACITY, not a second opinion.
    /// Adding a model-private slot is one edit here, and it stays one edit as long as the
    /// backends have room -- which is why they are sized well above this rather than
    /// exactly at it. `check_buf_table` turns "the backend is smaller than the enum" from
    /// an out-of-bounds index into a message at init.
    ///
    /// Three declarations of one number is how NO_WEIGHT ended up with two different
    /// values in this codebase, so there is one declaration and a check.
    pub const COUNT: usize = 38;
}

/// Words per row in `BufId::RowLayout`:
///
/// ```text
/// [0]      position     where the row is roped (start + depth for a tree)
/// [1]      depth        steps from the batch's first row along the row's own path
/// [2]      rows 0..31   bit j set: the row attends to batch row j
/// [3]      rows 32..63
/// [4..12]  ancestors    the batch row 1..8 steps up the path, read only for steps <= depth
/// ```
///
/// Every row attends to the whole cache below the batch. A chain is the layout whose row t
/// has position start + t, depth t, rows 0..=t visible and ancestors t - 1, t - 2, ...
/// A row may also see rows after its own: every row of a drafted block sees the whole block.
pub const ROW_LAYOUT_WORDS: usize = 12;
/// Rows one layout can describe: the width of its visibility mask.
pub const ROW_LAYOUT_MAX_ROWS: usize = 64;
/// Ancestors a layout row carries: the longest short-convolution history it serves.
pub const ROW_LAYOUT_ANCESTORS: usize = 8;

/// Whether row-layout `words` (`ROW_LAYOUT_WORDS` per row) hold a tree depth-first with every
/// node shallower than `depth_limit`: the first row at depth 0, and each later row a child of
/// the latest row one level up, at most one level below the row before it. A kernel that walks
/// the rows in order, as the gated delta rule's tree verify does, then finds every node's parent
/// state where the parent left it.
#[must_use]
pub fn row_layout_depth_first(words: &[u32], depth_limit: u32) -> bool {
    if words.is_empty() || words.len() % ROW_LAYOUT_WORDS != 0 {
        return false;
    }
    // The latest row at each depth of the current path.
    let mut path: Vec<u32> = Vec::new();
    for (t, row) in words.chunks_exact(ROW_LAYOUT_WORDS).enumerate() {
        let Ok(t) = u32::try_from(t) else {
            return false;
        };
        let (depth, parent) = (row[1] as usize, row[4]);
        let fits = row[1] < depth_limit
            && if t == 0 {
                depth == 0
            } else {
                depth >= 1 && depth <= path.len() && path[depth - 1] == parent
            };
        if !fits {
            return false;
        }
        path.truncate(depth);
        path.push(t);
    }
    true
}

#[cfg(test)]
mod row_layout_depth_first_tests {
    use super::{ROW_LAYOUT_WORDS, row_layout_depth_first};

    /// Layout words holding only what the check reads: each row's depth and parent.
    fn words(rows: &[(u32, u32)]) -> Vec<u32> {
        rows.iter()
            .flat_map(|&(depth, parent)| {
                let mut row = [0_u32; ROW_LAYOUT_WORDS];
                row[1] = depth;
                row[4] = parent;
                row
            })
            .collect()
    }

    #[test]
    fn a_depth_first_tree_passes_and_other_orders_do_not() {
        // 0 -> 1 -> 2, then 3 under 0, then 4 under 3.
        assert!(row_layout_depth_first(
            &words(&[(0, 0), (1, 0), (2, 1), (1, 0), (2, 3)]),
            16
        ));
        // 4 under 1, which 3 has replaced at depth 1.
        assert!(!row_layout_depth_first(
            &words(&[(0, 0), (1, 0), (2, 1), (1, 0), (2, 1)]),
            16
        ));
        // A row two levels below the row before it.
        assert!(!row_layout_depth_first(&words(&[(0, 0), (2, 0)]), 16));
        // A second root.
        assert!(!row_layout_depth_first(
            &words(&[(0, 0), (1, 0), (0, 0)]),
            16
        ));
    }

    #[test]
    fn a_node_at_the_depth_limit_is_refused() {
        let chain: Vec<(u32, u32)> =
            (0_u32..17).map(|t| (t, t.saturating_sub(1))).collect();
        assert!(row_layout_depth_first(&words(&chain[..16]), 16));
        assert!(!row_layout_depth_first(&words(&chain), 16));
    }
}

/// The rows of row-layout `words` that have more than one child, as a mask (bit t for row t):
/// in a depth-first layout (`row_layout_depth_first`) a later child starts again from the state
/// its parent left, after the first child's subtree, so a kernel that walks the tree keeps the
/// state after exactly these rows. Rows past the mask's 64 bits are not marked.
#[must_use]
pub fn row_layout_branching_rows(words: &[u32]) -> u64 {
    let mut children = [0_u32; ROW_LAYOUT_MAX_ROWS];
    for row in words.chunks_exact(ROW_LAYOUT_WORDS) {
        if row[1] > 0
            && let Some(count) = children.get_mut(row[4] as usize)
        {
            *count += 1;
        }
    }
    children
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 1)
        .fold(0, |mask, (t, _)| mask | (1_u64 << t))
}

#[cfg(test)]
mod row_layout_branching_rows_tests {
    use super::{ROW_LAYOUT_WORDS, row_layout_branching_rows};

    #[test]
    fn only_rows_with_a_second_child_are_marked() {
        // 0 -> 1 -> 2, then 3 under 0 and 4 under 3: row 0 has children 1 and 3.
        let rows: [(u32, u32); 5] = [(0, 0), (1, 0), (2, 1), (1, 0), (2, 3)];
        let words: Vec<u32> = rows
            .iter()
            .flat_map(|&(depth, parent)| {
                let mut row = [0_u32; ROW_LAYOUT_WORDS];
                row[1] = depth;
                row[4] = parent;
                row
            })
            .collect();
        assert_eq!(row_layout_branching_rows(&words), 0b1);
        assert_eq!(row_layout_branching_rows(&words[..3 * ROW_LAYOUT_WORDS]), 0);
    }
}

/// Fail loudly when a backend's buffer table cannot hold every `BufId`.
///
/// # Errors
/// Returns a message naming both sizes.
pub fn check_buf_table(backend: &str, capacity: usize) -> Result<(), String> {
    if capacity < BufId::COUNT {
        return Err(format!(
            "{backend} buffer table holds {capacity} entries but BufId needs {};              the backend would index out of bounds",
            BufId::COUNT
        ));
    }
    Ok(())
}

/// The weight-type -> kernel table index (see imparo-cpu's WeightKind, whose
/// discriminants match this wire).
pub type WeightKindWire = u32;

/// Sentinel w_off meaning "no weight vector" for rms_norm (normalize only).
///
/// THE definition -- backends must compare against this and nothing else. It was
/// 0xFFFF_FFFF, and two other places disagreed with it: imparo-metal declared its own copy
/// and the CUDA backend tested `w_off != u64::MAX`, so a no-weight norm on CUDA read a
/// weight vector at offset 4294967295. All-ones in 64 bits cannot be a real tensor offset,
/// which is what the Metal kernel's IMPARO_NO_WEIGHT now is too.
pub const NO_WEIGHT: u64 = u64::MAX;

/// The mixer of an LFM2 decode layer, for [`Backend::mega_lfm2_layer`].
#[derive(Clone, Copy, Debug)]
pub enum Lfm2MegaMixer {
    /// The tail alone: `add` already holds the mixer's output.
    None,
    /// The short convolution: the operator norm of `x`, `in_proj` (3 x n_embd rows) into
    /// `bcx`, the conv step over the `kernel`-tap history at `state[state_off..]`
    /// (elements), `out_proj` into `add`; the history advances in place, and `snap`
    /// (buffer, element offset) receives the advanced history too when a checkpoint is
    /// armed at this token.
    ShortConv {
        op_norm_off: u64,
        in_kind: WeightKindWire,
        in_off: u64,
        conv_off: u64,
        out_kind: WeightKindWire,
        out_off: u64,
        kernel: u32,
        bcx: BufId,
        state: BufId,
        /// The plane the conv READS.
        state_off: u32,
        /// The plane its shift WRITES; equal to `state_off` is the in-place form. Two
        /// offsets because the shift carries values forward from the plane it read
        /// (task #165).
        state_out_off: u32,
        snap: Option<(BufId, u32)>,
    },
    /// Full attention at one token: the operator norm of `x`, the q/k/v projections, per-head
    /// `rms(w_qn)` + NEOX rope + the score scale on Q, `rms(w_kn)` + rope on K, K and V into
    /// layer `kv_layer`'s cache at `start_pos`, attention over the cache into `attn`, then
    /// `wo` into `add`. `q`, `k`, `v` receive the rows (scratch afterwards).
    Attention {
        op_norm_off: u64,
        wq_kind: WeightKindWire,
        wq_off: u64,
        wk_kind: WeightKindWire,
        wk_off: u64,
        wv_kind: WeightKindWire,
        wv_off: u64,
        wo_kind: WeightKindWire,
        wo_off: u64,
        q_norm_off: u64,
        k_norm_off: u64,
        head_dim: u32,
        n_heads: u32,
        n_kv: u32,
        kv_width: u32,
        kv_layer: u32,
        start_pos: u32,
        window: u32,
        ring: u32,
        rope_dim: u32,
        rope_base: f32,
        q_scale: f32,
        q: BufId,
        k: BufId,
        v: BufId,
        attn: BufId,
        /// The cache basis (see [`MegaAttn::had_k`] / [`MegaAttn::had_v`]).
        had_k: u32,
        had_v: u32,
    },
}

/// The operands of one LFM2 decode layer, one variant of [`MegaLayer`]: the mixer (which
/// forms the layer's operator norm from `x` itself), then the tail `x = x + add;
/// cur = rms(x) * w_ffn; x = x + down(act(gate*cur) * (up*cur))`. `g` and `u` are scratch.
#[derive(Clone, Copy, Debug)]
pub struct Lfm2MegaLayer {
    pub mixer: Lfm2MegaMixer,
    pub gate_kind: WeightKindWire,
    pub gate_off: u64,
    pub up_kind: WeightKindWire,
    pub up_off: u64,
    pub down_kind: WeightKindWire,
    pub down_off: u64,
    pub ffn_norm_off: u64,
    pub n_embd: u32,
    pub n_ff: u32,
    pub eps: f32,
    pub x: BufId,
    pub add: BufId,
    pub g: BufId,
    pub u: BufId,
}

/// The operands of one gemma4 decode layer, one variant of [`MegaLayer`]. The tail runs from
/// the FFN input `src`: `x = add + rms(down(act(gate*src) * (up*src))) * w1`, then the
/// per-layer block `back = pp(act(pg * x) * per_layer[per_layer_off..])`, then
/// `x = (x + rms(back) * w2) * out_scale` and, when `next_norm_off` is a weight,
/// `next_out = rms(x) * w3`. `front` moves the entry's start earlier in the layer.
/// `g`, `u`, `gate`, `back` are scratch.
#[derive(Clone, Copy, Debug)]
pub struct Gemma4MegaLayer<'a> {
    pub gate_kind: WeightKindWire,
    pub gate_off: u64,
    pub up_kind: WeightKindWire,
    pub up_off: u64,
    pub down_kind: WeightKindWire,
    pub down_off: u64,
    pub post_ffw_norm_off: u64,
    pub pg_kind: WeightKindWire,
    pub pg_off: u64,
    pub pp_kind: WeightKindWire,
    pub pp_off: u64,
    pub post_norm_off: u64,
    pub next_norm_off: u64,
    pub out_scale: f32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub ple: u32,
    pub per_layer_off: u32,
    pub eps: f32,
    pub src: BufId,
    pub x: BufId,
    pub add: BufId,
    pub g: BufId,
    pub u: BufId,
    pub gate: BufId,
    pub per_layer: BufId,
    pub back: BufId,
    pub next_out: BufId,
    pub front: Option<MegaFront<'a>>,
}

/// What precedes qwen35's shared tail. `None` is the tail alone, with `add` already holding
/// the mixer's output; the two mixers (full attention with a packed `[query | gate]`
/// projection, and the gated delta net) join as their phases land (task #165).
#[derive(Clone, Copy, Debug)]
pub enum Qwen35MegaMixer {
    /// `add` holds the mixer's output: the dispatch path ran it.
    None,
}

/// The operands of one Qwen3.8 decode layer, one variant of [`MegaLayer`]: the mixer, then
/// the tail `x' = x + add; cur = rms(x') * w_ffn; x = x' + down(act(gate*cur) * (up*cur))`.
///
/// EVERY WEIGHT CARRIES ITS OWN KIND, and that is the architecture, not caution: this
/// file's quants are assigned per tensor by importance, so two projections of one layer
/// routinely differ (one UD file: 53 distinct per-block signatures over 65 blocks). A
/// backend that stamps one format into its pipeline cannot serve it.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35MegaLayer {
    pub mixer: Qwen35MegaMixer,
    pub gate_kind: WeightKindWire,
    pub gate_off: u64,
    pub up_kind: WeightKindWire,
    pub up_off: u64,
    pub down_kind: WeightKindWire,
    pub down_off: u64,
    pub ffn_norm_off: u64,
    pub n_embd: u32,
    pub n_ff: u32,
    pub eps: f32,
    pub x: BufId,
    pub add: BufId,
    pub g: BufId,
    pub u: BufId,
    /// The FFN-normalised row. Whether it lives in device or threadgroup memory is the
    /// backend's choice (task #175); the model only says which buffer holds it.
    pub cur: BufId,
}

/// The ROUTED feed-forward of one LFM2-MoE decode layer, one variant of [`MegaLayer`]. The
/// mixer ran on the dispatch path and left its output in `add`; the entry forms
/// `x' = x + add; cur = rms(x') * w_ffn`, the router's rows into `scores`, the route (gating,
/// the pick over the probabilities plus `bias`, the weights), the k experts' gated rows into
/// `g` (k rows of `n_ff`), and ends with `x = x' + sum_j w_j * down_j(g_j)` summed in slot
/// order. Each expert stack is addressed by its base offset and its per-expert byte stride.
#[derive(Clone, Copy, Debug)]
pub struct Lfm2MoeMegaLayer {
    pub router_kind: WeightKindWire,
    pub router_off: u64,
    /// The file's selection bias (`exp_probs_b`), [`NO_WEIGHT`] when it has none -- the
    /// convention `Backend::moe_route` takes.
    pub bias_off: u64,
    pub gate_kind: WeightKindWire,
    pub gate_off: u64,
    pub gate_stride: u64,
    pub up_kind: WeightKindWire,
    pub up_off: u64,
    pub up_stride: u64,
    pub down_kind: WeightKindWire,
    pub down_off: u64,
    pub down_stride: u64,
    pub ffn_norm_off: u64,
    pub n_embd: u32,
    /// One expert's hidden width.
    pub n_ff: u32,
    pub n_expert: u32,
    pub k: u32,
    pub gating: ExpertGating,
    pub normalise: bool,
    pub weights_scale: f32,
    pub eps: f32,
    pub x: BufId,
    pub add: BufId,
    pub g: BufId,
    pub scores: BufId,
    /// The next layer's operator-norm weight, [`NO_WEIGHT`] for none: when given, the entry
    /// ends by writing `rms(x) * w` into `cur`, which is that layer's mixer input.
    pub next_norm_off: u64,
    pub cur: BufId,
}

/// What one decode layer IS, per architecture. A backend that runs the whole layer as one
/// operation reads the variant it knows; a new architecture is a new variant, never a new
/// `Backend` method.
#[derive(Clone, Copy, Debug)]
pub enum MegaLayer<'a> {
    Gemma4(Gemma4MegaLayer<'a>),
    Lfm2(Lfm2MegaLayer),
    Qwen35(Qwen35MegaLayer),
    Lfm2Moe(Lfm2MoeMegaLayer),
}

/// One decode layer offered to the backend as a MEGA ENTRY (see [`Backend::mega_layer`]).
#[derive(Clone, Copy, Debug)]
pub struct MegaEntry<'a> {
    pub layer: MegaLayer<'a>,
    /// Tokens in this forward. The mega route is a decode route: a backend refuses anything
    /// but 1 rather than growing a batched form.
    pub n_tok: u32,
}

/// One execution backend. Method-for-method the surface `gemma4_metal.rs` actually
/// calls; buffer arguments are `BufId`, weights are addressed by byte offset into
/// the backend's mapped weight blob.
///
/// KV ownership (alloc_kv/grow_kv/kv_store/kv_dequant/read_kv_bytes) sits behind
/// this trait ON PURPOSE: the unified-kv-pool + disk tier implements these same
/// calls without touching any workflow.
#[allow(clippy::too_many_arguments)]
/// The gated activation a matmul kernel fuses into its write-back.
///
/// The wire values are what the kernels compare against, so they are FIXED; `EPI_NONE`,
/// `EPI_GELU` and `EPI_SILU` in imparo.metal must agree with them.
///
/// This was a `bool`. "Epilogue on" meant GELU, because gemma4 was the only model, and a
/// SwiGLU model that turned it on would have got GELU and plausible wrong numbers -- the
/// knob table carried a comment warning not to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Epilogue {
    /// Write the product; no activation.
    None = 0,
    /// `y = gelu(y) * product`. gemma4's feed-forward.
    Gelu = 1,
    /// `y = silu(y) * product`. SwiGLU -- LFM2's, and most other architectures'.
    Silu = 2,
}

/// How a co-batched decode step's projections serve their rows (`Backend::set_decode_rows`,
/// docs/continuous-batching.md, section 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowRoute {
    /// Every row takes the one-row decode's arithmetic, so its bits equal its lone decode on
    /// the dispatch path. The route that proves the per-row plumbing: any difference is a
    /// block table, position or state slot, never arithmetic.
    Exact,
    /// Whichever kernel the backend measured fastest at the step's row count. A row's tokens
    /// match its lone decode's to the standard the mega route keeps against the dispatch path.
    Fast,
}

/// One row of a co-batched decode step (`Backend::kv_store_slot_rows`,
/// `Backend::attention_slot_rows`): the slot whose cache it reads and writes, and the
/// position of its token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotRow {
    pub slot: u32,
    pub pos: u32,
}

/// One row of a co-batched recurrent step (`Backend::causal_conv_slot_rows`,
/// `Backend::delta_net_slot_rows`): its slot and where its state -- a convolution's history,
/// a delta rule's matrix -- is read and written, as element offsets into that slot's state
/// buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotStateRow {
    pub slot: u32,
    pub state_off: u32,
    pub state_out_off: u32,
}

/// What a causal depthwise convolution reads, and what it does with the tap sum.
///
/// The convolution itself -- one channel, `kernel` taps, a per-conversation history of
/// `kernel - 1` past values, causal -- is the same in every architecture that has one.
/// What differs is the VALUE the taps run over and the epilogue, and those two facts are
/// this enum. They are a compile-time choice on the device, not a runtime branch: a
/// runtime branch in a Metal kernel costs speed and moves the floating-point answer
/// (`set_activation` carries the measurement).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ConvForm {
    /// LFM2's gated short convolution. The source row is `[b | c | x]`, three widths per
    /// token; the value is `b * x` and the output is `c * sum`.
    GatedBcx = 0,
    /// Qwen3.8's delta-net convolution. The source row is ONE width per token, the value
    /// is the row itself, and the output is `silu(sum)`.
    ///
    /// The history holds the RAW value, never the activated output. Storing the output
    /// would decay the history through the activation -- a slow drift that still runs.
    PlainSilu = 1,
}

impl ConvForm {
    /// Elements one token occupies in the source buffer, for a convolution of `width`
    /// channels. The gated form packs three chunks per token; the plain form packs one.
    #[must_use]
    pub fn src_stride(self, width: u32) -> u32 {
        match self {
            Self::GatedBcx => 3 * width,
            Self::PlainSilu => width,
        }
    }
}

/// One gated delta-net step for every value head, over a whole batch.
///
/// A struct because the op takes fourteen operands and a positional list of them is a
/// silent-wrong-answer waiting to happen -- `k_heads` and `v_heads` are both small
/// integers, and swapping them still runs.
///
/// THE RULE, per value head `h`, reading key head `h % k_heads`:
///
/// ```text
///   S      *= exp(g[h])                   g = a[h] * softplus(alpha[h] + dt_bias[h])
///   sk[j]   = SUM_i S[j][i] * khat[i]     what the state already remembers of k
///   d[j]    = (v[j] - sk[j]) * beta[h]    beta = sigmoid(the beta projection)
///   S[j][i]+= khat[i] * d[j]              rank-one update
///   o[j]    = SUM_i S[j][i] * qhat[i]     read it back with the query
/// ```
///
/// `qhat` and `khat` are the L2-normalised query and key of that head, `qhat` also scaled
/// by `1 / sqrt(key_dim)` BEFORE the state product -- the same order the reference rounds
/// in. `S` is `[value_head][value_coord][key_coord]` with the key coordinate contiguous.
///
/// KEY HEAD MAPPING. The reference widens Q and K from `k_heads` to `v_heads` with a
/// repeat, and a repeat TILES: value head `h` reads key head `h % k_heads`, NOT
/// `h / group`. A grouped-query attention in the same model uses `h / group`, because
/// there the widening is a strided view. Both are right for their own tensor.
/// The gated-RMS epilogue that follows the delta rule in every architecture that has one:
/// `out = rms_norm(core, w) * silu(gate)`, per VALUE HEAD.
///
/// It is a SEPARATE type, not two more fields, because it is optional as a UNIT -- a
/// backend either does both steps inside the rule or neither, and half of it is not a
/// state anyone should be able to construct.
/// A checkpoint boundary inside a delta-rule batch: after `row` tokens of the batch the
/// rule writes the matrix as it stands to `buf` at `off` and carries on.
///
/// The convolution history is written aside the same way (`causal_conv_snapshot`), and a
/// recurrent state is BOTH: a checkpoint that carries the history without the matrix
/// restores a model that saw the prefix's last three tokens and none of the rest.
/// Measured on Qwen3.8-27B: a conversation adopting another's 768-token prefix answered
/// differently from its first reasoning token, because the matrix region of the
/// snapshot plane was never written (docs/kv-identity-grid.md).
#[derive(Clone, Copy, Debug)]
pub struct DeltaSnapshot {
    pub buf: BufId,
    /// Element offset of this layer's matrix region in `buf` -- the snapshot plane's
    /// copy of `DeltaNet::state_off`.
    pub off: u32,
    /// Tokens of the batch before the boundary; the matrix after the update of token
    /// `row - 1` is what is written.
    pub row: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DeltaEpilogue {
    /// `ssm_norm` in the weight buffer: ONE head's worth of weights (`value_dim`),
    /// shared by every value head, which is exactly `rms_norm`'s row geometry.
    pub norm_w_off: u64,
    /// The SiLU gate, `v_heads * value_dim` per token, READ AND WRITTEN: the fused form
    /// leaves `silu(gate) * norm(core)` here, which is what the out projection reads.
    pub gate: BufId,
}

/// A packed matrix can declare an orthogonal input basis independently of its codec.
/// Model metadata owns this contract; it is never a performance knob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedHeads {
    pub head_dim: u32,
    pub key_heads: u32,
    pub value_heads: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightInputTransform {
    pub weight_offset: u64,
    pub width: u32,
    pub block_size: u32,
    pub signs: Vec<i8>,
    /// Embedding rows apply the inverse basis after decoding, instead of rotating
    /// activations before a matrix product.
    pub inverse: bool,
    pub permutation: Option<GroupedHeads>,
}

pub struct DeltaNet {
    /// The convolved projection, `2 * k_heads * key_dim + v_heads * value_dim` elements
    /// per token, packed `[Q | K | V]`.
    pub qkv: BufId,
    /// The raw alpha projection, `v_heads` per token. `softplus` and `dt_bias` are the
    /// op's, so a caller cannot apply them in the wrong order.
    pub alpha: BufId,
    /// The raw beta projection, `v_heads` per token; the op takes its sigmoid.
    pub beta: BufId,
    /// `ssm_a` in the weight buffer: one value per value head, ALREADY negated in the
    /// file (`-exp(A_log)`), so `a * softplus(dt)` is the negative log decay.
    pub a_off: u64,
    /// `ssm_dt.bias` in the weight buffer, one value per value head.
    pub dt_bias_off: u64,
    /// The recurrent matrix, `v_heads * value_dim * key_dim` floats at `state_off`.
    pub state: BufId,
    /// The plane the step READS.
    pub state_off: u32,
    /// The plane it WRITES. `state_out_off == state_off` is the in-place form, which is
    /// what prefill and every non-recurrent caller pass.
    ///
    /// WHY A SECOND OFFSET AND NOT A COPY: the rule reads and writes EVERY element of the
    /// head's matrix on every call (the kernel loads `s[r][c]` for all `j < value_dim`,
    /// `i < key_dim` and stores all of them back), so directing the store at another plane
    /// costs an address, not traffic. That is what lets a failed decode step be rolled
    /// back by NOT ADVANCING an index instead of by copying 149.6 MiB before every step.
    pub state_out_off: u32,
    /// `v_heads * value_dim` per token. With a fused `epilogue` this buffer is NOT
    /// written -- the normalised, gated row goes straight to `epilogue.gate`.
    pub out: BufId,
    /// The gated-RMS epilogue, folded into the rule by a backend that advertises
    /// `delta_net_fuses_epilogue`. `None`, or a backend that does not advertise it,
    /// leaves the caller to dispatch `rms_norm` and `act_mul` itself.
    pub epilogue: Option<DeltaEpilogue>,
    /// The boundary snapshot, when a checkpoint boundary falls inside this batch.
    pub snap: Option<DeltaSnapshot>,
    pub k_heads: u32,
    pub v_heads: u32,
    /// The Q/K head width, which is also the state's key coordinate.
    pub key_dim: u32,
    /// The V head width, which is also the state's value coordinate.
    pub value_dim: u32,
    pub n_tok: u32,
    /// The floor of the L2 normalisation: `max(norm, eps)`, not `norm + eps`. The second
    /// shrinks every vector slightly, which is a different function.
    pub eps: f32,
}

/// Model-owned FFN boundary for optional decode-graph experiments.
/// This describes semantic work; backends may ignore it without changing outputs.
#[derive(Clone, Copy, Debug)]
pub struct DecodeFfnPhase {
    pub layer: u32,
    pub gate_kind: WeightKindWire,
    pub up_kind: WeightKindWire,
    pub down_kind: WeightKindWire,
    pub gate_off: u64,
    pub up_off: u64,
    pub down_off: u64,
    pub n_in: u32,
    pub n_mid: u32,
    pub n_out: u32,
    pub src: BufId,
    pub dst: BufId,
    pub tokens: u32,
    pub activation: Epilogue,
}

/// Stable, typed policy for resolving the block width of the orthonormal
/// Hadamard rotation used before quantized KV storage.
///
/// This is a numerical-route property, not a performance knob: changing it changes
/// persisted cache bytes and can change recurrent decode. Backends may select a route
/// compatible with their reference implementation without leaking backend cfgs into a
/// model workflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HadamardWidth {
    Disabled,
    FullHead,
    Fixed(u32),
}

impl HadamardWidth {
    /// Resolve and validate the route for one attention head.
    ///
    /// Zero disables the transform. Any enabled width must be a power of two that
    /// divides the head dimension, matching the kernel contract.
    pub fn resolve(self, head_dim: u32) -> Result<u32, &'static str> {
        let width = match self {
            Self::Disabled => return Ok(0),
            Self::FullHead => head_dim,
            Self::Fixed(width) => width,
        };
        if width == 0 || !width.is_power_of_two() || head_dim % width != 0 {
            return Err(
                "Hadamard width must be a nonzero power-of-two divisor of head_dim",
            );
        }
        Ok(width)
    }
}

/// Key/value rotation widths are kept together because they jointly define the cache
/// representation. The default exactly preserves the pre-CUDA shared workflow/Metal
/// route; CUDA overrides it with its separately gated reference-compatible route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvQuantizationRoute {
    pub key: HadamardWidth,
    pub value: HadamardWidth,
}

impl Default for KvQuantizationRoute {
    fn default() -> Self {
        Self {
            key: HadamardWidth::FullHead,
            value: HadamardWidth::Fixed(128),
        }
    }
}

/// Stable byte codec used to encode one KV side after any model-owned basis transform.
///
/// This is a durable-format contract, not a backend name. Backends may advertise the
/// same value only when they produce byte-identical blocks for every finite input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvByteCodec {
    /// IEEE-754 binary16 values stored little-endian with round-to-nearest-even.
    F16LeRneV1,
    /// llama.cpp q4_0: one f16 scale and 32 unsigned nibbles per block.
    Q4_0LlamaV1,
    /// q8_0 with halfway cases rounded to the nearest even integer.
    Q8_0RintEvenV1,
    /// q8_0 with halfway cases rounded away from zero.
    Q8_0RoundAwayV1,
    /// Reserved for rejected profiles; no backend may write this format.
    Rejected,
}

/// Codec declarations for the three supported KV wire types. The model layer selects
/// the entry for K and V independently after resolving their configured types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvByteCodecRoute {
    pub f16: KvByteCodec,
    pub q4_0: KvByteCodec,
    pub q8_0: KvByteCodec,
}

impl Default for KvByteCodecRoute {
    fn default() -> Self {
        Self {
            f16: KvByteCodec::F16LeRneV1,
            q4_0: KvByteCodec::Q4_0LlamaV1,
            q8_0: KvByteCodec::Q8_0RintEvenV1,
        }
    }
}

/// Logical KV geometry passed to backends that maintain a device page table.
///
/// This descriptor deliberately carries slots and the independent K/V byte strides.
/// A backend must never infer either from an allocation byte count: mixed cache types,
/// window regions and alignment can all make that inference ambiguous. The fields use
/// fixed-width integers so CUDA can mirror the contract in its versioned C ABI.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct KvLayout {
    pub layer: u32,
    pub reserved: u32,
    pub logical_slots: u64,
    pub k_stride: u64,
    pub v_stride: u64,
}

impl KvLayout {
    /// Number of page-table entries needed to address this layer.
    ///
    /// TAKES the page rather than naming it. There was a `KV_PAGE_CELLS = 64` here,
    /// which made a fourth independent copy of a number the backend already declares
    /// as `PoolCaps::page_cells` and the pool already holds as
    /// `imparo_kv::identity::page_cells()`. A page that only one of the four knows
    /// about is how a device page table and the placement addressing it drift apart
    /// silently. Pass `pool_caps().page_cells`.
    pub fn page_count(self, page_cells: u32) -> Result<u32, &'static str> {
        if self.reserved != 0 {
            return Err("KV layout reserved field must be zero");
        }
        if self.logical_slots > 0 && (self.k_stride == 0 || self.v_stride == 0) {
            return Err("non-empty KV layout requires nonzero K/V strides");
        }
        if page_cells == 0 || !page_cells.is_power_of_two() {
            return Err("KV page must be a non-zero power of two");
        }
        let page = u64::from(page_cells);
        let pages =
            self.logical_slots / page + u64::from(self.logical_slots % page != 0);
        u32::try_from(pages).map_err(|_| "KV page count exceeds u32")
    }

    /// Largest legal physical page entry, or `None` for an empty layer.
    pub fn max_page_entry(self, page_cells: u32) -> Result<Option<u32>, &'static str> {
        Ok(self.page_count(page_cells)?.checked_sub(1))
    }
}

/// Semantic phase of one model batch. This is execution identity, not a performance knob.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum BatchPhase {
    Prefill = 0,
    Decode = 1,
}

/// Absolute geometry shared by every op in one device batch.
///
/// Backends whose numerical routes depend on canonical token cells use this descriptor
/// instead of inferring position from whichever layer happened to run most recently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchGeometry {
    pub absolute_start: u64,
    pub active_tokens: u32,
    pub phase: BatchPhase,
    /// The cold batch interval whose numerical family this physical suffix inherits.
    pub canonical_start: u64,
    pub canonical_tokens: u32,
}

impl BatchGeometry {
    /// Constructs a non-empty range whose exclusive end is representable.
    pub fn try_new(
        absolute_start: u64,
        active_tokens: u32,
        phase: BatchPhase,
    ) -> Result<Self, &'static str> {
        if active_tokens == 0 {
            return Err("batch geometry must contain at least one token");
        }
        absolute_start
            .checked_add(u64::from(active_tokens))
            .ok_or("batch geometry end overflows u64")?;
        Ok(Self {
            absolute_start,
            active_tokens,
            phase,
            canonical_start: absolute_start,
            canonical_tokens: active_tokens,
        })
    }

    /// Describe a physical suffix in an absolute prefill cell. The cell width
    /// belongs to the model scheduler; backends must not guess a fixed512 grid.
    pub fn try_prefill(
        absolute_start: u64,
        active_tokens: u32,
        cell_tokens: u32,
    ) -> Result<Self, &'static str> {
        if cell_tokens == 0 {
            return Err("prefill cell must contain at least one token");
        }
        let mut geometry =
            Self::try_new(absolute_start, active_tokens, BatchPhase::Prefill)?;
        let prefix = (absolute_start % u64::from(cell_tokens)) as u32;
        let canonical_tokens = prefix
            .checked_add(active_tokens)
            .ok_or("canonical token count overflows u32")?;
        if canonical_tokens > cell_tokens {
            return Err("physical batch crosses a canonical cell");
        }
        geometry.canonical_start = absolute_start - u64::from(prefix);
        geometry.canonical_tokens = canonical_tokens;
        Ok(geometry)
    }

    #[must_use]
    pub fn canonical_offset(self, cell_tokens: u32, local_token: u32) -> Option<u32> {
        if cell_tokens == 0 || local_token >= self.active_tokens {
            return None;
        }
        let absolute = self.absolute_start.checked_add(u64::from(local_token))?;
        Some((absolute % u64::from(cell_tokens)) as u32)
    }
}

#[cfg(test)]
mod batch_geometry_tests {
    use super::{BatchGeometry, BatchPhase, StreamedWeightSpan};

    #[test]
    fn cold_and_split_rows_share_canonical_offsets() {
        let cold = BatchGeometry::try_new(512, 232, BatchPhase::Prefill).unwrap();
        let split = BatchGeometry::try_new(704, 40, BatchPhase::Prefill).unwrap();
        assert_eq!(cold.canonical_offset(512, 192), Some(192));
        assert_eq!(split.canonical_offset(512, 0), Some(192));
        assert_eq!(cold.canonical_offset(512, 231), Some(231));
        assert_eq!(split.canonical_offset(512, 39), Some(231));
    }

    #[test]
    fn canonical_family_survives_suffix_pruning_with_different_cell_widths() {
        for cell in [128, 256, 384, 512, 1024] {
            let start = u64::from(cell) * 2;
            let cold = BatchGeometry::try_prefill(start, 66, cell).unwrap();
            let tail = BatchGeometry::try_prefill(start + 64, 2, cell).unwrap();
            assert_eq!(
                (cold.canonical_start, cold.canonical_tokens),
                (tail.canonical_start, tail.canonical_tokens)
            );
            assert_ne!(cold.active_tokens, tail.active_tokens);
            let truly_small = BatchGeometry::try_prefill(start, 2, cell).unwrap();
            assert_eq!(truly_small.canonical_tokens, 2);
        }
        assert!(BatchGeometry::try_prefill(0, 1, 0).is_err());
        assert!(BatchGeometry::try_prefill(127, 2, 128).is_err());
        assert!(BatchGeometry::try_prefill(u64::MAX, 1, 512).is_err());
    }

    #[test]
    fn invalid_geometry_fails_closed() {
        assert!(BatchGeometry::try_new(0, 0, BatchPhase::Prefill).is_err());
        assert!(BatchGeometry::try_new(u64::MAX, 1, BatchPhase::Decode).is_err());
    }

    #[test]
    fn phase_is_part_of_geometry_identity() {
        let prefill = BatchGeometry::try_new(7, 1, BatchPhase::Prefill).unwrap();
        let decode = BatchGeometry::try_new(7, 1, BatchPhase::Decode).unwrap();
        assert_ne!(prefill, decode);
    }
    #[test]
    fn streamed_weight_span_has_stable_wire_layout() {
        assert_eq!(core::mem::size_of::<StreamedWeightSpan>(), 16);
        assert_eq!(core::mem::align_of::<StreamedWeightSpan>(), 8);
        let span = StreamedWeightSpan {
            offset: 7,
            bytes: 11,
        };
        assert_eq!(span, span);
    }
}

/// A byte range in the mapped model that the backend may keep outside its hot
/// device-resident weight allocation. Offsets are absolute within the mapping.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamedWeightSpan {
    pub offset: u64,
    pub bytes: u64,
}

/// Where one contiguous byte range of the weight file LIVES (docs/memory-tiers-and-fit.md).
/// The tier names the place, not a transport: how a slow-tier layer is executed (fetched
/// into a fast-tier slot ahead of use, or computed where it lies) is the backend's per-layer
/// decision, measured by the tuner, and not part of the placement.
///
/// `Fast`: wired unified memory on Metal, the VRAM copy on CUDA.
/// `Slow { layer }`: a whole layer the fast tier could not hold; the pageable mapping on
/// Metal, host RAM on CUDA. `HostStaged`: a row-gathered tensor (an embedding table read by token
/// id) that lives in no tier: the host gathers the rows a batch needs into a small
/// fast-tier staging buffer. `Unread`: a block the forward never runs; no tier holds it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightTier {
    Fast,
    Slow {
        layer: u32,
    },
    /// Never bound: a row-gathered tensor (rows indexed by token id, e.g. gemma4's
    /// per-layer token embeddings) whose rows the host copies into a small device buffer
    /// per batch (`Backend::stage_rows`). The smallest wired set, and the reason a decode
    /// step on such a model needs the next token on the host before it is encoded.
    HostStaged,
    /// Never read: a block past the plan's layer count, such as the multi-token-prediction
    /// head a Qwen3.8 file carries and the plan drops. It gets no buffer and is never wired
    /// or copied; a backend that streams its non-fast spans lists it and never touches it.
    Unread,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightSegment {
    pub offset: u64,
    pub bytes: u64,
    pub tier: WeightTier,
}

/// The arithmetic behind a placement, kept so the startup line and a later reader can see
/// why the split came out as it did. All bytes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TierBudget {
    /// The fast tier the backend reported (None: unknown, everything resident as before).
    pub total: Option<u64>,
    pub reserve_kv: u64,
    pub reserve_activations: u64,
    pub reserve_scratch: u64,
    pub margin: u64,
    /// `total - reserve`, or u64::MAX when `total` is unknown.
    pub weight_budget: u64,
}

/// One load-time repack job: a tensor's file span, its dimensions and the types it moves
/// between (see `Backend::transform_weights`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WeightTransform {
    pub offset: u64,
    pub bytes: u64,
    pub from_type: u32,
    pub to_type: u32,
    pub n_in: u32,
    pub n_out: u32,
    /// THE LAYOUT, as data. A repack moves a block's scale bytes to the head of a unit and
    /// its payload after them; which bytes are which is stated once, in imparo-gguf's
    /// `TmRule`, and travels here. A backend that re-derived the layout from `from_type`
    /// would carry a second copy of it, and a copy puts every payload at a wrong offset
    /// the moment a format is added -- a slightly wrong weight, never a crash.
    pub layout: WeightBlockLayout,
}

/// Two scale spans is the most any format needs (Q2_K and IQ3_S split them front and back),
/// leaving at most three payload spans between and around them.
pub const MAX_BLOCK_SPANS: usize = 5;

/// Where a block's bytes are, and where the repack puts them.
///
/// Spans are the SCALE spans first (`n_scale_spans` of them), then the payload spans, in
/// source order. The repack is format-agnostic: it moves the spans it is given.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightBlockLayout {
    pub block_elems: u32,
    pub block_bytes: u32,
    pub unit_rows: u32,
    pub n_spans: u32,
    pub n_scale_spans: u32,
    pub span_off: [u32; MAX_BLOCK_SPANS],
    pub span_len: [u32; MAX_BLOCK_SPANS],
}

/// The common runtime's placement of a model's weights across the tiers: segments sorted by
/// file offset, non-overlapping, covering every tensor. Computed once at load from the plan,
/// the configured context and the backend's budget; a backend implements each tier its own
/// way and must not second-guess the split.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WeightPlacement {
    pub segments: Vec<WeightSegment>,
    pub budget: TierBudget,
    pub fast_layers: u32,
    pub total_layers: u32,
}

impl WeightPlacement {
    #[must_use]
    pub fn bytes_in(&self, pick: impl Fn(WeightTier) -> bool) -> u64 {
        self.segments
            .iter()
            .filter(|s| pick(s.tier))
            .map(|s| s.bytes)
            .sum()
    }
    /// Every segment outside the fast tier as a span: the shape the CUDA backend's existing
    /// residency entry consumes (tables and slow-tier layers alike go through its bounded
    /// staging cache until it implements the ring).
    #[must_use]
    pub fn slow_spans(&self) -> Vec<StreamedWeightSpan> {
        self.segments
            .iter()
            .filter(|s| s.tier != WeightTier::Fast)
            .map(|s| StreamedWeightSpan {
                offset: s.offset,
                bytes: s.bytes,
            })
            .collect()
    }
}

/// One immutable quantized matrix that a backend may transform into a persistent,
/// architecture-owned execution layout during model admission. The common engine
/// supplies only wire facts; unsupported backends return `Ok(false)` and retain their
/// established weight representation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantizedWeightPrepack {
    pub offset: u64,
    pub n_in: u32,
    pub n_out: u32,
    pub kind: WeightKindWire,
    pub reserved: u32,
}

/// How a router's scores become the weights a token's picked experts carry.
///
/// The file names it (`<arch>.expert_gating_func`), and the two the models here declare are
/// the two spelled out; llama.cpp's enum has more, and one of those arriving must be READ
/// and added rather than folded into the nearer of these.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpertGating {
    /// Scores are a distribution over ALL experts before the pick (value 1 in the file).
    Softmax,
    /// Each score is squashed independently, so the picked weights do not sum to 1 and
    /// the renormalisation step is what makes them a weighting (value 2 in the file).
    Sigmoid,
}

/// Process-wide backend handle.
///
/// The composition root publishes one static handle and the server may move a
/// serialized engine containing that handle between connection threads. Backend
/// implementations must therefore make shared access safe; per-request mutation
/// remains serialized by the engine owner.
pub trait Backend: Sync {
    // --- session ---
    /// Static target-only Q8 numerical admission, called once before any forward.
    /// Unsupported backends return false. This is not a dynamic plugin ABI.
    fn prepare_batch_invariant_q8_v1(&self) -> Result<bool, i32> {
        Ok(false)
    }
    fn begin(&self);
    /// Begins a forward pass while allowing the backend to select decode-specific
    /// lifecycle machinery. The default deliberately preserves the established
    /// `begin` behavior for backends without a distinct decode path.
    fn begin_forward(&self, _decode: bool) {
        self.begin();
    }
    /// Static laboratory declaration of the complete work in the next device batch.
    /// Called between closed forwards before graph preparation. A supporting backend
    /// binds capture identity and atomic FFN completion to this demand; `false`
    /// means this experiment is unsupported. This is not a dynamic plugin ABI.
    fn set_forward_demand_lab(
        &self,
        _ffn_layers: u32,
        _logits_wanted: bool,
    ) -> Result<bool, i32> {
        Ok(false)
    }
    /// Eager static experiment: projection numerical reference, independent of
    /// truthful active batch/attention/KV geometry; reset at each begin/end.
    fn set_projection_reference_lab(
        &self,
        _start: u32,
        _tokens: u32,
    ) -> Result<bool, i32> {
        Ok(false)
    }
    /// Prepares one single-token decode. `Ok(true)` means the backend has already
    /// submitted a reusable execution graph, so the semantic operations for this
    /// forward must not be encoded again. Backends without replay support retain
    /// the normal encode path through this conservative default.
    fn decode_prepare(
        &self,
        _token: u32,
        _start_pos: u32,
        _argmax: bool,
    ) -> Result<bool, i32> {
        Ok(false)
    }
    /// Prepares an exact-shape multi-token prefill. `Ok(true)` means the backend
    /// has already submitted a reusable execution graph for the supplied token
    /// payload and the semantic operations must not be encoded again. The
    /// conservative default keeps every non-CUDA backend on its existing path.
    fn prefill_prepare(
        &self,
        _tokens: &[u32],
        _start_pos: u32,
        _argmax: bool,
    ) -> Result<bool, i32> {
        Ok(false)
    }
    /// Marks the boundary after request-dependent token embedding and any
    /// per-layer-input embedding have been materialized. `Ok(true)` means the
    /// backend submitted a reusable graph for the remaining model body. The
    /// default keeps Metal and every backend without split-Prefill replay on the
    /// established workflow.
    fn prefill_body_prepare(
        &self,
        _tokens: &[u32],
        _start_pos: u32,
        _argmax: bool,
    ) -> Result<bool, i32> {
        Ok(false)
    }
    /// Select the projection weight representation before graph preparation.
    /// This lifecycle hint does not change numerical batch geometry: verification
    /// may deliberately retain a Prefill-shaped batch. Other backends need no work.
    fn prepare_projection_phase(&self, _decode_or_verify: bool) -> Result<(), i32> {
        Ok(())
    }
    /// Optionally capture the already-materialized verification body until end().
    /// Capture encodes current operations, while Replay has already submitted the
    /// body. Both defer external feature publication until after end().
    fn verification_body_begin(
        &self,
        _rows: u32,
        _start: u32,
    ) -> Result<VerificationSubmission, i32> {
        Ok(VerificationSubmission::Eager)
    }
    /// Sets the immutable absolute geometry for the next device batch. Backends that
    /// do not use position-dependent numerical routes deliberately inherit this no-op.
    fn set_batch_geometry(&self, _geometry: BatchGeometry) -> Result<(), i32> {
        Ok(())
    }
    /// Commit what is encoded and return WITHOUT waiting, so the GPU runs it while the
    /// CPU encodes what follows. A read after this sees whatever was there before.
    fn flush(&self);
    /// Commit and WAIT. The only call after which a `read` is meaningful.
    ///
    /// A read in the middle of a forward is `end`, the read, then `begin` before anything else is
    /// encoded. On Metal `end` leaves no encoder, so a dispatch encoded before the next `begin` is
    /// dropped with no error.
    ///
    /// # Errors
    /// Returns the command buffer's error code.
    fn end(&self) -> Result<(), i32>;
    /// Commit WITHOUT waiting and keep the region outstanding, so the next region can be
    /// encoded and committed behind it while this one runs (pipelined decode:
    /// docs/decode-turnaround.md). `wait_outstanding` retires the oldest such region;
    /// a `read` of what it wrote is meaningful only after that. A backend without the
    /// capability (`decode_pipelining` false) ends synchronously here.
    ///
    /// # Errors
    /// Returns the command buffer's error code.
    fn end_async(&self) -> Result<(), i32> {
        self.end()
    }
    /// Wait for the OLDEST region committed by `end_async` and retire it. A no-op when
    /// none is outstanding.
    ///
    /// # Errors
    /// Returns that region's command buffer error code.
    fn wait_outstanding(&self) -> Result<(), i32> {
        Ok(())
    }
    /// After a region failed (a mega-kernel barrier timeout) and the engine has retired
    /// every outstanding region and rolled the failed step's state back: make the backend
    /// ready to run the step again. The Metal backend resets its kernel's sync words and
    /// keeps the persistent route closed for a backoff of regions, so the re-run takes the
    /// dispatch path. A backend without such a route has nothing to do.
    ///
    /// # Errors
    /// The backend's own code when it cannot recover (a region still outstanding).
    /// Reserves the persistent kernel's scratch once at load, so no region ever regrows it
    /// (Metal: the sync buffer that carries the sticky error): the model's widest FFN row
    /// and the deep attention body's partials for `attn_heads` query heads at the widest
    /// head `attn_hd`.
    fn mega_reserve(
        &self,
        _n_mid: u32,
        _attn_heads: u32,
        _attn_hd: u32,
    ) -> Result<(), i32> {
        Ok(())
    }
    fn mega_recover(&self) -> Result<(), i32> {
        Ok(())
    }
    /// Negotiate the recurrent state planes requested by the model plan. This is a
    /// static execution capability: allocation, placement and step cursors must use
    /// the same count, independently of runtime tuning choices. Backends that rotate
    /// planes must bind each step's state addresses when replaying cached work.
    fn recurrent_plane_count(&self, requested: u32) -> u32 {
        requested.max(1)
    }
    /// Whether `end_async` / `wait_outstanding` overlap regions on this backend, and
    /// `argmax_feed` writes the pick where the next step's gather reads it.
    fn decode_pipelining(&self) -> bool {
        false
    }
    /// Whether this backend stages any row-gathered table on the host per token
    /// (`stage_rows` will answer true for it). A staged table needs the token id on the
    /// host at encode time, which pipelined decode does not have.
    fn stages_rows(&self) -> bool {
        false
    }
    /// Enter or leave an isolated tuning session. Backends with reusable execution
    /// caches must prevent capture/replay while enabled and invalidate any cached
    /// executable when the mode changes. The conservative default is a no-op.
    fn set_tuner_mode(&self, _enabled: bool) -> Result<(), i32> {
        Ok(())
    }
    /// Whether tuning this backend must fail closed when device-event timing is absent.
    /// CPU and legacy backends may use wall time; CUDA overrides this because sub-ms
    /// launch ranking is otherwise dominated by host scheduling jitter.
    fn tuner_requires_device_timing(&self) -> bool {
        false
    }
    /// Validate that all backend-specific discovery evidence needed to write a tuning
    /// result was established. CUDA fails closed here; legacy backends may accept the
    /// explicit zero sentinels they already used.
    fn validate_tuner_profile(&self, _profile: &DeviceProfile) -> Result<(), String> {
        Ok(())
    }
    /// Reset proof for one measured region. CUDA records the choice epoch observed by
    /// the actual kernel entry, so changing a knob without reaching its workload cannot
    /// be mistaken for a measured candidate.
    fn reset_tuner_dispatch_proof(&self) {}
    /// Declare which named choices the next measured workload must actually consult.
    /// Backends without dispatch instrumentation keep the conservative no-op default;
    /// CUDA maps names to stable native slots and fails closed on an unknown name.
    fn set_tuner_dispatch_expectation(&self, _knobs: &[&str]) -> Result<(), String> {
        Ok(())
    }
    fn validate_tuner_dispatch_proof(&self) -> Result<(), String> {
        Ok(())
    }
    /// GPU microseconds of the last `begin`/`end` region, or 0.0 if this backend cannot
    /// report it. Timing a candidate with a wall clock also charges submission latency;
    /// where the device can report its own busy time, the tuner uses that instead.
    fn last_gpu_us(&self) -> f64 {
        0.0
    }
    /// Backend-owned proof that every component of a coupled tuner workload reached
    /// its intended implementation in the last completed submission. Zero is the
    /// conservative default; it prevents a shared tuner from mistaking equal timings
    /// or a silent fallback for candidate execution.
    fn tuner_route_evidence(&self, _workload: Workload) -> u32 {
        0
    }
    /// Queried and measured device limits, for the derivations. Default is all zero --
    /// "nothing established" -- so a backend that has not implemented it cannot have a
    /// derivation silently compute against a made-up number.
    fn device_profile(&self) -> DeviceProfile {
        DeviceProfile::default()
    }
    /// MEASURE how many threadgroups of `architecture`'s mega layer pipeline at head-dim
    /// `slot` (0 = the smaller attention geometry, 1 = the larger) this GPU holds resident
    /// at the seated width -- the LIMIT the grid knob for that pipeline is ranked under
    /// (task #203). The value is remembered by the backend for `candidates` / `legal`
    /// to read; 0 = no such pipeline, or the probe could not run.
    ///
    /// A DISCOVERY probe, called by the tuner between regions like `spill_rate`: it
    /// dispatches the real pipeline in probe mode and clears the grid-barrier counters,
    /// so it must never run with a region open. The engine never calls it -- an untuned
    /// host runs one threadgroup per core, a tuned one applies the stored value as
    /// written, and the tuner only ever writes a value it measured under this limit.
    fn mega_admission(&self, _architecture: &str, _slot: u32) -> u32 {
        0
    }
    /// Which form `architecture`'s mega entries run in on this backend, or `None` when it
    /// has no pipeline family for them. The model side (whether the workflow offers
    /// entries at all) is the tuner's to combine with this; see [`MegaSeat`].
    fn mega_seat_form(&self, _architecture: &str) -> MegaSeat {
        MegaSeat::None
    }
    /// TFLOPS holding the `idx`-th candidate accumulator count live. Sweeping it finds
    /// the spill cliff. 0.0 where a backend has no such probe.
    fn spill_rate(&self, _idx: u32, _tgs: u32, _tpg: u32, _iters: u32) -> f64 {
        0.0
    }
    /// Streaming read GB/s at a given working set. Sweeping it finds the cache knee.
    fn bw_read(&self, _bytes: u64, _reps: u32, _tgs: u32, _tpg: u32) -> f64 {
        0.0
    }
    /// TFLOPS with the prefill attention score loop's own operand mix. This is what the
    /// score phase's measured rate should be compared against. 0.0 where a backend has no
    /// such probe.
    fn scoremix_rate(
        &self,
        _tgs: u32,
        _sgs: u32,
        _iters: u32,
        _stride: u32,
        _kspan: u32,
    ) -> f64 {
        0.0
    }
    /// Microseconds one extra command-buffer boundary costs the DEVICE: the idle gap
    /// between consecutive back-to-back buffers, which is what a mid-graph flush exposes
    /// when the host runs ahead (it does at decode). 0.0 where a backend has no such probe.
    fn commit_overhead(&self, _n: u32) -> f64 {
        0.0
    }
    /// Microseconds per submit AND WAIT -- a different, much larger quantity.
    fn sync_overhead(&self, _n: u32) -> f64 {
        0.0
    }
    /// The cache types this backend is ACTUALLY configured with, as a short tag.
    ///
    /// A self-check, not a convenience: the tuner keys its stored config by the cache type
    /// it believes it is tuning for, and that belief comes from a flag. If the flag did not
    /// reach the backend, every per-cache-type answer it records is mislabelled. Asking the
    /// backend what it actually holds is the only way to catch that.
    fn kv_tag(&self) -> String {
        "f16".to_string()
    }
    /// Correctness-defining rotation route for quantized KV storage.
    ///
    /// The default is the existing shared workflow route, which keeps Metal and the
    /// host backend byte-for-byte unchanged. A backend override must be covered by its
    /// own correctness receipt before it is admitted as a tuned route.
    fn kv_quantization_route(&self) -> KvQuantizationRoute {
        KvQuantizationRoute::default()
    }
    /// Optional backend-owned rotation for workflows whose established cache basis is
    /// canonical. `None` preserves that workflow byte-for-byte; an accelerator may opt
    /// in only when its route has independent numerical evidence.
    fn kv_quantization_route_override(&self) -> Option<KvQuantizationRoute> {
        None
    }
    /// Durable byte codecs implemented by this backend's KV store kernels.
    fn kv_byte_codec_route(&self) -> KvByteCodecRoute {
        KvByteCodecRoute::default()
    }
    /// Microseconds the host spends ENCODING one dispatch, with no execution in it.
    fn encode_cost(&self, _n: u32) -> f64 {
        0.0
    }
    /// Command-buffer length policy: layers per flush at decode / prefill.
    fn flush_layers(&self, decode: bool) -> u32;
    /// GPU seconds of the LONGEST single command buffer of the last region, or 0.0 from a
    /// backend that does not measure -- which leaves every policy built on it inert.
    ///
    /// The longest ONE, not the region's sum: what starves the rest of the machine is a
    /// single buffer holding the GPU with no gap in it.
    fn longest_cb_gpu_seconds(&self) -> f64 {
        0.0
    }

    /// Rows retained after the last state-writing operator when all remaining
    /// work is row-local and only the final logit row is observable. Backends
    /// may make this a measured per-request decision from the original batch.
    fn row_local_prefill_tail_rows(&self, _n_tok: u32) -> u32 {
        64
    }

    /// Whether a workflow may omit Q when its only consumer, attention, is dead
    /// after the final required state write. K/V preparation remains mandatory.
    /// Backends retain their existing schedule until they admit this capability.
    /// Admit compact PLE layer-prefix production only with a matching strided
    /// gather implementation. The workflow proves which layer slices are dead.
    /// Update the currently materialized Prefill suffix without rewriting the
    /// reference batch used for numerical route selection or graph identity.
    fn set_materialized_prefill_tail(
        &self,
        _start: u32,
        _tokens: u32,
    ) -> Result<(), i32> {
        Ok(())
    }

    fn prefill_ple_prefix_elision(&self, _n_tok: u32) -> bool {
        false
    }

    fn prefill_unused_query_elision(&self) -> bool {
        false
    }

    /// Optional row count for the final shared-KV tail. The model must prove that
    /// all KV writes are complete and that discarded query rows have no consumer.
    /// This is a separate numerical route from the established aligned schedule.
    fn shared_kv_prefill_tail_rows(&self) -> Option<u32> {
        None
    }

    /// Minimum retained rows for a proven finite-history suffix. The model owns
    /// history/checkpoint demand; the backend owns numerical shape admission.
    fn finite_history_prefill_tail_rows(&self) -> Option<u32> {
        None
    }

    // --- buffers and arena ---
    fn alloc(&self, id: BufId, bytes: u64) -> Result<(), i32>;
    fn arena(&self, bytes: u64) -> Result<(), i32>;
    fn place(&self, id: BufId, offset: u64, bytes: u64) -> Result<(), i32>;
    fn page_round(&self, n: u64) -> u64;
    fn alloc_kv(&self, bytes: &[u64]) -> Result<(), i32>;
    fn grow_kv(&self, bytes: &[u64]) -> Result<(), i32>;
    /// Allocate KV with explicit logical geometry for backends that own page tables.
    ///
    /// The default preserves existing CPU/Metal behavior. CUDA overrides this method so
    /// arena replacement and page-table allocation are one transaction.
    fn alloc_kv_layout(&self, bytes: &[u64], _layouts: &[KvLayout]) -> Result<(), i32> {
        self.alloc_kv(bytes)
    }
    /// Grow KV while preserving both bytes and any installed page-table prefix.
    /// The default keeps non-paged backends byte-for-byte unchanged.
    fn grow_kv_layout(&self, bytes: &[u64], _layouts: &[KvLayout]) -> Result<(), i32> {
        self.grow_kv(bytes)
    }
    /// Whether KV supports independent per-layer commits, with zero bytes meaning
    /// "leave this layer alone", without losing its paging layout. Opt in only when
    /// prepared commits and unchanged-layer semantics are implemented. Other backends
    /// keep full-capacity pool storage and the existing layout-aware growth path.
    fn supports_kv_incremental_commit(&self) -> bool {
        false
    }
    /// Allocate KV that can grow without moving a row (docs/memory-tiers-and-fit.md section
    /// 12): `reserve[i]` bytes of room per layer side, `bytes[i]` of it committed now.
    /// The default allocates `bytes` and keeps growing by copy, as before.
    fn alloc_kv_reserved(
        &self,
        bytes: &[u64],
        reserve: &[u64],
        layouts: &[KvLayout],
    ) -> Result<(), i32> {
        let _ = reserve;
        self.alloc_kv_layout(bytes, layouts)
    }
    /// Get a commit of `bytes` per layer ready off the step's path, so the forward that
    /// needs it does not wait (0 leaves a layer alone). Smaller than now is a release,
    /// adopted once it still covers what the next forward needs. Default: nothing -- the next
    /// grow does the work in place.
    fn kv_prefetch(&self, _bytes: &[u64]) {}
    /// Shrink each layer's KV storage to `bytes` (0 leaves a layer alone), with nothing
    /// in flight that addresses the part given up. The storage above goes back to the
    /// system. Default: nothing -- a backend whose storage is sized once keeps it.
    ///
    /// # Errors
    /// A backend code when the smaller storage cannot be installed.
    fn kv_release(&self, _bytes: &[u64]) -> Result<(), i32> {
        Ok(())
    }
    /// Whether the system has reported memory pressure since the last call (the report is
    /// taken). The KV tier gives idle conversations back when it has. Default: never.
    fn take_memory_pressure(&self) -> bool {
        false
    }
    /// How long the device's residency is held after the last work before it is released,
    /// None when it is held for good. The KV tier releases idle conversations on the same
    /// clock, so a server that has gone quiet gives back its KV and its wiring together.
    fn idle_release_after(&self) -> Option<std::time::Duration> {
        None
    }
    /// Take a prepared commit if it covers `need` per layer (0 = no need for that layer): a
    /// forward entry's chance to adopt a prepared release. Default: nothing.
    fn kv_adopt(&self, _need: &[u64]) {}
    /// Bytes the KV cache holds committed now, K and V together; 0 when the backend does not
    /// track it.
    fn kv_committed_bytes(&self) -> u64 {
        0
    }
    /// True when committed KV memory follows the storage in use rather than the storage
    /// reserved: a KV tier larger than one context then costs address space until blocks are
    /// written, and the pool may size its device tier from the fit. False (the default): a
    /// reservation is an allocation, and the pool's device tier stays one context.
    fn kv_commits_on_demand(&self) -> bool {
        false
    }
    /// The largest KV storage one layer side can have, in bytes; None when the backend sets
    /// no limit of its own. The fit keeps every reservation under it.
    fn kv_max_view_bytes(&self) -> Option<u64> {
        None
    }
    /// Load is over: make every weight the device will read ready to be read.
    ///
    /// A LOAD THAT RETURNS WITH WORK STILL OWED IS CLAIMING READY WHEN IT IS NOT. On Metal
    /// this is the residency set's page wiring: measured 3.7-8.7 s for Qwen3.8-27B's
    /// 15509 MiB, and until this hook existed every millisecond of it landed inside the
    /// first request. Backends with nothing to wire inherit the no-op.
    ///
    /// `stall_budget_s` is how long ONE uninterruptible step may hold the host. A backend
    /// that wires in pieces stops when a piece exceeds it and leaves the rest pageable: a
    /// tier that does not fit must DEGRADE, never freeze. Measured on Metal, one call over
    /// a 15509 MiB tier: 26.8 ms with the memory free, 122177.6 ms with it not -- two
    /// minutes in which nothing else on the machine can run.
    fn wire_weights(&self, stall_budget_s: f64) {
        let _ = stall_budget_s;
    }
    /// The file the weight mapping came from, so a backend that CONVERTS weights can read
    /// them without the page cache.
    ///
    /// A converted tensor is read once and its source is dead the moment the twin exists, so
    /// caching it means the file's bytes sit in the page cache beside the twin that replaced
    /// them -- measured at +9588 MiB of file-backed pages across a Qwen3.8-27B load. A
    /// tensor used DIRECTLY has no such twin: the mapping is the resident copy, one copy, and
    /// should stay mapped. Backends that convert nothing inherit the no-op.
    fn set_weight_path(&self, _path: &std::path::Path) {}
    /// The model's size in bytes, which keys measured configuration: the end of its tensor
    /// data. A paired drafter's mapping runs past it. Called before init; default: unused.
    fn set_model_bytes(&self, _bytes: u64) {}
    /// The prefill chunk this host's tuned config stores for the model keyed by
    /// `model_bytes`, read WITHOUT applying anything, or `None` for no config. Asked before
    /// placement, because the KV and activation reserves are sized from the chunk.
    fn stored_prefill_batch(&self, _model_bytes: u64) -> Option<usize> {
        None
    }
    fn write(&self, id: BufId, off: u64, src: &[f32]);
    /// Set `elems` floats of a device buffer to zero, starting at `off`.
    ///
    /// SEPARATE FROM `write` BECAUSE THE SOURCE DOES NOT EXIST. Writing zeros through
    /// `write` means the caller builds them: a recurrent state cleared that way allocated
    /// a host `Vec` the size of the whole state times its plane count, faulted it in, and
    /// memcpy'd it across -- 3374 ms on Qwen3.8-27B's first conversation and 65 ms on
    /// every one after, for a buffer a backend can fill in place.
    ///
    /// The default keeps the semantics with a bounded staging buffer, so a backend that
    /// has no in-place fill still never allocates more than a megabyte. A backend whose
    /// memory the host can address overrides it with a `memset`.
    fn zero(&self, id: BufId, off: u64, elems: u64) {
        const CHUNK: usize = 256 * 1024; // 1 MiB of f32
        let zeros = vec![0.0_f32; CHUNK.min(elems as usize)];
        let mut done = 0u64;
        while done < elems {
            let n = CHUNK.min((elems - done) as usize);
            self.write(id, off + done, &zeros[..n]);
            done += n as u64;
        }
    }
    fn write_u32(&self, id: BufId, off: u64, src: &[u32]);
    fn read(&self, id: BufId, off: u64, dst: &mut [f32]);
    fn read_kv_bytes(&self, layer: u32, is_v: bool, off: u64, dst: &mut [u8]);
    /// The KV pool's restore path: raw bytes back into a layer's cache (device idle).
    fn write_kv_bytes(&self, layer: u32, is_v: bool, off: u64, src: &[u8]);
    /// Lends the host bytes of `ranges` of the KV cache to `with`, one slice per range in
    /// order, so a restore reads a file straight into the cache. On storage the GPU shares
    /// with the host, that read is the whole transfer. Device idle, as for
    /// `write_kv_bytes`; the ranges must not overlap.
    ///
    /// `None` when this backend's KV storage is not host-addressable: the caller reads
    /// into buffers of its own and writes them through `write_kv_bytes`.
    fn lend_kv_ranges(
        &self,
        _ranges: &[KvRange],
        _with: &mut dyn FnMut(&mut [&mut [u8]]) -> Result<(), String>,
    ) -> Option<Result<(), String>> {
        None
    }
    /// The pool's placement decision for one layer: entry i maps positions
    /// [i*64, i*64+64) to physical block entries[i]. Capability-gated on
    /// `pool_caps().paged_reads` growing true per backend.
    fn set_kv_page_table(&self, layer: u32, entries: &[u32]);
    /// Which REGION of a windowed layer's cache the resident conversation owns, as a byte
    /// offset into that layer's K and V caches.
    ///
    /// A windowed layer holds one ring per conversation that may be live at once. Pointing
    /// the layer at a conversation's own ring keeps every kernel addressing 0..ring-1, so
    /// the ring rule, the wrap test and the quantized dequant scratch are untouched by
    /// regions -- and a conversation's window stops being clobbered by whoever ran last.
    /// Pooled (full-attention) layers place through the block table and stay at 0.
    /// Default: no-op, for a backend that has not grown regions yet.
    fn set_kv_region(&self, _layer: u32, _k_off: u64, _v_off: u64) {}
    /// Tell the OS the pages backing a freed KV range are discardable
    /// (MADV_FREE-style). The pool calls this when residency is dropped so the
    /// footprint actually falls; content is gone, the buffer stays valid.
    /// Default: no-op (a backend without page-level release just keeps the pages).
    fn kv_advise_free(&self, _layer: u32, _is_v: bool, _off: u64, _len: u64) {}
    /// Recommit a previously advise-freed range (call before the block is
    /// written again). Default: no-op.
    fn kv_advise_reuse(&self, _layer: u32, _is_v: bool, _off: u64, _len: u64) {}

    /// Allocate persistent host-resident KV storage owned by the backend.
    ///
    /// A discrete backend uses page-locked memory here. The opaque handle keeps raw
    /// host pointers out of the common pool and lets the backend reject stale or
    /// double-freed allocations. Shared-address backends deliberately keep the
    /// unsupported default: manufacturing a Host copy there would be pure overhead.
    fn kv_host_alloc(&self, _bytes: u64) -> Result<KvHostHandle, i32> {
        Err(KV_HOST_UNSUPPORTED)
    }
    /// Release one persistent host allocation. Implementations must reject stale
    /// handles and double free.
    fn kv_host_free(&self, _handle: KvHostHandle) -> Result<(), i32> {
        Err(KV_HOST_UNSUPPORTED)
    }
    /// Copy a complete set of physical KV spans from device storage into one
    /// persistent host allocation. The batch is one transaction: on error neither
    /// allocator ownership nor the caller-visible host contents may change.
    fn kv_demote(&self, _spans: &[KvTransferSpan]) -> Result<(), i32> {
        Err(KV_HOST_UNSUPPORTED)
    }
    /// Copy a complete set of spans from persistent host storage into device KV.
    /// The caller owns already-recommitted destination placements; on error it rolls
    /// those placements back while the Host copy remains valid for retry.
    fn kv_promote(&self, _spans: &[KvTransferSpan]) -> Result<(), i32> {
        Err(KV_HOST_UNSUPPORTED)
    }
    /// Copy bytes between a persistent backend-owned Host allocation and ordinary
    /// CPU memory. These are for canonical disk interchange and diagnostics, never
    /// the per-token decode path.
    fn kv_host_read(
        &self,
        _handle: KvHostHandle,
        _off: u64,
        _dst: &mut [u8],
    ) -> Result<(), i32> {
        Err(KV_HOST_UNSUPPORTED)
    }
    fn kv_host_write(
        &self,
        _handle: KvHostHandle,
        _off: u64,
        _src: &[u8],
    ) -> Result<(), i32> {
        Err(KV_HOST_UNSUPPORTED)
    }
    /// Bytes currently retained in backend-owned persistent Host allocations.
    fn kv_host_allocated_bytes(&self) -> u64 {
        0
    }
    /// Once-per-machine facts used by the common pool's Host-tier auto-fit.
    /// Unknown or failed measurements return `None`; callers must not invent
    /// bandwidth or headroom in their place.
    fn kv_host_profile(&self) -> Option<HostTierProfile> {
        None
    }

    // --- compute ---
    /// The fingerprint and model bytes this backend's stored config was keyed by, or `None`
    /// when it loaded none.
    ///
    /// A caller that wants to store its OWN measurements beside the tuner's -- keyed by the
    /// same host, backend, search space, cache type and model -- asks here instead of
    /// rebuilding the key and drifting from it.
    fn config_key(&self) -> Option<(String, u64)> {
        None
    }

    fn matmat(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        n_tok: u32,
    );
    fn matmat_from(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        n_in: u32,
        n_out: u32,
        src: BufId,
        dst: BufId,
        n_tok: u32,
        src_row: u32,
    );
    /// Try the state-selected gated projection
    /// `dst = activation(gate * src) * (up * src)` as one backend operation.
    ///
    /// Returning `false` promises no writes and asks the workflow to issue the two
    /// projections plus `act_mul`. The activation is the process-wide value selected by
    /// `set_activation`; a backend must reject unsupported activation or layout pairs.
    #[allow(clippy::too_many_arguments)]
    fn matmat_gated(
        &self,
        _gate_kind: WeightKindWire,
        _gate_off: u64,
        _up_kind: WeightKindWire,
        _up_off: u64,
        _n_in: u32,
        _n_out: u32,
        _src: BufId,
        _dst: BufId,
        _tmp: BufId,
        _n_tok: u32,
    ) -> bool {
        false
    }
    /// Observe an FFN boundary without submitting work or changing its semantics.
    /// Default backends do nothing; an experiment may record capture dependencies.
    fn decode_ffn_boundary(
        &self,
        _phase: DecodeFfnPhase,
        _entering: bool,
    ) -> Result<(), i32> {
        Ok(())
    }
    /// Try the complete gated FFN projection
    /// `dst = down * (activation(gate * src) * (up * src))` as one backend
    /// operation.
    ///
    /// Returning `true` promises that the final `dst` has been written completely.
    /// The backend may use `gated_tmp` as scratch, but callers must not assume that
    /// it contains a valid gated intermediate after a successful call. Returning
    /// `false` promises that no public buffer has been written and asks the workflow
    /// to execute its established gated-projection and down-projection sequence.
    #[allow(clippy::too_many_arguments)]
    fn ffn_gated_down(
        &self,
        _gate_kind: WeightKindWire,
        _gate_off: u64,
        _up_kind: WeightKindWire,
        _up_off: u64,
        _down_kind: WeightKindWire,
        _down_off: u64,
        _n_in: u32,
        _n_mid: u32,
        _n_out: u32,
        _src: BufId,
        _gated_tmp: BufId,
        _dst: BufId,
        _n_tok: u32,
    ) -> bool {
        false
    }
    /// Whether `mega_layer` should be offered a gemma4 layer's FRONT as well (o_proj and
    /// the sandwich norm), i.e. called before the o_proj projection with `front = Some(..)`.
    fn mega_front_wanted(&self) -> bool {
        false
    }
    /// Whether the decode attention step should be offered to the mega entry as well
    /// (`MegaFront::attention = Some(..)`, called before `attention`).
    fn mega_attn_wanted(&self) -> bool {
        false
    }
    /// Run one whole decode layer as ONE backend operation (see [`MegaEntry`]). Returning
    /// `true` promises the layer's outputs are written -- for gemma4 `x` and, with a next
    /// norm, `next_out`; for LFM2 `x` and its mixer state advanced -- and that the scratch
    /// buffers the variant names are the only others touched. Returning `false` promises no
    /// public buffer has been written and asks for the established dispatch sequence.
    fn mega_layer(&self, _entry: &MegaEntry<'_>) -> bool {
        false
    }
    /// The mega program (task #153): the model's layer loop has ended; a backend that records
    /// the layers into one dispatch per token encodes the pending run now. Default: nothing.
    fn mega_program_end(&self) {}
    /// Whether the q/k/v front should be offered to the mega block as well
    /// (`MegaFront::qkv = Some(..)`, called before the layer's input norm).
    fn mega_qkv_wanted(&self) -> bool {
        false
    }
    /// Two independent equal-shape projections from one activation. Quantized
    /// backends may share activation conversion and launch setup; the default is the
    /// exact two-matmul sequence in call order.
    #[allow(clippy::too_many_arguments)]
    fn matmat_pair(
        &self,
        first_kind: WeightKindWire,
        first_off: u64,
        first_dst: BufId,
        second_kind: WeightKindWire,
        second_off: u64,
        second_dst: BufId,
        n_in: u32,
        n_out: u32,
        src: BufId,
        n_tok: u32,
    ) {
        self.matmat(first_kind, first_off, n_in, n_out, src, first_dst, n_tok);
        self.matmat(second_kind, second_off, n_in, n_out, src, second_dst, n_tok);
    }
    /// Per-layer embedding projection. The default preserves the established
    /// projection, process activation, layer-vector multiply, projection sequence.
    /// Backends may fuse its intermediates without exposing scheduling to the model.
    #[allow(clippy::too_many_arguments)]
    fn ple_project(
        &self,
        gate_kind: WeightKindWire,
        gate_off: u64,
        proj_kind: WeightKindWire,
        proj_off: u64,
        n_embd: u32,
        ple_width: u32,
        src: BufId,
        gate: BufId,
        per_layer: BufId,
        per_layer_off: u32,
        per_layer_stride: u32,
        back: BufId,
        n_tok: u32,
    ) {
        self.matmat(gate_kind, gate_off, n_embd, ple_width, src, gate, n_tok);
        self.act(gate, n_tok * ple_width);
        self.mul_strided(
            gate,
            per_layer,
            ple_width,
            per_layer_off,
            per_layer_stride,
            ple_width,
            n_tok,
        );
        self.matmat(proj_kind, proj_off, ple_width, n_embd, gate, back, n_tok);
    }
    /// Which gated activation the matmul kernels fuse into their write-back, or None.
    ///
    /// Sticky: it applies to every `matmat`/`matvec` until set again, so a caller sets it,
    /// dispatches, and sets it back.
    ///
    /// Workflows must consult `supports_epilogue` before selecting a fused route. This
    /// separate capability prevents an unsupported nonzero wire value from being
    /// mistaken for whichever activation a native backend happened to implement first.
    fn set_epilogue(&self, epi: Epilogue);
    /// Whether `set_epilogue(epi)` followed by a projection is numerically implemented.
    /// `None` is mandatory; activation-specific fusion is opt-in.
    fn supports_epilogue(&self, epi: Epilogue) -> bool {
        epi == Epilogue::None
    }

    /// Try an input projection followed by gated short convolution and state advance.
    /// A successful call writes `bcx`, `out`, and the advanced recurrent `state`.
    /// Returning `false` promises no public writes; the caller retains its ordinary
    /// projection, optional snapshot, and short-convolution sequence. Callers needing
    /// a snapshot of the pre-advance history must use that established sequence.
    #[allow(clippy::too_many_arguments)]
    fn matmat_shortconv(
        &self,
        _wkind: WeightKindWire,
        _proj_off: u64,
        _conv_off: u64,
        _src: BufId,
        _bcx: BufId,
        _state: BufId,
        _state_off: u32,
        _out: BufId,
        _width: u32,
        _kernel: u32,
        _n_tok: u32,
    ) -> bool {
        false
    }

    /// A causal depthwise convolution over `n_tok` tokens of `width` channels.
    ///
    /// `src` is the input projection, token-major with `form.src_stride(width)` elements
    /// per token. `conv_w` is at `w_off`, channel-major with the tap fastest. `state`
    /// holds `kernel - 1` past values per channel, oldest first, at element offset
    /// `state_off`, and the advanced history is written at `state_out_off`.
    ///
    /// `form` says what the taps run over and what the epilogue does; see [`ConvForm`].
    /// LFM2's gated short convolution and Qwen3.8's delta-net convolution differ in
    /// exactly those two places and in nothing else, which is why they are one op.
    ///
    /// The state advance is a second dispatch, not part of this one: the new state is the
    /// tail of the same sequence the outputs read, so a single dispatch would have
    /// threads overwriting slots other threads still need, and a kernel cannot barrier
    /// its whole grid.
    ///
    /// `state_out_off` is the plane the advanced history is written to; equal to
    /// `state_off` is the in-place form. The shift rewrites every value it read, so
    /// aiming it at another plane costs an address, not traffic.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv(
        &self,
        form: ConvForm,
        src: BufId,
        w_off: u64,
        state: BufId,
        state_off: u32,
        state_out_off: u32,
        out: BufId,
        width: u32,
        kernel: u32,
        n_tok: u32,
    );
    /// The convolution state as of `n_tok` tokens into this batch, written to `snap` and
    /// leaving `state` alone.
    ///
    /// This is what lets a batch run past a checkpoint boundary instead of being cut to
    /// end on one. The state after `n_tok` tokens is just the last `kernel - 1` values of
    /// the form's value sequence ending there, and `src` already holds them, so the
    /// mid-batch state is COMPUTED rather than stood on.
    ///
    /// Must be dispatched BEFORE `causal_conv` advances `state`: with `n_tok` shorter
    /// than the history, part of the answer is the pre-batch state.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv_snapshot(
        &self,
        form: ConvForm,
        src: BufId,
        state: BufId,
        state_off: u32,
        snap: BufId,
        snap_off: u32,
        width: u32,
        kernel: u32,
        n_tok: u32,
    );
    /// Whether this backend serves a batch described by `BufId::RowLayout` at this head
    /// dim through `attention_rows`, `head_norm_rope_rows`, `causal_conv_rows` and
    /// `causal_conv_row_inputs`. Each entry still returns false for a call it cannot serve.
    fn supports_row_layout(&self, _head_dim: u32) -> bool {
        false
    }
    /// `attention` in which row t sees the cache in `key_lo..start_pos` and the batch rows its
    /// layout names, instead of rows 0..=t. No window, no ring. `key_lo` is 0 for every caller
    /// but a drafter whose cache holds this request's rows only from a restore point up; it may
    /// not exceed `start_pos`. `float_q`: Q enters the scores in float where the backend
    /// otherwise rounds it to half (an option; every caller leaves it off by default). False:
    /// nothing was encoded.
    #[allow(clippy::too_many_arguments)]
    fn attention_rows(
        &self,
        _kv_layer: u32,
        _head_dim: u32,
        _n_heads: u32,
        _n_kv: u32,
        _kv_width: u32,
        _start_pos: u32,
        _key_lo: u32,
        _scale: f32,
        _n_tok: u32,
        _float_q: bool,
    ) -> bool {
        false
    }
    /// `head_norm_rope_hadamard` with no rotation and each row roped at its layout position.
    /// False: nothing was encoded.
    #[allow(clippy::too_many_arguments)]
    fn head_norm_rope_rows(
        &self,
        _buf: BufId,
        _w_off: u64,
        _head_dim: u32,
        _eps: f32,
        _n_heads: u32,
        _n_tok: u32,
        _rope_dim: u32,
        _rope_base: f32,
        _freqs: Option<&[f32]>,
    ) -> bool {
        false
    }
    /// `causal_conv`'s outputs for a layout batch: tap k of row t reads the row
    /// `kernel - 1 - k` steps up its own path, or `state` where the path leaves the batch.
    /// `state` is read and not advanced. False: nothing was encoded.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv_rows(
        &self,
        _form: ConvForm,
        _src: BufId,
        _w_off: u64,
        _state: BufId,
        _state_off: u32,
        _out: BufId,
        _width: u32,
        _kernel: u32,
        _n_tok: u32,
    ) -> bool {
        false
    }
    /// CO-BATCHED DECODE (docs/continuous-batching.md). A slot holds one conversation's
    /// device state: its KV page table per layer, its recurrent-state buffers
    /// (`BufId::Recur`, `BufId::RecurSnap`) and the rings of the windowed layers
    /// `ring_layers` names (a windowed layer keeps no page table: each conversation's window is
    /// its own ring). The one-row paths -- prefill, a lone decode, capture and restore -- read
    /// and write the SELECTED slot; a co-batched step names each row's slot. `set_slots(n, ..)`
    /// makes slots `0..n` exist, each new one to hold buffers and rings of their current
    /// sizes, made zeroed on its first select (a slot never used costs no memory); slot 0 is
    /// the one every process starts in, and its rings are the ones the load allocated.
    /// False: this backend has no slots, and nothing changed.
    fn set_slots(&self, _n: u32, _ring_layers: &[u32]) -> bool {
        false
    }
    /// The slot the one-row paths address from now on. False: the slot does not exist or its
    /// buffers could not be made on this first select, and the selection is unchanged. A
    /// backend without slots has slot 0 only.
    fn select_slot(&self, slot: u32) -> bool {
        slot == 0
    }
    /// Gives back a slot's per-conversation buffers; its next select makes them again, as its
    /// first did. For a slot no conversation occupies, other than the selected one, with
    /// nothing in flight. True when buffers went back; false for the selected slot, one that
    /// does not exist, one holding none, with work in flight, or on a backend without slots.
    fn release_slot(&self, _slot: u32) -> bool {
        false
    }
    /// While `Some`, a projection of 2+ rows is that many independent decode rows, served on
    /// `route` (docs/continuous-batching.md, section 6); `None` ends the step, and 2+ rows are
    /// one prompt's chunk again. False: this backend has no decode rows, and nothing changed.
    fn set_decode_rows(&self, _route: Option<RowRoute>) -> bool {
        false
    }
    /// Hint that one speculative round uses small causal/tree batches. This is
    /// not an independent-slot co-batch and must not enter its ownership domain
    /// or disable verification graphs. False leaves the existing route unchanged.
    /// A backend accepting `true` must also accept `false` to end the hint.
    fn set_speculative_rows(&self, _on: bool) -> bool {
        false
    }
    /// False keeps a row-layout attention from splitting its keys across threadgroups, so a
    /// chain as a row layout computes what the causal forward computes; true restores the
    /// backend's configured split. False: this backend has no split, and nothing changed.
    fn set_verify_split(&self, _on: bool) -> bool {
        false
    }
    /// The most rows one co-batched step serves on `route`; 0 when this backend has no decode
    /// rows. The exact route ends where the one-row arithmetic does; the fast route serves any
    /// width (its widest steps on the GEMM), so memory bounds it, not this.
    fn decode_rows_max(&self, _route: RowRoute) -> usize {
        0
    }
    /// Cache codecs implemented by this backend's independent row consumers.
    /// Model workflows separately declare their quantized transforms.
    fn supports_decode_rows_kv(&self, k: u32, v: u32) -> bool {
        k == 1 && v == 1
    }
    /// Independent slot-state consumers: (0, 0) requests convolution-only support;
    /// nonzero dimensions additionally require the recurrent matrix operation.
    fn supports_recurrent_slot_rows(&self, _key_dim:u32, _value_dim:u32)->bool {
        false
    }
    /// Per-position head processing including the existing quantized-cache rotation.
    #[allow(clippy::too_many_arguments)]
    fn head_norm_rope_hadamard_at(
        &self, buf: BufId, w: u64, hd: u32, eps: f32, heads: u32,
        pos: &[u32], rd: u32, base: f32, freqs: Option<&[f32]>, nrot: u32,
    ) -> bool {
        if !self.head_norm_rope_at(buf, w, hd, eps, heads, pos, rd, base, freqs) {
            return false;
        }
        if nrot != 0 { self.hadamard(buf, pos.len() as u32 * heads * hd, nrot); }
        true
    }
    /// K and V retain the one-row fused normalization/rotation arithmetic.
    #[allow(clippy::too_many_arguments)]
    fn kv_head_postprocess_at(
        &self, k: BufId, v: BufId, w: u64, hd: u32, eps: f32, heads: u32,
        pos: &[u32], rd: u32, base: f32, freqs: Option<&[f32]>, hk: u32, hv: u32,
    ) -> bool {
        if !self.head_norm_rope_hadamard_at(k, w, hd, eps, heads, pos, rd, base, freqs, hk) {
            return false;
        }
        self.rms_norm(v, NO_WEIGHT, hd, eps, pos.len() as u32 * heads, hd, 0);
        if hv != 0 { self.hadamard(v, pos.len() as u32 * heads * hd, hv); }
        true
    }
    /// `kv_store` for co-batched rows: row r of `src` is stored at `rows[r].pos` in
    /// `rows[r].slot`'s cache, by the one-row store's kernel. False: nothing was encoded.
    #[allow(clippy::too_many_arguments)]
    fn kv_store_slot_rows(
        &self,
        _src: BufId,
        _layer: u32,
        _width: u32,
        _rows: &[SlotRow],
        _is_v: bool,
        _ring: u32,
    ) -> bool {
        false
    }
    /// `attention` for co-batched rows: row r of `BufId::Q` attends over `rows[r].slot`'s
    /// cache through position `rows[r].pos` into row r of `BufId::Attn`, by the one-row
    /// decode's kernel; `max_scores[r]` is that one-row call's score count. False: nothing
    /// was encoded.
    #[allow(clippy::too_many_arguments)]
    fn attention_slot_rows(
        &self,
        _kv_layer: u32,
        _head_dim: u32,
        _n_heads: u32,
        _n_kv: u32,
        _kv_width: u32,
        _scale: f32,
        _window: u32,
        _rows: &[SlotRow],
        _max_scores: &[u32],
        _ring: u32,
    ) -> bool {
        false
    }
    /// `causal_conv` at one token for co-batched rows: row r reads row r of `src`, its
    /// history at `rows[r].state_off` in its slot's `state` buffer, writes the advanced
    /// history at `rows[r].state_out_off` there and row r of `out`, by the one-token step's
    /// kernel. False: nothing was encoded.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv_slot_rows(
        &self,
        _form: ConvForm,
        _src: BufId,
        _w_off: u64,
        _state: BufId,
        _rows: &[SlotStateRow],
        _out: BufId,
        _width: u32,
        _kernel: u32,
    ) -> bool {
        false
    }
    /// `delta_net` at one token for co-batched rows: row r reads row r of `op.qkv`,
    /// `op.alpha` and `op.beta`, its matrix at `rows[r].state_off` in its slot's `op.state`,
    /// writes the advanced matrix at `rows[r].state_out_off` there and row r of `op.out` (of the
    /// epilogue's gate when fused), by the one-token step's kernel. `op.state_off`,
    /// `op.state_out_off`, `op.n_tok` and `op.snap` are not read. False: the step failed.
    fn delta_net_slot_rows(&self, _op: &DeltaNet, _rows: &[SlotStateRow]) -> bool {
        false
    }
    /// `head_norm_rope_hadamard` without a rotation for rows at unrelated positions: row r of
    /// `buf` (`n_heads` heads) is normed and roped at `pos[r]`, by the one-row call's kernel.
    /// False: nothing was encoded.
    #[allow(clippy::too_many_arguments)]
    fn head_norm_rope_at(
        &self,
        _buf: BufId,
        _w_off: u64,
        _head_dim: u32,
        _eps: f32,
        _n_heads: u32,
        _pos: &[u32],
        _rope_dim: u32,
        _rope_base: f32,
        _freqs: Option<&[f32]>,
    ) -> bool {
        false
    }
    /// Each batch row's input to a convolution window -- the value `causal_conv` shifts into
    /// the state for that row -- `width` values written to `inputs` at element
    /// `inputs_off + row * row_elems`. Reads `src` only. False: nothing was encoded.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv_row_inputs(
        &self,
        _form: ConvForm,
        _src: BufId,
        _inputs: BufId,
        _inputs_off: u32,
        _row_elems: u32,
        _width: u32,
        _n_tok: u32,
    ) -> bool {
        false
    }
    /// Copy cache rows of one full-attention layer between logical positions: row `from[i]`
    /// to row `to[i]`, K and V, in the order given, through the layer's own block table. A
    /// row is `k_stride` / `v_stride` bytes, the widths the KV pool addresses. Call with the
    /// device idle.
    ///
    /// The copies run in order, so the caller keeps every destination off the sources of
    /// later copies. A tree commit does: it moves each accepted node down to its depth, and a
    /// node's depth is never larger than its batch row.
    ///
    /// False: nothing was copied -- the default, or a row outside the layer's cache.
    fn kv_move_rows(
        &self,
        _layer: u32,
        _k_stride: u64,
        _v_stride: u64,
        _from: &[u32],
        _to: &[u32],
    ) -> bool {
        false
    }
    /// The gated delta rule over `n_tok` tokens, advancing the recurrent matrix in place.
    ///
    /// Returns false when this backend has no kernel for it, and writes nothing: a model
    /// must then refuse rather than read an untouched output buffer, which is a plausible
    /// answer and a wrong one. Default: no support.
    fn delta_net(&self, _op: &DeltaNet) -> bool {
        false
    }
    /// Whether this backend serves a gated delta-net mixer at all: `delta_net`, the
    /// `PlainSilu` causal-conv form and `mul_strided_sigmoid`.
    ///
    /// ONE question asked once, before the first layer, rather than three checks spread
    /// through a forward. A backend that answers false is never dispatched into those
    /// three entries -- they assert rather than write nothing, because a no-op leaves a
    /// buffer holding the previous layer's values, which is a plausible answer.
    fn supports_gated_delta(&self) -> bool {
        false
    }
    /// Whether `delta_net` applies `DeltaNet::epilogue` itself.
    ///
    /// ASKED BEFORE THE DISPATCH, not after: the caller must know whether to issue the
    /// `rms_norm` + `act_mul` pair, and `delta_net`'s bool already means "the rule ran".
    /// A backend that answers true and then ignores the field would leave the gate
    /// holding an un-normalised row, which is a plausible answer and a wrong one, so the
    /// two are read from the same place.
    fn delta_net_fuses_epilogue(&self) -> bool {
        false
    }
    /// `dst[r * width + i] = src[r * src_stride + src_off + i]` for `i < width`,
    /// `r < n_row`: ONE sub-block out of every row.
    ///
    /// The op a projection that packs two tensors per head needs -- Qwen3.8's Q weight is
    /// 24 heads of `[query(256) | gate(256)]`, and reading it as two halves puts every
    /// head after the first on the wrong side. The default walks the rows with
    /// `copy_range`, which is correct everywhere and one dispatch per row; a backend with
    /// a strided copy overrides it with one.
    fn copy_strided(
        &self,
        dst: BufId,
        src: BufId,
        width: u32,
        src_off: u32,
        src_stride: u32,
        n_row: u32,
    ) {
        for r in 0..n_row {
            self.copy_range(dst, r * width, src, r * src_stride + src_off, width);
        }
    }
    /// `dst[r * dst_stride + dst_off + i] = src[r * width + i]` for `i < width`,
    /// `r < n_row`: contiguous rows written into ONE sub-block of every row, the inverse of
    /// `copy_strided`.
    ///
    /// A drafter's feature taps write each tapped layer's residual rows beside the other
    /// taps' this way. The default walks the rows with `copy_range`, one dispatch per row; a
    /// backend with a strided copy overrides it with one.
    fn scatter_strided(
        &self,
        dst: BufId,
        src: BufId,
        width: u32,
        dst_off: u32,
        dst_stride: u32,
        n_row: u32,
    ) {
        for r in 0..n_row {
            self.copy_range(dst, r * dst_stride + dst_off, src, r * width, width);
        }
    }
    /// `a[r * a_stride + i] *= sigmoid(b[r * b_stride + b_off + i])`, the strided sigmoid
    /// gate. Same parameter order as `mul_strided`, which is the same op without the
    /// sigmoid.
    ///
    /// Sigmoid is not an `Epilogue`: `set_activation` bakes ONE activation into every
    /// epilogue for the whole process, and Qwen3.8 needs SiLU there for its feed-forward
    /// while its attention output gate is a plain sigmoid. Two gates, two activations, in
    /// one architecture.
    #[allow(clippy::too_many_arguments)]
    fn mul_strided_sigmoid(
        &self,
        a: BufId,
        b: BufId,
        width: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_row: u32,
    );
    /// One embedding-table row, dequantised and scaled into `dst`.
    ///
    /// `wkind` is the same weight-type table index `matmat` takes. It used to be absent,
    /// which made the op Q4_0 by assumption: a table with another layout would have been
    /// unpacked by the Q4_0 reader and produced plausible garbage.
    fn row(
        &self,
        wkind: WeightKindWire,
        w_off: u64,
        width: u32,
        index: u32,
        scale: f32,
        dst: BufId,
        dst_off: u32,
    );
    /// Every embedding row of a batch in ONE dispatch: `idx` holds `n_rows` u32 table
    /// indices, written by `write_u32`. Returns false when the backend has no kernel for
    /// this kind or cannot vectorise this width, in which case the caller must fall back
    /// to `row` per token -- which is what the prefill path did for every token before
    /// this existed. Default: no backend support.
    fn gather_rows(
        &self,
        _wkind: WeightKindWire,
        _w_off: u64,
        _width: u32,
        _table_rows: u32,
        _scale: f32,
        _dst: BufId,
        _dst_off: u32,
        _idx: BufId,
        _n_rows: u32,
    ) -> bool {
        false
    }
    fn rms_norm(
        &self,
        buf: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    );
    fn rms_norm_from(
        &self,
        buf: BufId,
        src: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    );
    /// Whether one-token projection boundaries should publish a backend-private
    /// prepared activation. The default keeps existing backends on their established
    /// normalization path; a backend may enable this only through its tuned policy.
    fn use_decode_projection_preparation(&self) -> bool {
        false
    }
    /// Whether batched projection boundaries should publish the backend's prepared
    /// activation layout. This is separate from Decode because the producer layout,
    /// consumers and end-to-end crossover differ.
    fn use_prefill_projection_preparation(&self) -> bool {
        false
    }
    /// Normalize an activation that will feed one or more projections. The default is
    /// exactly `rms_norm_from`; discrete quantized backends may also prepare a reusable
    /// activation representation while producing the same public f32 buffer.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_projection(
        &self,
        buf: BufId,
        src: BufId,
        w_off: u64,
        width: u32,
        eps: f32,
        n_row: u32,
        row_stride: u32,
        base_off: u32,
    ) {
        self.rms_norm_from(buf, src, w_off, width, eps, n_row, row_stride, base_off);
    }
    /// Normalize projected heads, apply RoPE, and optionally apply the orthonormal
    /// block transform used by quantized KV. The default preserves the established
    /// three-operation sequence; backends may fuse only the implementation.
    #[allow(clippy::too_many_arguments)]
    fn head_norm_rope_hadamard(
        &self,
        buf: BufId,
        w_off: u64,
        head_dim: u32,
        eps: f32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        rope_dim: u32,
        rope_base: f32,
        freqs: Option<&[f32]>,
        hadamard_nrot: u32,
    ) {
        self.rms_norm(buf, w_off, head_dim, eps, n_tok * n_heads, head_dim, 0);
        self.rope(
            buf, rope_dim, rope_base, head_dim, n_heads, start_pos, n_tok, freqs,
        );
        if hadamard_nrot != 0 {
            self.hadamard(buf, n_tok * n_heads * head_dim, hadamard_nrot);
        }
    }
    /// Postprocess paired K/V tensors while preserving the existing five-operation
    /// ordering. The default keeps Metal and other backends byte-for-byte on their
    /// established path; CUDA may fuse memory traffic behind the same contract.
    #[allow(clippy::too_many_arguments)]
    fn kv_head_postprocess(
        &self,
        k: BufId,
        v: BufId,
        k_norm_off: u64,
        head_dim: u32,
        eps: f32,
        n_kv: u32,
        start_pos: u32,
        n_tok: u32,
        rope_dim: u32,
        rope_base: f32,
        freqs: Option<&[f32]>,
        k_hadamard_nrot: u32,
        v_hadamard_nrot: u32,
    ) {
        self.rms_norm(k, k_norm_off, head_dim, eps, n_tok * n_kv, head_dim, 0);
        self.rms_norm(v, NO_WEIGHT, head_dim, eps, n_tok * n_kv, head_dim, 0);
        self.rope(
            k, rope_dim, rope_base, head_dim, n_kv, start_pos, n_tok, freqs,
        );
        if k_hadamard_nrot != 0 {
            self.hadamard(k, n_tok * n_kv * head_dim, k_hadamard_nrot);
        }
        if v_hadamard_nrot != 0 {
            self.hadamard(v, n_tok * n_kv * head_dim, v_hadamard_nrot);
        }
    }
    /// Try to postprocess K/V and write their quantized cache representation as
    /// one backend transaction. Returning `false` guarantees that neither the
    /// source buffers nor the cache were modified, so the workflow can execute
    /// the portable postprocess + store sequence unchanged.
    #[allow(clippy::too_many_arguments)]
    fn kv_head_postprocess_store(
        &self,
        _k: BufId,
        _v: BufId,
        _k_norm_off: u64,
        _v_rms_norm: bool,
        _head_dim: u32,
        _eps: f32,
        _n_kv: u32,
        _start_pos: u32,
        _n_tok: u32,
        _rope_dim: u32,
        _rope_base: f32,
        _freqs: Option<&[f32]>,
        _k_hadamard_nrot: u32,
        _v_hadamard_nrot: u32,
        _layer: u32,
        _ring: u32,
    ) -> bool {
        false
    }
    /// Try to evaluate `dst = rms_norm(src, weight) + add` in one backend operation.
    ///
    /// This is an optional, model-agnostic fusion. Returning `false` promises that no
    /// buffers were modified; the caller must then issue `rms_norm_from` and `add`
    /// separately. Backends must also return `false` for layouts they cannot fuse.
    /// Keeping the fallback at this seam lets CUDA match its native reduction/add
    /// association without changing Metal or forcing every backend to expose a
    /// model-named entry point.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_add(
        &self,
        _dst: BufId,
        _src: BufId,
        _w_off: u64,
        _width: u32,
        _eps: f32,
        _n_row: u32,
        _row_stride: u32,
        _base_off: u32,
        _add: BufId,
        _output_scale: f32,
    ) -> bool {
        false
    }
    /// Try `dst = rms_norm(resid + other) * w` with `resid += other` as one operation:
    /// the PRE-norm residual order (LFM2), where every residual add is followed by the
    /// next norm reading the sum. Returning `false` promises that no buffers were
    /// modified; the caller then issues `add` and `rms_norm_from` separately. Bits must
    /// match that two-step form (same float adds, same reduction over the same values).
    #[allow(clippy::too_many_arguments)]
    fn add_rms_norm(
        &self,
        _dst: BufId,
        _resid: BufId,
        _other: BufId,
        _w_off: u64,
        _width: u32,
        _eps: f32,
        _n_row: u32,
        _row_stride: u32,
        _base_off: u32,
    ) -> bool {
        false
    }
    /// Try the in-place RMSNorm+residual operation while also preparing the
    /// quantized activation representation consumed by an immediate projection.
    /// Returning `false` promises no mutation; the caller retains the ordinary
    /// `rms_norm_add` fallback. Backends without a discrete projection layout do
    /// not need to implement this capability.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm_add_projection(
        &self,
        _dst: BufId,
        _src: BufId,
        _w_off: u64,
        _width: u32,
        _eps: f32,
        _n_row: u32,
        _row_stride: u32,
        _base_off: u32,
        _add: BufId,
    ) -> bool {
        false
    }
    /// Try the adjacent transformer boundary as one backend operation:
    ///
    /// mid = rms(src, first_weight) + residual
    /// out = rms(mid, second_weight)
    ///
    /// Returning false promises no writes. This optional seam lets a backend
    /// keep mid on chip while still materializing it for the later residual;
    /// backends without a proven implementation retain the two established
    /// operations unchanged.
    #[allow(clippy::too_many_arguments)]
    /// `mid = (residual + norm(src) * w1) * output_scale`, then `out = norm(mid) * w2`:
    /// a norm-and-residual followed by the NEXT consumer's input norm, one dispatch. The
    /// scale is the layer's output scalar (1.0 where a model has none).
    fn rms_norm_add_dual_projection(
        &self,
        _src: BufId,
        _residual: BufId,
        _first_w_off: u64,
        _mid: BufId,
        _second_w_off: u64,
        _out: BufId,
        _width: u32,
        _eps: f32,
        _n_row: u32,
        _row_stride: u32,
        _base_off: u32,
        _output_scale: f32,
    ) -> bool {
        false
    }
    fn rope(
        &self,
        buf: BufId,
        n_rot: u32,
        base: f32,
        head_dim: u32,
        n_heads: u32,
        start_pos: u32,
        n_tok: u32,
        freqs: Option<&[f32]>,
    );
    fn hadamard(&self, buf: BufId, n: u32, nrot: u32);
    fn kv_store(
        &self,
        src: BufId,
        layer: u32,
        width: u32,
        start_pos: u32,
        n_tok: u32,
        is_v: bool,
        ring: u32,
    );
    fn attention(
        &self,
        kv_layer: u32,
        head_dim: u32,
        n_heads: u32,
        n_kv: u32,
        kv_width: u32,
        start_pos: u32,
        // THE SCORE SCALE IS THE OP'S: softmax(scale * q.k). Each backend applies it inside
        // this entry (a scale of Q before the kernel -- softmax((scale q).k) is the same
        // function), so a model cannot forget it: forgetting is a missing argument, not
        // logits that are right at n = 1 and wrong at n >= 2 (a softmax over one position
        // is 1 whatever the scale). 1.0 where the file folds it into the weights, as
        // gemma4's q_norm does; 1/sqrt(head_dim) otherwise.
        scale: f32,
        window: u32,
        n_tok: u32,
        max_scores: u32,
        ring: u32,
    );
    /// Which activation every epilogue applies, for this process.
    ///
    /// SPECIALISES the kernels, so it must be called before `init_weights`; afterwards it
    /// is a no-op. One value per process because that is what a model is -- gemma4 is
    /// GELU in every layer, LFM2 is SiLU in every layer.
    ///
    /// It is not a per-dispatch argument because making it one MEASURABLY changed the
    /// arithmetic: with the value still GELU, a runtime branch in the epilogue moved
    /// gemma4's n=16 logits from 25.582184 to 25.582018, reproducibly.
    fn set_activation(&self, act: Epilogue);
    /// The attention head dims THIS model uses, distinct, before `init_weights`.
    ///
    /// A specialised attention kernel is compiled per head dim because the dim sizes a
    /// REGISTER array, which the Metal compiler requires to be a constant expression
    /// (`simdgroup_float8x8 o[n]` is rejected: "array size is not a constant expression").
    /// The backend compiles its shader from source on the device at every start, so the
    /// right set to compile is the one the MODEL actually uses -- not a list written in
    /// the shader, which both misses dims nobody wrote down and compiles dims this model
    /// can never dispatch.
    ///
    /// Must be called before `init_weights`, for the same reason `set_activation` must:
    /// the value is baked in when the library is built, so setting it afterwards is a
    /// silent no-op. Default does nothing -- a backend that does not specialise on the
    /// dim ignores it.
    fn set_attention_head_dims(&self, _dims: &[u32]) {}
    /// The K/V row width (kv heads x head dim, in elements) for each head dim passed to
    /// `set_attention_head_dims`, same order. A backend that specialises its attention
    /// kernels at library compile takes the row stride as a compile-time constant from it
    /// (every K/V tile load carries that stride); a backend that does not, ignores it.
    fn set_attention_kv_widths(&self, _widths: &[u32]) {}
    /// The delta-net head widths this model uses -- `key_dim` is the Q/K head width and
    /// the state's key coordinate, `value_dim` the V head width and its value coordinate.
    ///
    /// Same timing and same reason as `set_attention_head_dims`: the recurrence holds one
    /// state row per lane in REGISTERS, and a register array's size must be a constant
    /// expression, so the dims are compiled in. A backend that does not specialise on
    /// them ignores this. Zero means the model has no recurrent mixer.
    fn set_recurrent_dims(&self, _key_dim: u32, _value_dim: u32) {}
    /// `a = act(a)`, elementwise.
    fn act(&self, a: BufId, n: u32);
    /// `a = act(a) * b`, elementwise -- the DECODE form of the fused epilogue.
    ///
    /// Prefill folds this into the up projection's write-back. At one token that measured
    /// worse (38.2-38.6 -> 37.9-38.2 tok/s): a read-modify-write per output row inside
    /// the GEMV, against a wide vectorised pass here.
    fn act_mul(&self, a: BufId, b: BufId, n: u32);
    fn add(&self, a: BufId, b: BufId, n: u32);
    fn add_scale(&self, a: BufId, b: BufId, k: f32, n: u32);
    fn scale(&self, a: BufId, k: f32, n: u32);
    fn copy(&self, dst: BufId, src: BufId, n: u32);
    /// `n` floats from `src[src_off..]` to `dst[dst_off..]` (element offsets). `dst` may
    /// be `src` when the two ranges do not overlap; the caller guarantees that.
    fn copy_range(&self, dst: BufId, dst_off: u32, src: BufId, src_off: u32, n: u32);
    /// Copy after a completed synchronous forward, outside its reusable graph.
    /// A backend with retained decode state must preserve that lifecycle here.
    fn copy_range_after_forward(
        &self,
        dst: BufId,
        dst_off: u32,
        src: BufId,
        src_off: u32,
        n: u32,
    ) -> Result<(), i32> {
        self.begin();
        self.copy_range(dst, dst_off, src, src_off, n);
        self.end()
    }
    fn mul_strided(
        &self,
        a: BufId,
        b: BufId,
        n: u32,
        b_off: u32,
        b_stride: u32,
        a_stride: u32,
        n_tok: u32,
    );
    fn softcap(&self, a: BufId, cap: f32, n: u32);
    fn argmax(&self, src: BufId, dst: BufId, n: u32);
    /// The greedy pick stored twice: into `tokens[0]`, where the next decode step's
    /// embedding gather reads it, and into `pick[pick_slot]` for the host to read one step
    /// later. Same rule as `argmax`. Reached only when `decode_pipelining` is true; the
    /// default writes `pick[0]` alone so a backend that never claims the capability is
    /// not silently half-wired.
    fn argmax_feed(
        &self,
        src: BufId,
        _tokens: BufId,
        pick: BufId,
        _pick_slot: u32,
        n: u32,
    ) {
        self.argmax(src, pick, n);
    }
    /// Whether `argmax_rows` is served. Asked before a forward that needs it, so a backend
    /// that answers false is never dispatched into `argmax_rows`.
    fn supports_argmax_rows(&self) -> bool {
        false
    }
    /// One greedy pick per row: `dst[r]` receives, as one u32, the index within row `r` of
    /// the `rows` rows of `width` floats in `src`, smallest on ties. That is the rule of
    /// `argmax`, so a row's pick is `argmax` over that row alone.
    fn argmax_rows(&self, _src: BufId, _dst: BufId, _width: u32, _rows: u32) {
        unreachable!(
            "argmax_rows on a backend whose supports_argmax_rows returned false"
        )
    }
    /// Whether `top_k_rows` is served at `k` entries per row.
    fn supports_top_k_rows(&self, _k: u32) -> bool {
        false
    }
    /// Elements `top_k_rows` needs in `dst` for `rows` rows of `width` floats at `k` entries
    /// per row: the output, then the entry's working space.
    fn top_k_rows_len(&self, _width: u32, rows: u32, k: u32) -> u64 {
        2 * u64::from(rows) * u64::from(k)
    }
    /// The `k` largest entries of each of the `rows` rows of `width` floats in `src`. Row `r`'s
    /// `j`-th entry lands in `dst[r * k + j]`, its index within the row as a u32, and in
    /// `dst[rows * k + r * k + j]`, its value. Larger values come first, and the smaller index
    /// first among equal values: the rule of `argmax`, so a row's entry 0 is its `argmax`. The
    /// rest of `dst`, up to `top_k_rows_len` elements, is working space. Rows hold no NaN and
    /// `k <= width`. False when refused, with nothing written.
    fn top_k_rows(
        &self,
        _src: BufId,
        _dst: BufId,
        _width: u32,
        _rows: u32,
        _k: u32,
    ) -> bool {
        false
    }
    /// Whether a ROUTED feed-forward is served: `moe_gate`, `moe_plan`, `moe_grouped` and
    /// `moe_combine` together. A backend answers for the set, because a model that has only
    /// some of them cannot run a routed layer at all.
    fn supports_moe(&self) -> bool {
        false
    }
    /// A routed layer's gating. `scores` holds `n_tok` rows of `n_expert` router outputs.
    /// Writes `probs`, the probabilities (softmax over the row, or the logistic of each
    /// entry), and `sel`, those probabilities PLUS the file's per-expert selection bias at
    /// `bias_off` (`u64::MAX` when the file carries none).
    ///
    /// TWO OUTPUTS BECAUSE THE BIAS SELECTS AND DOES NOT WEIGH: the pick reads `sel`, the
    /// weights read `probs`. One buffer would silently weigh by the biased value, which
    /// still produces fluent text.
    ///
    /// False when refused, with nothing written.
    fn moe_gate(
        &self,
        _scores: BufId,
        _probs: BufId,
        _sel: BufId,
        _bias_off: u64,
        _n_tok: u32,
        _n_expert: u32,
        _gating: ExpertGating,
    ) -> bool {
        false
    }
    /// The work rows of a routed layer, sorted by expert. `topk` holds each token's `k`
    /// picked expert ids as u32 (`top_k_rows`' index half over `sel`), `probs` the gating's
    /// unbiased probabilities.
    ///
    /// Writes, over `n_tok * k` work rows: `perm[w]`, the token work row `w` belongs to;
    /// `wgt[w]`, its routing weight (the picked probabilities, renormalised over the `k`
    /// picks when `normalise` is set, then scaled by `scale`); `seg[e..=e+1]`, expert `e`'s
    /// half-open range of work rows, `n_expert + 1` entries; and `inv[t * k + j]`, the work
    /// row of token `t`'s `j`-th pick, which the combine reads back.
    ///
    /// False when refused, with nothing written.
    fn moe_plan(
        &self,
        _topk: BufId,
        _probs: BufId,
        _perm: BufId,
        _wgt: BufId,
        _seg: BufId,
        _inv: BufId,
        _n_tok: u32,
        _n_expert: u32,
        _k: u32,
        _normalise: bool,
        _scale: f32,
    ) -> bool {
        false
    }
    /// One matmul over `rows` work rows, each reading the weight of the expert whose segment
    /// it falls in: `dst[w]` is `src[perm[w]]` times expert `e`'s `n_in` by `n_out` matrix,
    /// where `seg[e] <= w < seg[e + 1]`. The stacked weight starts at `w_off` and expert `e`
    /// begins `e * expert_stride` bytes into it.
    ///
    /// `dst` holds one row per WORK ROW. Where `src`'s rows live is `src_work_rows`: false
    /// means one row per TOKEN and work row `w` reads `perm[w]` (the gate and up
    /// projections), true means one row per work row and `w` reads `w` (the down
    /// projection, whose input the other two just wrote). Both counts are passed and
    /// neither is derived from the other.
    ///
    /// `prefill_chunk`: this forward is a prompt chunk of one sequence. Its positions must
    /// then compute the same whatever the chunk's width -- a resumed prompt is chunked
    /// differently from a cold one, and the KV pool reuses one for the other byte for
    /// byte. False for a verify batch, a decode step and co-batched rows.
    ///
    /// False when refused, with nothing written.
    fn moe_grouped(
        &self,
        _wkind: u32,
        _w_off: u64,
        _expert_stride: u64,
        _src: BufId,
        _dst: BufId,
        _perm: BufId,
        _seg: BufId,
        _n_in: u32,
        _n_out: u32,
        _n_expert: u32,
        _n_tok: u32,
        _rows: u32,
        _src_work_rows: bool,
        _prefill_chunk: bool,
    ) -> bool {
        false
    }
    /// The gate, the top-k pick and the work-row plan as ONE dispatch: everything
    /// `moe_gate`, `top_k_rows` and `moe_plan` write, written by one kernel.
    ///
    /// A fold of ALREADY DEPENDENT work -- the pick reads what the gate wrote and the plan
    /// reads what the pick wrote -- so no dispatch that could have overlapped is serialised.
    ///
    /// False when refused, with nothing written; the caller then runs the three.
    #[allow(clippy::too_many_arguments)]
    fn moe_route(
        &self,
        _scores: BufId,
        _probs: BufId,
        _sel: BufId,
        _topk: BufId,
        _perm: BufId,
        _wgt: BufId,
        _seg: BufId,
        _inv: BufId,
        _bias_off: u64,
        _n_tok: u32,
        _n_expert: u32,
        _k: u32,
        _gating: ExpertGating,
        _normalise: bool,
        _scale: f32,
    ) -> bool {
        false
    }
    /// The gate and up projections and the activation between them, as ONE routed dispatch:
    /// `dst[w] = act(src[perm[w]] . gate_e) * (src[perm[w]] . up_e)`, one row per work row,
    /// for the expert `e` whose segment `w` falls in. The two stacks share `expert_stride`
    /// and a weight kind.
    ///
    /// A GRID MERGE, NOT A FOLD. Gate and up both read the token's hidden state and neither
    /// is an input to the other, so running them in one kernel introduces no dependency the
    /// three dispatches did not have.
    ///
    /// False when refused, with nothing written -- the caller then runs `moe_grouped` twice
    /// and `act_mul`, which computes the same thing. `prefill_chunk` as for `moe_grouped`.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_pair(
        &self,
        _wkind: u32,
        _gate_off: u64,
        _up_off: u64,
        _expert_stride: u64,
        _src: BufId,
        _dst: BufId,
        _perm: BufId,
        _seg: BufId,
        _n_in: u32,
        _n_out: u32,
        _n_expert: u32,
        _n_tok: u32,
        _rows: u32,
        _prefill_chunk: bool,
    ) -> bool {
        false
    }
    /// The same routed gate/up operation with caller-owned temporary storage.
    /// The contents of scratch are dead on entry and after this operation.
    /// A provider may use it for gathered input and the up result; existing
    /// providers retain their pair implementation.
    ///
    /// False promises no writes. An execution failure after dispatch must remain
    /// a backend error and must not authorize the caller's ordinary fallback.
    #[allow(clippy::too_many_arguments)]
    fn moe_grouped_pair_with_scratch(
        &self,
        wkind: u32,
        gate_off: u64,
        up_off: u64,
        expert_stride: u64,
        src: BufId,
        dst: BufId,
        _scratch: BufId,
        perm: BufId,
        seg: BufId,
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        n_tok: u32,
        rows: u32,
        prefill_chunk: bool,
    ) -> bool {
        self.moe_grouped_pair(
            wkind, gate_off, up_off, expert_stride, src, dst, perm, seg,
            n_in, n_out, n_expert, n_tok, rows, prefill_chunk,
        )
    }
    /// Each token's expert outputs, summed by routing weight: `dst[t]` is the sum over the
    /// `k` slots of `wgt[inv[t * k + j]] * src[inv[t * k + j]]`, `n_embd` floats a row.
    ///
    /// SLOT ORDER, EVERY RUN. Atomics would order the sum differently each time and this
    /// engine's determinism gate fails on the next step.
    ///
    /// False when refused, with nothing written.
    fn moe_combine(
        &self,
        _src: BufId,
        _wgt: BufId,
        _inv: BufId,
        _dst: BufId,
        _n_embd: u32,
        _k: u32,
        _n_tok: u32,
    ) -> bool {
        false
    }
    /// `resid += W . src`: a projection whose output only ever feeds the residual stream,
    /// added into it in the matmul's store instead of written to a scratch row that
    /// `add_rms_norm` then adds. Bits as `matmat` into a scratch followed by that add.
    /// Returning `false` promises nothing a result depends on was written; the caller then
    /// runs `matmat` into its scratch. A caller that got `true` normalises with
    /// `rms_norm_resid`, never `rms_norm_from` (the reduction must be the pre-add one).
    #[allow(clippy::too_many_arguments)]
    fn matmat_resid(
        &self,
        _kind: u32,
        _w_off: u64,
        _n_in: u32,
        _n_out: u32,
        _src: BufId,
        _resid: BufId,
        _n_tok: u32,
    ) -> bool {
        false
    }
    /// `dst = rms_norm(resid) * w` over rows already holding the residual sum, with the same
    /// floats `add_rms_norm` computes on that sum. False when refused, nothing written.
    fn rms_norm_resid(
        &self,
        _dst: BufId,
        _resid: BufId,
        _w_off: u64,
        _width: u32,
        _eps: f32,
        _n_row: u32,
    ) -> bool {
        false
    }
    /// `moe_combine` into a residual that the next block's norm reads, as one operation:
    /// `resid[t] += sum over the k slots of wgt * src`, then `dst = rms_norm(resid) * w` --
    /// what `moe_combine` into a scratch row followed by `add_rms_norm` computes, with the
    /// same floats (slot-order sum, the same reduction). Returning `false` promises nothing
    /// was written; the caller runs the two steps.
    #[allow(clippy::too_many_arguments)]
    fn moe_combine_add_rms_norm(
        &self,
        _src: BufId,
        _wgt: BufId,
        _inv: BufId,
        _k: u32,
        _dst: BufId,
        _resid: BufId,
        _w_off: u64,
        _width: u32,
        _eps: f32,
        _n_row: u32,
    ) -> bool {
        false
    }
    /// Whether `logistic_rows` is served.
    fn supports_logistic_rows(&self) -> bool {
        false
    }
    /// Elements `logistic_rows` needs in `dst` from `dst_off`: the `rows` outputs, then the
    /// entry's working space.
    fn logistic_rows_len(&self, _a_width: u32, _b_width: u32, rows: u32) -> u64 {
        u64::from(rows)
    }
    /// One logistic output per row: `dst[dst_off + r] = 1 / (1 + exp(-z))` with
    /// `z = w . [a_r ; b_r] + bias`, over the `rows` rows of `a_width` floats in `a` and of
    /// `b_width` floats in `b`. `w` holds the `a_width + b_width` weights from `w_off`, then the
    /// bias. `dst` from `dst_off`, `logistic_rows_len` elements, may share `w`'s buffer where the
    /// two do not overlap. Each row sums in one fixed order and adds the bias last, so a row's
    /// output does not change between calls. False when refused, with nothing written.
    fn logistic_rows(
        &self,
        _a: BufId,
        _a_width: u32,
        _b: BufId,
        _b_width: u32,
        _w: BufId,
        _w_off: u32,
        _dst: BufId,
        _dst_off: u32,
        _rows: u32,
    ) -> bool {
        false
    }
    /// Optional whole-logit greedy selection with accepted-prefix recurrent restore.
    fn supports_greedy_verification(&self) -> bool {
        false
    }
    /// dst[0..3] stores status/consumed/next as u32 bits; dst[3..3+2*rows]
    /// is row workspace. Status zero is success. All logits must be finite;
    /// equal maxima select the first index, matching the host verifier.
    /// Restore snap[(consumed+1)*elems..] only for a partial accepted prefix.
    /// Validate every range/alias before launch; unsupported backends return Err.
    fn verify_greedy_and_restore(
        &self,
        _src: BufId,
        _tokens: BufId,
        _dst: BufId,
        _recur: BufId,
        _snap: BufId,
        _vocab: u32,
        _rows: u32,
        _elems: u32,
    ) -> Result<(), i32> {
        Err(-1)
    }
    fn ple_gather_combine(
        &self,
        proj: BufId,
        tokens_buf: BufId,
        w_offset: u64,
        width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    );
    /// Host-staged tier: copy row `ids[t]` of the row-gathered tensor at file offset
    /// `file_off` (rows of `row_bytes`) into `dst` at `t * row_bytes`, for every t. The
    /// host does the gather, so the device never maps the table. Returns false when this
    /// backend reads tables itself (the caller then uses `ple_gather_combine`).
    fn stage_rows(
        &self,
        _file_off: u64,
        _row_bytes: u32,
        _ids: &[u32],
        _dst: BufId,
    ) -> bool {
        false
    }
    /// `ple_gather_combine` over rows already staged by `stage_rows`: row t is token t.
    fn ple_gather_combine_staged(
        &self,
        _proj: BufId,
        _rows: BufId,
        _width: u32,
        _emb_scale: f32,
        _comb_scale: f32,
        _n_tok: u32,
    ) {
        unreachable!(
            "ple_gather_combine_staged on a backend whose stage_rows returned false"
        )
    }

    /// Combine a compact output prefix with rows from the original wider table.
    /// Callers may use unequal widths only when prefill_ple_prefix_elision is true.
    /// Output prefix with the reference projection's original output width.
    /// The admitted backend preserves its numerical family while omitting dead rows.
    fn matmat_output_prefix(
        &self,
        kind: u32,
        offset: u64,
        n_in: u32,
        canonical_out: u32,
        output_out: u32,
        src: BufId,
        dst: BufId,
        n_tok: u32,
    ) {
        assert_eq!(canonical_out, output_out, "projection prefix not admitted");
        self.matmat(kind, offset, n_in, output_out, src, dst, n_tok);
    }

    /// Fuse projection scaling, per-layer normalization and token-row addition.
    /// Returning false leaves all buffers unchanged and selects the caller fallback.
    /// source_width is the immutable table pitch; output_layers may be a prefix.
    fn ple_norm_gather_combine_prefix(
        &self,
        _proj: BufId,
        _tokens: BufId,
        _norm_offset: u64,
        _table_offset: u64,
        _ple_width: u32,
        _output_layers: u32,
        _source_width: u32,
        _input_scale: f32,
        _eps: f32,
        _emb_scale: f32,
        _comb_scale: f32,
        _n_tok: u32,
    ) -> bool {
        false
    }

    fn ple_gather_combine_prefix(
        &self,
        proj: BufId,
        tokens_buf: BufId,
        w_offset: u64,
        source_width: u32,
        output_width: u32,
        emb_scale: f32,
        comb_scale: f32,
        n_tok: u32,
    ) {
        assert_eq!(source_width, output_width, "PLE prefix gather not admitted");
        self.ple_gather_combine(
            proj,
            tokens_buf,
            w_offset,
            output_width,
            emb_scale,
            comb_scale,
            n_tok,
        );
    }

    // --- weights + config + identity ---
    /// Make the mapped weight blob available to the backend (Metal: share the mmap;
    /// CUDA: upload). Called once at load by the composition root, so imparo-cpu need
    /// not know any backend exists. `base`/`len` describe the CPU-visible mapping.
    ///
    /// # Safety
    /// `base` must point to `len` readable bytes that outlive every kernel.
    unsafe fn init_weights(&self, base: *const u8, len: u64) -> Result<(), i32>;

    /// Initialize weights with optional model-owned residency hints. Backends with a
    /// unified address space or no separate device memory deliberately ignore them.
    ///
    /// # Safety
    /// `base` must point to `len` readable bytes that outlive every kernel, and every
    /// span must lie wholly within that mapping.
    unsafe fn init_weights_with_residency(
        &self,
        base: *const u8,
        len: u64,
        _streamed: &[StreamedWeightSpan],
    ) -> Result<(), i32> {
        unsafe { self.init_weights(base, len) }
    }

    /// The fast tier's size in bytes, when the backend can report one: Metal's recommended
    /// working set, CUDA's VRAM. `None` means unknown and the common runtime places every
    /// weight in the fast tier, as it did before placement existed.
    fn fast_tier_budget(&self) -> Option<u64> {
        None
    }

    /// Additional conversation-state capacity outside an already allocated KV pool.
    /// `Some(bytes)` is a bounded, separate slot budget; charging it must not also
    /// remove KV pages whose storage is already committed. `None` retains the shared
    /// working-set policy of withholding pool pages. Query at scheduler setup after
    /// model allocation; native allocation remains the final authority under pressure.
    fn slot_state_budget_bytes(&self) -> Option<u64> {
        None
    }


    /// Initialize weights under the common runtime's placement. The default hands every
    /// segment outside the fast tier to `init_weights_with_residency` as a span, which is
    /// exactly what a backend without per-tier support did before; a backend that
    /// implements the tiers overrides this.
    ///
    /// # Safety
    /// As `init_weights`; every segment lies inside `[base, base + len)`.
    unsafe fn init_weights_with_placement(
        &self,
        base: *const u8,
        len: u64,
        placement: &WeightPlacement,
    ) -> Result<(), i32> {
        let slow = placement.slow_spans();
        unsafe { self.init_weights_with_residency(base, len, &slow) }
    }

    /// Whether load-time conversion is implemented. Reading a prepacked file alone
    /// does not imply the backend transforms canonical weights during loading.
    /// Use the existing split-plane Q8 codec for Gate/Up fast-tier runtime copies.
    /// False preserves the common per-unit codec; no file format is redefined.
    fn q8_split_gate_up_repack(&self) -> bool { false }

    fn supports_load_time_repack(&self) -> bool {
        false
    }

    /// Load-time repack of fast-tier tensors into a backend-private copy
    /// (docs/memory-tiers-and-fit.md section 7). Each job names one tensor by its file
    /// span and its row-major and tile-major ggml types; the rule for the bytes is
    /// `imparo_gguf::weights::TM_RULES`. Returns one flag per job: true when the backend
    /// now serves that tensor in the `to` layout (the caller then rewrites the tensor's
    /// type so dispatch routes to the tile-major kernels), false when it left the tensor
    /// as it was (no readers for `to`, or the tensor is not in its fast tier). The default
    /// transforms nothing, which is what a backend without tile-major kernels wants.
    ///
    /// # Errors
    /// A backend failure while transforming (allocation, GPU error); the weights are then
    /// in an undefined state and the load must fail.
    fn transform_weights(&self, jobs: &[WeightTransform]) -> Result<Vec<bool>, String> {
        Ok(vec![false; jobs.len()])
    }
    /// Whether this backend has kernels for a weight ggml type. Checked for every tensor at
    /// load: a file carrying a layout this backend cannot read (a tile-major kind without
    /// readers here) is refused with the tensor's name instead of dispatched to a kernel
    /// built for another layout. The default is the row-major set every backend reads.
    fn serves_weight_type(&self, ggml_type: u32) -> bool {
        matches!(ggml_type, 0 | 2 | 8) // F32, Q4_0, Q8_0
    }
    /// Tells the backend which GGML TYPE each compact wire kind is, once at load.
    ///
    /// A backend switches its kernels on the wire kind but its DECODE depends on the ggml
    /// type -- and it must not re-derive one from the other, because that is a second copy
    /// of a mapping imparo-gguf already states (`ggml_type_of_wire`). The common runtime
    /// reads it there and hands the pairs over; a backend that does not need them ignores
    /// this.
    fn set_weight_kind_types(&self, _pairs: &[(WeightKindWire, u32)]) {}
    /// Registers validated model-defined input bases at load, before any forward.
    /// Backends must reject a nonempty unsupported basis rather than produce
    /// plausible output with the wrong model mathematics.
    fn register_weight_input_transforms(
        &self,
        transforms: &[WeightInputTransform],
    ) -> Result<(), String> {
        if transforms.is_empty() {
            Ok(())
        } else {
            Err("backend does not implement this model's weight input basis".into())
        }
    }

    /// Dispatches the backend REFUSED rather than encoding: a weight kind no kernel
    /// serves, a width no route can walk, a span past a kernel's slices. Every refusal
    /// logs, and a log line is not something a harness can act on -- a tuner run refused
    /// 38005 k-quant matmats, printed 38005 lines and still wrote a config in which every
    /// prefill-tile candidate had measured the same empty dispatch. A harness that must
    /// have dispatched reads this and records nothing when it is nonzero. 0 by default:
    /// a backend that cannot count them says so by never rising above zero, so a caller
    /// gets no false all-clear from a backend that simply does not implement it -- it
    /// gets the same answer as a clean run, which is why the tuner ALSO checks the count
    /// moved during a workload it knows dispatches.
    fn refused_dispatches(&self) -> u64 {
        0
    }
    /// Matmul calls this process encoded, per route -- the kernel each projection ran -- by
    /// name. A harness prints the counts that grew while it measured, so a number names the
    /// kernel behind it; a route whose name ends in `_fallback` ran because the chosen kernel
    /// could not. Empty by default: a backend that does not count names no kernel.
    fn matmul_routes(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
    /// Read `dst.len()` weight bytes as the backend will serve them at file offset
    /// `file_off` (after any load-time transform). For verification only; false when the
    /// backend cannot read back (default).
    fn read_weight_bytes(&self, _file_off: u64, _dst: &mut [u8]) -> bool {
        false
    }

    /// Whether the selected per-model/per-device policy wants admission-time
    /// transformed weights. False keeps common model loading allocation-free.
    fn quantized_weight_cache_enabled(&self) -> bool {
        false
    }

    /// Which model-owned projection spans the selected backend policy needs at
    /// admission. The common workflow supplies offsets and shapes; it never names a
    /// device format or interprets backend tuning state.
    fn quantized_weight_cache_plan(&self) -> QuantizedWeightCachePlan {
        QuantizedWeightCachePlan::default()
    }

    /// Prepare a complete persistent weight-cache transaction after activation and KV
    /// admission. `Ok(true)` means every requested matrix is ready; `Ok(false)` is the
    /// conservative unsupported/insufficient-memory fallback. Partial publication is
    /// forbidden so a request never discovers half of a model-specific hot set.
    fn prepare_quantized_weight_cache(
        &self,
        _weights: &[QuantizedWeightPrepack],
    ) -> Result<bool, i32> {
        Ok(false)
    }

    /// Configure KV cache storage types (f16=1, q4_0=2, q8_0=8) before build.
    fn set_kv_types(&self, k: u32, v: u32);

    /// GPU profiling snapshot since the last read (empty when unsupported).
    fn prof_stats(&self) -> ProfStats;
    fn prof_enable(&self, on: bool);

    /// Bytes the backend currently has allocated on the device.
    fn allocated_bytes(&self) -> u64;

    /// Short device tag contributed to the host fingerprint (the #14 seam).
    fn device_tag(&self) -> String;

    /// The KV pool's capability descriptor for this backend.
    fn pool_caps(&self) -> PoolCaps;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct QuantizedWeightCachePlan {
    pub include_down: bool,
    pub include_full_ffn: bool,
    pub include_head: bool,
}

/// One storage tier a backend can place KV state in, nearest-compute first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Tier {
    /// Discrete device memory (CUDA/ROCm): explicit allocations, no lazy commit.
    Device,
    /// Host RAM as a real intermediate (discrete platforms only).
    Host,
    /// One address space for CPU and GPU (Apple Silicon, integrated).
    Unified,
    /// Durable storage; never a backing store for live decode.
    Disk,
}

/// Default error for a backend that does not implement an explicit Host mover.
pub const KV_HOST_UNSUPPORTED: i32 = -38;

/// Opaque, generation-checked handle to backend-owned persistent Host storage.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct KvHostHandle(pub u64);

/// One contiguous physical KV range in a Device <-> Host batch.
///
/// `device_offset` addresses the selected layer/side's raw allocation and
/// `host_offset` addresses `host_handle`. A single batch may name independent
/// content-unit handles, so a whole conversation promotes with one completion
/// barrier without coupling those units' lifetimes. Every byte range is checked by
/// the backend; the common pool derives it from actual geometry and
/// `UnitPlacement`, never from a model-specific constant.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvTransferSpan {
    pub host_handle: u64,
    pub layer: u32,
    pub is_v: u32,
    pub device_offset: u64,
    pub host_offset: u64,
    pub len: u64,
}

/// `len` bytes of one layer side's KV cache at `off`, in the coordinates
/// `Backend::write_kv_bytes` takes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvRange {
    pub layer: u32,
    pub is_v: bool,
    pub off: u64,
    pub len: usize,
}

/// Measured facts for deciding whether a discrete Host tier is useful and safe.
/// Policy (speed margin and retained OS headroom) remains in the common pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostTierProfile {
    /// Currently available pageable+page-lockable host memory reported by the OS.
    pub available_host_bytes: u64,
    /// Measured pinned Host -> Device transfer throughput.
    pub pinned_h2d_bytes_per_second: u64,
    /// Measured Device -> pinned Host transfer throughput.
    pub pinned_d2h_bytes_per_second: u64,
}

/// The pool's capability descriptor (docs/unified-kv-pool.md "Backend capability
/// descriptor").
///
/// What is ENFORCED, and what is only declared, because the difference has already
/// cost a wrong answer once:
///
/// ```text
/// paged_reads      enforced. The server refuses to run the pool without it: a block
///                  table installed for a kernel that ignores it reads somebody else's
///                  rows, and nothing would fail.
/// page_cells       Paged attention's PAGE: `kv_slot` maps through the block table one
///                  page at a time. NOT checked against the identity grid, because the
///                  grid takes its VALUE from it -- one block per placement entry means
///                  one extent per page, so the two agree by construction.
/// finest_cut_tokens    enforced by the same gate, and a DIFFERENT question -- where a
///                  prefill may be cut, set by the attention kernel's query group.
///                  grid_tokens() must be a multiple of it. See docs/kv-identity-grid.md.
/// shared_address   enforced together with tiers when a caller enters the common pool.
///                  Shared backends start at Unified; discrete backends must expose an
///                  explicit Device -> Host path.
/// tiers            implemented capabilities, not aspirations. A backend adds Host only
///                  when its explicit transfer path is ready.
/// ```
///
/// The reason for the last two is that a "copy" between CPU and GPU is a change of
/// reader on a shared address space and a PCIe transfer on a discrete one; a tier list
/// that omits Host on Metal makes the wasteful case unrepresentable. Call
/// [`PoolCaps::validate_for_pool`] before constructing the common pool.
#[derive(Clone, Copy, Debug)]
pub struct PoolCaps {
    /// PAGED ATTENTION'S PAGE, in KV cells: the span one block-table entry maps.
    ///
    /// `imparo_kv`'s identity grid TAKES ITS VALUE from this, so an extent is exactly one
    /// page -- which is what the placement structure requires (`BTreeMap<layer,
    /// BlockIdx>`, one block per layer). A backend with a different page therefore works
    /// unchanged: it gets a grid of that size, and its store is a different root anyway
    /// (`config_root` hashes `device_tag`).
    ///
    /// MUST NOT BECOME TUNER-OWNED. The disk grid is cut on it, so a swept page would
    /// re-cut the layout under every stored conversation on each retune
    /// (docs/unified-kv-pool.md).
    ///
    /// Windowed layers do not page -- they address through a ring -- but the pool
    /// allocates every layer on this quantum.
    pub page_cells: u32,
    /// The FINEST spacing at which a prefill may be CUT -- resumed, or split across
    /// requests -- and still reproduce a cold pass byte for byte.
    ///
    /// A FLOOR, not a grid, which is why it is not named like one: the engine picks its
    /// own quantum and this says how fine that may go. Any multiple of it also
    /// reproduces, so the gate is `grid_tokens() % finest_cut_tokens == 0`.
    ///
    /// A different question from `page_cells`, and a different number: this one comes
    /// from the attention prefill kernel, whose KV scan bounds are taken from a query
    /// GROUP rather than a query, and whose position blocks split into whole 8-runs
    /// plus a scalar remainder. Measured at 8 and at 16 on Metal while `page_cells`
    /// stayed 64 (docs/kv-identity-grid.md).
    ///
    /// Report a CEILING over every kernel this backend may select, not the value of
    /// the current selection: the selection varies per layer (head_dim picks different
    /// instantiations), per request (the window engages with length), per device (a
    /// threadgroup limit) and per KV type. Too coarse costs reuse depth; too fine is a
    /// wrong answer.
    pub finest_cut_tokens: u32,
    /// Whether attention accepts a block table; without it the pool uses the
    /// gather-into-scratch or contiguous fallback.
    pub paged_reads: bool,
    /// Whether device and host address the same bytes.
    pub shared_address: bool,
    /// Ordered, nearest compute first. Disk last everywhere.
    pub tiers: &'static [Tier],
}

/// How the common pool may make KV bytes visible to the compute device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoolAddressing {
    /// CPU and GPU see the same allocation; residency changes never emit a copy.
    Shared,
    /// Device and host are distinct tiers; residency changes require an explicit mover.
    ExplicitHostTransfers,
}

/// One independent reason a backend cannot safely enter the common KV pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoolCapsIssue {
    PagedReadsUnavailable,
    InvalidPageCells { declared: u32 },
    IncompatibleFinestCut { declared: u32, grid_tokens: u32 },
    SharedAddressNeedsUnifiedDisk,
    DiscreteAddressNeedsDeviceHostDisk,
}

impl core::fmt::Display for PoolCapsIssue {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PagedReadsUnavailable => write!(f, "paged_reads=false"),
            Self::InvalidPageCells { declared } => {
                write!(f, "page_cells={declared}, required a non-zero power of two")
            }
            Self::IncompatibleFinestCut {
                declared,
                grid_tokens,
            } => write!(
                f,
                "finest_cut_tokens={declared} does not divide grid_tokens={grid_tokens}"
            ),
            Self::SharedAddressNeedsUnifiedDisk => {
                write!(f, "shared_address=true requires exactly [Unified, Disk]")
            }
            Self::DiscreteAddressNeedsDeviceHostDisk => write!(
                f,
                "shared_address=false requires exactly implemented [Device, Host, Disk]"
            ),
        }
    }
}

impl PoolCaps {
    /// Validate every property needed before the common pool may install block tables.
    ///
    /// All issues are returned together so a backend under bring-up cannot appear to be
    /// one flag flip away from safety. `tiers` is deliberately treated as an
    /// implemented-capability list: advertising Host on a discrete backend promises that
    /// the explicit mover exists. `grid_tokens` is the caller's selected identity grid;
    /// the page feeds that grid today, while the explicit argument keeps the cut contract
    /// testable if copy-on-write later makes the grid finer than a page.
    pub fn validate_for_pool(
        &self,
        grid_tokens: u32,
    ) -> Result<PoolAddressing, Vec<PoolCapsIssue>> {
        let mut issues = Vec::new();
        if !self.paged_reads {
            issues.push(PoolCapsIssue::PagedReadsUnavailable);
        }
        if self.page_cells == 0 || !self.page_cells.is_power_of_two() {
            issues.push(PoolCapsIssue::InvalidPageCells {
                declared: self.page_cells,
            });
        }
        if self.finest_cut_tokens == 0
            || grid_tokens == 0
            || grid_tokens % self.finest_cut_tokens != 0
        {
            issues.push(PoolCapsIssue::IncompatibleFinestCut {
                declared: self.finest_cut_tokens,
                grid_tokens,
            });
        }

        let addressing = if self.shared_address {
            if self.tiers != [Tier::Unified, Tier::Disk] {
                issues.push(PoolCapsIssue::SharedAddressNeedsUnifiedDisk);
            }
            PoolAddressing::Shared
        } else {
            if self.tiers != [Tier::Device, Tier::Host, Tier::Disk] {
                issues.push(PoolCapsIssue::DiscreteAddressNeedsDeviceHostDisk);
            }
            PoolAddressing::ExplicitHostTransfers
        };

        if issues.is_empty() {
            Ok(addressing)
        } else {
            Err(issues)
        }
    }
}

/// A backend-agnostic profiling snapshot (the wrapper in the model layer prints it
/// without any per-backend cfg).
#[derive(Clone, Debug, Default)]
pub struct ProfStats {
    pub gpu_s: f64,
    pub wall_s: f64,
    pub cbs: u64,
    pub dispatches: u64,
    /// Buffer barriers the encoder emitted. One per dispatch means concurrent encoding
    /// buys nothing -- every kernel's tail waits for the next one's head.
    pub barriers: u64,
    /// (category name, ticks, calls) since the last read.
    /// name, seconds, calls, WEIGHT BYTES (0 = the site did not declare any).
    pub categories: Vec<(String, f64, u64, u64)>,
}

/// The five-category knob taxonomy (see imparo-tune's knobs.rs for the doctrine):
/// `Benched` knobs are micro-swept, `EndToEnd` knobs require a whole-engine admission
/// bracket, and the others are computed or profiled once.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KnobCategory {
    ModelShape,
    DeviceProfile,
    Arithmetic,
    Benched,
    EndToEnd,
}
/// The widest token tile any backend's prefill GEMM uses, and therefore how far a
/// half-activation mirror has to be PADDED.
///
/// A GEMM walks whole token tiles, so a conversion pass fills `ceil(n_tok/tile)*tile`
/// rows and a kernel reading its operand straight from the mirror loads whole 8-row
/// fragments. Sizing the mirror at exactly `batch` rows makes both of those read or write
/// past the end: for LFM2's down projection at 455 tokens, `n_in` is 10752 and the
/// conversion writes 480 rows into a buffer holding 455 -- 537 KB past it, landing in
/// whatever the arena packed next.
///
/// The Metal table's widest tile is 128 (shape 8, 32x128); `q8_token_tiles_fit_the_mirror`
/// in imparo-metal asserts this constant covers it, so the two cannot drift apart.
pub const MAX_GEMM_TOKEN_TILE: usize = 128;

/// The stage-1 workload a Micro knob is judged on, spoken in Backend-trait ops so the
/// shared tuner builds it for any backend from the model's shapes.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Workload {
    /// The decode-step matmuls at n_tok=1, at the model's real weight offsets, summed.
    DecodeMix,
    /// One complete single-token dense gated-FFN transaction. Unlike `DecodeMix`, this
    /// reaches `matmat_gated`, so Decode-only projection fusion is ranked on the path it
    /// actually changes instead of being diluted by unrelated standalone matmuls.
    DecodeFfnTransaction,
    /// One single-token short-convolution state transition. The tuner restores the
    /// model-sized recurrent state before every timed repetition.
    DecodeShortconvTransaction,
    /// A narrow batch through the layer's big matmuls (the MTP verification shape).
    NarrowMix(u32),
    /// Single-query attention against a long context.
    AttentionDecode,
    /// Single-query attention against a context DEEP enough to reach the streaming
    /// decode kernel.
    ///
    /// AttentionDecode runs at position 512, and the streaming path only engages at
    /// `n_pos >= attn_stream_min_pos`, which is thousands. So every knob that governs
    /// the streaming kernel -- head sharing, slice count, the threadgroup floor -- was
    /// being ranked on a workload that never reached it, and their candidates duly
    /// landed within noise of each other. Same defect as a prefill tile knob declared on
    /// DecodeMix: the workload has to exercise the thing the knob selects.
    ///
    /// This one FORCES THE SCORE-TILE KERNEL (it ranks attn_min_tgs and attn_threads,
    /// which govern that kernel), so the streaming knobs need the workload below.
    AttentionDecodeDeep,
    /// Single-query attention against the same deep context with the STREAMING kernel
    /// forced: the vector route off and the score-tile boundary at zero. What
    /// attn_stream_hq and attn_stream_slices select. Ranked on AttentionDecodeDeep
    /// (score-tile forced) every head-sharing value read the same 842.5 us on
    /// Qwen3.8-27B while the engine measured 601 / 334 / 238 ms per 24 steps for 1 / 2
    /// / 3 heads per threadgroup at 16k keys -- the knob ranked on a kernel it could
    /// not move, the defect this enum's doc describes, one level down.
    AttentionDecodeStream,
    /// Batched prefill attention against a SHALLOW context -- a cold first chunk.
    AttentionPrefill,
    /// A full prefill chunk against a DEEP context: the shape a 16k prefill spends
    /// almost all of its attention time in, since the scan is the context length and a
    /// 16k scan is 32x a 512 one. Judging a prefill attention shape on the shallow
    /// workload ranks it on 3% of the work it will actually do.
    AttentionPrefillDeep,
    /// ONE DECODE STEP'S ATTENTION, across every layer at ITS OWN geometry and span.
    ///
    /// Every other attention workload here times a single dispatch, and for a model with
    /// one attention geometry that is the same thing. For a model with two it is not:
    /// gemma4 E4B runs 7 full-attention layers at 512 dims among 35 windowed ones at 256
    /// whose spans never exceed their window, so a step's attention is a MIX and a deep
    /// dispatch is 7/42 of it.
    ///
    /// The gap that produced this: the streaming-decode boundary measured on the isolated
    /// deep dispatch reported the score-tile path ahead at 16k, while the engine at 17k
    /// measured streaming 7.3% CHEAPER for exactly those layers (differenced against a run
    /// with IMPARO_SKIP_ATTN=2). Opposite signs, so the isolated dispatch is not what the
    /// engine does.
    DecodeAttentionStep,
    /// A SHORT prefill tail against a DEEP context -- what a prefix-matched turn runs.
    ///
    /// This is the agentic pattern, not an edge case: a continuation re-sends the whole
    /// conversation, the cache matches almost all of it, and only the tail is prefilled
    /// (STATUS.md records reused=640 of 721). The shape is neither of the other two --
    /// too few tokens for the wide prefill tile, too many for the decode path -- and
    /// until now no workload measured it, so no knob was ever chosen for the turn shape
    /// the engine most often sees.
    AttentionPrefillReuse,
    /// The geometry MOST layers run, as a second regime for the prefill-attention knobs:
    /// the deep workload ranks them at `deep_head_dim` with no window (E4B's few global
    /// layers), while most of E4B's layers are head dim 256 inside a sliding window. A
    /// model with one geometry gets the same geometry at the short position instead, so
    /// the cross-check is always a different regime from the primary.
    AttentionPrefillMajor,
    /// The full-chunk GEMM the large st_gemm shape actually serves: the pair projection
    /// at the prefill chunk width (512 tokens), not the 128-token pair tile. Ranking the
    /// large shape at 128 tokens read shape 11 as 0.8% faster while the engine's 512-token
    /// chunks measured it 2% slower (2026-09-02).
    PrefillGemmChunk,
    /// The layer's two widest matmuls at the PREFILL tile: n_embd->n_ff and n_ff->n_embd
    /// at 512 tokens. This is the shape the GEMM tile knobs actually govern. Judging them
    /// on DecodeMix instead -- n_tok=1 matvecs -- is measuring a different kernel path,
    /// and is why a micro-bench once ranked a tile 1.6% faster that was 11% slower in a
    /// real prefill. That was read as "micro-benchmarks cannot rank GEMM tiles, move the
    /// knob end-to-end"; it actually meant the workload was wrong.
    PrefillGemm,
    /// The same two matmuls at a MIX of ubatch widths, which is what a real prefill sends.
    ///
    /// PrefillGemm times ONE token count. That is enough to rank a knob whose answer is
    /// the same for every dispatch, and not enough for one whose answer depends on the
    /// dispatch's own token count -- which is why the second prefill tile could never be
    /// ranked and its threshold was excluded from the registry entirely. A prefill sends
    /// full ubatches and one remainder, so this sends both: the widths a request actually
    /// produces, each let through the engine's own per-dispatch tile rule.
    PrefillGemmWidths,
    /// The same two matmuls over the model's FULL projection at the fewest tokens that
    /// exercise every candidate (the widest token tile in the shape table). For the SECOND
    /// tile of the prefill pair, which the dispatch selects only where it pads no worse
    /// than the first; a remainder width never runs it, so on `PrefillGemmWidths` half of
    /// every measurement of that knob was dilution, and that workload's cache-resident
    /// slice hid the re-read trade besides. Measured on LFM2 (Q8, M3 Pro): end to end by
    /// env pin the 64x64 second tile is worth +3.1% / +3.0% at 5963 / 17123 tokens; on
    /// PrefillGemmWidths it read -0.1% to +1.8%, always inside the noise floor; here
    /// +10.4% on a 2.3% floor at 128 tokens (+4.0% on 2.0% at 512, same order).
    PrefillGemmUbatch,
    /// One complete dense gated-FFN transaction at a tunable token width:
    /// gate/up projection, activation-product, then down projection. This is distinct
    /// from `PrefillGemm`: fusion and private intermediate layouts can only be ranked by
    /// timing the whole semantic operation against its established unfused fallback.
    PrefillFfnTransaction,
    /// The exact 128-token SM86 route bundle. This is intentionally separate from
    /// the general FFN threshold: one value owns D256 Attention, PLE, ready-Q8 and
    /// Down together, so the tuner must time and prove every controlled stage rather
    /// than persist an untested mixture or rank the bundle on FFN alone.
    PrefillFfnExact128,
    /// ONE DECODE STEP THROUGH THE MEGA KERNEL: every layer of the model as one
    /// `Backend::mega_layer` entry, over the tuner's synthetic weights and its own cache,
    /// at a SHORT span -- every layer's dispatch runs at the grid seat. The regime of the
    /// persistent grid knobs (`mega_tgs`, `mega_tgs_large`, `mega_nsg`; task #203):
    /// `DecodeMix` dispatches independent matmuls and never reaches a grid barrier, so it
    /// once called them INERT. A layer the backend refuses is a candidate rejected, never
    /// a timing.
    MegaDecodeStep,
    /// The same step at a DEEP span: the engine's real mix at depth, where a layer whose
    /// span exceeds the vector-attention cap takes the deep variant at one threadgroup per
    /// core and the rest stay at the seat. The cross-check regime of the grid knobs.
    MegaDecodeStepDeep,
}

/// Stable identity of the frozen dynamic-program search surface. The three catalog
/// hashes remain zero until their canonical encoding is activated by the receipt work;
/// consumers must check `identity_ready` before persisting them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProgramCatalogIdentity {
    pub program_pack_abi: u32,
    pub identity_ready: bool,
    pub pack_set_sha256: [u8; 32],
    pub candidate_catalog_sha256: [u8; 32],
    pub eligible_candidate_set_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProgramProvider {
    BuiltIn,
    ProgramPack,
}

/// One opaque implementation of an engine-owned operation contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramCandidateDecl {
    pub variant_id: [u8; 32],
    pub config_id: [u8; 32],
    pub provider: ProgramProvider,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramChoiceDecl {
    pub choice_group_id: String,
    pub candidates: Vec<ProgramCandidateDecl>,
    pub workload: Workload,
    pub cross_check: Option<Workload>,
    pub screened: bool,
    pub bit_affecting: bool,
    pub joint_with: Vec<String>,
}

/// Backend-generic dynamic program surface. Static backends inherit an inert default,
/// so adding CUDA Program Packs cannot add a Metal/model workflow branch.
pub trait BackendPrograms: Sync {
    fn program_catalog_identity(&self) -> ProgramCatalogIdentity {
        ProgramCatalogIdentity::default()
    }

    fn program_choices(
        &self,
        _facts: &ModelFacts,
        _profile: &DeviceProfile,
    ) -> Vec<ProgramChoiceDecl> {
        Vec::new()
    }

    fn bind_program_choice(
        &self,
        _group: &str,
        _variant: &[u8; 32],
    ) -> Result<(), String> {
        Err("backend has no dynamic program catalog".into())
    }

    fn current_program_choice(&self, _group: &str) -> Option<[u8; 32]> {
        None
    }

    fn freeze_program_catalog(&self) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TunerScratchRegion {
    BufferF32 {
        id: BufId,
        off: u64,
        elements: usize,
    },
    KvBytes {
        layer: u32,
        is_v: bool,
        off: u64,
        bytes: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkloadEffects {
    ReadOnly,
    Mutable(&'static [TunerScratchRegion]),
    /// Mutable recurrent state whose exact size comes from the model plan rather than
    /// a backend-global constant.
    ModelRecurrentState,
    /// The mega decode step's writes, sized from the model: the residual and the
    /// activation scratch rows of one token, and this token's K/V row in every layer
    /// that owns a cache. Captured by the tuner from the entries it built.
    MegaDecodeState,
}

impl Workload {
    /// Every mutable tuning workload must declare the exact device ranges it changes.
    /// Adding a workload forces this exhaustive match to be updated instead of silently
    /// benchmarking state left behind by the previous candidate.
    pub const fn effects(self) -> WorkloadEffects {
        match self {
            Self::DecodeMix
            | Self::DecodeFfnTransaction
            | Self::NarrowMix(_)
            | Self::AttentionDecode
            | Self::AttentionDecodeDeep
            | Self::AttentionDecodeStream
            | Self::AttentionPrefill
            | Self::AttentionPrefillDeep
            | Self::DecodeAttentionStep
            | Self::AttentionPrefillReuse
            | Self::AttentionPrefillMajor
            | Self::PrefillGemm
            | Self::PrefillGemmChunk
            | Self::PrefillGemmWidths
            | Self::PrefillGemmUbatch
            | Self::PrefillFfnTransaction
            | Self::PrefillFfnExact128 => WorkloadEffects::ReadOnly,
            Self::DecodeShortconvTransaction => WorkloadEffects::ModelRecurrentState,
            Self::MegaDecodeStep | Self::MegaDecodeStepDeep => {
                WorkloadEffects::MegaDecodeState
            }
        }
    }
}

/// How a Benched knob's value is found.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SweepKind {
    /// Not swept at all -- see `KnobDecl::derive`. Present so a derived knob can sit in
    /// the registry beside the benched ones instead of hiding in backend init.
    Derived,
    /// A policy whose effect exists only across a whole workflow/graph boundary. The
    /// micro tuner records the current value without timing it. Promotion requires an
    /// external whole-engine bracket plus the normal versioned correctness receipt.
    External,
    /// Interleaved sweep over `values` with a per-axis noise floor.
    Values,
    /// A routing boundary: at each rung `n` of `ladder`, a matmat of `n` tokens is timed
    /// with the knob forced to `hi` (its kernel covers n) against `lo` (it does not).
    /// The pick is the last width where the hi-side kernel keeps winning; scans run three
    /// times and must agree within one rung, else the compiled default stands.
    Crossing {
        ladder: &'static [u32],
        hi: u32,
        lo: u32,
    },
    /// A token-count routing boundary whose candidate loses at small batches and wins
    /// from some minimum width onward. At every rung the candidate is forced with `hi`
    /// and the safe route with `lo`; the first stable winning rung becomes the stored
    /// minimum. This is deliberately separate from `Crossing`, whose `hi` route owns
    /// the small side of the boundary.
    TokenMinCrossing {
        ladder: &'static [u32],
        hi: u32,
        lo: u32,
    },
    /// Transaction-aware counterpart of TokenMinCrossing whose candidate owns
    /// the small side. The last stable winning rung is an inclusive maximum.
    /// Unlike Crossing, this measures the declared complete semantic workload.
    TokenMaxCrossing {
        ladder: &'static [u32],
        hi: u32,
        lo: u32,
    },
    /// A routing boundary measured on CONTEXT SPANS rather than token counts: at each
    /// rung the two decode-attention kernels race a single-query dispatch at that span,
    /// and the pick is the FIRST span where the `hi` side wins and keeps winning (the
    /// knob is a `>= threshold` test, so the pick is the rung itself).
    ///
    /// Separate from `Crossing` because the direction is inverted (the hi side loses
    /// while shallow and wins deep) and because the timed op is an attention dispatch,
    /// which has to rotate KV layers -- reps against one layer are served from the SLC
    /// and stop measuring memory at all.
    SpanCrossing {
        ladder: &'static [u32],
        hi: u32,
        lo: u32,
    },
    /// `Crossing`, timed as the independent decode rows of a co-batched step: the backend's
    /// fast decode-rows route is on while each rung runs, so `hi` keeps the rows on the
    /// decode-rows GEMV and `lo` sends them to the GEMM. Without the route on, both arms
    /// would dispatch the GEMM and the sweep would measure nothing.
    RowsCrossing {
        ladder: &'static [u32],
        hi: u32,
        lo: u32,
    },
}

/// One backend-owned performance knob. The shared tuner machinery consumes these;
/// a backend developer adds knobs HERE (their crate), never in shared code. REGISTRY
/// ORDER IS SWEEP ORDER: a knob whose measurement depends on another's pick (nb8_max's
/// crossing runs against the nb8_shape winner) is declared after it.
/// There is no implicit `stage` field. Most knobs are chosen by math or a per-kernel
/// micro-bench. A policy that can only be judged by running the whole engine must declare
/// `EndToEnd` + `External`; the micro tuner then preserves its incumbent instead of
/// manufacturing a no-op timing decision. Command-buffer length remains DERIVED from
/// measured encode cost and buffer turnaround because it has an explicit arithmetic
/// model; Graph/workflow policies without such a model require external admission.
/// What the tuner knows about the model in front of it, for deciding whether a knob
/// APPLIES at all. Shapes come from the GGUF header, so this costs no tensor mapping.
///
/// Not every knob is meaningful for every model. A knob selecting a MoE routing path has
/// nothing to say about a dense model; one that picks between head-dim-512 attention
/// kernels is irrelevant to a model with no such layers. Offering an inapplicable knob is
/// worse than useless: the tuner spends measurements on a value that changes nothing, and
/// then RECORDS a pick for it, which reads as a decision when nothing was decided.
/// What the tuner knows about the DEVICE. Queried where the API exposes it, measured
/// where it does not; a zero means "not established on this host yet", and a derivation
/// that needs one must say so rather than invent a number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceProfile {
    /// Threadgroup memory a single threadgroup may declare. QUERIED.
    pub threadgroup_bytes: u64,
    /// Threads a single threadgroup may hold. QUERIED.
    pub max_threads: u32,
    /// Working set past which reads fall from cache rate to DRAM rate. MEASURED by
    /// sweeping the set and finding the knee; 0 until that probe runs at init.
    pub cache_knee_bytes: u64,
    /// Streaming read rate past the knee, MB/s. MEASURED; 0 until probed.
    pub dram_read_mbs: u32,
    /// Accumulator fragments one THREAD may hold before the compiler spills them to
    /// device memory. MEASURED by sweeping the count and finding the cliff; 0 until
    /// probed. Per thread and flat -- it does NOT divide among a threadgroup's threads,
    /// and assuming it did produced a bound that rejected a legal shape.
    pub max_accumulators: u32,
    /// The prefill attention score loop's OWN matrix-op ceiling, GFLOPS. MEASURED with
    /// that loop's operand mix -- staged Q re-read from threadgroup memory, K streamed
    /// from device -- not with a generic matrix peak. 0 until probed.
    ///
    /// Why it is a separate number: the score phase was judged against a peak measured on
    /// a DIFFERENT access pattern, which made it look like it was leaving most of the
    /// machine unused -- 37% of 7.3 TFLOPS. Measured with its own mix the ceiling is about
    /// 4.1, so it sits near 65%. A ceiling measured on the wrong access pattern is the
    /// ceiling of a kernel that does not exist.
    pub attn_score_ceiling_gflops: u32,
    /// THREADS a memory-bound dispatch needs in flight before the machine stops scaling.
    /// MEASURED; 0 until probed.
    ///
    /// Separate from `fill_threadgroups` because they are different quantities: an
    /// ALU-dense kernel saturates when the arithmetic units are busy, a latency-bound one
    /// not until enough loads are in flight to hide the memory. `attn_min_tgs` gates a
    /// decode attention dispatch, which is the latter.
    ///
    /// IN THREADS, NOT THREADGROUPS, and that is the whole portability of it. A
    /// threadgroup count only means something alongside the threads-per-threadgroup it was
    /// measured at -- the probe runs 256, the decode attention kernel runs `attn_threads`,
    /// and a consumer that took the count as-is would silently import the probe's choice.
    /// Threads divide out cleanly: a consumer divides by its OWN threadgroup size.
    pub fill_threads_membound: u32,
    /// Nanoseconds to SUBMIT an empty command buffer, not waited on -- what a mid-graph
    /// flush costs. MEASURED; 0 until probed.
    pub commit_overhead_ns: u32,
    /// Nanoseconds the host spends encoding one dispatch. MEASURED; 0 until probed.
    ///
    /// Together with `commit_overhead_ns` and the measured cost of a layer's work, these
    /// are what decide how many layers belong in one command buffer -- a trade between
    /// paying the fixed submission cost too often and letting the GPU idle while the host
    /// is still encoding. Neither cost is reported by any API, and a literal in their
    /// place is one machine's balance written down.
    pub encode_cost_ns: u32,
    /// GPU nanoseconds for ONE layer's decode matmuls, at THIS model's dimensions.
    /// MEASURED (the DecodeMix workload); 0 until probed.
    ///
    /// The one entry here that is not pure device ground truth, and it is deliberate: the
    /// trade it feeds -- how many layers belong in a command buffer -- is a model x device
    /// question, not a device one. A 2-billion-parameter layer and a 70-billion one cover
    /// the same fixed submission cost with very different amounts of work. Naming it for
    /// what it is beats pretending it belongs to the device.
    pub layer_work_ns: u32,
    /// GPU nanoseconds for ONE layer's PREFILL matmuls, at this model's dimensions and a
    /// full chunk. MEASURED (the PrefillGemm workload, scaled back up from its slice);
    /// 0 until probed. Checked against the engine once (qwen35: 92 ms per layer here, 81 ms
    /// real) -- 14% over, close enough that the floor claim holds.
    ///
    /// The term that separates the two flush knobs. A prefill layer carries hundreds of
    /// times the GPU work of a decode layer while costing the host the SAME encode, so the
    /// encode lag that justifies flushing during decode is noise during prefill.
    pub layer_work_prefill_ns: u32,

    /// Threadgroups needed before the GPU stops scaling -- the point at which adding more
    /// stops buying parallelism and starts queueing. MEASURED; 0 until probed.
    ///
    /// This is what a dispatch has to reach to be worth dispatching whole rather than
    /// sliced, and it is a pure DEVICE property: it follows core count, and a literal in
    /// its place is one machine's core count written down.
    pub fill_threadgroups: u32,
}

/// What the tuned mega grid seat governs for a model's decode -- the APPLICABILITY fact
/// behind `mega_tgs` / `mega_nsg` (task #203).
///
/// Two owners feed it, and neither alone can answer: the model's workflow decides whether
/// it offers mega entries at all (qwen35's does not yet), the backend decides which FORM
/// those entries run in (LFM2's default is the per-token program at one threadgroup per
/// core, measured in #153; gemma4's is the per-layer dispatch at the seat). A knob offered
/// where it governs nothing is worse than useless: the tuner spends measurements on it and
/// then RECORDS a pick that reads as a decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MegaSeat {
    /// This model's decode dispatches no mega entry; neither knob applies.
    None,
    /// Per-layer entries at the seat: `mega_tgs` sets the grid, `mega_nsg` the width.
    Grid,
    /// Entries run one threadgroup per core (the program form): the grid is derived and
    /// `mega_tgs` governs nothing; `mega_nsg` is still the width floor.
    OnePerCore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelFacts {
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    /// Largest head dim across layers -- architectures mix geometries (gemma4 E4B runs
    /// 512 on its full-attention layers and 256 on its windowed ones).
    pub deep_head_dim: u32,
    /// 0 for a dense model. Non-zero enables the MoE knobs and nothing else.
    pub n_experts: u32,
    /// Experts a token picks (0 for a dense model). With `n_experts`, it says how many
    /// work rows a prefill chunk gives each expert: `chunk * experts_used / n_experts`.
    pub experts_used: u32,
    /// Windowed-attention layers. Non-zero means a chunk boundary can land inside a window
    /// and move that layer's answer, so the chunk must stay at the compiled width.
    pub windowed_layers: u32,
    /// Blocks in the model. From the GGUF header.
    pub n_layers: u32,
    /// Compute dispatches ONE layer encodes in a decode step. Counted from the model's
    /// forward code, not measured -- the tuner never runs the graph, and the engine's
    /// dispatch counter reads 0 unless profiling is on.
    ///
    /// It is here because it is the work term in the command-buffer trade: what a flush
    /// costs the host is a submission, and what it buys is that the GPU can start on the
    /// layers already encoded.
    pub layer_dispatches: u32,
    /// Weight-kind wire values present on this model's layer projections, as a bitmask
    /// over `WeightKind` discriminants (bit 0 = F32, bit 1 = Q4_0, bit 2 = Q8_0).
    ///
    /// A knob that governs one quant's kernels must not be swept on a model that has no
    /// tensor of that quant: the workload would dispatch a different kernel family and
    /// rank the candidates on noise. `applies` reads this.
    ///
    /// u64 BECAUSE THE DISCRIMINANTS REACH 32. It was u32, and `1u32 << 32` does not
    /// overflow in release Rust -- it shifts by 32 % 32 = 0 -- so a model carrying an
    /// IQ2_S_TM tensor set bit 0 and read as "F32 present, IQ2_S_TM absent". Two wrong
    /// answers, no error. The assertion below is what stops the same silence at 64.
    pub weight_kinds: u64,
    /// What the mega grid seat governs on this model (see [`MegaSeat`]). The tuner
    /// resolves it from the model's workflow and the backend's form policy; a bench
    /// literal that has not asked says `None`.
    pub mega_seat: MegaSeat,
}

/// The front of a gemma4 layer offered to `mega_layer` (when `mega_front_wanted`):
/// `o = W_o . attn`, then the sandwich `mid = x + rms(o) * w_post_attn`,
/// `ffn_in = rms(mid) * w_ffn_norm`. With it the block starts at the attention output; `src`
/// (the FFN input buffer) is then unused and `add` receives o_proj's output.
#[derive(Clone, Copy, Debug)]
pub struct MegaFront<'a> {
    pub wo_kind: WeightKindWire,
    pub wo_off: u64,
    pub attn: BufId,
    pub attn_in: u32,
    pub post_attn_norm_off: u64,
    pub ffn_norm_off: u64,
    /// The layer's attention head dim (selects the block's instantiation).
    pub head_dim: u32,
    /// With `mega_attn_wanted`: the decode attention over the cache joins the block too;
    /// the call then replaces `attention` as well and `attn` receives its output.
    pub attention: Option<MegaAttn>,
    /// With `mega_qkv_wanted` (and `attention` given): the layer's input norm, the q/k/v
    /// projections, head norm + rope and the KV store join too -- the call then replaces the
    /// whole layer from the residual `x` on, and `attention.q`, `k`, `v` receive the rows.
    pub qkv: Option<MegaQkv<'a>>,
}

/// The q/k/v front offered to the mega block: `cur = rms(x) * w_in`, the three projections,
/// per-head `rms(w_qn)` + rope on Q, `rms(w_kn)` + rope on K, unweighted rms on V, then K
/// and V into this layer's cache at `start_pos`. `wk`/`wv` are None for a shared-KV layer.
#[derive(Clone, Copy, Debug)]
pub struct MegaQkv<'a> {
    pub wq_kind: WeightKindWire,
    pub wq_off: u64,
    pub wk: Option<(WeightKindWire, u64)>,
    pub wv: Option<(WeightKindWire, u64)>,
    pub q_norm_off: u64,
    pub k_norm_off: u64,
    pub in_norm_off: u64,
    pub rope_dim: u32,
    pub rope_base: f32,
    pub freqs: Option<&'a [f32]>,
    pub k: BufId,
    pub v: BufId,
    /// This layer's index (the cache written is this layer's; must equal `attention.kv_layer`).
    pub layer: u32,
}

/// The decode attention step offered to the mega block: the same arguments `attention`
/// takes at one token (scale 1.0 -- the query is pre-scaled where a model scales).
#[derive(Clone, Copy, Debug)]
pub struct MegaAttn {
    pub kv_layer: u32,
    pub n_heads: u32,
    pub n_kv: u32,
    pub kv_width: u32,
    pub start_pos: u32,
    pub window: u32,
    pub ring: u32,
    pub q: BufId,
    /// The cache basis: the Hadamard block width the workflow rotates Q and K by before a
    /// quantized K store (0 = none). The block rotates the rows it forms the same way.
    pub had_k: u32,
    /// The same for V (0 = none); the block rotates its attention output back by it.
    pub had_v: u32,
}

pub struct KnobDecl {
    pub name: &'static str,
    /// A SECOND regime the winner must not be bad in, or `None` to skip the check.
    ///
    /// A REGIME is a runtime state the engine works in -- a deep prefill chunk, a
    /// prefix-matched tail, a decode step. A VARIANT is a per-regime VALUE: one knob, or
    /// one tuple, answering differently in different regimes. A flip here is the signal
    /// that this knob needs variants rather than one value.
    ///
    /// A knob is swept on the regime where its kernel spends the most time, which is the
    /// right thing to optimise -- but the engine runs other regimes too, and one value
    /// has to serve all of them. This re-measures the winner against the runner-up on a
    /// second workload and says so if the order FLIPS. A flip is the signal that no
    /// single value serves both and the knob wants a per-regime value.
    ///
    /// The prefill attention shape is swept on the deep chunk (256 threadgroups) and
    /// cross-checked on the prefix-matched tail (32 threadgroups, under the ~72 that
    /// fill this GPU) -- very different occupancy, same kernel. Today they agree; the
    /// point of the check is to notice when they stop.
    pub cross_check: Option<Workload>,
    /// Whether this knob applies to the model under test. `None` means always.
    ///
    /// Checked BEFORE the screen, so an inapplicable knob costs no measurement and gets
    /// no recorded pick. The tuner says which knobs it skipped and why, because a knob
    /// silently missing from a config file is indistinguishable from one that was
    /// measured and left at its default.
    pub applies: Option<fn(&ModelFacts) -> bool>,
    /// Whether a VALUE can run at all here, or `None` if every declared value can.
    ///
    /// A candidate that provably cannot run should never reach a dispatch. Today such a
    /// value is rejected by MEASURING it: the screen catches it because it reads 292x off
    /// the pace, which costs a dispatch of a register-spilling kernel saturating the
    /// memory bus -- the thing that stalled a machine for thirty seconds. Legality is
    /// arithmetic; it belongs before the screen, not in it.
    ///
    /// The predicate should ASK THE BACKEND rather than restate its rule. The backend
    /// owns the shape tables and the register model; a copy of that arithmetic in the
    /// registry is a second source of truth that will drift, which is the failure this
    /// codebase keeps finding.
    pub legal: Option<fn(u32, &ModelFacts, &DeviceProfile) -> bool>,
    /// Whether moving this knob can change output BITS.
    ///
    /// The tuner ranks on TIME. A knob that also moves the numerics can therefore trade
    /// accuracy for speed with nothing noticing -- staging Q as half is worth real time
    /// and moved the q4 agreement from 0.301 to 1.025 against a 1.0 tolerance, and that
    /// trade was caught by a human, not by the tuner.
    ///
    /// Declaring it does not decide it. The tuner takes the choice as an argument:
    /// bit-affecting knobs are held at their incumbent unless the caller opts in, and
    /// opting in prints what it obliges -- regenerate the pins, re-check agreement.
    pub bit_affecting: bool,
    /// COMPUTED from ground truth instead of measured, when set.
    ///
    /// This is the "math decides" half of the design given a mechanism. A knob with a
    /// derivation is never swept: its value is a function of the model's shape and the
    /// device's limits, and searching for something you can compute is how a literal
    /// ends up frozen at whatever the author's machine happened to be.
    ///
    /// A derived value is REPORTED in the stored config and never APPLIED from it. The
    /// config records what this host computed; another host must compute its own, and
    /// restoring a number derived elsewhere is exactly the bug deriving it prevents.
    pub derive: Option<fn(&ModelFacts, &DeviceProfile) -> u32>,
    /// CANDIDATES computed from ground truth, when set, instead of `values`.
    ///
    /// The middle case between "math decides" and "the bench searches". Some knobs are
    /// not derivable -- their optimum is a trade the device settles, so it must be
    /// measured -- but the RANGE worth measuring still follows from a measured quantity,
    /// and a hand-typed list is one machine's range written down.
    ///
    /// `attn_min_tgs` is the example. It targets a threadgroup count for the sliced decode
    /// attention dispatch, so the memory-bound fill point is what sets its scale; but more
    /// slices also means more partials to combine, and where that trade lands is not
    /// something arithmetic answers. So the fill point picks the candidates and the bench
    /// picks among them.
    pub candidates: Option<fn(&ModelFacts, &DeviceProfile) -> Vec<u32>>,
    /// Knobs that must be SETTLED before this one is measured.
    ///
    /// This axis was rejected once, on the grounds that every real instance was a boundary
    /// inside the tuple whose regimes it defines -- which the tuple already orders. Then a
    /// measurement produced an instance that is neither, and across tuples:
    ///
    ///   attn_min_tgs = 72 (untuned)   the span scan finds streaming FASTER at every rung,
    ///                                 crossing at 2048
    ///   attn_min_tgs = 48 (tuned)     the same scan finds it SLOWER at every rung, no
    ///                                 crossing at all
    ///
    /// attn_min_tgs sets how the score-tile path slices, and the score-tile path is the
    /// side attn_stream_min_pos races against. Settle it first and the boundary moves.
    ///
    /// The registry order already satisfies this -- by accident of where the declarations
    /// sit in the file, which is exactly the fragility that motivated the axis. Declaring
    /// it does not reorder anything; it lets the tuner CHECK that the order holds, and say
    /// so if an edit breaks it.
    pub after: &'static [&'static str],
    /// The TUPLE this knob belongs to, or `None` for one that stands alone.
    ///
    /// A tuple is several knobs that collapse into ONE virtual knob: they sit at the same
    /// level and meet inside a single selection, so no member has a best value on its
    /// own, and what the tuner picks is the TUPLE -- that composite IS the value.
    ///
    /// The default search is coordinate-wise: hold everything, move one knob, keep it if
    /// it wins. That is only valid when the knobs are independent, and several here are
    /// not -- they meet inside one selection and their best values are joint:
    ///
    ///   lanes x nr0        one pipeline table, indexed by BOTH: p_q4mm_lanes[lanes][nr0]
    ///   nb8_shape x nb8_max   which narrow tile, and the batch size at which it engages;
    ///                      the right boundary depends on which tile sits behind it
    ///   the prefill attention shape   NSG, BLK and PT are each forced by a different
    ///                      limit (accumulator registers, unit count reaching NSG,
    ///                      threadgroup bytes), so no single move reaches the optimum --
    ///                      about ten attempts across two sessions failed exactly here
    ///
    /// A coordinate sweep over a coupled group reports "nothing beat the incumbent" and
    /// is believed, because every single step really is worse. Only the joint move wins.
    pub tuple: Option<&'static str>,
    pub category: KnobCategory,
    /// Candidate values for Values-swept knobs; empty for Crossing knobs.
    pub values: &'static [u32],
    pub apply: fn(u32),
    pub current: fn() -> u32,
    /// Candidates must pass the tiny-size screen before any full-size sweep -- the
    /// machine-safety rule: a register-spilling candidate reads ~35-80x there and
    /// would otherwise freeze the host with multi-second command buffers.
    pub screened: bool,
    pub sweep: SweepKind,
    /// Judged on this workload (Micro stage; EndToEnd knobs are measured on the model).
    /// The regime this knob is measured in.
    ///
    /// READ ONLY for `SweepKind::Values`. A Crossing or SpanCrossing knob drives its own
    /// ladder -- it is looking for where two paths cross, which is a different
    /// measurement from timing one -- and a Derived knob is not measured at all. The
    /// field is inert for those, and `--explain` says "own ladder" rather than repeating
    /// a value nothing reads.
    pub workload: Workload,
}

/// The registry half of a backend: its knobs and its OWN search-space version.
/// Versions are per backend so a CUDA space bump never invalidates a Metal config;
/// the stored-config fingerprint carries device_tag + this version.
pub trait BackendKnobs {
    fn knob_registry(&self) -> &'static [KnobDecl];
    fn space_version(&self) -> u32;

    /// PER-TENSOR SEATS, which a knob cannot express: one value for each
    /// `(wire kind, n_in, n_out)` the model carries rather than one for the whole engine.
    ///
    /// `tensors` is `(wire kind, weight offset, n_in, n_out)`; the answer is
    /// `(wire kind, n_in, n_out, value)`, which the tuner writes as `blk_rows.K.IN.OUT=V`.
    ///
    /// A backend that times BOTH candidate kernels on each tensor returns its winners. The
    /// default returns nothing, so a backend without this axis writes no such lines and
    /// nothing else changes.
    ///
    /// It is a measurement, not a sweep, and that is the point: a sweep reads one workload
    /// total, and a total dominated by one tensor group picks that group's kernel for every
    /// tensor. Timing each tensor on its own has no aggregate to hide in.
    fn measure_row_kernel_seats(
        &self,
        _tensors: &[(u32, u64, u32, u32)],
        _rows: u32,
    ) -> Vec<(u32, u32, u32, u32)> {
        Vec::new()
    }

    /// The prefill chunk this backend derives for the model, starting from the engine's
    /// `compiled` chunk, or `None` to keep the stored one (or the compiled one). The tuner
    /// writes it as the config's `batch` line, which the engine applies at load;
    /// `IMPARO_BATCH` still overrides it.
    fn prefill_batch(&self, _m: &ModelFacts, _compiled: usize) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod kv_quantization_route_tests {
    use super::{HadamardWidth, KvByteCodec, KvByteCodecRoute, KvQuantizationRoute};

    #[test]
    fn shared_default_preserves_established_metal_route() {
        let route = KvQuantizationRoute::default();
        assert_eq!(route.key.resolve(256), Ok(256));
        assert_eq!(route.key.resolve(512), Ok(512));
        assert_eq!(route.value.resolve(256), Ok(128));
        assert_eq!(route.value.resolve(512), Ok(128));
    }

    #[test]
    fn fixed_width_must_be_power_of_two_divisor() {
        assert!(HadamardWidth::Fixed(0).resolve(256).is_err());
        assert!(HadamardWidth::Fixed(96).resolve(256).is_err());
        assert!(HadamardWidth::Fixed(512).resolve(256).is_err());
        assert_eq!(HadamardWidth::Disabled.resolve(256), Ok(0));
        assert_eq!(HadamardWidth::Fixed(64).resolve(256), Ok(64));
    }

    #[test]
    fn shared_default_declares_the_common_f16_q4_and_rint_q8_codecs() {
        let route = KvByteCodecRoute::default();
        assert_eq!(route.f16, KvByteCodec::F16LeRneV1);
        assert_eq!(route.q4_0, KvByteCodec::Q4_0LlamaV1);
        assert_eq!(route.q8_0, KvByteCodec::Q8_0RintEvenV1);
    }
}

#[cfg(test)]
mod kv_layout_tests {
    use super::KvLayout;

    /// The page a backend declares in `PoolCaps::page_cells`. Named once here so the
    /// test cannot become a second place that decides what a page is.
    const PAGE: u32 = 64;

    fn layout(slots: u64) -> KvLayout {
        KvLayout {
            layer: 3,
            reserved: 0,
            logical_slots: slots,
            k_stride: 16,
            v_stride: 32,
        }
    }

    #[test]
    fn page_count_covers_boundaries_without_becoming_a_knob() {
        assert_eq!(core::mem::size_of::<KvLayout>(), 32);
        assert_eq!(core::mem::align_of::<KvLayout>(), 8);
        for (slots, pages, max) in [
            (0, 0, None),
            (63, 1, Some(0)),
            (64, 1, Some(0)),
            (127, 2, Some(1)),
        ] {
            assert_eq!(layout(slots).page_count(PAGE), Ok(pages));
            assert_eq!(layout(slots).max_page_entry(PAGE), Ok(max));
        }
    }

    #[test]
    fn non_empty_layout_rejects_missing_stride_and_reserved_bits() {
        for invalid in [
            KvLayout {
                k_stride: 0,
                ..layout(1)
            },
            KvLayout {
                v_stride: 0,
                ..layout(1)
            },
            KvLayout {
                reserved: 1,
                ..layout(1)
            },
        ] {
            assert!(invalid.page_count(PAGE).is_err());
        }
    }

    #[test]
    fn a_page_that_is_not_a_power_of_two_is_refused_rather_than_rounded() {
        for page in [0, 24, 100] {
            assert!(layout(128).page_count(page).is_err());
        }
        assert_eq!(layout(128).page_count(128), Ok(1));
    }
}

#[cfg(test)]
mod pool_caps_tests {
    use super::{PoolAddressing, PoolCaps, PoolCapsIssue, Tier};

    const GRID: u32 = 64;

    #[test]
    fn unified_caps_remain_eligible_without_transfer_tiers() {
        let caps = PoolCaps {
            page_cells: GRID,
            finest_cut_tokens: GRID,
            paged_reads: true,
            shared_address: true,
            tiers: &[Tier::Unified, Tier::Disk],
        };
        assert_eq!(caps.validate_for_pool(GRID), Ok(PoolAddressing::Shared));
    }

    #[test]
    fn current_cuda_shape_is_rejected_for_both_independent_reasons() {
        let caps = PoolCaps {
            page_cells: GRID,
            finest_cut_tokens: GRID,
            paged_reads: false,
            shared_address: false,
            tiers: &[Tier::Device],
        };
        assert_eq!(
            caps.validate_for_pool(GRID),
            Err(vec![
                PoolCapsIssue::PagedReadsUnavailable,
                PoolCapsIssue::DiscreteAddressNeedsDeviceHostDisk,
            ])
        );
    }

    #[test]
    fn page_shape_and_numerical_cut_are_independent_gates() {
        let invalid = PoolCaps {
            page_cells: 0,
            finest_cut_tokens: 24,
            paged_reads: true,
            shared_address: true,
            tiers: &[Tier::Unified, Tier::Disk],
        };
        assert_eq!(
            invalid.validate_for_pool(GRID),
            Err(vec![
                PoolCapsIssue::InvalidPageCells { declared: 0 },
                PoolCapsIssue::IncompatibleFinestCut {
                    declared: 24,
                    grid_tokens: GRID,
                },
            ])
        );
    }

    #[test]
    fn discrete_caps_require_the_complete_unique_ordered_ladder() {
        for tiers in [
            &[Tier::Device, Tier::Disk][..],
            &[Tier::Host, Tier::Device, Tier::Disk][..],
            &[Tier::Device, Tier::Host][..],
            &[Tier::Device, Tier::Host, Tier::Disk, Tier::Disk][..],
            &[Tier::Device, Tier::Host, Tier::Disk, Tier::Unified][..],
        ] {
            let caps = PoolCaps {
                page_cells: GRID,
                finest_cut_tokens: GRID,
                paged_reads: true,
                shared_address: false,
                tiers,
            };
            assert!(
                caps.validate_for_pool(GRID)
                    .unwrap_err()
                    .contains(&PoolCapsIssue::DiscreteAddressNeedsDeviceHostDisk)
            );
        }

        let ready = PoolCaps {
            page_cells: GRID,
            finest_cut_tokens: GRID,
            paged_reads: true,
            shared_address: false,
            tiers: &[Tier::Device, Tier::Host, Tier::Disk],
        };
        assert_eq!(
            ready.validate_for_pool(GRID),
            Ok(PoolAddressing::ExplicitHostTransfers)
        );
    }

    #[test]
    fn shared_caps_require_the_complete_unique_disk_last_ladder() {
        for tiers in [
            &[Tier::Unified][..],
            &[Tier::Unified, Tier::Disk, Tier::Disk][..],
            &[Tier::Disk, Tier::Unified][..],
            &[Tier::Unified, Tier::Host, Tier::Disk][..],
        ] {
            let caps = PoolCaps {
                page_cells: GRID,
                finest_cut_tokens: GRID,
                paged_reads: true,
                shared_address: true,
                tiers,
            };
            assert_eq!(
                caps.validate_for_pool(GRID),
                Err(vec![PoolCapsIssue::SharedAddressNeedsUnifiedDisk])
            );
        }
    }
}

#[cfg(test)]
mod kv_host_transfer_tests {
    use super::{KvHostHandle, KvTransferSpan};

    #[test]
    fn transfer_span_wire_layout_matches_native_abi_23() {
        assert_eq!(size_of::<KvHostHandle>(), 8);
        assert_eq!(size_of::<KvTransferSpan>(), 40);
        assert_eq!(std::mem::offset_of!(KvTransferSpan, host_handle), 0);
        assert_eq!(std::mem::offset_of!(KvTransferSpan, layer), 8);
        assert_eq!(std::mem::offset_of!(KvTransferSpan, is_v), 12);
        assert_eq!(std::mem::offset_of!(KvTransferSpan, device_offset), 16);
        assert_eq!(std::mem::offset_of!(KvTransferSpan, host_offset), 24);
        assert_eq!(std::mem::offset_of!(KvTransferSpan, len), 32);
    }

    #[test]
    fn opaque_host_handle_is_not_a_pointer_contract() {
        let handle = KvHostHandle(0x0000_0007_0000_0003);
        let span = KvTransferSpan {
            host_handle: handle.0,
            layer: 9,
            is_v: 1,
            device_offset: 128,
            host_offset: 256,
            len: 64,
        };
        assert_eq!(span.host_handle, handle.0);
    }
}
