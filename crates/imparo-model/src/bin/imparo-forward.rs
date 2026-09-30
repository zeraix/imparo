//! Dev tool: run the CPU reference forward pass over explicit token ids and print top logits.
//!
//! Token ids rather than text so this milestone does not depend on the tokenizer. The output
//! is meant to be compared against llama.cpp for the same ids.

#[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
mod w4a16_ffn_check;

mod continuation_score;

use std::path::PathBuf;

use imparo_model::build_plan;
// No `use` for Model or KvPoolMember: the tool holds a Box<dyn Model>, and a trait
// object dispatches its own methods -- including the supertrait's -- without them.
use imparo_model::weights::Weights;

fn top10_string(logits: &[f32], out: &mut String) {
    use std::fmt::Write as _;
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in idx.iter().take(10) {
        let _ = write!(out, " {i}:{:.6}", logits[i]);
    }
}

/// `--verify-tree`'s tree: node i's parent, -1 for the root (the anchor, the trail's first token).
/// Nodes 0..=8 are the trail's first nine tokens in a chain; node 9 branches off node 2 and node 12
/// off node 5, each a run that starts with a token the trail did not take there.
const TREE_PARENTS: [i32; 16] = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 2, 9, 10, 5, 12, 13, 14];
/// The leaves of `TREE_PARENTS`; their paths cover every node.
const TREE_LEAVES: [usize; 3] = [8, 11, 15];
/// `--verify-tree`'s commit cases over `TREE_PARENTS`: the nodes that take a token the trail did not
/// take there (its token + 1), and the path a verify must accept when the target's picks are the
/// trail. The cursor's verify is bounded by the tree; repetition 6's commit verify (branch_at_3)
/// by 9 nodes.
///
/// ```text
/// case         wrong nodes   accepted path            KV rows moved
/// chain        9, 12         0 1 2 3 4 5 6 7 8        none
/// branch_at_3  3, 12         0 1 2 9 10 11            9 10 11 -> 3 4 5
/// branch_at_6  6, 9          0 1 2 3 4 5 12 13 14 15  12 13 14 15 -> 6 7 8 9
/// root_only    1, 9, 12      0                        none
/// ```
const TREE_CASES: [(&str, &[usize], &[i32]); 4] = [
    ("chain", &[9, 12], &[0, 1, 2, 3, 4, 5, 6, 7, 8]),
    ("branch_at_3", &[3, 12], &[0, 1, 2, 9, 10, 11]),
    // The branch's fourth node (15) is accepted too: a tree's path is bounded by the tree, not
    // by the drafter's block of 9.
    ("branch_at_6", &[6, 9], &[0, 1, 2, 3, 4, 5, 12, 13, 14, 15]),
    ("root_only", &[1, 9, 12], &[0]),
];
/// Tokens `--verify-tree`'s cursor run emits.
const TREE_EMIT: usize = 96;
/// One-token decoding steps `--verify-tree` runs from a committed tree.
const TREE_POST: usize = 16;
/// The path a verify of `wide_tree` accepts when the target's picks are the trail.
const WIDE_PATH: [i32; 9] = [0, 1, 2, 56, 57, 58, 59, 60, 61];

/// `--verify-tree`'s wide tree over `trail` from index `at`: 64 nodes, so rows 32 to 63 use the
/// layout's second mask word. With the target's picks equal to the trail, a verify with limit 9
/// accepts `WIDE_PATH` and moves rows 56 to 61 down to depths 3 to 8.
///
/// ```text
/// rows 0..=8    the trail as a chain, node 3 taking the trail's token + 1
/// rows 9..=55   wrong children of chain nodes 0..=7 in turn: the trail's token at their depth + 2..6
/// rows 56..=61  the trail's tokens at depths 3..=8, a chain hung under node 2
/// rows 62, 63   wrong children of rows 61 and 60
/// ```
fn wide_tree(
    trail: &[u32],
    at: usize,
    vocab: u32,
) -> imparo_model::speculative::DraftTree {
    let token = |depth: usize, offset: u32| {
        (trail.get(at + depth).copied().unwrap_or(0) + offset) % vocab
    };
    let mut parents = vec![-1_i32];
    let mut tokens = vec![token(0, 0)];
    for depth in 1..=8 {
        parents.push(depth as i32 - 1);
        tokens.push(token(depth, u32::from(depth == 3)));
    }
    for row in 9..56_usize {
        let parent = (row - 9) % 8;
        parents.push(parent as i32);
        tokens.push(token(parent + 1, 2 + (row % 5) as u32));
    }
    for depth in 3..=8 {
        parents.push(if depth == 3 {
            2
        } else {
            parents.len() as i32 - 1
        });
        tokens.push(token(depth, 0));
    }
    parents.push(61);
    tokens.push(token(9, 2));
    parents.push(60);
    tokens.push(token(8, 2));
    imparo_model::speculative::DraftTree::plain(tokens, parents)
}

/// Two captures' committed rows at positions `base..base + rows` in every full-attention layer's K
/// and V, and their whole recurrent states, as `verify-tree` line fields: rows byte-equal, the
/// first unequal (layer, row), the recurrent values that differ, the largest difference, and the
/// model layers those values belong to (`layout` is the plan's recurrent layout).
fn committed_state_fields(
    a: &imparo_kv::KvState,
    b: &imparo_kv::KvState,
    base: usize,
    rows: usize,
    layout: &[(u32, u32, u32, u32)],
) -> String {
    let (mut compared, mut equal) = (0_usize, 0_usize);
    let mut first_unequal: Option<(u32, usize)> = None;
    for (x, y) in a.full.iter().zip(&b.full) {
        let k_stride = x.k.len() / x.positions.max(1);
        let v_stride = x.v.len() / x.positions.max(1);
        for row in 0..rows {
            let p = base + row;
            compared += 1;
            let k_range = p * k_stride..(p + 1) * k_stride;
            let v_range = p * v_stride..(p + 1) * v_stride;
            if x.k.get(k_range.clone()) == y.k.get(k_range)
                && x.v.get(v_range.clone()) == y.v.get(v_range)
            {
                equal += 1;
            } else if first_unequal.is_none() {
                first_unequal = Some((x.layer, row));
            }
        }
    }
    let (mut differ, mut max_abs) = (0_usize, 0.0_f32);
    let mut layers_differ: Vec<usize> = Vec::new();
    let pairs = a.recurrent.chunks_exact(4).zip(b.recurrent.chunks_exact(4));
    for (index, (x, y)) in pairs.enumerate() {
        if x == y {
            continue;
        }
        differ += 1;
        let x = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
        let y = f32::from_le_bytes([y[0], y[1], y[2], y[3]]);
        max_abs = max_abs.max((x - y).abs());
        let at = index as u32;
        let owner = layout
            .iter()
            .position(|&(r_off, _, r, s)| at >= r_off && at < r_off + r + s);
        if let Some(layer) = owner
            && layers_differ.last() != Some(&layer)
        {
            layers_differ.push(layer);
        }
    }
    let kv_layer_ids: Vec<u32> = a.full.iter().map(|l| l.layer).collect();
    format!(
        "kv_layers={kv_layer_ids:?} kv_rows_equal={equal}/{compared} first_unequal={first_unequal:?} recurrent_values_differ={differ}/{} recurrent_max_abs={max_abs:.6} recurrent_layers_differ={layers_differ:?}",
        a.recurrent.len() / 4
    )
}

/// `TREE_PARENTS` over `trail` from index `at`: node i takes the trail's token at its depth, or that
/// token + 1 when `wrong` names it.
fn scripted_tree(
    trail: &[u32],
    at: usize,
    wrong: &[usize],
    vocab: u32,
) -> imparo_model::speculative::DraftTree {
    let mut depth = [0_usize; TREE_PARENTS.len()];
    for (i, &parent) in TREE_PARENTS.iter().enumerate().skip(1) {
        depth[i] = depth[parent as usize] + 1;
    }
    let tokens = (0..TREE_PARENTS.len())
        .map(|i| {
            let token = trail.get(at + depth[i]).copied().unwrap_or(0);
            if wrong.contains(&i) {
                (token + 1) % vocab
            } else {
                token
            }
        })
        .collect();
    imparo_model::speculative::DraftTree::plain(tokens, TREE_PARENTS.to_vec())
}

/// `--verify-tree`'s cursor run: a draft provider that proposes the trail as a chain and offers
/// `TREE_PARENTS` over it, cycling through `TREE_CASES`. It counts the tree commits the cursor hands
/// back, those whose path leaves the chain, and those that accepted the case's path.
struct ScriptedTree {
    trail: Vec<u32>,
    base: usize,
    vocab: u32,
    rounds: usize,
    anchor_mismatches: usize,
    tree_commits: usize,
    branch_commits: usize,
    paths_as_scripted: usize,
}

impl imparo_model::speculative::DraftProvider for ScriptedTree {
    fn block_size(&self) -> usize {
        9
    }

    fn draft(
        &mut self,
        _target: &mut dyn imparo_model::Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String> {
        let at = start
            .checked_sub(self.base)
            .ok_or("scripted tree asked before the trail starts")?;
        if self.trail.get(at) != Some(&anchor) {
            self.anchor_mismatches += 1;
        }
        self.rounds += 1;
        Ok((1..9)
            .map(|k| self.trail.get(at + k).copied().unwrap_or(0))
            .collect())
    }

    fn tree_proposal(
        &mut self,
        start: usize,
        _anchor: u32,
        _chain: &[u32],
        _stops: &[u32],
    ) -> Result<Option<imparo_model::speculative::DraftTree>, String> {
        let at = start
            .checked_sub(self.base)
            .ok_or("scripted tree asked before the trail starts")?;
        let (_, wrong, _) = TREE_CASES[(self.rounds - 1) % TREE_CASES.len()];
        Ok(Some(scripted_tree(&self.trail, at, wrong, self.vocab)))
    }

    fn commit_tree(
        &mut self,
        _start: usize,
        _inputs: &[u32],
        path: &[i32],
        _next: u32,
    ) -> Result<(), String> {
        self.tree_commits += 1;
        if path
            .iter()
            .enumerate()
            .any(|(depth, &node)| usize::try_from(node) != Ok(depth))
        {
            self.branch_commits += 1;
        }
        let (_, _, expect) = TREE_CASES[(self.rounds - 1) % TREE_CASES.len()];
        if path == expect {
            self.paths_as_scripted += 1;
        }
        Ok(())
    }
}

/// What `--verify-tree` found for one node, over every leaf path it lies on.
#[derive(Clone, Copy)]
struct TreeNodeCheck {
    /// The node's ancestors are batch rows 0..depth, the key layout a chain has.
    chain_rows: bool,
    /// Every logit bit-equal to the causal path forward's row.
    exact: bool,
    /// The node's top-10 order against the causal row's, worst over the paths through it.
    order: RowOrder,
    /// The largest difference of a top-10 id's delta to the top-1.
    delta: f32,
}

/// `--verify-chain` repetitions 4 to 6: a draft provider that proposes the reference trail, with
/// one token corrupted at a depth that cycles every round and, optionally, a stop token forced in:
/// first in the proposal on even rounds, second on odd rounds. It counts proposals holding a stop
/// token, proposals that begin with one, and anchors that left the trail.
struct ScriptedDraft {
    trail: Vec<u32>,
    base: usize,
    block: usize,
    vocab: u32,
    inject: Option<u32>,
    stops: Vec<u32>,
    rounds: usize,
    proposals_with_stop: usize,
    proposals_led_by_stop: usize,
    anchor_mismatches: usize,
}

impl imparo_model::speculative::DraftProvider for ScriptedDraft {
    fn block_size(&self) -> usize {
        self.block
    }

    fn draft(
        &mut self,
        _target: &mut dyn imparo_model::Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String> {
        let at = start
            .checked_sub(self.base)
            .ok_or("scripted draft asked before the trail starts")?;
        if self.trail.get(at) != Some(&anchor) {
            self.anchor_mismatches += 1;
        }
        let mut proposal: Vec<u32> = (1..self.block)
            .map(|k| self.trail.get(at + k).copied().unwrap_or(0))
            .collect();
        let depth = self.rounds % self.block;
        if depth > 0 {
            proposal[depth - 1] = (proposal[depth - 1] + 1) % self.vocab;
        }
        if let (Some(token), Some(slot)) =
            (self.inject, proposal.get_mut(self.rounds % 2))
        {
            *slot = token;
        }
        if proposal.iter().any(|t| self.stops.contains(t)) {
            self.proposals_with_stop += 1;
        }
        if proposal.first().is_some_and(|t| self.stops.contains(t)) {
            self.proposals_led_by_stop += 1;
        }
        self.rounds += 1;
        Ok(proposal)
    }
}

/// The ten largest logits' ids, largest first. Equal logits keep the smaller id first, the
/// order `top10_string`'s stable sort gives.
fn top10_ids(logits: &[f32]) -> Vec<usize> {
    let mut top: Vec<usize> = Vec::with_capacity(11);
    for (i, &v) in logits.iter().enumerate() {
        if top.len() == 10 && v <= logits[top[9]] {
            continue;
        }
        let at = top.iter().position(|&j| v > logits[j]).unwrap_or(top.len());
        top.insert(at, i);
        top.truncate(10);
    }
    top
}

/// One logits row against its reference under the verify-row rule
/// (docs/speculator-design.md section 8): the rows agree when their top-1 ids are equal and every
/// reference top-10 id's delta to the top-1 moves by at most the tolerance. An order swap below
/// rank 1 is reported, not failed.
/// A logits row's top-10 order against its reference's, from agreeing to not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum RowOrder {
    Equal,
    /// The top-1 ids match; ids below them swap.
    SwapBelowTop1,
    Top1Differs,
}

struct RowLogits {
    reference_top: Vec<usize>,
    row_top: Vec<usize>,
    /// The largest change of a reference top-10 id's delta to the top-1.
    delta: f32,
}

impl RowLogits {
    fn compare(reference: &[f32], row: &[f32]) -> Self {
        let (reference_top, row_top) = (top10_ids(reference), top10_ids(row));
        let delta = reference_top
            .iter()
            .map(|&id| {
                ((reference[id] - reference[reference_top[0]])
                    - (row[id] - row[row_top[0]]))
                    .abs()
            })
            .fold(0.0_f32, f32::max);
        Self {
            reference_top,
            row_top,
            delta,
        }
    }

    fn top1_equal(&self) -> bool {
        self.reference_top.first() == self.row_top.first()
    }

    fn order(&self) -> RowOrder {
        if !self.top1_equal() {
            RowOrder::Top1Differs
        } else if self.first_swap().is_some() {
            RowOrder::SwapBelowTop1
        } else {
            RowOrder::Equal
        }
    }

    /// The first rank, counted from 0, where the two top-10 orders part.
    fn first_swap(&self) -> Option<usize> {
        self.reference_top
            .iter()
            .zip(&self.row_top)
            .position(|(a, b)| a != b)
    }
}

fn print_top10(logits: &[f32]) {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    print!("top10");
    for &i in idx.iter().take(10) {
        print!(" {i}:{:.6}", logits[i]);
    }
    println!();
}

/// Test-only full-distribution witness. The environment variable is deliberately
/// opt-in so normal CLI output and hot-path behavior do not change. The file is a
/// headerless sequence of IEEE-754 f32 values in vocabulary order and little-endian
/// byte order, which lets a cross-process gate compare the actual logits byte-for-byte.
fn dump_logits_to_path(
    logits: &[f32],
    path: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    let mut raw = Vec::with_capacity(std::mem::size_of_val(logits));
    for value in logits {
        raw.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, raw)
}

fn dump_logits_for_env(logits: &[f32], variable: &str) -> std::io::Result<()> {
    let Some(path) = std::env::var_os(variable) else {
        return Ok(());
    };
    dump_logits_to_path(logits, path)
}

/// Lab-only full recurrent witnesses, sampled after prefill and before decode.
/// This readback is deliberately outside the measured prefill interval.
fn dump_recurrent_witness(
    model: &dyn imparo_model::Model,
    rep: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let Some(dir) = std::env::var_os("IMPARO_RECURRENT_DUMP_DIR") else {
        return Ok(());
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir)?;
    let filled = model.kv_runtime().filled;
    let live = model
        .kv_recurrent_blob(filled)
        .ok_or("live recurrent state missing")?;
    std::fs::write(dir.join(format!("rep-{rep:03}-live.raw")), &live)?;
    let note = model.kv_recurrent_note();
    if let Some((_, bytes)) = &note {
        std::fs::write(dir.join(format!("rep-{rep:03}-checkpoint.raw")), bytes)?;
    }
    let meta = serde_json::json!({
        "live_boundary": filled,
        "recurrent_elements": model.plan().recurrent_elems(),
        "checkpoint_boundary": note.as_ref().map(|(at, _)| *at),
        "checkpoint_bytes": note.as_ref().map(|(_, bytes)| bytes.len()),
        "scope": "read-only prefill state witness; not performance evidence",
    });
    std::fs::write(
        dir.join(format!("rep-{rep:03}.json")),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    Ok(())
}

fn dump_logits_if_requested(logits: &[f32]) -> std::io::Result<()> {
    dump_logits_for_env(logits, "IMPARO_LOGITS_DUMP")
}

/// The scramble probe's store-side witness: the first bytes of four K rows of one
/// full-attention layer, read at their identity slots (`swapped` false) or at the slots
/// the pair-swap map sends them to. The harness requires the rows to move intact.
fn print_k_rows(geom: &imparo_kv::LayerStateGeom, swapped: bool) {
    let layer = geom.layer;
    for pos in [0usize, 63, 64, 127] {
        let phys = if swapped {
            ((pos / 64) ^ 1) * 64 + pos % 64 // pair-swap map
        } else {
            pos
        };
        let mut row = vec![0u8; geom.k_stride];
        imparo_model::backend::active().unwrap().read_kv_bytes(
            layer,
            false,
            (phys * geom.k_stride) as u64,
            &mut row,
        );
        let head: Vec<String> = row[..8].iter().map(|b| format!("{b:02x}")).collect();
        if swapped {
            println!("k{layer} pos={pos} phys={phys} head={}", head.join(""));
        } else {
            println!("k{layer}-identity pos={pos} head={}", head.join(""));
        }
    }
}

fn phase_a1_prefill_wall_line(
    enabled: bool,
    rep: usize,
    token_count: usize,
    start_pos: usize,
    split_mode: &str,
    split_at: Option<usize>,
    split_parts: Option<&[usize]>,
    elapsed: std::time::Duration,
) -> Result<Option<String>, serde_json::Error> {
    if !enabled {
        return Ok(None);
    }
    serde_json::to_string(&serde_json::json!({
        "schema": 1,
        "phase": "A1",
        "event": "prefill_wall",
        "lab_only": true,
        "production_authority": false,
        "rep": rep,
        "token_count": token_count,
        "start_pos": start_pos,
        "split_state": {
            "mode": split_mode,
            "split_at": split_at,
            "parts": split_parts,
        },
        "prefill_wall_ms": elapsed.as_secs_f64() * 1e3,
    }))
    .map(Some)
}

/// The first index of the largest value.
fn host_argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// `--decode N --cobatch B [--fast]`: the co-batched decode gate. Slot s holds the prompt cut
/// by `5 s` tokens, in blocks `[s * span, (s + 1) * span)`, so the rows sit at different
/// positions in different blocks. Each slot first decodes N steps alone (the one-row dispatch
/// path); then every slot is prefilled again and the N steps run co-batched, fed the same
/// input tokens.
///
/// On the exact route every row's logits must equal its lone step's, bit for bit: a difference
/// is a block-table, position or state-slot bug. On the fast route (`--fast`) the projections
/// may take another kernel, so the standard is the mega route's: every row picks the token its
/// lone step picks, at every step. The largest top-10 logit difference and the top-10 order
/// swaps are reported beside it.
/// The indices of the ten largest logits, largest first.
fn top_ten(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    let k = idx.len().min(10);
    idx.select_nth_unstable_by(k.saturating_sub(1), |a, b| v[*b].total_cmp(&v[*a]));
    idx.truncate(k);
    idx.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
    idx
}

fn cobatch_gate(
    model: &mut dyn imparo_model::Model,
    tokens: &[u32],
    rows: u32,
    steps: usize,
    capacity: usize,
    fast: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let route = if fast {
        imparo_backend::RowRoute::Fast
    } else {
        imparo_backend::RowRoute::Exact
    };
    let be =
        imparo_model::backend::active().ok_or("--cobatch needs a device backend")?;
    let most = be.decode_rows_max(route);
    if rows == 0 || rows as usize > most {
        return Err(
            format!("--cobatch takes 1..{most} rows on the {route:?} route").into(),
        );
    }
    if std::env::var("IMPARO_MEGA_FFN").as_deref() != Ok("0") {
        return Err(
            "--cobatch compares with the dispatch path: set IMPARO_MEGA_FFN=0".into(),
        );
    }
    // IMPARO_COBATCH_BLK_ROWS_GEMV_MAX=N and IMPARO_COBATCH_BLK_ROWS_MMA_MAX=N (diagnostics, as
    // imparo-metalbench's IMPARO_BENCH_BLK_ROWS_*): the block formats' two crossings for this
    // run, so one binary times a co-batched step on either of their row kernels.
    #[cfg(target_os = "macos")]
    {
        let env_u32 =
            |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u32>().ok());
        if let Some(n) = env_u32("IMPARO_COBATCH_BLK_ROWS_GEMV_MAX") {
            imparo_metal::set_blk_rows_gemv_max(n);
        }
        if let Some(n) = env_u32("IMPARO_COBATCH_BLK_ROWS_MMA_MAX") {
            imparo_metal::set_blk_rows_mma_max(n);
        }
    }
    const CUT: usize = 5;
    let b = rows as usize;
    if tokens.len() < CUT * (b - 1) + 8 {
        return Err(format!(
            "--cobatch {b} needs at least {} prompt tokens",
            CUT * (b - 1) + 8
        )
        .into());
    }
    let page = imparo_kv::page_cells();
    let span = (tokens.len() + steps).div_ceil(page);
    let positions = b * span * page;
    if positions > capacity {
        return Err(format!("--cobatch needs IMPARO_KV_CAP >= {positions}").into());
    }
    let full: Vec<u32> = model
        .kv_state_geometry()
        .iter()
        .filter(|g| matches!(g.kind, imparo_kv::StateKind::Full))
        .map(|g| g.layer)
        .collect();
    model.set_slots(rows)?;
    model.kv_fit(positions)?;
    let place = |model: &mut dyn imparo_model::Model, s: usize| -> Result<(), String> {
        model.select_slot(s as u32)?;
        let table: Vec<u32> = (0..span).map(|p| (s * span + p) as u32).collect();
        for &layer in &full {
            be.set_kv_page_table(layer, &table);
        }
        Ok(())
    };
    let prompt = |s: usize| &tokens[..tokens.len() - CUT * s];

    // Each slot alone: its step logits and the input token of every step.
    let mut alone_ms = Vec::with_capacity(b * steps);
    let mut alone: Vec<Vec<Vec<f32>>> = Vec::with_capacity(b);
    let mut trail: Vec<Vec<u32>> = Vec::with_capacity(b);
    // IMPARO_PROF=1 (a diagnostic): each slot's lone steps and the co-batched steps get their own
    // per-class GPU report; each report closes the profiler's window, so the prefills before them
    // are reported apart.
    for s in 0..b {
        place(model, s)?;
        let p = prompt(s);
        let mut lg = Vec::new();
        let t = std::time::Instant::now();
        model.forward_into(p, 0, &mut lg)?;
        imparo_model::host::prof_log(
            "cobatch gate prefill",
            t.elapsed().as_secs_f64() * 1e3,
        );
        let mut tok = host_argmax(&lg);
        let mut logs = Vec::with_capacity(steps);
        let mut toks = Vec::with_capacity(steps);
        let mut slot_ms = 0.0;
        for k in 0..steps {
            toks.push(tok);
            let t = std::time::Instant::now();
            model.forward_into(&[tok], p.len() + k, &mut lg)?;
            let dt = t.elapsed().as_secs_f64() * 1e3;
            alone_ms.push(dt);
            slot_ms += dt;
            tok = host_argmax(&lg);
            logs.push(lg.clone());
        }
        imparo_model::host::prof_log(
            &format!("cobatch gate slot {s} lone steps"),
            slot_ms,
        );
        alone.push(logs);
        trail.push(toks);
    }

    // The same conversations again, prefilled, then co-batched.
    for (s, toks) in trail.iter().enumerate() {
        place(model, s)?;
        let mut lg = Vec::new();
        model.forward_into(prompt(s), 0, &mut lg)?;
        if host_argmax(&lg) != toks[0] {
            return Err(format!(
                "slot {s}: the second prefill picked another first token"
            )
            .into());
        }
    }
    let mut diffs = 0_usize;
    let mut pick_diffs = 0_usize;
    let (mut top_delta, mut order_swaps) = (0.0_f32, 0_usize);
    let mut step_ms = Vec::with_capacity(steps);
    imparo_model::host::prof_log("cobatch gate second prefills", 0.0);
    let routes_before = be.matmul_routes();
    for k in 0..steps {
        let step: Vec<(u32, u32)> = (0..b).map(|s| (s as u32, trail[s][k])).collect();
        let mut lg = Vec::new();
        let t = std::time::Instant::now();
        let picks = model.decode_rows(&step, Some(&mut lg), route)?;
        step_ms.push(t.elapsed().as_secs_f64() * 1e3);
        let v = lg.len() / b;
        for s in 0..b {
            let got = &lg[s * v..(s + 1) * v];
            let want = &alone[s][k];
            if fast {
                let (want_top, got_top) = (top_ten(want), top_ten(got));
                for &i in &want_top {
                    top_delta = top_delta.max((got[i] - want[i]).abs());
                }
                order_swaps += want_top
                    .iter()
                    .zip(&got_top)
                    .filter(|(a, b)| a != b)
                    .count();
            } else if let Some(i) = got
                .iter()
                .zip(want)
                .position(|(g, w)| g.to_bits() != w.to_bits())
            {
                diffs += 1;
                println!(
                    "COBATCH DIFF slot={s} step={k} pos={} index={i} cobatch={} alone={}",
                    prompt(s).len() + k,
                    got[i],
                    want[i]
                );
            }
            let host = host_argmax(want);
            if picks[s] != host {
                pick_diffs += 1;
                // The lone step's own margin between its first two tokens: a flip on the fast
                // route across a gap smaller than the top-10 delta is a near tie.
                let top = top_ten(want);
                let gap = want[top[0]] - want[top.get(1).copied().unwrap_or(top[0])];
                println!(
                    "COBATCH PICK slot={s} step={k} device={} host={host} alone_top2_gap={gap:.5}",
                    picks[s]
                );
            }
        }
    }
    imparo_model::host::prof_log("cobatch gate co-batched steps", step_ms.iter().sum());
    // IMPARO_COBATCH_STEPS=1 (a diagnostic): every co-batched step's time in step order, so a
    // drift across the run (a clock that falls under sustained load) is seen, not averaged away.
    if std::env::var("IMPARO_COBATCH_STEPS").as_deref() == Ok("1") {
        let ms: Vec<String> = step_ms.iter().map(|t| format!("{t:.1}")).collect();
        println!("COBATCH STEPS ms={}", ms.join(","));
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let verdict = if diffs == 0 && pick_diffs == 0 {
        "PASS"
    } else {
        "FAIL"
    };
    // The kernels the co-batched steps ran: the matmul routes that grew, most calls first.
    let kernels = route_growth(&routes_before, &be.matmul_routes());
    println!(
        "COBATCH {verdict} route={} rows={b} steps={steps} logit_diffs={diffs} \
         pick_diffs={pick_diffs} top10_max_delta={top_delta:.5} top10_order_swaps={order_swaps} \
         median_ms_alone_step={:.2} median_ms_cobatch_step={:.2} kernels={kernels}",
        if fast { "fast" } else { "exact" },
        median(&mut alone_ms),
        median(&mut step_ms)
    );
    if kernels.contains("_fallback") {
        println!("COBATCH WARNING a matmul ran a fallback route: {kernels}");
    }
    if verdict == "FAIL" {
        return Err("co-batched rows differ from their lone decode".into());
    }
    Ok(())
}

/// `name:calls` for every matmul route whose count grew between two snapshots, most calls first;
/// `none` when nothing grew.
fn route_growth(
    before: &[(&'static str, u64)],
    after: &[(&'static str, u64)],
) -> String {
    let mut grew: Vec<(&str, u64)> = after
        .iter()
        .map(|&(name, n)| {
            let was = before
                .iter()
                .find(|(b, _)| *b == name)
                .map_or(0, |&(_, c)| c);
            (name, n - was)
        })
        .filter(|&(_, d)| d > 0)
        .collect();
    grew.sort_by_key(|&(_, calls)| std::cmp::Reverse(calls));
    if grew.is_empty() {
        return "none".into();
    }
    grew.iter()
        .map(|(name, d)| format!("{name}:{d}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn full_history_control_requested(
    arguments: &[String],
    gate_mode: bool,
) -> Result<bool, String> {
    const FLAG: &str = "--lfm-full-history-control";
    let count = arguments
        .iter()
        .filter(|argument| argument.as_str() == FLAG)
        .count();
    if count == 0 {
        return Ok(false);
    }
    if !gate_mode || count != 1 || arguments.first().map(String::as_str) != Some(FLAG) {
        return Err("--lfm-full-history-control must be the first option, occur once, and requires IMPARO_CORRECTNESS_GATE=1".into());
    }
    if arguments.iter().any(|argument| {
        matches!(
            argument.as_str(),
            "--correctness-template" | "--w4a16-ffn-check"
        )
    }) {
        return Err("full-history control cannot create a correctness template or run another fixed checker".into());
    }
    Ok(true)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let full_history_control = full_history_control_requested(
        &arguments,
        std::env::var("IMPARO_CORRECTNESS_GATE").as_deref() == Ok("1"),
    )?;
    let mut args = arguments.into_iter();
    let first = args
        .next()
        .ok_or("usage: imparo-forward MODEL.gguf TOKEN...")?;
    let first = if full_history_control {
        args.next()
            .ok_or("--lfm-full-history-control needs forward arguments")?
    } else {
        first
    };
    if first == "--w4a16-ffn-check" {
        #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
        {
            let model = PathBuf::from(
                args.next()
                    .ok_or("--w4a16-ffn-check needs MODEL INPUT_F32 LAYER")?,
            );
            let input = PathBuf::from(
                args.next()
                    .ok_or("--w4a16-ffn-check needs MODEL INPUT_F32 LAYER")?,
            );
            let layer: usize = args
                .next()
                .ok_or("--w4a16-ffn-check needs MODEL INPUT_F32 LAYER")?
                .parse()?;
            if args.next().is_some() {
                return Err(
                    "--w4a16-ffn-check accepts exactly MODEL INPUT_F32 LAYER".into()
                );
            }
            let result = w4a16_ffn_check::run(&model, &input, layer)?;
            println!("{}", serde_json::to_string(&result)?);
            return Ok(());
        }
        #[cfg(not(all(feature = "cuda-speculative", target_os = "windows")))]
        return Err("--w4a16-ffn-check requires Windows cuda-speculative".into());
    }
    if first == "--correctness-template" {
        #[cfg_attr(
            not(any(feature = "cuda", feature = "cuda-dynamic")),
            allow(unused_variables)
        )]
        let config = PathBuf::from(
            args.next()
                .ok_or("--correctness-template needs CONFIG and MODEL")?,
        );
        #[cfg_attr(
            not(any(feature = "cuda", feature = "cuda-dynamic")),
            allow(unused_variables)
        )]
        let model_path = PathBuf::from(
            args.next()
                .ok_or("--correctness-template needs CONFIG and MODEL")?,
        );
        let kv = if let Some(flag) = args.next() {
            if flag != "--kv" {
                return Err("--correctness-template optional argument is --kv".into());
            }
            let value = args
                .next()
                .ok_or("--correctness-template --kv needs a value")?;
            if args.next().is_some() {
                return Err(
                    "--correctness-template accepts only CONFIG MODEL [--kv TYPE]"
                        .into(),
                );
            }
            value
        } else {
            "q4_0".to_string()
        };
        if kv != "q4_0" && kv != "q8_0" && kv != "f16" {
            return Err(
                "--correctness-template --kv supports q4_0, q8_0 or qualified f16"
                    .into(),
            );
        }
        #[cfg(any(feature = "cuda", feature = "cuda-dynamic"))]
        {
            unsafe { std::env::set_var("IMPARO_HOST_CONFIG", &config) };
            let document = imparo_gguf::read(&model_path)?;
            let plan = build_plan(&document, &model_path)?;
            let weights = Weights::open_with(&document, &model_path)?;
            let kv_layout_sha256 =
                imparo_model::kv::effective_kv_byte_layout_profile(&plan, &kv, &kv)?
                    .sha256_identity();
            let receipt = imparo_cuda::correctness_receipt_template(
                &config,
                weights.byte_len(),
                weights.full_file_sha256(),
                *plan.sha256_identity().as_bytes(),
                kv_layout_sha256,
                &kv,
                &kv,
            )?;
            println!("{}", serde_json::to_string(&receipt)?);
            return Ok(());
        }
        #[cfg(not(any(feature = "cuda", feature = "cuda-dynamic")))]
        return Err("--correctness-template requires a CUDA-enabled build".into());
    }
    // --repeat N: run the forward N times in ONE process, printing stats+top10 per rep.
    // The det gate uses this to sample determinism without paying the ~2s Metal init per
    // sample; it still runs multiple PROCESSES too, since the visibility race was
    // observed across fresh processes. Any cross-rep state leak shows up as a
    // determinism failure, which is exactly what the gate exists to catch.
    let mut first = first;
    let score_prefix = if first == "--score-continuation" {
        if std::env::var("IMPARO_CORRECTNESS_GATE").as_deref() != Ok("1") {
            return Err(
                "--score-continuation requires IMPARO_CORRECTNESS_GATE=1".into()
            );
        }
        let prefix = args
            .next()
            .ok_or("--score-continuation needs a prefix length")?
            .parse::<usize>()?;
        first = args
            .next()
            .ok_or("--score-continuation PREFIX MODEL -t TOKENS")?;
        Some(prefix)
    } else {
        None
    };
    let mut repeat = 1_usize;
    let mut decode_n = 0_usize;
    let mut graph_decode_n = 0_usize;
    let mut graph_decode_paged = false;
    if first == "--repeat" {
        repeat = args.next().ok_or("--repeat needs a count")?.parse()?;
        first = args.next().ok_or("--repeat N MODEL.gguf TOKEN...")?;
    }
    // Dev-only same-shape Graph witness. Each repetition replaces the final token
    // with BASE+rep while retaining one model/backend process. Combined with
    // IMPARO_LOGITS_DUMP_DIR this proves that replay consumes fresh request data,
    // not merely that one repeated prompt is deterministic.
    let mut vary_last: Option<u32> = None;
    if first == "--vary-last" {
        vary_last = Some(
            args.next()
                .ok_or("--vary-last needs a base token")?
                .parse()?,
        );
        first = args.next().ok_or("--vary-last BASE MODEL.gguf TOKEN...")?;
    }
    // --split K: forward tokens[..K] at 0, then tokens[K..] at K -- the KV-pool
    // continuation instrument. The harness compares the opt-in raw final-logit dump;
    // top10 remains diagnostic output only.
    //
    // --split K1,K2,...: one rep per cut in THIS process (0 = the cold forward), so a
    // gate that compares several cuts of one prompt loads the model once instead of once
    // per cut -- on Qwen3.8-27B a load is 5 s and the resume gate ran 16 of them. Each
    // rep's logits go to IMPARO_LOGITS_DUMP_DIR as rep-NNN.raw; a rep starting at
    // position 0 resets the recurrent state and refills the cache exactly as a fresh
    // process does (the reset lives in the forward, not here).
    let mut split: Option<usize> = None;
    let mut split_series: Vec<usize> = Vec::new();
    // --decode N: after the prefill, run N greedy single-token steps and print the
    // token trail plus an FNV hash of every step's logits. With --repeat this is
    // the engine-level decode-determinism gate (task #26's reproducer).
    if first == "--decode" {
        decode_n = args.next().ok_or("--decode needs a count")?.parse()?;
        first = args.next().ok_or("--decode N MODEL.gguf TOKEN...")?;
    }
    // --dspark DRAFTER, after --decode N: the N tokens come through the DSpark provider over the
    // drafter file DRAFTER (the file IMPARO_DRAFT names otherwise): the prompt as an
    // observed prefill with a commit per chunk, then the greedy cursor, stopping at the model's
    // end-of-sequence token. One line per verified round (start, anchor, drafts, consumed), then
    // the trail. Needs IMPARO_KV_CAP for the prompt, the N tokens and a block.
    let mut dspark_draft: Option<PathBuf> = None;
    if first == "--dspark" {
        if decode_n == 0 {
            return Err("--dspark follows --decode N".into());
        }
        dspark_draft = Some(PathBuf::from(
            args.next().ok_or("--dspark needs a drafter file")?,
        ));
        first = args
            .next()
            .ok_or("--decode N --dspark DRAFTER.gguf MODEL.gguf TOKEN...")?;
    }
    // --decode-pipe N: the same N greedy steps through the PIPELINED decode API
    // (queue_step / wait_step: step k+1 is encoded and committed before step k's pick is
    // read; docs/decode-turnaround.md). The host never sees the logits, so the output is
    // the token trail and the wall time per step; the trail must equal --decode's.
    let mut decode_pipe = false;
    if first == "--decode-pipe" {
        decode_n = args.next().ok_or("--decode-pipe needs a count")?.parse()?;
        decode_pipe = true;
        first = args.next().ok_or("--decode-pipe N MODEL.gguf TOKEN...")?;
    }
    // Real CUDA-Graph probe: unlike --decode, this uses `forward_next` so device
    // argmax is active and the native backend may capture/replay. The paged form
    // installs a non-identity table before prefill, so captured attention nodes use
    // the dedicated paged symbols rather than merely validating identity graphs.
    if first == "--graph-decode" || first == "--paged-graph-decode" {
        graph_decode_paged = first == "--paged-graph-decode";
        graph_decode_n = args.next().ok_or("--graph-decode needs a count")?.parse()?;
        first = args.next().ok_or("--graph-decode N MODEL.gguf TOKEN...")?;
    }
    // --dbatch B: feed B rows per decode forward instead of 1. NOT a correctness path
    // -- the rows carry the same token at consecutive positions, so the logits are
    // meaningless -- it exists to price the SHAPE. A co-batched decode (two
    // conversations in one forward) reads the weights once for B rows, and what that
    // is worth is exactly `B x t(1) - t(B)`.
    let mut dbatch = 1_usize;
    if first == "--dbatch" {
        dbatch = args.next().ok_or("--dbatch needs a count")?.parse()?;
        first = args.next().ok_or("--dbatch B MODEL.gguf TOKEN...")?;
    }
    // --cobatch B, after --decode N: the co-batched decode gate on the exact route
    // (docs/continuous-batching.md, section 6). B conversations -- the prompt cut to B
    // lengths, each in its own slot and block range -- decode N greedy steps each alone on
    // the dispatch path, then together, one forward per step; every row's logits must be
    // bit-equal to its lone step's. Needs IMPARO_MEGA_FFN=0 and IMPARO_KV_CAP for B slots.
    let mut cobatch = 0_u32;
    let mut cobatch_fast = false;
    if first == "--cobatch" {
        if decode_n == 0 {
            return Err("--cobatch follows --decode N".into());
        }
        cobatch = args.next().ok_or("--cobatch needs a row count")?.parse()?;
        first = args
            .next()
            .ok_or("--decode N --cobatch B [--fast] MODEL.gguf TOKEN...")?;
        if first == "--fast" {
            cobatch_fast = true;
            first = args
                .next()
                .ok_or("--decode N --cobatch B --fast MODEL.gguf TOKEN...")?;
        }
    }
    if first == "--split" {
        let spec = args.next().ok_or("--split needs a position")?;
        let cuts = spec
            .split(',')
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()?;
        if cuts.len() > 1 {
            if repeat != 1 {
                return Err(
                    "--split K1,K2,... sets the rep count itself; drop --repeat".into(),
                );
            }
            repeat = cuts.len();
            split_series = cuts;
        } else {
            split = cuts.first().copied();
        }
        first = args
            .next()
            .ok_or("--split K[,K2,...] MODEL.gguf TOKEN...")?;
    }
    // --spill DIR / --restore DIR: the disk-tier instrument. --spill runs the full
    // forward, captures the KV state at the unit boundary and commits it to the
    // store; --restore (a FRESH process: the durability proof) probes the store
    // with the request's own hashes, restores, and forwards only the tail.
    let mut spill_dir: Option<PathBuf> = None;
    let mut restore_dir: Option<PathBuf> = None;
    if first == "--spill" {
        spill_dir = Some(PathBuf::from(args.next().ok_or("--spill needs a dir")?));
        first = args.next().ok_or("--spill DIR MODEL.gguf TOKEN...")?;
    }
    if first == "--restore" {
        restore_dir = Some(PathBuf::from(args.next().ok_or("--restore needs a dir")?));
        first = args.next().ok_or("--restore DIR MODEL.gguf TOKEN...")?;
    }
    // --scramble-kv: forward twice in one process -- identity placement, then
    // pair-swapped block tables -- and compare top10 internally. Proves physical
    // KV placement is invisible to the output (the isolation invariant).
    let mut scramble = false;
    if first == "--scramble-kv" {
        scramble = true;
        first = args.next().ok_or("--scramble-kv MODEL.gguf TOKEN...")?;
    }
    // --verify-chain B,N,M: the greedy block verifier against one-token decoding.
    // Repetition 0 decodes N + M tokens one at a time. Repetition 1 prefills again and
    // emits N tokens through verifications of B rows (an anchor and the next B - 1 tokens
    // of repetition 0's trail), each with one input replaced at a depth that cycles every
    // round, then decodes M tokens one at a time from the verified state. Every token must
    // match repetition 0. IMPARO_VERIFY_LOG=1 names the path each verification took.
    // Repetitions 2 and 3 compare logits on the first N tokens of that trail: 2 forwards them
    // one row at a time, 3 in B-row batches with every row's logits. Each row must meet the
    // verify-row rule against repetition 2 (`RowLogits`): the same top-1, and deltas to the
    // top-1 within 2e-2; an order swap below rank 1 is printed, not failed.
    // Repetitions 4 to 6 run `continue_greedy` with a scripted provider that proposes the trail,
    // corrupted at a depth that cycles every round: 4 with no stop token, where the output must be
    // the trail; 5 with a stop token the target never picks forced into every proposal, first on
    // even rounds and second on odd rounds, where the output must still be the trail; 6 with the
    // last trail token that does not repeat an earlier one declared a stop, where the output must
    // end just before it. In every case each draft must be verified, except one that begins with
    // a stop token.
    let mut verify_chain: Option<(usize, usize, usize)> = None;
    if first == "--verify-chain" {
        let spec = args.next().ok_or("--verify-chain needs B,N,M")?;
        let counts = spec
            .split(',')
            .map(str::parse::<usize>)
            .collect::<Result<Vec<_>, _>>()?;
        let [rows, emit, post] = counts[..] else {
            return Err("--verify-chain B,N,M takes three counts".into());
        };
        if rows < 2 || emit == 0 || post == 0 {
            return Err("--verify-chain needs B >= 2, N >= 1 and M >= 1".into());
        }
        if repeat != 1
            || decode_n > 0
            || graph_decode_n > 0
            || dbatch != 1
            || split.is_some()
            || !split_series.is_empty()
        {
            return Err(
                "--verify-chain runs alone and sets the rep count itself".into()
            );
        }
        repeat = 7;
        verify_chain = Some((rows, emit, post));
        first = args
            .next()
            .ok_or("--verify-chain B,N,M MODEL.gguf TOKEN...")?;
    }
    // `--verify-tree` checks the tree forward's logits on TREE_PARENTS, built from repetition 0's
    // one-token trail:
    //   repetition 1: the trail's first 9 tokens as a chain row layout against the causal 9-row
    //                 forward, every logit bit for bit with the verify's key split off; with it
    //                 on, a measurement line
    //   repetition 2: the tree as one row-layout forward, every node's logits kept
    //   repetitions 3 to 5: each leaf's own path as a chain row layout, each node on it held to
    //                 its tree row -- bit for bit where the node's ancestors are rows 0..depth,
    //                 the verify-row rule (`RowLogits`) for every node
    //   repetition 6: one verify of TREE_CASES' branch_at_3 tree at the prompt's end, then
    //                 TREE_POST steps of one-token decoding from the committed state
    //   repetition 7: the same accepted tokens as a chain verify; its KV rows and recurrent state
    //                 against repetition 6's, byte for byte (a measurement, no verdict)
    //   repetition 8: the cursor over TREE_EMIT tokens with a scripted tree provider cycling
    //                 TREE_CASES; the tokens must equal one-token decoding
    //   repetition 9: the 64-node wide tree's forward, each node on WIDE_PATH against a causal
    //                 forward over the same tokens (the verify-row rule)
    //   repetition 10: one verify of the wide tree, then TREE_POST steps of one-token decoding
    //   repetition 11: the same accepted tokens as a chain verify, byte for byte against
    //                 repetition 10's commit (a measurement, no verdict)
    // --decode-vs-verify R: is a verify row one-token decode, bit for bit?
    //   repetition 0: the prompt, then R tokens decoded one at a time (`forward_next`): the trail
    //   repetition 1: the prompt, then the trail's first R tokens forwarded ONE ROW AT A TIME,
    //                 every row's logits kept (a one-token forward with a logits demand)
    //   repetition 2: the prompt, then the same R tokens as ONE chain row layout on the rows
    //                 route a speculative round takes (fast; IMPARO_DVV_ROUTE=exact for the
    //                 exact one): each row's logits against repetition 1's, bit for bit, and its
    //                 argmax against the trail
    let mut decode_vs_verify: Option<usize> = None;
    if first == "--decode-vs-verify" {
        let rows: usize = args
            .next()
            .ok_or("--decode-vs-verify needs a row count")?
            .parse()
            .map_err(|_| "--decode-vs-verify: the row count is not a number")?;
        if !(2..=64).contains(&rows) {
            return Err("--decode-vs-verify takes 2..64 rows".into());
        }
        if repeat != 1 || decode_n > 0 || verify_chain.is_some() {
            return Err("--decode-vs-verify runs alone and sets the rep count itself".into());
        }
        repeat = 3;
        decode_vs_verify = Some(rows);
        first = args
            .next()
            .ok_or("--decode-vs-verify R MODEL.gguf TOKEN...")?;
    }
    let mut verify_tree = false;
    if first == "--verify-tree" {
        if repeat != 1
            || decode_n > 0
            || graph_decode_n > 0
            || dbatch != 1
            || split.is_some()
            || !split_series.is_empty()
            || verify_chain.is_some()
        {
            return Err("--verify-tree runs alone and sets the rep count itself".into());
        }
        repeat = 9 + TREE_LEAVES.len();
        verify_tree = true;
        first = args.next().ok_or("--verify-tree MODEL.gguf TOKEN...")?;
    }
    if dspark_draft.is_some() {
        if cfg!(not(feature = "speculative")) {
            return Err("--dspark needs a build with the speculative feature".into());
        }
        if repeat != 1
            || dbatch != 1
            || decode_pipe
            || graph_decode_n > 0
            || split.is_some()
            || !split_series.is_empty()
            || verify_chain.is_some()
            || verify_tree
        {
            return Err("--dspark runs alone after --decode N".into());
        }
    }
    // Dev-only exact-128 mask sweep: one model load, one mask per repeat.
    // This accelerates numerical-route search without changing backend ABI or
    // production policy; ordinary runs never set the environment variable.
    let exact128_mask_sweep: Option<Vec<String>> =
        std::env::var("IMPARO_EXACT128_MASK_SWEEP")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
                    .map(str::to_owned)
                    .collect()
            });
    if let Some(masks) = &exact128_mask_sweep {
        if masks.is_empty() {
            return Err("IMPARO_EXACT128_MASK_SWEEP contains no masks".into());
        }
        if !split_series.is_empty() {
            return Err("IMPARO_EXACT128_MASK_SWEEP and --split K1,K2,... both set the rep count".into());
        }
        repeat = masks.len();
    }
    let path = PathBuf::from(first);
    // Token ids as args, or -t FILE (whitespace-separated) -- long contexts
    // exceed the OS argv limit around 90k tokens.
    let mut rest: Vec<String> = args.collect();
    let tokens: Vec<u32> = if rest.first().map(String::as_str) == Some("-t") {
        let tf = rest.get(1).ok_or("-t needs a file")?;
        std::fs::read_to_string(tf)?
            .split_whitespace()
            .map(str::parse)
            .collect::<Result<_, _>>()?
    } else {
        rest.drain(..)
            .map(|a| a.parse::<u32>())
            .collect::<Result<_, _>>()?
    };
    if tokens.is_empty() {
        return Err("give at least one token id".into());
    }
    if let Some(prefix) = score_prefix {
        if prefix == 0
            || prefix >= tokens.len()
            || repeat != 1
            || decode_n != 0
            || graph_decode_n != 0
            || dbatch != 1
            || split.is_some()
            || !split_series.is_empty()
            || spill_dir.is_some()
            || restore_dir.is_some()
            || scramble
            || vary_last.is_some()
            || full_history_control
            || exact128_mask_sweep.is_some()
        {
            return Err("scoring requires a nonempty continuation and cannot combine with other probes".into());
        }
    }
    // Phase-A1-only denominator instrument. Exact value "1" is required; when
    // absent (the normal CLI), this changes no output, route, math, or model state.
    let phase_a1_prefill_wall =
        std::env::var("IMPARO_CUDA_PHASE_A1_PREFILL_WALL").as_deref() == Ok("1");

    let t0 = std::time::Instant::now();
    let document = imparo_gguf::read(&path)?;
    let plan = build_plan(&document, &path)?;
    if full_history_control && plan.config.architecture != "lfm2" {
        return Err("full-history control requires the retained LFM model".into());
    }
    // IMPARO_DRAFT=DRAFTER or --dspark DRAFTER: map the drafter file right after MODEL, in the
    // same range, and place the drafter's tensors.
    #[cfg(feature = "speculative")]
    let mut draft_pairing = match dspark_draft
        .clone()
        .or_else(|| std::env::var_os("IMPARO_DRAFT").map(PathBuf::from))
    {
        Some(draft) => Some(load_drafter(&draft)?),
        None => None,
    };
    #[cfg(feature = "speculative")]
    let mut weights = match &draft_pairing {
        Some((draft, _)) => Weights::open_with_appended(&document, &path, draft)?,
        None => Weights::open(&path)?,
    };
    #[cfg(not(feature = "speculative"))]
    let mut weights = Weights::open(&path)?;
    // IMPARO_KV_CAP: force the KV capacity (test instrument). The server sizes
    // capacity from -c, not the prompt; this reproduces that shape here. Known before
    // the GPU is enabled because the weight placement reserves the KV for it.
    let capacity = std::env::var("IMPARO_KV_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| tokens.len().max(8));
    if score_prefix.is_some()
        && (capacity < tokens.len()
            || tokens.iter().any(|&t| t >= plan.config.vocab_size))
    {
        return Err(
            "score tokens exceed model vocabulary or configured capacity".into(),
        );
    }
    #[cfg(feature = "speculative")]
    let draft_descriptor = match &draft_pairing {
        Some((_, pairing)) => Some(pairing.descriptor(&weights, plan.config.n_layers)?),
        None => None,
    };
    #[cfg(feature = "speculative")]
    let appended = match &draft_descriptor {
        Some(descriptor) => {
            let spans = descriptor.appended_spans();
            eprintln!(
                "[imparo] draft pairing: {} drafter tensors, {} bytes placed past the target",
                spans.len(),
                spans.iter().map(|s| s.1).sum::<u64>()
            );
            spans
        }
        None => Vec::new(),
    };
    #[cfg(not(feature = "speculative"))]
    let appended: Vec<(u64, u64)> = Vec::new();
    // The drafter's caches, its attention dims and its feature rows are sized from the plan,
    // so the plan carries the drafter before the device is enabled.
    #[cfg(feature = "speculative")]
    let plan = imparo_model::ModelPlan {
        drafter: draft_descriptor
            .as_ref()
            .map(imparo_model::dspark::DsparkDescriptor::drafter_plan),
        ..plan
    };
    imparo_model::backend::enable_gpu_with_appended(
        &mut weights,
        &plan,
        capacity,
        &appended,
    )?;
    let tensor_count = weights.tensors.len();
    let layer_count = plan.config.n_layers;
    let vocab_size = plan.config.vocab_size;

    let mut model = imparo_model::load(weights, plan, capacity)?;
    // BOTH questions, because they are different ones: a backend is active whenever one
    // is compiled in (and IMPARO_BACKEND=cpu makes the CPU backend the active one), while
    // `has_device_workflow` says whether THIS architecture has a device forward to make
    // ready. Asking only the first sent a CPU-only architecture into a refusal.
    if model.has_device_workflow() && imparo_model::backend::active().is_some() {
        model.ensure_gpu_ready()?;
    }
    if full_history_control {
        let domain =
            imparo_model::lfm2::workflow_gpu::enable_lfm_full_history_control()?;
        eprintln!(
            "[lfm-full-history-control] version=1 domain={domain} finite_history=64 crop=disabled production_authority=false"
        );
    }
    // Use the server's exact opt-in laboratory selection for isolated diagnostics.
    // This does not issue a correctness receipt or alter normal production defaults.
    imparo_model::backend::apply_lab_knobs_from_env()?;
    if cobatch > 0 {
        return cobatch_gate(
            &mut *model,
            &tokens,
            cobatch,
            decode_n,
            capacity,
            cobatch_fast,
        );
    }
    #[cfg(feature = "speculative")]
    if dspark_draft.is_some() {
        let (_, pairing) = draft_pairing
            .take()
            .ok_or("--dspark: the pairing did not load")?;
        let stops: Vec<u32> = document
            .unsigned_value("tokenizer.ggml.eos_token_id")
            .and_then(|t| u32::try_from(t).ok())
            .into_iter()
            .collect();
        return dspark_decode(&mut *model, pairing, &tokens, decode_n, &stops);
    }
    // With a pairing, the drafter's feature taps ride every forward when any of these is set:
    //   IMPARO_DRAFT_TAP_PROBE=1         each forward reports whether the feature rows equal
    //                                    the tapped layers' outputs
    //   IMPARO_DSPARK_REPLAY=FILE        a speculative run's rounds replayed through the
    //                                    drafter (dspark_replay_rounds); nothing else runs
    //   --verify-chain / --verify-tree   the drafter attached as a provider holds it: the taps
    //                                    capturing and the activation rows at its verify width
    #[cfg(feature = "speculative")]
    let dspark_replay = std::env::var_os("IMPARO_DSPARK_REPLAY");
    #[cfg(feature = "speculative")]
    let verify_with_drafter = verify_chain.is_some() || verify_tree;
    #[cfg(feature = "speculative")]
    let feature_taps = match &draft_descriptor {
        Some(descriptor)
            if dspark_replay.is_some()
                || verify_with_drafter
                || std::env::var("IMPARO_DRAFT_TAP_PROBE").as_deref() == Ok("1") =>
        {
            let rows = u32::try_from(imparo_model::prefill_batch())
                .map_err(|_| "prefill batch too large")?;
            Some(imparo_model::dspark::FeatureTaps::attach(
                &mut *model,
                descriptor,
                rows,
            )?)
        }
        _ => None,
    };
    #[cfg(feature = "speculative")]
    let _drafter = match &draft_descriptor {
        Some(descriptor) if verify_with_drafter => {
            let drafter = imparo_model::dspark_forward::DsparkForward::attach(
                &mut *model,
                descriptor,
            )?;
            eprintln!(
                "[imparo] drafter attached: feature taps capturing, a block of {} rows",
                drafter.block_size()
            );
            Some(drafter)
        }
        _ => None,
    };
    #[cfg(feature = "speculative")]
    if let Some(rounds) = &dspark_replay {
        let descriptor = draft_descriptor
            .as_ref()
            .ok_or("IMPARO_DSPARK_REPLAY needs IMPARO_DRAFT")?;
        let taps = feature_taps
            .as_ref()
            .ok_or("IMPARO_DSPARK_REPLAY needs the feature taps")?;
        return dspark_replay_rounds(
            &mut *model,
            descriptor,
            taps,
            &tokens,
            std::path::Path::new(rounds),
        );
    }
    let load_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "load  ms={load_ms:.1} tensors={tensor_count} layers={layer_count} vocab={vocab_size}"
    );
    if let Some(prefix) = score_prefix {
        use sha2::{Digest, Sha256};
        use std::fmt::Write as _;
        let rows = continuation_score::score(&tokens, prefix, |ids, pos| {
            model.forward(ids, pos)
        })?;
        let sum: f64 = rows.iter().map(|row| row.nll).sum();
        let token_bytes: Vec<u8> =
            tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        let report = serde_json::json!({
            "schema": 1, "kind": "teacher_forced_continuation", "formal_admission": false,
            "prefix_tokens": prefix, "scored_tokens": rows.len(), "vocab_size": vocab_size,
            "tokens_u32le_sha256": Sha256::digest(&token_bytes).iter().fold(String::new(), |mut hex, b| {
                let _ = write!(hex, "{b:02x}");
                hex
            }),
            "nll_sum": sum, "nll_mean": sum / rows.len() as f64,
            "rows": rows.iter().map(|r| serde_json::json!({"position": r.position, "token": r.token, "nll": r.nll, "argmax": r.argmax})).collect::<Vec<_>>()
        });
        println!("continuation-score {report}");
        return Ok(());
    }
    if graph_decode_paged {
        model.kv_prepare_pool()?;
        model.kv_set_scrambled_tables(false);
    }
    // IMPARO_TREE_SCRAMBLE=1 with --verify-tree: every repetition writes and reads the cache through
    // pair-swapped block tables, so a tree commit's row move maps through a table that is not the
    // identity. The byte comparisons read physical rows, which that placement scatters, so they are
    // skipped; every other verify-tree line must equal a run with identity tables.
    let tree_scramble =
        verify_tree && std::env::var("IMPARO_TREE_SCRAMBLE").as_deref() == Ok("1");
    if tree_scramble {
        model.kv_prepare_pool()?;
        model.kv_set_scrambled_tables(false);
    }

    let logits_dump_dir = std::env::var_os("IMPARO_LOGITS_DUMP_DIR").map(PathBuf::from);
    if let Some(dir) = &logits_dump_dir {
        std::fs::create_dir_all(dir)?;
    }
    let mut logits = Vec::new();
    let mut chain_reference: Option<Vec<u32>> = None;
    let mut chain_one_row: Option<Vec<f32>> = None;
    let mut tree_reference: Option<Vec<u32>> = None;
    let mut dvv_trail: Option<Vec<u32>> = None;
    let mut dvv_one_row: Option<Vec<f32>> = None;
    let mut tree_logits: Option<Vec<f32>> = None;
    let mut tree_checks: Vec<Option<TreeNodeCheck>> = vec![None; TREE_PARENTS.len()];
    let mut tree_commit_state: Option<imparo_kv::KvState> = None;
    let mut wide_commit_state: Option<imparo_kv::KvState> = None;
    for rep in 0..repeat {
        let mut toks_rep = tokens.clone();
        if let Some(base) = vary_last {
            let replacement = base
                .checked_add(u32::try_from(rep)?)
                .ok_or("--vary-last token overflow")?;
            *toks_rep.last_mut().expect("non-empty tokens checked above") = replacement;
        }
        let split = if split_series.is_empty() {
            split
        } else {
            Some(split_series[rep])
        };
        // A scramble rep after the first starts on the tables the previous rep left
        // pair-swapped; its identity leg needs identity placement back.
        if scramble && rep > 0 {
            model.kv_set_scrambled_tables(true);
        }
        let t1 = std::time::Instant::now();
        let mut wall_token_count = toks_rep.len();
        let mut wall_start_pos = 0_usize;
        let mut wall_split_mode = "single";
        let mut wall_split_at = None;
        let mut wall_split_parts: Option<Vec<usize>> = None;
        if let Some(dir) = &restore_dir {
            // Restart-durability path: nothing of this conversation has run in this
            // process. The request's tokens are the whole key.
            let digest = imparo_model::kv::model_digest(&path)?;
            let root = imparo_model::kv::config_root(
                model.plan(),
                &digest,
                &imparo_model::backend::active().map_or_else(
                    || "none".to_string(),
                    imparo_backend::Backend::device_tag,
                ),
            );
            let store = imparo_kv::Store::open(dir, &root).map_err(|e| e.clone())?;
            let state =
                store.read_whole(&root, &model.kv_state_geometry(), &toks_rep)?;
            let boundary = state.boundary;
            model.kv_restore(&state)?;
            println!("restored boundary={boundary}");
            wall_token_count = toks_rep.len() - boundary;
            wall_start_pos = boundary;
            wall_split_mode = "restore-tail";
            wall_split_at = Some(boundary);
            if phase_a1_prefill_wall {
                wall_split_parts = Some(vec![wall_token_count]);
            }
            logits = model.forward(&toks_rep[boundary..], boundary)?;
        } else {
            logits = match split {
                Some(k) if k > 0 && k < toks_rep.len() => {
                    wall_split_mode = "explicit-split";
                    wall_split_at = Some(k);
                    if phase_a1_prefill_wall {
                        wall_split_parts = Some(vec![k, toks_rep.len() - k]);
                    }
                    let _ = model.forward(&toks_rep[..k], 0)?;
                    model.forward(&toks_rep[k..], k)?
                }
                // --spill: forward to the BOUNDARY, checkpoint there, then forward
                // the tail. A recurrent state is one buffer holding "now", so a
                // checkpoint is only valid at the position the device is at -- which is
                // why a server pauses at its branch point rather than spilling after the
                // fact. Position-indexed KV would not care; this makes both correct.
                _ if spill_dir.is_some() => {
                    let b = toks_rep.len() / imparo_kv::grid_tokens()
                        * imparo_kv::grid_tokens();
                    wall_split_mode = "spill-boundary";
                    wall_split_at = Some(b);
                    if phase_a1_prefill_wall {
                        wall_split_parts = Some(if b == toks_rep.len() {
                            vec![b]
                        } else {
                            vec![b, toks_rep.len() - b]
                        });
                    }
                    if b > 0 {
                        model.forward(&toks_rep[..b], 0)?;
                    }
                    spill_now(
                        model.as_mut(),
                        spill_dir.as_ref().expect("checked"),
                        &path,
                        &toks_rep,
                    )?;
                    model.forward(&toks_rep[b..], b)?
                }
                _ => {
                    let splits = std::env::var("IMPARO_PREFILL_SPLITS")
                        .ok()
                        .map(|raw| {
                            raw.split(',')
                                .map(str::parse::<usize>)
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()?;
                    let chunk = std::env::var("IMPARO_PREFILL_CHUNK")
                        .ok()
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|&value| value > 0);
                    if let Some(splits) = splits {
                        if splits.iter().sum::<usize>() != toks_rep.len()
                            || splits.contains(&0)
                        {
                            return Err("IMPARO_PREFILL_SPLITS must be positive and sum to the prompt length".into());
                        }
                        wall_split_mode = "env-splits";
                        if phase_a1_prefill_wall {
                            wall_split_parts = Some(splits.clone());
                        }
                        model.ensure_gpu_ready()?;
                        model.kv_fit(toks_rep.len())?;
                        let mut last = Vec::new();
                        let mut start = 0;
                        for count in splits {
                            last = model
                                .forward(&toks_rep[start..start + count], start)?;
                            start += count;
                        }
                        last
                    } else if let Some(chunk) = chunk {
                        wall_split_mode = "env-chunk";
                        if phase_a1_prefill_wall {
                            wall_split_parts = Some(
                                toks_rep.chunks(chunk).map(<[u32]>::len).collect(),
                            );
                        }
                        let mut last = Vec::new();
                        for (index, part) in toks_rep.chunks(chunk).enumerate() {
                            last = model.forward(part, index * chunk)?;
                        }
                        last
                    } else {
                        model.forward(&toks_rep, 0)?
                    }
                }
            };
        }
        let prefill_wall = t1.elapsed();
        imparo_model::host::prof_log("prefill", prefill_wall.as_secs_f64() * 1e3);
        if let Some(line) = phase_a1_prefill_wall_line(
            phase_a1_prefill_wall,
            rep,
            wall_token_count,
            wall_start_pos,
            wall_split_mode,
            wall_split_at,
            wall_split_parts.as_deref(),
            prefill_wall,
        )? {
            println!("{line}");
        }
        dump_recurrent_witness(model.as_ref(), rep)?;
        if scramble {
            // Both legs run in this process, so each needs its own machine-readable
            // full-distribution witness for the fail-closed KV placement gate.
            dump_logits_for_env(&logits, "IMPARO_SCRAMBLE_IDENTITY_DUMP")?;

            let mut a = String::new();
            top10_string(&logits, &mut a);
            // Store-side witness on the FIRST full-attention layer -- the one kind whose
            // placement the scramble moves. Which layer that is belongs to the
            // architecture (gemma4: 5, qwen35: 3); asking for a fixed number panicked on
            // the second architecture that ran this probe.
            let witness = std::env::var("IMPARO_SCRAMBLE_DUMP")
                .ok()
                .map(|_| {
                    model
                        .kv_state_geometry()
                        .into_iter()
                        .find(|g| matches!(g.kind, imparo_kv::StateKind::Full))
                        .ok_or(
                            "scramble witness: the model has no full-attention layer",
                        )
                })
                .transpose()?;
            if let Some(geom) = &witness {
                print_k_rows(geom, false);
            }
            model.kv_set_scrambled_tables(false);
            // With --split K, the scrambled run RESUMES at K instead of running cold:
            // paged placement AND a resume, which is the cell neither half covered.
            // Identity+cold vs identity+resume is the split gate; identity+cold vs
            // paged+cold is this gate's own row; paged+RESUME is where a quantized
            // cache was found to diverge.
            let l2 = if let Some(k) = split {
                model.forward(&toks_rep[..k], 0)?;
                model.forward(&toks_rep[k..], k)?
            } else {
                model.forward(&toks_rep, 0)?
            };
            let mut b = String::new();
            top10_string(&l2, &mut b);
            dump_logits_for_env(&l2, "IMPARO_SCRAMBLE_PAGED_DUMP")?;
            if let Some(dir) = &logits_dump_dir {
                dump_logits_to_path(&l2, dir.join(format!("rep-{rep:03}.paged.raw")))?;
            }
            println!("identity {a}");
            println!("scrambled{b}");
            println!(
                "scramble-kv: {}",
                if a == b { "BYTE-EQUAL" } else { "MISMATCH" }
            );
            // Store-side verification: position p's K row must sit at the
            // pair-swapped physical slot.
            if let Some(geom) = &witness {
                print_k_rows(geom, true);
            }
        }
        if let Some(rows) = decode_vs_verify {
            let base = toks_rep.len();
            if base + rows + 1 > model.kv_runtime().capacity {
                return Err("--decode-vs-verify: KV capacity too small: set IMPARO_KV_CAP".into());
            }
            let vocab = vocab_size as usize;
            let first_token = imparo_cpu::ops::argmax_f32(&logits);
            if rep == 0 {
                let mut trail = vec![first_token];
                let mut next = first_token;
                for i in 0..rows {
                    next = model.forward_next(next, base + i)?;
                    trail.push(next);
                }
                println!("dvv trail rows={rows} trail={trail:?}");
                dvv_trail = Some(trail);
                continue;
            }
            let trail = dvv_trail.as_ref().ok_or("--decode-vs-verify: no trail")?;
            if trail[0] != first_token {
                return Err("--decode-vs-verify: this prefill's first token differs".into());
            }
            if rep == 1 {
                let mut all = Vec::with_capacity(rows * vocab);
                let mut row = Vec::new();
                for (i, &tok) in trail[..rows].iter().enumerate() {
                    model.forward_all_logits_into(&[tok], base + i, &mut row)?;
                    if row.len() != vocab {
                        return Err(format!("--decode-vs-verify: {} one-row logits", row.len()).into());
                    }
                    all.extend_from_slice(&row);
                }
                let argmax_equal = all
                    .chunks(vocab)
                    .zip(&trail[1..])
                    .filter(|(r, t)| imparo_cpu::ops::argmax_f32(r) == **t)
                    .count();
                println!("dvv one_row rows={rows} argmax_equal_trail={argmax_equal}/{rows}");
                dvv_one_row = Some(all);
                continue;
            }
            let one_row = dvv_one_row.as_ref().ok_or("--decode-vs-verify: no one-row logits")?;
            let exact = std::env::var("IMPARO_DVV_ROUTE").as_deref() == Ok("exact");
            let route = if exact {
                imparo_backend::RowRoute::Exact
            } else {
                imparo_backend::RowRoute::Fast
            };
            let chain = imparo_model::speculative::DraftTree::plain(
                trail[..rows].to_vec(),
                (-1..i32::try_from(rows).unwrap_or(i32::MAX) - 1).collect(),
            );
            let be = imparo_model::backend::active();
            if let Some(be) = be {
                be.set_decode_rows(Some(route));
            }
            let mut verify = Vec::new();
            let ran = model.forward_tree_logits(&chain, base, &mut verify);
            if let Some(be) = be {
                be.set_decode_rows(None);
            }
            ran?;
            if verify.len() != rows * vocab {
                return Err(format!("--decode-vs-verify: {} verify logits", verify.len()).into());
            }
            let mut bit_equal = 0_usize;
            let mut argmax_trail = 0_usize;
            let mut worst = 0.0_f32;
            for (i, (v, d)) in verify.chunks(vocab).zip(one_row.chunks(vocab)).enumerate() {
                let equal = v.iter().zip(d).all(|(a, b)| a.to_bits() == b.to_bits());
                bit_equal += usize::from(equal);
                argmax_trail += usize::from(imparo_cpu::ops::argmax_f32(v) == trail[i + 1]);
                let row_max = v.iter().zip(d).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
                worst = worst.max(row_max);
                let differing = v.iter().zip(d).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                println!(
                    "dvv row={i} pos={} bit_equal={equal} differing_logits={differing} max_abs={row_max:.6}",
                    base + i
                );
            }
            println!(
                "dvv verify route={route:?} rows={rows} bit_equal_rows={bit_equal}/{rows} argmax_equal_trail={argmax_trail}/{rows} max_abs={worst:.6} verdict={}",
                if bit_equal == rows { "PASS" } else { "FAIL" }
            );
            // ROW-COUNT INVARIANCE: the first r rows verified as their own chain must be the
            // R-row verify's first r rows, bit for bit, for every r -- a verify's width never
            // changes a row it shares with a narrower one.
            let mut unequal = Vec::new();
            for r in 2..rows {
                let prefix = imparo_model::speculative::DraftTree::plain(
                    trail[..r].to_vec(),
                    (-1..i32::try_from(r).unwrap_or(i32::MAX) - 1).collect(),
                );
                if let Some(be) = be {
                    be.set_decode_rows(Some(route));
                }
                let mut narrow = Vec::new();
                let ran = model.forward_tree_logits(&prefix, base, &mut narrow);
                if let Some(be) = be {
                    be.set_decode_rows(None);
                }
                ran?;
                let equal = narrow
                    .iter()
                    .zip(&verify[..r * vocab])
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                if !equal {
                    unequal.push(r);
                }
            }
            println!(
                "dvv width_invariance rows=2..{} unequal_widths={unequal:?} verdict={}",
                rows - 1,
                if unequal.is_empty() { "PASS" } else { "FAIL" }
            );
            continue;
        }
        if verify_tree {
            let base = toks_rep.len();
            let nodes = TREE_PARENTS.len();
            let mut depth = vec![0_usize; nodes];
            for (i, &parent) in TREE_PARENTS.iter().enumerate().skip(1) {
                depth[i] = depth[parent as usize] + 1;
            }
            let trail_len = TREE_EMIT + depth.iter().copied().max().unwrap_or(0) + 1;
            let needed = base + trail_len + nodes;
            let capacity = model.kv_runtime().capacity;
            if needed > capacity {
                return Err(format!(
                    "--verify-tree needs KV capacity {needed}, have {capacity}: set IMPARO_KV_CAP"
                )
                .into());
            }
            let first_token = imparo_cpu::ops::argmax_f32(&logits);
            if rep == 0 {
                let mut trail = vec![first_token];
                let mut next = first_token;
                for i in 0..trail_len {
                    next = model.forward_next(next, base + i)?;
                    trail.push(next);
                }
                println!(
                    "verify-tree reference tokens={} trail={trail:?}",
                    trail.len()
                );
                tree_reference = Some(trail);
                continue;
            }
            let reference = tree_reference
                .as_ref()
                .ok_or("--verify-tree: the reference repetition did not run")?;
            if reference[0] != first_token {
                return Err(
                    "--verify-tree: this prefill and repetition 0's disagree on the first token"
                        .into(),
                );
            }
            let vocab = vocab_size as usize;
            // Node i takes the trail's token at its depth; a node whose parent already has an
            // earlier child takes the next id, a token the trail did not take there.
            let tokens: Vec<u32> = (0..nodes)
                .map(|i| {
                    let later_sibling =
                        (1..i).any(|j| TREE_PARENTS[j] == TREE_PARENTS[i]);
                    if later_sibling {
                        (reference[depth[i]] + 1) % vocab_size
                    } else {
                        reference[depth[i]]
                    }
                })
                .collect();
            if rep == 1 {
                // A chain as a row layout sees exactly what causality sees, so every logit of
                // the layout forward must equal the causal forward's -- with the verify's key
                // split off, the one part of a row layout's grid that reassociates the softmax.
                // With it on (what a verify runs), the same rows are a measurement line.
                let chain = &reference[..9];
                let tree = imparo_model::speculative::DraftTree::plain(
                    chain.to_vec(),
                    (-1..8).collect(),
                );
                let be = imparo_model::backend::active();
                let mut split_rows = Vec::new();
                model.forward_tree_logits(&tree, base, &mut split_rows)?;
                if let Some(be) = be {
                    be.set_verify_split(false);
                }
                let mut layout_rows = Vec::new();
                let unsplit = model.forward_tree_logits(&tree, base, &mut layout_rows);
                if let Some(be) = be {
                    be.set_verify_split(true);
                }
                unsplit?;
                let mut causal = Vec::new();
                model.forward_all_logits_into(chain, base, &mut causal)?;
                let split_bit_equal = split_rows
                    .chunks(vocab)
                    .zip(causal.chunks(vocab))
                    .filter(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()))
                    .count();
                let split_delta = split_rows
                    .chunks(vocab)
                    .zip(causal.chunks(vocab))
                    .map(|(row, reference)| RowLogits::compare(reference, row))
                    .fold((0.0_f32, 0_usize), |(worst, top1), check| {
                        (worst.max(check.delta), top1 + usize::from(!check.top1_equal()))
                    });
                let split_max_abs = split_rows
                    .iter()
                    .zip(&causal)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                println!(
                    "verify-tree chain_layout_split rows=9 bit_equal_rows={split_bit_equal} top1_differs={} delta={:.6} max_abs={split_max_abs:.6} (a measurement, no verdict)",
                    split_delta.1, split_delta.0
                );
                if layout_rows.len() != 9 * vocab || causal.len() != 9 * vocab {
                    return Err(format!(
                        "--verify-tree: layout {} and causal {} logits for 9 rows",
                        layout_rows.len(),
                        causal.len()
                    )
                    .into());
                }
                let bit_equal_rows = layout_rows
                    .chunks(vocab)
                    .zip(causal.chunks(vocab))
                    .filter(|(a, b)| {
                        a.iter()
                            .zip(b.iter())
                            .all(|(x, y)| x.to_bits() == y.to_bits())
                    })
                    .count();
                let max_abs = layout_rows
                    .iter()
                    .zip(&causal)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max);
                println!(
                    "verify-tree chain_layout rows=9 bit_equal_rows={bit_equal_rows} max_abs={max_abs:.6} verdict={}",
                    if bit_equal_rows == 9 { "PASS" } else { "FAIL" }
                );
                continue;
            }
            if rep == 2 {
                let tree = imparo_model::speculative::DraftTree::plain(
                    tokens.clone(),
                    TREE_PARENTS.to_vec(),
                );
                let mut out = Vec::new();
                model.forward_tree_logits(&tree, base, &mut out)?;
                if out.len() != nodes * vocab {
                    return Err(format!(
                        "--verify-tree: the tree forward returned {} logits for {nodes} nodes",
                        out.len()
                    )
                    .into());
                }
                let top1: Vec<usize> =
                    out.chunks(vocab).map(|row| top10_ids(row)[0]).collect();
                println!(
                    "verify-tree tree nodes={nodes} parents={TREE_PARENTS:?} tokens={tokens:?} top1={top1:?}"
                );
                tree_logits = Some(out);
                continue;
            }
            let commit_rep = 3 + TREE_LEAVES.len();
            if rep == commit_rep {
                // One verify of the branch_at_3 tree at the prompt's end: its path leaves the chain
                // after node 2, so the commit moves three nodes' KV rows and takes node 11's state.
                let (case, wrong, expect) = TREE_CASES[1];
                let tree = scripted_tree(reference, 0, wrong, vocab_size);
                if !model.prepare_greedy_tree(&tree, base)? {
                    return Err(
                        "--verify-tree: the model declined the tree at the prompt's end".into(),
                    );
                }
                let v = model.verify_greedy_tree(&tree, base, 9, &[])?;
                let filled = model.kv_runtime().filled;
                let state = model
                    .kv_spill()
                    .ok_or("--verify-tree: nothing to capture after the tree commit")?;
                let expect_next = reference[expect.len()];
                // One-token decoding from the committed state must continue the trail.
                let mut post_equal = v.next_token == expect_next;
                let mut next = v.next_token;
                for i in 0..TREE_POST {
                    if !post_equal {
                        break;
                    }
                    next = model.forward_next(next, filled + i)?;
                    post_equal = next == reference[expect.len() + 1 + i];
                }
                let pass =
                    v.path == expect && filled == base + expect.len() && post_equal;
                println!(
                    "verify-tree commit case={case} path={:?} expect={expect:?} next={} expect_next={expect_next} filled={filled} post_steps={TREE_POST} post_equal={post_equal} verdict={}",
                    v.path,
                    v.next_token,
                    if pass { "PASS" } else { "FAIL" }
                );
                tree_commit_state = Some(state);
                continue;
            }
            if rep == commit_rep + 1 {
                if tree_scramble {
                    println!(
                        "verify-tree commit_vs_chain case=branch_at_3 skipped=scrambled_placement"
                    );
                    continue;
                }
                // The same accepted tokens as a chain verify at the same start. Its committed KV
                // rows and recurrent state against the tree commit's: a measurement, no verdict.
                let (case, _, expect) = TREE_CASES[1];
                let v = model.verify_greedy_block(&reference[..expect.len()], base)?;
                let chain = model.kv_spill().ok_or(
                    "--verify-tree: nothing to capture after the chain verify",
                )?;
                let tree_state = tree_commit_state
                    .as_ref()
                    .ok_or("--verify-tree: the tree commit repetition did not run")?;
                println!(
                    "verify-tree commit_vs_chain case={case} chain_consumed={} chain_next={} {}",
                    v.consumed,
                    v.next_token,
                    committed_state_fields(
                        tree_state,
                        &chain,
                        base,
                        expect.len(),
                        &model.plan().recurrent_layout()
                    )
                );
                continue;
            }
            if rep == commit_rep + 2 {
                use imparo_model::speculative::{StopReason, continue_greedy};
                let mut provider = ScriptedTree {
                    trail: reference.clone(),
                    base,
                    vocab: vocab_size,
                    rounds: 0,
                    anchor_mismatches: 0,
                    tree_commits: 0,
                    branch_commits: 0,
                    paths_as_scripted: 0,
                };
                let run = continue_greedy(
                    model.as_mut(),
                    &mut provider,
                    base,
                    reference[0],
                    TREE_EMIT,
                    &[],
                )?;
                let tokens_equal = run.tokens == reference[..TREE_EMIT];
                let pass = tokens_equal
                    && run.reason == StopReason::Limit
                    && provider.anchor_mismatches == 0
                    && provider.tree_commits > 0
                    && provider.branch_commits > 0
                    && provider.paths_as_scripted == provider.tree_commits;
                println!(
                    "verify-tree cursor rows=9 tokens={} expect={TREE_EMIT} tokens_equal={tokens_equal} reason={:?} draft_calls={} verified_blocks={} sequential_steps={} tree_commits={} branch_commits={} paths_as_scripted={} anchor_mismatches={} verdict={}",
                    run.tokens.len(),
                    run.reason,
                    run.draft_calls,
                    run.verified_blocks,
                    run.sequential_steps,
                    provider.tree_commits,
                    provider.branch_commits,
                    provider.paths_as_scripted,
                    provider.anchor_mismatches,
                    if pass { "PASS" } else { "FAIL" }
                );
                continue;
            }
            if rep == commit_rep + 3 {
                // The wide tree's forward: each node on WIDE_PATH against a causal forward over the
                // trail's first nine tokens, which are the path's own tokens.
                let tree = wide_tree(reference, 0, vocab_size);
                let mut out = Vec::new();
                model.forward_tree_logits(&tree, base, &mut out)?;
                let mut causal = Vec::new();
                model.forward_all_logits_into(
                    &reference[..WIDE_PATH.len()],
                    base,
                    &mut causal,
                )?;
                if out.len() != tree.tokens.len() * vocab
                    || causal.len() != WIDE_PATH.len() * vocab
                {
                    return Err(format!(
                        "--verify-tree: the wide tree returned {} logits and its path forward {}",
                        out.len(),
                        causal.len()
                    )
                    .into());
                }
                let (mut exact, mut order_differs, mut top1_differs, mut worst) =
                    (0_usize, 0_usize, 0_usize, 0.0_f32);
                for (k, &node) in WIDE_PATH.iter().enumerate() {
                    let node = node as usize;
                    let chain_row = &causal[k * vocab..(k + 1) * vocab];
                    let tree_row = &out[node * vocab..(node + 1) * vocab];
                    if chain_row
                        .iter()
                        .zip(tree_row)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                    {
                        exact += 1;
                    }
                    let check = RowLogits::compare(chain_row, tree_row);
                    let (chain_top, tree_top) = (&check.reference_top, &check.row_top);
                    if !check.top1_equal() {
                        top1_differs += 1;
                    }
                    if let Some(rank) = check.first_swap() {
                        order_differs += 1;
                        // The first rank where the two orders part, and how far apart that pair
                        // sits in each arm: the witness step 1b prints for verify rows.
                        let (a, b) = (chain_top[rank], tree_top[rank]);
                        let gap = chain_top.get(rank + 1).map_or_else(
                            || "n/a".to_string(),
                            |&next| {
                                format!(
                                    "{:.6}",
                                    chain_row[chain_top[rank]] - chain_row[next]
                                )
                            },
                        );
                        println!(
                            "verify-tree wide_order node={node} depth={k} rank={} chain_id={a} tree_id={b} chain_gap_to_next={gap} chain_deltas={a}:{:.6},{b}:{:.6} tree_deltas={a}:{:.6},{b}:{:.6}",
                            rank + 1,
                            chain_row[a] - chain_row[chain_top[0]],
                            chain_row[b] - chain_row[chain_top[0]],
                            tree_row[a] - tree_row[tree_top[0]],
                            tree_row[b] - tree_row[tree_top[0]]
                        );
                    }
                    worst = worst.max(check.delta);
                }
                let tol = 2e-2_f32;
                println!(
                    "verify-tree wide nodes={} path={WIDE_PATH:?} path_exact={exact}/{} top1_differs={top1_differs} order_differs={order_differs} worst_delta={worst:.6} tol={tol} verdict={}",
                    tree.tokens.len(),
                    WIDE_PATH.len(),
                    if top1_differs == 0 && worst <= tol {
                        "PASS"
                    } else {
                        "FAIL"
                    }
                );
                continue;
            }
            if rep == commit_rep + 4 {
                // One verify of the wide tree: rows 56 to 61 move down to depths 3 to 8.
                let tree = wide_tree(reference, 0, vocab_size);
                if !model.prepare_greedy_tree(&tree, base)? {
                    return Err(
                        "--verify-tree: the model declined the wide tree".into()
                    );
                }
                let v = model.verify_greedy_tree(&tree, base, WIDE_PATH.len(), &[])?;
                let filled = model.kv_runtime().filled;
                let state = model
                    .kv_spill()
                    .ok_or("--verify-tree: nothing to capture after the wide commit")?;
                let expect_next = reference[WIDE_PATH.len()];
                let mut post_equal = v.next_token == expect_next;
                let mut next = v.next_token;
                for i in 0..TREE_POST {
                    if !post_equal {
                        break;
                    }
                    next = model.forward_next(next, filled + i)?;
                    post_equal = next == reference[WIDE_PATH.len() + 1 + i];
                }
                let pass = v.path == WIDE_PATH
                    && filled == base + WIDE_PATH.len()
                    && post_equal;
                println!(
                    "verify-tree commit case=wide path={:?} expect={WIDE_PATH:?} next={} expect_next={expect_next} filled={filled} post_steps={TREE_POST} post_equal={post_equal} verdict={}",
                    v.path,
                    v.next_token,
                    if pass { "PASS" } else { "FAIL" }
                );
                wide_commit_state = Some(state);
                continue;
            }
            if rep == commit_rep + 5 {
                if tree_scramble {
                    println!(
                        "verify-tree commit_vs_chain case=wide skipped=scrambled_placement"
                    );
                    continue;
                }
                // The wide path's tokens as a chain verify, byte for byte against the wide commit.
                let v =
                    model.verify_greedy_block(&reference[..WIDE_PATH.len()], base)?;
                let chain = model.kv_spill().ok_or(
                    "--verify-tree: nothing to capture after the chain verify",
                )?;
                let wide = wide_commit_state
                    .as_ref()
                    .ok_or("--verify-tree: the wide commit repetition did not run")?;
                println!(
                    "verify-tree commit_vs_chain case=wide chain_consumed={} chain_next={} {}",
                    v.consumed,
                    v.next_token,
                    committed_state_fields(
                        wide,
                        &chain,
                        base,
                        WIDE_PATH.len(),
                        &model.plan().recurrent_layout()
                    )
                );
                continue;
            }
            let leaf = TREE_LEAVES[rep - 3];
            let mut path = vec![leaf];
            while let Ok(parent) = usize::try_from(TREE_PARENTS[path[path.len() - 1]]) {
                path.push(parent);
            }
            path.reverse();
            let path_tokens: Vec<u32> = path.iter().map(|&i| tokens[i]).collect();
            // The leaf's path as a chain row layout: the verify's grid on both sides, so a node
            // whose ancestors are rows 0..depth computes exactly its chain row. Repetition 1
            // links the chain layout to the causal forward.
            let path_chain = imparo_model::speculative::DraftTree::plain(
                path_tokens.clone(),
                (-1..i32::try_from(path.len()).unwrap_or(i32::MAX) - 1).collect(),
            );
            let mut causal = Vec::new();
            model.forward_tree_logits(&path_chain, base, &mut causal)?;
            if causal.len() != path.len() * vocab {
                return Err(format!(
                    "--verify-tree: a {}-row path forward returned {} logits",
                    path.len(),
                    causal.len()
                )
                .into());
            }
            let tree_out = tree_logits
                .as_ref()
                .ok_or("--verify-tree: the tree repetition did not run")?;
            for (k, &node) in path.iter().enumerate() {
                let chain_row = &causal[k * vocab..(k + 1) * vocab];
                let tree_row = &tree_out[node * vocab..(node + 1) * vocab];
                let exact = chain_row
                    .iter()
                    .zip(tree_row)
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                let row = RowLogits::compare(chain_row, tree_row);
                let (order, delta) = (row.order(), row.delta);
                let top1_equal = order != RowOrder::Top1Differs;
                let order_equal = order == RowOrder::Equal;
                let chain_rows = path[..=k].iter().enumerate().all(|(j, &n)| j == n);
                println!(
                    "verify-tree node={node} leaf={leaf} depth={k} chain_rows={chain_rows} exact={exact} top1_equal={top1_equal} order_equal={order_equal} delta={delta:.6}"
                );
                let check = tree_checks[node].get_or_insert(TreeNodeCheck {
                    chain_rows,
                    exact: true,
                    order: RowOrder::Equal,
                    delta: 0.0,
                });
                check.exact &= exact;
                check.order = check.order.max(order);
                check.delta = check.delta.max(delta);
            }
            if rep + 1 == commit_rep {
                let tol = 2e-2_f32;
                let checks: Vec<TreeNodeCheck> =
                    tree_checks.iter().flatten().copied().collect();
                let chain_nodes = checks.iter().filter(|c| c.chain_rows).count();
                let chain_exact =
                    checks.iter().filter(|c| c.chain_rows && c.exact).count();
                let branch_exact =
                    checks.iter().filter(|c| !c.chain_rows && c.exact).count();
                let top1_differs = checks
                    .iter()
                    .filter(|c| c.order == RowOrder::Top1Differs)
                    .count();
                let order_differs =
                    checks.iter().filter(|c| c.order != RowOrder::Equal).count();
                let worst = checks.iter().map(|c| c.delta).fold(0.0_f32, f32::max);
                println!(
                    "verify-tree nodes={} chain_nodes={chain_nodes} chain_exact={chain_exact} branch_nodes={} branch_exact={branch_exact} top1_differs={top1_differs} order_differs={order_differs} worst_delta={worst:.6} tol={tol} chain_verdict={} logit_verdict={}",
                    checks.len(),
                    checks.len() - chain_nodes,
                    if chain_exact == chain_nodes {
                        "PASS"
                    } else {
                        "FAIL"
                    },
                    if top1_differs == 0 && worst <= tol {
                        "PASS"
                    } else {
                        "FAIL"
                    }
                );
            }
            continue;
        }
        if let Some((rows, emit, post)) = verify_chain {
            let base = toks_rep.len();
            let needed = base + emit + post + rows;
            let capacity = model.kv_runtime().capacity;
            if needed > capacity {
                return Err(format!(
                    "--verify-chain needs KV capacity {needed}, have {capacity}: set IMPARO_KV_CAP"
                )
                .into());
            }
            let first_token = imparo_cpu::ops::argmax_f32(&logits);
            if rep == 0 {
                let mut trail = vec![first_token];
                let mut next = first_token;
                for i in 0..emit + post {
                    next = model.forward_next(next, base + i)?;
                    trail.push(next);
                }
                println!(
                    "verify-chain reference tokens={} trail={trail:?}",
                    trail.len()
                );
                chain_reference = Some(trail);
                continue;
            }
            let reference = chain_reference
                .as_ref()
                .ok_or("--verify-chain: the reference repetition did not run")?;
            if reference[0] != first_token {
                return Err(
                    "--verify-chain: this prefill and repetition 0's disagree on the first token"
                        .into(),
                );
            }
            let vocab = vocab_size as usize;
            if rep == 2 {
                let mut one_row = Vec::with_capacity(emit * vocab);
                let mut row = Vec::new();
                for (i, &token) in reference[..emit].iter().enumerate() {
                    model.forward_into(&[token], base + i, &mut row)?;
                    if row.len() != vocab {
                        return Err(format!(
                            "--verify-chain: a one-row forward returned {} logits",
                            row.len()
                        )
                        .into());
                    }
                    one_row.extend_from_slice(&row);
                }
                chain_one_row = Some(one_row);
                continue;
            }
            if rep == 3 {
                let one_row = chain_one_row
                    .as_ref()
                    .ok_or("--verify-chain: the one-row repetition did not run")?;
                let cell = imparo_model::prefill_batch();
                let tol = 2e-2_f32;
                let (mut compared, mut order_differs, mut top1_differs) =
                    (0_usize, 0_usize, 0_usize);
                let (mut batches, mut single_rows) = (0_usize, 0_usize);
                let (mut worst, mut worst_pos) = (0_f32, 0_usize);
                let mut batch = Vec::new();
                let mut at = 0_usize;
                while at < emit {
                    // A batch that would cross the prefill cell runs one row at a time, so each
                    // batch stops at the cell's end.
                    let pos = base + at;
                    let take = rows.min(emit - at).min(cell - pos % cell);
                    model.forward_all_logits_into(
                        &reference[at..at + take],
                        pos,
                        &mut batch,
                    )?;
                    if batch.len() != take * vocab {
                        return Err(format!(
                            "--verify-chain: a {take}-row forward returned {} logits",
                            batch.len()
                        )
                        .into());
                    }
                    batches += 1;
                    if take == 1 {
                        single_rows += 1;
                    }
                    for j in 0..take {
                        let one = &one_row[(at + j) * vocab..(at + j + 1) * vocab];
                        let many = &batch[j * vocab..(j + 1) * vocab];
                        let check = RowLogits::compare(one, many);
                        let (one_top, many_top) =
                            (&check.reference_top, &check.row_top);
                        compared += 1;
                        if !check.top1_equal() {
                            top1_differs += 1;
                        }
                        if let Some(rank) = check.first_swap() {
                            order_differs += 1;
                            // At rank 10 the next id is outside the top 10, so there is no gap.
                            let gap = one_top.get(rank + 1).map_or_else(
                                || "n/a".to_string(),
                                |&next| {
                                    format!("{:.6}", one[one_top[rank]] - one[next])
                                },
                            );
                            // Both ids' deltas to the top-1 in both arms: how far the pair moved.
                            let (a, b) = (one_top[rank], many_top[rank]);
                            println!(
                                "verify-logits order pos={} rank={} one_row_id={a} batch_id={b} one_row_gap_to_next={gap} one_row_deltas={a}:{:.6},{b}:{:.6} batch_deltas={a}:{:.6},{b}:{:.6}",
                                pos + j,
                                rank + 1,
                                one[a] - one[one_top[0]],
                                one[b] - one[one_top[0]],
                                many[a] - many[many_top[0]],
                                many[b] - many[many_top[0]]
                            );
                        }
                        if check.delta > worst {
                            worst = check.delta;
                            worst_pos = pos + j;
                        }
                    }
                    at += take;
                }
                println!(
                    "verify-logits rows={rows} batches={batches} single_row_batches={single_rows} compared={compared} top1_differs={top1_differs} order_differs={order_differs} worst_delta={worst:.6} worst_pos={worst_pos} tol={tol} verdict={}",
                    if top1_differs == 0 && worst <= tol {
                        "PASS"
                    } else {
                        "FAIL"
                    }
                );
                continue;
            }
            if rep >= 4 {
                use imparo_model::speculative::{StopReason, continue_greedy};
                let (case, stops, inject, stop_index) = match rep {
                    4 => ("plain", Vec::new(), None, None),
                    5 => {
                        let unused = (0..vocab_size)
                            .rev()
                            .find(|t| !reference.contains(t) && !toks_rep.contains(t))
                            .ok_or("--verify-chain: every token id is in the prompt or the trail")?;
                        ("injected_stop", vec![unused], Some(unused), None)
                    }
                    _ => {
                        // The last trail token that does not repeat an earlier one: one-token
                        // decoding first picks it at `s`, so the output must end just before `s`.
                        let s = (1..emit)
                            .rev()
                            .find(|&s| !reference[..s].contains(&reference[s]))
                            .ok_or("--verify-chain: every trail token after the first repeats an earlier one")?;
                        ("trail_stop", vec![reference[s]], None, Some(s))
                    }
                };
                let (expect_tokens, expect_reason) = match stop_index {
                    Some(s) => (&reference[..s], StopReason::StopToken(reference[s])),
                    None => (&reference[..emit], StopReason::Limit),
                };
                let mut provider = ScriptedDraft {
                    trail: reference.clone(),
                    base,
                    block: rows,
                    vocab: vocab_size,
                    inject,
                    stops: stops.clone(),
                    rounds: 0,
                    proposals_with_stop: 0,
                    proposals_led_by_stop: 0,
                    anchor_mismatches: 0,
                };
                let run = continue_greedy(
                    model.as_mut(),
                    &mut provider,
                    base,
                    reference[0],
                    emit,
                    &stops,
                )?;
                let tokens_equal = run.tokens == expect_tokens;
                let reason_equal = run.reason == expect_reason;
                // Speculation must survive a proposal that held a stop token: every draft is
                // verified, except one that began with a stop token, where the cut leaves nothing
                // to verify and the round runs one plain step.
                let speculated = run.verified_blocks > 0
                    && run.draft_calls
                        == run.verified_blocks + provider.proposals_led_by_stop;
                let pass = tokens_equal
                    && reason_equal
                    && speculated
                    && provider.anchor_mismatches == 0;
                println!(
                    "verify-draft case={case} rows={rows} tokens={} expect={} tokens_equal={tokens_equal} reason={:?} reason_equal={reason_equal} draft_calls={} verified_blocks={} sequential_steps={} proposals_with_stop={} proposals_led_by_stop={} anchor_mismatches={} verdict={}",
                    run.tokens.len(),
                    expect_tokens.len(),
                    run.reason,
                    run.draft_calls,
                    run.verified_blocks,
                    run.sequential_steps,
                    provider.proposals_with_stop,
                    provider.proposals_led_by_stop,
                    provider.anchor_mismatches,
                    if pass { "PASS" } else { "FAIL" }
                );
                continue;
            }
            let mut consumed_hist = vec![0_usize; rows + 1];
            // `at` indexes the reference: reference[at] is the anchor at position base + at.
            let mut at = 0_usize;
            let mut round = 0_usize;
            let mut diverged: Option<usize> = None;
            while at < emit {
                let pos = base + at;
                let take = rows.min(emit - at + 1);
                let mut input = reference[at..at + take].to_vec();
                let corrupt = round % take;
                if corrupt > 0 {
                    input[corrupt] = (input[corrupt] + 1) % vocab_size;
                }
                let v = model.verify_greedy_block(&input, pos)?;
                let filled = model.kv_runtime().filled;
                if v.consumed == 0 || v.consumed > take || filled != pos + v.consumed {
                    return Err(format!(
                        "--verify-chain round {round}: consumed {} of {take}, filled {filled}, expected {}",
                        v.consumed,
                        pos + v.consumed
                    )
                    .into());
                }
                consumed_hist[v.consumed] += 1;
                let expect = reference[at + v.consumed];
                println!(
                    "verify-chain round={round} pos={pos} rows={take} corrupt_at={corrupt} consumed={} next={} expect={expect}",
                    v.consumed, v.next_token
                );
                let accepted_as_scripted = if corrupt > 0 {
                    v.consumed == corrupt
                } else {
                    v.consumed == take
                };
                if v.next_token != expect || !accepted_as_scripted {
                    diverged = Some(at + v.consumed);
                    break;
                }
                at += v.consumed;
                round += 1;
            }
            let post_steps = post.min(reference.len() - 1 - at);
            let mut post_diverged: Option<usize> = None;
            if diverged.is_none() {
                let mut next = reference[at];
                for i in 0..post_steps {
                    next = model.forward_next(next, base + at + i)?;
                    if next != reference[at + i + 1] {
                        post_diverged = Some(at + i + 1);
                        break;
                    }
                }
            }
            println!(
                "verify-chain rows={rows} emitted={at} rounds={round} consumed_hist={consumed_hist:?} trail_equal={} post_steps={post_steps} post_equal={}",
                diverged.is_none(),
                diverged.is_none() && post_diverged.is_none()
            );
            if let Some(index) = diverged.or(post_diverged) {
                println!("verify-chain first_divergence_index={index}");
            }
            continue;
        }
        if graph_decode_n > 0 {
            let mut next = logits
                .iter()
                .enumerate()
                .max_by(|left, right| {
                    left.1
                        .partial_cmp(right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(index, _)| index as u32)
                .ok_or("graph decode prefill produced no logits")?;
            let base = toks_rep.len();
            let mut trail = vec![next];
            let mut step_us = Vec::with_capacity(graph_decode_n);
            for step in 0..graph_decode_n {
                let step_start = std::time::Instant::now();
                next = model.forward_next(next, base + step)?;
                step_us.push(step_start.elapsed().as_secs_f64() * 1e6);
                trail.push(next);
                println!(
                    "graph-dstep placement={} i={step} token={next}",
                    if graph_decode_paged {
                        "paged"
                    } else {
                        "identity"
                    }
                );
            }
            // Step 0 warms the route and step 1 captures the graph. Report only the
            // steady replay suffix so model load and graph construction cannot be
            // mistaken for per-token Decode latency.
            let warmup_steps = step_us.len().min(2);
            let mut steady = step_us[warmup_steps..].to_vec();
            steady.sort_by(f64::total_cmp);
            if !steady.is_empty() {
                let median_us = if steady.len() % 2 == 0 {
                    steady[steady.len() / 2 - 1].midpoint(steady[steady.len() / 2])
                } else {
                    steady[steady.len() / 2]
                };
                let mean_us = steady.iter().sum::<f64>() / steady.len() as f64;
                println!(
                    concat!(
                        "graph-decode-timing warmup_steps={} samples={} ",
                        "median_us={:.3} mean_us={:.3} min_us={:.3} max_us={:.3}"
                    ),
                    warmup_steps,
                    steady.len(),
                    median_us,
                    mean_us,
                    steady[0],
                    steady[steady.len() - 1]
                );
            }
            println!(
                "graph-decode placement={} steps={graph_decode_n} trail={trail:?}",
                if graph_decode_paged {
                    "paged"
                } else {
                    "identity"
                }
            );
        }
        if decode_n > 0 {
            // FNV-1a 64-bit: standard offset basis and prime. Order-sensitive byte
            // fingerprint -- two runs must produce identical hashes at every step.
            const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
            const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
            let fnv = |bytes: &mut dyn Iterator<Item = u8>| {
                bytes.fold(FNV_OFFSET, |h, b| {
                    (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
                })
            };
            let argmax = |v: &[f32]| {
                let mut best = 0usize;
                for (i, &x) in v.iter().enumerate() {
                    if x > v[best] {
                        best = i;
                    }
                }
                best as u32
            };
            let base = toks_rep.len();
            let print_decode_top10 = std::env::var("IMPARO_DECODE_TOP10").is_ok();
            if print_decode_top10 {
                let mut row = String::new();
                top10_string(&logits, &mut row);
                println!("decode-top10{row}");
            }
            let mut next = argmax(&logits);
            let mut trail: Vec<u32> = vec![next];
            let mut lg = Vec::new();
            let mut th = FNV_OFFSET;
            let mut hash_ms = 0.0_f64;
            let t_dec = std::time::Instant::now();
            if decode_pipe {
                if !model.decode_pipelined() {
                    return Err(
                        "--decode-pipe: decode is not pipelined on this backend/model"
                            .into(),
                    );
                }
                model.queue_step(Some(next), base)?;
                for i in 0..decode_n {
                    if i + 1 < decode_n {
                        model.queue_step(None, base + i + 1)?;
                    }
                    next = model.wait_step()?;
                    trail.push(next);
                }
                let dt_ms = t_dec.elapsed().as_secs_f64() * 1e3;
                println!(
                    "decode-pipe rep={rep} steps={decode_n} ms_per_forward={:.3} pos0={base}",
                    dt_ms / decode_n as f64
                );
                println!("decode-pipe rep={rep} steps={decode_n} trail={trail:?}");
                continue;
            }
            let mut step = vec![next; dbatch];
            for i in 0..decode_n {
                // Refilled every step: the greedy trail is what the determinism gate
                // reads, and a `step` built once would forward the FIRST token forever
                // while still printing a plausible hash per step.
                step.fill(next);
                model.forward_into(&step, base + i * dbatch, &mut lg)?;
                if std::env::var("IMPARO_KV_PROBE_STEP")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .is_some_and(|wanted| wanted == i)
                {
                    let layer = std::env::var("IMPARO_KV_PROBE_LAYER")
                        .ok()
                        .and_then(|value| value.parse::<u32>().ok())
                        .unwrap_or(0);
                    let geom = model
                        .kv_state_geometry()
                        .into_iter()
                        .find(|geom| geom.layer == layer)
                        .ok_or_else(|| {
                            format!("missing KV geometry for layer {layer}")
                        })?;
                    let slots = match geom.kind {
                        imparo_kv::StateKind::Window { ring, .. } => ring,
                        imparo_kv::StateKind::Full => base + (i + 1) * dbatch,
                    };
                    let is_v = std::env::var("IMPARO_KV_PROBE_SIDE")
                        .is_ok_and(|side| side.eq_ignore_ascii_case("v"));
                    let stride = if is_v { geom.v_stride } else { geom.k_stride };
                    let mut bytes = vec![0_u8; slots * stride];
                    imparo_model::backend::active()
                        .unwrap()
                        .read_kv_bytes(layer, is_v, 0, &mut bytes);
                    let path = std::env::var("IMPARO_KV_PROBE_DUMP")
                        .map_err(|_| "IMPARO_KV_PROBE_DUMP is required")?;
                    std::fs::write(&path, bytes)?;
                    eprintln!(
                        "[gpu] KV probe layer={layer} V={is_v} step={i} path={path}"
                    );
                }
                if print_decode_top10 {
                    let mut row = String::new();
                    top10_string(&lg, &mut row);
                    println!("decode-top10{row}");
                }
                // THE HASH IS THE HARNESS'S, NOT THE ENGINE'S, and it is not free: FNV over
                // 128000 f32 is 512 KB a step. Timed and subtracted below so
                // ms_per_forward is the forward and nothing else. It cannot simply be
                // skipped -- the stephash is built from it.
                let t_hash = std::time::Instant::now();
                let h = fnv(&mut lg.iter().flat_map(|v| v.to_le_bytes()));
                hash_ms += t_hash.elapsed().as_secs_f64() * 1e3;
                if dbatch == 1 {
                    println!("dstep rep={rep} i={i} h={h:016x}");
                }
                // IMPARO_DECODE_DUMP_DIR: the decode step's logits as raw f32 LE, one file per
                // step, for a bit-level A/B between two routes (the prefill dump above cannot
                // see a decode-only route).
                if let Some(dir) = std::env::var_os("IMPARO_DECODE_DUMP_DIR") {
                    let dir = PathBuf::from(dir);
                    std::fs::create_dir_all(&dir)?;
                    dump_logits_to_path(
                        &lg,
                        dir.join(format!("dstep-{rep:03}-{i:03}.raw")),
                    )?;
                }
                th = fnv(&mut h.to_le_bytes().into_iter().chain(th.to_le_bytes()));
                next = argmax(&lg);
                trail.push(next);
            }
            // The forward alone. The step hash above is harness bookkeeping over the
            // logits and sat inside this bracket until 2026-09-21, inflating every decode
            // number this binary printed by about a millisecond a token.
            let dt_ms = t_dec.elapsed().as_secs_f64() * 1e3 - hash_ms;
            // Plain wall clock, always printed: IMPARO_PROF=1 costs three orders of
            // magnitude here (see host::prof_log), so it can never answer "what does
            // this shape cost".
            println!(
                "decode rep={rep} steps={decode_n} rows={dbatch} \
                 ms_per_forward={:.3} hash_ms_per_step={:.3} pos0={base}",
                dt_ms / decode_n as f64,
                hash_ms / decode_n as f64
            );
            if dbatch == 1 {
                println!(
                    "decode rep={rep} steps={decode_n} stephash={th:016x} trail={trail:?}"
                );
            }
            imparo_model::host::prof_log("decode", dt_ms);
        }
        if let Some(dir) = &logits_dump_dir {
            dump_logits_to_path(&logits, dir.join(format!("rep-{rep:03}.raw")))?;
        }
        if rep + 1 < repeat {
            print_top10(&logits);
        }
    }

    dump_logits_if_requested(&logits)?;
    let finite = logits.iter().filter(|v| v.is_finite()).count();
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    // NO TIMING IS PRINTED by default, deliberately. Nothing parses one -- not the gates, not
    // bracket.py, which is the only sanctioned speed comparison and drives the server.
    // The `fwd ms=` line that used to be here existed solely to be grepped by ad-hoc
    // benchmarking, and a full engine forward is the wrong instrument for kernel speed:
    // it is ~32 s at 16k, pins the GPU, and answers questions a seconds-long micro-bench
    // answers better. Speed work goes through the tuner's micro-benches (#39, #44).
    // The exact-value Phase-A1 lab opt-in above is the sole exception and remains
    // explicitly non-production-authoritative.
    println!("logits={} finite={finite}", logits.len());
    println!("stats min={min:.4} max={max:.4} mean={mean:.4}");

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|a, b| {
        logits[*b]
            .partial_cmp(&logits[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    print!("top10");
    // Six decimals, not four: this output is diffed against llama.cpp's logprobs, and at
    // four the print resolution was the same order as the difference being measured.
    for &i in idx.iter().take(10) {
        print!(" {i}:{:.6}", logits[i]);
    }
    println!();

    if finite != logits.len() {
        return Err("non-finite logits".into());
    }
    Ok(())
}

/// Writes the current state to the store, AT the position the device is at.
///
/// Split out of `main` because the order matters: the caller forwards to the boundary,
/// calls this, and only then forwards the tail. A recurrent state is one buffer holding
/// "now", so a checkpoint taken after the whole forward has absorbed the tail -- and
/// reprocessing that tail on restore would apply it twice.
fn spill_now(
    model: &mut dyn imparo_model::Model,
    dir: &std::path::Path,
    path: &std::path::Path,
    tokens: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    let state = model.kv_spill().ok_or("spill: nothing to spill")?;
    let digest = imparo_model::kv::model_digest(path)?;
    let root = imparo_model::kv::config_root(
        model.plan(),
        &digest,
        &imparo_model::backend::active()
            .map_or_else(|| "none".to_string(), imparo_backend::Backend::device_tag),
    );
    // One call: the extents, the anchor link and the manifest come from the store,
    // which is also what the server's non-pool commit uses. This was a copy of that.
    let store = imparo_kv::Store::open(dir, &root).map_err(|e| e.clone())?;
    let staged = store.stage_whole(&root, tokens, &state)?;
    store.commit("cli", &staged.manifest(false), &staged.at)?;
    Ok(())
}

/// IMPARO_DSPARK_REPLAY: a speculative run's rounds replayed through the drafter. `tokens` are
/// the run's committed tokens (the prompt, then the output) and `rounds` holds each round's start
/// position, ascending: where its anchor sits. Before a round the tokens below its start go
/// through the target in forwards of at most one prefill chunk, each followed by the drafter's
/// append; the round then drafts one block from its anchor and prints it.
#[cfg(feature = "speculative")]
fn dspark_replay_rounds(
    model: &mut (dyn imparo_model::Model + Send),
    descriptor: &imparo_model::dspark::DsparkDescriptor,
    taps: &imparo_model::dspark::FeatureTaps,
    tokens: &[u32],
    rounds: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let starts: Vec<usize> = std::fs::read_to_string(rounds)?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let drafter =
        imparo_model::dspark_forward::DsparkForward::attach(&mut *model, descriptor)?;
    let chunk = imparo_model::prefill_batch();
    let mut cursor = 0;
    for &start in &starts {
        if start < cursor || start >= tokens.len() {
            return Err(format!(
                "replay round at {start}: rounds must ascend and stay below the {} tokens",
                tokens.len()
            )
            .into());
        }
        while cursor < start {
            let n = chunk.min(start - cursor);
            model.forward(&tokens[cursor..cursor + n], cursor)?;
            let (at, rows) = (u32::try_from(cursor)?, u32::try_from(n)?);
            taps.check_rows(at, rows)?;
            drafter.append(at, rows)?;
            cursor += n;
        }
        let ids = drafter
            .generate(&mut *model, u32::try_from(start)?, tokens[start], 0)?
            .ids;
        println!(
            "dspark round start={start} anchor={} drafts={}",
            tokens[start],
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        );
    }
    Ok(())
}

/// `--decode N --dspark DRAFTER`: N greedy tokens through the DSpark provider, the way a
/// request runs them: the provider sees the prompt as an observed prefill with a commit per
/// chunk, then the greedy cursor drafts, verifies and commits. Prints one line per verified round
/// and the trail in `--decode`'s form.
#[cfg(feature = "speculative")]
fn dspark_decode(
    model: &mut (dyn imparo_model::Model + Send),
    pairing: imparo_model::dspark::Pairing,
    tokens: &[u32],
    limit: usize,
    stops: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    use imparo_model::speculative::{DraftSpec, GreedyCursor};
    let spec = DraftSpec::Dspark(pairing);
    model.with_draft(&spec, &mut |model, provider| {
        // IMPARO_KV_PREGROW=1: the cache at its capacity before the prompt, as the server's pool
        // sizes it, so no decode round grows it.
        if std::env::var("IMPARO_KV_PREGROW").as_deref() == Ok("1") {
            model.kv_prepare_pool()?;
            imparo_model::host::log_footprint("kv pregrow");
        }
        let t_prefill = std::time::Instant::now();
        provider.initialize_at(0)?;
        provider.set_capture(true)?;
        let mut logits = Vec::new();
        let (mut committed, mut chunks) = (0_usize, 0_usize);
        model.forward_prefill_observed(tokens, 0, &mut logits, None, &mut |at, chunk| {
            if at != committed {
                return Err(format!(
                    "draft prefill chunk at {at}, the history ends at {committed}"
                ));
            }
            provider.commit(at, chunk)?;
            committed = at + chunk.len();
            chunks += 1;
            Ok(())
        })?;
        if committed != tokens.len() {
            return Err(format!(
                "draft prompt history {committed} of {} tokens",
                tokens.len()
            ));
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
        imparo_model::host::prof_log("prefill", prefill_ms);
        // The smallest index attaining the maximum, as --decode picks.
        let first = logits
            .iter()
            .enumerate()
            .fold(0_usize, |best, (i, &x)| if x > logits[best] { i } else { best });
        let first = u32::try_from(first).map_err(|_| "first token id overflows")?;
        let clock = std::env::var("IMPARO_DSPARK_CLOCK").as_deref() == Ok("1");
        let mut log = RoundLog {
            inner: provider,
            drafted: None,
            tree: std::cell::Cell::new(None),
            path: None,
            clock,
            draft_ms: 0.0,
            tree_ms: std::cell::Cell::new(0.0),
            append_ms: 0.0,
        };
        let t_decode = std::time::Instant::now();
        let mut cursor =
            GreedyCursor::new(&*model, &log, tokens.len(), first, limit, stops)?;
        let mut trail = Vec::with_capacity(limit);
        loop {
            let (blocks, consumed) = (cursor.verified_blocks, cursor.consumed);
            let round_at = clock.then(std::time::Instant::now);
            let token = cursor.next(&mut *model, &mut log)?;
            if cursor.verified_blocks > blocks {
                let (start, anchor, drafts) = log
                    .drafted
                    .take()
                    .ok_or("dspark: a verified round with no drafts")?;
                println!(
                    "dspark round start={start} anchor={anchor} drafts={} consumed={} next={}",
                    drafts
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    cursor.consumed - consumed,
                    cursor.next_token
                );
                let nodes = log.tree.take();
                let path = log.path.take();
                if let Some(nodes) = nodes {
                    println!(
                        "dspark tree start={start} nodes={nodes} path={}",
                        path.map_or_else(
                            || "-".to_string(),
                            |p| p.iter().map(i32::to_string).collect::<Vec<_>>().join(",")
                        )
                    );
                }
                if let Some(at) = round_at {
                    let round_ms = at.elapsed().as_secs_f64() * 1e3;
                    let tree_ms = log.tree_ms.get();
                    println!(
                        "dspark clock start={start} form={} rows={} round_ms={round_ms:.3} draft_ms={:.3} tree_ms={tree_ms:.3} verify_ms={:.3} append_ms={:.3}",
                        if nodes.is_some() { "tree" } else { "chain" },
                        nodes.unwrap_or(drafts.len() + 1),
                        log.draft_ms,
                        round_ms - log.draft_ms - tree_ms - log.append_ms,
                        log.append_ms
                    );
                }
            }
            match token {
                Some(t) => trail.push(t),
                None => break,
            }
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1e3;
        println!(
            "dspark decode prompt={} chunks={chunks} steps={limit} tokens={} reason={:?} draft_calls={} verified_blocks={} sequential_steps={} prefill_ms={prefill_ms:.1} decode_ms={decode_ms:.1} trail={trail:?}",
            tokens.len(),
            trail.len(),
            cursor.reason,
            cursor.draft_calls,
            cursor.verified_blocks,
            cursor.sequential_steps,
        );
        imparo_model::host::prof_log("dspark decode", decode_ms);
        imparo_model::host::log_footprint("dspark decode");
        Ok(())
    })?;
    Ok(())
}

/// `--dspark`'s view of the provider: every call passes through, and a round's drafts are kept
/// until the round is printed.
#[cfg(feature = "speculative")]
struct RoundLog<'a> {
    inner: &'a mut dyn imparo_model::speculative::DraftProvider,
    drafted: Option<(usize, u32, Vec<u32>)>,
    /// The round's tree when the provider proposed one and the round verified it: its nodes, and
    /// the accepted path once the tree is committed.
    tree: std::cell::Cell<Option<usize>>,
    path: Option<Vec<i32>>,
    /// `IMPARO_DSPARK_CLOCK=1`: the round's drafter forward, tree build and drafter append, in ms;
    /// the verify is the rest of the round.
    clock: bool,
    draft_ms: f64,
    tree_ms: std::cell::Cell<f64>,
    append_ms: f64,
}

#[cfg(feature = "speculative")]
impl imparo_model::speculative::DraftProvider for RoundLog<'_> {
    fn tree_proposal(
        &mut self,
        start: usize,
        anchor: u32,
        chain: &[u32],
        stops: &[u32],
    ) -> Result<Option<imparo_model::speculative::DraftTree>, String> {
        let at = self.clock.then(std::time::Instant::now);
        let tree = self.inner.tree_proposal(start, anchor, chain, stops)?;
        if let Some(at) = at {
            self.tree_ms.set(at.elapsed().as_secs_f64() * 1e3);
        }
        self.tree.set(tree.as_ref().map(|t| t.tokens.len()));
        Ok(tree)
    }

    fn commit_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        path: &[i32],
        next: u32,
    ) -> Result<(), String> {
        let at = self.clock.then(std::time::Instant::now);
        self.inner.commit_tree(start, inputs, path, next)?;
        if let Some(at) = at {
            self.append_ms = at.elapsed().as_secs_f64() * 1e3;
        }
        self.path = Some(path.to_vec());
        Ok(())
    }

    fn keep_tree(
        &mut self,
        start: usize,
        inputs: &[u32],
        path: &[i32],
    ) -> Result<(), String> {
        let at = self.clock.then(std::time::Instant::now);
        self.inner.keep_tree(start, inputs, path)?;
        if let Some(at) = at {
            self.append_ms = at.elapsed().as_secs_f64() * 1e3;
        }
        self.path = Some(path.to_vec());
        Ok(())
    }

    // EVERY METHOD IS FORWARDED BY HAND, so a new one on the trait defaults here and the
    // wrapped provider never sees it. That is how the verify-cost table first measured
    // nothing: 121 rounds ran, all at the same 8 rows, with every step of the table empty.
    fn keeps_rows(&self, start: usize, n: usize) -> bool {
        self.inner.keeps_rows(start, n)
    }

    fn observe_verify(&mut self, rows: usize, elapsed: std::time::Duration) {
        self.inner.observe_verify(rows, elapsed);
    }

    fn observe_round_fixed(&mut self, elapsed: std::time::Duration) {
        self.inner.observe_round_fixed(elapsed);
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn minimum_remaining(&self) -> usize {
        self.inner.minimum_remaining()
    }

    fn initialize(&mut self) -> Result<(), String> {
        self.inner.initialize()
    }

    fn initialize_at(&mut self, start: usize) -> Result<(), String> {
        self.inner.initialize_at(start)
    }

    fn set_capture(&mut self, enabled: bool) -> Result<(), String> {
        self.inner.set_capture(enabled)
    }

    fn can_draft(&self, start: usize, remaining: usize) -> bool {
        self.inner.can_draft(start, remaining)
    }

    fn can_bridge(&self, start: usize, remaining: usize) -> bool {
        self.inner.can_bridge(start, remaining)
    }

    fn draft(
        &mut self,
        target: &mut dyn imparo_model::Model,
        start: usize,
        anchor: u32,
    ) -> Result<Vec<u32>, String> {
        let ids = self.inner.draft(target, start, anchor)?;
        self.drafted = Some((start, anchor, ids.clone()));
        Ok(ids)
    }

    fn try_draft(
        &mut self,
        target: &mut dyn imparo_model::Model,
        start: usize,
        anchor: u32,
    ) -> Result<Option<Vec<u32>>, String> {
        let at = self.clock.then(std::time::Instant::now);
        let ids = self.inner.try_draft(target, start, anchor)?;
        if let Some(at) = at {
            self.draft_ms = at.elapsed().as_secs_f64() * 1e3;
            self.tree_ms.set(0.0);
            self.append_ms = 0.0;
        }
        if let Some(ids) = &ids {
            self.drafted = Some((start, anchor, ids.clone()));
        }
        Ok(ids)
    }

    fn commit(&mut self, start: usize, inputs: &[u32]) -> Result<(), String> {
        let at = self.clock.then(std::time::Instant::now);
        self.inner.commit(start, inputs)?;
        if self.drafted.is_some() {
            // This round verified a chain: no tree was proposed, or the target declined it.
            self.tree.set(None);
            if let Some(at) = at {
                self.append_ms = at.elapsed().as_secs_f64() * 1e3;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), String> {
        self.inner.finish()
    }
}

/// The DSpark drafter in the file `draft`, opened as it is (`Pairing::open`; the mask token is
/// the drafter's own). Returns the drafter's path, which the caller maps after MODEL.
#[cfg(feature = "speculative")]
fn load_drafter(
    draft: &std::path::Path,
) -> Result<(PathBuf, imparo_model::dspark::Pairing), Box<dyn std::error::Error>> {
    let (draft, pairing) = imparo_model::dspark::Pairing::open(draft, None)?;
    eprintln!("[imparo] drafter admitted: {}", draft.display());
    Ok((draft, pairing))
}

#[cfg(test)]
mod phase_a1_prefill_wall_tests {
    use super::phase_a1_prefill_wall_line;

    #[test]
    fn absent_opt_in_produces_no_line() {
        let line = phase_a1_prefill_wall_line(
            false,
            0,
            512,
            0,
            "single",
            None,
            None,
            std::time::Duration::from_millis(7),
        )
        .unwrap();
        assert!(line.is_none());
    }

    #[test]
    fn opt_in_line_is_strict_lab_only_json() {
        let line = phase_a1_prefill_wall_line(
            true,
            3,
            384,
            128,
            "restore-tail",
            Some(128),
            Some(&[384]),
            std::time::Duration::from_micros(12_345),
        )
        .unwrap()
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["schema"], 1);
        assert_eq!(value["phase"], "A1");
        assert_eq!(value["event"], "prefill_wall");
        assert_eq!(value["lab_only"], true);
        assert_eq!(value["production_authority"], false);
        assert_eq!(value["rep"], 3);
        assert_eq!(value["token_count"], 384);
        assert_eq!(value["start_pos"], 128);
        assert_eq!(value["split_state"]["mode"], "restore-tail");
        assert_eq!(value["split_state"]["split_at"], 128);
        assert_eq!(value["split_state"]["parts"], serde_json::json!([384]));
        assert_eq!(value["prefill_wall_ms"], 12.345);
        assert!(!line.contains('\n'));
    }
}

#[cfg(test)]
mod full_history_control_tests {
    use super::full_history_control_requested;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn control_requires_explicit_gate_prefix_and_preserves_forward_options() {
        let ordinary = args(&["--repeat", "2", "--decode", "8", "model.gguf", "2"]);
        assert_eq!(full_history_control_requested(&ordinary, false), Ok(false));
        let mut control = args(&["--lfm-full-history-control"]);
        control.extend(ordinary);
        assert!(full_history_control_requested(&control, false).is_err());
        assert_eq!(full_history_control_requested(&control, true), Ok(true));
        assert!(
            full_history_control_requested(
                &args(&["model.gguf", "--lfm-full-history-control"]),
                true,
            )
            .is_err()
        );
        control.push("--lfm-full-history-control".into());
        assert!(full_history_control_requested(&control, true).is_err());
    }

    #[test]
    fn control_cannot_enter_template_or_other_fixed_checker() {
        for other in ["--correctness-template", "--w4a16-ffn-check"] {
            assert!(
                full_history_control_requested(
                    &args(&["--lfm-full-history-control", other, "model.gguf"]),
                    true,
                )
                .is_err()
            );
            assert!(
                full_history_control_requested(
                    &args(&[other, "--lfm-full-history-control", "model.gguf"]),
                    true,
                )
                .is_err()
            );
        }
    }
}
