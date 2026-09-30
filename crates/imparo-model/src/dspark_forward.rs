//! The DSpark drafter's forward on `Backend` ops.
//!
//! The same two operations as the CUDA session in `imparo-cuda` (`dspark.cuh`), written against
//! the backend-neutral ops:
//!
//! ```text
//! append(start, n)         the tapped features of the target's last forward, its first n rows
//!                          -> fc -> enc_norm -> per layer: k, v -> K norm + RoPE -> store at start..
//! generate(start, anchor)  [anchor, mask x (block - 1)] at start.. -> embedding
//!                          -> per layer: norm, q/k/v, Q/K norm + RoPE, store, attention over the
//!                             committed cache and the whole block, o, residual, norm, FFN, residual
//!                          -> output norm -> head -> DraftLogits (block x vocab)
//!                          -> the Markov chain on the device -> the block's ids, read back once
//! ```
//!
//! The block's K/V rows sit above the committed context. The next `append` overwrites them, and
//! no attention reads above `start + block`, so no stale row is ever read.

use crate::dspark::{DraftTensor, DsparkDescriptor};
use imparo_backend::{
    Backend, BufId, ROW_LAYOUT_ANCESTORS, ROW_LAYOUT_MAX_ROWS, ROW_LAYOUT_WORDS,
};
use imparo_gguf::weights::{WeightKind, weight_kind};

/// A weight the backend reads by kind and offset in the mapping.
#[derive(Clone, Copy, Debug)]
struct Weight {
    kind: u32,
    off: u64,
}

impl Weight {
    fn of(t: &DraftTensor, name: &str) -> Result<Self, String> {
        let kind = weight_kind(t.ggml_type).ok_or_else(|| {
            format!("DSpark {name}: ggml type {} has no kernel", t.ggml_type)
        })?;
        Ok(Self {
            kind: kind as u32,
            off: t.offset,
        })
    }

    /// A norm's weight vector: the norm kernels read F32.
    fn norm(t: &DraftTensor, name: &str) -> Result<u64, String> {
        if weight_kind(t.ggml_type) != Some(WeightKind::F32) {
            return Err(format!(
                "DSpark {name}: a norm weight must be F32, the file has ggml type {}",
                t.ggml_type
            ));
        }
        Ok(t.offset)
    }
}

struct Layer {
    attn_norm: u64,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    q_norm: u64,
    k_norm: u64,
    ffn_norm: u64,
    gate: Weight,
    up: Weight,
    down: Weight,
}

/// One drafted block, as read back after the drafter's forward.
#[derive(Clone, Debug)]
pub struct DraftBlock {
    /// The Markov chain's pick at each drafted position.
    pub ids: Vec<u32>,
    /// Candidates per position; 0 when the drafter runs without candidates, and then the
    /// candidates and confidences below are empty.
    pub candidates: usize,
    /// Position `i`'s candidates at `[i * candidates..]`: the largest entries of its biased column
    /// (its head logits plus the Markov bias of the chain's previous id), largest first and the
    /// smaller id first among equal values. Entry 0 is the chain's pick.
    pub candidate_ids: Vec<u32>,
    /// Those candidates' biased values.
    pub candidate_values: Vec<f32>,
    /// The confidence head at each position: `sigmoid(w . [h ; W1[previous id]] + b)`, `h` the
    /// position's row after the output norm.
    pub confidence: Vec<f32>,
}

/// A DSpark drafter bound to its target: the weights it reads and the geometry it runs at.
pub struct DsparkForward {
    hidden: u32,
    ffn: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    block: u32,
    vocab: u32,
    rank: u32,
    feature_width: u32,
    mask_token: u32,
    eps: f32,
    rope_theta: f32,
    /// The drafter's first layer in the per-layer KV arrays, which follow the target's.
    first_kv_layer: u32,
    embedding: Weight,
    head: Weight,
    fc: Weight,
    enc_norm: u64,
    out_norm: u64,
    markov1: Weight,
    markov2: Weight,
    /// Candidates kept per drafted position; 0 runs neither the candidates nor the confidence
    /// head.
    candidates: u32,
    /// The confidence head as floats: its `hidden + rank` weights, then its bias. The device
    /// copy heads `DraftConfidence`.
    confidence: Vec<f32>,
    layers: Vec<Layer>,
}

impl DsparkForward {
    /// Binds the drafter to a target whose plan carries it, allocates the drafter's own
    /// vocabulary-wide buffers, and keeps the target's activation layout at least as wide as
    /// a verify of the block (anchor and drafts).
    ///
    /// # Errors
    /// When the target has no device forward or does not carry this drafter, the drafter's
    /// rows do not fit the target's buffers, the backend cannot serve its attention or its
    /// weights, or an allocation fails.
    pub fn attach<M: crate::Model + ?Sized>(
        target: &mut M,
        d: &DsparkDescriptor,
    ) -> Result<Self, String> {
        if target.state().host_forward {
            return Err("DSpark drafter: the target has no device forward".into());
        }
        let plan = target.plan();
        if plan.drafter.as_ref() != Some(&d.drafter_plan()) {
            return Err(
                "DSpark drafter: the target's plan does not carry this drafter".into(),
            );
        }
        let c = &plan.config;
        // The drafter gathers the target's embedding, projects with its head and runs on its
        // per-row scratch: its residual is the target's, and no other row may be wider.
        if d.hidden != c.n_embd || d.target_hidden != c.n_embd {
            return Err(format!(
                "DSpark drafter: hidden {} and features {} wide, the target's residual is {}",
                d.hidden, d.target_hidden, c.n_embd
            ));
        }
        let attention_hd = plan
            .layers
            .iter()
            .filter(|l| l.attention.is_attention())
            .map(|l| l.attention.head_dim())
            .max()
            .unwrap_or(0);
        let ffn_width = plan
            .layers
            .iter()
            .map(|l| l.ffn.max_hidden())
            .max()
            .unwrap_or(0)
            .max(c.n_ff);
        if d.heads * d.head_dim > c.n_heads * attention_hd
            || d.kv_heads * d.head_dim > c.n_kv_heads * attention_hd
            || d.ffn > ffn_width
        {
            return Err(
                "DSpark drafter: its attention or FFN rows are wider than the target's buffers"
                    .into(),
            );
        }
        let block = d.block_size;
        if !(2..=ROW_LAYOUT_MAX_ROWS as u32).contains(&block) {
            return Err(format!(
                "DSpark drafter: a block of {block} rows; the row-layout attention serves 2 to {ROW_LAYOUT_MAX_ROWS}"
            ));
        }
        let be = crate::gpu_support::be();
        if !be.supports_row_layout(d.head_dim) {
            return Err(format!(
                "DSpark drafter: this backend has no row-layout attention at head dim {}",
                d.head_dim
            ));
        }
        let candidates = candidates_per_position();
        if candidates > d.vocab {
            return Err(format!(
                "DSpark drafter: {candidates} candidates per position, the vocabulary holds {}",
                d.vocab
            ));
        }
        if candidates > 0
            && (!be.supports_top_k_rows(candidates) || !be.supports_logistic_rows())
        {
            return Err(format!(
                "DSpark drafter: this backend has no top-{candidates} row entry or no logistic row entry"
            ));
        }
        let confidence = d.confidence_head(target.weights())?;
        if crate::kv::KvType::k() != crate::kv::KvType::F16
            || crate::kv::KvType::v() != crate::kv::KvType::F16
        {
            return Err(
                "DSpark drafter: the drafter's caches are f16 only so far".into()
            );
        }
        let layers = d
            .layers
            .iter()
            .map(|l| {
                Ok(Layer {
                    attn_norm: Weight::norm(&l.attn_norm, "attn_norm")?,
                    q: Weight::of(&l.q, "attn_q")?,
                    k: Weight::of(&l.k, "attn_k")?,
                    v: Weight::of(&l.v, "attn_v")?,
                    o: Weight::of(&l.o, "attn_output")?,
                    q_norm: Weight::norm(&l.q_norm, "attn_q_norm")?,
                    k_norm: Weight::norm(&l.k_norm, "attn_k_norm")?,
                    ffn_norm: Weight::norm(&l.ffn_norm, "ffn_norm")?,
                    gate: Weight::of(&l.gate, "ffn_gate")?,
                    up: Weight::of(&l.up, "ffn_up")?,
                    down: Weight::of(&l.down, "ffn_down")?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let drafter = Self {
            hidden: d.hidden,
            ffn: d.ffn,
            heads: d.heads,
            kv_heads: d.kv_heads,
            head_dim: d.head_dim,
            block,
            vocab: d.vocab,
            rank: d.rank,
            feature_width: u32::try_from(d.target_layers.len())
                .ok()
                .and_then(|n| n.checked_mul(d.target_hidden))
                .ok_or("DSpark drafter: feature width overflows")?,
            mask_token: d.mask_token,
            eps: d.eps,
            rope_theta: d.rope_theta,
            first_kv_layer: c.n_layers,
            embedding: Weight::of(&d.embedding, "token_embd")?,
            head: Weight::of(&d.head, "output")?,
            fc: Weight::of(&d.fc, "fc")?,
            enc_norm: Weight::norm(&d.enc_norm, "enc.output_norm")?,
            out_norm: Weight::norm(&d.out_norm, "output_norm")?,
            markov1: Weight::of(&d.markov1, "markov_w1")?,
            markov2: Weight::of(&d.markov2, "markov_w2")?,
            candidates,
            confidence,
            layers,
        };
        for (id, elems) in [
            (BufId::DraftLogits, u64::from(block) * u64::from(d.vocab)),
            (BufId::DraftColumn, u64::from(d.vocab)),
            (BufId::DraftBias, u64::from(d.vocab)),
            (BufId::DraftRank, u64::from(block) * (u64::from(d.rank) + 1)),
            (
                BufId::DraftConfidence,
                drafter.confidence.len() as u64
                    + be.logistic_rows_len(d.hidden, d.rank, block),
            ),
            (
                BufId::RowLayout,
                (ROW_LAYOUT_MAX_ROWS * ROW_LAYOUT_WORDS) as u64,
            ),
        ] {
            be.alloc(id, elems * 4)
                .map_err(|rc| format!("DSpark drafter: {id:?} allocation rc={rc}"))?;
        }
        if candidates > 0 {
            let elems = be.top_k_rows_len(d.vocab, block, candidates);
            be.alloc(BufId::DraftTop, elems * 4)
                .map_err(|rc| format!("DSpark drafter: DraftTop allocation rc={rc}"))?;
        }
        be.write(BufId::DraftConfidence, 0, &drafter.confidence);
        let state = target.state_mut();
        state.activation_floor_rows =
            state.activation_floor_rows.max(block as usize + 1);
        Ok(drafter)
    }

    /// Rows per drafted block, the anchor included.
    #[must_use]
    pub fn block_size(&self) -> u32 {
        self.block
    }

    /// The tapped features of the first `n` rows of the target's last forward, at positions
    /// `start..start + n`, into the drafter's caches. The caller checks that the last forward
    /// started at `start` with at least `n` rows (`FeatureTaps::check_rows`): the target's
    /// buffers then hold those rows.
    ///
    /// # Errors
    /// When `n` is 0 or the backend fails.
    pub fn append(&self, start: u32, n: u32) -> Result<(), String> {
        if n == 0 {
            return Err("DSpark append: no rows".into());
        }
        let be = crate::gpu_support::be();
        let (h, kw) = (self.hidden, self.kv_heads * self.head_dim);
        be.begin();
        be.matmat(
            self.fc.kind,
            self.fc.off,
            self.feature_width,
            h,
            BufId::DraftFeatures,
            BufId::X,
            n,
        );
        be.rms_norm(BufId::X, self.enc_norm, h, self.eps, n, h, 0);
        for (i, l) in self.layers.iter().enumerate() {
            let layer = self.first_kv_layer + i as u32;
            be.matmat(l.k.kind, l.k.off, h, kw, BufId::X, BufId::K, n);
            be.matmat(l.v.kind, l.v.off, h, kw, BufId::X, BufId::V, n);
            be.head_norm_rope_hadamard(
                BufId::K,
                l.k_norm,
                self.head_dim,
                self.eps,
                self.kv_heads,
                start,
                n,
                self.head_dim,
                self.rope_theta,
                None,
                0,
            );
            be.kv_store(BufId::K, layer, kw, start, n, false, 0);
            be.kv_store(BufId::V, layer, kw, start, n, true, 0);
        }
        be.end()
            .map_err(|rc| format!("DSpark append failed rc={rc}"))?;
        if let Some(file) = context_dump() {
            // `BufId::X` still holds the rows after `enc_norm`: the layers only read it.
            let mut rows = vec![0.0_f32; n as usize * h as usize];
            be.read(BufId::X, 0, &mut rows);
            write_context_record(file, [1, start, n, h], &rows)?;
        }
        Ok(())
    }

    /// One block at `start`: the anchor and `block - 1` mask tokens through the drafter, then
    /// the Markov head's chain on the device, `block` drafted ids read back once. The block's
    /// attention reads the committed cache from `floor` up: 0 for a history that starts at the
    /// prompt, the restore point for one rebuilt there (design 5.5), whose cache holds nothing of
    /// this request below it.
    ///
    /// `IMPARO_DSPARK_TOP3=1` also prints each position's top 3 with their probabilities over
    /// the top 10, the numbers upstream's `-lv 5` log prints. `IMPARO_DSPARK_CANDIDATES=1` prints
    /// each position's candidates and confidence, and `IMPARO_DSPARK_CHECK=1` checks both against
    /// a host computation; both need the candidates on (`IMPARO_DSPARK_TOPK=K`).
    ///
    /// # Errors
    /// When the block does not fit the target's buffers or cache, the backend cannot serve an
    /// op, or it fails.
    pub fn generate<M: crate::Model + ?Sized>(
        &self,
        target: &mut M,
        start: u32,
        anchor: u32,
        floor: u32,
    ) -> Result<DraftBlock, String> {
        let m = self.block;
        let rows = target.state().gpu_batch;
        if m as usize > rows {
            return Err(format!(
                "DSpark generate: a block of {m} rows, the target's buffers hold {rows}"
            ));
        }
        if anchor >= self.vocab {
            return Err(format!(
                "DSpark generate: anchor {anchor} past the vocabulary"
            ));
        }
        let end = start as usize + m as usize;
        let capacity = target.state().kv_rt.capacity;
        if end > capacity {
            return Err(format!(
                "DSpark generate: the block ends at {end}, past the KV capacity {capacity}"
            ));
        }
        target.kv_fit(end)?;
        let mut tokens = vec![self.mask_token; m as usize];
        tokens[0] = anchor;
        let layout = block_row_layout(start, m as usize);
        let top_probe = top_probe_on();
        if (candidates_probe_on() || check_on()) && self.candidates == 0 {
            return Err(
                "IMPARO_DSPARK_CHECK and IMPARO_DSPARK_CANDIDATES need IMPARO_DSPARK_TOPK=K".into(),
            );
        }
        let be = crate::gpu_support::be();
        be.begin();
        let encoded = self.encode_block(
            be,
            start,
            floor,
            &tokens,
            &layout,
            top_probe,
            attn_probe_on(),
        );
        let ended = be
            .end()
            .map_err(|rc| format!("DSpark generate failed rc={rc}"));
        encoded?;
        ended?;
        let (rows, k) = (m as usize, self.candidates as usize);
        let mut picks = vec![0.0_f32; rows];
        be.read(
            BufId::DraftRank,
            u64::from(m) * u64::from(self.rank),
            &mut picks,
        );
        let mut top = vec![0.0_f32; 2 * rows * k];
        be.read(BufId::DraftTop, 0, &mut top);
        let mut confidence = vec![0.0_f32; rows];
        be.read(
            BufId::DraftConfidence,
            self.confidence.len() as u64,
            &mut confidence,
        );
        let (top_ids, top_values) = top.split_at(rows * k);
        let block = DraftBlock {
            ids: picks.iter().map(|p| p.to_bits()).collect(),
            candidates: k,
            candidate_ids: top_ids.iter().map(|x| x.to_bits()).collect(),
            candidate_values: top_values.to_vec(),
            confidence,
        };
        if block
            .ids
            .iter()
            .chain(&block.candidate_ids)
            .any(|&t| t >= self.vocab)
        {
            return Err("DSpark generate: a drafted id past the vocabulary".into());
        }
        if block.confidence.iter().any(|c| !(0.0..=1.0).contains(c)) {
            return Err("DSpark generate: a confidence outside 0..1".into());
        }
        if top_probe {
            self.print_top(be, start);
        }
        if candidates_probe_on() {
            Self::print_candidates(start, &block);
        }
        if check_on() {
            self.check_block(be, start, &block);
        }
        if let Some(file) = hidden_dump() {
            let file = file.as_ref().map_err(Clone::clone)?;
            self.dump_hidden(be, file, start, anchor, &block)?;
        }
        if let Some(file) = context_dump() {
            write_context_record(file, [2, start, anchor, self.block], &[])?;
        }
        Ok(block)
    }

    /// One record per drafted block for an OFFLINE replay of what online training of the drafter's
    /// output side would have done: an adapter on the rows the head reads, overlays on the Markov
    /// tables, a refit confidence head. The replay rebuilds every biased column from these rows and
    /// the file's weights, so a record carries only what the weights cannot give back.
    ///
    /// ```text
    ///   u32 x 6          start, anchor, rows m, hidden, rank, candidates k
    ///   u32 x m          the chain's picks
    ///   f32 x m          the confidence head
    ///   u32 x m*k        candidate ids          f32 x m*k   their biased values
    ///   f32 x m*hidden   the rows after the output norm: `BufId::Cur`, which nothing writes after
    ///                    the norm inside the block's region (the head, the chain, the candidates
    ///                    and the confidence head only read it)
    /// ```
    fn dump_hidden(
        &self,
        be: &dyn Backend,
        file: &std::sync::Mutex<std::fs::File>,
        start: u32,
        anchor: u32,
        block: &DraftBlock,
    ) -> Result<(), String> {
        use std::io::Write;
        let (m, h) = (self.block as usize, self.hidden as usize);
        let mut rows = vec![0.0_f32; m * h];
        be.read(BufId::Cur, 0, &mut rows);
        let k = u32::try_from(block.candidates)
            .map_err(|_| "DSpark hidden dump: k overflows")?;
        let mut out: Vec<u8> =
            Vec::with_capacity(4 * (6 + 2 * m + 2 * m * block.candidates + m * h));
        for v in [start, anchor, self.block, self.hidden, self.rank, k] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for w in &block.ids {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for c in &block.confidence {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for w in &block.candidate_ids {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for v in &block.candidate_values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for x in &rows {
            out.extend_from_slice(&x.to_le_bytes());
        }
        file.lock()
            .map_err(|_| "DSpark hidden dump: the file's lock is poisoned".to_string())?
            .write_all(&out)
            .map_err(|e| format!("DSpark hidden dump: {e}"))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_block(
        &self,
        be: &dyn Backend,
        start: u32,
        floor: u32,
        tokens: &[u32],
        layout: &[u32],
        probe: bool,
        attn_probe: bool,
    ) -> Result<(), String> {
        let (m, h) = (self.block, self.hidden);
        let (qw, kw) = (self.heads * self.head_dim, self.kv_heads * self.head_dim);
        be.write_u32(BufId::Tokens, 0, tokens);
        be.write_u32(BufId::RowLayout, 0, layout);
        if !be.gather_rows(
            self.embedding.kind,
            self.embedding.off,
            h,
            self.vocab,
            1.0,
            BufId::X,
            0,
            BufId::Tokens,
            m,
        ) {
            return Err(
                "DSpark generate: no row gather for the embedding's kind".into()
            );
        }
        for (i, l) in self.layers.iter().enumerate() {
            let layer = self.first_kv_layer + i as u32;
            be.rms_norm_from(BufId::Cur, BufId::X, l.attn_norm, h, self.eps, m, h, 0);
            be.matmat(l.q.kind, l.q.off, h, qw, BufId::Cur, BufId::Q, m);
            be.matmat(l.k.kind, l.k.off, h, kw, BufId::Cur, BufId::K, m);
            be.matmat(l.v.kind, l.v.off, h, kw, BufId::Cur, BufId::V, m);
            for (buf, norm, heads) in [
                (BufId::Q, l.q_norm, self.heads),
                (BufId::K, l.k_norm, self.kv_heads),
            ] {
                be.head_norm_rope_hadamard(
                    buf,
                    norm,
                    self.head_dim,
                    self.eps,
                    heads,
                    start,
                    m,
                    self.head_dim,
                    self.rope_theta,
                    None,
                    0,
                );
            }
            be.kv_store(BufId::K, layer, kw, start, m, false, 0);
            be.kv_store(BufId::V, layer, kw, start, m, true, 0);
            let host_attn = host_attn_mode();
            let probe_q = ((attn_probe && i == 0) || host_attn.is_some()).then(|| {
                let _ = be.end();
                let mut q = vec![0.0_f32; (m * qw) as usize];
                be.read(BufId::Q, 0, &mut q);
                be.begin();
                q
            });
            if !be.attention_rows(
                layer,
                self.head_dim,
                self.heads,
                self.kv_heads,
                kw,
                start,
                floor,
                1.0 / (self.head_dim as f32).sqrt(),
                m,
                dspark_float_q(),
            ) {
                return Err(format!(
                    "DSpark generate: row-layout attention not served at drafter layer {i}"
                ));
            }
            if let Some(q) = probe_q {
                if attn_probe && i == 0 {
                    self.probe_attention(be, layer, start, floor, &q);
                }
                if let Some(round_q) = host_attn {
                    let out: Vec<f32> = self
                        .host_attention(be, layer, start, floor, &q, round_q)
                        .into_iter()
                        .map(|x| x as f32)
                        .collect();
                    be.write(BufId::Attn, 0, &out);
                    eprintln!(
                        "[imparo] dspark host attention start={start} layer={layer} q={}",
                        if round_q { "half" } else { "float" }
                    );
                }
            }
            be.matmat(l.o.kind, l.o.off, qw, h, BufId::Attn, BufId::O, m);
            be.add(BufId::X, BufId::O, m * h);
            be.rms_norm_from(BufId::Cur, BufId::X, l.ffn_norm, h, self.eps, m, h, 0);
            let fused = be.ffn_gated_down(
                l.gate.kind,
                l.gate.off,
                l.up.kind,
                l.up.off,
                l.down.kind,
                l.down.off,
                h,
                self.ffn,
                h,
                BufId::Cur,
                BufId::G,
                BufId::O,
                m,
            );
            if !fused {
                if !be.matmat_gated(
                    l.gate.kind,
                    l.gate.off,
                    l.up.kind,
                    l.up.off,
                    h,
                    self.ffn,
                    BufId::Cur,
                    BufId::G,
                    BufId::U,
                    m,
                ) {
                    be.matmat(
                        l.gate.kind,
                        l.gate.off,
                        h,
                        self.ffn,
                        BufId::Cur,
                        BufId::G,
                        m,
                    );
                    be.matmat(
                        l.up.kind,
                        l.up.off,
                        h,
                        self.ffn,
                        BufId::Cur,
                        BufId::U,
                        m,
                    );
                    be.act_mul(BufId::G, BufId::U, m * self.ffn);
                }
                be.matmat(l.down.kind, l.down.off, self.ffn, h, BufId::G, BufId::O, m);
            }
            be.add(BufId::X, BufId::O, m * h);
        }
        be.rms_norm_from(BufId::Cur, BufId::X, self.out_norm, h, self.eps, m, h, 0);
        be.matmat(
            self.head.kind,
            self.head.off,
            h,
            self.vocab,
            BufId::Cur,
            BufId::DraftLogits,
            m,
        );
        // The Markov chain: position i's column is its logits plus W2 . W1[previous id], and
        // its argmax is the next position's previous id. Tokens[0] holds the anchor first and
        // then each pick; DraftRank keeps position i's rank row at row i and the picks after
        // the block's rows. With candidates on, or the TOP3 probe, each biased column is written
        // back over its head logits.
        for i in 0..m {
            if !be.gather_rows(
                self.markov1.kind,
                self.markov1.off,
                self.rank,
                self.vocab,
                1.0,
                BufId::DraftRank,
                i * self.rank,
                BufId::Tokens,
                1,
            ) {
                return Err(
                    "DSpark generate: no row gather for the Markov table's kind".into(),
                );
            }
            be.matmat_from(
                self.markov2.kind,
                self.markov2.off,
                self.rank,
                self.vocab,
                BufId::DraftRank,
                BufId::DraftBias,
                1,
                i,
            );
            be.copy_range(
                BufId::DraftColumn,
                0,
                BufId::DraftLogits,
                i * self.vocab,
                self.vocab,
            );
            be.add(BufId::DraftColumn, BufId::DraftBias, self.vocab);
            be.argmax(BufId::DraftColumn, BufId::Tokens, self.vocab);
            be.copy_range(BufId::DraftRank, m * self.rank + i, BufId::Tokens, 0, 1);
            if probe || self.candidates > 0 {
                be.copy_range(
                    BufId::DraftLogits,
                    i * self.vocab,
                    BufId::DraftColumn,
                    0,
                    self.vocab,
                );
            }
        }
        if self.candidates > 0 {
            // The candidates: the largest entries of every position's biased column, in one pass
            // after the chain (design 6.1, level 1).
            if !be.top_k_rows(
                BufId::DraftLogits,
                BufId::DraftTop,
                self.vocab,
                m,
                self.candidates,
            ) {
                return Err(
                    "DSpark generate: the backend refused the candidates".into()
                );
            }
            // The confidence head: position i's row after the output norm beside its rank row.
            let (h, rank) = (self.hidden, self.rank);
            if !be.logistic_rows(
                BufId::Cur,
                h,
                BufId::DraftRank,
                rank,
                BufId::DraftConfidence,
                0,
                BufId::DraftConfidence,
                h + rank + 1,
                m,
            ) {
                return Err(
                    "DSpark generate: the backend refused the confidence head".into()
                );
            }
        }
        Ok(())
    }

    /// One drafter layer's attention for the block at `start`, recomputed on the host in f64 from
    /// the Q rows and the layer's f16 cache from `floor` up, laid out as `BufId::Attn` holds it:
    /// from Q as the host holds it, or from Q rounded to half as the FA entry stages it. The cache
    /// is read by region offset, not through the block table, so this holds only while the
    /// drafter's block table is the identity (the replay's is).
    fn host_attention(
        &self,
        be: &dyn Backend,
        layer: u32,
        start: u32,
        floor: u32,
        q: &[f32],
        round_q: bool,
    ) -> Vec<f64> {
        let (m, hd) = (self.block as usize, self.head_dim as usize);
        let (heads, kvh) = (self.heads as usize, self.kv_heads as usize);
        let keys = start as usize + m;
        let _ = be.end();
        let mut kb = vec![0_u8; keys * kvh * hd * 2];
        let mut vb = vec![0_u8; keys * kvh * hd * 2];
        be.read_kv_bytes(layer, false, 0, &mut kb);
        be.read_kv_bytes(layer, true, 0, &mut vb);
        be.begin();
        let half = |b: &[u8], i: usize| {
            half_to_f64(u16::from_le_bytes([b[2 * i], b[2 * i + 1]]))
        };
        let scale = 1.0 / (hd as f64).sqrt();
        let group = heads / kvh;
        let mut out = vec![0.0_f64; m * heads * hd];
        let mut weights = vec![0.0_f64; keys];
        for t in 0..m {
            for h in 0..heads {
                let g = h / group;
                let qv = &q[(t * heads + h) * hd..(t * heads + h + 1) * hd];
                for (j, w) in weights.iter_mut().enumerate() {
                    if j < floor as usize {
                        *w = f64::NEG_INFINITY;
                        continue;
                    }
                    *w = (0..hd)
                        .map(|d| {
                            let qd = if round_q {
                                f32_to_half_f64(qv[d])
                            } else {
                                f64::from(qv[d])
                            };
                            qd * half(&kb, (j * kvh + g) * hd + d)
                        })
                        .sum::<f64>()
                        * scale;
                }
                let top = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let z: f64 = weights.iter().map(|s| (s - top).exp()).sum();
                for w in &mut weights {
                    *w = (*w - top).exp() / z;
                }
                let row = &mut out[(t * heads + h) * hd..(t * heads + h + 1) * hd];
                for (d, o) in row.iter_mut().enumerate() {
                    *o = weights
                        .iter()
                        .enumerate()
                        .map(|(j, w)| w * half(&vb, (j * kvh + g) * hd + d))
                        .sum();
                }
            }
        }
        out
    }

    /// `IMPARO_DSPARK_ATTN_PROBE=1`: the first drafter layer's attention on the device against
    /// `host_attention`, from Q as the host holds it and from Q rounded to half.
    fn probe_attention(
        &self,
        be: &dyn Backend,
        layer: u32,
        start: u32,
        floor: u32,
        q: &[f32],
    ) {
        let (m, hd, heads) = (
            self.block as usize,
            self.head_dim as usize,
            self.heads as usize,
        );
        let _ = be.end();
        let mut attn = vec![0.0_f32; m * heads * hd];
        be.read(BufId::Attn, 0, &mut attn);
        be.begin();
        let float_q = self.host_attention(be, layer, start, floor, q, false);
        let half_q = self.host_attention(be, layer, start, floor, q, true);
        let (mut worst, mut worst_at, mut worst_half_q, mut max_abs) =
            (0.0_f64, 0, 0.0_f64, 0.0_f64);
        for (i, &a) in attn.iter().enumerate() {
            let e = (f64::from(a) - float_q[i]).abs();
            if e > worst {
                worst = e;
                worst_at = i;
            }
            worst_half_q = worst_half_q.max((f64::from(a) - half_q[i]).abs());
            max_abs = max_abs.max(float_q[i].abs());
        }
        eprintln!(
            "[imparo] dspark attn probe start={start} start_mod8={} layer={layer} keys={} \
             max_abs_diff={worst:.3e} at row {} head {} max_abs_diff_half_q={worst_half_q:.3e} \
             max_abs={max_abs:.3e}",
            start % 8,
            start as usize + m,
            worst_at / (heads * hd),
            worst_at / hd % heads
        );
    }

    /// `IMPARO_DSPARK_TOP3=1`: each position's biased logits (the chain wrote them back), top 3
    /// with probabilities over the top 10, and the logit gap between the first two.
    fn print_top(&self, be: &dyn Backend, start: u32) {
        let v = self.vocab as usize;
        let mut logits = vec![0.0_f32; self.block as usize * v];
        be.read(BufId::DraftLogits, 0, &mut logits);
        for (pos, row) in logits.chunks_exact(v).enumerate() {
            let mut order: Vec<usize> = (0..v).collect();
            order.select_nth_unstable_by(9, |&a, &b| {
                row[b].total_cmp(&row[a]).then(a.cmp(&b))
            });
            order.truncate(10);
            order.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            let top = row[order[0]];
            let norm: f32 = order.iter().map(|&i| (row[i] - top).exp()).sum();
            let shown: Vec<String> = order
                .iter()
                .take(3)
                .map(|&i| format!("{i}:{:.6}", (row[i] - top).exp() / norm))
                .collect();
            eprintln!(
                "[imparo] dspark top start={start} pos={pos} {}",
                shown.join(" ")
            );
            eprintln!(
                "[imparo] dspark gap start={start} pos={pos} logit_gap={:.6}",
                top - row[order[1]]
            );
        }
    }

    /// `IMPARO_DSPARK_CANDIDATES=1`: each position's pick, confidence, candidates and their biased
    /// values.
    fn print_candidates(start: u32, block: &DraftBlock) {
        let k = block.candidates;
        let rows = block
            .candidate_ids
            .chunks_exact(k)
            .zip(block.candidate_values.chunks_exact(k));
        for (pos, (ids, values)) in rows.enumerate() {
            eprintln!(
                "[imparo] dspark candidates start={start} pos={pos} pick={} conf={} ids={} values={}",
                block.ids[pos],
                block.confidence[pos],
                csv(ids),
                csv(values)
            );
        }
    }

    /// `IMPARO_DSPARK_CHECK=1`: the block's candidates against a host sort of the read-back biased
    /// columns (ids and value bits), each position's pick against its first candidate, and its
    /// confidence against a host computation in f64 from the read-back rows and the host copy of
    /// the head.
    fn check_block(&self, be: &dyn Backend, start: u32, block: &DraftBlock) {
        let (m, v, k) = (self.block as usize, self.vocab as usize, block.candidates);
        let (h, r) = (self.hidden as usize, self.rank as usize);
        let mut logits = vec![0.0_f32; m * v];
        be.read(BufId::DraftLogits, 0, &mut logits);
        let mut normed = vec![0.0_f32; m * h];
        be.read(BufId::Cur, 0, &mut normed);
        let mut rank_rows = vec![0.0_f32; m * r];
        be.read(BufId::DraftRank, 0, &mut rank_rows);
        let (w, bias) = (&self.confidence[..h + r], f64::from(self.confidence[h + r]));
        let (mut differ, mut picks_differ, mut conf_worst) =
            (0_usize, 0_usize, 0.0_f64);
        for pos in 0..m {
            let row = &logits[pos * v..(pos + 1) * v];
            let order = largest_first(row, k);
            let ids = &block.candidate_ids[pos * k..(pos + 1) * k];
            let values = &block.candidate_values[pos * k..(pos + 1) * k];
            if order
                .iter()
                .zip(ids.iter().zip(values))
                .any(|(&i, (&id, value))| {
                    i as u32 != id || row[i].to_bits() != value.to_bits()
                })
            {
                differ += 1;
            }
            if ids[0] != block.ids[pos] {
                picks_differ += 1;
            }
            let features = normed[pos * h..(pos + 1) * h]
                .iter()
                .chain(&rank_rows[pos * r..(pos + 1) * r]);
            let z = features
                .zip(w)
                .map(|(x, w)| f64::from(*x) * f64::from(*w))
                .sum::<f64>()
                + bias;
            let host = 1.0 / (1.0 + (-z).exp());
            conf_worst =
                conf_worst.max((host - f64::from(block.confidence[pos])).abs());
        }
        let verdict = if differ == 0 && picks_differ == 0 && conf_worst <= 1e-5 {
            "PASS"
        } else {
            "FAIL"
        };
        eprintln!(
            "[imparo] dspark check start={start} rows={m} k={k} candidate_rows_differ={differ} picks_differ={picks_differ} conf_worst={conf_worst:.3e} verdict={verdict}"
        );
    }
}

/// `IMPARO_DSPARK_HOST_ATTN=float|half`: every drafter layer's attention output replaced by
/// `host_attention` (Q as held, or rounded to half), to see what the FA entry's half Q moves.
fn host_attn_mode() -> Option<bool> {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("IMPARO_DSPARK_HOST_ATTN").as_deref() {
            Ok("float") => Some(false),
            Ok("half") => Some(true),
            _ => None,
        },
    )
}

/// `IMPARO_DSPARK_FLOAT_Q=1`: the drafter's block attention keeps Q in float instead of rounding it
/// to half. Off by default; the step 3d evidence records what it moves and costs.
fn dspark_float_q() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_FLOAT_Q").as_deref() == Ok("1"))
}

fn attn_probe_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_ATTN_PROBE").as_deref() == Ok("1"))
}

/// `x` rounded to the nearest half (ties to even), as a float-to-half conversion stores it.
fn f32_to_half_f64(x: f32) -> f64 {
    let v = f64::from(x);
    let a = v.abs();
    if a == 0.0 || !a.is_finite() {
        return v;
    }
    if a >= 65520.0 {
        return v.signum() * f64::INFINITY;
    }
    // The step is 2^(e - 10) for a normal half of exponent e, 2^-24 below the normal range.
    let e = a.log2().floor().max(-14.0);
    let step = 2f64.powf(e - 10.0);
    v.signum() * (a / step).round_ties_even() * step
}

fn half_to_f64(bits: u16) -> f64 {
    let sign = if bits >> 15 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let mant = f64::from(bits & 0x3ff);
    match exp {
        0 => sign * mant * 2f64.powi(-24),
        31 => f64::NAN,
        _ => sign * (1.0 + mant / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// The indices of the `k` largest values of `row`, larger first and the smaller index first among
/// equal values: the order `top_k_rows` writes. `k` is at least 1.
fn largest_first(row: &[f32], k: usize) -> Vec<usize> {
    let order_of = |a: &usize, b: &usize| {
        row[*b]
            .partial_cmp(&row[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    };
    let mut order: Vec<usize> = (0..row.len()).collect();
    if k < order.len() {
        order.select_nth_unstable_by(k - 1, order_of);
        order.truncate(k);
    }
    order.sort_unstable_by(order_of);
    order
}

/// Comma-separated, for the probes' lines.
fn csv<T: ToString>(items: impl IntoIterator<Item = T>) -> String {
    items
        .into_iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// `IMPARO_DSPARK_CANDIDATES=1`: `print_candidates` after each block.
fn candidates_probe_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_CANDIDATES").as_deref() == Ok("1"))
}

/// `IMPARO_DSPARK_CHECK=1`: `check_block` after each block.
fn check_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_CHECK").as_deref() == Ok("1"))
}

/// The file `IMPARO_DSPARK_HIDDEN_DUMP=PATH` names, appended to by `dump_hidden` after each block.
/// A path that cannot be opened fails every block rather than leaving a run without its records.
type HiddenDump = Result<std::sync::Mutex<std::fs::File>, String>;

fn hidden_dump() -> Option<&'static HiddenDump> {
    static FILE: std::sync::OnceLock<Option<HiddenDump>> = std::sync::OnceLock::new();
    FILE.get_or_init(|| {
        let path = std::env::var("IMPARO_DSPARK_HIDDEN_DUMP").ok()?;
        Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map(std::sync::Mutex::new)
                .map_err(|e| format!("IMPARO_DSPARK_HIDDEN_DUMP={path}: {e}")),
        )
    })
    .as_ref()
}

/// The file `IMPARO_DSPARK_CONTEXT_DUMP=PATH` names: the drafter's inputs in call order, for an
/// OFFLINE replay of training the drafter's layers. With the hidden dump's records (one per block,
/// in the same order as the blocks here) and the weights, a replay rebuilds every block's forward.
///
/// ```text
///   u32 1, start, n, hidden     f32 x n*hidden   an `append`: the rows after `enc_norm` it stores at
///                                                start..start + n (a later append may overwrite them)
///   u32 2, start, anchor, block                  a `generate`, after its block was drafted
/// ```
fn context_dump() -> Option<&'static HiddenDump> {
    static FILE: std::sync::OnceLock<Option<HiddenDump>> = std::sync::OnceLock::new();
    FILE.get_or_init(|| {
        let path = std::env::var("IMPARO_DSPARK_CONTEXT_DUMP").ok()?;
        Some(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map(std::sync::Mutex::new)
                .map_err(|e| format!("IMPARO_DSPARK_CONTEXT_DUMP={path}: {e}")),
        )
    })
    .as_ref()
}

fn write_context_record(
    file: &HiddenDump,
    head: [u32; 4],
    rows: &[f32],
) -> Result<(), String> {
    use std::io::Write;
    let file = file.as_ref().map_err(Clone::clone)?;
    let mut out: Vec<u8> = Vec::with_capacity(16 + 4 * rows.len());
    for v in head {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for x in rows {
        out.extend_from_slice(&x.to_le_bytes());
    }
    file.lock()
        .map_err(|_| "DSpark context dump: the file's lock is poisoned".to_string())?
        .write_all(&out)
        .map_err(|e| format!("DSpark context dump: {e}"))
}

/// The top-K a tree is built from when `IMPARO_DSPARK_TOPK` does not say: the K every tree
/// measurement was taken at, and the key the acceptance model's stored state is read under.
const DEFAULT_TOPK: u32 = 8;

/// Candidates per drafted position: `IMPARO_DSPARK_TOPK=K`, else `DEFAULT_TOPK` whenever a tree is
/// built -- the default -- and 0 for a chain verify, which reads none of them (at 8 the two
/// entries cost about 0.7 ms per drafted block).
///
/// A default of 0 under a tree made every speculative request fail ("a tree needs
/// IMPARO_DSPARK_TOPK of 2 or more"): the budget is the default, so the default had no tree.
pub(crate) fn candidates_per_position() -> u32 {
    std::env::var("IMPARO_DSPARK_TOPK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(match crate::dspark::tree_width() {
            Ok(None) => 0,
            _ => DEFAULT_TOPK,
        })
}

fn top_probe_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IMPARO_DSPARK_TOP3").as_deref() == Ok("1"))
}

/// Row-layout words for a drafted block at `start`: every row sees the committed cache below
/// `start` and every row of the block, since the drafter's attention is not causal. Depth and
/// ancestors are those of a chain; the attention reads only the position mask.
#[must_use]
pub fn block_row_layout(start: u32, rows: usize) -> Vec<u32> {
    let seen: u64 = if rows >= 64 {
        u64::MAX
    } else {
        (1_u64 << rows) - 1
    };
    let mut words = vec![0_u32; rows * ROW_LAYOUT_WORDS];
    for (t, row) in words.chunks_exact_mut(ROW_LAYOUT_WORDS).enumerate() {
        row[0] = start + t as u32;
        row[1] = t as u32;
        row[2] = seen as u32;
        row[3] = (seen >> 32) as u32;
        for k in 1..=ROW_LAYOUT_ANCESTORS.min(t) {
            row[3 + k] = (t - k) as u32;
        }
    }
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_row_sees_every_row_of_the_block() {
        let words = block_row_layout(100, 9);
        assert_eq!(words.len(), 9 * ROW_LAYOUT_WORDS);
        for (t, row) in words.chunks_exact(ROW_LAYOUT_WORDS).enumerate() {
            assert_eq!(row[0], 100 + t as u32);
            assert_eq!(row[1], t as u32);
            assert_eq!((row[2], row[3]), (0x1ff, 0));
        }
        assert_eq!(
            &words[3 * ROW_LAYOUT_WORDS + 4..3 * ROW_LAYOUT_WORDS + 8],
            &[2, 1, 0, 0]
        );
    }

    #[test]
    fn wide_blocks_fill_both_mask_words() {
        let words = block_row_layout(0, 33);
        assert_eq!((words[2], words[3]), (u32::MAX, 1));
        let words = block_row_layout(0, 64);
        assert_eq!((words[2], words[3]), (u32::MAX, u32::MAX));
    }
}
