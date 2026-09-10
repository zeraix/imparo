#![forbid(unsafe_code)]
#![doc = "GGUF-driven BPE tokenizer."]
//! Model-agnostic: the vocabulary, merge list, and special ids all come from the file.
//! Adding a model adds no code here unless its algorithm differs.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::path::Path;

use imparo_gguf::scan;

/// Where the token strings live.
///
/// `Mapped` BORROWS them from the GGUF file mapping: 8.2 MiB of token bytes stay
/// file-backed (clean pages, reclaimable) instead of anonymous heap; the resident cost is
/// the 2 MiB span table. `Owned` is the fallback when mapping fails and holds the old
/// blob + prefix offsets.
enum TokenText {
    Mapped {
        map: std::sync::Arc<imparo_gguf::scan::Map>,
        spans: Vec<(u32, u32)>,
    },
    Owned {
        blob: Vec<u8>,
        offsets: Vec<u32>,
    },
}

impl TokenText {
    fn len(&self) -> usize {
        match self {
            Self::Mapped { spans, .. } => spans.len(),
            Self::Owned { offsets, .. } => offsets.len().saturating_sub(1),
        }
    }
    fn get(&self, i: usize) -> &str {
        match self {
            Self::Mapped { map, spans } => {
                let Some(&(o, n)) = spans.get(i) else {
                    return "";
                };
                std::str::from_utf8(&map.bytes()[o as usize..o as usize + n as usize])
                    .unwrap_or("")
            }
            Self::Owned { blob, offsets } => {
                let (Some(&a), Some(&b)) = (offsets.get(i), offsets.get(i + 1)) else {
                    return "";
                };
                std::str::from_utf8(&blob[a as usize..b as usize]).unwrap_or("")
            }
        }
    }
    /// Anonymous (resident-charged) bytes only; mapped token bytes are file-backed.
    fn anon_bytes(&self) -> usize {
        match self {
            Self::Mapped { spans, .. } => {
                spans.len() * std::mem::size_of::<(u32, u32)>()
            }
            Self::Owned { blob, offsets } => {
                blob.len() + offsets.len() * std::mem::size_of::<u32>()
            }
        }
    }
}

pub struct Tokenizer {
    /// Token strings as ONE blob plus offsets, not `Vec<String>`.
    ///
    /// 262144 separate `String`s cost 24 bytes of header each plus a heap allocation each,
    /// on top of the 8.2 MiB of actual bytes -- about 19 MiB for 8.2 MiB of data. The blob
    /// holds the same bytes contiguously with a 4-byte offset per token.
    text: TokenText,
    /// Open-addressed index into `tokens`, storing `index + 1` (0 means empty).
    ///
    /// A `HashMap<String, u32>` held a SECOND copy of all 262144 token strings on top of
    /// `tokens` itself. Storing indices and comparing against `tokens` removes that copy
    /// entirely; the table is a flat `Vec<u32>`.
    id_slots: Vec<u32>,
    id_mask: usize,
    /// Merge ranks for token-id pairs, lower merges first: ONE sorted u64 per entry.
    ///
    /// The pair fits in 36 bits (each id < 2^18 for this vocab) and the rank in 20, so
    /// `(a<<18|b) << 20 | rank` packs both -- key in the HIGH bits, so sorting the packed
    /// word still sorts by key, and the first of any duplicate key keeps its lowest rank.
    /// One array of 8-byte words instead of parallel 8+4: 4.1 MiB instead of 6.2 for
    /// 514906 entries. (`HashMap<(u32,u32),u32>` was 13.6 MiB before that.) Binary search
    /// costs ~19 comparisons; affordable since `bpe_into` stopped rescanning every pair.
    ranks: Vec<u64>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub eot: Option<u32>,
    pub unk: Option<u32>,
    pub add_bos: bool,
    pub add_space_prefix: bool,
    /// Exact spellings of CONTROL / USER_DEFINED tokens, longest first. Chat templates
    /// contain these literally, so encode must map them to their ids rather than to bytes.
    specials: Vec<(String, u32)>,
    /// `special_ids[id]`: whether `id` is a CONTROL / USER_DEFINED spelling. `decode` used to
    /// scan `specials` linearly for EVERY id of the whole generated prefix on every token --
    /// quadratic in generation length (0.093 ms per token at 1024 generated, review #116).
    special_ids: Vec<bool>,
    /// Which algorithm the file's discriminator tuple admitted.
    pre_kind: PreTokenizer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreTokenizer {
    /// SentencePiece-shaped BPE used by the original gemma4 path.
    SentencePiece,
    /// Llama-3 pre-tokenization plus GPT-2 byte encoding, selected by
    /// `tokenizer.ggml.pre = "lfm2"`.
    Lfm2,
    /// Qwen3.5 pre-tokenization plus GPT-2 byte encoding, selected by
    /// `tokenizer.ggml.pre = "qwen35"`. The same splitter as Llama-3's with two
    /// alternatives written differently; see [`PieceShape`].
    Qwen35,
}

/// WHERE THE TWO BYTE-BPE SPLITTERS DIFFER, and nowhere else.
///
/// ```text
/// llama-3   ...|[^\r\n\p{L}\p{N}]?\p{L}+       |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+     [\r\n]*|...
/// qwen3.5   ...|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}    | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|...
/// ```
///
/// Two knobs, taken from llama.cpp's `LLAMA_VOCAB_PRE_TYPE_QWEN35` and
/// `LLAMA_VOCAB_PRE_TYPE_LLAMA3` regexes. The digit one is not cosmetic: "12345" is
/// ["123", "45"] under Llama-3 and five pieces under Qwen3.5, so a splitter that got it
/// wrong would produce a different token count for the same text.
#[derive(Clone, Copy)]
struct PieceShape {
    /// Digits one piece may take: 3 for Llama-3, 1 for Qwen3.5.
    digit_run: usize,
    /// Whether a combining mark continues a letter run -- and is therefore excluded from
    /// the punctuation class. True for Qwen3.5, false for Llama-3.
    marks_join_letters: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TokenizerAlgorithm {
    architecture: &'static str,
    model: &'static str,
    pre: Option<&'static str>,
    kind: PreTokenizer,
    default_add_bos: bool,
    default_add_space_prefix: bool,
}

/// Tokenizer algorithms are admitted by their complete GGUF discriminator tuple.
/// Shape facts such as vocabulary size and token ids deliberately do not live here.
const TOKENIZER_ALGORITHMS: &[TokenizerAlgorithm] = &[
    TokenizerAlgorithm {
        architecture: "gemma4",
        model: "gemma4",
        // The reference Gemma4 tokenizer selects this pre-tokenizer from `model`
        // itself, so both old conversions without `pre` and explicit conversions are
        // the same supported algorithm.
        pre: None,
        kind: PreTokenizer::SentencePiece,
        default_add_bos: true,
        default_add_space_prefix: false,
    },
    TokenizerAlgorithm {
        architecture: "gemma4",
        model: "gemma4",
        pre: Some("gemma4"),
        kind: PreTokenizer::SentencePiece,
        default_add_bos: true,
        default_add_space_prefix: false,
    },
    TokenizerAlgorithm {
        architecture: "lfm2",
        model: "gpt2",
        pre: Some("lfm2"),
        kind: PreTokenizer::Lfm2,
        default_add_bos: true,
        default_add_space_prefix: false,
    },
    TokenizerAlgorithm {
        architecture: "qwen35",
        model: "gpt2",
        pre: Some("qwen35"),
        kind: PreTokenizer::Qwen35,
        // Qwen3.8's file sets `tokenizer.ggml.add_bos_token = false`, and the read of
        // that key wins over this default; it is here for a conversion that omits it.
        default_add_bos: false,
        default_add_space_prefix: false,
    },
];

fn select_tokenizer_algorithm(
    architecture: Option<&str>,
    model: Option<&str>,
    pre: Option<&str>,
) -> Result<&'static TokenizerAlgorithm, String> {
    TOKENIZER_ALGORITHMS
        .iter()
        .find(|algorithm| {
            architecture == Some(algorithm.architecture)
                && model == Some(algorithm.model)
                && pre == algorithm.pre
        })
        .ok_or_else(|| {
            format!(
                "unsupported tokenizer algorithm tuple: general.architecture={architecture:?}, \
                 tokenizer.ggml.model={model:?}, tokenizer.ggml.pre={pre:?}"
            )
        })
}

/// A symbol in the working sequence: a slice of the normalised input.
#[derive(Clone, Copy)]
struct Sym {
    text_start: usize,
    text_len: usize,
    prev: i32,
    next: i32,
}

impl Tokenizer {
    /// Loads vocabulary, merges, and special ids from a GGUF file.
    ///
    /// # Errors
    ///
    /// Returns an error when the vocabulary or merge list cannot be read.
    pub fn from_gguf(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        // ONE metadata pass. `read_metadata_array` re-reads the whole file per call, so
        // tokens + merges + token_type + the scalars cost five parses of a 4 GB file --
        // which was most of a 6.4 s startup.
        // Mapped first: token bytes then BORROW from the file. Plain read is the
        // fallback and behaves exactly as before.
        let mut scan = match scan::Metadata::read_mapped(path) {
            Ok(m) => m,
            Err(_) => scan::Metadata::read(path)?,
        };
        // scalars first: the array takes below need &mut, so no closure may hold a borrow
        let begin_token = scan.u32_at("tokenizer.ggml.bos_token_id");
        let stop_token = scan.u32_at("tokenizer.ggml.eos_token_id");
        let turn_token = scan.u32_at("tokenizer.ggml.eot_token_id");
        let unknown_token = scan.u32_at("tokenizer.ggml.unknown_token_id");
        // WHICH ALGORITHM, admitted by the file's complete discriminator tuple rather
        // than guessed from any one field. An unknown tuple is an ERROR: running a GPT-2
        // byte-level vocabulary through the SentencePiece path produces tokens, and they
        // are the wrong tokens.
        let algo = select_tokenizer_algorithm(
            scan.str_at("general.architecture"),
            scan.str_at("tokenizer.ggml.model"),
            scan.str_at("tokenizer.ggml.pre"),
        )?;
        let add_bos = scan
            .bool_at("tokenizer.ggml.add_bos_token")
            .unwrap_or(algo.default_add_bos);
        let add_space_prefix = scan
            .bool_at("tokenizer.ggml.add_space_prefix")
            .unwrap_or(algo.default_add_space_prefix);
        let text = if let (Some(spans), Some(map)) =
            (scan.take_spans("tokenizer.ggml.tokens"), scan.file_arc())
        {
            TokenText::Mapped { map, spans }
        } else {
            let tokens = scan
                .take_strings("tokenizer.ggml.tokens")
                .unwrap_or_default();
            let mut blob: Vec<u8> =
                Vec::with_capacity(tokens.iter().map(String::len).sum());
            let mut offsets: Vec<u32> = Vec::with_capacity(tokens.len() + 1);
            for t in &tokens {
                offsets.push(
                    u32::try_from(blob.len()).map_err(|_| "vocab blob too large")?,
                );
                blob.extend_from_slice(t.as_bytes());
            }
            offsets
                .push(u32::try_from(blob.len()).map_err(|_| "vocab blob too large")?);
            TokenText::Owned { blob, offsets }
        };
        let merges = scan
            .take_strings("tokenizer.ggml.merges")
            .unwrap_or_default();
        // flat index table, power-of-two capacity at <= 50% load
        let cap = (text.len() * 2).next_power_of_two().max(16);
        let mut id_slots = vec![0_u32; cap];
        let id_mask = cap - 1;
        for i in 0..text.len() {
            let t = text.get(i);
            let mut slot = str_hash(t) as usize & id_mask;
            loop {
                let cur = id_slots[slot];
                if cur == 0 {
                    id_slots[slot] = i as u32 + 1;
                    break;
                }
                // first spelling wins, matching the previous `or_insert`
                if text.get(cur as usize - 1) == t {
                    break;
                }
                slot = (slot + 1) & id_mask;
            }
        }
        let lookup =
            |slots: &Vec<u32>, toks: &TokenText, mask: usize, q: &str| -> Option<u32> {
                let mut slot = str_hash(q) as usize & mask;
                loop {
                    let cur = slots[slot];
                    if cur == 0 {
                        return None;
                    }
                    if toks.get(cur as usize - 1) == q {
                        return Some(cur - 1);
                    }
                    slot = (slot + 1) & mask;
                }
            };
        // The packing needs ids in 18 bits and ranks in 20. True for this model family
        // (vocab exactly 2^18, 515k merges); a model that breaks either must fail loudly
        // here, not corrupt the table.
        if text.len() > (1 << 18) || merges.len() > (1 << 20) {
            return Err(format!(
                "merge packing limits exceeded: vocab {} merges {}",
                text.len(),
                merges.len()
            )
            .into());
        }
        let mut pairs: Vec<(u64, ())> = Vec::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            // a merge is "left right"; the space is the separator, and the pieces
            // themselves never contain one (SentencePiece uses U+2581 for space).
            if let Some(sp) = m.find(' ') {
                let (l, r) = (&m[..sp], &m[sp + 1..]);
                if let (Some(a), Some(b)) = (
                    lookup(&id_slots, &text, id_mask, l),
                    lookup(&id_slots, &text, id_mask, r),
                ) {
                    debug_assert!(a < (1 << 18) && b < (1 << 18) && rank < (1 << 20));
                    pairs.push((
                        (((u64::from(a) << 18) | u64::from(b)) << 20) | rank as u64,
                        (),
                    ));
                }
            }
        }
        drop(merges); // 515k strings, dead once the rank table exists
        // Sort by key THEN rank, so the first of any duplicate key is its lowest rank --
        // which is what `HashMap::entry().or_insert()` kept when the merges were walked
        // in rank order.
        pairs.sort_unstable();
        // Dedup on the KEY bits only: the sort put each key's lowest rank first.
        pairs.dedup_by_key(|(k, ())| *k >> 20);
        let ranks: Vec<u64> = pairs.into_iter().map(|(k, ())| k).collect();
        let types = scan
            .take_ints("tokenizer.ggml.token_type")
            .unwrap_or_default();
        let mut specials: Vec<(String, u32)> = (0..text.len())
            .filter_map(|i| {
                let t = text.get(i);
                (!t.is_empty() && matches!(types.get(i).copied(), Some(3 | 4)))
                    .then(|| (t.to_string(), i as u32))
            })
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));
        let special_ids = {
            let mut v = vec![false; text.len()];
            for &(_, id) in &specials {
                if (id as usize) < v.len() {
                    v[id as usize] = true;
                }
            }
            v
        };
        Ok(Self {
            text,
            id_slots,
            id_mask,
            ranks,
            bos: begin_token,
            eos: stop_token,
            eot: turn_token,
            unk: unknown_token,
            add_bos,
            add_space_prefix,
            specials,
            special_ids,
            pre_kind: algo.kind,
        })
    }

    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.text.len()
    }

    /// The token's bytes, or an empty string when the id is out of range.
    #[must_use]
    pub fn token_str(&self, id: usize) -> &str {
        self.text.get(id)
    }

    /// Merge pairs held in the rank table.
    #[must_use]
    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    /// Bytes held by the token strings themselves, ignoring map overhead. The id map
    /// currently stores a SECOND copy of every one of them.
    #[must_use]
    pub fn token_bytes(&self) -> usize {
        self.text.anon_bytes()
    }

    /// Encodes text to token ids, optionally prefixing BOS.
    #[must_use]
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        // split on literal special-token spellings first; only the gaps get BPE
        let mut rest = text;
        while !rest.is_empty() {
            let hit = self
                .specials
                .iter()
                .filter_map(|(sp, id)| {
                    rest.find(sp.as_str()).map(|at| (at, sp.len(), *id))
                })
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            if let Some((at, len, id)) = hit {
                if at > 0 {
                    self.encode_plain(&rest[..at], &mut out);
                }
                out.push(id);
                rest = &rest[at + len..];
            } else {
                self.encode_plain(rest, &mut out);
                break;
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        match self.pre_kind {
            PreTokenizer::SentencePiece => {
                let mut norm = String::with_capacity(text.len() + 8);
                if self.add_space_prefix && !text.starts_with(' ') {
                    norm.push('\u{2581}');
                }
                for ch in text.chars() {
                    if ch == ' ' {
                        norm.push('\u{2581}');
                    } else {
                        norm.push(ch);
                    }
                }
                self.bpe_into(&norm, out);
            }
            PreTokenizer::Lfm2 | PreTokenizer::Qwen35 => {
                for piece in byte_bpe_pieces(text, self.pre_kind.piece_shape()) {
                    let encoded = gpt2_byte_encode(piece);
                    // `ignore_merges = true`: a whole pre-tokenized word that is in the
                    // vocabulary wins even when no merge sequence would build it.
                    if let Some(id) = self.lookup(&encoded) {
                        out.push(id);
                    } else {
                        self.bpe_into(&encoded, out);
                    }
                }
            }
        }
    }

    /// Pushes the pair (li, ri) onto the candidate heap if the merge table ranks it.
    ///
    /// The recorded lengths are what make a stale entry detectable: symbols only ever grow
    /// and a merged-away symbol is stamped with length 0, so an entry whose recorded
    /// lengths still match describes exactly the pair that was pushed.
    fn push_pair(
        &self,
        heap: &mut BinaryHeap<Reverse<(u32, u32, u32, u32, u32)>>,
        syms: &[Sym],
        ids: &[Option<u32>],
        li: i32,
        ri: i32,
    ) {
        if li < 0 || ri < 0 {
            return;
        }
        let (lu, ru) = (li as usize, ri as usize);
        if let (Some(a), Some(b)) = (ids[lu], ids[ru]) {
            let key = (u64::from(a) << 18) | u64::from(b);
            // First packed word with these key bits; its low 20 bits are the rank.
            let at = self.ranks.partition_point(|&p| p >> 20 < key);
            if let Some(&p) = self.ranks.get(at) {
                if p >> 20 == key {
                    heap.push(Reverse((
                        (p & 0xF_FFFF) as u32,
                        li as u32,
                        ri as u32,
                        syms[lu].text_len as u32,
                        syms[ru].text_len as u32,
                    )));
                }
            }
        }
    }

    /// Greedy lowest-rank-first merging over a doubly linked list of symbols.
    ///
    /// Candidates come off a heap rather than from a rescan. The rescan re-read EVERY
    /// adjacent pair on every merge, and resolved both sides to ids each time, so a prompt
    /// cost O(n^2) hash lookups: 3033 characters took 149 ms. Each merge can only create
    /// two new adjacent pairs, so pushing just those two is enough.
    fn bpe_into(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // seed with one symbol per character
        let mut syms: Vec<Sym> = Vec::new();
        for (i, ch) in text.char_indices() {
            let n = syms.len() as i32;
            syms.push(Sym {
                text_start: i,
                text_len: ch.len_utf8(),
                prev: n - 1,
                next: n + 1,
            });
        }
        if let Some(last) = syms.last_mut() {
            last.next = -1;
        }

        let piece =
            |s: &Sym| -> &str { &text[s.text_start..s.text_start + s.text_len] };
        // One id per symbol, kept current across merges, instead of two lookups per pair
        // per pass.
        let mut ids: Vec<Option<u32>> =
            syms.iter().map(|s| self.lookup(piece(s))).collect();

        let mut heap: BinaryHeap<Reverse<(u32, u32, u32, u32, u32)>> =
            BinaryHeap::new();
        for i in 0..syms.len() {
            self.push_pair(&mut heap, &syms, &ids, i as i32, syms[i].next);
        }

        // Ties break on the smaller left index, which is the leftmost pair -- the same
        // choice the rescan made by keeping the first pair of equal rank it saw.
        while let Some(Reverse((_, li, ri, len_l, len_r))) = heap.pop() {
            let (lu, ru) = (li as usize, ri as usize);
            if syms[lu].next != ri as i32 {
                continue;
            }
            if syms[lu].text_len as u32 != len_l || syms[ru].text_len as u32 != len_r {
                continue;
            }
            // merge right into left
            syms[lu].text_len += syms[ru].text_len;
            let rn = syms[ru].next;
            syms[lu].next = rn;
            if rn >= 0 {
                syms[rn as usize].prev = li as i32;
            }
            syms[ru].text_len = 0; // stamp dead, so stale entries naming it fail above
            ids[lu] = self.lookup(piece(&syms[lu]));
            let lp = syms[lu].prev;
            self.push_pair(&mut heap, &syms, &ids, lp, li as i32);
            self.push_pair(&mut heap, &syms, &ids, li as i32, rn);
        }

        let mut i = 0_i32;
        while i >= 0 {
            let s = syms[i as usize];
            let p = piece(&s);
            if let Some(id) = self.lookup(p) {
                out.push(id);
            } else {
                // byte fallback: SentencePiece spells raw bytes as <0xNN>
                for b in p.as_bytes() {
                    let name = format!("<0x{b:02X}>");
                    if let Some(id) = self.lookup(&name) {
                        out.push(id);
                    } else if let Some(u) = self.unk {
                        out.push(u);
                    }
                }
            }
            i = s.next;
        }
    }

    /// Decodes ids back to text, undoing the space marker and byte spellings.
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        self.decode_bytes_into(ids, &mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The byte stream `decode` builds, appended to `out` -- so a caller that decodes a
    /// growing sequence one id at a time keeps the bytes and pays for the new id only;
    /// `String::from_utf8_lossy(&out)` is then exactly `decode(&all_ids_so_far)`.
    pub fn decode_bytes_into(&self, ids: &[u32], bytes: &mut Vec<u8>) {
        for &id in ids {
            if id as usize >= self.vocab_size() {
                continue;
            }
            let t = self.token_str(id as usize);
            if self.pre_kind != PreTokenizer::SentencePiece {
                // CONTROL / USER_DEFINED spellings are literal markup; every other piece
                // is in the reversible GPT-2 byte alphabet.
                if self.special_ids.get(id as usize).copied().unwrap_or(false) {
                    bytes.extend_from_slice(t.as_bytes());
                } else {
                    for ch in t.chars() {
                        if let Some(byte) = gpt2_char_to_byte(ch) {
                            bytes.push(byte);
                        } else {
                            let mut buf = [0_u8; 4];
                            bytes
                                .extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                    }
                }
                continue;
            }
            if t.len() == 6 && t.starts_with("<0x") && t.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&t[3..5], 16) {
                    bytes.push(b);
                    continue;
                }
            }
            for ch in t.chars() {
                if ch == '\u{2581}' {
                    bytes.push(b' ');
                } else {
                    let mut buf = [0_u8; 4];
                    bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }

    /// Token id for an exact literal piece, e.g. a control token spelling.
    #[must_use]
    pub fn id_of(&self, piece: &str) -> Option<u32> {
        self.lookup(piece)
    }

    fn lookup(&self, q: &str) -> Option<u32> {
        let mut slot = str_hash(q) as usize & self.id_mask;
        loop {
            let cur = self.id_slots[slot];
            if cur == 0 {
                return None;
            }
            if self.token_str(cur as usize - 1) == q {
                return Some(cur - 1);
            }
            slot = (slot + 1) & self.id_mask;
        }
    }
}

/// Stable string hash for the flat id table.
fn str_hash(s: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// The Unicode general category M (Mn, Mc, Me) as sorted, non-overlapping code-point
/// ranges -- 321 of them, generated from Python's `unicodedata`
/// 16.0.0 and never edited by hand.
///
/// Qwen3.5's pre-tokenizer classes carry `\p{M}` where Llama-3's do not, so a combining
/// mark continues a letter run there instead of opening a punctuation run. Rust's standard
/// library has `char::is_alphabetic` but no mark test, and an approximation would split
/// Devanagari, Arabic and Hebrew text differently from the reference -- which reads as a
/// worse model, not as a tokenizer defect.
static MARK_RANGES: &[(u32, u32)] = &[
    (0x0300, 0x036F), (0x0483, 0x0489), (0x0591, 0x05BD), (0x05BF, 0x05BF), (0x05C1, 0x05C2),
    (0x05C4, 0x05C5), (0x05C7, 0x05C7), (0x0610, 0x061A), (0x064B, 0x065F), (0x0670, 0x0670),
    (0x06D6, 0x06DC), (0x06DF, 0x06E4), (0x06E7, 0x06E8), (0x06EA, 0x06ED), (0x0711, 0x0711),
    (0x0730, 0x074A), (0x07A6, 0x07B0), (0x07EB, 0x07F3), (0x07FD, 0x07FD), (0x0816, 0x0819),
    (0x081B, 0x0823), (0x0825, 0x0827), (0x0829, 0x082D), (0x0859, 0x085B), (0x0897, 0x089F),
    (0x08CA, 0x08E1), (0x08E3, 0x0903), (0x093A, 0x093C), (0x093E, 0x094F), (0x0951, 0x0957),
    (0x0962, 0x0963), (0x0981, 0x0983), (0x09BC, 0x09BC), (0x09BE, 0x09C4), (0x09C7, 0x09C8),
    (0x09CB, 0x09CD), (0x09D7, 0x09D7), (0x09E2, 0x09E3), (0x09FE, 0x09FE), (0x0A01, 0x0A03),
    (0x0A3C, 0x0A3C), (0x0A3E, 0x0A42), (0x0A47, 0x0A48), (0x0A4B, 0x0A4D), (0x0A51, 0x0A51),
    (0x0A70, 0x0A71), (0x0A75, 0x0A75), (0x0A81, 0x0A83), (0x0ABC, 0x0ABC), (0x0ABE, 0x0AC5),
    (0x0AC7, 0x0AC9), (0x0ACB, 0x0ACD), (0x0AE2, 0x0AE3), (0x0AFA, 0x0AFF), (0x0B01, 0x0B03),
    (0x0B3C, 0x0B3C), (0x0B3E, 0x0B44), (0x0B47, 0x0B48), (0x0B4B, 0x0B4D), (0x0B55, 0x0B57),
    (0x0B62, 0x0B63), (0x0B82, 0x0B82), (0x0BBE, 0x0BC2), (0x0BC6, 0x0BC8), (0x0BCA, 0x0BCD),
    (0x0BD7, 0x0BD7), (0x0C00, 0x0C04), (0x0C3C, 0x0C3C), (0x0C3E, 0x0C44), (0x0C46, 0x0C48),
    (0x0C4A, 0x0C4D), (0x0C55, 0x0C56), (0x0C62, 0x0C63), (0x0C81, 0x0C83), (0x0CBC, 0x0CBC),
    (0x0CBE, 0x0CC4), (0x0CC6, 0x0CC8), (0x0CCA, 0x0CCD), (0x0CD5, 0x0CD6), (0x0CE2, 0x0CE3),
    (0x0CF3, 0x0CF3), (0x0D00, 0x0D03), (0x0D3B, 0x0D3C), (0x0D3E, 0x0D44), (0x0D46, 0x0D48),
    (0x0D4A, 0x0D4D), (0x0D57, 0x0D57), (0x0D62, 0x0D63), (0x0D81, 0x0D83), (0x0DCA, 0x0DCA),
    (0x0DCF, 0x0DD4), (0x0DD6, 0x0DD6), (0x0DD8, 0x0DDF), (0x0DF2, 0x0DF3), (0x0E31, 0x0E31),
    (0x0E34, 0x0E3A), (0x0E47, 0x0E4E), (0x0EB1, 0x0EB1), (0x0EB4, 0x0EBC), (0x0EC8, 0x0ECE),
    (0x0F18, 0x0F19), (0x0F35, 0x0F35), (0x0F37, 0x0F37), (0x0F39, 0x0F39), (0x0F3E, 0x0F3F),
    (0x0F71, 0x0F84), (0x0F86, 0x0F87), (0x0F8D, 0x0F97), (0x0F99, 0x0FBC), (0x0FC6, 0x0FC6),
    (0x102B, 0x103E), (0x1056, 0x1059), (0x105E, 0x1060), (0x1062, 0x1064), (0x1067, 0x106D),
    (0x1071, 0x1074), (0x1082, 0x108D), (0x108F, 0x108F), (0x109A, 0x109D), (0x135D, 0x135F),
    (0x1712, 0x1715), (0x1732, 0x1734), (0x1752, 0x1753), (0x1772, 0x1773), (0x17B4, 0x17D3),
    (0x17DD, 0x17DD), (0x180B, 0x180D), (0x180F, 0x180F), (0x1885, 0x1886), (0x18A9, 0x18A9),
    (0x1920, 0x192B), (0x1930, 0x193B), (0x1A17, 0x1A1B), (0x1A55, 0x1A5E), (0x1A60, 0x1A7C),
    (0x1A7F, 0x1A7F), (0x1AB0, 0x1ACE), (0x1B00, 0x1B04), (0x1B34, 0x1B44), (0x1B6B, 0x1B73),
    (0x1B80, 0x1B82), (0x1BA1, 0x1BAD), (0x1BE6, 0x1BF3), (0x1C24, 0x1C37), (0x1CD0, 0x1CD2),
    (0x1CD4, 0x1CE8), (0x1CED, 0x1CED), (0x1CF4, 0x1CF4), (0x1CF7, 0x1CF9), (0x1DC0, 0x1DFF),
    (0x20D0, 0x20F0), (0x2CEF, 0x2CF1), (0x2D7F, 0x2D7F), (0x2DE0, 0x2DFF), (0x302A, 0x302F),
    (0x3099, 0x309A), (0xA66F, 0xA672), (0xA674, 0xA67D), (0xA69E, 0xA69F), (0xA6F0, 0xA6F1),
    (0xA802, 0xA802), (0xA806, 0xA806), (0xA80B, 0xA80B), (0xA823, 0xA827), (0xA82C, 0xA82C),
    (0xA880, 0xA881), (0xA8B4, 0xA8C5), (0xA8E0, 0xA8F1), (0xA8FF, 0xA8FF), (0xA926, 0xA92D),
    (0xA947, 0xA953), (0xA980, 0xA983), (0xA9B3, 0xA9C0), (0xA9E5, 0xA9E5), (0xAA29, 0xAA36),
    (0xAA43, 0xAA43), (0xAA4C, 0xAA4D), (0xAA7B, 0xAA7D), (0xAAB0, 0xAAB0), (0xAAB2, 0xAAB4),
    (0xAAB7, 0xAAB8), (0xAABE, 0xAABF), (0xAAC1, 0xAAC1), (0xAAEB, 0xAAEF), (0xAAF5, 0xAAF6),
    (0xABE3, 0xABEA), (0xABEC, 0xABED), (0xFB1E, 0xFB1E), (0xFE00, 0xFE0F), (0xFE20, 0xFE2F),
    (0x101FD, 0x101FD), (0x102E0, 0x102E0), (0x10376, 0x1037A), (0x10A01, 0x10A03),
    (0x10A05, 0x10A06), (0x10A0C, 0x10A0F), (0x10A38, 0x10A3A), (0x10A3F, 0x10A3F),
    (0x10AE5, 0x10AE6), (0x10D24, 0x10D27), (0x10D69, 0x10D6D), (0x10EAB, 0x10EAC),
    (0x10EFC, 0x10EFF), (0x10F46, 0x10F50), (0x10F82, 0x10F85), (0x11000, 0x11002),
    (0x11038, 0x11046), (0x11070, 0x11070), (0x11073, 0x11074), (0x1107F, 0x11082),
    (0x110B0, 0x110BA), (0x110C2, 0x110C2), (0x11100, 0x11102), (0x11127, 0x11134),
    (0x11145, 0x11146), (0x11173, 0x11173), (0x11180, 0x11182), (0x111B3, 0x111C0),
    (0x111C9, 0x111CC), (0x111CE, 0x111CF), (0x1122C, 0x11237), (0x1123E, 0x1123E),
    (0x11241, 0x11241), (0x112DF, 0x112EA), (0x11300, 0x11303), (0x1133B, 0x1133C),
    (0x1133E, 0x11344), (0x11347, 0x11348), (0x1134B, 0x1134D), (0x11357, 0x11357),
    (0x11362, 0x11363), (0x11366, 0x1136C), (0x11370, 0x11374), (0x113B8, 0x113C0),
    (0x113C2, 0x113C2), (0x113C5, 0x113C5), (0x113C7, 0x113CA), (0x113CC, 0x113D0),
    (0x113D2, 0x113D2), (0x113E1, 0x113E2), (0x11435, 0x11446), (0x1145E, 0x1145E),
    (0x114B0, 0x114C3), (0x115AF, 0x115B5), (0x115B8, 0x115C0), (0x115DC, 0x115DD),
    (0x11630, 0x11640), (0x116AB, 0x116B7), (0x1171D, 0x1172B), (0x1182C, 0x1183A),
    (0x11930, 0x11935), (0x11937, 0x11938), (0x1193B, 0x1193E), (0x11940, 0x11940),
    (0x11942, 0x11943), (0x119D1, 0x119D7), (0x119DA, 0x119E0), (0x119E4, 0x119E4),
    (0x11A01, 0x11A0A), (0x11A33, 0x11A39), (0x11A3B, 0x11A3E), (0x11A47, 0x11A47),
    (0x11A51, 0x11A5B), (0x11A8A, 0x11A99), (0x11C2F, 0x11C36), (0x11C38, 0x11C3F),
    (0x11C92, 0x11CA7), (0x11CA9, 0x11CB6), (0x11D31, 0x11D36), (0x11D3A, 0x11D3A),
    (0x11D3C, 0x11D3D), (0x11D3F, 0x11D45), (0x11D47, 0x11D47), (0x11D8A, 0x11D8E),
    (0x11D90, 0x11D91), (0x11D93, 0x11D97), (0x11EF3, 0x11EF6), (0x11F00, 0x11F01),
    (0x11F03, 0x11F03), (0x11F34, 0x11F3A), (0x11F3E, 0x11F42), (0x11F5A, 0x11F5A),
    (0x13440, 0x13440), (0x13447, 0x13455), (0x1611E, 0x1612F), (0x16AF0, 0x16AF4),
    (0x16B30, 0x16B36), (0x16F4F, 0x16F4F), (0x16F51, 0x16F87), (0x16F8F, 0x16F92),
    (0x16FE4, 0x16FE4), (0x16FF0, 0x16FF1), (0x1BC9D, 0x1BC9E), (0x1CF00, 0x1CF2D),
    (0x1CF30, 0x1CF46), (0x1D165, 0x1D169), (0x1D16D, 0x1D172), (0x1D17B, 0x1D182),
    (0x1D185, 0x1D18B), (0x1D1AA, 0x1D1AD), (0x1D242, 0x1D244), (0x1DA00, 0x1DA36),
    (0x1DA3B, 0x1DA6C), (0x1DA75, 0x1DA75), (0x1DA84, 0x1DA84), (0x1DA9B, 0x1DA9F),
    (0x1DAA1, 0x1DAAF), (0x1E000, 0x1E006), (0x1E008, 0x1E018), (0x1E01B, 0x1E021),
    (0x1E023, 0x1E024), (0x1E026, 0x1E02A), (0x1E08F, 0x1E08F), (0x1E130, 0x1E136),
    (0x1E2AE, 0x1E2AE), (0x1E2EC, 0x1E2EF), (0x1E4EC, 0x1E4EF), (0x1E5EE, 0x1E5EF),
    (0x1E8D0, 0x1E8D6), (0x1E944, 0x1E94A), (0xE0100, 0xE01EF),
];

/// True when `ch` is in Unicode general category M.
fn is_combining_mark(ch: char) -> bool {
    let cp = ch as u32;
    MARK_RANGES
        .binary_search_by(|&(lo, hi)| {
            if cp < lo {
                std::cmp::Ordering::Greater
            } else if cp > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

impl PreTokenizer {
    /// The two alternatives that differ between the byte-BPE splitters. Asked of the
    /// SentencePiece arm it returns Llama-3's, which that arm never uses.
    fn piece_shape(self) -> PieceShape {
        match self {
            Self::Qwen35 => PieceShape { digit_run: 1, marks_join_letters: true },
            Self::SentencePiece | Self::Lfm2 => {
                PieceShape { digit_run: 3, marks_join_letters: false }
            }
        }
    }
}

/// Splits raw text the way the reference's byte-BPE pre-tokenizers do, with `shape`
/// carrying the two places Llama-3 (`pre = "lfm2"`) and Qwen3.5 (`pre = "qwen35"`)
/// disagree. Everything else below is one splitter, not two.
fn byte_bpe_pieces(text: &str, shape: PieceShape) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pieces = Vec::new();
    let mut pos = 0_usize;

    let byte_at = |index: usize| chars.get(index).map_or(text.len(), |&(byte, _)| byte);
    let get = |index: usize| chars.get(index).map(|&(_, ch)| ch);
    // `\p{L}` approximated by `char::is_alphabetic` (which also admits Nl and
    // Other_Alphabetic), as it has been since LFM2; `\p{M}` is exact.
    let is_letter = |index: usize| {
        get(index).is_some_and(|c| {
            c.is_alphabetic() || (shape.marks_join_letters && is_combining_mark(c))
        })
    };
    let is_number = |index: usize| get(index).is_some_and(char::is_numeric);
    let is_space = |index: usize| get(index).is_some_and(char::is_whitespace);
    let mut push = |start: usize, end: usize| {
        if end > start {
            pieces.push(&text[byte_at(start)..byte_at(end)]);
        }
    };

    while pos < chars.len() {
        let start = pos;
        let ch = get(pos).expect("position is in range");

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if ch == '\'' {
            if let Some(next) = get(pos + 1).map(|c| c.to_ascii_lowercase()) {
                if matches!(next, 's' | 't' | 'm' | 'd') {
                    pos += 2;
                    push(start, pos);
                    continue;
                }
                if let Some(after) = get(pos + 2).map(|c| c.to_ascii_lowercase()) {
                    if matches!((next, after), ('r' | 'v', 'e') | ('l', 'l')) {
                        pos += 3;
                        push(start, pos);
                        continue;
                    }
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+ -- and [\p{L}\p{M}]+ under Qwen3.5's shape. The
        // OPTIONAL PREFIX class carries no \p{M} in either regex, so only the run's test
        // moves; a mark that opens the piece is the run's first character, not the prefix.
        if ch != '\r'
            && ch != '\n'
            && !is_number(pos)
            && (is_letter(pos) || is_letter(pos + 1))
        {
            pos += 1;
            while is_letter(pos) {
                pos += 1;
            }
            push(start, pos);
            continue;
        }

        // \p{N}{1,3} for Llama-3, a single \p{N} for Qwen3.5
        if is_number(pos) {
            while pos < chars.len() && is_number(pos) && pos - start < shape.digit_run {
                pos += 1;
            }
            push(start, pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]* -- with \p{M} excluded too under Qwen3.5,
        // which `is_letter` already carries.
        let probe = if ch == ' ' { pos + 1 } else { pos };
        let is_other = |index: usize| {
            get(index).is_some()
                && !is_space(index)
                && !is_letter(index)
                && !is_number(index)
        };
        if is_other(probe) {
            pos = probe;
            while is_other(pos) {
                pos += 1;
            }
            while matches!(get(pos), Some('\r' | '\n')) {
                pos += 1;
            }
            push(start, pos);
            continue;
        }

        let mut whitespace_end = pos;
        let mut last_newline_end = None;
        while is_space(whitespace_end) {
            if matches!(get(whitespace_end), Some('\r' | '\n')) {
                last_newline_end = Some(whitespace_end + 1);
            }
            whitespace_end += 1;
        }

        // \s*[\r\n]+
        if let Some(end) = last_newline_end {
            pos = end;
            push(start, pos);
            continue;
        }

        // \s+(?!\S): before a following non-space, leave its final space for the
        // next token. The final \s+ alternative consumes all trailing whitespace.
        let whitespace_count = whitespace_end - pos;
        if whitespace_count > 1 && whitespace_end < chars.len() {
            pos = whitespace_end - 1;
            push(start, pos);
            continue;
        }
        if whitespace_count > 0 {
            pos = whitespace_end;
            push(start, pos);
            continue;
        }

        pos += 1;
        push(start, pos);
    }

    pieces
}

/// GPT-2's reversible byte alphabet. This is byte-oriented on purpose: a UTF-8
/// code point first becomes its encoded bytes, then each byte becomes one vocabulary
/// character.
fn gpt2_byte_char(byte: u8) -> char {
    let codepoint = match byte {
        0..=32 => 256 + u32::from(byte),
        33..=126 | 161..=172 | 174..=255 => u32::from(byte),
        127..=160 => 289 + u32::from(byte - 127),
        173 => 323,
    };
    char::from_u32(codepoint).expect("GPT-2 byte map only contains scalar values")
}

fn gpt2_char_to_byte(ch: char) -> Option<u8> {
    match u32::from(ch) {
        value @ (33..=126 | 161..=172 | 174..=255) => u8::try_from(value).ok(),
        value @ 256..=288 => u8::try_from(value - 256).ok(),
        value @ 289..=322 => u8::try_from(127 + value - 289).ok(),
        323 => Some(173),
        _ => None,
    }
}

fn gpt2_byte_encode(text: &str) -> String {
    text.as_bytes()
        .iter()
        .map(|&byte| gpt2_byte_char(byte))
        .collect()
}

#[cfg(test)]
mod tests {
    /// Incremental decode (bytes appended one id at a time) must equal the whole-prefix
    /// decode at every step, including special ids and multi-byte pieces. Runs on the GGUFs
    /// named by `IMPARO_TOKENIZE_GGUFS`, a `:`-separated list of paths; skips (passes) when
    /// the variable is unset or a path is not a file, so the test names no machine's own
    /// directories.
    #[test]
    fn incremental_decode_matches_whole_prefix_decode() {
        let Ok(list) = std::env::var("IMPARO_TOKENIZE_GGUFS") else {
            return;
        };
        for path in list.split(':').filter(|s| !s.is_empty()) {
            let p = std::path::Path::new(path);
            if !p.is_file() {
                continue;
            }
            let tok = super::Tokenizer::from_gguf(p).expect("tokenizer");
            let vocab = tok.vocab_size() as u32;
            let specials: Vec<u32> =
                tok.specials.iter().map(|(_, id)| *id).take(4).collect();
            let mut ids: Vec<u32> =
                (0..600u32).map(|i| (i * 7919 + 13) % vocab).collect();
            ids.extend_from_slice(&specials);
            ids.extend((0..64u32).map(|i| 1000 + i % 500));
            let mut bytes = Vec::new();
            for n in 1..=ids.len() {
                tok.decode_bytes_into(&ids[n - 1..n], &mut bytes);
                let incremental = String::from_utf8_lossy(&bytes).into_owned();
                assert_eq!(incremental, tok.decode(&ids[..n]), "{path} at n={n}");
            }
        }
    }

    use super::{
        PreTokenizer, gpt2_byte_char, gpt2_byte_encode, gpt2_char_to_byte,
        PieceShape, byte_bpe_pieces, select_tokenizer_algorithm,
    };

    fn llama3_pieces_(text: &str) -> Vec<&str> {
        byte_bpe_pieces(text, PreTokenizer::Lfm2.piece_shape())
    }
    fn qwen35_pieces(text: &str) -> Vec<&str> {
        byte_bpe_pieces(text, PreTokenizer::Qwen35.piece_shape())
    }

    /// The TWO alternatives that differ between the byte-BPE splitters, and nothing else.
    /// Both expectations are read off llama.cpp's regexes, not off this implementation.
    #[test]
    fn qwen35_splits_digits_singly_and_keeps_marks_with_their_letter() {
        // \p{N}{1,3} against a single \p{N}: the same string, two token counts.
        assert_eq!(llama3_pieces_("1234567"), ["123", "456", "7"]);
        assert_eq!(qwen35_pieces("1234567"), ["1", "2", "3", "4", "5", "6", "7"]);

        // [\p{L}\p{M}]+ against \p{L}+. A combining acute (U+0301, category Mn) after a
        // Latin letter: Qwen3.5 keeps them in one piece, Llama-3 breaks between them and
        // hands the mark to the punctuation alternative.
        assert_eq!(qwen35_pieces("e\u{301}"), ["e\u{301}"]);
        assert_eq!(llama3_pieces_("e\u{301}"), ["e", "\u{301}"]);

        // NOT A DIFFERENCE HERE, and it should be: U+093F DEVANAGARI VOWEL SIGN I is
        // category Mc AND Other_Alphabetic, so `char::is_alphabetic` -- this file's
        // standing approximation of \p{L} -- already admits it on BOTH arms. Qwen3.5's
        // answer is right; Llama-3's disagrees with its own regex, which is a
        // pre-existing approximation in the LFM2 path and not something this change
        // introduced or fixed.
        let devanagari = "\u{928}\u{93f}";
        assert_eq!(qwen35_pieces(devanagari), [devanagari]);
        assert_eq!(llama3_pieces_(devanagari), [devanagari]);

        // Everything else is the same splitter.
        for text in ["Hello, world!", "can't I'M we'd", "a  b   c ", "line 1\nline 2\r\n"] {
            assert_eq!(llama3_pieces_(text), qwen35_pieces(text), "{text:?}");
        }
    }

    #[test]
    fn the_mark_table_is_the_unicode_category_and_not_a_guess() {
        // Spot values from three scripts plus the boundaries either side of the first
        // range, so an off-by-one in the binary search cannot pass.
        for ch in ['\u{300}', '\u{36F}', '\u{93F}', '\u{5B0}', '\u{20DD}', '\u{FE20}'] {
            assert!(super::is_combining_mark(ch), "{ch:?} is category M");
        }
        for ch in ['\u{2FF}', '\u{370}', 'a', '0', ' ', '\u{4E00}'] {
            assert!(!super::is_combining_mark(ch), "{ch:?} is not category M");
        }
    }

    #[test]
    fn tokenizer_algorithm_selection_is_explicit_and_fail_closed() {
        assert_eq!(
            select_tokenizer_algorithm(Some("lfm2"), Some("gpt2"), Some("lfm2"))
                .map(|algorithm| algorithm.kind),
            Ok(PreTokenizer::Lfm2)
        );
        assert_eq!(
            select_tokenizer_algorithm(Some("gemma4"), Some("gemma4"), Some("gemma4"))
                .map(|algorithm| algorithm.kind),
            Ok(PreTokenizer::SentencePiece)
        );
        assert_eq!(
            select_tokenizer_algorithm(Some("gemma4"), Some("gemma4"), None)
                .map(|algorithm| algorithm.kind),
            Ok(PreTokenizer::SentencePiece)
        );
        assert!(select_tokenizer_algorithm(Some("lfm2"), None, Some("lfm2")).is_err());
        assert!(select_tokenizer_algorithm(Some("lfm2"), Some("gpt2"), None).is_err());
        assert!(
            select_tokenizer_algorithm(Some("lfm2"), Some("gpt2"), Some("llama3"))
                .is_err()
        );
        assert!(
            select_tokenizer_algorithm(Some("unknown"), Some("gpt2"), Some("lfm2"))
                .is_err()
        );
        assert!(select_tokenizer_algorithm(None, Some("gpt2"), Some("lfm2")).is_err());
    }
    #[test]
    fn lfm2_llama3_pretokenizer_matches_reference_boundaries() {
        assert_eq!(
            llama3_pieces_("Hello, world!"),
            ["Hello", ",", " world", "!"]
        );
        assert_eq!(
            llama3_pieces_("can't I'M we'd"),
            ["can", "'t", " I", "'M", " we", "'d"]
        );
        assert_eq!(llama3_pieces_("1234567"), ["123", "456", "7"]);
        assert_eq!(
            llama3_pieces_("a  b   c "),
            ["a", " ", " b", "  ", " c", " "]
        );
        assert_eq!(
            llama3_pieces_("line 1\nline 2\r\n"),
            ["line", " ", "1", "\n", "line", " ", "2", "\r\n"]
        );
        assert_eq!(
            llama3_pieces_("你好，世界! café 👋"),
            ["你好", "，世界", "!", " café", " 👋"]
        );
    }
    #[test]
    fn gpt2_byte_alphabet_is_a_reversible_bijection() {
        let mut seen = std::collections::HashSet::new();
        for byte in 0_u8..=u8::MAX {
            let ch = gpt2_byte_char(byte);
            assert!(seen.insert(ch));
            assert_eq!(gpt2_char_to_byte(ch), Some(byte));
        }
        assert_eq!(gpt2_byte_encode(" hello"), "Ġhello");
        assert_eq!(gpt2_byte_encode("\n"), "Ċ");
        assert_eq!(gpt2_byte_encode("👋"), "ðŁĳĭ");
    }
}
