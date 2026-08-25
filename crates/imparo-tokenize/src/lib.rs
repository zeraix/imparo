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
        let bos = scan.u32_at("tokenizer.ggml.bos_token_id");
        let eos = scan.u32_at("tokenizer.ggml.eos_token_id");
        let eot = scan.u32_at("tokenizer.ggml.eot_token_id");
        let unk = scan.u32_at("tokenizer.ggml.unknown_token_id");
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
                        ((u64::from(a) << 18) | u64::from(b)) << 20 | rank as u64,
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
        Ok(Self {
            text,
            id_slots,
            id_mask,
            ranks,
            bos,
            eos,
            eot,
            unk,
            add_bos,
            add_space_prefix,
            specials,
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
            PreTokenizer::Lfm2 => {
                for piece in llama3_pieces(text) {
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
        for &id in ids {
            if id as usize >= self.vocab_size() {
                continue;
            }
            let t = self.token_str(id as usize);
            if self.pre_kind == PreTokenizer::Lfm2 {
                // CONTROL / USER_DEFINED spellings are literal markup; every other piece
                // is in the reversible GPT-2 byte alphabet.
                if self.specials.iter().any(|(_, sid)| *sid == id) {
                    bytes.extend_from_slice(t.as_bytes());
                } else {
                    for ch in t.chars() {
                        if let Some(byte) = gpt2_char_to_byte(ch) {
                            bytes.push(byte);
                        } else {
                            let mut buf = [0_u8; 4];
                            bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
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
        String::from_utf8_lossy(&bytes).into_owned()
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

/// Splits raw text with the exact custom Llama-3 pre-tokenizer shape used by
/// the reference fork for `tokenizer.ggml.pre = "lfm2"`.
fn llama3_pieces(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pieces = Vec::new();
    let mut pos = 0_usize;

    let byte_at = |index: usize| chars.get(index).map_or(text.len(), |&(byte, _)| byte);
    let get = |index: usize| chars.get(index).map(|&(_, ch)| ch);
    let is_letter = |index: usize| get(index).is_some_and(char::is_alphabetic);
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

        // [^\r\n\p{L}\p{N}]?\p{L}+
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

        // \p{N}{1,3}
        if is_number(pos) {
            while pos < chars.len() && is_number(pos) && pos - start < 3 {
                pos += 1;
            }
            push(start, pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
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
    use super::{
        PreTokenizer, gpt2_byte_char, gpt2_byte_encode, gpt2_char_to_byte, llama3_pieces,
        select_tokenizer_algorithm,
    };

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
            llama3_pieces("Hello, world!"),
            ["Hello", ",", " world", "!"]
        );
        assert_eq!(
            llama3_pieces("can't I'M we'd"),
            ["can", "'t", " I", "'M", " we", "'d"]
        );
        assert_eq!(llama3_pieces("1234567"), ["123", "456", "7"]);
        assert_eq!(
            llama3_pieces("a  b   c "),
            ["a", " ", " b", "  ", " c", " "]
        );
        assert_eq!(
            llama3_pieces("line 1\nline 2\r\n"),
            ["line", " ", "1", "\n", "line", " ", "2", "\r\n"]
        );
        assert_eq!(
            llama3_pieces("你好，世界! café 👋"),
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
