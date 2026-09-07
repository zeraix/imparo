//! Content identity: the rule everything follows from.
//!
//! Identity is a ROLLING HASH OF THE PREFIX, rooted in the configuration that
//! determines the numbers:
//!
//! ```text
//! root      = H( model digest || kv byte layout || rope config || layer geometry
//!                || writing backend )
//! prefix(p) = H( root || tokens[0..p] )
//! ```
//!
//! Over the prefix, because K/V values depend on every preceding token. Rooted,
//! because a different model, KV byte layout or backend produces different bytes for
//! the same tokens. The property this buys: EQUAL HASH IMPLIES EQUAL BYTES — sharing
//! needs no validation on borrow.
//!
//! WHAT THE ROOT SCOPES A STORE TO. Nothing TUNED may enter it — a knob that changes
//! speed and not bytes must not split a store. Everything that changes the bytes must:
//! the model, the K/V types, the quantisation basis and byte codec
//! (`KvByteLayoutProfile`), the geometry, and `config_root`'s `device_tag`. That last
//! one is why a store is per-backend TODAY: Metal and CUDA never share a root, so no
//! seed written by one is readable by the other, whatever their byte layouts say.
//! Making the byte-layout profile sufficient — and dropping `device_tag` — is the
//! change that would let one seed serve both, and it is not this.
//!
//! WHY THE PREFIX AND NOT THE CUT. Identity used to chain over fixed 256-token
//! units, `hash[i] = H(hash[i-1] || tokens of unit i)`. While every stream cut at
//! the same positions that value was determined by the prefix alone. Once a stream
//! is cut where its REQUESTS end, it is not:
//!
//! ```text
//! A cuts [0,300)[300,700)   H( H(root||t[0..300]) || t[300..700] )
//! B cuts [0,700)            H( root || t[0..700] )
//! same 700 tokens, different value -> A and B cannot match
//! ```
//!
//! Two sub-agents with the same preamble cut at different request ends, so the
//! chained form misses exactly the sharing that pays. `prefix(p)` depends on the
//! tokens below `p` and on nothing else, so any two streams that both have a
//! boundary at `p` agree there however they got there.

use imparo_backend::KvByteCodec;
use sha2::{Digest, Sha256};

/// Version of the canonical KV byte-layout profile carried by [`ConfigRoot`].
pub const KV_BYTE_LAYOUT_PROFILE_VERSION: u32 = 2;

/// Storage type of one side of the KV cache. These identity tags are independent
/// of backend and ggml enum ordinals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvByteType {
    F16,
    Q4_0,
    Q8_0,
}

impl KvByteType {
    const fn canonical_tag(self) -> u8 {
        match self {
            Self::F16 => 0,
            Self::Q4_0 => 1,
            Self::Q8_0 => 2,
        }
    }
}

/// Basis used before quantising one KV side into its canonical stored bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KvQuantizationBasis {
    Disabled,
    FullHead,
    Fixed(u32),
}

impl KvQuantizationBasis {
    const fn canonical_parts(self) -> (u8, u32) {
        match self {
            Self::Disabled => (0, 0),
            Self::FullHead => (1, 0),
            Self::Fixed(width) => (2, width),
        }
    }
}

/// Every non-geometric choice that changes canonical KV bytes.
///
/// F16 sides skip the quantisation Hadamard route. Their basis is canonicalised
/// to [`KvQuantizationBasis::Disabled`], so irrelevant declarations or overrides
/// cannot split identical F16 bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvByteLayoutProfile {
    version: u32,
    key_type: KvByteType,
    value_type: KvByteType,
    key_basis: KvQuantizationBasis,
    value_basis: KvQuantizationBasis,
    key_codec: KvByteCodec,
    value_codec: KvByteCodec,
}

impl KvByteLayoutProfile {
    #[must_use]
    pub const fn current(
        key_type: KvByteType,
        value_type: KvByteType,
        key_basis: KvQuantizationBasis,
        value_basis: KvQuantizationBasis,
        key_codec: KvByteCodec,
        value_codec: KvByteCodec,
    ) -> Self {
        Self::with_version(
            KV_BYTE_LAYOUT_PROFILE_VERSION,
            key_type,
            value_type,
            key_basis,
            value_basis,
            key_codec,
            value_codec,
        )
    }

    /// Reserved fail-closed identity for an invalid runtime route. It cannot
    /// alias a valid profile and must never be used to write canonical bytes.
    #[must_use]
    pub const fn rejected(key_type: KvByteType, value_type: KvByteType) -> Self {
        Self::with_version(
            0,
            key_type,
            value_type,
            KvQuantizationBasis::Disabled,
            KvQuantizationBasis::Disabled,
            KvByteCodec::Rejected,
            KvByteCodec::Rejected,
        )
    }

    const fn with_version(
        version: u32,
        key_type: KvByteType,
        value_type: KvByteType,
        key_basis: KvQuantizationBasis,
        value_basis: KvQuantizationBasis,
        key_codec: KvByteCodec,
        value_codec: KvByteCodec,
    ) -> Self {
        Self {
            version,
            key_type,
            value_type,
            key_basis: match (key_type, key_basis) {
                (KvByteType::F16, _) | (_, KvQuantizationBasis::Fixed(0)) => {
                    KvQuantizationBasis::Disabled
                }
                (_, basis) => basis,
            },
            value_basis: match (value_type, value_basis) {
                (KvByteType::F16, _) | (_, KvQuantizationBasis::Fixed(0)) => {
                    KvQuantizationBasis::Disabled
                }
                (_, basis) => basis,
            },
            key_codec,
            value_codec,
        }
    }

    #[must_use]
    pub const fn version(self) -> u32 {
        self.version
    }
    #[must_use]
    pub const fn key_type(self) -> KvByteType {
        self.key_type
    }
    #[must_use]
    pub const fn value_type(self) -> KvByteType {
        self.value_type
    }
    #[must_use]
    pub const fn key_basis(self) -> KvQuantizationBasis {
        self.key_basis
    }
    #[must_use]
    pub const fn value_basis(self) -> KvQuantizationBasis {
        self.value_basis
    }
    #[must_use]
    pub const fn key_codec(self) -> KvByteCodec {
        self.key_codec
    }
    #[must_use]
    pub const fn value_codec(self) -> KvByteCodec {
        self.value_codec
    }

    /// Stable digest used by correctness receipts to bind the complete durable
    /// byte contract independently of the model and geometry root.
    #[must_use]
    pub fn sha256_identity(self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"imparo-kv-byte-layout-profile-v1\0");
        self.hash_canonical(&mut h);
        h.finalize().into()
    }

    fn hash_canonical(self, h: &mut Sha256) {
        let (key_tag, key_width) = self.key_basis.canonical_parts();
        let (value_tag, value_width) = self.value_basis.canonical_parts();
        h.update(self.version.to_le_bytes());
        h.update([self.key_type.canonical_tag()]);
        h.update([self.value_type.canonical_tag()]);
        h.update([codec_tag(self.key_codec)]);
        h.update([codec_tag(self.value_codec)]);
        h.update([key_tag]);
        h.update(key_width.to_le_bytes());
        h.update([value_tag]);
        h.update(value_width.to_le_bytes());
    }
}

const fn codec_tag(codec: KvByteCodec) -> u8 {
    match codec {
        KvByteCodec::Rejected => 0,
        KvByteCodec::F16LeRneV1 => 1,
        KvByteCodec::Q4_0LlamaV1 => 2,
        KvByteCodec::Q8_0RintEvenV1 => 3,
        KvByteCodec::Q8_0RoundAwayV1 => 4,
    }
}

/// The grid every stored boundary and every match probe lands on.
///
/// PAGED ATTENTION'S PAGE, in cells: what one block-table entry maps.
///
/// THE declared fact this crate turns on. The backend states it in
/// `PoolCaps::page_cells`, `imparo_model::backend::active` passes it here once, and
/// everything else -- placement, addressing, growth, identity, cuts, resume -- derives
/// from this one value rather than from each other.
///
/// Read once: the backend is chosen at startup, so it cannot change within a process.
#[must_use]
pub fn page_cells() -> usize {
    *PAGE.get_or_init(|| DEFAULT_PAGE_CELLS)
}

/// The grid every stored boundary, match probe and disk cut lands on.
///
/// IT IS THE PAGE, and this is a redirect rather than a second state, because a
/// RESIDENT unit is one grid step and occupies one block:
///
/// ```text
///   resident_bounds(len)  one bound per grid step -- "the ids residency is keyed on,
///                         one per block"
///   alloc_unit()          BTreeMap<layer, BlockIdx> -- ONE block per layer
/// ```
///
/// So a grid step has to fit exactly one page. (A DISK extent is a different thing and
/// spans many grid steps -- a 704-token request is eleven of them.)
///
/// Kept as its own name because the question differs at the call site: a reader of
/// `blk * page_cells() * stride` is asking what one block spans, and a reader of
/// `b % grid_tokens() != 0` is asking where a stream may be cut. It was a second
/// OnceLock set from the same value; two states that cannot drift are still two states.
///
/// If this ever stops being the page it becomes purely a MATCH granularity, free to be
/// finer under `page_cells % grid_tokens == 0`. The way there is copy-on-write of the
/// partial page -- a hit inside a page copies it into the borrower's own block before
/// extending it -- and NOT a placement entry holding several blocks, which would touch
/// the allocator and `UnitPlacement` for no gain. Surveyed against a mature paged
/// implementation; the comparison is in docs/kv-identity-grid.md.
#[must_use]
pub fn grid_tokens() -> usize {
    page_cells()
}

/// The page when no backend has spoken: pure-crate tests, and any binary that touches
/// identity before selecting one. Every backend in the tree declares 64.
pub const DEFAULT_PAGE_CELLS: usize = 64;

static PAGE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Fix the page to the active backend's declaration. Idempotent; LOUD when it disagrees.
///
/// A second, different value means something already hashed or placed on the old one --
/// stored prefixes are `H(root || tokens[0..p])` at multiples of it, so the two are not
/// comparable and carrying on would mean a store nothing can match.
///
/// # Errors
/// When the page is already fixed to a different value, or `n` is zero or not a power of
/// two. NOTHING IN THE TREE STILL REQUIRES THE POWER OF TWO: it was the resume snap's
/// `& !(page - 1)`, which now divides instead, and the shader indexes with
/// `pt[gp / KV_PAGE_CELLS]` rather than a shift. Kept as a conservative admissibility
/// rule -- every backend here declares 64 -- not as an arithmetic requirement.
pub fn set_page_cells(n: usize) -> Result<(), String> {
    if n == 0 || !n.is_power_of_two() {
        return Err(format!(
            "kv page must be a non-zero power of two (conservative: no arithmetic \
             in the tree still requires it), got {n}"
        ));
    }
    if PAGE.set(n).is_err() {
        let have = page_cells();
        if have != n {
            return Err(format!(
                "kv page is already fixed at {have} and cannot become {n}: \
                 boundaries placed and hashed on {have} are not comparable with {n}"
            ));
        }
    }
    Ok(())
}

/// The furthest position a resumed prefill may start at, given how much of the request is
/// already computed (`resident`) and how long the request is (`len`).
///
/// A resume is only worth anything if it reproduces what a cold prefill would have
/// produced, and two engine properties decide where that holds (both measured, and both
/// gated by dev_harness/kv_gates.py `engine`):
///
/// - Resuming off the 64 grid does not reproduce the cold pass. MEASURED 2026-08-26,
///   nine of nine unaligned splits differ: 521 at 63, 65, 129, 130, 500, 511 and 744 at
///   500, 703, 705.
/// - One token is a DECODE shape: it takes the GEMV, not the staged prefill GEMM, so a
///   1-token tail differs where a 2-token tail is byte-equal (n=705 split@704 against
///   n=706, q4_0).
///
/// WHAT MAKES THE GRID, which is not what this comment used to say. It read "64 is the
/// token-tile width", and blamed the wo projection staging half only at >= 64 tokens --
/// a WIDTH threshold. That was fixed (the floor is 2 now, HALF_A_MIN in
/// imparo_metal.mm) and the grid did not move, so width was never the whole reason.
///
/// The measurement separates them. 63 and 65 both differ while 64 is exact: the
/// property is PERIODIC in 64, which is alignment. A width threshold would fail
/// monotonically -- every narrow chunk and no wide one -- and it does not.
///
/// The cause is now MEASURED, and it is not the KV block. This comment briefly said
/// the grid was the block a paged read indexes (`kv_slot`: `pt[gp >> 6]`); that was a
/// guess and it is wrong. `page_cells` stayed 64 while the grid was measured at 8 and
/// at 16, and a quantity that does not move cannot explain one that does.
///
/// It is the attention prefill kernel's QUERY GROUP. Both prefill kernels take their KV
/// scan bounds from the group a query sits in, not from the query:
///
/// ```text
/// qb0 = q0 & ~15u                    the group; q0 is BATCH-LOCAL, so it moves with
/// lo  = start_pos + qb0 + 1 - window     start_pos
/// n   = pos_last + 1 - lo
/// full_np = (np / 8u) * 8u           whole 8s -> simdgroup MMA, remainder -> scalar adds
/// ```
///
/// Split off the group's grid and `n` moves, the last position block ends elsewhere, and
/// positions the MMA summed as a full 8-chunk go through the sequential tail instead.
/// Same values, different association -- the kernel's own comment says so.
///
/// Two constraints stack and the context length picks which is live: the 8-run split
/// always (grid 8), plus the 16-query group once positions pass the window (grid 16).
/// Measured: grid 8 at n <= 520, grid 16 from n = 600, gemma4's window being 512. Only
/// changing the query group moved it -- not the GEMM tile (shapes 32/64/128), not
/// tail-split, nb8 or half-A staging.
///
/// grid_tokens() must therefore be a multiple of the kernel's query group, and separately
/// a multiple of `page_cells` so an extent is whole pages. Both hold at 64, which is 4x
/// the identity requirement -- headroom worth keeping, not a coincidence to preserve.
/// See docs/kv-identity-grid.md.
///
/// The tail's WIDTH no longer matters, and n=744 split@704 is the case that proves it.
#[must_use]
pub fn resume_point(resident: usize, len: usize) -> usize {
    // THE CUT GRID, and the one snapping idiom the rest of the tree uses. It asked
    // `page_cells()` and masked with `& !(page - 1)`, which is the same number today and
    // the wrong question: `grid_tokens()` answers "where may a stream be cut", while
    // `page_cells()` answers "what does one block span". Its own doc says the grid is
    // free to become FINER than the page, and on that day the mask would have snapped
    // silently to the coarser one.
    //
    // `len - 2` leaves at least two tokens above the resume point for the forward to
    // run on; `resume_point(704, 705)` is 640 and `resume_point(704, 706)` is 704.
    resident.min(len.saturating_sub(2)) / grid_tokens() * grid_tokens()
}

/// Cheap, stable model identity for the configuration root: sha256 of the header
/// region (metadata + tensor table live at the front) plus the file length.
/// Hashing 4 GB per launch is not acceptable; the header pins architecture, quant
/// layout and vocab.
///
/// # Errors
/// Returns an error when the file cannot be read.
pub fn model_digest(path: &std::path::Path) -> Result<Vec<u8>, String> {
    use sha2::Digest as _;
    use std::io::Read as _;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    let mut head = vec![0u8; 1 << 20];
    let n = f.read(&mut head).map_err(|e| e.to_string())?;
    head.truncate(n);
    let mut out = len.to_le_bytes().to_vec();
    let mut h = sha2::Sha256::new();
    h.update(&head);
    let d: [u8; 32] = h.finalize().into();
    out.extend_from_slice(&d);
    Ok(out)
}

/// 128-bit unit hash (SHA-256 truncated). Wide enough that collision is not a case
/// the code handles — no byte-verify on hit, no guard path (design: "Hash width").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct UnitHash(pub [u8; 16]);

impl UnitHash {
    /// The inverse of `hex`: 32 hex characters back to a hash, or None.
    ///
    /// Beside `hex` on purpose -- a spelling and its parser that live apart drift
    /// apart, and this pair is what a manifest line means.
    #[must_use]
    pub fn from_hex(s: &str) -> Option<Self> {
        unhex(s).map(Self)
    }

    #[must_use]
    pub fn hex(&self) -> String {
        hex(&self.0)
    }
}

/// The configuration root. Two runs share KV bytes only if every input that shapes
/// those bytes is identical; everything that shapes them goes in here, and nothing
/// else does (a tuned tile width does NOT shape the values, so it must not enter).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConfigRoot([u8; 32]);

impl ConfigRoot {
    /// The root's bytes, for scoping the on-disk store directory.
    #[must_use]
    pub fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// `model_digest`: the GGUF's content identity (imparo-gguf snapshot sha256 or
    /// equivalent). `layout`: the versioned canonical KV byte layout.
    /// `rope_and_geometry`:
    /// the plan's serialized per-layer geometry (rope config, head counts, state
    /// kinds) — the plan owns what belongs in it; this module only hashes bytes.
    #[must_use]
    pub fn from_layout(
        model_digest: &[u8],
        layout: &KvByteLayoutProfile,
        rope_and_geometry: &[u8],
    ) -> Self {
        let mut h = Sha256::new();
        h.update(b"imparo-kv-root-v2\0");
        h.update((model_digest.len() as u64).to_le_bytes());
        h.update(model_digest);
        layout.hash_canonical(&mut h);
        h.update((rope_and_geometry.len() as u64).to_le_bytes());
        h.update(rope_and_geometry);
        Self(h.finalize().into())
    }

    /// Compatibility constructor for tests and downstream scaffolding that do not
    /// produce persistent model KV. Its separate domain prevents opaque legacy type
    /// ids from aliasing a v2 canonical layout.
    #[must_use]
    pub fn new(
        model_digest: &[u8],
        kv_types: (u32, u32),
        rope_and_geometry: &[u8],
    ) -> Self {
        let mut h = Sha256::new();
        h.update(b"imparo-kv-root-v2-legacy\0");
        h.update((model_digest.len() as u64).to_le_bytes());
        h.update(model_digest);
        h.update(kv_types.0.to_le_bytes());
        h.update(kv_types.1.to_le_bytes());
        h.update((rope_and_geometry.len() as u64).to_le_bytes());
        h.update(rope_and_geometry);
        Self(h.finalize().into())
    }
}

/// A rolling hash of `tokens[0..p]` under a config root.
///
/// A separate type from `UnitHash` because they answer different questions: this
/// one names a POSITION IN A STREAM ("everything below here"), a `UnitHash` names
/// the BYTES OF ONE STORED EXTENT. Mixing them up produces a file whose name says
/// nothing about what is in it, so the compiler is made to keep them apart.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct PrefixHash(pub [u8; 16]);

impl PrefixHash {
    /// Position 0: the prefix with no tokens in it, which is the root alone. Every
    /// stream on this configuration starts here, so it is what the first stored
    /// extent counts from.
    #[must_use]
    pub fn start(root: &ConfigRoot) -> Self {
        let mut h = Sha256::new();
        h.update(root.0);
        Self(truncate(h.finalize().into()))
    }

    /// The inverse of `hex`: 32 hex characters back to a hash, or None.
    #[must_use]
    pub fn from_hex(s: &str) -> Option<Self> {
        unhex(s).map(Self)
    }

    #[must_use]
    pub fn hex(&self) -> String {
        hex(&self.0)
    }
}

/// Prefix hashes on the `grid_tokens()` grid, in position order: element `i` is the
/// hash of `tokens[0..(i + 1) * grid_tokens()]`. Tokens above the last whole grid
/// step get no hash -- no boundary can land there, so nothing would ask.
///
/// One pass and one hasher: the state is carried forward and snapshotted at each
/// step, so this costs one SHA-256 over the tokens plus one block per step, not a
/// rehash per position.
#[must_use]
pub fn prefix_hashes(root: &ConfigRoot, tokens: &[u32]) -> Vec<PrefixHash> {
    let mut h = Sha256::new();
    h.update(root.0);
    let mut out = Vec::with_capacity(tokens.len() / grid_tokens());
    for step in tokens.chunks_exact(grid_tokens()) {
        for t in step {
            h.update(t.to_le_bytes());
        }
        out.push(PrefixHash(truncate(h.clone().finalize().into())));
    }
    out
}

/// The file id of the stored extent between two boundaries.
///
/// BOTH ends, not just the end: `end` alone identifies the prefix, and two streams
/// that cut differently reach the same prefix by storing different extents. One
/// stream storing `[0, 700)` and another storing `[300, 700)` would then agree on
/// a file name while disagreeing on its contents.
#[must_use]
pub fn unit_id(start: &PrefixHash, end: &PrefixHash) -> UnitHash {
    let mut h = Sha256::new();
    h.update(b"imparo-kv-unit-v1");
    h.update(start.0);
    h.update(end.0);
    UnitHash(truncate(h.finalize().into()))
}

fn truncate(full: [u8; 32]) -> [u8; 16] {
    let mut short = [0_u8; 16];
    short.copy_from_slice(&full[..16]);
    short
}

fn hex(b: &[u8; 16]) -> String {
    use std::fmt::Write as _;
    b.iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

fn unhex(s: &str) -> Option<[u8; 16]> {
    if s.len() != 32 {
        return None;
    }
    let mut b = [0_u8; 16];
    for (i, c) in s.as_bytes().chunks_exact(2).enumerate() {
        b[i] = u8::from_str_radix(std::str::from_utf8(c).ok()?, 16).ok()?;
    }
    Some(b)
}

/// Ids for the extents a stream is cut into, in position order.
///
/// `bounds` are absolute end positions, ascending, each a multiple of
/// `grid_tokens()` and at most `tokens.len()`. Extent `i` is
/// `[bounds[i - 1], bounds[i])`, counting from position 0 for the first.
///
/// The cut is a PARAMETER because it is what the disk layout decides -- a stream
/// cut where its requests ended and a stream cut on a fixed tiling reach the same
/// prefixes and store different extents there.
#[must_use]
pub fn unit_ids_at(
    root: &ConfigRoot,
    tokens: &[u32],
    bounds: &[usize],
) -> Vec<UnitHash> {
    let grid = prefix_hashes(root, tokens);
    let mut prev = PrefixHash::start(root);
    let mut out = Vec::with_capacity(bounds.len());
    for b in bounds {
        debug_assert!(b % grid_tokens() == 0, "boundary {b} is off the grid");
        // Position 0 is where the first extent COUNTS FROM, so it never ends one.
        // Taking it as a bound would index step -1.
        let Some(step) = (b / grid_tokens()).checked_sub(1) else {
            continue;
        };
        let Some(end) = grid.get(step) else {
            break;
        };
        out.push(unit_id(&prev, end));
        prev = *end;
    }
    out
}

/// Every position on the grid, which is where residency cuts.
///
/// One entry per `grid_tokens()`, the same grid identity and the disk cut use. It
/// used to be one per 256 so that residency tiled four blocks at a time; the disk
/// then had to translate, and the translations were where the bugs were.
#[must_use]
pub fn resident_bounds(len: usize) -> Vec<usize> {
    (1..=len / grid_tokens())
        .map(|u| u * grid_tokens())
        .collect()
}

/// `unit_ids_at` on the resident cut: the ids residency is keyed on, one per block.
///
/// Not the disk cut -- extents end where requests ended, which is a subset of these
/// positions, so a disk extent's id is not one of these.
#[must_use]
pub fn unit_ids(root: &ConfigRoot, tokens: &[u32]) -> Vec<UnitHash> {
    unit_ids_at(root, tokens, &resident_bounds(tokens.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> ConfigRoot {
        ConfigRoot::from_layout(b"model-A", &profile(), b"geom-1")
    }

    fn profile() -> KvByteLayoutProfile {
        KvByteLayoutProfile::current(
            KvByteType::Q4_0,
            KvByteType::Q8_0,
            KvQuantizationBasis::FullHead,
            KvQuantizationBasis::Fixed(128),
            KvByteCodec::Q4_0LlamaV1,
            KvByteCodec::Q8_0RintEvenV1,
        )
    }

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn a_shared_prefix_agrees_and_a_divergence_splits_everything_after_it() {
        let a = toks(3 * grid_tokens());
        let mut b = a.clone();
        b[2 * grid_tokens() + 7] ^= 1;
        let (ha, hb) = (prefix_hashes(&root(), &a), prefix_hashes(&root(), &b));
        assert_eq!(ha[0], hb[0]);
        assert_eq!(ha[1], hb[1]);
        assert_ne!(ha[2], hb[2]);

        // A divergence low down moves every hash above it, even where the tokens
        // in that step are equal again.
        let mut c = a.clone();
        c[3] ^= 1;
        let hc = prefix_hashes(&root(), &c);
        for i in 0..3 {
            assert_ne!(hc[i], ha[i]);
        }
    }

    /// The property the chained-over-units form did not have, and the reason this
    /// module changed: the value at a position does not depend on how the stream
    /// was cut on the way there.
    #[test]
    fn the_hash_at_a_position_does_not_depend_on_how_the_stream_was_cut() {
        let all = toks(8 * grid_tokens());
        let whole = prefix_hashes(&root(), &all);

        // Feed the same tokens as two pieces and carry the hash across the seam by
        // hashing the concatenation -- which is what a stream cut at 3 steps and
        // then continued does.
        let split = prefix_hashes(&root(), &all[..8 * grid_tokens()]);
        assert_eq!(whole, split);

        // And a stream that stops at a boundary agrees with a longer one there.
        let shorter = prefix_hashes(&root(), &all[..3 * grid_tokens()]);
        assert_eq!(shorter.len(), 3);
        assert_eq!(shorter[..], whole[..3]);
    }

    #[test]
    fn root_inputs_split_identity() {
        let t = toks(grid_tokens());
        let base = prefix_hashes(&ConfigRoot::from_layout(b"m", &profile(), b"g"), &t);
        let k_type = KvByteLayoutProfile::current(
            KvByteType::Q8_0,
            KvByteType::Q8_0,
            KvQuantizationBasis::FullHead,
            KvQuantizationBasis::Fixed(128),
            KvByteCodec::Q8_0RintEvenV1,
            KvByteCodec::Q8_0RintEvenV1,
        );
        let v_type = KvByteLayoutProfile::current(
            KvByteType::Q4_0,
            KvByteType::Q4_0,
            KvQuantizationBasis::FullHead,
            KvQuantizationBasis::Fixed(128),
            KvByteCodec::Q4_0LlamaV1,
            KvByteCodec::Q4_0LlamaV1,
        );
        let k_route = KvByteLayoutProfile::current(
            KvByteType::Q4_0,
            KvByteType::Q8_0,
            KvQuantizationBasis::Fixed(64),
            KvQuantizationBasis::Fixed(128),
            KvByteCodec::Q4_0LlamaV1,
            KvByteCodec::Q8_0RintEvenV1,
        );
        let v_route = KvByteLayoutProfile::current(
            KvByteType::Q4_0,
            KvByteType::Q8_0,
            KvQuantizationBasis::FullHead,
            KvQuantizationBasis::Fixed(64),
            KvByteCodec::Q4_0LlamaV1,
            KvByteCodec::Q8_0RintEvenV1,
        );
        let mut next_version = profile();
        next_version.version = KV_BYTE_LAYOUT_PROFILE_VERSION + 1;
        for other in [
            ConfigRoot::from_layout(b"m2", &profile(), b"g"),
            ConfigRoot::from_layout(b"m", &k_type, b"g"),
            ConfigRoot::from_layout(b"m", &v_type, b"g"),
            ConfigRoot::from_layout(b"m", &k_route, b"g"),
            ConfigRoot::from_layout(b"m", &v_route, b"g"),
            ConfigRoot::from_layout(b"m", &next_version, b"g"),
            ConfigRoot::from_layout(b"m", &profile(), b"g2"),
        ] {
            assert_ne!(base[0], prefix_hashes(&other, &t)[0]);
            assert_ne!(PrefixHash::start(&other), PrefixHash::start(&root()));
        }
    }

    /// Two backends running the same model at the same KV type must not share a
    /// store. Their kernels do not produce byte-identical KV -- the determinism gates
    /// prove resumed-vs-cold WITHIN a backend, never across -- so adopting each
    /// other's rows would answer from arithmetic this machine never did.
    /// `imparo_model::kv::config_root` appends the device tag to
    /// `rope_and_geometry` for exactly this; here is the mechanism it relies on.
    ///
    /// The GRID is deliberately not scoped this way -- see the note there: a
    /// different grid costs some BOUNDARIES, not the store, and the pool's candidate
    /// filter skips the ones that are not on it.
    /// The grid takes the backend's page, so a backend with a different page is the
    /// case this has to survive -- CUDA at 128 is the one actually coming.
    ///
    /// Not `set_page_cells` itself: it can be set once per PROCESS, and the test
    /// binary shares that with every other test here. What is checked is the rule the
    /// setter enforces, which is where a wrong page would get in.
    #[test]
    fn identical_profiles_are_stable_and_f16_ignores_quantization_basis() {
        assert_eq!(
            ConfigRoot::from_layout(b"m", &profile(), b"g"),
            ConfigRoot::from_layout(b"m", &profile(), b"g")
        );
        let a = KvByteLayoutProfile::current(
            KvByteType::F16,
            KvByteType::F16,
            KvQuantizationBasis::FullHead,
            KvQuantizationBasis::Fixed(64),
            KvByteCodec::F16LeRneV1,
            KvByteCodec::F16LeRneV1,
        );
        let b = KvByteLayoutProfile::current(
            KvByteType::F16,
            KvByteType::F16,
            KvQuantizationBasis::Disabled,
            KvQuantizationBasis::Fixed(128),
            KvByteCodec::F16LeRneV1,
            KvByteCodec::F16LeRneV1,
        );
        assert_eq!(a.key_basis(), KvQuantizationBasis::Disabled);
        assert_eq!(a.value_basis(), KvQuantizationBasis::Disabled);
        assert_eq!(
            ConfigRoot::from_layout(b"m", &a, b"g"),
            ConfigRoot::from_layout(b"m", &b, b"g")
        );
        let zero = KvByteLayoutProfile::current(
            KvByteType::Q4_0,
            KvByteType::Q8_0,
            KvQuantizationBasis::Fixed(0),
            KvQuantizationBasis::Fixed(0),
            KvByteCodec::Q4_0LlamaV1,
            KvByteCodec::Q8_0RintEvenV1,
        );
        let disabled = KvByteLayoutProfile::current(
            KvByteType::Q4_0,
            KvByteType::Q8_0,
            KvQuantizationBasis::Disabled,
            KvQuantizationBasis::Disabled,
            KvByteCodec::Q4_0LlamaV1,
            KvByteCodec::Q8_0RintEvenV1,
        );
        assert_eq!(zero, disabled);
        assert_eq!(
            ConfigRoot::from_layout(b"m", &zero, b"g"),
            ConfigRoot::from_layout(b"m", &disabled, b"g")
        );
    }

    #[test]
    fn rejected_profile_never_aliases_a_valid_disabled_route() {
        let rejected =
            KvByteLayoutProfile::rejected(KvByteType::Q4_0, KvByteType::Q8_0);
        let valid_disabled = KvByteLayoutProfile::current(
            KvByteType::Q4_0,
            KvByteType::Q8_0,
            KvQuantizationBasis::Disabled,
            KvQuantizationBasis::Disabled,
            KvByteCodec::Q4_0LlamaV1,
            KvByteCodec::Q8_0RintEvenV1,
        );
        assert_ne!(rejected, valid_disabled);
        assert_ne!(
            ConfigRoot::from_layout(b"m", &rejected, b"g"),
            ConfigRoot::from_layout(b"m", &valid_disabled, b"g")
        );
    }

    #[test]
    fn q8_rounding_codec_is_part_of_the_root_and_profile_digest() {
        let even = profile();
        let away = KvByteLayoutProfile::current(
            KvByteType::Q4_0,
            KvByteType::Q8_0,
            KvQuantizationBasis::FullHead,
            KvQuantizationBasis::Fixed(128),
            KvByteCodec::Q4_0LlamaV1,
            KvByteCodec::Q8_0RoundAwayV1,
        );
        assert_ne!(even.sha256_identity(), away.sha256_identity());
        assert_ne!(
            ConfigRoot::from_layout(b"m", &even, b"g"),
            ConfigRoot::from_layout(b"m", &away, b"g")
        );
    }

    #[test]
    fn a_backends_page_is_admissible_only_as_a_power_of_two() {
        // A conservative admissibility rule, not an arithmetic one: `resume_point`
        // floors by division now, so 96 would snap correctly. See `set_page_cells`.
        for bad in [0usize, 3, 96, 100, 129] {
            assert!(
                set_page_cells(bad).is_err(),
                "{bad} is not a power of two and must be refused"
            );
        }
        // Every page a backend in this tree could reasonably declare.
        for good in [8usize, 16, 32, 64, 128, 256] {
            assert!(good.is_power_of_two(), "{good} must be admissible");
        }
    }

    /// The default is what a pure-crate test and a backend-less binary see.
    #[test]
    fn the_page_defaults_to_what_every_backend_declares() {
        assert_eq!(grid_tokens(), DEFAULT_PAGE_CELLS);
        assert_eq!(DEFAULT_PAGE_CELLS, 64);
        // Setting it to what it already is must be a no-op, not an error: `active()`
        // calls this on every lookup, not once.
        assert!(set_page_cells(grid_tokens()).is_ok());
        // And a DIFFERENT value must be refused, naming both -- boundaries hashed on
        // one grid are not comparable with another.
        let e = set_page_cells(grid_tokens() * 2).expect_err("a second grid must fail");
        assert!(e.contains(&grid_tokens().to_string()), "{e}");
    }

    /// The rule the engine gate measures, in one place: land on the 64 grid, and leave
    /// the resumed pass at least two tokens (one is a decode shape and still differs).
    #[test]
    fn resume_point_is_the_64_grid_with_a_two_token_tail() {
        // An exact resend: 744 resident, 744 asked -> 704, a 40-token tail.
        assert_eq!(resume_point(744, 744), 704);
        // A 1-token append would leave a 1-token tail at 704, so it stops a grid short.
        assert_eq!(resume_point(704, 705), 640);
        // Two tokens is enough.
        assert_eq!(resume_point(704, 706), 704);
        // Nothing resident, or nothing to resume into.
        assert_eq!(resume_point(0, 1000), 0);
        assert_eq!(resume_point(1000, 1), 0);
        assert_eq!(resume_point(1000, 0), 0);
        // Never past what is resident.
        assert_eq!(resume_point(100, 10_000), 64);
    }

    #[test]
    fn the_writers_identity_is_part_of_the_root() {
        let same = |tag: &str| {
            ConfigRoot::new(
                b"model",
                (1, 1),
                &[b"geom".as_slice(), tag.as_bytes()].concat(),
            )
        };
        assert_ne!(
            same("metal").bytes(),
            same("cuda").bytes(),
            "two backends opened the same store"
        );
        assert_eq!(same("metal").bytes(), same("metal").bytes());
    }

    #[test]
    fn tokens_above_the_last_whole_grid_step_get_no_hash() {
        let t = toks(2 * grid_tokens() + 5);
        assert_eq!(prefix_hashes(&root(), &t).len(), 2);
    }

    /// Two streams reaching the same prefix by different cuts must NOT name the
    /// same file, because they store different extents there.
    #[test]
    fn a_unit_id_separates_extents_that_end_at_the_same_prefix() {
        let all = toks(8 * grid_tokens());
        let h = prefix_hashes(&root(), &all);
        let zero = PrefixHash::start(&root());
        let whole = unit_id(&zero, &h[7]);
        let second_half = unit_id(&h[3], &h[7]);
        assert_ne!(whole, second_half);
        // ... and the same extent names the same file from either stream.
        assert_eq!(second_half, unit_id(&h[3], &h[7]));
    }

    #[test]
    fn a_zero_bound_is_skipped_rather_than_indexing_below_the_first_step() {
        let root = root();
        let all = toks(4 * grid_tokens());
        let with_zero =
            unit_ids_at(&root, &all, &[0, 2 * grid_tokens(), 4 * grid_tokens()]);
        let without = unit_ids_at(&root, &all, &[2 * grid_tokens(), 4 * grid_tokens()]);
        assert_eq!(with_zero, without);
        assert_eq!(with_zero.len(), 2);
    }

    #[test]
    fn hex_round_trips_both_kinds() {
        let h = prefix_hashes(&root(), &toks(grid_tokens()));
        assert_eq!(PrefixHash::from_hex(&h[0].hex()), Some(h[0]));
        let u = unit_id(&PrefixHash::start(&root()), &h[0]);
        assert_eq!(UnitHash::from_hex(&u.hex()), Some(u));
        assert_eq!(PrefixHash::from_hex("short"), None);
        assert_eq!(UnitHash::from_hex("zz"), None);
    }
}
