//! The disk tier: sealed units (immutable, content-addressed, independent) and
//! per-conversation manifests + checkpoint blobs (mutable, superseded at each
//! boundary). Design: docs/unified-kv-pool.md "Disk tier" / "Ownership, deletion,
//! and restart".
//!
//! Everything here is opaque bytes and hashes -- no model knowledge, no device
//! knowledge. Crash safety is tmp+rename everywhere, manifest written LAST: a
//! kill at any moment leaves either the old committed state or the new one,
//! never a half state; orphaned unit files are reclaimed by GC, never read
//! (nothing references them).
//!
//! Store layout, scoped by the ConfigRoot (a different model/quant is a
//! different world that must never be probed):
//!
//! ```text
//! <base>/<root-hex>/units/<hash-hex>      sealed extent blobs
//! <base>/<root-hex>/ckpt/<hash-hex>       checkpoint blobs, content-addressed
//! <base>/<root-hex>/conv/<name>.manifest  cuts + checkpoints for one conversation
//! <base>/<root-hex>/INDEX                 prefix -> conversation name
//! ```
//!
//! Checkpoints are reached through `Manifest::ckpts[].blob`, never by a name derived
//! from the manifest's path. Two readers that did the latter -- `checkpoint_for`
//! (`<manifest>.ckpt`) and `checkpoint` (`conv/<name>.ckpt`) -- were deleted on
//! 2026-08-26: nothing has ever written either file, so both returned None always, and
//! the server's non-pool restore branch was unreachable because of it.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::identity::{
    ConfigRoot, PrefixHash, UnitHash, grid_tokens, prefix_hashes, unit_id,
};

// THE MANIFEST, and why the magic no longer carries a number.
//
// ```text
//   imparo-kv-manifest      the magic, stable forever
//   format=2                bumped only when an existing key CHANGES MEANING
//   boundary=1234           where the committed state describes
//   cut=<end>:<32 hex>      one per stored extent, in order: where it ends, and
//                           prefix(end) -- the identity of that position
// ```
//
// `cut` replaced `unit=<32 hex>`, which named an extent and left its position to be
// derived by multiplying an index by 256. Extents end where a request ended, so
// there is no constant to multiply by; the position is written down. The extent's
// own file id is DERIVED from the pair of prefixes around it rather than stored,
// because a name kept beside the thing it names is a name that can disagree with it.
//
// Key = value, one per line, ANY order, and a key this build does not know is
// SKIPPED. That is the whole point: adding a key -- media descriptors for a vision
// model, a checkpoint chain, a pin flag -- costs nothing, where a positional format
// costs a version bump and every bump invalidates every store. The tokens between
// the last unit and the boundary are a key here for exactly that reason; adding them
// used to be a new magic.
const MANIFEST_MAGIC: &str = "imparo-kv-manifest";
/// See `MANIFEST_MAGIC`. Read `1..=FORMAT`; refuse anything newer, because a newer
/// writer may have changed what a key this build thinks it understands means.
const MANIFEST_FORMAT: u32 = 2;

/// Clone is cheap and deliberate: the disk WRITER thread owns one handle while
/// request threads read through another. A `Store` is two paths and one shared
/// cache of the index.
#[derive(Clone)]
pub struct Store {
    dir: PathBuf,
    base: PathBuf,
    /// The index, parsed once and refreshed by APPENDED BYTES ONLY.
    ///
    /// Shared across clones on purpose: the writer thread appends and the request
    /// threads read, and one cache serves both. The file is append-only, so its
    /// LENGTH is an exact version marker -- unchanged means the cache stands, longer
    /// means parse the new tail, shorter means a compaction rewrote it.
    index: std::sync::Arc<std::sync::RwLock<IndexCache>>,
}

#[derive(Default)]
struct IndexCache {
    /// Bytes of the index file this map was built from.
    len: u64,
    map: Index,
    /// What the file measured the last time it was rewritten from live manifests.
    /// Everything past it is a mix of live appends and lines rewinds left behind.
    compacted: u64,
}

/// One committed BOUNDARY: where it sits, the blob holding its state, and the tokens
/// between the last whole unit and it.
///
/// A conversation commits one of these per turn and KEEPS the older ones, because a
/// rewind or a branch aims at a turn boundary that is not the newest. They are content
/// addressed, so a boundary is written once however often it is committed, and two
/// conversations that reach identical state share the blob.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ckpt {
    pub boundary: u64,
    /// Where this link's own rows START. Everything below is the predecessor's, exactly
    /// as in the resident chain: a restore walks `from` down until the window is covered.
    /// Equal to `boundary` would mean a link that owns nothing; `0` is an anchor.
    pub from: u64,
    pub blob: UnitHash,
    /// Tokens between the last extent's end and `boundary`, proving that stretch.
    pub tail: Vec<u32>,
    /// Whether this boundary STARTS a user turn (a user message begins here) rather
    /// than ending one of that turn's tool-call steps.
    ///
    /// One state per user turn: the steps within a turn replace each other, so the
    /// kept set collapses to the newest boundary in each turn -- and that needs to
    /// know where a turn begins. A model that takes ten tool calls in a turn would
    /// otherwise spend ten of the cap's slots on one turn and evict the rewind
    /// targets the cap exists to protect.
    pub turn: bool,
}

/// One stored extent: where it ends, the identity of that position, and the file
/// holding its rows.
///
/// The file id is `unit_id(prefix before, prefix here)` and so looks derivable. It
/// is written down anyway, because the SWEEP walks the directories of config roots
/// other than the loaded one -- a root cannot be reconstructed from its directory
/// name, so nothing there can recompute an id. A manifest has to be able to say
/// which files it keeps alive without help. Nothing re-derives it on READ -- the id is
/// computed once by `Cut::for_stream` and trusted thereafter, which is what
/// `an_extents_id_is_the_hash_of_its_own_span` pins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    /// On the `grid_tokens()` grid: a position off it could not be resumed from.
    pub end: u64,
    /// The extent's file, `unit_id(prefix before, prefix here)`.
    ///
    /// The PREFIX at `end` is not here: it lives in the index, which is the whole
    /// point of having one. The unit id does, because the sweep walks the directories
    /// of config roots other than the loaded one and cannot recompute an id there.
    pub unit: UnitHash,
}

impl Cut {
    /// The cut list for `tokens` cut at `bounds` (absolute end positions, ascending,
    /// each a multiple of `grid_tokens()`).
    ///
    /// One pass: the prefix hashes come from a single rolling hash over the tokens,
    /// and each extent's file id is the pair of prefixes around it.
    /// Returns the cuts AND the prefix at each of them, in step.
    ///
    /// Both halves at once because they are computed together and stored apart: the
    /// manifest takes the cuts, `commit` hands the prefixes to the index. Returning
    /// only the cuts would mean rolling the same hash twice.
    #[must_use]
    pub fn for_stream(
        root: &ConfigRoot,
        tokens: &[u32],
        bounds: &[usize],
    ) -> (Vec<Self>, Vec<PrefixHash>) {
        let grid = prefix_hashes(root, tokens);
        let mut prev = PrefixHash::start(root);
        let mut cuts = Vec::with_capacity(bounds.len());
        let mut at = Vec::with_capacity(bounds.len());
        for &end in bounds {
            // OFF THE GRID: `end / grid` truncates, so a bound of 100 on a 64 grid
            // would push a cut claiming end=100 carrying the id of the prefix at 64 --
            // an extent whose name describes a different span than its `end` says.
            // Every bound is `resume_point(..)` today and so a grid multiple; this
            // refuses the case rather than mis-naming it, because the result would be
            // a store that reads back wrong instead of failing.
            if end % grid_tokens() != 0 {
                continue;
            }
            let Some(step) = (end / grid_tokens()).checked_sub(1) else {
                continue;
            };
            let Some(prefix) = grid.get(step).copied() else {
                break;
            };
            cuts.push(Self {
                end: end as u64,
                unit: unit_id(&prev, &prefix),
            });
            at.push(prefix);
            prev = prefix;
        }
        (cuts, at)
    }
}

/// A conversation's durable record: the unit-hash chain up to `boundary`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    /// Where the committed state describes -- the conversation's tip, which is
    /// NOT on the unit grid.
    pub boundary: u64,
    /// The stored extents below `boundary`, in order. Extent `i` covers
    /// `[cuts[i - 1].end, cuts[i].end)`, counting from 0 for the first.
    pub cuts: Vec<Cut>,
    /// EVERY boundary this conversation still has state for, shallowest first. The
    /// newest is also spelled by `boundary`/`tail` above, which is what a build that
    /// has never heard of this field reads.
    pub ckpts: Vec<Ckpt>,
    /// Whether the label this was written under is CONTENT-DERIVED (a client that sent
    /// no conversation id) rather than a name the client can ask for again.
    ///
    /// It decides who may delete this file. A content-derived name moves every time the
    /// conversation grows past a unit boundary and is minted fresh on every restart, so
    /// one conversation leaves a trail of manifests that nothing can address and nothing
    /// removes until the store hits its cap. A newer one that covers this one may
    /// therefore supersede it. A client-supplied id is an identity: only `erase` removes
    /// it. Absent (an older file) reads as false, which is the safe direction.
    pub keyless: bool,
}

/// A whole state written to the store, waiting to be named.
pub struct StagedState {
    /// The extents, in order -- what the manifest records.
    pub cuts: Vec<Cut>,
    /// The prefix AT each cut, which is what the index is fed.
    pub at: Vec<PrefixHash>,
    blob: UnitHash,
    boundary: u64,
}

impl StagedState {
    /// The extent ids, for a caller that derives a conversation name from its content.
    #[must_use]
    pub fn unit_hashes(&self) -> Vec<UnitHash> {
        self.cuts.iter().map(|c| c.unit).collect()
    }

    /// The manifest naming this state: one anchor at the boundary.
    #[must_use]
    pub fn manifest(&self, keyless: bool) -> Manifest {
        Manifest {
            boundary: self.boundary,
            cuts: self.cuts.clone(),
            keyless,
            ckpts: vec![Ckpt {
                boundary: self.boundary,
                from: 0,
                blob: self.blob,
                tail: Vec::new(),
                // The only boundary these paths write: a whole-conversation state,
                // which the per-turn collapse must never drop.
                turn: true,
            }],
        }
    }
}

// THE INDEX, and why the restore path needs one.
//
// ```text
//   <prefix hex> <conv file stem>     this conversation has a boundary here
//   -<conv file stem>                 forget everything recorded for it
// ```
//
// WHY IT EXISTS. With a conversation id the manifest is O(1) by name and nothing
// here is needed to CONTINUE. It is needed for the two questions a name cannot
// answer: identifying a KEYLESS conversation, whose only handle is its content and
// whose prefix hashes change every turn; and BORROWING across conversations, where a
// prompt wants whoever already has the preamble it carries -- keyed or keyless.
//
// It is an inverse map, prefix -> conversation, and it is MAINTAINED rather than
// merely appended to. That is what makes a line trustworthy, and one trustworthy hit
// is all the agreement test needs:
//
//   prefix(p) equal  =>  tokens[0..p] identical  =>  every boundary below p agrees
//
// So the deepest position a prompt matches settles it: the agreed extents are the
// manifest's cuts ending at or below that position. Nothing per-extent is stored
// here, and nothing about a position is stored twice.
//
// A REWIND is why the `-name` record exists. It abandons the boundaries above its
// branch point, and their prefixes must stop naming that conversation -- a stale
// prefix would report agreement at a position whose rows the conversation no longer
// has. Appending `-name` followed by its current prefixes replaces them.
/// Whether to print. Read once: this is on the restore path.
fn log_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("IMPARO_LOG").is_ok_and(|v| v != "0" && !v.is_empty())
    })
}

const INDEX_FILE: &str = "index";

/// Bytes of the files directly in `dir`.
fn dir_bytes(dir: &Path) -> u64 {
    fs::read_dir(dir).map_or(0, |rd| {
        rd.flatten()
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    })
}

/// Bytes ONE config root occupies: extents, checkpoints, manifests and the index.
///
/// The index used to be left out of both places this was computed -- the live root and
/// every stale one -- so the cap was compared against a number smaller than what was on
/// disk. It is bounded by compaction rather than unbounded, but a disk cap that does not
/// count a file on disk is simply wrong.
fn root_bytes(dir: &Path) -> u64 {
    dir_bytes(&dir.join("units"))
        + dir_bytes(&dir.join("conv"))
        + dir_bytes(&dir.join("ckpt"))
        + fs::metadata(dir.join(INDEX_FILE)).map_or(0, |m| m.len())
}

/// Below this, the file is too small for its garbage to be worth a rewrite.
const INDEX_COMPACT_MIN: u64 = 1 << 20;

/// A position's identity to the conversations that have a boundary there, and the
/// reverse -- what each conversation currently claims.
///
/// The reverse half is what lets a commit say "these are my boundaries now" rather
/// than "here are some more": a rewind has to retract, and retracting needs to know
/// what was there.
///
/// Sets per position: a shared preamble puts every conversation on one prefix, and
/// de-duplicating that by scanning a growing vector is quadratic in the number of
/// conversations -- 71 ms at 3000 where the lookup itself is nothing.
#[derive(Default)]
struct Index {
    by_prefix: std::collections::BTreeMap<PrefixHash, BTreeSet<String>>,
    by_name: std::collections::BTreeMap<String, BTreeSet<PrefixHash>>,
}

impl Index {
    fn add(&mut self, prefix: PrefixHash, name: &str) {
        self.by_prefix
            .entry(prefix)
            .or_default()
            .insert(name.to_string());
        self.by_name
            .entry(name.to_string())
            .or_default()
            .insert(prefix);
    }

    fn forget(&mut self, name: &str) {
        for p in self.by_name.remove(name).unwrap_or_default() {
            if let Some(e) = self.by_prefix.get_mut(&p) {
                e.remove(name);
                if e.is_empty() {
                    self.by_prefix.remove(&p);
                }
            }
        }
    }
}

/// How many distinct conversations a single probe will parse.
const PROBE_FANOUT: usize = 32;

/// How many to take FROM ONE POSITION.
///
/// Every conversation the index names at a position agrees with this prompt exactly
/// there and nowhere deeper by construction, so one of them serves as well as
/// another -- taking a hundred means a hundred manifest parses for one answer. A
/// shared preamble is where this bites: it names every conversation in the store.
const PER_POSITION: usize = 4;

impl Store {
    fn index_path(&self) -> PathBuf {
        self.dir.join(INDEX_FILE)
    }

    /// Runs `f` against the index, refreshing only what was appended since last time.
    ///
    /// Re-reading the whole file per call cost 32 ms at 3000 conversations and grew
    /// with the store; the restore path pays this on every request that misses
    /// residency.
    fn with_index<R>(&self, f: impl FnOnce(&Index) -> R) -> R {
        let len = fs::metadata(self.index_path()).map_or(0, |m| m.len());
        {
            let g = self
                .index
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if g.len == len {
                return f(&g.map);
            }
        }
        let mut g = self
            .index
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if g.len != len {
            if len < g.len {
                // Rewritten under us -- a sweep, an erase, or another handle's
                // compaction. Start over, and take the new length as the baseline.
                *g = IndexCache {
                    compacted: len,
                    ..IndexCache::default()
                };
            }
            let from = g.len;
            let mut body = String::new();
            if let Ok(mut fh) = fs::File::open(self.index_path()) {
                use std::io::{Read as _, Seek as _, SeekFrom};
                if fh.seek(SeekFrom::Start(from)).is_ok()
                    && fh.read_to_string(&mut body).is_ok()
                {
                    for l in body.lines() {
                        if let Some(name) = l.strip_prefix('-') {
                            g.map.forget(name);
                            continue;
                        }
                        let mut it = l.split(' ');
                        let (Some(h), Some(name)) = (it.next(), it.next()) else {
                            continue;
                        };
                        let Some(prefix) = PrefixHash::from_hex(h) else {
                            continue;
                        };
                        g.map.add(prefix, name);
                    }
                    g.len = len;
                }
            }
        }
        f(&g.map)
    }

    /// Records what `name`'s boundaries ARE, not merely that it has some more.
    ///
    /// `at[i]` is the prefix at `m.cuts[i].end`. Growth appends only the new ones. A
    /// REWIND -- anything the index holds for this conversation that it no longer
    /// claims -- appends `-name` first, so the abandoned prefixes stop naming it.
    fn extend_index(&self, name: &str, at: &[PrefixHash]) {
        use std::io::Write as _;
        let want: BTreeSet<PrefixHash> = at.iter().copied().collect();
        let (retract, missing) = self.with_index(|idx| {
            let have = idx.by_name.get(name).cloned().unwrap_or_default();
            let retract = !have.is_subset(&want);
            let missing: Vec<PrefixHash> = if retract {
                at.to_vec()
            } else {
                at.iter().copied().filter(|p| !have.contains(p)).collect()
            };
            (retract, missing)
        });
        if !retract && missing.is_empty() {
            return;
        }
        let mut body = String::new();
        if retract {
            body.push_str(&format!("-{name}\n"));
        }
        for p in &missing {
            body.push_str(&format!("{} {name}\n", p.hex()));
        }
        // The append happens UNDER the write lock, and `g.len` is advanced by what
        // this call wrote rather than by asking the filesystem afterwards.
        //
        // Reading the length after the write recorded bytes this handle had not
        // parsed, whenever anything appended in between:
        //
        //   A write_all(a)      file = X + a
        //   B write_all(b)      file = X + a + b
        //   A g.len = metadata  g.len covers b, but g.map does not
        //   A with_index        lengths agree, so the re-read is skipped and b's
        //                       entries are invisible to this handle
        //
        // Costs reuse, never correctness -- a conversation whose prefix went missing
        // cold-starts once. The guard is to leave `g.len` ALONE when the file was not
        // what this handle had parsed: `with_index` then sees a mismatch and re-reads,
        // which is the self-correcting path that already exists.
        let mut g = self
            .index
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let pre = fs::metadata(self.index_path()).map_or(0, |m| m.len());
        let wrote = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.index_path())
            .and_then(|mut f| f.write_all(body.as_bytes()))
            .is_ok();
        if wrote {
            if retract {
                g.map.forget(name);
            }
            for p in &missing {
                g.map.add(*p, name);
            }
            if pre == g.len {
                g.len = pre + body.len() as u64;
            }
        }
        drop(g);
        self.compact_if_doubled();
    }

    /// Rewrites the index in canonical form: the map it currently describes, with no
    /// retractions left to replay and no conversation whose manifest is gone.
    ///
    /// A REPLAY, not a regeneration: the prefixes exist nowhere else, so the file
    /// cannot be rebuilt from the manifests. Losing it costs reuse -- every
    /// conversation cold-starts once -- and never correctness.
    fn compact_index(dir: &Path) {
        let path = dir.join(INDEX_FILE);
        let Ok(body) = fs::read_to_string(&path) else {
            return;
        };
        let mut map = Index::default();
        for l in body.lines() {
            if let Some(name) = l.strip_prefix('-') {
                map.forget(name);
                continue;
            }
            let mut it = l.split(' ');
            let (Some(h), Some(name)) = (it.next(), it.next()) else {
                continue;
            };
            let Some(prefix) = PrefixHash::from_hex(h) else {
                continue;
            };
            map.add(prefix, name);
        }
        let live: BTreeSet<String> = Self::manifests_in(dir)
            .into_iter()
            .filter_map(|(p, _)| Some(p.file_stem()?.to_str()?.to_string()))
            .collect();
        let mut out = String::with_capacity(body.len());
        for (name, prefixes) in &map.by_name {
            if !live.contains(name) {
                continue;
            }
            for prefix in prefixes {
                out.push_str(&format!("{} {name}\n", prefix.hex()));
            }
        }
        let _ = write_atomic(&path, out.as_bytes());
    }

    /// Rewrites the index once it has doubled since the last rewrite, so retractions
    /// and the lines they retracted stay under half of it.
    fn compact_if_doubled(&self) {
        let len = fs::metadata(self.index_path()).map_or(0, |m| m.len());
        {
            let g = self
                .index
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if len < INDEX_COMPACT_MIN || len < g.compacted.saturating_mul(2) {
                return;
            }
        }
        Self::compact_index(&self.dir);
        let mut g = self
            .index
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *g = IndexCache {
            compacted: fs::metadata(self.index_path()).map_or(0, |m| m.len()),
            ..IndexCache::default()
        };
    }
}

/// How many of a manifest's leading extents a prompt agrees with, given the DEEPEST
/// position the index says it agrees at.
///
/// One position settles it. `prefix(p)` is a hash of exactly the tokens below `p`, so
/// two streams agreeing there agree on every token below it, and therefore at every
/// boundary below it. The agreed extents are the ones ending at or below `p`.
#[must_use]
fn agreed_cuts(m: &Manifest, deepest: u64) -> usize {
    m.cuts.iter().take_while(|c| c.end <= deepest).count()
}

/// Content address for a state blob: the same 16-byte shortening `unit_hashes` uses, so
/// one spelling of "what this content is called" serves units and checkpoints alike.
#[must_use]
pub fn blob_hash(bytes: &[u8]) -> UnitHash {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let full: [u8; 32] = h.finalize().into();
    let mut short = [0_u8; 16];
    short.copy_from_slice(&full[..16]);
    UnitHash(short)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    let mut f =
        fs::File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    // Durability copies are written once and rarely re-read; keep them OUT of
    // the page cache so spill traffic never displaces weight pages -- the page
    // cache is the speed budget when a model pages from disk.
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        unsafe {
            libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1);
        }
    }
    f.write_all(bytes)
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    f.sync_all()
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Conversation ids come from clients; the on-disk name never trusts them.
fn conv_file_name(id: &str) -> String {
    let mut h = Sha256::new();
    h.update(id.as_bytes());
    let d: [u8; 32] = h.finalize().into();
    d[..12].iter().fold(String::new(), |mut s, b| {
        use std::fmt::Write as _;
        let _ = write!(s, "{b:02x}");
        s
    })
}

impl Store {
    /// Opens (creating if needed) the store for one configuration root.
    ///
    /// # Errors
    /// Returns an error when the directories cannot be created.
    pub fn open(base: &Path, root: &ConfigRoot) -> Result<Self, String> {
        let mut h = Sha256::new();
        h.update(root.bytes());
        let tag: [u8; 32] = h.finalize().into();
        let hex = tag[..12].iter().fold(String::new(), |mut s, b| {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
            s
        });
        let dir = base.join(&hex);
        for sub in ["units", "conv"] {
            fs::create_dir_all(dir.join(sub))
                .map_err(|e| format!("{}: {e}", dir.join(sub).display()))?;
        }
        let store = Self {
            dir,
            base: base.to_path_buf(),
            index: std::sync::Arc::default(),
        };
        // IN MEMORY FROM THE START. Reading it lazily put the whole parse on the first
        // request that missed residency -- measured 23.76 ms at 3000 conversations,
        // growing with the store -- where it belongs to opening the store.
        store.with_index(|_| ());
        Ok(store)
    }

    fn unit_path(&self, h: &UnitHash) -> PathBuf {
        self.dir.join("units").join(h.hex())
    }

    /// Writes a sealed unit if absent (immutable: an existing file is already
    /// byte-identical by the identity rule).
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn put_unit(&self, h: &UnitHash, bytes: &[u8]) -> Result<(), String> {
        let p = self.unit_path(h);
        if p.exists() {
            return Ok(());
        }
        write_atomic(&p, bytes)
    }

    #[must_use]
    pub fn get_unit(&self, h: &UnitHash) -> Option<Vec<u8>> {
        fs::read(self.unit_path(h)).ok()
    }

    #[must_use]
    pub fn has_unit(&self, h: &UnitHash) -> bool {
        self.unit_path(h).exists()
    }

    /// Commits a conversation's durable record. The manifest is the commit LINE: every
    /// blob it names -- units and checkpoints alike -- must already be on disk, so a
    /// crash before this leaves the previously committed turn intact and a crash after
    /// it leaves nothing dangling.
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    /// `at[i]` is the prefix at `m.cuts[i].end` -- what `Cut::for_stream` returned
    /// beside the cuts. The manifest keeps the cuts, the index keeps the prefixes.
    pub fn commit(
        &self,
        conv: &str,
        m: &Manifest,
        at: &[PrefixHash],
    ) -> Result<(), String> {
        let name = conv_file_name(conv);
        let mut body = String::new();
        body.push_str(MANIFEST_MAGIC);
        body.push('\n');
        body.push_str(&format!("format={MANIFEST_FORMAT}\n"));
        body.push_str(&format!("boundary={}\n", m.boundary));
        for c in &m.cuts {
            body.push_str(&format!("cut={}:{}\n", c.end, c.unit.hex()));
        }
        // One line per kept boundary: position, blob, and the sub-unit tokens that prove
        // the stretch above the last whole unit. A build that predates the key skips it
        // and reads `boundary`/`tail` -- which is why this needs no format bump.
        for c in &m.ckpts {
            let tail: Vec<String> = c.tail.iter().map(u32::to_string).collect();
            body.push_str(&format!(
                "ckpt={}:{}:{}:{}:{}\n",
                c.boundary,
                c.from,
                c.blob.hex(),
                tail.join(","),
                u8::from(c.turn)
            ));
        }
        if m.keyless {
            body.push_str("keyless=1\n");
        }
        let mp = self.dir.join("conv").join(format!("{name}.manifest"));
        write_atomic(&mp, body.as_bytes())?;
        // AFTER the manifest lands. An index line naming a manifest that does not
        // exist costs a wasted parse; a manifest no line names is invisible to every
        // later probe, which is the failure that matters.
        self.extend_index(&name, at);
        Ok(())
    }

    fn read_manifest_file(path: &Path) -> Option<Manifest> {
        let body = fs::read_to_string(path).ok()?;
        let mut lines = body.lines();
        if lines.next()? != MANIFEST_MAGIC {
            return None; // not ours, or a pre-format file: aged out, never guessed at
        }
        let mut format = 0_u32;
        let mut boundary = 0_u64;
        let mut cuts: Vec<Cut> = Vec::new();
        let mut ckpts: Vec<Ckpt> = Vec::new();
        let mut keyless = false;
        for l in lines {
            let Some((key, val)) = l.split_once('=') else {
                continue; // not a key line: skip, do not fail
            };
            match key {
                "format" => format = val.parse().ok()?,
                "boundary" => boundary = val.parse().ok()?,
                "cut" => {
                    let (end, unit) = val.split_once(':')?;
                    cuts.push(Cut {
                        end: end.parse().ok()?,
                        unit: UnitHash::from_hex(unit)?,
                    });
                }
                "keyless" => keyless = val == "1",
                "ckpt" => {
                    let mut it = val.splitn(5, ':');
                    let at: u64 = it.next()?.parse().ok()?;
                    let from: u64 = it.next()?.parse().ok()?;
                    let blob = UnitHash::from_hex(it.next()?)?;
                    let t = it.next().unwrap_or("");
                    let tail: Vec<u32> = t
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse().ok())
                        .collect::<Option<_>>()?;
                    let turn = it.next() == Some("1");
                    ckpts.push(Ckpt {
                        boundary: at,
                        from,
                        blob,
                        tail,
                        turn,
                    });
                }
                // A key a later build writes and this one has never heard of. Skipping
                // it is what makes adding one free.
                _ => {}
            }
        }
        if format == 0 || format > MANIFEST_FORMAT {
            return None;
        }
        ckpts.sort_by_key(|c| c.boundary);
        Some(Manifest {
            boundary,
            cuts,
            ckpts,
            keyless,
        })
    }

    /// Where this conversation's manifest lives. The file name is a hash of the id, so
    /// a path cannot be turned back into a label -- callers that hold a path compare it,
    /// they do not decode it.
    #[must_use]
    pub fn manifest_path(&self, conv: &str) -> PathBuf {
        self.dir
            .join("conv")
            .join(format!("{}.manifest", conv_file_name(conv)))
    }

    /// Delete manifests BY PATH, then sweep whatever they were the last reference to.
    ///
    /// `erase` deletes by conversation id, across every config root, because a client
    /// asked. This is the other case: the caller just read these files and knows a newer
    /// manifest covers them. Only the root they live in is swept, because that is the
    /// only place they exist.
    ///
    /// # Errors
    /// Returns an error on I/O failure while removing or sweeping.
    pub fn erase_paths(&self, paths: &[PathBuf]) -> Result<(usize, usize), String> {
        // Manifests actually REMOVED, not paths asked for. The caller logged
        // `paths.len()`, which counts the ones that were already gone as dropped.
        let mut removed = 0;
        for p in paths {
            let mut hit = false;
            for ext in ["manifest", "ckpt"] {
                let f = p.with_extension(ext);
                if f.exists() {
                    fs::remove_file(&f).map_err(|e| format!("{}: {e}", f.display()))?;
                    hit = true;
                }
            }
            removed += usize::from(hit);
        }
        let swept = if removed > 0 {
            Self::sweep_dir(&self.dir)?
        } else {
            0
        };
        Ok((removed, swept))
    }

    #[must_use]
    pub fn manifest(&self, conv: &str) -> Option<Manifest> {
        let name = conv_file_name(conv);
        Self::read_manifest_file(
            &self.dir.join("conv").join(format!("{name}.manifest")),
        )
    }

    /// Store one boundary's state, addressed by its content.
    ///
    /// Written once: a boundary that is committed again -- every turn, for as long as it
    /// stays in the kept set -- finds its blob already there and costs nothing. Two
    /// conversations that reach byte-identical state share one file.
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn put_checkpoint(&self, bytes: &[u8]) -> Result<UnitHash, String> {
        let h = blob_hash(bytes);
        let p = self.ckpt_path(&h);
        if p.exists() {
            return Ok(h);
        }
        if let Some(d) = p.parent() {
            fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        write_atomic(&p, bytes)?;
        Ok(h)
    }

    #[must_use]
    pub fn get_checkpoint(&self, h: &UnitHash) -> Option<Vec<u8>> {
        fs::read(self.ckpt_path(h)).ok()
    }

    #[must_use]
    pub fn has_checkpoint(&self, h: &UnitHash) -> bool {
        self.ckpt_path(h).exists()
    }

    fn ckpt_path(&self, h: &UnitHash) -> PathBuf {
        self.dir.join("ckpt").join(h.hex())
    }

    /// Stage a WHOLE captured state: one extent, one anchor link, ready to commit.
    ///
    /// The pool writes a state incrementally -- an extent per request, a link per turn.
    /// Two callers instead have a finished state and no request structure to cut on:
    /// `imparo-forward --spill`, and the server's commit when the pool is off. They had
    /// a copy of this each, ~35 lines apiece, and the comments admitted it ("for the
    /// same reason as imparo-forward"). That duplication is where the second blob
    /// format came from.
    ///
    /// ONE extent, because a whole state written in one go IS one request by the rule
    /// the disk layout follows; `resident_bounds` used to split it on the 256 grid that
    /// design replaced. Snapped to the identity grid like any other cut.
    ///
    /// The caller names the conversation and commits, because naming is where the two
    /// differ: the server may derive a label from the unit hashes.
    ///
    /// # Errors
    /// When an extent or the checkpoint blob cannot be written.
    pub fn stage_whole(
        &self,
        root: &ConfigRoot,
        tokens: &[u32],
        state: &crate::state::KvState,
    ) -> Result<StagedState, String> {
        let cut = state.boundary / crate::grid_tokens() * crate::grid_tokens();
        let bounds: Vec<usize> = if cut > 0 { vec![cut] } else { Vec::new() };
        let covered = tokens.get(..state.boundary).unwrap_or(tokens);
        let (cuts, at) = Cut::for_stream(root, covered, &bounds);
        for (c, b) in cuts
            .iter()
            .zip(&crate::state::unit_blobs_at(state, &bounds))
        {
            self.put_unit(&c.unit, b)?;
        }
        let unit_end = bounds.last().copied().unwrap_or(0);
        let blob = self.put_checkpoint(&crate::state::delta_blob(
            &crate::state::anchor_link(state, unit_end),
        ))?;
        Ok(StagedState {
            cuts,
            at,
            blob,
            boundary: state.boundary as u64,
        })
    }

    /// Read a whole state back: the extents a prompt matches, plus its link chain.
    ///
    /// The counterpart of `stage_whole`, and the only restore path that does not go
    /// through the pool's residency. Newest link first, as `ckpts` names them.
    ///
    /// # Errors
    /// When nothing matches, a blob is missing, or the chain does not cover a window.
    pub fn read_whole(
        &self,
        root: &ConfigRoot,
        geom: &[crate::state::LayerStateGeom],
        tokens: &[u32],
    ) -> Result<crate::state::KvState, String> {
        let grid = prefix_hashes(root, tokens);
        let (_, manifest) = self
            .best_match(&grid)
            .ok_or("restore: no committed prefix matches these tokens")?;
        let units: Vec<Vec<u8>> = manifest
            .cuts
            .iter()
            .map(|c| {
                self.get_unit(&c.unit)
                    .ok_or_else(|| "restore: extent missing".to_string())
            })
            .collect::<Result<_, _>>()?;
        let mut cks: Vec<&Ckpt> = manifest.ckpts.iter().collect();
        cks.sort_by_key(|c| std::cmp::Reverse(c.boundary));
        let links: Vec<Vec<u8>> = cks
            .iter()
            .map(|c| {
                self.get_checkpoint(&c.blob)
                    .ok_or_else(|| "restore: checkpoint missing".to_string())
            })
            .collect::<Result<_, _>>()?;
        crate::state::state_from_links(geom, &units, &links)
    }

    /// Bytes this config root occupies: extents, checkpoints, manifests AND the index.
    ///
    /// The index was left out, so the cap it is compared against was the cap plus
    /// whatever the index had grown to. It is bounded by compaction rather than
    /// unbounded, but a disk cap that does not count a file on disk is simply wrong.
    fn root_bytes(&self) -> u64 {
        root_bytes(&self.dir)
    }

    /// Every committed manifest (for content probes and the deletion sweep).
    #[must_use]
    pub fn manifests(&self) -> Vec<(PathBuf, Manifest)> {
        Self::manifests_in(&self.dir)
    }

    fn manifests_in(dir: &Path) -> Vec<(PathBuf, Manifest)> {
        let mut out = Vec::new();
        if let Ok(rd) = fs::read_dir(dir.join("conv")) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "manifest") {
                    if let Some(m) = Self::read_manifest_file(&p) {
                        out.push((p, m));
                    }
                }
            }
        }
        out
    }

    /// The deepest committed conversation whose manifest chain is a prefix of
    /// `hashes` -- the content probe: no id consulted, the request's own tokens
    /// are the key. Returns (manifest path, manifest).
    #[must_use]
    pub fn best_match(&self, grid: &[PrefixHash]) -> Option<(PathBuf, Manifest)> {
        self.probe(grid)
            .into_iter()
            .filter(|(_, m, k)| !m.cuts.is_empty() && *k == m.cuts.len())
            .max_by_key(|(_, m, _)| m.cuts.len())
            .map(|(p, m, _)| (p, m))
    }

    /// Every conversation that SHARES a boundary with this prompt, deepest agreement
    /// first, paired with how many of its leading extents agree.
    ///
    /// `best_match` answers a narrower question -- "whose whole chain is a prefix of
    /// mine" -- which only a conversation that never continued past the shared part
    /// can satisfy. A real donor answered its own question afterwards, so its chain
    /// diverges, and it can still serve every extent up to the divergence.
    #[must_use]
    pub fn prefix_matches(
        &self,
        grid: &[PrefixHash],
    ) -> Vec<(PathBuf, Manifest, usize)> {
        let mut out = self.probe(grid);
        out.sort_by_key(|m| std::cmp::Reverse(m.2));
        out
    }

    /// The conversations that have a boundary at a position this prompt agrees with,
    /// found by INDEX LOOKUP rather than by reading every manifest in the root, each
    /// with how many of its leading extents agree.
    ///
    /// Deepest first, because the deepest agreement is the one worth having and the
    /// shallow end of the grid is where a shared preamble names everybody. Bounded by
    /// `PER_POSITION` and `PROBE_FANOUT`, and it says what it dropped.
    fn probe(&self, grid: &[PrefixHash]) -> Vec<(PathBuf, Manifest, usize)> {
        let (deepest, dropped) = self.with_index(|idx| {
            // Name -> the deepest position this prompt matches it at. One position is
            // the whole answer: equal prefixes there mean identical tokens below, so
            // every boundary the conversation has below it agrees too.
            let mut deepest: std::collections::BTreeMap<String, u64> =
                std::collections::BTreeMap::new();
            let mut dropped = 0_usize;
            for (step, ph) in grid.iter().enumerate().rev() {
                let Some(here) = idx.by_prefix.get(ph) else {
                    continue;
                };
                let pos = ((step + 1) * crate::identity::grid_tokens()) as u64;
                let mut took = 0_usize;
                for n in here {
                    if deepest.contains_key(n) {
                        continue; // already matched deeper
                    }
                    if took >= PER_POSITION || deepest.len() >= PROBE_FANOUT {
                        dropped += 1;
                        continue;
                    }
                    deepest.insert(n.clone(), pos);
                    took += 1;
                }
            }
            (deepest, dropped)
        });
        if dropped > 0 && log_on() {
            eprintln!(
                "[imparo] kv index: {dropped} conversation(s) share a boundary with this \
                 prompt beyond the {PROBE_FANOUT} deepest; not considered"
            );
        }
        deepest
            .into_iter()
            .filter_map(|(n, pos)| {
                let p = self.dir.join("conv").join(format!("{n}.manifest"));
                let m = Self::read_manifest_file(&p)?;
                let k = agreed_cuts(&m, pos);
                (k > 0).then_some((p, m, k))
            })
            .collect()
    }

    /// Deletes a set of conversations in one pass, then removes every unit no
    /// surviving manifest references (deletion is user intent: prompt, not lazy).
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn erase(&self, convs: &[String]) -> Result<usize, String> {
        // EVERY config root, not just the loaded one. A conversation is a client-side
        // thing: the same id may have been answered by this model at f16 and at q4_0,
        // or by two different models, and each of those opens its own root under
        // `base`. The client deletes it once, whichever configuration happens to be
        // running -- so deleting only `self.dir` leaves the others holding a manifest
        // with no id left anywhere to name it, and an orphan manifest keeps its units
        // reachable against the sweep forever.
        let mut swept = 0;
        for dir in self.roots() {
            let mut touched = false;
            for c in convs {
                let name = conv_file_name(c);
                for ext in ["manifest", "ckpt"] {
                    let p = dir.join("conv").join(format!("{name}.{ext}"));
                    if p.exists() {
                        fs::remove_file(&p)
                            .map_err(|e| format!("{}: {e}", p.display()))?;
                        touched = true;
                    }
                }
            }
            // Units are per-root (their hash folds the root in), so each root sweeps
            // its own -- and only the ones that lost something need sweeping at all.
            if touched || dir == self.dir {
                swept += Self::sweep_dir(&dir)?;
            }
        }
        Ok(swept)
    }

    /// Every config root under the store base, the loaded one included.
    fn roots(&self) -> Vec<PathBuf> {
        let mut out = vec![self.dir.clone()];
        if let Ok(rd) = fs::read_dir(&self.base) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() && p != self.dir {
                    out.push(p);
                }
            }
        }
        out
    }

    /// The reachability sweep over ONE config root: a unit or checkpoint file survives
    /// exactly while some surviving manifest in that root names it.
    ///
    /// Checkpoints are swept the same way units are, and for the same reason: they are
    /// content addressed, so two conversations can name one file and neither owns it.
    fn sweep_dir(dir: &Path) -> Result<usize, String> {
        let manifests = Self::manifests_in(dir);
        let n = Self::sweep_against(dir, &manifests)?;
        // Compaction is not part of freeing blobs, and it rewrites a whole file. It
        // used to sit in here, which put it inside `gc_measured`'s eviction loop.
        Self::compact_index(dir);
        Ok(n)
    }

    /// The blob sweep against an ALREADY-PARSED manifest set.
    ///
    /// Split out because `gc_measured` evicts one conversation at a time and has to
    /// know what each removal freed. Re-reading every manifest for each of those was
    /// the whole cost of an eviction pass: 400 conversations took 3360.9 ms to shed
    /// 351 of them, nearly all of it re-parsing manifests that had not changed.
    fn sweep_against(
        dir: &Path,
        manifests: &[(PathBuf, Manifest)],
    ) -> Result<usize, String> {
        let live_units: BTreeSet<String> = manifests
            .iter()
            .flat_map(|(_, m)| m.cuts.iter().map(|c| c.unit.hex()))
            .collect();
        let live_ckpts: BTreeSet<String> = manifests
            .iter()
            .flat_map(|(_, m)| m.ckpts.iter().map(|c| c.blob.hex()))
            .collect();
        let mut removed = 0;
        for (sub, live) in [("units", &live_units), ("ckpt", &live_ckpts)] {
            if let Ok(rd) = fs::read_dir(dir.join(sub)) {
                for e in rd.flatten() {
                    let p = e.path();
                    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if !live.contains(name) {
                        fs::remove_file(&p)
                            .map_err(|e| format!("{}: {e}", p.display()))?;
                        removed += 1;
                    }
                }
            }
        }
        Ok(removed)
    }

    /// LRU aging under the disk cap: oldest-touched conversations go first (their
    /// manifests removed, then the sweep reclaims what nothing else holds).
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn gc(&self, cap_bytes: u64) -> Result<(), String> {
        self.gc_measured(cap_bytes).map(|_| ())
    }

    /// `gc`, returning the store's size in bytes after the pass, so a caller can tell
    /// how much headroom is left before another walk could find anything.
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn gc_measured(&self, cap_bytes: u64) -> Result<u64, String> {
        // The cap covers the WHOLE store, every config root included. A config
        // change (KV types, model, plan) opens a new root and strands the old
        // one -- unreadable by the current config, so stale roots are the first
        // thing reclaimed, whole and oldest-first.
        let mut stale: Vec<(std::time::SystemTime, PathBuf, u64)> = Vec::new();
        let mut total = 0u64;
        if let Ok(rd) = fs::read_dir(&self.base) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_dir() {
                    continue;
                }
                let sz = root_bytes(&p);
                total += sz;
                if p != self.dir {
                    if let Ok(t) = e.metadata().and_then(|m| m.modified()) {
                        stale.push((t, p, sz));
                    }
                }
            }
        }
        if total <= cap_bytes {
            return Ok(total);
        }
        stale.sort();
        for (_, p, sz) in stale {
            if total <= cap_bytes {
                break;
            }
            let _ = fs::remove_dir_all(&p);
            total = total.saturating_sub(sz);
        }
        if total <= cap_bytes {
            return Ok(total);
        }
        let mut by_age: Vec<(std::time::SystemTime, PathBuf)> = self
            .manifests()
            .into_iter()
            .filter_map(|(p, _)| {
                let t = fs::metadata(&p).ok()?.modified().ok()?;
                Some((t, p))
            })
            .collect();
        by_age.sort();
        // PARSED ONCE. Every removal changes which blobs are still named, so the sweep
        // has to run per eviction -- but it does not have to re-read the manifests that
        // survived, which is what `sweep_unreferenced` did. Dropping the removed one
        // from an in-memory list says the same thing.
        let mut live: Vec<(PathBuf, Manifest)> = self.manifests();
        for (_, p) in by_age {
            if total <= cap_bytes {
                break;
            }
            let _ = fs::remove_file(&p);
            live.retain(|(q, _)| q != &p);
            Self::sweep_against(&self.dir, &live)?;
            total = self.root_bytes();
        }
        // Once, at the end: the index names conversations, and a pass that removed
        // several has one file to rewrite, not one per removal.
        Self::compact_index(&self.dir);
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ConfigRoot, grid_tokens, prefix_hashes, resident_bounds};

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("imparo-kv-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn root() -> ConfigRoot {
        ConfigRoot::new(b"m", (1, 1), b"g")
    }

    /// A stream's cuts, the prefix AT each cut (the index's half), and the full grid
    /// (what a matcher is asked with).
    fn stream(
        root: &ConfigRoot,
        n: usize,
    ) -> (Vec<Cut>, Vec<PrefixHash>, Vec<PrefixHash>) {
        let toks: Vec<u32> = (0..n as u32).collect();
        let (cuts, at) = Cut::for_stream(root, &toks, &resident_bounds(toks.len()));
        (cuts, at, prefix_hashes(root, &toks))
    }

    fn setup(tag: &str) -> (Store, Vec<Cut>, Vec<PrefixHash>) {
        let root = root();
        let (cuts, at, _) = stream(&root, 3 * grid_tokens());
        (Store::open(&tmpdir(tag), &root).unwrap(), cuts, at)
    }

    /// A conversation answered under two configurations (two KV types, say) leaves a
    /// manifest in each root. Deleting it once must reach both.
    #[test]
    fn erase_reaches_every_config_root() {
        let dir = tmpdir("erase-roots");
        let root_a = ConfigRoot::new(b"m", (1, 1), b"g");
        let root_b = ConfigRoot::new(b"m", (2, 2), b"g");
        let a = Store::open(&dir, &root_a).unwrap();
        let b = Store::open(&dir, &root_b).unwrap();
        for (st, root) in [(&a, &root_a), (&b, &root_b)] {
            let (cuts, at, _) = stream(root, grid_tokens());
            st.put_unit(&cuts[0].unit, b"unit-bytes").unwrap();
            let blob = st.put_checkpoint(b"ckpt").unwrap();
            st.commit(
                "conv-x",
                &Manifest {
                    boundary: grid_tokens() as u64,
                    cuts: cuts.clone(),
                    ckpts: vec![Ckpt {
                        boundary: grid_tokens() as u64,
                        from: 0,
                        blob,
                        tail: Vec::new(),
                        turn: true,
                    }],
                    keyless: false,
                },
                &at,
            )
            .unwrap();
        }
        assert!(b.manifest("conv-x").is_some());
        // erased through the store that is LOADED; the other root must lose it too
        a.erase(&["conv-x".to_string()]).unwrap();
        assert!(a.manifest("conv-x").is_none());
        assert!(
            b.manifest("conv-x").is_none(),
            "the other config root kept a manifest nothing can name any more"
        );
    }

    #[test]
    fn commit_probe_restart() {
        let (store, cuts, at) = setup("cpr");
        for (i, c) in cuts.iter().enumerate() {
            store.put_unit(&c.unit, &[i as u8; 64]).unwrap();
        }
        let blob = store.put_checkpoint(b"ckpt-bytes").unwrap();
        store
            .commit(
                "conv-a",
                &Manifest {
                    boundary: (3 * grid_tokens()) as u64,
                    cuts: cuts.clone(),
                    ckpts: vec![Ckpt {
                        boundary: (3 * grid_tokens()) as u64,
                        from: 0,
                        blob,
                        tail: Vec::new(),
                        turn: true,
                    }],
                    keyless: false,
                },
                &at,
            )
            .unwrap();
        // the content probe finds it with no id: the request's own prefixes are the key
        let root = root();
        let (_, _, grid) = stream(&root, 3 * grid_tokens());
        let (_, m) = store.best_match(&grid).unwrap();
        assert_eq!(m.cuts, cuts);
        let m0 = store.manifest("conv-a").unwrap();
        assert_eq!(
            store.get_checkpoint(&m0.ckpts[0].blob).unwrap(),
            b"ckpt-bytes"
        );
        // a longer request still matches the committed prefix
        let (_, _, longer) = stream(&root, 5 * grid_tokens());
        let (_, m2) = store.best_match(&longer).unwrap();
        assert_eq!(m2.cuts.len(), 3);
        // a diverged request does not
        let mut div: Vec<u32> = (0..5 * grid_tokens() as u32).collect();
        div[0] ^= 1;
        assert!(store.best_match(&prefix_hashes(&root, &div)).is_none());
    }

    #[test]
    fn erase_sweeps_only_the_unshared() {
        let (store, cuts, at) = setup("erase");
        for c in &cuts {
            store.put_unit(&c.unit, b"x").unwrap();
        }
        let blob = store.put_checkpoint(b"c").unwrap();
        let m = |n: usize| Manifest {
            boundary: (n * grid_tokens()) as u64,
            cuts: cuts[..n].to_vec(),
            ckpts: vec![Ckpt {
                boundary: (n * grid_tokens()) as u64,
                from: 0,
                blob,
                tail: Vec::new(),
                turn: true,
            }],
            keyless: false,
        };
        store.commit("a", &m(3), &at).unwrap();
        store.commit("b", &m(2), &at).unwrap();
        // erase a: unit 2 (only a's) goes; units 0,1 and the shared checkpoint blob
        // survive through b
        let removed = store.erase(&["a".into()]).unwrap();
        assert_eq!(removed, 1);
        assert!(store.has_checkpoint(&blob), "b still names it");
        assert!(store.has_unit(&cuts[0].unit) && store.has_unit(&cuts[1].unit));
        assert!(!store.has_unit(&cuts[2].unit));
        // erase b: everything goes, the checkpoint blob with it
        store.erase(&["b".into()]).unwrap();
        assert!(!store.has_unit(&cuts[0].unit));
        assert!(!store.has_checkpoint(&blob), "nothing names it now");
    }

    /// The lookup the scan was doing: who else has a boundary at a position I agree
    /// with? Answered by key, not by reading everyone.
    #[test]
    fn the_index_finds_a_conversation_by_a_boundary_it_shares() {
        let root = root();
        let store = Store::open(&tmpdir("index-find"), &root).unwrap();
        let blob = store.put_checkpoint(b"s").unwrap();
        let mk = |n: u32| -> Vec<u32> {
            (0..grid_tokens() as u32)
                .chain(std::iter::repeat_n(n, grid_tokens()))
                .collect()
        };
        for n in 0..3_u32 {
            let (cuts, at) =
                Cut::for_stream(&root, &mk(n), &resident_bounds(2 * grid_tokens()));
            store
                .commit(
                    &format!("c{n}"),
                    &Manifest {
                        boundary: (2 * grid_tokens()) as u64,
                        cuts,
                        ckpts: vec![Ckpt {
                            boundary: (2 * grid_tokens()) as u64,
                            from: 0,
                            blob,
                            tail: Vec::new(),
                            turn: true,
                        }],
                        keyless: false,
                    },
                    &at,
                )
                .unwrap();
        }
        // c1's own prompt agrees with c1 on BOTH extents and with the others on the
        // shared preamble only.
        let hits = store.prefix_matches(&prefix_hashes(&root, &mk(1)));
        let best = hits.first().expect("the index named nobody");
        assert_eq!(best.2, 2, "the deepest agreement is the whole of c1");
        assert!(
            hits.iter().any(|(_, _, n)| *n == 1),
            "the preamble sharers too"
        );
    }

    /// A REWIND drops the boundaries above its branch point, and the index keeps the
    /// lines naming them. The old prompt must not resurrect the old state: the index
    /// is a shortlist, the manifest is the authority, and agreement is recomputed
    /// from what the manifest actually holds.
    #[test]
    fn a_rewound_conversation_does_not_answer_for_the_path_it_abandoned() {
        let root = root();
        let store = Store::open(&tmpdir("index-rewind"), &root).unwrap();
        let blob = store.put_checkpoint(b"s").unwrap();
        let bounds = resident_bounds(2 * grid_tokens());
        let shared: Vec<u32> = (0..grid_tokens() as u32).collect();
        let before: Vec<u32> = shared
            .iter()
            .copied()
            .chain(std::iter::repeat_n(7, grid_tokens()))
            .collect();
        let after: Vec<u32> = shared
            .iter()
            .copied()
            .chain(std::iter::repeat_n(9, grid_tokens()))
            .collect();
        let commit = |toks: &[u32]| {
            let (cuts, at) = Cut::for_stream(&root, toks, &bounds);
            store
                .commit(
                    "c",
                    &Manifest {
                        boundary: (2 * grid_tokens()) as u64,
                        cuts,
                        ckpts: vec![Ckpt {
                            boundary: (2 * grid_tokens()) as u64,
                            from: 0,
                            blob,
                            tail: Vec::new(),
                            turn: true,
                        }],
                        keyless: false,
                    },
                    &at,
                )
                .unwrap();
        };
        commit(&before);
        let old_grid = prefix_hashes(&root, &before);
        assert_eq!(store.prefix_matches(&old_grid).first().unwrap().2, 2);

        // The rewind: same conversation, the second extent replaced.
        commit(&after);

        // The index still names "c" under the abandoned prefix -- that is what
        // append-only means -- but the answer comes from the manifest.
        let hits = store.prefix_matches(&old_grid);
        assert_eq!(hits.len(), 1, "the stale line still shortlists it");
        assert_eq!(
            hits[0].2, 1,
            "agreement is the shared preamble only: the abandoned extent is gone"
        );
        // ... and the path it actually took answers in full.
        assert_eq!(
            store
                .prefix_matches(&prefix_hashes(&root, &after))
                .first()
                .unwrap()
                .2,
            2
        );
    }

    /// The bound on the append-only file: rewinds leave lines behind, so it is
    /// rewritten from the live manifests once it has doubled. Without it a
    /// conversation that rewinds often grows the index without limit.
    ///
    /// A LONG conversation on purpose. Two lines per commit never reaches the
    /// threshold, so a test built that way asserts nothing while appearing to pass --
    /// which is what the first version of this did.
    #[test]
    fn the_index_is_rewritten_once_its_garbage_could_be_half_of_it() {
        const EXTENTS: usize = 200;
        const COMMITS: u32 = 220;
        let root = root();
        let store = Store::open(&tmpdir("index-compact"), &root).unwrap();
        let blob = store.put_checkpoint(b"s").unwrap();
        let bounds = resident_bounds(EXTENTS * grid_tokens());
        // Every commit REPLACES the last extent, so all but one line of the previous
        // commit stays live and the divergent one dies -- a rewind, repeated.
        let mk = |n: u32| -> Vec<u32> {
            (0..((EXTENTS - 1) * grid_tokens()) as u32)
                .chain(std::iter::repeat_n(n, grid_tokens()))
                .collect()
        };
        for n in 0..COMMITS {
            let (cuts, at) = Cut::for_stream(&root, &mk(n), &bounds);
            store
                .commit(
                    "rewinder",
                    &Manifest {
                        boundary: (EXTENTS * grid_tokens()) as u64,
                        cuts,
                        ckpts: vec![Ckpt {
                            boundary: (EXTENTS * grid_tokens()) as u64,
                            from: 0,
                            blob,
                            tail: Vec::new(),
                            turn: true,
                        }],
                        keyless: false,
                    },
                    &at,
                )
                .unwrap();
        }
        let len = fs::metadata(store.index_path()).unwrap().len();
        let appended = u64::from(COMMITS) * EXTENTS as u64 * 75;
        assert!(
            appended > 2 * INDEX_COMPACT_MIN,
            "the test appended {appended} bytes and never reached the threshold: it \
             would assert nothing"
        );
        assert!(
            len < appended / 2,
            "the index is {len} bytes of {appended} appended: compaction never ran"
        );
        // ... and it still answers for the path the conversation actually took.
        let hits = store.prefix_matches(&prefix_hashes(&root, &mk(COMMITS - 1)));
        assert_eq!(hits.first().map(|h| h.2), Some(EXTENTS));
    }

    /// A conversation that is erased leaves its index lines behind. That must cost a
    /// wasted lookup and nothing more -- never a hit on a manifest that is gone.
    #[test]
    fn a_stale_index_line_names_nobody() {
        let root = root();
        let store = Store::open(&tmpdir("index-stale"), &root).unwrap();
        let toks: Vec<u32> = (0..2 * grid_tokens() as u32).collect();
        let (cuts, at) = Cut::for_stream(&root, &toks, &resident_bounds(toks.len()));
        let blob = store.put_checkpoint(b"s").unwrap();
        store
            .commit(
                "gone",
                &Manifest {
                    boundary: (2 * grid_tokens()) as u64,
                    cuts,
                    ckpts: vec![Ckpt {
                        boundary: (2 * grid_tokens()) as u64,
                        from: 0,
                        blob,
                        tail: Vec::new(),
                        turn: true,
                    }],
                    keyless: false,
                },
                &at,
            )
            .unwrap();
        let grid = prefix_hashes(&root, &toks);
        assert_eq!(store.prefix_matches(&grid).len(), 1);
        store.erase(&["gone".to_string()]).unwrap();
        assert!(
            store.prefix_matches(&grid).is_empty(),
            "an erased conversation still answered a probe"
        );
    }

    /// What the per-request manifest scan COSTS as a store fills up.
    ///
    /// WHAT AN EVICTION PASS COSTS, and specifically whether the per-eviction index
    /// compaction inside `sweep_dir` is worth hoisting out.
    ///
    /// `gc_measured` evicts oldest-first and calls the sweep after EACH removal, so
    /// with E evictions it re-parses every manifest, rewrites the whole INDEX and
    /// walks units/ and ckpt/ E times. IMPARO_SCAN_BENCH=1 to run it.
    #[test]
    fn what_an_eviction_pass_costs() {
        if !std::env::var("IMPARO_SCAN_BENCH").is_ok_and(|v| v == "1") {
            return;
        }
        let root = root();
        let base = tmpdir("gc-bench");
        let store = Store::open(&base, &root).unwrap();
        let blob = store.put_checkpoint(b"state").unwrap();
        let toks = |n: usize| -> Vec<u32> {
            (0..grid_tokens() as u32)
                .chain(std::iter::repeat_n(n as u32, grid_tokens()))
                .collect()
        };
        let n_conv = 400usize;
        for i in 0..n_conv {
            let t = toks(i);
            let (cuts, at) =
                Cut::for_stream(&root, &t, &[grid_tokens(), 2 * grid_tokens()]);
            let m = Manifest {
                boundary: (2 * grid_tokens()) as u64,
                cuts,
                keyless: false,
                ckpts: vec![Ckpt {
                    boundary: (2 * grid_tokens()) as u64,
                    from: 0,
                    blob,
                    tail: Vec::new(),
                    turn: true,
                }],
            };
            store.commit(&format!("c{i}"), &m, &at).unwrap();
        }
        let before = store.gc_measured(u64::MAX).unwrap();
        // A cap that forces most of them out in one pass.
        let t0 = std::time::Instant::now();
        let after = store.gc_measured(before / 8).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let left = store.manifests().len();
        println!(
            "gc: {n_conv} conversations, {before} B -> {after} B, {left} manifests left, \
{ms:.1} ms"
        );
    }

    /// An extent's id IS the hash of the span it claims, and an off-grid bound cannot
    /// produce one that isn't.
    ///
    /// Nothing re-derives the id on READ -- `Cut::for_stream` computes it once and the
    /// store trusts it -- so this is where the two are checked against each other. The
    /// doc on `Cut` used to point at a `Manifest::check_unit_ids` that was never
    /// written; it points here now.
    #[test]
    fn an_extents_id_is_the_hash_of_its_own_span() {
        let root = root();
        let g = grid_tokens();
        let toks: Vec<u32> = (0..4 * g as u32).collect();
        let (cuts, at) = Cut::for_stream(&root, &toks, &[g, 3 * g]);
        assert_eq!(cuts.len(), 2);

        // Recompute from the tokens alone: id[i] = unit_id(prefix before, prefix here).
        let grid = prefix_hashes(&root, &toks);
        let mut prev = PrefixHash::start(&root);
        for (c, a) in cuts.iter().zip(&at) {
            let here = grid[c.end as usize / g - 1];
            assert_eq!(*a, here, "the recorded prefix is the prefix AT the cut");
            assert_eq!(c.unit, unit_id(&prev, &here), "the id names its own span");
            prev = here;
        }

        // An OFF-GRID bound is dropped, not given the id of a different position.
        let (off, _) = Cut::for_stream(&root, &toks, &[g + 1]);
        assert!(off.is_empty(), "a bound off the grid must not become a cut");
        // And it does not shift the ones around it.
        let (mixed, _) = Cut::for_stream(&root, &toks, &[g, g + 1, 2 * g]);
        assert_eq!(
            mixed.iter().map(|c| c.end).collect::<Vec<_>>(),
            vec![g as u64, 2 * g as u64]
        );
        assert_eq!(mixed[1].unit, unit_id(&at[0], &grid[1]), "ids stay chained");
    }

    /// `prefix_matches` reads and parses every manifest in the config root on any
    /// request that misses residency, so the restore path is O(stored conversations).
    /// Whether that matters decides whether the index needs to change shape at all --
    /// so it is measured here rather than argued about. IMPARO_SCAN_BENCH=1 to run it;
    /// it writes thousands of files and is not part of the normal suite.
    #[test]
    fn the_manifest_scan_cost_as_the_store_fills() {
        if !std::env::var("IMPARO_SCAN_BENCH").is_ok_and(|v| v == "1") {
            return;
        }
        let root = root();
        // Kept: tmpdir() WIPES its directory, so re-opening the filled store below
        // must reuse this path rather than ask for it again.
        let base = tmpdir("scan-bench");
        let store = Store::open(&base, &root).unwrap();
        let blob = store.put_checkpoint(b"state").unwrap();
        // What an agent fleet looks like: every conversation carries the same
        // preamble and then goes its own way. The shallow boundary is shared by all
        // of them; the deep one belongs to exactly one.
        let toks = |n: usize| -> Vec<u32> {
            (0..grid_tokens() as u32)
                .chain(std::iter::repeat_n(n as u32, grid_tokens()))
                .collect()
        };
        let grid = prefix_hashes(&root, &toks(0));
        let mk = |n: usize| {
            let (cuts, at) =
                Cut::for_stream(&root, &toks(n), &resident_bounds(2 * grid_tokens()));
            (
                Manifest {
                    boundary: (2 * grid_tokens()) as u64,
                    cuts,
                    // 20 boundaries: the disk cap, so this is the widest a manifest
                    // gets.
                    ckpts: (0..20)
                        .map(|i| Ckpt {
                            boundary: (grid_tokens() + i) as u64,
                            from: 0,
                            blob,
                            tail: vec![n as u32; i],
                            turn: true,
                        })
                        .collect(),
                    keyless: true,
                },
                at,
            )
        };
        let mut made = 0usize;
        for step in [100usize, 400, 500, 1000, 3000] {
            while made < step {
                let (m, at) = mk(made);
                store.commit(&format!("conv-{made}"), &m, &at).unwrap();
                made += 1;
            }
            // COLD: the index gained `made` conversations' lines since the last
            // probe, so this pays for the whole batch's tail.
            let t = std::time::Instant::now();
            let hits = store.prefix_matches(&grid);
            let cold = t.elapsed().as_secs_f64() * 1e3;
            // WARM: what a request actually pays -- nothing appended since, so the
            // index cache stands and only the candidate manifests are parsed.
            let t = std::time::Instant::now();
            let again = store.prefix_matches(&grid);
            let warm = t.elapsed().as_secs_f64() * 1e3;
            // BOOT: a real open of the filled store plus the first probe. `open` reads
            // the index, so this is what a server pays at startup, not per request.
            let t = std::time::Instant::now();
            let fresh = Store::open(&base, &root).unwrap();
            let first = fresh.prefix_matches(&grid);
            let boot = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "manifests={made} boot={boot:.2} ms cold={cold:.2} ms \
                 warm={warm:.2} ms hits={}/{}/{}",
                first.len(),
                hits.len(),
                again.len()
            );
        }
    }

    /// Same bytes, different file counts: what does one extent COST in file ops?
    ///
    /// A 2000-token turn is ~8 units of 4 MiB on E4B f16, against one ~32 MiB file if the
    /// unit were the turn. Same content either way, so this is purely the per-file price
    /// -- create, write, fsync, rename on the way in; open and read on the way back.
    /// IMPARO_SCAN_BENCH=1 to run it.
    #[test]
    fn what_a_unit_costs_in_file_ops() {
        if !std::env::var("IMPARO_SCAN_BENCH").is_ok_and(|v| v == "1") {
            return;
        }
        let root = ConfigRoot::new(b"m", (1, 1), b"g");
        let store = Store::open(&tmpdir("fileops"), &root).unwrap();
        const MIB: usize = 1 << 20;
        // 4.0 MiB was the measured size of a 256-token extent at gemma-4-E4B f16;
        // kept as the bench's fixed payload so the numbers stay comparable.
        for (label, n, each) in
            [("8 x 4 MiB", 8usize, 4 * MIB), ("1 x 32 MiB", 1, 32 * MIB)]
        {
            let blob = vec![7u8; each];
            let hs: Vec<UnitHash> = (0..n)
                .map(|i| {
                    let mut h = [0u8; 16];
                    h[0] = label.len() as u8;
                    h[1] = i as u8;
                    UnitHash(h)
                })
                .collect();
            let t = std::time::Instant::now();
            for h in &hs {
                store.put_unit(h, &blob).unwrap();
            }
            let write = t.elapsed().as_secs_f64() * 1e3;
            let t = std::time::Instant::now();
            let got: usize = hs
                .iter()
                .filter_map(|h| store.get_unit(h))
                .map(|b| b.len())
                .sum();
            let read = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "{label}: write={write:.2} ms read={read:.2} ms bytes={}",
                got / MIB
            );
        }
    }

    /// A keyless manifest says so, and supersession deletes BY PATH -- which is all the
    /// caller has, since the file name is a one-way hash of the id.
    #[test]
    fn a_keyless_manifest_says_so_and_can_be_dropped_by_path() {
        let (store, cuts, at) = setup("keyless-flag");
        store.put_unit(&cuts[0].unit, b"unit").unwrap();
        let blob = store.put_checkpoint(b"state").unwrap();
        let m = |keyless: bool| Manifest {
            boundary: grid_tokens() as u64,
            cuts: cuts[..1].to_vec(),
            ckpts: vec![Ckpt {
                boundary: grid_tokens() as u64,
                from: 0,
                blob,
                tail: Vec::new(),
                turn: true,
            }],
            keyless,
        };
        store.commit("keyed", &m(false), &at).unwrap();
        store.commit("keyless-abc", &m(true), &at).unwrap();
        assert!(!store.manifest("keyed").unwrap().keyless);
        assert!(store.manifest("keyless-abc").unwrap().keyless);

        // Dropping the keyless one leaves the keyed one, and its unit alive with it.
        let p = store.manifest_path("keyless-abc");
        store.erase_paths(&[p]).unwrap();
        assert!(store.manifest("keyless-abc").is_none());
        assert!(store.manifest("keyed").is_some());
        assert!(store.get_unit(&cuts[0].unit).is_some());

        // With the last namer gone, the sweep takes the unit and the blob too.
        store.erase(&["keyed".to_string()]).unwrap();
        assert!(store.get_unit(&cuts[0].unit).is_none());
        assert!(!store.has_checkpoint(&blob));
    }

    /// Every boundary a conversation reached stays namable, because a rewind or a branch
    /// aims at a turn that is not the newest -- and the seed case is exactly that: the
    /// preamble's boundary must survive every later turn.
    #[test]
    fn a_manifest_keeps_every_boundary_it_reached() {
        let (store, cuts, prefixes) = setup("boundaries");
        for c in &cuts {
            store.put_unit(&c.unit, b"x").unwrap();
        }
        let mut kept = Vec::new();
        for at in [256_u64, 512, 800] {
            let blob = store
                .put_checkpoint(format!("state-at-{at}").as_bytes())
                .unwrap();
            kept.push(Ckpt {
                boundary: at,
                from: 0,
                blob,
                // 800 sits above the unit grid: its tail proves the stretch above 768.
                tail: if at % grid_tokens() as u64 == 0 {
                    Vec::new()
                } else {
                    (0..32).collect()
                },
                turn: true,
            });
            let units = (at as usize) / grid_tokens();
            store
                .commit(
                    "conv",
                    &Manifest {
                        boundary: at,
                        cuts: cuts[..units.min(cuts.len())].to_vec(),
                        ckpts: kept.clone(),
                        keyless: false,
                    },
                    &prefixes,
                )
                .unwrap();
        }
        let m = store.manifest("conv").unwrap();
        assert_eq!(
            m.ckpts.iter().map(|c| c.boundary).collect::<Vec<_>>(),
            vec![256, 512, 800],
            "the older boundaries survived the newer commits"
        );
        for c in &m.ckpts {
            assert_eq!(
                store.get_checkpoint(&c.blob).unwrap(),
                format!("state-at-{}", c.boundary).into_bytes()
            );
        }
        // and the sub-unit tail rides with the boundary it belongs to
        assert_eq!(m.ckpts[2].tail.len(), 32);
        assert!(m.ckpts[0].tail.is_empty());
    }

    #[test]
    fn gc_ages_oldest_first() {
        let (store, cuts, prefixes) = setup("gc");
        for c in &cuts {
            store.put_unit(&c.unit, &[0u8; 4096]).unwrap();
        }
        let blob = store.put_checkpoint(b"c").unwrap();
        let at = |b: u64| {
            vec![Ckpt {
                boundary: b,
                from: 0,
                blob,
                tail: Vec::new(),
                turn: true,
            }]
        };
        store
            .commit(
                "old",
                &Manifest {
                    boundary: 256,
                    cuts: cuts[..1].to_vec(),
                    ckpts: at(256),
                    keyless: false,
                },
                &prefixes[..1],
            )
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        store
            .commit(
                "new",
                &Manifest {
                    boundary: 512,
                    cuts: cuts[1..3].to_vec(),
                    ckpts: at(512),
                    keyless: false,
                },
                &prefixes[1..3],
            )
            .unwrap();
        store.gc(9000).unwrap(); // forces dropping one conversation
        assert!(store.manifest("new").is_some());
        assert!(store.manifest("old").is_none());
        assert!(!store.has_unit(&cuts[0].unit));
        assert!(store.has_unit(&cuts[1].unit));
    }
}
