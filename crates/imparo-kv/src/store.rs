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
//! <base>/<root-hex>/units/<hash-hex>      sealed unit blobs
//! <base>/<root-hex>/conv/<name>.manifest  hash chain + boundary
//! <base>/<root-hex>/conv/<name>.ckpt      windowed-state checkpoint blob
//! ```

use std::collections::BTreeSet;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::identity::{ConfigRoot, UnitHash};

// THE MANIFEST, and why the magic no longer carries a number.
//
// ```text
//   imparo-kv-manifest      the magic, stable forever
//   format=1                bumped only when an existing key CHANGES MEANING
//   boundary=1234           where the committed state describes
//   tail=7,9,11             tokens between the last unit and the boundary
//   unit=<32 hex>           one per content unit, in order
// ```
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
const MANIFEST_FORMAT: u32 = 1;

/// Clone is cheap and deliberate: the disk WRITER thread owns one handle while
/// request threads read through another. A `Store` is two paths, no state.
#[derive(Clone)]
pub struct Store {
    dir: PathBuf,
    base: PathBuf,
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
    /// Tokens between `hashes.len() * 256` and `boundary`, proving the sub-unit stretch.
    pub tail: Vec<u32>,
}

/// A conversation's durable record: the unit-hash chain up to `boundary`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Manifest {
    /// Where the committed state describes -- the conversation's tip, which is
    /// NOT on the unit grid.
    pub boundary: u64,
    /// The content units below `boundary`, in order. `hashes.len() * 256 <=
    /// boundary < (hashes.len() + 1) * 256`.
    pub hashes: Vec<UnitHash>,
    /// The tokens between the last unit and `boundary`.
    ///
    /// The unit hashes prove a shared prefix only to the unit's end. Everything
    /// above it is proved by these: a conversation restoring from this manifest
    /// must carry the same tokens there, or it would adopt rows describing
    /// somebody else's text.
    pub tail: Vec<u32>,
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
        Ok(Self {
            dir,
            base: base.to_path_buf(),
        })
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
    pub fn commit(&self, conv: &str, m: &Manifest) -> Result<(), String> {
        let name = conv_file_name(conv);
        let mut body = String::new();
        body.push_str(MANIFEST_MAGIC);
        body.push('\n');
        body.push_str(&format!("format={MANIFEST_FORMAT}\n"));
        body.push_str(&format!("boundary={}\n", m.boundary));
        let tail: Vec<String> = m.tail.iter().map(u32::to_string).collect();
        body.push_str(&format!("tail={}\n", tail.join(",")));
        for h in &m.hashes {
            body.push_str(&format!("unit={}\n", h.hex()));
        }
        // One line per kept boundary: position, blob, and the sub-unit tokens that prove
        // the stretch above the last whole unit. A build that predates the key skips it
        // and reads `boundary`/`tail` -- which is why this needs no format bump.
        for c in &m.ckpts {
            let tail: Vec<String> = c.tail.iter().map(u32::to_string).collect();
            body.push_str(&format!(
                "ckpt={}:{}:{}:{}\n",
                c.boundary,
                c.from,
                c.blob.hex(),
                tail.join(",")
            ));
        }
        if m.keyless {
            body.push_str("keyless=1\n");
        }
        let mp = self.dir.join("conv").join(format!("{name}.manifest"));
        write_atomic(&mp, body.as_bytes())
    }

    fn read_manifest_file(path: &Path) -> Option<Manifest> {
        let body = fs::read_to_string(path).ok()?;
        let mut lines = body.lines();
        if lines.next()? != MANIFEST_MAGIC {
            return None; // not ours, or a pre-format file: aged out, never guessed at
        }
        let mut format = 0_u32;
        let mut boundary = 0_u64;
        let mut tail: Vec<u32> = Vec::new();
        let mut hashes = Vec::new();
        let mut ckpts: Vec<Ckpt> = Vec::new();
        let mut keyless = false;
        for l in lines {
            let Some((key, val)) = l.split_once('=') else {
                continue; // not a key line: skip, do not fail
            };
            match key {
                "format" => format = val.parse().ok()?,
                "boundary" => boundary = val.parse().ok()?,
                "tail" => {
                    tail = val
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse().ok())
                        .collect::<Option<_>>()?;
                }
                "unit" => hashes.push(UnitHash::from_hex(val)?),
                "keyless" => keyless = val == "1",
                "ckpt" => {
                    let mut it = val.splitn(4, ':');
                    let at: u64 = it.next()?.parse().ok()?;
                    let from: u64 = it.next()?.parse().ok()?;
                    let blob = UnitHash::from_hex(it.next()?)?;
                    let t = it.next().unwrap_or("");
                    let tail: Vec<u32> = t
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| s.parse().ok())
                        .collect::<Option<_>>()?;
                    ckpts.push(Ckpt { boundary: at, from, blob, tail });
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
            hashes,
            tail,
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
    pub fn erase_paths(&self, paths: &[PathBuf]) -> Result<usize, String> {
        let mut touched = false;
        for p in paths {
            for ext in ["manifest", "ckpt"] {
                let f = p.with_extension(ext);
                if f.exists() {
                    fs::remove_file(&f).map_err(|e| format!("{}: {e}", f.display()))?;
                    touched = true;
                }
            }
        }
        if touched { Self::sweep_dir(&self.dir) } else { Ok(0) }
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

    /// The checkpoint blob committed beside a manifest found by content probe.
    #[must_use]
    pub fn checkpoint_for(&self, manifest_path: &Path) -> Option<Vec<u8>> {
        fs::read(manifest_path.with_extension("ckpt")).ok()
    }

    #[must_use]
    pub fn checkpoint(&self, conv: &str) -> Option<Vec<u8>> {
        let name = conv_file_name(conv);
        fs::read(self.dir.join("conv").join(format!("{name}.ckpt"))).ok()
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
    pub fn best_match(&self, hashes: &[UnitHash]) -> Option<(PathBuf, Manifest)> {
        self.manifests()
            .into_iter()
            .filter(|(_, m)| {
                let n = m.hashes.len();
                n > 0 && n <= hashes.len() && m.hashes[..] == hashes[..n]
            })
            .max_by_key(|(_, m)| m.hashes.len())
    }

    /// Every manifest that SHARES a unit prefix with `hashes`, longest first, paired
    /// with how many leading units match.
    ///
    /// `best_match` answers a narrower question -- "whose whole chain is a prefix of
    /// mine" -- which only a conversation that never continued past the shared part
    /// can satisfy. A real donor answered its own question afterwards, so its chain
    /// diverges, and it can still serve every unit up to the divergence. That is the
    /// difference between a seed being borrowable and every conversation being
    /// borrowable.
    #[must_use]
    pub fn prefix_matches(&self, hashes: &[UnitHash]) -> Vec<(PathBuf, Manifest, usize)> {
        let mut out: Vec<(PathBuf, Manifest, usize)> = self
            .manifests()
            .into_iter()
            .filter_map(|(p, m)| {
                let n = m
                    .hashes
                    .iter()
                    .zip(hashes)
                    .take_while(|(a, b)| a == b)
                    .count();
                (n > 0).then_some((p, m, n))
            })
            .collect();
        out.sort_by_key(|m| std::cmp::Reverse(m.2));
        out
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

    /// Removes units referenced by no manifest. The manifests are the truth;
    /// nothing persistent counts references.
    ///
    /// # Errors
    /// Returns an error on I/O failure.
    pub fn sweep_unreferenced(&self) -> Result<usize, String> {
        Self::sweep_dir(&self.dir)
    }

    /// The reachability sweep over ONE config root: a unit or checkpoint file survives
    /// exactly while some surviving manifest in that root names it.
    ///
    /// Checkpoints are swept the same way units are, and for the same reason: they are
    /// content addressed, so two conversations can name one file and neither owns it.
    fn sweep_dir(dir: &Path) -> Result<usize, String> {
        let manifests = Self::manifests_in(dir);
        let live_units: BTreeSet<String> = manifests
            .iter()
            .flat_map(|(_, m)| m.hashes.iter().map(UnitHash::hex))
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
        let size = |dir: &Path| -> u64 {
            fs::read_dir(dir).map_or(0, |rd| {
                rd.flatten()
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
        };
        // The cap covers the WHOLE store, every config root included. A config
        // change (KV types, model, plan) opens a new root and strands the old
        // one -- unreadable by the current config, so stale roots are the first
        // thing reclaimed, whole and oldest-first.
        let root_size = |dir: &Path| {
            size(&dir.join("units")) + size(&dir.join("conv")) + size(&dir.join("ckpt"))
        };
        let mut stale: Vec<(std::time::SystemTime, PathBuf, u64)> = Vec::new();
        let mut total = 0u64;
        if let Ok(rd) = fs::read_dir(&self.base) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_dir() {
                    continue;
                }
                let sz = root_size(&p);
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
        for (_, p) in by_age {
            if total <= cap_bytes {
                break;
            }
            let ck = p.with_extension("ckpt");
            let _ = fs::remove_file(&p);
            let _ = fs::remove_file(&ck);
            self.sweep_unreferenced()?;
            total = size(&self.dir.join("units"))
                + size(&self.dir.join("conv"))
                + size(&self.dir.join("ckpt"));
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{ConfigRoot, UNIT_TOKENS, unit_hashes};

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join(format!("imparo-kv-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn setup(tag: &str) -> (Store, Vec<UnitHash>) {
        let root = ConfigRoot::new(b"m", (1, 1), b"g");
        let toks: Vec<u32> = (0..3 * UNIT_TOKENS as u32).collect();
        let hashes = unit_hashes(&root, &toks);
        (Store::open(&tmpdir(tag), &root).unwrap(), hashes)
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
            let toks: Vec<u32> = (0..UNIT_TOKENS as u32).collect();
            let hashes = unit_hashes(root, &toks);
            st.put_unit(&hashes[0], b"unit-bytes").unwrap();
            let blob = st.put_checkpoint(b"ckpt").unwrap();
            st.commit(
                "conv-x",
                &Manifest {
                    boundary: UNIT_TOKENS as u64,
                    hashes: hashes.clone(),
                    tail: Vec::new(),
                    ckpts: vec![Ckpt {
                        boundary: UNIT_TOKENS as u64,
                        from: 0,
                        blob,
                        tail: Vec::new(),
                    }],
                    keyless: false,
                },
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
        let (store, hashes) = setup("cpr");
        for (i, h) in hashes.iter().enumerate() {
            store.put_unit(h, &[i as u8; 64]).unwrap();
        }
        let blob = store.put_checkpoint(b"ckpt-bytes").unwrap();
        store
            .commit(
                "conv-a",
                &Manifest {
                    boundary: (3 * UNIT_TOKENS) as u64,
                    hashes: hashes.clone(),
                    tail: Vec::new(),
                    ckpts: vec![Ckpt {
                        boundary: (3 * UNIT_TOKENS) as u64,
                        from: 0,
                        blob,
                        tail: Vec::new(),
                    }],
                    keyless: false,
                },
            )
            .unwrap();
        // the content probe finds it with no id: the request's hashes are the key
        let (_, m) = store.best_match(&hashes).unwrap();
        assert_eq!(m.hashes, hashes);
        let m0 = store.manifest("conv-a").unwrap();
        assert_eq!(store.get_checkpoint(&m0.ckpts[0].blob).unwrap(), b"ckpt-bytes");
        // a longer request still matches the committed prefix
        let root = ConfigRoot::new(b"m", (1, 1), b"g");
        let longer: Vec<u32> = (0..5 * UNIT_TOKENS as u32).collect();
        let (_, m2) = store.best_match(&unit_hashes(&root, &longer)).unwrap();
        assert_eq!(m2.hashes.len(), 3);
        // a diverged request does not
        let mut div = longer;
        div[0] ^= 1;
        assert!(store.best_match(&unit_hashes(&root, &div)).is_none());
    }

    #[test]
    fn erase_sweeps_only_the_unshared() {
        let (store, hashes) = setup("erase");
        for h in &hashes {
            store.put_unit(h, b"x").unwrap();
        }
        let blob = store.put_checkpoint(b"c").unwrap();
        let m = |n: usize| Manifest {
            boundary: (n * UNIT_TOKENS) as u64,
            hashes: hashes[..n].to_vec(),
            tail: Vec::new(),
            ckpts: vec![Ckpt {
                boundary: (n * UNIT_TOKENS) as u64,
                from: 0,
                blob,
                tail: Vec::new(),
            }],
            keyless: false,
        };
        store.commit("a", &m(3)).unwrap();
        store.commit("b", &m(2)).unwrap();
        // erase a: unit 2 (only a's) goes; units 0,1 and the shared checkpoint blob
        // survive through b
        let removed = store.erase(&["a".into()]).unwrap();
        assert_eq!(removed, 1);
        assert!(store.has_checkpoint(&blob), "b still names it");
        assert!(store.has_unit(&hashes[0]) && store.has_unit(&hashes[1]));
        assert!(!store.has_unit(&hashes[2]));
        // erase b: everything goes, the checkpoint blob with it
        store.erase(&["b".into()]).unwrap();
        assert!(!store.has_unit(&hashes[0]));
        assert!(!store.has_checkpoint(&blob), "nothing names it now");
    }

    /// What the per-request manifest scan COSTS as a store fills up.
    ///
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
        let (store, hashes) = setup("scan-bench");
        let blob = store.put_checkpoint(b"state").unwrap();
        let mk = |n: usize| Manifest {
            boundary: UNIT_TOKENS as u64,
            hashes: hashes[..1].to_vec(),
            tail: Vec::new(),
            // 20 boundaries: the disk cap, so this is the widest a manifest gets.
            ckpts: (0..20)
                .map(|i| Ckpt {
                    boundary: (UNIT_TOKENS + i) as u64,
                    from: 0,
                    blob,
                    tail: vec![n as u32; i],
                })
                .collect(),
            keyless: true,
        };
        let mut made = 0usize;
        for step in [100usize, 400, 500, 1000, 3000] {
            while made < step {
                store.commit(&format!("conv-{made}"), &mk(made)).unwrap();
                made += 1;
            }
            let t = std::time::Instant::now();
            let hits = store.prefix_matches(&hashes);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!("manifests={made} prefix_matches={ms:.2} ms hits={}", hits.len());
        }
    }

    /// Same bytes, different file counts: what does the 256-token unit COST in file ops?
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
        // 4.0 MiB is the measured size of one 256-token unit at gemma-4-E4B f16.
        for (label, n, each) in [("8 x 4 MiB", 8usize, 4 * MIB), ("1 x 32 MiB", 1, 32 * MIB)] {
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
            let got: usize = hs.iter().filter_map(|h| store.get_unit(h)).map(|b| b.len()).sum();
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
        let (store, hashes) = setup("keyless-flag");
        store.put_unit(&hashes[0], b"unit").unwrap();
        let blob = store.put_checkpoint(b"state").unwrap();
        let m = |keyless: bool| Manifest {
            boundary: UNIT_TOKENS as u64,
            hashes: hashes[..1].to_vec(),
            tail: Vec::new(),
            ckpts: vec![Ckpt {
                boundary: UNIT_TOKENS as u64,
                from: 0,
                blob,
                tail: Vec::new(),
            }],
            keyless,
        };
        store.commit("keyed", &m(false)).unwrap();
        store.commit("keyless-abc", &m(true)).unwrap();
        assert!(!store.manifest("keyed").unwrap().keyless);
        assert!(store.manifest("keyless-abc").unwrap().keyless);

        // Dropping the keyless one leaves the keyed one, and its unit alive with it.
        let p = store.manifest_path("keyless-abc");
        store.erase_paths(&[p]).unwrap();
        assert!(store.manifest("keyless-abc").is_none());
        assert!(store.manifest("keyed").is_some());
        assert!(store.get_unit(&hashes[0]).is_some());

        // With the last namer gone, the sweep takes the unit and the blob too.
        store.erase(&["keyed".to_string()]).unwrap();
        assert!(store.get_unit(&hashes[0]).is_none());
        assert!(!store.has_checkpoint(&blob));
    }

    /// Every boundary a conversation reached stays namable, because a rewind or a branch
    /// aims at a turn that is not the newest -- and the seed case is exactly that: the
    /// preamble's boundary must survive every later turn.
    #[test]
    fn a_manifest_keeps_every_boundary_it_reached() {
        let (store, hashes) = setup("boundaries");
        for h in &hashes {
            store.put_unit(h, b"x").unwrap();
        }
        let mut kept = Vec::new();
        for (i, at) in [256_u64, 512, 800].into_iter().enumerate() {
            let blob = store.put_checkpoint(format!("state-at-{at}").as_bytes()).unwrap();
            kept.push(Ckpt {
                boundary: at,
                from: 0,
                blob,
                // 800 sits above the unit grid: its tail proves the stretch above 768.
                tail: if at % UNIT_TOKENS as u64 == 0 {
                    Vec::new()
                } else {
                    (0..32).collect()
                },
            });
            let units = (at as usize) / UNIT_TOKENS;
            store
                .commit(
                    "conv",
                    &Manifest {
                        boundary: at,
                        hashes: hashes[..units.min(hashes.len())].to_vec(),
                        tail: kept[i].tail.clone(),
                        ckpts: kept.clone(),
                        keyless: false,
                    },
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
        let (store, hashes) = setup("gc");
        for h in &hashes {
            store.put_unit(h, &[0u8; 4096]).unwrap();
        }
        let blob = store.put_checkpoint(b"c").unwrap();
        let at = |b: u64| {
            vec![Ckpt {
                boundary: b,
                from: 0,
                blob,
                tail: Vec::new(),
            }]
        };
        store
            .commit(
                "old",
                &Manifest {
                    boundary: 256,
                    hashes: hashes[..1].to_vec(),
                    tail: Vec::new(),
                    ckpts: at(256),
                    keyless: false,
                },
            )
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1100));
        store
            .commit(
                "new",
                &Manifest {
                    boundary: 512,
                    hashes: hashes[1..3].to_vec(),
                    tail: Vec::new(),
                    ckpts: at(512),
                    keyless: false,
                },
            )
            .unwrap();
        store.gc(9000).unwrap(); // forces dropping one conversation
        assert!(store.manifest("new").is_some());
        assert!(store.manifest("old").is_none());
        assert!(!store.has_unit(&hashes[0]));
        assert!(store.has_unit(&hashes[1]));
    }
}
