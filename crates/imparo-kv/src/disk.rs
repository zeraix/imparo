//! The disk tier's WRITER: one thread, one queue, everything in order.
//!
//! Why a queue rather than writing where the decision is made. A turn's commit is
//! 21.6-22.1 MiB of checkpoint on gemma-4-E4B, measured at 20-38 ms of a ~730 ms
//! turn -- paid on the request thread, for durability nothing is waiting on. The
//! bytes are already in RAM by then (they were read off the device to build the
//! blob), so the only thing left is file I/O, and file I/O does not need the
//! engine lock or the client.
//!
//! Order is the other reason. Units, manifests, erases and GC sweeps are decided
//! by four different call sites; run them concurrently and a sweep can delete the
//! units a commit is about to name. One FIFO thread gives:
//!
//! ```text
//!   turn end   -> Unit(a) Unit(b) Commit(conv)        in that order, always
//!   erase      -> Erase(ids)   after every commit already queued
//!   shutdown   -> Barrier      returns when the queue is empty
//! ```
//!
//! Crash window: a commit writes its checkpoint file before its manifest, so a
//! reader that sees the new manifest sees the new checkpoint. The reverse pairing
//! (old manifest, new checkpoint) fails the boundary check in `state_from_links`
//! and reprocesses -- fail-closed, which is the only acceptable direction.

use std::collections::HashSet;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

use crate::identity::UnitHash;
use crate::{Manifest, PrefixHash, Store};

enum Job {
    Unit(UnitHash, Vec<u8>),
    UnitDurable(UnitHash, Vec<u8>, Sender<Result<(), String>>),
    /// One ordered transaction: checkpoints first, then the manifest plus its prefix
    /// index, and only after that may manifests covered by the new one be erased.
    Commit {
        conversation: String,
        manifest: Manifest,
        blobs: Vec<(UnitHash, Vec<u8>)>,
        at: Vec<PrefixHash>,
        superseded: Vec<std::path::PathBuf>,
        ack: Option<Sender<Result<(), String>>>,
    },
    Erase(Vec<String>, Sender<Result<usize, String>>),
    /// Manifests a newer commit has covered; see `DiskQueue::drop_manifests`.
    DropManifests(Vec<std::path::PathBuf>),
    Gc(u64),
    Barrier(Sender<Result<(), String>>),
}

pub struct DiskQueue {
    tx: Sender<Job>,
    /// Units handed to the queue but not yet on disk. Without it, the second turn
    /// re-reads and re-queues a unit the first turn already sent, because the file
    /// it would check for does not exist yet.
    queued: Arc<Mutex<HashSet<UnitHash>>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl DiskQueue {
    /// Starts the writer thread. The queue owns its own handle to the store; the
    /// request threads keep reading through theirs.
    #[must_use]
    pub fn new(store: Store) -> Self {
        let (tx, rx) = channel::<Job>();
        let queued: Arc<Mutex<HashSet<UnitHash>>> =
            Arc::new(Mutex::new(HashSet::new()));
        let q = Arc::clone(&queued);
        let handle = std::thread::spawn(move || {
            // Bytes handed to this thread since the last GC pass, and what that pass
            // measured the store at. Together they say whether a walk could possibly
            // find the store over its cap.
            let mut written: u64 = 0;
            let mut last_gc_total: u64 = 0;
            // A barrier reports failures from every job before it.  Merely reaching
            // the end of the FIFO is not a durability acknowledgement when the disk
            // was full or a manifest rename failed.
            let mut pending_error: Option<String> = None;
            while let Ok(job) = rx.recv() {
                match job {
                    Job::Unit(h, bytes) => {
                        written += bytes.len() as u64;
                        if let Err(e) = store.put_unit(&h, &bytes) {
                            eprintln!("[imparo] kv unit write: {e}");
                            pending_error.get_or_insert(e);
                        }
                        q.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(&h);
                    }
                    Job::UnitDurable(h, bytes, ack) => {
                        written += bytes.len() as u64;
                        let result = store.put_unit(&h, &bytes);
                        if let Err(e) = &result {
                            eprintln!("[imparo] kv durable unit write: {e}");
                            pending_error.get_or_insert_with(|| e.clone());
                        }
                        q.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(&h);
                        // The receiver closing does not change durability. The write
                        // result is still the authoritative ownership handoff.
                        let _ = ack.send(result);
                    }
                    Job::Commit {
                        conversation,
                        manifest,
                        blobs,
                        at,
                        mut superseded,
                        ack,
                    } => {
                        // Blobs first, manifest last: the manifest is the commit line and
                        // must never name a checkpoint that is not yet on disk.
                        let mut job_error: Option<String> = None;
                        for (h, bytes) in &blobs {
                            written += bytes.len() as u64;
                            match store.put_checkpoint(bytes) {
                                Ok(got) if got == *h => {}
                                Ok(got) => {
                                    eprintln!(
                                        "[imparo] kv commit: checkpoint addressed {} but stored as {}",
                                        h.hex(),
                                        got.hex()
                                    );
                                    job_error.get_or_insert_with(|| {
                                        format!(
                                            "checkpoint addressed {} but stored as {}",
                                            h.hex(),
                                            got.hex()
                                        )
                                    });
                                }
                                Err(e) => {
                                    eprintln!("[imparo] kv commit: {e}");
                                    job_error.get_or_insert(e);
                                }
                            }
                        }
                        let committed = if job_error.is_none() {
                            if let Err(e) = store.commit(&conversation, &manifest, &at)
                            {
                                eprintln!("[imparo] kv commit: {e}");
                                job_error.get_or_insert(e);
                                false
                            } else {
                                true
                            }
                        } else {
                            false
                        };
                        let mine = store.manifest_path(&conversation);
                        superseded.retain(|path| *path != mine);
                        // Commit and supersede are one transaction in this worker:
                        // an old manifest is never deleted unless its replacement is
                        // already durable.
                        if committed && !superseded.is_empty() {
                            match store.erase_paths(&superseded) {
                                Ok((removed, swept)) => {
                                    if crate::log_on() && removed != 0 {
                                        eprintln!(
                                            "[imparo] kv disk: dropped {removed} superseded \
manifest(s), swept {swept} unit(s)"
                                        );
                                    }
                                }
                                Err(e) => {
                                    eprintln!("[imparo] kv drop: {e}");
                                    job_error.get_or_insert(e);
                                }
                            }
                        }
                        // On disk now (or failed, and then the store is the truth):
                        // drop the in-flight marks so `holds_checkpoint` reads the
                        // filesystem from here on.
                        {
                            let mut qq = q
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner);
                            for (h, _) in &blobs {
                                qq.remove(h);
                            }
                        }
                        if let Some(error) = &job_error {
                            pending_error.get_or_insert_with(|| error.clone());
                        }
                        if let Some(ack) = ack {
                            let _ = ack.send(job_error.map_or(Ok(()), Err));
                        }
                    }
                    Job::Erase(ids, ack) => {
                        let result = match store.erase(&ids) {
                            Ok(n) => Ok(n),
                            Err(e) => {
                                eprintln!("[imparo] kv erase: {e}");
                                pending_error.get_or_insert_with(|| e.clone());
                                Err(e)
                            }
                        };
                        let _ = ack.send(result);
                    }
                    Job::Gc(cap) => {
                        // A GC pass stats every unit file and parses every manifest,
                        // twice. Bytes written since the last pass are a cheap upper
                        // bound on how much the store can have grown, so a store that
                        // cannot have crossed its cap is never walked at all.
                        if written < cap.saturating_sub(last_gc_total) {
                            continue;
                        }
                        match store.gc_measured(cap) {
                            Ok(total) => {
                                last_gc_total = total;
                                written = 0;
                            }
                            Err(e) => {
                                eprintln!("[imparo] kv gc: {e}");
                                pending_error.get_or_insert(e);
                            }
                        }
                    }
                    Job::DropManifests(paths) => match store.erase_paths(&paths) {
                        Ok((0, _)) => {}
                        Ok((removed, swept)) => {
                            if crate::log_on() {
                                // `removed`, not `paths.len()`: a path already gone is
                                // not a manifest this dropped.
                                eprintln!(
                                    "[imparo] kv disk: dropped {removed} superseded \
manifest(s), swept {swept} unit(s)"
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("[imparo] kv drop: {e}");
                            pending_error.get_or_insert(e);
                        }
                    },
                    Job::Barrier(ack) => {
                        // Sticky by design: a later empty barrier cannot repair a write
                        // that already failed.  Restarting with a new queue is the
                        // explicit recovery boundary.
                        let result = pending_error.clone().map_or(Ok(()), Err);
                        let _ = ack.send(result);
                    }
                }
            }
        });
        Self {
            tx,
            queued,
            handle: Some(handle),
        }
    }

    /// Whether this unit is already durable or on its way there.
    #[must_use]
    pub fn holds(&self, store: &Store, h: &UnitHash) -> bool {
        self.queued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(h)
            || store.has_unit(h)
    }

    /// The same question for a CHECKPOINT blob.
    ///
    /// Asking the filesystem alone says "no" for a blob this queue is still writing,
    /// which is the common case: turns arrive back to back and the writer is behind by
    /// design. That made a conversation re-write an anchor it had just committed --
    /// 20 MiB per turn, measured on the lifecycle gate after a restart.
    #[must_use]
    pub fn holds_checkpoint(&self, store: &Store, h: &UnitHash) -> bool {
        self.queued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .contains(h)
            || store.has_checkpoint(h)
    }

    #[cfg(test)]
    pub(crate) fn disconnected_for_test() -> Self {
        let (tx, rx) = channel();
        drop(rx);
        Self {
            tx,
            queued: Arc::new(Mutex::new(HashSet::new())),
            handle: None,
        }
    }

    /// Hand an immutable unit to the asynchronous writer.
    ///
    /// Success means the queue owns `bytes`, not that the file is durable. Call
    /// `ensure_unit_durable` before releasing the last Host copy under pressure.
    pub fn put_unit(&self, h: UnitHash, bytes: Vec<u8>) -> Result<(), String> {
        self.queued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(h);
        if self.tx.send(Job::Unit(h, bytes)).is_err() {
            self.queued
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&h);
            return Err("kv disk writer is not running".to_string());
        }
        Ok(())
    }

    /// Establish a durable disk copy before the pool releases its last Host copy.
    ///
    /// This is intentionally a separate slow-path operation. Ordinary turn commits
    /// remain asynchronous; Host pressure needs a positive acknowledgement because
    /// `queued` only means ownership was accepted, not that the write succeeded.
    pub fn ensure_unit_durable(
        &self,
        store: &Store,
        h: UnitHash,
        bytes: Vec<u8>,
    ) -> Result<(), String> {
        if store.has_unit(&h) {
            return Ok(());
        }
        let (ack, wait) = channel();
        self.queued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(h);
        if self.tx.send(Job::UnitDurable(h, bytes, ack)).is_err() {
            self.queued
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(&h);
            return Err("kv disk writer is not running".to_string());
        }
        wait.recv().map_err(|_| {
            "kv disk writer stopped before durability acknowledgement".to_string()
        })?
    }

    pub fn commit(
        &self,
        conv: &str,
        m: Manifest,
        blobs: Vec<(UnitHash, Vec<u8>)>,
        at: Vec<PrefixHash>,
        superseded: Vec<std::path::PathBuf>,
    ) -> Result<(), String> {
        let hashes: Vec<_> = blobs.iter().map(|(h, _)| *h).collect();
        // Superseding transfers ownership, so wait for that uncommon transaction.
        // Ordinary commits remain asynchronous and off the inference hot path.
        let (ack, wait) = if superseded.is_empty() {
            (None, None)
        } else {
            let (ack, wait) = channel();
            (Some(ack), Some(wait))
        };
        {
            let mut q = self
                .queued
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (h, _) in &blobs {
                q.insert(*h);
            }
        }
        if self
            .tx
            .send(Job::Commit {
                conversation: conv.to_string(),
                manifest: m,
                blobs,
                at,
                superseded,
                ack,
            })
            .is_err()
        {
            let mut q = self
                .queued
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for h in hashes {
                q.remove(&h);
            }
            return Err("kv disk writer is not running".to_string());
        }
        match wait {
            Some(wait) => wait.recv().map_err(|_| {
                "kv disk writer stopped before commit acknowledgement".to_string()
            })?,
            None => Ok(()),
        }
    }

    /// Delete manifests by path off the request thread.
    ///
    /// Retained for callers that have already durably committed the covering
    /// manifest separately. New commit paths should pass `superseded` to `commit`,
    /// which keeps replacement and deletion in one worker transaction.
    pub fn drop_manifests(&self, paths: Vec<std::path::PathBuf>) {
        let _ = self.tx.send(Job::DropManifests(paths));
    }

    /// Erase a conversation set and WAIT for the sweep, returning how many units it
    /// removed.
    ///
    /// Queued, so it lands behind whatever those conversations' last turns were
    /// still writing -- a sweep that overtook a commit would delete the units that
    /// commit is about to name. Blocking, because the caller's reply states the
    /// number.
    pub fn erase_blocking(&self, ids: Vec<String>) -> Result<usize, String> {
        let (ack, wait) = channel();
        if self.tx.send(Job::Erase(ids, ack)).is_err() {
            return Err("kv disk writer is not running".to_string());
        }
        wait.recv().map_err(|_| {
            "kv disk writer stopped before erase acknowledgement".to_string()
        })?
    }

    pub fn gc(&self, cap: u64) {
        let _ = self.tx.send(Job::Gc(cap));
    }

    /// Blocks until everything queued so far has been written.
    ///
    /// The erase endpoint needs it (its reply says how many units were swept, so
    /// the sweep has to have happened) and so does shutdown.
    pub fn drain(&self) -> Result<(), String> {
        let (ack, wait) = channel();
        self.tx
            .send(Job::Barrier(ack))
            .map_err(|_| "kv disk writer is not running".to_string())?;
        wait.recv()
            .map_err(|_| "kv disk writer stopped before barrier".to_string())?
    }
}

impl Drop for DiskQueue {
    fn drop(&mut self) {
        // Use the same explicit FIFO barrier on every platform. Unix signal
        // shutdown calls it earlier; normal Windows/server teardown reaches it
        // here. A dead writer is handled below by dropping the sender and join.
        let _ = self.drain();
        // Dropping the sender ends the loop; the join is what makes "the process
        // exited" mean "the bytes are on disk".
        let (tx, _) = channel();
        let dead = std::mem::replace(&mut self.tx, tx);
        drop(dead);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConfigRoot;
    use crate::store::Cut;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch(name: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "imparo-disk-queue-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn durable_handoff_waits_for_a_readable_unit() {
        let dir = scratch("durable");
        let root = ConfigRoot::new(b"model", (1, 1), b"geometry");
        let store = Store::open(&dir, &root).expect("open scratch store");
        let queue = DiskQueue::new(store.clone());
        let hash = UnitHash([7; 16]);
        let bytes = vec![3_u8; 4096];

        queue
            .ensure_unit_durable(&store, hash, bytes.clone())
            .expect("durability acknowledgement");
        assert_eq!(store.get_unit(&hash), Some(bytes));
        queue.drain().expect("writer barrier");
        drop(queue);
        drop(store);
        std::fs::remove_dir_all(&dir).expect("remove scratch store");
    }

    #[test]
    fn normal_drop_drains_an_asynchronous_unit_on_every_platform() {
        let dir = scratch("drop-drain");
        let root = ConfigRoot::new(b"model", (1, 1), b"geometry");
        let store = Store::open(&dir, &root).expect("open scratch store");
        let hash = UnitHash([8; 16]);
        let bytes = vec![4_u8; 4096];
        {
            let queue = DiskQueue::new(store.clone());
            queue
                .put_unit(hash, bytes.clone())
                .expect("enqueue asynchronous unit");
        }
        assert_eq!(store.get_unit(&hash), Some(bytes));
        drop(store);
        std::fs::remove_dir_all(&dir).expect("remove scratch store");
    }

    #[test]
    fn rejected_enqueue_rolls_back_the_in_flight_mark() {
        let (tx, rx) = channel();
        drop(rx);
        let queued = Arc::new(Mutex::new(HashSet::new()));
        let queue = DiskQueue {
            tx,
            queued: Arc::clone(&queued),
            handle: None,
        };
        let hash = UnitHash([9; 16]);

        assert!(queue.put_unit(hash, vec![1, 2, 3]).is_err());
        assert!(
            !queued
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .contains(&hash)
        );
    }

    #[test]
    fn barrier_failure_is_reported_instead_of_inferred() {
        let (tx, rx) = channel();
        drop(rx);
        let queue = DiskQueue {
            tx,
            queued: Arc::new(Mutex::new(HashSet::new())),
            handle: None,
        };
        assert!(queue.drain().is_err());
        assert!(queue.drain().is_err(), "a failure must remain sticky");
    }

    #[test]
    fn commit_durably_replaces_a_superseded_manifest_in_one_job() {
        let dir = scratch("atomic-supersede");
        let root = ConfigRoot::new(b"model", (1, 1), b"geometry");
        let store = Store::open(&dir, &root).expect("open scratch store");
        let manifest = Manifest {
            boundary: 0,
            cuts: Vec::new(),
            ckpts: Vec::new(),
            keyless: true,
        };
        store.commit("old", &manifest, &[]).expect("old manifest");
        let old = store.manifest_path("old");
        let new = store.manifest_path("new");
        let queue = DiskQueue::new(store.clone());
        queue
            .commit("new", manifest, Vec::new(), Vec::new(), vec![old.clone()])
            .expect("enqueue replacement");
        queue.drain().expect("replacement barrier");
        assert!(!old.exists());
        assert!(new.exists());
        drop(queue);
        drop(store);
        std::fs::remove_dir_all(&dir).expect("remove scratch store");
    }

    #[test]
    fn commit_forwards_prefix_index_before_the_barrier_acknowledges() {
        let dir = scratch("prefix-index");
        let root = ConfigRoot::new(b"model", (1, 1), b"geometry");
        let store = Store::open(&dir, &root).expect("open scratch store");
        let tokens = vec![7_u32; crate::grid_tokens()];
        let (cuts, at) = Cut::for_stream(&root, &tokens, &[tokens.len()]);
        let manifest = Manifest {
            boundary: tokens.len() as u64,
            cuts,
            ckpts: Vec::new(),
            keyless: false,
        };
        let queue = DiskQueue::new(store.clone());

        queue
            .commit(
                "indexed",
                manifest.clone(),
                Vec::new(),
                at.clone(),
                Vec::new(),
            )
            .expect("enqueue indexed commit");
        queue.drain().expect("indexed commit barrier");

        assert_eq!(store.manifest("indexed"), Some(manifest));
        assert!(
            store
                .prefix_matches(&at)
                .iter()
                .any(|(_, _, matched)| *matched == 1)
        );
        drop(queue);
        drop(store);
        std::fs::remove_dir_all(&dir).expect("remove scratch store");
    }
}
