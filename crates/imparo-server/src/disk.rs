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
//! (old manifest, new checkpoint) fails the boundary check in `state_from_blobs`
//! and reprocesses -- fail-closed, which is the only acceptable direction.

use std::collections::HashSet;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

use imparo_kv::identity::UnitHash;
use imparo_kv::{Manifest, Store};

enum Job {
    Unit(UnitHash, Vec<u8>),
    Commit(String, Manifest, Vec<(imparo_kv::UnitHash, Vec<u8>)>),
    Erase(Vec<String>, Sender<usize>),
    /// Manifests a newer commit has covered; see `DiskQueue::drop_manifests`.
    DropManifests(Vec<std::path::PathBuf>),
    Gc(u64),
    Barrier(Sender<()>),
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
        let queued: Arc<Mutex<HashSet<UnitHash>>> = Arc::new(Mutex::new(HashSet::new()));
        let q = Arc::clone(&queued);
        let handle = std::thread::spawn(move || {
            // Bytes handed to this thread since the last GC pass, and what that pass
            // measured the store at. Together they say whether a walk could possibly
            // find the store over its cap.
            let mut written: u64 = 0;
            let mut last_gc_total: u64 = 0;
            while let Ok(job) = rx.recv() {
                match job {
                    Job::Unit(h, bytes) => {
                        written += bytes.len() as u64;
                        if let Err(e) = store.put_unit(&h, &bytes) {
                            eprintln!("[imparo] kv unit write: {e}");
                        }
                        q.lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .remove(&h);
                    }
                    Job::Commit(conv, m, blobs) => {
                        // Blobs first, manifest last: the manifest is the commit line and
                        // must never name a checkpoint that is not yet on disk.
                        let mut ok = true;
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
                                    ok = false;
                                }
                                Err(e) => {
                                    eprintln!("[imparo] kv commit: {e}");
                                    ok = false;
                                }
                            }
                        }
                        if ok {
                            if let Err(e) = store.commit(&conv, &m) {
                                eprintln!("[imparo] kv commit: {e}");
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
                    }
                    Job::Erase(ids, ack) => {
                        let swept = match store.erase(&ids) {
                            Ok(n) => n,
                            Err(e) => {
                                eprintln!("[imparo] kv erase: {e}");
                                0
                            }
                        };
                        let _ = ack.send(swept);
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
                            Err(e) => eprintln!("[imparo] kv gc: {e}"),
                        }
                    }
                    Job::DropManifests(paths) => {
                        match store.erase_paths(&paths) {
                            Ok(0) => {}
                            Ok(n) => {
                                if imparo_model::log_on() {
                                    eprintln!(
                                        "[imparo] kv disk: dropped {} superseded manifest(s), swept {n} unit(s)",
                                        paths.len()
                                    );
                                }
                            }
                            Err(e) => eprintln!("[imparo] kv drop: {e}"),
                        }
                    }
                    Job::Barrier(ack) => {
                        let _ = ack.send(());
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

    pub fn put_unit(&self, h: UnitHash, bytes: Vec<u8>) {
        self.queued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(h);
        let _ = self.tx.send(Job::Unit(h, bytes));
    }

    /// Delete manifests by PATH, off the request thread.
    ///
    /// Ordered behind everything already queued, which is what makes supersession safe:
    /// the commit that covers them is written first.
    pub fn drop_manifests(&self, paths: Vec<std::path::PathBuf>) {
        let _ = self.tx.send(Job::DropManifests(paths));
    }

    pub fn commit(
        &self,
        conv: &str,
        m: Manifest,
        blobs: Vec<(imparo_kv::UnitHash, Vec<u8>)>,
    ) {
        {
            let mut q = self
                .queued
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for (h, _) in &blobs {
                q.insert(*h);
            }
        }
        let _ = self.tx.send(Job::Commit(conv.to_string(), m, blobs));
    }

    /// Erase a conversation set and WAIT for the sweep, returning how many units it
    /// removed.
    ///
    /// Queued, so it lands behind whatever those conversations' last turns were
    /// still writing -- a sweep that overtook a commit would delete the units that
    /// commit is about to name. Blocking, because the caller's reply states the
    /// number.
    pub fn erase_blocking(&self, ids: Vec<String>) -> usize {
        let (ack, wait) = channel();
        if self.tx.send(Job::Erase(ids, ack)).is_err() {
            return 0;
        }
        wait.recv().unwrap_or(0)
    }

    pub fn gc(&self, cap: u64) {
        let _ = self.tx.send(Job::Gc(cap));
    }

    /// Blocks until everything queued so far has been written.
    ///
    /// The erase endpoint needs it (its reply says how many units were swept, so
    /// the sweep has to have happened) and so does shutdown.
    pub fn drain(&self) {
        let (ack, wait) = channel();
        if self.tx.send(Job::Barrier(ack)).is_ok() {
            let _ = wait.recv();
        }
    }
}

impl Drop for DiskQueue {
    fn drop(&mut self) {
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
