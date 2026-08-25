//! The unified KV pool (design: docs/unified-kv-pool.md).
//!
//! One crate, below imparo-model and above the Backend trait, identical on mac-metal
//! and win-cuda. NO model knowledge (a layer is "a state kind with a geometry"), NO
//! device knowledge beyond the backend's declared capability descriptor, no `cfg`
//! anywhere.
//!
//! Landed so far (the design doc's implementation sequencing):
//!   step 1  content identity — the chained unit hash and the content index
//!   step 2  refcounted sharing — adopt on lookup, merge on seal
//!   step 3  deterministic allocation — lowest free block, a pure function of the
//!           free set
//!   step 10/11 (partial) disk tier — sealed-unit store, manifests (commit line
//!           written last), content probe, erase-with-sweep, LRU GC
//! Everything here is compute/bookkeeping only until the wiring steps land; engine
//! behavior is unchanged while this crate is not called.

pub mod alloc;
pub mod identity;
pub mod index;
pub mod resident;
pub mod state;
pub mod store;

pub use alloc::BlockAllocator;
pub use identity::{ConfigRoot, UNIT_TOKENS, UnitHash, unit_hashes};
pub use index::{ConversationId, Pool, Probe, UnitId};
pub use resident::{ResidentKv, UNIT_BLOCKS};
pub use state::{KvDelta, KvLayerState, KvState, LayerStateGeom, StateKind};
pub use store::{Manifest, Store};
