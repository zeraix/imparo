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
pub mod disk;
pub mod identity;
pub mod index;
pub mod pool;
pub mod resident;
pub mod state;
pub mod store;
pub mod tenant;

pub use alloc::BlockAllocator;
pub use disk::DiskQueue;
pub use identity::{
    ConfigRoot, DEFAULT_PAGE_CELLS, KV_BYTE_LAYOUT_PROFILE_VERSION,
    KvByteLayoutProfile, KvByteType, KvQuantizationBasis, PrefixHash, UnitHash,
    grid_tokens, page_cells, prefix_hashes, resident_bounds, set_page_cells, unit_id,
    unit_ids, unit_ids_at,
};
pub use index::{AdoptError, ConversationId, Lookup, Pool, Probe, UnitId};
pub use pool::PoolMode;
pub use resident::{
    DemotionPlan, DroppedConversation, EvictedResidency, HostResidency,
    PlacementLifecycle, PromotionPlan, ResidencyError, ResidentKv, ResidentProbe,
    ResidentProbePlan, SealOutcome, TableError, UNIT_BLOCKS, UnitPlacement,
    UnitResidency,
};
pub use state::{
    CheckpointShape, KvDelta, KvLayerState, KvState, LayerStateGeom, StateKind,
};
pub use store::{Manifest, Store};
pub use tenant::PoolTenant;

/// Whether the engine's own logging is on (`IMPARO_LOG`).
///
/// A copy of `imparo_model::log_on` rather than a call to it: this crate sits below
/// the model crate, and one env read is not worth a dependency edge.
#[must_use]
pub fn log_on() -> bool {
    std::env::var("IMPARO_LOG").is_ok_and(|v| v != "0" && !v.is_empty())
}
