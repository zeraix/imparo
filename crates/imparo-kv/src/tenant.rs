//! What the pool needs from whoever owns the cache.
//!
//! THE POOL IS A KV STRUCTURE, so it lives here -- but it has to drive a model, and the
//! model crate sits above this one. Rather than invert the crates (which would carry
//! `KvState`, `KvDelta` and the store up into `imparo-model`, leaving a crate called
//! `imparo-kv` that no longer holds `KvState`), the DEPENDENCY is inverted: the lower
//! crate owns the interface and the upper one conforms.
//!
//! `imparo_model::kv::KvPoolMember` is the model-side half. It has 32 methods, because
//! it also carries the machinery that lets a model implement them from two accessors --
//! `plan()` and `state()` -- rather than writing 170 lines. The pool calls TEN of them,
//! and every one is expressible in this crate's own types once two substitutions are
//! made:
//!
//! ```text
//!   plan() -> &ModelPlan      the pool wanted one number from it   -> recurrent_elems()
//!   backend::active()         a registry that lives in the model crate -> backend()
//! ```
//!
//! So the trait that looked model-shaped was two traits sharing a name: the CONTRACT (a
//! tenant of the pool) and the CONVENIENCE (how a model satisfies it cheaply). Only the
//! first belongs down here.

use crate::state::{KvDelta, KvRuntime, KvState, LayerStateGeom};
use imparo_backend::Backend;
use std::collections::BTreeMap;

/// A cache owner the pool can place, capture, spill and resume.
///
/// Implemented for every `KvPoolMember` by a blanket impl in `imparo-model`, so a model
/// gains it by being a pool member and never names this trait.
pub trait PoolTenant {
    /// Elements in the recurrent state, or 0 for a model with none.
    ///
    /// The pool asks only whether a captured checkpoint's recurrent half is required:
    /// restoring a convolution model with a zero history gives the right shape and the
    /// wrong numbers, which is a failure that reads as success.
    fn recurrent_elems(&self) -> usize;

    /// The device this tenant computes on, when there is one.
    ///
    /// The pool needs it to advise pages and to move rows; `imparo-backend` is already a
    /// dependency here, so only the REGISTRY (which backend is active) lived too high.
    fn backend(&self) -> Option<&'static dyn Backend>;

    /// Capacity, allocated slots, positions filled.
    fn kv_runtime(&self) -> &KvRuntime;

    /// Per-layer capture geometry: what is windowed, what is full, and the row strides.
    fn kv_state_geometry(&self) -> Vec<LayerStateGeom>;

    /// Install per-layer block tables, so reads follow the pool's placement.
    fn kv_apply_tables(&self, tables: &BTreeMap<u32, Vec<u32>>);

    /// Point the windowed rings at one region of the ring arena.
    fn kv_apply_region(&self, region: usize);

    /// The recurrent buffer and the position it describes, if the model keeps one.
    fn kv_recurrent_note(&self) -> Option<(usize, Vec<u8>)>;

    /// The whole windowed state at `boundary` -- an anchor's worth of rows.
    fn kv_capture_windows(&self, boundary: usize) -> KvState;

    /// One chain link: the window rows this turn OWNS, `[from, boundary)`.
    fn kv_capture_delta(&self, boundary: usize, from: usize) -> KvDelta;

    /// Walk a chain newest-first back into a whole windowed state.
    fn kv_assemble_chain(&self, chain: &[&KvDelta]) -> Option<KvState>;

    /// Write a captured state back and take it as the live cache.
    ///
    /// # Errors
    /// When the state does not fit the model's geometry, or the device refuses it.
    fn kv_resume(&mut self, state: &KvState) -> Result<(), String>;
}
