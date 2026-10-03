//! Fixed, default-off selection of the three retained LFM2 execution domains.
//! Native model state is authoritative; this is not a second configuration loader.
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
static APPLY_ERROR: AtomicI32 = AtomicI32::new(0);
static TREE_GRAPH_APPLY_ERROR: AtomicI32 = AtomicI32::new(0);
static TREE_CANDIDATES: AtomicU32 = AtomicU32::new(0);
static TREE_CANDIDATES_APPLY_ERROR: AtomicI32 = AtomicI32::new(0);

#[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
unsafe extern "C" {
    fn imparo_cuda_set_lfm_retained_policy(value: u32) -> i32;
    fn imparo_cuda_lfm_retained_policy() -> u32;
    fn imparo_cuda_lfm_retained_domain() -> u32;
    fn imparo_cuda_set_lfm_tree_graph_short(value: u32) -> i32;
    fn imparo_cuda_lfm_tree_graph_short() -> u32;
    fn imparo_cuda_prepare_lfm_retained_policy(
        capacity: u32,
        batch: u32,
        hidden: u32,
        mid: u32,
        layers: u32,
        heads: u32,
        kv_heads: u32,
        down_kind: u32,
    ) -> i32;
}

pub(crate) fn current() -> u32 {
    #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
    {
        unsafe { imparo_cuda_lfm_retained_policy() }
    }
    #[cfg(not(all(feature = "cuda-speculative", target_os = "windows")))]
    {
        0
    }
}
pub(crate) fn apply(value: u32) {
    #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
    let rc = unsafe { imparo_cuda_set_lfm_retained_policy(value) };
    #[cfg(not(all(feature = "cuda-speculative", target_os = "windows")))]
    let rc = if value == 0 { 0 } else { 3 };
    APPLY_ERROR.store(rc, Ordering::Relaxed);
    if rc != 0 {
        eprintln!("[imparo] retained LFM policy rejected value={value} rc={rc}");
    }
}
pub(crate) fn status() -> Result<(), i32> {
    let candidates = TREE_CANDIDATES_APPLY_ERROR.load(Ordering::Relaxed);
    if candidates != 0 {
        return Err(candidates);
    }
    let graph = TREE_GRAPH_APPLY_ERROR.load(Ordering::Relaxed);
    if graph != 0 {
        return Err(graph);
    }
    match APPLY_ERROR.load(Ordering::Relaxed) {
        0 => Ok(()),
        rc => Err(rc),
    }
}
/// Default-off, registered Graph reuse in the admitted short target domain.
pub(crate) fn tree_graph_short() -> u32 {
    #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
    { unsafe { imparo_cuda_lfm_tree_graph_short() } }
    #[cfg(not(all(feature = "cuda-speculative", target_os = "windows")))]
    { 0 }
}
pub(crate) fn apply_tree_graph_short(value: u32) {
    #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
    let rc = unsafe { imparo_cuda_set_lfm_tree_graph_short(value) };
    #[cfg(not(all(feature = "cuda-speculative", target_os = "windows")))]
    let rc = if value == 0 { 0 } else { 3 };
    TREE_GRAPH_APPLY_ERROR.store(rc, Ordering::Relaxed);
    if rc != 0 {
        eprintln!("[imparo] LFM short tree Graph rejected value={value} rc={rc}");
    }
}
/// Reuse cached draft candidates in the admitted short target domain.
/// This host-only selector is frozen alongside the native model owner.
pub(crate) fn tree_candidates() -> u32 {
    TREE_CANDIDATES.load(Ordering::Relaxed)
}
fn tree_candidates_change_allowed(
    value: u32,
    previous: u32,
    policy: u32,
    bound_domain: u32,
    supported: bool,
) -> bool {
    value <= 3
        && (value == 0 || (supported && policy == 1 && bound_domain <= 1))
        && (bound_domain == 0 || value == previous)
}
pub(crate) fn apply_tree_candidates(value: u32) {
    let accepted = tree_candidates_change_allowed(
        value,
        tree_candidates(),
        current(),
        domain(),
        cfg!(all(feature = "cuda-static", feature = "cuda-speculative", target_os = "windows")),
    );
    let rc = if accepted { 0 } else { 3 };
    if accepted {
        TREE_CANDIDATES.store(value, Ordering::Relaxed);
    }
    TREE_CANDIDATES_APPLY_ERROR.store(rc, Ordering::Relaxed);
    if rc != 0 {
        eprintln!("[imparo] LFM short tree candidates rejected value={value} rc={rc}");
    }
}
pub(crate) fn allow_workflow_change(current: u32, next: u32) -> bool {
    if domain() != 0 && current != next {
        APPLY_ERROR.store(3, Ordering::Relaxed);
        eprintln!("[imparo] retained LFM workflow is model-owner frozen");
        false
    } else {
        true
    }
}
pub(crate) fn domain() -> u32 {
    #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
    {
        unsafe { imparo_cuda_lfm_retained_domain() }
    }
    #[cfg(not(all(feature = "cuda-speculative", target_os = "windows")))]
    {
        0
    }
}
/// A mismatching explicit selection is an error, never an implicit short route.
pub(crate) fn domain_for(capacity: u32, batch: u32, down_kind: u32) -> Option<u32> {
    match (capacity, batch, down_kind) {
        (1024, 512, 3) => Some(1),
        (6656, 1920, 2) => Some(2),
        (16896, 1920, 2) => Some(3),
        _ => None,
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare(
    capacity: u32,
    batch: u32,
    hidden: u32,
    mid: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    down_kind: u32,
) -> Result<(), i32> {
    status()?;
    if tree_candidates() >= 1
        && (current() != 1 || domain_for(capacity, batch, down_kind) != Some(1))
    {
        return Err(3);
    }
    // The first adaptive domain prices eager verification only. A Graph capture
    // or replay is a different cost regime and cannot share its learned table.
    if tree_candidates() == 3 && tree_graph_short() != 0 {
        return Err(3);
    }
    if current() == 0 {
        return Ok(());
    }
    if let Err(error) = crate::correctness::validate_environment(std::env::vars_os()) {
        eprintln!("[imparo] retained LFM environment rejected: {error}");
        return Err(3);
    }
    if domain_for(capacity, batch, down_kind).is_none()
        || crate::knobs::finite_history_prefill_tail_rows() != Some(64)
        || crate::knobs::prefill_bcx_shortconv_ready_enabled()
    {
        return Err(3);
    }
    #[cfg(all(feature = "cuda-speculative", target_os = "windows"))]
    let rc = unsafe {
        imparo_cuda_prepare_lfm_retained_policy(
            capacity, batch, hidden, mid, layers, heads, kv_heads, down_kind,
        )
    };
    #[cfg(not(all(feature = "cuda-speculative", target_os = "windows")))]
    let rc = {
        let _ = (hidden, mid, layers, heads, kv_heads);
        3
    };
    match rc {
        0 => Ok(()),
        rc => Err(rc),
    }
}

#[cfg(test)]
mod tests {
    use super::{domain_for, tree_candidates_change_allowed};
    #[test]
    fn tree_candidates_require_short_retained_owner_and_freeze_after_binding() {
        assert!(tree_candidates_change_allowed(0, 0, 0, 0, false));
        for mode in [1, 2, 3] {
            assert!(tree_candidates_change_allowed(mode, 0, 1, 0, true));
            assert!(tree_candidates_change_allowed(mode, mode, 1, 1, true));
            assert!(!tree_candidates_change_allowed(mode, 0, 0, 0, true));
            assert!(!tree_candidates_change_allowed(mode, 0, 1, 0, false));
            for bound in [1, 2, 3] {
                assert!(!tree_candidates_change_allowed(mode, 0, 1, bound, true));
                assert!(!tree_candidates_change_allowed(0, mode, 1, bound, true));
                let other = if mode == 1 { 2 } else { 1 };
                assert!(!tree_candidates_change_allowed(other, mode, 1, bound, true));
            }
            for wide in [2, 3] {
                assert!(!tree_candidates_change_allowed(mode, mode, 1, wide, true));
            }
        }
        for invalid in [4, u32::MAX] {
            assert!(!tree_candidates_change_allowed(invalid, 0, 1, 0, true));
        }
    }
    #[test]
    fn retained_domains_do_not_cross_layout_or_batch_boundaries() {
        for (capacity, batch, down, domain) in
            [(1024, 512, 3, 1), (6656, 1920, 2, 2), (16896, 1920, 2, 3)]
        {
            assert_eq!(domain_for(capacity, batch, down), Some(domain));
            for c in [capacity - 1, capacity + 1] {
                assert_eq!(domain_for(c, batch, down), None);
            }
            assert_eq!(
                domain_for(capacity, if batch == 512 { 1920 } else { 512 }, down),
                None
            );
            assert_eq!(
                domain_for(capacity, batch, if down == 2 { 3 } else { 2 }),
                None
            );
        }
    }

    #[test]
    fn short_layout_cannot_expand_to_n2000_or_wide_capacities() {
        assert_eq!(domain_for(1024, 512, 3), Some(1));
        for capacity in [2016, 6656, 16896] {
            assert_eq!(domain_for(capacity, 512, 3), None);
        }
    }
}
