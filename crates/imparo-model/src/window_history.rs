//! Experimental physical window reach, carried with the native owner workflow.
//! Regions are existing KV arenas; these records hold no tensor/cache payload.
use imparo_kv::{KvState, LayerStateGeom, StateKind};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::OnceLock;

// The forward diagnostic owns this process-local, one-way reference control.
// The first LFM device batch freezes its value; it cannot change a live owner.
static LFM_FULL_HISTORY_CONTROL: OnceLock<bool> = OnceLock::new();

fn initialize_lfm_full_history_control(
    control: &OnceLock<bool>,
    gate_mode: bool,
    domain: u32,
    finite_history_rows: Option<u32>,
) -> Result<u32, String> {
    if !gate_mode || !(1..=3).contains(&domain) || finite_history_rows != Some(64) {
        return Err("full-history control requires correctness-gate mode and a prepared retained LFM finite64 owner".into());
    }
    control.set(true).map_err(|_| {
        "full-history control must be initialized once before the first LFM forward"
            .to_string()
    })?;
    Ok(domain)
}

/// Forward-only correctness reference. This leaves the registered finite64
/// selection and prepared target/draft policies intact and disables only the
/// proposed finite-history crop. The server has no caller or environment switch.
#[doc(hidden)]
pub fn enable_lfm_full_history_control() -> Result<u32, String> {
    initialize_lfm_full_history_control(
        &LFM_FULL_HISTORY_CONTROL,
        std::env::var("IMPARO_CORRECTNESS_GATE").as_deref() == Ok("1"),
        crate::lfm_retained_domain(),
        crate::backend::active()
            .and_then(imparo_backend::Backend::finite_history_prefill_tail_rows),
    )
}

pub(crate) fn lfm_full_history_control_enabled() -> bool {
    *LFM_FULL_HISTORY_CONTROL.get_or_init(|| false)
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Reach {
    floor: usize,
    end: usize,
    valid: bool,
}
#[derive(Clone)]
pub(crate) struct WindowHistory {
    enabled: bool,
    region: Cell<usize>,
    regions: RefCell<BTreeMap<usize, Reach>>,
}
impl WindowHistory {
    pub(crate) fn new(windowed: bool) -> Self {
        Self {
            enabled: windowed
                && cfg!(feature = "cuda-speculative")
                && (crate::e4b_retained_decode_policy_enabled()
                    || std::env::var("IMPARO_KV_HISTORY_LAB").as_deref() == Ok("1")),
            region: Cell::new(0),
            regions: RefCell::new(BTreeMap::new()),
        }
    }
    pub(crate) fn select_region(&self, region: usize) {
        if self.enabled {
            self.region.set(region);
        }
    }
    pub(crate) fn allows(&self, boundary: usize) -> Option<bool> {
        self.enabled.then(|| {
            self.regions
                .borrow()
                .get(&self.region.get())
                .is_some_and(|r| r.valid && r.floor <= boundary && boundary <= r.end)
        })
    }
    /// Staging must preserve every boundary advertised by the old history.
    pub(crate) fn can_stage(&self, start: usize, end: usize, slack: usize) -> bool {
        if !self.enabled || start == 0 {
            return true;
        }
        self.regions
            .borrow()
            .get(&self.region.get())
            .is_some_and(|r| {
                r.valid
                    && r.floor <= start
                    && start <= r.end
                    && end.saturating_sub(r.floor) <= slack
            })
    }
    /// Reserve ring slack without inventing stored history. Only discard older
    /// rollback boundaries; retain the current start and the original committed end.
    /// Even failure must retain this narrower floor because staging may overwrite
    /// slots belonging to the discarded prefixes.
    pub(crate) fn for_verification(
        &self,
        start: usize,
        end: usize,
        slack: usize,
    ) -> Option<Self> {
        if end < start || end - start > slack {
            return None;
        }
        if self.can_stage(start, end, slack) {
            return Some(self.clone());
        }
        if !self.enabled || self.allows(start) != Some(true) {
            return None;
        }
        let floor = end.saturating_sub(slack);
        if floor > start {
            return None;
        }
        let narrowed = self.clone();
        {
            let mut regions = narrowed.regions.borrow_mut();
            let reach = regions.get_mut(&narrowed.region.get())?;
            reach.floor = reach.floor.max(floor);
        }
        narrowed.can_stage(start, end, slack).then_some(narrowed)
    }
    /// Save only history that survives all writes of this step. A failed
    /// dispatch may retry from start, but must not resurrect overwritten slots.
    pub(crate) fn begin_recoverable(
        &mut self,
        start: usize,
        end: usize,
        slack: usize,
    ) -> Result<(Self, Option<Reach>), String> {
        let retry = if self.enabled {
            self.for_verification(start, end, slack).ok_or_else(|| {
                format!("window history cannot retain retry at {start}")
            })?
        } else {
            self.clone()
        };
        *self = retry.clone();
        let ticket = self.begin(start)?;
        Ok((retry, ticket))
    }
    pub(crate) fn begin(&self, start: usize) -> Result<Option<Reach>, String> {
        if !self.enabled {
            return Ok(None);
        }
        if start != 0 && self.allows(start) != Some(true) {
            return Err(format!("window history does not cover resume {start}"));
        }
        let mut regions = self.regions.borrow_mut();
        let r = regions.entry(self.region.get()).or_default();
        let before = *r;
        r.valid = false;
        Ok(Some(before))
    }
    pub(crate) fn commit(
        &self,
        before: Option<Reach>,
        start: usize,
        end: usize,
        slack: usize,
    ) {
        let Some(before) = before else {
            return;
        };
        let floor = if start == 0 {
            end.saturating_sub(slack)
        } else {
            before.floor.max(end.saturating_sub(slack))
        };
        self.regions.borrow_mut().insert(
            self.region.get(),
            Reach {
                floor,
                end,
                valid: true,
            },
        );
    }
    pub(crate) fn restored(&self, state: &KvState, geom: &[LayerStateGeom]) {
        if !self.enabled {
            return;
        }
        if state.window.is_empty() {
            // An elided restore keeps the region's actual reach. Boundary zero
            // without window bytes instead starts an unknown/cold region.
            if state.boundary == 0 {
                self.regions.borrow_mut().remove(&self.region.get());
            }
            return;
        }
        let complete = geom
            .iter()
            .filter_map(|g| match g.kind {
                StateKind::Window { window, .. } => Some((g, window)),
                StateKind::Full => None,
            })
            .all(|(g, window)| {
                state.window.iter().any(|s| {
                    s.layer == g.layer
                        && s.base_pos <= state.boundary.saturating_sub(window)
                        && s.base_pos
                            .checked_add(s.positions)
                            .is_some_and(|end| end >= state.boundary)
                })
            });
        self.regions.borrow_mut().insert(
            self.region.get(),
            Reach {
                floor: state.boundary,
                end: state.boundary,
                valid: complete,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_history_control_requires_gate_and_prepared_finite64_owner() {
        for (gate, domain, rows) in [
            (false, 1, Some(64)),
            (true, 0, Some(64)),
            (true, 4, Some(64)),
            (true, 1, None),
            (true, 1, Some(0)),
        ] {
            let control = OnceLock::new();
            assert!(
                initialize_lfm_full_history_control(&control, gate, domain, rows)
                    .is_err()
            );
            assert!(control.get().is_none());
        }
        for domain in 1..=3 {
            let control = OnceLock::new();
            assert_eq!(
                initialize_lfm_full_history_control(&control, true, domain, Some(64)),
                Ok(domain)
            );
            assert_eq!(control.get(), Some(&true));
            assert!(
                initialize_lfm_full_history_control(&control, true, domain, Some(64))
                    .is_err()
            );
        }
    }

    #[test]
    fn first_ordinary_forward_freezes_full_history_control_off() {
        let control = OnceLock::new();
        assert!(!*control.get_or_init(|| false));
        assert!(
            initialize_lfm_full_history_control(&control, true, 1, Some(64)).is_err()
        );
        assert_eq!(control.get(), Some(&false));
    }

    fn tracked() -> WindowHistory {
        WindowHistory {
            enabled: true,
            region: Cell::new(0),
            regions: RefCell::new(BTreeMap::new()),
        }
    }
    #[test]
    fn failed_step_retries_without_resurrecting_overwritten_floor() {
        let mut history = tracked();
        let first = history.begin(0).unwrap();
        history.commit(first, 0, 512, 512);
        assert_eq!(history.allows(0), Some(true));

        let (retry, _) = history.begin_recoverable(512, 513, 512).unwrap();
        assert_eq!(history.allows(512), Some(false)); // in flight
        history = retry; // device failure restores the returned safe history
        assert_eq!(history.allows(0), Some(false));
        assert_eq!(history.allows(1), Some(true));
        assert_eq!(history.allows(512), Some(true));
        assert_eq!(history.allows(513), Some(false));

        let (_, ticket) = history.begin_recoverable(512, 513, 512).unwrap();
        history.commit(ticket, 512, 513, 512);
        assert_eq!(history.allows(0), Some(false));
        assert_eq!(history.allows(1), Some(true));
        assert_eq!(history.allows(513), Some(true));
    }
    #[test]
    fn cold_prefill_verification_retains_real_rollback_window() {
        let h = tracked();
        let before = h.begin(0).unwrap();
        h.commit(before, 0, 512, 512);
        assert!(!h.can_stage(512, 515, 512));
        let staged = h.for_verification(512, 515, 512).unwrap();
        assert!(staged.can_stage(512, 515, 512));
        assert_eq!(staged.allows(2), Some(false));
        assert_eq!(staged.allows(3), Some(true));
        assert_eq!(staged.allows(512), Some(true));
        assert_eq!(staged.allows(513), Some(false)); // no uncommitted coverage
        assert_eq!(h.allows(0), Some(true)); // preparation does not mutate owner
        let rollback = staged.clone();
        assert!(rollback.begin(512).is_ok());
        assert!(staged.for_verification(512, 1025, 512).is_none());
        assert!(staged.for_verification(2, 5, 512).is_none());
        assert!(staged.for_verification(513, 516, 512).is_none());
    }
    #[test]
    fn restored_window_cannot_resurrect_older_history() {
        // Boundaries independently reproduced with native ring bytes.
        let h = tracked();
        let ticket = h.begin(0).unwrap();
        h.commit(ticket, 0, 2080, 512);
        assert_eq!(h.allows(1536), Some(false));
        assert_eq!(h.allows(1920), Some(true));
        let geom = [LayerStateGeom {
            layer: 0,
            kind: StateKind::Window {
                window: 512,
                ring: 1024,
            },
            k_stride: 1,
            v_stride: 1,
        }];
        let state = KvState {
            boundary: 1536,
            full: vec![],
            recurrent: vec![],
            window: vec![imparo_kv::KvLayerState {
                layer: 0,
                base_pos: 1024,
                positions: 512,
                k: vec![0; 512],
                v: vec![0; 512],
            }],
        };
        h.restored(&state, &geom);
        assert_eq!(h.allows(1536), Some(true));
        assert_eq!(h.allows(1408), Some(false));
        let ticket = h.begin(1536).unwrap();
        h.commit(ticket, 1536, 1991, 512);
        assert_eq!(h.allows(1408), Some(false));
        assert_eq!(h.allows(1536), Some(true));
    }
    #[test]
    fn unfinished_write_invalidates_only_its_physical_region() {
        let h = tracked();
        let ticket = h.begin(0).unwrap();
        h.commit(ticket, 0, 640, 512);
        h.select_region(1);
        assert_eq!(h.allows(512), Some(false));
        let ticket = h.begin(0).unwrap();
        h.commit(ticket, 0, 672, 512);
        let _uncommitted = h.begin(640).unwrap();
        assert_eq!(h.allows(640), Some(false));
        assert!(h.begin(640).is_err());
        h.select_region(0);
        assert_eq!(h.allows(512), Some(true));
    }
}
