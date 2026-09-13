//! Experimental physical window reach, carried with the native owner workflow.
//! Regions are existing KV arenas; these records hold no tensor/cache payload.
use imparo_kv::{KvState, LayerStateGeom, StateKind};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
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
                && std::env::var("IMPARO_KV_HISTORY_LAB").as_deref() == Ok("1"),
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
