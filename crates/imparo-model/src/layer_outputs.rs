//! Optional dense layer outputs required by an attached consumer.
//! Model workflows publish only after the full residual is available.
use imparo_backend::BufId;

pub struct LayerOutputCapture {
    pub(crate) enabled: std::sync::Weak<std::sync::atomic::AtomicBool>,
    pub(crate) layers: Vec<u32>,
    pub(crate) capture: fn(u32, u32, u32, BufId) -> Result<(), String>,
}
impl LayerOutputCapture {
    // Only the speculative drafters (dspark, gemma4_mtp) ask this.
    #[cfg(feature = "cuda-speculative")]
    pub(crate) fn attached(&self) -> bool {
        self.enabled.strong_count() != 0
    }
    pub(crate) fn active(&self) -> bool {
        self.enabled
            .upgrade()
            .is_some_and(|x| x.load(std::sync::atomic::Ordering::Relaxed))
    }
    /// A crop after this layer is legal only when no active subscriber needs
    /// dense rows from a later layer. Publication of this layer must finish first.
    pub(crate) fn observes_after(&self, layer: u32) -> bool {
        self.active() && self.layers.iter().any(|&required| required > layer)
    }
    pub(crate) fn record(
        &self,
        layer: u32,
        start: u32,
        tokens: u32,
        source: BufId,
    ) -> Result<(), String> {
        if self.active() && self.layers.binary_search(&layer).is_ok() {
            (self.capture)(layer, start, tokens, source)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    #[test]
    fn suffix_crop_waits_for_last_active_subscriber() {
        let enabled = Arc::new(AtomicBool::new(true));
        let capture = LayerOutputCapture {
            enabled: Arc::downgrade(&enabled),
            layers: vec![3, 12, 27],
            capture: |_, _, _, _| Ok(()),
        };
        assert!(capture.observes_after(26));
        assert!(!capture.observes_after(27));
        enabled.store(false, Ordering::Relaxed);
        assert!(!capture.observes_after(0));
        enabled.store(true, Ordering::Relaxed);
        assert!(capture.observes_after(12));
        drop(enabled);
        assert!(!capture.observes_after(0));
    }
}
