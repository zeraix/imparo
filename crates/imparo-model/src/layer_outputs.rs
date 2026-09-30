//! Optional dense layer outputs required by an attached consumer.
//! Model workflows publish only after the full residual is available.
use imparo_backend::BufId;

/// Called as `(layer, start position, rows, source)` once a subscribed layer's output is
/// complete in `source`. Observers carry their layout in a closure; the native CUDA
/// subscriber has an explicit identity so graph replay cannot skip arbitrary observers.
pub(crate) enum CaptureFn {
    Observer(Box<dyn Fn(u32, u32, u32, BufId) -> Result<(), String> + Send + Sync>),
    // This variant names the callback itself, not an assertion about an arbitrary
    // observer. Only its native metadata can be republished by CUDA tree replay.
    #[cfg(feature = "cuda-speculative")]
    CudaDspark,
}
impl CaptureFn {
    pub(crate) fn observer(
        f: impl Fn(u32, u32, u32, BufId) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        Self::Observer(Box::new(f))
    }
    fn record(
        &self,
        layer: u32,
        start: u32,
        rows: u32,
        source: BufId,
    ) -> Result<(), String> {
        match self {
            Self::Observer(f) => f(layer, start, rows, source),
            #[cfg(feature = "cuda-speculative")]
            Self::CudaDspark => {
                crate::dspark::capture_layer_output(layer, start, rows, source)
            }
        }
    }
    #[cfg(feature = "cuda-speculative")]
    pub(crate) fn is_native_dspark(&self) -> bool {
        matches!(self, Self::CudaDspark)
    }
}

pub struct LayerOutputCapture {
    pub(crate) enabled: std::sync::Weak<std::sync::atomic::AtomicBool>,
    pub(crate) layers: Vec<u32>,
    pub(crate) capture: CaptureFn,
}
impl LayerOutputCapture {
    // Only the speculative drafters (dspark, gemma4_mtp) ask this.
    #[cfg(feature = "speculative")]
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
            self.capture.record(layer, start, tokens, source)?;
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
            capture: CaptureFn::observer(|_, _, _, _| Ok(())),
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
    #[test]
    fn observer_identity_and_errors_are_preserved() {
        let enabled = Arc::new(AtomicBool::new(true));
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let capture = LayerOutputCapture {
            enabled: Arc::downgrade(&enabled),
            layers: vec![3],
            capture: CaptureFn::observer(move |_, _, _, _| {
                seen.fetch_add(1, Ordering::Relaxed);
                Err("observer error".into())
            }),
        };
        #[cfg(feature = "cuda-speculative")]
        {
            assert!(!capture.capture.is_native_dspark());
            assert!(CaptureFn::CudaDspark.is_native_dspark());
        }
        assert!(capture.record(2, 0, 1, BufId::Cur).is_ok());
        assert_eq!(
            capture.record(3, 0, 1, BufId::Cur).unwrap_err(),
            "observer error"
        );
        enabled.store(false, Ordering::Relaxed);
        assert!(capture.record(3, 0, 1, BufId::Cur).is_ok());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
