//! Optional owner-local Decode Down MMVQ policy. Missing extensions preserve
//! the original MoE table and grouped arithmetic in older dynamic plugins.
#[cfg(feature = "cuda-static")]
unsafe extern "C" {
    pub(crate) fn imparo_cuda_moe_down_mmvq_v1(enabled: u32) -> i32;
}

#[cfg(feature = "cuda-dynamic")]
mod loaded {
    use core::ffi::c_void;
    use std::sync::OnceLock;
    type Control = unsafe extern "C" fn(u32) -> i32;
    fn resolve(mut lookup: impl FnMut(&'static [u8]) -> Option<*mut c_void>) -> Option<Control> {
        let address = lookup(b"imparo_cuda_moe_down_mmvq_v1\0")?;
        if address.is_null() { return None; }
        Some(unsafe { std::mem::transmute::<*mut c_void, Control>(address) })
    }
    pub(crate) unsafe fn imparo_cuda_moe_down_mmvq_v1(enabled: u32) -> i32 {
        static CONTROL: OnceLock<Option<Control>> = OnceLock::new();
        let Some(control) = CONTROL.get_or_init(|| resolve(crate::ffi::optional_backend_symbol)) else {
            return -70;
        };
        unsafe { control(enabled) }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn missing_down_mmvq_extension_is_optional() {
            assert!(resolve(|name| { assert_eq!(name, b"imparo_cuda_moe_down_mmvq_v1\0"); None }).is_none());
            assert!(resolve(|_| Some(std::ptr::null_mut())).is_none());
        }
    }
}
#[cfg(feature = "cuda-dynamic")]
pub(crate) use loaded::imparo_cuda_moe_down_mmvq_v1;

fn status(rc: i32) -> Result<(), i32> {
    match rc { 0 | -70 => Ok(()), rc => Err(rc) }
}
pub(crate) fn apply_at_boundary() -> Result<(), i32> {
    status(unsafe { imparo_cuda_moe_down_mmvq_v1(u32::from(crate::knobs::moe_down_mmvq_enabled())) })
}
#[cfg(test)]
mod tests {
    #[test]
    fn only_missing_down_mmvq_capability_is_a_silent_fallback() {
        assert_eq!(super::status(0), Ok(()));
        assert_eq!(super::status(-70), Ok(()));
        for rc in [-1, 1, 2, 3] { assert_eq!(super::status(rc), Err(rc)); }
    }
}
