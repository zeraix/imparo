//! Optional owner-local Decode GateUp MMVQ policy and fused operation.
//! The ABI keeps the workspace argument; the fused native provider does not use it.
//! Both exports are optional; the original MoE plugin table stays unchanged.
#[cfg(feature = "cuda-static")]
unsafe extern "C" {
    pub(crate) fn imparo_cuda_moe_gateup_mmvq_v1(enabled: u32) -> i32;
    pub(crate) fn imparo_cuda_moe_gateup_mmvq_pair_v1(
        kind: u32, gate_off: u64, up_off: u64, stride: u64,
        src: u32, dst: u32, scratch: u32, perm: u32, seg: u32,
        ni: u32, no: u32, ne: u32, nt: u32, rows: u32,
    ) -> i32;
}

#[cfg(feature = "cuda-dynamic")]
mod loaded {
    use core::ffi::c_void;
    use std::sync::OnceLock;
    type Control = unsafe extern "C" fn(u32) -> i32;
    type Pair = unsafe extern "C" fn(
        u32, u64, u64, u64, u32, u32, u32, u32, u32, u32, u32, u32, u32, u32,
    ) -> i32;
    struct Api { control: Control, pair: Pair }
    fn resolve(mut lookup: impl FnMut(&'static [u8]) -> Option<*mut c_void>) -> Option<Api> {
        let control = lookup(b"imparo_cuda_moe_gateup_mmvq_v1\0")?;
        let pair = lookup(b"imparo_cuda_moe_gateup_mmvq_pair_v1\0")?;
        if control.is_null() || pair.is_null() { return None; }
        Some(Api {
            control: unsafe { std::mem::transmute::<*mut c_void, Control>(control) },
            pair: unsafe { std::mem::transmute::<*mut c_void, Pair>(pair) },
        })
    }
    fn api() -> Option<&'static Api> {
        static API: OnceLock<Option<Api>> = OnceLock::new();
        API.get_or_init(|| resolve(crate::ffi::optional_backend_symbol)).as_ref()
    }
    pub(crate) unsafe fn imparo_cuda_moe_gateup_mmvq_v1(enabled: u32) -> i32 {
        let Some(api) = api() else { return -70; };
        unsafe { (api.control)(enabled) }
    }
    pub(crate) unsafe fn imparo_cuda_moe_gateup_mmvq_pair_v1(
        kind: u32, gate_off: u64, up_off: u64, stride: u64,
        src: u32, dst: u32, scratch: u32, perm: u32, seg: u32,
        ni: u32, no: u32, ne: u32, nt: u32, rows: u32,
    ) -> i32 {
        let Some(api) = api() else { return -70; };
        unsafe { (api.pair)(kind, gate_off, up_off, stride, src, dst, scratch, perm, seg, ni, no, ne, nt, rows) }
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        unsafe extern "C" fn control(_: u32) -> i32 { 0 }
        unsafe extern "C" fn pair(
            _: u32, _: u64, _: u64, _: u64, _: u32, _: u32, _: u32,
            _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32,
        ) -> i32 { 0 }
        #[test]
        fn missing_or_partial_gateup_mmvq_extension_is_optional() {
            assert!(resolve(|_| None).is_none());
            assert!(resolve(|_| Some(std::ptr::null_mut())).is_none());
            assert!(resolve(|name| (name == b"imparo_cuda_moe_gateup_mmvq_v1\0")
                .then_some(control as *const () as *mut c_void)).is_none());
            assert!(resolve(|name| (name == b"imparo_cuda_moe_gateup_mmvq_pair_v1\0")
                .then_some(pair as *const () as *mut c_void)).is_none());
            assert!(resolve(|name| Some(if name == b"imparo_cuda_moe_gateup_mmvq_v1\0" {
                control as *const () as *mut c_void
            } else {
                assert_eq!(name, b"imparo_cuda_moe_gateup_mmvq_pair_v1\0");
                pair as *const () as *mut c_void
            })).is_some());
        }
    }
}
#[cfg(feature = "cuda-dynamic")]
pub(crate) use loaded::{imparo_cuda_moe_gateup_mmvq_pair_v1, imparo_cuda_moe_gateup_mmvq_v1};

fn compute_status(rc: i32) -> Result<bool, i32> {
    match rc { 0 => Ok(true), -70 => Ok(false), rc => Err(rc) }
}
pub(crate) fn apply_at_boundary() -> Result<(), i32> {
    compute_status(unsafe {
        imparo_cuda_moe_gateup_mmvq_v1(u32::from(crate::knobs::moe_gateup_mmvq_enabled()))
    }).map(|_| ())
}
pub(crate) unsafe fn try_pair(
    kind: u32, gate_off: u64, up_off: u64, stride: u64,
    src: u32, dst: u32, scratch: u32, perm: u32, seg: u32,
    ni: u32, no: u32, ne: u32, nt: u32, rows: u32,
) -> Result<bool, i32> {
    compute_status(unsafe {
        imparo_cuda_moe_gateup_mmvq_pair_v1(kind, gate_off, up_off, stride, src, dst, scratch, perm, seg, ni, no, ne, nt, rows)
    })
}
#[cfg(test)]
mod tests {
    #[test]
    fn only_unsupported_gateup_mmvq_allows_arithmetic_fallback() {
        assert_eq!(super::compute_status(0), Ok(true));
        assert_eq!(super::compute_status(-70), Ok(false));
        for rc in [-1, 1, 2, 3] { assert_eq!(super::compute_status(rc), Err(rc)); }
    }
}
