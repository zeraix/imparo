//! Single-token route fusion is independent of the original MoE v1 table and
//! of the optional grouped pair. An older plugin retains both original fallbacks.
#[cfg(feature = "cuda-static")]
unsafe extern "C" {
    pub(crate) fn imparo_cuda_moe_route_v1(
        scores: u32, probs: u32, sel: u32, top: u32,
        perm: u32, wgt: u32, seg: u32, inv: u32,
        bias: u64, nt: u32, ne: u32, k: u32, gating: u32, norm: u32, scale: f32,
    ) -> i32;
}

#[cfg(feature = "cuda-dynamic")]
mod loaded {
    use core::ffi::c_void;
    use std::sync::OnceLock;

    type Route = unsafe extern "C" fn(
        u32, u32, u32, u32, u32, u32, u32, u32,
        u64, u32, u32, u32, u32, u32, f32,
    ) -> i32;

    fn resolve(mut lookup: impl FnMut(&'static [u8]) -> Option<*mut c_void>) -> Option<Route> {
        let address = lookup(b"imparo_cuda_moe_route_v1\0")?;
        if address.is_null() { return None; }
        Some(unsafe { std::mem::transmute::<*mut c_void, Route>(address) })
    }

    pub(crate) unsafe fn imparo_cuda_moe_route_v1(
        scores: u32, probs: u32, sel: u32, top: u32,
        perm: u32, wgt: u32, seg: u32, inv: u32,
        bias: u64, nt: u32, ne: u32, k: u32, gating: u32, norm: u32, scale: f32,
    ) -> i32 {
        static ROUTE: OnceLock<Option<Route>> = OnceLock::new();
        let Some(route) = ROUTE.get_or_init(|| resolve(crate::ffi::optional_backend_symbol)) else {
            return -70;
        };
        unsafe { route(scores, probs, sel, top, perm, wgt, seg, inv, bias, nt, ne, k, gating, norm, scale) }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn missing_route_extension_is_optional() {
            assert!(resolve(|_| None).is_none());
            assert!(resolve(|_| Some(std::ptr::null_mut())).is_none());
        }
    }
}

#[cfg(feature = "cuda-dynamic")]
pub(crate) use loaded::imparo_cuda_moe_route_v1;
