//! Independent optional MoE pair extension. Do not add it to the MoE v1 table:
//! old plugins must keep their existing routed execution when this is absent.
#[cfg(feature = "cuda-static")]
unsafe extern "C" {
    pub(crate) fn imparo_cuda_moe_grouped_pair_v1(
        kind: u32, gate_off: u64, up_off: u64, stride: u64,
        src: u32, dst: u32, perm: u32, seg: u32,
        ni: u32, no: u32, ne: u32, nt: u32, rows: u32,
    ) -> i32;
}

#[cfg(feature = "cuda-dynamic")]
mod loaded {
    use core::ffi::c_void;
    use std::sync::OnceLock;

    type Pair = unsafe extern "C" fn(
        u32, u64, u64, u64, u32, u32, u32, u32, u32, u32, u32, u32, u32,
    ) -> i32;

    fn resolve(mut lookup: impl FnMut(&'static [u8]) -> Option<*mut c_void>) -> Option<Pair> {
        let address = lookup(b"imparo_cuda_moe_grouped_pair_v1\0")?;
        if address.is_null() { return None; }
        Some(unsafe { std::mem::transmute::<*mut c_void, Pair>(address) })
    }

    pub(crate) unsafe fn imparo_cuda_moe_grouped_pair_v1(
        kind: u32, gate_off: u64, up_off: u64, stride: u64,
        src: u32, dst: u32, perm: u32, seg: u32,
        ni: u32, no: u32, ne: u32, nt: u32, rows: u32,
    ) -> i32 {
        static PAIR: OnceLock<Option<Pair>> = OnceLock::new();
        let Some(pair) = PAIR.get_or_init(|| resolve(crate::ffi::optional_backend_symbol)) else {
            return -70;
        };
        unsafe { pair(kind, gate_off, up_off, stride, src, dst, perm, seg, ni, no, ne, nt, rows) }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn missing_pair_extension_is_optional() {
            assert!(resolve(|_| None).is_none());
            assert!(resolve(|_| Some(std::ptr::null_mut())).is_none());
        }
    }
}

#[cfg(feature = "cuda-dynamic")]
pub(crate) use loaded::imparo_cuda_moe_grouped_pair_v1;
