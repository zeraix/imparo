//! Versioned, optional CoBatch extension. Existing ABI plugins remain loadable.
//! Admission is all-or-nothing: slots, state budget, row consumers and the matching
//! single-row recurrent operations must all resolve before any capability is exposed.
use super::{SlotRowWire, SlotStateRowWire};
use crate::ffi::GatedDeltaWire;

macro_rules! fields {
    ($declare:ident) => { $declare! {
            imparo_cuda_cobatch_version() -> u32 = 0;
            imparo_cuda_argmax_rows(src:u32,dst:u32,width:u32,rows:u32) -> () = ();
            imparo_cuda_cobatch_route(route: u32) -> i32 = -70;
            imparo_cuda_cobatch_head(buf: u32, weight: u64, hd: u32, eps: f32, heads: u32, pos: *const u32, count: u32, rd: u32, base: f32, freqs: *const f32, nrot: u32) -> i32 = -70;
            imparo_cuda_cobatch_kv_head(k: u32, v: u32, weight: u64, hd: u32, eps: f32, heads: u32, pos: *const u32, count: u32, rd: u32, base: f32, freqs: *const f32, hk: u32, hv: u32) -> i32 = -70;
            imparo_cuda_cobatch_store(src: u32, layer: u32, width: u32, rows: *const SlotRowWire, count: u32, is_v: u32, ring: u32) -> i32 = -70;
            imparo_cuda_cobatch_attention(layer: u32, hd: u32, heads: u32, kvheads: u32, width: u32, scale: f32, window: u32, rows: *const SlotRowWire, count: u32, ring: u32) -> i32 = -70;
            imparo_cuda_cobatch_conv(form: u32, src: u32, weight: u64, state: u32, rows: *const SlotStateRowWire, count: u32, out: u32, width: u32, kernel: u32) -> i32 = -70;
            imparo_cuda_cobatch_delta(op: *const GatedDeltaWire, rows: *const SlotStateRowWire, count: u32) -> i32 = -70;
            imparo_cuda_set_slots(n: u32, rings: *const u32, nrings: u32) -> i32 = -70;
            imparo_cuda_select_slot(slot: u32) -> i32 = -70;
            imparo_cuda_release_slot(slot: u32) -> i32 = -70;
            imparo_cuda_cobatch_committed_runtime(kv: *mut u64, activations: *mut u64) -> i32 = -70;
            imparo_cuda_set_placement_reserve(bytes: u64) -> i32 = -70;
            imparo_cuda_delta_net_run(op: *const GatedDeltaWire) -> i32 = -70;
            imparo_cuda_plain_conv(src: u32, w_off: u64, state: u32, state_off: u32, state_out_off: u32, out: u32, width: u32, kernel: u32, n_tok: u32) -> () = ();
            imparo_cuda_plain_conv_snapshot(src: u32, state: u32, state_off: u32, snap: u32, snap_off: u32, width: u32, kernel: u32, n_tok: u32) -> () = ();
            imparo_cuda_mul_strided_sigmoid(a: u32, b: u32, width: u32, b_off: u32, b_stride: u32, a_stride: u32, n_rows: u32) -> () = ();
            imparo_cuda_copy_strided(dst: u32, src: u32, width: u32, src_off: u32, src_stride: u32, n_row: u32) -> () = ();
    } };
}
#[cfg(feature = "cuda-static")]
macro_rules! declare {
    ($( $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $fallback:expr; )*) => {
        unsafe extern "C" { $(pub(crate) fn $name($($arg:$ty),*) -> $ret;)* }
    };
}
#[cfg(feature = "cuda-static")]
fields!(declare);
#[cfg(feature = "cuda-static")]
pub(super) fn available() -> bool {
    unsafe { imparo_cuda_cobatch_version() == 1 }
}

#[cfg(feature = "cuda-dynamic")]
mod loaded {
    use super::*;
    use core::ffi::c_void;
    use std::sync::OnceLock;
    macro_rules! declare {
        ($( $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty = $fallback:expr; )*) => {
            struct Api { $( $name: unsafe extern "C" fn($($ty),*) -> $ret, )* }
            impl Api {
                fn load(mut lookup:impl FnMut(&'static [u8])->Option<*mut c_void>)->Option<Self> {
                    let api=Self { $( $name:unsafe {
                        std::mem::transmute::<*mut c_void,unsafe extern "C" fn($($ty),*) -> $ret>(
                            lookup(concat!(stringify!($name), "\0").as_bytes())?)
                    }, )* };
                    (unsafe { (api.imparo_cuda_cobatch_version)() } == 1).then_some(api)
                }
            }
            fn api()->Option<&'static Api> {
                static API:OnceLock<Option<Api>>=OnceLock::new();
                API.get_or_init(||Api::load(crate::ffi::optional_backend_symbol)).as_ref()
            }
            pub(super) fn available()->bool {unsafe {imparo_cuda_cobatch_version()==1}}
            $(pub(crate) unsafe fn $name($($arg:$ty),*) -> $ret {
                let Some(api)=api() else {return $fallback;};
                unsafe {(api.$name)($($arg),*)}
            })*
        };
    }
    fields!(declare);
    #[cfg(test)]
    mod tests {
        use super::*;
        unsafe extern "C" fn v1() -> u32 {
            1
        }
        unsafe extern "C" fn v2() -> u32 {
            2
        }
        #[test]
        fn missing_extension_or_one_missing_member_is_not_admitted() {
            assert!(Api::load(|_| None).is_none());
            assert!(
                Api::load(|name| if name == b"imparo_cuda_cobatch_store\0" {
                    None
                } else {
                    Some(v1 as *const () as *mut c_void)
                })
                .is_none()
            );
        }
        #[test]
        fn unsupported_extension_version_is_not_admitted() {
            assert!(Api::load(|_| Some(v2 as *const () as *mut c_void)).is_none());
            // Only the version function is called; the other pointers are resolution stubs.
            assert!(Api::load(|_| Some(v1 as *const () as *mut c_void)).is_some());
        }
    }
}
#[cfg(feature = "cuda-dynamic")]
pub(crate) use loaded::*;
#[cfg(feature = "cuda-dynamic")]
pub(super) fn available() -> bool {
    loaded::available()
}
