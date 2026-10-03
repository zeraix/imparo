//! Optional versioned MoE extension; old plugins retain dense execution.
macro_rules! fields {
    ($declare:ident) => { $declare! {
        imparo_cuda_moe_version() -> u32 = 0;
        imparo_cuda_top_k_rows(src:u32,dst:u32,width:u32,rows:u32,k:u32) -> i32 = -70;
        imparo_cuda_moe_gate(scores:u32,probs:u32,sel:u32,bias:u64,nt:u32,ne:u32,gating:u32) -> i32 = -70;
        imparo_cuda_moe_plan(top:u32,probs:u32,perm:u32,wgt:u32,seg:u32,inv:u32,nt:u32,ne:u32,k:u32,norm:u32,scale:f32) -> i32 = -70;
        imparo_cuda_moe_grouped(kind:u32,off:u64,stride:u64,src:u32,dst:u32,perm:u32,seg:u32,ni:u32,no:u32,ne:u32,nt:u32,rows:u32,work:u32) -> i32 = -70;
        imparo_cuda_moe_combine(src:u32,wgt:u32,inv:u32,dst:u32,width:u32,k:u32,nt:u32) -> i32 = -70;
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
    unsafe { imparo_cuda_moe_version() == 1 }
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
                    (unsafe { (api.imparo_cuda_moe_version)() } == 1).then_some(api)
                }
            }
            fn api()->Option<&'static Api> {
                static API:OnceLock<Option<Api>>=OnceLock::new();
                API.get_or_init(||Api::load(crate::ffi::optional_backend_symbol)).as_ref()
            }
            pub(super) fn available()->bool {unsafe {imparo_cuda_moe_version()==1}}
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
                Api::load(|name| if name == b"imparo_cuda_moe_grouped\0" {
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
        #[test]
        fn missing_optional_pair_keeps_existing_moe_api() {
            assert!(Api::load(|name| {
                if name == b"imparo_cuda_moe_grouped_pair_v1\0" {
                    None
                } else {
                    Some(v1 as *const () as *mut c_void)
                }
            }).is_some());
        }
        #[test]
        fn missing_optional_route_keeps_existing_moe_api() {
            assert!(Api::load(|name| {
                if name == b"imparo_cuda_moe_route_v1\0" {
                    None
                } else {
                    Some(v1 as *const () as *mut c_void)
                }
            }).is_some());
        }
        #[test]
        fn missing_active_policy_keeps_existing_moe_api() {
            assert!(Api::load(|name| {
                if name == b"imparo_cuda_moe_active_experts_v1\0" {
                    None
                } else {
                    Some(v1 as *const () as *mut c_void)
                }
            }).is_some());
        }
    }
}
#[cfg(feature = "cuda-dynamic")]
pub(crate) use loaded::*;
#[cfg(feature = "cuda-dynamic")]
pub(super) fn available() -> bool {
    loaded::available()
}
