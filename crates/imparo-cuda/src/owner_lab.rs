//! Static development-only execution views for one shared model.
//!
//! This is not a concurrent backend API. Every call requires exclusive,
//! serialized access to the process's native backend and paired model state.

unsafe extern "C" {
    fn imparo_cuda_execution_owner_create_lab(id: *mut u64) -> i32;
    fn imparo_cuda_execution_owner_select_lab(id: u64) -> i32;
    fn imparo_cuda_execution_owner_release_lab(id: u64) -> i32;
}

/// # Safety
/// Own exclusive access to the initialized backend until all views are retired.
/// Retain the model weights and pair each view with its own WorkflowState.
pub unsafe fn create() -> Result<u64, i32> {
    let mut id = 0;
    let rc = unsafe { imparo_cuda_execution_owner_create_lab(&raw mut id) };
    if rc == 0 { Ok(id) } else { Err(rc) }
}

/// # Safety
/// Select only between closed forwards, and install the matching host workflow
/// state before any further backend/model operation. No other caller may submit.
pub unsafe fn select(id: u64) -> Result<(), i32> {
    let rc = unsafe { imparo_cuda_execution_owner_select_lab(id) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

/// # Safety
/// The view must be inactive; discard its host state only after success.
/// On failure retain the model/owner lifetime and do not reuse its handle.
pub unsafe fn release(id: u64) -> Result<(), i32> {
    let rc = unsafe { imparo_cuda_execution_owner_release_lab(id) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

#[repr(C)]
struct FfnPhaseWire {
    gate_off: u64,
    up_off: u64,
    down_off: u64,
    layer: u32,
    gate_kind: u32,
    up_kind: u32,
    down_kind: u32,
    n_in: u32,
    n_mid: u32,
    n_out: u32,
    src: u32,
    dst: u32,
    tokens: u32,
    activation: u32,
    reserved: u32,
}
unsafe extern "C" {
    fn imparo_cuda_record_ffn_phase_lab(
        wire: *const FfnPhaseWire,
        entering: u32,
    ) -> i32;
}
/// # Safety
/// Called in the same serialized model encode as the described FFN.
/// This records metadata during capture; it does not change GPU work.
pub unsafe fn record_ffn(
    p: imparo_backend::DecodeFfnPhase,
    entering: bool,
) -> Result<(), i32> {
    let wire = FfnPhaseWire {
        gate_off: p.gate_off,
        up_off: p.up_off,
        down_off: p.down_off,
        layer: p.layer,
        gate_kind: p.gate_kind,
        up_kind: p.up_kind,
        down_kind: p.down_kind,
        n_in: p.n_in,
        n_mid: p.n_mid,
        n_out: p.n_out,
        src: p.src as u32,
        dst: p.dst as u32,
        tokens: p.tokens,
        activation: p.activation as u32,
        reserved: 0,
    };
    let rc = unsafe {
        imparo_cuda_record_ffn_phase_lab(&raw const wire, u32::from(entering))
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

unsafe extern "C" {
    fn imparo_cuda_ffn_group_create_lab(
        ids: *const u64,
        count: u32,
        out: *mut u64,
    ) -> i32;
    fn imparo_cuda_ffn_group_step_lab(
        id: u64,
        members: *const u32,
        tokens: *const u32,
        positions: *const u32,
        count: u32,
        grouped: u32,
        out: *mut u32,
    ) -> i32;
    fn imparo_cuda_ffn_group_read_lab(
        id: u64,
        member: u32,
        buf: u32,
        out: *mut f32,
        n: u64,
    ) -> i32;
    fn imparo_cuda_ffn_group_read_kv_lab(
        id: u64,
        member: u32,
        layer: u32,
        is_v: u32,
        off: u64,
        out: *mut u8,
        n: u64,
    ) -> i32;
    fn imparo_cuda_ffn_group_release_lab(id: u64) -> i32;
}
/// # Safety
/// Exclusive native access; all owners have matching captured model graphs.
/// Retain weights, host states and owner lifetimes until group release succeeds.
pub unsafe fn group_create(ids: &[u64]) -> Result<u64, i32> {
    if !(2..=5).contains(&ids.len()) {
        return Err(4);
    }
    let mut id = 0;
    let rc = unsafe {
        imparo_cuda_ffn_group_create_lab(ids.as_ptr(), ids.len() as u32, &raw mut id)
    };
    if rc == 0 { Ok(id) } else { Err(rc) }
}
/// # Safety
/// Exclusive access, default view selected, live group and matching host states.
/// Tokens and positions must be valid for the retained model and KV history.
/// Advance host state only after success. On failure do not retry submitted work.
pub unsafe fn group_step(
    id: u64,
    members: &[u32],
    tokens: &[u32],
    positions: &[u32],
    grouped: bool,
    out: &mut [u32],
) -> Result<(), i32> {
    let n = members.len();
    if !(1..=4).contains(&n)
        || tokens.len() != n
        || positions.len() != n
        || out.len() != n
    {
        return Err(4);
    }
    let rc = unsafe {
        imparo_cuda_ffn_group_step_lab(
            id,
            members.as_ptr(),
            tokens.as_ptr(),
            positions.as_ptr(),
            n as u32,
            u32::from(grouped),
            out.as_mut_ptr(),
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
/// # Safety
/// Exclusive access to the idle group, all retained lifetimes remain valid.
pub unsafe fn group_read(
    id: u64,
    member: u32,
    buf: imparo_backend::BufId,
    out: &mut [f32],
) -> Result<(), i32> {
    let rc = unsafe {
        imparo_cuda_ffn_group_read_lab(
            id,
            member,
            buf as u32,
            out.as_mut_ptr(),
            out.len() as u64,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
/// # Safety
/// Exclusive access to the idle group and a valid bounded KV byte interval.
pub unsafe fn group_read_kv(
    id: u64,
    member: u32,
    layer: u32,
    is_v: bool,
    off: u64,
    out: &mut [u8],
) -> Result<(), i32> {
    let rc = unsafe {
        imparo_cuda_ffn_group_read_kv_lab(
            id,
            member,
            layer,
            u32::from(is_v),
            off,
            out.as_mut_ptr(),
            out.len() as u64,
        )
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
/// # Safety
/// Exclusive access and default view selected; retain owners on failure.
pub unsafe fn group_release(id: u64) -> Result<(), i32> {
    let rc = unsafe { imparo_cuda_ffn_group_release_lab(id) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

unsafe extern "C" {
    fn imparo_cuda_execution_owner_graph_range_lab(
        id: u64,
        lo: *mut u32,
        hi: *mut u32,
    ) -> i32;
}
/// # Safety
/// Exclusive backend access, live owner and default view selected. A None result
/// means use normal model execution; errors from group_step must never be retried.
pub unsafe fn graph_range(id: u64) -> Result<Option<(u32, u32)>, i32> {
    let (mut lo, mut hi) = (0, 0);
    let rc = unsafe {
        imparo_cuda_execution_owner_graph_range_lab(id, &raw mut lo, &raw mut hi)
    };
    match rc {
        1 => Ok(Some((lo, hi))),
        0 => Ok(None),
        other => Err(-other),
    }
}

pub use crate::execution::{set_forward_demand, set_projection_reference};

// Keep this numerical control inside one target verification forward. It is
// owner-local and process-configured; ordinary Prefill and M1 stay unchanged.
unsafe extern "C" {
    fn imparo_cuda_verification_m1_quant_exchange_lab(enabled: i32) -> i32;
}

pub struct VerificationM1Quant {
    previous: i32,
    _serialized: std::marker::PhantomData<*mut ()>,
}
impl VerificationM1Quant {
    /// # Safety
    /// Own serialized access to the selected native owner until this guard drops.
    /// Do not switch owners or change this policy for an already captured graph.
    pub unsafe fn configured() -> Option<Self> {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED.get_or_init(|| {
            std::env::var("IMPARO_LAB_VERIFY_M1_QUANT").as_deref() == Ok("1")
        }) {
            return None;
        }
        let previous = unsafe { imparo_cuda_verification_m1_quant_exchange_lab(1) };
        Some(Self {
            previous,
            _serialized: std::marker::PhantomData,
        })
    }
}
impl Drop for VerificationM1Quant {
    fn drop(&mut self) {
        unsafe {
            imparo_cuda_verification_m1_quant_exchange_lab(self.previous);
        }
    }
}
