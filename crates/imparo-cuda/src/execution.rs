//! Private static execution geometry for speculative inference.

unsafe extern "C" {
    fn imparo_cuda_prepare_batch_invariant_q8_v1() -> i32;
}
/// Select the fixed target Q8 numerical family once, before any target forward.
/// This static bridge changes neither batch geometry nor output/state demand.
///
/// # Safety
/// Hold serialized access to a freshly prepared target owner with no live Graph
/// leases or numerical cache. Never call while the DSpark draft owner is selected.
pub unsafe fn prepare_batch_invariant_q8_v1() -> Result<(), i32> {
    let rc = unsafe { imparo_cuda_prepare_batch_invariant_q8_v1() };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

unsafe extern "C" {
    fn imparo_cuda_execution_work_demand_lab(
        ffn_layers: u32,
        logits_wanted: u32,
    ) -> i32;
}
/// Declare the complete FFN work expected by the selected output demand.
/// This invalidates a Prefill graph when output demand changes. Full-model packed
/// weight preparation remains atomic; no dynamic/public backend ABI is added.
///
/// # Safety
/// Exclusive access to a prepared, unleased owner between closed forwards. The
/// count must describe the actual required model work; full output requires all
/// model layers. Pair every subsequent forward with its actual demand.
pub unsafe fn set_forward_demand(
    ffn_layers: u32,
    logits_wanted: bool,
) -> Result<(), i32> {
    let rc = unsafe {
        imparo_cuda_execution_work_demand_lab(ffn_layers, u32::from(logits_wanted))
    };
    if rc == 0 { Ok(()) } else { Err(rc) }
}

unsafe extern "C" {
    fn imparo_cuda_projection_reference_lab(start: u32, tokens: u32) -> i32;
}
/// # Safety
/// Hold exclusive ownership inside an eager non-decode forward. Reference extent
/// must describe the known cold output tail, not invented active work.
pub unsafe fn set_projection_reference(start: u32, tokens: u32) -> Result<(), i32> {
    let rc = unsafe { imparo_cuda_projection_reference_lab(start, tokens) };
    if rc == 0 { Ok(()) } else { Err(rc) }
}
