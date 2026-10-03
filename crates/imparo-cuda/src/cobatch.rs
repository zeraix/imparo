//! Additive CUDA CoBatch bridge. Dynamic plugin admission follows the native gates.
use imparo_backend::SlotRow;
#[path = "cobatch_api.rs"]
pub(crate) mod native;
use native::*;
pub(crate) fn available() -> bool {
    native::available()
}

#[repr(C)]
pub(crate) struct SlotRowWire {
    slot: u32,
    pos: u32,
}
fn wire(rows: &[SlotRow]) -> Vec<SlotRowWire> {
    rows.iter()
        .map(|r| SlotRowWire {
            slot: r.slot,
            pos: r.pos,
        })
        .collect()
}

pub(crate) fn route(route: Option<imparo_backend::RowRoute>) -> bool {
    let value = match route {
        None => 0,
        Some(imparo_backend::RowRoute::Fast) => 1,
        _ => return false,
    };
    unsafe { imparo_cuda_cobatch_route(value) == 0 }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn head(
    buf: u32,
    weight: u64,
    hd: u32,
    eps: f32,
    heads: u32,
    pos: &[u32],
    rd: u32,
    base: f32,
    freqs: Option<&[f32]>,
    nrot: u32,
) -> bool {
    if pos.is_empty()
        || pos.len() > 64
        || freqs.is_some_and(|f| f.len() < (rd / 2) as usize)
    {
        return false;
    }
    unsafe {
        imparo_cuda_cobatch_head(
            buf,
            weight,
            hd,
            eps,
            heads,
            pos.as_ptr(),
            pos.len() as u32,
            rd,
            base,
            freqs.map_or(core::ptr::null(), |f| f.as_ptr()),
            nrot,
        ) == 0
    }
}
pub(crate) fn store(
    src: u32,
    layer: u32,
    width: u32,
    rows: &[SlotRow],
    is_v: bool,
    ring: u32,
) -> bool {
    if rows.is_empty() || rows.len() > 64 {
        return false;
    }
    let rows = wire(rows);
    unsafe {
        imparo_cuda_cobatch_store(
            src,
            layer,
            width,
            rows.as_ptr(),
            rows.len() as u32,
            u32::from(is_v),
            ring,
        ) == 0
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    layer: u32,
    hd: u32,
    heads: u32,
    kvheads: u32,
    width: u32,
    scale: f32,
    window: u32,
    rows: &[SlotRow],
    ring: u32,
) -> bool {
    if rows.is_empty() || rows.len() > 64 {
        return false;
    }
    let rows = wire(rows);
    unsafe {
        imparo_cuda_cobatch_attention(
            layer,
            hd,
            heads,
            kvheads,
            width,
            scale,
            window,
            rows.as_ptr(),
            rows.len() as u32,
            ring,
        ) == 0
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn kv_head(
    k: u32,
    v: u32,
    w: u64,
    hd: u32,
    eps: f32,
    heads: u32,
    pos: &[u32],
    rd: u32,
    base: f32,
    freqs: Option<&[f32]>,
    hk: u32,
    hv: u32,
) -> bool {
    if pos.is_empty()
        || pos.len() > 64
        || freqs.is_some_and(|f| f.len() < (rd / 2) as usize)
    {
        return false;
    }
    unsafe {
        imparo_cuda_cobatch_kv_head(
            k,
            v,
            w,
            hd,
            eps,
            heads,
            pos.as_ptr(),
            pos.len() as u32,
            rd,
            base,
            freqs.map_or(core::ptr::null(), |f| f.as_ptr()),
            hk,
            hv,
        ) == 0
    }
}

#[repr(C)]
pub(crate) struct SlotStateRowWire {
    slot: u32,
    state_off: u32,
    state_out_off: u32,
}

fn state_wire(rows: &[imparo_backend::SlotStateRow]) -> Vec<SlotStateRowWire> {
    rows.iter()
        .map(|r| SlotStateRowWire {
            slot: r.slot,
            state_off: r.state_off,
            state_out_off: r.state_out_off,
        })
        .collect()
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv(
    form: imparo_backend::ConvForm,
    src: u32,
    weight: u64,
    state: u32,
    rows: &[imparo_backend::SlotStateRow],
    out: u32,
    width: u32,
    kernel: u32,
) -> bool {
    if rows.is_empty() || rows.len() > 64 {
        return false;
    }
    let rows = state_wire(rows);
    unsafe {
        imparo_cuda_cobatch_conv(
            form as u32,
            src,
            weight,
            state,
            rows.as_ptr(),
            rows.len() as u32,
            out,
            width,
            kernel,
        ) == 0
    }
}
pub(crate) fn delta(
    op: &imparo_backend::DeltaNet,
    rows: &[imparo_backend::SlotStateRow],
) -> bool {
    if rows.is_empty() || rows.len() > 64 || op.epilogue.is_some() {
        return false;
    }
    let wire = crate::ffi::GatedDeltaWire {
        a_off: op.a_off,
        dt_off: op.dt_bias_off,
        qkv: op.qkv as u32,
        alpha: op.alpha as u32,
        beta: op.beta as u32,
        state: op.state as u32,
        state_off: 0,
        state_out_off: 0,
        out: op.out as u32,
        snap: u32::MAX,
        snap_off: 0,
        snap_row: 0,
        k_heads: op.k_heads,
        v_heads: op.v_heads,
        key_dim: op.key_dim,
        value_dim: op.value_dim,
        n_tok: 1,
        eps: op.eps,
    };
    let rows = state_wire(rows);
    unsafe { imparo_cuda_cobatch_delta(&wire, rows.as_ptr(), rows.len() as u32) == 0 }
}

#[cfg(test)]
mod wire_tests {
    #[test]
    fn extension_rows_have_fixed_c_layout() {
        use super::*;
        assert_eq!(std::mem::size_of::<SlotRowWire>(),8);
        assert_eq!(std::mem::offset_of!(SlotRowWire,pos),4);
        assert_eq!(std::mem::size_of::<SlotStateRowWire>(),12);
        assert_eq!(std::mem::offset_of!(SlotStateRowWire,state_out_off),8);
        assert_eq!(std::mem::size_of::<crate::ffi::GatedDeltaWire>(),80);
        assert_eq!(std::mem::offset_of!(crate::ffi::GatedDeltaWire,eps),76);
    }
}
