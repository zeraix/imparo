//! CUDA launch parameters consumed by native kernels.
//!
//! A parameter belongs here only after the native launch reads the corresponding
//! slot. Keeping the slot constants beside the declarations makes the stored tuner
//! configuration an auditable contract instead of a second, implicit knob table.

use crate::ffi::{imparo_cuda_knob, imparo_cuda_set_knob};
use imparo_backend::{
    BackendKnobs, KnobCategory, KnobDecl, SweepKind as Sw, Workload as Wl,
};

const SLOT_GEMV_WARPS: u32 = 6;
const SLOT_ATTN_THREADS: u32 = 7;
const SLOT_RMS_THREADS: u32 = 8;
// Slots 9 and 10 are intentionally unassigned. Never reuse them without bumping the
// CUDA space version: persisted host configs address knobs by name, while native code
// addresses the same choices by slot.
const SLOT_MMQ_FULL_ROWS: u32 = 11;
const SLOT_MMQ_FULL_TILE_MIN_EFFICIENCY: u32 = 12;
const SLOT_ATTN_D256_WORKSPACE_MIB: u32 = 13;
const SLOT_ATTN_D512_WORKSPACE_MIB: u32 = 14;
const SLOT_ATTN_SCORE_KEY_GROUPS: u32 = 15;
const SLOT_ATTN_SOFTMAX_WARPS: u32 = 16;
const SLOT_ATTN_VALUE_TILES: u32 = 17;
const SLOT_ATTN_STREAM_PART_CAP: u32 = 18;
const SLOT_NARROW_GEMV_WARPS: u32 = 19;
const SLOT_NARROW_GEMV_MAX_WIDTH: u32 = 20;
const SLOT_NARROW_GEMV_ROWS_PER_CTA: u32 = 21;
const SLOT_BATCH_MMVQ_ROWS_PER_CTA: u32 = 22;
const SLOT_D512_MMA_MIN_SCHEDULE: u32 = 23;
const SLOT_STREAMK_NUMERIC: u32 = 24;
const SLOT_MMQ_LLAMA_COMPAT: u32 = 25;
const SLOT_MMQ_VIRTUAL_512: u32 = 26;
const SLOT_MMQ_CANONICAL_FULL_TILE: u32 = 27;
const SLOT_RMS_NORM_ADD: u32 = 28;
const SLOT_ATTN_D256_TILED: u32 = 29;
const SLOT_ATTN_D256_VEC: u32 = 30;
const SLOT_ATTN_D256_VIRTUAL_STREAM: u32 = 31;
const SLOT_ATTN_D256_FUSED: u32 = 32;
const SLOT_ATTN_D512_MMA: u32 = 33;
const SLOT_ATTN_D512_VIRTUAL_STREAM: u32 = 34;
const SLOT_ATTN_D512_VIRTUAL_CELL_POLICY: u32 = 35;
const SLOT_ATTN_DECODE_SPECIALIZED: u32 = 36;
const SLOT_ATTN_D64_MMA_PREFILL: u32 = 37;
const SLOT_FFN_SIDECAR_MIN_TOKENS: u32 = 38;
const SLOT_PREFILL_EXACT128_SM86_ROUTE: u32 = 39;
const SLOT_PREFILL_EXACT128_GRAPH: u32 = 40;
const SLOT_ATTN_D64_Q8_VEC: u32 = 41;
const SLOT_MMQ_Q8_ALIGNED_WHOLE_K: u32 = 42;
const SLOT_MMQ_Q8_TM_SILU_PAIR: u32 = 43;
const SLOT_MMQ_Q8_TM_SILU_Q8_SIDECAR_MIN_TOKENS: u32 = 44;
const SLOT_MMQ_Q8_TM_SILU_PRIVATE_DOWN_MIN_TOKENS: u32 = 45;
const SLOT_ATTN_D64_MMA_SHARED_KV_MIN_TOKENS: u32 = 46;
const SLOT_MMQ_Q8_TM_ASYNC_WEIGHT_STAGE: u32 = 47;
const SLOT_MMQ_Q8_TM_GATE_UP_ROW_PAIR_MIN_TOKENS: u32 = 48;
const SLOT_ATTN_D64_Q8_GQA4: u32 = 49;
const SLOT_MMQ_Q8_TM_DECODE_SILU_PAIR: u32 = 50;
const SLOT_SHORTCONV_DECODE_FUSED: u32 = 51;
const SLOT_DECODE_GRAPH_Q8_PRODUCER_REUSE: u32 = 52;
const SLOT_DECODE_FFN_Q5_LAYER_MASK: u32 = 53;
const SLOT_DECODE_DOWN_Q4_LAYER_MASK: u32 = 54;
const SLOT_DECODE_DOWN_Q5_LAYER_MASK: u32 = 55;
const SLOT_PREFILL_PROJECTION_Q8_D4: u32 = 56;
const SLOT_PREFILL_DOWN_Q4_LAYER_MASK: u32 = 57;
const SLOT_ROW_LOCAL_PREFILL_TAIL_ROWS: u32 = 58;
const SLOT_PREFILL_HEAD_POST_THREADS: u32 = 59;

pub(crate) fn slot_for_name(name: &str) -> Option<u32> {
    CUDA_KNOBS
        .iter()
        .position(|decl| decl.name == name)
        .and_then(|registry_index| {
            const SLOTS: &[u32] = &[
                SLOT_GEMV_WARPS,
                SLOT_ATTN_THREADS,
                SLOT_RMS_THREADS,
                SLOT_MMQ_FULL_ROWS,
                SLOT_MMQ_FULL_TILE_MIN_EFFICIENCY,
                SLOT_ATTN_D256_WORKSPACE_MIB,
                SLOT_ATTN_D512_WORKSPACE_MIB,
                SLOT_ATTN_SCORE_KEY_GROUPS,
                SLOT_ATTN_SOFTMAX_WARPS,
                SLOT_ATTN_VALUE_TILES,
                SLOT_ATTN_STREAM_PART_CAP,
                SLOT_NARROW_GEMV_WARPS,
                SLOT_NARROW_GEMV_ROWS_PER_CTA,
                SLOT_NARROW_GEMV_MAX_WIDTH,
                SLOT_BATCH_MMVQ_ROWS_PER_CTA,
                SLOT_D512_MMA_MIN_SCHEDULE,
                SLOT_STREAMK_NUMERIC,
                SLOT_MMQ_LLAMA_COMPAT,
                SLOT_MMQ_VIRTUAL_512,
                SLOT_MMQ_CANONICAL_FULL_TILE,
                SLOT_RMS_NORM_ADD,
                SLOT_ATTN_D256_TILED,
                SLOT_ATTN_D256_VEC,
                SLOT_ATTN_D256_VIRTUAL_STREAM,
                SLOT_ATTN_D256_FUSED,
                SLOT_ATTN_D512_MMA,
                SLOT_ATTN_D512_VIRTUAL_STREAM,
                SLOT_ATTN_D512_VIRTUAL_CELL_POLICY,
                SLOT_ATTN_DECODE_SPECIALIZED,
                SLOT_ATTN_D64_MMA_PREFILL,
                SLOT_ATTN_D64_MMA_SHARED_KV_MIN_TOKENS,
                SLOT_FFN_SIDECAR_MIN_TOKENS,
                SLOT_PREFILL_EXACT128_SM86_ROUTE,
                SLOT_PREFILL_EXACT128_GRAPH,
                SLOT_ATTN_D64_Q8_VEC,
                SLOT_ATTN_D64_Q8_GQA4,
                SLOT_MMQ_Q8_TM_DECODE_SILU_PAIR,
                SLOT_SHORTCONV_DECODE_FUSED,
                SLOT_DECODE_GRAPH_Q8_PRODUCER_REUSE,
                SLOT_MMQ_Q8_ALIGNED_WHOLE_K,
                SLOT_MMQ_Q8_TM_ASYNC_WEIGHT_STAGE,
                SLOT_MMQ_Q8_TM_SILU_PAIR,
                SLOT_MMQ_Q8_TM_SILU_Q8_SIDECAR_MIN_TOKENS,
                SLOT_MMQ_Q8_TM_SILU_PRIVATE_DOWN_MIN_TOKENS,
                SLOT_MMQ_Q8_TM_GATE_UP_ROW_PAIR_MIN_TOKENS,
                SLOT_DECODE_FFN_Q5_LAYER_MASK,
                SLOT_DECODE_DOWN_Q4_LAYER_MASK,
                SLOT_DECODE_DOWN_Q5_LAYER_MASK,
                SLOT_PREFILL_PROJECTION_Q8_D4,
                SLOT_PREFILL_DOWN_Q4_LAYER_MASK,
                SLOT_ROW_LOCAL_PREFILL_TAIL_ROWS,
                SLOT_PREFILL_HEAD_POST_THREADS,
            ];
            SLOTS.get(registry_index).copied()
        })
}

const CUDA_SPACE_VERSION: u32 = 49;
const D512_SCHEDULE_KEYS: u32 = 256;
const D512_MMA_CONSERVATIVE_FLOOR: u32 = 4 * D512_SCHEDULE_KEYS;
const D512_MMA_SPAN_LADDER: &[u32] =
    &[1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384];
// Exact 128 is reserved for the atomic route below. Keeping the general
// threshold at 129+ prevents a scalar boundary from selecting the FFN half of
// an exact-shape bundle without its PLE and Down halves.
const FFN_SIDECAR_TOKEN_LADDER: &[u32] =
    &[129, 192, 255, 256, 257, 320, 384, 448, 449, 511, 512];
const ATTN_D64_SHARED_KV_TOKEN_LADDER: &[u32] =
    &[129, 192, 256, 384, 449, 512, 1024, 2048, 4096];

macro_rules! slot {
    ($idx:expr) => {
        (
            |v| unsafe { imparo_cuda_set_knob($idx, v) },
            || unsafe { imparo_cuda_knob($idx) },
        )
    };
}

#[must_use]
pub(crate) fn rms_norm_add_enabled() -> bool {
    unsafe { imparo_cuda_knob(SLOT_RMS_NORM_ADD) != 0 }
}

#[must_use]
pub(crate) fn ffn_sidecar_min_tokens() -> u32 {
    unsafe { imparo_cuda_knob(SLOT_FFN_SIDECAR_MIN_TOKENS) }
}

#[must_use]
pub(crate) fn q8_tm_silu_pair_enabled() -> bool {
    unsafe { imparo_cuda_knob(SLOT_MMQ_Q8_TM_SILU_PAIR) != 0 }
}

#[must_use]
pub(crate) fn q8_tm_decode_silu_pair_enabled() -> bool {
    unsafe { imparo_cuda_knob(SLOT_MMQ_Q8_TM_DECODE_SILU_PAIR) != 0 }
}

#[must_use]
pub(crate) fn decode_graph_q8_producer_reuse_enabled() -> bool {
    unsafe { imparo_cuda_knob(SLOT_DECODE_GRAPH_Q8_PRODUCER_REUSE) != 0 }
}

#[must_use]
pub(crate) fn decode_ffn_q5_layer_mask() -> u32 {
    unsafe { imparo_cuda_knob(SLOT_DECODE_FFN_Q5_LAYER_MASK) }
}

#[must_use]
pub(crate) fn decode_shadow_cache_enabled() -> bool {
    unsafe {
        imparo_cuda_knob(SLOT_DECODE_FFN_Q5_LAYER_MASK) != 0
            || imparo_cuda_knob(SLOT_DECODE_DOWN_Q4_LAYER_MASK) != 0
            || imparo_cuda_knob(SLOT_DECODE_DOWN_Q5_LAYER_MASK) != 0
    }
}

#[must_use]
pub(crate) fn prefill_projection_q8_d4_enabled() -> bool {
    unsafe { imparo_cuda_knob(SLOT_PREFILL_PROJECTION_Q8_D4) != 0 }
}

#[must_use]
pub(crate) fn prefill_projection_q8_d4_mode() -> u32 {
    unsafe { imparo_cuda_knob(SLOT_PREFILL_PROJECTION_Q8_D4) }
}

#[must_use]
pub(crate) fn prefill_down_q4_layer_mask() -> u32 {
    unsafe { imparo_cuda_knob(SLOT_PREFILL_DOWN_Q4_LAYER_MASK) }
}

#[must_use]
pub(crate) fn row_local_prefill_tail_rows() -> u32 {
    match unsafe { imparo_cuda_knob(SLOT_ROW_LOCAL_PREFILL_TAIL_ROWS) } {
        1 | 4 | 8 | 16 | 32 | 64 => unsafe {
            imparo_cuda_knob(SLOT_ROW_LOCAL_PREFILL_TAIL_ROWS)
        },
        _ => 64,
    }
}

#[must_use]
pub(crate) fn q8_tm_silu_private_down_min_tokens() -> u32 {
    unsafe { imparo_cuda_knob(SLOT_MMQ_Q8_TM_SILU_PRIVATE_DOWN_MIN_TOKENS) }
}

#[must_use]
pub(crate) fn prefill_exact128_sm86_route_enabled() -> bool {
    matches!(
        unsafe { imparo_cuda_knob(SLOT_PREFILL_EXACT128_SM86_ROUTE) },
        1 | 2
    )
}

#[must_use]
pub(crate) fn prefill_exact128_fast_transaction_enabled() -> bool {
    matches!(
        unsafe { imparo_cuda_knob(SLOT_PREFILL_EXACT128_SM86_ROUTE) },
        4 | 5
    )
}

#[must_use]
pub(crate) fn prefill_exact128_graph_enabled() -> bool {
    unsafe { imparo_cuda_knob(SLOT_PREFILL_EXACT128_GRAPH) == 1 }
}

macro_rules! value_knob {
    (
        $name:literal, $slot:expr, $values:expr, $screened:expr, $workload:expr,
        bit_affecting = $bit_affecting:expr
        $(, applies = $applies:expr)?
        $(, tuple = $tuple:literal)?
    ) => {
        KnobDecl {
            name: $name,
            legal: None,
            bit_affecting: $bit_affecting,
            derive: None,
            candidates: None,
            after: &[],
            cross_check: None,
            applies: value_knob!(@optional $( $applies )?),
            tuple: value_knob!(@tuple $( $tuple )?),
            category: KnobCategory::Benched,
            values: $values,
            apply: slot!($slot).0,
            current: slot!($slot).1,
            screened: $screened,
            sweep: Sw::Values,
            workload: $workload,
        }
    };
    (@optional $value:expr) => { Some($value) };
    (@optional) => { None };
    (@tuple $value:literal) => { Some($value) };
    (@tuple) => { None };
}

/// All native CUDA choices that the shared tuner may persist.
///
/// Registry order is sweep order. Coupled MMQ and narrow-MMVQ choices therefore sit
/// together, with routing decisions measured after the shapes they route between.
pub static CUDA_KNOBS: &[KnobDecl] = &[
    value_knob!(
        "gemv_warps",
        SLOT_GEMV_WARPS,
        &[0, 2, 4, 8],
        false,
        Wl::DecodeMix,
        bit_affecting = true,
        tuple = "decode_mmvq"
    ),
    value_knob!(
        "attn_threads",
        SLOT_ATTN_THREADS,
        &[0, 64, 128, 256],
        true,
        Wl::AttentionDecode,
        bit_affecting = true
    ),
    value_knob!(
        "rms_threads",
        SLOT_RMS_THREADS,
        &[0, 256, 1024],
        true,
        Wl::DecodeMix,
        bit_affecting = true
    ),
    value_knob!(
        "mmq_full_rows",
        SLOT_MMQ_FULL_ROWS,
        &[0, 64, 128],
        true,
        Wl::PrefillGemm,
        bit_affecting = true,
        tuple = "prefill_mmq"
    ),
    value_knob!(
        "mmq_full_tile_min_efficiency",
        SLOT_MMQ_FULL_TILE_MIN_EFFICIENCY,
        &[0, 50, 70, 80, 85, 90, 95],
        true,
        Wl::PrefillGemm,
        bit_affecting = true,
        tuple = "prefill_mmq"
    ),
    value_knob!(
        "attn_d256_workspace_mib",
        SLOT_ATTN_D256_WORKSPACE_MIB,
        &[0, 8, 12, 16, 20, 24, 32, 64],
        false,
        Wl::AttentionPrefillDeep,
        bit_affecting = true,
        applies = |m| m.head_dim == 256
    ),
    value_knob!(
        "attn_d512_workspace_mib",
        SLOT_ATTN_D512_WORKSPACE_MIB,
        &[0, 8, 12, 16, 20, 24, 32, 64],
        false,
        Wl::AttentionPrefillDeep,
        bit_affecting = true,
        applies = |m| m.deep_head_dim == 512
    ),
    value_knob!(
        "attn_score_key_groups",
        SLOT_ATTN_SCORE_KEY_GROUPS,
        &[0, 1, 2, 4],
        true,
        Wl::AttentionPrefillDeep,
        bit_affecting = true
    ),
    value_knob!(
        "attn_softmax_warps",
        SLOT_ATTN_SOFTMAX_WARPS,
        &[0, 1, 2, 4, 8],
        true,
        Wl::AttentionDecode,
        bit_affecting = true
    ),
    value_knob!(
        "attn_value_tiles",
        SLOT_ATTN_VALUE_TILES,
        &[0, 1, 2],
        true,
        Wl::AttentionDecode,
        bit_affecting = false,
        applies = |m| m.head_dim != 64
    ),
    value_knob!(
        "attn_stream_part_cap",
        SLOT_ATTN_STREAM_PART_CAP,
        &[0, 4, 8, 12, 16, 24, 32],
        false,
        Wl::AttentionDecodeDeep,
        bit_affecting = true
    ),
    value_knob!(
        "narrow_gemv_warps",
        SLOT_NARROW_GEMV_WARPS,
        &[0, 2, 4, 8],
        true,
        Wl::DecodeMix,
        bit_affecting = true,
        tuple = "decode_mmvq"
    ),
    value_knob!(
        "narrow_gemv_rows_per_cta",
        SLOT_NARROW_GEMV_ROWS_PER_CTA,
        &[0, 1, 2, 4],
        true,
        Wl::DecodeMix,
        bit_affecting = false,
        applies = |m| m.weight_kinds & (1 << 1) != 0,
        tuple = "decode_mmvq"
    ),
    value_knob!(
        "narrow_gemv_max_width",
        SLOT_NARROW_GEMV_MAX_WIDTH,
        &[0, 256, 512, 1024],
        false,
        Wl::DecodeMix,
        bit_affecting = true,
        tuple = "decode_mmvq"
    ),
    value_knob!(
        "batch_mmvq_rows_per_cta",
        SLOT_BATCH_MMVQ_ROWS_PER_CTA,
        &[0, 1, 2, 4],
        false,
        Wl::NarrowMix(4),
        bit_affecting = false,
        applies = |m| m.weight_kinds & (1 << 1) != 0
    ),
    KnobDecl {
        name: "d512_mma_min_schedule",
        legal: None,
        // This threshold selects a different reduction route. It must stay at the
        // compiled correctness floor unless the caller explicitly opts into tuning
        // bit-affecting choices and then re-runs the numerical gates.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["attn_stream_part_cap"],
        cross_check: None,
        applies: Some(|m| m.deep_head_dim == 512),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: slot!(SLOT_D512_MMA_MIN_SCHEDULE).0,
        current: slot!(SLOT_D512_MMA_MIN_SCHEDULE).1,
        screened: false,
        sweep: Sw::SpanCrossing {
            ladder: D512_MMA_SPAN_LADDER,
            // Never ask the tuner to enable the MMA route below the verified 1024-key
            // floor. UINT32_MAX is the conservative direct-Q4 leg.
            hi: D512_MMA_CONSERVATIVE_FLOOR,
            lo: u32::MAX,
        },
        workload: Wl::AttentionDecodeDeep,
    },
    value_knob!(
        "streamk_numeric",
        SLOT_STREAMK_NUMERIC,
        &[0, 1],
        false,
        Wl::PrefillGemm,
        bit_affecting = true,
        tuple = "prefill_mmq"
    ),
    value_knob!(
        "mmq_llama_compat",
        SLOT_MMQ_LLAMA_COMPAT,
        &[0, 1],
        false,
        Wl::PrefillGemm,
        bit_affecting = true,
        tuple = "prefill_mmq"
    ),
    value_knob!(
        "mmq_virtual_512",
        SLOT_MMQ_VIRTUAL_512,
        &[0, 1],
        false,
        Wl::PrefillGemm,
        bit_affecting = true,
        tuple = "prefill_mmq"
    ),
    value_knob!(
        "mmq_canonical_full_tile",
        SLOT_MMQ_CANONICAL_FULL_TILE,
        &[0, 1],
        false,
        Wl::PrefillGemm,
        bit_affecting = true,
        tuple = "prefill_mmq"
    ),
    value_knob!(
        "rms_norm_add",
        SLOT_RMS_NORM_ADD,
        &[0, 1],
        false,
        Wl::DecodeMix,
        bit_affecting = true
    ),
    value_knob!(
        "attn_d256_tiled",
        SLOT_ATTN_D256_TILED,
        &[0, 1],
        false,
        Wl::AttentionPrefillDeep,
        bit_affecting = true,
        applies = |m| m.head_dim == 256
    ),
    value_knob!(
        "attn_d256_vec",
        SLOT_ATTN_D256_VEC,
        &[0, 1],
        false,
        Wl::AttentionDecodeDeep,
        bit_affecting = true,
        applies = |m| m.head_dim == 256
    ),
    value_knob!(
        "attn_d256_virtual_stream",
        SLOT_ATTN_D256_VIRTUAL_STREAM,
        &[0, 1],
        false,
        Wl::AttentionPrefillDeep,
        bit_affecting = true,
        applies = |m| m.head_dim == 256
    ),
    value_knob!(
        "attn_d256_fused",
        SLOT_ATTN_D256_FUSED,
        &[0, 1],
        false,
        Wl::AttentionPrefillDeep,
        bit_affecting = true,
        applies = |m| m.head_dim == 256
    ),
    value_knob!(
        "attn_d512_mma",
        SLOT_ATTN_D512_MMA,
        &[0, 1],
        false,
        Wl::AttentionDecodeDeep,
        bit_affecting = true,
        applies = |m| m.deep_head_dim == 512
    ),
    value_knob!(
        "attn_d512_virtual_stream",
        SLOT_ATTN_D512_VIRTUAL_STREAM,
        &[0, 1],
        false,
        Wl::AttentionPrefillDeep,
        bit_affecting = true,
        applies = |m| m.deep_head_dim == 512
    ),
    value_knob!(
        "attn_d512_virtual_cell_policy",
        SLOT_ATTN_D512_VIRTUAL_CELL_POLICY,
        &[0, 1],
        false,
        Wl::AttentionPrefillDeep,
        bit_affecting = true,
        applies = |m| m.deep_head_dim == 512
    ),
    value_knob!(
        "attn_decode_specialized",
        SLOT_ATTN_DECODE_SPECIALIZED,
        &[0, 1],
        false,
        Wl::AttentionDecodeDeep,
        bit_affecting = true
    ),
    value_knob!(
        "attn_d64_mma_prefill",
        SLOT_ATTN_D64_MMA_PREFILL,
        &[0, 1],
        false,
        Wl::AttentionPrefillDeep,
        bit_affecting = true,
        applies =
            |m| { m.deep_head_dim == 64 && m.n_kv != 0 && m.n_head == 4 * m.n_kv }
    ),
    KnobDecl {
        name: "attn_d64_mma_shared_kv_min_tokens",
        legal: None,
        // The candidate preserves the whole-K MMA/softmax order but changes the
        // physical staging graph. Keep its promotion independently receipt-bound.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["attn_d64_mma_prefill"],
        cross_check: None,
        applies: Some(|m| {
            m.deep_head_dim == 64 && m.n_kv != 0 && m.n_head == 4 * m.n_kv
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: slot!(SLOT_ATTN_D64_MMA_SHARED_KV_MIN_TOKENS).0,
        current: slot!(SLOT_ATTN_D64_MMA_SHARED_KV_MIN_TOKENS).1,
        screened: false,
        sweep: Sw::TokenMinCrossing {
            ladder: ATTN_D64_SHARED_KV_TOKEN_LADDER,
            hi: 1,
            lo: 0,
        },
        workload: Wl::AttentionPrefillDeep,
    },
    KnobDecl {
        name: "ffn_sidecar_min_tokens",
        legal: None,
        // The sidecar changes the quantization/fusion route. A timing pick is never
        // authority by itself: the tuner requires --allow-bit-changes and the exact
        // resulting config still needs the fixed correctness receipt.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["mmq_full_tile_min_efficiency", "mmq_llama_compat"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd.checked_mul(4) == Some(m.n_ff)
                && m.weight_kinds & (1 << 1) != 0
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[],
        apply: slot!(SLOT_FFN_SIDECAR_MIN_TOKENS).0,
        current: slot!(SLOT_FFN_SIDECAR_MIN_TOKENS).1,
        screened: false,
        sweep: Sw::TokenMinCrossing {
            ladder: FFN_SIDECAR_TOKEN_LADDER,
            // 1 forces the candidate at every measured rung; zero disables it. The
            // winning ladder value, not either force value, is persisted.
            hi: 1,
            lo: 0,
        },
        workload: Wl::PrefillFfnTransaction,
    },
    KnobDecl {
        name: "prefill_exact128_sm86_route",
        legal: None,
        // The micro transaction remains a diagnostic, not admission authority:
        // on the RTX 3060 it ranked value 1 ahead of safe-off while a real E4B
        // bracket measured value 1 slower. Only a whole-engine bracket plus the
        // numerical receipt may persist either nonzero route.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd == 2560
                && m.n_ff == 10240
                && m.weight_kinds & (1 << 1) != 0
        }),
        tuple: None,
        category: KnobCategory::EndToEnd,
        values: &[0, 1, 2, 3, 4, 5],
        apply: slot!(SLOT_PREFILL_EXACT128_SM86_ROUTE).0,
        current: slot!(SLOT_PREFILL_EXACT128_SM86_ROUTE).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::PrefillFfnExact128,
    },
    KnobDecl {
        name: "prefill_exact128_graph",
        legal: None,
        // Scheduling-only: the fixed replay gate must prove byte equality.
        // Zero remains the architecture-independent safe default.
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &["prefill_exact128_sm86_route"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd == 2560
                && m.n_ff == 10240
                && m.weight_kinds & (1 << 1) != 0
        }),
        tuple: None,
        category: KnobCategory::EndToEnd,
        values: &[0, 1],
        apply: slot!(SLOT_PREFILL_EXACT128_GRAPH).0,
        current: slot!(SLOT_PREFILL_EXACT128_GRAPH).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::PrefillFfnExact128,
    },
    value_knob!(
        "attn_d64_q8_vec",
        SLOT_ATTN_D64_Q8_VEC,
        &[0, 1],
        false,
        Wl::AttentionDecodeDeep,
        bit_affecting = true,
        applies =
            |m| { m.deep_head_dim == 64 && m.n_kv != 0 && m.n_head == 4 * m.n_kv }
    ),
    KnobDecl {
        name: "attn_d64_q8_gqa4",
        legal: None,
        // Grouped ownership changes the partition reduction graph even though it
        // preserves the Q8 codec. Keep it independent, default-off and receipt-bound.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["attn_d64_q8_vec"],
        cross_check: None,
        applies: Some(|m| {
            m.deep_head_dim == 64 && m.n_kv != 0 && m.n_head == 4 * m.n_kv
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 1],
        apply: slot!(SLOT_ATTN_D64_Q8_GQA4).0,
        current: slot!(SLOT_ATTN_D64_Q8_GQA4).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeAttentionStep,
    },
    KnobDecl {
        name: "mmvq_q8_tm_decode_silu_pair",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd % 32 == 0
                && m.n_ff % 8 == 0
                && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 1],
        apply: slot!(SLOT_MMQ_Q8_TM_DECODE_SILU_PAIR).0,
        current: slot!(SLOT_MMQ_Q8_TM_DECODE_SILU_PAIR).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeFfnTransaction,
    },
    KnobDecl {
        name: "shortconv_decode_fused",
        legal: None,
        // The fused implementation preserves the shortconv arithmetic and state layout.
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        // Availability is decided by the model-plan/GGUF-derived workload shape. It is
        // intentionally independent of CUDA SM and projection quantization.
        applies: None,
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 1],
        apply: slot!(SLOT_SHORTCONV_DECODE_FUSED).0,
        current: slot!(SLOT_SHORTCONV_DECODE_FUSED).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::DecodeShortconvTransaction,
    },
    KnobDecl {
        name: "decode_graph_q8_producer_reuse",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        applies: Some(|m| m.n_embd % 32 == 0 && m.weight_kinds & (1 << 3) != 0),
        tuple: None,
        category: KnobCategory::EndToEnd,
        values: &[0, 1],
        apply: slot!(SLOT_DECODE_GRAPH_Q8_PRODUCER_REUSE).0,
        current: slot!(SLOT_DECODE_GRAPH_Q8_PRODUCER_REUSE).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::DecodeMix,
    },
    value_knob!(
        "mmq_q8_aligned_whole_k",
        SLOT_MMQ_Q8_ALIGNED_WHOLE_K,
        &[0, 1],
        false,
        Wl::PrefillGemm,
        bit_affecting = false,
        applies = |m| m.weight_kinds & ((1 << 2) | (1 << 3)) != 0
    ),
    KnobDecl {
        name: "mmq_q8_tm_async_weight_stage",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &["mmq_q8_aligned_whole_k"],
        cross_check: None,
        applies: Some(|m| m.weight_kinds & (1 << 3) != 0),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 1, 2],
        apply: slot!(SLOT_MMQ_Q8_TM_ASYNC_WEIGHT_STAGE).0,
        current: slot!(SLOT_MMQ_Q8_TM_ASYNC_WEIGHT_STAGE).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::PrefillGemm,
    },
    KnobDecl {
        name: "mmq_q8_tm_silu_pair",
        legal: None,
        // Fusing SiLU and the gate product changes floating-point evaluation order.
        // Selection therefore requires the exact fixed correctness receipt.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["mmq_q8_aligned_whole_k", "mmq_q8_tm_async_weight_stage"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd % 256 == 0
                && m.n_ff % 128 == 0
                && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 1],
        apply: slot!(SLOT_MMQ_Q8_TM_SILU_PAIR).0,
        current: slot!(SLOT_MMQ_Q8_TM_SILU_PAIR).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::PrefillFfnTransaction,
    },
    KnobDecl {
        name: "mmq_q8_tm_silu_q8_sidecar_min_tokens",
        legal: None,
        // The producer must remain byte-identical to the established standalone
        // D4 quantizer. Keep the first selection receipt-bound until that claim is
        // proven by the fixed model gates as well as the sidecar contract test.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["mmq_q8_tm_silu_pair"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd % 256 == 0
                && m.n_ff % 128 == 0
                && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 9, 128, 129, 257, 513],
        apply: slot!(SLOT_MMQ_Q8_TM_SILU_Q8_SIDECAR_MIN_TOKENS).0,
        current: slot!(SLOT_MMQ_Q8_TM_SILU_Q8_SIDECAR_MIN_TOKENS).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::PrefillFfnTransaction,
    },
    KnobDecl {
        name: "mmq_q8_tm_silu_private_down_min_tokens",
        legal: None,
        // This complete transaction may leave its intermediate private. Keep it
        // receipt-bound and independently selectable from the dense v35 sidecar.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["mmq_q8_tm_silu_q8_sidecar_min_tokens"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd % 256 == 0
                && m.n_ff % 128 == 0
                && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 9, 128, 129, 257, 513],
        apply: slot!(SLOT_MMQ_Q8_TM_SILU_PRIVATE_DOWN_MIN_TOKENS).0,
        current: slot!(SLOT_MMQ_Q8_TM_SILU_PRIVATE_DOWN_MIN_TOKENS).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::PrefillFfnTransaction,
    },
    KnobDecl {
        name: "mmq_q8_tm_gate_up_row_pair_min_tokens",
        legal: None,
        // The paired producer changes both projection ownership and floating-point
        // evaluation order. Selection remains receipt-bound and independent from
        // the established private-Down transaction so it can fail closed.
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["mmq_q8_tm_silu_private_down_min_tokens"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd % 256 == 0
                && m.n_ff % 128 == 0
                && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 9, 128, 129, 257, 513],
        apply: slot!(SLOT_MMQ_Q8_TM_GATE_UP_ROW_PAIR_MIN_TOKENS).0,
        current: slot!(SLOT_MMQ_Q8_TM_GATE_UP_ROW_PAIR_MIN_TOKENS).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::PrefillFfnTransaction,
    },
    KnobDecl {
        name: "decode_ffn_q5_layer_mask",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["mmvq_q8_tm_decode_silu_pair"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0 && m.n_layers <= 32 && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: Some("decode_precision_map"),
        category: KnobCategory::EndToEnd,
        values: &[0, 0x3aaa_ffbf],
        apply: slot!(SLOT_DECODE_FFN_Q5_LAYER_MASK).0,
        current: slot!(SLOT_DECODE_FFN_Q5_LAYER_MASK).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::DecodeFfnTransaction,
    },
    KnobDecl {
        name: "decode_down_q4_layer_mask",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["decode_ffn_q5_layer_mask"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0 && m.n_layers <= 32 && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: Some("decode_precision_map"),
        category: KnobCategory::EndToEnd,
        values: &[0, 0x0200_5dda, 0x0200_5dd8],
        apply: slot!(SLOT_DECODE_DOWN_Q4_LAYER_MASK).0,
        current: slot!(SLOT_DECODE_DOWN_Q4_LAYER_MASK).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::DecodeFfnTransaction,
    },
    KnobDecl {
        name: "decode_down_q5_layer_mask",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["decode_down_q4_layer_mask"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0 && m.n_layers <= 32 && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: Some("decode_precision_map"),
        category: KnobCategory::EndToEnd,
        values: &[0, 0x1800_2225],
        apply: slot!(SLOT_DECODE_DOWN_Q5_LAYER_MASK).0,
        current: slot!(SLOT_DECODE_DOWN_Q5_LAYER_MASK).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::DecodeFfnTransaction,
    },
    KnobDecl {
        name: "prefill_projection_q8_d4",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["mmq_q8_aligned_whole_k"],
        cross_check: None,
        applies: Some(|m| m.n_embd % 32 == 0 && m.weight_kinds & (1 << 3) != 0),
        tuple: None,
        category: KnobCategory::EndToEnd,
        values: &[0, 1, 2],
        apply: slot!(SLOT_PREFILL_PROJECTION_Q8_D4).0,
        current: slot!(SLOT_PREFILL_PROJECTION_Q8_D4).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::PrefillFfnTransaction,
    },
    KnobDecl {
        name: "prefill_down_q4_layer_mask",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &["prefill_projection_q8_d4"],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0 && m.n_layers <= 32 && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: Some("prefill_precision_map"),
        category: KnobCategory::EndToEnd,
        values: &[0, 0x3fff_ffff],
        apply: slot!(SLOT_PREFILL_DOWN_Q4_LAYER_MASK).0,
        current: slot!(SLOT_PREFILL_DOWN_Q4_LAYER_MASK).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::PrefillFfnTransaction,
    },
    KnobDecl {
        name: "row_local_prefill_tail_rows",
        legal: None,
        bit_affecting: false,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        applies: Some(|m| {
            m.n_experts == 0
                && m.n_embd % 128 == 0
                && m.n_ff % 256 == 0
                && m.weight_kinds & (1 << 3) != 0
        }),
        tuple: None,
        category: KnobCategory::Benched,
        values: &[0, 1, 4, 8, 16, 32, 64],
        apply: slot!(SLOT_ROW_LOCAL_PREFILL_TAIL_ROWS).0,
        current: slot!(SLOT_ROW_LOCAL_PREFILL_TAIL_ROWS).1,
        screened: false,
        sweep: Sw::Values,
        workload: Wl::PrefillFfnTransaction,
    },
    KnobDecl {
        name: "prefill_head_post_threads",
        legal: None,
        bit_affecting: true,
        derive: None,
        candidates: None,
        after: &[],
        cross_check: None,
        applies: Some(|m| m.head_dim == 64),
        tuple: None,
        category: KnobCategory::EndToEnd,
        values: &[0, 64, 128, 256],
        apply: slot!(SLOT_PREFILL_HEAD_POST_THREADS).0,
        current: slot!(SLOT_PREFILL_HEAD_POST_THREADS).1,
        screened: false,
        sweep: Sw::External,
        workload: Wl::AttentionPrefill,
    },
];
impl BackendKnobs for crate::CudaBackend {
    fn knob_registry(&self) -> &'static [KnobDecl] {
        CUDA_KNOBS
    }

    /// CUDA versions independently of Metal. Version 20 declared the D512 MMA
    /// schedule boundary. Version 21 moves every accepted SM86 numerical route from
    /// process environment into the persisted registry. Version 22 binds the coupled
    /// SM86 decode route. Version 23 makes Q8 MMQ use one maximal token tile and a
    /// whole-K owner so recurrent state is invariant across prefill/resume boundaries.
    /// Version 24 receipts the SM86 D64/GQA4 MMA prefill route that closes the LFM2
    /// Q4-KV recurrent agreement gate. Version 25 adds the receipt-bound minimum-token
    /// boundary for the complete dense FFN sidecar transaction; zero remains safe-off.
    /// Version 26 reserves exact 128 for one atomic SM86 route controlling the FFN,
    /// PLE Direct-K/ready-Q8 and Down-R128 choices together.
    /// Version 27 binds exact-128 full-logits CUDA Graph replay to a safe-off knob.
    /// Production cannot widen beyond the already receipted exact-128 route, while
    /// the fixed suite proves replay is byte-identical to the ordinary DAG.
    /// Version 28 adds value 2 to the same atomic route: it retains value 1's
    /// Attention/FFN/PLE numerical bundle and splits only the exact-128 majority
    /// local-Q MMQ token tile from J128 to J64. Old receipts cannot prove that
    /// scheduling choice, so they fail closed until the complete v28 gate is sealed.
    /// Version 29 adds value 3 as an isolated SM86/exact-128 staged-F32 Attention
    /// candidate. Unlike values 1 and 2, it does not enable the FFN/PLE bundle.
    /// Value 4 evaluates staged-F32 Attention together with the established fast
    /// Sidecar/PLE/shape-hybrid MMQ transaction; it remains default-off.
    /// Version 30 adds value 5 for the same fast transaction without staged-F32
    /// Attention, allowing the fixed receipt to isolate combined-route drift.
    /// Version 31 adds the default-off SM86 D64/GQA4 direct-Q8 vector Decode
    /// Attention route. Its Q8 cache codec changes numerical ordering, so only a
    /// current-fingerprint correctness receipt may admit it.
    /// Version 32 adds a default-off aligned whole-K Q8 MMQ specialization. It
    /// removes replay, K-tail, and row-tail control only when the runtime proves
    /// the resident projection is fully aligned; the arithmetic loop and output
    /// bytes remain unchanged. The normal MMQ route handles every ineligible
    /// projection and every older configuration.
    /// Version 33 adds CUDA readers for Q8_0_TM projections. MMVQ, MMQ and the
    /// conservative F32 route all share the same tile-major wire contract, while
    /// paging preserves complete 8-row ownership units. The existing aligned
    /// whole-K knob applies to either Q8 layout; model identity keeps their tuning
    /// and correctness receipts separate.
    /// Version 34 adds a default-off Q8_0_TM SwiGLU pair transaction. The
    /// aligned whole-K up projection reads the completed gate value and writes
    /// `silu(gate) * up` directly, while every unselected or ineligible model
    /// retains the established materialized fallback.
    /// Version 35 lets that same aligned producer publish a D4 Q8 activation
    /// sidecar while its result tile is still on chip. Down then reuses the
    /// ordinary transient-cache contract and skips the standalone quantizer;
    /// the dense SwiGLU output remains authoritative for fallback and probes.
    /// Version 36 adds a default-off complete Q8_0_TM SwiGLU-to-Down transaction.
    /// Its private sidecar can omit the dense gated-intermediate write, while the
    /// backend capability contract retains the materialized workflow fallback.
    /// Version 37 adds an independently tuned D64/GQA4 whole-K Attention staging
    /// variant. One CTA cooperatively stages each 64x64 K/V update and lets its
    /// four query warps reuse it instead of repeating identical global reads.
    /// Version 38 adds a default-off Q8_0_TM MMQ variant that stages aligned
    /// 16-byte weight vectors directly into shared memory with Ampere `cp.async`
    /// while retaining the established scale loads and MMA arithmetic order.
    /// Version 40 adds an independently receipted D64/Q8 GQA4 Decode route that
    /// shares each K/V source tile across the four query heads which consume it.
    ///
    /// Version 41 adds a receipt-bound single-token Q8_0_TM Gate/Up/SiLU route.
    /// During tuning the candidate fails closed unless its fused SM86 kernel
    /// actually runs; production retains the established fallback.
    ///
    /// Version 42 adds a default-off short-convolution Decode fusion selector and a
    /// model-plan-derived stateful micro workload. Older measurements did not rank this
    /// route or restore its recurrent state between candidates.
    ///
    /// Version 43 adds a default-off, end-to-end selector for capture-local reuse of
    /// Q8 activations produced by projection RMSNorm. Ownership is valid only within
    /// one Decode capture generation; other backends retain the established sequence.
    /// Older files cannot prove which route produced their numerical evidence and must
    /// not be reused.
    /// Version 44 promotes admission-time mixed Q4/Q5 Decode projection maps from a
    /// laboratory environment surface into receipt-bound, end-to-end tuner choices.
    /// Their zero values preserve the exact Q8 path on every backend and model.
    /// The same version also admits the previously validated batched RMS-to-D4
    /// projection producer through an independent, safe-off end-to-end selector.
    /// Version 45 adds a receipt-bound per-layer map for admission-time Q8-TileMajor
    /// to Q4 Down shadows. Zero keeps every Prefill projection on its original Q8
    /// route; accepted maps are tied to the complete model.forward gate.
    fn space_version(&self) -> u32 {
        CUDA_SPACE_VERSION
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const CONSUMED_SLOTS: &[u32] = &[
        SLOT_GEMV_WARPS,
        SLOT_ATTN_THREADS,
        SLOT_RMS_THREADS,
        SLOT_MMQ_FULL_ROWS,
        SLOT_MMQ_FULL_TILE_MIN_EFFICIENCY,
        SLOT_ATTN_D256_WORKSPACE_MIB,
        SLOT_ATTN_D512_WORKSPACE_MIB,
        SLOT_ATTN_SCORE_KEY_GROUPS,
        SLOT_ATTN_SOFTMAX_WARPS,
        SLOT_ATTN_VALUE_TILES,
        SLOT_ATTN_STREAM_PART_CAP,
        SLOT_NARROW_GEMV_WARPS,
        SLOT_NARROW_GEMV_ROWS_PER_CTA,
        SLOT_NARROW_GEMV_MAX_WIDTH,
        SLOT_BATCH_MMVQ_ROWS_PER_CTA,
        SLOT_D512_MMA_MIN_SCHEDULE,
        SLOT_STREAMK_NUMERIC,
        SLOT_MMQ_LLAMA_COMPAT,
        SLOT_MMQ_VIRTUAL_512,
        SLOT_MMQ_CANONICAL_FULL_TILE,
        SLOT_RMS_NORM_ADD,
        SLOT_ATTN_D256_TILED,
        SLOT_ATTN_D256_VEC,
        SLOT_ATTN_D256_VIRTUAL_STREAM,
        SLOT_ATTN_D256_FUSED,
        SLOT_ATTN_D512_MMA,
        SLOT_ATTN_D512_VIRTUAL_STREAM,
        SLOT_ATTN_D512_VIRTUAL_CELL_POLICY,
        SLOT_ATTN_DECODE_SPECIALIZED,
        SLOT_ATTN_D64_MMA_PREFILL,
        SLOT_ATTN_D64_MMA_SHARED_KV_MIN_TOKENS,
        SLOT_FFN_SIDECAR_MIN_TOKENS,
        SLOT_PREFILL_EXACT128_SM86_ROUTE,
        SLOT_PREFILL_EXACT128_GRAPH,
        SLOT_ATTN_D64_Q8_VEC,
        SLOT_ATTN_D64_Q8_GQA4,
        SLOT_MMQ_Q8_TM_DECODE_SILU_PAIR,
        SLOT_SHORTCONV_DECODE_FUSED,
        SLOT_DECODE_GRAPH_Q8_PRODUCER_REUSE,
        SLOT_MMQ_Q8_ALIGNED_WHOLE_K,
        SLOT_MMQ_Q8_TM_ASYNC_WEIGHT_STAGE,
        SLOT_MMQ_Q8_TM_SILU_PAIR,
        SLOT_MMQ_Q8_TM_SILU_Q8_SIDECAR_MIN_TOKENS,
        SLOT_MMQ_Q8_TM_SILU_PRIVATE_DOWN_MIN_TOKENS,
        SLOT_MMQ_Q8_TM_GATE_UP_ROW_PAIR_MIN_TOKENS,
        SLOT_DECODE_FFN_Q5_LAYER_MASK,
        SLOT_DECODE_DOWN_Q4_LAYER_MASK,
        SLOT_DECODE_DOWN_Q5_LAYER_MASK,
        SLOT_PREFILL_PROJECTION_Q8_D4,
        SLOT_PREFILL_DOWN_Q4_LAYER_MASK,
        SLOT_ROW_LOCAL_PREFILL_TAIL_ROWS,
        SLOT_PREFILL_HEAD_POST_THREADS,
    ];

    #[test]
    fn registry_covers_each_consumed_slot_once() {
        assert_eq!(CUDA_KNOBS.len(), CONSUMED_SLOTS.len());
        for (decl, slot) in CUDA_KNOBS.iter().zip(CONSUMED_SLOTS) {
            assert_eq!(slot_for_name(decl.name), Some(*slot), "{}", decl.name);
        }
        assert_eq!(
            CONSUMED_SLOTS.iter().copied().collect::<HashSet<_>>().len(),
            CONSUMED_SLOTS.len()
        );
        assert!(!CONSUMED_SLOTS.contains(&9));
        assert!(!CONSUMED_SLOTS.contains(&10));
        assert_eq!(
            CUDA_KNOBS
                .iter()
                .map(|d| d.name)
                .collect::<HashSet<_>>()
                .len(),
            CUDA_KNOBS.len()
        );
    }

    #[test]
    fn value_sweeps_have_nonempty_unique_candidates() {
        for decl in CUDA_KNOBS {
            if decl.sweep == Sw::Values {
                assert!(!decl.values.is_empty(), "{} has no candidates", decl.name);
                assert_eq!(
                    decl.values.iter().copied().collect::<HashSet<_>>().len(),
                    decl.values.len(),
                    "{} repeats a candidate",
                    decl.name
                );
            }
        }
    }

    #[test]
    fn d512_boundary_cannot_tune_below_the_correctness_floor() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|d| d.name == "d512_mma_min_schedule")
            .unwrap();
        assert!(decl.bit_affecting);
        assert_eq!(decl.workload, Wl::AttentionDecodeDeep);
        assert!(decl.values.is_empty());
        match decl.sweep {
            Sw::SpanCrossing { ladder, hi, lo } => {
                assert_eq!(hi, D512_MMA_CONSERVATIVE_FLOOR);
                assert_eq!(lo, u32::MAX);
                assert!(ladder.iter().all(|&v| {
                    v >= D512_MMA_CONSERVATIVE_FLOOR && v % D512_SCHEDULE_KEYS == 0
                }));
            }
            _ => panic!("D512 route threshold must remain a span boundary"),
        }
    }

    #[test]
    fn registry_bump_invalidates_pre_boundary_cuda_configs() {
        assert_eq!(CUDA_SPACE_VERSION, 49);
    }

    #[test]
    fn decode_precision_maps_are_receipted_and_safe_off() {
        let names = [
            "decode_ffn_q5_layer_mask",
            "decode_down_q4_layer_mask",
            "decode_down_q5_layer_mask",
        ];
        let expected_slots = [53, 54, 55];
        for (name, slot) in names.into_iter().zip(expected_slots) {
            let decl = CUDA_KNOBS
                .iter()
                .find(|decl| decl.name == name)
                .expect("Decode precision-map registry entry");
            assert!(decl.bit_affecting);
            assert_eq!(decl.category, KnobCategory::EndToEnd);
            assert_eq!(decl.sweep, Sw::External);
            assert_eq!(decl.workload, Wl::DecodeFfnTransaction);
            assert_eq!(decl.tuple, Some("decode_precision_map"));
            assert_eq!(decl.values[0], 0);
            assert_eq!((decl.current)(), 0);
            assert_eq!(slot_for_name(name), Some(slot));
        }
        let down_q4 = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "decode_down_q4_layer_mask")
            .expect("Decode Down Q4 precision-map entry");
        assert_eq!(down_q4.values, &[0, 0x0200_5dda, 0x0200_5dd8]);

        let native = include_str!("../native/imparo_cuda.cu");
        for slot in expected_slots {
            assert!(native.contains(&format!("g.knobs[{slot}] = 0;")));
            assert!(native.contains(&format!("tuner_knob({slot})")));
        }
        let workflow = include_str!("../../imparo-model/src/lfm2/workflow_gpu.rs");
        assert!(workflow.contains("quantized_weight_cache_plan()"));
        assert!(!workflow.contains("IMPARO_LAB_LFM2_Q8_TM_DOWN_Q4_SHADOW"));
    }

    #[test]
    fn prefill_projection_preparation_is_receipted_and_safe_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "prefill_projection_q8_d4")
            .expect("Prefill Q8 producer registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.category, KnobCategory::EndToEnd);
        assert_eq!(decl.sweep, Sw::External);
        assert_eq!(decl.values, &[0, 1, 2]);
        assert_eq!(slot_for_name(decl.name), Some(56));
        assert_eq!((decl.current)(), 0);
    }

    #[test]
    fn prefill_down_precision_map_is_receipted_and_safe_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "prefill_down_q4_layer_mask")
            .expect("Prefill Down Q4 map registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.category, KnobCategory::EndToEnd);
        assert_eq!(decl.sweep, Sw::External);
        assert_eq!(decl.workload, Wl::PrefillFfnTransaction);
        assert_eq!(decl.tuple, Some("prefill_precision_map"));
        assert_eq!(decl.values, &[0, 0x3fff_ffff]);
        assert_eq!(slot_for_name(decl.name), Some(57));
        assert_eq!((decl.current)(), 0);

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[57] = 0;"));
        assert!(native.contains("tuner_knob(57)"));
    }

    #[test]
    fn row_local_prefill_tail_is_tuned_and_safe_by_default() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "row_local_prefill_tail_rows")
            .expect("row-local Prefill tail registry entry");
        assert!(!decl.bit_affecting);
        assert_eq!(decl.category, KnobCategory::Benched);
        assert_eq!(decl.values, &[0, 1, 4, 8, 16, 32, 64]);
        assert_eq!(slot_for_name(decl.name), Some(58));
        assert_eq!((decl.current)(), 0);
        assert_eq!(row_local_prefill_tail_rows(), 64);

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[58] = 0;"));
    }

    #[test]
    fn prefill_head_post_threads_is_receipted_and_safe_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "prefill_head_post_threads")
            .expect("Prefill head-post thread registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.category, KnobCategory::EndToEnd);
        assert_eq!(decl.sweep, Sw::External);
        assert_eq!(decl.values, &[0, 64, 128, 256]);
        assert_eq!(slot_for_name(decl.name), Some(59));
        assert_eq!((decl.current)(), 0);

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[59] = 0;"));
        assert!(native.contains("tuner_knob(59)"));
    }

    #[test]
    fn shortconv_decode_fusion_is_reachable_and_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "shortconv_decode_fused")
            .expect("short-convolution Decode registry entry");
        assert!(!decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.workload, Wl::DecodeShortconvTransaction);
        assert!(decl.applies.is_none());
        assert_eq!(slot_for_name(decl.name), Some(51));
        assert_eq!((decl.current)(), 0);
    }

    #[test]
    fn decode_specializations_are_one_receipted_arch_route() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "attn_decode_specialized")
            .expect("decode specialization registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.workload, Wl::AttentionDecodeDeep);

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[36] = verified_sm86;"));
        assert!(
            native.contains("const bool specialized_decode = tuner_knob(36) != 0;")
        );
        assert_eq!(native.matches("specialized_decode &&").count(), 4);
        assert!(native.contains("(!g.forward_decode || specialized_decode)"));
        assert!(native.contains("start_pos % selected_d512_query_tokens == 0"));
    }

    #[test]
    fn decode_graph_q8_producer_reuse_is_external_receipted_and_safe_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "decode_graph_q8_producer_reuse")
            .expect("Decode Graph Q8 producer reuse registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.category, KnobCategory::EndToEnd);
        assert_eq!(decl.sweep, Sw::External);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.workload, Wl::DecodeMix);
        assert!(decl.applies.is_some());
        assert_eq!(slot_for_name(decl.name), Some(52));
        assert_eq!((decl.current)(), 0);
    }

    #[test]
    fn d64_q8_vector_decode_is_receipted_and_shape_scoped() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "attn_d64_q8_vec")
            .expect("D64 Q8 vector Decode registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.workload, Wl::AttentionDecodeDeep);
        let applies = decl.applies.expect("D64 Q8 vector shape predicate");
        assert!(applies(&imparo_backend::ModelFacts {
            n_embd: 0,
            n_ff: 0,
            n_head: 32,
            n_kv: 8,
            head_dim: 64,
            deep_head_dim: 64,
            n_experts: 0,
            n_layers: 0,
            layer_dispatches: 0,
            weight_kinds: 0,
        }));
        assert!(!applies(&imparo_backend::ModelFacts {
            n_embd: 0,
            n_ff: 0,
            n_head: 32,
            n_kv: 8,
            head_dim: 256,
            deep_head_dim: 256,
            n_experts: 0,
            n_layers: 0,
            layer_dispatches: 0,
            weight_kinds: 0,
        }));
    }

    #[test]
    fn d64_q8_gqa4_decode_is_receipted_and_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "attn_d64_q8_gqa4")
            .expect("D64 Q8 GQA4 Decode registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.after, &["attn_d64_q8_vec"]);
        assert_eq!(decl.workload, Wl::DecodeAttentionStep);
        assert_eq!((decl.current)(), 0);

        let applies = decl.applies.expect("D64 Q8 GQA4 shape predicate");
        let facts = imparo_backend::ModelFacts {
            n_embd: 0,
            n_ff: 0,
            n_head: 32,
            n_kv: 8,
            head_dim: 64,
            deep_head_dim: 64,
            n_experts: 0,
            n_layers: 0,
            layer_dispatches: 0,
            weight_kinds: 0,
        };
        assert!(applies(&facts));

        let native = include_str!("../native/imparo_cuda.cu");
        let kernel = include_str!("../native/sm80/attention_decode_vec_d64_q8.cuh");
        assert!(native.contains("g.knobs[49] = 0;"));
        assert!(native.contains("tuner_knob(49) != 0"));
        assert!(native.contains("IMPARO_CUDA_NO_ATTN_D64_Q8_GQA4"));
        assert!(kernel.contains("partial_q8_gqa4"));
        assert!(kernel.contains("combine_q8_gqa4_final"));
    }

    #[test]
    fn q8_tm_decode_silu_pair_is_receipted_and_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "mmvq_q8_tm_decode_silu_pair")
            .expect("Q8_0_TM Decode Gate/Up/SiLU registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1]);
        assert!(decl.after.is_empty());
        assert_eq!(decl.workload, Wl::DecodeFfnTransaction);
        assert_eq!(decl.tuple, None);
        assert_eq!((decl.current)(), 0);

        let applies = decl.applies.expect("dense Q8_0_TM Decode FFN predicate");
        let mut facts = imparo_backend::ModelFacts {
            n_embd: 2048,
            n_ff: 10752,
            n_head: 32,
            n_kv: 8,
            head_dim: 64,
            deep_head_dim: 64,
            n_experts: 0,
            n_layers: 30,
            layer_dispatches: 9,
            weight_kinds: 1 << 3,
        };
        assert!(applies(&facts));
        facts.n_experts = 8;
        assert!(!applies(&facts));
        facts.n_experts = 0;
        facts.n_ff += 1;
        assert!(!applies(&facts));

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[50] = 0;"));
        assert!(native.contains("tuner_knob(50) != 0"));
        assert!(native.contains("IMPARO_CUDA_NO_Q8_TM_GATED_MMVQ"));
        let kernel = include_str!("../native/sm86/mmvq_q8_tm_gate_up_silu.cuh");
        assert!(kernel.contains("constexpr uint32_t kTotalWarps = 2 * kCohortWarps;"));
        assert!(kernel.contains("projection_tid = projection_warp * 32 + lane"));
        let backend = include_str!("backend_impl.rs");
        assert!(backend.contains("let q8_tm_decode_silu = gate_kind == 3"));
    }

    #[test]
    fn aligned_whole_k_q8_mmq_is_shape_checked_and_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "mmq_q8_aligned_whole_k")
            .expect("aligned whole-K Q8 MMQ registry entry");
        assert!(!decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.workload, Wl::PrefillGemm);
        assert_eq!(decl.tuple, None);
        let applies = decl.applies.expect("Q8 weight-kind predicate");
        let mut facts = imparo_backend::ModelFacts {
            n_embd: 2048,
            n_ff: 10752,
            n_head: 32,
            n_kv: 8,
            head_dim: 64,
            deep_head_dim: 64,
            n_experts: 0,
            n_layers: 30,
            layer_dispatches: 0,
            weight_kinds: 1 << 2,
        };
        assert!(applies(&facts));
        facts.weight_kinds = 1 << 1;
        assert!(!applies(&facts));

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[42] = 0;"));
        assert!(native.contains("tuner_knob(42) != 0"));
        assert!(native.contains("launch_aligned_whole_k("));
    }

    #[test]
    fn q8_tm_async_weight_stage_is_independent_and_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "mmq_q8_tm_async_weight_stage")
            .expect("Q8_0_TM async weight-stage registry entry");
        assert!(!decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1, 2]);
        assert_eq!(decl.after, &["mmq_q8_aligned_whole_k"]);
        assert_eq!(decl.workload, Wl::PrefillGemm);
        assert_eq!((decl.current)(), 0);

        let native = include_str!("../native/imparo_cuda.cu");
        let kernel = include_str!("../native/sm80/mmq_q8_q8_1.cuh");
        assert!(native.contains("g.knobs[47] = 0;"));
        assert!(native.contains("tuner_knob(47) != 0"));
        assert!(native.contains("IMPARO_CUDA_NO_Q8_TM_ASYNC_STAGE"));
        assert!(kernel.contains("bool AsyncTileMajor = false"));
        assert!(kernel.contains("copy_global_to_shared_16(staged, packed)"));
    }

    #[test]
    fn q8_tm_silu_pair_is_receipted_shape_scoped_and_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "mmq_q8_tm_silu_pair")
            .expect("Q8_0_TM SiLU pair registry entry");

        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.workload, Wl::PrefillFfnTransaction);
        assert_eq!(
            decl.after,
            &["mmq_q8_aligned_whole_k", "mmq_q8_tm_async_weight_stage",]
        );
        assert_eq!((decl.current)(), 0);

        let applies = decl.applies.expect("dense Q8_0_TM FFN predicate");
        let mut facts = imparo_backend::ModelFacts {
            n_embd: 2048,
            n_ff: 10752,
            n_head: 32,
            n_kv: 8,
            head_dim: 64,
            deep_head_dim: 64,
            n_experts: 0,
            n_layers: 30,
            layer_dispatches: 0,
            weight_kinds: 1 << 3,
        };
        assert!(applies(&facts));
        facts.weight_kinds = 1 << 2;
        assert!(!applies(&facts));
        facts.weight_kinds = 1 << 3;
        facts.n_experts = 1;
        assert!(!applies(&facts));

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[43] = 0;"));
        assert!(native.contains("tuner_knob(43) != 0"));
        assert!(native.contains("launch_aligned_whole_k<true, 2>"));
        let backend = include_str!("backend_impl.rs");
        assert!(backend.contains("crate::knobs::q8_tm_silu_pair_enabled()"));
        assert!(backend.contains("if q8_tm_silu || q8_tm_decode_silu"));
    }

    #[test]
    fn q8_tm_silu_q8_sidecar_min_tokens_depends_on_pair_and_is_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "mmq_q8_tm_silu_q8_sidecar_min_tokens")
            .expect("Q8_0_TM SwiGLU Q8 sidecar registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 9, 128, 129, 257, 513]);
        assert_eq!(decl.workload, Wl::PrefillFfnTransaction);
        assert_eq!(decl.after, &["mmq_q8_tm_silu_pair"]);
        assert_eq!((decl.current)(), 0);

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[44] = 0;"));
        assert!(native.contains("n_tok >= q8_tm_silu_sidecar_min_tokens"));
        assert!(native.contains("launch_aligned_whole_k<true, 3>"));
        assert!(native.contains("Q8_LAYOUT_MMQ_D4"));
    }

    #[test]
    fn q8_tm_silu_private_down_is_independent_and_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "mmq_q8_tm_silu_private_down_min_tokens")
            .expect("Q8_0_TM private SwiGLU-to-Down registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 9, 128, 129, 257, 513]);
        assert_eq!(decl.workload, Wl::PrefillFfnTransaction);
        assert_eq!(decl.after, &["mmq_q8_tm_silu_q8_sidecar_min_tokens"]);
        assert_eq!((decl.current)(), 0);
        assert_eq!(q8_tm_silu_private_down_min_tokens(), 0);

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[45] = 0;"));
        assert!(native.contains("launch_aligned_whole_k<true, 4>"));
    }
    #[test]
    fn q8_tm_gate_up_row_pair_is_receipted_and_default_off() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "mmq_q8_tm_gate_up_row_pair_min_tokens")
            .expect("Q8_0_TM Gate/Up row-pair registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 9, 128, 129, 257, 513]);
        assert_eq!(decl.workload, Wl::PrefillFfnTransaction);
        assert_eq!(decl.after, &["mmq_q8_tm_silu_private_down_min_tokens"]);
        assert_eq!((decl.current)(), 0);

        let applies = decl.applies.expect("dense Q8_0_TM FFN predicate");
        let mut facts = imparo_backend::ModelFacts {
            n_embd: 2048,
            n_ff: 10752,
            n_head: 32,
            n_kv: 8,
            head_dim: 64,
            deep_head_dim: 64,
            n_experts: 0,
            n_layers: 30,
            layer_dispatches: 0,
            weight_kinds: 1 << 3,
        };
        assert!(applies(&facts));
        facts.n_ff = 10751;
        assert!(!applies(&facts));

        let native = include_str!("../native/imparo_cuda.cu");
        let kernel = include_str!("../native/sm86/mmq_q8_tm_gate_up_row_pair.cuh");
        assert!(native.contains("g.knobs[48] = 0;"));
        assert!(native.contains("tuner_knob(48)"));
        assert!(native.contains("IMPARO_CUDA_NO_Q8_TM_GATE_UP_ROW_PAIR"));
        assert!(kernel.contains("q8_tm_gate_up_row_pair"));
        assert!(!kernel.contains("float * __restrict__ dst"));
    }
    #[test]
    fn d64_mma_prefill_is_receipted_and_sm86_owned() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "attn_d64_mma_prefill")
            .expect("D64 MMA prefill registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.workload, Wl::AttentionPrefillDeep);
        assert!(decl.applies.is_some());

        let native = include_str!("../native/imparo_cuda.cu");
        assert!(native.contains("g.knobs[37] = verified_sm86;"));
        assert!(native.contains("const bool d64_mma_prefill = tuner_knob(37) != 0"));
        assert!(!native.contains("IMPARO_CUDA_ATTN_D64_MMA_PREFILL\") != nullptr"));
        assert!(native.contains("IMPARO_CUDA_NO_ATTN_D64_MMA_PREFILL"));
    }

    #[test]
    fn d64_shared_kv_staging_is_a_safe_off_token_boundary() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "attn_d64_mma_shared_kv_min_tokens")
            .expect("D64 shared K/V registry entry");
        assert!(decl.bit_affecting);
        assert!(decl.values.is_empty());
        assert_eq!(decl.after, &["attn_d64_mma_prefill"]);
        assert_eq!(decl.workload, Wl::AttentionPrefillDeep);
        assert_eq!((decl.current)(), 0);
        match decl.sweep {
            Sw::TokenMinCrossing { ladder, hi, lo } => {
                assert_eq!(ladder, ATTN_D64_SHARED_KV_TOKEN_LADDER);
                assert_eq!((hi, lo), (1, 0));
                assert!(ladder.windows(2).all(|pair| pair[0] < pair[1]));
            }
            _ => panic!("D64 shared K/V must remain a minimum-token boundary"),
        }

        let native = include_str!("../native/imparo_cuda.cu");
        let kernel = include_str!("../native/sm80/attention_prefill_mma_d64_f16.cuh");
        assert!(native.contains("g.knobs[46] = 0;"));
        assert!(native.contains("whole_k_tile<true>"));
        assert!(native.contains("n_tok >= shared_kv_min_tokens"));
        assert!(native.contains("IMPARO_CUDA_NO_ATTN_D64_SHARED_KV"));
        assert!(kernel.contains("template <bool SharedKv>"));
        assert!(kernel.contains("e < shared_kv_half2"));
    }

    #[test]
    fn ffn_sidecar_is_a_safe_off_minimum_token_boundary() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "ffn_sidecar_min_tokens")
            .expect("FFN sidecar boundary registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.workload, Wl::PrefillFfnTransaction);
        assert_eq!(ffn_sidecar_min_tokens(), 0);
        match decl.sweep {
            Sw::TokenMinCrossing { ladder, hi, lo } => {
                assert_eq!(ladder, FFN_SIDECAR_TOKEN_LADDER);
                assert_eq!(hi, 1);
                assert_eq!(lo, 0);
                assert!(ladder.windows(2).all(|pair| pair[0] < pair[1]));
            }
            _ => panic!("FFN sidecar must remain a minimum-token boundary"),
        }
        assert!(FFN_SIDECAR_TOKEN_LADDER.iter().all(|value| *value > 128));
    }

    #[test]
    fn exact128_is_one_safe_off_atomic_route() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "prefill_exact128_sm86_route")
            .expect("exact-128 route registry entry");
        assert!(decl.bit_affecting);
        assert_eq!(decl.category, KnobCategory::EndToEnd);
        assert_eq!(decl.sweep, Sw::External);
        assert_eq!(decl.values, &[0, 1, 2, 3, 4, 5]);
        assert_eq!(decl.workload, Wl::PrefillFfnExact128);
        assert_eq!((decl.current)(), 0);
        assert!(decl.applies.is_some());
    }

    #[test]
    fn exact128_graph_is_safe_off_and_ordered_after_its_route() {
        let decl = CUDA_KNOBS
            .iter()
            .find(|decl| decl.name == "prefill_exact128_graph")
            .expect("exact-128 Graph registry entry");
        assert!(!decl.bit_affecting);
        assert_eq!(decl.category, KnobCategory::EndToEnd);
        assert_eq!(decl.sweep, Sw::External);
        assert_eq!(decl.values, &[0, 1]);
        assert_eq!(decl.workload, Wl::PrefillFfnExact128);
        assert_eq!(decl.after, &["prefill_exact128_sm86_route"]);
        assert_eq!((decl.current)(), 0);
        assert!(decl.applies.is_some());
    }
}
