// SPDX-License-Identifier: MIT
//
// Copyright (c) 2026 Imparo contributors

// Pure arithmetic contract for the Ampere Q8_0 x Q8_1 MMQ selector and
// numerical Stream-K replay. This header intentionally has no CUDA runtime
// dependency so route planning can be tested on hosts without a GPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define IMPARO_Q8_REPLAY_HD __host__ __device__
#else
#define IMPARO_Q8_REPLAY_HD
#endif

namespace imparo_sm80_q8_replay {

constexpr uint32_t kRows = 128;
constexpr uint32_t kBlockValues = 32;
constexpr uint32_t kStageBlocks = 8;
constexpr uint32_t kWeightStride = 304;
constexpr uint32_t kActivationStride = 144;

enum class PlanStatus : uint8_t {
    Ok,
    NotApplicable,
    InvalidShape,
    UnsupportedTile,
    Overflow,
    GridLimit,
    WorkLimit,
};

enum class ExecutionRoute : uint8_t {
    Mmq,
    SafeF32,
    Reject,
};

struct PlanLimits {
    uint64_t max_grid_x = 0;
    uint64_t max_grid_y = 0;
    uint64_t max_shared_bytes = 0;
    uint64_t max_total_work = 0;
    uint16_t enabled_tile_mask = 0;
};

struct ReplayPlan {
    PlanStatus status = PlanStatus::InvalidShape;
    uint32_t tile_tokens = 0;
    uint32_t row_tiles = 0;
    uint32_t token_tiles = 0;
    uint32_t logical_tiles = 0;
    uint32_t physical_grid = 0;
    uint32_t prefix_phases = 0;
    uint32_t blocks = 0;
    uint64_t total_work = 0;
    uint64_t workspace_bytes_per_row = 0;
    bool fallback_rows = false;
    bool replay = false;
    bool multi_seam = false;
    bool needs_workspace = false;
};

struct Segment {
    uint32_t begin = 0;
    uint32_t end = 0;
    bool valid = false;
};

struct SliceTiles {
    uint32_t first = 0;
    uint32_t count = 0;
    bool valid = false;
};

IMPARO_Q8_REPLAY_HD constexpr bool checked_add_u64(
        uint64_t a, uint64_t b, uint64_t * out) {
    if (!out || b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

IMPARO_Q8_REPLAY_HD constexpr bool checked_mul_u64(
        uint64_t a, uint64_t b, uint64_t * out) {
    if (!out || (a && b > std::numeric_limits<uint64_t>::max() / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

IMPARO_Q8_REPLAY_HD constexpr bool checked_ceil_div_u32_to_u64(
        uint32_t value, uint32_t divisor, uint64_t * out) {
    if (!out || !divisor) return false;
    uint64_t numerator = 0;
    if (!checked_add_u64(value, uint64_t(divisor) - 1, &numerator)) {
        return false;
    }
    *out = numerator / divisor;
    return true;
}

IMPARO_Q8_REPLAY_HD constexpr bool checked_workspace_bytes(
        uint32_t rows, uint32_t tokens, uint64_t * out) {
    uint64_t count = 0;
    return checked_mul_u64(rows, tokens, &count)
        && checked_mul_u64(count, sizeof(float), out);
}

IMPARO_Q8_REPLAY_HD constexpr bool registered_tile(
        uint32_t tokens, bool fallback) {
    if (fallback) {
        return tokens == 8 || tokens == 16 || tokens == 32
            || tokens == 64 || tokens == 128;
    }
    return tokens == 8 || tokens == 16 || tokens == 24
        || tokens == 32 || tokens == 40 || tokens == 48
        || tokens == 64 || tokens == 80 || tokens == 96
        || tokens == 112 || tokens == 128;
}

IMPARO_Q8_REPLAY_HD constexpr uint16_t tile_bit(uint32_t tokens) {
    return tokens >= 8 && tokens <= 128 && tokens % 8 == 0
        ? uint16_t(1u << (tokens / 8 - 1)) : uint16_t{0};
}

IMPARO_Q8_REPLAY_HD constexpr uint16_t all_registered_tile_mask() {
    uint16_t mask = 0;
    for (uint32_t tokens = 8; tokens <= 128; tokens += 8) {
        if (registered_tile(tokens, false) || registered_tile(tokens, true)) {
            mask = uint16_t(mask | tile_bit(tokens));
        }
    }
    return mask;
}

IMPARO_Q8_REPLAY_HD constexpr uint64_t shared_bytes(uint32_t tokens) {
    return uint64_t(kRows) * kWeightStride
        + uint64_t(2) * tokens * kActivationStride;
}

IMPARO_Q8_REPLAY_HD constexpr uint32_t select_tile_tokens(
        uint32_t n_tok, bool fallback, const PlanLimits & limits) {
    if (!n_tok) return 0;
    uint32_t best = 0;
    // Token-tile width is part of the numerical route because each template owns
    // a different MMA accumulator layout. Use one maximal supported template for
    // every MMQ-eligible batch, including partial tiles at a resume boundary.
    for (uint32_t candidate = 8; candidate <= 128; candidate += 8) {
        if (!registered_tile(candidate, fallback)
            || !(limits.enabled_tile_mask & tile_bit(candidate))
            || shared_bytes(candidate) > limits.max_shared_bytes) {
            continue;
        }
        best = candidate;
    }
    return best;
}

IMPARO_Q8_REPLAY_HD constexpr ReplayPlan canonical_whole_k_plan(
        ReplayPlan plan) {
    if (plan.status != PlanStatus::Ok) return plan;
    // Physical Stream-K improves occupancy by splitting one output tile's K
    // reduction between workers, but the split point depends on caller width.
    // Recurrent models feed that reassociation into persistent state, so the safe
    // production route keeps one ascending whole-K owner per logical output tile.
    plan.physical_grid = plan.logical_tiles;
    plan.prefix_phases = 0;
    plan.replay = false;
    plan.multi_seam = false;
    plan.needs_workspace = false;
    plan.workspace_bytes_per_row = 0;
    return plan;
}

IMPARO_Q8_REPLAY_HD constexpr uint32_t pinned_physical_grid(
        uint32_t logical_tiles, uint32_t sm_count) {
    if (!logical_tiles || !sm_count) return 0;
    const uint64_t waves =
        (uint64_t(logical_tiles) + sm_count - 1) / sm_count;
    const uint64_t efficiency = uint64_t(100) * logical_tiles
        / (uint64_t(sm_count) * waves);
    return efficiency >= 90 ? logical_tiles : sm_count;
}

IMPARO_Q8_REPLAY_HD constexpr ReplayPlan make_plan(
        uint32_t n_in, uint32_t out_stride, uint32_t n_tok,
        uint32_t sm_count, const PlanLimits & limits) {
    ReplayPlan plan{};
    if (!n_in || !out_stride || !n_tok || !sm_count || n_in % 32 != 0) {
        plan.status = PlanStatus::InvalidShape;
        return plan;
    }
    if (n_in % 128 != 0) {
        plan.status = PlanStatus::NotApplicable;
        return plan;
    }
    if (n_tok <= 8) {
        plan.status = PlanStatus::NotApplicable;
        return plan;
    }
    plan.fallback_rows = out_stride % kRows != 0;
    plan.tile_tokens = select_tile_tokens(n_tok, plan.fallback_rows, limits);
    if (!plan.tile_tokens) {
        plan.status = PlanStatus::UnsupportedTile;
        return plan;
    }

    uint64_t row_tiles = 0;
    uint64_t token_tiles = 0;
    uint64_t logical_tiles = 0;
    if (!checked_ceil_div_u32_to_u64(out_stride, kRows, &row_tiles)
        || !checked_ceil_div_u32_to_u64(
            n_tok, plan.tile_tokens, &token_tiles)
        || !checked_mul_u64(row_tiles, token_tiles, &logical_tiles)
        || logical_tiles > std::numeric_limits<uint32_t>::max()) {
        plan.status = PlanStatus::Overflow;
        return plan;
    }
    if (!limits.max_grid_x || !limits.max_grid_y
        || row_tiles > limits.max_grid_x || token_tiles > limits.max_grid_y) {
        plan.status = PlanStatus::GridLimit;
        return plan;
    }
    plan.row_tiles = uint32_t(row_tiles);
    plan.token_tiles = uint32_t(token_tiles);
    plan.logical_tiles = uint32_t(logical_tiles);
    plan.blocks = n_in / kBlockValues;
    if (!checked_mul_u64(logical_tiles, plan.blocks, &plan.total_work)) {
        plan.status = PlanStatus::Overflow;
        return plan;
    }
    if (limits.max_total_work && plan.total_work > limits.max_total_work) {
        plan.status = PlanStatus::WorkLimit;
        return plan;
    }
    plan.physical_grid = pinned_physical_grid(plan.logical_tiles, sm_count);
    if (!plan.physical_grid) {
        plan.status = PlanStatus::InvalidShape;
        return plan;
    }
    const bool imbalanced = plan.logical_tiles % plan.physical_grid != 0;
    const uint32_t seam_slots = (plan.blocks - 1) / kStageBlocks;
    const bool potentially_multi = imbalanced
        && plan.physical_grid > plan.logical_tiles;
    const uint32_t raw_prefix_phases = !imbalanced ? 0u : (potentially_multi
        ? uint32_t((uint64_t(plan.physical_grid - 1)
            + plan.logical_tiles - 1) / plan.logical_tiles)
        : 1u);
    plan.prefix_phases = raw_prefix_phases < seam_slots
        ? raw_prefix_phases : seam_slots;
    plan.replay = plan.prefix_phases != 0;
    plan.multi_seam = plan.prefix_phases > 1;
    plan.needs_workspace = plan.multi_seam;
    if (plan.needs_workspace && !checked_workspace_bytes(
            1, n_tok, &plan.workspace_bytes_per_row)) {
        plan.status = PlanStatus::Overflow;
        return plan;
    }
    plan.status = PlanStatus::Ok;
    return plan;
}

IMPARO_Q8_REPLAY_HD constexpr bool slice_workspace_bytes(
        const ReplayPlan & plan, uint32_t rows, uint64_t * out) {
    if (!out || plan.status != PlanStatus::Ok) return false;
    if (!plan.needs_workspace) {
        *out = 0;
        return true;
    }
    return checked_mul_u64(rows, plan.workspace_bytes_per_row, out);
}

IMPARO_Q8_REPLAY_HD constexpr ExecutionRoute choose_execution(
        const ReplayPlan & plan, bool workspace_available) {
    switch (plan.status) {
        case PlanStatus::InvalidShape:
        case PlanStatus::Overflow:
        case PlanStatus::GridLimit: return ExecutionRoute::Reject;
        case PlanStatus::Ok: break;
        default: return ExecutionRoute::SafeF32;
    }
    return plan.needs_workspace && !workspace_available
        ? ExecutionRoute::SafeF32 : ExecutionRoute::Mmq;
}

IMPARO_Q8_REPLAY_HD constexpr uint64_t stream_boundary(
        uint32_t worker, uint64_t total_work, uint32_t physical_grid,
        uint32_t blocks) {
    if (!physical_grid || !blocks) return 0;
    uint64_t boundary = uint64_t(worker) * total_work / physical_grid;
    boundary -= (boundary % blocks) % kStageBlocks;
    return boundary;
}

IMPARO_Q8_REPLAY_HD constexpr bool previous_stream_seam(
        uint64_t logical_tile, uint32_t segment_end, uint64_t total_work,
        uint32_t physical_grid, uint32_t blocks, uint32_t * seam) {
    if (!seam || !blocks) return false;
    uint32_t best = 0;
    for (uint32_t worker = 1; worker < physical_grid; ++worker) {
        const uint64_t boundary = stream_boundary(
            worker, total_work, physical_grid, blocks);
        if (boundary / blocks != logical_tile) continue;
        const uint32_t k = uint32_t(boundary % blocks);
        if (k && k < segment_end && k > best) best = k;
    }
    *seam = best;
    return best != 0;
}

IMPARO_Q8_REPLAY_HD constexpr Segment segment_for_phase(
        uint64_t logical_tile, uint32_t blocks, uint64_t total_work,
        uint32_t physical_grid, uint32_t replay_phase) {
    Segment segment{0, blocks, blocks != 0};
    if (!segment.valid || !physical_grid) return segment;
    for (uint32_t step = 0; step <= replay_phase; ++step) {
        uint32_t seam = 0;
        const bool found = previous_stream_seam(
            logical_tile, segment.end, total_work, physical_grid, blocks, &seam);
        if (step < replay_phase) {
            if (!found) return Segment{};
            segment.end = seam;
        } else {
            segment.begin = found ? seam : 0;
        }
    }
    segment.valid = segment.begin < segment.end;
    return segment;
}

IMPARO_Q8_REPLAY_HD constexpr SliceTiles slice_tiles(
        uint32_t row_base, uint32_t rows, uint32_t out_stride) {
    SliceTiles result{};
    if (!rows || row_base > out_stride || rows > out_stride - row_base) {
        return result;
    }
    result.first = row_base / kRows;
    const uint64_t first_offset = row_base % kRows;
    const uint64_t covered = first_offset + rows;
    result.count = uint32_t((covered + kRows - 1) / kRows);
    result.valid = true;
    return result;
}

} // namespace imparo_sm80_q8_replay

#undef IMPARO_Q8_REPLAY_HD
