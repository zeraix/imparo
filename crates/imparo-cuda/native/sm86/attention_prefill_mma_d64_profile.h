#pragma once

// Pinned SM86 policy for the common D64/GQA4 whole-K prefill kernel.
// Keep device facts out of the semantic kernel so future SM families can provide
// their own profile without editing global dispatch logic.
namespace imparo_sm86_d64_mma_profile {

constexpr uint32_t kThreads = 128;
constexpr uint32_t kLaunchOccupancy = 2;
constexpr uint32_t kDynamicSharedBytes = 0;

constexpr bool applies_to(int sm_version) {
    return sm_version == 86;
}

static_assert(kThreads == imparo_sm80_d64_mma_plan::kThreads);

} // namespace imparo_sm86_d64_mma_profile
