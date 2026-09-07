#include "lfm2_ops.cuh"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

using imparo_cuda_lfm2::ShortconvLayout;
using imparo_cuda_lfm2::checked_shortconv_layout;
using imparo_cuda_lfm2::shortconv_reference;
using imparo_cuda_lfm2::shortconv_state_reference;

static void advance(const std::vector<float> & bcx, std::vector<float> & state,
                    uint32_t width, uint32_t kernel, uint32_t n_tok) {
    std::vector<float> next(state.size());
    shortconv_state_reference(bcx.data(), state.data(), next.data(),
                              width, kernel, n_tok);
    state = std::move(next);
}

static void batched_equals_stepwise() {
    constexpr uint32_t width = 4, kernel = 3, n_tok = 5;
    std::vector<float> bcx(n_tok * 3 * width);
    std::vector<float> weights(width * kernel);
    for (uint32_t i = 0; i < bcx.size(); ++i) bcx[i] = float(i % 7) - 3.0f;
    for (uint32_t i = 0; i < weights.size(); ++i) weights[i] = (float(i % 5) - 2.0f) * 0.25f;
    std::vector<float> batch_state((kernel - 1) * width, 0.0f);
    std::vector<float> batch_out(n_tok * width);
    shortconv_reference(bcx.data(), weights.data(), batch_state.data(), batch_out.data(),
                        width, kernel, n_tok);
    advance(bcx, batch_state, width, kernel, n_tok);

    std::vector<float> step_state((kernel - 1) * width, 0.0f);
    std::vector<float> step_out(n_tok * width);
    for (uint32_t token = 0; token < n_tok; ++token) {
        std::vector<float> row(bcx.begin() + token * 3 * width,
                               bcx.begin() + (token + 1) * 3 * width);
        shortconv_reference(row.data(), weights.data(), step_state.data(),
                            step_out.data() + token * width, width, kernel, 1);
        advance(row, step_state, width, kernel, 1);
    }
    assert(batch_out == step_out);
    assert(batch_state == step_state);
}

static void tap_gate_and_snapshot_contracts() {
    constexpr uint32_t width = 1, kernel = 4, n_tok = 2;
    const std::vector<float> bcx = {2, 0, 3, 2, 0, 5};
    const std::vector<float> state = {11, 13, 17};
    const std::vector<float> oldest = {1, 0, 0, 0};
    std::vector<float> output(n_tok);
    shortconv_reference(bcx.data(), oldest.data(), state.data(), output.data(),
                        width, kernel, n_tok);
    assert(output[0] == 0.0f && output[1] == 0.0f);
    std::vector<float> next(kernel - 1);
    shortconv_state_reference(bcx.data(), state.data(), next.data(), width, kernel, n_tok);
    assert((next == std::vector<float>{17, 6, 10}));

    const std::vector<float> unit_c = {1, 1, 1, 1, 1, 1};
    const std::vector<float> state2 = {5, 7, 9};
    shortconv_reference(unit_c.data(), oldest.data(), state2.data(), output.data(),
                        width, kernel, n_tok);
    assert((output == std::vector<float>{5, 7}));
}

static void one_channel_owner_makes_in_place_advance_safe() {
    // history=2,n_tok=1 is the race-sensitive shape: slot 0 reads old slot 1 while
    // slot 1 is replaced. The shared channel helper must preserve {7, 2*3}.
    const std::vector<float> bcx = {2, 1, 3};
    std::vector<float> state = {5, 7};
    imparo_cuda_lfm2::shortconv_state_channel(
        bcx.data(), state.data(), state.data(), 0, 1, 3, 1);
    assert((state == std::vector<float>{7, 6}));
}

static void validation_and_silu_contracts() {
    ShortconvLayout layout;
    assert(checked_shortconv_layout(2048, 4, 512, &layout));
    assert(layout.weight_bytes == 2048ull * 4 * sizeof(float));
    assert(!checked_shortconv_layout(1, 1, 1, &layout));
    assert(!checked_shortconv_layout(0, 4, 1, &layout));
    assert(!checked_shortconv_layout(UINT32_MAX, UINT32_MAX, UINT32_MAX, &layout));
    assert(std::fabs(imparo_cuda_lfm2::silu(1.0f) - 0.7310586f) < 1e-6f);
    assert(imparo_cuda_lfm2::silu(0.0f) == 0.0f);
}

int main() {
    batched_equals_stepwise();
    tap_gate_and_snapshot_contracts();
    one_channel_owner_makes_in_place_advance_safe();
    validation_and_silu_contracts();
    return 0;
}
