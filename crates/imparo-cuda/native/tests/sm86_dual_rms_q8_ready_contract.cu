#include "../sm86/rms_norm_add_dual_q8_ready.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace Dual = imparo_sm86_dual_rms_q8_ready;
namespace Ready = imparo_q8_mma_ready_a0_v1_authority_lab;

void check_cuda(cudaError_t status, const char * expression, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA failure line=%d expression=%s error=%s\n",
            line, expression, cudaGetErrorString(status));
        std::exit(3);
    }
}
#define CHECK_CUDA(expression) check_cuda((expression), #expression, __LINE__)

__launch_bounds__(Dual::kThreads, 1)
__global__ void control_first_rms_add(
        const float *__restrict__ src,
        const float *__restrict__ residual,
        const float *__restrict__ weight,
        float *__restrict__ mid,
        uint32_t n_tok,
        float eps) {
#if __CUDA_ARCH__ == 860
    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok) return;
    const uint64_t row_offset = uint64_t(tok) * Dual::kWidth;
    const float *src_row = src + row_offset;
    const float *residual_row = residual + row_offset;
    float *mid_row = mid + row_offset;
    __shared__ float reduction[Dual::kWarps];
    float sum = 0.0f;
    for (uint32_t col = tid; col < Dual::kWidth; col += Dual::kThreads) {
        const float value = src_row[col];
        sum += value * value;
    }
    sum = Dual::block_sum<Dual::kThreads>(sum, reduction);
    const float scale = rsqrtf(sum / Dual::kWidth + eps);
    for (uint32_t col = tid; col < Dual::kWidth; col += Dual::kThreads) {
        mid_row[col] = scale * src_row[col] * weight[col]
            + residual_row[col];
    }
#else
    (void)src; (void)residual; (void)weight; (void)mid; (void)n_tok; (void)eps;
#endif
}

__launch_bounds__(Dual::kThreads, 1)
__global__ void control_second_rms_q8_ready(
        const float *__restrict__ mid,
        const float *__restrict__ weight,
        float *__restrict__ norm,
        uint16_t *__restrict__ quant_u16,
        float *__restrict__ d8_sideplane,
        uint32_t n_tok,
        uint32_t token_tiles,
        float eps) {
#if __CUDA_ARCH__ == 860
    const uint32_t tok = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (tok >= n_tok) return;
    const uint64_t row_offset = uint64_t(tok) * Dual::kWidth;
    const float *mid_row = mid + row_offset;
    float *norm_row = norm + row_offset;
    __shared__ float reduction[Dual::kWarps];
    float sum = 0.0f;
    for (uint32_t col = tid; col < Dual::kWidth; col += Dual::kThreads) {
        const float value = mid_row[col];
        sum += value * value;
    }
    sum = Dual::block_sum<Dual::kThreads>(sum, reduction);
    const float scale = rsqrtf(sum / Dual::kWidth + eps);
    const uint32_t token_tile = tok / Ready::kTokenTile;
    const uint32_t token_in_tile = tok % Ready::kTokenTile;
    for (uint32_t i0 = tid * 4; i0 < Dual::kWidth;
            i0 += Dual::kThreads * 4) {
        const float4 source = reinterpret_cast<const float4 *>(mid_row + i0)[0];
        float4 value;
        value.x = scale * source.x * weight[i0 + 0];
        value.y = scale * source.y * weight[i0 + 1];
        value.z = scale * source.z * weight[i0 + 2];
        value.w = scale * source.w * weight[i0 + 3];
        reinterpret_cast<float4 *>(norm_row)[i0 / 4] = value;
        float amax = fabsf(value.x);
        amax = fmaxf(amax, fabsf(value.y));
        amax = fmaxf(amax, fabsf(value.z));
        amax = fmaxf(amax, fabsf(value.w));
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            amax = fmaxf(
                amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));
        }
        const float d_inv = 127.0f / amax;
        char4 quant;
        quant.x = int8_t(roundf(value.x * d_inv));
        quant.y = int8_t(roundf(value.y * d_inv));
        quant.z = int8_t(roundf(value.z * d_inv));
        quant.w = int8_t(roundf(value.w * d_inv));
        const uint32_t block = i0 / Ready::kValuesPerQBlock;
        const uint32_t group = block / Ready::kQBlocksPerGroup;
        const uint32_t qblock = block % Ready::kQBlocksPerGroup;
        const uint32_t kpair = (i0 % Ready::kValuesPerQBlock) / 2;
        quant_u16[Ready::quant_u16_index(
            token_tiles, group, token_tile, qblock, kpair,
            token_in_tile)] = uint16_t(uint8_t(quant.x))
                | (uint16_t(uint8_t(quant.y)) << 8);
        quant_u16[Ready::quant_u16_index(
            token_tiles, group, token_tile, qblock, kpair + 1,
            token_in_tile)] = uint16_t(uint8_t(quant.z))
                | (uint16_t(uint8_t(quant.w)) << 8);
        if ((i0 % Ready::kValuesPerQBlock) == 0) {
            d8_sideplane[Ready::scale_index(
                token_tiles, group, token_tile, qblock,
                token_in_tile)] =
                    __half2float(__float2half(1.0f / d_inv));
        }
    }
#else
    (void)mid; (void)weight; (void)norm; (void)quant_u16;
    (void)d8_sideplane; (void)n_tok; (void)token_tiles; (void)eps;
#endif
}

template <typename T>
T * device_allocate(std::size_t count) {
    T * pointer = nullptr;
    CHECK_CUDA(cudaMalloc(&pointer, count * sizeof(T)));
    return pointer;
}

template <typename T>
void upload(T * destination, const std::vector<T> & source) {
    CHECK_CUDA(cudaMemcpy(destination, source.data(),
        source.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> download(const T * source, std::size_t count) {
    std::vector<T> destination(count);
    CHECK_CUDA(cudaMemcpy(destination.data(), source,
        count * sizeof(T), cudaMemcpyDeviceToHost));
    return destination;
}

template <typename T>
bool byte_equal(const std::vector<T> & first, const std::vector<T> & second) {
    return first.size() == second.size()
        && std::memcmp(first.data(), second.data(), first.size() * sizeof(T)) == 0;
}

double max_abs_delta(const std::vector<float> & first,
        const std::vector<float> & second) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < first.size(); ++index) {
        maximum = std::max(maximum,
            std::fabs(double(first[index]) - double(second[index])));
    }
    return maximum;
}

bool run_shape(uint32_t n_tok) {
    const std::size_t dense_count = std::size_t(n_tok) * Dual::kWidth;
    std::vector<float> src(dense_count);
    std::vector<float> residual(dense_count);
    std::vector<float> first_weight(Dual::kWidth);
    std::vector<float> second_weight(Dual::kWidth);
    for (std::size_t index = 0; index < dense_count; ++index) {
        const int centered = int((index * 37u + 11u) % 257u) - 128;
        const int residual_centered = int((index * 19u + 23u) % 193u) - 96;
        src[index] = float(centered) * 0.0078125f;
        residual[index] = float(residual_centered) * 0.00390625f;
    }
    for (uint32_t index = 0; index < Dual::kWidth; ++index) {
        first_weight[index] = 0.75f + float((index * 13u) % 31u) / 64.0f;
        second_weight[index] = 0.625f + float((index * 17u) % 29u) / 64.0f;
    }

    Ready::Layout layout{};
    const uint32_t padded_tokens = ((n_tok + 127u) / 128u) * 128u;
    if (!Ready::make_layout(Dual::kWidth, padded_tokens, &layout)) {
        std::fprintf(stderr, "layout rejected n_tok=%u\n", n_tok);
        return false;
    }
    layout.n_tok = n_tok;
    const std::size_t quant_count = std::size_t(layout.quant_u16_count);
    const std::size_t scale_count = std::size_t(layout.scale_count);

    float *d_src = device_allocate<float>(dense_count);
    float *d_residual = device_allocate<float>(dense_count);
    float *d_first_weight = device_allocate<float>(Dual::kWidth);
    float *d_second_weight = device_allocate<float>(Dual::kWidth);
    float *d_candidate_mid = device_allocate<float>(dense_count);
    float *d_candidate_norm = device_allocate<float>(dense_count);
    float *d_control_mid = device_allocate<float>(dense_count);
    float *d_control_norm = device_allocate<float>(dense_count);
    uint16_t *d_candidate_quant = device_allocate<uint16_t>(quant_count);
    uint16_t *d_control_quant = device_allocate<uint16_t>(quant_count);
    float *d_candidate_scale = device_allocate<float>(scale_count);
    float *d_control_scale = device_allocate<float>(scale_count);
    upload(d_src, src);
    upload(d_residual, residual);
    upload(d_first_weight, first_weight);
    upload(d_second_weight, second_weight);
    CHECK_CUDA(cudaMemset(d_candidate_quant, 0x5a,
        quant_count * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemset(d_control_quant, 0x5a,
        quant_count * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemset(d_candidate_scale, 0x5a,
        scale_count * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_control_scale, 0x5a,
        scale_count * sizeof(float)));

    constexpr float eps = 1.0e-6f;
    const auto launched = Dual::launch(
        d_src, d_residual, d_first_weight, d_second_weight,
        d_candidate_mid, d_candidate_norm, d_candidate_quant,
        d_candidate_scale, Dual::kWidth, n_tok, layout, eps, eps, 86, 0);
    if (launched != Dual::LaunchResult::Launched) {
        std::fprintf(stderr, "candidate rejected n_tok=%u result=%u\n",
            n_tok, unsigned(launched));
        return false;
    }
    control_first_rms_add<<<n_tok, Dual::kThreads>>>(
        d_src, d_residual, d_first_weight, d_control_mid, n_tok, eps);
    control_second_rms_q8_ready<<<n_tok, Dual::kThreads>>>(
        d_control_mid, d_second_weight, d_control_norm,
        d_control_quant, d_control_scale, n_tok, layout.token_tiles, eps);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const auto candidate_mid = download(d_candidate_mid, dense_count);
    const auto candidate_norm = download(d_candidate_norm, dense_count);
    const auto candidate_quant = download(d_candidate_quant, quant_count);
    const auto candidate_scale = download(d_candidate_scale, scale_count);
    const auto control_mid = download(d_control_mid, dense_count);
    const auto control_norm = download(d_control_norm, dense_count);
    const auto control_quant = download(d_control_quant, quant_count);
    const auto control_scale = download(d_control_scale, scale_count);

    const bool mid_equal = byte_equal(candidate_mid, control_mid);
    const bool norm_equal = byte_equal(candidate_norm, control_norm);
    const bool quant_equal = byte_equal(candidate_quant, control_quant);
    const bool scale_equal = byte_equal(candidate_scale, control_scale);
    std::printf(
        "{\"tokens\":%u,\"padded_tokens\":%u,\"mid_byte_equal\":%s,"
        "\"norm_byte_equal\":%s,\"quant_byte_equal\":%s,"
        "\"scale_byte_equal\":%s,\"mid_max_abs\":%.9g,"
        "\"norm_max_abs\":%.9g}\n",
        n_tok, padded_tokens, mid_equal ? "true" : "false",
        norm_equal ? "true" : "false", quant_equal ? "true" : "false",
        scale_equal ? "true" : "false",
        max_abs_delta(candidate_mid, control_mid),
        max_abs_delta(candidate_norm, control_norm));

    cudaFree(d_src);
    cudaFree(d_residual);
    cudaFree(d_first_weight);
    cudaFree(d_second_weight);
    cudaFree(d_candidate_mid);
    cudaFree(d_candidate_norm);
    cudaFree(d_control_mid);
    cudaFree(d_control_norm);
    cudaFree(d_candidate_quant);
    cudaFree(d_control_quant);
    cudaFree(d_candidate_scale);
    cudaFree(d_control_scale);
    return mid_equal && norm_equal && quant_equal && scale_equal;
}

}  // namespace

int main() {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major * 10 + properties.minor != 86) {
        std::fprintf(stderr, "requires SM86, found SM%d%d\n",
            properties.major, properties.minor);
        return 2;
    }
    bool passed = true;
    for (const uint32_t tokens : {9u, 128u, 449u, 512u}) {
        passed = run_shape(tokens) && passed;
    }
    Ready::Layout invalid_layout{};
    const auto zero = Dual::launch(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        Dual::kWidth, 0, invalid_layout, 1.0e-6f, 1.0e-6f, 86, 0);
    const auto wrong_sm = Dual::launch(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        Dual::kWidth, 128, invalid_layout, 1.0e-6f, 1.0e-6f, 80, 0);
    const bool fallback = zero == Dual::LaunchResult::NotSupported
        && wrong_sm == Dual::LaunchResult::NotSupported;
    std::printf("{\"fallback_zero_tokens\":%s,\"fallback_wrong_sm\":%s,"
        "\"passed\":%s}\n", zero == Dual::LaunchResult::NotSupported
            ? "true" : "false", wrong_sm == Dual::LaunchResult::NotSupported
            ? "true" : "false", passed && fallback ? "true" : "false");
    return passed && fallback ? 0 : 1;
}
