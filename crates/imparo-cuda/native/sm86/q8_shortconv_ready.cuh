#pragma once

// Include after the native MMQ header and lfm2_ops.cuh. The generic runtime is
// included at translation-unit scope so CUDA library declarations stay there.
namespace imparo_sm86_q8_shortconv_ready {

constexpr unsigned kWidth = 2048;
constexpr unsigned kTokens = 54;
constexpr unsigned kKernel = 3;
constexpr unsigned kGroups = 8;
constexpr unsigned kProducerTasks = 48;
constexpr unsigned kConsumerTasks = 432;
constexpr unsigned kWorkers = 30;
constexpr unsigned kThreads = 256;
constexpr unsigned kSharedBytes = imparo_sm80_q8_mmq::shared_bytes<128>();
using Queue = imparo_device_ready_tasks::Queue<kGroups>;
static_assert(kSharedBytes == 75776, "retained MMQ tile shared storage");
static_assert(sizeof(Queue) == 19 * sizeof(unsigned), "retained ready queue layout");

struct Policy {
    const uint8_t * w;
    const BlockQ8_1Mmq * q;
    float * bcx;
    const float * cw;
    const float * state;
    float * out;

    __device__ __forceinline__ void produce(unsigned id) const {
        const unsigned group = id / 6;
        const unsigned part = (id % 6) / 2;
        const unsigned half = id % 2;
        const unsigned logical = part * 16 + group * 2 + half;
        imparo_sm80_q8_mmq::q8_0_q8_1_mma_tile<
            128, true, false, 0, false, 8, 4, 128, true>(
                w, q, bcx, nullptr, 2048, 6144, 54, 6144, 0,
                0, 0, nullptr, 0, logical);
    }

    __device__ __forceinline__ void consume(unsigned id, unsigned tid) const {
        const unsigned token = id / 8;
        const unsigned channel = (id % 8) * 256 + tid;
        float sum = 0.f;
        for (unsigned tap = 0; tap < 3; tap++) {
            sum += cw[channel * 3 + tap]
                * imparo_cuda_lfm2::signal_at(
                    state, bcx, uint64_t(token) + tap, channel, 2048, 2);
        }
        out[uint64_t(token) * 2048 + channel] =
            bcx[uint64_t(token) * 6144 + 2048 + channel] * sum;
    }
};

// Host dispatch admits only the measured SM86/30-worker/M54/W2048/kernel3 case,
// clears Queue on the same stream, and retains the original state kernel after
// this kernel completes. This kernel does not advance or own recurrent state.
__global__ __launch_bounds__(256, 1) void kernel(
        const uint8_t * w, const BlockQ8_1Mmq * q, float * bcx,
        const float * cw, const float * state, float * out, Queue * queue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned tid = threadIdx.y * 32 + threadIdx.x;
    const Policy policy{w, q, bcx, cw, state, out};
    imparo_device_ready_tasks::run<kGroups, kProducerTasks, kTokens>(
        queue, tid, policy);
#else
    (void)w; (void)q; (void)bcx; (void)cw;
    (void)state; (void)out; (void)queue;
#endif
}

} // namespace imparo_sm86_q8_shortconv_ready
