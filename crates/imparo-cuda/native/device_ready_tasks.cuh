#pragma once

#include <cuda/atomic>

// Include this header outside implementation namespaces. The owner supplies
// zeroed queue storage, block geometry and operator policy for each invocation.
namespace imparo_device_ready_tasks {

template <unsigned GroupCount>
struct Queue {
    unsigned next_producer;
    unsigned ready_mask;
    unsigned producer_count[GroupCount];
    unsigned next_token[GroupCount];
    unsigned claimed;
};

__device__ __forceinline__
cuda::atomic_ref<unsigned, cuda::thread_scope_device> atomic(unsigned & value) {
    return cuda::atomic_ref<unsigned, cuda::thread_scope_device>(value);
}

// Producer IDs are contiguous within dependency groups. Each group's consumers
// become eligible only after every producer's output stores have been published.
// Policy::produce(id) and Policy::consume(id, tid) are collective CTA operations;
// consumer IDs are item * GroupCount + group. Every thread must enter this loop.
template <unsigned GroupCount, unsigned ProducerCount,
          unsigned ConsumersPerGroup, class Policy>
__device__ __forceinline__ void run(
        Queue<GroupCount> * queue, unsigned tid, const Policy & policy) {
    static_assert(GroupCount > 0 && GroupCount <= 32, "ready mask width");
    static_assert(ProducerCount > 0 && ProducerCount % GroupCount == 0,
                  "complete producer groups");
    static_assert(ConsumersPerGroup > 0, "nonempty consumer groups");
    constexpr unsigned producers_per_group = ProducerCount / GroupCount;
    constexpr unsigned consumer_count = GroupCount * ConsumersPerGroup;
    __shared__ unsigned type, task;
    bool producers_exhausted = false;
    for (;;) {
        if (tid == 0) {
            type = 2;
            if (!producers_exhausted) {
                const unsigned producer = atomic(queue->next_producer).fetch_add(
                    1, cuda::memory_order_relaxed);
                if (producer < ProducerCount) {
                    type = 0;
                    task = producer;
                } else {
                    producers_exhausted = true;
                }
            }
            if (producers_exhausted) {
                for (;;) {
                    if (atomic(queue->claimed).load(cuda::memory_order_relaxed)
                            == consumer_count) {
                        type = 2;
                        break;
                    }
                    const unsigned mask = atomic(queue->ready_mask).load(
                        cuda::memory_order_acquire);
                    if (!mask) {
                        __nanosleep(64);
                        continue;
                    }
                    const unsigned group = __ffs(mask) - 1;
                    const unsigned token = atomic(queue->next_token[group]).fetch_add(
                        1, cuda::memory_order_relaxed);
                    if (token < ConsumersPerGroup) {
                        task = token * GroupCount + group;
                        type = 1;
                        atomic(queue->claimed).fetch_add(1, cuda::memory_order_relaxed);
                        if (token == ConsumersPerGroup - 1) {
                            atomic(queue->ready_mask).fetch_and(
                                ~(1u << group), cuda::memory_order_relaxed);
                        }
                        break;
                    }
                    atomic(queue->ready_mask).fetch_and(
                        ~(1u << group), cuda::memory_order_relaxed);
                }
            }
        }
        __syncthreads();
        if (type == 2) return;
        const unsigned id = task;
        if (type == 0) {
            policy.produce(id);
            // Every output-owning thread fences before lane zero publishes the
            // CTA's completion; the counter's acquire chain joins the producers.
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                const unsigned group = id / producers_per_group;
                const unsigned done = atomic(queue->producer_count[group]).fetch_add(
                    1, cuda::memory_order_acq_rel);
                if (done == producers_per_group - 1) {
                    atomic(queue->ready_mask).fetch_or(
                        1u << group, cuda::memory_order_release);
                }
            }
        } else {
            policy.consume(id, tid);
        }
        __syncthreads();
    }
}

} // namespace imparo_device_ready_tasks
