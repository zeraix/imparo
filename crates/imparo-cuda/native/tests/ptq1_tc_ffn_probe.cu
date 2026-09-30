// A single real-shaped FFN experiment: only its three matrix providers change.
// Inputs/weights/signs come from the frozen root-owned fixture directory. The
// current fixture is model-derived embedding + FFN norm, not captured FFN state.
// This program does NOT claim model-quality admission; independent FP64 oracle
// checks consume its captured stage files after execution.
#include "../ptq1_tc.cuh"
#include "../weight_basis.cuh"
#include "../lfm2_ops.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr uint32_t K = 5120, F = 17408, M = 128;
constexpr float kGuard = 1234567.0f;

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
static void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> struct Device {
    T *p = nullptr;
    size_t count;
    explicit Device(size_t n): count(n) {
        check(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)));
    }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    void put(const std::vector<T> &v) {
        require(v.size() <= count, "upload size");
        check(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get(size_t n) const {
        require(n <= count, "download size");
        std::vector<T> out(n);
        check(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
};
template<class T> static std::vector<T> read(const fs::path &p, size_t n) {
    std::ifstream file(p, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(n * sizeof(T)))
        throw std::runtime_error("missing/wrong-sized fixture: " + p.string());
    file.seekg(0);
    std::vector<T> data(n);
    file.read(reinterpret_cast<char *>(data.data()), std::streamsize(n * sizeof(T)));
    if (!file) throw std::runtime_error("fixture read failed: " + p.string());
    return data;
}
static void write(const fs::path &p, const std::vector<float> &v) {
    std::ofstream file(p, std::ios::binary);
    file.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(float)));
    if (!file) throw std::runtime_error("output write failed: " + p.string());
}
static std::vector<int8_t> signs(const fs::path &p, size_t n) {
    auto f = read<float>(p, n);
    std::vector<int8_t> result(n);
    for (size_t i = 0; i < n; ++i) {
        require(f[i] == 1.0f || f[i] == -1.0f, "basis sign is not +/-1");
        result[i] = int8_t(f[i]);
    }
    return result;
}
static void finite(const std::vector<float> &v, const char *stage) {
    for (float x: v) if (!std::isfinite(x))
        throw std::runtime_error(std::string("nonfinite ") + stage);
}
static void half_range(const std::vector<float> &v, const char *stage) {
    finite(v, stage);
    for (float x: v) if (std::abs(x) > 65504.0f)
        throw std::runtime_error(std::string("half overflow ") + stage);
}

struct Ffn {
    Device<uint8_t> wg, wu, wd;
    Device<int8_t> sk, sf;
    Device<float> input, basis, gate, up, output;
    explicit Ffn(const fs::path &dir):
        wg(size_t(F) * (K / 128) * 28), wu(wg.count), wd(size_t(K) * (F / 128) * 28),
        sk(K), sf(F), input(size_t(M) * K), basis(size_t(M) * F),
        gate(size_t(M) * F), up(gate.count), output(size_t(M) * (K + 64) + 64) {
        wg.put(read<uint8_t>(dir / "gate.ptq.bin", wg.count));
        wu.put(read<uint8_t>(dir / "up.ptq.bin", wu.count));
        wd.put(read<uint8_t>(dir / "down.ptq.bin", wd.count));
        sk.put(signs(dir / "signs5120.f32.bin", K));
        sf.put(signs(dir / "signs17408.f32.bin", F));
        auto x = read<float>(dir / "input.f32.bin", input.count);
        finite(x, "input"); input.put(x);
    }
    void matrix(bool tc, const Device<uint8_t> &w, float *out,
                uint32_t k, uint32_t n, uint32_t m,
                uint32_t stride, uint32_t row_base) {
        if (tc) check(imparo_cuda_ptq1_tc::launch(
            w.p, basis.p, out, k, n, m, stride, row_base, nullptr));
        else {
            require(stride == n && row_base == 0, "baseline output must be contiguous");
            check(imparo_cuda_ptq1::launch_ptq1_matmat(w.p, basis.p, out, k, n, m, nullptr));
        }
    }
    void capture(const fs::path &dir, const std::string &prefix,
                 const char *name, const Device<float> &buffer, size_t count) {
        auto v = buffer.get(count);
        finite(v, name);
        write(dir / (prefix + "." + name + ".f32.bin"), v);
    }
    // Empty directory means the timed chain: no host readbacks, no extra copies.
    void run(bool tc, uint32_t m, uint32_t out_stride, uint32_t row_base,
             const fs::path &dump_dir = {}, const std::string &prefix = {}) {
        check(imparo_cuda_weight_basis::launch(input.p, basis.p, sk.p,
            K, 1024, m, false, 0, 0, 0, nullptr));
        if (!dump_dir.empty()) half_range(basis.get(size_t(m) * K), "gate input");
        matrix(tc, wg, gate.p, K, F, m, F, 0);
        if (!dump_dir.empty()) capture(dump_dir, prefix, "gate", gate, size_t(m) * F);

        // Repeat the source transform just as the current runtime does. Sharing
        // it would be another optimization and confound this matrix-only A/B.
        check(imparo_cuda_weight_basis::launch(input.p, basis.p, sk.p,
            K, 1024, m, false, 0, 0, 0, nullptr));
        matrix(tc, wu, up.p, K, F, m, F, 0);
        if (!dump_dir.empty()) capture(dump_dir, prefix, "up", up, size_t(m) * F);
        imparo_cuda_lfm2::silu_mul_kernel<<<(m * F + 255) / 256, 256>>>(gate.p, up.p, m * F);
        check(cudaGetLastError());
        if (!dump_dir.empty()) capture(dump_dir, prefix, "act", gate, size_t(m) * F);
        check(imparo_cuda_weight_basis::launch(gate.p, basis.p, sf.p,
            F, 1024, m, false, 0, 0, 0, nullptr));
        if (!dump_dir.empty()) {
            auto v = basis.get(size_t(m) * F);
            half_range(v, "down input");
            write(dump_dir / (prefix + ".down-input.f32.bin"), v);
        }
        matrix(tc, wd, output.p, F, K, m, out_stride, row_base);
    }
    std::vector<float> checked_output(uint32_t m, uint32_t stride, uint32_t row_base) {
        auto raw = output.get(output.count);
        std::vector<float> out(size_t(m) * K);
        for (size_t i = 0; i < raw.size(); ++i) {
            const size_t row = i / stride, column = i % stride;
            const bool valid = row < m && column >= row_base && column < row_base + K;
            if (valid) out[row * K + column - row_base] = raw[i];
            else require(std::memcmp(&raw[i], &kGuard, sizeof(float)) == 0, "output guard overwritten");
        }
        finite(out, "output");
        return out;
    }
    std::vector<float> correctness(bool tc, uint32_t m, const fs::path &dir) {
        output.put(std::vector<float>(output.count, kGuard));
        const uint32_t stride = tc && m == 115 ? K + 64 : K;
        const uint32_t base = tc && m == 115 ? 32 : 0;
        const std::string prefix = tc ? "candidate" : "baseline";
        run(tc, m, stride, base, m == M ? dir : fs::path{}, prefix);
        auto out = checked_output(m, stride, base);
        write(dir / (prefix + (m == M ? ".out.f32.bin" : ".m115.out.f32.bin")), out);
        return out;
    }
    float timed(bool tc) {
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        check(cudaEventRecord(start));
        run(tc, M, K, 0);
        check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
        float ms = 0.0f; check(cudaEventElapsedTime(&ms, start, stop));
        check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
        return ms;
    }
};
struct Difference {
    double max_abs = 0, relative_l2 = 0, max_over_rms = 0, cosine = 0;
};
static Difference difference(const std::vector<float> &a, const std::vector<float> &b) {
    require(a.size() == b.size(), "comparison shape");
    double aa = 0, bb = 0, dd = 0, ab = 0;
    Difference r;
    for (size_t i = 0; i < a.size(); ++i) {
        const double av = a[i], bv = b[i], d = av - bv;
        r.max_abs = std::max(r.max_abs, std::abs(d));
        aa += av * av; bb += bv * bv; dd += d * d; ab += av * bv;
    }
    r.relative_l2 = std::sqrt(dd / std::max(aa, 1e-300));
    r.max_over_rms = r.max_abs / std::sqrt(std::max(aa / a.size(), 1e-300));
    r.cosine = ab / std::sqrt(std::max(aa * bb, 1e-300));
    return r;
}
static void write_difference(std::ostream &f, const Difference &r) {
    f << "{\"max_abs\":" << r.max_abs << ",\"relative_l2\":" << r.relative_l2
      << ",\"max_over_rms\":" << r.max_over_rms << ",\"cosine\":" << r.cosine << "}";
}
int main(int argc, char **argv) {
    try {
        require(argc == 3, "usage: ptq1_tc_ffn_probe.exe DATA_DIR OUTPUT_DIR");
        const fs::path data = argv[1], out = argv[2];
        fs::create_directories(out);
        int device = 0; check(cudaGetDevice(&device));
        cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props, device));
        require(props.major >= 8, "SM80+ required by half/F32 MMA contract");
        Ffn ffn(data);
        require(imparo_cuda_ptq1_tc::launch(ffn.wg.p, ffn.basis.p, ffn.gate.p,
            K, F, 1, F, 0, nullptr) == cudaErrorNotSupported, "M1 must retain fallback");
        const auto a = ffn.correctness(false, M, out);
        const auto b = ffn.correctness(true, M, out);
        // One fixed configuration; one warmup then one measured complete FFN per
        // route. This is an initial small-graph observation, not stable service.
        ffn.run(false, M, K, 0); ffn.run(true, M, K, 0);
        check(cudaDeviceSynchronize());
        const float ams = ffn.timed(false), bms = ffn.timed(true);
        const auto a_tail = ffn.correctness(false, 115, out);
        const auto b_tail = ffn.correctness(true, 115, out);
        require(std::memcmp(a.data(), a_tail.data(), a_tail.size() * sizeof(float)) == 0,
            "baseline M115 prefix differs from M128");
        require(std::memcmp(b.data(), b_tail.data(), b_tail.size() * sizeof(float)) == 0,
            "candidate M115/strided prefix differs from M128");
        std::ofstream result(out / "result.json");
        result << std::setprecision(12)
          << "{\n  \"scope\":\"one complete FFN, not model quality or service admission\",\n"
          << "  \"input_origin\":\"model-derived embedding inverse-basis + FFN norm surrogate; not captured FFN activation\",\n"
          << "  \"tile\":[32,64,128],\n  \"timed_rows\":128,\n  \"correctness_tail_rows\":115,\n"
          << "  \"warmups_per_route\":1,\n  \"timed_iterations_per_route\":1,\n"
          << "  \"baseline_ms\":" << ams << ",\n  \"candidate_ms\":" << bms
          << ",\n  \"speedup\":" << ams / bms << ",\n"
          << "  \"finite_and_output_guards\":true,\n  \"m1_rejected_without_launch\":true,\n"
          << "  \"m115_strided_and_prefix_bitwise\":true,\n  \"observed_candidate_vs_baseline_m128\":";
        write_difference(result, difference(a, b));
        result << ",\n  \"observed_candidate_vs_baseline_m115\":";
        write_difference(result, difference(a_tail, b_tail));
        result << ",\n  \"independent_oracle_admission\":\"PENDING external frozen FP64 checks\"\n}\n";
        require(bool(result), "result write failed");
        std::printf("FFN M128 baseline=%.6f ms candidate=%.6f ms speedup=%.6fx\n", ams, bms, ams / bms);
        std::puts("Finite/guard/M115-prefix checks passed; independent FP64 and whole-model admission PENDING.");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
