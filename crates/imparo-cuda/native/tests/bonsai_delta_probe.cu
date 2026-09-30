// Standalone numerical/state probe for the actual Bonsai DeltaNet geometry.
// This does not benchmark the model and is not a claim of whole-model admission.
#include "../gated_delta.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gd = imparo_cuda_gated_delta;
static void ck(cudaError_t rc) {
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(rc));
        std::exit(2);
    }
}
struct Device {
    float * p = nullptr;
    size_t n;
    explicit Device(size_t count) : n(count) { ck(cudaMalloc(&p, n * sizeof(float))); }
    explicit Device(const std::vector<float>& x) : Device(x.size()) {
        ck(cudaMemcpy(p, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~Device() { cudaFree(p); }
    std::vector<float> read() const {
        std::vector<float> x(n);
        ck(cudaMemcpy(x.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
        return x;
    }
};
static uint64_t rng = 981372;
static float random_float() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return float(rng >> 40) / 16777216.0f - 0.5f;
}
static std::vector<float> randoms(size_t n, float scale = 1) {
    std::vector<float> x(n);
    for (auto& v : x) v = random_float() * scale;
    return x;
}
static double sig(double x) {
    if (x >= 0) return 1 / (1 + std::exp(-x));
    const double e = std::exp(x); return e / (1 + e);
}
static bool close(const char * name, const float * got,
                  const std::vector<double>& ref, double tol = 2e-5) {
    double error = 0, scale = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) return false;
        error = std::max(error, std::abs(double(got[i]) - ref[i]));
        scale = std::max(scale, std::abs(ref[i]));
    }
    std::printf("%s max_abs=%.9g relative=%.9g\n", name, error,
                error / std::max(scale, 1e-9));
    return error <= tol * std::max(scale, 1e-3);
}
static bool guards(const std::vector<float>& x, size_t offset, size_t count) {
    for (size_t i = 0; i < x.size(); ++i)
        if ((i < offset || i >= offset + count) && x[i] != 9876.0f) return false;
    return true;
}
int main() {
    constexpr uint32_t kh = 16, vh = 48, d = 128, n = 9, snap_row = 3;
    constexpr uint32_t qw = 2 * kh * d + vh * d, vw = vh * d, se = vh * d * d;
    constexpr uint32_t in_off = 23, out_off = 37, snap_off = 51;
    constexpr float eps = 1e-6f;
    auto qkv = randoms(size_t(n) * qw), a = randoms(n * vh, 8), b = randoms(n * vh, 8);
    auto wa = randoms(vh), dt = randoms(vh);
    for (auto& x : wa) x = -std::abs(x) - 0.1f;
    // Exercise the L2 floor and the tiled [0..15,0..15,0..15] head mapping.
    std::fill(qkv.begin(), qkv.begin() + d, 0.0f);
    for (uint32_t i = 0; i < d; ++i) qkv[kh * d + i] *= 1e-9f;
    const auto initial = randoms(se, 0.1f);
    std::vector<float> sin(in_off + se + 17, 9876), sout(out_off + se + 17, 9876);
    std::vector<float> ssnap(snap_off + se + 17, 9876);
    std::copy(initial.begin(), initial.end(), sin.begin() + in_off);
    std::vector<double> state(initial.begin(), initial.end()), reference(n * vw), snap;
    for (uint32_t t = 0; t < n; ++t) {
        for (uint32_t h = 0; h < vh; ++h) {
            double q[d], k[d], qs = 0, ks = 0;
            const size_t kb = size_t(t) * qw + (h % kh) * d;
            for (uint32_t i = 0; i < d; ++i) {
                q[i] = qkv[kb + i]; k[i] = qkv[kb + kh * d + i];
                qs += q[i] * q[i]; ks += k[i] * k[i];
            }
            qs = std::max(std::sqrt(qs), double(eps));
            ks = std::max(std::sqrt(ks), double(eps));
            for (uint32_t i = 0; i < d; ++i) {
                q[i] = q[i] / qs / std::sqrt(double(d)); k[i] /= ks;
            }
            const double x = double(a[t * vh + h]) + dt[h];
            const double decay = std::exp(wa[h] * (x > 20 ? x : std::log1p(std::exp(x))));
            const double beta = sig(b[t * vh + h]);
            for (uint32_t j = 0; j < d; ++j) {
                const size_t s = (size_t(h) * d + j) * d;
                double remembered = 0;
                for (uint32_t i = 0; i < d; ++i) {
                    state[s + i] *= decay; remembered += state[s + i] * k[i];
                }
                const double delta = (qkv[size_t(t) * qw + 2 * kh * d + h * d + j] - remembered) * beta;
                double result = 0;
                for (uint32_t i = 0; i < d; ++i) {
                    state[s + i] += k[i] * delta; result += state[s + i] * q[i];
                }
                reference[size_t(t) * vw + h * d + j] = result;
            }
        }
        if (t + 1 == snap_row) snap = state;
    }
    Device dq(qkv), da(a), db(b), dwa(wa), ddt(dt), din(sin), dout(sout), dsnap(ssnap), result(n * vw);
    ck(gd::launch_delta(dq.p, da.p, db.p, dwa.p, ddt.p, din.p + in_off,
        dout.p + out_off, result.p, dsnap.p + snap_off, snap_row,
        kh, vh, d, d, n, eps, nullptr));
    ck(cudaDeviceSynchronize());
    const auto got = result.read(), final_state = dout.read(), captured = dsnap.read();
    bool ok = close("delta-output", got.data(), reference)
           && close("delta-final-state", final_state.data() + out_off, state)
           && close("delta-mid-snapshot", captured.data() + snap_off, snap)
           && din.read() == sin && guards(final_state, out_off, se)
           && guards(captured, snap_off, se);
    // Same operation broken into decode-sized calls must preserve exact f32 state
    // and outputs: no different reduction tree is introduced by token grouping.
    Device stepped(sin), step_result(n * vw);
    for (uint32_t t = 0; t < n; ++t)
        ck(gd::launch_delta(dq.p + size_t(t) * qw, da.p + t * vh, db.p + t * vh,
            dwa.p, ddt.p, stepped.p + in_off, stepped.p + in_off,
            step_result.p + size_t(t) * vw, nullptr, 0,
            kh, vh, d, d, 1, eps, nullptr));
    ck(cudaDeviceSynchronize());
    const auto stepped_state = stepped.read();
    const bool exact = step_result.read() == got
        && std::equal(stepped_state.begin() + in_off, stepped_state.begin() + in_off + se,
                      final_state.begin() + out_off) && guards(stepped_state, in_off, se);
    std::printf("decode-vs-batch exact=%s\n", exact ? "true" : "false");
    ok = ok && exact;
    // Real 10240-channel, four-tap convolution. Check the early snapshot (one token,
    // before history length three) as well as a nonzero source/destination offset.
    constexpr uint32_t taps = 4, ch = qw, hist = (taps - 1) * ch;
    auto convx = randoms(n * ch), convw = randoms(taps * ch), prev = randoms(hist);
    std::vector<float> convstate(11 + hist + 17, 9876), convnext(19 + hist + 17, 9876);
    std::vector<float> convsnap(29 + hist + 17, 9876);
    std::copy(prev.begin(), prev.end(), convstate.begin() + 11);
    auto value = [&](uint32_t e, uint32_t c) -> double {
        return e < taps - 1 ? prev[size_t(e) * ch + c] : convx[size_t(e - (taps - 1)) * ch + c];
    };
    std::vector<double> cref(n * ch), csref(hist), cnref(hist);
    for (uint32_t t = 0; t < n; ++t) for (uint32_t c = 0; c < ch; ++c) {
        double sum = 0;
        for (uint32_t k = 0; k < taps; ++k) sum += convw[c * taps + k] * value(t + k, c);
        cref[t * ch + c] = sum * sig(sum);
    }
    for (uint32_t s = 0; s < taps - 1; ++s) for (uint32_t c = 0; c < ch; ++c) {
        csref[s * ch + c] = value(1 + s, c); cnref[s * ch + c] = value(n + s, c);
    }
    Device cx(convx), cw(convw), ci(convstate), co(convnext), cp(convsnap), cy(n * ch);
    ck(gd::launch_plain_conv_snapshot(cx.p, ci.p + 11, cp.p + 29, ch, taps, 1, nullptr));
    ck(gd::launch_plain_conv(cx.p, cw.p, ci.p + 11, co.p + 19, cy.p, ch, taps, n, nullptr));
    ck(cudaDeviceSynchronize());
    const auto cs = cp.read(), cn = co.read(), cv = cy.read();
    ok = close("conv-output", cv.data(), cref) && close("conv-snapshot", cs.data() + 29, csref)
       && close("conv-state", cn.data() + 19, cnref) && ci.read() == convstate
       && guards(cs, 29, hist) && guards(cn, 19, hist) && ok;
    Device cstep(convstate), cstepout(n * ch);
    for (uint32_t t = 0; t < n; ++t)
        ck(gd::launch_plain_conv(cx.p + size_t(t) * ch, cw.p,
            cstep.p + 11, cstep.p + 11, cstepout.p + size_t(t) * ch,
            ch, taps, 1, nullptr));
    ck(cudaDeviceSynchronize());
    const auto cstepstate = cstep.read();
    const bool conv_exact = cstepout.read() == cv
        && std::equal(cstepstate.begin() + 11, cstepstate.begin() + 11 + hist,
                      cn.begin() + 19) && guards(cstepstate, 11, hist);
    std::printf("conv-decode-vs-batch exact=%s\n", conv_exact ? "true" : "false");
    ok = conv_exact && ok;
    constexpr uint32_t heads = 24, hd = 256;
    auto attn = randoms(heads * hd), packed = randoms(heads * hd * 2, 100);
    std::vector<double> gate_ref(attn.begin(), attn.end());
    for (uint32_t h = 0; h < heads; ++h) for (uint32_t c = 0; c < hd; ++c)
        gate_ref[h * hd + c] *= sig(packed[h * hd * 2 + hd + c]);
    Device ga(attn), gb(packed);
    ck(gd::launch_mul_sigmoid(ga.p, gb.p, hd, hd, 2 * hd, hd, heads, nullptr));
    ck(cudaDeviceSynchronize());
    const auto gate_got = ga.read();
    ok = close("strided-sigmoid", gate_got.data(), gate_ref) && ok;
    std::printf("bonsai-delta-native-probe %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
