// Independent CPU nearest-half oracle and pinned FD16 answers for F24 reduction.
#include "f16_compat.cuh"
#include "../tests/fixtures/upstream_numeric.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
using H = __half;
void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
struct Random {
    uint32_t state;
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};
uint32_t u32(const unsigned char *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
std::vector<uint16_t> fixture(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Missing fixture");
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), {});
    if (b.size() < 12 || std::memcmp(b.data(), "NRNUM001", 8))
        throw std::runtime_error("Bad fixture");
    size_t at = 12;
    for (uint32_t i = 0; i < u32(b.data() + 8); ++i) {
        if (at + 8 > b.size())
            throw std::runtime_error("Short fixture header");
        std::string tag(reinterpret_cast<const char *>(b.data() + at), 4);
        size_t length = u32(b.data() + at + 4);
        at += 8;
        if (length > b.size() - at)
            throw std::runtime_error("Short fixture section");
        if (tag == "FD16") {
            if (length != 32768)
                throw std::runtime_error("Unexpected FD16 count");
            std::vector<uint16_t> out(length / 2);
            for (size_t j = 0; j < out.size(); ++j)
                out[j] = uint16_t(b[at + 2 * j]) | (uint16_t(b[at + 2 * j + 1]) << 8);
            return out;
        }
        at += length;
    }
    throw std::runtime_error("Missing FD16 section");
}
double value(H h) {
    return num::f16ToF32(nr::halfBits(h));
}
H nearest(double x) {
    static const std::vector<double> grid = [] {
        std::vector<double> v(0x7c00);
        for (size_t i = 0; i < v.size(); ++i)
            v[i] = num::f16ToF32(uint16_t(i));
        return v;
    }();
    const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
    if (std::isnan(x))
        return nr::halfFromBits(sign | 0x7e00);
    const double mag = std::abs(x);
    if (mag >= 65520)
        return nr::halfFromBits(sign | 0x7c00);
    auto it = std::lower_bound(grid.begin(), grid.end(), mag);
    if (it == grid.end())
        return nr::halfFromBits(sign | 0x7bff);
    const size_t hi = it - grid.begin();
    if (!hi || *it == mag)
        return nr::halfFromBits(sign | uint16_t(hi));
    const size_t lo = hi - 1;
    const double down = mag - grid[lo], up = grid[hi] - mag;
    const size_t chosen = down < up ? lo : up < down ? hi : (lo % 2 ? hi : lo);
    return nr::halfFromBits(sign | uint16_t(chosen));
}
int exponent(double x) {
    int p;
    std::frexp(std::abs(x), &p);
    return std::max(p - 1, -14);
}
H reference(const H *a, int as, const H *b, int bs, H incoming) {
    const double initial = value(incoming);
    if (!std::isfinite(initial))
        return incoming;
    int maximum = initial == 0 ? -21 : exponent(initial);
    for (int i = 0; i < 8; ++i)
        if (value(a[i * as]) != 0 && value(b[i * bs]) != 0)
            maximum = std::max(maximum, exponent(value(a[i * as])) + exponent(value(b[i * bs])));
    const double alignment = std::ldexp(1., 24 - maximum);
    int64_t units = int64_t(std::trunc(initial * alignment));
    for (int i = 0; i < 8; ++i)
        units += int64_t(std::trunc(value(a[i * as]) * value(b[i * bs]) * alignment));
    return nearest(std::ldexp(double(units), maximum - 24));
}
H approximation(const H *a, const H *b, H incoming) {
    float sum = float(value(incoming));
    for (int i = 0; i < 8; ++i)
        sum = std::fma(float(value(a[i])), float(value(b[i])), sum);
    return nr::halfFromBits(num::f16Bits(sum));
}
size_t differences(const std::vector<H> &a, const std::vector<H> &b) {
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i)
        count += nr::halfBits(a[i]) != nr::halfBits(b[i]);
    return count;
}
void same(const std::vector<H> &a, const std::vector<H> &b, const char *name) {
    const size_t count = differences(a, b);
    if (count) {
        std::fprintf(stderr, "%s: %zu mismatches\n", name, count);
        throw std::runtime_error("F16 bit mismatch");
    }
}
// Complete pageable default-stream H2D before nonblocking-stream kernels.
template <class T> struct Device {
    T *data = nullptr;
    explicit Device(size_t n) {
        check(cudaMalloc(&data, n * sizeof(T)));
    }
    ~Device() {
        if (data)
            cudaFree(data);
    }
    void upload(const std::vector<T> &x) {
        check(cudaMemcpy(data, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
        check(cudaStreamSynchronize(nullptr));
    }
    void download(std::vector<T> &x) {
        check(cudaMemcpy(x.data(), data, x.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
};
__global__ void primitive(const H *in, H *out, H *approx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const H *p = in + i * 17;
    out[i] = nr::f16_compat_detail::step8(p, 1, p + 8, 1, p[16]);
    float sum = nr::halfToFloat(p[16]);
    for (int j = 0; j < 8; ++j)
        sum = __fmaf_rn(nr::halfToFloat(p[j]), nr::halfToFloat(p[8 + j]), sum);
    approx[i] = nr::halfRne(sum);
}
struct Fixed {
    int units, binary_exponent;
};
__global__ void conversions(const Fixed *in, H *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = nr::f16_compat_detail::fixed_to_half(in[i].units, in[i].binary_exponent);
}
size_t conversion_cases(bool gpu, cudaStream_t stream) {
    std::vector<Fixed> input;
    constexpr int limit = (1 << 30) - 1;
    for (int e = -45; e <= 6; ++e) {
        for (int v : {0, 1, 2, 3, 7, 31, 63, 127, 1023, 1024, (1 << 24) - 1, 1 << 24, (1 << 24) + 1,
                      limit}) {
            input.push_back({v, e});
            if (v)
                input.push_back({-v, e});
        }
        // Every finite half rounding boundary, including zero/subnormal, the
        // normal transition and overflow. Test the nearest integer units on
        // both sides and the midpoint itself whenever it is representable.
        for (unsigned h = 0; h < 0x7c00; ++h) {
            const double lower = value(nr::halfFromBits(uint16_t(h)));
            const double upper = h == 0x7bff ? 65536. : value(nr::halfFromBits(uint16_t(h + 1)));
            const double midpoint = std::ldexp((lower + upper) * .5, -e);
            if (midpoint < 1 || midpoint > limit - 2)
                continue;
            const int center = int(std::floor(midpoint));
            for (int d = -1; d <= 1; ++d) {
                int v = center + d;
                input.push_back({v, e});
                if (v)
                    input.push_back({-v, e});
            }
        }
    }
    Random rng{0xf24c0deu};
    for (int i = 0; i < 65536; ++i) {
        const int v = int(rng.next() & limit);
        input.push_back({(rng.next() & 1) ? v : -v, int(rng.next() % 52) - 45});
    }
    std::vector<H> expected(input.size()), actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const Fixed x = input[i];
        expected[i] = nearest(std::ldexp(double(x.units), x.binary_exponent));
        actual[i] = nr::f16_compat_detail::fixed_to_half(x.units, x.binary_exponent);
    }
    same(actual, expected, "CPU round-to-odd conversion vs nearest-half oracle");
    if (gpu) {
        Device<Fixed> di(input.size());
        Device<H> dout(input.size());
        di.upload(input);
        conversions<<<unsigned((input.size() + 255) / 256), 256, 0, stream>>>(di.data, dout.data,
                                                                              int(input.size()));
        check(cudaGetLastError());
        check(cudaStreamSynchronize(stream));
        dout.download(actual);
        same(actual, expected, "GPU round-to-odd conversion vs nearest-half oracle");
    }
    return input.size();
}
int matrices(bool gpu, cudaStream_t stream) {
    Random random{0xf16c0deu};
    int count = 0;
    struct Shape {
        int M, K, N;
    };
    for (Shape s : {Shape{17, 16, 32}, Shape{19, 32, 4}, Shape{3, 8, 5}})
        for (bool quant : {false, true}) {
            std::vector<H> a(s.M * s.K), w(s.K * s.N), expected(s.M * s.N),
                expected_q(expected.size()), actual(expected.size()), actual_q(expected.size());
            for (H &x : a)
                x = nearest((int(random.next() % 16385) - 8192) / 4096.);
            for (H &x : w)
                x = nearest((int(random.next() % 16385) - 8192) / 4096.);
            for (int m = 0; m < s.M; ++m)
                for (int n = 0; n < s.N; ++n) {
                    H sum = nearest(0);
                    for (int k = 0; k < s.K; k += 8)
                        sum =
                            reference(a.data() + m * s.K + k, 1, w.data() + k * s.N + n, s.N, sum);
                    expected[m * s.N + n] = sum;
                    expected_q[m * s.N + n] =
                        nr::halfRne(num::e4m3ToF32(num::e4m3FromF16Bits(nr::halfBits(sum))));
                }
            if (gpu) {
                Device<H> da(a.size()), dw(w.size()), dout(actual.size()), dq(actual_q.size());
                da.upload(a);
                dw.upload(w);
                auto launch = [&] {
                    nr::nr_f16_compat(da.data, dw.data, dout.data, quant ? dq.data : nullptr, s.M,
                                      s.K, s.N, stream);
                };
                check(cudaMemsetAsync(dout.data, 0xff, actual.size() * sizeof(H), stream));
                check(cudaMemsetAsync(dq.data, 0xff, actual_q.size() * sizeof(H), stream));
                launch();
                check(cudaStreamSynchronize(stream));
                dout.download(actual);
                same(actual, expected, "Matrix half");
                if (quant) {
                    dq.download(actual_q);
                    same(actual_q, expected_q, "Matrix E4");
                }
                if (count == 0) {
                    cudaGraph_t graph;
                    cudaGraphExec_t exe;
                    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                    launch();
                    check(cudaStreamEndCapture(stream, &graph));
                    check(cudaGraphInstantiate(&exe, graph, 0));
                    check(cudaMemsetAsync(dout.data, 0xff, actual.size() * sizeof(H), stream));
                    check(cudaGraphLaunch(exe, stream));
                    check(cudaStreamSynchronize(stream));
                    dout.download(actual);
                    same(actual, expected, "Graph replay");
                    check(cudaGraphExecDestroy(exe));
                    check(cudaGraphDestroy(graph));
                }
            }
            ++count;
        }
    return count;
}
} // namespace
int main(int argc, char **argv) {
    try {
        bool host = false;
        std::string path = "tests/fixtures/numerics.bin";
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--host-only")
                host = true;
            else if (std::string(argv[i]) == "--fixture" && i + 1 < argc)
                path = argv[++i];
            else
                throw std::runtime_error("Usage: nr-f24-test [--host-only] [--fixture PATH]");
        }
        auto bits = fixture(path);
        Random random{0xc2b2ae35u};
        const size_t total = bits.size() + 30;
        std::vector<H> in(total * 17), expected(total), cpu(total), approx(total), actual(total),
            gpu_approx(total);
        for (size_t i = 0; i < bits.size(); ++i) {
            for (int j = 0; j < 17; ++j) {
                uint16_t b = random.next();
                if ((b & 0x7c00) == 0x7c00)
                    b &= 0x7bff;
                in[i * 17 + j] = nr::halfFromBits(b);
            }
            expected[i] = nr::halfFromBits(bits[i]);
        }
        // Exact cancellation cases: large + half-an-FP32-ULP - large. F24 retains
        // the tiny term; sequential FP32 FMAs lose it at the tie-to-even addition.
        for (int exponent = 0; exponent < 15; ++exponent)
            for (int sign : {1, -1}) {
                const size_t i = bits.size() + exponent * 2 + (sign < 0);
                for (int j = 0; j < 17; ++j)
                    in[i * 17 + j] = nearest(0);
                const int large_a = exponent / 2, large_b = exponent - large_a;
                const int small_a = (exponent - 24) / 2, small_b = exponent - 24 - small_a;
                in[i * 17] = nearest(sign * std::ldexp(1., large_a));
                in[i * 17 + 8] = nearest(std::ldexp(1., large_b));
                in[i * 17 + 1] = nearest(sign * std::ldexp(1., small_a));
                in[i * 17 + 9] = nearest(std::ldexp(1., small_b));
                in[i * 17 + 2] = nearest(-sign * std::ldexp(1., large_a));
                in[i * 17 + 10] = nearest(std::ldexp(1., large_b));
                expected[i] = nearest(sign * std::ldexp(1., exponent - 24));
            }
        for (size_t i = 0; i < total; ++i) {
            cpu[i] = reference(in.data() + i * 17, 1, in.data() + i * 17 + 8, 1, in[i * 17 + 16]);
            approx[i] = approximation(in.data() + i * 17, in.data() + i * 17 + 8, in[i * 17 + 16]);
        }
        same(cpu, expected, "CPU oracle vs pinned FD16");
        cudaStream_t stream = nullptr;
        if (!host) {
            cudaDeviceProp p{};
            check(cudaGetDeviceProperties(&p, 0));
            if (p.major != 7 || p.minor != 5)
                throw std::runtime_error("Test requires SM75");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            Device<H> di(in.size()), dout(actual.size()), da(approx.size());
            di.upload(in);
            primitive<<<unsigned((total + 255) / 256), 256, 0, stream>>>(di.data, dout.data,
                                                                         da.data, int(total));
            check(cudaGetLastError());
            check(cudaStreamSynchronize(stream));
            dout.download(actual);
            da.download(gpu_approx);
            same(actual, expected, "GPU vs pinned FD16 and cancellation cases");
            same(gpu_approx, approx, "Current approximate K8 CPU/GPU agreement");
        }
        const size_t convert_count = conversion_cases(!host, stream);
        const int count = matrices(!host, stream);
        if (stream)
            check(cudaStreamDestroy(stream));
        size_t fixture_differences = 0, cancellation_differences = 0;
        for (size_t i = 0; i < total; ++i)
            if (nr::halfBits(approx[i]) != nr::halfBits(expected[i])) {
                if (i < bits.size())
                    ++fixture_differences;
                else
                    ++cancellation_differences;
            }
        std::printf("{\"status\":\"pass\",\"gpu_executed\":%s,\"pinned_fd16_cases\":%zu,\"current_"
                    "approximation_fixture_mismatches\":%zu,\"cancellation_cases\":30,\"current_"
                    "approximation_cancellation_mismatches\":%zu,\"conversion_boundary_cases\":%zu,"
                    "\"matrix_cases\":%d,\"graph_replay_checked\":%s}\n",
                    host ? "false" : "true", bits.size(), fixture_differences,
                    cancellation_differences, convert_count, count, host ? "false" : "true");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "F16 COMPAT TEST FAILED: %s\n", e.what());
        return 1;
    }
}
