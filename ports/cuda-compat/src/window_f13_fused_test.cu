// Fused window tests against an independent CPU oracle and staged attention.
#include "window_f13_fused.cuh"
#include "nr_cuda.h"
#include "../tests/fixtures/upstream_numeric.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <vector>

namespace {
using H = __half;
void check(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(s));
}
float value(H x) {
    return num::f16ToF32(nr::halfBits(x));
}
H rne(float x) {
    return nr::halfFromBits(num::f16Bits(x));
}
H rtz(float x) {
    uint16_t bits = num::f16Bits(x);
    if (std::abs(num::f16ToF32(bits)) > std::abs(x))
        --bits;
    return nr::halfFromBits(bits);
}
H add(H a, H b) {
    return rne(value(a) + value(b));
}
H mul(H a, H b) {
    return rne(value(a) * value(b));
}
H publish(H h) {
    return rne(num::e4m3ToF32(num::e4m3FromF16Bits(nr::halfBits(h))));
}
H exponential(H h) {
    const H affine = rne(std::fma(value(h), .044921875f, 1.30078125f));
    const float bounded =
        std::max(num::f16ToF32(0x3c20), std::min(num::f16ToF32(0x3e47), value(affine)));
    return nr::halfFromBits(uint16_t((uint32_t(num::f16Bits(bounded)) << 5) ^ 0x8000u));
}
int exponent(double x, int floor) {
    int p = 0;
    std::frexp(std::abs(x), &p);
    return std::max(p - 1, floor);
}
H dot16(const H *a, int as, const H *b, int bs, H incoming) {
    const double initial = value(incoming);
    if (!std::isfinite(initial))
        return incoming;
    int maximum = initial == 0 ? -21 : exponent(initial, -14);
    for (int i = 0; i < 16; ++i) {
        const double x = value(a[i * as]), y = value(b[i * bs]);
        if (x && y)
            maximum = std::max(maximum, exponent(x, -6) + exponent(y, -6));
    }
    const double alignment = std::ldexp(1., 13 - maximum);
    int64_t units = int64_t(std::trunc(initial * alignment));
    for (int i = 0; i < 16; ++i)
        units += int64_t(std::trunc(double(value(a[i * as])) * value(b[i * bs]) * alignment));
    const H expected = rne(float(std::ldexp(double(units), maximum - 13)));
    uint32_t a_compact[16], b_compact[16];
    for (int i = 0; i < 16; ++i) {
        a_compact[i] = nr_window_f13_detail::encode(a[i * as]);
        b_compact[i] = nr_window_f13_detail::encode(b[i * bs]);
    }
    const H compact =
        rne(nr_window_f13_detail::compact_step(a_compact, 1, b_compact, 1, value(incoming)));
    if (nr::halfBits(expected) != nr::halfBits(compact))
        throw std::runtime_error("Independent window dot oracle mismatch");
    return expected;
}
H inverse(const H *a) {
    H sums[16];
    for (int i = 0; i < 16; ++i) {
        const volatile float product = value(a[i]) * value(a[i]);
        sums[i] = rtz(product + value(mul(a[i + 16], a[i + 16])));
    }
    for (int stride = 8; stride; stride /= 2)
        for (int i = 0; i < stride; ++i)
            sums[i] = add(sums[i], sums[i + stride]);
    return rtz(float(1. / std::sqrt(double(value(sums[0])))));
}
H total64(const H *e) {
    H t[8];
    for (int g = 0; g < 8; ++g)
        t[g] =
            add(add(add(add(e[g], e[g + 8]), add(e[g + 16], e[g + 24])), add(e[g + 32], e[g + 40])),
                add(e[g + 48], e[g + 56]));
    return add(add(add(add(t[0], t[2]), t[4]), t[6]), add(add(add(t[1], t[3]), t[5]), t[7]));
}
int physical(int t) {
    return ((t >> 3) >> 2) * 32 + ((t & 7) >> 2) * 16 + ((t >> 3) & 3) * 4 + (t & 3);
}
struct Shape {
    int w, h, heads, phase;
};
struct Input {
    Shape shape;
    std::vector<H> qkv, prior, expected;
    std::vector<float> scales;
};
struct Random {
    uint32_t x = 0xf130cafe;
    uint32_t next() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return x;
    }
};
Input random_input(Shape s, bool zero) {
    Input in{s};
    in.qkv.resize(size_t(s.w) * s.h * s.heads * 96);
    in.prior.resize(s.heads * 4096);
    in.scales.resize(s.heads);
    Random r;
    for (H &h : in.qkv)
        h = rne(zero ? 0.f : (int(r.next() % 65535) - 32767) / 32768.f);
    for (H &h : in.prior)
        h = rne((int(r.next() % 32767) - 16383) / 65536.f);
    for (int h = 0; h < s.heads; ++h)
        in.scales[h] = .5f + (r.next() % 100003) / 65536.f;
    // Include valid zero norms separately from padded tokens.
    for (int i = 0; i < s.heads * 96; ++i)
        in.qkv[i] = rne(0.f);
    return in;
}
std::vector<H> reference(const Input &in) {
    const auto s = in.shape;
    const int sx = s.phase == 1 || s.phase == 2 ? 4 : 0, sy = s.phase == 1 || s.phase == 3 ? 4 : 0;
    std::vector<H> out(size_t(s.w) * s.h * s.heads * 32);
    for (int wy = 0; wy < (s.h + sy + 7) / 8; ++wy)
        for (int wx = 0; wx < (s.w + sx + 7) / 8; ++wx)
            for (int head = 0; head < s.heads; ++head) {
                H q[2048], k[2048], v[2048];
                std::fill_n(q, 2048, rne(0.f));
                std::fill_n(k, 2048, rne(0.f));
                std::fill_n(v, 2048, rne(0.f));
                const H scale = rtz(in.scales[head]);
                for (int t = 0; t < 64; ++t) {
                    const int x = wx * 8 - sx + (t & 7), y = wy * 8 - sy + (t >> 3);
                    if (x < 0 || y < 0 || x >= s.w || y >= s.h)
                        continue;
                    const H *a = in.qkv.data() + ((size_t(y) * s.w + x) * s.heads + head) * 96;
                    const H qi = inverse(a), ki = inverse(a + 32);
                    const int p = physical(t);
                    for (int c = 0; c < 32; ++c) {
                        q[t * 32 + c] = publish(mul(mul(a[c], qi), scale));
                        k[p * 32 + c] = publish(mul(a[32 + c], ki));
                        v[p * 32 + c] = publish(a[64 + c]);
                    }
                }
                for (int row = 0; row < 64; ++row) {
                    const int x = wx * 8 - sx + (row & 7), y = wy * 8 - sy + (row >> 3);
                    if (x < 0 || y < 0 || x >= s.w || y >= s.h)
                        continue;
                    H e[64], p[64];
                    for (int col = 0; col < 64; ++col) {
                        H sum = in.prior[head * 4096 + row * 64 + col];
                        for (int c = 0; c < 32; c += 16)
                            sum = dot16(q + row * 32 + c, 1, k + col * 32 + c, 1, sum);
                        e[col] = exponential(sum);
                    }
                    const H reciprocal = rtz(1.f / value(total64(e)));
                    for (int j = 0; j < 64; ++j)
                        p[j] = publish(mul(e[j], reciprocal));
                    for (int c = 0; c < 32; ++c) {
                        H sum = rne(0.f);
                        for (int j = 0; j < 64; j += 16)
                            sum = dot16(p + j, 1, v + j * 32 + c, 32, sum);
                        out[((size_t(y) * s.w + x) * s.heads + head) * 32 + c] = publish(sum);
                    }
                }
            }
    return out;
}
void equal(const std::vector<H> &actual, const std::vector<H> &expected, const char *label) {
    if (actual.size() != expected.size())
        throw std::runtime_error("Wrong window output size");
    for (size_t i = 0; i < actual.size(); ++i)
        if (nr::halfBits(actual[i]) != nr::halfBits(expected[i])) {
            std::fprintf(stderr, "%s[%zu]: %04x != %04x\n", label, i, nr::halfBits(actual[i]),
                         nr::halfBits(expected[i]));
            throw std::runtime_error("Exact window bit mismatch");
        }
}
template <class T> struct Device {
    T *p = nullptr;
    explicit Device(size_t n) {
        check(cudaMalloc(&p, n * sizeof(T)));
    }
    ~Device() {
        if (p)
            cudaFree(p);
    }
    Device(const Device &) = delete;
    // Pageable H2D may return after staging. Complete the default-stream copy
    // before either the fused kernel or staged reference launches on a nonblocking stream.
    void upload(const std::vector<T> &x) {
        check(cudaMemcpy(p, x.data(), x.size() * sizeof(T), cudaMemcpyHostToDevice));
        check(cudaStreamSynchronize(nullptr));
    }
};
struct Guarded {
    static constexpr size_t guard = 16;
    size_t size;
    Device<H> buffer;
    explicit Guarded(size_t n) : size(n), buffer(n + 2 * guard) {
        std::vector<H> x(n + 2 * guard, nr::halfFromBits(0x3555));
        buffer.upload(x);
    }
    H *data() {
        return buffer.p + guard;
    }
    std::vector<H> read() {
        std::vector<H> x(size + 2 * guard);
        check(cudaMemcpy(x.data(), buffer.p, x.size() * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < guard; ++i)
            if (nr::halfBits(x[i]) != 0x3555 || nr::halfBits(x[guard + size + i]) != 0x3555)
                throw std::runtime_error("Window output guard overwritten");
        return std::vector<H>(x.begin() + guard, x.begin() + guard + size);
    }
};
struct Baseline {
    void *library = nullptr, *context = nullptr;
    int (*create)(void **, int, int) = nullptr;
    int (*destroy)(void *) = nullptr;
    int (*sync)(void *) = nullptr;
    int (*window)(void *, const H *, const H *, const float *, H *, int, int, int, int) = nullptr;
    int (*begin)(void *) = nullptr;
    int (*end)(void *) = nullptr;
    int (*replay)(void *, float *) = nullptr;
    const char *(*error)() = nullptr;
    template <class T> void get(T &fn, const char *name) {
        fn = reinterpret_cast<T>(dlsym(library, name));
        if (!fn)
            throw std::runtime_error(std::string("Missing baseline symbol ") + name);
    }
    explicit Baseline(const std::string &path) {
        library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!library)
            throw std::runtime_error(dlerror());
        get(create, "nr_create");
        get(destroy, "nr_destroy");
        get(sync, "nr_sync");
        get(window, "nr_window");
        get(begin, "nr_capture_begin");
        get(end, "nr_capture_end");
        get(replay, "nr_replay");
        get(error, "nr_error");
        uint32_t (*abi)() = nullptr;
        get(abi, "nr_abi_version");
        if (abi() != NR_CUDA_ABI_VERSION)
            throw std::runtime_error("Wrong baseline ABI version");
        const int status = create(&context, NR_CUDA_BROWSER_COMPAT_V1, 0);
        checked(status);
    }
    ~Baseline() {
        if (context)
            destroy(context);
        if (library)
            dlclose(library);
    }
    void checked(int status) {
        if (status)
            throw std::runtime_error(error());
    }
    void run(const Input &in, const H *q, const H *p, const float *s, H *out) {
        auto a = in.shape;
        checked(window(context, q, p, s, out, a.w, a.h, a.heads, a.phase));
    }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaStream_t stream;
    template <class F> Graph(cudaStream_t s, F fn) : stream(s) {
        check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        fn();
        check(cudaStreamEndCapture(s, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
    }
    ~Graph() {
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
    }
    void run() {
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
    }
};
void launch_window(const H *q, const H *p, const float *scale, H *out, const Shape &s,
                   cudaStream_t stream) {
    nr_window_f13_fused(q, p, scale, out, s.w, s.h, s.heads, s.phase, stream);
}
void gpu_case(Input &in, cudaStream_t stream, Baseline *baseline, bool graph) {
    const auto s = in.shape;
    const size_t count = size_t(s.w) * s.h * s.heads * 32;
    Device<H> q(in.qkv.size()), p(in.prior.size());
    Device<float> scale(in.scales.size());
    Guarded out(count);
    q.upload(in.qkv);
    p.upload(in.prior);
    scale.upload(in.scales);
    if (baseline) {
        baseline->run(in, q.p, p.p, scale.p, out.data());
        baseline->checked(baseline->sync(baseline->context));
        auto expected = out.read();
        if (in.expected.empty())
            in.expected = expected;
        else
            equal(expected, in.expected, "Baseline versus independent oracle");
    }
    if (in.expected.empty())
        throw std::runtime_error("A GPU case needs independent or staged expected data");
    auto launch = [&] { launch_window(q.p, p.p, scale.p, out.data(), s, stream); };
    // A missing kernel write must not reuse the staged result or a prior replay.
    check(cudaMemsetAsync(out.data(), 0xff, count * sizeof(H), stream));
    launch();
    check(cudaStreamSynchronize(stream));
    equal(out.read(), in.expected, "Fused window");
    if (graph) {
        Graph capture(stream, launch);
        for (int i = 0; i < 3; ++i) {
            check(cudaMemsetAsync(out.data(), 0xff, count * sizeof(H), stream));
            capture.run();
            equal(out.read(), in.expected, "Fused window graph");
        }
    }
}
Input fixture(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open window fixture");
    char magic[8];
    uint32_t dims[4];
    f.read(magic, 8);
    f.read(reinterpret_cast<char *>(dims), 16);
    if (std::string(magic, 8) != "NRWF1301" || dims[0] == 0 || dims[1] == 0 || dims[2] == 0 ||
        dims[2] > 16 || dims[3] > 3 ||
        uint64_t(dims[0]) * dims[1] * dims[2] > uint64_t(INT_MAX) / 96)
        throw std::runtime_error("Invalid window fixture header");
    Input in{{int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])}};
    in.qkv.resize(size_t(dims[0]) * dims[1] * dims[2] * 96);
    in.prior.resize(dims[2] * 4096);
    in.scales.resize(dims[2]);
    in.expected.resize(in.qkv.size() / 3);
    f.read(reinterpret_cast<char *>(in.qkv.data()), in.qkv.size() * 2);
    f.read(reinterpret_cast<char *>(in.prior.data()), in.prior.size() * 2);
    f.read(reinterpret_cast<char *>(in.scales.data()), in.scales.size() * 4);
    f.read(reinterpret_cast<char *>(in.expected.data()), in.expected.size() * 2);
    if (!f || f.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Wrong window fixture length");
    return in;
}
} // namespace

int main(int argc, char **argv) {
    try {
        bool host = false;
        std::string library, fixture_path;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--host-only")
                host = true;
            else if (option == "--reference-library" && i + 1 < argc)
                library = argv[++i];
            else if (option == "--fixture" && i + 1 < argc)
                fixture_path = argv[++i];
            else
                throw std::runtime_error("Usage: nr-window-test [--host-only] [--reference-library "
                                         "LIB] [--fixture FILE]");
        }
        // Byte-overlap rejection occurs before any CUDA API call, including in
        // host-only mode. These pointers are never dereferenced or launched.
        for (uintptr_t alias : {uintptr_t(0x10002), uintptr_t(0x30002), uintptr_t(0x50000)}) {
            bool rejected = false;
            try {
                nr_window_f13_fused(reinterpret_cast<const H *>(0x10000),
                                    reinterpret_cast<const H *>(0x30000),
                                    reinterpret_cast<const float *>(0x50000),
                                    reinterpret_cast<H *>(alias), 1, 1, 1, 0, nullptr);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            if (!rejected)
                throw std::runtime_error("Output overlap was not rejected");
        }
        for (int code = 0; code < 256; ++code)
            if ((code & 127) != 127) {
                const H h = rne(num::e4m3ToF32(uint8_t(code)));
                const uint32_t packet = nr_window_f13_detail::encode(h);
                if (num::f32Bits(value(h)) !=
                    num::f32Bits(nr_window_f13_detail::compact_value(packet)))
                    throw std::runtime_error("Compact operand loses E4 value bits");
                const int expected = value(h) == 0 ? -6 : exponent(value(h), -6);
                if (int(packet & 15u) - 6 != expected)
                    throw std::runtime_error("Compact operand floor exponent mismatch");
            }
        cudaStream_t stream = nullptr;
        Baseline *baseline = nullptr;
        if (!host) {
            cudaDeviceProp p{};
            check(cudaGetDeviceProperties(&p, 0));
            if (p.major != 7 || p.minor != 5)
                throw std::runtime_error("Window test requires SM75");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            if (!library.empty())
                baseline = new Baseline(library);
        }
        int cpu_cases = 0, gpu_cases = 0;
        for (const auto shape : {Shape{1, 1, 1, 0}, Shape{1, 9, 1, 1}, Shape{4, 4, 1, 2},
                                 Shape{8, 8, 1, 0}, Shape{12, 20, 2, 1}, Shape{5, 9, 1, 3}}) {
            auto in = random_input(shape, false);
            in.expected = reference(in);
            if (!host) {
                gpu_case(in, stream, baseline, cpu_cases == 0);
                ++gpu_cases;
            }
            ++cpu_cases;
        }
        if (!fixture_path.empty()) {
            auto in = fixture(fixture_path);
            if (host)
                equal(reference(in), in.expected, "Observed CPU window fixture");
            else {
                gpu_case(in, stream, baseline, true);
                ++gpu_cases;
            }
        }
        if (!host && baseline)
            for (auto shape : {Shape{4, 4, 1, 0}, Shape{8, 8, 1, 0}, Shape{12, 20, 2, 0},
                               Shape{32, 24, 4, 0}, Shape{65, 33, 16, 0}})
                for (int phase = 0; phase < 4; ++phase)
                    for (bool zero : {false, true}) {
                        shape.phase = phase;
                        auto in = random_input(shape, zero);
                        gpu_case(in, stream, baseline, false);
                        ++gpu_cases;
                    }
        delete baseline;
        baseline = nullptr;
        std::printf("{\"status\":\"pass\",\"gpu_executed\":%s,\"independent_cpu_cases\":%d,\"gpu_"
                    "cases\":%d,\"warps\":32,\"bitwise_required\":true}\n",
                    host ? "false" : "true", cpu_cases, gpu_cases);
        if (stream)
            check(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "EXACT WINDOW TEST FAILED: %s\n", e.what());
        return 1;
    }
}
