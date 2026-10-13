// Browser-compatible GEMM tests against independent oracles and fixed answers.
#include "gemm_f13.cuh"
#include "test_support.cuh"

namespace {
using namespace nr_test;
int group_repeats = 1;
double browser_step(const H *a, int as, const H *b, int bs, double incoming) {
    int maximum = incoming == 0 ? -21 : exponent(incoming, -14);
    for (int i = 0; i < 16; ++i) {
        const double x = number(a[i * as]), y = number(b[i * bs]);
        if (x != 0 && y != 0)
            maximum = std::max(maximum, exponent(x, -6) + exponent(y, -6));
    }
    const double alignment = std::ldexp(1., 13 - maximum);
    int64_t units = int64_t(std::trunc(incoming * alignment));
    for (int i = 0; i < 16; ++i)
        units += int64_t(std::trunc(number(a[i * as]) * number(b[i * bs]) * alignment));
    const double result = std::ldexp(double(units), maximum - 13);
    if (double(float(result)) != result)
        throw std::runtime_error("Browser oracle unexpectedly exceeds exact FP32 range");
    return result;
}

H toward_zero(double value) {
    H result = nearest(value);
    if (std::abs(number(result)) > std::abs(value))
        result = nr::halfFromBits(nr::halfBits(result) - 1);
    return result;
}

H browser_publication(double value) {
    const bool negative = value < 0;
    const double mag = std::min(std::abs(value), 448.);
    unsigned best = 0;
    double distance = std::abs(mag - number(decode(0)));
    for (unsigned code = 1; code < 127; ++code) {
        const double d = std::abs(mag - number(decode(code)));
        if (d < distance || (d == distance && !(code & 1))) {
            best = code;
            distance = d;
        }
    }
    return decode(best | (negative ? 128 : 0));
}

float browser_activation(H input) {
    const float x = float(number(input)), bounded = std::max(-4.f, std::min(4.f, x));
    const float inner = std::fma(-0.055908203125f, std::abs(bounded), 0.447265625f);
    const float polynomial = std::fma(bounded, inner, 0.89453125f);
    return x * polynomial;
}

int matrix_cases(bool gpu, cudaStream_t stream) {
    Random random{0xc001d00du};
    int cases = 0;
    for (int experts : {1, 2, 3})
        for (bool broadcast : {false, true})
            for (int partition : {0, 32, 48})
                for (bool residual_enabled : {false, true}) {
                    const int M = 17, K = 64, N = 19, as = K * (broadcast ? 1 : experts),
                              os = N * experts;
                    std::vector<H> a(M * as), w(experts * K * N), res(M * os), scale(os),
                        expected(M * os), raw(M * os), actual(M * os), expected_raw(M * os);
                    for (H &v : a)
                        v = decode((random.next() % 89) | ((random.next() & 1) << 7));
                    for (H &v : w)
                        v = decode((random.next() % 89) | ((random.next() & 1) << 7));
                    for (H &v : res)
                        v = nearest((int(random.next() % 129) - 64) / 16.0);
                    for (int i = 0; i < os; ++i)
                        scale[i] = nearest(i % 2 ? -.75 : 1.25);
                    const bool silu = residual_enabled, quant = partition != 48;
                    for (int row = 0; row < M; ++row)
                        for (int expert = 0; expert < experts; ++expert)
                            for (int column = 0; column < N; ++column) {
                                const int at = row * os + expert * N + column;
                                double wide =
                                    residual_enabled
                                        ? number(res[at]) * number(scale[expert * N + column])
                                        : 0;
                                double wide_total = 0;
                                for (int start = 0; start < K; start += 16) {
                                    const H *left =
                                        a.data() + row * as + (broadcast ? 0 : expert * K) + start;
                                    const H *right = w.data() + (expert * K + start) * N + column;
                                    wide = browser_step(left, 1, right, N, wide);
                                    if (partition &&
                                        ((start + 16) % partition == 0 || start + 16 == K)) {
                                        wide_total = start < partition
                                                         ? wide
                                                         : double(float(wide_total + wide));
                                        wide = 0;
                                    }
                                }
                                if (partition)
                                    wide = wide_total;
                                H value = toward_zero(wide);
                                const float activated =
                                    silu ? browser_activation(value) : float(wide);
                                if (silu)
                                    value = toward_zero(activated);
                                expected_raw[at] = value;
                                expected[at] =
                                    quant ? browser_publication(
                                                silu && number(value) == 0 ? 0. : activated)
                                          : value;
                            }
                    if (gpu) {
                        Device<H> da(a.size()), dw(w.size()), dr(res.size()), ds(scale.size()),
                            dout(actual.size()), draw(raw.size());
                        da.upload(a);
                        dw.upload(w);
                        dr.upload(res);
                        ds.upload(scale);
                        auto launch = [&] {
                            nr::nr_gemm_f13(da.data, dw.data, dout.data, draw.data,
                                            residual_enabled ? dr.data : nullptr,
                                            residual_enabled ? ds.data : nullptr, M, K, N, experts,
                                            int(broadcast), int(silu), partition, int(quant),
                                            stream);
                        };
                        check(cudaMemsetAsync(dout.data, 0xff, actual.size() * sizeof(H), stream));
                        check(cudaMemsetAsync(draw.data, 0xff, raw.size() * sizeof(H), stream));
                        launch();
                        check(cudaStreamSynchronize(stream));
                        dout.download(actual);
                        draw.download(raw);
                        same(actual, expected, "GEMM output");
                        same(raw, expected_raw, "GEMM raw");
                        if (cases == 0) {
                            cudaGraph_t graph;
                            cudaGraphExec_t executable;
                            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                            launch();
                            check(cudaStreamEndCapture(stream, &graph));
                            check(cudaGraphInstantiate(&executable, graph, 0));
                            check(cudaMemsetAsync(dout.data, 0xff, actual.size() * sizeof(H), stream));
                            check(cudaMemsetAsync(draw.data, 0xff, raw.size() * sizeof(H), stream));
                            check(cudaGraphLaunch(executable, stream));
                            check(cudaStreamSynchronize(stream));
                            dout.download(actual);
                            draw.download(raw);
                            same(actual, expected, "Captured GEMM");
                            same(raw, expected_raw, "Captured GEMM raw");
                            check(cudaGraphExecDestroy(executable));
                            check(cudaGraphDestroy(graph));
                        }
                    }
                    ++cases;
                }
    return cases;
}

int browser_witnesses(bool gpu, cudaStream_t stream) {
    // Retained K16 precision; final RTZ; distinct raw-half/direct-E4 stores.
    const uint16_t raw_bits[3] = {0x1000, 0x3c00, 0x3cbf};
    const float e4_values[3] = {0.f, 1.f, 1.125f};
    for (int which = 0; which < 3; ++which) {
        std::vector<H> a(32, nearest(0)), w(32, nearest(0)), out(1), raw(1);
        a[0] = nearest(which == 2 ? 1.125 : 1);
        w[0] = nearest(1);
        if (which < 2) {
            a[1] = nearest(1. / 512);
            w[1] = nearest(which == 0 ? .25 : .375);
        } else {
            a[1] = nearest(.0625);
            w[1] = nearest(1);
            a[2] = nearest(-1. / 512);
            w[2] = nearest(.125);
        }
        if (which == 0) {
            a[16] = nearest(-1);
            w[16] = nearest(1);
        }
        double value = browser_step(a.data(), 1, w.data(), 1, 0);
        value = browser_step(a.data() + 16, 1, w.data() + 16, 1, value);
        const std::vector<H> expected_raw{nr::halfFromBits(raw_bits[which])},
            expected{nearest(e4_values[which])};
        same(std::vector<H>{toward_zero(value)}, expected_raw, "Browser literal half witness");
        same(std::vector<H>{browser_publication(value)}, expected, "Browser literal E4 witness");
        if (gpu) {
            Device<H> da(32), dw(32), dout(1), draw(1);
            da.upload(a);
            dw.upload(w);
            nr::nr_gemm_f13(da.data, dw.data, dout.data, draw.data, nullptr, nullptr, 1, 32, 1, 1,
                            0, 0, 0, 1, stream);
            check(cudaStreamSynchronize(stream));
            dout.download(out);
            draw.download(raw);
            same(raw, expected_raw, "Browser GPU literal half witness");
            same(out, expected, "Browser GPU literal E4 witness");
        }
    }
    // Independent literal partition witnesses. Narrowing the first partial
    // [1+2^-11,-1] or the running total [1,2^-11,-1] to half loses 2^-11.
    for (int which = 0; which < 2; ++which) {
        const int K = which ? 48 : 32;
        std::vector<H> a(K, nearest(0)), w(K, nearest(0)), out(1), raw(1);
        a[0] = nearest(1);
        w[0] = nearest(1);
        const int small = which ? 16 : 1, last = which ? 32 : 16;
        a[small] = nearest(1. / 512);
        w[small] = nearest(.25);
        a[last] = nearest(-1);
        w[last] = nearest(1);
        double total = 0;
        for (int k = 0; k < K; k += 16) {
            const double partial = browser_step(a.data() + k, 1, w.data() + k, 1, 0);
            total = k ? double(float(total + partial)) : partial;
        }
        const std::vector<H> expected{nr::halfFromBits(0x1000)};
        same(std::vector<H>{toward_zero(total)}, expected, "Browser literal partition witness");
        if (gpu) {
            Device<H> da(K), dw(K), dout(1), draw(1);
            da.upload(a);
            dw.upload(w);
            nr::nr_gemm_f13(da.data, dw.data, dout.data, draw.data, nullptr, nullptr, 1, K, 1, 1, 0,
                            0, 16, 0, stream);
            check(cudaStreamSynchronize(stream));
            dout.download(out);
            draw.download(raw);
            same(raw, expected, "Browser GPU partition raw witness");
            same(out, expected, "Browser GPU partition output witness");
        }
    }
    return 5;
}

struct GroupRecord {
    H a[16], b[16];
    float incoming, expected;
};
static_assert(sizeof(GroupRecord) == 72, "Independent group record schema");
std::vector<unsigned char> read_bytes(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot read F13 fixture: " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(input)), {});
}
void same_float(float actual, float expected, const char *label, size_t index) {
    if (nr::browser_detail::floatBits(actual) != nr::browser_detail::floatBits(expected)) {
        std::fprintf(stderr, "%s[%zu] actual=%.9g/%08x expected=%.9g/%08x\n", label, index, actual,
                     nr::browser_detail::floatBits(actual), expected,
                     nr::browser_detail::floatBits(expected));
        throw std::runtime_error("F13 retained-FP32 mismatch");
    }
}
float group_cpu(const GroupRecord &record) {
    uint32_t a[16], b[16];
    for (int k = 0; k < 16; ++k) {
        a[k] = nr::f13_gemm_detail::pack_compact(record.a[k]);
        b[k] = nr::f13_gemm_detail::pack_compact(record.b[k]);
    }
    return nr::f13_gemm_detail::step_compact(a, 1, b, 1, record.incoming);
}
__global__ void group_gpu(const GroupRecord *source, float *output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        uint32_t a[16], b[16];
        for (int k = 0; k < 16; ++k) {
            a[k] = nr::f13_gemm_detail::pack_compact(source[i].a[k]);
            b[k] = nr::f13_gemm_detail::pack_compact(source[i].b[k]);
        }
        output[i] = nr::f13_gemm_detail::step_compact(a, 1, b, 1, source[i].incoming);
    }
}
int group_fixtures(const std::string &directory, bool gpu, cudaStream_t stream) {
    const auto bytes = read_bytes(directory + "/f13-integer-groups.bin");
    if (bytes.size() != 4096 * sizeof(GroupRecord))
        throw std::runtime_error("Wrong group fixture bytes");
    std::vector<GroupRecord> records(4096);
    std::memcpy(records.data(), bytes.data(), bytes.size());
    for (size_t i = 0; i < records.size(); ++i) {
        same_float(group_cpu(records[i]), records[i].expected, "CPU independent groups", i);
    }
    if (gpu)
        for (int repeat = 0; repeat < group_repeats; ++repeat) {
            Device<GroupRecord> input(records.size());
            Device<float> output(records.size());
            input.upload(records);
            group_gpu<<<16, 256, 0, stream>>>(input.data, output.data, int(records.size()));
            check(cudaGetLastError());
            check(cudaStreamSynchronize(stream));
            std::vector<float> result(records.size());
            output.download(result);
            for (size_t i = 0; i < records.size(); ++i)
                same_float(result[i], records[i].expected, "GPU independent groups", i);
        }
    return int(records.size());
}
template <class T>
std::vector<T> consume(const std::vector<unsigned char> &bytes, size_t &offset, size_t count) {
    if (count > (bytes.size() - offset) / sizeof(T))
        throw std::runtime_error("Short matrix fixture");
    std::vector<T> result(count);
    std::memcpy(result.data(), bytes.data() + offset, count * sizeof(T));
    offset += count * sizeof(T);
    return result;
}
int full_fixtures(const std::string &directory, bool gpu, cudaStream_t stream) {
    int cases = 0;
    for (int fixture = 0; fixture < 6; ++fixture) {
        char filename[64];
        std::snprintf(filename, sizeof(filename), "/f13-integer-gemm-%02d.bin", fixture);
        const auto bytes = read_bytes(directory + filename);
        if (bytes.size() < 16)
            throw std::runtime_error("Short fixture header");
        const int M = u32(bytes.data()), K = u32(bytes.data() + 4), N = u32(bytes.data() + 8),
                  partition = u32(bytes.data() + 12);
        if (M < 1 || M > 17 || K < 16 || K > 4096 || K % 16 || N < 1 || N > 19 || partition < 0 ||
            partition % 16)
            throw std::runtime_error("Unexpected fixture dimensions");
        size_t offset = 16;
        const auto a = consume<H>(bytes, offset, size_t(M) * K),
                   w = consume<H>(bytes, offset, size_t(K) * N);
        const auto res = consume<H>(bytes, offset, size_t(M) * N),
                   scale = consume<H>(bytes, offset, N);
        const auto expected_wide = consume<float>(bytes, offset, size_t(M) * N);
        const auto expected_raw = consume<H>(bytes, offset, size_t(M) * N);
        const auto codes = consume<unsigned char>(bytes, offset, size_t(M) * N);
        if (offset != bytes.size())
            throw std::runtime_error("Trailing fixture bytes");
        std::vector<H> expected(M * N);
        for (int i = 0; i < M * N; ++i)
            expected[i] = decode(codes[i]);
        for (int row = 0; row < M; ++row)
            for (int col = 0; col < N; ++col) {
                const int at = row * N + col;
                float value = float(number(res[at]) * number(scale[col])), total = 0;
                for (int k = 0; k < K; k += 16) {
                    GroupRecord group{};
                    group.incoming = value;
                    for (int j = 0; j < 16; ++j) {
                        group.a[j] = a[row * K + k + j];
                        group.b[j] = w[(k + j) * N + col];
                    }
                    value = group_cpu(group);
                    if (partition && ((k + 16) % partition == 0 || k + 16 == K)) {
                        total = k < partition ? value : float(total + value);
                        value = 0;
                    }
                }
                const float retained = partition ? total : value;
                same_float(retained, expected_wide[at], "CPU independent matrix",
                           at);
                if (nr::halfBits(nr::browserHalfRtz(retained)) != nr::halfBits(expected_raw[at]))
                    throw std::runtime_error("CPU independent raw FP16 mismatch");
                if (nr::browserE4Code(retained) != codes[at] ||
                    nr::halfBits(nr::browserPublishE4(retained)) != nr::halfBits(expected[at]))
                    throw std::runtime_error("CPU independent E4 output mismatch");
            }
        if (gpu) {
            Device<H> da(a.size()), dw(w.size()), dr(res.size()), ds(scale.size()),
                out(expected.size() + 32), raw(expected.size() + 32);
            Device<float> wide(expected.size());
            da.upload(a);
            dw.upload(w);
            dr.upload(res);
            ds.upload(scale);
            std::vector<H> guarded(expected.size() + 32, nr::halfFromBits(0x3555));
            out.upload(guarded);
            raw.upload(guarded);
            nr::nr_gemm_f13(da.data, dw.data, out.data + 16, raw.data + 16, dr.data, ds.data, M, K,
                            N, 1, 0, 0, partition, 1, stream, wide.data);
            check(cudaStreamSynchronize(stream));
            std::vector<H> actual(guarded.size()), actual_raw(guarded.size());
            out.download(actual);
            raw.download(actual_raw);
            for (int i = 0; i < 16; ++i)
                if (nr::halfBits(actual[i]) != 0x3555 ||
                    nr::halfBits(actual[expected.size() + 16 + i]) != 0x3555 ||
                    nr::halfBits(actual_raw[i]) != 0x3555 ||
                    nr::halfBits(actual_raw[expected.size() + 16 + i]) != 0x3555)
                    throw std::runtime_error("GEMM output guard changed");
            same(std::vector<H>(actual.begin() + 16, actual.end() - 16), expected,
                 "GPU independent E4");
            same(std::vector<H>(actual_raw.begin() + 16, actual_raw.end() - 16), expected_raw,
                 "GPU independent raw");
            std::vector<float> actual_wide(expected.size());
            wide.download(actual_wide);
            for (size_t i = 0; i < actual_wide.size(); ++i)
                same_float(actual_wide[i], expected_wide[i], "GPU independent retained FP32", i);
        }
        ++cases;
    }
    return cases;
}

int overlap_guards() {
    H *a = reinterpret_cast<H *>(uintptr_t(0x10000)),
      *w = reinterpret_cast<H *>(uintptr_t(0x20000));
    H *r = reinterpret_cast<H *>(uintptr_t(0x30000)),
      *s = reinterpret_cast<H *>(uintptr_t(0x40000));
    int rejected = 0;
    for (int which = 0; which < 7; ++which) {
        H *out = reinterpret_cast<H *>(uintptr_t(0x50000)),
          *raw = reinterpret_cast<H *>(uintptr_t(0x60000));
        float *wide = nullptr;
        if (which == 0)
            out = a + 1;
        else if (which == 1)
            out = w + 1;
        else if (which == 2)
            raw = out + 1;
        else if (which == 3)
            raw = a + 1;
        else if (which == 4)
            out = s;
        else if (which == 5)
            wide = reinterpret_cast<float *>(r);
        else
            out = r + 1;
        try {
            nr::nr_gemm_f13(a, w, out, raw, r, s, 16, 16, 16, 1, 0, 0, 0, 1, nullptr, wide);
        } catch (const std::invalid_argument &) {
            ++rejected;
            continue;
        }
        throw std::runtime_error("Overlap guard reached CUDA");
    }
    return rejected;
}

} // namespace

int main(int argc, char **argv) {
    try {
        bool host = false;
        std::string directory = "tests/fixtures/f13";
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--host-only")
                host = true;
            else if (option == "--fixtures" && i + 1 < argc)
                directory = argv[++i];
            else if (option == "--group-repeats" && i + 1 < argc) {
                group_repeats = std::stoi(argv[++i]);
                if (group_repeats < 1 || group_repeats > 100)
                    throw std::runtime_error("group-repeats must be 1..100");
            } else
                throw std::runtime_error("Usage: nr-compact-gemm-test [--host-only] [--fixtures "
                                         "DIR] [--group-repeats N]");
        }
        cudaStream_t stream = nullptr;
        if (!host) {
            cudaDeviceProp properties{};
            check(cudaGetDeviceProperties(&properties, 0));
            if (properties.major != 7 || properties.minor != 5)
                throw std::runtime_error("Requires SM75");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        }
        const int overlap_cases = overlap_guards();
        const int groups = group_fixtures(directory, !host, stream),
                  full = full_fixtures(directory, !host, stream);
        const int cases = matrix_cases(!host, stream), literals = browser_witnesses(!host, stream);
        if (stream)
            check(cudaStreamDestroy(stream));
        std::printf("{\"status\":\"pass\",\"gpu_executed\":%s,\"independent_groups\":%d,"
                    "\"independent_full_matrices\":%d,\"expert_matrix_cases\":%d,\"literal_cases\":"
                    "%d,\"overlap_guard_cases\":%d,\"output_guards_checked\":%s,\"graph_replay_"
                    "checked\":%s,\"group_gpu_repeats\":%d}\n",
                    host ? "false" : "true", groups, full, cases, literals, overlap_cases,
                    host ? "false" : "true", host ? "false" : "true", host ? 0 : group_repeats);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "F13 GEMM FAILED: %s\n", error.what());
        return 1;
    }
}
