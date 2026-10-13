// F13 dot and staged-attention tests against independent scalar oracles.
#include "attention_f13.cuh"
#include "test_support.cuh"

namespace {
using namespace nr_test;
std::vector<uint16_t> fixture(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open numeric fixture");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "NRNUM001", 8))
        throw std::runtime_error("Bad fixture magic");
    size_t offset = 12;
    for (uint32_t section = 0; section < u32(bytes.data() + 8); ++section) {
        if (offset + 8 > bytes.size())
            throw std::runtime_error("Short fixture header");
        const std::string tag(reinterpret_cast<const char *>(bytes.data() + offset), 4);
        const size_t count = u32(bytes.data() + offset + 4);
        offset += 8;
        if (count > bytes.size() - offset)
            throw std::runtime_error("Short fixture section");
        if (tag == "FDP8") {
            if (count != 32768)
                throw std::runtime_error("Unexpected FDP8 fixture size");
            std::vector<uint16_t> answer(count / 2);
            for (size_t i = 0; i < answer.size(); ++i)
                answer[i] =
                    uint16_t(bytes[offset + i * 2]) | (uint16_t(bytes[offset + i * 2 + 1]) << 8);
            return answer;
        }
        offset += count;
    }
    throw std::runtime_error("Missing FDP8 fixture");
}

H reference_step(const H *a, int as, const H *b, int bs, H incoming) {
    const double initial = number(incoming);
    if (!std::isfinite(initial))
        return incoming;
    int maximum = initial == 0 ? -21 : exponent(initial, -14);
    for (int i = 0; i < 16; ++i) {
        const double x = number(a[i * as]), y = number(b[i * bs]);
        if (x != 0 && y != 0)
            maximum = std::max(maximum, exponent(x, -6) + exponent(y, -6));
    }
    const double alignment = std::ldexp(1.0, 13 - maximum);
    int64_t units = int64_t(std::trunc(initial * alignment));
    for (int i = 0; i < 16; ++i)
        units += int64_t(std::trunc(number(a[i * as]) * number(b[i * bs]) * alignment));
    return nearest(std::ldexp(double(units), maximum - 13));
}

__global__ void primitive(const H *inputs, H *output, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = nr::f13_detail::step16(inputs + i * 33, 1, inputs + i * 33 + 16, 1,
                                           inputs[i * 33 + 32]);
}

int attention_cases(bool gpu, cudaStream_t stream) {
    Random random{0xa77e11u};
    int cases = 0;
    struct Case {
        int T, count, heads, first;
        bool prior, global;
    };
    for (const Case shape : {Case{64, 3, 2, 1, true, false}, Case{64, 2, 2, 0, false, true},
                             Case{128, 2, 2, 0, false, true}}) {
        const int T = shape.T, count = shape.count;
        std::vector<H> q(count * T * 32), k(q.size()), v(q.size()), p(count * T * T),
            prior(shape.heads * 4096), reciprocal(count * T);
        std::vector<H> scores(p.size()), expected_scores(p.size()), output(v.size()),
            expected_output(v.size());
        for (H &x : q)
            x = decode((random.next() % 73) | ((random.next() & 1) << 7));
        for (H &x : k)
            x = decode((random.next() % 73) | ((random.next() & 1) << 7));
        for (H &x : v)
            x = decode((random.next() % 89) | ((random.next() & 1) << 7));
        for (H &x : p)
            x = decode(random.next() % 56);
        for (H &x : prior)
            x = nearest((int(random.next() % 65) - 32) / 32.0);
        for (H &x : reciprocal)
            x = nearest((1 + random.next() % 32) / 64.0);
        for (int batch = 0; batch < count; ++batch)
            for (int row = 0; row < T; ++row) {
                for (int column = 0; column < T; ++column) {
                    H sum = shape.prior ? prior[((shape.first + batch) % shape.heads) * 4096 +
                                                row * 64 + column]
                                        : nearest(0);
                    for (int channel = 0; channel < 32; channel += 16)
                        sum =
                            reference_step(q.data() + (batch * T + row) * 32 + channel, 1,
                                           k.data() + (batch * T + column) * 32 + channel, 1, sum);
                    expected_scores[(batch * T + row) * T + column] = sum;
                }
                for (int column = 0; column < 32; ++column) {
                    H sum = nearest(0);
                    for (int start = 0; start < T; start += 16)
                        sum = reference_step(p.data() + (batch * T + row) * T + start, 1,
                                             v.data() + (batch * T + start) * 32 + column, 32, sum);
                    if (shape.global)
                        sum = nearest(number(sum) * number(reciprocal[batch * T + row]));
                    expected_output[(batch * T + row) * 32 + column] = publication(sum);
                }
            }
        if (gpu) {
            Device<H> dq(q.size()), dk(k.size()), dv(v.size()), dp(p.size()), dprior(prior.size()),
                dr(reciprocal.size()), ds(scores.size()), dout(output.size());
            dq.upload(q);
            dk.upload(k);
            dv.upload(v);
            dp.upload(p);
            dprior.upload(prior);
            dr.upload(reciprocal);
            auto launch = [&] {
                nr::nr_qk_f13(dq.data, dk.data, shape.prior ? dprior.data : nullptr, ds.data, T,
                              shape.heads, shape.first, count, stream);
                nr::nr_pv_f13(dp.data, dv.data, dout.data, shape.global ? dr.data : nullptr, T,
                              count, int(shape.global), stream);
            };
            check(cudaMemsetAsync(ds.data, 0xff, scores.size() * sizeof(H), stream));
            check(cudaMemsetAsync(dout.data, 0xff, output.size() * sizeof(H), stream));
            launch();
            check(cudaStreamSynchronize(stream));
            ds.download(scores);
            dout.download(output);
            same(scores, expected_scores, "QK F13");
            same(output, expected_output, "PV F13");
            if (cases == 0) {
                cudaGraph_t graph;
                cudaGraphExec_t executable;
                check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                launch();
                check(cudaStreamEndCapture(stream, &graph));
                check(cudaGraphInstantiate(&executable, graph, 0));
                check(cudaMemsetAsync(ds.data, 0xff, scores.size() * sizeof(H), stream));
                check(cudaMemsetAsync(dout.data, 0xff, output.size() * sizeof(H), stream));
                check(cudaGraphLaunch(executable, stream));
                check(cudaStreamSynchronize(stream));
                ds.download(scores);
                dout.download(output);
                same(scores, expected_scores, "Captured QK");
                same(output, expected_output, "Captured PV");
                check(cudaGraphExecDestroy(executable));
                check(cudaGraphDestroy(graph));
            }
        }
        ++cases;
    }
    return cases;
}
} // namespace

int main(int argc, char **argv) {
    try {
        bool host_only = false;
        std::string path = "tests/fixtures/numerics.bin";
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--host-only")
                host_only = true;
            else if (std::string(argv[i]) == "--fixture" && i + 1 < argc)
                path = argv[++i];
            else
                throw std::runtime_error("Usage: nr-f13-test [--host-only] [--fixture PATH]");
        }
        const auto bits = fixture(path);
        Random random{0x85ebca6bu};
        std::vector<H> inputs(bits.size() * 33), expected(bits.size()), cpu(bits.size()),
            actual(bits.size());
        for (size_t row = 0; row < bits.size(); ++row) {
            for (int i = 0; i < 32; ++i)
                inputs[row * 33 + i] = decode(random.next() & 255);
            uint16_t acc = random.next();
            if ((acc & 0x7c00) == 0x7c00)
                acc &= 0x7bff;
            inputs[row * 33 + 32] = nr::halfFromBits(acc);
            expected[row] = nr::halfFromBits(bits[row]);
            cpu[row] = reference_step(inputs.data() + row * 33, 1, inputs.data() + row * 33 + 16, 1,
                                      inputs[row * 33 + 32]);
        }
        same(cpu, expected, "CPU vs pinned FDP8");
        cudaStream_t stream = nullptr;
        if (!host_only) {
            cudaDeviceProp device{};
            check(cudaGetDeviceProperties(&device, 0));
            if (device.major != 7 || device.minor != 5)
                throw std::runtime_error("F13 test requires SM75");
            check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            Device<H> source(inputs.size()), destination(actual.size());
            source.upload(inputs);
            primitive<<<unsigned((bits.size() + 255) / 256), 256, 0, stream>>>(
                source.data, destination.data, int(bits.size()));
            check(cudaGetLastError());
            check(cudaStreamSynchronize(stream));
            destination.download(actual);
            same(actual, expected, "GPU vs pinned FDP8");
        }
        const int attention = attention_cases(!host_only, stream);
        if (stream)
            check(cudaStreamDestroy(stream));
        std::printf("{\"status\":\"pass\",\"gpu_executed\":%s,\"pinned_fdp8_cases\":%zu,"
                    "\"attention_cases\":%d,\"graph_replay_checked\":%s}\n",
                    host_only ? "false" : "true", bits.size(), attention,
                    host_only ? "false" : "true");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "F13 TEST FAILED: %s\n", error.what());
        return 1;
    }
}
