// CPU/GPU tests against captured immutable browser answers. --host-only makes
// no CUDA runtime calls, allowing execution in the CPU-only build container.
#include "browser_numerics.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Result {
    uint16_t half;
    uint8_t code;
    uint8_t reserved;
};
static_assert(sizeof(Result) == 4, "fixture record layout");
void checked(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
bool finiteHalf(unsigned bits) {
    return (bits & 0x7c00u) != 0x7c00u;
}
size_t failures = 0;
void compare(const char *group, size_t index, unsigned actual, unsigned expected) {
    if (actual == expected)
        return;
    if (failures < 12)
        std::fprintf(stderr, "%s[%zu]: actual=0x%x expected=0x%x\n", group, index, actual,
                     expected);
    ++failures;
}
std::vector<Result> fixture(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read observed browser fixture: " + path);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.size() != 65536 * 4)
        throw std::runtime_error("unexpected browser fixture size");
    std::vector<Result> records(65536);
    for (size_t i = 0; i < records.size(); ++i) {
        const uint8_t *p = bytes.data() + 4 * i;
        if (p[3])
            throw std::runtime_error("nonzero reserved fixture byte");
        records[i] = {uint16_t(p[0] | (uint16_t(p[1]) << 8)), p[2], 0};
    }
    return records;
}
double independentE4(unsigned code) {
    const unsigned exponent = code >> 3, fraction = code & 7;
    return exponent ? std::ldexp(1.0 + fraction / 8.0, int(exponent) - 7)
                    : std::ldexp(double(fraction), -9);
}
uint8_t nearestE4(float value) {
    if (std::isnan(value) || value == 0)
        return 0;
    const uint8_t sign = std::signbit(value) ? 128 : 0;
    const double magnitude = std::abs(double(value));
    if (magnitude >= 448)
        return sign | 126;
    unsigned upper = 0;
    while (independentE4(upper) < magnitude)
        ++upper;
    if (independentE4(upper) == magnitude || upper == 0)
        return sign | upper;
    const unsigned lower = upper - 1;
    const double down = magnitude - independentE4(lower), up = independentE4(upper) - magnitude;
    return sign | (down < up ? lower : up < down ? upper : (lower & 1) ? upper : lower);
}
uint16_t towardZeroHalf(float value) {
    if (std::isinf(value))
        return std::signbit(value) ? 0xfc00 : 0x7c00;
    const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    const double magnitude = std::abs(double(value));
    // Independent search over the positive half grid, avoiding bit conversion.
    unsigned lo = 0, hi = 0x7c00;
    while (lo + 1 < hi) {
        const unsigned mid = (lo + hi) / 2;
        if (nr::halfToFloat(nr::halfFromBits(uint16_t(mid))) <= magnitude)
            lo = mid;
        else
            hi = mid;
    }
    return sign | uint16_t(lo);
}
std::vector<float> publicationInputs() {
    std::vector<float> result;
    for (unsigned i = 0; i < 126; ++i) {
        const float mid = float((independentE4(i) + independentE4(i + 1)) * .5);
        for (float x : {std::nextafter(mid, -std::numeric_limits<float>::infinity()), mid,
                        std::nextafter(mid, std::numeric_limits<float>::infinity())}) {
            result.push_back(x);
            result.push_back(-x);
        }
    }
    for (float x : {0.f, -0.f, 448.f, -448.f, 65504.f, -65504.f, 100000.f, -100000.f,
                    std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()})
        result.push_back(x);
    return result;
}
__global__ void activationKernel(Result *output) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536 && (i & 0x7c00u) != 0x7c00u) {
        const __half x = nr::halfFromBits(uint16_t(i));
        output[i] = {nr::halfBits(nr::browserSiluHalf(x)), nr::browserSiluCode(x), 0};
    }
}
__global__ void publicationKernel(const float *input, Result *output, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = {nr::halfBits(nr::browserHalfRtz(input[i])), nr::browserE4Code(input[i]), 0};
}
} // namespace

int main(int argc, char **argv) {
    try {
        bool hostOnly = false;
        std::string path = "tests/fixtures/browser-silu-observed.bin";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--host-only")
                hostOnly = true;
            else if (arg == "--fixture" && i + 1 < argc)
                path = argv[++i];
            else
                throw std::runtime_error(
                    "usage: nr-browser-numerics-test [--host-only] [--fixture PATH]");
        }
        const auto expected = fixture(path);
        size_t finiteInputs = 0;
        for (unsigned i = 0; i < 65536; ++i)
            if (finiteHalf(i)) {
                ++finiteInputs;
                const __half value = nr::halfFromBits(uint16_t(i));
                compare("host-SiLU-half", i, nr::halfBits(nr::browserSiluHalf(value)),
                        expected[i].half);
                compare("host-SiLU-E4", i, nr::browserSiluCode(value), expected[i].code);
            }
        const auto inputs = publicationInputs();
        std::vector<Result> publicationExpected(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            const float value = inputs[i];
            publicationExpected[i] = {std::isnan(value) ? uint16_t(0) : towardZeroHalf(value),
                                      nearestE4(value), 0};
            compare("host-direct-E4", i, nr::browserE4Code(value), publicationExpected[i].code);
            if (!std::isnan(value))
                compare("host-half-RTZ", i, nr::halfBits(nr::browserHalfRtz(value)),
                        publicationExpected[i].half);
        }
        const size_t hostFailures = failures;
        if (!hostOnly) {
            Result *output = nullptr;
            float *values = nullptr;
            checked(cudaMalloc(&output, 65536 * sizeof(Result)));
            activationKernel<<<256, 256>>>(output);
            checked(cudaGetLastError());
            std::vector<Result> actual(65536);
            checked(
                cudaMemcpy(actual.data(), output, 65536 * sizeof(Result), cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < 65536; ++i)
                if (finiteHalf(i)) {
                    compare("GPU-SiLU-half", i, actual[i].half, expected[i].half);
                    compare("GPU-SiLU-E4", i, actual[i].code, expected[i].code);
                }
            checked(cudaMalloc(&values, inputs.size() * sizeof(float)));
            checked(cudaMemcpy(values, inputs.data(), inputs.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
            publicationKernel<<<(inputs.size() + 255) / 256, 256>>>(values, output, inputs.size());
            checked(cudaGetLastError());
            checked(cudaMemcpy(actual.data(), output, inputs.size() * sizeof(Result),
                               cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < inputs.size(); ++i) {
                compare("GPU-direct-E4", i, actual[i].code, publicationExpected[i].code);
                if (!std::isnan(inputs[i]))
                    compare("GPU-half-RTZ", i, actual[i].half, publicationExpected[i].half);
            }
            checked(cudaFree(values));
            checked(cudaFree(output));
        }
        std::printf("{\"passed\":%s,\"finite_half_inputs\":%zu,\"publication_inputs\":%zu,\"host_"
                    "failures\":%zu,\"gpu_executed\":%s,\"gpu_failures\":%zu}\n",
                    failures ? "false" : "true", finiteInputs, inputs.size(), hostFailures,
                    hostOnly ? "false" : "true", failures - hostFailures);
        return failures ? 1 : 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
