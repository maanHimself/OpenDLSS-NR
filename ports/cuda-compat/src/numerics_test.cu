// Standalone scalar CUDA validation. The checked-in upstream fixture, rather than
// this implementation, supplies the exhaustive activation/publication answers.
#include "kernels/numerics/numerics.cuh"
#include "../tests/fixtures/upstream_numeric.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NR_NUMERICS_FIXTURE_PATH
#define NR_NUMERICS_FIXTURE_PATH "tests/fixtures/numerics.bin"
#endif

namespace {
constexpr size_t halfCount = 65536, halfFields = 8, arithmeticFields = 5;
struct Fixture {
    std::map<std::string, std::vector<uint8_t>> sections;
};
uint32_t read32(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint16_t read16(const uint8_t *p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}
Fixture readFixture(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot read fixture: " + path);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "NRNUM001", 8))
        throw std::runtime_error("invalid fixture magic");
    const uint32_t count = read32(bytes.data() + 8);
    if (count > 64)
        throw std::runtime_error("invalid fixture section count");
    Fixture result;
    size_t offset = 12;
    for (uint32_t i = 0; i < count; ++i) {
        if (offset > bytes.size() || bytes.size() - offset < 8)
            throw std::runtime_error("short fixture header");
        const std::string tag(reinterpret_cast<const char *>(bytes.data() + offset), 4);
        const uint32_t length = read32(bytes.data() + offset + 4);
        offset += 8;
        if (length > bytes.size() - offset)
            throw std::runtime_error("short fixture section " + tag);
        if (!result.sections
                 .emplace(tag, std::vector<uint8_t>(bytes.begin() + offset,
                                                    bytes.begin() + offset + length))
                 .second)
            throw std::runtime_error("duplicate fixture section " + tag);
        offset += length;
    }
    if (offset != bytes.size())
        throw std::runtime_error("trailing fixture data");
    for (const auto &entry :
         std::array<std::pair<const char *, size_t>, 5>{{{"F16B", 32768 * 2},
                                                         {"E4EN", 65536},
                                                         {"E4DE", 256 * 4},
                                                         {"SILU", 65536 * 2},
                                                         {"EXPW", 65536 * 2}}}) {
        const auto it = result.sections.find(entry.first);
        if (it == result.sections.end() || it->second.size() != entry.second)
            throw std::runtime_error(std::string("missing/wrong-size fixture section ") +
                                     entry.first);
    }
    return result;
}

struct Xorshift {
    uint32_t state;
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
};
bool halfNaN(uint32_t bits) {
    return (bits & 0x7c00u) == 0x7c00u && (bits & 0x3ffu);
}
bool finiteHalf(uint32_t bits) {
    return (bits & 0x7c00u) != 0x7c00u;
}
uint16_t makeFiniteHalf(uint32_t bits) {
    uint16_t h = static_cast<uint16_t>(bits);
    if (!finiteHalf(h))
        h &= 0x7bff;
    return h;
}

// Independent arithmetic oracle: choose the closest number in the half value
// table. This does not share the implementation's bit-shift rounding algorithm.
uint16_t nearestHalf(double value) {
    static const std::vector<double> grid = [] {
        std::vector<double> values(0x7c00);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = num::f16ToF32(static_cast<uint16_t>(i));
        return values;
    }();
    const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    if (std::isnan(value))
        return sign | 0x7e00;
    const double magnitude = std::abs(value);
    if (magnitude >= 65520.0)
        return sign | 0x7c00;
    const auto it = std::lower_bound(grid.begin(), grid.end(), magnitude);
    if (it == grid.end())
        return sign | 0x7bff;
    const size_t upper = static_cast<size_t>(it - grid.begin());
    if (upper == 0 || *it == magnitude)
        return sign | static_cast<uint16_t>(upper);
    const size_t lower = upper - 1;
    const double down = magnitude - grid[lower], up = grid[upper] - magnitude;
    const size_t chosen = down < up ? lower : up < down ? upper : ((lower & 1) ? upper : lower);
    return sign | static_cast<uint16_t>(chosen);
}
uint16_t vitOracle(uint16_t input) {
    // Source: upstream browser numerics.js vitExpWeight. No original native ViT
    // scalar fixture exists, so report this weaker independent transcription apart.
    const double value = num::f16ToF32(input);
    const uint16_t affine = nearestHalf(value * 0.08953857421875 + 1.708984375);
    const double clamped =
        std::min(std::max(double(num::f16ToF32(affine)), 1.439453125), 1.9775390625);
    return static_cast<uint16_t>((uint32_t(nearestHalf(clamped)) << 4) + 0x4000u);
}

struct Check {
    const char *name;
    size_t tested = 0, skipped = 0, nanEquivalent = 0, failures = 0;
    void compare(size_t index, uint32_t actual, uint32_t expected, bool half = false) {
        ++tested;
        if (actual == expected)
            return;
        if (half && halfNaN(actual) && halfNaN(expected)) {
            ++nanEquivalent;
            return;
        }
        if (failures++ < 4)
            std::fprintf(stderr, "%s[%zu]: got 0x%08x, expected 0x%08x\n", name, index, actual,
                         expected);
    }
    bool finish(const char *backend) const {
        std::printf("%s %-32s tested=%zu skipped=%zu nan_equivalent=%zu failures=%zu\n", backend,
                    name, tested, skipped, nanEquivalent, failures);
        return failures == 0;
    }
};

__host__ __device__ void evaluateHalf(uint32_t index, uint32_t *out) {
    const __half h = nr::halfFromBits(static_cast<uint16_t>(index));
    out[0] = nr::e4m3FromHalfBits(static_cast<uint16_t>(index));
    out[1] = nr::e4m3FromFloat(nr::halfToFloat(h));
    out[2] = nr::halfBits(nr::halfRne(nr::halfToFloat(h)));
    out[3] = nr::halfBits(nr::siluHalf(h));
    out[4] = nr::halfBits(nr::windowExpWeight(h));
    out[5] = nr::halfBits(nr::vitExpWeight(h));
    out[6] = nr::halfBits(nr::publishE4(h));
    const float decoded = nr::e4m3ToFloat(static_cast<uint8_t>(index));
#if defined(__CUDA_ARCH__)
    out[7] = __float_as_uint(decoded);
#else
    out[7] = num::f32Bits(decoded);
#endif
}
__host__ __device__ void evaluateArithmetic(const uint16_t *input, uint32_t *out) {
    const __half a = nr::halfFromBits(input[0]), b = nr::halfFromBits(input[1]),
                 c = nr::halfFromBits(input[2]);
    out[0] = nr::halfBits(nr::addHalf(a, b));
    out[1] = nr::halfBits(nr::subHalf(a, b));
    out[2] = nr::halfBits(nr::mulHalf(a, b));
    out[3] = nr::halfBits(nr::fmaHalf(a, b, c));
    out[4] = nr::halfBits(nr::addHalf(nr::mulHalf(a, b), c));
}
__global__ void halfKernel(uint32_t *out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfCount)
        evaluateHalf(i, out + i * halfFields);
}
__global__ void floatKernel(const uint32_t *input, size_t count, uint32_t *out) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = __uint_as_float(input[i]);
        out[i * 2] = nr::halfBits(nr::halfRne(value));
        out[i * 2 + 1] = nr::e4m3FromFloat(value);
    }
}
__global__ void arithmeticKernel(const uint16_t *input, size_t count, uint32_t *out) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        evaluateArithmetic(input + i * 3, out + i * arithmeticFields);
}

bool checkHalves(const Fixture &fixture, const std::vector<uint32_t> &output, const char *backend) {
    std::array<Check, halfFields> checks{{{"E4 encode/all half patterns"},
                                          {"float-half-E4/all halves"},
                                          {"half roundtrip/all patterns"},
                                          {"SiLU/upstream fixture"},
                                          {"window exp/upstream fixture"},
                                          {"ViT exp/independent oracle"},
                                          {"E4 publication to half"},
                                          {"E4 decode/all byte patterns"}}};
    const auto &encode = fixture.sections.at("E4EN");
    const auto &decode = fixture.sections.at("E4DE");
    for (size_t i = 0; i < halfCount; ++i) {
        const auto *got = output.data() + i * halfFields;
        checks[0].compare(i, got[0], encode[i]);
        checks[1].compare(i, got[1], encode[i]);
        checks[2].compare(i, got[2], num::f16Bits(num::f16ToF32(static_cast<uint16_t>(i))), true);
        if (finiteHalf(static_cast<uint32_t>(i))) {
            checks[3].compare(i, got[3], read16(fixture.sections.at("SILU").data() + i * 2), true);
            checks[4].compare(i, got[4], read16(fixture.sections.at("EXPW").data() + i * 2), true);
            checks[5].compare(i, got[5], vitOracle(static_cast<uint16_t>(i)), true);
        } else {
            ++checks[3].skipped;
            ++checks[4].skipped;
            ++checks[5].skipped;
        }
        const uint32_t publishedFloat = read32(decode.data() + size_t(encode[i]) * 4);
        checks[6].compare(i, got[6], num::f16Bits(num::f32FromBits(publishedFloat)), true);
        if (i < 256)
            checks[7].compare(i, got[7], read32(decode.data() + i * 4));
    }
    bool ok = true;
    for (const auto &check : checks)
        ok &= check.finish(backend);
    return ok;
}

std::vector<uint32_t> floatInputs() {
    std::vector<uint32_t> input;
    Xorshift rng{0x9e3779b9u};
    for (size_t i = 0; i < 32768; ++i)
        input.push_back(rng.next());
    auto around = [&](float x) {
        for (float value : {std::nextafter(x, -std::numeric_limits<float>::infinity()), x,
                            std::nextafter(x, std::numeric_limits<float>::infinity())}) {
            input.push_back(num::f32Bits(value));
            input.push_back(num::f32Bits(-value));
        }
    };
    // Both sides of every finite half midpoint expose ties and subnormal/normal transitions.
    for (uint32_t h = 0; h < 0x7bff; ++h)
        around((num::f16ToF32(static_cast<uint16_t>(h)) +
                num::f16ToF32(static_cast<uint16_t>(h + 1))) *
               0.5f);
    around(65520.0f);
    // E4 midpoints expose accidental direct f32 -> E4 conversions (double rounding).
    for (uint32_t code = 0; code < 0x7e; ++code)
        around((num::e4m3ToF32(static_cast<uint8_t>(code)) +
                num::e4m3ToF32(static_cast<uint8_t>(code + 1))) *
               0.5f);
    for (uint32_t bits :
         {0u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 1u, 0x80000001u})
        input.push_back(bits);
    return input;
}
bool checkFloats(const Fixture &fixture, const std::vector<uint32_t> &input,
                 const std::vector<uint32_t> &out, const char *backend) {
    Check half{"f32-to-half/fixture+midpoints"}, e4{"two-stage f32-to-E4"},
        independent{"CPU oracle vs value-table oracle"};
    for (size_t i = 0; i < input.size(); ++i) {
        const float value = num::f32FromBits(input[i]);
        const uint16_t expected =
            i < 32768 ? read16(fixture.sections.at("F16B").data() + i * 2) : num::f16Bits(value);
        half.compare(i, out[i * 2], expected, true);
        e4.compare(i, out[i * 2 + 1], num::e4m3FromF16Bits(expected));
        independent.compare(i, expected, nearestHalf(value), true);
    }
    const bool a = half.finish(backend), b = e4.finish(backend), c = independent.finish(backend);
    return a && b && c;
}
std::vector<uint16_t> arithmeticInputs() {
    std::vector<uint16_t> input{0x3c01, 0x3c01,
                                0xbc02}; // FMA -> 2^-20, separate multiply/add -> +0.
    Xorshift rng{0xd2511f53u};
    for (size_t i = 1; i < halfCount; ++i) {
        input.push_back(static_cast<uint16_t>(i));
        input.push_back(makeFiniteHalf(rng.next()));
        input.push_back(makeFiniteHalf(rng.next()));
    }
    return input;
}
bool checkArithmetic(const std::vector<uint16_t> &input, const std::vector<uint32_t> &out,
                     const char *backend) {
    std::array<Check, arithmeticFields> checks{{{"half add/independent oracle"},
                                                {"half subtract/independent oracle"},
                                                {"half multiply/independent oracle"},
                                                {"half FMA/independent oracle"},
                                                {"separate multiply then add"}}};
    for (size_t i = 0; i < input.size() / 3; ++i) {
        const double a = num::f16ToF32(input[i * 3]), b = num::f16ToF32(input[i * 3 + 1]),
                     c = num::f16ToF32(input[i * 3 + 2]);
        const std::array<uint16_t, arithmeticFields> expected{
            {nearestHalf(a + b), nearestHalf(a - b), nearestHalf(a * b),
             nearestHalf(std::fma(a, b, c)),
             nearestHalf(double(num::f16ToF32(nearestHalf(a * b))) + c)}};
        for (size_t field = 0; field < arithmeticFields; ++field)
            checks[field].compare(i, out[i * arithmeticFields + field], expected[field], true);
    }
    bool ok = true;
    for (const auto &check : checks)
        ok &= check.finish(backend);
    if (out[3] != 0x0010 || out[4] != 0) {
        std::fprintf(stderr, "%s FMA/separate-rounding witness failed\n", backend);
        ok = false;
    }
    return ok;
}

void cudaCheck(cudaError_t error, const char *expression) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(error));
}
#define CUDA_CHECK(expression) cudaCheck((expression), #expression)
template <class T> struct DeviceBuffer {
    T *data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (data)
            cudaFree(data);
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};
} // namespace

int main(int argc, char **argv) {
    try {
        bool hostOnly = false;
        std::string path = NR_NUMERICS_FIXTURE_PATH;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--host-only")
                hostOnly = true;
            else if (option == "--fixture" && i + 1 < argc)
                path = argv[++i];
            else
                throw std::runtime_error("usage: nr-numerics-test [--host-only] [--fixture PATH]");
        }
        const Fixture fixture = readFixture(path);
        const auto f32 = floatInputs();
        const auto arithmetic = arithmeticInputs();
        std::vector<uint32_t> halves(halfCount * halfFields), floats(f32.size() * 2),
            operations((arithmetic.size() / 3) * arithmeticFields);
        for (size_t i = 0; i < halfCount; ++i)
            evaluateHalf(static_cast<uint32_t>(i), halves.data() + i * halfFields);
        for (size_t i = 0; i < f32.size(); ++i) {
            const float value = num::f32FromBits(f32[i]);
            floats[i * 2] = nr::halfBits(nr::halfRne(value));
            floats[i * 2 + 1] = nr::e4m3FromFloat(value);
        }
        for (size_t i = 0; i < arithmetic.size() / 3; ++i)
            evaluateArithmetic(arithmetic.data() + i * 3, operations.data() + i * arithmeticFields);
        bool ok = checkHalves(fixture, halves, "HOST");
        ok &= checkFloats(fixture, f32, floats, "HOST");
        ok &= checkArithmetic(arithmetic, operations, "HOST");
        if (!ok)
            return 1;
        if (hostOnly) {
            std::puts("HOST NUMERICS PASS; GPU checks explicitly not run");
            return 0;
        }
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        std::printf("GPU %s sm_%d%d\n", properties.name, properties.major, properties.minor);
        DeviceBuffer<uint32_t> dHalves(halves.size()), dFloats(floats.size()), dF32(f32.size()),
            dOperations(operations.size());
        DeviceBuffer<uint16_t> dArithmetic(arithmetic.size());
        CUDA_CHECK(cudaMemcpy(dF32.data, f32.data(), f32.size() * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dArithmetic.data, arithmetic.data(),
                              arithmetic.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        halfKernel<<<(halfCount + 255) / 256, 256>>>(dHalves.data);
        CUDA_CHECK(cudaGetLastError());
        floatKernel<<<(f32.size() + 255) / 256, 256>>>(dF32.data, f32.size(), dFloats.data);
        CUDA_CHECK(cudaGetLastError());
        arithmeticKernel<<<(arithmetic.size() / 3 + 255) / 256, 256>>>(
            dArithmetic.data, arithmetic.size() / 3, dOperations.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(halves.data(), dHalves.data, halves.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(floats.data(), dFloats.data, floats.size() * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(operations.data(), dOperations.data,
                              operations.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        ok = checkHalves(fixture, halves, "GPU");
        ok &= checkFloats(fixture, f32, floats, "GPU");
        ok &= checkArithmetic(arithmetic, operations, "GPU");
        std::puts(ok ? "GPU NUMERICS PASS" : "GPU NUMERICS FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "NUMERICS ERROR: %s\n", error.what());
        return 2;
    }
}
