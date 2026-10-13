// Shared native-test utilities and independent half-value oracle.
#pragma once
#include <cuda_runtime.h>
#include "kernels/numerics/numerics.cuh"
#include "../tests/fixtures/upstream_numeric.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace nr_test {
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
H decode(unsigned byte) {
    // The model decoder maps both E4 NaN codes to signed zero.
    const float value =
        (byte & 127) == 127 ? ((byte & 128) ? -0.f : 0.f) : num::e4m3ToF32(uint8_t(byte));
    return nr::halfRne(value);
}
// Independent nearest-value half oracle; no CUDA rounding intrinsic is used.
H nearest(double value) {
    static const std::vector<double> grid = [] {
        std::vector<double> values(0x7c00);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = num::f16ToF32(uint16_t(i));
        return values;
    }();
    const uint16_t sign = std::signbit(value) ? 0x8000 : 0;
    if (std::isnan(value))
        return nr::halfFromBits(sign | 0x7e00);
    const double magnitude = std::abs(value);
    if (magnitude >= 65520)
        return nr::halfFromBits(sign | 0x7c00);
    const auto found = std::lower_bound(grid.begin(), grid.end(), magnitude);
    if (found == grid.end())
        return nr::halfFromBits(sign | 0x7bff);
    const size_t upper = found - grid.begin();
    if (!upper || *found == magnitude)
        return nr::halfFromBits(sign | uint16_t(upper));
    const size_t lower = upper - 1;
    const double down = magnitude - grid[lower], up = grid[upper] - magnitude;
    const size_t chosen = down < up ? lower : up < down ? upper : (lower % 2 ? upper : lower);
    return nr::halfFromBits(sign | uint16_t(chosen));
}
double number(H value) {
    return num::f16ToF32(nr::halfBits(value));
}
int exponent(double value, int lower) {
    int power = 0;
    std::frexp(std::abs(value), &power);
    return std::max(power - 1, lower);
}
H publication(H value) {
    return decode(num::e4m3FromF16Bits(nr::halfBits(value)));
}
void same(const std::vector<H> &actual, const std::vector<H> &expected, const char *name) {
    if (actual.size() != expected.size())
        throw std::runtime_error("Wrong result size");
    for (size_t i = 0; i < actual.size(); ++i)
        if (nr::halfBits(actual[i]) != nr::halfBits(expected[i])) {
            std::fprintf(stderr, "%s mismatch at %zu: %04x != %04x\n", name, i,
                         nr::halfBits(actual[i]), nr::halfBits(expected[i]));
            throw std::runtime_error("F13 bit mismatch");
        }
}
template <class T> struct Device {
    T *data = nullptr;
    explicit Device(size_t size) {
        check(cudaMalloc(&data, size * sizeof(T)));
    }
    ~Device() {
        if (data)
            cudaFree(data);
    }
    void upload(const std::vector<T> &source) {
        check(cudaMemcpy(data, source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice));
        // Pageable H2D can return after host staging, before device DMA ends.
        // Tests use a nonblocking compute stream, so order it after this copy.
        check(cudaStreamSynchronize(nullptr));
    }
    void download(std::vector<T> &target) {
        check(cudaMemcpy(target.data(), data, target.size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
};
} // namespace nr_test
