// F24 K8 reduction for the adapter and head.
// Numerical semantics derive from OpenDLSS-NR src/reference.cpp (MIT, 2026 maan).
#pragma once
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include "kernels/numerics/numerics.cuh"

namespace nr {
namespace f16_compat_detail {
using H = __half;
__device__ __forceinline__ int exponent(H value) {
    const int e = int((halfBits(value) >> 10) & 31) - 15;
    return e < -14 ? -14 : e;
}
// Convert through FP32 with round-to-odd, not round-to-nearest. Any discarded
// integer bits set the FP32 low significand bit, so they cannot become a false
// half midpoint. FP32 has 24 bits versus half's <=11; this preserves final RNE
// for normals, subnormals, zero and overflow. The power-of-two scale is exact:
// |fixed_sum| < 2^30 and binary_exponent is in [-45,6], far inside FP32's range.
__host__ __device__ __forceinline__ H fixed_to_half(int fixed_sum, int binary_exponent) {
#if defined(__CUDA_ARCH__)
    const float truncated = __int2float_rz(fixed_sum);
    const unsigned sticky = unsigned(__float2int_rz(truncated) != fixed_sum);
    const float odd = __uint_as_float(__float_as_uint(truncated) | sticky);
    const float scale = __int_as_float((binary_exponent + 127) << 23);
    return halfRne(__fmul_rn(odd, scale));
#else
    float truncated = float(fixed_sum);
    if (std::abs(double(truncated)) > std::abs(double(fixed_sum)))
        truncated = std::nextafter(truncated, 0.f);
    unsigned bits;
    std::memcpy(&bits, &truncated, sizeof(bits));
    bits |= unsigned(int(truncated) != fixed_sum);
    std::memcpy(&truncated, &bits, sizeof(bits));
    return halfRne(std::ldexp(truncated, binary_exponent));
#endif
}
__device__ __forceinline__ H step8(const H *a, int as, const H *b, int bs, H incoming) {
    const unsigned incoming_bits = halfBits(incoming);
    // Nonfinite accumulators propagate; the pinned finite-input fixture covers
    // the exact arithmetic below. Normal model adapter/head operands are finite.
    if ((incoming_bits & 0x7c00u) == 0x7c00u)
        return incoming;
    int maximum = (incoming_bits & 0x7fffu) ? exponent(incoming) : -21;
    float products[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const H left = a[i * as], right = b[i * bs];
        products[i] = __fmul_rn(halfToFloat(left), halfToFloat(right));
        if ((halfBits(left) & 0x7fffu) && (halfBits(right) & 0x7fffu)) {
            const int e = exponent(left) + exponent(right);
            if (e > maximum)
                maximum = e;
        }
    }
    const float alignment = __int_as_float((151 - maximum) << 23);
    // A finite half with exponent e has |x| < 2^(e+1), including the
    // clamped subnormal exponent. Each aligned product is < 2^26 and
    // the incoming term is < 2^25. Thus every signed partial sum has
    // magnitude < 8*2^26 + 2^25 = 17*2^25 < 2^30: int32 is sufficient.
    int units = __float2int_rz(__fmul_rn(halfToFloat(incoming), alignment));
#pragma unroll
    for (int i = 0; i < 8; ++i)
        units += __float2int_rz(__fmul_rn(products[i], alignment));
    return fixed_to_half(units, maximum - 24);
}
template <int FixedK = 0, int FixedN = 0>
static __global__ void kernel(const H *a, const H *weights, H *output, H *quantized, int M, int K,
                              int N) {
    const int width = FixedK ? FixedK : K, columns = FixedN ? FixedN : N;
    // The entry point bounds M*N by INT_MAX. The two production shapes
    // turn these divisions into shifts/masks; byte addresses remain size_t.
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= unsigned(M) * unsigned(columns))
        return;
    const unsigned row = i / unsigned(columns), column = i % unsigned(columns);
    H accumulator = halfRne(0.f);
    if constexpr (FixedK) {
#pragma unroll
        for (int start = 0; start < FixedK; start += 8)
            accumulator = step8(a + size_t(row) * width + start, 1,
                                weights + size_t(start) * columns + column, columns, accumulator);
    } else {
        for (int start = 0; start < width; start += 8)
            accumulator = step8(a + size_t(row) * width + start, 1,
                                weights + size_t(start) * columns + column, columns, accumulator);
    }
    output[i] = accumulator;
    if (quantized)
        quantized[i] = publishE4(accumulator);
}
} // namespace f16_compat_detail

// A:[M,K], W:[K,N], output/optional decoded-E4 output:[M,N]. No allocation,
// synchronization or capture restrictions. The caller owns the supplied stream.
inline void nr_f16_compat(const __half *a, const __half *weights, __half *output, __half *quantized,
                          int M, int K, int N, cudaStream_t stream) {
    if (!a || !weights || !output || M <= 0 || K <= 0 || N <= 0 || K % 8 || size_t(M) * N > INT_MAX)
        throw std::invalid_argument("Invalid F16 compatibility GEMM arguments");
    const unsigned blocks = unsigned((size_t(M) * N + 255) / 256);
    if (K == 16 && N == 32)
        f16_compat_detail::kernel<16, 32>
            <<<blocks, 256, 0, stream>>>(a, weights, output, quantized, M, K, N);
    else if (K == 32 && N == 4)
        f16_compat_detail::kernel<32, 4>
            <<<blocks, 256, 0, stream>>>(a, weights, output, quantized, M, K, N);
    else
        f16_compat_detail::kernel<>
            <<<blocks, 256, 0, stream>>>(a, weights, output, quantized, M, K, N);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace nr
