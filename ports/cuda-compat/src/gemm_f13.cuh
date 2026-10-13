// Browser-compatible F13 GEMM with compact decoded-E4 operands.
#pragma once
#include <cuda_runtime.h>
#include <climits>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include "browser_numerics.cuh"

#define NR_F13_GEMM_HD __host__ __device__ __forceinline__
namespace nr {
namespace f13_gemm_detail {
using H = __half;
NR_F13_GEMM_HD int exponent(float x, int floor) {
    const int e = int((browser_detail::floatBits(x) >> 23) & 255u) - 127;
    return e < floor ? floor : e;
}
NR_F13_GEMM_HD bool nonfinite(float x) {
    return (browser_detail::floatBits(x) & 0x7f800000u) == 0x7f800000u;
}
NR_F13_GEMM_HD int start_exponent(float x) {
    return (browser_detail::floatBits(x) & 0x7fffffffu) ? exponent(x, -14) : -21;
}
NR_F13_GEMM_HD float power(int e) {
#if defined(__CUDA_ARCH__)
    return __int_as_float((e + 127) << 23);
#else
    return std::ldexp(1.f, e);
#endif
}
NR_F13_GEMM_HD float multiply(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
NR_F13_GEMM_HD int truncate(float x) {
#if defined(__CUDA_ARCH__)
    return __float2int_rz(x);
#else
    return int(std::trunc(x));
#endif
}

// Decoded E4 FP32 values leave low mantissa bits available for exponents.
NR_F13_GEMM_HD uint32_t pack_compact(H value) {
    const int e = int((halfBits(value) >> 10) & 31) - 15;
    return browser_detail::floatBits(halfToFloat(value)) | uint32_t((e < -6 ? -6 : e) + 6);
}
NR_F13_GEMM_HD float compact_value(uint32_t record) {
    const uint32_t bits = record & ~15u;
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}
NR_F13_GEMM_HD float step_compact(const uint32_t *a, int as, const uint32_t *b, int bs,
                                  float incoming) {
    if (nonfinite(incoming))
        return incoming;
    int maximum = start_exponent(incoming);
    float products[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t left = a[i * as], right = b[i * bs];
        products[i] = multiply(compact_value(left), compact_value(right));
        const int e = int(left & 15u) + int(right & 15u) - 12;
        if (products[i] != 0.f && e > maximum)
            maximum = e;
    }
    const float alignment = power(13 - maximum);
    int units = truncate(multiply(incoming, alignment));
#pragma unroll
    for (int i = 0; i < 16; ++i)
        units += truncate(multiply(products[i], alignment));
    return multiply(float(units), power(maximum - 13));
}

static __global__ void gemm(const H *a, const H *weights, H *output, H *raw, const H *residual,
                            const H *scale, float *diagnostic_wide, int M, int K, int N,
                            int experts, bool broadcast, bool silu, int partition, bool quantize) {
    const int tx = threadIdx.x, ty = threadIdx.y, row = int(blockIdx.y) * 16 + ty,
              column = int(blockIdx.x) * 16 + tx, expert = blockIdx.z;
    const size_t as = size_t(K) * (broadcast ? 1 : experts), os = size_t(N) * experts;
    const size_t at = size_t(row) * os + size_t(expert) * N + column;
    const bool valid = row < M && column < N;
    float accumulator = valid && residual ? __fmul_rn(halfToFloat(residual[at]),
                                                      halfToFloat(scale[expert * N + column]))
                                          : 0.f,
          total = 0.f;
    __shared__ uint32_t av[16][16], bv[16][16];
    for (int start = 0; start < K; start += 16) {
        const H left = row < M
                           ? a[size_t(row) * as + (broadcast ? 0 : size_t(expert) * K) + start + tx]
                           : halfRne(0.f);
        const H right =
            column < N ? weights[(size_t(expert) * K + start + ty) * N + column] : halfRne(0.f);
        av[ty][tx] = pack_compact(left);
        bv[ty][tx] = pack_compact(right);
        __syncthreads();
        if (valid) {
            accumulator = step_compact(&av[ty][0], 1, &bv[0][tx], 16, accumulator);
            if (partition && ((start + 16) % partition == 0 || start + 16 == K)) {
                total = start < partition ? accumulator : __fadd_rn(total, accumulator);
                accumulator = 0.f;
            }
        }
        __syncthreads();
    }
    if (valid) {
        const float value = partition ? total : accumulator;
        if (diagnostic_wide)
            diagnostic_wide[at] = value;
        const H index = browserHalfRtz(value), published = silu ? browserSiluHalf(index) : index;
        if (raw)
            raw[at] = published;
        output[at] = quantize ? (silu ? browserSiluE4(index) : browserPublishE4(value)) : published;
    }
}
inline bool ranges_overlap(const void *first, size_t first_bytes, const void *second,
                           size_t second_bytes) {
    if (!first || !second)
        return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(first), b = reinterpret_cast<uintptr_t>(second);
    return a <= b ? b - a < first_bytes : a - b < second_bytes;
}

} // namespace f13_gemm_detail

inline void nr_gemm_f13(const __half *a, const __half *weights, __half *output, __half *raw,
                        const __half *residual, const __half *scale, int M, int K, int N,
                        int experts, int broadcast, int silu, int partition, int quantize,
                        cudaStream_t stream, float *diagnostic_wide = nullptr) {
    if (!a || !weights || !output || M <= 0 || K <= 0 || N <= 0 || K % 16 || experts < 1 ||
        experts > 65535 || M > 65535 * 16 || N > INT_MAX - 15 || partition < 0 || partition % 16 ||
        (residual && !scale) || (broadcast != 0 && broadcast != 1) || (silu != 0 && silu != 1) ||
        (quantize != 0 && quantize != 1))
        throw std::invalid_argument("Invalid F13 GEMM shape or pointers");
    const size_t os = size_t(N) * experts;
    if (os > INT_MAX || size_t(M) > size_t(INT_MAX) / os || size_t(K) > size_t(INT_MAX) / os ||
        size_t(K) * (broadcast ? 1 : experts) > INT_MAX)
        throw std::invalid_argument("F13 GEMM tensor exceeds bounded count");
    const size_t input_bytes = size_t(M) * K * (broadcast ? 1 : experts) * sizeof(__half),
                 weights_bytes = size_t(K) * os * sizeof(__half);
    const size_t output_bytes = size_t(M) * os * sizeof(__half), scale_bytes = os * sizeof(__half);
    for (const void *target : {static_cast<const void *>(output), static_cast<const void *>(raw)})
        if (f13_gemm_detail::ranges_overlap(target, output_bytes, a, input_bytes) ||
            f13_gemm_detail::ranges_overlap(target, output_bytes, weights, weights_bytes) ||
            f13_gemm_detail::ranges_overlap(target, output_bytes, scale, scale_bytes) ||
            (target != residual &&
             f13_gemm_detail::ranges_overlap(target, output_bytes, residual, output_bytes)))
            throw std::invalid_argument("F13 GEMM output overlaps an input");
    if (f13_gemm_detail::ranges_overlap(output, output_bytes, raw, output_bytes))
        throw std::invalid_argument("F13 GEMM output overlaps raw");
    if (diagnostic_wide) {
        const size_t wide_bytes = size_t(M) * os * sizeof(float);
        if (f13_gemm_detail::ranges_overlap(diagnostic_wide, wide_bytes, a, input_bytes) ||
            f13_gemm_detail::ranges_overlap(diagnostic_wide, wide_bytes, weights, weights_bytes) ||
            f13_gemm_detail::ranges_overlap(diagnostic_wide, wide_bytes, residual, output_bytes) ||
            f13_gemm_detail::ranges_overlap(diagnostic_wide, wide_bytes, scale, scale_bytes) ||
            f13_gemm_detail::ranges_overlap(diagnostic_wide, wide_bytes, output, output_bytes) ||
            f13_gemm_detail::ranges_overlap(diagnostic_wide, wide_bytes, raw, output_bytes))
            throw std::invalid_argument("F13 GEMM wide output overlaps a tensor");
    }
    const dim3 grid((N + 15) / 16, (M + 15) / 16, experts), block(16, 16);
    f13_gemm_detail::gemm<<<grid, block, 0, stream>>>(a, weights, output, raw, residual, scale,
                                                      diagnostic_wide, M, K, N, experts, broadcast,
                                                      silu, partition, quantize);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace nr
#undef NR_F13_GEMM_HD
