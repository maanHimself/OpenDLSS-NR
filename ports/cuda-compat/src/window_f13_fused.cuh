// Browser-compatible fused window attention with 32 warps per block.
#pragma once
#include <cuda_runtime.h>
#include <climits>
#include <stdexcept>
#include "browser_numerics.cuh"
#include "window_common.cuh"

namespace nr_window_f13_detail {
using H = __half;
using nr::halfRne;
using nr::halfToFloat;
using nr::mulHalf;
using nr::publishE4;
constexpr int Warps = 32;

__host__ __device__ __forceinline__ float compact_value(uint32_t record) {
    const uint32_t bits = record & ~15u;
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}

__host__ __device__ __forceinline__ uint32_t encode(H value) {
    // Finite E4 values leave low FP32 mantissa bits free for the floor exponent.
    const int e = int((nr::halfBits(value) >> 10) & 31) - 15;
    return nr::browser_detail::floatBits(halfToFloat(value)) | uint32_t((e < -6 ? -6 : e) + 6);
}

__host__ __device__ __forceinline__ float compact_step(const uint32_t *a, int as, const uint32_t *b,
                                                       int bs, float incoming) {
    const unsigned bits = nr::browser_detail::floatBits(incoming);
    if ((bits & 0x7f800000u) == 0x7f800000u)
        return incoming;
    const int ie = int((bits >> 23) & 255) - 127;
    int maximum = (bits & 0x7fffffffu) ? (ie < -14 ? -14 : ie) : -21;
    float products[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t left = a[i * as], right = b[i * bs];
#if defined(__CUDA_ARCH__)
        products[i] = __fmul_rn(__uint_as_float(left & ~15u), __uint_as_float(right & ~15u));
#else
        products[i] = compact_value(left) * compact_value(right);
#endif
        const int e = int(left & 15u) + int(right & 15u) - 12;
        if (products[i] != 0.f && e > maximum)
            maximum = e;
    }
#if defined(__CUDA_ARCH__)
    const float alignment = __int_as_float((140 - maximum) << 23);
    int units = __float2int_rz(__fmul_rn(incoming, alignment));
#pragma unroll
    for (int i = 0; i < 16; ++i)
        units += __float2int_rz(__fmul_rn(products[i], alignment));
    return __fmul_rn(float(units), __int_as_float((maximum + 114) << 23));
#else
    const float alignment = std::ldexp(1.f, 13 - maximum);
    int units = int(std::trunc(incoming * alignment));
    for (int i = 0; i < 16; ++i)
        units += int(std::trunc(products[i] * alignment));
    return std::ldexp(float(units), maximum - 13);
#endif
}

inline bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < an : x - y < bn;
}

__global__ __launch_bounds__(Warps * 32) void window(const H *__restrict__ qkv,
                                                     const H *__restrict__ prior,
                                                     const float *__restrict__ scales,
                                                     H *__restrict__ output, int width, int height,
                                                     int heads, int sx, int sy, int windows_x) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int head = blockIdx.x % heads, win = blockIdx.x / heads;
    const int ox = (win % windows_x) * 8 - sx, oy = (win / windows_x) * 8 - sy;
    const H scale = __float2half_rz(scales[head]);
    // Transposed K gives contiguous per-warp
    // key loads; Q/P are broadcasts and V uses contiguous channel loads.
    // Each warp reuses just 64 probability packets for its current query.
    __shared__ __align__(32) uint32_t q[64 * 32], k[32 * 64], v[64 * 32];
    __shared__ __align__(32) uint32_t probabilities[Warps][64];
    for (int token = warp; token < 64; token += Warps) {
        const int x = ox + (token & 7), y = oy + (token >> 3);
        const bool valid = x >= 0 && y >= 0 && x < width && y < height;
        H qh = halfRne(0.f), kh = qh, vh = qh;
        if (valid) {
            const size_t at = (size_t(y) * width + x) * heads * 96 + head * 96 + lane;
            qh = qkv[at];
            kh = qkv[at + 32];
            vh = qkv[at + 64];
        }
        const H qi = nr_window_detail::inverse_norm_warp(qh, lane);
        const H ki = nr_window_detail::inverse_norm_warp(kh, lane);
        const int physical = nr_window_detail::physical_token(token);
        q[token * 32 + lane] = valid ? encode(publishE4(mulHalf(mulHalf(qh, qi), scale))) : 0;
        k[lane * 64 + physical] = valid ? encode(publishE4(mulHalf(kh, ki))) : 0;
        v[physical * 32 + lane] = valid ? encode(publishE4(vh)) : 0;
    }
    __syncthreads();
    for (int row = warp; row < 64; row += Warps) {
        H score0 = prior[head * 4096 + row * 64 + lane];
        H score1 = prior[head * 4096 + row * 64 + lane + 32];
#pragma unroll
        for (int channel = 0; channel < 32; channel += 16) {
            score0 = halfRne(compact_step(q + row * 32 + channel, 1, k + channel * 64 + lane, 64,
                                          halfToFloat(score0)));
            score1 = halfRne(compact_step(q + row * 32 + channel, 1, k + channel * 64 + lane + 32,
                                          64, halfToFloat(score1)));
        }
        const H e0 = nr::windowExpWeight(score0), e1 = nr::windowExpWeight(score1);
        const H total = nr_window_detail::softmax_total_warp(e0, e1, lane);
        H reciprocal = halfRne(0.f);
        if (lane == 0)
            reciprocal = __float2half_rz(__fdiv_rn(1.f, halfToFloat(total)));
        reciprocal = nr_window_detail::shuffle_half(reciprocal, 0);
        probabilities[warp][lane] = encode(publishE4(mulHalf(e0, reciprocal)));
        probabilities[warp][lane + 32] = encode(publishE4(mulHalf(e1, reciprocal)));
        __syncwarp();
        H result = halfRne(0.f);
#pragma unroll
        for (int key = 0; key < 64; key += 16)
            result = halfRne(compact_step(probabilities[warp] + key, 1, v + key * 32 + lane, 32,
                                          halfToFloat(result)));
        const int x = ox + (row & 7), y = oy + (row >> 3);
        if (x >= 0 && y >= 0 && x < width && y < height)
            output[(size_t(y) * width + x) * heads * 32 + head * 32 + lane] = publishE4(result);
        __syncwarp(); // Every lane finishes reading before this warp reuses P.
    }
}

} // namespace nr_window_f13_detail

inline void nr_window_f13_fused(const __half *qkv, const __half *prior, const float *scales,
                                __half *output, int width, int height, int heads, int phase,
                                cudaStream_t stream) {
    if (!qkv || !prior || !scales || !output || width <= 0 || height <= 0 || heads <= 0 ||
        heads > 16 || phase < 0 || phase > 3 ||
        size_t(width) > size_t(INT_MAX) / 96 / size_t(height) / size_t(heads))
        throw std::invalid_argument("Invalid fused window shape or pointers");
    const size_t pixels = size_t(width) * height * heads,
                 output_bytes = pixels * 32 * sizeof(__half);
    if (nr_window_f13_detail::overlaps(output, output_bytes, qkv, pixels * 96 * sizeof(__half)) ||
        nr_window_f13_detail::overlaps(output, output_bytes, prior,
                                       size_t(heads) * 4096 * sizeof(__half)) ||
        nr_window_f13_detail::overlaps(output, output_bytes, scales, size_t(heads) * sizeof(float)))
        throw std::invalid_argument("Exact fused window output overlaps an input");
    const int sx = phase == 1 || phase == 2 ? 4 : 0, sy = phase == 1 || phase == 3 ? 4 : 0;
    const int wx = (width + sx + 7) / 8, wy = (height + sy + 7) / 8;
    const size_t tasks = size_t(wx) * wy * heads;
    if (tasks > INT_MAX)
        throw std::invalid_argument("Exact fused window grid exceeds CUDA limit");
    nr_window_f13_detail::window<<<unsigned(tasks), 1024, 0, stream>>>(
        qkv, prior, scales, output, width, height, heads, sx, sy, wx);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
