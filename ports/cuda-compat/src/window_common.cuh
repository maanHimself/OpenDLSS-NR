// Warp reductions and token ordering for browser-compatible window attention.
#pragma once
#include "kernels/numerics/numerics.cuh"

namespace nr_window_detail {
using Half = __half;
using nr::addHalf;
using nr::halfBits;
using nr::halfFromBits;
using nr::halfToFloat;
using nr::mulHalf;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ __forceinline__ Half shuffle_half(Half value, int source_lane) {
    const unsigned bits = __shfl_sync(kWarpMask, unsigned(halfBits(value)), source_lane);
    return halfFromBits(static_cast<uint16_t>(bits));
}

// Each lane holds one of the 32 input channels. Match the pinned WGSL's FP32
// pair sum followed by half publication, then the half strides 8,4,2,1 tree.
// All lanes execute all shuffles, including lanes outside each active subtree.
__device__ __forceinline__ Half inverse_norm_warp(Half value, int lane) {
    const Half partner = shuffle_half(value, lane ^ 16);
    const float low_square = __fmul_rn(halfToFloat(value), halfToFloat(value));
    const float high_square = halfToFloat(mulHalf(partner, partner));
    const float pair = __fadd_rn(low_square, high_square);
    Half partial = __float2half_rz(pair);
#pragma unroll
    for (int stride = 8; stride > 0; stride >>= 1) {
        const Half other = shuffle_half(partial, (lane + stride) & 31);
        if (lane < stride)
            partial = addHalf(partial, other);
    }
    if (lane == 0) {
        partial = __float2half_rz(__frsqrt_rn(halfToFloat(partial)));
    }
    return shuffle_half(partial, 0);
}

// Exact upstream physical-key reduction. Each lane holds two exponentials:
// e0=e[lane], e1=e[lane+32]. Lanes 0..7 produce t[g], lanes 0/1 produce the
// even/odd chains, then lane 0's total is broadcast. Every addition is half.
__device__ __forceinline__ Half softmax_total_warp(Half e0, Half e1, int lane) {
    const Half b0 = addHalf(e0, shuffle_half(e0, (lane + 8) & 31));
    const Half b1 = addHalf(shuffle_half(e0, (lane + 16) & 31), shuffle_half(e0, (lane + 24) & 31));
    const Half b2 = addHalf(e1, shuffle_half(e1, (lane + 8) & 31));
    const Half b3 = addHalf(shuffle_half(e1, (lane + 16) & 31), shuffle_half(e1, (lane + 24) & 31));
    const Half t = addHalf(addHalf(addHalf(b0, b1), b2), b3);
    const Half t2 = shuffle_half(t, (lane + 2) & 31);
    const Half t4 = shuffle_half(t, (lane + 4) & 31);
    const Half t6 = shuffle_half(t, (lane + 6) & 31);
    const Half parity_sum = addHalf(addHalf(addHalf(t, t2), t4), t6);
    const Half even = shuffle_half(parity_sum, 0);
    const Half odd = shuffle_half(parity_sum, 1);
    return addHalf(even, odd);
}

__device__ __forceinline__ int physical_token(int natural) {
    const int x = natural & 7, y = natural >> 3;
    return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3);
}

} // namespace nr_window_detail
