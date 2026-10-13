// F13 K16 reduction with half publication, used by staged attention.
// Numerical semantics derive from OpenDLSS-NR (MIT, Copyright (c) 2026 maan).
#pragma once
#include <cuda_runtime.h>
#include <climits>
#include <stdexcept>
#include "kernels/numerics/numerics.cuh"

namespace nr {
namespace f13_detail {
using Half = __half;

__device__ __forceinline__ int exponent(Half value, int floor) {
    const int e = int((halfBits(value) >> 10) & 31) - 15;
    return e < floor ? floor : e;
}

// Inputs must be finite E4M3 values decoded exactly to half. A half accumulator
// is aligned with sixteen products, each term is truncated toward zero, and
// the integer total is rounded once to half. Every aligned term fits 15 bits;
// all seventeen terms fit 20 bits, so int32 addition is exact and order-free.
__device__ __forceinline__ Half step16(const Half *a, int a_stride, const Half *b, int b_stride,
                                       Half incoming) {
    const unsigned incoming_bits = halfBits(incoming);
    if ((incoming_bits & 0x7c00u) == 0x7c00u)
        return incoming;
    int maximum = (incoming_bits & 0x7fffu) ? exponent(incoming, -14) : -21;
    float products[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const Half left = a[i * a_stride], right = b[i * b_stride];
        products[i] = __fmul_rn(halfToFloat(left), halfToFloat(right));
        if ((halfBits(left) & 0x7fffu) && (halfBits(right) & 0x7fffu)) {
            const int product_exponent = exponent(left, -6) + exponent(right, -6);
            if (product_exponent > maximum)
                maximum = product_exponent;
        }
    }
    const float alignment = __int_as_float((140 - maximum) << 23);
    int units = __float2int_rz(__fmul_rn(halfToFloat(incoming), alignment));
#pragma unroll
    for (int i = 0; i < 16; ++i)
        units += __float2int_rz(__fmul_rn(products[i], alignment));
    const float unit = __int_as_float((maximum + 114) << 23);
    return halfRne(__fmul_rn(float(units), unit));
}

} // namespace f13_detail
} // namespace nr
