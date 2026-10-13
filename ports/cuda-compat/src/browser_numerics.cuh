// Publication rules for the captured Chrome/Vulkan compatibility profile.
// Exhaustive finite-half answers are in tests/fixtures/browser-silu-observed.bin.
#pragma once
#include "kernels/numerics/numerics.cuh"

#define NR_BROWSER_HD __host__ __device__ __forceinline__

namespace nr {
namespace browser_detail {
NR_BROWSER_HD uint32_t floatBits(float value) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}
NR_BROWSER_HD float madd(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
NR_BROWSER_HD float multiply(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
} // namespace browser_detail

NR_BROWSER_HD __half browserHalfRtz(float value) {
#if defined(__CUDA_ARCH__)
    return __float2half_rz(value);
#else
    __half result = halfRne(value);
    if (std::fabs(halfToFloat(result)) > std::fabs(value))
        result = halfFromBits(static_cast<uint16_t>(halfBits(result) - 1u));
    return result;
#endif
}

// Direct FP32 -> E4, with no intermediate half publication. Normal and
// subnormal E4 use ties-to-even; finite overflow saturates at 448. Exact zero
// and NaN publish +0, matching the packed WGSL publication function.
NR_BROWSER_HD uint8_t browserE4Code(float value) {
    const uint32_t original = browser_detail::floatBits(value);
    const uint32_t magnitude_bits = original & 0x7fffffffu;
    if (magnitude_bits == 0u || magnitude_bits > 0x7f800000u)
        return 0;
    const uint32_t sign = (original >> 24) & 0x80u;
    if (magnitude_bits >= 0x43e00000u)
        return static_cast<uint8_t>(sign | 0x7eu);
    uint32_t code;
    if (magnitude_bits < 0x3c800000u) {
        // E4 subnormal grid is 2^-9. Decode the f32 significand as an
        // integer, so host/device rounding modes cannot alter ties.
        const uint32_t exponent = magnitude_bits >> 23;
        if (exponent < 117u)
            code = 0; // strictly below half the smallest E4
        else {
            const uint32_t significand = (magnitude_bits & 0x7fffffu) | 0x800000u;
            code = numeric_detail::shiftRne(significand, 141u - exponent);
        }
    } else {
        const uint32_t rounded =
            (magnitude_bits + 0x7ffffu + ((magnitude_bits >> 20) & 1u)) & 0xfff00000u;
        code = (rounded >> 20) - 960u;
    }
    return static_cast<uint8_t>(sign | code);
}
NR_BROWSER_HD __half browserPublishE4(float value) {
    return e4m3ToHalf(browserE4Code(value));
}

// Captured table behavior: two FP32 fused multiply-adds, one FP32 multiply,
// and no half rounding between these operations. The caller supplies the
// half-bit table index, formed from its accumulator with browserHalfRtz.
NR_BROWSER_HD float browserSiluValue(__half input) {
    const float value = halfToFloat(input);
    const float bounded = value < -4.f ? -4.f : value > 4.f ? 4.f : value;
    const float absolute = bounded < 0.f ? -bounded : bounded;
    const float inner = browser_detail::madd(-0.055908203125f, absolute, 0.447265625f);
    const float polynomial = browser_detail::madd(bounded, inner, 0.89453125f);
    return browser_detail::multiply(value, polynomial);
}
NR_BROWSER_HD __half browserSiluHalf(__half input) {
    return browserHalfRtz(browserSiluValue(input));
}
NR_BROWSER_HD uint8_t browserSiluCode(__half input) {
    const float activated = browserSiluValue(input);
    // Observed input 0x8001 produces half -0 and packed +0. All other finite
    // inputs already agree with direct FP32 E4 publication. Express this as
    // the captured table's half-zero branch, not an input-specific exception.
    if ((halfBits(browserHalfRtz(activated)) & 0x7fffu) == 0u)
        return 0;
    return browserE4Code(activated);
}
NR_BROWSER_HD __half browserSiluE4(__half input) {
    return e4m3ToHalf(browserSiluCode(input));
}
} // namespace nr

#undef NR_BROWSER_HD
