// Numeric publication rules derived from OpenDLSS-NR, MIT, Copyright (c) 2026 maan.
// These scalar operations do not emulate Ada's FP8 matrix accumulation.
#pragma once
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#define NR_NUM_HD __host__ __device__ __forceinline__

namespace nr {

NR_NUM_HD uint16_t halfBits(__half value) { return __half_as_ushort(value); }
NR_NUM_HD __half halfFromBits(uint16_t bits) { return __ushort_as_half(bits); }
NR_NUM_HD __half halfRne(float value) { return __float2half_rn(value); }
NR_NUM_HD float halfToFloat(__half value) { return __half2float(value); }
NR_NUM_HD float roundF16(float value) { return halfToFloat(halfRne(value)); }

namespace numeric_detail {
// Host-only fallback for true half arithmetic. Rounding a double through float can
// double-round an FMA at a half midpoint, so convert directly from the double bits.
inline __half halfFromDoubleRne(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 48) & 0x8000);
  const unsigned exponent = static_cast<unsigned>((bits >> 52) & 0x7ff);
  const uint64_t fraction = bits & 0x000fffffffffffffull;
  if (exponent == 0x7ff) return halfFromBits(sign | (fraction ? 0x7e00 : 0x7c00));
  if (exponent == 0) return halfFromBits(sign);
  int he = static_cast<int>(exponent) - 1008;
  if (he >= 31) return halfFromBits(sign | 0x7c00);
  if (he < -10) return halfFromBits(sign);
  const uint64_t significand = fraction | (uint64_t{1} << 52);
  const unsigned shift = he <= 0 ? static_cast<unsigned>(43 - he) : 42u;
  uint64_t rounded = significand >> shift;
  const uint64_t remainder = significand & ((uint64_t{1} << shift) - 1);
  const uint64_t midpoint = uint64_t{1} << (shift - 1);
  if (remainder > midpoint || (remainder == midpoint && (rounded & 1))) ++rounded;
  if (he <= 0) return halfFromBits(sign | static_cast<uint16_t>(rounded));
  if (rounded == 2048) { rounded = 1024; ++he; }
  if (he >= 31) return halfFromBits(sign | 0x7c00);
  return halfFromBits(sign | static_cast<uint16_t>((he << 10) | (rounded - 1024)));
}

NR_NUM_HD uint32_t shiftRne(uint32_t value, unsigned shift) {
  const uint32_t quotient = value >> shift;
  const uint32_t remainder = value & ((1u << shift) - 1u);
  const uint32_t midpoint = 1u << (shift - 1u);
  return quotient + ((remainder > midpoint || (remainder == midpoint && (quotient & 1u))) ? 1u : 0u);
}

NR_NUM_HD __half clampHalf(__half value, uint16_t lo, uint16_t hi) {
  const float x = halfToFloat(value);
  if (x < halfToFloat(halfFromBits(lo))) return halfFromBits(lo);
  if (x > halfToFloat(halfFromBits(hi))) return halfFromBits(hi);
  return value;
}
}  // namespace numeric_detail

// The *_rn intrinsics prevent a separately rounded multiply and add from being
// contracted. Use fmaHalf only where the network explicitly specifies an FMA.
NR_NUM_HD __half addHalf(__half a, __half b) {
#if defined(__CUDA_ARCH__)
  return __hadd_rn(a, b);
#else
  return numeric_detail::halfFromDoubleRne(static_cast<double>(halfToFloat(a)) + halfToFloat(b));
#endif
}
NR_NUM_HD __half subHalf(__half a, __half b) {
#if defined(__CUDA_ARCH__)
  return __hsub_rn(a, b);
#else
  return numeric_detail::halfFromDoubleRne(static_cast<double>(halfToFloat(a)) - halfToFloat(b));
#endif
}
NR_NUM_HD __half mulHalf(__half a, __half b) {
#if defined(__CUDA_ARCH__)
  return __hmul_rn(a, b);
#else
  return numeric_detail::halfFromDoubleRne(static_cast<double>(halfToFloat(a)) * halfToFloat(b));
#endif
}
NR_NUM_HD __half fmaHalf(__half a, __half b, __half c) {
#if defined(__CUDA_ARCH__)
  return __hfma(a, b, c);
#else
  return numeric_detail::halfFromDoubleRne(std::fma(static_cast<double>(halfToFloat(a)),
                                                 static_cast<double>(halfToFloat(b)),
                                                 static_cast<double>(halfToFloat(c))));
#endif
}

// Every finite E4M3 value is exactly representable as binary16. Match the shader
// policy at 0x7f/0xff: decode to +0/-0, unlike the upstream host NaN decoder.
NR_NUM_HD __half e4m3ToHalf(uint8_t code) {
  const uint16_t sign = static_cast<uint16_t>(code & 0x80u) << 8;
  const unsigned exponent = (code >> 3) & 15u;
  const unsigned mantissa = code & 7u;
  if (exponent == 15u && mantissa == 7u) return halfFromBits(sign);
  if (exponent != 0u)
    return halfFromBits(sign | static_cast<uint16_t>(((exponent + 8u) << 10) | (mantissa << 7)));
  if (mantissa >= 4u) return halfFromBits(sign | static_cast<uint16_t>(0x2000u | ((mantissa - 4u) << 8)));
  if (mantissa >= 2u) return halfFromBits(sign | static_cast<uint16_t>(0x1c00u | ((mantissa - 2u) << 9)));
  return halfFromBits(sign | (mantissa ? 0x1800u : 0u));
}
NR_NUM_HD float e4m3ToFloat(uint8_t code) { return halfToFloat(e4m3ToHalf(code)); }

// RNE, finite saturation at +/-448, NaN -> +0, and preservation of signed zero.
NR_NUM_HD uint8_t e4m3FromHalfBits(uint16_t bits) {
  const uint32_t exponent = (bits >> 10) & 31u;
  const uint32_t mantissa = bits & 1023u;
  if (exponent == 31u && mantissa != 0u) return 0;
  const uint32_t sign = (bits >> 8) & 0x80u;
  if (exponent == 31u) return static_cast<uint8_t>(sign | 0x7eu);
  uint32_t code;
  if (exponent <= 8u) {
    const uint32_t significand = exponent ? 1024u + mantissa : mantissa;
    code = numeric_detail::shiftRne(significand, exponent ? 16u - exponent : 15u);
    if (code > 8u) code = 8u;
  } else {
    uint32_t e = exponent - 8u;
    uint32_t m = numeric_detail::shiftRne(mantissa, 7u);
    if (m == 8u) { m = 0; ++e; }
    code = (e > 15u || (e == 15u && m > 6u)) ? 0x7eu : ((e << 3) | m);
  }
  return static_cast<uint8_t>(sign | code);
}
NR_NUM_HD uint8_t e4m3FromHalf(__half value) { return e4m3FromHalfBits(halfBits(value)); }
// Do not replace this two-stage conversion with a direct float -> E4M3 rounding.
NR_NUM_HD uint8_t e4m3FromFloat(float value) { return e4m3FromHalf(halfRne(value)); }
NR_NUM_HD __half publishE4(__half value) { return e4m3ToHalf(e4m3FromHalf(value)); }
NR_NUM_HD __half publishE4(float value) { return e4m3ToHalf(e4m3FromFloat(value)); }

NR_NUM_HD __half siluHalf(__half value) {
  const __half bounded = numeric_detail::clampHalf(value, 0xc400, 0x4400);
  const __half absolute = halfFromBits(halfBits(bounded) & 0x7fffu);
  const __half inner = fmaHalf(halfFromBits(0xab28), absolute, halfFromBits(0x3728));
  const __half polynomial = fmaHalf(bounded, inner, halfFromBits(0x3b28));
  return mulHalf(value, polynomial);
}

// Window affine uses a FLOAT FMA followed by half publication.
NR_NUM_HD __half windowExpWeight(__half score) {
#if defined(__CUDA_ARCH__)
  const float affine32 = __fmaf_rn(halfToFloat(score), 0.044921875f, 1.30078125f);
#else
  const float affine32 = std::fma(halfToFloat(score), 0.044921875f, 1.30078125f);
#endif
  const __half affine = numeric_detail::clampHalf(halfRne(affine32), 0x3c20, 0x3e47);
  return halfFromBits(static_cast<uint16_t>((static_cast<uint32_t>(halfBits(affine)) << 5) ^ 0x8000u));
}

// ViT affine uses a HALF FMA with half-rounded constants (0x2dbb, 0x3ed6).
NR_NUM_HD __half vitExpWeight(__half score) {
  const __half affine = numeric_detail::clampHalf(
      fmaHalf(score, halfFromBits(0x2dbb), halfFromBits(0x3ed6)), 0x3dc2, 0x3fe9);
  return halfFromBits(static_cast<uint16_t>((static_cast<uint32_t>(halfBits(affine)) << 4) + 0x4000u));
}

}  // namespace nr
#undef NR_NUM_HD
