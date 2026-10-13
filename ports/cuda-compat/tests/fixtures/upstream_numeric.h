// CPU-side E4M3FN / IEEE binary16 helpers. These mirror the shaders' fp8 / half publication functions so
// host-side preprocessing rounds exactly like the kernels.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace num {

inline uint32_t f32Bits(float value) { uint32_t bits; memcpy(&bits, &value, 4); return bits; }
inline float f32FromBits(uint32_t bits) { float value; memcpy(&value, &bits, 4); return value; }

// IEEE RNE f32 -> f16 (the shaders' roundF16).
inline uint16_t f16Bits(float value) {
  uint32_t bits = f32Bits(value);
  uint32_t sign = (bits >> 16) & 0x8000;
  uint32_t exponent = (bits >> 23) & 0xff;
  uint32_t mantissa = bits & 0x7fffff;
  if (exponent == 0xff) return (uint16_t)(sign | (mantissa ? 0x7e00 : 0x7c00));
  int halfExponent = (int)exponent - 112;
  if (halfExponent >= 31) return (uint16_t)(sign | 0x7c00);
  if (halfExponent <= 0) {
    if (halfExponent < -10) return (uint16_t)sign;
    uint32_t normalized = mantissa | 0x800000;
    uint32_t shift = (uint32_t)(14 - halfExponent);
    uint32_t rounded = normalized >> shift;
    uint32_t remainder = normalized & ((1u << shift) - 1);
    uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1))) rounded += 1;
    return (uint16_t)(sign | rounded);
  }
  uint32_t rounded = mantissa >> 13;
  uint32_t remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder == 0x1000 && (rounded & 1))) {
    rounded += 1;
    if (rounded == 0x400) { rounded = 0; halfExponent += 1; }
  }
  if (halfExponent >= 31) return (uint16_t)(sign | 0x7c00);
  return (uint16_t)(sign | ((uint32_t)halfExponent << 10) | rounded);
}

inline float f16ToF32(uint16_t bits) {
  uint32_t sign = (uint32_t)(bits & 0x8000) << 16;
  uint32_t exponent = (bits >> 10) & 0x1f;
  uint32_t mantissa = bits & 0x3ff;
  if (exponent == 0) {
    if (mantissa == 0) return f32FromBits(sign);
    // subnormal
    int shift = 0;
    while (!(mantissa & 0x400)) { mantissa <<= 1; shift += 1; }
    mantissa &= 0x3ff;
    return f32FromBits(sign | ((uint32_t)(113 - shift) << 23) | (mantissa << 13));
  }
  if (exponent == 31) return f32FromBits(sign | 0x7f800000 | (mantissa << 13));
  return f32FromBits(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

inline float roundF16(float value) { return f16ToF32(f16Bits(value)); }

// e4m3ToNumber: E4M3FN byte to float (NaN code 0x7f/0xff decodes to NaN).
inline float e4m3ToF32(uint8_t byte) {
  int sign = (byte & 0x80) ? -1 : 1;
  int exponent = (byte >> 3) & 0xf;
  int mantissa = byte & 0x7;
  if (exponent == 0) return sign * (float)mantissa * (1.0f / 512.0f);
  if (exponent == 0xf && mantissa == 0x7) return NAN;
  return sign * (1.0f + mantissa / 8.0f) * std::ldexp(1.0f, exponent - 7);
}

// Already-rounded half -> E4M3FN code with RNE, finite saturation at 448 and NaN -> +0; the sign of a zero
// survives, as in the shaders' publication (common.glsl e4m3CodeFromF16Bits).
inline uint8_t e4m3FromF16Bits(uint16_t half) {
  bool isNaN = (half & 0x7c00) == 0x7c00 && (half & 0x03ff) != 0;
  if (isNaN) return 0;
  if ((half & 0x7fff) == 0) return (uint8_t)((half >> 8) & 0x80);
  uint32_t negative = (half & 0x8000) ? 0x80 : 0;
  uint32_t exponent = (half >> 10) & 0x1f;
  uint32_t mantissa = half & 0x3ff;
  auto shiftRne = [](uint32_t value, uint32_t shift) {
    uint32_t quotient = value >> shift;
    uint32_t remainder = value & ((1u << shift) - 1);
    uint32_t halfway = 1u << (shift - 1);
    return quotient + ((remainder > halfway || (remainder == halfway && (quotient & 1))) ? 1u : 0u);
  };
  uint32_t code;
  if (exponent == 31) {
    code = 0x7e;
  } else if (exponent <= 8) {
    uint32_t significand = exponent == 0 ? mantissa : 1024 + mantissa;
    uint32_t shift = exponent == 0 ? 15 : 16 - exponent;
    code = shiftRne(significand, shift);
    if (code > 8) code = 8;
  } else {
    uint32_t e4Exponent = exponent - 8;
    uint32_t e4Mantissa = shiftRne(mantissa, 7);
    if (e4Mantissa == 8) { e4Mantissa = 0; e4Exponent += 1; }
    code = (e4Exponent > 15 || (e4Exponent == 15 && e4Mantissa > 6)) ? 0x7e : ((e4Exponent << 3) | e4Mantissa);
  }
  return (uint8_t)(negative | code);
}

inline uint8_t e4m3FromF32(float value) { return e4m3FromF16Bits(f16Bits(value)); }

}  // namespace num
