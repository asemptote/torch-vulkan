// The 16-bit types of the matrix products (gemmcm.slang, splitk.slang): T is
// the storage type (bfloat16 is stored as uint16_t). Values are read into
// float32 and written rounded to nearest even, as cast.slang does (bfloat16
// NaN as 0x7fc0).

#pragma once

#include "_half.slang"

#if defined(T_bf16)
uint bits_of(T x) { return uint(x); }
float to_f(T x) { return bit_cast<float>(uint(x) << 16); }
T from_f(float f)
{
    if (isnan(f))
        return uint16_t(0x7fc0);
    uint u = bit_cast<uint>(f);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return uint16_t(u >> 16);
}
#else
uint bits_of(T x) { return uint(bit_cast<uint16_t>(x)); }
float to_f(T x) { return float(x); }
T from_f(float f) { return to_half(f); }
#endif
