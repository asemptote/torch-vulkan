// float -> float16 with round-to-nearest-even, done on the bits: Vulkan
// leaves the rounding of OpFConvert to half up to the driver, and PyTorch
// rounds to nearest even.

#pragma once

uint16_t f16_bits_rne(float f)
{
    uint x = bit_cast<uint>(f);
    uint sign = (x >> 16) & 0x8000u;
    uint ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u)
        return uint16_t(sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));  // NaN, inf
    if (ax >= 0x477ff000u)
        return uint16_t(sign | 0x7c00u);  // >= 65520 rounds to inf
    uint e = ax >> 23;
    uint q, rem, halfway;
    if (ax >= 0x38800000u)
    {  // normal half
        uint mant = ax & 0x7fffffu;
        q = ((e - 112u) << 10) | (mant >> 13);
        rem = mant & 0x1fffu;
        halfway = 0x1000u;
    }
    else
    {  // subnormal half (or zero): units of 2^-24
        if (e < 102u)
            return uint16_t(sign);
        uint mant = (ax & 0x7fffffu) | 0x800000u;
        uint s = 126u - e;
        q = mant >> s;
        rem = mant & ((1u << s) - 1u);
        halfway = 1u << (s - 1u);
    }
    if (rem > halfway || (rem == halfway && (q & 1u) != 0u))
        q += 1u;  // a carry into the exponent is still correct
    return uint16_t(sign | q);
}

half to_half(float f)
{
    return bit_cast<half>(f16_bits_rne(f));
}
