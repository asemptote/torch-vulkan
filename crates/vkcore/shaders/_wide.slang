// 64-bit integer arithmetic in 32-bit words: a value is uint2 (lo, hi), two's
// complement where signed. For kernels that run on GPUs without 64-bit
// integers (the Int64 capability: Mali GPUs lack it).

#pragma once

// a * b as a 64-bit value, from 16-bit halves.
uint2 mul_wide32(uint a, uint b)
{
    uint a0 = a & 0xffffu, a1 = a >> 16, b0 = b & 0xffffu, b1 = b >> 16;
    uint p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return uint2((p00 & 0xffffu) | (mid << 16), p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16));
}

// v + c, mod 2^64.
uint2 add_wide32(uint2 v, uint c)
{
    uint lo = v.x + c;
    return uint2(lo, v.y + (lo < c ? 1u : 0u));
}

// a + b, mod 2^64.
uint2 add_wide(uint2 a, uint2 b)
{
    uint lo = a.x + b.x;
    return uint2(lo, a.y + b.y + (lo < a.x ? 1u : 0u));
}

// a - b, mod 2^64.
uint2 sub_wide(uint2 a, uint2 b)
{
    return uint2(a.x - b.x, a.y - b.y - (a.x < b.x ? 1u : 0u));
}

// a < b, unsigned.
bool ult_wide(uint2 a, uint2 b)
{
    return a.y < b.y || (a.y == b.y && a.x < b.x);
}

// a mod b, unsigned, b != 0 (shift-subtract, one bit a step).
uint2 umod_wide(uint2 a, uint2 b)
{
    if (b.y == 0u && a.y == 0u)
        return uint2(a.x % b.x, 0u);
    uint2 r = uint2(0u, 0u);
    for (int i = 63; i >= 0; i--)
    {
        r = uint2(r.x << 1, (r.y << 1) | (r.x >> 31));
        uint bit = i >= 32 ? (a.y >> uint(i - 32)) & 1u : (a.x >> uint(i)) & 1u;
        r.x |= bit;
        if (!ult_wide(r, b))
            r = sub_wide(r, b);
    }
    return r;
}

// An element of an int64 tensor (indices, offsets, targets) as two words.
// Kernels use these values as 32-bit element offsets (vkcore's limit), so
// they read the low word, checking the high one where the value comes from
// the user, and write the sign-extended int.
typedef uint2 i64w;
i64w i64w_from(int v) { return uint2(asuint(v), v < 0 ? 0xffffffffu : 0u); }
int i64w_int(i64w v) { return asint(v.x); }
bool i64w_fits(i64w v) { return v.y == (asint(v.x) < 0 ? 0xffffffffu : 0u); }  // in int32's range

// The pointer whose 64-bit address is w (lo, hi): an OpBitcast from the two
// words, which slang's casts would route through a 64-bit integer.
uint* words_to_uint_ptr(uint2 w)
{
    return spirv_asm { result:$$uint* = OpBitcast $w };
}

// float(v), rounded to nearest even, for the signed 64-bit v: the top 32 bits
// of |v| with a sticky bit for the rest, so float's one rounding is the
// 64-bit conversion's.
float int64_words_to_float(uint lo, uint hi)
{
    bool neg = (hi >> 31) != 0u;
    if (neg)
    {
        lo = ~lo + 1u;
        hi = ~hi + (lo == 0u ? 1u : 0u);
    }
    float f;
    if (hi == 0u)
    {
        f = float(lo);
    }
    else
    {
        uint lz = 31u - uint(firstbithigh(hi));
        uint top = lz == 0u ? hi : (hi << lz) | (lo >> (32u - lz));
        uint rest = lz == 0u ? lo : lo << lz;
        f = ldexp(float(top | (rest != 0u ? 1u : 0u)), float(32u - lz));
    }
    return neg ? -f : f;
}
