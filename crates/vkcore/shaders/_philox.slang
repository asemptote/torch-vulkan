// Philox4x32-10 and the uniform / normal variates drawn from one block of its
// output, shared by rng.slang and sample.slang. The including file defines
// R (the precision distributions are computed in) and IS_FLOAT for float
// kernels, and includes _fmath.slang and _wide.slang.

#pragma once

uint4 philox(uint4 c, uint2 k)
{
    for (int r = 0; r < 10; r++)
    {
        uint2 p0 = mul_wide32(0xD2511F53u, c.x);
        uint2 p1 = mul_wide32(0xCD9E8D57u, c.z);
        c = uint4(p1.y ^ c.y ^ k.x, p1.x, p0.y ^ c.w ^ k.y, p0.x);
        k += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

#ifdef IS_FLOAT
// Uniform in [0, 1) from the bits: 24 bits for float, 53 for double.
R uniform(uint4 x, int which)
{
#if defined(PARAM_DOUBLE)
    uint hi = which == 0 ? x.x : x.z, lo = which == 0 ? x.y : x.w;
    // ((hi >> 5) 2^26 + (lo >> 6)) 2^-53, both parts and their sum exact in a double
    return (R(hi >> 5) * 67108864.0l + R(lo >> 6)) * 1.1102230246251565e-16l;
#else
    return R((which == 0 ? x.x : x.z) >> 8) * 5.9604644775390625e-8f;  // 2^-24
#endif
}

R standard_normal(uint4 x)
{
    R u1 = R(1) - uniform(x, 0);  // (0, 1]
    R u2 = uniform(x, 1);
    return sqrt(R(-2) * m_log(u1)) * m_cos(LIT(6.283185307179586) * u2);
}
#endif
