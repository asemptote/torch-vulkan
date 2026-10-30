// axis T: f32=float f64=double i32=int i64=int64_t
//
// Random fills of a contiguous tensor with Philox4x32-10: element i draws
// its bits from counter (i, offset) under key `seed`, so a (seed, offset)
// pair gives the same numbers however the work is split across threads, and
// the host advances `offset` once per call. OP (parameters a, b as raw
// 64-bit slots: PARAM_T values, or int64 for RNG_RANDOM):
//   RNG_UNIFORM      a + (b - a) u,  u in [0, 1)
//   RNG_NORMAL       a + b z,        z standard normal (Box-Muller)
//   RNG_RANDOM       a + bits mod b  (integers; b == 0 means all 64 bits)
//   RNG_EXPONENTIAL  -log(1 - u) / a
//   RNG_GEOMETRIC    ceil(log(1 - u) / log1p(-a))
//   RNG_CAUCHY       a + b tan(pi (u - 1/2))
//   RNG_LOG_NORMAL   exp(a + b z)
// Integer kernels do RNG_RANDOM only (the host computes the others in float
// and casts).

#include "_types.slang"
#include "opcodes.slang"
#if defined(T_f32) || defined(T_f64)
#define IS_FLOAT 1
#include "_fmath.slang"
#endif

struct Params
{
    T* out;
    uint64_t seed;
    uint64_t offset;
    uint64_t a_bits;
    uint64_t b_bits;
    uint n;
    uint stride;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = RNG_UNIFORM;

uint4 philox(uint4 c, uint2 k)
{
    for (int r = 0; r < 10; r++)
    {
        uint64_t p0 = uint64_t(0xD2511F53u) * uint64_t(c.x);
        uint64_t p1 = uint64_t(0xCD9E8D57u) * uint64_t(c.z);
        c = uint4(uint(p1 >> 32) ^ c.y ^ k.x, uint(p1), uint(p0 >> 32) ^ c.w ^ k.y, uint(p0));
        k += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

#ifdef IS_FLOAT
typedef T R;  // the precision distributions are computed in
#if defined(PARAM_DOUBLE)
R param(uint64_t bits) { return bit_cast<double>(bits); }
#else
R param(uint64_t bits) { return bit_cast<float>(uint(bits)); }
#endif

// Uniform in [0, 1) from the bits: 24 bits for float, 53 for double.
R uniform(uint4 x, int which)
{
#if defined(PARAM_DOUBLE)
    uint hi = which == 0 ? x.x : x.z, lo = which == 0 ? x.y : x.w;
    return R((uint64_t(hi >> 5) << 26) | uint64_t(lo >> 6)) * 1.1102230246251565e-16l;  // 2^-53
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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint2 key = uint2(uint(p.seed), uint(p.seed >> 32));
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        uint4 x = philox(uint4(i, 0u, uint(p.offset), uint(p.offset >> 32)), key);
        if (OP == RNG_RANDOM)
        {
            uint64_t range = p.b_bits;
            uint64_t bits = range != 0ull && range <= 0x100000000ull ? uint64_t(x.x) % range
                                                                     : ((uint64_t(x.x) << 32) | uint64_t(x.y));
            if (range > 0x100000000ull)
                bits %= range;
            p.out[i] = T(int64_t(p.a_bits) + int64_t(bits));
            continue;
        }
#ifdef IS_FLOAT
        R a = param(p.a_bits), b = param(p.b_bits), u = uniform(x, 0), v;
        switch (OP)
        {
        case RNG_UNIFORM: v = a + (b - a) * u; break;
        case RNG_NORMAL: v = a + b * standard_normal(x); break;
        case RNG_EXPONENTIAL: v = -m_log1p(-u) / a; break;
        case RNG_GEOMETRIC: v = ceil(m_log1p(-u) / m_log1p(-a)); break;
        case RNG_CAUCHY: v = a + b * m_tan(LIT(3.141592653589793) * (u - R(0.5))); break;
        default: v = m_exp(a + b * standard_normal(x)); break;  // RNG_LOG_NORMAL
        }
        p.out[i] = v;
#endif
    }
}
