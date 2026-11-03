// axis T: f32=float f64=double i32=int i64=I64
//
// Random fills of a contiguous tensor with Philox4x32-10: element i draws
// its bits from counter (i, offset) under key `seed`, so a (seed, offset)
// pair gives the same numbers however the work is split across threads, and
// the host advances `offset` once per call. OP (parameters a, b as raw
// 64-bit slots, _slot.slang: PARAM_T values, or int64 for RNG_RANDOM):
//   RNG_UNIFORM      a + (b - a) u,  u in [0, 1)
//   RNG_NORMAL       a + b z,        z standard normal (Box-Muller)
//   RNG_RANDOM       a + bits mod b  (integers; b == 0 means all 64 bits)
//   RNG_EXPONENTIAL  -log(1 - u) / a
//   RNG_GEOMETRIC    ceil(log(1 - u) / log1p(-a))
//   RNG_CAUCHY       a + b tan(pi (u - 1/2))
//   RNG_LOG_NORMAL   exp(a + b z)
// Integer kernels do RNG_RANDOM only (the host computes the others in float
// and casts).

#include "_int64.slang"  // (the i64 variant's T: _int64.slang)
#include "_types.slang"
#include "_wide.slang"
#include "opcodes.slang"
#if defined(T_f32) || defined(T_f64)
#define IS_FLOAT 1
#include "_fmath.slang"
#endif

struct Params
{
    T* out;
    slot64 seed;
    slot64 offset;
    slot64 a_bits;
    slot64 b_bits;
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
        uint2 p0 = mul_wide32(0xD2511F53u, c.x);
        uint2 p1 = mul_wide32(0xCD9E8D57u, c.z);
        c = uint4(p1.y ^ c.y ^ k.x, p1.x, p0.y ^ c.w ^ k.y, p0.x);
        k += uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

#ifdef IS_FLOAT
typedef T R;  // the precision distributions are computed in
#if defined(PARAM_DOUBLE)
R param(slot64 bits) { return slot_double(bits); }
#else
R param(slot64 bits) { return slot_float(bits); }
#endif

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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint2 key = p.seed;
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        uint4 x = philox(uint4(i, 0u, p.offset.x, p.offset.y), key);
        if (OP == RNG_RANDOM)
        {
            // As the CPU: 64 random bits from a range of 2^28 on (32 bits
            // modulo a larger range skew it), range 0 being the full 2^64.
            // (In 32-bit words: _wide.slang.)
            uint2 range = p.b_bits;
            bool small = range.y == 0u && range.x != 0u && range.x < 0x10000000u;
            uint2 bits = small ? uint2(x.x % range.x, 0u) : uint2(x.y, x.x);
            if (!small && (range.x | range.y) != 0u)
                bits = umod_wide(bits, range);
            uint2 v = add_wide(p.a_bits, bits);
#if defined(T_i64)
            p.out[i] = I64(v.x, v.y);
#elif defined(T_f64)
            p.out[i] = T(asint(v.y)) * 4294967296.0l + T(v.x);  // (exact parts, one rounding)
#elif defined(IS_FLOAT)
            p.out[i] = T(int64_words_to_float(v.x, v.y));
#else
            p.out[i] = T(int(v.x));
#endif
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
