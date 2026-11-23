// axis T: f32=float f16=uint16_t bf16=uint16_t
//
// Helpers of attention.slang:
//   MODE 0, the backward's stats: for each row (b, h, r) of out and
//     grad_out ([B, L, H, EV], n = B H L rows), (lse, delta = sum_d
//     grad_out out in float32) at stats[4 + 2 ((b H + h) L + r)], after a
//     header of the dropout's key, ctr and p (as bits) at stats[0..4).
//   MODE 1, the dropout's keep mask [B, H, L, S] (n elements, 1 kept, 0
//     dropped) as attention.slang draws it, for tests.

#include "_wide.slang"
#include "_philox.slang"

struct Params
{
    T* o;          // 0   out
    T* go;         // 8   grad_out
    float* lse;    // 16  [B, H, L]
    uint* out;     // 24  stats (MODE 0) or the mask (MODE 1)
    uint2 key;     // 32
    uint ctr;      // 40
    uint pbits;    // 44
    uint n;        // 48
    uint gstride;  // 52
    uint H;        // 56
    uint L;        // 60
    uint W;        // 64  EV (MODE 0) or S (MODE 1)
    uint pad;      // 68
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int MODE = 0;

// float16's bits widened exactly with integer ops (f16tof32 becomes a
// half conversion, which needs shaderFloat16).
float f16_bits_to_float(uint h)
{
    uint s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    if (e == 0u)
        return asfloat(s | asuint(float(m) * 5.9604644775390625e-8f));  // subnormal: m 2^-24
    if (e == 31u)
        return asfloat(s | 0x7f800000u | (m << 13));
    return asfloat(s | ((e + 112u) << 23) | (m << 13));
}

float ld(T v)
{
#if defined(T_bf16)
    return asfloat(uint(v) << 16);
#elif defined(T_f16)
    return f16_bits_to_float(uint(v));
#else
    return v;
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (MODE == 0 && id.x == 0)
    {
        p.out[0] = p.key.x;
        p.out[1] = p.key.y;
        p.out[2] = p.ctr;
        p.out[3] = p.pbits;
    }
    for (uint i = id.x; i < p.n; i += p.gstride)
    {
        if (MODE == 0)
        {
            uint r = i % p.L, bh = i / p.L;
            uint b = bh / p.H, h = bh % p.H;
            uint e = ((b * p.L + r) * p.H + h) * p.W;
            float s = 0.0f;
            for (uint d = 0; d < p.W; d++)
                s = fma(ld(p.go[e + d]), ld(p.o[e + d]), s);
            p.out[4 + 2 * i] = asuint(p.lse[i]);
            p.out[5 + 2 * i] = asuint(s);
        }
        else
        {
            uint c = i % p.W, rest = i / p.W;
            uint r = rest % p.L, bh = rest / p.L;
            uint w = philox(uint4(c / 4, r, bh, p.ctr), p.key)[c % 4];
            p.out[i] = float(w >> 8u) * 5.9604644775390625e-8f >= asfloat(p.pbits) ? 1u : 0u;
        }
    }
}
