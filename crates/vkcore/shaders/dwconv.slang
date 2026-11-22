// axis T: f32=float f64=double i32=int i64=I64 f16=half bf16=uint16_t
//
// Depthwise convolution (one input channel per group, M output channels per
// group), 1 to 3 spatial dims as D, H, W (size-1 dims for fewer), directly.
//   MODE 0, forward: one invocation per output,
//     y[n, c M + m, od, oh, ow] = sum_t w[c M + m, t] x[n, c, i(o, t)] (+ bias)
//   for the taps t = (kd, kh, kw) in order, the bias added last.
//   MODE 1, the weight's gradient: one workgroup per (tap, output channel),
//     gw[c M + m, t] = sum_{n, o} gy[n, c M + m, o] x[n, c, i(o, t)],
//   each lane summing its share of the positions (n outer, o inner, 64
//   apart), then a tree over the lanes.
// i(o, t) = (od sd - pd + kd dd, oh sh - ph + kh dh, ow sw - pw + kw dw).
// Taps outside x are skipped, as in the CPU's (oneDNN's) depthwise
// convolution: an infinite or NaN weight does not reach the outputs whose
// window meets it only in the padding. Negative padding crops. Sums in T,
// fused multiply-adds for floats; float16 and bfloat16 sum in float32 and
// round once (to nearest even). Integers wrap, as the CPU's.

#include "_half.slang"
#include "_int64.slang"

struct Params
{
    T* x;          // 0   [N, C, Di, Hi, Wi] contiguous
    T* w;          // 8   [C M, KD, KH, KW] contiguous (MODE 1: written)
    T* y;          // 16  [N, C M, Do, Ho, Wo] contiguous (MODE 1: gy, read)
    T* bias;       // 24  [C M] (USE_BIAS)
    uint total;    // 32  outputs (MODE 1: N)
    uint gstride;  // 36  grid stride
    uint cm;       // 40  output channels (C M)
    uint m;        // 44  output channels per input channel
    uint4 osz;     // 48  Do, Ho, Wo, KD
    uint4 isz;     // 64  Di, Hi, Wi, KH
    uint4 st;      // 80  strides (d, h, w), KW
    uint4 dl;      // 96  dilations (d, h, w), -
    int4 pd;       // 112 padding before (d, h, w; negative crops), -
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int USE_BIAS = 0;
[vk::constant_id(1)] const int MODE = 0;

#define WG 64

#if defined(T_f16) || defined(T_bf16)
typedef float A;
#else
typedef T A;
#endif

A ld(T v)
{
#if defined(T_bf16)
    return asfloat(uint(v) << 16);
#elif defined(T_f16)
    return float(v);
#else
    return v;
#endif
}

T st(A f)
{
#if defined(T_f16)
    return to_half(f);
#elif defined(T_bf16)
    if (isnan(f))
        return uint16_t(0x7fc0);
    uint u = asuint(f);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return uint16_t(u >> 16);
#else
    return f;
#endif
}

A mac(A acc, A a, A b)
{
#if defined(T_i32) || defined(T_i64)
    return acc + a * b;
#else
    return fma(a, b, acc);
#endif
}

groupshared A part[WG];

// MODE 1: gw[oc, t] for the workgroup's tap t (x) and output channel oc (y).
void weight_grad(uint lane, uint t, uint oc)
{
    uint Do = p.osz.x, Ho = p.osz.y, Wo = p.osz.z;
    uint Di = p.isz.x, Hi = p.isz.y, Wi = p.isz.z;
    uint KH = p.isz.w, KW = p.st.w;
    uint L = Do * Ho * Wo, plane = Di * Hi * Wi, C = p.cm / p.m;
    int kw = int(t % KW), kh = int((t / KW) % KH), kd = int(t / (KW * KH));
    int ed = kd * int(p.dl.x) - p.pd.x, eh = kh * int(p.dl.y) - p.pd.y, ew = kw * int(p.dl.z) - p.pd.z;
    A acc = A(0);
    for (uint n = 0; n < p.total; n++)
    {
        uint xbase = (n * C + oc / p.m) * plane, gbase = (n * p.cm + oc) * L;
        for (uint o = lane; o < L; o += WG)
        {
            uint r = o;
            int xw = int(r % Wo) * int(p.st.z) + ew;
            r /= Wo;
            int xh = int(r % Ho) * int(p.st.y) + eh;
            int xd = int(r / Ho) * int(p.st.x) + ed;
            if (xd >= 0 && xd < int(Di) && xh >= 0 && xh < int(Hi) && xw >= 0 && xw < int(Wi))
                acc = mac(acc, ld(p.y[gbase + o]), ld(p.x[xbase + uint((xd * int(Hi) + xh) * int(Wi) + xw)]));
        }
    }
    part[lane] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint h = WG / 2; h > 0; h /= 2)
    {
        if (lane < h)
            part[lane] = part[lane] + part[lane + h];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        p.w[oc * (p.osz.w * KH * KW) + t] = st(part[0]);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    if (MODE == 1)
    {
        weight_grad(lid.x, gid.x, gid.y);
        return;
    }
    uint Do = p.osz.x, Ho = p.osz.y, Wo = p.osz.z;
    uint Di = p.isz.x, Hi = p.isz.y, Wi = p.isz.z;
    uint KD = p.osz.w, KH = p.isz.w, KW = p.st.w;
    uint plane = Di * Hi * Wi, taps = KD * KH * KW;
    for (uint i = id.x; i < p.total; i += p.gstride)
    {
        uint r = i;
        int ow = int(r % Wo);
        r /= Wo;
        int oh = int(r % Ho);
        r /= Ho;
        int od = int(r % Do);
        r /= Do;
        uint oc = r % p.cm;
        uint n = r / p.cm;
        uint xbase = (n * (p.cm / p.m) + oc / p.m) * plane;
        uint wbase = oc * taps;
        int d0 = od * int(p.st.x) - p.pd.x, h0 = oh * int(p.st.y) - p.pd.y, w0 = ow * int(p.st.z) - p.pd.z;
        A acc = A(0);
        uint t = 0;
        for (uint kd = 0; kd < KD; kd++)
        {
            int xd = d0 + int(kd * p.dl.x);
            bool okd = xd >= 0 && xd < int(Di);
            for (uint kh = 0; kh < KH; kh++)
            {
                int xh = h0 + int(kh * p.dl.y);
                bool okh = okd && xh >= 0 && xh < int(Hi);
                uint row = xbase + uint((xd * int(Hi) + xh) * int(Wi));
                for (uint kw = 0; kw < KW; kw++)
                {
                    int xw = w0 + int(kw * p.dl.z);
                    if (okh && xw >= 0 && xw < int(Wi))
                        acc = mac(acc, ld(p.x[row + uint(xw)]), ld(p.w[wbase + t]));
                    t++;
                }
            }
        }
        if (USE_BIAS != 0)
            acc = acc + ld(p.bias[oc]);
        p.y[i] = st(acc);
    }
}
