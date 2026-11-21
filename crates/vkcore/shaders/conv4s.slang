// axis T: f32=float f16=half bf16=uint16_t f16h=half
// axis TILE: r8n1=int r4n2=int r6n2=int r8n2=int r4n4=int r12n1=int r8n1w1=int r8n1w4=int
//
// Direct convolution (stride 1, 1 to 3 spatial dims) for GPUs with 16-lane
// subgroups, on gemm4s.slang's scheme (A shared across a subgroup by
// shuffles), reading the input in place instead of an im2col copy:
//
//   y[n, g Cog + m, o] = sum_{c, t} W[g, m, c, t] x[n, g Cg + c, o + off(t)]
//
// over the zero-padded input x, whose spatial dims are flattened: an output
// position o is a "virtual column" v = (od Hp + oh) Wp + ow of the padded
// grid (Dp x Hp x Wp), and tap t = (kd, kh, kw) reads x at v + kd dd Hp Wp +
// kh dh Wp + kw dw. Columns with ow >= Wo or oh >= Ho (wrapping into the
// next row or plane) are computed and not stored.
//
// The host passes the padded input with Wp (2-D, 3-D) and each channel's
// plane a multiple of 4 values, 16-byte aligned (8 for 16-bit T), with
// slack after the last plane for the columns and taps past the end; the
// weight as A [G, Cog, Cg KO, KW4]: rows of the KO = KD KH outer taps'
// KW inner taps, zero-padded to KW4 (a multiple of 4).
//
// A subgroup computes R rows x 16 NC 4 columns: lane l the columns
// 4 NC l .. 4 NC l + 4 NC - 1. Per channel and outer tap, each lane loads
// its pieces of A (R rows x KW4 / 4 pieces over the 16 lanes) and the input
// around its columns once: tap kw's values are 4-value windows at kw dw past
// them, made from the 16-byte loads of the row (shifts known once KW and dw
// are specialised, so a load serves every tap it overlaps). Sums are float32
// with fused multiply-adds, rounded once (to nearest even) with the bias
// added, as gemm4s.slang's.
//
// f16h: float16 arithmetic (torch_vulkan.set_fp16_arithmetic): float16
// storage, products and sums in float16 (packed half2 multiply-adds), so a
// lane holds twice the columns in the same registers and a shuffle carries
// two taps' weights; the bias is added in float32 and the sum rounded once
// more. Sums of many products lose accuracy and overflow past 65504.
//
// Needs exactly 16 lanes per subgroup (vkcore: EXACT_SUBGROUP_KERNELS).

#include "_half.slang"
#include "_slot.slang"

#define WX 16  // lanes: a subgroup
// TILE: rows per subgroup (R), 4-value column groups per lane (NC),
// subgroups per workgroup (WY)
#if defined(TILE_r4n2)
#define R 4
#define NC 2
#elif defined(TILE_r6n2)
#define R 6
#define NC 2
#elif defined(TILE_r8n2)
#define R 8
#define NC 2
#elif defined(TILE_r4n4)
#define R 4
#define NC 4
#elif defined(TILE_r12n1)
#define R 12
#define NC 1
#else
#define R 8
#define NC 1
#endif
#if defined(TILE_r8n1w1)
#define WY 1
#elif defined(TILE_r8n1w4)
#define WY 4
#else
#define WY 2
#endif
#define MAXKW 16                         // inner taps at most
#define MAXRQ ((R * (MAXKW / 4) + WX - 1) / WX)  // pieces of A per lane at most

struct Params
{
    T* x;            // 0: padded input
    T* w;            // 8: A [G, Cog, Cg KO, KW4]
    T* y;            // 16: output [N, G Cog, O]
    T* bias;         // 24: [G Cog] (USE_BIAS)
    uint cog;        // 32: rows per group
    uint cg;         // 36: channels per group
    uint ko;         // 40: outer taps
    uint kh;         // 44: outer taps along h (ko = kd kh)
    int off_d;       // 48: x offset per kd (dd Hp Wp)
    int off_h;       // 52: x offset per kh (dh Wp)
    uint nv;         // 56: virtual columns
    uint plane;      // 60: padded values per channel
    int x_sn;        // 64: x stride per sample
    int w_sm;        // 68: A's row stride (Cg KO KW4)
    int y_sn;        // 72: y stride per sample
    int y_sc;        // 76: y stride per channel (the output's positions)
    uint wp;         // 80: padded row (Wp; 1-D: nv)
    uint wo;         // 84: output row (Wo; 1-D: nv)
    uint hp;         // 88: padded rows per plane (3-D; else nv)
    uint ho;         // 92: output rows per plane (3-D; else nv)
    uint groups;     // 96
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int KW = 3;        // inner taps (<= MAXKW)
[vk::constant_id(1)] const int DW = 1;        // inner dilation
[vk::constant_id(2)] const int USE_BIAS = 0;

#if defined(T_f32)
typedef float4 S4;  // four values as stored
typedef float4 V4;  // and as computed
typedef float V;
V4 ld4(S4* q) { return loadAligned<16>(q); }
#elif defined(T_f16h)
typedef uint2 S4;
typedef half4 V4;
typedef half V;
V4 ld4(S4* q) { return bit_cast<half4>(loadAligned<8>(q)); }
#else
typedef uint2 S4;
typedef float4 V4;
typedef float V;
V4 ld4(S4* q)
{
    uint2 v = loadAligned<8>(q);
#if defined(T_f16)
    return float4(f16tof32(v.x & 0xffffu), f16tof32(v.x >> 16), f16tof32(v.y & 0xffffu), f16tof32(v.y >> 16));
#else
    return float4(asfloat(v.x << 16), asfloat(v.x & 0xffff0000u), asfloat(v.y << 16), asfloat(v.y & 0xffff0000u));
#endif
}
#endif

float ld(T v)
{
#if defined(T_bf16)
    return asfloat(uint(v) << 16);
#else
    return float(v);
#endif
}

T st(float f)
{
#if defined(T_f16) || defined(T_f16h)
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

// Values s .. s + 3 of the 8 in a, b (s known at compile time once specialised).
V4 window(V4 a, V4 b, int s)
{
    if (s == 0)
        return a;
    if (s == 1)
        return V4(a.y, a.z, a.w, b.x);
    if (s == 2)
        return V4(a.z, a.w, b.x, b.y);
    return V4(a.w, b.x, b.y, b.z);
}

[shader("compute")]
[numthreads(WX, WY, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    const int KW4 = (KW + 3) / 4 * 4;
    const int KC = KW4 / 4;                 // pieces of A per row per (c, outer tap)
    const int RQ = (R * KC + WX - 1) / WX;  // pieces per lane
    uint lane = WaveGetLaneIndex();
    uint n = gid.z / p.groups, grp = gid.z % p.groups;
    uint g = gid.x * WX + lane;  // this lane's column group (of 4 NC columns)
    uint row0 = gid.y * (WY * R) + tid.y * R;
    S4* x4 = (S4*)p.x + (int(n) * p.x_sn + int(grp * p.cg * p.plane)) / 4 + int(g * NC);
    S4* w4 = (S4*)p.w + int(grp * p.cog) * (p.w_sm / 4);
    // This lane's pieces of A: piece i of a (c, outer tap) block is row
    // i / KC's piece i % KC (pieces past R KC repeat the last, unused).
    S4* pa[MAXRQ];
    [ForceUnroll]
    for (int q = 0; q < MAXRQ; q++)
    {
        uint i = min(uint(q * WX) + lane, uint(R * KC - 1));
        pa[q] = w4 + int(min(row0 + i / uint(KC), p.cog - 1)) * (p.w_sm / 4) + int(i % uint(KC));
    }

    V4 acc[R][NC];
    [ForceUnroll]
    for (int i = 0; i < R; i++)
        [ForceUnroll]
        for (int j = 0; j < NC; j++)
            acc[i][j] = V4(0);

    for (uint c = 0; c < p.cg; c++)
    {
        for (uint o = 0; o < p.ko; o++)
        {
            int off = int(o / p.kh) * p.off_d + int(o % p.kh) * p.off_h;
            S4* xb = x4 + (int(c * p.plane) + off) / 4;
#if defined(T_f16h)
            // A's pieces as stored (4 halves in a uint2): a shuffle of one
            // uint carries two taps' weights for a row.
            uint2 av[MAXRQ];
            [ForceUnroll]
            for (int q = 0; q < MAXRQ; q++)
            {
                if (q < RQ)
                    av[q] = loadAligned<8>(pa[q] + int(c * p.ko + o) * KC);
            }
            [ForceUnroll]
            for (int t2 = 0; t2 < MAXKW; t2 += 2)
            {
                if (t2 < KW)
                {
                    uint pair[R];
                    [ForceUnroll]
                    for (int i = 0; i < R; i++)
                    {
                        const int piece = i * KC + t2 / 4;
                        pair[i] = WaveReadLaneAt(av[piece / WX][(t2 % 4) / 2], piece % WX);
                    }
                    [ForceUnroll]
                    for (int h = 0; h < 2; h++)
                    {
                        const int t = t2 + h;
                        if (t < KW)
                        {
                            const int e = t * DW, e4 = e >> 2, s = e & 3;
                            V4 bv[NC];
                            [ForceUnroll]
                            for (int j = 0; j < NC; j++)
                            {
                                V4 lo = ld4(xb + e4 + j);
                                bv[j] = s == 0 ? lo : window(lo, ld4(xb + e4 + j + 1), s);
                            }
                            [ForceUnroll]
                            for (int i = 0; i < R; i++)
                            {
                                half a = bit_cast<half>(uint16_t(h == 0 ? pair[i] & 0xffffu : pair[i] >> 16));
                                [ForceUnroll]
                                for (int j = 0; j < NC; j++)
                                    acc[i][j] += a * bv[j];
                            }
                        }
                    }
                }
            }
#else
            V4 av[MAXRQ];
            [ForceUnroll]
            for (int q = 0; q < MAXRQ; q++)
            {
                if (q < RQ)
                    av[q] = ld4(pa[q] + int(c * p.ko + o) * KC);
            }
            [ForceUnroll]
            for (int t = 0; t < MAXKW; t++)
            {
                if (t < KW)
                {
                    const int e = t * DW, e4 = e >> 2, s = e & 3;
                    V4 bv[NC];
                    [ForceUnroll]
                    for (int j = 0; j < NC; j++)
                    {
                        V4 lo = ld4(xb + e4 + j);
                        bv[j] = s == 0 ? lo : window(lo, ld4(xb + e4 + j + 1), s);
                    }
                    [ForceUnroll]
                    for (int i = 0; i < R; i++)
                    {
                        const int piece = i * KC + t / 4;
                        V a = WaveReadLaneAt(av[piece / WX][t % 4], piece % WX);
                        [ForceUnroll]
                        for (int j = 0; j < NC; j++)
                            acc[i][j] += a * bv[j];
                    }
                }
            }
#endif
        }
    }

    // Store the valid columns: v -> (od, oh, ow), stored at (od Ho + oh) Wo + ow.
    T* yb = p.y + int(n) * p.y_sn + int(grp * p.cog) * p.y_sc;
    [ForceUnroll]
    for (int j = 0; j < NC; j++)
    {
        [ForceUnroll]
        for (int e = 0; e < 4; e++)
        {
            uint v = (g * NC + uint(j)) * 4 + uint(e);
            if (v >= p.nv)
                continue;
            uint ow = v % p.wp, q = v / p.wp;
            uint oh = q % p.hp, od = q / p.hp;
            if (ow >= p.wo || oh >= p.ho)
                continue;
            uint idx = (od * p.ho + oh) * p.wo + ow;
            [ForceUnroll]
            for (int i = 0; i < R; i++)
            {
                uint m = row0 + uint(i);
                if (m < p.cog)
                {
                    float val = float(acc[i][j][e]);
                    if (USE_BIAS != 0)
                        val += ld(p.bias[grp * p.cog + m]);
                    yb[int(m) * p.y_sc + int(idx)] = st(val);
                }
            }
        }
    }
}
