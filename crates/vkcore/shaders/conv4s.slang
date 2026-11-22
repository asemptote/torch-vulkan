// axis T: f32=float f16=half bf16=uint16_t f16h=half
// axis D: d1=int d2=int d3=int d5=int
// axis TILE: r8n1=int r4n2=int r8n2=int r8n1w4=int
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
// slack after the last plane for the columns and taps read past the end;
// the weight as A [G, Cog, Cg KO, KW4]: rows of the KO = KD KH outer taps'
// KW inner taps, zero-padded to KW4 (a multiple of 4).
//
// A subgroup computes R rows x 16 NC 4 columns: lane l the columns
// 4 NC l .. 4 NC l + 4 NC - 1. The inner taps are taken 4 at a time (a
// group): lane i < R loads row i's 4 weights of the group (one 4-value
// piece of A) and the others take them by shuffles; the group's 4 taps read
// the lane's columns at 0, dw, 2 dw and 3 dw, which are 4-value windows of
// the NW 4-value pieces of the input row around them (dw is the variant's
// D, so the windows' shifts are compile-time constants). The next group
// starts dw pieces further on, so the pieces it shares with this one stay in
// registers: for dw = 1, one load of the input and one of A per 4 taps, where
// a product of im2col columns loads 4 of B. Sums are float32 with fused
// multiply-adds, rounded once (to nearest even) with the bias added, as
// gemm4s.slang's.
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
#elif defined(TILE_r8n2)
#define R 8
#define NC 2
#else
#define R 8
#define NC 1
#endif
#if defined(TILE_r8n1w4)
#define WY 4
#else
#define WY 2
#endif
#if defined(D_d2)
#define DW 2
#elif defined(D_d3)
#define DW 3
#elif defined(D_d5)
#define DW 5
#else
#define DW 1
#endif
#define NW ((3 * DW + 3) / 4 + NC)  // 4-value pieces of the input row a group reads

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
    uint kg;         // 100: groups of 4 inner taps (KW4 / 4)
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int USE_BIAS = 0;

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
    return float4(bit_cast<half4>(v));  // (exact)
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

// The input's pieces are kept as loaded (W4: 16-bit values for f16 and
// bf16 storage, widened per tap after the window is taken: widening each
// piece once as it was loaded ran 3x slower on the HD 530).
#if defined(T_f16)
typedef half4 W4;
W4 ldw(S4* q) { return bit_cast<half4>(loadAligned<8>(q)); }
V4 widen(W4 v) { return float4(v); }
#elif defined(T_bf16)
typedef vector<uint16_t, 4> W4;
W4 ldw(S4* q)
{
    uint2 v = loadAligned<8>(q);
    return W4(uint16_t(v.x & 0xffffu), uint16_t(v.x >> 16), uint16_t(v.y & 0xffffu), uint16_t(v.y >> 16));
}
V4 widen(W4 v) { return asfloat(uint4(v) << 16); }
#else
typedef V4 W4;
W4 ldw(S4* q) { return ld4(q); }
V4 widen(W4 v) { return v; }
#endif

// Values s .. s + 3 of the 8 in a, b.
W4 window(W4 a, W4 b, int s)
{
    if (s == 0)
        return a;
    if (s == 1)
        return W4(a.y, a.z, a.w, b.x);
    if (s == 2)
        return W4(a.z, a.w, b.x, b.y);
    return W4(a.w, b.x, b.y, b.z);
}

[shader("compute")]
[numthreads(WX, WY, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint lane = WaveGetLaneIndex();
    uint n = gid.z / p.groups, grp = gid.z % p.groups;
    uint g = gid.x * WX + lane;  // this lane's column group (of 4 NC columns)
    uint row0 = gid.y * (WY * R) + tid.y * R;
    S4* x4 = (S4*)p.x + (int(n) * p.x_sn + int(grp * p.cg * p.plane)) / 4 + int(g * NC);
    // This lane's row of A (lanes past R repeat the rows, unused).
    uint arow = min(row0 + lane % R, p.cog - 1);
    S4* pa = (S4*)p.w + int(grp * p.cog + arow) * (p.w_sm / 4);

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
            S4* pab = pa + int(c * p.ko + o) * int(p.kg);
            W4 win[NW];
            [ForceUnroll]
            for (int i = 0; i < NW; i++)
                win[i] = ldw(xb + i);
            for (uint kg = 0; kg < p.kg; kg++)
            {
#if defined(T_f16h)
                uint2 av = loadAligned<8>(pab + int(kg));  // (4 halves: this group's taps of row lane % R)
#else
                V4 av = ld4(pab + int(kg));
#endif
                [ForceUnroll]
                for (int t = 0; t < 4; t++)
                {
                    const int e = t * DW, e4 = e >> 2, s = e & 3;
                    V4 bv[NC];
                    [ForceUnroll]
                    for (int j = 0; j < NC; j++)
                        bv[j] = widen(window(win[e4 + j], win[e4 + j + (s == 0 ? 0 : 1)], s));
                    [ForceUnroll]
                    for (int i = 0; i < R; i++)
                    {
#if defined(T_f16h)
                        uint pair = WaveReadLaneAt(t < 2 ? av.x : av.y, i);
                        half a = bit_cast<half>(uint16_t((t % 2) == 0 ? pair & 0xffffu : pair >> 16));
#else
                        V a = WaveReadLaneAt(av[t], i);
#endif
                        [ForceUnroll]
                        for (int j = 0; j < NC; j++)
                            acc[i][j] += a * bv[j];
                    }
                }
                // The next group: DW pieces on; keep the pieces it shares.
                xb = xb + DW;
                [ForceUnroll]
                for (int i = 0; i < NW; i++)
                    win[i] = i + DW < NW ? win[i + DW] : ldw(xb + i);
            }
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
