// axis T: f32=float f16=half bf16=uint16_t
//
// Batched GEMM for Arm Mali GPUs with 16-lane subgroups (Valhall), as
// shaders/gemm4.slang (same parameters, C[b] = alpha * A[b] @ B[b]
// (+ beta * Cin[b])) but sharing A across a subgroup by shuffles. Operands
// and C are T: float32, or float16 / bfloat16 storage, whose products are
// accumulated in float32 and rounded once (to nearest even), as PyTorch's
// CPU does.
//
// The host passes row-major A and B (unit k and n strides) whose rows and
// batches start 16-byte aligned (8-byte for 16-bit T), K a multiple of 4 and
// B's rows padded to a multiple of 4 columns (zeros past N, never written to
// C), as gemm4.slang's.
//
// A workgroup of WY subgroups of 16 lanes computes a (WY R) x 64 tile of C:
// subgroup y the rows y R .. y R + R - 1, lane l the columns 4l .. 4l + 3.
// Per k a lane loads its 4 values of B's row (one 16-byte load) and does
// R x 4 multiply-adds, taking A[row][k] from the lane that loaded it: per
// block of KB values of k, each lane loads one 16-byte piece of A (the R
// rows' KB values are R KB / 4 pieces, spread over the subgroup's lanes),
// and the multiply-adds read them by shuffles (WaveReadLaneAt with constant
// lanes once unrolled). On Mali every load, from cache too, costs as much
// as about 12 multiply-adds of the warp (and a shuffle about 1), so this
// does 32 multiply-adds per load and 8 shuffles, against gemm4.slang's 2 x 8
// outputs per thread (16 per 1.25 loads) that also reloads A per lane.
//
// Measured on the Mali-G57 MC2 (Galaxy A13; float32, 2048^3): 67 GFLOP/s,
// against gemm4.slang's 33; with operands from cache 75, about 62% of the
// 122 the FMA units could do. 8 rows per subgroup and blocks of 8 values of
// k were best (4, 6, 10 and 12 rows, 4, 16 and 32 values tried; 12 rows
// spill). Needs exactly 16 lanes per subgroup (vkcore:
// EXACT_SUBGROUP_KERNELS), in order along x.

#include "_half.slang"
#include "_slot.slang"

#define WX 16  // lanes: a subgroup
#define WY 2   // subgroups per workgroup
#define R 8    // rows per subgroup
#if defined(T_bf16)
#define KB 4   // values of k per block of A (bfloat16 spilled with 8: 43 GFLOP/s against 58)
#else
#define KB 8
#endif
#define KC (KB / 4)              // 4-value pieces of a row per block
#define RQ ((R * KC + WX - 1) / WX)  // pieces of A per lane per block
#define BM (WY * R)
#define BN (WX * 4)

struct Params
{
    T* a;            // 0
    T* b;            // 8
    T* c;            // 16
    T* cin;          // 24
    slot64 alpha;    // 32  (float bits)
    slot64 beta;     // 40
    uint m;          // 48
    uint n;          // 52
    uint k;          // 56
    uint _pad;       // 60
    int a_sm, a_sk, a_sb;        // 64
    int b_sk, b_sn, b_sb;        // 76
    int c_sm, c_sn, c_sb;        // 88
    int cin_sm, cin_sn, cin_sb;  // 100 .. 112
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int USE_CIN = 0;

// Four consecutive values of T in one load (S4), as floats.
#if defined(T_f32)
typedef float4 S4;
float4 widen(S4 v) { return v; }
#else
typedef uint2 S4;
#if defined(T_f16)
float4 widen(S4 v)
{
    return float4(f16tof32(v.x & 0xffffu), f16tof32(v.x >> 16), f16tof32(v.y & 0xffffu), f16tof32(v.y >> 16));
}
#else
float4 widen(S4 v)
{
    return float4(asfloat(v.x << 16), asfloat(v.x & 0xffff0000u), asfloat(v.y << 16), asfloat(v.y & 0xffff0000u));
}
#endif
#endif

float4 ld4(S4* q)
{
#if defined(T_f32)
    return loadAligned<16>(q);
#else
    return widen(loadAligned<8>(q));
#endif
}

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

[shader("compute")]
[numthreads(WX, WY, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint lane = WaveGetLaneIndex();
    int batch = int(gid.z);
    uint n4 = (p.n + 3) / 4;  // (B's padded column groups)
    uint g = gid.x * WX + lane;  // this lane's column group
    uint row0 = gid.y * BM + tid.y * R;
    S4* a4 = (S4*)p.a + (batch * p.a_sb) / 4;
    int bstep = p.b_sk / 4;
    S4* pb = (S4*)p.b + (batch * p.b_sb) / 4 + int(min(g, n4 - 1));
    // This lane's pieces of A: piece i of a block is row i / KC's piece i % KC
    // (pieces past R * KC repeat the last row's, unused).
    S4* pa[RQ];
    [ForceUnroll]
    for (uint q = 0; q < RQ; q++)
    {
        uint i = min(q * WX + lane, uint(R * KC - 1));
        pa[q] = a4 + int(min(row0 + i / KC, p.m - 1)) * (p.a_sm / 4) + int(i % KC);
    }

    float4 acc[R];
    [ForceUnroll]
    for (uint i = 0; i < R; i++)
        acc[i] = float4(0);

    uint blocks = p.k / KB;
    for (uint kb = 0; kb < blocks; kb++)
    {
        float4 av[RQ];
        [ForceUnroll]
        for (uint q = 0; q < RQ; q++)
            av[q] = ld4(pa[q] + int(kb * KC));
        [ForceUnroll]
        for (uint kk = 0; kk < KB; kk++)
        {
            float4 bv = ld4(pb + int(kk) * bstep);
            [ForceUnroll]
            for (uint i = 0; i < R; i++)
            {
                const uint piece = i * KC + kk / 4;
                acc[i] += WaveReadLaneAt(av[piece / WX][kk % 4], piece % WX) * bv;
            }
        }
        pb = pb + KB * bstep;
    }
    // The last K % KB values of k (K is a multiple of 4), A read directly.
    for (uint k4 = blocks * KC; k4 < p.k / 4; k4++)
    {
        float4 av[R];
        [ForceUnroll]
        for (uint i = 0; i < R; i++)
            av[i] = ld4(a4 + int(min(row0 + i, p.m - 1)) * (p.a_sm / 4) + int(k4));
        [ForceUnroll]
        for (uint kk = 0; kk < 4; kk++)
        {
            float4 bv = ld4(pb + int(kk) * bstep);
            [ForceUnroll]
            for (uint i = 0; i < R; i++)
                acc[i] += av[i][kk] * bv;
        }
        pb = pb + 4 * bstep;
    }

    float alpha = slot_float(p.alpha), beta = slot_float(p.beta);
    if (g >= n4)
        return;
    [ForceUnroll]
    for (uint i = 0; i < R; i++)
    {
        uint gm = row0 + i;
        if (gm >= p.m)
            break;
        [ForceUnroll]
        for (uint e = 0; e < 4; e++)
        {
            uint gn = g * 4 + e;
            if (gn >= p.n)
                break;
            float v = alpha * acc[i][e];
            if (USE_CIN != 0)
                v += beta * ld(p.cin[batch * p.cin_sb + int(gm) * p.cin_sm + int(gn) * p.cin_sn]);
            p.c[batch * p.c_sb + int(gm) * p.c_sm + int(gn) * p.c_sn] = st(v);
        }
    }
}
