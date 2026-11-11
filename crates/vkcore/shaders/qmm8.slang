// axis T: f32=float f16=half bf16=uint16_t
//
// aten::_weight_int8pack_mm: C = A @ (B * scales).T for A [M, K] (float32,
// float16 or bfloat16, any strides), int8 weights B [N, K] (contiguous) and
// per-output-channel scales [N] in A's dtype. As the CPU's kernel
// (ATen/native/cpu/int8mm_kernel.cpp) computes it: each weight dequantised
// to float32 as float(b) * float(scale), products accumulated in float32,
// the result rounded to A's dtype once (to nearest even).
//
// ROWS 0: gemm.slang's tiling: each 256-thread workgroup computes a 64x64
// tile of C, staging A's and the dequantised weights' slices of TK values of
// k in shared memory (the weights dequantised as they are staged: no float
// copy of B), each thread a 4x4 block. Grid: (N / 64, M / 64).
// ROWS r (1 to 8; a decoding step's few rows): matrix-vector products. A
// workgroup takes 8 weight rows (output columns), 32 threads each along k
// (contiguous reads of the weights), each accumulating all r rows of A;
// the 32 partial sums are added up in shared memory. Grid: N / 8.

#include "_half.slang"

struct Params
{
    T* a;            // 0
    int8_t* b;       // 8
    T* scales;       // 16
    T* c;            // 24  contiguous [M, N]
    uint m;          // 32
    uint n;          // 36
    uint k;          // 40
    int a_sm;        // 44
    int a_sk;        // 48
    uint _pad;       // 52
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int ROWS = 0;

#define BM 64
#define BN 64
#define TK 16
#define RM (BM / 16)
#define RN (BN / 16)

float ld(T x)
{
#if defined(T_bf16)
    return bit_cast<float>(uint(x) << 16);
#else
    return float(x);
#endif
}

T st(float f)
{
#if defined(T_f16)
    return to_half(f);
#elif defined(T_bf16)
    if (isnan(f))
        return uint16_t(0x7fc0);
    uint u = bit_cast<uint>(f);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return uint16_t(u >> 16);
#else
    return f;
#endif
}

// Shared memory, 8.3 KiB for either path: A's slice transposed [TK][BM + 1]
// (+1: no bank conflicts), the weights' [TK][BN], the scales [BN]; or the
// matrix-vector partial sums [column][row][lane] (lanes padded to 33).
groupshared float Sh[TK * (BM + 1) + TK * BN + BN];
#define AS(kk, r) Sh[(kk) * (BM + 1) + (r)]
#define BS(kk, c) Sh[TK * (BM + 1) + (kk) * BN + (c)]
#define SS(c) Sh[TK * (BM + 1) + TK * BN + (c)]
#define PART(col, r, lane) Sh[((col) * 8 + (r)) * 33 + (lane)]

void small_m(uint t, uint g)
{
    uint col = t / 32, lane = t % 32, n = g * 8 + col;
    uint nn = min(n, p.n - 1);
    float scale = ld(p.scales[nn]);
    float acc[8];
    [ForceUnroll]
    for (uint r = 0; r < 8; r++)
        acc[r] = 0.0;
    for (uint k = lane; k < p.k; k += 32)
    {
        float w = float(int(p.b[nn * p.k + k])) * scale;
        [ForceUnroll]
        for (uint r = 0; r < 8; r++)
            if (r < ROWS)
                acc[r] += ld(p.a[int(r) * p.a_sm + int(k) * p.a_sk]) * w;
    }
    [ForceUnroll]
    for (uint r = 0; r < 8; r++)
        PART(col, r, lane) = acc[r];
    GroupMemoryBarrierWithGroupSync();
    if (lane < ROWS && n < p.n)
    {
        float sum = 0.0;
        for (uint i = 0; i < 32; i++)
            sum += PART(col, lane, i);
        p.c[lane * p.n + n] = st(sum);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint tx = tid.x, ty = tid.y, lin = ty * 16 + tx;
    if (ROWS != 0)
    {
        small_m(lin, gid.x);
        return;
    }
    uint row0 = gid.y * BM, col0 = gid.x * BN;
    if (lin < BN)
        SS(lin) = ld(p.scales[min(col0 + lin, p.n - 1)]);

    float acc[RM][RN];
    for (uint i = 0; i < RM; i++)
        for (uint j = 0; j < RN; j++)
            acc[i][j] = 0.0;

    for (uint k0 = 0; k0 < p.k; k0 += TK)
    {
        GroupMemoryBarrierWithGroupSync();  // (Ss written; the previous slice consumed)
        // consecutive threads along k: contiguous reads of A's and B's rows
        // (indices clamped: lavapipe may perform a load before its test)
        for (uint e = lin; e < BM * TK; e += 256)
        {
            uint r = e / TK, kk = e % TK, gm = row0 + r, gk = k0 + kk;
            float v = ld(p.a[int(min(gm, p.m - 1)) * p.a_sm + int(min(gk, p.k - 1)) * p.a_sk]);
            AS(kk, r) = gm < p.m && gk < p.k ? v : 0.0;
        }
        for (uint e = lin; e < BN * TK; e += 256)
        {
            uint cb = e / TK, kk = e % TK, gn = col0 + cb, gk = k0 + kk;
            float w = float(int(p.b[min(gn, p.n - 1) * p.k + min(gk, p.k - 1)])) * SS(cb);
            BS(kk, cb) = gn < p.n && gk < p.k ? w : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint kk = 0; kk < TK; kk++)
        {
            float av[RM], bv[RN];
            for (uint i = 0; i < RM; i++)
                av[i] = AS(kk, ty + 16 * i);
            for (uint j = 0; j < RN; j++)
                bv[j] = BS(kk, tx + 16 * j);
            for (uint i = 0; i < RM; i++)
                for (uint j = 0; j < RN; j++)
                    acc[i][j] += av[i] * bv[j];
        }
    }

    for (uint i = 0; i < RM; i++)
    {
        uint gm = row0 + ty + 16 * i;
        for (uint j = 0; j < RN; j++)
        {
            uint gn = col0 + tx + 16 * j;
            if (gm < p.m && gn < p.n)
                p.c[gm * p.n + gn] = st(acc[i][j]);
        }
    }
}
