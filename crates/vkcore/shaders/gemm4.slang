// axis T: f32=float i32=int
//
// Batched float32 / int32 GEMM for Arm Mali GPUs, as shaders/gemm.slang
// (same parameters, C[b] = alpha * A[b] @ B[b] (+ beta * Cin[b]); int32
// wraps) but without shared memory and with 16-byte loads: Mali's shared
// memory is ordinary cached memory, and its load/store unit favours vector
// loads. The host uses it for row-major A and B (unit k and n strides) whose
// rows and batches start 16-byte aligned, with K a multiple of 4 and B's
// rows padded to a multiple of 4 columns (zeros past N, never written to C):
// other operands are copied into that layout first.
//
// Each WX x WY workgroup computes a (WY*RM) x (WX*RN4*4) tile of C; thread
// (tx, ty) the rows ty + WY*i and the float4 column groups tx + WX*j.
//
// Tile shape. On the Mali-G57 MC2 (2048^3 product): 2 rows x 2 float4 per
// thread in 16x4 workgroups, 31 GFLOP/s, against 18-24 for gemm.slang's
// tiles; 3 or 4 rows per thread fall to 3-9 GFLOP/s (unrolled or not).
#define WX 16
#define WY 4
#define RM 2
#define RN4 2
#define BM (WY * RM)
#define BN (WX * RN4 * 4)

#include "_slot.slang"

struct Params
{
    T* a;            // 0
    T* b;            // 8
    T* c;            // 16
    T* cin;          // 24
    slot64 alpha;    // 32  (raw slots: see param)
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

typedef vector<T, 4> T4;

// A raw 64-bit parameter slot, as gemm.slang's: float bits in the low word
// for float32, an int64 for int32.
T param(slot64 bits)
{
#if defined(T_f32)
    return slot_float(bits);
#else
    return T(slot_integer(bits));
#endif
}

[vk::constant_id(0)] const int USE_CIN = 0;

[shader("compute")]
[numthreads(WX, WY, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint tx = tid.x, ty = tid.y;
    uint row0 = gid.y * BM, g0 = gid.x * (WX * RN4);
    int batch = int(gid.z);
    T4* a4 = (T4*)p.a;
    T4* b4 = (T4*)p.b;
    uint n4 = (p.n + 3) / 4;  // (B's padded column groups)
    int aoff[RM], boff[RN4];  // (in float4s)
    [ForceUnroll]
    for (uint i = 0; i < RM; i++)
        aoff[i] = (batch * p.a_sb + int(min(row0 + ty + WY * i, p.m - 1)) * p.a_sm) / 4;
    [ForceUnroll]
    for (uint j = 0; j < RN4; j++)
        boff[j] = (batch * p.b_sb) / 4 + int(min(g0 + tx + WX * j, n4 - 1));
    int bstep = p.b_sk / 4;

    T4 acc[RM][RN4];
    [ForceUnroll]
    for (uint i = 0; i < RM; i++)
        [ForceUnroll]
        for (uint j = 0; j < RN4; j++)
            acc[i][j] = T4(T(0));

    for (uint k4 = 0; k4 < p.k / 4; k4++)
    {
        T4 av[RM];
        [ForceUnroll]
        for (uint i = 0; i < RM; i++)
            av[i] = loadAligned<16>(a4 + aoff[i] + int(k4));
        int brow = int(k4) * 4 * bstep;
        [ForceUnroll]
        for (uint kk = 0; kk < 4; kk++)
        {
            T4 bv[RN4];
            [ForceUnroll]
            for (uint j = 0; j < RN4; j++)
                bv[j] = loadAligned<16>(b4 + boff[j] + brow + int(kk) * bstep);
            [ForceUnroll]
            for (uint i = 0; i < RM; i++)
                [ForceUnroll]
                for (uint j = 0; j < RN4; j++)
                    acc[i][j] += av[i][kk] * bv[j];
        }
    }

    T alpha = param(p.alpha), beta = param(p.beta);
    [ForceUnroll]
    for (uint i = 0; i < RM; i++)
    {
        uint gm = row0 + ty + WY * i;
        if (gm >= p.m)
            continue;
        [ForceUnroll]
        for (uint j = 0; j < RN4; j++)
        {
            uint g = g0 + tx + WX * j;
            if (g >= n4)
                continue;
            [ForceUnroll]
            for (uint e = 0; e < 4; e++)
            {
                uint gn = g * 4 + e;
                if (gn >= p.n)
                    continue;
                T v = alpha * acc[i][j][e];
                if (USE_CIN != 0)
                    v += beta * p.cin[batch * p.cin_sb + int(gm) * p.cin_sm + int(gn) * p.cin_sn];
                p.c[batch * p.c_sb + int(gm) * p.c_sm + int(gn) * p.c_sn] = v;
            }
        }
    }
}
