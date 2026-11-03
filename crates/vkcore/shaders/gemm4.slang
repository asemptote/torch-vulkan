// axis T: f32=float
//
// Batched float32 GEMM for Arm Mali GPUs, as shaders/gemm.slang (same
// parameters, C[b] = alpha * A[b] @ B[b] (+ beta * Cin[b])) but without
// shared memory and with 16-byte loads: Mali's shared memory is ordinary
// cached memory, and its load/store unit favours vector loads. The host uses
// it for row-major A and B (unit k and n strides) whose rows and batches
// start 16-byte aligned, with K and N multiples of 4.
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
    float* a;        // 0
    float* b;        // 8
    float* c;        // 16
    float* cin;      // 24
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

// A raw 64-bit parameter slot (float bits in the low word, as gemm.slang's).
float param(slot64 bits)
{
    return slot_float(bits);
}

[vk::constant_id(0)] const int USE_CIN = 0;

[shader("compute")]
[numthreads(WX, WY, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint tx = tid.x, ty = tid.y;
    uint row0 = gid.y * BM, g0 = gid.x * (WX * RN4);
    int batch = int(gid.z);
    float4* a4 = (float4*)p.a;
    float4* b4 = (float4*)p.b;
    uint n4 = p.n / 4;
    int aoff[RM], boff[RN4];  // (in float4s)
    [ForceUnroll]
    for (uint i = 0; i < RM; i++)
        aoff[i] = (batch * p.a_sb + int(min(row0 + ty + WY * i, p.m - 1)) * p.a_sm) / 4;
    [ForceUnroll]
    for (uint j = 0; j < RN4; j++)
        boff[j] = (batch * p.b_sb) / 4 + int(min(g0 + tx + WX * j, n4 - 1));
    int bstep = p.b_sk / 4;

    float4 acc[RM][RN4];
    [ForceUnroll]
    for (uint i = 0; i < RM; i++)
        [ForceUnroll]
        for (uint j = 0; j < RN4; j++)
            acc[i][j] = float4(0);

    for (uint k4 = 0; k4 < p.k / 4; k4++)
    {
        float4 av[RM];
        [ForceUnroll]
        for (uint i = 0; i < RM; i++)
            av[i] = loadAligned<16>(a4 + aoff[i] + int(k4));
        int brow = int(k4) * 4 * bstep;
        [ForceUnroll]
        for (uint kk = 0; kk < 4; kk++)
        {
            float4 bv[RN4];
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

    float alpha = param(p.alpha), beta = param(p.beta);
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
                float v = alpha * acc[i][j][e];
                if (USE_CIN != 0)
                    v += beta * p.cin[batch * p.cin_sb + int(gm) * p.cin_sm + int(gn) * p.cin_sn];
                p.c[batch * p.c_sb + int(gm) * p.c_sm + int(gn) * p.c_sn] = v;
            }
        }
    }
}
