// axis T: f32=float f64=double i32=int i64=int64_t
//
// Batched GEMM: C[b] = alpha * A[b] @ B[b] (+ beta * Cin[b] when USE_CIN).
// Integer variants wrap on overflow, as the CPU's integer matmul does.
// Every operand is addressed through (row, column, batch) element strides,
// so transposed or broadcast operands need no copies.
//
// Each 16x16 workgroup computes a BM x BN tile of C, stepping through K in
// slices of TK staged in shared memory; each thread accumulates an RM x RN
// block (rows ty + 16i, columns tx + 16j). Workgroup grid: (N/BN, M/BM, batch).

// Tile shape. On the UHD 620, 64x64 tiles with 4x4 per thread beat
// 128x128 (8x8), 128x64 and 64x128: larger per-thread blocks spill registers
// (especially for float64).
#define BM 64
#define BN 64
#define TK 16
#define RM (BM / 16)
#define RN (BN / 16)

struct Params
{
    T* a;            // 0
    T* b;            // 8
    T* c;            // 16
    T* cin;          // 24
    double alpha;    // 32
    double beta;     // 40
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

// +1 column of padding: A tiles are stored transposed, and without it the
// threads of a load would hit the same shared-memory bank.
groupshared T As[TK][BM + 1];
groupshared T Bs[TK][BN];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint tx = tid.x, ty = tid.y;
    uint lin = ty * 16 + tx;
    uint row0 = gid.y * BM, col0 = gid.x * BN;
    int batch = int(gid.z);
    int abase = batch * p.a_sb, bbase = batch * p.b_sb;

    T acc[RM][RN];
    for (uint i = 0; i < RM; i++)
        for (uint j = 0; j < RN; j++)
            acc[i][j] = T(0);

    for (uint k0 = 0; k0 < p.k; k0 += TK)
    {
        // Stage A[row0 .. +BM, k0 .. +TK] and B[k0 .. +TK, col0 .. +BN].
        // Consecutive threads take consecutive k (for A) and consecutive
        // columns (for B): contiguous reads for row-major operands.
        for (uint e = lin; e < BM * TK; e += 256)
        {
            uint r = e / TK, kk = e % TK;
            uint gm = row0 + r, gk = k0 + kk;
            As[kk][r] = (gm < p.m && gk < p.k) ? p.a[abase + int(gm) * p.a_sm + int(gk) * p.a_sk] : T(0);
        }
        for (uint e = lin; e < TK * BN; e += 256)
        {
            uint kb = e / BN, cb = e % BN;
            uint gkb = k0 + kb, gn = col0 + cb;
            Bs[kb][cb] = (gkb < p.k && gn < p.n) ? p.b[bbase + int(gkb) * p.b_sk + int(gn) * p.b_sn] : T(0);
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint kk = 0; kk < TK; kk++)
        {
            T av[RM], bv[RN];
            for (uint i = 0; i < RM; i++)
                av[i] = As[kk][ty + 16 * i];
            for (uint j = 0; j < RN; j++)
                bv[j] = Bs[kk][tx + 16 * j];
            for (uint i = 0; i < RM; i++)
                for (uint j = 0; j < RN; j++)
                    acc[i][j] += av[i] * bv[j];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    T alpha = T(p.alpha), beta = T(p.beta);
    for (uint i = 0; i < RM; i++)
    {
        uint gm = row0 + ty + 16 * i;
        if (gm >= p.m)
            continue;
        for (uint j = 0; j < RN; j++)
        {
            uint gn = col0 + tx + 16 * j;
            if (gn >= p.n)
                continue;
            T v = alpha * acc[i][j];
            if (USE_CIN != 0)  // beta == 0 must ignore Cin entirely (it may hold NaN)
                v += beta * p.cin[batch * p.cin_sb + int(gm) * p.cin_sm + int(gn) * p.cin_sn];
            p.c[batch * p.c_sb + int(gm) * p.c_sm + int(gn) * p.c_sn] = v;
        }
    }
}
