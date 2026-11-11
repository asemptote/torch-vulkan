// axis T: f32=float f64=double
//
// Batched GEMM with register tiles and subgroup broadcasts, for GPUs with
// subgroup size control, shuffles and broadcasts: C[b] = alpha * A[b] @ B[b]
// (+ beta * Cin[b] when USE_CIN), operands addressed through element
// strides as in gemm.slang (whose Params this is, its padding slot unused).
//
// A workgroup of WG threads in subgroups of S lanes computes a
// (WG / S) MR x S NR tile of C: subgroup g its rows g MR .., lane l the
// columns l + j S (j < NR). Per block of S values of k, the workgroup stages
// B's S x S NR block in shared memory and each lane loads A[row, k0 + l] for
// its subgroup's MR rows into registers; then for kk < S a lane reads its NR
// values of B's row kk from shared memory and takes A[row, k0 + kk] from
// lane kk by a broadcast (WaveReadLaneAt with a constant lane once the loop
// is unrolled): MR NR multiply-adds per NR shared-memory loads, against
// gemm.slang's 16 per 8.
//
// vkcore creates the pipeline with a required subgroup size (kernels.rs
// SUBGROUP_KERNELS: 16 lanes for float32, 8 for float64, whose lanes then
// have twice the registers), every subgroup full; the kernel is instantiated
// for 8, 16 and 32 lanes and picks the size it runs with. On the UHD 620 at
// n = 2048: float32 about 2.2x gemm.slang, float64 about 1.35x.

#include "_slot.slang"

struct Params
{
    T* a;
    T* b;
    T* c;
    T* cin;
    slot64 alpha;
    slot64 beta;
    uint m;
    uint n;
    uint k;
    uint _pad;
    int a_sm, a_sk, a_sb;
    int b_sk, b_sn, b_sb;
    int c_sm, c_sn, c_sb;
    int cin_sm, cin_sn, cin_sb;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

T param(slot64 bits)
{
#if defined(T_f64)
    return slot_double(bits);
#else
    return slot_float(bits);
#endif
}

[vk::constant_id(0)] const int USE_CIN = 0;

#define WG 128
#define MR 8
// B's S x S NR block, within Vulkan's guaranteed 16 KiB of shared memory
// (float64 with 32 lanes takes NR = 2; csrc/linalg.cpp mirrors the tile shapes).
#if defined(T_f64)
#define BS_MAX 2048
#else
#define BS_MAX 4096
#endif
groupshared T Bs[BS_MAX];

void run<let S : int, let NR : int>(uint t, uint lane, uint2 grp, int batch)
{
    const uint CN = S * NR;
    uint row0 = (grp.y * (WG / S) + t / S) * MR, col0 = grp.x * CN;
    int abase = batch * p.a_sb, bbase = batch * p.b_sb;
    T acc[MR][NR];
    [ForceUnroll]
    for (int i = 0; i < MR; i++)
        [ForceUnroll]
        for (int j = 0; j < NR; j++)
            acc[i][j] = T(0);
    for (uint k0 = 0; k0 < p.k; k0 += S)
    {
        // B's block (zeros past k and n) and this lane's A values (zeros past
        // m and k), so the tail of k adds nothing.
        for (uint e = t; e < S * CN; e += WG)
        {
            uint r = e / CN, cc = e % CN;
            T v = T(0);
            if (k0 + r < p.k && col0 + cc < p.n)
                v = p.b[bbase + int(k0 + r) * p.b_sk + int(col0 + cc) * p.b_sn];
            Bs[e] = v;
        }
        uint ka = k0 + lane;
        T av[MR];
        [ForceUnroll]
        for (int i = 0; i < MR; i++)
        {
            av[i] = T(0);
            if (row0 + i < p.m && ka < p.k)
                av[i] = p.a[abase + int(row0 + i) * p.a_sm + int(ka) * p.a_sk];
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (int kk = 0; kk < S; kk++)
        {
            T bv[NR];
            [ForceUnroll]
            for (int j = 0; j < NR; j++)
                bv[j] = Bs[kk * CN + lane + j * S];
            [ForceUnroll]
            for (int i = 0; i < MR; i++)
            {
                T ai = WaveReadLaneAt(av[i], kk);
                [ForceUnroll]
                for (int j = 0; j < NR; j++)
                    acc[i][j] += ai * bv[j];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    T alpha = param(p.alpha), beta = param(p.beta);
    [ForceUnroll]
    for (int i = 0; i < MR; i++)
    {
        uint row = row0 + i;
        if (row >= p.m)
            continue;
        [ForceUnroll]
        for (int j = 0; j < NR; j++)
        {
            uint col = col0 + lane + uint(j * S);
            if (col >= p.n)
                continue;
            T v = alpha * acc[i][j];
            if (USE_CIN != 0)  // beta == 0 must ignore Cin entirely (it may hold NaN)
                v += beta * p.cin[batch * p.cin_sb + int(row) * p.cin_sm + int(col) * p.cin_sn];
            p.c[batch * p.c_sb + int(row) * p.c_sm + int(col) * p.c_sn] = v;
        }
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint S = WaveGetLaneCount(), lane = WaveGetLaneIndex();
    int batch = int(gid.z);
    if (S == 8)
        run<8, 4>(tid.x, lane, gid.xy, batch);
    else if (S == 16)
        run<16, 4>(tid.x, lane, gid.xy, batch);
#if defined(T_f64)
    else
        run<32, 2>(tid.x, lane, gid.xy, batch);
#else
    else
        run<32, 4>(tid.x, lane, gid.xy, batch);
#endif
}
