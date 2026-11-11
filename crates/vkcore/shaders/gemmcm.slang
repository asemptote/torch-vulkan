// axis T: f16=half
//
// Batched float16 GEMM on cooperative matrices (VK_KHR_cooperative_matrix):
// C[b] = alpha * A[b] @ B[b] (+ beta * Cin[b] when USE_CIN), operands
// addressed through element strides as in gemm.slang (whose Params this is).
// Products of float16 values accumulate in float32 (as the CPU's float16
// matmul and the other kernels' float32 path do; tensor cores sum them in
// another order, with float32-like error), and each result is rounded to
// float16 once, to nearest even.
//
// A workgroup of WG threads (subgroups of 32 lanes: vkcore requires that
// size) computes a BM x BN tile of C, its 8 subgroups in a 2 x 4 grid, each
// WM x WN fragments of 16 x 16 (64 x 64 values). Per block of BK values of k it stages A's
// BM x BK and B's BK x BN blocks in shared memory as float16, in 16-byte
// pieces of 8 values (zeros past the edges): one 16-byte load each where an
// operand is contiguous along k, m or n (A_MODE, B_MODE; csrc/linalg.cpp
// checks the strides and alignment), else 8 strided loads. The next block's
// pieces are loaded into registers while the subgroups multiply the current
// one's fragments (coopMatMulAdd), so global memory latency overlaps the
// products. The accumulators go through shared memory, one fragment per
// subgroup at a time, to be scaled, added to Cin and stored with bounds
// checks.
//
// Shared memory: the A and B blocks with 8 values of padding per row (rows
// stay 16-byte aligned, as cooperative-matrix loads need, and start in
// different banks), plus one float32 fragment per subgroup: 38 KiB
// (vkcore requires that much for this kernel).

#include "_slot.slang"
#include "_half.slang"

using namespace linalg;

struct Params
{
    T* a;
    T* b;
    T* c;
    T* cin;
    slot64 alpha;    // float bits (csrc/linalg.cpp packs them as float32's)
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

[vk::constant_id(0)] const int USE_CIN = 0;
// How A and B are staged: 0 element by element (any strides), 1 contiguous
// along k (A) or n (B), 2 contiguous along m (A) or k (B); 1 and 2 need
// 16-byte aligned pieces (csrc/linalg.cpp's coop_mode).
[vk::constant_id(1)] const int A_MODE = 0;
[vk::constant_id(2)] const int B_MODE = 0;

// Tile shape (csrc/linalg.cpp mirrors BM and BN). Measured at 2048^3 to
// 8192^3 (TFLOP/s, H100 NVL / L40S): 128 x 256 x 32 with 4 x 4 fragments
// per subgroup 145-149 / 147-157; 256 x 128 140-146 / 143-150; 128 x 128
// 112-118 / 123-138, with BK 64 123-126 (H100); 64 x 128 100-109 (H100).
#define LANES 32
#define BM 128
#define BN 256
#define BK 32
#define WM 4
#define WN 4
// subgroups in a grid of BM / (16 WM) rows and BN / (16 WN) columns, each
// WM x WN fragments of 16 x 16
#define SG_COLS (BN / (16 * WN))
#define WG (LANES * (BM / (16 * WM)) * SG_COLS)
// Shared rows in 16-byte units (8 values, plus 8 of padding).
#define RK ((BK + 8) / 8)  // rows along k
#define RM ((BM + 8) / 8)  // rows along m
#define RN ((BN + 8) / 8)  // rows along n
#define PA (BM * BK / 8 / WG)  // 16-byte pieces per thread: A
#define PB (BN * BK / 8 / WG)  // and B
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typealias FragA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias FragB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias FragC = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

groupshared uint4 As[MAX(BM * RK, BK * RM)];  // [m][k] (A_MODE 0, 1) or [k][m] (A_MODE 2)
groupshared uint4 Bs[MAX(BK * RN, BN * RK)];  // [k][n] (B_MODE 0, 1) or [n][k] (B_MODE 2)
groupshared float Cs[WG / LANES * 256];

uint pack2(half lo, half hi)
{
    return uint(bit_cast<uint16_t>(lo)) | (uint(bit_cast<uint16_t>(hi)) << 16);
}

// Piece v (< BM BK / 8) of an operand block in its [outer][inner] shared
// layout: the outer index and the inner offset (a multiple of 8) of its
// first value; the piece's 8 values run along the inner dim.
void piece_at(uint v, uint per_row, out uint outer, out uint inner)
{
    outer = v / per_row;
    inner = (v % per_row) * 8;
}

// The 8 values of A from (m0, k0) along k (k_inner) or along m, zeros past
// the edges; one 16-byte load when `vec`.
uint4 load_a(int base, uint gm, uint gk, bool k_inner, bool vec)
{
    if (vec && gm < p.m && gk < p.k && (k_inner ? gk + 8 <= p.k : gm + 8 <= p.m))
    {
        uint4* q = (uint4*)(p.a + (base + int(gm) * p.a_sm + int(gk) * p.a_sk));
        return loadAligned<16>(q);
    }
    half h[8];
    [ForceUnroll]
    for (uint i = 0; i < 8; i++)
    {
        uint r = k_inner ? gm : gm + i, c = k_inner ? gk + i : gk;
        half v = p.a[base + int(min(r, p.m - 1)) * p.a_sm + int(min(c, p.k - 1)) * p.a_sk];
        h[i] = r < p.m && c < p.k ? v : half(0);
    }
    return uint4(pack2(h[0], h[1]), pack2(h[2], h[3]), pack2(h[4], h[5]), pack2(h[6], h[7]));
}

// The 8 values of B from (k0, n0) along n (n_inner) or along k.
uint4 load_b(int base, uint gk, uint gn, bool n_inner, bool vec)
{
    if (vec && gk < p.k && gn < p.n && (n_inner ? gn + 8 <= p.n : gk + 8 <= p.k))
    {
        uint4* q = (uint4*)(p.b + (base + int(gk) * p.b_sk + int(gn) * p.b_sn));
        return loadAligned<16>(q);
    }
    half h[8];
    [ForceUnroll]
    for (uint i = 0; i < 8; i++)
    {
        uint r = n_inner ? gk : gk + i, c = n_inner ? gn + i : gn;
        half v = p.b[base + int(min(r, p.k - 1)) * p.b_sk + int(min(c, p.n - 1)) * p.b_sn];
        h[i] = r < p.k && c < p.n ? v : half(0);
    }
    return uint4(pack2(h[0], h[1]), pack2(h[2], h[3]), pack2(h[4], h[5]), pack2(h[6], h[7]));
}

// A's piece (outer o, inner offset ii) of the block at (row0, k0): [m][k]
// (pieces along k) unless A_MODE 2 ([k][m], pieces along m).
uint4 load_a_piece(int base, uint row0, uint k0, uint o, uint ii)
{
    if (A_MODE != 2)
        return load_a(base, row0 + o, k0 + ii, true, A_MODE == 1);
    return load_a(base, row0 + ii, k0 + o, false, true);
}

// B's piece of the block at (k0, col0): [k][n] (pieces along n) unless
// B_MODE 2 ([n][k], pieces along k).
uint4 load_b_piece(int base, uint k0, uint col0, uint o, uint ii)
{
    if (B_MODE != 2)
        return load_b(base, k0 + o, col0 + ii, true, B_MODE == 1);
    return load_b(base, k0 + ii, col0 + o, false, true);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint t = tid.x, sg = t / LANES, lane = t % LANES;
    uint row0 = gid.y * BM, col0 = gid.x * BN;
    int batch = int(gid.z);
    int abase = batch * p.a_sb, bbase = batch * p.b_sb;
    uint wr = (sg / SG_COLS) * WM * 16, wc = (sg % SG_COLS) * WN * 16;  // the subgroup's block of the tile
    // shared layouts: A [m][k] unless A_MODE 2 ([k][m]); B [k][n] unless
    // B_MODE 2 ([n][k]); pieces of 8 values along the inner dim
    uint a_row = A_MODE != 2 ? BK / 8 : BM / 8, b_row = B_MODE != 2 ? BN / 8 : BK / 8;  // pieces per row
    uint4 ra[PA], rb[PB];

    FragC acc[WM][WN];
    [ForceUnroll]
    for (uint i = 0; i < WM; i++)
        [ForceUnroll]
        for (uint j = 0; j < WN; j++)
            acc[i][j] = FragC(0.0);

    // the first block's pieces
    [ForceUnroll]
    for (uint s = 0; s < PA; s++)
    {
        uint o, ii;
        piece_at(t + s * WG, a_row, o, ii);
        ra[s] = load_a_piece(abase, row0, 0, o, ii);
    }
    [ForceUnroll]
    for (uint s = 0; s < PB; s++)
    {
        uint o, ii;
        piece_at(t + s * WG, b_row, o, ii);
        rb[s] = load_b_piece(bbase, 0, col0, o, ii);
    }

    for (uint k0 = 0; k0 < p.k; k0 += BK)
    {
        [ForceUnroll]
        for (uint s = 0; s < PA; s++)
        {
            uint o, ii;
            piece_at(t + s * WG, a_row, o, ii);
            As[o * (A_MODE != 2 ? RK : RM) + ii / 8] = ra[s];
        }
        [ForceUnroll]
        for (uint s = 0; s < PB; s++)
        {
            uint o, ii;
            piece_at(t + s * WG, b_row, o, ii);
            Bs[o * (B_MODE != 2 ? RN : RK) + ii / 8] = rb[s];
        }
        GroupMemoryBarrierWithGroupSync();
        uint k1 = k0 + BK;
        if (k1 < p.k)
        {
            // the next block's pieces, in flight during the products
            [ForceUnroll]
            for (uint s = 0; s < PA; s++)
            {
                uint o, ii;
                piece_at(t + s * WG, a_row, o, ii);
                ra[s] = load_a_piece(abase, row0, k1, o, ii);
            }
            [ForceUnroll]
            for (uint s = 0; s < PB; s++)
            {
                uint o, ii;
                piece_at(t + s * WG, b_row, o, ii);
                rb[s] = load_b_piece(bbase, k1, col0, o, ii);
            }
        }
        [ForceUnroll]
        for (uint kk = 0; kk < BK; kk += 16)
        {
            FragA fa[WM];
            FragB fb[WN];
            [ForceUnroll]
            for (uint i = 0; i < WM; i++)
            {
                uint r = wr + i * 16;
                if (A_MODE != 2)
                    fa[i] = FragA.Load<CoopMatMatrixLayout.RowMajor>(As, r * RK + kk / 8, RK);
                else
                    fa[i] = FragA.Load<CoopMatMatrixLayout.ColumnMajor>(As, kk * RM + r / 8, RM);
            }
            [ForceUnroll]
            for (uint j = 0; j < WN; j++)
            {
                uint c = wc + j * 16;
                if (B_MODE != 2)
                    fb[j] = FragB.Load<CoopMatMatrixLayout.RowMajor>(Bs, kk * RN + c / 8, RN);
                else
                    fb[j] = FragB.Load<CoopMatMatrixLayout.ColumnMajor>(Bs, c * RK + kk / 8, RK);
            }
            [ForceUnroll]
            for (uint i = 0; i < WM; i++)
                [ForceUnroll]
                for (uint j = 0; j < WN; j++)
                    acc[i][j] = coopMatMulAdd<float, false>(fa[i], fb[j], acc[i][j]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float alpha = slot_float(p.alpha), beta = slot_float(p.beta);
    [ForceUnroll]
    for (uint i = 0; i < WM; i++)
    {
        [ForceUnroll]
        for (uint j = 0; j < WN; j++)
        {
            // (each subgroup its own fragment of Cs: no workgroup barrier)
            acc[i][j].Store<CoopMatMatrixLayout.RowMajor>(Cs, sg * 256, 16);
            GroupMemoryBarrierWithWaveSync();
            for (uint e = lane; e < 256; e += LANES)
            {
                uint gm = row0 + wr + i * 16 + e / 16, gn = col0 + wc + j * 16 + e % 16;
                if (gm < p.m && gn < p.n)
                {
                    float v = alpha * Cs[sg * 256 + e];
                    if (USE_CIN != 0)  // beta == 0 must ignore Cin entirely (it may hold NaN)
                        v += beta * float(p.cin[batch * p.cin_sb + int(gm) * p.cin_sm + int(gn) * p.cin_sn]);
                    p.c[batch * p.c_sb + int(gm) * p.c_sm + int(gn) * p.c_sn] = to_half(v);
                }
            }
            GroupMemoryBarrierWithWaveSync();
        }
    }
}
