// axis T: f32=float f64=double c64=float2 c128=double2
//
// The panel of csrc/dense.cpp's blocked LU (LAPACK's getrf, partial
// pivoting) over many workgroups: a column per dispatch. factor.slang's
// LU_PANEL factors a panel on one workgroup per matrix, a column after
// another, and at n = 4096 that was nearly all of linalg.solve's time.
//
// Within a panel (columns [k, k + b), rows [k, rows)) the rows stay where
// they are: a row's position (where LAPACK's interchanges would have put it)
// is kept in pos, double-buffered by the step's parity (as are the
// workgroups' pivot candidates: a step reads one set and writes the next). At step j the pivot
// is the row of largest |a[., j]| (|re| + |im| for complex, izamax's) among
// the rows at positions >= j, the lowest position on ties, as LAPACK picks
// it; it moves to position j and the row there to the pivot's old position.
// The other rows at positions > j are eliminated against it, within the
// panel's columns. SWAP then applies the panel's interchanges to whole rows,
// which leaves the matrix as LAPACK's getf2 leaves it.
//
// Matrices are row-major and batched (matrix z at a + z * sa, entry (i, j)
// at [i * ld + j]); pivots (1-based, absolute) at piv + z * spiv, the first
// zero pivot (1-based) into info[z]. With UNIT, no pivoting (pivots j + 1).
//
// OP (GETRF_*), grid (groups, batch) with RW rows per workgroup:
//   START  positions = rows, and per workgroup the best candidate of column k
//          over its rows (pval, pidx: value, then position and row)
//   STEP   step j: every workgroup reduces the candidates (the same order in
//          each) to the pivot; workgroup 0 records it. Rows at positions
//          > j: l = a[r, j] / pivot, a[r, j + 1:k + b] -= l pivot_row (none
//          for a zero pivot, as LAPACK leaves that column unscaled); new
//          positions; candidates for column j + 1.
//   SWAP   the interchanges of piv[k:k + b] in order on every column, a
//          thread per column; grid (columns / WG, batch).

#include "opcodes.slang"

#if defined(T_c64) || defined(T_c128)
#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif
#include "_complex.slang"
typedef S R;
T mul(T a, T b) { return cmul(a, b); }
T dv(T a, T b) { return cdiv(a, b); }
R mag1(T a) { return abs(a.x) + abs(a.y); }  // izamax's |re| + |im|
T fromr(R r) { return T(r, R(0)); }
#else
typedef T R;
T mul(T a, T b) { return a * b; }
T dv(T a, T b) { return a / b; }
R mag1(T a) { return abs(a); }
T fromr(R r) { return r; }
#endif
#define ZERO fromr(R(0))

struct Params
{
    T* a;
    int* piv;
    int* info;
    R* pval;   // candidates: [batch, 2, groups] (by the step's parity, as pos)
    int* pidx; // their position and row: [batch, 2, groups, 2]
    int* pos;  // [batch, 2, rows]
    uint rows;
    uint cols;
    uint ld;
    uint sa;
    uint k;
    uint b;
    uint j;
    uint spiv;
    uint groups;
    uint pad;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = GETRF_STEP;
[vk::constant_id(1)]
const int UNIT = 0;

#define WG 256
#define LANES 8           // threads per row in STEP
#define RW (WG / LANES)   // rows per workgroup (kGetrfRows in csrc/dense.cpp)
#define NONE 0x7fffffff

#define MAXC 8          // columns per lane in STEP: panels of at most LANES * MAXC = 64 (kBlock)

groupshared R sv[WG];
groupshared int sp[WG];
groupshared int sr[WG];

// (shared memory only: no thread reads another's global writes within a dispatch)
void barrier()
{
    GroupMemoryBarrierWithGroupSync();
}

// Whether candidate (v, pos) beats (bv, bpos): larger, then lower position;
// NaN beats any number.
bool better(R v, int pos, R bv, int bpos)
{
    if (isnan(v))
        return !isnan(bv) || pos < bpos;
    if (isnan(bv))
        return false;
    return v > bv || (v == bv && pos < bpos);
}

// The best of the candidates in sv, sp, sr [0, width) (width a power of
// two, the same in every thread) into entry 0.
void reduce_best(uint t, uint width)
{
    barrier();
    for (uint s = width / 2; s > 0; s >>= 1)
    {
        if (t < s && better(sv[t + s], sp[t + s], sv[t], sp[t]))
        {
            sv[t] = sv[t + s];
            sp[t] = sp[t + s];
            sr[t] = sr[t + s];
        }
        barrier();
    }
}

void put_candidate(uint t, uint gx, uint z, uint par)
{
    if (t == 0)
    {
        uint g = (z * 2 + par) * p.groups + gx;
        p.pval[g] = sv[0];
        p.pidx[2 * g] = sp[0];
        p.pidx[2 * g + 1] = sr[0];
    }
}

uint pow2_at_least(uint x)
{
    uint w = 1;
    while (w < x)
        w <<= 1;
    return w;
}

void start(uint t, uint gx, uint z)
{
    T* a = p.a + z * p.sa;
    uint m = p.rows, k = p.k;
    uint r = k + gx * RW + t;
    uint rr = min(r, m - 1);
    if (t < RW)
    {
        R v = R(-1);
        int pos = NONE;
        if (r < m)
        {
            p.pos[(z * 2) * m + r] = int(r);
            v = mag1(a[rr * p.ld + k]);
            pos = int(r);
        }
        sv[t] = v;
        sp[t] = pos;
        sr[t] = int(rr);
    }
    reduce_best(t, RW);
    put_candidate(t, gx, z, 0);
}

void step(uint t, uint gx, uint z)
{
    T* a = p.a + z * p.sa;
    uint m = p.rows, ld = p.ld, k = p.k, j = p.j, cend = k + p.b;
    uint par = (j - k) & 1u;
    int* cur = p.pos + (z * 2 + par) * m;
    int* nxt = p.pos + (z * 2 + 1 - par) * m;
    // this thread's part of its row, loaded first (it does not depend on
    // the pivot): lane `lane` has columns j + 1 + lane + LANES i
    uint rl = t / LANES, lane = t % LANES;
    uint r = k + gx * RW + rl;
    bool valid = r < m;
    uint rr = min(r, m - 1);
    int pc = cur[rr];
    T aj = a[rr * ld + j];
    T x[MAXC];
    [ForceUnroll]
    for (uint i = 0; i < MAXC; i++)
        x[i] = a[rr * ld + min(j + 1 + lane + LANES * i, cend - 1)];
    // the pivot: the best of the workgroups' candidates
    uint groups = (m - k + RW - 1) / RW;
    {
        R bv = R(-1);
        int bpos = NONE, brow = 0;
        for (uint g = t; g < groups; g += WG)
        {
            uint at = (z * 2 + par) * p.groups + min(g, groups - 1);
            R v = p.pval[at];
            int pos = p.pidx[2 * at];
            if (better(v, pos, bv, bpos))
            {
                bv = v;
                bpos = pos;
                brow = p.pidx[2 * at + 1];
            }
        }
        sv[t] = bv;
        sp[t] = bpos;
        sr[t] = brow;
    }
    reduce_best(t, min(pow2_at_least(groups), WG));
    uint prow = uint(sr[0]), q = uint(sp[0]);
    if (UNIT != 0)
    {
        prow = j;
        q = j;
    }
    T pivot = a[prow * ld + j];
    bool zero = mag1(pivot) == R(0);
    if (gx == 0 && t == 0)
    {
        p.piv[z * p.spiv + j] = int(q) + 1;
        if (zero && p.info[z] == 0)
            p.info[z] = int(j) + 1;
    }
    // rows at positions >= j other than the pivot's (the row at j moves to
    // q): l = a[r, j] / pivot (every lane of the row alike), then the update
    bool active = valid && pc >= int(j) && rr != prow && !zero;
    T l = active ? dv(aj, pivot) : ZERO;
    int np = rr == prow ? int(j) : (pc == int(j) ? int(q) : pc);
    if (lane == 0 && valid)
    {
        if (active)
            a[rr * ld + j] = l;
        nxt[rr] = np;
    }
    [ForceUnroll]
    for (uint i = 0; i < MAXC; i++)
    {
        uint c = j + 1 + lane + LANES * i;
        if (c < cend && active)
        {
            x[i] = x[i] - mul(l, a[prow * ld + c]);
            a[rr * ld + c] = x[i];
        }
    }
    // candidates for column j + 1 (lane 0 holds it): rows at positions > j
    if (j + 1 < cend)
    {
        barrier();  // (sv: everyone has read the pivot)
        if (lane == 0)
        {
            bool cand = valid && np > int(j);
            sv[rl] = cand ? mag1(x[0]) : R(-1);
            sp[rl] = cand ? np : NONE;
            sr[rl] = int(rr);
        }
        reduce_best(t, RW);
        put_candidate(t, gx, z, 1 - par);
    }
}

void swap_rows(uint t, uint gx, uint z)
{
    T* a = p.a + z * p.sa;
    uint c = gx * WG + t;
    if (c >= p.cols)
        return;
    for (uint i = p.k; i < p.k + p.b; i++)
    {
        uint r = min(uint(p.piv[z * p.spiv + i] - 1), p.rows - 1);
        if (r != i)
        {
            T x = a[i * p.ld + c];
            a[i * p.ld + c] = a[r * p.ld + c];
            a[r * p.ld + c] = x;
        }
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint t = lid.x;
    switch (OP)
    {
    case GETRF_START: start(t, gid.x, gid.y); break;
    case GETRF_STEP: step(t, gid.x, gid.y); break;
    case GETRF_SWAP: swap_rows(t, gid.x, gid.y); break;
    }
}
