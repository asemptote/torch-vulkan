// axis T: c64=float2 c128=double2
// axis V: e16=uint e8=uint
//
// One radix-R Stockham pass of a power-of-two FFT, R = 128 to 1024, with the
// values in registers: csrc/fft.cpp's fft_pow2 takes longer rows in two or
// three such passes (fft_radix.slang's, r <= 128, need one more). The pass is
// fft_radix.slang's; for each group j < n/R (k = j % ns):
//
//     u[q] = in[j + q n/R] * w^(k q (n / (R ns))),  q < R
//     out[(j / ns) R ns + k + q ns] = sum_q' u[q'] W_R^(q' q)
//
// w = exp(-+2 pi i / n) from the two-level `table` (fft_radix.slang's), W_R^t
// from `rtab` (R entries); both computed on the host in float64. INVERSE
// conjugates every twiddle.
//
// A workgroup takes g = WG E / R consecutive groups (columns), TC = R / E
// threads per column, E values per thread (axis V: 16 or 8). Each thread
// loads values q = i + a TC (a < E) of its column into registers
// (consecutive threads take consecutive columns: runs of g values),
// multiplies by the twiddles and takes their E-point DFT. Radix-rho
// Stockham stages follow through shared memory (rho = E while it fits, then
// R / m: E / rho butterflies per thread), and the last stage's results go
// to global memory directly. The DFTs of 8 and 16 points are split in
// registers (2 x 4 and 4 x 4).
//
// Shared memory: WG E values (32 KiB, or 16 KiB for complex64 with E = 8);
// the host checks that the device has it.
//
// EPILOGUE = 1 (the last pass of a real FFT's half-length transform, rfft:
// csrc/fft.cpp's rfft_last_even): the output is the onesided spectrum of
// the real row of 2n values whose complex pairs this row is, as fft.slang's
// FFT_R2C computes it from the complex FFT Z:
//     X[k] = (A + B) / 2 + w^k (A - B) / (2i),  A = Z[k], B = conj(Z[n - k]),
// w = exp(-2 pi i / 2n) (`etab`, the two-level table for 2n), k <= n, rows
// of n + 1. Z[k + q ns]'s mirror Z[n - k - q ns] is in column ns - k (the
// last pass has ns = n / R), so a workgroup takes g / 2 columns k and their
// mirrors ns - k (columns 0 and ns / 2 are their own: their second copy is
// computed, not stored), its last stage writes shared memory, and the
// epilogue reads each value and its mirror from there. (The pairing halves
// the contiguous runs of the pass's loads and stores: g / 2 columns
// ascending and g / 2 descending.)
//
// EPILOGUE = 2 (the first pass of an inverse real FFT's half-length
// transform, irfft: csrc/fft.cpp's fft_c2r_compute): the input rows are the
// onesided spectrum X of the real row of 2n values (rows of `pad` >= n + 1
// values), folded on load as fft.slang's FFT_C2R folds it:
//     Z'[k] = (A + B) + i conj(w^k) (A - B),  A = X[k], B = conj(X[n - k]),
// with X[0]'s and X[n]'s imaginary parts dropped. In the first pass (ns = 1)
// Z'[j + q nr]'s mirror is at column nr - j, element R - 1 - q, so the
// columns are paired as for EPILOGUE = 1: a workgroup loads both columns'
// X into shared memory, and each thread folds its values from there.

#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif
#include "_complex.slang"

#if defined(V_e8)
#define E 8
#define WG 256
#else
#define E 16
#if defined(T_c64)
#define WG 256
#else
#define WG 128
#endif
#endif
#define BUF (WG * E)
// Stage twiddles loaded once before the first block (PRE), or at each
// stage's start, before its barrier (complex128: the registers they take)
#if defined(T_c128)
#define PRE 0
#else
#define PRE 1
#endif

struct Params
{
    T* in;
    T* out;
    T* table;  // two-level w^e (fft_radix.slang)
    T* rtab;   // W_R^t, t < R
    uint n;
    uint ns;
    uint batch;
    uint r;
    uint ngroups;
    uint pad;
    T* etab;   // EPILOGUE: the two-level table for 2n
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int INVERSE = 0;
[vk::constant_id(1)] const int EPILOGUE = 0;

groupshared T buf[BUF];

// Element (row pos, column c) of the g columns in shared memory, rows
// XOR-swizzled within blocks of E: the first stage writes rows E apart
// (pos = i E + b), which without it put every thread of a phase on one bank;
// later stages access runs of consecutive rows, which the swizzle only
// permutes within their block.
uint slot(uint pos, uint lg, uint c)
{
    return ((pos ^ ((pos >> firstbithigh(uint(E))) & (E - 1))) << lg) + c;
}

T fwd(T w)
{
    return INVERSE != 0 ? cconj(w) : w;
}

// w^e for e < n, from the two-level table.
T tw_n(uint e)
{
    uint ln = firstbithigh(p.n), L = (ln + 1) >> 1;
    return fwd(cmul(p.table[e & ((1u << L) - 1)], p.table[(1u << L) + min(e >> L, (p.n >> L) - 1)]));
}

// exp(-2 pi i e / 2n) for e <= n, from the two-level table for 2n (EPILOGUE).
T tw_2n(uint e)
{
    uint ln2 = firstbithigh(p.n) + 1, L = (ln2 + 1) >> 1;
    return cmul(p.etab[e & ((1u << L) - 1)], p.etab[(1u << L) + min(e >> L, ((2 * p.n) >> L) - 1)]);
}

// X[k] = (A + B) / 2 + w (A - B) / (2i) (EPILOGUE; fft.slang's FFT_R2C).
T r2c_point(T a, T c, T w)
{
    T e = (a + c) * S(0.5), d = (a - c) * S(0.5);
    return e + cmul(w, T(d.y, -d.x));
}

// W_R^e.
T tw_r(uint e)
{
    return fwd(p.rtab[e & (p.r - 1)]);
}

// z W_4 = -i z forward, i z inverse.
T times_w4(T z)
{
    return INVERSE != 0 ? T(-z.y, z.x) : T(z.y, -z.x);
}

#define KC1 S(0.92387953251128675613l)
#define KS1 S(0.38268343236508977173l)
#define KC2 S(0.70710678118654752440l)

void bfly2(inout T a, inout T b)
{
    T d = a - b;
    a = a + b;
    b = d;
}

void bfly4(inout T a, inout T b, inout T c, inout T d)
{
    T a0 = a + c, a1 = a - c, b0 = b + d, b1 = times_w4(b - d);
    a = a0 + b0;
    b = a1 + b1;
    c = a0 - b0;
    d = a1 - b1;
}

// In-place DFTs of v[o .. o + rho), inputs in natural order; output X[k] is
// left at v[o + at(rho, k)].
uint at(uint rho, uint k)
{
    if (rho == 16)
        return 4 * (k & 3) + (k >> 2);  // 4 x 4: X[k1 + 4 k2] at 4 k1 + k2
    if (rho == 8)
        return 4 * (k & 1) + (k >> 1);  // 2 x 4: X[k1 + 2 k2] at 4 k1 + k2
    return k;
}

void dft8(inout T v[E], uint o)
{
    // X[k1 + 2 k2] = sum_n2 W4^(n2 k2) W8^(n2 k1) sum_n1 x[4 n1 + n2] W2^(n1 k1)
    [ForceUnroll]
    for (uint n2 = 0; n2 < 4; n2++)
        bfly2(v[o + n2], v[o + n2 + 4]);
    v[o + 5] = cmul(v[o + 5], fwd(T(KC2, -KC2)));   // W8^1
    v[o + 6] = times_w4(v[o + 6]);                // W8^2
    v[o + 7] = cmul(v[o + 7], fwd(T(-KC2, -KC2)));  // W8^3
    bfly4(v[o], v[o + 1], v[o + 2], v[o + 3]);
    bfly4(v[o + 4], v[o + 5], v[o + 6], v[o + 7]);
}

#if E == 16
void dft16(inout T v[E], uint o)
{
    // X[k1 + 4 k2] = sum_n2 W4^(n2 k2) W16^(n2 k1) sum_n1 x[4 n1 + n2] W4^(n1 k1)
    [ForceUnroll]
    for (uint n2 = 0; n2 < 4; n2++)
        bfly4(v[o + n2], v[o + n2 + 4], v[o + n2 + 8], v[o + n2 + 12]);
    v[o + 5] = cmul(v[o + 5], fwd(T(KC1, -KS1)));    // W16^1
    v[o + 9] = cmul(v[o + 9], fwd(T(KC2, -KC2)));    // W16^2
    v[o + 13] = cmul(v[o + 13], fwd(T(KS1, -KC1)));  // W16^3
    v[o + 6] = cmul(v[o + 6], fwd(T(KC2, -KC2)));    // W16^2
    v[o + 10] = times_w4(v[o + 10]);               // W16^4
    v[o + 14] = cmul(v[o + 14], fwd(T(-KC2, -KC2))); // W16^6
    v[o + 7] = cmul(v[o + 7], fwd(T(KS1, -KC1)));    // W16^3
    v[o + 11] = cmul(v[o + 11], fwd(T(-KC2, -KC2))); // W16^6
    v[o + 15] = cmul(v[o + 15], fwd(T(-KC1, KS1)));  // W16^9
    [ForceUnroll]
    for (uint k1 = 0; k1 < 4; k1++)
        bfly4(v[o + 4 * k1], v[o + 4 * k1 + 1], v[o + 4 * k1 + 2], v[o + 4 * k1 + 3]);
}
#endif

// The RHO-point DFTs of v's E / RHO slices.
void dfts<let RHO : uint>(inout T v[E])
{
    [ForceUnroll]
    for (uint o = 0; o < E; o += RHO)
    {
        if (RHO == 2)
            bfly2(v[o], v[o + 1]);
        else if (RHO == 4)
            bfly4(v[o], v[o + 1], v[o + 2], v[o + 3]);
        else if (RHO == 8)
            dft8(v, o);
#if E == 16
        else
            dft16(v, o);
#endif
    }
}

uint log2c(uint x)
{
    return x >= 16 ? 4 : x >= 8 ? 3 : x >= 4 ? 2 : 1;
}

// Per-block constants for the stages.
struct Block
{
    uint i, c, lg, lT, lr, lns, rowout;
    bool store;  // (this thread's group exists: a partial last block repeats the last one)
};

// A radix-RHO Stockham stage over m (= 2^lm) through shared memory: E / RHO
// butterflies per thread; with LAST, the results go to global memory.
// Stage twiddles for this thread's values (W_{RHO m}^(a kk), 1 where a or kk
// is 0): the same for every block, so loaded once, before the first block's
// data, and in flight together with it.
void stage_tw<let RHO : uint>(uint i, uint lT, uint lr, uint m, uint lm, out T tw[E])
{
    uint lrho = log2c(RHO);
    [ForceUnroll]
    for (uint u = 0; u < E; u++)
    {
        uint ib = i + ((u / RHO) << lT), a = u % RHO, kk = ib & (m - 1);
        tw[u] = tw_r((a * kk) << (lr - lrho - lm));
    }
}

void stage<let RHO : uint>(Block k, uint m, uint lm, inout T v[E], T tw[E], bool last)
{
    uint lrho = log2c(RHO);
    uint lcnt = k.lr - lrho;  // butterflies per column, R / RHO
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint u = 0; u < E; u++)
    {
        // value u: butterfly i + (u / RHO) TC, input a = u % RHO
        uint ib = k.i + ((u / RHO) << k.lT), a = u % RHO, kk = ib & (m - 1);
        T x = buf[slot(ib + (a << lcnt), k.lg, k.c)];
        v[u] = a == 0 ? x : cmul(x, tw[u]);
    }
    dfts<RHO>(v);
    if (!last || EPILOGUE == 1)  // (uniform)
        GroupMemoryBarrierWithGroupSync();
    [ForceUnroll]
    for (uint u = 0; u < E; u++)
    {
        uint ib = k.i + ((u / RHO) << k.lT), b = u % RHO, kk = ib & (m - 1);
        uint pos = ((ib >> lm) << (lm + lrho)) + kk + (b << lm);
        T x = v[(u / RHO) * RHO + at(RHO, b)];
        if (last && EPILOGUE != 1)
        {
            if (k.store)
                p.out[k.rowout + (pos << k.lns)] = x;
        }
        else
            buf[slot(pos, k.lg, k.c)] = x;  // (EPILOGUE: read back by the epilogue)
    }
}

// A pass whose N stages after the first are radix S1, S2, S3 (unused ones 2,
// never run).
void run<let N : uint, let S1 : uint, let S2 : uint, let S3 : uint>(uint t, uint gx)
{
    uint R = p.r, lr = firstbithigh(R);
    uint lg = firstbithigh(BUF) - lr, g = 1u << lg;  // columns per workgroup
    uint lE = firstbithigh(E), lT = lr - lE;          // TC = 2^lT threads per column
    uint nr = p.n >> lr, lnr = firstbithigh(nr), lns = firstbithigh(p.ns), ln = firstbithigh(p.n);
    uint lstep = ln - lr - lns;  // the outer twiddle's exponent unit, n / (R ns)
    uint groups = p.batch << lnr, blocks = (groups + g - 1) >> lg;
    uint c = t & (g - 1), i = t >> lg;  // this thread's column and index in it
    // EPILOGUE: pairs of mirrored columns (k, nr - k), k <= nr / 2, g / 2 a block
    uint hg = g >> 1, pairs = (nr >> 1) + 1, npairs = p.batch * pairs;
    if (EPILOGUE != 0)
        blocks = (npairs + hg - 1) / hg;
    uint m1 = E, m2 = E * S1, m3 = E * S1 * S2;
    T tw1[E], tw2[E], tw3[E];
    if (PRE != 0)
    {
    if (N >= 1)
        stage_tw<S1>(i, lT, lr, m1, firstbithigh(m1), tw1);
    if (N >= 2)
        stage_tw<S2>(i, lT, lr, m2, firstbithigh(m2), tw2);
    if (N >= 3)
        stage_tw<S3>(i, lT, lr, m3, firstbithigh(m3), tw3);
    }
    for (uint w = gx; w < blocks; w += p.ngroups)  // (uniform per workgroup)
    {
        uint G = (w << lg) + c, Gc = min(G, groups - 1);  // (a partial last block repeats the last group)
        if (EPILOGUE != 0)
        {
            uint pi = w * hg + (c & (hg - 1)), pic = min(pi, npairs - 1);
            uint b = pic / pairs, kk = pic - b * pairs;
            Gc = (b << lnr) + (c < hg ? kk : (nr - kk) & (nr - 1));
            // (a self-mirrored column's second copy is computed, not stored)
            G = pi < npairs && (EPILOGUE == 1 || c < hg || (Gc & (nr - 1)) != kk) ? Gc : groups;
        }
        uint j = Gc & (nr - 1);
        uint rowin = (Gc >> lnr) << ln;
        uint rowout = rowin + ((j >> lns) << (lr + lns)) + (j & (p.ns - 1));
        T v[E];
        if (EPILOGUE == 2)
        {
            // both columns of each pair into shared memory, then each value folded with its mirror
            uint rowx = (Gc >> lnr) * p.pad;  // (this thread's row of X, rows of pad)
            [ForceUnroll]
            for (uint s = 0; s < E; s++)
            {
                uint e = t + s * WG, cs = e & (g - 1), pos = e >> lg;
                uint pi = min(w * hg + (cs & (hg - 1)), npairs - 1);
                uint b = pi / pairs, kk = pi - b * pairs;
                uint jj = cs < hg ? kk : (nr - kk) & (nr - 1);
                buf[slot(pos, lg, cs)] = p.in[b * p.pad + jj + (pos << lnr)];
            }
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint a = 0; a < E; a++)
            {
                uint q = i + (a << lT), idx = j + (q << lnr);
                T xa = buf[slot(q, lg, c)];
                T xb = j != 0 ? buf[slot(R - 1 - q, lg, c ^ hg)] : q != 0 ? buf[slot(R - q, lg, c)] : p.in[rowx + p.n];
                if (idx == 0)
                {
                    xa.y = S(0);  // (X[0] and X[n] count as real, as the full-length inverse's real part has them)
                    xb.y = S(0);
                }
                T bb = cconj(xb);
                T d = cmul(cconj(tw_2n(idx)), xa - bb);
                v[a] = (xa + bb) + T(-d.y, d.x);
            }
            GroupMemoryBarrierWithGroupSync();  // (the stages write buf next)
        }
        else
        {
        // the first stage: radix E from global memory (m = 1, a butterfly per thread)
        [ForceUnroll]
        for (uint a = 0; a < E; a++)
        {
            uint q = i + (a << lT);
            T x = p.in[rowin + j + (q << lnr)];
            v[a] = p.ns == 1 || q == 0 ? x : cmul(x, tw_n(((j & (p.ns - 1)) * q) << lstep));
        }
        }
        dfts<E>(v);
        Block k = { i, c, lg, lT, lr, lns, rowout, G < groups };
        [ForceUnroll]
        for (uint b = 0; b < E; b++)
        {
            uint pos = (i << lE) + b;  // ((i / 1) E 1) + 0 + b
            T x = v[at(E, b)];
            if (N == 0)
            {
                if (k.store)
                    p.out[rowout + (pos << lns)] = x;
            }
            else
                buf[slot(pos, lg, c)] = x;
        }
        if (N >= 1)
        {
            if (PRE == 0)
                stage_tw<S1>(i, lT, lr, m1, firstbithigh(m1), tw1);
            stage<S1>(k, m1, firstbithigh(m1), v, tw1, N == 1);
        }
        if (N >= 2)
        {
            if (PRE == 0)
                stage_tw<S2>(i, lT, lr, m2, firstbithigh(m2), tw1);
            stage<S2>(k, m2, firstbithigh(m2), v, PRE != 0 ? tw2 : tw1, N == 2);
        }
        if (N >= 3)
        {
            if (PRE == 0)
                stage_tw<S3>(i, lT, lr, m3, firstbithigh(m3), tw1);
            stage<S3>(k, m3, firstbithigh(m3), v, PRE != 0 ? tw3 : tw1, true);
        }
        if (EPILOGUE == 1)
        {
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint s = 0; s < E; s++)
            {
                // element (column slot cs, frequency pos) and its mirror's slot
                uint e = t + s * WG, cs = e & (g - 1), pos = e >> lg;
                uint pi = w * hg + (cs & (hg - 1));
                uint b = min(pi, npairs - 1) / pairs, kk = min(pi, npairs - 1) - b * pairs;
                uint jj = cs < hg ? kk : (nr - kk) & (nr - 1);
                if (pi >= npairs || (cs >= hg && jj == kk))
                    continue;  // (past the end, or a self-mirrored column's copy)
                uint posm = jj == 0 ? (R - pos) & (R - 1) : R - 1 - pos;
                T z = buf[slot(pos, lg, cs)], zm = buf[slot(posm, lg, cs ^ hg)];
                uint idx = jj + (pos << lnr), row = b * (p.n + 1);
                p.out[row + idx] = r2c_point(z, cconj(zm), tw_2n(idx));
                if (idx == 0)
                    p.out[row + p.n] = r2c_point(z, cconj(zm), T(S(-1), S(0)));
            }
        }
        GroupMemoryBarrierWithGroupSync();  // (buf is reused by the next block)
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    // the stages after the first: radix E while it fits, then R / m
    uint R = p.r;
#if E == 16
    if (R == 128)
        run<1, 8, 2, 2>(tid.x, gid.x);
    else if (R == 256)
        run<1, 16, 2, 2>(tid.x, gid.x);
    else if (R == 512)
        run<2, 16, 2, 2>(tid.x, gid.x);
    else
        run<2, 16, 4, 2>(tid.x, gid.x);
#else
    if (R == 128)
        run<2, 8, 2, 2>(tid.x, gid.x);
    else if (R == 256)
        run<2, 8, 4, 2>(tid.x, gid.x);
    else if (R == 512)
        run<2, 8, 8, 2>(tid.x, gid.x);
    else
        run<3, 8, 8, 2>(tid.x, gid.x);
#endif
}
