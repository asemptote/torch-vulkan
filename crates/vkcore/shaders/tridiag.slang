// axis T: f32=float f64=double c64=float2 c128=double2
//
// Eigenvalues and eigenvectors of real symmetric tridiagonal matrices by
// implicit QL (Numerical Recipes' tqli), for csrc/dense.cpp: eigh's
// tridiagonal below divide and conquer's size, and divide and conquer's
// leaves (stedc.slang). (The reduction to tridiagonal form is latrd.slang.)
//
// Per matrix z (n x n): d and e (real) at z * n, the transposed eigenvector
// matrix Z^T at zt + z * n * n, the rotation log at z * cap (row index) and
// 2 (z * cap) (c, s), state at z * 8, and w[z * n] (QL's scale).
//
// OP (TRIDIAG_*):
//   QL        eigenvalues of the tridiagonal (d, e) by implicit QL with
//             Wilkinson's shift (tqli), d and e in shared memory while it runs
//             (n <= QL_SHARED), iterated by thread 0. With LOG, the
//             plane rotations are appended to the log; a pass stops before an
//             iteration that would overflow it (state: 0 l, 1 iterations of
//             l, 2 rotations logged, 3 done, 4 failed, 5 scaled) and the host
//             runs ROT, then QL again.
//   ROT       Z^T <- the logged rotations applied to rows i, i + 1 of Z^T,
//             one thread per column of Z^T (a row of Z): coalesced, with the
//             row a chain of rotations shares kept in a register. Grid
//             (column groups, batch).

#include "opcodes.slang"

#if defined(T_c64) || defined(T_c128)
#define IS_COMPLEX 1
#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif
#include "_complex.slang"
typedef S R;  // the real type
R re(T a) { return a.x; }
T fromr(R r) { return T(r, R(0)); }
T scale(T a, R s) { return T(a.x * s, a.y * s); }
#else
typedef T R;
R re(T a) { return a; }
T fromr(R r) { return r; }
T scale(T a, R s) { return a * s; }
#endif
#define ZERO fromr(R(0))

struct Params
{
    T* w;
    R* d;
    R* e;
    T* zt;
    int* logi;
    R* logcs;
    int* state;
    uint n;
    uint cap;    // rotations per matrix in the log
    uint groups; // ROT: workgroups along x
    uint pad;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = TRIDIAG_QL;
[vk::constant_id(1)]
const int LOG = 0;

#define WG 256
void barrier()
{
    AllMemoryBarrierWithGroupSync();
}

// The smallest normal R, and the unit roundoff (half an ulp of 1).
R min_normal()
{
#if defined(T_f64) || defined(T_c128)
    return asdouble(0u, 0x00100000u);  // 2^-1022
#else
    return asfloat(0x00800000u);  // 2^-126
#endif
}
R epsilon()
{
#if defined(T_f64) || defined(T_c128)
    return asdouble(0u, 0x3cb00000u);  // 2^-52
#else
    return asfloat(0x34000000u);  // 2^-23
#endif
}

// sqrt(x^2 + y^2) without overflow or harmful underflow (LAPACK's dlapy2).
R hypot2(R x, R y)
{
    R ax = abs(x), ay = abs(y);
    R big = max(ax, ay), small = min(ax, ay);
    if (big == R(0) || small == R(0))
        return big;
    R q = small / big;
    return big * sqrt(R(1) + q * q);
}

// ---------------------------------------------------------------- tridiagonal QL

// d and e during QL: in shared memory when they fit (a single thread's
// dependent loads would otherwise wait on global memory each time), else
// the global arrays. Sized so that the kernel's shared memory stays within
// Vulkan's guaranteed 16 KiB (maxComputeSharedMemorySize).
#if defined(T_f64) || defined(T_c128)
#define QL_SHARED 512
#else
#define QL_SHARED 1024
#endif
groupshared R qd[QL_SHARED];
groupshared R qe[QL_SHARED];
R dget(bool sh, R* g, int i) { if (sh) return qd[i]; return g[i]; }
R eget(bool sh, R* g, int i) { if (sh) return qe[i]; return g[i]; }
void dset(bool sh, R* g, int i, R v) { if (sh) qd[i] = v; else g[i] = v; }
void eset(bool sh, R* g, int i, R v) { if (sh) qe[i] = v; else g[i] = v; }

// Implicit QL with Wilkinson's shift, as Numerical Recipes' tqli (unlike
// EISPACK's tql2, no accumulated shift and so no O(n) update of d per
// iteration), d and e copied to shared memory by the workgroup and iterated
// by thread 0; deflation where |e[m]| <= eps (|d[m]| + |d[m + 1]|).
// e[i] couples d[i] and d[i + 1]; e[n - 1] = 0.
void ql(uint t, uint z)
{
    R* gd = p.d + z * p.n;
    R* ge = p.e + z * p.n;
    int* st = p.state + z * 8;
    if (st[3] != 0)
    {
        if (t == 0)
            st[2] = 0;  // (finished in an earlier pass: ROT must not apply its log again)
        return;
    }
    uint un = p.n;
    bool sh = un <= QL_SHARED;
    if (sh)
    {
        for (uint i = t; i < un; i += WG)
        {
            uint ic = min(i, un - 1);
            qd[i] = gd[ic];
            qe[i] = ge[ic];
        }
    }
    barrier();
    if (t == 0)
    {
        int* logi = p.logi + z * p.cap;
        R* logcs = p.logcs + z * 2 * p.cap;
        int n = int(un), nlog = 0;
        int l = st[0], iter = st[1];
        R eps = epsilon(), tiny = min_normal();
        // T scaled to largest |entry| 1 on the first pass (as LAPACK's
        // dsteqr scales into a safe range), so that r = sqrt(f^2 + g^2)
        // below cannot overflow; eigenvalues are scaled back when done.
        // (The scale is kept in w[0]: the reduction is done with w.)
        T* scale_at = p.w + z * p.n;
        if (st[5] == 0)
        {
            R anorm = R(0);
            for (int i = 0; i < n; i++)
                anorm = max(anorm, max(abs(dget(sh, gd, i)), abs(eget(sh, ge, i))));
            R sc = R(1);
            if (anorm > tiny && !isinf(anorm))
            {
                sc = anorm;
                R inv = R(1) / anorm;
                for (int i = 0; i < n; i++)
                {
                    dset(sh, gd, i, dget(sh, gd, i) * inv);
                    eset(sh, ge, i, eget(sh, ge, i) * inv);
                }
            }
            scale_at[0] = fromr(sc);
            st[5] = 1;
        }
        while (l < n)
        {
            int m = l;
            while (m < n - 1)
            {
                R em = abs(eget(sh, ge, m));
                if (em <= eps * (abs(dget(sh, gd, m)) + abs(dget(sh, gd, m + 1))) || em <= tiny)
                    break;
                m++;
            }
            if (m == l)
            {
                l++;
                iter = 0;
                continue;
            }
            if (LOG != 0 && nlog + (m - l) > int(p.cap))
                break;  // the log is full: the host applies it and calls again
            if (iter >= 60)
            {
                st[4] = 1;  // no convergence (LAPACK's info > 0)
                break;
            }
            iter++;
            R dl = dget(sh, gd, l), el = eget(sh, ge, l);
            R g = (dget(sh, gd, l + 1) - dl) / (R(2) * el);
            R r = hypot2(g, R(1));
            g = dget(sh, gd, m) - dl + el / (g + (g >= R(0) ? r : -r));
            R s = R(1), c = R(1), pp = R(0);
            bool restart = false;
            for (int i = m - 1; i >= l; i--)
            {
                R ei = eget(sh, ge, i);
                R f = s * ei, b = c * ei;
                r = sqrt(f * f + g * g);  // (T scaled: no overflow; underflow restarts below)
                eset(sh, ge, i + 1, r);
                if (r == R(0))
                {
                    // (underflow: deflate here and start again)
                    dset(sh, gd, i + 1, dget(sh, gd, i + 1) - pp);
                    eset(sh, ge, m, R(0));
                    restart = true;
                    break;
                }
                R rinv = R(1) / r;
                s = f * rinv;
                c = g * rinv;
                g = dget(sh, gd, i + 1) - pp;
                r = (dget(sh, gd, i) - g) * s + R(2) * c * b;
                pp = s * r;
                dset(sh, gd, i + 1, g + pp);
                g = c * r - b;
                if (LOG != 0)
                {
                    logi[nlog] = i;
                    logcs[2 * nlog] = c;
                    logcs[2 * nlog + 1] = s;
                    nlog++;
                }
            }
            if (restart)
                continue;
            dset(sh, gd, l, dget(sh, gd, l) - pp);
            eset(sh, ge, l, g);
            eset(sh, ge, m, R(0));
        }
        if (l >= n)
        {
            R sc = re(scale_at[0]);
            for (int i = 0; i < n; i++)
                dset(sh, gd, i, dget(sh, gd, i) * sc);
        }
        st[0] = l;
        st[1] = iter;
        st[2] = nlog;
        st[3] = l >= n || st[4] != 0 ? 1 : 0;
    }
    barrier();
    if (sh)
    {
        for (uint i = t; i < un; i += WG)
        {
            gd[i] = qd[i];
            ge[i] = qe[i];
        }
    }
}

// A QL iteration's rotations come in a chain i = m - 1, m - 2, ..., l, each
// sharing a row with the next: the row both touch stays in a register
// (held), so a rotation reads one row of Z^T and writes one.
void rot(uint t, uint gx, uint z)
{
    uint n = p.n, col = gx * WG + t;
    int nlog = p.state[z * 8 + 2];
    if (col >= n || nlog == 0)
        return;
    T* zt = p.zt + z * n * n;
    int* logi = p.logi + z * p.cap;
    R* logcs = p.logcs + z * 2 * p.cap;
    int held_row = -1;
    T held = ZERO;
    for (int j = 0; j < nlog; j++)
    {
        int jc = min(j, nlog - 1);
        int i = logi[jc];
        R c = logcs[2 * jc], s = logcs[2 * jc + 1];
        T hi;
        if (held_row == i + 1)
            hi = held;
        else
        {
            if (held_row >= 0)
                zt[uint(held_row) * n + col] = held;
            hi = zt[uint(i + 1) * n + col];
        }
        T lo = zt[uint(i) * n + col];
        zt[uint(i + 1) * n + col] = scale(lo, s) + scale(hi, c);
        held = scale(lo, c) - scale(hi, s);
        held_row = i;
    }
    if (held_row >= 0)
        zt[uint(held_row) * n + col] = held;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint t = lid.x;
    switch (OP)
    {
    case TRIDIAG_QL: ql(t, gid.x); break;
    case TRIDIAG_ROT: rot(t, gid.x, gid.y); break;
    }
}
