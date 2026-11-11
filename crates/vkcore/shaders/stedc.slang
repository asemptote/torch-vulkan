// axis T: f32=float f64=double
//
// Divide and conquer for the symmetric tridiagonal eigenproblem (LAPACK's
// dstedc: dlaed0 - dlaed4), for csrc/dense.cpp's eigh. The tridiagonal is
// torn into leaves (solved by tridiag.slang's QL, batched); each level then
// merges pairs of solved halves, all merges of a level at once (G merges of
// m = 2h rows each, merge g on rows [g m, g m + m) of the batch's flat
// diagonal). For the halves' eigen-decompositions D1, Q1 and D2, Q2:
//
//   T = diag(T1, T2) + |e| u u^T,  u = e_{h-1} + sign(e) e_h  (the tear),
//   so with QM = diag(Q1, Q2):  T = QM (D + rho z z^T) QM^T,
//   z = QM^T u / sqrt(2), rho = 2 |e|.
//
// The rank-one update D + rho z z^T is solved as LAPACK solves it:
//
// OP (STEDC_*):
//   PREP     one workgroup per merge: z from QM's rows h - 1 and h; the two
//            sorted halves of d merged; deflation on one thread, in sorted
//            order (dlaed2): an entry with rho |z_i| <= tol is an eigenpair
//            already, and two close entries are rotated so that one of them
//            is (the rotations logged for ROT). The k others go to dlam
//            (ascending) with their z, the deflated ones after them.
//   ROT      the logged rotations applied to QM's columns, a thread per row.
//   SECULAR  root j < k of the secular equation 1 / rho + sum z_i^2 /
//            (dlam_i - lambda) = 0 (dlaed4's "middle way": both poles that
//            bound the root modelled as rational functions, the quadratic
//            for the step solved, bisection where a step leaves the
//            bracket), a thread per root. A root is kept as an origin pole
//            o and an offset tau (lambda = dlam_o + tau), so that every
//            dlam_i - lambda is computed as (dlam_i - dlam_o) - tau,
//            accurately near the poles.
//   ZHAT     Gu and Eisenstat's z (dlaed3): from the computed roots, so that
//            the eigenvectors come out orthogonal; a thread per i < k.
//   VEC      eigenvector j of the rank-one problem, zhat_i / (dlam_i -
//            lambda_j) normalised, written into U's row of its column in QM
//            (and a unit column for each deflated entry), and the merged
//            eigenvalue; one workgroup per column. csrc/dense.cpp then sorts
//            the eigenvalues and computes QM U (a batched GEMM).
//
// Work arrays (per merge g, offsets in units of G m): rw (real): z, dlam,
// tau, zhat, vals, then the rotations' (c, s) (2 G m), then rho [G]; iw
// (int): col, org, the rotations' column pairs (2 G m), then info [4 G]
// (k, number of rotations, whether a rotation mixed the two halves).

#include "opcodes.slang"

typedef T R;

struct Params
{
    R* d;     // [G m] the halves' eigenvalues, each half ascending
    R* e;     // [G m] the off-diagonal: merge g's tear at g m + h - 1
    R* q;     // [G, m, m] QM, rotated in place by ROT
    R* u;     // [G, m, m] U (zeroed by the host)
    R* rw;    // real work
    int* iw;  // int work
    uint m;
    uint G;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = STEDC_PREP;

// (a bare literal is float32: double constants need the l suffix)
#if defined(T_f64)
#define LIT(c) c##l
#else
#define LIT(c) c##f
#endif
#if defined(T_f64)
static const R EPS = 1.1102230246251565e-16l;  // unit roundoff
#else
static const R EPS = 5.9604645e-8f;
#endif


#define WG 64

// real work: z, dlam, tau, zhat, vals at multiples of G m; rotations after
uint gm() { return p.G * p.m; }
uint RZ(uint g, uint i) { return g * p.m + i; }
uint RDLAM(uint g, uint i) { return gm() + g * p.m + i; }
uint RTAU(uint g, uint i) { return 2 * gm() + g * p.m + i; }
uint RZHAT(uint g, uint i) { return 3 * gm() + g * p.m + i; }
uint RVALS(uint g, uint i) { return 4 * gm() + g * p.m + i; }
uint RCS(uint g, uint r) { return 5 * gm() + 2 * (g * p.m + r); }
uint RRHO(uint g) { return 7 * gm() + g; }
// int work: col, org, rotation pairs, info
uint ICOL(uint g, uint i) { return g * p.m + i; }
uint IORG(uint g, uint i) { return gm() + g * p.m + i; }
uint IROT(uint g, uint r) { return 2 * gm() + 2 * (g * p.m + r); }
uint IINFO(uint g) { return 4 * gm() + 4 * g; }

groupshared R red[WG];

R sum_wg(uint t, R v)
{
    red[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
            red[t] += red[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    R r = red[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

R max_wg(uint t, R v)
{
    red[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
            red[t] = max(red[t], red[t + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    R r = red[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

R hypot2(R a, R b)
{
    R x = abs(a), y = abs(b), big = max(x, y), small = min(x, y);
    if (big == R(0))
        return R(0);
    R r = small / big;
    return big * sqrt(R(1) + r * r);
}

// PREP's scratch: the full z (by merge-local index) in the tau area and the
// working d in the vals area; the merged order in the org area.
void prep(uint t, uint g)
{
    uint m = p.m, h = m / 2, base = g * m;
    R raw = p.e[base + h - 1];
    R sgn = raw < R(0) ? R(-1) : R(1);
    R inv_sqrt2 = R(LIT(0.70710678118654752440));
    uint qb = g * m * m;
    for (uint i = t; i < m; i += WG)
    {
        R zi = i < h ? p.q[qb + (h - 1) * m + i] : sgn * p.q[qb + h * m + i];
        p.rw[RTAU(g, i)] = zi * inv_sqrt2;
        p.rw[RVALS(g, i)] = p.d[base + i];
    }
    AllMemoryBarrierWithGroupSync();
    if (t != 0)
        return;
    R rho = R(2) * abs(raw);
    p.rw[RRHO(g)] = rho;
    // merge the two ascending halves
    uint i1 = 0, i2 = h;
    R dmax = R(0), zmax = R(0);
    for (uint o = 0; o < m; o++)
    {
        uint pick;
        if (i2 >= m || (i1 < h && p.rw[RVALS(g, i1)] <= p.rw[RVALS(g, i2)]))
            pick = i1++;
        else
            pick = i2++;
        p.iw[IORG(g, o)] = int(pick);
        dmax = max(dmax, abs(p.rw[RVALS(g, pick)]));
        zmax = max(zmax, abs(p.rw[RTAU(g, pick)]));
    }
    R tol = R(8) * EPS * max(dmax, zmax);
    uint k = 0, ndef = 0, nrot = 0, cross = 0;
    int pj = -1;
    for (uint o = 0; o < m; o++)
    {
        uint j = uint(p.iw[IORG(g, o)]);
        R zj = p.rw[RTAU(g, j)];
        if (rho * abs(zj) <= tol)
        {
            ndef++;
            p.rw[RDLAM(g, m - ndef)] = p.rw[RVALS(g, j)];
            p.iw[ICOL(g, m - ndef)] = int(j);
            continue;
        }
        if (pj < 0)
        {
            pj = int(j);
            continue;
        }
        uint i = uint(pj);
        R s = p.rw[RTAU(g, i)], c = zj;
        R tau = hypot2(c, s);
        R tt = p.rw[RVALS(g, j)] - p.rw[RVALS(g, i)];
        c /= tau;
        s = -s / tau;
        if (abs(tt * c * s) <= tol)
        {
            // rotate columns i and j so that z_i = 0: i is then deflated
            p.rw[RTAU(g, j)] = tau;
            p.rw[RTAU(g, i)] = R(0);
            p.iw[IROT(g, nrot)] = int(i);
            p.iw[IROT(g, nrot) + 1] = int(j);
            p.rw[RCS(g, nrot)] = c;
            p.rw[RCS(g, nrot) + 1] = s;
            nrot++;
            if ((i < h) != (j < h))
                cross = 1;  // (QM no longer block diagonal)
            R di = p.rw[RVALS(g, i)], dj = p.rw[RVALS(g, j)];
            p.rw[RVALS(g, i)] = di * c * c + dj * s * s;
            p.rw[RVALS(g, j)] = di * s * s + dj * c * c;
            ndef++;
            p.rw[RDLAM(g, m - ndef)] = p.rw[RVALS(g, i)];
            p.iw[ICOL(g, m - ndef)] = int(i);
        }
        else
        {
            p.rw[RDLAM(g, k)] = p.rw[RVALS(g, i)];
            p.rw[RZ(g, k)] = p.rw[RTAU(g, i)];
            p.iw[ICOL(g, k)] = int(i);
            k++;
        }
        pj = int(j);
    }
    if (pj >= 0)
    {
        uint i = uint(pj);
        p.rw[RDLAM(g, k)] = p.rw[RVALS(g, i)];
        p.rw[RZ(g, k)] = p.rw[RTAU(g, i)];
        p.iw[ICOL(g, k)] = int(i);
        k++;
    }
    p.iw[IINFO(g)] = int(k);
    p.iw[IINFO(g) + 1] = int(nrot);
    p.iw[IINFO(g) + 2] = int(cross);
}

void rot(uint r, uint g)
{
    uint m = p.m;
    if (r >= m)
        return;
    uint nrot = uint(p.iw[IINFO(g) + 1]);
    uint row = g * m * m + r * m;
    for (uint i = 0; i < nrot; i++)
    {
        uint a = uint(p.iw[IROT(g, i)]), b = uint(p.iw[IROT(g, i) + 1]);
        R c = p.rw[RCS(g, i)], s = p.rw[RCS(g, i) + 1];
        R x = p.q[row + a], y = p.q[row + b];
        p.q[row + a] = c * x + s * y;
        p.q[row + b] = c * y - s * x;
    }
}

// Roots j of the secular equation: SEC_LANES lanes per root sum the poles'
// terms (dlam_i, z_i) between them, added up in shared memory, and the
// root's first lane updates it; a workgroup's SEC_ROOTS roots iterate
// together until all have converged. (A thread per root summed k terms one
// after another each iteration: latency-bound, and too few threads to fill
// a large GPU.) Per root: origin pole o and offset tau.
#define SEC_LANES 32
#define SEC_ROOTS (WG / SEC_LANES)
groupshared R sec_sum[5][WG];
groupshared R sec_tau[SEC_ROOTS];
groupshared R sec_dorg[SEC_ROOTS];
groupshared uint sec_done[SEC_ROOTS];
groupshared uint sec_any;

void secular(uint t, uint gx, uint g)
{
    uint k = uint(p.iw[IINFO(g)]);
    uint r = t / SEC_LANES, lane = t % SEC_LANES;
    uint j = gx * SEC_ROOTS + r;
    if (gx * SEC_ROOTS >= k)
        return;  // (uniform: the whole workgroup)
    bool mine = j < k;
    uint jj = min(j, k - 1);
    R rho = p.rw[RRHO(g)];
    R rhoinv = R(1) / rho;
    R dj = p.rw[RDLAM(g, jj)];
    bool last = jj + 1 == k;
    R gap = last ? R(0) : p.rw[RDLAM(g, jj + 1)] - dj;
    R mid = gap * R(0.5);
    // (the root's state, kept by its first lane)
    uint o = jj;
    R lo = R(0), hi = R(0);
    bool choosing = true;
    if (lane == 0)
    {
        sec_tau[r] = last ? R(0) : mid;  // pass 0 picks the origin
        sec_dorg[r] = dj;
        sec_done[r] = mine ? 0u : 1u;
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint it = 0; it < 101; it++)
    {
        R tau = sec_tau[r], dorg = sec_dorg[r];
        bool done = sec_done[r] != 0u;
        R psi = R(0), dpsi = R(0), phi = R(0), dphi = R(0), zz = R(0);
        if (!done)
        {
            for (uint i = lane; i < k; i += SEC_LANES)
            {
                R zi = p.rw[RZ(g, i)];
                R del = (p.rw[RDLAM(g, i)] - dorg) - tau;
                R tmp = zi / del;
                zz += zi * zi;
                if (i <= jj)
                {
                    psi += zi * tmp;
                    dpsi += tmp * tmp;
                }
                else
                {
                    phi += zi * tmp;
                    dphi += tmp * tmp;
                }
            }
        }
        sec_sum[0][t] = psi;
        sec_sum[1][t] = dpsi;
        sec_sum[2][t] = phi;
        sec_sum[3][t] = dphi;
        sec_sum[4][t] = zz;
        GroupMemoryBarrierWithGroupSync();
        for (uint w = SEC_LANES / 2; w > 0; w >>= 1)
        {
            if (lane < w)
            {
                [ForceUnroll]
                for (uint q = 0; q < 5; q++)
                    sec_sum[q][t] += sec_sum[q][t + w];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0 && !done)
        {
            psi = sec_sum[0][t];
            dpsi = sec_sum[1][t];
            phi = sec_sum[2][t];
            dphi = sec_sum[3][t];
            zz = sec_sum[4][t];
            R f = rhoinv + psi + phi;
            if (choosing)
            {
                choosing = false;
                if (last)
                {
                    // the largest root: in (dlam_j, dlam_j + rho |z|^2]
                    lo = R(0);
                    hi = rho * zz * (R(1) + R(4) * EPS);
                }
                else if (f >= R(0))
                {
                    lo = R(0);
                    hi = mid;
                }
                else
                {
                    o = jj + 1;
                    dorg = p.rw[RDLAM(g, o)];
                    lo = -(gap - mid);
                    hi = R(0);
                }
                tau = (lo + hi) * R(0.5);
            }
            else if (f == R(0))
                done = true;
            else
            {
                if (f < R(0))
                    lo = tau;
                else
                    hi = tau;
                R d1 = (dj - dorg) - tau;  // dlam_j - lambda (< 0)
                R eta = R(0);
                bool ok = true;
                if (last)
                {
                    // one pole: c + b1 / (d1 - eta) = 0
                    R c = f - d1 * dpsi;
                    R b1 = d1 * d1 * dpsi;
                    if (c == R(0))
                        ok = false;
                    else
                        eta = d1 + b1 / c;
                }
                else
                {
                    R d2 = (p.rw[RDLAM(g, jj + 1)] - dorg) - tau;  // > 0
                    R b1 = d1 * d1 * dpsi, b2 = d2 * d2 * dphi;
                    R c = f - d1 * dpsi - d2 * dphi;
                    R bq = c * (d1 + d2) + b1 + b2;
                    R cq = c * d1 * d2 + b1 * d2 + b2 * d1;
                    if (c == R(0))
                    {
                        if (b1 + b2 == R(0))
                            ok = false;
                        else
                            eta = cq / bq;
                    }
                    else
                    {
                        R disc = bq * bq - R(4) * c * cq;
                        if (disc < R(0))
                            ok = false;
                        else
                        {
                            R sq = sqrt(disc);
                            // the two roots, stably; the one between the poles (d1, d2)
                            R r1 = bq >= R(0) ? (bq + sq) / (R(2) * c) : (R(2) * cq) / (bq - sq);
                            R r2 = bq >= R(0) ? (R(2) * cq) / (bq + sq) : (bq - sq) / (R(2) * c);
                            bool in1 = r1 > d1 && r1 < d2, in2 = r2 > d1 && r2 < d2;
                            if (in1 && in2)
                                eta = abs(r1) < abs(r2) ? r1 : r2;
                            else if (in1)
                                eta = r1;
                            else if (in2)
                                eta = r2;
                            else
                                ok = false;
                        }
                    }
                }
                R tnew = tau + eta;
                if (!ok || !(tnew > lo && tnew < hi) || isnan(tnew))
                    tnew = (lo + hi) * R(0.5);
                R step = abs(tnew - tau);
                tau = tnew;
                if (step <= R(2) * EPS * abs(tau) || hi - lo <= R(2) * EPS * max(abs(lo), abs(hi)))
                    done = true;
            }
            sec_tau[r] = tau;
            sec_dorg[r] = dorg;
            sec_done[r] = done ? 1u : 0u;
        }
        if (t == 0)
            sec_any = 0;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0 && sec_done[r] == 0u)
            sec_any = 1;
        GroupMemoryBarrierWithGroupSync();
        if (sec_any == 0)
            break;
    }
    if (mine && lane == 0)
    {
        p.rw[RTAU(g, j)] = sec_tau[r];
        p.iw[IORG(g, j)] = int(o);
    }
}

// dlam_i - lambda_j, accurately
R delta(uint g, uint i, uint j)
{
    uint o = uint(p.iw[IORG(g, j)]);
    return (p.rw[RDLAM(g, i)] - p.rw[RDLAM(g, o)]) - p.rw[RTAU(g, j)];
}

void zhat(uint i, uint g)
{
    uint k = uint(p.iw[IINFO(g)]);
    if (i >= k)
        return;
    R di = p.rw[RDLAM(g, i)];
    R w = delta(g, i, i);
    for (uint j = 0; j < k; j++)
        if (j != i)
            w *= delta(g, i, j) / (di - p.rw[RDLAM(g, j)]);
    R zi = p.rw[RZ(g, i)];
    R z = sqrt(max(-w, R(0)));
    p.rw[RZHAT(g, i)] = zi < R(0) ? -z : z;
}

void vec(uint t, uint j, uint g)
{
    uint m = p.m, k = uint(p.iw[IINFO(g)]);
    uint ub = g * m * m;
    if (j >= k)
    {
        if (t == 0)
        {
            p.u[ub + uint(p.iw[ICOL(g, j)]) * m + j] = R(1);
            p.rw[RVALS(g, j)] = p.rw[RDLAM(g, j)];
        }
        return;
    }
    R big = R(0);
    for (uint i = t; i < k; i += WG)
        big = max(big, abs(p.rw[RZHAT(g, i)] / delta(g, i, j)));
    big = max_wg(t, big);
    R ss = R(0);
    for (uint i = t; i < k; i += WG)
    {
        R v = p.rw[RZHAT(g, i)] / delta(g, i, j) / big;
        ss += v * v;
    }
    R nrm = big * sqrt(sum_wg(t, ss));
    for (uint i = t; i < k; i += WG)
        p.u[ub + uint(p.iw[ICOL(g, i)]) * m + j] = p.rw[RZHAT(g, i)] / delta(g, i, j) / nrm;
    if (t == 0)
    {
        uint o = uint(p.iw[IORG(g, j)]);
        p.rw[RVALS(g, j)] = p.rw[RDLAM(g, o)] + p.rw[RTAU(g, j)];
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID, uint3 dtid: SV_DispatchThreadID)
{
    if (OP == STEDC_PREP)
        prep(tid.x, gid.x);
    else if (OP == STEDC_ROT)
        rot(dtid.x, gid.y);
    else if (OP == STEDC_SECULAR)
        secular(tid.x, gid.x, gid.y);
    else if (OP == STEDC_ZHAT)
        zhat(dtid.x, gid.y);
    else
        vec(tid.x, gid.x, gid.y);
}
