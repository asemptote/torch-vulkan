// axis T: c64=float2 c128=double2
//
// General (nonsymmetric) eigendecomposition for csrc/dense.cpp (M14), in
// complex arithmetic (real matrices are converted first). One workgroup per
// matrix; matrices are row-major n x n, matrix z at a + z * n * n.
//
// OP (EIG_*):
//   HESSENBERG  reduce A (at a) to upper Hessenberg form H by Householder
//               similarities, accumulating the unitary Z (at z, A = Z H Z^H);
//               aux is scratch for the reflector (n per matrix)
//   QR          up to `budget` shifted QR sweeps (single Wilkinson shift,
//               Givens rotations chasing the bulge) on H, accumulated into Z,
//               deflating from the bottom; state[2 z] is the last unconverged
//               row (-1 when H is triangular: the complex Schur form), and
//               state[2 z + 1] the sweeps since the last deflation
//   VECTORS     eigenvectors of the triangular T (at a): column k of X (at
//               aux, n x n) solves (T - t_kk) x = 0 with x_k = 1 by back
//               substitution, one thread per k; small pivots are replaced by
//               smin (aux2[z].x), as LAPACK's ztrevc does

#include "opcodes.slang"

#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif
#include "_complex.slang"

struct Params
{
    T* a;
    T* z;
    T* aux;
    T* aux2;
    int* state;
    uint n;
    uint budget;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = EIG_HESSENBERG;

#define WG 256
groupshared T sval[WG];
groupshared S rval[WG];

void barrier()
{
    AllMemoryBarrierWithGroupSync();
}

T czero() { return T(S(0), S(0)); }
T cone() { return T(S(1), S(0)); }
S abs1(T x) { return abs(x.x) + abs(x.y); }
S norm2(T x) { return x.x * x.x + x.y * x.y; }

T reduce_sum(uint t, T v)
{
    sval[t] = v;
    barrier();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
            sval[t] += sval[t + s];
        barrier();
    }
    T r = sval[0];
    barrier();
    return r;
}

S reduce_sum_real(uint t, S v)
{
    rval[t] = v;
    barrier();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
            rval[t] += rval[t + s];
        barrier();
    }
    S r = rval[0];
    barrier();
    return r;
}

// ---------------------------------------------------------------- Hessenberg reduction

void hessenberg(uint t, uint zi)
{
    uint n = p.n;
    T* h = p.a + zi * n * n;
    T* q = p.z + zi * n * n;
    T* v = p.aux + zi * n;
    for (uint e = t; e < n * n; e += WG)
        q[e] = (e / n == e % n) ? cone() : czero();
    barrier();
    for (uint k = 0; k + 2 < n; k++)
    {
        // zlarfg on x = h[k+1 .. n, k]
        S part = S(0);
        for (uint i = k + 2 + t; i < n; i += WG)
            part += norm2(h[i * n + k]);
        S xnorm2 = reduce_sum_real(t, part);
        T alpha = h[(k + 1) * n + k];
        if (xnorm2 == S(0) && alpha.y == S(0))
            continue;  // H(k) = I
        S beta = (alpha.x < S(0) ? S(1) : S(-1)) * sqrt(norm2(alpha) + xnorm2);
        T tau = T((beta - alpha.x) / beta, -alpha.y / beta);
        T scale = cdiv(cone(), alpha - T(beta, S(0)));
        for (uint i = k + 1 + t; i < n; i += WG)
            v[i] = i == k + 1 ? cone() : cmul(h[i * n + k], scale);
        barrier();
        // Left: H <- P^H H on columns k+1.., P = I - tau v v^H (column k is (.., beta, 0..)).
        T ctau = cconj(tau);
        for (uint j = k + 1 + t; j < n; j += WG)
        {
            T w = czero();
            for (uint i = k + 1; i < n; i++)
                w += cmul(cconj(v[i]), h[i * n + j]);
            w = cmul(ctau, w);
            for (uint i = k + 1; i < n; i++)
                h[i * n + j] -= cmul(v[i], w);
        }
        barrier();
        if (t == 0)
        {
            h[(k + 1) * n + k] = T(beta, S(0));
            for (uint i = k + 2; i < n; i++)
                h[i * n + k] = czero();
        }
        // Right: H <- H P and Z <- Z P on columns k+1.. of every row.
        for (uint r = t; r < 2 * n; r += WG)
        {
            T* row = r < n ? h + r * n : q + (r - n) * n;
            T w = czero();
            for (uint j = k + 1; j < n; j++)
                w += cmul(row[j], v[j]);
            w = cmul(w, tau);
            for (uint j = k + 1; j < n; j++)
                row[j] -= cmul(w, cconj(v[j]));
        }
        barrier();
    }
}

// ---------------------------------------------------------------- shifted QR

// G = [[c, s], [-conj(s), c]] with G (f, g)^T = (r, 0)^T.
void givens(T f, T g, out S c, out T s)
{
    S af = cabs(f), ag = cabs(g);
    if (ag == S(0))
    {
        c = S(1);
        s = czero();
        return;
    }
    if (af == S(0))
    {
        c = S(0);
        s = cconj(g) / ag;
        return;
    }
    S r = sqrt(af * af + ag * ag);
    c = af / r;
    s = cmul(f / af, cconj(g)) / r;
}

// H <- G H G^H on rows/columns k, k+1 (rows from col0 on the left, columns
// up to row1 on the right), Z <- Z G^H.
void rotate(uint t, T* h, T* q, uint n, uint k, uint col0, uint row1, S c, T s)
{
    T ns = -cconj(s);
    for (uint j = col0 + t; j < n; j += WG)
    {
        T x = h[k * n + j], y = h[(k + 1) * n + j];
        h[k * n + j] = c * x + cmul(s, y);
        h[(k + 1) * n + j] = cmul(ns, x) + c * y;
    }
    barrier();
    T cs = cconj(s);
    for (uint r = t; r < (row1 + 1) + n; r += WG)
    {
        T* row = r <= row1 ? h + r * n : q + (r - row1 - 1) * n;
        T x = row[k], y = row[k + 1];
        row[k] = c * x + cmul(cs, y);
        row[k + 1] = c * y - cmul(s, x);
    }
    barrier();
}

void qr(uint t, uint zi)
{
    uint n = p.n;
    T* h = p.a + zi * n * n;
    T* q = p.z + zi * n * n;
#if defined(T_c64)
    const S eps = S(5.9604644775390625e-8);
#else
    const S eps = S(1.1102230246251565e-16);
#endif
    int hi = p.state[2 * zi];
    int its = p.state[2 * zi + 1];
    for (uint sweep = 0; sweep < p.budget && hi >= 0; sweep++)
    {
        // Deflate: the lowest l with a negligible subdiagonal h[l, l-1].
        int l = 0;
        for (int m = hi; m > 0; m--)
        {
            uint u = uint(max(m, 1));  // (m >= 1; see the note on the shift below)
            T sub = h[u * n + u - 1];
            S near = abs1(h[(u - 1) * n + u - 1]) + abs1(h[u * n + u]);
            if (abs1(sub) <= eps * near || abs1(sub) < S(1.17549435e-38))
            {
                l = m;
                break;
            }
        }
        if (l == hi)
        {
            barrier();
            if (t == 0 && hi > 0)
                h[hi * n + hi - 1] = czero();
            barrier();
            hi--;
            its = 0;
            continue;
        }
        if (t == 0 && l > 0)
            h[l * n + l - 1] = czero();
        barrier();
        // Wilkinson shift: the eigenvalue of the trailing 2 x 2 nearer h[hi, hi];
        // an exceptional shift every 10 sweeps without deflation.
        // (hi >= 1 here. But lavapipe's generated code also performs loads on
        // paths a lane does not take, e.g. these for a lane that deflated
        // above with hi = 0, and a loop's first loads before its condition:
        // every index is kept in range even where it is not used.)
        uint u = uint(max(hi, 1));
        T a = h[(u - 1) * n + u - 1], b = h[(u - 1) * n + u], cc = h[u * n + u - 1], d = h[u * n + u];
        T mu;
        its++;
        if (its % 10 == 0)
        {
            mu = d + T(S(0.75) * abs1(cc), S(0));
        }
        else
        {
            T half = (a - d) * S(0.5);
            T disc = cmul(half, half) + cmul(b, cc);
            // sqrt of disc (principal), sign chosen to move away from half
            S r = cabs(disc);
            T root = T(sqrt(max(S(0.5) * (r + disc.x), S(0))), S(0));
            S im = sqrt(max(S(0.5) * (r - disc.x), S(0)));
            root.y = disc.y < S(0) ? -im : im;
            if (half.x * root.x + half.y * root.y < S(0))
                root = -root;
            T den = half + root;
            mu = (den.x == S(0) && den.y == S(0)) ? d : d - cdiv(cmul(b, cc), den);
        }
        // One sweep on rows/columns l .. hi.
        for (int k = l; k < hi; k++)
        {
            // The column the rotation zeroes: l itself, then the bulge at k - 1.
            // (Choosing the index, not the loaded value: a compiler may load
            // both sides of a branch, and h[k * n + k - 1] for k = 0 is out of
            // bounds.)
            uint col = k == l ? uint(l) : uint(k - 1);
            T f = h[uint(k) * n + col] - (k == l ? mu : czero());
            T g = h[uint(k + 1) * n + col];
            S c;
            T s;
            givens(f, g, c, s);
            barrier();
            rotate(t, h, q, n, uint(k), k == l ? uint(l) : uint(k - 1), uint(min(k + 2, hi)), c, s);
            if (k > l)
            {
                if (t == 0)
                    h[(k + 1) * n + k - 1] = czero();
                barrier();
            }
        }
    }
    barrier();
    if (t == 0)
    {
        p.state[2 * zi] = hi;
        p.state[2 * zi + 1] = its;
    }
}

// ---------------------------------------------------------------- eigenvectors of T

void vectors(uint t, uint zi)
{
    uint n = p.n;
    T* tm = p.a + zi * n * n;
    T* x = p.aux + zi * n * n;
    S smin = p.aux2[zi].x;
    for (uint k = t; k < n; k += WG)
    {
        T lam = tm[k * n + k];
        for (uint j = k + 1; j < n; j++)
            x[j * n + k] = czero();
        x[k * n + k] = cone();
        for (int j = int(k) - 1; j >= 0; j--)
        {
            uint u = uint(max(j, 0));  // (j >= 0; see the note in qr)
            T s = czero();
            for (uint m = u + 1; m <= k; m++)
                s += cmul(tm[u * n + m], x[m * n + k]);
            T dd = tm[u * n + u] - lam;
            if (abs1(dd) < smin)
                dd = T(smin, S(0));
            x[u * n + k] = -cdiv(s, dd);
        }
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    switch (OP)
    {
    case EIG_HESSENBERG: hessenberg(lid.x, gid.x); break;
    case EIG_QR: qr(lid.x, gid.x); break;
    default: vectors(lid.x, gid.x); break;  // EIG_VECTORS
    }
}
