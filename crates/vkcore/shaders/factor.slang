// axis T: f32=float f64=double
//
// Small dense factorization steps for csrc/dense.cpp (M13), the parts of
// blocked LU, Cholesky, QR and Jacobi eigen/SVD that LAPACK would do on the
// CPU. Matrices are row-major and batched: matrix z starts at a + z * sa
// (row i, column j at [i * ld + j]). One workgroup works on one matrix (or
// one Jacobi pair of one matrix), so steps that depend on each other are
// separated by workgroup barriers; global memory is used throughout.
//
// OP (FACTOR_*):
//   LU_PANEL    partial-pivoting LU of columns [k, k + b) over rows [k, rows),
//               swapping whole rows; pivots (1-based, absolute) into piv,
//               the first zero pivot into info (LAPACK's getrf2 semantics)
//   LASWP       apply the first b row interchanges in piv to the rows x cols
//               matrix (in order, or in reverse with UPPER)
//   TRI_INV     inverse of the b x b triangular block at (k, k) (lower or
//               upper per UPPER, unit diagonal per UNIT) into aux (b x b)
//   CHOL_BLOCK  in-place lower Cholesky of the b x b block at (k, k); a
//               non-positive pivot sets info (once) to its 1-based index
//   QR_PANEL    Householder QR of columns [k, k + b) over rows [k, rows)
//               (LAPACK's dlarfg/dlarf: beta = -sign(alpha) |x|, v(0) = 1
//               implicit, stored below the diagonal), tau into aux[k + j]
//   LARFT       the b x b upper-triangular T of the panel's block reflector
//               H = I - V T V^T (forward, columnwise) into aux2
//   SVD_ROUND   one-sided Jacobi: for column pair `pairs[gid.x]` of the
//               rows x cols matrix W and the cols x cols matrix V (at aux),
//               rotate the two columns to make them orthogonal; the
//               pair's |w_p . w_q| / (|w_p| |w_q|) into aux2
//   EIG_COLS    Jacobi eigenvalues, first half of a round: the rotation
//               zeroing A[p, q] for each pair, applied to columns p, q of A
//               and of V (at aux); (c, s) into aux2
//   EIG_ROWS    second half: the same rotations applied to rows p, q of A

#include "opcodes.slang"

struct Params
{
    T* a;
    int* piv;
    T* aux;
    T* aux2;
    int* info;
    uint2* pairs;
    uint rows;
    uint cols;
    uint ld;
    uint k;
    uint b;
    uint sa;     // elements between matrices of a
    uint saux;   // ... of aux
    uint saux2;  // ... of aux2
    uint npairs;
    uint spiv;   // elements between pivot vectors
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = FACTOR_LU_PANEL;
[vk::constant_id(1)]
const int UPPER = 0;
[vk::constant_id(2)]
const int UNIT = 0;

#define WG 256
groupshared T sval[WG];
groupshared int sidx[WG];
groupshared T sval2[WG];
groupshared T sval3[WG];

void barrier()
{
    AllMemoryBarrierWithGroupSync();
}

// Sum of v over the workgroup (all threads get it).
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

// Three sums at once.
void reduce_sum3(uint t, inout T x, inout T y, inout T z)
{
    sval[t] = x;
    sval2[t] = y;
    sval3[t] = z;
    barrier();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            sval[t] += sval[t + s];
            sval2[t] += sval2[t + s];
            sval3[t] += sval3[t + s];
        }
        barrier();
    }
    x = sval[0];
    y = sval2[0];
    z = sval3[0];
    barrier();
}

T sign_of(T x)
{
    return x < T(0) ? T(-1) : T(1);
}

// ---------------------------------------------------------------- LU

void lu_panel(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    uint n = p.rows, ld = p.ld;
    for (uint j = p.k; j < p.k + p.b; j++)
    {
        // Pivot: the first row with the largest |a[i, j]|, i >= j.
        T best = T(-1);
        int at = int(j);
        for (uint i = j + t; i < n; i += WG)
        {
            T v = abs(a[i * ld + j]);
            if (v > best || (isnan(v) && !isnan(best)))
            {
                best = v;
                at = int(i);
            }
        }
        sval[t] = best;
        sidx[t] = at;
        barrier();
        for (uint s = WG / 2; s > 0; s >>= 1)
        {
            if (t < s)
            {
                T o = sval[t + s];
                int oi = sidx[t + s];
                if (o > sval[t] || (o == sval[t] && oi < sidx[t]))
                {
                    sval[t] = o;
                    sidx[t] = oi;
                }
            }
            barrier();
        }
        uint piv = uint(sidx[0]);
        T pivot_abs = sval[0];
        barrier();
        if (t == 0)
            p.piv[z * p.spiv + j] = int(piv) + 1;
        if (piv != j)
        {
            for (uint c = t; c < p.cols; c += WG)
            {
                T tmp = a[j * ld + c];
                a[j * ld + c] = a[piv * ld + c];
                a[piv * ld + c] = tmp;
            }
        }
        barrier();
        if (pivot_abs == T(0))
        {
            if (t == 0 && p.info[z] == 0)
                p.info[z] = int(j) + 1;
            continue;  // LAPACK leaves the column unscaled
        }
        T d = a[j * ld + j];
        for (uint i = j + 1 + t; i < n; i += WG)
            a[i * ld + j] = a[i * ld + j] / d;
        barrier();
        uint width = p.k + p.b - j - 1;
        uint height = n - j - 1;
        for (uint e = t; e < width * height; e += WG)
        {
            uint i = j + 1 + e / width, c = j + 1 + e % width;
            a[i * ld + c] = a[i * ld + c] - a[i * ld + j] * a[j * ld + c];
        }
        barrier();
    }
}

void laswp(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    uint ld = p.ld;
    for (uint step = 0; step < p.b; step++)
    {
        uint i = UPPER != 0 ? p.b - 1 - step : step;
        uint r = uint(p.piv[z * p.spiv + i] - 1);
        if (r != i)
        {
            for (uint c = t; c < p.cols; c += WG)
            {
                T tmp = a[i * ld + c];
                a[i * ld + c] = a[r * ld + c];
                a[r * ld + c] = tmp;
            }
        }
        barrier();
    }
}

// ---------------------------------------------------------------- triangular inverse, Cholesky

void tri_inv(uint t, uint z)
{
    // Thread t computes column t of the inverse by substitution.
    T* a = p.a + z * p.sa;
    T* x = p.aux + z * p.saux;
    uint b = p.b, ld = p.ld, k = p.k;
    for (uint c = t; c < b; c += WG)
    {
        if (UPPER == 0)
        {
            for (uint i = 0; i < b; i++)
            {
                T v = i == c ? T(1) : T(0);
                for (uint m = c; m < i; m++)
                    v -= a[(k + i) * ld + k + m] * x[m * b + c];
                x[i * b + c] = (UNIT != 0 || i < c) ? v : v / a[(k + i) * ld + k + i];
            }
        }
        else
        {
            for (int ii = int(b) - 1; ii >= 0; ii--)
            {
                uint i = uint(ii);
                T v = i == c ? T(1) : T(0);
                for (uint m = i + 1; m <= c; m++)
                    v -= a[(k + i) * ld + k + m] * x[m * b + c];
                x[i * b + c] = (UNIT != 0 || i > c) ? v : v / a[(k + i) * ld + k + i];
            }
        }
    }
}

void chol_block(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    uint b = p.b, ld = p.ld, k = p.k;
    for (uint j = 0; j < b; j++)
    {
        T d = a[(k + j) * ld + k + j];
        bool bad = !(d > T(0));  // also NaN
        if (bad)
        {
            if (t == 0 && p.info[z] == 0)
                p.info[z] = int(k + j) + 1;
            return;  // uniform across the workgroup
        }
        T l = sqrt(d);
        barrier();
        if (t == 0)
            a[(k + j) * ld + k + j] = l;
        for (uint i = j + 1 + t; i < b; i += WG)
            a[(k + i) * ld + k + j] = a[(k + i) * ld + k + j] / l;
        barrier();
        // Trailing lower triangle of the block: a[i, c] -= l_ij l_cj, j < c <= i.
        uint m = b - j - 1;
        for (uint e = t; e < m * m; e += WG)
        {
            uint i = j + 1 + e / m, c = j + 1 + e % m;
            if (c <= i)
                a[(k + i) * ld + k + c] = a[(k + i) * ld + k + c] - a[(k + i) * ld + k + j] * a[(k + c) * ld + k + j];
        }
        barrier();
    }
}

// ---------------------------------------------------------------- QR

void qr_panel(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    T* tau = p.aux + z * p.saux;
    uint m = p.rows, ld = p.ld;
    for (uint j = p.k; j < p.k + p.b; j++)
    {
        // dlarfg on x = a[j:m, j]
        T part = T(0);
        for (uint i = j + 1 + t; i < m; i += WG)
        {
            T v = a[i * ld + j];
            part += v * v;
        }
        T xnorm2 = reduce_sum(t, part);
        T alpha = a[j * ld + j];
        T tj = T(0);
        if (xnorm2 != T(0))
        {
            T beta = -sign_of(alpha) * sqrt(alpha * alpha + xnorm2);
            tj = (beta - alpha) / beta;
            T scale = T(1) / (alpha - beta);
            for (uint i = j + 1 + t; i < m; i += WG)
                a[i * ld + j] = a[i * ld + j] * scale;
            barrier();
            if (t == 0)
                a[j * ld + j] = beta;
        }
        if (t == 0)
            tau[j] = tj;
        barrier();
        if (tj == T(0))
            continue;
        // Apply H = I - tau v v^T to the panel's later columns.
        for (uint c = j + 1; c < p.k + p.b; c++)
        {
            T dot = T(0);
            for (uint i = j + t; i < m; i += WG)
            {
                T v = i == j ? T(1) : a[i * ld + j];
                dot += v * a[i * ld + c];
            }
            T w = reduce_sum(t, dot) * tj;
            for (uint i = j + t; i < m; i += WG)
            {
                T v = i == j ? T(1) : a[i * ld + j];
                a[i * ld + c] = a[i * ld + c] - v * w;
            }
            barrier();
        }
    }
}

// T (b x b, upper) with H_k ... H_{k+b-1} = I - V T V^T: T[j, j] = tau_j,
// T[0:j, j] = -tau_j T[0:j, 0:j] (V[:, 0:j]^T v_j).
void larft(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    T* tau = p.aux + z * p.saux;
    T* tm = p.aux2 + z * p.saux2;
    uint m = p.rows, ld = p.ld, k = p.k, b = p.b;
    for (uint j = 0; j < b; j++)
    {
        T tj = tau[k + j];
        // y[i] = V[:, i] . v_j for i < j, over rows k + j .. m (v_j is zero above).
        for (uint i = 0; i < j; i++)
        {
            T part = T(0);
            for (uint r = k + j + t; r < m; r += WG)
            {
                T vi = a[r * ld + k + i];  // r > k + i: below V[:, i]'s unit diagonal
                T vj = r == k + j ? T(1) : a[r * ld + k + j];
                part += vi * vj;
            }
            T y = reduce_sum(t, part);
            if (t == 0)
                tm[i * b + j] = y;  // temporarily
        }
        barrier();
        if (t == 0)
        {
            // T[0:j, j] = -tau_j T[0:j, 0:j] y  (T upper triangular; y in column j)
            for (uint i = 0; i < j; i++)
            {
                T s = T(0);
                for (uint l = i; l < j; l++)
                    s += tm[i * b + l] * tm[l * b + j];
                sval[i] = -tj * s;
            }
            for (uint i = 0; i < j; i++)
                tm[i * b + j] = sval[i];
            tm[j * b + j] = tj;
            for (uint i = j + 1; i < b; i++)
                tm[i * b + j] = T(0);
        }
        barrier();
    }
}

// ---------------------------------------------------------------- Jacobi

void svd_round(uint t, uint pair, uint z)
{
    T* w = p.a + z * p.sa;
    T* v = p.aux + z * p.saux;
    uint m = p.rows, n = p.cols, ld = p.ld;
    uint2 pq = p.pairs[pair];
    uint col_p = pq.x, col_q = pq.y;
    if (col_p >= n || col_q >= n)
    {
        if (t == 0)
            p.aux2[z * p.saux2 + pair] = T(0);
        return;  // the dummy partner of an odd column count
    }
    T alpha = T(0), beta = T(0), gamma = T(0);
    for (uint i = t; i < m; i += WG)
    {
        T x = w[i * ld + col_p], y = w[i * ld + col_q];
        alpha += x * x;
        beta += y * y;
        gamma += x * y;
    }
    reduce_sum3(t, alpha, beta, gamma);
    T off = (alpha > T(0) && beta > T(0)) ? abs(gamma) / sqrt(alpha * beta) : T(0);
    if (t == 0)
        p.aux2[z * p.saux2 + pair] = off;
    if (gamma == T(0))
        return;
    T zeta = (beta - alpha) / (T(2) * gamma);
    T tt = sign_of(zeta) / (abs(zeta) + sqrt(T(1) + zeta * zeta));
    T c = T(1) / sqrt(T(1) + tt * tt), s = c * tt;
    for (uint i = t; i < m; i += WG)
    {
        T x = w[i * ld + col_p], y = w[i * ld + col_q];
        w[i * ld + col_p] = c * x - s * y;
        w[i * ld + col_q] = s * x + c * y;
    }
    for (uint i = t; i < n; i += WG)
    {
        T x = v[i * n + col_p], y = v[i * n + col_q];
        v[i * n + col_p] = c * x - s * y;
        v[i * n + col_q] = s * x + c * y;
    }
}

void eig_cols(uint t, uint pair, uint z)
{
    T* a = p.a + z * p.sa;
    T* v = p.aux + z * p.saux;
    T* rot = p.aux2 + z * p.saux2;
    uint n = p.cols, ld = p.ld;
    uint2 pq = p.pairs[pair];
    uint i_p = pq.x, i_q = pq.y;
    T c = T(1), s = T(0);
    if (i_p < n && i_q < n)
    {
        T apq = a[i_p * ld + i_q];
        if (apq != T(0))
        {
            T theta = (a[i_q * ld + i_q] - a[i_p * ld + i_p]) / (T(2) * apq);
            T tt = sign_of(theta) / (abs(theta) + sqrt(theta * theta + T(1)));
            c = T(1) / sqrt(tt * tt + T(1));
            s = tt * c;
        }
    }
    if (t == 0)
    {
        rot[2 * pair] = c;
        rot[2 * pair + 1] = s;
    }
    if (s == T(0))
        return;
    // A <- A J, V <- V J for J = [[c, s], [-s, c]] in the (p, q) plane.
    for (uint i = t; i < n; i += WG)
    {
        T x = a[i * ld + i_p], y = a[i * ld + i_q];
        a[i * ld + i_p] = c * x - s * y;
        a[i * ld + i_q] = s * x + c * y;
        T vx = v[i * n + i_p], vy = v[i * n + i_q];
        v[i * n + i_p] = c * vx - s * vy;
        v[i * n + i_q] = s * vx + c * vy;
    }
}

void eig_rows(uint t, uint pair, uint z)
{
    T* a = p.a + z * p.sa;
    T* rot = p.aux2 + z * p.saux2;
    uint n = p.cols, ld = p.ld;
    uint2 pq = p.pairs[pair];
    uint i_p = pq.x, i_q = pq.y;
    T c = rot[2 * pair], s = rot[2 * pair + 1];
    if (s == T(0) || i_p >= n || i_q >= n)
        return;
    // A <- J^T A
    for (uint j = t; j < n; j += WG)
    {
        T x = a[i_p * ld + j], y = a[i_q * ld + j];
        a[i_p * ld + j] = c * x - s * y;
        a[i_q * ld + j] = s * x + c * y;
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    uint t = lid.x;
    switch (OP)
    {
    case FACTOR_LU_PANEL: lu_panel(t, gid.x); break;
    case FACTOR_LASWP: laswp(t, gid.x); break;
    case FACTOR_TRI_INV: tri_inv(t, gid.x); break;
    case FACTOR_CHOL_BLOCK: chol_block(t, gid.x); break;
    case FACTOR_QR_PANEL: qr_panel(t, gid.x); break;
    case FACTOR_LARFT: larft(t, gid.x); break;
    case FACTOR_SVD_ROUND: svd_round(t, gid.x, gid.y); break;
    case FACTOR_EIG_COLS: eig_cols(t, gid.x, gid.y); break;
    default: eig_rows(t, gid.x, gid.y); break;  // FACTOR_EIG_ROWS
    }
}
