// axis T: f32=float f64=double c64=float2 c128=double2
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
//               H = I - V T V^H (forward, columnwise) into aux2
//   SVD_ROUND   one-sided Jacobi: for column pair `pairs[gid.x]` of the
//               rows x cols matrix W and the cols x cols matrix V (at aux),
//               rotate the two columns to make them orthogonal; the
//               pair's |w_p^H w_q| / (|w_p| |w_q|) into aux2
//   EIG_COLS    Jacobi eigenvalues, first half of a round: the rotation
//               zeroing A[p, q] for each pair, applied to columns p, q of A
//               and of V (at aux); the rotation into aux2
//   EIG_ROWS    second half: the same rotations applied to rows p, q of A
//   LDL         Bunch-Kaufman LDL^T (LDL^H with UPPER = hermitian) of the
//               rows x rows matrix's lower triangle in place, as LAPACK's
//               sytf2 / hetf2 (lower): 1-based pivots into piv, negative
//               for 2 x 2 blocks; the first zero pivot into info; aux is
//               scratch (2 rows per matrix)
//   LDL_SOLVE   solve A X = B from LDL's factors (sytrs / hetrs, lower) for
//               the rows x cols right-hand sides at aux (row stride cols),
//               one thread per column
//
// Complex matrices (c64, c128) follow LAPACK's z* routines: LU pivots on
// |re| + |im| (izamax), Cholesky reads the real part of the diagonal and
// gives A = L L^H, Householder reflectors are H = I - tau v v^H with a real
// beta (zlarfg) and H^H is applied to the panel. The Jacobi steps first
// rotate column q by the phase of the (p, q) entry, which turns each 2 x 2
// problem into the real one.

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
T mul(T a, T b) { return cmul(a, b); }
T dv(T a, T b) { return cdiv(a, b); }
T cj(T a) { return cconj(a); }
R re(T a) { return a.x; }
R mag(T a) { return cabs(a); }
R mag1(T a) { return abs(a.x) + abs(a.y); }  // izamax's |re| + |im|
R norm2(T a) { return a.x * a.x + a.y * a.y; }
T fromr(R r) { return T(r, R(0)); }
bool is_zero(T a) { return a.x == R(0) && a.y == R(0); }
#else
typedef T R;
T mul(T a, T b) { return a * b; }
T dv(T a, T b) { return a / b; }
T cj(T a) { return a; }
R re(T a) { return a; }
R mag(T a) { return abs(a); }
R mag1(T a) { return abs(a); }
R norm2(T a) { return a * a; }
T fromr(R r) { return r; }
bool is_zero(T a) { return a == T(0); }
#endif
#define ONE fromr(R(1))
#define ZERO fromr(R(0))

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
groupshared R rval[WG];
groupshared R rval2[WG];

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

R reduce_sum_real(uint t, R v)
{
    rval[t] = v;
    barrier();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
            rval[t] += rval[t + s];
        barrier();
    }
    R r = rval[0];
    barrier();
    return r;
}

// Two real sums and one T sum at once.
void reduce_sum3(uint t, inout R x, inout R y, inout T z)
{
    rval[t] = x;
    rval2[t] = y;
    sval[t] = z;
    barrier();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
        {
            rval[t] += rval[t + s];
            rval2[t] += rval2[t + s];
            sval[t] += sval[t + s];
        }
        barrier();
    }
    x = rval[0];
    y = rval2[0];
    z = sval[0];
    barrier();
}

R sign_of(R x)
{
    return x < R(0) ? R(-1) : R(1);
}

// ---------------------------------------------------------------- LU

void lu_panel(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    uint n = p.rows, ld = p.ld;
    for (uint j = p.k; j < p.k + p.b; j++)
    {
        // Pivot: the first row with the largest |a[i, j]|, i >= j.
        R best = R(-1);
        int at = int(j);
        for (uint i = j + t; i < n; i += WG)
        {
            R v = mag1(a[i * ld + j]);
            if (v > best || (isnan(v) && !isnan(best)))
            {
                best = v;
                at = int(i);
            }
        }
        rval[t] = best;
        sidx[t] = at;
        barrier();
        for (uint s = WG / 2; s > 0; s >>= 1)
        {
            if (t < s)
            {
                R o = rval[t + s];
                int oi = sidx[t + s];
                if (o > rval[t] || (o == rval[t] && oi < sidx[t]))
                {
                    rval[t] = o;
                    sidx[t] = oi;
                }
            }
            barrier();
        }
        uint piv = uint(sidx[0]);
        R pivot_abs = rval[0];
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
        if (pivot_abs == R(0))
        {
            if (t == 0 && p.info[z] == 0)
                p.info[z] = int(j) + 1;
            continue;  // LAPACK leaves the column unscaled
        }
        T d = a[j * ld + j];
        for (uint i = j + 1 + t; i < n; i += WG)
            a[i * ld + j] = dv(a[i * ld + j], d);
        barrier();
        uint width = p.k + p.b - j - 1;
        uint height = n - j - 1;
        for (uint e = t; e < width * height; e += WG)
        {
            uint i = j + 1 + e / width, c = j + 1 + e % width;
            a[i * ld + c] = a[i * ld + c] - mul(a[i * ld + j], a[j * ld + c]);
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
    // Thread t computes column t of the inverse by forward substitution. An
    // upper block U is inverted as the lower J U J (J reverses the order of
    // rows and columns): inv(U) = J inv(J U J) J. (One loop shape for both:
    // lavapipe's code for a bottom-up loop here read row -1 of the matrix.)
    T* a = p.a + z * p.sa;
    T* x = p.aux + z * p.saux;
    uint b = p.b, ld = p.ld, k = p.k;
    for (uint c = t; c < b; c += WG)
    {
        uint cc = UPPER != 0 ? b - 1 - c : c;
        for (uint i = 0; i < b; i++)
        {
            uint ii = UPPER != 0 ? b - 1 - i : i;
            T v = i == c ? ONE : ZERO;
            for (uint m = c; m < i; m++)
            {
                uint mm = UPPER != 0 ? b - 1 - m : m;
                v -= mul(a[(k + ii) * ld + k + mm], x[mm * b + cc]);
            }
            x[ii * b + cc] = (UNIT != 0 || i < c) ? v : dv(v, a[(k + ii) * ld + k + ii]);
        }
    }
}

void chol_block(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    uint b = p.b, ld = p.ld, k = p.k;
    for (uint j = 0; j < b; j++)
    {
        R d = re(a[(k + j) * ld + k + j]);
        bool bad = !(d > R(0));  // also NaN
        if (bad)
        {
            if (t == 0 && p.info[z] == 0)
                p.info[z] = int(k + j) + 1;
            return;  // uniform across the workgroup
        }
        R l = sqrt(d);
        barrier();
        if (t == 0)
            a[(k + j) * ld + k + j] = fromr(l);
        for (uint i = j + 1 + t; i < b; i += WG)
            a[(k + i) * ld + k + j] = a[(k + i) * ld + k + j] / l;
        barrier();
        // Trailing lower triangle of the block: a[i, c] -= l_ij conj(l_cj), j < c <= i.
        uint m = b - j - 1;
        for (uint e = t; e < m * m; e += WG)
        {
            uint i = j + 1 + e / m, c = j + 1 + e % m;
            if (c <= i)
                a[(k + i) * ld + k + c] =
                    a[(k + i) * ld + k + c] - mul(a[(k + i) * ld + k + j], cj(a[(k + c) * ld + k + j]));
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
        // dlarfg / zlarfg on x = a[j:m, j]
        R part = R(0);
        for (uint i = j + 1 + t; i < m; i += WG)
            part += norm2(a[i * ld + j]);
        R xnorm2 = reduce_sum_real(t, part);
        T alpha = a[j * ld + j];
        R alphr = re(alpha);
        T tj = ZERO;
        if (xnorm2 != R(0) || !is_zero(alpha - fromr(alphr)))
        {
            R beta = -sign_of(alphr) * sqrt(norm2(alpha) + xnorm2);
#ifdef IS_COMPLEX
            tj = T((beta - alphr) / beta, -alpha.y / beta);
#else
            tj = (beta - alpha) / beta;
#endif
            T scale = dv(ONE, alpha - fromr(beta));
            for (uint i = j + 1 + t; i < m; i += WG)
                a[i * ld + j] = mul(a[i * ld + j], scale);
            barrier();
            if (t == 0)
                a[j * ld + j] = fromr(beta);
        }
        if (t == 0)
            tau[j] = tj;
        barrier();
        if (is_zero(tj))
            continue;
        // Apply H^H = I - conj(tau) v v^H to the panel's later columns.
        T ctj = cj(tj);
        for (uint c = j + 1; c < p.k + p.b; c++)
        {
            T dot = ZERO;
            for (uint i = j + t; i < m; i += WG)
            {
                T v = i == j ? ONE : a[i * ld + j];
                dot += mul(cj(v), a[i * ld + c]);
            }
            T w = mul(reduce_sum(t, dot), ctj);
            for (uint i = j + t; i < m; i += WG)
            {
                T v = i == j ? ONE : a[i * ld + j];
                a[i * ld + c] = a[i * ld + c] - mul(v, w);
            }
            barrier();
        }
    }
}

// T (b x b, upper) with H_k ... H_{k+b-1} = I - V T V^H: T[j, j] = tau_j,
// T[0:j, j] = -tau_j T[0:j, 0:j] (V[:, 0:j]^H v_j).
void larft(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    T* tau = p.aux + z * p.saux;
    T* tm = p.aux2 + z * p.saux2;
    uint m = p.rows, ld = p.ld, k = p.k, b = p.b;
    for (uint j = 0; j < b; j++)
    {
        T tj = tau[k + j];
        // y[i] = V[:, i]^H v_j for i < j, over rows k + j .. m (v_j is zero above).
        for (uint i = 0; i < j; i++)
        {
            T part = ZERO;
            for (uint r = k + j + t; r < m; r += WG)
            {
                T vi = a[r * ld + k + i];  // r > k + i: below V[:, i]'s unit diagonal
                T vj = r == k + j ? ONE : a[r * ld + k + j];
                part += mul(cj(vi), vj);
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
                T s = ZERO;
                for (uint l = i; l < j; l++)
                    s += mul(tm[i * b + l], tm[l * b + j]);
                sval[i] = -mul(tj, s);
            }
            for (uint i = 0; i < j; i++)
                tm[i * b + j] = sval[i];
            tm[j * b + j] = tj;
            for (uint i = j + 1; i < b; i++)
                tm[i * b + j] = ZERO;
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
            p.aux2[z * p.saux2 + pair] = ZERO;
        return;  // the dummy partner of an odd column count
    }
    R alpha = R(0), beta = R(0);
    T gamma = ZERO;
    for (uint i = t; i < m; i += WG)
    {
        T x = w[i * ld + col_p], y = w[i * ld + col_q];
        alpha += norm2(x);
        beta += norm2(y);
        gamma += mul(cj(x), y);
    }
    reduce_sum3(t, alpha, beta, gamma);
    R g = mag(gamma);
    R off = (alpha > R(0) && beta > R(0)) ? g / sqrt(alpha * beta) : R(0);
    if (t == 0)
        p.aux2[z * p.saux2 + pair] = fromr(off);
    if (g == R(0))
        return;
#ifdef IS_COMPLEX
    // Column q times conj(phase of gamma): then w_p^H w_q' = |gamma|.
    T ph = cj(gamma / g);
    R gr = g;
#else
    T ph = T(1);
    R gr = gamma;
#endif
    R zeta = (beta - alpha) / (R(2) * gr);
    R tt = sign_of(zeta) / (abs(zeta) + sqrt(R(1) + zeta * zeta));
    R c = R(1) / sqrt(R(1) + tt * tt), s = c * tt;
    for (uint i = t; i < m; i += WG)
    {
        T x = w[i * ld + col_p], y = mul(ph, w[i * ld + col_q]);
        w[i * ld + col_p] = c * x - s * y;
        w[i * ld + col_q] = s * x + c * y;
    }
    for (uint i = t; i < n; i += WG)
    {
        T x = v[i * n + col_p], y = mul(ph, v[i * n + col_q]);
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
    R c = R(1), s = R(0);
    T ph = ONE;  // A[p, q] / |A[p, q]|
    if (i_p < n && i_q < n)
    {
        T apq = a[i_p * ld + i_q];
        if (!is_zero(apq))
        {
#ifdef IS_COMPLEX
            R g = mag(apq);
            ph = apq / g;
#else
            R g = apq;
#endif
            R theta = (re(a[i_q * ld + i_q]) - re(a[i_p * ld + i_p])) / (R(2) * g);
            R tt = sign_of(theta) / (abs(theta) + sqrt(theta * theta + R(1)));
            c = R(1) / sqrt(tt * tt + R(1));
            s = tt * c;
        }
    }
    if (t == 0)
    {
#ifdef IS_COMPLEX
        rot[2 * pair] = T(c, s);
        rot[2 * pair + 1] = ph;
#else
        rot[2 * pair] = c;
        rot[2 * pair + 1] = s;
#endif
    }
    if (s == R(0))
        return;
    // A <- A J, V <- V J for J = diag(1, conj(ph)) [[c, s], [-s, c]] in the
    // (p, q) plane.
    T cph = cj(ph);
    for (uint i = t; i < n; i += WG)
    {
        T x = a[i * ld + i_p], y = mul(cph, a[i * ld + i_q]);
        a[i * ld + i_p] = c * x - s * y;
        a[i * ld + i_q] = s * x + c * y;
        T vx = v[i * n + i_p], vy = mul(cph, v[i * n + i_q]);
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
#ifdef IS_COMPLEX
    R c = rot[2 * pair].x, s = rot[2 * pair].y;
    T ph = rot[2 * pair + 1];
#else
    R c = rot[2 * pair], s = rot[2 * pair + 1];
    T ph = ONE;
#endif
    if (s == R(0) || i_p >= n || i_q >= n)
        return;
    // A <- J^H A
    for (uint j = t; j < n; j += WG)
    {
        T x = a[i_p * ld + j], y = mul(ph, a[i_q * ld + j]);
        a[i_p * ld + j] = c * x - s * y;
        a[i_q * ld + j] = s * x + c * y;
    }
}

// ---------------------------------------------------------------- LDL (Bunch-Kaufman)

groupshared int g_kp;
groupshared int g_kstep;

// The diagonal's magnitude as the pivot tests see it (hetf2 reads the real part).
R diag_mag(T x)
{
    return UPPER != 0 ? abs(re(x)) : mag1(x);
}

// The conjugate for hermitian factorizations, the value for symmetric ones.
T hc(T x)
{
    return UPPER != 0 ? cj(x) : x;
}

T real_part(T x)
{
    return fromr(re(x));
}

void ldl(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    T* w = p.aux + z * p.saux;
    uint n = p.rows, ld = p.ld;
    const R alpha = (R(1) + sqrt(R(17))) / R(8);
    uint k = 0;
    while (k < n)
    {
        if (t == 0)
        {
            int kstep = 1;
            uint kp = k;
            R absakk = diag_mag(a[k * ld + k]);
            uint imax = k;
            R colmax = R(0);
            for (uint i = k + 1; i < n; i++)
            {
                R v = mag1(a[i * ld + k]);
                if (v > colmax)
                {
                    colmax = v;
                    imax = i;
                }
            }
            if (max(absakk, colmax) == R(0) || isnan(absakk))
            {
                if (p.info[z] == 0)
                    p.info[z] = int(k) + 1;
                kstep = 0;  // no update (LAPACK continues with the next column)
            }
            else if (!(absakk >= alpha * colmax))
            {
                R rowmax = R(0);
                for (uint j = k; j < imax; j++)
                    rowmax = max(rowmax, mag1(a[imax * ld + j]));
                for (uint j = imax + 1; j < n; j++)
                    rowmax = max(rowmax, mag1(a[j * ld + imax]));
                if (absakk >= alpha * colmax * (colmax / rowmax))
                    kp = k;
                else if (diag_mag(a[imax * ld + imax]) >= alpha * rowmax)
                    kp = imax;
                else
                {
                    kp = imax;
                    kstep = 2;
                }
            }
            g_kp = int(kp);
            g_kstep = kstep;
            uint steps = kstep == 0 ? 1u : uint(kstep);
            if (steps == 1)
                p.piv[z * p.spiv + k] = int(kp) + 1;
            else
            {
                p.piv[z * p.spiv + k] = -(int(kp) + 1);
                p.piv[z * p.spiv + k + 1] = -(int(kp) + 1);
            }
        }
        barrier();
        int kstep = g_kstep;
        uint kp = uint(g_kp);
        barrier();
        if (kstep == 0)
        {
            k += 1;
            continue;
        }
        uint kk = k + uint(kstep) - 1;
        if (kp != kk)
        {
            // Interchange rows and columns kk and kp of the trailing submatrix.
            for (uint i = kp + 1 + t; i < n; i += WG)
            {
                T tmp = a[i * ld + kk];
                a[i * ld + kk] = a[i * ld + kp];
                a[i * ld + kp] = tmp;
            }
            for (uint j = kk + 1 + t; j < kp; j += WG)
            {
                T tmp = hc(a[j * ld + kk]);
                a[j * ld + kk] = hc(a[kp * ld + j]);
                a[kp * ld + j] = tmp;
            }
            if (t == 0)
            {
                if (UPPER != 0)
                    a[kp * ld + kk] = cj(a[kp * ld + kk]);
                T tmp = a[kk * ld + kk];
                a[kk * ld + kk] = a[kp * ld + kp];
                a[kp * ld + kp] = tmp;
                if (kstep == 2)
                {
                    tmp = a[(k + 1) * ld + k];
                    a[(k + 1) * ld + k] = a[kp * ld + k];
                    a[kp * ld + k] = tmp;
                }
            }
            barrier();
        }
        if (UPPER != 0 && t == 0)
        {
            a[k * ld + k] = real_part(a[k * ld + k]);
            if (kstep == 2)
                a[(k + 1) * ld + k + 1] = real_part(a[(k + 1) * ld + k + 1]);
        }
        barrier();
        if (kstep == 1)
        {
            if (k + 1 < n)
            {
                // A22 -= d11 x x^T (x^H), then x *= d11, x = A[k+1.., k].
                T d11 = UPPER != 0 ? fromr(R(1) / re(a[k * ld + k])) : dv(ONE, a[k * ld + k]);
                uint m = n - k - 1;
                for (uint e = t; e < m * m; e += WG)
                {
                    uint i = k + 1 + e / m, j = k + 1 + e % m;
                    if (j <= i)
                    {
                        T upd = a[i * ld + j] + mul(a[i * ld + k], -mul(d11, hc(a[j * ld + k])));
                        a[i * ld + j] = (UPPER != 0 && i == j) ? real_part(upd) : upd;
                    }
                }
                barrier();
                for (uint i = k + 1 + t; i < n; i += WG)
                    a[i * ld + k] = mul(d11, a[i * ld + k]);
                barrier();
            }
        }
        else if (k + 2 < n)
        {
            T d11, d22, d21, dd, tt;
            if (UPPER != 0)
            {
                R d = mag(a[(k + 1) * ld + k]);
                d11 = fromr(re(a[(k + 1) * ld + k + 1]) / d);
                d22 = fromr(re(a[k * ld + k]) / d);
                tt = fromr(R(1) / (re(d11) * re(d22) - R(1)));
                d21 = a[(k + 1) * ld + k] / d;
                dd = fromr(re(tt) / d);
            }
            else
            {
                T e21 = a[(k + 1) * ld + k];
                d11 = dv(a[(k + 1) * ld + k + 1], e21);
                d22 = dv(a[k * ld + k], e21);
                tt = dv(ONE, mul(d11, d22) - ONE);
                dd = dv(tt, e21);
                d21 = ONE;
            }
            for (uint j = k + 2 + t; j < n; j += WG)
            {
                T ajk = a[j * ld + k], ajk1 = a[j * ld + k + 1];
                w[j] = mul(dd, mul(d11, ajk) - mul(d21, ajk1));
                w[n + j] = mul(dd, mul(d22, ajk1) - mul(cj(d21), ajk));
            }
            barrier();
            uint m = n - k - 2;
            for (uint e = t; e < m * m; e += WG)
            {
                uint i = k + 2 + e / m, j = k + 2 + e % m;
                if (j <= i)
                {
                    T upd = a[i * ld + j] - mul(a[i * ld + k], hc(w[j])) - mul(a[i * ld + k + 1], hc(w[n + j]));
                    a[i * ld + j] = (UPPER != 0 && i == j) ? real_part(upd) : upd;
                }
            }
            barrier();
            for (uint j = k + 2 + t; j < n; j += WG)
            {
                a[j * ld + k] = w[j];
                a[j * ld + k + 1] = w[n + j];
            }
            barrier();
        }
        k += uint(kstep);
    }
}

void ldl_solve(uint t, uint z)
{
    T* a = p.a + z * p.sa;
    T* b = p.aux + z * p.saux;
    int* piv = p.piv + z * p.spiv;
    uint n = p.rows, ld = p.ld, nr = p.cols;
    // Indices stay in range on every path, also those a lane does not take
    // (lavapipe's code performs loads there too: see eig.slang's qr): the
    // pivot is decoded before the branch, and k / k + 1 / k - 1 are clamped.
    for (uint c = t; c < nr; c += WG)
    {
        // L D y = P b
        uint k = 0;
        while (k < n)
        {
            int pk = piv[min(k, n - 1)];
            uint kp = min(uint(abs(pk) - 1), n - 1);
            if (pk > 0)
            {
                if (kp != k)
                {
                    T tmp = b[k * nr + c];
                    b[k * nr + c] = b[kp * nr + c];
                    b[kp * nr + c] = tmp;
                }
                T bk = b[k * nr + c];
                for (uint i = k + 1; i < n; i++)
                    b[i * nr + c] = b[i * nr + c] - mul(a[i * ld + k], bk);
                T inv = UPPER != 0 ? fromr(R(1) / re(a[k * ld + k])) : dv(ONE, a[k * ld + k]);
                b[k * nr + c] = mul(inv, b[k * nr + c]);
                k += 1;
            }
            else
            {
                uint k1 = min(k + 1, n - 1);  // (k + 1 < n in a 2 x 2 block)
                if (kp != k1)
                {
                    T tmp = b[k1 * nr + c];
                    b[k1 * nr + c] = b[kp * nr + c];
                    b[kp * nr + c] = tmp;
                }
                T bk0 = b[k * nr + c], bk1 = b[k1 * nr + c];
                for (uint i = k + 2; i < n; i++)
                    b[i * nr + c] = b[i * nr + c] - mul(a[i * ld + k], bk0);
                for (uint i = k + 2; i < n; i++)
                    b[i * nr + c] = b[i * nr + c] - mul(a[i * ld + k1], bk1);
                T akm1k = a[k1 * ld + k];
                T akm1 = dv(a[k * ld + k], hc(akm1k));
                T ak = dv(a[k1 * ld + k1], akm1k);
                T denom = mul(akm1, ak) - ONE;
                T bkm1 = dv(b[k * nr + c], hc(akm1k));
                T bkk = dv(b[k1 * nr + c], akm1k);
                b[k * nr + c] = dv(mul(ak, bkm1) - bkk, denom);
                b[k1 * nr + c] = dv(mul(akm1, bkk) - bkm1, denom);
                k += 2;
            }
        }
        // L^T x = y (L^H for hermitian), then undo the interchanges.
        int kb = int(n) - 1;
        while (kb >= 0)
        {
            uint k2 = uint(max(kb, 0));
            int pk = piv[k2];
            uint kp = min(uint(abs(pk) - 1), n - 1);
            if (pk > 0)
            {
                T s = ZERO;
                for (uint i = k2 + 1; i < n; i++)
                    s += mul(hc(a[i * ld + k2]), b[i * nr + c]);
                b[k2 * nr + c] = b[k2 * nr + c] - s;
                if (kp != k2)
                {
                    T tmp = b[k2 * nr + c];
                    b[k2 * nr + c] = b[kp * nr + c];
                    b[kp * nr + c] = tmp;
                }
                kb -= 1;
            }
            else
            {
                uint km1 = max(k2, 1u) - 1;  // (k2 >= 1 in a 2 x 2 block)
                T s = ZERO, s1 = ZERO;
                for (uint i = k2 + 1; i < n; i++)
                {
                    s += mul(hc(a[i * ld + k2]), b[i * nr + c]);
                    s1 += mul(hc(a[i * ld + km1]), b[i * nr + c]);
                }
                b[k2 * nr + c] = b[k2 * nr + c] - s;
                b[km1 * nr + c] = b[km1 * nr + c] - s1;
                if (kp != k2)
                {
                    T tmp = b[k2 * nr + c];
                    b[k2 * nr + c] = b[kp * nr + c];
                    b[kp * nr + c] = tmp;
                }
                kb -= 2;
            }
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
    case FACTOR_LU_PANEL: lu_panel(t, gid.x); break;
    case FACTOR_LASWP: laswp(t, gid.x); break;
    case FACTOR_TRI_INV: tri_inv(t, gid.x); break;
    case FACTOR_CHOL_BLOCK: chol_block(t, gid.x); break;
    case FACTOR_QR_PANEL: qr_panel(t, gid.x); break;
    case FACTOR_LARFT: larft(t, gid.x); break;
    case FACTOR_SVD_ROUND: svd_round(t, gid.x, gid.y); break;
    case FACTOR_EIG_COLS: eig_cols(t, gid.x, gid.y); break;
    case FACTOR_LDL: ldl(t, gid.x); break;
    case FACTOR_LDL_SOLVE: ldl_solve(t, gid.x); break;
    default: eig_rows(t, gid.x, gid.y); break;  // FACTOR_EIG_ROWS
    }
}
