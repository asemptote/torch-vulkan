// axis T: f32=float f64=double c64=float2 c128=double2
//
// Blocked reduction of a Hermitian matrix to real tridiagonal form, for
// csrc/dense.cpp's eigh: LAPACK's zhetrd with zlatrd's panels (lower). A
// panel of NB columns is reduced one column at a time against the matrix as
// it was at the panel's start (A22 stale); the panel's reflectors V and the
// vectors W with A_now = A_start - V W^H - W V^H are kept, and after the
// panel the host updates the trailing matrix by GEMM (A22 -= V W^H + W V^H,
// zher2k). A step therefore reads A22 once (SYMV) instead of reading it and
// writing it back (tridiag.slang's SYMV and RANK2).
//
// Matrices are row-major and batched (matrix z at a + z * sa, entry (i, j) at
// [i * ld + j]), both triangles kept. Per matrix z, indexed by global row:
// tau[z n + k], e[z n + k] (beta, real); the panel's V and W column j (global
// column k0 + j) at v + (z NB + j) n and w + (z NB + j) n; c[z n + ...], the
// column being reflected (contiguous); y[z n + ...] = A22_start v; part, the
// SYMV workgroups' partial sums (PS per workgroup, `groups` per matrix).
//
// OP (LATRD_*), step k (panel column i = k - k0; A22 = A[k+1:n, k+1:n],
// m = n - k - 1 rows):
//   SYMV  every workgroup forms the reflector of column k from c (zlarfg as
//         tridiag.slang's REFLECT: H = I - tau v v^H, v_0 = 1; the same
//         arithmetic in each, so the same v); workgroup 0 also writes v to
//         A[k+1:n, k] and V's column i, tau[k] and e[k]. Then y = A22 v,
//         ROWS rows per workgroup, 32 lanes along each row, with partial
//         sums over its rows of y^H v, V^H v and W^H v. Grid (m / ROWS,
//         batch).
//   FIN   W's column i: with x1 = V^H v, x2 = W^H v (V, W: the panel's
//         first i columns) and u = A_now v = y - V x2 - W x1,
//         W_i = tau u - 1/2 |tau|^2 (u^H v) v, u^H v = y^H v - x2^H x1 -
//         x1^H x2 (zlatrd's w). The sums come from SYMV's partials, added
//         in the same order by every workgroup. Unless i is the panel's last
//         column, column k + 1 of A (rows k + 1..) is brought up to date,
//         A -= V W^H + W V^H with W_i, into A and c, for the next SYMV. A
//         thread per row; grid (m / WG, batch).

#include "opcodes.slang"

#if defined(T_c64) || defined(T_c128)
#define IS_COMPLEX 1
#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif
#include "_complex.slang"
typedef S R;
T mul(T a, T b) { return cmul(a, b); }
T dv(T a, T b) { return cdiv(a, b); }
T cj(T a) { return cconj(a); }
R re(T a) { return a.x; }
R mag1(T a) { return abs(a.x) + abs(a.y); }
R norm2(T a) { return a.x * a.x + a.y * a.y; }
T fromr(R r) { return T(r, R(0)); }
T scale(T a, R s) { return T(a.x * s, a.y * s); }
bool is_zero(T a) { return a.x == R(0) && a.y == R(0); }
#else
typedef T R;
T mul(T a, T b) { return a * b; }
T dv(T a, T b) { return a / b; }
T cj(T a) { return a; }
R re(T a) { return a; }
R mag1(T a) { return abs(a); }
R norm2(T a) { return a * a; }
T fromr(R r) { return r; }
T scale(T a, R s) { return a * s; }
bool is_zero(T a) { return a == T(0); }
#endif
#define ONE fromr(R(1))
#define ZERO fromr(R(0))

#define NB 32         // panel columns (kLatrdNb in csrc/dense.cpp)
#define PS (2 * NB)   // partial sums per SYMV workgroup: y^H v, then x1 at 1 + j, x2 at NB + j
#define WG 256
#define ROWS (WG / 32)  // SYMV rows per workgroup (kLatrdRows)

struct Params
{
    T* a;
    T* tau;
    R* e;
    T* v;
    T* w;
    T* c;
    T* y;
    T* part;
    uint n;
    uint ld;
    uint sa;      // elements between matrices of a
    uint k;       // the step
    uint i;       // its column in the panel
    uint groups;  // SYMV workgroups per matrix in part
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = LATRD_SYMV;

groupshared T sval[WG];
groupshared R rval[WG];
groupshared T srow[2 * NB];  // FIN: row k + 1 of V and W; SYMV: the rows' y
groupshared T sx[PS];        // FIN: the summed partials

void barrier()
{
    AllMemoryBarrierWithGroupSync();
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

R reduce_max_real(uint t, R v)
{
    rval[t] = v;
    barrier();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
            rval[t] = max(rval[t], rval[t + s]);
        barrier();
    }
    R r = rval[0];
    barrier();
    return r;
}

R min_normal()
{
#if defined(T_f64) || defined(T_c128)
    return asdouble(0u, 0x00100000u);  // 2^-1022
#else
    return asfloat(0x00800000u);  // 2^-126
#endif
}

R sign_of(R x)
{
    return x < R(0) ? R(-1) : R(1);
}

// ---------------------------------------------------------------- SYMV

void symv(uint t, uint gx, uint z)
{
    uint n = p.n, ld = p.ld, k = p.k, top = k + 1, m = n - top, i = p.i;
    T* a = p.a + z * p.sa;
    T* c = p.c + z * n;
    // zlarfg on alpha = c[top], x = c[top+1:n] (tridiag.slang's REFLECT):
    // norms scaled by the largest |part|; a column whose largest part is
    // subnormal counts as zero (H = I, x left as it is).
    T alpha = c[top];
    R amax = R(0);
    for (uint r = top + 1 + t; r < n; r += WG)
        amax = max(amax, mag1(c[min(r, n - 1)]));
    amax = max(reduce_max_real(t, amax), mag1(alpha));
    R part = R(0);
    if (amax >= min_normal())
    {
        for (uint r = top + 1 + t; r < n; r += WG)
            part += norm2(scale(c[min(r, n - 1)], R(1) / amax));
    }
    R xnorm2 = reduce_sum_real(t, part);
    // (beta, tau and the scale of x by thread 0 alone, shared: see REFLECT)
    if (t == 0)
    {
        R alphr = re(alpha);
        T tk = ZERO, sc = ZERO;
        R beta = alphr;
        if (amax >= min_normal() && (xnorm2 != R(0) || !is_zero(alpha - fromr(alphr))))
        {
            beta = -sign_of(alphr) * amax * sqrt(norm2(scale(alpha, R(1) / amax)) + xnorm2);
#ifdef IS_COMPLEX
            tk = T((beta - alphr) / beta, -alpha.y / beta);
#else
            tk = (beta - alpha) / beta;
#endif
            sc = dv(ONE, alpha - fromr(beta));
        }
        sval[0] = sc;
        if (gx == 0)
        {
            p.tau[z * n + k] = tk;
            p.e[z * n + k] = beta;
        }
    }
    barrier();
    T sc = sval[0];
    bool scaled = !is_zero(sc);
    barrier();
    // v_r (global row r >= top): 1 at top, then x scaled
    T* vi = p.v + (z * NB + i) * n;
    if (gx == 0)
    {
        for (uint r = top + t; r < n; r += WG)
        {
            T xr = c[r];
            T vr = r == top ? ONE : (scaled ? mul(xr, sc) : xr);
            a[r * ld + k] = vr;
            vi[r] = vr;
        }
    }
    // y = A22 v, a row per 32 lanes
    uint row = gx * ROWS + t / 32, lane = t % 32;
    T acc = ZERO;
    if (row < m)
    {
        uint ar = (top + row) * ld + top;
        for (uint j = lane; j < m; j += 32)
        {
            T xj = c[top + j];
            T vj = j == 0 ? ONE : (scaled ? mul(xj, sc) : xj);
            acc += mul(a[ar + j], vj);
        }
    }
    sval[t] = acc;
    barrier();
    for (uint s = 16; s > 0; s >>= 1)
    {
        if (lane < s)
            sval[t] += sval[t + s];
        barrier();
    }
    if (lane == 0)
    {
        T yr = sval[t];
        if (row < m)
            p.y[z * n + top + row] = yr;
        srow[t / 32] = yr;
    }
    barrier();
    // partial sums over the workgroup's rows: y^H v (q = 0), V_j^H v
    // (q = 1 + j) and W_j^H v (q = NB + j), j < i
    uint q = t;
    bool active = q == 0 || (q >= 1 && q <= i) || (q >= NB && q < NB + i);
    if (active)
    {
        T* col = q == 0 ? p.v : (q < NB ? p.v + (z * NB + q - 1) * n : p.w + (z * NB + q - NB) * n);
        T s = ZERO;
        for (uint rl = 0; rl < ROWS; rl++)
        {
            uint r = gx * ROWS + rl;
            if (r < m)
            {
                T xr = c[top + r];
                T vr = r == 0 ? ONE : (scaled ? mul(xr, sc) : xr);
                T lhs = q == 0 ? srow[rl] : col[top + r];
                s += mul(cj(lhs), vr);
            }
        }
        p.part[(z * p.groups + gx) * PS + q] = s;
    }
}

// ---------------------------------------------------------------- FIN

// u_r = y_r - V_r x2 - W_r x1 (row r of the panel's first i columns)
T u_at(uint z, uint r, uint i, uint n)
{
    T u = p.y[z * n + r];
    for (uint j = 0; j < i; j++)
        u -= mul(p.v[(z * NB + j) * n + r], sx[NB + j]) + mul(p.w[(z * NB + j) * n + r], sx[1 + j]);
    return u;
}

void fin(uint t, uint gx, uint z)
{
    uint n = p.n, k = p.k, top = k + 1, m = n - top, i = p.i;
    uint groups = (m + ROWS - 1) / ROWS;
    // the partial sums, PS of them per SYMV workgroup: thread t adds
    // quantity t % PS of every (WG / PS)-th workgroup, then the WG / PS
    // threads of a quantity in a fixed order
    {
        uint q = t % PS, ch = t / PS;
        T s = ZERO;
        for (uint g = ch; g < groups; g += WG / PS)
            s += p.part[(z * p.groups + g) * PS + q];
        sval[t] = s;
    }
    barrier();
    if (t < PS)
    {
        T s = sval[t];
        for (uint ch = 1; ch < WG / PS; ch++)
            s += sval[ch * PS + t];
        sx[t] = s;
    }
    if (t < i)
    {
        srow[t] = p.v[(z * NB + t) * n + top];
        srow[NB + t] = p.w[(z * NB + t) * n + top];
    }
    barrier();
    T tk = p.tau[z * n + k];
    if (t == 0)
    {
        // u^H v = y^H v - x2^H x1 - x1^H x2
        T uv = sx[0];
        for (uint j = 0; j < i; j++)
            uv -= mul(cj(sx[NB + j]), sx[1 + j]) + mul(cj(sx[1 + j]), sx[NB + j]);
        T alpha = scale(uv, R(-0.5) * norm2(tk));
        sval[0] = alpha;
        // W_i at row top (v = 1 there), for column k + 1's update
        sval[1] = mul(tk, u_at(z, top, i, n)) + alpha;
    }
    barrier();
    T alpha = sval[0], wtop = sval[1];
    uint r = top + gx * WG + t;
    if (r >= n)
        return;
    T* vi = p.v + (z * NB + i) * n;
    T vr = vi[r];
    T wr = r == top ? wtop : mul(tk, u_at(z, r, i, n)) + mul(alpha, vr);
    p.w[(z * NB + i) * n + r] = wr;
    if (i + 1 < NB)
    {
        // column top of A: A[r, top] -= sum_j V_j[r] conj(W_j[top]) + W_j[r] conj(V_j[top]), j <= i
        T* a = p.a + z * p.sa;
        uint at = r * p.ld + top;
        T x = a[at] - mul(vr, cj(wtop)) - wr;  // (V_i[top] = 1)
        for (uint j = 0; j < i; j++)
            x -= mul(p.v[(z * NB + j) * n + r], cj(srow[NB + j])) + mul(p.w[(z * NB + j) * n + r], cj(srow[j]));
        a[at] = x;
        p.c[z * n + r] = x;
    }
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint t = tid.x;
    switch (OP)
    {
    case LATRD_SYMV: symv(t, gid.x, gid.y); break;
    case LATRD_FIN: fin(t, gid.x, gid.y); break;
    }
}
