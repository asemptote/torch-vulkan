// axis T: f32=float f64=double c64=float2 c128=double2
//
// The panel of csrc/dense.cpp's blocked Householder QR (LAPACK's geqrf) over
// many workgroups, two dispatches per column. factor.slang's QR_PANEL
// factored a panel on one workgroup per matrix, a column after another: for a
// tall-skinny matrix (1000 x 64) the panel is the whole matrix, and that was
// 20-40 times CUDA's time (#62).
//
// Matrices are row-major and batched (matrix z at a + z * sa, entry (i, j) at
// [i * ld + j]); the panel is columns [k, k + b), b <= 64, rows [k, rows);
// tau[z * ntau + j]. Rows are split in blocks of RW, one per workgroup, from
// row j. Per matrix, refl[z * 4 ...] holds column j's beta, the scale of x
// and tau, for UPD.
//
// OP (GEQRF_*), step j, grid (groups, batch):
//   DOT  every workgroup forms column j's reflector (zlarfg as QR_PANEL: H =
//        I - tau v v^H, v_0 = 1, norms scaled by the largest |part|, a column
//        whose largest part is subnormal counts as zero, H = I; the same
//        arithmetic in each) and leaves the partial sums of v^H a[:, c] over
//        its rows for the panel's later columns c; workgroup 0 records beta,
//        the scale and tau.
//   UPD  every workgroup adds up the partial sums (the same order in each),
//        a[i, c] -= v_i conj(tau) (v^H a[:, c]) for its rows i and the
//        panel's later columns c, then stores v (x scaled; beta at the
//        diagonal) in column j.

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

struct Params
{
    T* a;
    T* tau;
    T* part;  // [batch, groups, PB] partial sums
    T* refl;  // [batch, 4]: beta, scale, tau
    uint rows;
    uint ld;
    uint sa;
    uint k;
    uint b;
    uint j;
    uint groups;
    uint ntau;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = GEQRF_DOT;

#define WG 256
#define PB 64              // panel columns (kBlock in csrc/dense.cpp)
#define LANES (WG / PB)    // row lanes per column
#define RW 256             // rows per workgroup (kGeqrfRows)

groupshared R rval[WG];
groupshared T sval[WG];
groupshared T sv[RW];
groupshared T sref[3];

void barrier()
{
    GroupMemoryBarrierWithGroupSync();
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

void dot(uint t, uint gx, uint z)
{
    T* a = p.a + z * p.sa;
    uint m = p.rows, ld = p.ld, j = p.j, cend = p.k + p.b;
    // column j's reflector, from all of its rows
    T alpha = a[j * ld + j];
    R amax = R(0);
    for (uint i = j + 1 + t; i < m; i += WG)
        amax = max(amax, mag1(a[min(i, m - 1) * ld + j]));
    amax = max(reduce_max_real(t, amax), mag1(alpha));
    R part = R(0);
    if (amax >= min_normal())
    {
        for (uint i = j + 1 + t; i < m; i += WG)
            part += norm2(scale(a[min(i, m - 1) * ld + j], R(1) / amax));
    }
    R xnorm2 = reduce_sum_real(t, part);  // (|x|^2 / amax^2)
    // (beta, tau and the scale by thread 0 alone, shared: per-lane float64
    // divisions came out infinite in partly active SIMD threads on the UHD 620)
    if (t == 0)
    {
        R alphr = re(alpha);
        T tj = ZERO, sc = ZERO;
        R beta = alphr;
        if (amax >= min_normal() && (xnorm2 != R(0) || !is_zero(alpha - fromr(alphr))))
        {
            beta = -sign_of(alphr) * amax * sqrt(norm2(scale(alpha, R(1) / amax)) + xnorm2);
#ifdef IS_COMPLEX
            tj = T((beta - alphr) / beta, -alpha.y / beta);
#else
            tj = (beta - alpha) / beta;
#endif
            sc = dv(ONE, alpha - fromr(beta));
        }
        sref[0] = fromr(beta);
        sref[1] = sc;
        sref[2] = tj;
        if (gx == 0)
        {
            p.refl[z * 4] = sref[0];
            p.refl[z * 4 + 1] = sc;
            p.refl[z * 4 + 2] = tj;
            p.tau[z * p.ntau + j] = tj;
        }
    }
    barrier();
    T sc = sref[1], tj = sref[2];
    if (is_zero(tj) || j + 1 >= cend)
        return;  // (uniform: H = I, or no later column in the panel)
    // v for this workgroup's rows, then partial sums of v^H a[:, c]
    uint r0 = j + gx * RW, r1 = min(r0 + RW, m);
    for (uint i = r0 + t; i < r1; i += WG)
        sv[i - r0] = i == j ? ONE : mul(a[i * ld + j], sc);
    barrier();
    uint co = t % PB, lane = t / PB, c = j + 1 + co;
    T acc = ZERO;
    if (c < cend)
    {
        for (uint i = r0 + lane; i < r1; i += LANES)
            acc += mul(cj(sv[i - r0]), a[i * ld + c]);
    }
    sval[t] = acc;
    barrier();
    if (t < PB)
    {
        T s = sval[t];
        for (uint l = 1; l < LANES; l++)
            s += sval[l * PB + t];
        p.part[(z * p.groups + gx) * PB + t] = s;
    }
}

void upd(uint t, uint gx, uint z)
{
    T* a = p.a + z * p.sa;
    uint m = p.rows, ld = p.ld, j = p.j, cend = p.k + p.b;
    T beta = p.refl[z * 4], sc = p.refl[z * 4 + 1], tj = p.refl[z * 4 + 2];
    if (is_zero(tj))
        return;  // (uniform: H = I leaves the column as it is)
    uint groups = (m - j + RW - 1) / RW;
    uint r0 = j + gx * RW, r1 = min(r0 + RW, m);
    if (j + 1 < cend)
    {
        // w_c = conj(tau) v^H a[:, c], the partial sums in a fixed order
        if (t < PB)
        {
            T s = ZERO;
            for (uint g = 0; g < groups; g++)
                s += p.part[(z * p.groups + g) * PB + t];
            sval[t] = mul(cj(tj), s);
        }
        for (uint i = r0 + t; i < r1; i += WG)
            sv[i - r0] = i == j ? ONE : mul(a[i * ld + j], sc);
        barrier();
        uint co = t % PB, lane = t / PB, c = j + 1 + co;
        if (c < cend)
        {
            T w = sval[co];
            for (uint i = r0 + lane; i < r1; i += LANES)
                a[i * ld + c] = a[i * ld + c] - mul(sv[i - r0], w);
        }
        barrier();
    }
    // column j: v's tail (x scaled) below the diagonal, beta on it
    for (uint i = r0 + t; i < r1; i += WG)
        a[i * ld + j] = i == j ? beta : mul(a[i * ld + j], sc);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    switch (OP)
    {
    case GEQRF_DOT: dot(lid.x, gid.x, gid.y); break;
    case GEQRF_UPD: upd(lid.x, gid.x, gid.y); break;
    }
}
