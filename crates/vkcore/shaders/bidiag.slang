// axis T: f32=float f64=double c64=float2 c128=double2
//
// One Householder reflector (LAPACK's zlarfg) per matrix, for
// csrc/dense.cpp's reduction to bidiagonal form (zgebd2): on the vector
// a[start + i stride], i < len, of each matrix (at a + z sa), alpha = its
// first entry, x the rest: H = I - tau v v^H with H^H (alpha, x) = (beta,
// 0), beta real; x is overwritten by v's tail (v_0 = 1, written in alpha's
// place), tau goes to tau[z ntau + k] and beta to beta[z ntau + k]. Norms
// are scaled by the largest |part|; a vector whose largest part is subnormal
// counts as zero (H = I), as in tridiag.slang's REFLECT. One workgroup per
// matrix.

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
    R* beta;
    uint sa;      // elements between matrices of a
    uint start;   // alpha's offset in a matrix
    uint len;     // alpha and x
    uint stride;  // between the vector's entries
    uint k;       // tau's and beta's index
    uint ntau;    // entries of tau and beta per matrix
};

[[vk::push_constant]] ConstantBuffer<Params> p;

#define WG 256
groupshared T sval[WG];
groupshared R rval[WG];

R reduce(uint t, R v, bool take_max)
{
    rval[t] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = WG / 2; s > 0; s >>= 1)
    {
        if (t < s)
            rval[t] = take_max ? max(rval[t], rval[t + s]) : rval[t] + rval[t + s];
        GroupMemoryBarrierWithGroupSync();
    }
    R r = rval[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

#if defined(T_f64) || defined(T_c128)
R min_normal() { return R(2.2250738585072014e-308l); }
#else
R min_normal() { return R(1.17549435e-38f); }
#endif
R sign_of(R x) { return x < R(0) ? R(-1) : R(1); }

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint t = tid.x, z = gid.x;
    T* a = p.a + z * p.sa + p.start;
    uint len = p.len, st = p.stride;
    T alpha = a[0];
    R amax = R(0);
    for (uint i = 1 + t; i < len; i += WG)
        amax = max(amax, mag1(a[i * st]));
    amax = max(reduce(t, amax, true), mag1(alpha));
    R part = R(0);
    if (amax >= min_normal())
    {
        for (uint i = 1 + t; i < len; i += WG)
            part += norm2(scale(a[i * st], R(1) / amax));
    }
    R xnorm2 = reduce(t, part, false);  // |x|^2 / amax^2
    // beta, tau and the scale of x on thread 0 alone (per-lane float64
    // divisions and square roots came out infinite in some lanes of a partly
    // active SIMD thread on the UHD 620: tridiag.slang)
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
        p.tau[z * p.ntau + p.k] = tk;
        p.beta[z * p.ntau + p.k] = beta;
    }
    GroupMemoryBarrierWithGroupSync();
    T sc = sval[0];
    if (!is_zero(sc))
    {
        for (uint i = 1 + t; i < len; i += WG)
            a[i * st] = mul(a[i * st], sc);
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
        a[0] = ONE;  // v_0
}
