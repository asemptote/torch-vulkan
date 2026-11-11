// Float math on a scalar type FT under one set of names (m_exp, m_log, ...):
// float for kernels with T_f32 (or FMATH_F32, e.g. complex64 kernels),
// double for T_f64 (or FMATH_F64).
//
// float32 uses _math32.slang (Cephes' algorithms) rather than the GPU's
// GLSL.std.450 transcendentals, whose accuracy varies by driver (lavapipe's
// asin is off by 4e-4, its tanh near 0 by 6%); only sqrt, exp2, log2 and
// special values come from built-ins. float64 uses the software library in
// _math64*.slang, since SPIR-V has no 64-bit transcendentals. A float32
// module never contains a double, so it runs on GPUs without float64.

// LIT(c): the constant c at FT's precision (a bare literal is float32).
#if defined(T_f64) || defined(FMATH_F64)
#define LIT(c) c##l
#else
#define LIT(c) c##f
#endif

#if defined(T_f32) || defined(FMATH_F32)
#include "_wide.slang"
#include "_math32.slang"

typedef float FT;  // the scalar type of these functions

typedef uint UBITS;
static const float M_PI = 3.14159265358979f;
static const float M_LN2 = 0.693147180559945f;
static const float M_INV_LN2 = 1.44269504088896f;
static const float M_INV_LN10 = 0.434294481903252f;

float m_inf() { return f32_inf(); }
float m_nan() { return f32_nan(); }
bool m_signbit(float x) { return (bit_cast<uint>(x) >> 31) != 0; }

float m_exp(float x) { return exp32(x); }
float m_exp2(float x) { return exp2(x); }
float m_expm1(float x) { return expm1_32(x); }
float m_log(float x) { return log32(x); }
float m_log2(float x) { return log2(x); }
float m_log10(float x) { return m_log(x) * M_INV_LN10; }
float m_log1p(float x) { return log1p32(x); }
// sin, cos and tan by reduction to [-pi/4, pi/4] (r, quadrant n) and the
// _math32 polynomials: the built-ins can be off by ~1e-5 for moderate
// arguments (sin, cos) or ~1e-4 (tan), and are not reduced exactly for
// large ones. Cody-Waite below |x| = 800, Payne-Hanek above.

// (2/pi) / 4 in 32-bit words, a zero word first (checked against pi computed
// with Machin's formula).
static const uint TWO_OVER_PI_32[9] = {0x00000000u, 0x28be60dbu, 0x9391054au, 0x7f09d5f4u, 0x7d4d3770u, 0x36d8a566u,
                                       0x4f10e410u, 0x7f9458eau, 0xf7aef158u};

// Payne-Hanek for finite |x| >= 800, after N. Juffa: x times 96 bits of 2/pi
// (chosen by x's exponent) in 2.62 fixed point, rounded to the nearest
// quadrant.
float reduce_pio2_32_large(float x, out int n)
{
    uint bits = bit_cast<uint>(abs(x));
    uint e = ((bits >> 23) & 0xffu) - 126u;              // |x| = m 2^e, m in [0.5, 1)
    uint ia = ((bits & 0x7fffffu) | 0x800000u) << 8;     // m 2^32
    uint i = e >> 5, s = e & 31u;
    uint w0 = TWO_OVER_PI_32[i], w1 = TWO_OVER_PI_32[i + 1], w2 = TWO_OVER_PI_32[i + 2], w3 = TWO_OVER_PI_32[i + 3];
    uint hi = s == 0u ? w0 : (w0 << s) | (w1 >> (32u - s));
    uint mid = s == 0u ? w1 : (w1 << s) | (w2 >> (32u - s));
    uint lo = s == 0u ? w2 : (w2 << s) | (w3 >> (32u - s));
    // (the 64-bit fixed-point product as two 32-bit words, p = (lo, hi): no
    // 64-bit integers, which Mali GPUs lack)
    uint2 p = mul_wide32(ia, lo);
    p = add_wide32(mul_wide32(ia, mid), p.y);
    p.y += ia * hi;
    int q = int(p.y >> 30);
    p.y &= 0x3fffffffu;
    if ((p.y & 0x20000000u) != 0u)  // fraction >= 1/2: the next quadrant
    {
        p.y -= 0x40000000u;  // (now negative, two's complement)
        q += 1;
    }
    float r = int64_words_to_float(p.x, p.y) * 3.4061215800865545e-19f;  // pi/2 2^-62
    n = x < 0.0f ? -q : q;
    return x < 0.0f ? -r : r;
}
float reduce_pio2_32(float x, out int n)
{
    if (!(abs(x) < 800.0f))
        return reduce_pio2_32_large(x, n);
    float k = round(x * 0.636619772367581f);
    n = int(k);
    return ((x - k * 1.5703125f) - k * 4.837512969970703125e-4f) - k * 7.54978995489188216e-8f;  // (Cephes: k c1, k c2 exact)
}
float m_sin(float x)
{
    if (isnan(x) || isinf(x)) return m_nan();
    int n;
    float r = reduce_pio2_32(x, n);
    float v = (n & 1) == 0 ? sin_poly32(r) : cos_poly32(r);
    return (n & 2) == 0 ? v : -v;
}
float m_cos(float x)
{
    if (isnan(x) || isinf(x)) return m_nan();
    int n;
    float r = reduce_pio2_32(x, n);
    float v = (n & 1) == 0 ? cos_poly32(r) : sin_poly32(r);
    return ((n + 1) & 2) == 0 ? v : -v;
}
float m_tan(float x)
{
    if (isnan(x) || isinf(x)) return m_nan();
    int n;
    float r = reduce_pio2_32(x, n);
    float s = sin_poly32(r), c = cos_poly32(r);
    return (n & 1) == 0 ? s / c : -c / s;
}
float m_asin(float x) { return abs(x) <= 1.0f ? asin32(x) : m_nan(); }
// acos via asin, keeping relative accuracy near x = 1 (as Cephes' acosf).
float m_acos(float x)
{
    if (!(abs(x) <= 1.0f)) return m_nan();
    if (abs(x) <= 0.5f) return (0.5f * M32_PI_HI - asin32(x)) + 0.5f * M32_PI_LO;
    float r = 2.0f * asin32(sqrt(0.5f * (1.0f - abs(x))));
    return x > 0.0f ? r : (M32_PI_HI - r) + M32_PI_LO;
}
float m_atan(float x) { return atan32(x); }

// Cephes' tanhf.
float m_tanh(float x)
{
    float a = abs(x), r;
    if (a > 44.0f) r = 1.0f;
    else if (a >= 0.625f) r = 1.0f - 2.0f / (exp32(a + a) + 1.0f);
    else
    {
        float z = a * a;
        r = ((((-5.70498872745e-3f * z + 2.06390887954e-2f) * z - 5.37397155531e-2f) * z + 1.33314422036e-1f) * z
             - 3.33332819422e-1f) * z * a + a;
    }
    return m_signbit(x) ? -r : r;
}

float m_sigmoid(float x) { return 1.0f / (1.0f + exp32(-x)); }

// Cephes' sinhf and coshf; e^|x| / 2 as (e^(|x|/2) / 2) e^(|x|/2) where e^|x| overflows.
float m_sinh(float x)
{
    float a = abs(x), r;
    if (a > 88.0f)
    {
        float h = exp32(0.5f * a);
        r = (0.5f * h) * h;
    }
    else if (a > 1.0f)
    {
        float e = exp32(a);
        r = 0.5f * e - 0.5f / e;
    }
    else
    {
        float z = a * a;
        r = ((2.03721912945e-4f * z + 8.33028376239e-3f) * z + 1.66667160211e-1f) * z * a + a;
    }
    return m_signbit(x) ? -r : r;
}

float m_cosh(float x)
{
    float a = abs(x);
    if (a > 88.0f)
    {
        float h = exp32(0.5f * a);
        return (0.5f * h) * h;
    }
    float e = exp32(a);
    return 0.5f * e + 0.5f / e;
}

float m_asinh(float x)
{
    float a = abs(x), r;
    if (a > 1e18f) r = log32(a) + M_LN2;
    else r = log1p32(a + a * a / (1.0f + sqrt(1.0f + a * a)));
    return m_signbit(x) ? -r : r;
}

float m_acosh(float x)
{
    if (x < 1.0f) return f32_nan();
    if (x > 1e18f) return log32(x) + M_LN2;
    float t = x - 1.0f;
    return log1p32(t + sqrt(2.0f * t + t * t));
}

float m_atanh(float x)
{
    float a = abs(x);
    float r = 0.5f * log1p32(2.0f * a / (1.0f - a));
    return m_signbit(x) ? -r : r;
}

float m_erf(float x) { return erf32(x); }
float m_erfc(float x) { return erfc32(x); }
float m_erfinv(float x) { return erfinv32(x); }
float m_lgamma(float x) { return lgamma32(x); }
float m_digamma(float x) { return digamma32(x); }
float m_sinc(float x) { return sinc32(x); }

float m_atan2(float y, float x)
{
    if (isnan(x) || isnan(y)) return f32_nan();
    bool xneg = m_signbit(x);
    float sy = m_signbit(y) ? -1.0f : 1.0f;
    if (isinf(x) && isinf(y)) return sy * (xneg ? 0.75f * M_PI : 0.25f * M_PI);
    if (y == 0.0f) return xneg ? sy * M_PI : y;  // atan2(+-0, -x) = +-pi, atan2(+-0, +x) = +-0
    if (x == 0.0f || isinf(y)) return sy * 0.5f * M_PI;
    if (isinf(x)) return xneg ? sy * M_PI : sy * 0.0f;
    float a = atan32(abs(y) / abs(x));  // (as Cephes' atan2f)
    return sy * (xneg ? (M32_PI_HI - a) + M32_PI_LO : a);
}

// x * 2^k exactly for 0 <= k <= 277 (k is split so each factor is a normal power of two).
float m_scale2(float x, int k)
{
    while (k > 0)
    {
        int s = min(k, 127);
        x *= bit_cast<float>(uint(s + 127) << 23);
        k -= s;
    }
    return x;
}

// floor(log2(|x|)) for finite nonzero x, subnormals included.
int m_ilogb(float x)
{
    uint bits = bit_cast<uint>(x) & 0x7fffffffu;
    int e = int(bits >> 23);
    return e != 0 ? e - 127 : firstbithigh(bits) - 149;
}

// a * b exactly as p + e: Dekker's product with Veltkamp's split, plain
// multiplications and additions (kernels are compiled with -fp-mode precise,
// which keeps them unfused), for |a|, |b| and |a b| within 2^+-64 of 1.
void div32_two_prod(float a, float b, out float p, out float e)
{
    p = a * b;
    float t = 4097.0f * a, ah = t - (t - a), al = a - ah;
    t = 4097.0f * b;
    float bh = t - (t - b), bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}

// a - q b exactly, for q within a few ulps of a / b (a, b in [1, 2), or
// a, b, q within 2^+-60 of 1: no overflow, Dekker's error terms normal).
float div32_residual(float a, float q, float b)
{
    float p, e;
    div32_two_prod(q, b, p, e);
    return (a - p) - e;
}

// |x|'s significand as an int in [2^23, 2^24) and its exponent: |x| = m 2^(e - 23).
uint div32_unpack(float x, out int e)
{
    uint u = bit_cast<uint>(x) & 0x7fffffffu;
    uint f = u >> 23;
    if (f != 0u)
    {
        e = int(f) - 127;
        return (u & 0x7fffffu) | 0x800000u;
    }
    int shift = 23 - firstbithigh(u);  // a subnormal, normalised on the bits
    e = -126 - shift;
    return u << uint(shift);
}

// a / b correctly rounded to nearest even, as the CPU divides, on any GPU:
// Vulkan allows OpFDiv 2.5 ulp, and an FMA-based correction needs the driver
// to fuse FMAs, which the UHD 620's does not (about 12% of its quotients came
// out 1 ulp off). The significands' quotient, in (0.5, 2), is corrected with
// its exact residual (Dekker's product) and chosen among its neighbours by
// their exact residuals; the exponent and the rounding of subnormal results
// (ties broken by the residual's sign) are done on the bits, so neither
// overflow nor a subnormal intermediate can interfere. Zeros, infinities and
// NaNs keep the GPU's division (exact there).
float m_div32(float a, float b)
{
    // Zeros, infinities and NaNs, told apart on the bits (a device that
    // flushes subnormals may compare them equal to 0; they are divided exactly
    // below). Their results are built here, not left to the GPU's division:
    // drivers divide as a * (1 / b), and 1 / b overflows for |b| < 2^-128, so
    // that 0 / 2^-149 came out 0 * inf = NaN (Mesa's ANV on Gen9). Only the
    // NaN results (0 / 0, inf / inf, NaN operands) keep the GPU's division.
    uint aa = bit_cast<uint>(a) & 0x7fffffffu, ab = bit_cast<uint>(b) & 0x7fffffffu;
    uint zero = (bit_cast<uint>(a) ^ bit_cast<uint>(b)) & 0x80000000u, inf = zero | 0x7f800000u;
    if (aa > 0x7f800000u || ab > 0x7f800000u || (aa == 0u && ab == 0u) || (aa == 0x7f800000u && ab == 0x7f800000u))
        return a / b;
    if (aa == 0u || ab == 0x7f800000u)
        return bit_cast<float>(zero);
    if (aa == 0x7f800000u || ab == 0u)
        return bit_cast<float>(inf);
    // The common case: a, b and a / b within 2^+-60 (no overflow in the
    // split, Dekker's error terms normal), the GPU's quotient corrected once
    // and kept when its exact residual is clearly under half an ulp of b q.
    int fa = int((bit_cast<uint>(a) >> 23) & 255u), fb = int((bit_cast<uint>(b) >> 23) & 255u);
    if (fa >= 67 && fa <= 187 && fb >= 67 && fb <= 187 && abs(fa - fb) <= 60)
    {
        float q = a / b;
        q += div32_residual(a, q, b) / b;
        float ulp = bit_cast<float>((bit_cast<uint>(q) & 0x7f800000u) - (23u << 23));  // 2^(exponent(q) - 23)
        if (abs(div32_residual(a, q, b)) < 0.4999f * ulp * abs(b))
            return q;
    }
    int ea, eb;
    uint ia = div32_unpack(a, ea), ib = div32_unpack(b, eb);
    float ma = bit_cast<float>(0x3f800000u | (ia & 0x7fffffu)), mb = bit_cast<float>(0x3f800000u | (ib & 0x7fffffu));
    float q = ma / mb;
    q += div32_residual(ma, q, mb) / mb;
    uint qb = bit_cast<uint>(q);
    float r = div32_residual(ma, q, mb);
    [ForceUnroll]
    for (int d = -1; d <= 1; d += 2)
    {
        float c = bit_cast<float>(uint(int(qb) + d));
        float rc = div32_residual(ma, c, mb);
        if (abs(rc) < abs(r) || (abs(rc) == abs(r) && (bit_cast<uint>(c) & 1u) == 0u))
        {
            q = c;
            r = rc;
        }
    }
    // q: the significands' quotient rounded to 24 bits; r = ma - q mb exactly
    qb = bit_cast<uint>(q);
    uint m = (qb & 0x7fffffu) | 0x800000u;
    int e = int(qb >> 23) - 127 + ea - eb;
    uint sign = (bit_cast<uint>(a) ^ bit_cast<uint>(b)) & 0x80000000u;
    if (e > 127)
        return bit_cast<float>(sign | 0x7f800000u);
    if (e >= -126)
        return bit_cast<float>(sign | (uint(e + 127) << 23) | (m & 0x7fffffu));
    // a subnormal (or zero) result: m 2^(e - 23) in units of 2^-149 is m / 2^d
    int d = -126 - e;
    if (d > 24)
        return bit_cast<float>(sign);
    uint keep = m >> uint(d), rem = m & ((1u << uint(d)) - 1u), half = 1u << uint(d - 1);
    // (|rem - half| >= 1 unit of q, more than q's error: only exact halves
    // depend on which side of q the quotient lies)
    if (rem > half || (rem == half && (r > 0.0f || (r == 0.0f && (keep & 1u) != 0u))))
        keep += 1u;
    return bit_cast<float>(sign | keep);
}

// sqrt(x) correctly rounded, as the CPU's: Vulkan's Sqrt is only as accurate
// as 1 / inversesqrt (ANV's float32 roots differed from the CPU's in 8% of
// cases). The GPU's root c of the significand M (x scaled by an even power
// of two into [1, 4)) is moved to a neighbour while the root lies beyond the
// midpoint on that side: with u = ulp(c), (c + u/2)^2 = c (c + u) + u^2 / 4
// and M - c (c + u) is a multiple of u^2, so sqrt(M) > c + u/2 exactly when
// M - c next(c) > 0, an exact residual (Dekker's product) whose sign the
// rounding keeps. Zeros (with their sign), +inf and NaNs as the CPU's.
float m_sqrt(float x)
{
    uint b = bit_cast<uint>(x);
    if ((b & 0x7fffffffu) == 0u || b == 0x7f800000u)
        return x;
    if (b > 0x7f800000u)
        return f32_nan();  // (negative or NaN)
    int e;
    uint m = div32_unpack(x, e);  // (subnormals included, on the bits)
    float M = bit_cast<float>(0x3f800000u | (m & 0x7fffffu));
    if ((e & 1) != 0)
    {
        M *= 2.0f;
        e -= 1;
    }
    float c = sqrt(M);
    [ForceUnroll]
    for (int i = 0; i < 3; i++)
    {
        uint cb = bit_cast<uint>(c);
        float up = bit_cast<float>(cb + 1u), dn = bit_cast<float>(cb - 1u);
        if (div32_residual(M, c, up) > 0.0f)
            c = up;
        else if (div32_residual(M, dn, c) <= 0.0f)
            c = dn;
    }
    // (c in [1, 2): the root of M < 4 rounds below 2)
    return bit_cast<float>((uint(e / 2 + 127) << 23) | (bit_cast<uint>(c) & 0x7fffffu));
}

#elif defined(T_f64) || defined(FMATH_F64)
#include "_math64.slang"
#include "_math64_ext.slang"

typedef uint64_t UBITS;
typedef double FT;
static const double M_LN2 = LN2;
static const double M_INV_LN2 = INV_LN2;

double m_inf() { return f64_inf(); }
double m_nan() { return f64_nan(); }
bool m_signbit(double x) { return sign_bit(x); }

double m_exp(double x) { return exp64(x); }
double m_exp2(double x) { return exp2_64(x); }
double m_expm1(double x) { return expm1_64(x); }
double m_log(double x) { return log64(x); }
double m_log2(double x) { return log2_64(x); }
double m_log10(double x) { return log10_64(x); }
double m_log1p(double x) { return log1p64(x); }
double m_sin(double x) { return sin64(x); }
double m_cos(double x) { return cos64(x); }
double m_tan(double x) { return tan64(x); }
double m_asin(double x) { return asin64(x); }
double m_acos(double x) { return acos64(x); }
double m_atan(double x) { return atan64(x); }
double m_tanh(double x) { return tanh64(x); }
double m_sigmoid(double x) { return sigmoid64(x); }
double m_sinh(double x) { return sinh64(x); }
double m_cosh(double x) { return cosh64(x); }
double m_asinh(double x) { return asinh64(x); }
double m_acosh(double x) { return acosh64(x); }
double m_atanh(double x) { return atanh64(x); }
double m_erf(double x) { return erf64(x); }
double m_erfc(double x) { return erfc64(x); }
double m_erfinv(double x) { return erfinv64(x); }
double m_lgamma(double x) { return lgamma64(x); }
double m_digamma(double x) { return digamma64(x); }
double m_sinc(double x) { return sinc64(x); }
double m_atan2(double y, double x) { return atan2_64(y, x); }

double m_scale2(double x, int k)
{
    while (k > 0)
    {
        int s = min(k, 1023);
        x *= bit_cast<double>(uint64_t(s + 1023) << 52);
        k -= s;
    }
    return x;
}

int m_ilogb(double x)
{
    uint64_t bits = bit_cast<uint64_t>(x) & 0x7fffffffffffffffull;
    int e = int(bits >> 52);
    if (e != 0) return e - 1023;
    uint hi = uint(bits >> 32), lo = uint(bits);
    return (hi != 0 ? firstbithigh(hi) + 32 : firstbithigh(lo)) - 1074;
}

// a * b exactly as p + e (Dekker's product, Veltkamp's split; unfused, as
// div32_two_prod), for |a|, |b| within 2^+-490 of 1.
void sqrt64_two_prod(double a, double b, out double p, out double e)
{
    p = a * b;
    double t = 134217729.0l * a, ah = t - (t - a), al = a - ah;
    t = 134217729.0l * b;
    double bh = t - (t - b), bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}

// a - q b, with the exact value's sign, for q b within a few ulps of a.
double sqrt64_residual(double a, double q, double b)
{
    double p, e;
    sqrt64_two_prod(q, b, p, e);
    return (a - p) - e;
}

// sqrt(x) correctly rounded, as the float32 m_sqrt (no FMAs: not every
// driver fuses them); x is first scaled by 2^+-600 into [2^-500, 2^500]
// (a subnormal x, on a device that flushes float64 subnormals, is lost).
double m_sqrt(double x)
{
    if (x == 0.0l || (isinf(x) && x > 0.0l))
        return x;
    if (!(x > 0.0l))
        return f64_nan();
    double back = 1.0l;
    if (x < bit_cast<double>(uint64_t(1023 - 500) << 52))
    {
        x *= bit_cast<double>(uint64_t(1023 + 600) << 52);
        back = bit_cast<double>(uint64_t(1023 - 300) << 52);
    }
    else if (x > bit_cast<double>(uint64_t(1023 + 500) << 52))
    {
        x *= bit_cast<double>(uint64_t(1023 - 600) << 52);
        back = bit_cast<double>(uint64_t(1023 + 300) << 52);
    }
    double c = sqrt(x);
    [ForceUnroll]
    for (int i = 0; i < 3; i++)
    {
        uint64_t cb = bit_cast<uint64_t>(c);
        double up = bit_cast<double>(cb + 1ull), dn = bit_cast<double>(cb - 1ull);
        if (sqrt64_residual(x, c, up) > 0.0l)
            c = up;
        else if (sqrt64_residual(x, dn, c) <= 0.0l)
            c = dn;
    }
    return c * back;
}
#endif

// |a| with the sign bit of b, on the bits: negating a NaN need not set its
// sign bit (NVIDIA's float64 negation and abs do not).
FT m_copysign(FT a, FT b)
{
    UBITS mask = ~(~UBITS(0) >> 1);
    return bit_cast<FT>((bit_cast<UBITS>(a) & ~mask) | (bit_cast<UBITS>(b) & mask));
}

// x rounded to the nearest integer, ties to even (torch.round), with x's
// sign (-0.4 gives -0). Not the built-in round: SPIR-V's Round leaves the
// direction of ties to the driver (lavapipe on an AVX2 CPU rounded 3.5 to 3).
FT m_round_even(FT x)
{
    FT r = floor(x);
    FT d = x - r;
    if (d > FT(0.5) || (d == FT(0.5) && floor(r * FT(0.5)) * FT(2) != r))
        r += FT(1);
    return m_copysign(r, x);
}

// log(1 + e^x) without overflow.
FT m_softplus(FT x)
{
    return x > FT(0) ? x + m_log1p(m_exp(-x)) : m_log1p(m_exp(x));
}

// fmod(a, b) exactly: subtract b scaled by powers of two (each step is exact by Sterbenz).
FT m_fmod(FT a, FT b)
{
    if (isnan(a) || isnan(b) || isinf(a) || b == FT(0)) return m_nan();
    if (isinf(b)) return a;
    FT r = abs(a), d = abs(b);
    int ed = m_ilogb(d);
    while (r >= d)
    {
        FT t = m_scale2(d, m_ilogb(r) - ed);
        if (t > r) t *= FT(0.5);
        r -= t;
    }
    return m_signbit(a) ? -r : r;
}

// a / b correctly rounded: Vulkan allows OpFDiv 2.5 ulp, so 6 / 3 can be
// 1.9999999 (and trunc it 1). float32: m_div32, without FMAs (not every
// driver fuses them). float64: one correction with the exact remainder (an
// FMA) fixes the last bit; for |b| > 2^126 both are quartered (exactly)
// first: GPUs may divide as a * (1 / b), and 1 / b is then subnormal, which
// devices without subnormals flush to 0.
FT m_div(FT a, FT b)
{
#if defined(T_f32) || defined(FMATH_F32)
    return m_div32(a, b);  // (no FMA needed)
#else
    if (abs(b) > LIT(8.5070591730234616e37))
    {
        a *= FT(0.25);
        b *= FT(0.25);
    }
    FT q = a / b;
    if (isnan(q) || isinf(q) || q == FT(0) || isinf(b)) return q;
    return q + fma(-q, b, a) / b;
#endif
}

// Python-style floor division, as PyTorch's div_floor for floats.
FT m_floor_div(FT a, FT b)
{
    if (b == FT(0)) return a / b;
    FT mod = m_fmod(a, b);
    FT div = m_div(a - mod, b);
    if (mod != FT(0) && (b < FT(0)) != (mod < FT(0))) div -= FT(1);
    if (div == FT(0)) return FT(0) * (a / b);  // keeps the sign of zero
    FT fl = floor(div);
    if (div - fl > FT(0.5)) fl += FT(1);
    return fl;
}

bool m_is_odd_integer(FT y)
{
    return y == floor(y) && abs(y) < FT(9007199254740992.0) && floor(y * FT(0.5)) != y * FT(0.5);
}

// x^y with C99 special cases. Integer exponents up to 64 use exact squaring;
// float64 recovers the rounding error of y log x with an FMA.
FT m_pow(FT x, FT y)
{
    if (y == FT(0) || x == FT(1)) return FT(1);
    if (isnan(x) || isnan(y)) return m_nan();
    bool neg = false;
    if (x < FT(0) || (x == FT(0) && m_signbit(x)))
    {
        if (!isinf(x) && x != FT(0) && y != floor(y) && !isinf(y)) return m_nan();
        neg = m_is_odd_integer(y);
        x = -x;
    }
    if (isinf(y))
    {
        if (x == FT(1)) return FT(1);
        return (x > FT(1)) == (y > FT(0)) ? m_inf() : FT(0);
    }
    FT r;
    if (x == FT(0))
        r = y > FT(0) ? FT(0) : m_inf();
    else if (isinf(x))
        r = y > FT(0) ? m_inf() : FT(0);
    else if (y == floor(y) && abs(y) <= FT(64))
    {
        FT base = y > FT(0) ? x : FT(1) / x;
        int e = int(abs(y));
        r = FT(1);
        while (e > 0)
        {
            if ((e & 1) != 0) r *= base;
            base *= base;
            e >>= 1;
        }
    }
    else
    {
#if defined(T_f64) || defined(FMATH_F64)
        double l = log64(x);
        double t = y * l;
        double tl = fma(y, l, -t);
        // (the correction only where exp(t) is finite and nonzero: for a huge
        // t, its rounding error tl is huge too, and 1 + tl can be negative)
        r = exp64(t);
        if (r != 0.0l && !isinf(r)) r *= 1.0l + tl;
#else
        r = pow32(x, y);
#endif
    }
    return neg ? -r : r;
}
