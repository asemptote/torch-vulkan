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
    uint64_t p = uint64_t(ia) * uint64_t(lo);
    p = uint64_t(ia) * uint64_t(mid) + (p >> 32);
    p = (uint64_t(ia * hi) << 32) + p;
    int q = int(uint(p >> 62));
    p &= 0x3fffffffffffffffull;
    if ((p & 0x2000000000000000ull) != 0ull)  // fraction >= 1/2: the next quadrant
    {
        p -= 0x4000000000000000ull;
        q += 1;
    }
    float r = float(int64_t(p)) * 3.4061215800865545e-19f;  // pi/2 2^-62
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
#endif

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
// 1.9999999 (and trunc it 1). One correction with the exact remainder (an
// FMA) fixes the last bit.
FT m_div(FT a, FT b)
{
    FT q = a / b;
    if (isnan(q) || isinf(q) || q == FT(0) || isinf(b)) return q;
    return q + fma(-q, b, a) / b;
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
        r = exp64(t) * (1.0l + tl);
#else
        r = pow32(x, y);
#endif
    }
    return neg ? -r : r;
}
