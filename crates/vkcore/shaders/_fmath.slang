// Float math on a scalar type FT under one set of names (m_exp, m_log, ...):
// float for kernels with T_f32 (or FMATH_F32, e.g. complex64 kernels),
// double for T_f64 (or FMATH_F64).
//
// float32 uses the GPU's GLSL.std.450 built-ins where they are accurate
// enough, _math32.slang where there is no built-in, and short wrappers where
// a built-in mishandles an edge case (tanh overflow, sinh/asinh cancellation,
// pow/atan2 special values). float64 uses the software library in
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

float m_exp(float x) { return exp(x); }
float m_exp2(float x) { return exp2(x); }
float m_expm1(float x) { return expm1_32(x); }
float m_log(float x) { return log(x); }
float m_log2(float x) { return log2(x); }
float m_log10(float x) { return log(x) * M_INV_LN10; }
float m_log1p(float x) { return log1p32(x); }
// sin, cos and tan by Cody-Waite reduction to [-pi/4, pi/4] (r, quadrant n)
// and the _math32 polynomials: the built-ins can be off by ~1e-5 for
// moderate arguments (sin, cos) or ~1e-4 (tan). The built-ins beyond |x| ~ 800.
float reduce_pio2_32(float x, out int n)
{
    float k = round(x * 0.636619772367581f);
    n = int(k);
    return ((x - k * 1.5707855225e+00f) - k * 1.0804273188e-05f) - k * 6.0770943833e-11f;
}
float m_sin(float x)
{
    if (!(abs(x) < 800.0f)) return sin(x);
    int n;
    float r = reduce_pio2_32(x, n);
    float v = (n & 1) == 0 ? sin_poly32(r) : cos_poly32(r);
    return (n & 2) == 0 ? v : -v;
}
float m_cos(float x)
{
    if (!(abs(x) < 800.0f)) return cos(x);
    int n;
    float r = reduce_pio2_32(x, n);
    float v = (n & 1) == 0 ? cos_poly32(r) : sin_poly32(r);
    return ((n + 1) & 2) == 0 ? v : -v;
}
float m_tan(float x)
{
    if (!(abs(x) < 800.0f)) return tan(x);
    int n;
    float r = reduce_pio2_32(x, n);
    float s = sin_poly32(r), c = cos_poly32(r);
    return (n & 1) == 0 ? s / c : -c / s;
}
float m_asin(float x) { return asin(x); }
// acos via asin, keeping relative accuracy near x = 1 (the built-in acos is
// only accurate to ~1e-4 on some GPUs).
float m_acos(float x)
{
    if (abs(x) < 0.5f) return 0.5f * M_PI - asin(x);
    float r = 2.0f * asin(sqrt(0.5f * (1.0f - abs(x))));
    return x > 0.0f ? r : M_PI - r;
}
float m_atan(float x) { return atan(x); }

float m_tanh(float x)
{
    if (abs(x) > 10.0f) return x > 0.0f ? 1.0f : -1.0f;  // e^2x overflows in the built-in
    if (abs(x) < 0.5f)
    {
        float e = expm1_32(2.0f * x);
        return e / (e + 2.0f);
    }
    return tanh(x);
}

float m_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

float m_sinh(float x)
{
    if (abs(x) < 1.0f)
    {
        float e = expm1_32(x);
        return 0.5f * (e + e / (e + 1.0f));
    }
    return sinh(x);
}

float m_cosh(float x) { return cosh(x); }

float m_asinh(float x)
{
    float a = abs(x), r;
    if (a > 1e18f) r = log(a) + M_LN2;
    else r = log1p32(a + a * a / (1.0f + sqrt(1.0f + a * a)));
    return m_signbit(x) ? -r : r;
}

float m_acosh(float x)
{
    if (x < 1.0f) return f32_nan();
    if (x > 1e18f) return log(x) + M_LN2;
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
    return atan2(y, x);
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

// Python-style floor division, as PyTorch's div_floor for floats.
FT m_floor_div(FT a, FT b)
{
    if (b == FT(0)) return a / b;
    FT mod = m_fmod(a, b);
    FT div = (a - mod) / b;
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
        r = pow(x, y);
#endif
    }
    return neg ? -r : r;
}
