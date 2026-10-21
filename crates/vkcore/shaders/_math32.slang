// float32 functions with no GLSL.std.450 built-in: log1p, expm1, erf, erfc,
// erfinv, lgamma, digamma, sinpi, sinc. float32 kernels use the GPU's
// built-ins for everything else (exp, log, sin, tanh, pow, atan2, ...), so
// their accuracy follows Vulkan's precision rules; these routines build on
// those built-ins and are similarly accurate. No double anywhere: float32
// kernels must run on GPUs without float64.

static const float M32_PI = 3.14159265358979f;
static const float M32_TWO_OVER_SQRT_PI = 1.12837916709551f;
static const float M32_INV_SQRT_PI = 0.564189583547756f;
static const float M32_HALF_LOG_2PI = 0.918938533204673f;
static const float M32_LOG_PI = 1.1447298858494f;

float f32_inf() { return 1.0f / 0.0f; }
float f32_nan() { return 0.0f / 0.0f; }

float log1p32(float x)
{
    float u = 1.0f + x;
    if (u == 1.0f) return x;
    if (isinf(u) || u <= 0.0f) return log(u);
    return log(u) * (x / (u - 1.0f));  // Goldberg: cancels the rounding of 1 + x
}

float expm1_32(float x)
{
    float u = exp(x);
    if (u == 1.0f) return x;
    float um1 = u - 1.0f;
    if (um1 == -1.0f || isinf(u)) return um1;
    return um1 * x / log(u);  // Kahan: cancels the rounding of exp(x)
}

// erf for 0 <= x < 2: 2/sqrt(pi) x e^{-x^2} sum (2x^2)^n / (1*3*...*(2n+1)).
float erf_series32(float x)
{
    float t = 2.0f * x * x, term = 1.0f, sum = 1.0f;
    for (int n = 1; n < 60; n++)
    {
        term *= t / float(2 * n + 1);
        sum += term;
        if (term < 1e-8f * sum) break;
    }
    return M32_TWO_OVER_SQRT_PI * x * exp(-x * x) * sum;
}

// erfc for x >= 0.5 by its continued fraction, evaluated backwards.
float erfc_cf32(float x)
{
    int terms = min(120, 10 + int(50.0f / (x * x)));
    float f = x;
    for (int n = terms; n >= 1; n--)
        f = x + (0.5f * float(n)) / f;
    return exp(-x * x) * M32_INV_SQRT_PI / f;
}

float erf32(float x)
{
    if (isnan(x)) return x;
    float a = abs(x), r;
    if (a < 2.0f) r = erf_series32(a);
    else if (a > 4.0f) r = 1.0f;
    else r = 1.0f - erfc_cf32(a);
    return x < 0.0f ? -r : r;
}

float erfc32(float x)
{
    if (isnan(x)) return x;
    if (x < 0.5f)
    {
        if (x < -4.0f) return 2.0f;
        return x < -0.5f ? 2.0f - erfc_cf32(-x) : 1.0f - erf32(x);
    }
    if (x > 10.1f) return 0.0f;
    return erfc_cf32(x);
}

// Winitzki's approximation, then Halley steps on erf(x) - y.
float erfinv32(float y)
{
    if (isnan(y) || abs(y) > 1.0f) return f32_nan();
    if (abs(y) == 1.0f) return y * f32_inf();
    if (y == 0.0f) return y;
    float a = abs(y);
    float l = log((1.0f - a) * (1.0f + a));
    float k = 2.0f / (M32_PI * 0.147f) + 0.5f * l;
    float x = sqrt(sqrt(k * k - l / 0.147f) - k);
    for (int i = 0; i < 2; i++)
    {
        float f = a < 0.5f ? erf32(x) - a : (1.0f - a) - erfc32(x);  // both = erf(x) - a
        float dfdx = M32_TWO_OVER_SQRT_PI * exp(-x * x);
        x -= f / (dfdx + x * f);
    }
    return y < 0.0f ? -x : x;
}

// sin and cos on [-pi/4, pi/4] by Taylor polynomials (error < 1e-9): GPU
// built-ins may only be accurate to ~2^-11 absolute.
float sin_poly32(float z)
{
    float z2 = z * z;
    return z + z * z2 * (-1.0f / 6.0f + z2 * (1.0f / 120.0f + z2 * (-1.0f / 5040.0f + z2 * (1.0f / 362880.0f))));
}

float cos_poly32(float z)
{
    float z2 = z * z;
    return 1.0f + z2 * (-0.5f + z2 * (1.0f / 24.0f + z2 * (-1.0f / 720.0f + z2 * (1.0f / 40320.0f))));
}

// sin(pi x) with exact argument reduction (exact zeros at integers).
float sinpi32(float x)
{
    if (isnan(x) || isinf(x)) return f32_nan();
    float r = x - 2.0f * round(0.5f * x);  // r in [-1, 1], exact
    float n = round(2.0f * r);
    float t = r - 0.5f * n;               // |t| <= 1/4, exact
    float arg = M32_PI * t;
    int q = int(n) & 3;
    float v = (q & 1) == 0 ? sin_poly32(arg) : cos_poly32(arg);
    return (q & 2) == 0 ? v : -v;
}

float cospi32(float x)
{
    if (isnan(x) || isinf(x)) return f32_nan();
    float r = x - 2.0f * round(0.5f * x);
    float n = round(2.0f * r);
    float t = r - 0.5f * n;
    float arg = M32_PI * t;
    int q = int(n) & 3;
    float v = (q & 1) == 0 ? cos_poly32(arg) : sin_poly32(arg);
    return (q == 0 || q == 3) ? v : -v;
}

float sinc32(float x)
{
    if (isnan(x)) return x;
    if (x == 0.0f) return 1.0f;
    if (isinf(x)) return 0.0f;
    return sinpi32(x) / (M32_PI * x);
}

// lgamma for x >= 0.5: shift to z >= 8, then Stirling's series.
float lgamma32_pos(float x)
{
    float z = x, prod = 1.0f;
    while (z < 8.0f)
    {
        prod *= z;
        z += 1.0f;
    }
    float r = 1.0f / z, r2 = r * r;
    float s = ((-1.0f / 1680.0f * r2 + 1.0f / 1260.0f) * r2 - 1.0f / 360.0f) * r2 + 1.0f / 12.0f;
    return (z - 0.5f) * log(z) - z + M32_HALF_LOG_2PI + s * r - log(prod);
}

float lgamma32(float x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return f32_inf();
    if (x <= 0.0f && x == floor(x)) return f32_inf();
    if (x < 0.5f)  // reflection
        return (M32_LOG_PI - log(abs(sinpi32(x)))) - lgamma32_pos(1.0f - x);
    return lgamma32_pos(x);
}

// digamma for x > 0: shift to z >= 6, then the asymptotic series.
float digamma32_pos(float x)
{
    float acc = 0.0f, z = x;
    while (z < 6.0f)
    {
        acc -= 1.0f / z;
        z += 1.0f;
    }
    float r2 = 1.0f / (z * z);
    float s = ((1.0f / 252.0f * r2 - 1.0f / 120.0f) * r2 + 1.0f / 12.0f) * r2;
    return acc + log(z) - 0.5f / z - s;
}

float digamma32(float x)
{
    if (isnan(x) || x == f32_inf()) return x;
    if (x == 0.0f) return (bit_cast<uint>(x) >> 31) != 0 ? f32_inf() : -f32_inf();
    if (x < 0.0f)
    {
        if (x == floor(x)) return f32_nan();
        return digamma32_pos(1.0f - x) - M32_PI * cospi32(x) / sinpi32(x);  // reflection
    }
    return digamma32_pos(x);
}
