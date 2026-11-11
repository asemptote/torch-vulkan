// float32 math without the GPU's transcendental built-ins (exp, log, asin,
// atan, pow, ...), whose accuracy varies by driver: Cephes' algorithms
// (about 1 ulp), and log1p, expm1, erf, erfc, erfinv, lgamma, digamma, sinpi,
// sinc on top of them. No double anywhere: float32 kernels must run on GPUs
// without float64. (Shaders are compiled with -fp-mode precise, so the
// error-free transformations below are not reassociated.)

static const float M32_PI = 3.14159265358979f;
static const float M32_PI_HI = 3.14159274101257324f;      // float(pi)
static const float M32_PI_LO = -8.74227765734758577e-8f;  // pi - float(pi)
static const float M32_TWO_OVER_SQRT_PI = 1.12837916709551f;
static const float M32_INV_SQRT_PI = 0.564189583547756f;
static const float M32_HALF_LOG_2PI = 0.918938533204673f;
static const float M32_LOG_PI = 1.1447298858494f;

float f32_inf() { return 1.0f / 0.0f; }
float f32_nan() { return 0.0f / 0.0f; }

// e^x by Cephes' expf (x = n ln 2 + r, |r| <= ln 2 / 2, a degree-5 polynomial):
// about 1 ulp, where the built-in exp can be a few. 2^n is applied in two
// halves, so results near overflow and subnormal ones come out right too; the
// built-in beyond |x| = 104 (inf, 0) and for NaN.
float exp32(float x)
{
    if (!(abs(x) < 104.0f)) return exp(x);
    float n = floor(1.44269504088896f * x + 0.5f);
    float r = (x - n * 0.693359375f) + n * 2.12194440e-4f;
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    int k = int(n);
    return ldexp(ldexp(p * r * r + r + 1.0f, k / 2), k - k / 2);
}
// ln x by Cephes' logf (x = m 2^e with m in [sqrt(1/2), sqrt(2)), a degree-8
// polynomial in m - 1): about 1 ulp, where the built-in log loses relative
// accuracy near 1 (~1e-5). The built-in for zero, negatives, inf and NaN.
float log32(float x)
{
    if (!(x > 0.0f) || isinf(x)) return log(x);
    float scale = 0.0f;
    if (x < 1.17549435e-38f)  // subnormal: scale into the normal range
    {
        x *= 33554432.0f;  // 2^25
        scale = -25.0f;
    }
    uint bits = bit_cast<uint>(x);
    float e = float(int((bits >> 23) & 0xffu) - 126) + scale;
    float m = bit_cast<float>((bits & 0x7fffffu) | 0x3f000000u);  // [0.5, 1)
    if (m < 0.707106781186547524f)
    {
        m = m + m;
        e = e - 1.0f;
    }
    float t = m - 1.0f;
    float z = t * t;
    float y = 7.0376836292e-2f;
    y = y * t - 1.1514610310e-1f;
    y = y * t + 1.1676998740e-1f;
    y = y * t - 1.2420140846e-1f;
    y = y * t + 1.4249322787e-1f;
    y = y * t - 1.6668057665e-1f;
    y = y * t + 2.0000714765e-1f;
    y = y * t - 2.4999993993e-1f;
    y = y * t + 3.3333331174e-1f;
    y = y * z * t;
    y += -2.12194440e-4f * e;
    y += -0.5f * z;
    return (t + y) + 0.693359375f * e;
}
// asin by Cephes' asinf.
float asin32(float x)
{
    float a = abs(x), z;
    if (a < 1e-4f) z = a;
    else
    {
        bool big = a > 0.5f;
        float w = big ? 0.5f * (1.0f - a) : a * a;
        float v = big ? sqrt(w) : a;
        z = ((((4.2163199048e-2f * w + 2.4181311049e-2f) * w + 4.5470025998e-2f) * w + 7.4953002686e-2f) * w
             + 1.6666752422e-1f) * w * v + v;
        if (big) z = (0.5f * M32_PI_HI - (z + z)) + 0.5f * M32_PI_LO;
    }
    return x < 0.0f ? -z : z;
}

// atan by Cephes' atanf.
float atan32(float x)
{
    float a = abs(x), y = 0.0f;
    if (a > 2.414213562373095f)
    {
        y = 0.5f * M32_PI;
        a = -1.0f / a;
    }
    else if (a > 0.4142135623730950f)
    {
        y = 0.25f * M32_PI;
        a = (a - 1.0f) / (a + 1.0f);
    }
    float z = a * a;
    y += (((8.05374449538e-2f * z - 1.38776856032e-1f) * z + 1.99777106478e-1f) * z - 3.33329491539e-1f) * z * a + a;
    return x < 0.0f ? -y : y;
}

// Exact sum and product of two floats as (hi, lo): Knuth's two-sum, Dekker's
// split product (no FMA: Vulkan does not promise a fused one).
float2 two_sum32(float a, float b)
{
    float s = a + b, bb = s - a;
    return float2(s, (a - (s - bb)) + (b - bb));
}
float2 split32(float a)
{
    float c = 4097.0f * a;
    float hi = c - (c - a);
    return float2(hi, a - hi);
}
float2 two_prod32(float a, float b)
{
    float p = a * b;
    float2 x = split32(a), y = split32(b);
    return float2(p, ((x.x * y.x - p) + x.x * y.y + x.y * y.x) + x.y * y.y);
}

// ln x as hi + lo (about 2^-40 relative) for finite x > 0.
float2 ln_hilo32(float x)
{
    uint bits = bit_cast<uint>(x);
    int e = int((bits >> 23) & 0xffu) - 127;
    if (e == -127)  // subnormal
    {
        bits = bit_cast<uint>(x * 33554432.0f);  // 2^25
        e = int((bits >> 23) & 0xffu) - 152;
    }
    float m = bit_cast<float>((bits & 0x7fffffu) | 0x3f800000u);  // [1, 2)
    if (m > 1.41421356f)
    {
        m *= 0.5f;
        e += 1;
    }
    // ln m = 2 atanh(s), s = f / (2 + f), f = m - 1 (exact), |s| < 0.172.
    float f = m - 1.0f;
    float2 d = two_sum32(2.0f, f);
    float s_hi = f / d.x;
    float2 q = two_prod32(s_hi, d.x);
    float s_lo = ((f - q.x) - q.y - s_hi * d.y) / d.x;
    float z = s_hi * s_hi;
    float p = (((((z * (1.0f / 15.0f) + 1.0f / 13.0f) * z + 1.0f / 11.0f) * z + 1.0f / 9.0f) * z + 1.0f / 7.0f) * z
               + 1.0f / 5.0f) * z + 1.0f / 3.0f;
    // ln x = e ln 2 + 2 s + 2 s z p, with ln 2 = 0.693145751953125 (e times it
    // is exact) + 1.428606820309417e-6.
    float2 l = two_sum32(float(e) * 0.693145751953125f, 2.0f * s_hi);
    float lo = l.y + (2.0f * s_lo + 2.0f * s_hi * z * p + float(e) * 1.428606820309417e-6f);
    return two_sum32(l.x, lo);
}

// x^y for finite x > 0 and finite y: ln x in two floats, y ln x with its
// rounding error, then e^hi (1 + lo). A few ulp, where pow(x, y) as
// exp2(y log2 x) in float loses about |y ln x| ulp.
float pow32(float x, float y)
{
    float2 lx = ln_hilo32(x);
    float t = y * lx.x;
    if (!(abs(t) < 104.0f))  // overflow, underflow
        return t > 0.0f ? f32_inf() : 0.0f;
    float2 yt = two_prod32(y, lx.x);
    float t_lo = yt.y + y * lx.y;
    float r = exp32(t);
    return r + r * t_lo;
}

// ln(1 + x): 1 + x as u + c exactly (two-sum), then ln u in two floats
// plus ln(1 + c / u) ~ c / u, rounded once: under 1 ulp (Goldberg's
// log(u) x / (u - 1) carried log32's error, up to 2.8 ulp).
float log1p32(float x)
{
    float2 u = two_sum32(1.0f, x);
    if (u.x == 1.0f && u.y == x) return x;  // (|x| < 2^-24: x - x^2 / 2 rounds to x)
    if (isnan(x) || isinf(u.x) || u.x <= 0.0f) return log32(u.x);
    float2 l = ln_hilo32(u.x);
    // c / u, corrected with its exact residual: OpFDiv may be 2.5 ulp off,
    // and for x near -2^-25 the result cancels down to c's own size
    float q = u.y / u.x;
    float2 qu = two_prod32(q, u.x);
    q += ((u.y - qu.x) - qu.y) / u.x;
    return l.x + (l.y + q);
}

float expm1_32(float x)
{
    float u = exp32(x);
    if (u == 1.0f) return x;
    float um1 = u - 1.0f;
    if (um1 == -1.0f || isinf(u)) return um1;
    return um1 * x / log32(u);  // Kahan: cancels the rounding of exp32(x)
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
    return M32_TWO_OVER_SQRT_PI * x * exp32(-x * x) * sum;
}

// erfc for x >= 0.5 by its continued fraction, evaluated backwards.
float erfc_cf32(float x)
{
    int terms = min(120, 10 + int(50.0f / (x * x)));
    float f = x;
    for (int n = terms; n >= 1; n--)
        f = x + (0.5f * float(n)) / f;
    return exp32(-x * x) * M32_INV_SQRT_PI / f;
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
    float l = log32((1.0f - a) * (1.0f + a));
    float k = 2.0f / (M32_PI * 0.147f) + 0.5f * l;
    float x = sqrt(sqrt(k * k - l / 0.147f) - k);
    for (int i = 0; i < 2; i++)
    {
        float f = a < 0.5f ? erf32(x) - a : (1.0f - a) - erfc32(x);  // both = erf(x) - a
        float dfdx = M32_TWO_OVER_SQRT_PI * exp32(-x * x);
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
    if (isinf(x)) return f32_nan();  // as sin(inf) / inf on the CPU
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
    return (z - 0.5f) * log32(z) - z + M32_HALF_LOG_2PI + s * r - log32(prod);
}

float lgamma32(float x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return f32_inf();
    if (x <= 0.0f && x == floor(x)) return f32_inf();
    if (x < 0.5f)  // reflection
        return (M32_LOG_PI - log32(abs(sinpi32(x)))) - lgamma32_pos(1.0f - x);
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
    return acc + log32(z) - 0.5f / z - s;
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
