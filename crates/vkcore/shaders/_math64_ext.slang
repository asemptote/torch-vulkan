// float64 special functions, built on _math64.slang (include that first):
// log1p, log10, atan/asin/acos/atan2, sinh/cosh/asinh/acosh/atanh, erf/erfc/
// erfinv, lgamma/digamma, sinpi/cospi/sinc.
//
// Argument reductions and series are chosen so truncation error is below one
// ULP; constants were generated with mpmath (40 digits) and rounded to double.

static const double PI_HI = 3.141592653589793l;
static const double PI_LO = 1.2246467991473532e-16l;
static const double PIO2_HI = 1.5707963267948966l;
static const double PIO2_LO = 6.123233995736766e-17l;
static const double INV_LN10 = 0.4342944819032518l;
static const double LOG10_2_HI = 0.3010299956639812l;
static const double LOG10_2_LO = -2.8037281277851704e-18l;
static const double TWO_OVER_SQRT_PI = 1.1283791670955126l;
static const double INV_SQRT_PI = 0.5641895835477563l;
static const double HALF_LOG_2PI = 0.9189385332046728l;
static const double LOG_PI = 1.1447298858494002l;

bool sign_bit(double x) { return (bit_cast<uint64_t>(x) >> 63) != 0; }

double log1p64(double x)
{
    if (isnan(x) || x == f64_inf()) return x;
    if (x < -1.0l) return f64_nan();
    if (x == -1.0l) return -f64_inf();
    double u = 1.0l + x;
    if (u == 1.0l) return x;              // |x| below half an ULP of 1
    return log64(u) * (x / (u - 1.0l));   // Goldberg: cancels the rounding of 1 + x
}

double log10_64(double x)
{
    if (isnan(x)) return x;
    if (x < 0.0l) return f64_nan();
    if (x == 0.0l) return -f64_inf();
    if (isinf(x)) return x;
    int e;
    double lm = log_mantissa(x, e);
    return double(e) * LOG10_2_HI + (lm * INV_LN10 + double(e) * LOG10_2_LO);
}

// atan for a in [0, 1]: a = tan(k pi/12 + atan(t)) with t = (a - c_k) / (1 + a c_k),
// |t| <= tan(pi/24) ~ 0.132; Taylor series of atan(t) through t^29.
double atan01(double a)
{
    double c, hi, lo;
    if (a < 0.13165249758739586l) { c = 0.0l; hi = 0.0l; lo = 0.0l; }
    else if (a < 0.41421356237309503l) { c = 0.2679491924311227l; hi = 0.26179938779914946l; lo = -2.6802044161277275e-17l; }
    else if (a < 0.7673269879789604l) { c = 0.5773502691896257l; hi = 0.5235987755982989l; lo = -5.360408832255455e-17l; }
    else { c = 1.0l; hi = 0.7853981633974483l; lo = 3.061616997868383e-17l; }
    double t = (a - c) / (1.0l + a * c);
    double z = t * t;
    double p = -1.0l / 29.0l;
    p = p * z + 1.0l / 27.0l;
    p = p * z - 1.0l / 25.0l;
    p = p * z + 1.0l / 23.0l;
    p = p * z - 1.0l / 21.0l;
    p = p * z + 1.0l / 19.0l;
    p = p * z - 1.0l / 17.0l;
    p = p * z + 1.0l / 15.0l;
    p = p * z - 1.0l / 13.0l;
    p = p * z + 1.0l / 11.0l;
    p = p * z - 1.0l / 9.0l;
    p = p * z + 1.0l / 7.0l;
    p = p * z - 1.0l / 5.0l;
    p = p * z + 1.0l / 3.0l;
    return hi + (lo + (t - t * z * p));
}

double atan64(double x)
{
    if (isnan(x)) return x;
    double a = abs(x);
    double r = a <= 1.0l ? atan01(a) : (PIO2_HI - atan01(1.0l / a)) + PIO2_LO;
    return x < 0.0l ? -r : r;
}

double atan2_64(double y, double x)
{
    if (isnan(x) || isnan(y)) return f64_nan();
    bool yneg = sign_bit(y), xneg = sign_bit(x);
    double sy = yneg ? -1.0l : 1.0l;
    if (isinf(x) && isinf(y))
        return sy * (xneg ? 2.356194490192345l : 0.7853981633974483l);
    if (y == 0.0l)
        return xneg ? sy * PI_HI : y;  // atan2(+-0, -x) = +-pi, atan2(+-0, +x) = +-0
    if (x == 0.0l || isinf(y))
        return sy * PIO2_HI;
    if (isinf(x))
        return xneg ? sy * PI_HI : sy * 0.0l;
    if (abs(y) <= abs(x))
    {
        double r = atan64(y / x);
        return xneg ? (r + sy * PI_HI) + sy * PI_LO : r;
    }
    return (sy * PIO2_HI - atan64(x / y)) + sy * PIO2_LO;
}

double asin64(double x)
{
    if (isnan(x) || abs(x) > 1.0l) return f64_nan();
    return atan2_64(x, sqrt((1.0l - x) * (1.0l + x)));
}

double acos64(double x)
{
    if (isnan(x) || abs(x) > 1.0l) return f64_nan();
    return atan2_64(sqrt((1.0l - x) * (1.0l + x)), x);
}

double sinh64(double x)
{
    if (isnan(x)) return x;
    double a = abs(x), r;
    if (a < 22.0l)
    {
        double t = expm1_64(a);
        r = 0.5l * (t + t / (t + 1.0l));
    }
    else if (a < 709.0l)
        r = 0.5l * exp64(a);
    else
    {
        double w = exp64(0.5l * a);  // avoid overflowing before the halving
        r = (0.5l * w) * w;
    }
    return x < 0.0l ? -r : r;
}

double cosh64(double x)
{
    if (isnan(x)) return x;
    double a = abs(x);
    if (a < 0.34657359027997264l)
    {
        double t = expm1_64(a);
        return 1.0l + (t * t) / (2.0l * (1.0l + t));
    }
    if (a < 22.0l)
    {
        double e = exp64(a);
        return 0.5l * e + 0.5l / e;
    }
    if (a < 709.0l) return 0.5l * exp64(a);
    double w = exp64(0.5l * a);
    return (0.5l * w) * w;
}

double asinh64(double x)
{
    if (isnan(x) || isinf(x)) return x;
    double a = abs(x), r;
    if (a > 268435456.0l)  // 2^28: sqrt(a^2 + 1) rounds to a
        r = log64(a) + LN2;
    else if (a > 2.0l)
        r = log64(2.0l * a + 1.0l / (sqrt(a * a + 1.0l) + a));
    else
        r = log1p64(a + a * a / (1.0l + sqrt(1.0l + a * a)));
    return x < 0.0l ? -r : r;
}

double acosh64(double x)
{
    if (isnan(x) || x < 1.0l) return f64_nan();
    if (isinf(x)) return x;
    if (x > 268435456.0l) return log64(x) + LN2;
    if (x > 2.0l) return log64(2.0l * x - 1.0l / (x + sqrt(x * x - 1.0l)));
    double t = x - 1.0l;
    return log1p64(t + sqrt(2.0l * t + t * t));
}

double atanh64(double x)
{
    if (isnan(x)) return x;
    double a = abs(x);
    if (a > 1.0l) return f64_nan();
    double r;
    if (a == 1.0l)
        r = f64_inf();
    else if (a < 0.5l)
        r = 0.5l * log1p64(2.0l * a + 2.0l * a * a / (1.0l - a));
    else
        r = 0.5l * log1p64(2.0l * a / (1.0l - a));
    return x < 0.0l ? -r : r;
}

// exp(-x^2) without the rounding error of x^2 (split with an FMA).
double exp_neg_sq(double x)
{
    double hi = x * x;
    double lo = fma(x, x, -hi);
    return exp64(-hi) * (1.0l - lo);
}

// erf(x) for 0 <= x < 2.5: 2/sqrt(pi) x e^{-x^2} sum (2x^2)^n / (1*3*...*(2n+1)) (positive terms).
double erf_series(double x)
{
    double t = 2.0l * x * x;
    double term = 1.0l, sum = 1.0l;
    for (int n = 1; n < 200; n++)
    {
        term *= t / double(2 * n + 1);
        sum += term;
        if (term < 1e-17l * sum) break;
    }
    return TWO_OVER_SQRT_PI * x * exp_neg_sq(x) * sum;
}

// erfc(x) for x >= 1 by the continued fraction
// e^{-x^2}/sqrt(pi) / (x + (1/2)/(x + 1/(x + (3/2)/(x + ...)))), evaluated backwards.
double erfc_cf(double x)
{
    int terms = min(400, 20 + int(160.0l / (x * x)));
    double f = x;
    for (int n = terms; n >= 1; n--)
        f = x + (0.5l * double(n)) / f;
    return exp_neg_sq(x) * INV_SQRT_PI / f;
}

double erf64(double x)
{
    if (isnan(x)) return x;
    double a = abs(x), r;
    if (a < 2.5l) r = erf_series(a);
    else if (a > 6.0l) r = 1.0l;
    else r = 1.0l - erfc_cf(a);
    return x < 0.0l ? -r : r;
}

double erfc64(double x)
{
    if (isnan(x)) return x;
    if (x < 1.0l)
    {
        if (x < -6.0l) return 2.0l;
        return x < -1.0l ? 2.0l - erfc_cf(-x) : 1.0l - erf64(x);
    }
    if (x > 27.3l) return 0.0l;
    return erfc_cf(x);
}

// erfinv: Winitzki's approximation (relative error ~2e-3), then Halley steps
// on erf(x) - y, or on (1 - y) - erfc(x) near +-1 to avoid cancellation.
double erfinv64(double y)
{
    if (isnan(y) || abs(y) > 1.0l) return f64_nan();
    if (y == 1.0l) return f64_inf();
    if (y == -1.0l) return -f64_inf();
    if (y == 0.0l) return y;
    double a = abs(y);
    double l = log64((1.0l - a) * (1.0l + a));
    double k = 2.0l / (PI_HI * 0.147l) + 0.5l * l;
    double x = sqrt(sqrt(k * k - l / 0.147l) - k);
    for (int i = 0; i < 4; i++)
    {
        double f = a < 0.5l ? erf64(x) - a : (1.0l - a) - erfc64(x);  // both = erf(x) - a
        double dfdx = TWO_OVER_SQRT_PI * exp_neg_sq(x);
        x -= f / (dfdx + x * f);  // Halley (f'' = -2x f')
    }
    return y < 0.0l ? -x : x;
}

// sin(pi x), cos(pi x) with exact argument reduction (exact zeros at integers).
double sinpi64(double x)
{
    if (isnan(x) || isinf(x)) return f64_nan();
    double r = x - 2.0l * round(0.5l * x);  // r in [-1, 1], exact
    double n = round(2.0l * r);
    double t = r - 0.5l * n;               // |t| <= 1/4, exact
    double arg = t * PI_HI + t * PI_LO;
    int q = int(int64_t(n) & 3);
    double v = (q & 1) == 0 ? sin_poly(arg) : cos_poly(arg);
    double s = (q & 2) == 0 ? v : -v;
    return s == 0.0l ? (x < 0.0l ? -0.0l : 0.0l) : s;
}

double cospi64(double x)
{
    if (isnan(x) || isinf(x)) return f64_nan();
    double r = x - 2.0l * round(0.5l * x);
    double n = round(2.0l * r);
    double t = r - 0.5l * n;
    double arg = t * PI_HI + t * PI_LO;
    int q = int(int64_t(n) & 3);
    double v = (q & 1) == 0 ? cos_poly(arg) : sin_poly(arg);
    return (q == 0 || q == 3) ? v : -v;
}

double sinc64(double x)
{
    if (isnan(x)) return x;
    if (x == 0.0l) return 1.0l;
    if (isinf(x)) return f64_nan();  // as sin(inf) / inf on the CPU
    return sinpi64(x) / (PI_HI * x);
}

// Stirling series for z >= 15 (Bernoulli terms through z^-19).
double lgamma_stirling(double z)
{
    double r = 1.0l / z, r2 = r * r;
    double s = -1.3924322169059011l;
    s = s * r2 + 0.17964437236883057l;
    s = s * r2 - 0.029550653594771242l;
    s = s * r2 + 0.00641025641025641l;
    s = s * r2 - 0.0019175269175269176l;
    s = s * r2 + 0.0008417508417508417l;
    s = s * r2 - 0.0005952380952380953l;
    s = s * r2 + 0.0007936507936507937l;
    s = s * r2 - 0.002777777777777778l;
    s = s * r2 + 0.08333333333333333l;
    return (z - 0.5l) * log64(z) - z + HALF_LOG_2PI + s * r;
}

// lgamma for x >= 0.5: lgamma(z) - log(x (x+1) ... (z-1)) with z >= 15.
double lgamma64_pos(double x)
{
    double z = x, prod = 1.0l;
    while (z < 15.0l)
    {
        prod *= z;
        z += 1.0l;
    }
    return lgamma_stirling(z) - log64(prod);
}

double lgamma64(double x)
{
    if (isnan(x)) return x;
    if (isinf(x)) return f64_inf();
    if (x <= 0.0l && x == floor(x)) return f64_inf();  // poles
    if (x < 0.5l)  // reflection: lgamma(x) = log(pi / |sin(pi x)|) - lgamma(1 - x)
        return (LOG_PI - log64(abs(sinpi64(x)))) - lgamma64_pos(1.0l - x);
    return lgamma64_pos(x);
}

// digamma for x > 0: shift up to z >= 10, then the asymptotic series.
double digamma64_pos(double x)
{
    double acc = 0.0l, z = x;
    while (z < 10.0l)
    {
        acc -= 1.0l / z;
        z += 1.0l;
    }
    double r2 = 1.0l / (z * z);
    double s = -26.456212121212122l;
    s = s * r2 + 3.0539543302701198l;
    s = s * r2 - 0.4432598039215686l;
    s = s * r2 + 0.08333333333333333l;
    s = s * r2 - 0.021092796092796094l;
    s = s * r2 + 0.007575757575757576l;
    s = s * r2 - 0.004166666666666667l;
    s = s * r2 + 0.003968253968253968l;
    s = s * r2 - 0.008333333333333333l;
    s = s * r2 + 0.08333333333333333l;
    return acc + log64(z) - 0.5l / z - s * r2;
}

double digamma64(double x)
{
    if (isnan(x)) return x;
    if (x == f64_inf()) return x;
    if (x == 0.0l) return sign_bit(x) ? f64_inf() : -f64_inf();
    if (x < 0.0l)
    {
        if (x == floor(x)) return f64_nan();
        return digamma64_pos(1.0l - x) - PI_HI * cospi64(x) / sinpi64(x);  // reflection
    }
    return digamma64_pos(x);
}
