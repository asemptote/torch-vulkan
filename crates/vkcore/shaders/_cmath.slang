// Complex functions on vector<S, 2> (real, imag) with S = float or double,
// built on _fmath.slang's real functions (the including file defines
// FMATH_F32 or FMATH_F64 and S). Formulas follow the usual principal
// branches (as std::complex / C99), arranged to avoid cancellation and
// overflow where that is cheap: hypot for |z|, log1p and expm1 for values
// near 1 and 0, asymptotic forms for large |z|. Infinite and NaN parts
// follow C99's Annex G (as NumPy).

#include "_complex.slang"

typedef vector<S, 2> C;

C c_make(S re, S im) { return C(re, im); }
S c_copysign(S x, S y) { return m_signbit(x) != m_signbit(y) ? -x : x; }
bool c_finite(S x) { return !isinf(x) && !isnan(x); }
C c_i_times(C z) { return C(-z.y, z.x); }    // i z
C c_neg_i_times(C z) { return C(z.y, -z.x); }  // -i z

C c_exp(C z)
{
    if (isinf(z.x) && !c_finite(z.y))  // e^(-inf + i inf/nan) = 0, e^(+inf + i inf/nan) = inf + i nan
        return z.x < S(0) ? C(S(0), S(0)) : C(z.x, m_nan());
    S e = m_exp(z.x);
    if (z.y == S(0))
        return C(e, z.y);  // keeps a real input real (and the sign of a zero imaginary part)
    return C(e * m_cos(z.y), e * m_sin(z.y));
}

C c_log(C z)
{
    return C(m_log(cabs(z)), m_atan2(z.y, z.x));
}

C c_sqrt(C z)
{
    if (z.x == S(0) && z.y == S(0))
        return C(S(0), z.y);
    if (isinf(z.y))
        return C(m_inf(), z.y);
    if (isinf(z.x))  // sqrt(+inf + iy) = inf + i0, sqrt(-inf + iy) = 0 + i inf (y finite; nan stays nan)
        return z.x > S(0) ? C(z.x, isnan(z.y) ? z.y : c_copysign(S(0), z.y))
                          : C(isnan(z.y) ? z.y : S(0), c_copysign(m_inf(), z.y));
    S r = cabs(z);
    if (z.x >= S(0))
    {
        S t = sqrt(S(0.5) * r + S(0.5) * z.x);  // (halves first: no overflow)
        return C(t, z.y / (S(2) * t));
    }
    S t = sqrt(S(0.5) * r - S(0.5) * z.x);
    return C(abs(z.y) / (S(2) * t), m_signbit(z.y) ? -t : t);
}

C c_sinh(C z)
{
    S x = z.x, y = z.y;
    if (!c_finite(y) && c_finite(x))  // sinh(x + i inf/nan) = nan + i nan, (+-0 + i nan) for x = 0
        return C(x == S(0) ? x : m_nan(), m_nan());
    if (isinf(x))
    {
        if (y == S(0)) return z;
        if (!c_finite(y)) return C(x, m_nan());
        return C(c_copysign(m_inf(), m_cos(y)) * (x < S(0) ? S(-1) : S(1)), c_copysign(m_inf(), m_sin(y)));
    }
    if (isnan(x))
        return C(x, y == S(0) ? y : m_nan());
    return C(m_sinh(x) * m_cos(y), m_cosh(x) * m_sin(y));
}
C c_cosh(C z)
{
    S x = z.x, y = z.y;
    if (!c_finite(y) && c_finite(x))  // cosh(x + i inf/nan) = nan + i nan, (nan + i0) for x = 0
        return C(m_nan(), x == S(0) ? S(0) : m_nan());
    if (isinf(x))
    {
        if (y == S(0)) return C(m_inf(), c_copysign(S(0), x) * (m_signbit(y) ? S(-1) : S(1)));
        if (!c_finite(y)) return C(m_inf(), m_nan());
        return C(c_copysign(m_inf(), m_cos(y)), c_copysign(m_inf(), m_sin(y)) * (x < S(0) ? S(-1) : S(1)));
    }
    if (isnan(x))
        return C(x, y == S(0) ? y : m_nan());
    return C(m_cosh(x) * m_cos(y), m_sinh(x) * m_sin(y));
}
C c_sin(C z) { return c_neg_i_times(c_sinh(c_i_times(z))); }  // sin z = -i sinh(i z)
C c_cos(C z) { return c_cosh(c_i_times(z)); }                  // cos z = cosh(i z)

// tanh(x + iy) = (t (1 + s^2) + i s (1 - t^2)) / (1 + t^2 s^2), t = tanh x,
// s = tan y (Kahan); saturates to +-1 for large |x|.
C c_tanh(C z)
{
    if (isinf(z.x))  // tanh(+-inf + iy) = +-1 + i0 sin 2y (+-0 for infinite or nan y)
        return C(c_copysign(S(1), z.x), c_finite(z.y) ? c_copysign(S(0), m_sin(S(2) * z.y)) : c_copysign(S(0), z.y));
    if (isnan(z.x))
        return C(z.x, z.y == S(0) ? z.y : m_nan());
    if (!c_finite(z.y))  // tanh(x + i inf/nan) = nan + i nan, (+-0 + i nan) for x = 0
        return C(z.x == S(0) ? z.x : m_nan(), m_nan());
    if (abs(z.x) > S(20))
        return C(m_signbit(z.x) ? S(-1) : S(1), S(4) * m_sin(z.y) * m_cos(z.y) * m_exp(S(-2) * abs(z.x)));
    S t = m_tanh(z.x), s = m_tan(z.y);
    S d = S(1) + t * t * s * s;
    return C(t * (S(1) + s * s) / d, s * (S(1) - t * t) / d);
}

C c_tan(C z) { return c_neg_i_times(c_tanh(c_i_times(z))); }  // tan z = -i tanh(i z)

C c_reciprocal(C z) { return cdiv(C(S(1), S(0)), z); }

C c_log1p(C z)
{
    // log|1 + z| = log1p(2x + x^2 + y^2) / 2 for small z.
    S x = z.x, y = z.y;
    S r2 = x * (S(2) + x) + y * y;
    if (abs(r2) < S(0.5))
        return C(S(0.5) * m_log1p(r2), m_atan2(y, S(1) + x));
    return c_log(C(S(1) + x, y));
}

C c_expm1(C z)
{
    // Re: expm1(x) cos y - 2 sin^2(y/2);  Im: e^x sin y
    S h = m_sin(S(0.5) * z.y);
    return C(m_expm1(z.x) * m_cos(z.y) - S(2) * h * h, m_exp(z.x) * m_sin(z.y));
}

// On the first quadrant (asinh(-z) = -asinh(z), asinh(conj z) = conj asinh(z)):
// log(z + sqrt(z^2 + 1)), which is log(2z) to working precision for large |z|.
C c_asinh(C z)
{
    S x = abs(z.x), y = abs(z.y);
    C w;
    if (isinf(x) || isinf(y))
        w = C(m_inf(), isnan(x) || isnan(y) ? m_nan() : isinf(x) && isinf(y) ? LIT(0.7853981633974483)
                                                     : isinf(x) ? S(0) : LIT(1.5707963267948966));
    else if (isnan(x) || isnan(y))
        w = C(m_nan(), y == S(0) ? y : m_nan());
    else if (max(x, y) > LIT(1e8))  // (z^2 + 1 = z^2: log 2 + log z, without squaring)
        w = C(LIT(0.6931471805599453) + m_log(cabs(C(x, y))), m_atan2(y, x));
    else
    {
        C v = C(x, y);
        w = c_log(v + c_sqrt(cmul(v, v) + C(S(1), S(0))));
    }
    return C(c_copysign(w.x, z.x), c_copysign(w.y, z.y));
}

// log(z + sqrt(z + 1) sqrt(z - 1)), which is log(2z) to working precision for
// large |z|; the imaginary part takes the sign of y (acosh(conj z) = conj acosh(z)).
C c_acosh(C z)
{
    S x = z.x, y = z.y;
    if (isinf(x) || isinf(y))
    {
        if (isnan(x) || isnan(y))
            return C(m_inf(), m_nan());
        S im = isinf(y) ? (isinf(x) ? (x < S(0) ? LIT(2.356194490192345) : LIT(0.7853981633974483))
                                    : LIT(1.5707963267948966))
                        : (x < S(0) ? LIT(3.141592653589793) : S(0));
        return C(m_inf(), c_copysign(im, y));
    }
    if (isnan(x) || isnan(y))
        return C(m_nan(), m_nan());
    if (max(abs(x), abs(y)) > LIT(1e8))
        return C(LIT(0.6931471805599453) + m_log(cabs(z)), m_atan2(y, x));
    C w = c_log(z + cmul(c_sqrt(z + C(S(1), S(0))), c_sqrt(z - C(S(1), S(0)))));
    return C(abs(w.x), c_copysign(abs(w.y), y));
}

C c_atanh(C z)
{
    S x = z.x, y = z.y;
    if (isinf(x) || isinf(y))  // atanh(z) -> 0 + i pi/2 sign y (an infinite x with a nan y: 0 + i nan)
        return C(c_copysign(S(0), x), isnan(y) ? y : c_copysign(LIT(1.5707963267948966), y));
    if (isnan(x) || isnan(y))
        return C(x == S(0) ? x : m_nan(), m_nan());
    S r = max(abs(x), abs(y));
    if (r > LIT(1e8))  // 1/z to working precision: x / |z|^2 + i pi/2 sign y
    {
        S h = cabs(z);
        return C((x / h) / h, c_copysign(LIT(1.5707963267948966), y));
    }
    return S(0.5) * (c_log1p(z) - c_log1p(-z));
}

C c_asin(C z) { return c_neg_i_times(c_asinh(c_i_times(z))); }  // asin z = -i asinh(i z)
C c_atan(C z) { return c_neg_i_times(c_atanh(c_i_times(z))); }  // atan z = -i atanh(i z)

C c_acos(C z)
{
    C a = c_asin(z);
    return C(LIT(1.5707963267948966) - a.x, -a.y);
}

C c_pow(C a, C b)
{
    if (b.x == S(0) && b.y == S(0))
        return C(S(1), S(0));
    if (a.x == S(0) && a.y == S(0))
        return b.y == S(0) && b.x > S(0) ? C(S(0), S(0)) : C(m_nan(), m_nan());
    return c_exp(cmul(b, c_log(a)));
}
