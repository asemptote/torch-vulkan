// Complex functions on vector<S, 2> (real, imag) with S = float or double,
// built on _fmath.slang's real functions (the including file defines
// FMATH_F32 or FMATH_F64 and S). Formulas follow the usual principal
// branches (as std::complex / C99), arranged to avoid cancellation and
// overflow where that is cheap: hypot for |z|, log1p and expm1 for values
// near 1 and 0.

#include "_complex.slang"

typedef vector<S, 2> C;

C c_make(S re, S im) { return C(re, im); }
C c_i_times(C z) { return C(-z.y, z.x); }    // i z
C c_neg_i_times(C z) { return C(z.y, -z.x); }  // -i z

C c_exp(C z)
{
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
    S r = cabs(z);
    if (z.x >= S(0))
    {
        S t = sqrt(S(0.5) * (r + z.x));
        return C(t, z.y / (S(2) * t));
    }
    S t = sqrt(S(0.5) * (r - z.x));
    return C(abs(z.y) / (S(2) * t), m_signbit(z.y) ? -t : t);
}

C c_sinh(C z) { return C(m_sinh(z.x) * m_cos(z.y), m_cosh(z.x) * m_sin(z.y)); }
C c_cosh(C z) { return C(m_cosh(z.x) * m_cos(z.y), m_sinh(z.x) * m_sin(z.y)); }
C c_sin(C z) { return C(m_sin(z.x) * m_cosh(z.y), m_cos(z.x) * m_sinh(z.y)); }
C c_cos(C z) { return C(m_cos(z.x) * m_cosh(z.y), -m_sin(z.x) * m_sinh(z.y)); }

// tanh(x + iy) = (t (1 + s^2) + i s (1 - t^2)) / (1 + t^2 s^2), t = tanh x,
// s = tan y (Kahan); saturates to +-1 for large |x|.
C c_tanh(C z)
{
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

C c_asinh(C z) { return c_log(z + c_sqrt(cmul(z, z) + C(S(1), S(0)))); }

C c_acosh(C z)
{
    return c_log(z + cmul(c_sqrt(z + C(S(1), S(0))), c_sqrt(z - C(S(1), S(0)))));
}

C c_atanh(C z)
{
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
