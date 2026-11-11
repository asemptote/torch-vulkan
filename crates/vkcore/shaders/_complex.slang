// Complex arithmetic on vector types: float2 / double2 hold (real, imag).
// The including file defines S as the part type (float or double).

vector<S, 2> cmul(vector<S, 2> a, vector<S, 2> b)
{
    return vector<S, 2>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vector<S, 2> cconj(vector<S, 2> a)
{
    return vector<S, 2>(a.x, -a.y);
}

// The parts' quotient in cdiv: _fmath.slang's correctly rounded m_div where
// the including file asks for it (CDIV_EXACT, after including _fmath.slang:
// the element-wise division, which must equal c10::complex's, whose
// divisions are IEEE ones), else OpFDiv (up to 2.5 ulp; the linear-algebra
// kernels' pivots and rotations).
S cdiv_part(S a, S b)
{
#ifdef CDIV_EXACT
    return m_div(a, b);
#else
    return a / b;
#endif
}

// Smith's algorithm, as c10::complex's operator/: avoids overflow/underflow
// of |b|^2 for large or small b.
vector<S, 2> cdiv(vector<S, 2> a, vector<S, 2> b)
{
    if (abs(b.x) >= abs(b.y))
    {
        if (b.x == S(0) && b.y == S(0))  // (as c10::complex: (a.x / 0, a.y / 0), inf or NaN by part)
            return vector<S, 2>(a.x / abs(b.x), a.y / abs(b.y));
        S r = cdiv_part(b.y, b.x);
        S d = b.x + b.y * r;
        return vector<S, 2>(cdiv_part(a.x + a.y * r, d), cdiv_part(a.y - a.x * r, d));
    }
    S r = cdiv_part(b.x, b.y);
    S d = b.x * r + b.y;
    return vector<S, 2>(cdiv_part(a.x * r + a.y, d), cdiv_part(a.y * r - a.x, d));
}

// |a| without overflow for large parts.
S cabs(vector<S, 2> a)
{
    S x = abs(a.x), y = abs(a.y);
    if (isinf(x) || isinf(y))  // (C99's hypot: inf even with a NaN part)
        return S(1) / S(0);
    if (isnan(x) || isnan(y))  // (max below drops NaNs on GPUs)
        return x + y;
    S m = max(x, y);
    if (m == S(0) || isinf(m))
        return m;
    S u = x / m, v = y / m;
    return m * sqrt(u * u + v * v);
}
