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

// Smith's algorithm: avoids overflow/underflow of |b|^2 for large or small b.
vector<S, 2> cdiv(vector<S, 2> a, vector<S, 2> b)
{
    if (abs(b.x) >= abs(b.y))
    {
        S r = b.y / b.x;
        S d = b.x + b.y * r;
        return vector<S, 2>((a.x + a.y * r) / d, (a.y - a.x * r) / d);
    }
    S r = b.x / b.y;
    S d = b.x * r + b.y;
    return vector<S, 2>((a.x * r + a.y) / d, (a.y * r - a.x) / d);
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
