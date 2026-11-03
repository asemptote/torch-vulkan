// int64 without the Int64 capability (Mali GPUs lack it; the same code runs
// on every GPU, so that it is tested everywhere): I64 holds an int64 as two
// 32-bit words, little-endian (the memory layout of int64_t), with the C++
// semantics of int64_t: two's complement, sums, differences and products
// mod 2^64, division truncating toward zero, shifts by the amount mod 64.
// U64 is the unsigned counterpart (sort keys).

#pragma once

#include "_wide.slang"

struct I64
{
    uint lo;
    uint hi;
    __init(int v)
    {
        lo = asuint(v);
        hi = v < 0 ? 0xffffffffu : 0u;
    }
    __init(uint l, uint h)
    {
        lo = l;
        hi = h;
    }
    __init(I64 v)
    {
        lo = v.lo;
        hi = v.hi;
    }
    __init(uint8_t v)
    {
        lo = uint(v);
        hi = 0u;
    }
};

I64 i64_from_uint(uint v) { return I64(v, 0u); }
I64 i64_min() { return I64(0u, 0x80000000u); }
int i64_low(I64 a) { return asint(a.lo); }  // (the value mod 2^32, as an int)
bool i64_nonzero(I64 a) { return (a.lo | a.hi) != 0u; }
bool i64_negative(I64 a) { return asint(a.hi) < 0; }

I64 operator+(I64 a, I64 b)
{
    uint lo = a.lo + b.lo;
    return I64(lo, a.hi + b.hi + (lo < a.lo ? 1u : 0u));
}
I64 operator-(I64 a, I64 b) { return I64(a.lo - b.lo, a.hi - b.hi - (a.lo < b.lo ? 1u : 0u)); }
I64 operator-(I64 a) { return I64(0) - a; }
I64 operator*(I64 a, I64 b)
{
    uint2 p = mul_wide32(a.lo, b.lo);
    return I64(p.x, p.y + a.hi * b.lo + a.lo * b.hi);
}
I64 operator&(I64 a, I64 b) { return I64(a.lo & b.lo, a.hi & b.hi); }
I64 operator|(I64 a, I64 b) { return I64(a.lo | b.lo, a.hi | b.hi); }
I64 operator^(I64 a, I64 b) { return I64(a.lo ^ b.lo, a.hi ^ b.hi); }
I64 operator~(I64 a) { return I64(~a.lo, ~a.hi); }

I64 shl(I64 a, uint s)
{
    s &= 63u;
    if (s == 0u) return a;
    if (s >= 32u) return I64(0u, a.lo << (s - 32u));
    return I64(a.lo << s, (a.hi << s) | (a.lo >> (32u - s)));
}
I64 sar(I64 a, uint s)  // arithmetic
{
    s &= 63u;
    if (s == 0u) return a;
    if (s >= 32u) return I64(uint(asint(a.hi) >> (s - 32u)), uint(asint(a.hi) >> 31));
    return I64((a.lo >> s) | (a.hi << (32u - s)), uint(asint(a.hi) >> s));
}
I64 shr(I64 a, uint s)  // logical
{
    s &= 63u;
    if (s == 0u) return a;
    if (s >= 32u) return I64(a.hi >> (s - 32u), 0u);
    return I64((a.lo >> s) | (a.hi << (32u - s)), a.hi >> s);
}
I64 operator<<(I64 a, I64 s) { return shl(a, s.lo); }
I64 operator>>(I64 a, I64 s) { return sar(a, s.lo); }

bool operator==(I64 a, I64 b) { return a.lo == b.lo && a.hi == b.hi; }
bool operator!=(I64 a, I64 b) { return a.lo != b.lo || a.hi != b.hi; }
bool operator<(I64 a, I64 b) { return asint(a.hi) < asint(b.hi) || (a.hi == b.hi && a.lo < b.lo); }
bool operator>(I64 a, I64 b) { return b < a; }
bool operator<=(I64 a, I64 b) { return !(b < a); }
bool operator>=(I64 a, I64 b) { return !(a < b); }

// |a| (int64 min stays itself, as in C++'s two's complement).
I64 abs(I64 a) { return i64_negative(a) ? -a : a; }
I64 min(I64 a, I64 b) { return b < a ? b : a; }
I64 max(I64 a, I64 b) { return a < b ? b : a; }
I64 clamp(I64 x, I64 lo, I64 hi) { return min(max(x, lo), hi); }
int sign(I64 a) { return i64_negative(a) ? -1 : (i64_nonzero(a) ? 1 : 0); }

// Unsigned a / b and a % b, b != 0 (one quotient bit a step).
void udivmod(I64 a, I64 b, out I64 q, out I64 r)
{
    if (a.hi == 0u && b.hi == 0u)
    {
        q = I64(a.lo / b.lo, 0u);
        r = I64(a.lo % b.lo, 0u);
        return;
    }
    q = I64(0);
    r = I64(0);
    for (int i = 63; i >= 0; i--)
    {
        r = I64(r.lo << 1, (r.hi << 1) | (r.lo >> 31));
        r.lo |= i >= 32 ? (a.hi >> uint(i - 32)) & 1u : (a.lo >> uint(i)) & 1u;
        bool ge = r.hi > b.hi || (r.hi == b.hi && r.lo >= b.lo);
        if (ge)
        {
            r = r - b;
            if (i >= 32) q.hi |= 1u << uint(i - 32);
            else q.lo |= 1u << uint(i);
        }
    }
}

// Signed division truncating toward zero, and its remainder (sign of a):
// C++'s int64 / and %. Division by zero gives 0 (the kernels' callers guard
// it, as the CPU raises), and int64 min / -1 wraps to int64 min.
I64 operator/(I64 a, I64 b)
{
    if (!i64_nonzero(b)) return I64(0);
    bool neg = i64_negative(a) != i64_negative(b);
    I64 q, r;
    udivmod(abs(a), abs(b), q, r);
    return neg ? -q : q;
}
I64 operator%(I64 a, I64 b)
{
    if (!i64_nonzero(b)) return I64(0);
    I64 q, r;
    udivmod(abs(a), abs(b), q, r);
    return i64_negative(a) ? -r : r;
}

I64 operator+=(inout I64 a, I64 b) { a = a + b; return a; }
I64 operator-=(inout I64 a, I64 b) { a = a - b; return a; }
I64 operator*=(inout I64 a, I64 b) { a = a * b; return a; }
I64 operator/=(inout I64 a, I64 b) { a = a / b; return a; }
I64 operator%=(inout I64 a, I64 b) { a = a % b; return a; }
I64 operator&=(inout I64 a, I64 b) { a = a & b; return a; }
I64 operator|=(inout I64 a, I64 b) { a = a | b; return a; }
I64 operator^=(inout I64 a, I64 b) { a = a ^ b; return a; }
I64 operator<<=(inout I64 a, I64 b) { a = a << b; return a; }
I64 operator>>=(inout I64 a, I64 b) { a = a >> b; return a; }

// float(a), rounded to nearest even, as the CPU converts int64 to float.
float i64_to_float(I64 a) { return int64_words_to_float(a.lo, a.hi); }

// float x truncated to int64, as the CPU converts: out of range and NaN give
// int64's "indefinite" value, int64 min.
I64 i64_from_float(float x)
{
    float a = abs(x);
    if (a < 2147483648.0f) return I64(int(x));
    if (isnan(x) || !(a < 9.223372036854775808e18f)) return i64_min();
    uint bits = asuint(a);
    uint sh = ((bits >> 23) & 0xffu) - 150u;  // a = m 2^sh, 24-bit m, 8 <= sh < 40
    I64 m = shl(I64(((bits & 0x7fffffu) | 0x800000u), 0u), sh);
    return x < 0.0f ? -m : m;
}

#if defined(I64_DOUBLE)
// double(a), rounded once (both parts exact), and double x truncated to int64
// (as the CPU: out of range and NaN give int64 min). For float64 kernels.
double i64_to_double(I64 a) { return double(asint(a.hi)) * 4294967296.0l + double(a.lo); }
I64 i64_from_double(double x)
{
    if (isnan(x) || !(abs(x) < 9.223372036854775808e18l)) return i64_min();  // (isnan: a driver may fold the NaN compare)
    double t = trunc(x);
    double h = floor(t / 4294967296.0l);  // (exact: t is an integer below 2^63)
    double l = t - h * 4294967296.0l;     // in [0, 2^32)
    return I64(uint(l), uint(int(h)));
}
#endif

// Unsigned 64-bit sort keys.
struct U64
{
    uint lo;
    uint hi;
    __init(uint l, uint h)
    {
        lo = l;
        hi = h;
    }
};
bool operator<(U64 a, U64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
bool operator>(U64 a, U64 b) { return b < a; }
bool operator<=(U64 a, U64 b) { return !(b < a); }
bool operator>=(U64 a, U64 b) { return !(a < b); }
bool operator==(U64 a, U64 b) { return a.lo == b.lo && a.hi == b.hi; }
bool operator!=(U64 a, U64 b) { return !(a == b); }
