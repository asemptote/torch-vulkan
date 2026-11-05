// axis T: f32=float f64=double i32=int i64=I64
//
// out = op(a, b), strided with broadcasting. OP is a BINARY_* code; float
// math comes from _fmath.slang (GPU built-ins for float32, software for
// float64), and float fmod/remainder are exact. B_SCALAR replaces b with the
// scalar parameter alpha. With a tensor b, add/sub multiply b by alpha
// (torch.add(a, b, alpha=...)); with a scalar b the host folds alpha into it.

#define OUT_T T
#define IN_T T
#include "_strided.slang"

[vk::constant_id(0)]
const int OP = BINARY_ADD;
[vk::constant_id(1)]
const int B_SCALAR = 0;

#if defined(T_f32) || defined(T_f64)
#define IS_FLOAT 1
#include "_fmath.slang"
T scalar() { return T(alpha()); }

T nextafter_(T x, T y)
{
    if (isnan(x) || isnan(y)) return x + y;
    if (x == y) return y;
    if (x == T(0)) return y > T(0) ? bit_cast<T>(UBITS(1)) : -bit_cast<T>(UBITS(1));
    UBITS bits = bit_cast<UBITS>(x);
    bits = ((x < y) == (x > T(0))) ? bits + UBITS(1) : bits - UBITS(1);
    return bit_cast<T>(bits);
}

#else
T scalar() { return T(alpha_int()); }
#if defined(T_i32)
#define BITS 32
#else
#define BITS 64
#endif

T ipow(T base, T e)
{
    if (e < T(0))
        return base == T(1) ? T(1) : (base == T(-1) ? ((e & T(1)) != T(0) ? T(-1) : T(1)) : T(0));
    T r = T(1);
    while (e > T(0))
    {
        if ((e & T(1)) != T(0)) r *= base;
        base *= base;
        e >>= T(1);
    }
    return r;
}

// Truncated remainder (the sign of a), b != 0. Not `%`: NVIDIA's driver
// (H100, 2026) computes SPIR-V OpSRem as an unsigned remainder when an
// operand is negative (-33 % 4 gave 3); its OpSDiv is right.
T irem(T a, T b)
{
    return a - (a / b) * b;
}

T igcd(T a, T b)
{
    a = abs(a);
    b = abs(b);
    while (b != T(0))
    {
        T t = a % b;
        a = b;
        b = t;
    }
    return a;
}
#endif

T apply(T a, T b)
{
#ifdef IS_FLOAT
    T mult = B_SCALAR != 0 ? T(1) : T(alpha());
#else
    T mult = B_SCALAR != 0 ? T(1) : T(alpha_int());
#endif
    switch (OP)
    {
    case BINARY_ADD: return a + mult * b;
    case BINARY_SUB: return a - mult * b;
    case BINARY_MUL: return a * b;
#ifdef IS_FLOAT
    case BINARY_MAXIMUM: return (isnan(a) || isnan(b)) ? T(a + b) : max(a, b);  // NaN propagates
    case BINARY_MINIMUM: return (isnan(a) || isnan(b)) ? T(a + b) : min(a, b);
    case BINARY_DIV: return m_div(a, b);
    case BINARY_DIV_TRUNC: return trunc(m_div(a, b));
    case BINARY_COPYSIGN: return m_copysign(a, b);
    case BINARY_HYPOT: {
        if (isinf(a) || isinf(b)) return T(1) / T(0);
        T m = max(abs(a), abs(b));
        if (m == T(0) || isnan(m)) return m;
        T u = m_div(a, m), v = m_div(b, m);  // (not /: see m_div, m near the largest float)
        return m * sqrt(u * u + v * v);
    }
    case BINARY_NEXTAFTER: return nextafter_(a, b);
    // Activation backward, a = grad_output: the CPU kernels' formulas.
    case BINARY_GELU_BACKWARD: {
        T cdf = T(0.5) * (T(1) + m_erf(b * LIT(0.7071067811865476)));
        T pdf = LIT(0.3989422804014327) * m_exp(b * b * T(-0.5));
        return a * (cdf + b * pdf);
    }
    case BINARY_GELU_TANH_BACKWARD: {
        T x_sq = b * b, x_cube = x_sq * b;
        T inner = LIT(0.7978845608028654) * (b + LIT(0.044715) * x_cube);
        T t = m_tanh(inner);
        T left = T(0.5) * b, right = T(1) + t;
        T left_derivative = T(0.5) * right;
        T inner_derivative = LIT(0.7978845608028654) * (T(1) + T(3) * LIT(0.044715) * x_sq);
        T right_derivative = left * (T(1) - t * t) * inner_derivative;
        return a * (left_derivative + right_derivative);
    }
    case BINARY_SILU_BACKWARD: {
        T s = T(1) / (T(1) + m_exp(-b));
        return a * s * (T(1) + b * (T(1) - s));
    }
    case BINARY_SIGMOID_BACKWARD: return a * (T(1) - b) * b;
    case BINARY_TANH_BACKWARD: return a * (T(1) - b * b);
    case BINARY_MISH_BACKWARD: {
        T s = T(1) / (T(1) + m_exp(-b));
        T ts = m_tanh(m_log1p(m_exp(b)));
        return a * (ts + b * s * (T(1) - ts * ts));
    }
    case BINARY_HARDSWISH_BACKWARD: return b <= T(-3) ? T(0) : (b < T(3) ? a * (b / T(3) + T(0.5)) : a);  // (open interval, as the CPU)
    case BINARY_HARDSIGMOID_BACKWARD: return (b > T(-3) && b < T(3)) ? a * (T(1) / T(6)) : T(0);
    // (alpha, beta, gamma: the op's scalars, in T)
    case BINARY_SOFTPLUS_BACKWARD: {  // beta, threshold
        T z = m_exp(b * T(alpha()));
        return (b * T(alpha())) > T(beta()) ? a : a * z / (z + T(1));
    }
    case BINARY_ELU_BACKWARD:  // negcoef = alpha scale, poscoef = scale, negiptcoef = input_scale
        return b <= T(0) ? a * T(gamma()) * T(alpha()) * m_exp(b * T(gamma())) : a * T(beta());
    case BINARY_ELU_BACKWARD_RESULT:  // (b: the output)
        return b <= T(0) ? a * T(gamma()) * (b + T(alpha())) : a * T(beta());
    case BINARY_HARDTANH_BACKWARD: return (b <= T(alpha()) || b >= T(beta())) ? T(0) : a;
    case BINARY_SMOOTH_L1_BACKWARD: {  // b: input - target; alpha: beta, beta: norm (the CPU's kernel)
        T norm = T(beta());
        return b <= -T(alpha()) ? -norm * a : (b >= T(alpha()) ? norm * a : norm * b * a / T(alpha()));
    }
    case BINARY_HUBER_BACKWARD: {  // b: input - target; alpha: delta, beta: norm
        T norm = T(beta()), delta = T(alpha());
        return b <= -delta ? -norm * a * delta : (b >= delta ? norm * a * delta : norm * b * a);
    }
    case BINARY_LEAKY_RELU_BACKWARD: return b > T(0) ? a : a * T(alpha());
    case BINARY_LOGIT_BACKWARD: {  // alpha: eps (< 0: none), beta: 1 - eps
        if (T(alpha()) < T(0))
            return (b < T(0) || b > T(1)) ? m_nan() : ((b == T(0) || b == T(1)) ? a * m_inf() : a / (b * (T(1) - b)));
        return (b < T(alpha()) || b > T(beta())) ? T(0) : a / (b * (T(1) - b));
    }
    case BINARY_LOG_SIGMOID_BACKWARD: {
        bool neg = b < T(0);
        T z = m_exp(-abs(b));
        return a * ((neg ? T(1) : T(0)) - (neg ? T(1) : T(-1)) * (z / (T(1) + z)));
    }
    case BINARY_HEAVISIDE: return a == T(0) ? b : (a > T(0) ? T(1) : T(0));  // NaN gives 0, as the CPU
    case BINARY_FMAX: return isnan(a) ? b : (isnan(b) ? a : max(a, b));
    case BINARY_FMIN: return isnan(a) ? b : (isnan(b) ? a : min(a, b));
    case BINARY_POW: return m_pow(a, b);
    case BINARY_ATAN2: return m_atan2(a, b);
    case BINARY_LOGADDEXP: {
        if (isinf(a) && a == b) return a;
        return max(a, b) + m_log1p(m_exp(-abs(a - b)));
    }
    case BINARY_LOGADDEXP2: {
        if (isinf(a) && a == b) return a;
        return max(a, b) + m_log1p(m_exp2(-abs(a - b))) * T(M_INV_LN2);
    }
    case BINARY_XLOGY: return a == T(0) && !isnan(b) ? T(0) : a * m_log(b);
    case BINARY_XLOG1PY: return a == T(0) && !isnan(b) ? T(0) : a * m_log1p(b);
    case BINARY_REMAINDER: {  // sign of the divisor
        T r = m_fmod(a, b);
        return (r != T(0) && (r < T(0)) != (b < T(0))) ? r + b : r;
    }
    case BINARY_FMOD: return m_fmod(a, b);
    case BINARY_FLOOR_DIVIDE: return m_floor_div(a, b);
#else
    case BINARY_MAXIMUM: return max(a, b);
    case BINARY_MINIMUM: return min(a, b);
    // (x / -1 is -x, wrapping for the minimum, and x % -1 is 0: a division
    // instruction may trap or saturate there)
    case BINARY_REMAINDER: {  // sign of the divisor
        if (b == T(0) || b == T(-1)) return T(0);
        T r = irem(a, b);
        return (r != T(0) && ((r < T(0)) != (b < T(0)))) ? r + b : r;
    }
    case BINARY_FMOD: return (b == T(0) || b == T(-1)) ? T(0) : irem(a, b);  // sign of the dividend
    case BINARY_FLOOR_DIVIDE: {
        if (b == T(0)) return T(0);
        if (b == T(-1)) return T(0) - a;
        T q = a / b;
        return (a - q * b != T(0) && ((a < T(0)) != (b < T(0)))) ? q - T(1) : q;
    }
    case BINARY_DIV_TRUNC: return b == T(0) ? T(0) : (b == T(-1) ? T(0) - a : a / b);
    case BINARY_POW: return ipow(a, b);
    case BINARY_BITWISE_AND: return a & b;
    case BINARY_BITWISE_OR: return a | b;
    case BINARY_BITWISE_XOR: return a ^ b;
    case BINARY_SHIFT_LEFT: return (b < T(0) || b >= T(BITS)) ? T(0) : a << b;
    case BINARY_SHIFT_RIGHT: return (b < T(0) || b >= T(BITS)) ? (a < T(0) ? T(-1) : T(0)) : a >> b;
    case BINARY_GCD: return igcd(a, b);
    case BINARY_LCM: {
        T g = igcd(a, b);
        return g == T(0) ? T(0) : abs(a / g * b);
    }
#endif
    default: return a;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        T b = B_SCALAR != 0 ? scalar() : p.b[o.b];
        p.out[o.o] = apply(p.a[o.a], b);
    }
}
