// axis T: f32=float f64=double i32=int i64=int64_t
//
// out = op(a), strided. OP (specialization constant 0) is a UNARY_* code
// from opcodes.txt; parameters alpha, beta, gamma. Float math
// comes from _fmath.slang (GPU built-ins for float32, software for float64).

#define OUT_T T
#define IN_T T
#include "_strided.slang"

[vk::constant_id(0)]
const int OP = UNARY_NEG;

#if defined(T_f32) || defined(T_f64)
#define IS_FLOAT 1
#include "_fmath.slang"
#endif

T apply(T x)
{
    switch (OP)
    {
    case UNARY_NEG: return -x;
    case UNARY_ABS: return abs(x);
#ifdef IS_FLOAT
    case UNARY_RELU: {
        // On the bits: drivers turn any `x < 0 ? 0 : x` (even with NaN checks
        // around it) into a max(), which drops NaN. Negative non-NaN values
        // become 0; NaN, -0.0 and the rest are kept, as on the CPU.
        UBITS bits = bit_cast<UBITS>(x);
        UBITS sign = UBITS(1) << (sizeof(UBITS) * 8 - 1);
        UBITS magnitude = bits & ~sign;
        UBITS inf_bits = bit_cast<UBITS>(m_inf());
        bool negative = (bits & sign) != UBITS(0) && magnitude != UBITS(0) && magnitude <= inf_bits;
        return negative ? T(0) : x;
    }
#else
    case UNARY_RELU: return x < T(0) ? T(0) : x;
#endif
    case UNARY_SQUARE: return x * x;
    case UNARY_SIGN: return T(sign(x));
#ifdef IS_FLOAT
    case UNARY_RECIPROCAL: return T(1) / x;
    case UNARY_SQRT: return sqrt(x);  // correctly rounded in SPIR-V
    case UNARY_RSQRT: return T(1) / sqrt(x);
    case UNARY_FLOOR: return floor(x);
    case UNARY_CEIL: return ceil(x);
    case UNARY_TRUNC: return trunc(x);
    case UNARY_ROUND: return round(x);  // half to even
    case UNARY_FRAC: return x - trunc(x);
    case UNARY_EXP: return m_exp(x);
    case UNARY_LOG: return m_log(x);
    case UNARY_SIN: return m_sin(x);
    case UNARY_COS: return m_cos(x);
    case UNARY_TANH: return m_tanh(x);
    case UNARY_SIGMOID: return m_sigmoid(x);
    case UNARY_EXP2: return m_exp2(x);
    case UNARY_LOG2: return m_log2(x);
    case UNARY_TAN: return m_tan(x);
    case UNARY_ASIN: return m_asin(x);
    case UNARY_ACOS: return m_acos(x);
    case UNARY_ATAN: return m_atan(x);
    case UNARY_SINH: return m_sinh(x);
    case UNARY_COSH: return m_cosh(x);
    case UNARY_ASINH: return m_asinh(x);
    case UNARY_ACOSH: return m_acosh(x);
    case UNARY_ATANH: return m_atanh(x);
    case UNARY_LOG10: return m_log10(x);
    case UNARY_LOG1P: return m_log1p(x);
    case UNARY_EXPM1: return m_expm1(x);
    case UNARY_ERF: return m_erf(x);
    case UNARY_ERFC: return m_erfc(x);
    case UNARY_ERFINV: return m_erfinv(x);
    case UNARY_LGAMMA: return m_lgamma(x);
    case UNARY_DIGAMMA: return m_digamma(x);
    case UNARY_SINC: return m_sinc(x);
    case UNARY_SILU: return x * m_sigmoid(x);
    case UNARY_MISH: return x * m_tanh(m_softplus(x));
    case UNARY_GELU: return T(0.5) * x * (T(1) + m_erf(x * LIT(0.7071067811865476)));
    case UNARY_GELU_TANH: {
        T inner = LIT(0.7978845608028654) * (x + LIT(0.044715) * x * x * x);
        return T(0.5) * x * (T(1) + m_tanh(inner));
    }
    case UNARY_LOG_SIGMOID: return -m_softplus(-x);
    case UNARY_SOFTPLUS: {  // beta = alpha, threshold = beta
        T b = T(alpha()), bx = b * x;
        return bx > T(beta()) ? x : m_softplus(bx) / b;
    }
    case UNARY_ELU:  // alpha, scale = beta, input_scale = gamma
        return x > T(0) ? T(beta()) * x : T(beta()) * T(alpha()) * m_expm1(x * T(gamma()));
    case UNARY_LOGIT: {  // eps = alpha, applied when gamma != 0
        if (isnan(x)) return x;  // explicit: float64 comparisons with NaN are not trusted to be false
        T z = x;
        if (T(gamma()) != T(0)) z = z < T(alpha()) ? T(alpha()) : (z > T(1) - T(alpha()) ? T(1) - T(alpha()) : z);  // NaN stays
        return m_log(z / (T(1) - z));
    }
    case UNARY_HARDTANH: return x < T(alpha()) ? T(alpha()) : (x > T(beta()) ? T(beta()) : x);  // NaN stays
    case UNARY_HARDSWISH: return isnan(x) ? x : x * min(max(x + T(3), T(0)), T(6)) / T(6);
    case UNARY_HARDSIGMOID: return isnan(x) ? x : min(max(x + T(3), T(0)), T(6)) / T(6);
    case UNARY_LEAKY_RELU: return x > T(0) ? x : x * T(alpha());
    case UNARY_SOFTSHRINK: {
        T l = T(alpha());
        // x * 0 keeps NaN and the sign of zero; the 0 is gamma() (unset, 0), as a literal 0 gets folded
        return x > l ? x - l : (x < -l ? x + l : x * T(gamma()));
    }
    case UNARY_HARDSHRINK: return (x >= -T(alpha()) && x <= T(alpha())) ? T(0) : x;  // NaN stays
    case UNARY_THRESHOLD: return x <= T(alpha()) ? T(beta()) : x;  // NaN stays
    case UNARY_NAN_TO_NUM: {  // nan, posinf, neginf
        if (isnan(x)) return T(alpha());
        if (isinf(x)) return x > T(0) ? T(beta()) : T(gamma());
        return x;
    }
    // NaN propagates; x itself where it equals a bound, so -0 stays -0, as the
    // CPU's std::max/min. (Masks on the bits: selects of this shape, even on
    // bit patterns, get compiled into min/max, which return the bound on ties.)
    case UNARY_CLAMP: {
        if (isnan(x)) return x;
        T lo = T(alpha()), hi = T(beta());
        // min(max(x, lo), hi): the upper bound applies to max(x, lo) (lo > hi gives hi)
        // (x's bits plus a zero the compiler cannot see, gamma = 0, hide the
        // select of x: see ternary.slang's clamp)
        UBITS zero = bit_cast<UBITS>(T(gamma()));
        UBITS below = UBITS(0) - UBITS(x < lo ? 1 : 0);
        UBITS r = (bit_cast<UBITS>(lo) & below) | ((bit_cast<UBITS>(x) + zero) & ~below);
        UBITS above = UBITS(0) - UBITS(bit_cast<T>(r) > hi ? 1 : 0);
        return bit_cast<T>((bit_cast<UBITS>(hi) & above) | (r & ~above));
    }
#else
    case UNARY_CLAMP: return min(max(x, T(alpha_int())), T(beta_int()));
    case UNARY_BITWISE_NOT: return ~x;
#endif
    default: return x;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        p.out[o.o] = apply(p.a[o.a]);
    }
}
