// axis T: c64=float2 c128=double2
//
// Complex out = op(a), strided. OP is a CUNARY_* code: NEG, CONJ, and the
// complex functions (their codes equal the UNARY_* codes of the same
// functions, so the host passes those through).

#if defined(T_c64)
typedef float S;
#define FMATH_F32 1
#else
typedef double S;
#define FMATH_F64 1
#endif

#define OUT_T T
#define IN_T T
#include "_strided.slang"
#include "_fmath.slang"
#include "_cmath.slang"

[vk::constant_id(0)]
const int OP = CUNARY_NEG;

T apply(T z)
{
    switch (OP)
    {
    case CUNARY_NEG: return -z;
    case CUNARY_CONJ: return T(z.x, -z.y);
    case CUNARY_SQUARE: return cmul(z, z);
    case CUNARY_RECIPROCAL: return c_reciprocal(z);
    case CUNARY_SQRT: return c_sqrt(z);
    case CUNARY_RSQRT: return c_reciprocal(c_sqrt(z));
    case CUNARY_EXP: return c_exp(z);
    case CUNARY_LOG: return c_log(z);
    case CUNARY_SIN: return c_sin(z);
    case CUNARY_COS: return c_cos(z);
    case CUNARY_TANH: return c_tanh(z);
    case CUNARY_SIGMOID: return c_reciprocal(T(S(1), S(0)) + c_exp(-z));
    case CUNARY_EXP2: return c_exp(z * LIT(0.6931471805599453));
    case CUNARY_LOG2: return c_log(z) * LIT(1.4426950408889634);
    case CUNARY_TAN: return c_tan(z);
    case CUNARY_ASIN: return c_asin(z);
    case CUNARY_ACOS: return c_acos(z);
    case CUNARY_ATAN: return c_atan(z);
    case CUNARY_SINH: return c_sinh(z);
    case CUNARY_COSH: return c_cosh(z);
    case CUNARY_ASINH: return c_asinh(z);
    case CUNARY_ACOSH: return c_acosh(z);
    case CUNARY_ATANH: return c_atanh(z);
    case CUNARY_LOG10: return c_log(z) * LIT(0.4342944819032518);
    case CUNARY_LOG1P: return c_log1p(z);
    case CUNARY_EXPM1: return c_expm1(z);
    default: return z;
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
