// axis T: f32=float f64=double i32=int i64=int64_t
//
// Three-input element-wise ops, strided with broadcasting:
//   TERNARY_LERP:     a + w (b - a), w = c (or alpha with C_SCALAR);
//                     PyTorch's formula (b - (b - a)(1 - w) for |w| >= 0.5)
//   TERNARY_ADDCMUL:  a + alpha * b * c
//   TERNARY_ADDCDIV:  a + alpha * b / c
//   TERNARY_CLAMP:    min(max(a, b), c) with b / c absent per HAS_LO /
//                     HAS_HI; NaN in a propagates (floats)

#define OUT_T T
#define A_T T
#define B_T T
#define C_T T
#include "_strided4.slang"

[vk::constant_id(0)] const int OP = TERNARY_LERP;
[vk::constant_id(1)] const int C_SCALAR = 0;
[vk::constant_id(2)] const int HAS_LO = 1;
[vk::constant_id(3)] const int HAS_HI = 1;

#if defined(T_f32) || defined(T_f64)
#define IS_FLOAT 1
T scalar() { return T(alpha()); }
#if defined(T_f32)
typedef uint TBITS;
#else
typedef uint64_t TBITS;
#endif
#else
T scalar() { return T(alpha_int()); }
#endif

T apply(T a, T b, T c)
{
    switch (OP)
    {
#ifdef IS_FLOAT
    case TERNARY_LERP: {
        T diff = b - a;
        return abs(c) < T(0.5) ? a + c * diff : b - diff * (T(1) - c);
    }
#endif
    case TERNARY_ADDCMUL: return a + scalar() * b * c;
    case TERNARY_ADDCDIV: return a + scalar() * b / c;
    case TERNARY_CLAMP: {
#ifdef IS_FLOAT
        if (isnan(a)) return a;
        if (HAS_LO != 0 && isnan(b)) return b;
        if (HAS_HI != 0 && isnan(c)) return c;
#endif
        T r = a;  // (comparisons, not max/min: x where it equals a bound, so -0 stays -0)
#ifdef IS_FLOAT
        // A select of a or a bound gets compiled into min/max (which return the
        // bound on ties, +0 for -0): a's bits plus a zero the compiler cannot
        // see (the scalar parameter, 0 for clamp) hide that it is a select of a.
        TBITS zero = bit_cast<TBITS>(scalar());
        // min(max(a, b), c): the upper bound applies to max(a, b) (b > c gives c)
        TBITS below = TBITS(0) - TBITS(HAS_LO != 0 && a < b ? 1 : 0);
        TBITS bits = (bit_cast<TBITS>(b) & below) | ((bit_cast<TBITS>(a) + zero) & ~below);
        TBITS above = TBITS(0) - TBITS(HAS_HI != 0 && bit_cast<T>(bits) > c ? 1 : 0);
        bits = (bit_cast<TBITS>(c) & above) | (bits & ~above);
        return bit_cast<T>(bits);
#else
        if (HAS_LO != 0 && r < b) r = b;
        if (HAS_HI != 0 && r > c) r = c;
#endif
        return r;
    }
    default: return a;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets4 o = offsets4(i);
        T a = p.a[o.a];
        T b = (OP == TERNARY_CLAMP && HAS_LO == 0) ? a : p.b[o.b];
        T c = C_SCALAR != 0 ? scalar() : ((OP == TERNARY_CLAMP && HAS_HI == 0) ? a : p.c[o.c]);
        p.out[o.o] = apply(a, b, c);
    }
}
