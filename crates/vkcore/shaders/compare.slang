// axis T: f32=float f64=double i32=int i64=int64_t b8=uint8_t c64=float2 c128=double2
//
// out (bool, one byte) = predicate of a (and b), strided. OP is a COMPARE_*
// code: comparisons (complex: eq, ne only), logical ops (nonzero = true) and
// predicates of a alone. B_SCALAR compares against the scalar parameter alpha
// (complex: real part in alpha, imaginary part in beta).

#define OUT_T uint8_t
#define IN_T T
#include "_strided.slang"

[vk::constant_id(0)]
const int OP = COMPARE_EQ;
[vk::constant_id(1)]
const int B_SCALAR = 0;

#if defined(T_c64) || defined(T_c128)
#define IS_COMPLEX 1
#endif
#if defined(T_f32)
#define IS_FLOAT 1
typedef uint UBITS;
#elif defined(T_f64)
#define IS_FLOAT 1
typedef uint64_t UBITS;
#endif

#if defined(IS_COMPLEX)
T scalar() { return T(alpha(), beta()); }
bool nonzero(T x) { return x.x != 0 || x.y != 0; }
#elif defined(IS_FLOAT)
T scalar() { return T(alpha()); }
bool nonzero(T x) { return x != T(0); }
#else
T scalar() { return T(alpha_int()); }
bool nonzero(T x) { return x != T(0); }
#endif

bool apply(T a, T b)
{
    switch (OP)
    {
#ifdef IS_COMPLEX
    case COMPARE_EQ: return all(a == b);
    case COMPARE_NE: return any(a != b);
    case COMPARE_ISNAN: return isnan(a.x) || isnan(a.y);
    case COMPARE_ISINF: return isinf(a.x) || isinf(a.y);
    case COMPARE_ISFINITE: return !(isnan(a.x) || isnan(a.y) || isinf(a.x) || isinf(a.y));
#else
    case COMPARE_EQ: return a == b;
    case COMPARE_NE: return a != b;
    case COMPARE_LT: return a < b;
    case COMPARE_LE: return a <= b;
    case COMPARE_GT: return a > b;
    case COMPARE_GE: return a >= b;
#ifdef IS_FLOAT
    case COMPARE_ISNAN: return isnan(a);
    case COMPARE_ISINF: return isinf(a);
    case COMPARE_ISPOSINF: return isinf(a) && a > T(0);
    case COMPARE_ISNEGINF: return isinf(a) && a < T(0);
    case COMPARE_ISFINITE: return !isnan(a) && !isinf(a);
    case COMPARE_SIGNBIT: return (bit_cast<UBITS>(a) >> (sizeof(UBITS) * 8 - 1)) != UBITS(0);
#else
    case COMPARE_ISNAN: case COMPARE_ISINF: case COMPARE_ISPOSINF: case COMPARE_ISNEGINF: return false;
    case COMPARE_ISFINITE: return true;
    case COMPARE_SIGNBIT: return a < T(0);
#endif
#endif
    case COMPARE_LOGICAL_AND: return nonzero(a) && nonzero(b);
    case COMPARE_LOGICAL_OR: return nonzero(a) || nonzero(b);
    case COMPARE_LOGICAL_XOR: return nonzero(a) != nonzero(b);
    case COMPARE_LOGICAL_NOT: return !nonzero(a);
    default: return false;
    }
}

bool unary()
{
    return OP == COMPARE_ISNAN || OP == COMPARE_ISINF || OP == COMPARE_ISPOSINF || OP == COMPARE_ISNEGINF ||
           OP == COMPARE_ISFINITE || OP == COMPARE_SIGNBIT || OP == COMPARE_LOGICAL_NOT;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        T a = p.a[o.a];
        // (predicates of a alone get no b: its address is null)
        T b = B_SCALAR != 0 || unary() ? scalar() : p.b[o.b];
#if defined(T_b8)
        a = a != T(0) ? T(1) : T(0);  // any nonzero byte is true
        b = b != T(0) ? T(1) : T(0);
#endif
        p.out[o.o] = apply(a, b) ? uint8_t(1) : uint8_t(0);
    }
}
