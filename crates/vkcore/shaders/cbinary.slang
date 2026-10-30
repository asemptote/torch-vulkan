// axis T: c64=float2 c128=double2
//
// Complex out = op(a, b), strided with broadcasting. OP: CBINARY_ADD
// (a + alpha b), CBINARY_SUB (a - alpha b), CBINARY_MUL, CBINARY_DIV,
// CBINARY_POW (a^b, principal branch); alpha is real. B_SCALAR replaces b
// with the complex scalar (alpha, beta), in which case the host has already
// folded alpha into it.

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
const int OP = CBINARY_ADD;
[vk::constant_id(1)]
const int B_SCALAR = 0;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    S mult = B_SCALAR != 0 ? S(1) : S(alpha());
    T scalar = T(S(alpha()), S(beta()));
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        T a = p.a[o.a];
        T b = B_SCALAR != 0 ? scalar : p.b[o.b];
        T r;
        switch (OP)
        {
        case CBINARY_ADD: r = a + mult * b; break;
        case CBINARY_SUB: r = a - mult * b; break;
        case CBINARY_MUL: r = cmul(a, b); break;
        case CBINARY_POW: r = c_pow(a, b); break;
        default: r = cdiv(a, b); break;  // CBINARY_DIV
        }
        p.out[o.o] = r;
    }
}
