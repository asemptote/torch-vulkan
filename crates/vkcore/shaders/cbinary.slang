// axis T: c64=float2 c128=double2
//
// Complex out = op(a, b), strided with broadcasting. OP: 0 add (a + alpha*b),
// 1 sub (a - alpha*b), 2 mul, 10 div; alpha is real. B_SCALAR replaces b
// with the complex scalar (alpha, bits of the imaginary part in alpha_i), in
// which case the host has already folded alpha into it.

#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif

#define OUT_T T
#define IN_T T
#include "_strided.slang"
#include "_complex.slang"

[vk::constant_id(0)]
const int OP = 0;
[vk::constant_id(1)]
const int B_SCALAR = 0;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    S alpha = B_SCALAR != 0 ? S(1) : S(p.alpha);
    T scalar = T(S(p.alpha), S(bit_cast<double>(p.alpha_i)));
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        T a = p.a[o.a];
        T b = B_SCALAR != 0 ? scalar : p.b[o.b];
        T r;
        switch (OP)
        {
        case 0: r = a + alpha * b; break;
        case 1: r = a - alpha * b; break;
        case 2: r = cmul(a, b); break;
        default: r = cdiv(a, b); break;
        }
        p.out[o.o] = r;
    }
}
