// axis T: c64=float2 c128=double2
//
// Complex out = op(a), strided. OP: 0 neg, 1 conj.

#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif

#define OUT_T T
#define IN_T T
#include "_strided.slang"

[vk::constant_id(0)]
const int OP = 0;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        T a = p.a[o.a];
        p.out[o.o] = OP == 0 ? -a : T(a.x, -a.y);
    }
}
