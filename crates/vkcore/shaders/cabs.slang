// axis T: c64=float2 c128=double2
//
// Real out = |a| for complex a, strided (overflow-safe).

#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif

#define OUT_T S
#define IN_T T
#include "_strided.slang"
#include "_complex.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        p.out[o.o] = cabs(p.a[o.a]);
    }
}
