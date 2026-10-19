// axis T: f32=float f64=double i32=int i64=int64_t
//
// out (bool, one byte) = a <op> b, strided. OP: 0 eq, 1 ne, 2 lt, 3 le,
// 4 gt, 5 ge. B_SCALAR: compare against the scalar in alpha/alpha_i.

#define OUT_T uint8_t
#define IN_T T
#include "_strided.slang"

[vk::constant_id(0)]
const int OP = 0;
[vk::constant_id(1)]
const int B_SCALAR = 0;

#if defined(T_f32) || defined(T_f64)
T scalar() { return T(p.alpha); }
#else
T scalar() { return T(p.alpha_i); }
#endif

bool apply(T a, T b)
{
    switch (OP)
    {
    case 0: return a == b;
    case 1: return a != b;
    case 2: return a < b;
    case 3: return a <= b;
    case 4: return a > b;
    default: return a >= b;
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
        p.out[o.o] = apply(p.a[o.a], b) ? uint8_t(1) : uint8_t(0);
    }
}
