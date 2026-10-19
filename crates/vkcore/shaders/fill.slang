// axis T: f32=float f64=double i32=int i64=int64_t b8=uint8_t
//
// out = scalar (alpha for float types, alpha_i otherwise), strided.

#define OUT_T T
#define IN_T T
#include "_strided.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
#if defined(T_f32) || defined(T_f64)
    T value = T(p.alpha);
#else
    T value = T(p.alpha_i);
#endif
    for (uint i = id.x; i < p.n; i += p.stride)
        p.out[offsets(i).o] = value;
}
