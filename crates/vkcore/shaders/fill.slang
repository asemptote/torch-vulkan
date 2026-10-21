// axis T: f32=float f64=double i32=int i64=int64_t b8=uint8_t c64=float2 c128=double2
//
// out = scalar, strided. The scalar is the parameter alpha; complex values
// carry the real part in alpha and the imaginary part in beta.

#define OUT_T T
#define IN_T T
#include "_strided.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
#if defined(T_c64) || defined(T_c128)
    T value = T(alpha(), beta());
#elif defined(T_f32) || defined(T_f64)
    T value = T(alpha());
#else
    T value = T(alpha_int());
#endif
    for (uint i = id.x; i < p.n; i += p.stride)
        p.out[offsets(i).o] = value;
}
