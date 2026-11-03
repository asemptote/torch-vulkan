// axis T: f32=float f64=double i32=int i64=I64 b8=uint8_t c64=float2 c128=double2
//
// out = cond ? a : b, strided with broadcasting (four-operand layout: out,
// cond in slot c). A_SCALAR / B_SCALAR replace a / b with the real scalar
// parameter alpha / beta (masked_fill, where with a Python number).

#define OUT_T T
#define A_T T
#define B_T T
#define C_T uint8_t
#include "_strided4.slang"

[vk::constant_id(0)] const int A_SCALAR = 0;
[vk::constant_id(1)] const int B_SCALAR = 0;

#if defined(T_c64) || defined(T_c128)
T a_scalar() { return T(alpha(), PARAM_T(0)); }
T b_scalar() { return T(beta(), PARAM_T(0)); }
#elif defined(T_f32) || defined(T_f64)
T a_scalar() { return T(alpha()); }
T b_scalar() { return T(beta()); }
#else
T a_scalar() { return T(alpha_int()); }
T b_scalar() { return T(beta_int()); }
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets4 o = offsets4(i);
        bool cond = p.c[o.c] != uint8_t(0);
        T v;
        if (cond)
            v = A_SCALAR != 0 ? a_scalar() : p.a[o.a];
        else
            v = B_SCALAR != 0 ? b_scalar() : p.b[o.b];
        p.out[o.o] = v;
    }
}
