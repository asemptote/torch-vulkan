// axis T: f32=float f64=double i32=int i64=int64_t f16=half
//
// Index sequences, strided over out (index i is the linear output index):
//   IOTA_ARANGE    out = alpha + i * beta
//   IOTA_LINSPACE  out = alpha + i * gamma for i < n / 2, else
//                  beta - (n - 1 - i) * gamma (PyTorch's symmetric formula;
//                  float types only). float16 rounds the product and the
//                  sum to half separately, as PyTorch's c10::Half does.

#define OUT_T T
#define IN_T T
#include "_strided.slang"
#if defined(T_f16)
#include "_half.slang"
#endif

[vk::constant_id(0)]
const int OP = IOTA_ARANGE;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
#if defined(T_f16)
        float v;
        if (OP == IOTA_ARANGE)
            v = fma(float(i), beta(), alpha());
        else if (i < p.n / 2)
            v = alpha() + float(to_half(gamma() * float(i)));
        else
            v = beta() - float(to_half(gamma() * float(p.n - 1 - i)));
        p.out[o.o] = to_half(v);
#elif defined(T_f32) || defined(T_f64)
        T v;
        if (OP == IOTA_ARANGE)
            v = fma(T(i), T(beta()), T(alpha()));
        else if (i < p.n / 2)
            v = T(alpha()) + T(gamma()) * T(i);
        else
            v = T(beta()) - T(gamma()) * T(p.n - 1 - i);
        p.out[o.o] = v;
#else
        p.out[o.o] = T(alpha_int() + slot_int_t(i) * beta_int());
#endif
    }
}
