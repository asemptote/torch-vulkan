// axis T: f32=float f64=double i32=int i64=int64_t
//
// out = op(a, b), strided, with broadcasting through zero strides.
// OP (specialization constant 0) selects the operation; B_SCALAR (constant 1)
// replaces b with the scalar in alpha/alpha_i. With a tensor b, add/sub use
// alpha as the multiplier of b (torch.add(a, b, alpha=...)); with a scalar b
// the host folds alpha into the scalar. Op codes must match _ops.py.

#define OUT_T T
#define IN_T T
#include "_strided.slang"

[vk::constant_id(0)]
const int OP = 0;
[vk::constant_id(1)]
const int B_SCALAR = 0;

#if defined(T_f32) || defined(T_f64)
#define IS_FLOAT 1
T scalar() { return T(p.alpha); }
#else
T scalar() { return T(p.alpha_i); }
#endif

T apply(T a, T b)
{
    T alpha = B_SCALAR != 0 ? T(1) : scalar();
    switch (OP)
    {
    case 0: return a + alpha * b;
    case 1: return a - alpha * b;
    case 2: return a * b;
    case 3: return max(a, b);
    case 4: return min(a, b);
#ifdef IS_FLOAT
    case 10: return a / b;
#endif
#ifdef T_f32
    case 20: return pow(a, b);
    case 21: return atan2(a, b);
#endif
#ifndef IS_FLOAT
    case 30: return a & b;
    case 31: return a | b;
    case 32: return a ^ b;
#endif
    default: return a;
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
        p.out[o.o] = apply(p.a[o.a], b);
    }
}
