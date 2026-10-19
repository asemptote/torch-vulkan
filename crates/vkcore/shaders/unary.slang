// axis T: f32=float f64=double i32=int i64=int64_t
//
// out = op(a), strided. OP (specialization constant 0) selects the operation;
// the driver compiles away every other case. Op codes must match
// python/torch_vulkan/_ops.py.
//
// GLSL.std.450 has no 64-bit exp/log/sin/cos/tanh, so float64 uses the
// software implementations in _math64.slang; float32 uses the GPU built-ins.

#define OUT_T T
#define IN_T T
#include "_strided.slang"
#ifdef T_f64
#include "_math64.slang"
#endif

[vk::constant_id(0)]
const int OP = 0;

#if defined(T_f32) || defined(T_f64)
#define IS_FLOAT 1
#endif

T apply(T x)
{
    switch (OP)
    {
    case 0: return -x;
    case 1: return abs(x);
    case 2: return x > T(0) ? x : T(0); // relu
    case 3: return x * x;               // square
    case 4: return T(sign(x));
#ifdef IS_FLOAT
    case 10: return T(1) / x;           // reciprocal
    case 11: return sqrt(x);
    case 12: return rsqrt(x);
    case 13: return floor(x);
    case 14: return ceil(x);
    case 15: return trunc(x);
    case 16: return round(x);           // round half to even
#endif
#ifdef T_f32
    case 20: return exp(x);
    case 21: return log(x);
    case 22: return sin(x);
    case 23: return cos(x);
    case 24: return tanh(x);
    case 25: return T(1) / (T(1) + exp(-x)); // sigmoid
    case 26: return exp2(x);
    case 27: return log2(x);
    case 28: return tan(x);
#endif
#ifdef T_f64
    case 20: return exp64(x);
    case 21: return log64(x);
    case 22: return sin64(x);
    case 23: return cos64(x);
    case 24: return tanh64(x);
    case 25: return sigmoid64(x);
    case 26: return exp2_64(x);
    case 27: return log2_64(x);
    case 28: return tan64(x);
#endif
    default: return x;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        p.out[o.o] = apply(p.a[o.a]);
    }
}
