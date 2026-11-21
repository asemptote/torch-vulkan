// axis T: f16=half bf16=uint16_t f32=float f64=double
//
// The last step of a split-k matrix product (gemmcm.slang and gemmsg.slang
// with SPLIT = 1, csrc/linalg.cpp's split_k): C[b] = alpha * sum_s W[b S + s]
// (+ beta * Cin[b] when USE_CIN), for the S partial sums W[b S + s] = A[b]'s
// columns s kc .. times B[b]'s rows s kc .., each an m x n row-major slice of
// the workspace. They are added in the order s = 0, 1, ..., S - 1, in
// float32 (float64 for float64), so the result does not depend on how the
// GPU scheduled the products; 16-bit results are rounded once, to nearest
// even, as in gemmcm.slang.

#include "_slot.slang"

#if defined(T_f16) || defined(T_bf16)
#include "_gemm16.slang"
typealias P = float;
P widen(T x) { return to_f(x); }
T narrow(P v) { return from_f(v); }
P param(slot64 bits) { return slot_float(bits); }
#else
typealias P = T;
P widen(T x) { return x; }
T narrow(P v) { return v; }
#if defined(T_f64)
P param(slot64 bits) { return slot_double(bits); }
#else
P param(slot64 bits) { return slot_float(bits); }
#endif
#endif

struct Params
{
    P* ws;   // [batch S][m][n]
    T* c;
    T* cin;
    slot64 alpha;  // float32 bits, float64's for float64
    slot64 beta;
    uint m;
    uint n;
    uint splits;  // S
    uint stride;  // grid stride over m n
    int c_sm, c_sn, c_sb;
    int cin_sm, cin_sn, cin_sb;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int USE_CIN = 0;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint b = id.y, mn = p.m * p.n;
    P alpha = param(p.alpha), beta = param(p.beta);
    for (uint i = id.x; i < mn; i += p.stride)
    {
        uint row = i / p.n, col = i % p.n, base = b * p.splits * mn + i;
        P sum = p.ws[base];
        for (uint s = 1; s < p.splits; s++)
            sum += p.ws[base + s * mn];
        P v = alpha * sum;
        if (USE_CIN != 0)  // beta == 0 must ignore Cin entirely (it may hold NaN)
            v += beta * widen(p.cin[int(b) * p.cin_sb + int(row) * p.cin_sm + int(col) * p.cin_sn]);
        p.c[int(b) * p.c_sb + int(row) * p.c_sm + int(col) * p.c_sn] = narrow(v);
    }
}
