// Parameter block and index math for strided kernels with four operands
// (out, a, b, c), e.g. where and lerp. The including file defines OUT_T,
// A_T, B_T and C_T; PARAM_T comes from _types.slang (the kernel's T axis).
//
// Layout (128 bytes; csrc/launch.cpp packs it as StridedLayout<4>), using
// 4-vectors rather than arrays as in _strided.slang:
//   0   out, a, b, c        element pointers (storage offset applied)
//   32  alpha, beta         raw 64-bit parameter slots, packed per T as in
//                           _strided.slang
//   48  n, ndim, stride, _pad  element count, merged dims (<= 3), grid stride
//   64  size                dim sizes, innermost first (w unused)
//   80  st0, st1, st2       element strides (0 = broadcast), 12 ints:
//                           out0..2, a0..2, b0..2, c0..2

#include "_types.slang"
#include "opcodes.slang"

struct Params4
{
    OUT_T* out;
    A_T* a;
    B_T* b;
    C_T* c;
    uint64_t alpha_bits;
    uint64_t beta_bits;
    uint n;
    uint ndim;
    uint stride;
    uint _pad;
    uint4 size;
    int4 st0;
    int4 st1;
    int4 st2;
};

[[vk::push_constant]] ConstantBuffer<Params4> p;

PARAM_T param(uint64_t bits)
{
#if defined(PARAM_DOUBLE)
    return bit_cast<double>(bits);
#else
    return bit_cast<float>(uint(bits));
#endif
}

PARAM_T alpha() { return param(p.alpha_bits); }
PARAM_T beta() { return param(p.beta_bits); }
int64_t alpha_int() { return int64_t(p.alpha_bits); }
int64_t beta_int() { return int64_t(p.beta_bits); }

struct Offsets4
{
    int o;
    int a;
    int b;
    int c;
};

// Stride j of the 12 (operand k, dim d at j = 3k + d).
int stride_at(uint j)
{
    int4 v = j < 4 ? p.st0 : (j < 8 ? p.st1 : p.st2);
    return v[j & 3];
}

Offsets4 offsets4(uint i)
{
    Offsets4 r = { 0, 0, 0, 0 };
    uint rem = i;
    [unroll]
    for (uint d = 0; d < 3; d++)
    {
        if (d < p.ndim)
        {
            uint s = p.size[d];
            int k = int(rem % s);
            rem /= s;
            r.o += k * stride_at(d);
            r.a += k * stride_at(3 + d);
            r.b += k * stride_at(6 + d);
            r.c += k * stride_at(9 + d);
        }
    }
    return r;
}
