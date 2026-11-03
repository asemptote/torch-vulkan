// Shared parameter block and index math for strided element-wise kernels.
//
// The including file defines OUT_T (output element type) and IN_T (input
// element type), and optionally B_T (b's element type, default IN_T);
// PARAM_T comes from _types.slang (the kernel's T axis).
// Kernels other than float64 ones never touch a double, so they run on GPUs
// without float64 support.
//
// Layout (128 bytes; csrc/launch.cpp packs it as StridedLayout<3>). Sizes and
// strides are 4-vectors rather than arrays: Intel's Windows driver misreads
// dynamically indexed arrays in push constants.
//   0   out, a, b               element pointers (storage offset applied)
//   24  alpha, beta, gamma      raw 64-bit parameter slots (_slot.slang),
//                               packed by the host per T: an int64 for
//                               integer and bool kernels, double bits for
//                               float64/complex128, float bits in the low
//                               word otherwise
//   48  n, ndim, stride, _pad   element count, merged dims (<= 4), grid stride
//   64  size                    dim sizes, innermost first
//   80  so, sa, sb              element strides of out, a, b (0 = broadcast)

#include "_types.slang"
#include "opcodes.slang"

#ifndef B_T
#define B_T IN_T
#endif

struct Params
{
    OUT_T* out;
    IN_T* a;
    B_T* b;
    slot64 alpha_bits;
    slot64 beta_bits;
    slot64 gamma_bits;
    uint n;
    uint ndim;
    uint stride;
    uint _pad;
    uint4 size;
    int4 so;
    int4 sa;
    int4 sb;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

PARAM_T param(slot64 bits)
{
#if defined(PARAM_DOUBLE)
    return slot_double(bits);
#else
    return slot_float(bits);
#endif
}

PARAM_T alpha() { return param(p.alpha_bits); }
PARAM_T beta() { return param(p.beta_bits); }
PARAM_T gamma() { return param(p.gamma_bits); }
slot_int_t alpha_int() { return slot_integer(p.alpha_bits); }
slot_int_t beta_int() { return slot_integer(p.beta_bits); }

struct Offsets
{
    int o;
    int a;
    int b;
};

// Element offsets of linear output index i in out, a and b.
Offsets offsets(uint i)
{
    Offsets r = { 0, 0, 0 };
    uint rem = i;
    [unroll]
    for (uint d = 0; d < 4; d++)
    {
        if (d < p.ndim)
        {
            uint s = p.size[d];
            int c = int(rem % s);
            rem /= s;
            r.o += c * p.so[d];
            r.a += c * p.sa[d];
            r.b += c * p.sb[d];
        }
    }
    return r;
}
