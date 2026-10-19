// Shared parameter block and index math for strided element-wise kernels.
//
// The including file defines OUT_T (output element type) and IN_T (input
// element type) before including this file.
//
// Layout (128 bytes, every field naturally aligned, so std430 and scalar
// layouts agree; the host packs it as `<QQQdqIIII8x4I4i4i4i`):
//   0  out, a, b      element pointers (storage offset already applied)
//   24 alpha          float scalar (alpha multiplier, scalar operand, fill value)
//   32 alpha_i        integer version of the same scalar
//   40 n              number of output elements
//   44 ndim           number of (coalesced) dims, <= 4; dim 0 is innermost
//   48 stride         grid stride (total threads launched)
//   64 size[4]        dim sizes
//   80 so/sa/sb[4]    element strides of out, a, b (0 = broadcast)

struct Params
{
    OUT_T* out;
    IN_T* a;
    IN_T* b;
    double alpha;
    int64_t alpha_i;
    uint n;
    uint ndim;
    uint stride;
    uint _pad0;
    uint4 size;
    int4 so;
    int4 sa;
    int4 sb;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

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
