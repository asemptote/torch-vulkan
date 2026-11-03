// axis I: i32=int i64=uint2
//
// Element offsets for indexing ops, accumulated one index tensor at a time:
//   out = (ACCUM ? a : 0) + wrap(b) * alpha
// out and a are int64 offsets, b the index tensor (broadcast to out's
// shape), alpha the indexed dim's element stride, beta its size. With WRAP,
// negative indices count from the end (valid range [-size, size)); without
// it the valid range is [0, size), as for gather and index_select. An index
// outside the range is replaced by 0 and recorded in the error word at
// address gamma, which the host checks when it next synchronizes (like
// CUDA's device-side asserts). The int64 tensors (out, a, and b for I_i64)
// are read and written as two words (_wide.slang): offsets are 32-bit, and
// an int64 index past int32's range is out of range of any dim.

#define OUT_T uint2
#define IN_T uint2
#define B_T I
#include "_strided.slang"
#include "_wide.slang"

[vk::constant_id(0)]
const int ACCUM = 0;
[vk::constant_id(1)]
const int WRAP = 1;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    int size = slot_int(p.beta_bits);
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
#if defined(I_i64)
        bool fits = i64w_fits(p.b[o.b]);
        int v = i64w_int(p.b[o.b]);
#else
        bool fits = true;
        int v = p.b[o.b];
#endif
        if (!fits || v < (WRAP != 0 ? -size : 0) || v >= size)
        {
            uint* error = words_to_uint_ptr(p.gamma_bits);
            *error = 1u;
            v = 0;
        }
        else if (v < 0)
        {
            v += size;
        }
        p.out[o.o] = i64w_from((ACCUM != 0 ? i64w_int(p.a[o.a]) : 0) + v * slot_int(p.alpha_bits));
    }
}
