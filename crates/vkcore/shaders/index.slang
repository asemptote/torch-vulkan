// axis I: i32=int i64=int64_t
//
// Element offsets for indexing ops, accumulated one index tensor at a time:
//   out = (ACCUM ? a : 0) + wrap(b) * alpha
// out and a are int64 offsets, b the index tensor (broadcast to out's
// shape), alpha the indexed dim's element stride, beta its size. With WRAP,
// negative indices count from the end (valid range [-size, size)); without
// it the valid range is [0, size), as for gather and index_select. An index
// outside the range is replaced by 0 and recorded in the error word at
// address gamma, which the host checks when it next synchronizes (like
// CUDA's device-side asserts).

#define OUT_T int64_t
#define IN_T int64_t
#define B_T I
#include "_strided.slang"

[vk::constant_id(0)]
const int ACCUM = 0;
[vk::constant_id(1)]
const int WRAP = 1;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    int64_t size = slot_int64(p.beta_bits);
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        int64_t v = int64_t(p.b[o.b]);
        if (v < (WRAP != 0 ? -size : int64_t(0)) || v >= size)
        {
            uint* error = (uint*)uint64_t(slot_int64(p.gamma_bits));
            *error = 1u;
            v = 0;
        }
        else if (v < 0)
        {
            v += size;
        }
        p.out[o.o] = (ACCUM != 0 ? p.a[o.a] : int64_t(0)) + v * slot_int64(p.alpha_bits);
    }
}
