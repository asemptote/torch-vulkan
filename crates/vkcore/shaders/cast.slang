// axis SRC: f32=float f64=double i32=int i64=int64_t b8=uint8_t
// axis DST: f32=float f64=double i32=int i64=int64_t b8=uint8_t
//
// out = DST(a), strided. With SRC == DST this is a strided copy (copy_,
// contiguous, clone). Conversion to bool maps nonzero to 1.

#define OUT_T DST
#define IN_T SRC
#include "_strided.slang"

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        SRC x = p.a[o.a];
#ifdef DST_b8
        p.out[o.o] = x != SRC(0) ? uint8_t(1) : uint8_t(0);
#else
        p.out[o.o] = DST(x);
#endif
    }
}
