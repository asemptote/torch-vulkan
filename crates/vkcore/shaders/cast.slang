// axis SRC: f32=float f64=double i32=int i64=int64_t b8=uint8_t c64=float2 c128=double2
// axis DST: f32=float f64=double i32=int i64=int64_t b8=uint8_t c64=float2 c128=double2
//
// out = DST(a), strided. With SRC == DST this is a strided copy (copy_,
// contiguous, clone). Conversion to bool maps nonzero to 1; complex to real
// keeps the real part (as PyTorch does, with a warning on its side); real to
// complex sets the imaginary part to 0.

#define OUT_T DST
#define IN_T SRC
#include "_strided.slang"

#if defined(SRC_c64) || defined(SRC_c128)
#define SRC_COMPLEX 1
#endif
#if defined(DST_c64)
#define DST_PART float
#elif defined(DST_c128)
#define DST_PART double
#endif

DST convert(SRC x)
{
#if defined(DST_b8)
#ifdef SRC_COMPLEX
    return (x.x != 0 || x.y != 0) ? uint8_t(1) : uint8_t(0);
#else
    return x != SRC(0) ? uint8_t(1) : uint8_t(0);
#endif
#elif defined(DST_PART)
#ifdef SRC_COMPLEX
    return DST(DST_PART(x.x), DST_PART(x.y));
#else
    return DST(DST_PART(x), DST_PART(0));
#endif
#else
#ifdef SRC_COMPLEX
    return DST(x.x);
#else
    return DST(x);
#endif
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        p.out[o.o] = convert(p.a[o.a]);
    }
}
