// axis SRC: f32=float f64=double i32=int i64=int64_t b8=uint8_t c64=float2 c128=double2 u8=uint8_t i8=int8_t i16=int16_t f16=half bf16=uint16_t
// axis DST: f32=float f64=double i32=int i64=int64_t b8=uint8_t c64=float2 c128=double2 u8=uint8_t i8=int8_t i16=int16_t f16=half bf16=uint16_t
//
// out = DST(a), strided, between dtypes (same-dtype copies use move.slang).
// Conversion to bool maps nonzero to 1; complex to real keeps the real part
// (as PyTorch does, with a warning on its side); real to complex sets the
// imaginary part to 0. bfloat16 is stored as its 16 bits; both 16-bit float
// types are converted through float with round-to-nearest-even.

#define OUT_T DST
#define IN_T SRC
#include "_strided.slang"

// The value types: bfloat16 is handled as float.
#if defined(SRC_bf16)
typedef float SV;
SV load(SRC x) { return bit_cast<float>(uint(x) << 16); }
#else
typedef SRC SV;
SV load(SRC x) { return x; }
#endif

#if defined(DST_f16)
#include "_half.slang"
typedef float DV;
DST store(float f) { return to_half(f); }
#elif defined(DST_bf16)
typedef float DV;
DST store(float f)
{
    uint u = bit_cast<uint>(f);
    if (isnan(f)) return uint16_t(0x7fc0);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return uint16_t(u >> 16);
}
#else
typedef DST DV;
DST store(DV x) { return x; }
#endif

#if defined(SRC_c64) || defined(SRC_c128)
#define SRC_COMPLEX 1
#endif
#if defined(DST_c64)
#define DST_PART float
#elif defined(DST_c128)
#define DST_PART double
#endif

DV convert(SV x)
{
#if defined(SRC_b8)
    x = x != SV(0) ? SV(1) : SV(0);  // any nonzero byte is true (as on the CPU)
#endif
#if defined(DST_b8)
#ifdef SRC_COMPLEX
    return (x.x != 0 || x.y != 0) ? uint8_t(1) : uint8_t(0);
#else
    return x != SV(0) ? uint8_t(1) : uint8_t(0);
#endif
#elif defined(DST_u8) || defined(DST_i8) || defined(DST_i16)
    // Through int64, so out-of-range values wrap as on the CPU (e.g. -3.0 -> 253).
#ifdef SRC_COMPLEX
    return DV(int64_t(x.x));
#else
    return DV(int64_t(x));
#endif
#elif defined(DST_PART)
#ifdef SRC_COMPLEX
    return DV(DST_PART(x.x), DST_PART(x.y));
#else
    return DV(DST_PART(x), DST_PART(0));
#endif
#else
#ifdef SRC_COMPLEX
    return DV(x.x);
#else
    return DV(x);
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
        p.out[o.o] = store(convert(load(p.a[o.a])));
    }
}
