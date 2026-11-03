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

#if (defined(DST_f64) || defined(DST_c128)) && (defined(SRC_f32) || defined(SRC_bf16) || defined(SRC_f16) || defined(SRC_c64))
// float to double keeping a NaN's sign and payload (shifted up, quiet bit
// set), as x86 does: NVIDIA's conversion returns a canonical NaN.
double widen(float x)
{
    if (!isnan(x)) return double(x);
    uint u = bit_cast<uint>(x);
    return bit_cast<double>((uint64_t(u >> 31) << 63) | 0x7FF8000000000000ull | (uint64_t(u & 0x7fffffu) << 29));
}
#define TO_WIDE(v) widen(float(v))
#endif

// The low 32 bits of float x truncated to int64, as the CPU converts: |x| <
// 2^31 directly; up to 2^63 from the mantissa shifted into place (the
// magnitude's low word, negated for x < 0); beyond, and NaN, int64's
// "indefinite" value 0x8000000000000000, whose low word is 0.
int wrap32(float x)
{
    float a = abs(x);
    if (a < 2147483648.0f)
        return int(x);
    if (!(a < 9.223372036854775808e18f))
        return 0;
    uint bits = asuint(a);
    uint sh = ((bits >> 23) & 0xffu) - 150u;  // a = m 2^sh, 24-bit m, sh >= 8 here
    uint low = sh >= 32u ? 0u : ((bits & 0x7fffffu) | 0x800000u) << sh;
    return x < 0.0f ? -int(low) : int(low);
}

// float x truncated to int32 as the CPU converts to int16 (through int32,
// not int64): out of range, and NaN, int32's "indefinite" 0x80000000, whose
// low 16 bits are 0.
int trunc32(float x)
{
    return abs(x) < 2147483648.0f ? int(x) : 0;
}

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
    // As the CPU, through int64, so out-of-range values wrap (e.g. -3.0 ->
    // 253): only the int64's low word matters, which wrap32 forms without
    // 64-bit integers (Mali GPUs lack them; float64 sources keep them).
    // (Half through float: Intel's UHD 620 driver saturated half -> int64 at
    // 127, e.g. 178.6 -> 127 as uint8; lavapipe and NVIDIA wrap as the CPU.)
#if defined(SRC_f64)
    return DV(int64_t(x));
#elif defined(SRC_c128)
    return DV(int64_t(x.x));
#elif defined(SRC_COMPLEX) && defined(DST_i16)
    return DV(trunc32(x.x));
#elif defined(SRC_COMPLEX)
    return DV(wrap32(x.x));
#elif (defined(SRC_f32) || defined(SRC_f16) || defined(SRC_bf16)) && defined(DST_i16)
    return DV(trunc32(float(x)));
#elif defined(SRC_f32) || defined(SRC_f16) || defined(SRC_bf16)
    return DV(wrap32(float(x)));
#else
    return DV(int(x));  // (integer sources fit)
#endif
#elif defined(DST_PART) && defined(TO_WIDE)
#ifdef SRC_COMPLEX
    return DV(TO_WIDE(x.x), TO_WIDE(x.y));
#else
    return DV(TO_WIDE(x), DST_PART(0));
#endif
#elif defined(DST_PART)
#ifdef SRC_COMPLEX
    return DV(DST_PART(x.x), DST_PART(x.y));
#else
    return DV(DST_PART(x), DST_PART(0));
#endif
#elif defined(TO_WIDE)
#ifdef SRC_COMPLEX
    return DV(TO_WIDE(x.x));
#else
    return DV(TO_WIDE(x));
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
