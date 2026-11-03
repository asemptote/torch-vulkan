// axis T: f32=float f64=double i32=int i64=I64 b8=uint8_t u8=uint8_t i8=int8_t i16=int16_t f16=half bf16=uint16_t
//
// Sort keys for shaders/sort.slang: each element of x (contiguous [rows, n])
// becomes an unsigned key whose integer order is PyTorch's sort order (NaN
// largest, -0 equal to +0; inverted when DESCENDING), plus its index in the
// row. Rows are padded to P (a power of two) and the buffer to `total`
// entries with keys that sort last. Keys are 64-bit for 8-byte types.

#include "_int64.slang"  // (the i64 variant's T: _int64.slang)
#if defined(T_f64)
typedef uint64_t K;
#define KMAX 0xffffffffffffffffull
#elif defined(T_i64)
typedef U64 K;  // (the layout of uint64_t: _int64.slang)
#define KMAX U64(0xffffffffu, 0xffffffffu)
#else
typedef uint K;
#define KMAX 0xffffffffu
#endif

struct Params
{
    T* x;
    K* keys;
    uint* idx;
    uint n;      // row length
    uint P;      // padded row length (power of two)
    uint total;  // entries in keys / idx (a multiple of P)
    uint rows;
    uint stride; // grid stride
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int DESCENDING = 0;

// (-0 made +0 on the bits: Mali's compiler folds f == 0 ? 0 : f into f)
uint key_of_float(float f)
{
    uint u = bit_cast<uint>(f);
    u = isnan(f) ? 0x7fc00000u : ((u & 0x7fffffffu) == 0u ? 0u : u);
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

K key_of(T v)
{
#if defined(T_f64)
    uint64_t u = bit_cast<uint64_t>(v);
    u = isnan(v) ? 0x7ff8000000000000ull : ((u & 0x7fffffffffffffffull) == 0ull ? 0ull : u);
    return (u >> 63) != 0ull ? ~u : (u | 0x8000000000000000ull);
#elif defined(T_i64)
    return U64(v.lo, v.hi ^ 0x80000000u);
#elif defined(T_f32)
    return key_of_float(v);
#elif defined(T_f16)
    return key_of_float(float(v));
#elif defined(T_bf16)
    return key_of_float(bit_cast<float>(uint(v) << 16));
#elif defined(T_b8)
    return v != T(0) ? 1u : 0u;  // any nonzero byte is true
#elif defined(T_u8)
    return uint(v);
#else  // i32, i8, i16
    return uint(int(v)) ^ 0x80000000u;
#endif
}

#if defined(T_i64)
K flip(K k) { return U64(~k.lo, ~k.hi); }
#else
K flip(K k) { return ~k; }
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.total; i += p.stride)
    {
        uint row = i / p.P, j = i % p.P;
        if (row < p.rows && j < p.n)
        {
            K k = key_of(p.x[row * p.n + j]);
            p.keys[i] = DESCENDING != 0 ? flip(k) : k;
        }
        else
        {
            p.keys[i] = KMAX;  // padding sorts after every real key
        }
        p.idx[i] = j;
    }
}
