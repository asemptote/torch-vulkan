// axis T: f32=float f64=double i32=int i64=int64_t b8=uint8_t u8=uint8_t i8=int8_t i16=int16_t f16=half bf16=uint16_t
//
// Sort keys for shaders/sort.slang: each element of x (contiguous [rows, n])
// becomes an unsigned key whose integer order is PyTorch's sort order (NaN
// largest, -0 equal to +0; inverted when DESCENDING), plus its index in the
// row. Rows are padded to P (a power of two) and the buffer to `total`
// entries with keys that sort last. Keys are 64-bit for 8-byte types.

#if defined(T_f64) || defined(T_i64)
typedef uint64_t K;
#define KMAX 0xffffffffffffffffull
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

uint key_of_float(float f)
{
    uint u = isnan(f) ? 0x7fc00000u : bit_cast<uint>(f == 0.0f ? 0.0f : f);
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

K key_of(T v)
{
#if defined(T_f64)
    uint64_t u = isnan(v) ? 0x7ff8000000000000ull : bit_cast<uint64_t>(v == 0.0l ? 0.0l : v);
    return (u >> 63) != 0ull ? ~u : (u | 0x8000000000000000ull);
#elif defined(T_i64)
    return uint64_t(v) ^ 0x8000000000000000ull;
#elif defined(T_f32)
    return key_of_float(v);
#elif defined(T_f16)
    return key_of_float(float(v));
#elif defined(T_bf16)
    return key_of_float(bit_cast<float>(uint(v) << 16));
#elif defined(T_b8) || defined(T_u8)
    return uint(v);
#else  // i32, i8, i16
    return uint(int(v)) ^ 0x80000000u;
#endif
}

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
            p.keys[i] = DESCENDING != 0 ? ~k : k;
        }
        else
        {
            p.keys[i] = KMAX;  // padding sorts after every real key
        }
        p.idx[i] = j;
    }
}
