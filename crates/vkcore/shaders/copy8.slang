// axis S: b1=uint8_t b2=uint16_t b4=uint b8=uint64_t b16=uint4
//
// out = a, strided over up to 8 dims, by element size (any dtype of that
// size). The strided kernels (_strided.slang) take 4 dims and the host loops
// over the rest, one dispatch per index: a 6-dim im2col view of a batch of
// images took 224. This two-operand layout has room for 8 dims in the same
// 128 bytes (csrc/launch.cpp: copy_many_dims). Sizes and strides are
// vectors, not arrays: Intel's Windows driver misreads dynamically indexed
// arrays in push constants.

struct Params
{
    S* out;        // 0
    S* a;          // 8
    uint n;        // 16  elements
    uint ndim;     // 20  dims (<= 8), innermost first
    uint stride;   // 24  grid stride
    uint _pad;     // 28
    uint4 size0;   // 32  sizes of dims 0..3
    uint4 size1;   // 48  dims 4..7
    int4 so0;      // 64  out's element strides
    int4 so1;      // 80
    int4 sa0;      // 96  a's element strides
    int4 sa1;      // 112
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        uint rem = i;
        int o = 0, a = 0;
        [unroll]
        for (uint d = 0; d < 4; d++)
        {
            if (d < p.ndim)
            {
                int c = int(rem % p.size0[d]);
                rem /= p.size0[d];
                o += c * p.so0[d];
                a += c * p.sa0[d];
            }
        }
        [unroll]
        for (uint d = 0; d < 4; d++)
        {
            if (d + 4 < p.ndim)
            {
                int c = int(rem % p.size1[d]);
                rem /= p.size1[d];
                o += c * p.so1[d];
                a += c * p.sa1[d];
            }
        }
        p.out[o] = p.a[a];
    }
}
