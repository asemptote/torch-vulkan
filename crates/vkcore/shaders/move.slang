// axis S: b1=uint8_t b2=uint16_t b4=uint b8=uint2 b16=uint4
// (8-byte elements as two words: no 64-bit integers, which Mali GPUs lack)
//
// Data movement by element size (any dtype of that size), strided. OP:
//   MOVE_COPY     out = a
//   MOVE_FILL     out = the value whose bits are in alpha (and beta for 16 bytes)
//   MOVE_GATHER   out = a[b]: a is the source restrided to out's shape with
//                 the indexed dims at stride 0, b the element offsets (int64)
//   MOVE_SCATTER  out[b] = a: out is the destination restrided the same way;
//                 with A_FILL the value comes from alpha/beta instead of a
// Duplicate offsets in a scatter write in no particular order, as on CUDA.

#define OUT_T S
#define IN_T S
#define B_T int64_t
#include "_strided.slang"

[vk::constant_id(0)]
const int OP = MOVE_COPY;
[vk::constant_id(1)]
const int A_FILL = 0;

S fill_value()
{
#if defined(S_b16)
    return uint4(p.alpha_bits.x, p.alpha_bits.y, p.beta_bits.x, p.beta_bits.y);
#elif defined(S_b8)
    return p.alpha_bits;
#else
    return S(p.alpha_bits.x);
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    S fill = fill_value();
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Offsets o = offsets(i);
        switch (OP)
        {
        case MOVE_COPY: p.out[o.o] = p.a[o.a]; break;
        case MOVE_FILL: p.out[o.o] = fill; break;
        case MOVE_GATHER: p.out[o.o] = p.a[int64_t(o.a) + p.b[o.b]]; break;
        default: p.out[int64_t(o.o) + p.b[o.b]] = A_FILL != 0 ? fill : p.a[o.a]; break;  // MOVE_SCATTER
        }
    }
}
