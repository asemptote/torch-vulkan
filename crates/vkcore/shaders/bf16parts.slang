// The bfloat16 parts of a float32 matrix product's operand, for float32
// products at torch.set_float32_matmul_precision's "medium" and "high"
// (csrc/linalg.cpp's bf16_parts; docs/log/M41), which then run as one
// bfloat16 product on cooperative matrices (gemmcm.slang with C_F32):
//
//   y[b][r][q kp + j] = part q of x[b][r][j] for j < k, 0 for k <= j < kp,
//
// for q < parts: x and y addressed through element strides (r: A's rows or
// B's columns; j: along k), k padded to kp (a multiple of 8, so that
// gemmcm.slang loads y in 16-byte pieces). Threads run along r first where
// x is contiguous along r (R_FAST; y is then too), else along k, so that
// both the reads and the writes are coalesced. Each part's kind is two bits
// of p.kinds, part q's at bit 2q:
// - 0, hi: x rounded to bfloat16, to nearest even (NaN as 0x7fc0); where
//   that would turn a finite x into infinity, x truncated instead, so that
//   finite operands stay finite;
// - 1, lo: x - hi (exact in float32) rounded to bfloat16; 0 for infinities
//   and NaN;
// - 2, hi for finite x, else 0: the factor that meets the other operand's
//   lo, so that an infinity there is not multiplied by a lo of 0.
// "medium" takes hi alone; "high" takes A as (lo, finite hi, hi) and B as
// (finite hi, lo, hi): over 3 kp, lo_a hi_b + hi_a lo_b + hi_a hi_b (Henry
// et al., arXiv:1904.06376), the small terms first.

struct Params
{
    float* x;
    uint16_t* y;
    uint rows;
    uint k;
    uint kp;
    uint parts;
    uint kinds;
    uint stride;  // grid stride over rows kp
    int sb, sr, sk;  // x's strides
    int yb, yr, yk;  // y's
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int R_FAST = 0;

uint bf16_rne(uint u)
{
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint b = id.y, n = p.rows * p.kp;
    for (uint i = id.x; i < n; i += p.stride)
    {
        uint r = R_FAST != 0 ? i % p.rows : i / p.kp, j = R_FAST != 0 ? i / p.rows : i % p.kp;
        uint u = j < p.k ? bit_cast<uint>(p.x[int(b) * p.sb + int(r) * p.sr + int(j) * p.sk]) : 0u;
        bool finite = (u & 0x7f800000u) != 0x7f800000u;
        uint hi = bf16_rne(u);
        if ((u & 0x7fffffffu) > 0x7f800000u)
            hi = 0x7fc0u;  // NaN
        else if (finite && (hi & 0x7f80u) == 0x7f80u)
            hi = u >> 16;  // (rounding would overflow: truncated)
        uint lo = finite ? bf16_rne(bit_cast<uint>(bit_cast<float>(u) - bit_cast<float>(hi << 16))) : 0u;
        int base = int(b) * p.yb + int(r) * p.yr + int(j) * p.yk;
        for (uint q = 0; q < p.parts; q++)
        {
            uint kind = (p.kinds >> (2 * q)) & 3u;
            uint v = kind == 0 ? hi : kind == 1 ? lo : (finite ? hi : 0u);
            p.y[base + int(q * p.kp) * p.yk] = uint16_t(v);
        }
    }
}
