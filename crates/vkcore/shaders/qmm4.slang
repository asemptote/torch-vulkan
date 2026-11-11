// axis T: f32=float bf16=uint16_t
//
// aten::_dyn_quant_matmul_4bit as the CPU's reference kernels compute it
// (ATen/native/cpu/int4mm_kernel.cpp, ref_dyn_quant_matmul_4bit_*), bit for
// bit: the input's rows quantised to int8 (asymmetric, per row), integer
// products with the int4 weights, scaled per block of k in float32.
//
// The packed weights are _dyn_quant_pack_4bit_weight's float32 vector (the
// CPU's layout, so either device's packing works with either's matmul):
// N rows of ceil(K / 2) bytes of 4-bit weights (low nibble first, value
// nibble - 8) as floats, then the float32 scales (N x K / bs, or N when
// channelwise: bs = K), then optionally N biases.
//
// PHASE 0, per input row m (one workgroup): min and max (with 0), then on
//   one thread the scale 255 / (rmax - rmin) and its reciprocal (correctly
//   rounded divisions: div_exact), the zero point (rounded to nearest even,
//   clamped), and the quantised row q = clamp(round(v * scale) + zp)
//   (round: halfway away from zero, as std::round) into q (int32 [M, K]);
//   rows[2m] = 1 / scale (float bits), rows[2m + 1] = -zp.
// PHASE 1, out[m, n]: per block of bs values of k (in order), the integer
//   sum of (q + (-zp)) * w, converted and multiplied by the block's scale
//   and added up in float32; then times 1 / scale, plus the bias, clamped to
//   the output type's finite range, stored (bfloat16 by truncation, as the
//   CPU's bf16 kernel does). A 256-thread workgroup computes an 8 x 32 tile,
//   one output per thread, staging 32 values of k of q and the weights.
//
// GROUPWISE (spec constant 1): the groupwise kernel adds the zero point's
// error terms left to right ((qmin + dmin) + qmax) + dmax, the channelwise
// ones as (qmin + dmin) + (qmax + dmax); the sign decides the zero point.

#define FMATH_F32 1
#include "_fmath.slang"

struct Params
{
    T* inp;          // 0  contiguous [M, K]
    int* q;          // 8  [M, K]
    uint* rows;      // 16 [M, 2]: 1 / scale (float bits), -zp (int bits)
    float* packed;   // 24
    T* out;          // 32 contiguous [M, N]
    uint m;          // 40
    uint n;          // 44
    uint k;          // 48
    uint bs;         // 52
    uint has_bias;   // 56
    uint _pad;       // 60
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int PHASE = 0;
[vk::constant_id(1)] const int GROUPWISE = 0;

float ld(T x)
{
#if defined(T_bf16)
    return bit_cast<float>(uint(x) << 16);
#else
    return x;
#endif
}

// a * b exactly as p + e (Dekker's product with Veltkamp's split: plain
// multiplications and additions, which this kernel's precise mode keeps
// unfused); |a b| below 2^100, so that the split cannot overflow.
void two_prod(float a, float b, out float p, out float e)
{
    p = a * b;
    float t = 4097.0 * a, ah = t - (t - a), al = a - ah;
    t = 4097.0 * b;
    float bh = t - (t - b), bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}

// a - q b exactly, for q within a few ulps of a / b.
float residual(float a, float q, float b)
{
    float pr, e;
    two_prod(q, b, pr, e);
    return (a - pr) - e;
}

// a / b correctly rounded (to nearest even), as the CPU divides: Vulkan
// allows OpFDiv 2.5 ulp, and _fmath.slang's m_div corrects with an FMA,
// which not every GPU fuses (the UHD 620's driver does not: a third of its
// float32 quotients came out 1 ulp off). The quotient's neighbours are
// compared by their exact residuals instead. (Positive finite a and b here,
// quotients well inside float32's range.)
float div_exact(float a, float b)
{
    float q = a / b;
    if (isnan(q) || isinf(q) || q == 0.0 || q > 1e30 || q < 1e-30)
        return q;
    q += residual(a, q, b) / b;
    uint bits = bit_cast<uint>(q);
    float best = q, best_r = abs(residual(a, q, b));
    [ForceUnroll]
    for (int d = -1; d <= 1; d += 2)
    {
        float c = bit_cast<float>(uint(int(bits) + d));
        float r = abs(residual(a, c, b));
        if (r < best_r || (r == best_r && (bit_cast<uint>(c) & 1u) == 0u))
        {
            best = c;
            best_r = r;
        }
    }
    return best;
}

// std::round: halfway cases away from zero (x - trunc(x) is exact).
float round_away(float x)
{
    float t = trunc(x);
    if (abs(x - t) >= 0.5)
        t += x < 0.0 ? -1.0 : 1.0;
    return t;
}

groupshared float red_min[256];
groupshared float red_max[256];
groupshared float row_scale;
groupshared int row_zp;
groupshared int Qs[8][32];
groupshared int Ws[32][33];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint t = tid.x;
    if (PHASE == 0)
    {
        uint m = gid.x;  // one workgroup per row (csrc/quant.cpp: M <= 65535)
        uint base = m * p.k;
        float mn = 3.40282347e38, mx = -3.40282347e38;
        for (uint k = t; k < p.k; k += 256)
        {
            float v = ld(p.inp[base + k]);
            mn = min(mn, v);
            mx = max(mx, v);
        }
        red_min[t] = mn;
        red_max[t] = mx;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = 128; s > 0; s /= 2)
        {
            if (t < s)
            {
                red_min[t] = min(red_min[t], red_min[t + s]);
                red_max[t] = max(red_max[t], red_max[t + s]);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (t == 0)
        {
            const float qmin = -128.0, qmax = 127.0;
            float rmin = min(0.0, red_min[0]), rmax = max(0.0, red_max[0]);
            float scale = rmin == rmax ? 1.0 : div_exact(qmax - qmin, rmax - rmin);
            float recip = scale != 0.0 ? div_exact(1.0, scale) : 0.0;
            float dmin = rmin * scale, dmax = rmax * scale;
            float err = GROUPWISE != 0 ? ((qmin + dmin) + qmax) + dmax : (qmin + dmin) + (qmax + dmax);
            float zp = err > 0.0 ? qmin - dmin : qmax - dmax;
            zp = min(max(zp, qmin), qmax);
            int izp = int(m_round_even(zp));
            row_scale = scale;
            row_zp = izp;
            // (as bits: an int read as a float may be a denormal, which a GPU may flush)
            p.rows[2 * m] = bit_cast<uint>(recip);
            p.rows[2 * m + 1] = uint(-izp);
        }
        GroupMemoryBarrierWithGroupSync();
        float scale = row_scale;
        int zp = row_zp;
        for (uint k = t; k < p.k; k += 256)
        {
            int v = int(round_away(ld(p.inp[base + k]) * scale)) + zp;
            p.q[base + k] = clamp(v, -128, 127);
        }
        return;
    }

    // PHASE 1
    uint mi = t / 32, ni = t % 32;
    uint m0 = gid.y * 8, n0 = gid.x * 32;
    uint m = m0 + mi, n = n0 + ni;
    uint rstride = (p.k + 1) / 2;                  // bytes per weight row
    uint wcount = p.n * p.k / 2;                   // (the CPU's weights_elements)
    uint nblocks = (p.k + p.bs - 1) / p.bs;
    int off = int(p.rows[2 * min(m, p.m - 1) + 1]);
    float main_acc = 0.0;
    bool first = true;
    int iacc = 0;
    for (uint k0 = 0; k0 < p.k; k0 += 32)
    {
        {   // q: one value per thread
            uint r = m0 + mi, k = k0 + ni;
            int v = p.q[min(r, p.m - 1) * p.k + min(k, p.k - 1)];
            Qs[mi][ni] = r < p.m && k < p.k ? v : 0;
        }
        {   // weights: 32 rows x 16 bytes, two bytes (four nibbles) per thread
            uint wr = t / 8, j = t % 8, row = min(n0 + wr, p.n - 1);
            for (uint h = 0; h < 2; h++)
            {
                uint kk = 4 * j + 2 * h, byte = (k0 + kk) / 2;
                uint b = uint(p.packed[row * rstride + min(byte, rstride - 1)]);
                Ws[wr][kk] = int(b & 15u) - 8;
                Ws[wr][kk + 1] = int(b >> 4) - 8;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        uint kend = min(32u, p.k - k0);
        for (uint kk = 0; kk < kend; kk++)
        {
            int w = Ws[ni][kk];
            iacc += Qs[mi][kk] * w;
            iacc += off * w;
        }
        GroupMemoryBarrierWithGroupSync();
        if ((k0 + 32) % p.bs == 0 || k0 + 32 >= p.k)
        {
            // the block's scale: groupwise [n][block], channelwise [n]
            uint blk = k0 / p.bs;
            float s = p.packed[wcount + min(n, p.n - 1) * nblocks + blk];
            float x = float(iacc) * s;
            main_acc = first ? x : main_acc + x;  // (the channelwise kernel assigns: keeps -0)
            first = false;
            iacc = 0;
        }
    }
    if (m >= p.m || n >= p.n)
        return;
    main_acc = main_acc * bit_cast<float>(p.rows[2 * m]);
    if (p.has_bias != 0)
        main_acc += p.packed[wcount + p.n * nblocks + n];
#if defined(T_bf16)
    const float hi = 3.38953139e38;
#else
    const float hi = 3.40282347e38;
#endif
    main_acc = main_acc < -hi ? -hi : main_acc;  // std::max(x, lo), std::min(x, hi): NaN passes
    main_acc = hi < main_acc ? hi : main_acc;
#if defined(T_bf16)
    p.out[m * p.n + n] = uint16_t(bit_cast<uint>(main_acc) >> 16);  // truncation, as the CPU's kernel
#else
    p.out[m * p.n + n] = main_acc;
#endif
}
