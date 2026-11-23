// axis V: f32=float f16=half bf16=uint16_t q8f32=float q8f16=half q8bf16=uint16_t q4f32=float q4f16=half q4bf16=uint16_t
// axis C: c1=uint c2=uint c4=uint c8=uint
//
// Matrix-vector products with vector loads, for GPUs whose load/store unit
// favours them (Arm Mali): one to eight vectors at once,
//
//     out[j][r] = alpha * sum_k W[r, k] x[j][k]  (+ beta * cin[j][r] when USE_CIN)
//
// for r < rows, j < ncol, with W's rows contiguous along k. Products are
// accumulated in float32 and rounded once to the output's type V (to nearest
// even). The label's prefix picks W's storage; V is x's, out's and cin's type:
//
//   f32, f16, bf16  W in V (mm, addmm, addmv, linear with one to eight columns)
//   q8<V>           int8 W with a scale per row, in V (_weight_int8pack_mm):
//                   sum_k float(q) x[k], times the row's scale
//   q4<V>           4-bit W, two per byte with the even k in the high nibble
//                   (_weight_int4pack_mm's input layout), a scale and a zero
//                   per row and group of k in V, sz[k / group][r][2]:
//                   W[r, k] = float(q - 8) * scale + zero, the sums taken per
//                   32 values of k as scale * sum (q - 8) x + zero * sum x
//
// and C the vectors' count rounded up to 1, 2, 4 or 8 (NC; the host passes
// ncol, and columns past it are read again and not stored).
//
// Threads: LANES per row (a specialisation constant) walk k in 16-byte chunks of W, consecutive lanes on consecutive
// chunks, so a warp's loads are contiguous. Each thread keeps RB rows, 64 /
// LANES rows apart; a 64-thread workgroup covers 64 / LANES * RB rows. A
// chunk is taken in steps of STEP values of k: the vectors' STEP values are
// loaded once per step and used for all RB rows, so loads of x are 1 / RB
// of W's. (On Mali every load, x's from cache too, goes through the same
// load/store unit, and registers are few: rows are taken one at a time
// within a step, so registers hold RB * NC sums, NC * STEP values of x and
// one row's STEP values of W.) The LANES partial sums of each output are
// added in shared memory (Mali's subgroups are not used).
//
// The host passes vectors that are contiguous along k, start 16-byte aligned
// and are zero-padded to a whole step. W's steps are vector loads where
// W_ALIGNED (rows start 16-byte aligned); otherwise, and for a row's last
// partial chunk, W's elements are loaded one at a time (past k: zeros).
// Grid: row blocks, block gid.y * 65535 + gid.x (more than 65,535 blocks,
// Vulkan's guaranteed grid limit, folded into y; blocks past the last load
// clamped rows and store nothing).

#include "_half.slang"
#include "_slot.slang"

#if defined(V_f32) || defined(V_q8f32) || defined(V_q4f32)
#define X32 1  // x and out are float32
#elif defined(V_bf16) || defined(V_q8bf16) || defined(V_q4bf16)
#define XBF 1  // bfloat16 (as uint16_t bits)
#else
#define XF16 1  // float16
#endif
#if defined(V_q8f32) || defined(V_q8f16) || defined(V_q8bf16)
#define WQ8 1
#define WBITS 8
#elif defined(V_q4f32) || defined(V_q4f16) || defined(V_q4bf16)
#define WQ4 1
#define WBITS 4
#elif defined(X32)
#define WBITS 32
#else
#define WBITS 16
#endif

#if defined(C_c1)
#define NC 1
#define RB 4
#elif defined(C_c2)
#define NC 2
#define RB 4
#elif defined(C_c4)
#define NC 4
#define RB 2
#else
#define NC 8
#define RB 2
#endif

// Values of k per 16-byte chunk of W, per step, and W's bytes per step.
#define CHUNK (128 / WBITS)
#if WBITS == 32 || NC >= 4
#define STEP 4
#else
#define STEP 8
#endif
#define STEPS (CHUNK / STEP)
#define SBYTES (STEP * WBITS / 8)

struct Params
{
    uint* w;         // 0   W
    V* x;            // 8   x[j][k] at j * x_sc + k
    V* out;          // 16  out[j][r] at j * o_sc + r * o_sr
    V* cin;          // 24
    V* scales;       // 32  q8: [rows]; q4: [k / group][rows][2]
    slot64 alpha;    // 40  (float bits)
    slot64 beta;     // 48
    uint rows;       // 56
    uint k;          // 60
    uint w_sr;       // 64  W's row stride in bytes
    int x_sc;        // 68
    int o_sr, o_sc;  // 72
    int c_sr, c_sc;  // 80
    uint group;      // 88  q4: k per scale and zero
    uint ncol;       // 92
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int USE_CIN = 0;
[vk::constant_id(1)] const int LANES = 16;  // threads per row: 1, 2, 4, ..., 64
[vk::constant_id(2)] const int W_ALIGNED = 1;

#define WG 64

groupshared float part[RB * NC * WG];

float ld(V v)
{
#if defined(XBF)
    return asfloat(uint(v) << 16);
#else
    return float(v);
#endif
}

V st(float f)
{
#if defined(XF16)
    return to_half(f);
#elif defined(XBF)
    if (isnan(f))
        return uint16_t(0x7fc0);
    uint u = asuint(f);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return uint16_t(u >> 16);
#else
    return f;
#endif
}

// The two 16-bit values of a word (float16 or bfloat16 bits), as floats.
float lo16(uint u)
{
#if defined(XBF)
    return asfloat(u << 16);
#else
    return f16tof32(u & 0xffffu);
#endif
}
float hi16(uint u)
{
#if defined(XBF)
    return asfloat(u & 0xffff0000u);
#else
    return f16tof32(u >> 16);
#endif
}

// x[j][k0 .. k0 + STEP) as floats (k0 a multiple of STEP).
void load_x(uint j, uint k0, out float xs[STEP])
{
    uint e = uint(int(j) * p.x_sc) + k0;  // (elements; a multiple of STEP)
#if defined(X32)
    [ForceUnroll]
    for (uint q = 0; q < STEP / 4; q++)
    {
        float4 v = loadAligned<16>((float4*)p.x + (e / 4 + q));
        [ForceUnroll]
        for (uint i = 0; i < 4; i++)
            xs[4 * q + i] = v[i];
    }
#elif STEP == 8
    uint4 v = loadAligned<16>((uint4*)p.x + e / 8);
    [ForceUnroll]
    for (uint i = 0; i < 4; i++)
    {
        xs[2 * i] = lo16(v[i]);
        xs[2 * i + 1] = hi16(v[i]);
    }
#else
    uint2 v = loadAligned<8>((uint2*)p.x + e / 4);
    [ForceUnroll]
    for (uint i = 0; i < 2; i++)
    {
        xs[2 * i] = lo16(v[i]);
        xs[2 * i + 1] = hi16(v[i]);
    }
#endif
}

// W[r, k0 .. k0 + STEP) as SBYTES raw bytes (k0 a multiple of STEP); past
// k, zeros.
uint4 load_w(uint r, uint k0)
{
    uint base = r * p.w_sr;  // (bytes: element-aligned, and 16-byte aligned where W_ALIGNED)
    if (W_ALIGNED != 0 && k0 + STEP <= p.k)
    {
        uint at = base + k0 * WBITS / 8;  // (a multiple of SBYTES)
#if SBYTES == 16
        return loadAligned<16>((uint4*)p.w + at / 16);
#elif SBYTES == 8
        return uint4(loadAligned<8>((uint2*)p.w + at / 8), 0u, 0u);
#elif SBYTES == 4
        return uint4(p.w[at / 4], 0u, 0u, 0u);
#else
        return uint4(uint(((uint16_t*)p.w)[at / 2]), 0u, 0u, 0u);
#endif
    }
    uint4 v = uint4(0);
#if defined(WQ4)
    // (never unaligned: rows are 16-byte aligned and k a multiple of 32)
#elif defined(WQ8)
    int8_t* w8 = (int8_t*)p.w + base;
    [ForceUnroll]
    for (uint i = 0; i < STEP; i++)
    {
        uint kk = k0 + i;
        uint b = kk < p.k ? uint(w8[min(kk, p.k - 1)]) & 0xffu : 0u;
        v[i / 4] |= b << (8 * (i % 4));
    }
#elif defined(X32)
    uint* w32 = p.w + base / 4;
    [ForceUnroll]
    for (uint i = 0; i < STEP; i++)
    {
        uint kk = k0 + i;
        v[i] = kk < p.k ? w32[min(kk, p.k - 1)] : 0u;
    }
#else
    uint16_t* w16 = (uint16_t*)p.w + base / 2;
    [ForceUnroll]
    for (uint i = 0; i < STEP; i++)
    {
        uint kk = k0 + i;
        uint h = kk < p.k ? uint(w16[min(kk, p.k - 1)]) : 0u;
        v[i / 2] |= h << (16 * (i % 2));
    }
#endif
    return v;
}

// A step of W as floats: W itself, q for int8, q - 8 for 4 bits (their
// scales are applied to the sums). Integers become floats exactly through
// the exponent of 2^23 (no integer-to-float conversion): asfloat(0x4B000000
// | u) is 2^23 + u for u < 2^23.
void decode(uint4 v, out float ws[STEP])
{
#if defined(WQ4)
    [ForceUnroll]
    for (uint e = 0; e < STEP; e++)
    {
        uint shift = 8 * (e / 2) + ((e & 1u) != 0 ? 0u : 4u);  // (even k: the high nibble)
        ws[e] = asfloat(0x4B000000u | ((v.x >> shift) & 15u)) - 8388616.0;  // q - 8
    }
#elif defined(WQ8)
    [ForceUnroll]
    for (uint e = 0; e < STEP; e++)  // (b ^ 0x80 = q + 128)
        ws[e] = asfloat(((v[e / 4] >> (8 * (e % 4))) & 0xffu) ^ 0x4B000080u) - 8388736.0;
#elif defined(X32)
    [ForceUnroll]
    for (uint e = 0; e < 4; e++)
        ws[e] = asfloat(v[e]);
#else
    [ForceUnroll]
    for (uint e = 0; e < STEP / 2; e++)
    {
        ws[2 * e] = lo16(v[e]);
        ws[2 * e + 1] = hi16(v[e]);
    }
#endif
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint t = tid.x;
    uint groups = WG / uint(LANES);  // row groups in the workgroup
    uint lane = t % uint(LANES), g = t / uint(LANES);
    uint row0 = (gid.y * 65535u + gid.x) * groups * RB;
    uint rr[RB];
    [ForceUnroll]
    for (uint i = 0; i < RB; i++)
        rr[i] = min(row0 + i * groups + g, p.rows - 1);
    float acc[RB][NC];
    [ForceUnroll]
    for (uint i = 0; i < RB; i++)
        [ForceUnroll]
        for (uint j = 0; j < NC; j++)
            acc[i][j] = 0.0;

    uint chunks = (p.k + CHUNK - 1) / CHUNK;
    for (uint c = lane; c < chunks; c += uint(LANES))
    {
#if defined(WQ4)
        // A chunk (32 values of k) lies in one group: its sums of x (q - 8)
        // and of x, times the row's scale and zero, then added to acc.
        float dot[RB][NC], xsum[NC];
        [ForceUnroll]
        for (uint j = 0; j < NC; j++)
        {
            xsum[j] = 0.0;
            [ForceUnroll]
            for (uint i = 0; i < RB; i++)
                dot[i][j] = 0.0;
        }
#define SUM dot
#else
#define SUM acc
#endif
        [ForceUnroll]
        for (uint s = 0; s < STEPS; s++)
        {
            uint k0 = c * CHUNK + s * STEP;
            float xs[NC][STEP];
            [ForceUnroll]
            for (uint j = 0; j < NC; j++)
            {
                load_x(min(j, p.ncol - 1), k0, xs[j]);
#if defined(WQ4)
                [ForceUnroll]
                for (uint e = 0; e < STEP; e++)
                    xsum[j] += xs[j][e];
#endif
            }
            [ForceUnroll]
            for (uint i = 0; i < RB; i++)
            {
                float ws[STEP];
                decode(load_w(rr[i], k0), ws);
                [ForceUnroll]
                for (uint j = 0; j < NC; j++)
                {
                    float a = SUM[i][j];
                    [ForceUnroll]
                    for (uint e = 0; e < STEP; e++)
                        a += ws[e] * xs[j][e];
                    SUM[i][j] = a;
                }
            }
        }
#if defined(WQ4)
        [ForceUnroll]
        for (uint i = 0; i < RB; i++)
        {
            uint at = ((c * CHUNK) / p.group * p.rows + rr[i]) * 2;
            float scale = ld(p.scales[at]), zero = ld(p.scales[at + 1]);
            [ForceUnroll]
            for (uint j = 0; j < NC; j++)
                acc[i][j] += scale * dot[i][j] + zero * xsum[j];
        }
#endif
#undef SUM
    }

    // The LANES partial sums of each (row, vector) added in shared memory.
    [ForceUnroll]
    for (uint i = 0; i < RB; i++)
        [ForceUnroll]
        for (uint j = 0; j < NC; j++)
            part[(i * NC + j) * WG + t] = acc[i][j];
    GroupMemoryBarrierWithGroupSync();
    float alpha = slot_float(p.alpha), beta = slot_float(p.beta);
    for (uint o = t; o < groups * RB * NC; o += WG)
    {
        uint og = o % groups, oij = o / groups;  // (row group, then i * NC + j)
        uint i = oij / NC, j = oij % NC;
        float sum = 0.0;
        for (uint l = 0; l < uint(LANES); l++)
            sum += part[oij * WG + og * uint(LANES) + l];
        uint r = row0 + i * groups + og;
        if (r < p.rows && j < p.ncol)
        {
#if defined(WQ8)
            sum *= ld(p.scales[r]);
#endif
            float v = alpha * sum;
            if (USE_CIN != 0)  // beta == 0 must ignore cin entirely (it may hold NaN)
                v += beta * ld(p.cin[int(j) * p.c_sc + int(r) * p.c_sr]);
            p.out[int(j) * p.o_sc + int(r) * p.o_sr] = st(v);
        }
    }
}
