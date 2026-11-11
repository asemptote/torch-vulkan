// axis T: c64=float2 c128=double2
//
// One radix-r Stockham pass of a power-of-two FFT over contiguous complex
// rows [batch, n]: the whole FFT in one pass (r = n, ns = 1) for rows up to
// BUDGET long, else one of ceil(log2(n) / 7) passes with r up to 128
// (csrc/fft.cpp; fft.slang's FFT_LOCAL takes rows up to 2 BUDGET). For each
// group j < n/r (k = j % ns):
//
//     u[q] = in[j + q n/r] * w^(k q (n / (r ns))),  q < r
//     out[(j / ns) r ns + k + q ns] = sum_q' u[q'] W_r^(q' q)
//
// with w = exp(-+2 pi i / n) from `table`, which holds w^t for t < n/2 in
// float64 accuracy (computed on the host; GPU sin/cos are far too
// inaccurate), so w^(t + n/2) = -w^t. INVERSE conjugates every twiddle.
//
// A workgroup takes g consecutive groups (r g = BUDGET; with r = n, g rows):
// it multiplies by the twiddles while loading them into shared memory (group
// fastest, or q fastest for whole rows, so the loads are contiguous),
// gathers W_r^t (t < r) into shared memory, runs the
// r-point DFTs as radix-4 Stockham stages (one butterfly per thread, in
// place: each thread reads its four values, a barrier, then writes them),
// with a last radix-2 stage when log2(r) is odd, and stores the results
// (group fastest once ns >= g, else q fastest). Every size is a power of
// two, so indices take shifts and masks (integer division is slow on GPUs).
// Shared memory: 8 KiB of complex64 or complex128 values plus up to 8 KiB of
// roots, Vulkan's guaranteed 16 KiB.

#if defined(T_c64)
typedef float S;
#define BUDGET 1024
#else
typedef double S;
#define BUDGET 512
#endif
#define WG (BUDGET / 4)
#define RMAX BUDGET
#include "_complex.slang"

struct Params
{
    T* in;           // 0
    T* out;          // 8
    T* table;        // 16
    uint n;          // 24
    uint ns;         // 28
    uint batch;      // 32
    uint r;          // 36
    uint g;          // 40
    uint ngroups;    // 44
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int INVERSE = 0;

groupshared T buf[BUDGET];
groupshared T roots[RMAX];

// w^e for e < n, from the half table.
T twiddle(uint e)
{
    uint h = p.n / 2;
    T w = e < h ? p.table[min(e, h - 1)] : -p.table[min(e - h, h - 1)];
    return INVERSE != 0 ? cconj(w) : w;
}

// z times W_4 = exp(-+ i pi / 2): -i z forward, i z inverse.
T times_w4(T z)
{
    return INVERSE != 0 ? T(-z.y, z.x) : T(z.y, -z.x);
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    // every size is a power of two: indices by shifts and masks
    uint t = tid.x;
    uint r = p.r, g = p.g, nr = p.n / r;
    uint lr = firstbithigh(r), lg = firstbithigh(g), lnr = firstbithigh(nr), lns = firstbithigh(p.ns);
    uint ln = firstbithigh(p.n);
    uint groups = p.batch << lnr, blocks = (groups + g - 1) >> lg;
    uint lstep = ln - lr - lns;  // the outer twiddle's exponent unit, n / (r ns)
    for (uint i = t; i < r; i += WG)
        roots[i] = twiddle(i << lnr);  // W_r^i
    bool load_q_fast = nr < g, store_q_fast = p.ns < g;
    for (uint w = gid.x; w < blocks; w += p.ngroups)  // uniform per workgroup
    {
        [ForceUnroll]
        for (uint s = 0; s < BUDGET / WG; s++)
        {
            // group fastest while a row has g groups, else (r = n) q fastest
            uint e = t + s * WG;
            uint q = load_q_fast ? e & (r - 1) : e >> lg, c = load_q_fast ? e >> lr : e & (g - 1);
            uint G = min((w << lg) + c, groups - 1), j = G & (nr - 1);  // (a partial last block repeats the last group)
            T v = p.in[((G >> lnr) << ln) + j + (q << lnr)];
            if (q != 0)
                v = cmul(v, twiddle(((j & (p.ns - 1)) * q) << lstep));
            buf[(q << lg) + c] = v;  // column c, element q
        }
        GroupMemoryBarrierWithGroupSync();
        uint m = 1, lm = 0;
        for (; m * 4 <= r; m *= 4, lm += 2)
        {
            // radix 4: butterflies (i, c), i < r/4, one per thread (r g = BUDGET)
            uint quarter = r >> 2, lunit = lr - 2 - lm;
            uint c = t & (g - 1), i = t >> lg, kk = i & (m - 1);
            T x0 = buf[(i << lg) + c];
            T x1 = cmul(buf[((i + quarter) << lg) + c], roots[kk << lunit]);
            T x2 = cmul(buf[((i + 2 * quarter) << lg) + c], roots[(2 * kk) << lunit]);
            T x3 = cmul(buf[((i + 3 * quarter) << lg) + c], roots[(3 * kk) << lunit]);
            T a0 = x0 + x2, a1 = x0 - x2, b0 = x1 + x3, b1 = times_w4(x1 - x3);
            uint o = ((((i >> lm) << (lm + 2)) + kk) << lg) + c;
            GroupMemoryBarrierWithGroupSync();
            buf[o] = a0 + b0;
            buf[o + (m << lg)] = a1 + b1;
            buf[o + (2 * m << lg)] = a0 - b0;
            buf[o + (3 * m << lg)] = a1 - b1;
            GroupMemoryBarrierWithGroupSync();
        }
        if (m < r)
        {
            // a last radix-2 stage (m = r/2): pairs (i, c), i < m, two per thread
            T lo[2], hi[2];
            uint at[2];
            [ForceUnroll]
            for (uint s = 0; s < 2; s++)
            {
                uint pi = t + s * WG;
                uint c = pi & (g - 1), i = pi >> lg;
                T x0 = buf[(i << lg) + c];
                T x1 = cmul(buf[((i + m) << lg) + c], roots[i]);
                lo[s] = x0 + x1;
                hi[s] = x0 - x1;
                at[s] = (i << lg) + c;
            }
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint s = 0; s < 2; s++)
            {
                buf[at[s]] = lo[s];
                buf[at[s] + (m << lg)] = hi[s];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        [ForceUnroll]
        for (uint s = 0; s < BUDGET / WG; s++)
        {
            uint e = t + s * WG;
            uint q = store_q_fast ? e & (r - 1) : e >> lg, c = store_q_fast ? e >> lr : e & (g - 1);
            uint G = (w << lg) + c;
            if (G < groups)
            {
                uint j = G & (nr - 1);
                p.out[((G >> lnr) << ln) + ((j >> lns) << (lr + lns)) + (j & (p.ns - 1)) + (q << lns)] = buf[(q << lg) + c];
            }
        }
        GroupMemoryBarrierWithGroupSync();  // buf is reused by the next blocks
    }
}
