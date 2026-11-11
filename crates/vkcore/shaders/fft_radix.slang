// axis T: c64=float2 c128=double2
//
// One radix-r Stockham pass of a power-of-two FFT over contiguous complex
// rows [batch, n], for rows longer than fft.slang's LOCAL_MAX (csrc/fft.cpp
// runs ceil(log2(n) / 7) of them, against log2(n) radix-2 passes). r is a
// power of two up to 128. For each group j < n/r (k = j % ns):
//
//     u[q] = in[j + q n/r] * w^(k q (n / (r ns))),  q < r
//     out[(j / ns) r ns + k + q ns] = sum_q' u[q'] W_r^(q' q)
//
// with w = exp(-+2 pi i / n) from `table`, which holds w^t for t < n/2 in
// float64 accuracy (computed on the host; GPU sin/cos are far too
// inaccurate), so w^(t + n/2) = -w^t. INVERSE conjugates every twiddle.
//
// A workgroup takes g consecutive groups (r g = BUDGET): it multiplies by
// the twiddles while loading them into shared memory (group fastest, so the
// loads are contiguous), gathers W_r^t (t < r) into shared memory, runs the
// r-point DFTs as radix-4 Stockham stages (one butterfly per thread, in
// place: each thread reads its four values, a barrier, then writes them),
// with a last radix-2 stage when log2(r) is odd, and stores the results
// (group fastest once ns >= g, else q fastest). Shared memory: 8 KiB of
// complex64 or complex128 values, plus the roots, within Vulkan's
// guaranteed 16 KiB, so several workgroups fit per core.

#if defined(T_c64)
typedef float S;
#define BUDGET 1024
#else
typedef double S;
#define BUDGET 512
#endif
#define WG (BUDGET / 4)
#define RMAX 128
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
    uint t = tid.x;
    uint r = p.r, g = p.g, nr = p.n / r;
    uint per_row = nr / g, total = p.batch * per_row;
    uint step = p.n / (r * p.ns);  // the outer twiddle's exponent unit
    for (uint i = t; i < r; i += WG)
        roots[i] = twiddle(i * nr);  // W_r^i
    for (uint w = gid.x; w < total; w += p.ngroups)  // uniform per workgroup
    {
        uint base = (w / per_row) * p.n, j0 = (w % per_row) * g;
        for (uint e = t; e < r * g; e += WG)
        {
            uint q = e / g, c = e % g, j = j0 + c;
            T v = p.in[base + j + q * nr];
            if (q != 0)
                v = cmul(v, twiddle((j % p.ns) * q * step));
            buf[e] = v;  // column c, element q at q g + c
        }
        GroupMemoryBarrierWithGroupSync();
        uint m = 1;
        for (; m * 4 <= r; m *= 4)
        {
            // radix 4: butterflies (i, c), i < r/4, one per thread when r g = BUDGET
            uint quarter = r / 4, unit = r / (4 * m);
            bool active = t < g * quarter;
            T y0, y1, y2, y3;
            uint o = 0;
            if (active)
            {
                uint c = t % g, i = t / g, kk = i % m;
                T x0 = buf[i * g + c];
                T x1 = cmul(buf[(i + quarter) * g + c], roots[kk * unit]);
                T x2 = cmul(buf[(i + 2 * quarter) * g + c], roots[2 * kk * unit]);
                T x3 = cmul(buf[(i + 3 * quarter) * g + c], roots[3 * kk * unit]);
                T a0 = x0 + x2, a1 = x0 - x2, b0 = x1 + x3, b1 = times_w4(x1 - x3);
                y0 = a0 + b0;
                y1 = a1 + b1;
                y2 = a0 - b0;
                y3 = a1 - b1;
                o = ((i / m) * 4 * m + kk) * g + c;
            }
            GroupMemoryBarrierWithGroupSync();
            if (active)
            {
                buf[o] = y0;
                buf[o + m * g] = y1;
                buf[o + 2 * m * g] = y2;
                buf[o + 3 * m * g] = y3;
            }
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
                if (pi < g * m)
                {
                    uint c = pi % g, i = pi / g;
                    T x0 = buf[i * g + c];
                    T x1 = cmul(buf[(i + m) * g + c], roots[i]);
                    lo[s] = x0 + x1;
                    hi[s] = x0 - x1;
                    at[s] = i * g + c;
                }
            }
            GroupMemoryBarrierWithGroupSync();
            [ForceUnroll]
            for (uint s = 0; s < 2; s++)
            {
                if (t + s * WG < g * m)
                {
                    buf[at[s]] = lo[s];
                    buf[at[s] + m * g] = hi[s];
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        for (uint e = t; e < r * g; e += WG)
        {
            uint q = p.ns >= g ? e / g : e % r, c = p.ns >= g ? e % g : e / r, j = j0 + c;
            p.out[base + (j / p.ns) * r * p.ns + (j % p.ns) + q * p.ns] = buf[q * g + c];
        }
        GroupMemoryBarrierWithGroupSync();  // buf is reused by the next groups
    }
}
