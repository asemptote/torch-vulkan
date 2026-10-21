// axis T: c64=float2 c128=double2
//
// FFT building blocks over contiguous complex rows [batch, n]:
//
// PHASE 0, one radix-2 Stockham pass (n a power of two; the host runs
// log2(n) passes with ns = 1, 2, 4, ..., n/2, ping-ponging buffers; the
// result comes out in natural order, no bit reversal):
//     for each butterfly j < n/2:  k = j % ns,  w = table[k * n / (2 ns)]
//         v0 = in[j], v1 = in[j + n/2] * w
//         out[(j / ns) * 2 ns + k] = v0 + v1,  out[... + ns] = v0 - v1
//   `table` holds exp(-2 pi i t / n) for t < n/2, computed on the host in
//   float64 (GPU sin/cos are far too inaccurate for twiddles); INVERSE
//   conjugates it.
//
// PHASE 1, Hermitian extension for C2R: out[k] = in[k] for k <= n/2, else
//   conj(in[n - k]); input rows have `m` elements.
//
// PHASE 2, a whole power-of-two FFT of rows with n <= LOCAL_MAX in shared
//   memory: one workgroup per row runs every Stockham pass between two
//   shared buffers, so the row is read and written once instead of log2(n)
//   times.

#include "opcodes.slang"

#if defined(T_c64)
typedef float S;
#else
typedef double S;
#endif
#include "_complex.slang"

struct Params
{
    T* in;           // 0
    T* out;          // 8
    T* table;        // 16
    uint n;          // 24
    uint ns;         // 28
    uint batch;      // 32
    uint m;          // 36  input row length (PHASE 1)
    uint ngroups;    // 40
    uint _pad;       // 44
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int PHASE = 0;
[vk::constant_id(1)] const int INVERSE = 0;

// Two shared buffers of LOCAL_MAX elements: 32 KiB, the guaranteed minimum.
#if defined(T_c64)
static const uint LOCAL_MAX = 2048;
#else
static const uint LOCAL_MAX = 1024;
#endif
groupshared T sbuf[2][LOCAL_MAX];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint stride = p.ngroups * 256;
    if (PHASE == FFT_STOCKHAM)
    {
        uint half_n = p.n / 2;
        uint total = p.batch * half_n;
        for (uint t = gid.x * 256 + tid.x; t < total; t += stride)
        {
            uint b = t / half_n, j = t % half_n;
            uint base = b * p.n;
            uint k = j % p.ns;
            T w = p.table[k * (p.n / (2 * p.ns))];
            if (INVERSE != 0)
                w = cconj(w);
            T v0 = p.in[base + j];
            T v1 = cmul(p.in[base + j + half_n], w);
            uint idx = (j / p.ns) * 2 * p.ns + k;
            p.out[base + idx] = v0 + v1;
            p.out[base + idx + p.ns] = v0 - v1;
        }
    }
    else if (PHASE == FFT_LOCAL)
    {
        uint half_n = p.n / 2;
        for (uint row = gid.x; row < p.batch; row += p.ngroups)  // uniform per workgroup
        {
            uint base = row * p.n;
            for (uint i = tid.x; i < p.n; i += 256)
                sbuf[0][i] = p.in[base + i];
            GroupMemoryBarrierWithGroupSync();
            uint cur = 0;
            for (uint ns = 1; ns < p.n; ns *= 2)
            {
                for (uint j = tid.x; j < half_n; j += 256)
                {
                    uint k = j % ns;
                    T w = p.table[k * (p.n / (2 * ns))];
                    if (INVERSE != 0)
                        w = cconj(w);
                    T v0 = sbuf[cur][j];
                    T v1 = cmul(sbuf[cur][j + half_n], w);
                    uint idx = (j / ns) * 2 * ns + k;
                    sbuf[1 - cur][idx] = v0 + v1;
                    sbuf[1 - cur][idx + ns] = v0 - v1;
                }
                cur = 1 - cur;
                GroupMemoryBarrierWithGroupSync();
            }
            for (uint i = tid.x; i < p.n; i += 256)
                p.out[base + i] = sbuf[cur][i];
            GroupMemoryBarrierWithGroupSync();  // sbuf is reused by the next row
        }
    }
    else
    {
        uint total = p.batch * p.n;
        for (uint t = gid.x * 256 + tid.x; t < total; t += stride)
        {
            uint b = t / p.n, k = t % p.n;
            T v = k <= p.n / 2 ? p.in[b * p.m + k] : cconj(p.in[b * p.m + (p.n - k)]);
            p.out[t] = v;
        }
    }
}
