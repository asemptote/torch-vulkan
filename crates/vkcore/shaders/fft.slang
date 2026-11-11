// axis T: c64=float2 c128=double2
//
// FFT building blocks over contiguous complex rows [batch, n]:
//
// `table` holds exp(-2 pi i t / n) for t < n/2, computed on the host in
// float64 (GPU sin/cos are far too inaccurate for twiddles); INVERSE
// conjugates it. Power-of-two rows up to LOCAL_MAX / 2 and longer than
// LOCAL_MAX take fft_radix.slang's radix-r passes (which replaced this
// file's radix-2 pass, opcode 0, now free).
//
// PHASE 1, Hermitian extension for C2R: out[k] = in[k] for k <= n/2, else
//   conj(in[n - k]); input rows have `m` elements.
//
// PHASE 2, a whole power-of-two FFT of rows with LOCAL_MAX / 2 < n <=
//   LOCAL_MAX in shared memory: one workgroup per row runs every pass between two
//   shared buffers, so the row is read and written once instead of log2(n)
//   times.
//
// PHASE 3 and 4, real transforms of even length n at half length (h = n/2):
//   a real row x read as complex z[j] = x[2j] + i x[2j + 1] has
//   FFT Z = E + i O (E, O the FFTs of x's even and odd samples), and
//   X[k] = E[k] + w^k O[k] with w = exp(-2 pi i / n) (table: n's).
//   R2C, after Z = FFT_h(z) (rows of h; out rows of h + 1), k <= h:
//       A = Z[k mod h], B = conj(Z[(h - k) mod h])
//       X[k] = (A + B) / 2 + w^k (A - B) / (2i)
//   C2R, before an unnormalized inverse FFT_h whose output, read as reals, is
//   n x (in rows of `m` >= h + 1 elements; out rows of h), k < h:
//       A = X[k], B = conj(X[h - k])  (only the real parts of X[0] and X[h])
//       Z'[k] = (A + B) + i conj(w^k) (A - B)

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
    if (PHASE == FFT_LOCAL)
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
    else if (PHASE == FFT_R2C)
    {
        uint h = p.n / 2, total = p.batch * (h + 1);
        for (uint t = gid.x * 256 + tid.x; t < total; t += stride)
        {
            uint b = t / (h + 1), k = t % (h + 1);
            T a = p.in[b * h + (k == h ? 0 : k)];
            T c = cconj(p.in[b * h + (k == 0 ? 0 : h - k)]);
            T w = k == h ? T(S(-1), S(0)) : p.table[min(k, h - 1)];
            T e = (a + c) * S(0.5), d = (a - c) * S(0.5);
            T o = T(d.y, -d.x);  // (A - B) / (2i)
            p.out[t] = e + cmul(w, o);
        }
    }
    else if (PHASE == FFT_C2R)
    {
        uint h = p.n / 2, total = p.batch * h;
        for (uint t = gid.x * 256 + tid.x; t < total; t += stride)
        {
            uint b = t / h, k = t % h;
            T a = p.in[b * p.m + k];
            T c = cconj(p.in[b * p.m + (h - k)]);
            if (k == 0)
            {
                // (X[0] and X[h] count as real, as the full-length inverse's
                // real part has them: their imaginary parts are dropped)
                a.y = S(0);
                c.y = S(0);
            }
            T d = cmul(cconj(p.table[min(k, max(h, 1u) - 1)]), a - c);
            p.out[t] = (a + c) + T(-d.y, d.x);  // + i d
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
