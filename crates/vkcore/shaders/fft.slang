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

// X[k] = (A + B) / 2 + w (A - B) / (2i), w = w^k (FFT_R2C).
T r2c_point(T a, T c, T w)
{
    T e = (a + c) * S(0.5), d = (a - c) * S(0.5);
    T o = T(d.y, -d.x);  // (A - B) / (2i)
    return e + cmul(w, o);
}

// Z'[k] = (A + B) + i cw (A - B), cw = conj(w^k), k < h (FFT_C2R).
T c2r_point(T a, T c, T cw)
{
    T d = cmul(cw, a - c);
    return (a + c) + T(-d.y, d.x);  // + i d
}

// w^k for k <= h/2 from the table (w^h = -1); the pair's other twiddle is
// w^(h - k) = w^h conj(w^k) = -conj(w^k), so each pair reads one entry and
// the passes read only the first half of the table.
T pair_twiddle(uint k, uint h)
{
    return k == h ? T(S(-1), S(0)) : p.table[min(k, max(h, 1u) - 1)];
}

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
        // One thread per pair (k, h - k), k <= h/2: Z[k] and Z[h - k] are
        // read once for both outputs (the pair's A and B swap and conjugate:
        // for X[h - k], A = Z[h - k] and B = conj(Z[k])).
        uint h = p.n / 2, pairs = h / 2 + 1, total = p.batch * pairs;
        for (uint t = gid.x * 256 + tid.x; t < total; t += stride)
        {
            uint b = t / pairs, k = t - b * pairs, km = k == 0 ? 0 : h - k;
            T zk = p.in[b * h + k], zm = p.in[b * h + km];
            T w = pair_twiddle(k, h);
            uint row = b * (h + 1);
            p.out[row + k] = r2c_point(zk, cconj(zm), w);
            if (h - k != k)
                p.out[row + h - k] = r2c_point(zm, cconj(zk), -cconj(w));
        }
    }
    else if (PHASE == FFT_C2R)
    {
        // One thread per pair (k, h - k), k <= h/2, as FFT_R2C; k = 0 has the
        // one output Z'[0] (from X[0] and X[h]).
        uint h = p.n / 2, pairs = h / 2 + 1, total = p.batch * pairs;
        for (uint t = gid.x * 256 + tid.x; t < total; t += stride)
        {
            uint b = t / pairs, k = t - b * pairs;
            if (k >= h)
                continue;  // (h = 1: the one pair is k = 0)
            T xk = p.in[b * p.m + k], xm = p.in[b * p.m + (h - k)];
            if (k == 0)
            {
                // (X[0] and X[h] count as real, as the full-length inverse's
                // real part has them: their imaginary parts are dropped)
                xk.y = S(0);
                xm.y = S(0);
            }
            T w = pair_twiddle(k, h);
            p.out[b * h + k] = c2r_point(xk, cconj(xm), cconj(w));
            if (k != 0 && h - k != k)
                p.out[b * h + h - k] = c2r_point(xm, cconj(xk), -w);  // (conj(w^(h - k)) = -w^k)
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
