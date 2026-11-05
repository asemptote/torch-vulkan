// axis T: f32=float f64=double
//
// Layer norm over the last n elements of a contiguous tensor seen as [m, n]:
// one workgroup per row. OP:
//   LAYERNORM_FORWARD:  mean, rstd = 1 / sqrt(var + eps) (biased variance,
//                       two passes), y = (x - mean) * rstd * w + b
//   LAYERNORM_BACKWARD: gx = rstd * g w + bc * x + cc, with ds = sum(g w x),
//                       db = sum(g w), bc = (db mean - ds) rstd^3 / n and
//                       cc = -bc mean - db rstd / n (the CPU's formula)
//   LAYERNORM_RMS_FORWARD:  RMS norm: rstd = 1 / sqrt(mean(x^2) + eps),
//                       y = x * rstd * w (mean not written)
//   LAYERNORM_RMS_BACKWARD: gx = rstd * g w - x rstd^3 sum(g w x) / n
// w and b are optional (HAS_W, HAS_B; absent: 1 and 0).

#include "_fmath.slang"
#include "_slot.slang"
#include "opcodes.slang"

struct Params
{
    T* x;            // 0
    T* g;            // 8   grad_output (backward)
    T* w;            // 16
    T* b;            // 24
    T* out;          // 32  y (forward) or gx (backward)
    T* mean;         // 40  [m]: written (forward) or read (backward)
    T* rstd;         // 48
    uint m;          // 56
    uint n;          // 60
    uint ngroups;    // 64
    uint _pad;       // 68
    slot64 eps;      // 72  raw slot (_slot.slang): T bits (float in the low word)
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = LAYERNORM_FORWARD;
[vk::constant_id(1)] const int HAS_W = 0;
[vk::constant_id(2)] const int HAS_B = 0;

static const uint G = 256;
groupshared T shared_[G];

T param(slot64 bits)
{
#if defined(T_f64)
    return slot_double(bits);
#else
    return slot_float(bits);
#endif
}

T reduce_sum(T v, uint tid)
{
    shared_[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            shared_[tid] += shared_[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    T r = shared_[0];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

T weight(uint j) { return HAS_W != 0 ? p.w[j] : T(1); }

[shader("compute")]
[numthreads(G, 1, 1)]
void main(uint3 tid3: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint tid = tid3.x;
    T n = T(p.n);
    for (uint row = gid.x; row < p.m; row += p.ngroups)
    {
        uint base = row * p.n;
        if (OP == LAYERNORM_RMS_FORWARD)
        {
            T q = T(0);
            for (uint j = tid; j < p.n; j += G)
            {
                T v = p.x[base + j];
                q += v * v;
            }
            T rstd = T(1) / sqrt(reduce_sum(q, tid) / n + param(p.eps));
            if (tid == 0)
                p.rstd[row] = rstd;
            for (uint j = tid; j < p.n; j += G)
                p.out[base + j] = p.x[base + j] * rstd * weight(j);
        }
        else if (OP == LAYERNORM_RMS_BACKWARD)
        {
            T rstd = p.rstd[row];
            T ds = T(0);
            for (uint j = tid; j < p.n; j += G)
                ds += p.g[base + j] * weight(j) * p.x[base + j];
            T bc = -reduce_sum(ds, tid) * rstd * rstd * rstd / n;
            for (uint j = tid; j < p.n; j += G)
                p.out[base + j] = rstd * p.g[base + j] * weight(j) + bc * p.x[base + j];
        }
        else if (OP == LAYERNORM_FORWARD)
        {
            T s = T(0);
            for (uint j = tid; j < p.n; j += G)
                s += p.x[base + j];
            T mean = reduce_sum(s, tid) / n;
            T q = T(0);
            for (uint j = tid; j < p.n; j += G)
            {
                T d = p.x[base + j] - mean;
                q += d * d;
            }
            T var = reduce_sum(q, tid) / n;
            T rstd = T(1) / sqrt(max(var, T(0)) + param(p.eps));
            if (tid == 0)
            {
                p.mean[row] = mean;
                p.rstd[row] = rstd;
            }
            for (uint j = tid; j < p.n; j += G)
            {
                T y = (p.x[base + j] - mean) * rstd * weight(j);
                p.out[base + j] = HAS_B != 0 ? y + p.b[j] : y;
            }
        }
        else
        {
            T mean = p.mean[row], rstd = p.rstd[row];
            T ds = T(0), db = T(0);
            for (uint j = tid; j < p.n; j += G)
            {
                T gw = p.g[base + j] * weight(j);
                ds += gw * p.x[base + j];
                db += gw;
            }
            ds = reduce_sum(ds, tid);
            db = reduce_sum(db, tid);
            T bc = (db * mean - ds) * rstd * rstd * rstd / n;
            T cc = -bc * mean - db * rstd / n;
            for (uint j = tid; j < p.n; j += G)
                p.out[base + j] = rstd * p.g[base + j] * weight(j) + bc * p.x[base + j] + cc;
        }
    }
}
