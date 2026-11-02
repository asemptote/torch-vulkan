// axis T: f32=float f64=double
//
// Batch norm and group norm backward over a contiguous tensor seen as
// [K, O, L] (batch norm: [N, C, HxW], slice o a channel; group norm:
// [1, N C, HxW], slice o an (n, c)). OP:
//   NORMGRAD_STATS:    one workgroup per slice o: sum_g[o] = sum of g,
//                      sum_gx[o] = sum of g (x - center[o]) (center 0
//                      without HAS_CENTER)
//   NORMGRAD_APPLY_BN: gx = ((g - (x - center[o]) b[o]) - c[o]) a[o]
//                      (PyTorch's batch norm backward; TRAIN = 0: g a[o])
//   NORMGRAD_APPLY_GN: gx = a[o] g + b[o] x + c[o] (group norm's)
// The per-slice coefficients a, b, c come from the host.

#include "opcodes.slang"

struct Params
{
    T* x;          // 0
    T* g;          // 8   grad_output
    T* center;     // 16  [O] (HAS_CENTER)
    T* a;          // 24  [O]
    T* b;          // 32
    T* c;          // 40
    T* out;        // 48  gx (APPLY)
    T* sum_g;      // 56  [O] (STATS)
    T* sum_gx;     // 64  [O] (STATS)
    uint outer;    // 72  K
    uint slices;   // 76  O
    uint inner;    // 80  L
    uint ngroups;  // 84
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = NORMGRAD_STATS;
[vk::constant_id(1)] const int HAS_CENTER = 0;
[vk::constant_id(2)] const int TRAIN = 1;

static const uint G = 256;
groupshared T shared_[G];

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

[shader("compute")]
[numthreads(G, 1, 1)]
void main(uint3 tid3: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint tid = tid3.x;
    uint L = p.inner, O = p.slices;
    if (OP == NORMGRAD_STATS)
    {
        uint count = p.outer * L;
        for (uint o = gid.x; o < O; o += p.ngroups)
        {
            T ctr = HAS_CENTER != 0 ? p.center[o] : T(0);
            T sg = T(0), sgx = T(0);
            for (uint i = tid; i < count; i += G)
            {
                uint k = i / L, j = i - k * L;
                uint off = (k * O + o) * L + j;
                T gv = p.g[off];
                sg += gv;
                sgx += gv * (p.x[off] - ctr);
            }
            sg = reduce_sum(sg, tid);
            sgx = reduce_sum(sgx, tid);
            if (tid == 0)
            {
                p.sum_g[o] = sg;
                p.sum_gx[o] = sgx;
            }
        }
        return;
    }
    uint total = p.outer * O * L;
    for (uint i = gid.x * G + tid; i < total; i += p.ngroups * G)
    {
        uint o = (i / L) % O;
        T gv = p.g[i];
        if (OP == NORMGRAD_APPLY_BN)
        {
            if (TRAIN != 0)
            {
                T ctr = HAS_CENTER != 0 ? p.center[o] : T(0);
                p.out[i] = ((gv - (p.x[i] - ctr) * p.b[o]) - p.c[o]) * p.a[o];
            }
            else
            {
                p.out[i] = gv * p.a[o];
            }
        }
        else
        {
            p.out[i] = p.a[o] * gv + p.b[o] * p.x[i] + p.c[o];
        }
    }
}
