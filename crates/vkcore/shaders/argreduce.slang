// axis T: f32=float f64=double i32=int i64=int64_t
//
// (value, index) reduction for argmax/argmin and max/min along a dim. Same
// chunking scheme as reduce.slang: each workgroup writes the best
// (value, index) of its chunk; later passes (LEVEL0 = 0) reduce those pairs,
// reading indices from in_idx. Ties and NaNs resolve to the smallest index,
// matching PyTorch (NaN counts as the maximum and the minimum).

#include "opcodes.slang"

struct Params
{
    T* in;              // 0
    T* out;             // 8   values: partials [n_out, nchunks] or results [n_out]
    int64_t* out_idx;   // 16  indices, same layout
    int64_t* in_idx;    // 24  indices of `in` when LEVEL0 == 0
    uint ond;           // 32
    uint rnd;           // 36
    uint n_out;         // 40
    uint n_red;         // 44
    uint chunk;         // 48
    uint nchunks;       // 52
    uint ngroups;       // 56
    uint _pad;          // 60
    uint4 osz;          // 64
    int4 ost;           // 80
    uint4 rsz;          // 96
    int4 rst;           // 112
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = ARGREDUCE_MAX;
[vk::constant_id(1)] const int LEVEL0 = 1;

static const uint WG = 256;
groupshared T svalue[WG];
groupshared int64_t sindex[WG];

#if defined(T_f32) || defined(T_f64)
bool nan_(T x) { return isnan(x); }
#else
bool nan_(T x) { return false; }
#endif

// Whether (vb, ib) should replace (va, ia). An index < 0 marks "empty".
bool better(T va, int64_t ia, T vb, int64_t ib)
{
    if (ib < 0) return false;
    if (ia < 0) return true;
    if (nan_(va)) return nan_(vb) && ib < ia;
    if (nan_(vb)) return true;
    if (vb == va) return ib < ia;
    return OP == ARGREDUCE_MAX ? vb > va : vb < va;
}

int offset(uint index, uint nd, uint4 sz, int4 st)
{
    if (nd == 1) return int(index) * st[0];  // common case: no integer division
    int off = 0;
    for (uint d = 0; d < 4; d++)
    {
        if (d < nd)
        {
            off += int(index % sz[d]) * st[d];
            index /= sz[d];
        }
    }
    return off;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint work = p.n_out * p.nchunks;
    for (uint w = gid.x; w < work; w += p.ngroups)
    {
        uint o = w / p.nchunks;
        uint c = w % p.nchunks;
        int base = offset(o, p.ond, p.osz, p.ost);
        uint end = min((c + 1) * p.chunk, p.n_red);
        T bv = T(0);
        int64_t bi = -1;
        for (uint i = c * p.chunk + tid.x; i < end; i += WG)
        {
            int at = base + offset(i, p.rnd, p.rsz, p.rst);
            T v = p.in[at];
            int64_t idx = LEVEL0 != 0 ? int64_t(i) : p.in_idx[at];
            if (better(bv, bi, v, idx))
            {
                bv = v;
                bi = idx;
            }
        }
        svalue[tid.x] = bv;
        sindex[tid.x] = bi;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2; s > 0; s >>= 1)
        {
            if (tid.x < s && better(svalue[tid.x], sindex[tid.x], svalue[tid.x + s], sindex[tid.x + s]))
            {
                svalue[tid.x] = svalue[tid.x + s];
                sindex[tid.x] = sindex[tid.x + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid.x == 0)
        {
            p.out[w] = svalue[0];
            p.out_idx[w] = sindex[0];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
