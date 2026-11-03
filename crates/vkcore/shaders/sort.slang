// axis K: k32=uint k64=U64
//
// Bitonic sort of (key, index) pairs in rows of P entries (a power of two),
// all rows at once: comparing the index on equal keys makes the result the
// stable sort. The host drives the network:
//   SORT_LOCAL_FULL   every 1024-entry block sorts itself in shared memory
//                     (stages k = 2 .. min(P, 1024))
//   SORT_GLOBAL       one compare-exchange step (j, k) with j >= 1024
//   SORT_LOCAL_MERGE  the remaining steps j = 512 .. 1 of stage k, in
//                     shared memory
// `total` (all rows) is a multiple of 1024; each pair sorts ascending when
// bit k of its position in the row is 0 (stage k merges runs of length k).

#include "_int64.slang"  // (the i64 variant's T: _int64.slang)
struct Params
{
    K* keys;
    uint* idx;
    uint total;
    uint P;
    uint j;
    uint k;
    uint stride;  // grid stride (SORT_GLOBAL)
    uint _pad;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = 0;

#define SORT_LOCAL_FULL 0
#define SORT_GLOBAL 1
#define SORT_LOCAL_MERGE 2
#define BLOCK 1024

groupshared K skey[BLOCK];
groupshared uint sidx[BLOCK];

bool greater(K ka, uint ia, K kb, uint ib)
{
    return ka > kb || (ka == kb && ia > ib);
}

// Ascending if bit k of the in-row position is clear.
bool ascending(uint pos, uint k)
{
    return ((pos & (p.P - 1u)) & k) == 0u;
}

void local_steps(uint base, uint t, uint k, uint jstart)
{
    for (uint j = jstart; j > 0u; j >>= 1)
    {
        uint lo = (t / j) * 2u * j + (t % j);
        uint hi = lo + j;
        bool up = ascending(base + lo, k);
        if (greater(skey[lo], sidx[lo], skey[hi], sidx[hi]) == up)
        {
            K tk = skey[lo];
            skey[lo] = skey[hi];
            skey[hi] = tk;
            uint ti = sidx[lo];
            sidx[lo] = sidx[hi];
            sidx[hi] = ti;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID, uint3 id: SV_DispatchThreadID)
{
    if (OP == SORT_GLOBAL)
    {
        for (uint t = id.x; t < p.total / 2u; t += p.stride)
        {
            uint lo = (t / p.j) * 2u * p.j + (t % p.j);
            uint hi = lo + p.j;
            K ka = p.keys[lo], kb = p.keys[hi];
            uint ia = p.idx[lo], ib = p.idx[hi];
            if (greater(ka, ia, kb, ib) == ascending(lo, p.k))
            {
                p.keys[lo] = kb;
                p.keys[hi] = ka;
                p.idx[lo] = ib;
                p.idx[hi] = ia;
            }
        }
        return;
    }
    uint base = gid.x * BLOCK;
    uint t = lid.x;
    skey[t] = p.keys[base + t];
    sidx[t] = p.idx[base + t];
    skey[t + 512u] = p.keys[base + t + 512u];
    sidx[t + 512u] = p.idx[base + t + 512u];
    GroupMemoryBarrierWithGroupSync();
    if (OP == SORT_LOCAL_FULL)
    {
        uint kmax = min(p.P, uint(BLOCK));
        for (uint k = 2u; k <= kmax; k <<= 1)
            local_steps(base, t, k, k >> 1);
    }
    else
    {
        local_steps(base, t, p.k, uint(BLOCK) >> 1);
    }
    p.keys[base + t] = skey[t];
    p.idx[base + t] = sidx[t];
    p.keys[base + t + 512u] = skey[t + 512u];
    p.idx[base + t + 512u] = sidx[t + 512u];
}
