// axis T: f32=float f64=double i32=int i64=int64_t b8=uint8_t
//
// Inclusive scan (cumsum / cumprod) along one dim, in three phases driven by
// the host:
//   PHASE 0: each workgroup scans a 1024-element chunk of one row into `out`
//            (contiguous [rows, n]) and writes the chunk total to totals.
//   (host)   the totals [rows, nchunks] are scanned recursively.
//   PHASE 1: chunk c > 0 of each row is combined with scanned total c - 1.
// A is T for floats and int64 for integers and bool (PyTorch's cumsum
// dtype). OP: 0 sum, 1 prod.

#include "opcodes.slang"

#if defined(T_f32)
typedef float A;
#elif defined(T_f64)
typedef double A;
#else
typedef int64_t A;
#endif

struct Params
{
    T* in;          // 0   PHASE 0 input (any strides)
    A* out;         // 8   contiguous [rows, n]
    A* totals;      // 16  [rows, nchunks]: written by PHASE 0, read (scanned) by PHASE 1
    uint ond;       // 24  number of row dims, <= 4, innermost first
    int dstride;    // 28  input stride along the scanned dim
    uint rows;      // 32
    uint n;         // 36  length of the scanned dim
    uint nchunks;   // 40
    uint ngroups;   // 44
    uint _pad0;     // 48
    uint _pad1;     // 52
    uint _pad2;     // 56
    uint _pad3;     // 60
    uint4 osz;      // 64  row dims: sizes, input strides
    int4 ost;       // 80
    uint4 _pad4;    // 96
    uint4 _pad5;    // 112
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = 0;
[vk::constant_id(1)] const int PHASE = 0;

static const uint WG = 256;
static const uint ITEMS = 4;
static const uint CHUNK = WG * ITEMS;
groupshared A s[2][WG];

A identity() { return OP == SCAN_PROD ? A(1) : A(0); }
A combine(A a, A b) { return OP == SCAN_PROD ? a * b : a + b; }

int row_offset(uint row)
{
    int off = 0;
    for (uint d = 0; d < 4; d++)
    {
        if (d < p.ond)
        {
            off += int(row % p.osz[d]) * p.ost[d];
            row /= p.osz[d];
        }
    }
    return off;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint work = p.rows * p.nchunks;
    for (uint w = gid.x; w < work; w += p.ngroups)
    {
        uint row = w / p.nchunks;
        uint c = w % p.nchunks;
        uint first = c * CHUNK + tid.x * ITEMS;

        if (PHASE == 1)  // add the scanned chunk totals
        {
            if (c > 0)
            {
                A carry = p.totals[row * p.nchunks + c - 1];
                for (uint k = 0; k < ITEMS; k++)
                {
                    uint i = first + k;
                    if (i < p.n)
                        p.out[row * p.n + i] = combine(carry, p.out[row * p.n + i]);
                }
            }
            continue;
        }

        // Serial scan of this thread's items.
        int base = row_offset(row);
        A items[ITEMS];
        A running = identity();
        for (uint k = 0; k < ITEMS; k++)
        {
            uint i = first + k;
#if defined(T_b8)
            A v = i < p.n ? A(p.in[base + int(i) * p.dstride] != T(0) ? 1 : 0) : identity();  // any nonzero byte is true
#else
            A v = i < p.n ? A(p.in[base + int(i) * p.dstride]) : identity();
#endif
            running = combine(running, v);
            items[k] = running;
        }

        // Hillis-Steele scan of the per-thread totals across the workgroup.
        uint cur = 0;
        s[cur][tid.x] = running;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 1; step < WG; step <<= 1)
        {
            A v = s[cur][tid.x];
            if (tid.x >= step)
                v = combine(s[cur][tid.x - step], v);
            s[1 - cur][tid.x] = v;
            cur = 1 - cur;
            GroupMemoryBarrierWithGroupSync();
        }
        A prefix = tid.x > 0 ? s[cur][tid.x - 1] : identity();

        for (uint k = 0; k < ITEMS; k++)
        {
            uint i = first + k;
            if (i < p.n)
                p.out[row * p.n + i] = combine(prefix, items[k]);
        }
        if (tid.x == WG - 1 && p.nchunks > 1)
            p.totals[w] = s[cur][WG - 1];
        GroupMemoryBarrierWithGroupSync();
    }
}
