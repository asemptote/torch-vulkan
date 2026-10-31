// axis T: f32=float f64=double i32=int i64=int64_t b8=uint8_t
//
// Tree reduction of `in` over its reduced dims, one result per output index.
// Each workgroup reduces chunk c of output o's reduction range and writes
// out[o * nchunks + c]; with nchunks > 1 the host reduces those partials in a
// further pass (LEVEL0 = 0, input already of the accumulator type A).
//
// A (accumulator and output type) is T for floats and int64 for integers and
// bool, as in PyTorch's sums. OP codes must match csrc/reduce.cpp.

#include "opcodes.slang"

#if defined(T_f32)
typedef float A;
#define A_FLOAT 1
#elif defined(T_f64)
typedef double A;
#define A_FLOAT 1
#else
typedef int64_t A;
#endif

// 128 bytes; every field naturally aligned.
struct Params
{
    T* in;          // 0
    A* out;         // 8   partials [n_out, nchunks] or the result [n_out]
    A* mean;        // 16  per-output mean (OP_SQDEV)
    uint ond;       // 24  number of outer (kept) dims, <= 4, innermost first
    uint rnd;       // 28  number of reduced dims, <= 4, innermost first
    double alpha;   // 32  scale for FINAL (1/N for mean, 1/(N-ddof) for var)
    uint n_out;     // 40
    uint n_red;     // 44  elements reduced per output
    uint chunk;     // 48  elements per workgroup chunk
    uint nchunks;   // 52
    uint ngroups;   // 56  workgroups launched (grid-stride over n_out * nchunks)
    uint _pad;      // 60
    uint4 osz;      // 64  outer dims: sizes, input strides
    int4 ost;       // 80
    uint4 rsz;      // 96  reduced dims: sizes, input strides
    int4 rst;       // 112
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = 0;
[vk::constant_id(1)] const int LEVEL0 = 1;
[vk::constant_id(2)] const int FINAL = FINAL_NONE;  // FINAL_SCALE: * alpha, FINAL_SQRT_SCALE: sqrt(* alpha)
// 1: each thread reduces one (output, chunk) alone. Used when the reduced
// dims are strided and outputs are adjacent in memory (e.g. summing columns):
// neighbouring threads then read neighbouring addresses.
[vk::constant_id(3)] const int PER_THREAD = 0;

#define OP_SUM REDUCE_SUM
#define OP_PROD REDUCE_PROD
#define OP_MAX REDUCE_MAX
#define OP_MIN REDUCE_MIN
#define OP_SUMABS REDUCE_SUMABS
#define OP_SUMSQ REDUCE_SUMSQ
#define OP_MAXABS REDUCE_MAXABS
#define OP_MINABS REDUCE_MINABS
#define OP_SQDEV REDUCE_SQDEV
#define OP_ANY REDUCE_ANY
#define OP_ALL REDUCE_ALL

static const uint WG = 256;
groupshared A sdata[WG];

bool is_max_like() { return OP == OP_MAX || OP == OP_MAXABS || OP == OP_ANY; }
bool is_min_like() { return OP == OP_MIN || OP == OP_MINABS || OP == OP_ALL; }

A identity()
{
    if (OP == OP_PROD || OP == OP_ALL) return A(1);
#ifdef A_FLOAT
    if (is_max_like() && OP != OP_ANY) return A(-1.0 / 0.0);
    if (is_min_like() && OP != OP_ALL) return A(1.0 / 0.0);
#else
    if (is_max_like() && OP != OP_ANY) return A(0x8000000000000000ull);
    if (is_min_like() && OP != OP_ALL) return A(0x7fffffffffffffffull);
#endif
    return A(0);
}

A combine(A a, A b)
{
    if (OP == OP_PROD) return a * b;
    if (is_max_like() || is_min_like())
    {
#ifdef A_FLOAT
        if (isnan(a)) return a;  // NaN propagates, as in torch.max/amax
        if (isnan(b)) return b;
#endif
        return is_max_like() ? max(a, b) : min(a, b);
    }
    return a + b;
}

// acc += v, for float sums with Kahan's compensation c: a thread's running
// sum over thousands of elements otherwise loses ~n ulp (the CPU's and
// NumPy's pairwise sums do not). Infinities skip the compensation (inf - inf).
void accumulate(inout A acc, inout A c, A v)
{
#ifdef A_FLOAT
    if (OP == OP_SUM || OP == OP_SUMABS || OP == OP_SUMSQ || OP == OP_SQDEV)
    {
        A y = v - c;
        A t = acc + y;
        c = isinf(t) || isnan(t) ? A(0) : (t - acc) - y;
        acc = t;
        return;
    }
#endif
    acc = combine(acc, v);
}

// First-level transform of an input element.
A load(T x, uint o)
{
#if defined(T_b8)
    A v = A(x != T(0) ? 1 : 0);  // any nonzero byte is true (as on the CPU)
#else
    A v = A(x);
#endif
    if (LEVEL0 == 0) return v;
    switch (OP)
    {
    case OP_SUMABS:
    case OP_MAXABS:
    case OP_MINABS: return abs(v);
    case OP_SUMSQ: return v * v;
    case OP_SQDEV: { A d = v - p.mean[o]; return d * d; }
    case OP_ANY:
    case OP_ALL: return x != T(0) ? A(1) : A(0);
    default: return v;
    }
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

A finish(A r)
{
#ifdef A_FLOAT
    if (FINAL == FINAL_SCALE) return r * A(p.alpha);
    if (FINAL == FINAL_SQRT_SCALE) return sqrt(r * A(p.alpha));
#endif
    return r;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint work = p.n_out * p.nchunks;
    if (PER_THREAD != 0)
    {
        for (uint t = gid.x * WG + tid.x; t < work; t += p.ngroups * WG)
        {
            // Adjacent threads take adjacent outputs (coalesced reads).
            uint o = t % p.n_out;
            uint c = t / p.n_out;
            int base = offset(o, p.ond, p.osz, p.ost);
            uint end = min((c + 1) * p.chunk, p.n_red);
            A acc = identity(), comp = A(0);
            for (uint i = c * p.chunk; i < end; i++)
                accumulate(acc, comp, load(p.in[base + offset(i, p.rnd, p.rsz, p.rst)], o));
            p.out[o * p.nchunks + c] = p.nchunks == 1 ? finish(acc) : acc;
        }
        return;
    }
    for (uint w = gid.x; w < work; w += p.ngroups)  // uniform per workgroup
    {
        uint o = w / p.nchunks;
        uint c = w % p.nchunks;
        int base = offset(o, p.ond, p.osz, p.ost);
        uint end = min((c + 1) * p.chunk, p.n_red);
        A acc = identity(), comp = A(0);
        for (uint i = c * p.chunk + tid.x; i < end; i += WG)
            accumulate(acc, comp, load(p.in[base + offset(i, p.rnd, p.rsz, p.rst)], o));
        sdata[tid.x] = acc;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = WG / 2; s > 0; s >>= 1)
        {
            if (tid.x < s)
                sdata[tid.x] = combine(sdata[tid.x], sdata[tid.x + s]);
            GroupMemoryBarrierWithGroupSync();
        }
        if (tid.x == 0)
            p.out[w] = p.nchunks == 1 ? finish(sdata[0]) : sdata[0];
        GroupMemoryBarrierWithGroupSync();  // sdata is reused by the next work item
    }
}
