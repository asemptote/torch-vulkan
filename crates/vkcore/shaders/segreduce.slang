// axis T: f32=float f64=double i32=int i64=I64
//
// Accumulation into repeated destinations (index_add, scatter_add,
// scatter_reduce, index_put_ with accumulate, bincount, ...) without atomics:
// `order` lists the source elements sorted (stably) by destination offset
// `offs`, so equal destinations form runs. The thread at the start of each
// run folds the run's values in source order, like the CPU's sequential
// loop, and writes dst[offset] once:
//   acc = INCLUDE_SELF ? dst[off] : identity;  acc = op(acc, vals[...]) ...
// OP: SEGREDUCE_SUM, _PROD, _MAX, _MIN (NaN propagates), _MEAN ((acc) / count,
// counting dst when included; integer means round toward -inf).
// Rows (width > 1): each source element is a row of `width` contiguous values
// (vals[i * width + col]) and offs its destination row's first element
// (dst[off + col]); a thread per (sorted position, col). scale_by_count (SUM)
// multiplies each value by 1 / (its run's length) before adding it, as
// embedding_dense_backward's scale_grad_by_freq does.

#include "_int64.slang"  // (the i64 variant's T: _int64.slang)
#include "opcodes.slang"
#include "_wide.slang"

struct Params
{
    T* dst;        // destination, offsets relative to this
    i64w* offs;    // per source element (int64 as words: _wide.slang)
    T* vals;       // per source element
    uint* order;   // source elements sorted by offset
    uint n;        // source elements (rows)
    uint stride;
    uint width;    // values per row (0: 1)
    uint scale_by_count;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = SEGREDUCE_SUM;
[vk::constant_id(1)]
const int INCLUDE_SELF = 1;

#if defined(T_f32) || defined(T_f64)
#define IS_FLOAT 1
#endif

T combine(T a, T b)
{
    switch (OP)
    {
    case SEGREDUCE_PROD: return a * b;
#ifdef IS_FLOAT
    case SEGREDUCE_MAX: return (isnan(a) || isnan(b)) ? a + b : max(a, b);
    case SEGREDUCE_MIN: return (isnan(a) || isnan(b)) ? a + b : min(a, b);
#else
    case SEGREDUCE_MAX: return max(a, b);
    case SEGREDUCE_MIN: return min(a, b);
#endif
    default: return a + b;  // SUM, MEAN
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    // (Every index into order stays in range, also on paths a lane does not
    // take: lavapipe's code performs those loads too.)
    uint last = max(p.n, 1u) - 1u;
    uint width = max(p.width, 1u);
    for (uint t = id.x; t < p.n * width; t += p.stride)
    {
        uint s = t / width, col = t - s * width;
        i64w offw = p.offs[p.order[min(s, last)]];
        int off = i64w_int(offw);
        if (s > 0u && all(p.offs[p.order[max(s, 1u) - 1u]] == offw))
            continue;  // not the start of a run
        T scale = T(1);
#ifdef IS_FLOAT
        if (p.scale_by_count != 0)  // (floating types only)
        {
            uint run = 0;
            for (uint j = s; j < p.n && all(p.offs[p.order[min(j, last)]] == offw); j++)
                run++;
            scale = T(1) / T(run);
        }
#endif
        bool have = INCLUDE_SELF != 0;
        T acc = have ? p.dst[off + col] : T(0);
        uint count = have ? 1u : 0u;
        for (uint j = s; j < p.n && all(p.offs[p.order[min(j, last)]] == offw); j++)
        {
            T v = p.vals[p.order[min(j, last)] * width + col];
            if (p.scale_by_count != 0)
                v = v * scale;
            acc = have ? combine(acc, v) : v;
            have = true;
            count++;
        }
        if (OP == SEGREDUCE_MEAN)
        {
#ifdef IS_FLOAT
            acc = acc / T(count);
#else
            T c = T(count);
            T q = acc / c;
            acc = (acc - q * c != T(0) && acc < T(0)) ? q - T(1) : q;  // (not %: see binary.slang's irem)
#endif
        }
        p.dst[off + col] = acc;
    }
}
