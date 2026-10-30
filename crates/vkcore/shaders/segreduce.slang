// axis T: f32=float f64=double i32=int i64=int64_t
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

#include "opcodes.slang"

struct Params
{
    T* dst;        // destination, offsets relative to this
    int64_t* offs; // per source element
    T* vals;       // per source element
    uint* order;   // source elements sorted by offset
    uint n;        // source elements
    uint stride;
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
    for (uint s = id.x; s < p.n; s += p.stride)
    {
        int64_t off = p.offs[p.order[s]];
        if (s > 0u && p.offs[p.order[s - 1u]] == off)
            continue;  // not the start of a run
        bool have = INCLUDE_SELF != 0;
        T acc = have ? p.dst[off] : T(0);
        uint count = have ? 1u : 0u;
        for (uint j = s; j < p.n && p.offs[p.order[j]] == off; j++)
        {
            T v = p.vals[p.order[j]];
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
            acc = (acc % c != T(0) && acc < T(0)) ? q - T(1) : q;
#endif
        }
        p.dst[off] = acc;
    }
}
