// axis T: f32=float f64=double i32=int i64=int64_t b8=uint8_t
//
// Sequential per-row passes, one thread per row of a contiguous [rows, n]
// input (simple and exact; rows are scanned in order like the CPU does):
//   ROWSCAN_CUMMAX, ROWSCAN_CUMMIN  out[j] = running max/min, idx[j] = its
//       index (the latest on ties; a NaN, once seen, stays)
//   ROWSCAN_LOGCUMSUMEXP            out[j] = log(sum exp(x[0..j])) (floats)
//   ROWSCAN_MODE   x holds each row sorted ascending and idx the matching
//       original indices; out[row] / out_idx[row] = the most frequent value
//       (the smallest on ties) and the last index it occurs at (PyTorch
//       leaves the index among equal elements unspecified)

#include "opcodes.slang"
#include "_wide.slang"
#if defined(T_f32) || defined(T_f64)
#include "_fmath.slang"
#define IS_FLOAT 1
#endif

struct Params
{
    T* x;
    T* out;
    i64w* idx;         // (int64 as words: _wide.slang)
    i64w* out_idx;     // ROWSCAN_MODE
    uint rows;
    uint n;
    uint stride;
    uint _pad;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = ROWSCAN_CUMMAX;

bool is_nan(T v)
{
#ifdef IS_FLOAT
    return isnan(v);
#else
    return false;
#endif
}

#if defined(T_b8)
#define LOADX(i) (p.x[i] != T(0) ? T(1) : T(0))  // any nonzero byte is true
#else
#define LOADX(i) p.x[i]
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint row = id.x; row < p.rows; row += p.stride)
    {
        uint base = row * p.n;
        if (OP == ROWSCAN_MODE)
        {
            T best = LOADX(base);
            i64w best_idx = p.idx[base];
            uint best_count = 0u, count = 0u;
            for (uint j = 0u; j < p.n; j++)
            {
                T v = LOADX(base + j);
                count = (j > 0u && v == LOADX(base + max(j, 1u) - 1u)) ? count + 1u : 1u;  // (index in range for j = 0)
                if (count > best_count)
                {
                    best_count = count;
                    best = v;
                }
                if (v == best && count == best_count)
                    best_idx = p.idx[base + j];  // last index of the current best run
            }
            p.out[row] = best;
            p.out_idx[row] = best_idx;
            continue;
        }
        T acc = LOADX(base);
        int at = 0;
        for (uint j = 0u; j < p.n; j++)
        {
            T v = LOADX(base + j);
            if (OP == ROWSCAN_LOGCUMSUMEXP)
            {
#ifdef IS_FLOAT
                if (j > 0u)
                {
                    T hi = max(acc, v), lo = min(acc, v);
                    if (isnan(acc) || isnan(v))
                        acc = acc + v;
                    else if (isinf(hi) && hi == lo)
                        acc = hi;  // both +inf or both -inf
                    else
                        acc = hi + m_log1p(m_exp(lo - hi));
                }
#endif
                p.out[base + j] = acc;
                continue;
            }
            bool take = OP == ROWSCAN_CUMMAX ? v >= acc : v <= acc;
            if (j == 0u || (!is_nan(acc) && (is_nan(v) || take)))
            {
                acc = v;
                at = int(j);
            }
            p.out[base + j] = acc;
            p.idx[base + j] = i64w_from(at);
        }
    }
}
