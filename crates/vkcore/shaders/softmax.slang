// axis T: f32=float f64=double
//
// Softmax along one dim of a contiguous tensor seen as [outer, n, inner]:
// one workgroup per row (outer, inner) of n elements (stride inner). OP:
//   SOFTMAX_FORWARD:         y = exp(x - max) / sum(exp(x - max))
//   SOFTMAX_LOG_FORWARD:     y = x - max - log(sum(exp(x - max)))
//   SOFTMAX_BACKWARD:        gx = y * (g - sum(g * y))         (a: g, b: y)
//   SOFTMAX_LOG_BACKWARD:    gx = g - exp(y) * sum(g)          (a: g, b: y)
// A NaN in a row makes its maximum NaN (as the CPU's vectorized max), so the
// whole row comes out NaN.

#include "_fmath.slang"
#include "opcodes.slang"

struct Params
{
    T* a;          // 0   x (forward) or grad_output (backward)
    T* b;          // 8   output of the forward (backward only)
    T* out;        // 16
    uint rows;     // 24  outer * inner
    uint n;        // 28  elements per row
    uint inner;    // 32  stride between a row's elements
    uint ngroups;  // 36
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = SOFTMAX_FORWARD;

static const uint G = 256;
groupshared T shared_[G];

// Workgroup reduction of v (max when IS_MAX, else sum); every thread gets it.
T reduce(T v, uint tid, bool is_max)
{
    shared_[tid] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = G / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            T o = shared_[tid + s], m = shared_[tid];
            shared_[tid] = is_max ? ((isnan(m) || m > o) ? m : o) : m + o;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    T r = shared_[0];
    GroupMemoryBarrierWithGroupSync();  // (shared_ is reused by the next reduction)
    return r;
}

[shader("compute")]
[numthreads(G, 1, 1)]
void main(uint3 tid3: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint tid = tid3.x;
    for (uint row = gid.x; row < p.rows; row += p.ngroups)
    {
        uint outer = row / p.inner, in_i = row - outer * p.inner;
        uint base = outer * p.n * p.inner + in_i;
        if (OP == SOFTMAX_FORWARD || OP == SOFTMAX_LOG_FORWARD)
        {
            T m = -m_inf();
            for (uint j = tid; j < p.n; j += G)
            {
                T v = p.a[base + j * p.inner];
                m = (isnan(m) || m > v) ? m : v;
            }
            m = reduce(m, tid, true);
            T s = T(0);
            for (uint j = tid; j < p.n; j += G)
                s += m_exp(p.a[base + j * p.inner] - m);
            s = reduce(s, tid, false);
            if (OP == SOFTMAX_FORWARD)
            {
                for (uint j = tid; j < p.n; j += G)
                    p.out[base + j * p.inner] = m_exp(p.a[base + j * p.inner] - m) / s;
            }
            else
            {
                T ls = m_log(s);
                for (uint j = tid; j < p.n; j += G)
                    p.out[base + j * p.inner] = p.a[base + j * p.inner] - m - ls;
            }
        }
        else
        {
            T s = T(0);
            for (uint j = tid; j < p.n; j += G)
            {
                uint e = base + j * p.inner;
                s += OP == SOFTMAX_BACKWARD ? p.a[e] * p.b[e] : p.a[e];
            }
            s = reduce(s, tid, false);
            for (uint j = tid; j < p.n; j += G)
            {
                uint e = base + j * p.inner;
                p.out[e] = OP == SOFTMAX_BACKWARD ? p.b[e] * (p.a[e] - s) : p.a[e] - m_exp(p.b[e]) * s;
            }
        }
    }
}
