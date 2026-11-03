// axis T: f32=float f64=double
//
// nll_loss over input [N, C] and int64 target [N], one thread per sample:
//   NLL_FORWARD:  loss[n] = -x[n, t] w[t], wsum[n] = w[t] (both 0 where
//                 t == ignore_index); the host reduces them
//   NLL_BACKWARD: gx[n, t] = w[t] g, gx zeroed by the host, with g =
//                 -gout[n] (PER_SAMPLE, reduction none) or gout[0] (the
//                 host's -(grad_output (/ total_weight)))
// w is weight (HAS_WEIGHT) or 1. A target outside [0, C) (and not
// ignore_index) is skipped and recorded in the error word, which the host
// raises as an IndexError when it next synchronizes.

#include "opcodes.slang"
#include "_wide.slang"

struct Params
{
    T* x;              // 0   (forward)
    i64w* target;      // 8   (int64 as words: _wide.slang)
    T* weight;         // 16
    T* out;            // 24  loss [N] (forward) or gx [N, C] (backward)
    T* wsum;           // 32  [N] (forward)
    T* gout;           // 40  (backward)
    uint* error;       // 48
    uint n;            // 56
    uint c;            // 60
    i64w ignore;       // 64
    uint stride;       // 72
    uint _pad;         // 76
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = NLL_FORWARD;
[vk::constant_id(1)] const int HAS_WEIGHT = 0;
[vk::constant_id(2)] const int PER_SAMPLE = 0;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        i64w tw = p.target[i];
        bool ignored = all(tw == p.ignore);
        int t = i64w_int(tw);
        bool valid = !ignored && i64w_fits(tw) && t >= 0 && t < int(p.c);
        if (!ignored && !valid)
            *p.error = 1u;
        uint ti = valid ? uint(t) : 0u;
        T w = (valid && HAS_WEIGHT != 0) ? p.weight[ti] : T(1);
        if (OP == NLL_FORWARD)
        {
            p.out[i] = valid ? -p.x[i * p.c + ti] * w : T(0);
            p.wsum[i] = valid ? w : T(0);
        }
        else if (valid)
        {
            T g = PER_SAMPLE != 0 ? -p.gout[i] : p.gout[0];
            p.out[i * p.c + ti] = HAS_WEIGHT != 0 ? w * g : g;
        }
    }
}
