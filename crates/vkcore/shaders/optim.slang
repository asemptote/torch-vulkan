// axis T: f32=float f64=double
//
// Fused optimiser steps (torch.optim with fused=True), one launch per
// parameter tensor, elementwise over its n contiguous elements, with the
// CPU's formulas (aten/src/ATen/native/cpu/Fused*Kernel.cpp, their
// vectorised loops). The host computes the per-tensor coefficients in double
// precision from the step count, as the CPU does, and passes them rounded to
// T. OP:
//   OPTIM_ADAM, OPTIM_ADAMW: buf1 = exp_avg, buf2 = exp_avg_sq,
//     buf3 = max_exp_avg_sq (AMSGRAD); c = gs, wd, 1 - lr wd, 1 - beta1,
//     beta2, 1 - beta2, sqrt(bias_correction2), eps, -lr / bias_correction1
//   OPTIM_SGD: buf1 = momentum_buffer (MOMENTUM); c = gs, wd, momentum,
//     1 - dampening, -lr
//   OPTIM_ADAGRAD: buf1 = state_sum; c = gs, wd, eps, lr / (1 + (step - 1) lr_decay)
// FLAGS (bits): SCALED (grad /= gs, written back, as the CPU), MAXIMIZE,
// DECAY (weight_decay != 0), AMSGRAD, MOMENTUM, FIRST (SGD's first step),
// NESTEROV.

#include "_slot.slang"
#include "opcodes.slang"

struct Params
{
    T* param;      // 0
    T* grad;       // 8
    T* buf1;       // 16
    T* buf2;       // 24
    T* buf3;       // 32
    uint n;        // 40
    uint stride;   // 44
    uint flags;    // 48
    uint _pad;     // 52
    slot64 c[9];   // 56  raw slots (_slot.slang): T bits
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = OPTIM_ADAM;

static const uint SCALED = 1;
static const uint MAXIMIZE = 2;
static const uint DECAY = 4;
static const uint AMSGRAD = 8;
static const uint MOMENTUM = 16;
static const uint FIRST = 32;
static const uint NESTEROV = 64;

T coef(uint i)
{
#if defined(T_f64)
    return slot_double(p.c[i]);
#else
    return slot_float(p.c[i]);
#endif
}

bool has(uint flag) { return (p.flags & flag) != 0; }

[numthreads(64, 1, 1)]  // (vkt_grid_1d's workgroup size)
void main(uint3 gid: SV_DispatchThreadID)
{
    for (uint i = gid.x; i < p.n; i += p.stride)
    {
        T param = p.param[i];
        T grad = p.grad[i];
        if (has(SCALED))
        {
            grad = grad / coef(0);
            p.grad[i] = grad;
        }
        if (has(MAXIMIZE))
            grad = -grad;
        if (OP == OPTIM_ADAM || OP == OPTIM_ADAMW)
        {
            if (has(DECAY))
            {
                if (OP == OPTIM_ADAM)
                    grad = grad + param * coef(1);
                else
                    param = param * coef(2);
            }
            // exp_avg.lerp_(grad, 1 - beta1), as the CPU's vectorised lerp
            T w = coef(3);
            T m = p.buf1[i];
            bool small = abs(w) < T(0.5);
            m = fma(small ? w : w - T(1), grad - m, small ? m : grad);
            T v = p.buf2[i] * coef(4) + coef(5) * grad * grad;
            p.buf1[i] = m;
            p.buf2[i] = v;
            if (has(AMSGRAD))
            {
                v = max(p.buf3[i], v);
                p.buf3[i] = v;
            }
            T denom = sqrt(v) / coef(6) + coef(7);
            p.param[i] = param + coef(8) * m / denom;
        }
        else if (OP == OPTIM_SGD)
        {
            if (has(DECAY))
                grad = fma(param, coef(1), grad);
            if (has(MOMENTUM))
            {
                T buf = has(FIRST) ? grad : fma(coef(3), grad, p.buf1[i] * coef(2));
                p.buf1[i] = buf;
                grad = has(NESTEROV) ? fma(buf, coef(2), grad) : buf;
            }
            p.param[i] = param + grad * coef(4);
        }
        else  // OPTIM_ADAGRAD
        {
            if (has(DECAY))
                grad = grad + param * coef(1);
            T sum = p.buf1[i] + grad * grad;
            p.buf1[i] = sum;
            T std_ = sqrt(sum) + coef(2);
            p.param[i] = param - coef(3) * grad / std_;
        }
    }
}
