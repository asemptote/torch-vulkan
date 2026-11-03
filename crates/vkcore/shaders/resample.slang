// axis T: f32=float f64=double
//
// Linear interpolation (upsample_linear1d / bilinear2d / trilinear3d) on
// contiguous [planes, D, H, W] tensors (fewer dims: leading sizes 1, left
// out of the sums, where a weight of 0 would turn an infinity into NaN),
// one thread per output, as the CPU:
// per dim, the source position is
//   align_corners: scale * o                   (scale = (in - 1) / (out - 1))
//   otherwise:     max(scale * (o + 0.5) - 0.5, 0)  (scale = 1 / scales or in / out)
// its floor i0 (at most in - 1) and i1 = i0 + 1 (i0 at the last input), with
// weights 1 - l and l = clamp(position - i0, 0, 1). The scales are computed
// by the host in T, as the CPU does in its opmath type.
//
// RESAMPLE_BACKWARD: one dim's backward, g [A, n_out, B] -> gx [A, n_in, B]
// (isz = A, n_in, B; osz.y = n_out; scale0): one thread per input position
// sums g over the outputs whose taps include it (also with weight 0, as the
// CPU adds those), in output order, w0 and w1 separately where both taps
// are this input.

#include "opcodes.slang"
#include "_slot.slang"

struct Params
{
    T* in;            // 0
    T* out;           // 8
    uint total;       // 16  planes * output size
    uint ngroups;     // 20
    uint align;       // 24  align_corners
    uint ndim;        // 28  spatial dims (1-3): the leading padded ones are not mixed in
    uint4 isz;        // 32  input D, H, W
    uint4 osz;        // 48  output D, H, W
    slot64 scale0;    // 64  raw slots (_slot.slang: T bits; float in the low word) per dim D, H, W
    slot64 scale1;    // 72
    slot64 scale2;    // 80
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = RESAMPLE_FORWARD;

T param(slot64 bits)
{
#if defined(T_f64)
    return slot_double(bits);
#else
    return slot_float(bits);
#endif
}

// (i0, i1) and (w0, w1) of output index o along one dim.
void taps(uint o, uint n_in, uint n_out, T scale, out uint i0, out uint i1, out T w0, out T w1)
{
    T src = p.align != 0 ? (n_out > 1 ? scale * T(o) : T(0)) : max(scale * (T(o) + T(0.5)) - T(0.5), T(0));
    i0 = min(uint(floor(src)), n_in - 1);
    i1 = i0 + (i0 < n_in - 1 ? 1u : 0u);
    w1 = clamp(src - T(i0), T(0), T(1));
    w0 = T(1) - w1;
}

// First and last output whose taps can include input i (a superset).
void output_range(uint i, uint n_in, uint n_out, T scale, out uint lo, out uint hi)
{
    lo = 0;
    hi = n_out - 1;
    if (!(scale > T(0)))
        return;  // (scale 0: every output samples input 0)
    T a = p.align != 0 ? (T(i) - T(1)) / scale : (T(i) - T(0.5)) / scale - T(0.5);
    T b = p.align != 0 ? (T(i) + T(1)) / scale : (T(i) + T(1.5)) / scale - T(0.5);
    if (i > 1)  // (outputs before the first input's centre clamp to 0: taps 0 and 1)
        lo = uint(clamp(floor(a) - T(1), T(0), T(n_out - 1)));
    if (i + 1 < n_in)  // (outputs past the last input clamp to it)
        hi = uint(clamp(ceil(b) + T(1), T(0), T(n_out - 1)));
}

void backward(uint t)
{
    uint B = p.isz.z, n_in = p.isz.y, n_out = p.osz.y;
    uint a = t / (n_in * B), r = t - a * (n_in * B);
    uint i = r / B, b = r - i * B;
    T scale = param(p.scale0);
    uint lo, hi;
    output_range(i, n_in, n_out, scale, lo, hi);
    T acc = T(0);
    for (uint o = lo; o <= hi; o++)
    {
        uint i0, i1;
        T w0, w1;
        taps(o, n_in, n_out, scale, i0, i1, w0, w1);
        T g = p.in[(a * n_out + o) * B + b];
        if (i0 == i)
            acc += w0 * g;
        if (i1 == i)
            acc += w1 * g;
    }
    p.out[t] = acc;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    if (OP == RESAMPLE_BACKWARD)
    {
        for (uint t = gid.x * 256 + tid.x; t < p.total; t += p.ngroups * 256)
            backward(t);
        return;
    }
    uint ov = p.osz.x * p.osz.y * p.osz.z;
    uint iv = p.isz.x * p.isz.y * p.isz.z;
    for (uint t = gid.x * 256 + tid.x; t < p.total; t += p.ngroups * 256)
    {
        uint plane = t / ov, r = t - plane * ov;
        uint ox = r % p.osz.z;
        r /= p.osz.z;
        uint oy = r % p.osz.y, oz = r / p.osz.y;
        uint z0, z1, y0, y1, x0, x1;
        T wz0, wz1, wy0, wy1, wx0, wx1;
        taps(oz, p.isz.x, p.osz.x, param(p.scale0), z0, z1, wz0, wz1);
        taps(oy, p.isz.y, p.osz.y, param(p.scale1), y0, y1, wy0, wy1);
        taps(ox, p.isz.z, p.osz.z, param(p.scale2), x0, x1, wx0, wx1);
        uint base = plane * iv;
        uint r00 = base + (z0 * p.isz.y + y0) * p.isz.z, r01 = base + (z0 * p.isz.y + y1) * p.isz.z;
        uint r10 = base + (z1 * p.isz.y + y0) * p.isz.z, r11 = base + (z1 * p.isz.y + y1) * p.isz.z;
        T a00 = wx0 * p.in[r00 + x0] + wx1 * p.in[r00 + x1];
        if (p.ndim == 1)
        {
            p.out[t] = a00;
            continue;
        }
        T v0 = wy0 * a00 + wy1 * (wx0 * p.in[r01 + x0] + wx1 * p.in[r01 + x1]);
        if (p.ndim == 2)
        {
            p.out[t] = v0;
            continue;
        }
        T v1 = wy0 * (wx0 * p.in[r10 + x0] + wx1 * p.in[r10 + x1]) + wy1 * (wx0 * p.in[r11 + x0] + wx1 * p.in[r11 + x1]);
        p.out[t] = wz0 * v0 + wz1 * v1;
    }
}
