// axis T: f32=float f64=double
//
// grid_sampler_2d / 3d, bilinear (MODE 0) and nearest (MODE 1), with
// PyTorch's CPU arithmetic: the grid coordinate is unnormalized
// (align_corners: (g + 1) / 2 (size - 1), else ((g + 1) size - 1) / 2),
// reflected (reflection padding) and clipped (border, reflection), and a
// coordinate outside int range or NaN becomes -100 (outside the input); the
// corners' weights are (i0 + 1 - x) and (x - i0) per dim, in the CPU's corner
// order, and corners outside the input are skipped (zeros padding); nearest
// rounds ties to even. One thread per (batch, output point), all channels.
//   GRID_FORWARD:  out[n, c, p] = sum over corners of input * weight
//   GRID_BACKWARD: ggrid[n, p, d] = sum over corners and channels of
//                  input * gout * dweight / dcoord_d, times the coordinate's
//                  derivative; idx/wgt[n, p, corner] = the corner's offset in
//                  the input plane (-1: outside) and weight, for grad_input's
//                  scatter-add.
// Coordinate d of the grid indexes the input's dim from the last: 0 W, 1 H, 2 D.

#include "_fmath.slang"
#include "opcodes.slang"

struct Params
{
    T* input;      // 0   [N, C, D, H, W] contiguous (D = 1 for 2-D)
    T* grid;       // 8   [N, P, k]
    T* out;        // 16  [N, C, P] (forward)
    T* gout;       // 24  [N, C, P] (backward)
    T* ggrid;      // 32  [N, P, k] (backward, when want_grid)
    int64_t* idx;  // 40  [N, P, 2^k] (backward)
    T* wgt;        // 48  [N, P, 2^k] (backward)
    uint n;        // 56
    uint c;        // 60
    uint p;        // 64  output points per batch element
    uint k;        // 68  2 or 3
    uint padding;  // 72  0 zeros, 1 border, 2 reflection
    uint align;    // 76
    uint ngroups;  // 80
    uint want_grid;  // 84  backward: write ggrid
    uint _pad1;    // 88
    uint _pad2;    // 92
    uint4 isz;     // 96  input W, H, D (by grid coordinate) and 0
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = GRID_FORWARD;
[vk::constant_id(1)] const int MODE = 0;

// Input coordinate of grid coordinate g along a dim of `size`, and its
// derivative d(coordinate) / d(g) (PyTorch's grid_sampler_compute_source_index_set_grad).
T source(T g, uint size, out T mult)
{
    T x;
    if (p.align != 0)
    {
        x = (g + T(1)) / T(2) * T(size - 1);
        mult = T(size - 1) / T(2);
    }
    else
    {
        x = ((g + T(1)) * T(size) - T(1)) / T(2);
        mult = T(size) / T(2);
    }
    if (p.padding == 2)  // reflection about [0, size - 1] (align) or [-0.5, size - 0.5]
    {
        T twice_low = p.align != 0 ? T(0) : T(-1);
        T twice_high = p.align != 0 ? T(2) * T(size - 1) : T(2) * T(size) - T(1);
        if (twice_low == twice_high)
        {
            x = T(0);
            mult = T(0);
        }
        else
        {
            T low = twice_low / T(2), span = (twice_high - twice_low) / T(2);
            T sign = x < low ? T(-1) : T(1);
            T in_ = abs(x - low);
            T extra = m_fmod(in_, span);
            T flips = floor(in_ / span);
            bool even = floor(flips * T(0.5)) * T(2) == flips;
            x = even ? extra + low : span - extra + low;
            mult *= even ? sign : -sign;
        }
    }
    if (p.padding != 0)  // clip into [0, size - 1] (border, and after reflection)
    {
        if (x <= T(0))
        {
            x = T(0);
            mult = T(0);
        }
        else if (x >= T(size - 1))
        {
            x = T(size - 1);
            mult = T(0);
        }
    }
    if (!(abs(x) < T(1073741824)))  // (outside int range, or NaN: outside the input)
        x = T(-100);
    return x;
}

uint dim_size(uint d) { return d == 0 ? p.isz.x : (d == 1 ? p.isz.y : p.isz.z); }

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint plane = p.isz.x * p.isz.y * p.isz.z;
    uint corners = 1u << p.k;
    for (uint t = gid.x * 64 + tid.x; t < p.n * p.p; t += p.ngroups * 64)
    {
        uint b = t / p.p, pt = t - b * p.p;
        T x[3], mult[3], w0[3], w1[3];
        int i0[3];
        for (uint d = 0; d < 3; d++)
        {
            x[d] = T(0);
            mult[d] = T(0);
            i0[d] = 0;
            w0[d] = T(1);
            w1[d] = T(0);
            if (d < p.k)
            {
                x[d] = source(p.grid[(b * p.p + pt) * p.k + d], dim_size(d), mult[d]);
                if (MODE == 1)
                {
                    i0[d] = int(m_round_even(x[d]));
                }
                else
                {
                    T f = floor(x[d]);
                    i0[d] = int(f);
                    w0[d] = f + T(1) - x[d];
                    w1[d] = x[d] - f;
                }
            }
        }
        uint ncorners = MODE == 1 ? 1u : corners;
        // corner q: bit d chooses i0 or i0 + 1 along dim d
        uint off[8];
        bool ok[8];
        T wq[8];
        T dots[8];
        for (uint q = 0; q < 8; q++)
        {
            off[q] = 0;
            ok[q] = false;
            wq[q] = T(0);
            dots[q] = T(0);
        }
        for (uint q = 0; q < ncorners; q++)
        {
            bool valid = true;
            uint offset = 0, stride = 1;
            T weight = T(1);
            for (uint d = 0; d < p.k; d++)
            {
                uint bit = (q >> d) & 1u;
                int i = i0[d] + int(bit);
                valid = valid && i >= 0 && i < int(dim_size(d));
                offset += uint(max(i, 0)) * stride;
                stride *= dim_size(d);
                weight *= bit != 0 ? w1[d] : w0[d];
            }
            off[q] = offset;
            ok[q] = valid;
            wq[q] = weight;
            if (OP == GRID_BACKWARD)
            {
                p.idx[(b * p.p + pt) * corners + q] = valid ? int64_t(offset) : int64_t(-1);
                p.wgt[(b * p.p + pt) * corners + q] = valid ? weight : T(0);
            }
        }
        if (OP == GRID_BACKWARD)
        {
            for (uint q = ncorners; q < corners; q++)  // (nearest: one corner; the rest unused)
            {
                p.idx[(b * p.p + pt) * corners + q] = int64_t(-1);
                p.wgt[(b * p.p + pt) * corners + q] = T(0);
            }
        }
        if (OP == GRID_FORWARD)
        {
            for (uint ch = 0; ch < p.c; ch++)
            {
                uint base = (b * p.c + ch) * plane;
                T sum = T(0);
                for (uint q = 0; q < ncorners; q++)
                    if (ok[q])
                        sum += p.input[base + off[q]] * wq[q];
                p.out[(b * p.c + ch) * p.p + pt] = sum;
            }
            continue;
        }
        if (p.want_grid != 0 && MODE == 0)
        {
            for (uint ch = 0; ch < p.c; ch++)
            {
                uint base = (b * p.c + ch) * plane;
                T g = p.gout[(b * p.c + ch) * p.p + pt];
                for (uint q = 0; q < corners; q++)
                    if (ok[q])
                        dots[q] += p.input[base + off[q]] * g;
            }
        }
        if (p.want_grid != 0)
        {
            for (uint d = 0; d < p.k; d++)
            {
                T g = T(0);
                if (MODE == 0)
                {
                    for (uint q = 0; q < corners; q++)
                    {
                        T others = T(1);
                        for (uint e = 0; e < p.k; e++)
                            if (e != d)
                                others *= ((q >> e) & 1u) != 0 ? w1[e] : w0[e];
                        g += dots[q] * others * (((q >> d) & 1u) != 0 ? T(1) : T(-1));
                    }
                }
                p.ggrid[(b * p.p + pt) * p.k + d] = g * mult[d];
            }
        }
    }
}
