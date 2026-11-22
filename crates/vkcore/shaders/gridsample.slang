// axis T: f32=float f64=double
//
// grid_sampler_2d / 3d, bilinear (MODE 0), nearest (MODE 1) and bicubic
// (MODE 2, below), with PyTorch's CPU arithmetic: the grid coordinate is
// unnormalized
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
//
// Bicubic (MODE 2), as PyTorch 2.15's CPU kernels (GridSampler.cpp's
// resolve_cubic_taps): per dim, the Keys coefficients (A = -0.75) of the
// unnormalized coordinate's fractional part, and four taps around its
// floor, each resolved by the padding (border: clipped; reflection:
// reflected, then clipped); a tap that is negative, not finite or past the
// input is dropped (it reads nothing). The coordinate itself is neither
// reflected nor clipped. The 4^k taps are the backward's corners (3-D: z,
// y, x; 2-D: x, y; the last fastest, the CPU's loop order). Sums in the
// CPU's order: 2-D, four values along x per row, then the rows along y;
// 3-D, the 64 terms sample * (z y) * x in turn. grad_grid: the CPU's sums
// of sample * gout times the coefficients' derivatives, times the
// unnormalization's scale.
//
// MODE 3 is upsample_bicubic2d (UpSample.h, UpSampleBicubic2d.cpp) on the
// same taps: the grid holds each output's source coordinates (computed on
// the host with the CPU's arithmetic), the tap index is floor(x) capped at
// size - 1 with its fraction clamped to [0, 1], and the taps are clamped
// into the input; the 2-D sums are the CPU's cubic_interp1d along x, then
// y. (The backward's weights only: no grid gradient.)

#include "_fmath.slang"
#include "opcodes.slang"
#include "_wide.slang"

struct Params
{
    T* input;      // 0   [N, C, D, H, W] contiguous (D = 1 for 2-D; bicubic: strides ist, sc, sn)
    T* grid;       // 8   [N, P, k]
    T* out;        // 16  [N, C, P] (forward)
    T* gout;       // 24  [N, C, P] (backward)
    T* ggrid;      // 32  [N, P, k] (backward, when want_grid)
    i64w* idx;     // 40  [N, P, 2^k or 4^k] (backward; int64 as words: _wide.slang)
    T* wgt;        // 48  [N, P, 2^k or 4^k] (backward)
    uint n;        // 56
    uint c;        // 60
    uint p;        // 64  output points per batch element
    uint k;        // 68  2 or 3
    uint padding;  // 72  0 zeros, 1 border, 2 reflection
    uint align;    // 76
    uint ngroups;  // 80
    uint want_grid;  // 84  backward: write ggrid
    int sc;        // 88  bicubic: input's element stride per channel
    int sn;        // 92  and per batch element
    uint4 isz;     // 96  input W, H, D (by grid coordinate) and 0
    int4 ist;      // 112 bicubic: input's element strides along W, H, D and 0
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
            // (divided correctly rounded: the GPU's division may round 455 / 455
            // below 1, a reflection off by a whole span)
            T flips = floor(m_div(in_, span));
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

// ---------------------------------------------------------------- bicubic (MODE 2)

// The unnormalized coordinate and its derivative: align_corners
// (g + 1) / 2 (size - 1), else ((g + 1) size - 1) / 2 with the product and
// the difference fused, as the CPU's compiler does.
T cubic_coord(T g, uint size, out T mult)
{
    if (p.align != 0)
    {
        mult = T(size - 1) / T(2);
        return (g + T(1)) / T(2) * T(size - 1);
    }
    mult = T(size) / T(2);
    return fma(g + T(1), T(size) / T(2), T(-0.5));
}

// UpSample.h's get_cubic_upsample_coefficients (A = -0.75), operation for
// operation.
void cubic_coeffs(T t, out T c[4])
{
    T x0 = t + T(1), x1 = t, x2 = T(1) - t, x3 = x2 + T(1);
    c[0] = ((T(-0.75) * x0 - T(-3.75)) * x0 + T(-6)) * x0 - T(-3);
    c[1] = (T(1.25) * x1 - T(2.25)) * x1 * x1 + T(1);
    c[2] = (T(1.25) * x2 - T(2.25)) * x2 * x2 + T(1);
    c[3] = ((T(-0.75) * x3 - T(-3.75)) * x3 + T(-6)) * x3 - T(-3);
}

// GridSampler.h's get_cubic_coefficients_grad.
void cubic_coeffs_grad(T t, out T c[4])
{
    T x = T(-1) - t;
    c[0] = (T(2.25) * x - T(-7.5)) * x - T(-6);
    x = -t;
    c[1] = (T(-3.75) * x - T(4.5)) * x;
    x = T(1) - t;
    c[2] = (T(3.75) * x - T(4.5)) * x;
    x = T(2) - t;
    c[3] = (T(-2.25) * x - T(-7.5)) * x + T(-6);
}

// GridSampler.h's clip_coordinates: min(size - 1, max(x, 0)), NaN -> size - 1.
T clip_coord(T x, uint size)
{
    return isnan(x) ? T(size - 1) : clamp(x, T(0), T(size - 1));
}

// The input index a tap at `tap` reads along a dim of `size` after the
// padding (compute_coordinates_sized), or -1 where it is dropped.
int cubic_tap(T tap, uint size)
{
    if (p.padding == 1)
    {
        tap = clip_coord(tap, size);
    }
    else if (p.padding == 2)
    {
        T low = p.align != 0 ? T(0) : T(-0.5);
        T span = p.align != 0 ? T(size - 1) : T(size);
        if (span == T(0))
        {
            tap = T(0);
        }
        else
        {
            T dist = abs(tap - low);
            T extra = m_fmod(dist, span);
            // (divided correctly rounded, as in source())
            bool odd = m_fmod(floor(m_div(dist, span)), T(2)) != T(0);
            tap = odd ? span - extra + low : extra + low;
        }
        tap = clip_coord(tap, size);
    }
    // (the bound compared as integers: a float may not hold the size exactly;
    // the host keeps sizes below 2^31)
    if (!(tap >= T(0) && tap < T(2147483648.0)))
        return -1;
    int i = int(tap);
    return i < int(size) ? i : -1;
}

// Offset of the tap at indices (tx, ty, tz) in a contiguous input plane
// (grad_input's), or -1 where one is dropped.
int tap_offset(int tx, int ty, int tz)
{
    if (tx < 0 || ty < 0 || tz < 0)
        return -1;
    return (tz * int(p.isz.y) + ty) * int(p.isz.x) + tx;
}

// The input's value at that tap, through its strides (0 where dropped).
T tap_value(uint base, int tx, int ty, int tz)
{
    if (tx < 0 || ty < 0 || tz < 0)
        return T(0);
    return p.input[int(base) + tz * p.ist.z + ty * p.ist.y + tx * p.ist.x];
}

void bicubic(uint b, uint pt)
{
    uint k = p.k;
    T cf[3][4], cg[3][4], mult[3];
    int tap[3][4];
    for (uint d = 0; d < 3; d++)
    {
        mult[d] = T(0);
        for (uint i = 0; i < 4; i++)
        {
            cf[d][i] = T(0);
            cg[d][i] = T(0);
            tap[d][i] = i == 0 ? 0 : -1;  // (2-D: z is one tap, at 0)
        }
        if (d < k && MODE == 3)
        {
            // upsample_bicubic2d: the grid holds source coordinates; UpSample.h's
            // guard_index_and_lambda, and taps clamped into the input
            uint size = dim_size(d);
            T x = p.grid[(b * p.p + pt) * k + d];
            int base = min(int(floor(x)), int(size) - 1);
            T c[4];
            cubic_coeffs(clamp(x - T(base), T(0), T(1)), c);
            for (uint i = 0; i < 4; i++)
            {
                cf[d][i] = c[i];
                tap[d][i] = clamp(base + int(i) - 1, 0, int(size) - 1);
            }
        }
        else if (d < k)
        {
            uint size = dim_size(d);
            T x = cubic_coord(p.grid[(b * p.p + pt) * k + d], size, mult[d]);
            T fl = floor(x);
            T c[4], g[4];
            cubic_coeffs(x - fl, c);
            cubic_coeffs_grad(x - fl, g);
            for (uint i = 0; i < 4; i++)
            {
                cf[d][i] = c[i];
                cg[d][i] = g[i];
                tap[d][i] = cubic_tap(fl + T(int(i) - 1), size);
            }
        }
    }
    uint ncorner = k == 3 ? 64u : 16u;
    uint row0 = (b * p.p + pt) * ncorner;
    if (OP == GRID_BACKWARD)
    {
        for (uint q = 0; q < ncorner; q++)
        {
            uint i = k == 3 ? q % 4 : q / 4, j = k == 3 ? (q / 4) % 4 : q % 4, l = k == 3 ? q / 16 : 0;
            int off = tap_offset(tap[0][i], tap[1][j], tap[2][l]);
            p.idx[row0 + q] = i64w_from(off);
            p.wgt[row0 + q] = off < 0 ? T(0) : (k == 3 ? cf[0][i] * cf[1][j] * cf[2][l] : cf[0][i] * cf[1][j]);
        }
        if (p.want_grid == 0)
            return;
        T gx = T(0), gy = T(0), gz = T(0);
        for (uint ch = 0; ch < p.c; ch++)
        {
            uint base = uint(int(b) * p.sn + int(ch) * p.sc);
            T go = p.gout[(b * p.c + ch) * p.p + pt];
            if (k == 3)
            {
                for (uint l = 0; l < 4; l++)
                {
                    for (uint j = 0; j < 4; j++)
                    {
                        for (uint i = 0; i < 4; i++)
                        {
                            T v = tap_value(base, tap[0][i], tap[1][j], tap[2][l]);
                            gx -= v * cg[0][i] * cf[1][j] * cf[2][l] * go;
                            gy -= v * cf[0][i] * cg[1][j] * cf[2][l] * go;
                            gz -= v * cf[0][i] * cf[1][j] * cg[2][l] * go;
                        }
                    }
                }
            }
            else
            {
                for (uint i = 0; i < 4; i++)
                {
                    for (uint j = 0; j < 4; j++)
                    {
                        T v = tap_value(base, tap[0][i], tap[1][j], 0);
                        gx = gx - v * go * cg[0][i] * cf[1][j];
                        gy = gy - v * go * cg[1][j] * cf[0][i];
                    }
                }
            }
        }
        uint gbase = (b * p.p + pt) * k;
        p.ggrid[gbase] = mult[0] * gx;
        p.ggrid[gbase + 1] = mult[1] * gy;
        if (k == 3)
            p.ggrid[gbase + 2] = mult[2] * gz;
        return;
    }
    for (uint ch = 0; ch < p.c; ch++)
    {
        uint base = uint(int(b) * p.sn + int(ch) * p.sc);
        T value = T(0);
        if (k == 3)
        {
            for (uint l = 0; l < 4; l++)
            {
                for (uint j = 0; j < 4; j++)
                {
                    T wzy = cf[2][l] * cf[1][j];
                    for (uint i = 0; i < 4; i++)
                    {
                        T v = tap_value(base, tap[0][i], tap[1][j], tap[2][l]);
                        value += v * wzy * cf[0][i];
                    }
                }
            }
        }
        else
        {
            for (uint j = 0; j < 4; j++)
            {
                T r[4];
                for (uint i = 0; i < 4; i++)
                {
                    r[i] = tap_value(base, tap[0][i], tap[1][j], 0) * cf[0][i];
                }
                T row = ((r[0] + r[1]) + r[2]) + r[3];
                value = j == 0 ? row * cf[1][j] : value + row * cf[1][j];
            }
        }
        p.out[(b * p.c + ch) * p.p + pt] = value;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint plane = p.isz.x * p.isz.y * p.isz.z;
    uint corners = 1u << p.k;
    for (uint t = gid.x * 64 + tid.x; t < p.n * p.p; t += p.ngroups * 64)
    {
        uint b = t / p.p, pt = t - b * p.p;
        if (MODE >= 2)
        {
            bicubic(b, pt);
            continue;
        }
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
                p.idx[(b * p.p + pt) * corners + q] = i64w_from(valid ? int(offset) : -1);
                p.wgt[(b * p.p + pt) * corners + q] = valid ? weight : T(0);
            }
        }
        if (OP == GRID_BACKWARD)
        {
            for (uint q = ncorners; q < corners; q++)  // (nearest: one corner; the rest unused)
            {
                p.idx[(b * p.p + pt) * corners + q] = i64w_from(-1);
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
