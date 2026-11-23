// axis T: f32=float f64=double
//
// torchvision::roi_align and its backward (_roi_align_backward), computed as
// torchvision's CPU kernels compute them: each region's geometry, sample
// positions and bilinear weights by the same operations in T, in the same
// order (divisions correctly rounded), so results agree with the CPU's to
// the last bit where its compiler does not reorder them.
//   ROI_FORWARD:  a thread per output (n, c, ph, pw): the mean of the
//                 bilinear samples of input's plane (batch of region n, c)
//                 over the bin's grid.
//   ROI_GRID:     a thread per region: its sampling grid (rows, columns),
//                 for the host's largest grid (adaptive sampling only).
//   ROI_PLAN:     backward, a thread per sample slot (n, ph, pw, iy, ix) of
//                 the largest grid, its four corners as entries e = 4 slot +
//                 corner: key[e] the input position (batch, y, x) as batch
//                 H W + y W + x, or SKIP (no sample there, out of the image,
//                 an invalid batch index), and weight[e].
//                 (and each region's count, for ROI_GATHER)
//   ROI_GATHER:   backward, after the host sorts the keys (stably): a thread
//                 per (sorted entry, channel), channels side by side (g and
//                 grad_input channels-last, so that they coalesce), that starts a run of equal
//                 keys adds the run's g[n, c, ph, pw] * weight / count in
//                 entry order, the CPU's order of additions into that input
//                 element, and writes the sum. grad_input is zeroed first;
//                 no atomics, deterministic.
// A region whose batch index is outside [0, batch) contributes nothing and
// sets the error word (the host raises it when it next synchronizes), where
// the CPU raises at once.

#include "opcodes.slang"
#include "_slot.slang"
#include "_fmath.slang"  // (m_div: OpFDiv may be 2.5 ulp off, the CPU's is not)

#define SKIP 0x7fffffff

struct Params
{
    T* input;        // 0   [batch, C, H, W] (forward); g [K, PH, PW, C] (ROI_GATHER), contiguous
    T* rois;         // 8   [K, 5] (batch index, x1, y1, x2, y2), contiguous; ROI_GATHER: counts [K]
    T* out;          // 16  [K, C, PH, PW] (forward); counts [K] (ROI_PLAN); grad_input [batch, H, W, C] (ROI_GATHER)
    int* key;        // 24  ROI_PLAN: [E]; ROI_GATHER: the same sorted; ROI_GRID: [K, 2] grids
    T* weight;       // 32  [E]
    uint* order;     // 40  [E] (ROI_GATHER): entries sorted by key
    uint* error;     // 48
    slot64 scale;    // 56  spatial_scale, as T
    uint batch;      // 64
    uint c;          // 68
    uint h;          // 72
    uint w;          // 76
    uint ph;         // 80
    uint pw;         // 84
    int sampling;    // 88  sampling_ratio
    uint aligned;    // 92
    uint total;      // 96  threads
    uint stride;     // 100 grid stride
    uint gh;         // 104 ROI_PLAN / ROI_GATHER: the largest grid's rows
    uint gw;         // 108 and columns
    uint k;          // 112 regions
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = ROI_FORWARD;

T scale()
{
#if defined(T_f64)
    return slot_double(p.scale);
#else
    return slot_float(p.scale);
#endif
}

// std::max as the CPU kernel calls it.
T cmax(T a, T b) { return a < b ? b : a; }

struct Region
{
    bool ok;        // batch index in [0, batch)
    uint batch;
    T start_h, start_w, bin_h, bin_w;
    int grid_h, grid_w;
    T count;        // forward: max(grid_h grid_w, 1); backward: grid_h grid_w
};

Region region(uint n, bool backward)
{
    Region r;
    T b = p.rois[5 * n];
    r.ok = b >= T(0) && b < T(p.batch);
    r.batch = r.ok ? uint(b) : 0u;
    T offset = p.aligned != 0u ? T(0.5) : T(0);
    T s = scale();
    r.start_w = p.rois[5 * n + 1] * s - offset;
    r.start_h = p.rois[5 * n + 2] * s - offset;
    T end_w = p.rois[5 * n + 3] * s - offset;
    T end_h = p.rois[5 * n + 4] * s - offset;
    T width = end_w - r.start_w, height = end_h - r.start_h;
    if (p.aligned == 0u)
    {
        width = cmax(width, T(1));  // (malformed regions as 1 x 1)
        height = cmax(height, T(1));
    }
    r.bin_h = m_div(height, T(p.ph));
    r.bin_w = m_div(width, T(p.pw));
    r.grid_h = p.sampling > 0 ? p.sampling : int(ceil(m_div(height, T(p.ph))));
    r.grid_w = p.sampling > 0 ? p.sampling : int(ceil(m_div(width, T(p.pw))));
    int cells = r.grid_h * r.grid_w;
    r.count = backward ? T(cells) : T(max(cells, 1));
    return r;
}

// The sample position along one dim: start + cell bin + (i + 0.5) bin / grid.
T sample_at(T start, uint cell, T bin, int i, int grid)
{
    return start + T(cell) * bin + m_div(T(float(i) + 0.5f) * bin, T(grid));
}

// Bilinear corners and weights of (y, x) in an H x W plane, as
// torchvision's pre_calc_for_bilinear_interpolate and
// bilinear_interpolate_gradient; false where the sample is outside.
bool bilinear(T y, T x, out uint pos[4], out T wt[4])
{
    for (uint k = 0; k < 4u; k++)
    {
        pos[k] = 0u;
        wt[k] = T(0);
    }
    int H = int(p.h), W = int(p.w);
    if (y < T(-1) || y > T(H) || x < T(-1) || x > T(W))
        return false;
    if (y <= T(0))
        y = T(0);
    if (x <= T(0))
        x = T(0);
    int y_low = int(y), x_low = int(x), y_high, x_high;
    if (y_low >= H - 1)
    {
        y_high = y_low = H - 1;
        y = T(y_low);
    }
    else
        y_high = y_low + 1;
    if (x_low >= W - 1)
    {
        x_high = x_low = W - 1;
        x = T(x_low);
    }
    else
        x_high = x_low + 1;
    T ly = y - T(y_low), lx = x - T(x_low);
    T hy = T(1) - ly, hx = T(1) - lx;
    wt[0] = hy * hx;
    wt[1] = hy * lx;
    wt[2] = ly * hx;
    wt[3] = ly * lx;
    pos[0] = uint(y_low * W + x_low);
    pos[1] = uint(y_low * W + x_high);
    pos[2] = uint(y_high * W + x_low);
    pos[3] = uint(y_high * W + x_high);
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint plane = p.h * p.w;
    for (uint t = id.x; t < p.total; t += p.stride)
    {
        if (OP == ROI_FORWARD)
        {
            uint pw = t % p.pw, ph = (t / p.pw) % p.ph, c = (t / (p.pw * p.ph)) % p.c, n = t / (p.pw * p.ph * p.c);
            Region r = region(n, false);
            if (!r.ok)
            {
                *p.error = 1u;
                p.out[t] = T(0);
                continue;
            }
            uint base = (r.batch * p.c + c) * plane;
            T acc = T(0);
            for (int iy = 0; iy < r.grid_h; iy++)
            {
                T y = sample_at(r.start_h, ph, r.bin_h, iy, r.grid_h);
                for (int ix = 0; ix < r.grid_w; ix++)
                {
                    T x = sample_at(r.start_w, pw, r.bin_w, ix, r.grid_w);
                    uint pos[4];
                    T wt[4];
                    bilinear(y, x, pos, wt);  // (outside: weights 0 at position 0, as the CPU's)
                    acc = acc + (((wt[0] * p.input[base + pos[0]] + wt[1] * p.input[base + pos[1]]) +
                                  wt[2] * p.input[base + pos[2]]) +
                                 wt[3] * p.input[base + pos[3]]);
                }
            }
            p.out[t] = m_div(acc, r.count);
        }
        else if (OP == ROI_GRID)
        {
            Region r = region(t, true);
            p.key[2 * t] = r.ok ? r.grid_h : 0;
            p.key[2 * t + 1] = r.ok ? r.grid_w : 0;
        }
        else if (OP == ROI_PLAN)
        {
            uint ix = t % p.gw, iy = (t / p.gw) % p.gh, bin = t / (p.gw * p.gh);
            uint pw = bin % p.pw, ph = (bin / p.pw) % p.ph, n = bin / (p.pw * p.ph);
            Region r = region(n, true);
            if (ix == 0u && iy == 0u && pw == 0u && ph == 0u)
            {
                p.out[n] = r.count;  // (for ROI_GATHER)
                if (!r.ok)
                    *p.error = 1u;
            }
            uint pos[4];
            T wt[4];
            bool inside = false;
            if (r.ok && int(iy) < r.grid_h && int(ix) < r.grid_w)
            {
                T y = sample_at(r.start_h, ph, r.bin_h, int(iy), r.grid_h);
                T x = sample_at(r.start_w, pw, r.bin_w, int(ix), r.grid_w);
                inside = bilinear(y, x, pos, wt);
            }
            for (uint k = 0; k < 4u; k++)
            {
                p.key[4 * t + k] = inside ? int(r.batch * plane + pos[k]) : SKIP;
                p.weight[4 * t + k] = inside ? wt[k] : T(0);
            }
        }
        else  // ROI_GATHER: t = sorted entry C + c
        {
            uint entries = p.k * p.ph * p.pw * p.gh * p.gw * 4u;
            uint i = t / p.c, c = t % p.c;
            int key = p.key[i];
            if (key == SKIP || (i > 0u && p.key[i - 1u] == key))
                continue;
            T acc = T(0);
            for (uint j = i; j < entries && p.key[j] == key; j++)
            {
                uint e = p.order[j];
                uint bin = e / (4u * p.gh * p.gw), n = bin / (p.ph * p.pw);
                T g = p.input[bin * p.c + c];  // (bin = (n PH + ph) PW + pw)
                acc += m_div(g * p.weight[e], p.rois[n]);
            }
            p.out[uint(key) * p.c + c] = acc;
        }
    }
}
