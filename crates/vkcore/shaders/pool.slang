// axis T: f32=float f64=double
//
// Max pooling over up to 3 spatial dims (2-D pooling has depth 1), on
// contiguous [planes, D, H, W] tensors, with the CPU kernel's semantics:
//   POOL_MAX_FORWARD:  out[p, o] = the largest input in o's window, idx[p, o]
//                      its flat index within the plane: NaN wins (each NaN
//                      replaces the maximum, so the last NaN's index), ties
//                      keep the first; the index starts at the window's first
//                      element inside the input (an all -inf window).
//   POOL_MAX_BACKWARD: gin[p, i] = the sum of gout[p, o] over the outputs o
//                      with idx[p, o] = i, in output order (as the CPU's
//                      scatter adds them): a gather over the outputs whose
//                      windows can cover i, so no atomics and deterministic.
//   POOL_AVG_FORWARD:  out[p, o] = the sum over o's window / its divisor, as
//                      the CPU's avg_pool: the window runs from o * s - pad
//                      to min(that + k, n + pad), the divisor is
//                      divisor_override, else its size (count_include_pad),
//                      else the size of its part inside the input; truncated
//                      for integer inputs (computed in float64).
//   POOL_AVG_BACKWARD: gin[p, i] = the sum of gout[p, o] / divisor(o) over the
//                      outputs whose windows cover i, in output order.
// Average pooling has no dilation; its settings take the dilation slots:
// dl = (divisor_override or 0, count_include_pad, truncate).

#include "opcodes.slang"

struct Params
{
    T* a;          // 0   forward: input; backward: grad_output
    T* out;        // 8   forward: output; backward: grad_input
    int64_t* idx;  // 16  [planes, output size]
    uint total;    // 24  elements written: planes * output size (forward) or input size (backward)
    uint ngroups;  // 28
    uint4 isz;     // 32  input D, H, W (and 0)
    uint4 osz;     // 48  output D, H, W
    uint4 ks;      // 64  kernel size
    uint4 st;      // 80  stride
    int4 pd;       // 96  padding
    uint4 dl;      // 112 dilation
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = POOL_MAX_FORWARD;

T neg_inf()
{
#if defined(T_f64)
    return bit_cast<double>(0xfff0000000000000ull);
#else
    return bit_cast<float>(0xff800000u);
#endif
}

// First output (inclusive) and end (exclusive) along one dim whose window,
// starting at o * s - pad, can reach input position i.
uint2 covering(uint i, uint n_out, uint k, uint s, int pad, uint d)
{
    int last = int(i) + pad;  // o * s <= i + pad
    if (last < 0)
        return uint2(0, 0);
    int first = last - int((k - 1) * d);  // o * s >= i + pad - (k - 1) d
    uint lo = first <= 0 ? 0u : (uint(first) + s - 1) / s;
    uint hi = min(n_out, uint(last) / s + 1);
    return uint2(lo, max(lo, hi));
}

// One dim of an average-pooling window: (start, end) inside the input and
// its full size (padding included).
int3 avg_window(uint o, uint k, uint s, int pad, uint n)
{
    int start = int(o * s) - pad;
    int end = min(start + int(k), int(n) + pad);
    return int3(max(start, 0), min(end, int(n)), end - start);
}

T avg_divisor(int3 z, int3 y, int3 x)
{
    if (p.dl.x != 0)
        return T(int(p.dl.x));  // (the override may be negative)
    if (p.dl.y != 0)
        return T(z.z * y.z * x.z);
    return T((z.y - z.x) * (y.y - y.x) * (x.y - x.x));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint ov = p.osz.x * p.osz.y * p.osz.z;
    uint iv = p.isz.x * p.isz.y * p.isz.z;
    for (uint t = gid.x * 256 + tid.x; t < p.total; t += p.ngroups * 256)
    {
        if (OP == POOL_AVG_FORWARD)
        {
            uint plane = t / ov, r = t - plane * ov;
            uint ox = r % p.osz.z;
            r /= p.osz.z;
            uint oy = r % p.osz.y, oz = r / p.osz.y;
            int3 wz = avg_window(oz, p.ks.x, p.st.x, p.pd.x, p.isz.x);
            int3 wy = avg_window(oy, p.ks.y, p.st.y, p.pd.y, p.isz.y);
            int3 wx = avg_window(ox, p.ks.z, p.st.z, p.pd.z, p.isz.z);
            T sum = T(0);
            for (int iz = wz.x; iz < wz.y; iz++)
                for (int iy = wy.x; iy < wy.y; iy++)
                    for (int ix = wx.x; ix < wx.y; ix++)
                        sum += p.a[plane * iv + (uint(iz) * p.isz.y + uint(iy)) * p.isz.z + uint(ix)];
            T q = sum / avg_divisor(wz, wy, wx);
            p.out[t] = p.dl.z != 0 ? trunc(q) : q;
        }
        else if (OP == POOL_AVG_BACKWARD)
        {
            uint plane = t / iv, r = t - plane * iv;
            uint ix = r % p.isz.z;
            r /= p.isz.z;
            uint iy = r % p.isz.y, iz = r / p.isz.y;
            uint2 rz = covering(iz, p.osz.x, p.ks.x, p.st.x, p.pd.x, 1);
            uint2 ry = covering(iy, p.osz.y, p.ks.y, p.st.y, p.pd.y, 1);
            uint2 rx = covering(ix, p.osz.z, p.ks.z, p.st.z, p.pd.z, 1);
            T sum = T(0);
            for (uint oz = rz.x; oz < rz.y; oz++)
            {
                int3 wz = avg_window(oz, p.ks.x, p.st.x, p.pd.x, p.isz.x);
                for (uint oy = ry.x; oy < ry.y; oy++)
                {
                    int3 wy = avg_window(oy, p.ks.y, p.st.y, p.pd.y, p.isz.y);
                    for (uint ox = rx.x; ox < rx.y; ox++)
                    {
                        int3 wx = avg_window(ox, p.ks.z, p.st.z, p.pd.z, p.isz.z);
                        sum += p.a[plane * ov + (oz * p.osz.y + oy) * p.osz.z + ox] / avg_divisor(wz, wy, wx);
                    }
                }
            }
            p.out[t] = sum;
        }
        else if (OP == POOL_MAX_FORWARD)
        {
            uint plane = t / ov, r = t - plane * ov;
            uint ox = r % p.osz.z;
            r /= p.osz.z;
            uint oy = r % p.osz.y, oz = r / p.osz.y;
            int z0 = int(oz * p.st.x) - p.pd.x, y0 = int(oy * p.st.y) - p.pd.y, x0 = int(ox * p.st.z) - p.pd.z;
            T best = neg_inf();
            int64_t at = -1;
            for (uint kz = 0; kz < p.ks.x; kz++)
            {
                int iz = z0 + int(kz * p.dl.x);
                if (iz < 0 || iz >= int(p.isz.x))
                    continue;
                for (uint ky = 0; ky < p.ks.y; ky++)
                {
                    int iy = y0 + int(ky * p.dl.y);
                    if (iy < 0 || iy >= int(p.isz.y))
                        continue;
                    for (uint kx = 0; kx < p.ks.z; kx++)
                    {
                        int ix = x0 + int(kx * p.dl.z);
                        if (ix < 0 || ix >= int(p.isz.z))
                            continue;
                        uint index = (uint(iz) * p.isz.y + uint(iy)) * p.isz.z + uint(ix);
                        T v = p.a[plane * iv + index];
                        if (at < 0)
                            at = int64_t(index);
                        if (v > best || isnan(v))
                        {
                            best = v;
                            at = int64_t(index);
                        }
                    }
                }
            }
            p.out[t] = best;
            p.idx[t] = at;
        }
        else
        {
            uint plane = t / iv, r = t - plane * iv;
            uint ix = r % p.isz.z;
            r /= p.isz.z;
            uint iy = r % p.isz.y, iz = r / p.isz.y;
            int64_t index = int64_t(t - plane * iv);
            uint2 rz = covering(iz, p.osz.x, p.ks.x, p.st.x, p.pd.x, p.dl.x);
            uint2 ry = covering(iy, p.osz.y, p.ks.y, p.st.y, p.pd.y, p.dl.y);
            uint2 rx = covering(ix, p.osz.z, p.ks.z, p.st.z, p.pd.z, p.dl.z);
            T sum = T(0);
            for (uint oz = rz.x; oz < rz.y; oz++)
                for (uint oy = ry.x; oy < ry.y; oy++)
                    for (uint ox = rx.x; ox < rx.y; ox++)
                    {
                        uint o = plane * ov + (oz * p.osz.y + oy) * p.osz.z + ox;
                        if (p.idx[o] == index)
                            sum += p.a[o];
                    }
            p.out[t] = sum;
        }
    }
}
