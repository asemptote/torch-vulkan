// axis T: f32=float f64=double
//
// torchvision::nms, in two launches over the boxes sorted by score
// (descending, stably: csrc/vision.cpp):
//   NMS_MASK: bit j % 32 of mask[i, j / 32] is set where box j (j > i, in
//             score order) overlaps box i by more than the threshold, as
//             torchvision's CPU kernel decides it (its IoU's operations in
//             T, in its order, the division correctly rounded); a thread
//             per (i, word)
//   NMS_SCAN: one workgroup walks the boxes in score order: a box not yet
//             suppressed is kept, and suppresses those its mask row sets
//             (torchvision's CUDA kernel does this walk on the host)
// The kept boxes' original indices go to keep[0, count), int64 in score
// order, as torchvision returns them.

#include "opcodes.slang"
#include "_slot.slang"
#include "_wide.slang"
#include "_fmath.slang"  // (m_div: OpFDiv may be 2.5 ulp off, the CPU's is not)

struct Params
{
    T* boxes;        // 0   [n, 4] (x1, y1, x2, y2) in score order, contiguous
    uint* mask;      // 8   [n, words]
    i64w* order;     // 16  [n]: each sorted box's original index (int64 as words)
    i64w* keep;      // 24  [n] out (NMS_SCAN)
    uint* count;     // 32  [1] out (NMS_SCAN)
    slot64 thr;      // 40  T: a box is suppressed where IoU > thr
    uint n;          // 48
    uint words;      // 52  ceil(n / 32)
    uint stride;     // 56  (NMS_MASK: grid stride, threads)
    uint _pad;       // 60
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int OP = NMS_MASK;

T threshold()
{
#if defined(T_f64)
    return slot_double(p.thr);
#else
    return slot_float(p.thr);
#endif
}

// std::max and std::min as the CPU kernel calls them (NaN: the first argument).
T cmax(T a, T b) { return a < b ? b : a; }
T cmin(T a, T b) { return b < a ? b : a; }

// NMS_SCAN's suppressed boxes, a bit each (n <= 32 * MAX_WORDS: the host checks).
#define MAX_WORDS 2048
groupshared uint removed[MAX_WORDS];
groupshared uint kept;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 tid: SV_GroupThreadID)
{
    if (OP == NMS_MASK)
    {
        uint total = p.n * p.words;
        for (uint t = id.x; t < total; t += p.stride)
        {
            uint i = t / p.words, w = t % p.words;
            uint first = max(w * 32u, i + 1u), last = min(w * 32u + 32u, p.n);
            uint bits = 0u;
            if (first < last)
            {
                T ix1 = p.boxes[4 * i], iy1 = p.boxes[4 * i + 1], ix2 = p.boxes[4 * i + 2], iy2 = p.boxes[4 * i + 3];
                T iarea = (ix2 - ix1) * (iy2 - iy1);
                T thr = threshold();
                for (uint j = first; j < last; j++)
                {
                    T x1 = p.boxes[4 * j], y1 = p.boxes[4 * j + 1], x2 = p.boxes[4 * j + 2], y2 = p.boxes[4 * j + 3];
                    T area = (x2 - x1) * (y2 - y1);
                    T xx1 = cmax(ix1, x1), yy1 = cmax(iy1, y1), xx2 = cmin(ix2, x2), yy2 = cmin(iy2, y2);
                    T w_ = cmax(T(0), xx2 - xx1), h = cmax(T(0), yy2 - yy1);
                    T inter = w_ * h;
                    T ovr = m_div(inter, iarea + area - inter);
                    if (ovr > thr)
                        bits |= 1u << (j - w * 32u);
                }
            }
            p.mask[t] = bits;
        }
        return;
    }
    // NMS_SCAN: one workgroup
    for (uint w = tid.x; w < p.words; w += 256u)
        removed[w] = 0u;
    if (tid.x == 0u)
        kept = 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint i = 0; i < p.n; i++)
    {
        // (uniform: every thread reads the same word)
        if (((removed[i / 32u] >> (i % 32u)) & 1u) != 0u)
            continue;
        if (tid.x == 0u)
        {
            p.keep[kept] = p.order[i];
            kept += 1u;
        }
        uint row = i * p.words;
        for (uint w = i / 32u + tid.x; w < p.words; w += 256u)
            removed[w] |= p.mask[row + w];
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid.x == 0u)
        *p.count = kept;
}
