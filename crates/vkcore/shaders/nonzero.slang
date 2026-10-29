// Coordinates of the nonzero elements of a contiguous bool tensor (shape
// `size`, up to 8 dims, row-major), given csum = its inclusive running count
// (int64): element i, if set, writes its coordinates to row csum[i] - 1 of
// out (count x ndim, int64).

struct Params
{
    uint8_t* mask;
    int64_t* csum;
    int64_t* out;
    uint n;
    uint ndim;
    uint stride;
    uint _pad;
    uint4 size_lo;  // dims 0-3
    uint4 size_hi;  // dims 4-7
};

[[vk::push_constant]] ConstantBuffer<Params> p;

uint dim_size(uint d)
{
    return d < 4 ? p.size_lo[d] : p.size_hi[d - 4];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        if (p.mask[i] == uint8_t(0))
            continue;
        int64_t row = p.csum[i] - 1;
        uint rem = i;
        for (int d = int(p.ndim) - 1; d >= 0; d--)
        {
            uint s = dim_size(uint(d));
            p.out[row * p.ndim + d] = int64_t(rem % s);
            rem /= s;
        }
    }
}
