// axis T: f32=float f64=double
//
// Row gather / scatter for 2-D strided matrices (used to apply pivoting):
//   MODE 0 (gather):  out[i, c] = in[idx[i], c]
//   MODE 1 (scatter): out[idx[i], c] = in[i, c]
// for i < nrows, c < ncols.

#include "opcodes.slang"

struct Params
{
    T* in;            // 0
    T* out;           // 8
    int64_t* idx;     // 16
    uint nrows;       // 24
    uint ncols;       // 28
    int in_rs;        // 32
    int in_cs;        // 36
    int out_rs;       // 40
    int out_cs;       // 44
    uint ngroups;     // 48
    uint _pad;        // 52
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int MODE = 0;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint total = p.nrows * p.ncols;
    for (uint t = gid.x * 256 + tid.x; t < total; t += p.ngroups * 256)
    {
        uint i = t / p.ncols, c = t % p.ncols;
        int r = int(p.idx[min(i, p.nrows - 1)]);  // (i < nrows; clamped for loads past the loop's end)
        if (MODE == ROWS_GATHER)
            p.out[int(i) * p.out_rs + int(c) * p.out_cs] = p.in[r * p.in_rs + int(c) * p.in_cs];
        else
            p.out[r * p.out_rs + int(c) * p.out_cs] = p.in[int(i) * p.in_rs + int(c) * p.in_cs];
    }
}
