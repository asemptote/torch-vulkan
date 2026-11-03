// axis T: f32=float f64=double i32=int i64=I64 b8=uint8_t c64=float2 c128=double2
//
// tril / triu over a batch of matrices [batch, rows, cols] (any strides):
// out[b, i, j] = in[b, i, j] where kept, else 0. Kept: j - i <= diag (tril),
// j - i >= diag (triu, UPPER = 1). In place when in == out.

#include "_int64.slang"  // (the i64 variant's T: _int64.slang)
struct Params
{
    T* in;          // 0
    T* out;         // 8
    uint batch;     // 16
    uint rows;      // 20
    uint cols;      // 24
    int diag;       // 28
    int in_sb, in_sr, in_sc;     // 32
    int out_sb, out_sr, out_sc;  // 44
    uint ngroups;   // 56
    uint _pad;      // 60
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int UPPER = 0;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint per = p.rows * p.cols;
    uint total = p.batch * per;
    for (uint t = gid.x * 256 + tid.x; t < total; t += p.ngroups * 256)
    {
        int b = int(t / per), i = int((t % per) / p.cols), j = int(t % p.cols);
        bool keep = UPPER != 0 ? (j - i >= p.diag) : (j - i <= p.diag);
        T v = keep ? p.in[b * p.in_sb + i * p.in_sr + j * p.in_sc] : T(0);
        p.out[b * p.out_sb + i * p.out_sr + j * p.out_sc] = v;
    }
}
