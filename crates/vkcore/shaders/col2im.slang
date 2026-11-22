// axis T: f32=float f64=double
//
// col2im (fold): out[n, c, y, x] = the sum of the columns' values that
// im2col would have read from (y, x),
//   cols[n, (c KH + kh) KW + kw, oh OW + ow] for y = oh sh - ph + kh dh,
//                                               x = ow sw - pw + kw dw,
// gathered per output in (kh, kw) order from zero: the order in which the
// CPU's col2im loops add them (over c KH KW + kh KW + kw, then the blocks),
// so the sums are the CPU's. One invocation per output element; no
// scatter, no sort.

struct Params
{
    T* cols;       // 0   [N, C KH KW, OH OW] contiguous
    T* out;        // 8   [N, C, H, W] contiguous
    uint total;    // 16  N C H W
    uint gstride;  // 20  grid stride
    uint H;        // 24
    uint W;        // 28
    uint OH;       // 32  blocks along h
    uint OW;       // 36  and w
    uint KH;       // 40
    uint KW;       // 44
    uint sh;       // 48  stride
    uint sw;       // 52
    uint dh;       // 56  dilation
    uint dw;       // 60
    uint ph;       // 64  padding
    uint pw;       // 68
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint L = p.OH * p.OW, K = p.KH * p.KW;
    for (uint i = id.x; i < p.total; i += p.gstride)
    {
        uint x = i % p.W, r = i / p.W;
        uint y = r % p.H;
        uint nc = r / p.H;  // n C + c: the column rows start at nc K
        T sum = T(0);
        for (uint kh = 0; kh < p.KH; kh++)
        {
            int ty = int(y + p.ph) - int(kh * p.dh);  // oh sh
            if (ty < 0 || uint(ty) % p.sh != 0 || uint(ty) / p.sh >= p.OH)
                continue;
            uint oh = uint(ty) / p.sh;
            for (uint kw = 0; kw < p.KW; kw++)
            {
                int tx = int(x + p.pw) - int(kw * p.dw);
                if (tx < 0 || uint(tx) % p.sw != 0 || uint(tx) / p.sw >= p.OW)
                    continue;
                sum += p.cols[(nc * K + kh * p.KW + kw) * L + oh * p.OW + uint(tx) / p.sw];
            }
        }
        p.out[i] = sum;
    }
}
