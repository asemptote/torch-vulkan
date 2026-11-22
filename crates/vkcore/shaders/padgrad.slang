// axis T: f32=float f64=double
//
// reflection_pad{1,2,3}d_backward (MODE 0) and replication_pad (MODE 1):
// grad_input[p, i] = the sum of grad_output[p, o] over the output positions
// o whose padded read is input position i, gathered per input element (no
// scatter): along each dim, the outputs that map to i, in ascending order,
// so the sum goes in the CPU's order (its loop over the outputs adds each
// one in turn into its input position) and is the CPU's. 1-3 dims as D, H,
// W (size 1, no padding, for fewer); p runs over the leading dims. The
// maps are the CPU's (ReflectionPad.cpp, ReplicationPadding.cpp), with
// i_start - o_start = -pad_lo for negative padding too:
//   reflection:  o < l: 2 l - o;  o < n + l: o;  else 2 (n + l - 1) - o;  minus l
//   replication: clamp(o, l, n + l - 1) - l
// so input i is read by o in {l - i, l + i, 2 n + l - 2 - i} (reflection)
// or by the range [i == 0 ? 0 : l + i, i == n - 1 ? m - 1 : l + i]
// (replication), m the output size, each checked against the map.

struct Params
{
    T* gout;       // 0   [planes, Do, Ho, Wo] contiguous
    T* gin;        // 8   [planes, Di, Hi, Wi] contiguous
    uint total;    // 16  grad_input's elements
    uint gstride;  // 20  grid stride
    uint4 isz;     // 24  Di, Hi, Wi, -
    uint4 osz;     // 40  Do, Ho, Wo, -
    int4 pl;       // 56  padding before (d, h, w), -
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int MODE = 0;

// Input index that output o reads along a dim (input size n, padding l).
int read_index(int o, int n, int l)
{
    int ip;
    if (MODE == 0)
        ip = o < l ? 2 * l - o : (o < n + l ? o : 2 * (n + l - 1) - o);
    else
        ip = clamp(o, l, n + l - 1);
    return ip - l;
}

// The outputs along one dim that read input i: first and last of a range
// (replication), or up to three single positions (reflection), ascending.
struct Reads
{
    int o[3];
    int count;
    int lo;
    int hi;
};

Reads reads(int i, int n, int m, int l)
{
    Reads r;
    r.count = 0;
    r.lo = 0;
    r.hi = -1;
    if (MODE == 1)
    {
        r.lo = max(i == 0 ? 0 : l + i, 0);
        r.hi = min(i == n - 1 ? m - 1 : l + i, m - 1);
        return r;
    }
    int c[3] = { l - i, l + i, 2 * n + l - 2 - i };
    for (int k = 0; k < 3; k++)
    {
        int o = c[k];
        bool fresh = r.count == 0 || o > r.o[r.count - 1];
        if (o >= 0 && o < m && fresh && read_index(o, n, l) == i)
        {
            r.o[r.count] = o;
            r.count++;
        }
    }
    return r;
}

// The k-th output of a dim's reads, and how many there are.
int reads_count(Reads r) { return MODE == 1 ? max(r.hi - r.lo + 1, 0) : r.count; }
int reads_at(Reads r, int k) { return MODE == 1 ? r.lo + k : r.o[k]; }

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint Di = p.isz.x, Hi = p.isz.y, Wi = p.isz.z;
    uint Do = p.osz.x, Ho = p.osz.y, Wo = p.osz.z;
    for (uint t = id.x; t < p.total; t += p.gstride)
    {
        uint x = t % Wi, r = t / Wi;
        uint y = r % Hi;
        r /= Hi;
        uint z = r % Di;
        uint plane = r / Di;
        Reads rd = reads(int(z), int(Di), int(Do), p.pl.x);
        Reads rh = reads(int(y), int(Hi), int(Ho), p.pl.y);
        Reads rw = reads(int(x), int(Wi), int(Wo), p.pl.z);
        int nd = reads_count(rd), nh = reads_count(rh), nw = reads_count(rw);
        T sum = T(0);
        for (int a = 0; a < nd; a++)
        {
            uint od = uint(reads_at(rd, a));
            for (int b = 0; b < nh; b++)
            {
                uint row = ((plane * Do + od) * Ho + uint(reads_at(rh, b))) * Wo;
                for (int c = 0; c < nw; c++)
                    sum += p.gout[row + uint(reads_at(rw, c))];
            }
        }
        p.gin[t] = sum;
    }
}
