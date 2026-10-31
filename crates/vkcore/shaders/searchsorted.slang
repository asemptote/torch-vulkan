// axis T: f32=float f64=double i32=int i64=int64_t
//
// searchsorted / bucketize: for each value v in row r of `vals`
// (contiguous [rows, q]), the insertion point in row r of the sorted
// sequence `seq` (contiguous, seq_row elements apart; 0 when one sequence
// serves every row). RIGHT: first i with v < seq[i] (upper bound), else
// first i with !(seq[i] < v) (lower bound), as PyTorch compares. OUT32
// writes int32 instead of int64.

struct Params
{
    T* seq;
    T* vals;
    int64_t* out;
    uint m;        // sequence length
    uint q;        // values per row
    uint seq_row;  // elements between sequence rows
    uint total;    // rows * q
    uint stride;
    uint _pad;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int RIGHT = 0;
[vk::constant_id(1)]
const int OUT32 = 0;

#if defined(T_f32) || defined(T_f64)
bool nan_(T x) { return isnan(x); }
#else
bool nan_(T x) { return false; }
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.total; i += p.stride)
    {
        uint row = i / p.q;
        uint base = row * p.seq_row;
        T v = p.vals[i];
        uint lo = 0u, hi = p.m;
        while (lo < hi)
        {
            uint mid = (lo + hi) >> 1;
            T s = p.seq[base + mid];
            // As the CPU's !(s > v) and !(s >= v): NaNs sort last. (Tested
            // explicitly: GPU compilers turn !(s > v) into s <= v.)
            bool unordered = nan_(v) || nan_(s);
            bool go_right = unordered || (RIGHT != 0 ? s <= v : s < v);
            if (go_right)
                lo = mid + 1u;
            else
                hi = mid;
        }
        if (OUT32 != 0)
            ((int*)p.out)[i] = int(lo);
        else
            p.out[i] = int64_t(lo);
    }
}
