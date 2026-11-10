// axis T: f32=float f64=double i32=int i64=I64
//
// Batched matrix-vector product, for the GEMMs csrc/linalg.cpp sees with one
// output column or one output row (addmv, mv, vec @ mat, a single sample
// through nn.Linear):
//
//     out[b][r] = alpha * sum_k M[b][r, k] v[b][k]  (+ beta * cin[b][r] when USE_CIN)
//
// gemm.slang's 64 x 64 tiles would leave 63 of every 64 columns idle; a
// matrix-vector product reads each matrix element once, so the kernel's job
// is to read M in whole cache lines. MODE picks the threads' layout from M's
// strides (the host's choice):
//
//   MODE 0 (M's k stride is 1, or neither stride is): 8 rows per workgroup,
//     32 threads per row striding through k, so a row's reads are contiguous;
//     the 32 partial sums are added in shared memory.
//   MODE 1 (M's row stride is 1, as for a transposed matrix): 64 rows per
//     workgroup, one per thread, in 4 groups of threads that take every 4th k;
//     the 64 threads of a group read 64 consecutive elements of a column.
//
// Integer variants wrap on overflow, as gemm.slang's do. Grid: (row blocks,
// batch).

#include "_int64.slang"  // (the i64 variant's T: _int64.slang)
#include "_slot.slang"

struct Params
{
    T* mat;          // 0
    T* vec;          // 8
    T* out;          // 16
    T* cin;          // 24
    slot64 alpha;    // 32  (raw slots, as in gemm.slang)
    slot64 beta;     // 40
    uint rows;       // 48
    uint k;          // 52
    int m_sr, m_sk, m_sb;  // 56
    int v_sk, v_sb;        // 68
    int o_sr, o_sb;        // 76
    int c_sr, c_sb;        // 84 .. 92
};

[[vk::push_constant]] ConstantBuffer<Params> p;

T param(slot64 bits)
{
#if defined(T_f64)
    return slot_double(bits);
#elif defined(T_f32)
    return slot_float(bits);
#else
    return T(slot_integer(bits));
#endif
}

[vk::constant_id(0)] const int USE_CIN = 0;
[vk::constant_id(1)] const int MODE = 0;

#define WG 256
#define LANES 32                // MODE 0: threads per row
#define ROWS0 (WG / LANES)      // MODE 0: rows per workgroup
#define ROWS1 64                // MODE 1: rows per workgroup
#define KGROUPS (WG / ROWS1)    // MODE 1: groups of threads splitting k

groupshared T part[WG];

// Partial sum of row r over k = first, first + step, ... (r < rows, k > 0).
// (Loads index min(kk, k - 1): lavapipe performs a loop's first loads before
// testing its condition, for every lane, as factor.slang notes.)
T row_sum(uint r, int mbase, int vbase, uint first, uint step)
{
    int row = mbase + int(r) * p.m_sr;
    uint last = p.k - 1;
    T acc = T(0);
    for (uint kk = first; kk < p.k; kk += step)
    {
        int kc = int(min(kk, last));
        acc += p.mat[row + kc * p.m_sk] * p.vec[vbase + kc * p.v_sk];
    }
    return acc;
}

void store(uint r, int batch, T acc)
{
    T v = param(p.alpha) * acc;
    if (USE_CIN != 0)  // beta == 0 must ignore cin entirely (it may hold NaN)
        v += param(p.beta) * p.cin[batch * p.c_sb + int(r) * p.c_sr];
    p.out[batch * p.o_sb + int(r) * p.o_sr] = v;
}

[shader("compute")]
[numthreads(WG, 1, 1)]
void main(uint3 tid: SV_GroupThreadID, uint3 gid: SV_GroupID)
{
    uint t = tid.x;
    int batch = int(gid.y);
    int mbase = batch * p.m_sb, vbase = batch * p.v_sb;
    if (MODE == 0)
    {
        uint lane = t % LANES, r = gid.x * ROWS0 + t / LANES;
        T acc = T(0);
        if (r < p.rows && p.k > 0)
            acc = row_sum(r, mbase, vbase, lane, LANES);
        part[t] = acc;
        GroupMemoryBarrierWithGroupSync();
        for (uint s = LANES / 2; s > 0; s >>= 1)
        {
            if (lane < s)
                part[t] += part[t + s];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0 && r < p.rows)
            store(r, batch, part[t]);
    }
    else
    {
        uint lane = t % ROWS1, group = t / ROWS1, r = gid.x * ROWS1 + lane;
        T acc = T(0);
        if (r < p.rows && p.k > 0)
            acc = row_sum(r, mbase, vbase, group, KGROUPS);
        part[t] = acc;
        GroupMemoryBarrierWithGroupSync();
        if (group == 0 && r < p.rows)
        {
            T sum = part[lane];
            for (uint g = 1; g < KGROUPS; g++)
                sum += part[lane + g * ROWS1];
            store(r, batch, sum);
        }
    }
}
