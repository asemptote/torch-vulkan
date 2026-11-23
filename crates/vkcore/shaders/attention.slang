// axis T: f32=float f16=uint16_t bf16=uint16_t
// axis D: d64=64 d128=128 d256=256
//
// Scaled dot-product attention without the attention matrix (FlashAttention
// 2's tiling): out = softmax(q k^T scale + mask) v, its gradients, all in
// float32 (16-bit inputs are widened as they are loaded, results rounded
// once to nearest even; float16 is handled as its bits, which reads faster
// than half loads on NVIDIA at the larger head dims and needs no
// shaderFloat16). q [B, H, L, E], k [B, H / GROUP, S, E], v [B, H / GROUP, S, EV]
// with any strides along b, h and the sequence (the head dim contiguous);
// query head h reads key / value head h / GROUP. D (64, 128, 256) bounds E
// and EV; the spec constants E and EV are the head dims themselves.
//
// A workgroup of 256 threads (16 x 16: tx, ty) takes 64 "own" rows and runs
// over the "other" positions in blocks of 64: thread (tx, ty) holds the
// scores of own rows 4 ty + i and other positions 4 tx + j (i, j < 4), and
// output columns 64 k + 4 tx + j of its 4 rows. Score tiles are products of
// 16-deep slices of both operands staged in shared memory; the products
// with v (or k, q, grad_out) read the block's weights from shared memory
// (64 x 64, sh_p) and 2048-element chunks of the rows.
//
//   MODE 0, forward: own rows = queries, grid (ceil(L / 64), H, B). Online
//     softmax (running row max m and sum l per row, as FlashAttention):
//     out = sum_c exp(s_c - m) z_c v_c / l, lse = m + log l, both in float32
//     until out is rounded. A row with every score -inf (fully masked) gives
//     zeros and lse = +inf, as the math path's _safe_softmax.
//   MODE 1, grad_q: own rows = queries, grid as MODE 0. Recomputes
//     P = exp(s - lse), dP = grad_out v^T, dS = P (dP z - delta) and sums
//     grad_q = scale dS k over the key blocks.
//   MODE 2, grad_k and grad_v: own rows = keys, grid (ceil(S / 64), H /
//     GROUP, B), over the GROUP query heads of the key head and their query
//     blocks: grad_v = (P z)^T grad_out, grad_k = scale dS^T q.
// s = q . k scale + mask (an additive mask, read with its own strides;
// MASK 1 for one with unit stride along keys, 2 for stride 0), -inf where
// CAUSAL hides key c from query r (c > r, as the math path's tril) and past
// the ends. Dropout (DROPOUT 1): z = keep / (1 - p), keep = u >= p for u
// in [0, 1) from Philox4x32-10 with counter (c / 4, r, b H + h, ctr) and key
// `key`, word c % 4, so all three modes (and attention_aux.slang's mask)
// draw the same z for each (b, h, r, c). The backward modes read lse,
// delta = rowsum(grad_out out) and the dropout parameters from the stats
// that attention_aux.slang writes.

#define FMATH_F32 1
#include "_fmath.slang"
#include "_half.slang"
#include "_philox.slang"

struct Params
{
    T* q;       // 0
    T* k;       // 8
    T* v;       // 16
    T* o;       // 24  MODE 0: out [B, L, H, EV] (written); 1, 2: grad_out, same layout
    float* st;  // 32  MODE 0: lse [B, H, L] (written); 1, 2: stats (header, then (lse, delta) per row)
    T* mask;    // 40
    uint2 w48;  // 48  MODE 0: dropout key; 1: grad_q [B, L, H, E]; 2: grad_k [B, S, H / GROUP, E]
    uint2 w56;  // 56  MODE 0: (ctr, p's bits); 2: grad_v [B, S, H / GROUP, EV]
    uint4 qs;   // 64  q's strides (b, h, n), H
    uint4 ks;   // 80  k's strides (b, h, n), L
    uint4 vs;   // 96  v's strides (b, h, n), S
    uint4 ms;   // 112 mask's strides (b, h, query), scale's bits
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)] const int MODE = 0;
[vk::constant_id(1)] const int E = 64;
[vk::constant_id(2)] const int EV = 64;
[vk::constant_id(3)] const int CAUSAL = 0;
[vk::constant_id(4)] const int MASK = 0;
[vk::constant_id(5)] const int DROPOUT = 0;
[vk::constant_id(6)] const int GROUP = 1;

static const uint NK = D / 64;  // column groups of 64 per output row
static const uint D4 = D / 4;   // float4s per staged row
static const uint CR = 2048 / D;  // rows per staged chunk
static const uint PS = 17;      // sh_p's row stride, float4s (64 + 4 floats)

groupshared float4 sh_u[512];      // operand slices, row chunks or row partials
groupshared float4 sh_p[64 * PS];  // the block's weights [own][other]

float ld(T v)
{
#if defined(T_bf16)
    return asfloat(uint(v) << 16);
#elif defined(T_f16)
    return f16tof32(uint(v));
#else
    return v;
#endif
}

T st(float f)
{
#if defined(T_f16)
    return f16_bits_rne(f);
#elif defined(T_bf16)
    if (isnan(f))
        return uint16_t(0x7fc0);
    uint u = asuint(f);
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return uint16_t(u >> 16);
#else
    return f;
#endif
}

T* as_ptr(uint2 w)
{
    return (T*)(uint64_t(w.x) | (uint64_t(w.y) << 32));
}

// Stages a 16-deep slice (depth d0..d0+15 of `depth`) of 64 rows (n valid,
// row r at base + r rs) at sh_u[at..at+256) as [16 depth][64 rows].
void stage_slice(T* a, uint base, uint rs, uint n, uint d0, uint depth, uint at, uint lid)
{
    uint r = lid >> 2, dd = (lid & 3) * 4;
    for (uint t = 0; t < 4; t++)
    {
        uint d = d0 + dd + t;
        float x = r < n && d < depth ? ld(a[base + r * rs + d]) : 0.0f;
        sh_u[at + (dd + t) * 16 + (r >> 2)][r & 3] = x;
    }
}

// x[i][j] = a_{4 ty + i} . b_{4 tx + j} over `depth`: rows of a (na valid)
// at abase + r ars, of b at bbase + r brs.
void scores(inout float4 x[4], T* a, uint abase, uint ars, uint na, T* b, uint bbase, uint brs, uint nb, uint depth,
            uint lid)
{
    uint tx = lid & 15, ty = lid >> 4;
    for (uint i = 0; i < 4; i++)
        x[i] = float4(0.0f);
    for (uint d0 = 0; d0 < depth; d0 += 16)
    {
        GroupMemoryBarrierWithGroupSync();
        stage_slice(a, abase, ars, na, d0, depth, 0, lid);
        stage_slice(b, bbase, brs, nb, d0, depth, 256, lid);
        GroupMemoryBarrierWithGroupSync();
        for (uint dd = 0; dd < 16; dd++)
        {
            float4 av = sh_u[dd * 16 + ty], bv = sh_u[256 + dd * 16 + tx];
            for (uint i = 0; i < 4; i++)
                x[i] = fma(float4(av[i]), bv, x[i]);
        }
    }
}

// acc[i][k] += sum_c sh_p[4 ty + i][c] c_c[64 k + 4 tx ..], c over the 64
// other rows (n valid) of c at base + r rs, `width` columns.
void weighted_rows(inout float4 acc[4][NK], T* c, uint base, uint rs, uint n, uint width, uint lid)
{
    uint tx = lid & 15, ty = lid >> 4;
    for (uint c0 = 0; c0 < 64; c0 += CR)
    {
        GroupMemoryBarrierWithGroupSync();
        for (uint s = lid; s < CR * D4; s += 256)
        {
            uint r = s / D4, d = (s % D4) * 4;
            float4 x = float4(0.0f);
            if (c0 + r < n)
            {
                uint e = base + (c0 + r) * rs + d;
                for (uint t = 0; t < 4; t++)
                    if (d + t < width)
                        x[t] = ld(c[e + t]);
            }
            sh_u[s] = x;
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint cc = 0; cc < CR; cc += 4)
        {
            float4 w[4];
            for (uint i = 0; i < 4; i++)
                w[i] = sh_p[(4 * ty + i) * PS + (c0 + cc) / 4];
            for (uint t = 0; t < 4; t++)
            {
                for (uint k = 0; k < NK; k++)
                {
                    float4 cv = sh_u[(cc + t) * D4 + 16 * k + tx];
                    for (uint i = 0; i < 4; i++)
                        acc[i][k] = fma(float4(w[i][t]), cv, acc[i][k]);
                }
            }
        }
    }
}

// Dropout: the 4 keep factors (0 or 1 / (1 - p)) of keys 4 cq .. 4 cq + 3
// for query r of row bh.
float4 keep4(uint cq, uint r, uint bh, uint2 key, uint ctr, float pdrop)
{
    uint4 w = philox(uint4(cq, r, bh, ctr), key);
    float4 u = float4(w >> 8u) * 5.9604644775390625e-8f;  // 2^-24
    float z = 1.0f / (1.0f - pdrop);
    return float4(u.x >= pdrop ? z : 0.0f, u.y >= pdrop ? z : 0.0f, u.z >= pdrop ? z : 0.0f, u.w >= pdrop ? z : 0.0f);
}

// The mask's value for query r, key c of head h, batch b.
float mask_at(uint b, uint h, uint r, uint c)
{
    uint e = b * p.ms.x + h * p.ms.y + r * p.ms.z + (MASK == 1 ? c : 0u);
    return ld(p.mask[e]);
}

// The score s for query r, key c (raw product x); -inf where hidden.
float score(float x, float scale, uint b, uint h, uint r, uint c, uint L, uint S)
{
    if (r >= L || c >= S || (CAUSAL != 0 && c > r))
        return -m_inf();
    float s = x * scale;
    if (MASK != 0)
        s += mask_at(b, h, r, c);
    return s;
}

void forward(uint3 gid, uint lid)
{
    uint tx = lid & 15, ty = lid >> 4;
    uint H = p.qs.w, L = p.ks.w, S = p.vs.w;
    uint h = gid.y, b = gid.z, r0 = gid.x * 64, hk = h / uint(GROUP);
    float scale = asfloat(p.ms.w);
    uint2 key = p.w48;
    uint ctr = p.w56.x;
    float pdrop = asfloat(p.w56.y);
    uint qbase = b * p.qs.x + h * p.qs.y + r0 * p.qs.z;
    uint kbase = b * p.ks.x + hk * p.ks.y, vbase = b * p.vs.x + hk * p.vs.y;
    uint nq = min(64u, L - r0);
    uint nblk = (S + 63) / 64;
    if (CAUSAL != 0)
        nblk = min(nblk, (r0 + 63) / 64 + 1);
    float m[4], l[4];
    float4 acc[4][NK];
    for (uint i = 0; i < 4; i++)
    {
        m[i] = -m_inf();
        l[i] = 0.0f;
        for (uint k = 0; k < NK; k++)
            acc[i][k] = float4(0.0f);
    }
    for (uint kb = 0; kb < nblk; kb++)
    {
        uint c0 = kb * 64, nk = min(64u, S - c0);
        float4 x[4];
        scores(x, p.q, qbase, p.qs.z, nq, p.k, kbase + c0 * p.ks.z, p.ks.z, nk, uint(E), lid);
        float4 bm = float4(-m_inf());
        for (uint i = 0; i < 4; i++)
        {
            for (uint j = 0; j < 4; j++)
            {
                x[i][j] = score(x[i][j], scale, b, h, r0 + 4 * ty + i, c0 + 4 * tx + j, L, S);
                bm[i] = max(bm[i], x[i][j]);
            }
        }
        // (the row maxima over the 16 threads sharing the rows)
        GroupMemoryBarrierWithGroupSync();
        sh_u[tx * 16 + ty] = bm;
        GroupMemoryBarrierWithGroupSync();
        for (uint t = 0; t < 16; t++)
            bm = max(bm, sh_u[t * 16 + ty]);
        for (uint i = 0; i < 4; i++)
        {
            float mn = max(m[i], bm[i]);
            float alpha = m[i] == -m_inf() ? 0.0f : m_exp(m[i] - mn);
            float ms = mn == -m_inf() ? 0.0f : mn;
            m[i] = mn;
            float4 pr;
            for (uint j = 0; j < 4; j++)
                pr[j] = m_exp(x[i][j] - ms);
            l[i] = l[i] * alpha + ((pr.x + pr.y) + (pr.z + pr.w));
            for (uint k = 0; k < NK; k++)
                acc[i][k] = acc[i][k] * alpha;
            if (DROPOUT != 0)
                pr = pr * keep4(c0 / 4 + tx, r0 + 4 * ty + i, b * H + h, key, ctr, pdrop);
            sh_p[(4 * ty + i) * PS + tx] = pr;
        }
        weighted_rows(acc, p.v, vbase + c0 * p.vs.z, p.vs.z, nk, uint(EV), lid);
    }
    // (the row sums over the 16 threads)
    GroupMemoryBarrierWithGroupSync();
    sh_u[tx * 16 + ty] = float4(l[0], l[1], l[2], l[3]);
    GroupMemoryBarrierWithGroupSync();
    float4 ls = float4(0.0f);
    for (uint t = 0; t < 16; t++)
        ls += sh_u[t * 16 + ty];
    for (uint i = 0; i < 4; i++)
    {
        uint r = r0 + 4 * ty + i;
        if (r >= L)
            continue;
        float li = ls[i];
        uint obase = ((b * L + r) * H + h) * uint(EV);
        for (uint k = 0; k < NK; k++)
        {
            for (uint j = 0; j < 4; j++)
            {
                uint d = 64 * k + 4 * tx + j;
                if (d < uint(EV))
                    p.o[obase + d] = st(li > 0.0f ? acc[i][k][j] / li : 0.0f);
            }
        }
        if (tx == 0)
            p.st[(b * H + h) * L + r] = li > 0.0f ? m[i] + m_log(li) : m_inf();
    }
}

// The stats' (lse, delta) of query r of head h in batch b.
float2 row_stats(uint b, uint h, uint r, uint H, uint L)
{
    uint e = 4 + 2 * ((b * H + h) * L + r);
    return float2(p.st[e], p.st[e + 1]);
}

void grad_q(uint3 gid, uint lid)
{
    uint tx = lid & 15, ty = lid >> 4;
    uint H = p.qs.w, L = p.ks.w, S = p.vs.w;
    uint h = gid.y, b = gid.z, r0 = gid.x * 64, hk = h / uint(GROUP);
    float scale = asfloat(p.ms.w);
    uint* hdr = (uint*)p.st;
    uint2 key = uint2(hdr[0], hdr[1]);
    uint ctr = hdr[2];
    float pdrop = asfloat(hdr[3]);
    uint qbase = b * p.qs.x + h * p.qs.y + r0 * p.qs.z;
    uint obase = ((b * L + r0) * H + h) * uint(EV), ors = H * uint(EV);
    uint kbase = b * p.ks.x + hk * p.ks.y, vbase = b * p.vs.x + hk * p.vs.y;
    uint nq = min(64u, L - r0);
    uint nblk = (S + 63) / 64;
    if (CAUSAL != 0)
        nblk = min(nblk, (r0 + 63) / 64 + 1);
    float2 rs[4];
    float4 acc[4][NK];
    for (uint i = 0; i < 4; i++)
    {
        uint r = r0 + 4 * ty + i;
        rs[i] = r < L ? row_stats(b, h, r, H, L) : float2(m_inf(), 0.0f);
        for (uint k = 0; k < NK; k++)
            acc[i][k] = float4(0.0f);
    }
    for (uint kb = 0; kb < nblk; kb++)
    {
        uint c0 = kb * 64, nk = min(64u, S - c0);
        float4 x[4], y[4];
        scores(x, p.q, qbase, p.qs.z, nq, p.k, kbase + c0 * p.ks.z, p.ks.z, nk, uint(E), lid);
        scores(y, p.o, obase, ors, nq, p.v, vbase + c0 * p.vs.z, p.vs.z, nk, uint(EV), lid);
        for (uint i = 0; i < 4; i++)
        {
            uint r = r0 + 4 * ty + i;
            float4 ds;
            float4 z = DROPOUT != 0 ? keep4(c0 / 4 + tx, r, b * H + h, key, ctr, pdrop) : float4(1.0f);
            for (uint j = 0; j < 4; j++)
            {
                float pr = m_exp(score(x[i][j], scale, b, h, r, c0 + 4 * tx + j, L, S) - rs[i].x);
                ds[j] = pr * (y[i][j] * z[j] - rs[i].y);
            }
            sh_p[(4 * ty + i) * PS + tx] = ds;
        }
        weighted_rows(acc, p.k, kbase + c0 * p.ks.z, p.ks.z, nk, uint(E), lid);
    }
    T* gq = as_ptr(p.w48);
    for (uint i = 0; i < 4; i++)
    {
        uint r = r0 + 4 * ty + i;
        if (r >= L)
            continue;
        uint gbase = ((b * L + r) * H + h) * uint(E);
        for (uint k = 0; k < NK; k++)
            for (uint j = 0; j < 4; j++)
            {
                uint d = 64 * k + 4 * tx + j;
                if (d < uint(E))
                    gq[gbase + d] = st(acc[i][k][j] * scale);
            }
    }
}

void grad_kv(uint3 gid, uint lid)
{
    uint tx = lid & 15, ty = lid >> 4;
    uint H = p.qs.w, L = p.ks.w, S = p.vs.w, HK = H / uint(GROUP);
    uint hk = gid.y, b = gid.z, c0 = gid.x * 64;
    float scale = asfloat(p.ms.w);
    uint* hdr = (uint*)p.st;
    uint2 key = uint2(hdr[0], hdr[1]);
    uint ctr = hdr[2];
    float pdrop = asfloat(hdr[3]);
    uint kbase = b * p.ks.x + hk * p.ks.y + c0 * p.ks.z, vbase = b * p.vs.x + hk * p.vs.y + c0 * p.vs.z;
    uint nk = min(64u, S - c0), ors = H * uint(EV);
    float4 acck[4][NK], accv[4][NK];
    for (uint i = 0; i < 4; i++)
        for (uint k = 0; k < NK; k++)
        {
            acck[i][k] = float4(0.0f);
            accv[i][k] = float4(0.0f);
        }
    for (uint g = 0; g < uint(GROUP); g++)
    {
        uint h = hk * uint(GROUP) + g;
        // (with CAUSAL, queries before c0 see none of these keys)
        for (uint r0 = CAUSAL != 0 ? c0 : 0u; r0 < L; r0 += 64)
        {
            uint nq = min(64u, L - r0);
            uint qbase = b * p.qs.x + h * p.qs.y + r0 * p.qs.z;
            uint obase = ((b * L + r0) * H + h) * uint(EV);
            float4 x[4], y[4];
            scores(x, p.k, kbase, p.ks.z, nk, p.q, qbase, p.qs.z, nq, uint(E), lid);
            scores(y, p.v, vbase, p.vs.z, nk, p.o, obase, ors, nq, uint(EV), lid);
            float4 pz[4], ds[4];
            for (uint j = 0; j < 4; j++)
            {
                uint r = r0 + 4 * tx + j;
                float2 rs = r < L ? row_stats(b, h, r, H, L) : float2(m_inf(), 0.0f);
                float4 z = DROPOUT != 0 ? keep4(c0 / 4 + ty, r, b * H + h, key, ctr, pdrop) : float4(1.0f);
                for (uint i = 0; i < 4; i++)
                {
                    float pr = m_exp(score(x[i][j], scale, b, h, r, c0 + 4 * ty + i, L, S) - rs.x);
                    pz[i][j] = pr * z[i];
                    ds[i][j] = pr * (y[i][j] * z[i] - rs.y);
                }
            }
            for (uint i = 0; i < 4; i++)
                sh_p[(4 * ty + i) * PS + tx] = pz[i];
            weighted_rows(accv, p.o, obase, ors, nq, uint(EV), lid);
            GroupMemoryBarrierWithGroupSync();
            for (uint i = 0; i < 4; i++)
                sh_p[(4 * ty + i) * PS + tx] = ds[i];
            weighted_rows(acck, p.q, qbase, p.qs.z, nq, uint(E), lid);
        }
    }
    T* gk = as_ptr(p.w48);
    T* gv = as_ptr(p.w56);
    for (uint i = 0; i < 4; i++)
    {
        uint c = c0 + 4 * ty + i;
        if (c >= S)
            continue;
        uint kb = ((b * S + c) * HK + hk) * uint(E), vb = ((b * S + c) * HK + hk) * uint(EV);
        for (uint k = 0; k < NK; k++)
            for (uint j = 0; j < 4; j++)
            {
                uint d = 64 * k + 4 * tx + j;
                if (d < uint(E))
                    gk[kb + d] = st(acck[i][k][j] * scale);
                if (d < uint(EV))
                    gv[vb + d] = st(accv[i][k][j]);
            }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid: SV_GroupID, uint3 lid: SV_GroupThreadID)
{
    if (MODE == 0)
        forward(gid, lid.x);
    else if (MODE == 1)
        grad_q(gid, lid.x);
    else
        grad_kv(gid, lid.x);
}
