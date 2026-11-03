// axis T: f32=float f64=double
//
// out[i] drawn with element i's own parameters, by the algorithms of
// PyTorch's CPU and CUDA kernels (ATen/native/Distributions.h), in T's
// precision (as CUDA; the CPU computes float32 ones in double). OP:
//   SAMPLE_GAMMA     standard gamma with shape a[i] (Marsaglia-Tsang, shapes
//                    below 1 boosted), at least the smallest normal T
//   SAMPLE_LOG_GAMMA its logarithm, at least log of the smallest normal
//                    double (Dirichlet samples are their softmax: no
//                    underflow for small shapes in float32)
//   SAMPLE_POISSON   Poisson with rate a[i] (Hoermann's transformed
//                    rejection from rate 10, a product of uniforms below)
//   SAMPLE_BINOMIAL  binomial with count a[i], probability b[i] (BTRS from
//                    a mean of 10, inversion below)
// and the reparameterized gradients of the first and of Dirichlet samples
// (no randomness):
//   SAMPLE_GAMMA_GRAD      of a gamma sample x = b[i] with shape alpha = a[i]
//   SAMPLE_DIRICHLET_GRAD  of a Beta sample x = a[i], alpha = b[i], total = c[i]
// Rejection loops stop after LOOPS tries (a NaN then: only reachable with
// non-finite parameters, which would otherwise never finish). The loops draw block j of element i from Philox counter
// (i, j + 1, offset): rng.slang's counters have j = 0, and the host gives
// each call its own offset.

#include "_types.slang"
#include "_wide.slang"
#include "opcodes.slang"
#define IS_FLOAT 1
#include "_fmath.slang"

typedef T R;
#include "_philox.slang"

struct Params
{
    T* out;
    T* a;
    T* b;
    T* c;
    slot64 seed;
    slot64 offset;
    uint n;
    uint stride;
};

[[vk::push_constant]] ConstantBuffer<Params> p;

[vk::constant_id(0)]
const int OP = SAMPLE_GAMMA;

static const int LOOPS = 100000;

// Element i's stream of Philox blocks.
struct Stream
{
    uint i;
    uint j;
    [mutating] uint4 next()
    {
        j += 1u;
        return philox(uint4(i, j, p.offset.x, p.offset.y), p.seed);
    }
};

R draw_uniform(inout Stream s) { return uniform(s.next(), 0); }
R draw_normal(inout Stream s) { return standard_normal(s.next()); }

R sample_gamma(R alpha, inout Stream s)
{
    if (!(alpha >= R(0)))
        return m_nan();
    R scale = R(1);
    if (alpha < R(1))  // boost: Gamma(a) = Gamma(a + 1) U^(1/a)
    {
        if (alpha == R(0))
            return R(0);
        scale *= m_pow(R(1) - draw_uniform(s), R(1) / alpha);
        alpha += R(1);
    }
    R d = alpha - R(1) / R(3);
    R c = R(1) / sqrt(R(9) * d);
    for (int t = 0; t < LOOPS; t++)
    {
        R x = draw_normal(s), y = R(1) + c * x;
        if (y <= R(0))
            continue;
        R v = y * y * y;
        R u = R(1) - draw_uniform(s);
        R xx = x * x;
        if (u < R(1) - LIT(0.0331) * xx * xx)
            return scale * d * v;
        if (m_log(u) < LIT(0.5) * xx + d * (R(1) - v + m_log(v)))
            return scale * d * v;
    }
    return m_nan();
}

// log of a standard gamma sample, the boost for shapes below 1 added as a log.
R sample_log_gamma(R alpha, inout Stream s)
{
    if (!(alpha >= R(0)))
        return m_nan();
    const R LOG_TINY = LIT(-708.3964185322641);  // log(2^-1022)
    if (alpha == R(0))
        return LOG_TINY;
    R lg = alpha < R(1) ? m_log(sample_gamma(alpha + R(1), s)) + m_log(R(1) - draw_uniform(s)) / alpha
                        : m_log(sample_gamma(alpha, s));
    return lg < LOG_TINY ? LOG_TINY : lg;
}

R sample_poisson(R lambda, inout Stream s)
{
    if (lambda >= R(10))
    {
        R slam = sqrt(lambda), loglam = m_log(lambda);
        R b = LIT(0.931) + LIT(2.53) * slam;
        R a = LIT(-0.059) + LIT(0.02483) * b;
        R invalpha = LIT(1.1239) + LIT(1.1328) / (b - LIT(3.4));
        R vr = LIT(0.9277) - LIT(3.6224) / (b - R(2));
        for (int t = 0; t < LOOPS; t++)
        {
            R U = draw_uniform(s) - LIT(0.5);
            R V = draw_uniform(s);
            R us = LIT(0.5) - abs(U);
            R k = floor((R(2) * a / us + b) * U + lambda + LIT(0.43));
            if (us >= LIT(0.07) && V <= vr)
                return k;
            if (!(k >= R(0)) || (us < LIT(0.013) && V > us))
                continue;
            if (m_log(V) + m_log(invalpha) - m_log(a / (us * us) + b) <= -lambda + k * loglam - m_lgamma(k + R(1)))
                return k;
        }
        return m_nan();
    }
    if (lambda == R(0))
        return R(0);
    R enlam = m_exp(-lambda), prod = R(1), X = R(0);
    for (int t = 0; t < LOOPS; t++)
    {
        prod *= draw_uniform(s);
        if (prod > enlam)
            X += R(1);
        else
            return X;
    }
    return m_nan();
}

R stirling_approx_tail(R k)
{
    static const R tail[10] = {LIT(0.0810614667953272), LIT(0.0413406959554092), LIT(0.0276779256849983),
                               LIT(0.02079067210376509), LIT(0.0166446911898211), LIT(0.0138761288230707),
                               LIT(0.0118967099458917), LIT(0.0104112652619720), LIT(0.00925546218271273),
                               LIT(0.00833056343336287)};
    if (k <= R(9))
        return tail[int(k)];
    R kp1sq = (k + R(1)) * (k + R(1));
    return (R(1) / R(12) - (R(1) / R(360) - R(1) / R(1260) / kp1sq) / kp1sq) / (k + R(1));
}

// Binomial transformed rejection with squeeze (Hoermann 1993).
R btrs(R count, R prob, inout Stream s)
{
    R stddev = sqrt(count * prob * (R(1) - prob));
    R b = LIT(1.15) + LIT(2.53) * stddev;
    R a = LIT(-0.0873) + LIT(0.0248) * b + LIT(0.01) * prob;
    R c = count * prob + LIT(0.5);
    R v_r = LIT(0.92) - LIT(4.2) / b;
    R r = prob / (R(1) - prob);
    R alpha = (LIT(2.83) + LIT(5.1) / b) * stddev;
    R m = floor((count + R(1)) * prob);
    for (int t = 0; t < LOOPS; t++)
    {
        R U = draw_uniform(s) - LIT(0.5);
        R V = draw_uniform(s);
        R us = LIT(0.5) - abs(U);
        R k = floor((R(2) * a / us + b) * U + c);
        if (!(k >= R(0) && k <= count))
            continue;
        if (us >= LIT(0.07) && V <= v_r)
            return k;
        V = m_log(V * alpha / (a / (us * us) + b));
        R upper = (m + LIT(0.5)) * m_log((m + R(1)) / (r * (count - m + R(1)))) +
                  (count + R(1)) * m_log((count - m + R(1)) / (count - k + R(1))) +
                  (k + LIT(0.5)) * m_log(r * (count - k + R(1)) / (k + R(1))) + stirling_approx_tail(m) +
                  stirling_approx_tail(count - m) - stirling_approx_tail(k) - stirling_approx_tail(count - k);
        if (V <= upper)
            return k;
    }
    return m_nan();
}

R binomial_inversion(R count, R prob, inout Stream s)
{
    R geom_sum = R(0), num_geom = R(0);
    R logprob = m_log1p(-prob);
    for (int t = 0; t < LOOPS; t++)
    {
        R geom = ceil(m_log(draw_uniform(s)) / logprob);
        geom_sum += geom;
        if (geom_sum > count)
            break;
        num_geom += R(1);
    }
    return geom_sum > count ? num_geom : m_nan();
}

R sample_binomial(R count, R prob, inout Stream s)
{
    if (isnan(count) || isinf(count) || isnan(prob))
        return m_nan();
    if (count <= R(0) || prob <= R(0))
        return R(0);
    if (prob >= R(1))
        return count;
    if (prob <= LIT(0.5))
        return count * prob >= R(10) ? btrs(count, prob, s) : binomial_inversion(count, prob, s);
    if (prob > LIT(0.5))
    {
        R q = R(1) - prob;
        return count - (count * q >= R(10) ? btrs(count, q, s) : binomial_inversion(count, q, s));
    }
    return m_nan();  // (a NaN probability)
}

// ---------------------------------------------------------------- gradients

// digamma as Distributions.h's digamma_one (Cephes).
R digamma_one(R x)
{
    const R PSI_10 = LIT(2.25175258906672110764);
    if (x == R(0))
        return m_inf();
    R extra = R(0);
    if (x < R(0))
    {
        if (x == floor(x))
            return m_inf();
        extra = -LIT(3.14159265358979323846) / m_tan(LIT(3.14159265358979323846) * x);
        x = R(1) - x;
    }
    R result = R(0);
    for (int t = 0; t < 64 && x < R(10); t++)
    {
        result -= R(1) / x;
        x += R(1);
    }
    if (x == R(10))
        return result + PSI_10 + extra;
    static const R A[7] = {LIT(8.33333333333333333333E-2), LIT(-2.10927960927960927961E-2), LIT(7.57575757575757575758E-3),
                           LIT(-4.16666666666666666667E-3), LIT(3.96825396825396825397E-3), LIT(-8.33333333333333333333E-3),
                           LIT(8.33333333333333333333E-2)};
    R y = R(0);
    if (x < LIT(1.0e17))
    {
        R z = R(1) / (x * x), poly = A[0];
        for (int k = 1; k < 7; k++)
            poly = poly * z + A[k];
        y = z * poly;
    }
    return result + m_log(x) - LIT(0.5) / x - y + extra;
}

// -(d/dalpha cdf(x; alpha)) / pdf(x; alpha) for a standard gamma sample x.
R standard_gamma_grad_one(R alpha, R x)
{
    if (x < LIT(0.8))  // Taylor series for small x
    {
        R numer = R(1), denom = alpha;
        R series1 = numer / denom, series2 = numer / (denom * denom);
        for (int i = 1; i <= 5; ++i)
        {
            numer *= -x / R(i);
            denom += R(1);
            series1 += numer / denom;
            series2 += numer / (denom * denom);
        }
        R pow_x_alpha = m_pow(x, alpha);
        R gamma_pdf = m_pow(x, alpha - R(1)) * m_exp(-x);
        R gamma_cdf = pow_x_alpha * series1;
        R gamma_cdf_alpha = (m_log(x) - digamma_one(alpha)) * gamma_cdf - pow_x_alpha * series2;
        R result = -gamma_cdf_alpha / gamma_pdf;
        return isnan(result) ? R(0) : result;
    }
    if (alpha > R(8))  // Rice saddle point expansion for large alpha
    {
        if (LIT(0.9) * alpha <= x && x <= LIT(1.1) * alpha)
        {
            R numer_1 = R(1) + R(24) * alpha * (R(1) + R(12) * alpha);
            R numer_2 = R(1440) * (alpha * alpha) + R(6) * x * (R(53) - R(120) * x) - R(65) * x * x / alpha +
                        alpha * (R(107) + R(3600) * x);
            R denom = R(1244160) * (alpha * alpha) * (alpha * alpha);
            return numer_1 * numer_2 / denom;
        }
        R denom = sqrt(R(8) * alpha);
        R term2 = denom / (alpha - x);
        R term3 = m_pow(x - alpha - alpha * m_log(x / alpha), LIT(-1.5));
        R term23 = x < alpha ? term2 - term3 : term2 + term3;
        R term1 = m_log(x / alpha) * term23 - sqrt(R(2) / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
        R stirling = R(1) + R(1) / (R(12) * alpha) * (R(1) + R(1) / (R(24) * alpha));
        return -stirling * (x * term1) / denom;
    }
    // bivariate rational approximation
    R u = m_log(x / alpha), v = m_log(alpha);
    static const R coef_uv[3][8] = {
        {LIT(0.16009398), LIT(-0.094634809), LIT(0.025146376), LIT(-0.0030648343), LIT(1.0), LIT(0.32668115),
         LIT(0.10406089), LIT(0.0014179084)},
        {LIT(0.53487893), LIT(0.1298071), LIT(0.065735949), LIT(-0.0015649758), LIT(0.16639465), LIT(0.020070113),
         LIT(-0.0035938915), LIT(-0.00058392623)},
        {LIT(0.040121004), LIT(-0.0065914022), LIT(-0.0026286047), LIT(-0.0013441777), LIT(0.017050642),
         LIT(-0.0021309326), LIT(0.00085092367), LIT(-1.5247877e-07)},
    };
    R cv[8];
    for (int i = 0; i < 8; ++i)
        cv[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
    R pp = cv[0] + v * (cv[1] + v * (cv[2] + v * cv[3]));
    R qq = cv[4] + v * (cv[5] + v * (cv[6] + v * cv[7]));
    return m_exp(pp / qq);
}

R beta_grad_alpha_small(R x, R alpha, R beta)
{
    R factor = digamma_one(alpha) - digamma_one(alpha + beta) - m_log(x);
    R numer = R(1);
    R series = numer / alpha * (factor + R(1) / alpha);
    for (int i = 1; i <= 10; ++i)
    {
        numer *= (R(i) - beta) * x / R(i);
        R denom = alpha + R(i);
        series += numer / denom * (factor + R(1) / denom);
    }
    R result = x * m_pow(R(1) - x, -beta) * series;
    return isnan(result) ? R(0) : result;
}

R beta_grad_beta_small(R x, R alpha, R beta)
{
    R factor = digamma_one(alpha + beta) - digamma_one(beta);
    R numer = R(1), betas = R(1), dbetas = R(0), series = factor / alpha;
    for (int i = 1; i <= 8; ++i)
    {
        numer *= -x / R(i);
        dbetas = dbetas * (beta - R(i)) + betas;
        betas = betas * (beta - R(i));
        series += numer / (alpha + R(i)) * (dbetas + factor * betas);
    }
    R result = -m_pow(R(1) - x, R(1) - beta) * series;
    return isnan(result) ? R(0) : result;
}

// (PyTorch computes this branch in double; in float32 its formula cancels
// badly near the mean: there the polynomial is used out to half a standard
// deviation, 0.8% worst error against PyTorch's, from 84% at its 0.1)
#if defined(T_f64)
static const R MID_POLY = LIT(0.1);
#else
static const R MID_POLY = LIT(0.5);
#endif

R beta_grad_alpha_mid(R x, R alpha, R beta)
{
    R total = alpha + beta, mean = alpha / total;
    R std = sqrt(alpha * beta / (total + R(1))) / total;
    if (mean - MID_POLY * std <= x && x <= mean + MID_POLY * std)
    {
        R poly = R(47) * x * (beta * beta) * (beta * beta) +
                 alpha * ((R(43) + R(20) * (R(16) + R(27) * beta) * x) * (beta * beta) * beta +
                          alpha * (R(3) * (R(59) + R(180) * beta - R(90) * x) * (beta * beta) +
                                   alpha * ((R(453) + R(1620) * beta * (R(1) - x) - R(455) * x) * beta +
                                            alpha * (R(8) * (R(1) - x) * (R(135) * beta - R(11))))));
        R prefactor_num = (R(1) + R(12) * alpha) * (R(1) + R(12) * beta) / (total * total);
        R prefactor_den = R(12960) * alpha * alpha * alpha * beta * beta * (R(1) + R(12) * total);
        return prefactor_num / (R(1) - x) * poly / prefactor_den;
    }
    R prefactor = -x / sqrt(R(2) * alpha * beta / total);
    R stirling = (R(1) + R(1) / (R(12) * alpha) + R(1) / (R(288) * alpha * alpha)) *
                 (R(1) + R(1) / (R(12) * beta) + R(1) / (R(288) * beta * beta)) /
                 (R(1) + R(1) / (R(12) * total) + R(1) / (R(288) * total * total));
    R term1_num = R(2) * (alpha * alpha) * (x - R(1)) + alpha * beta * (x - R(1)) - x * (beta * beta);
    R axbx = alpha * (x - R(1)) + beta * x;
    R term1_den = sqrt(R(2) * alpha / beta) * m_pow(total, LIT(1.5)) * axbx * axbx;
    R term1 = term1_num / term1_den;
    R term2 = LIT(0.5) * m_log(alpha / (total * x));
    R term3 = sqrt(R(8) * alpha * beta / total) / (beta * x + alpha * (x - R(1)));
    R term4 = m_pow(beta * m_log(beta / (total * (R(1) - x))) + alpha * m_log(alpha / (total * x)), LIT(-1.5));
    R term1234 = term1 + term2 * (term3 + (x < mean ? term4 : -term4));
    return stirling * prefactor * term1234;
}

// -(d/dalpha cdf(x; alpha, beta)) / pdf(x; alpha, beta) / (1 - x), total = alpha + beta.
R dirichlet_grad_one(R x, R alpha, R total)
{
    R beta = total - alpha;
    R boundary = total * x * (R(1) - x);
    if (x <= LIT(0.5) && boundary < LIT(2.5))
        return beta_grad_alpha_small(x, alpha, beta);
    if (x >= LIT(0.5) && boundary < LIT(0.75))
        return -beta_grad_beta_small(R(1) - x, beta, alpha);
    if (alpha > R(6) && beta > R(6))
        return beta_grad_alpha_mid(x, alpha, beta);
    static const R c[2][3][3][4] = {
        {{{LIT(1.003668233), LIT(-0.01061107488), LIT(-0.0657888334), LIT(0.01201642863)},
          {LIT(0.6336835991), LIT(-0.3557432599), LIT(0.05486251648), LIT(-0.001465281033)},
          {LIT(-0.03276231906), LIT(0.004474107445), LIT(0.002429354597), LIT(-0.0001557569013)}},
         {{LIT(0.221950385), LIT(-0.3187676331), LIT(0.01799915743), LIT(0.01074823814)},
          {LIT(-0.2951249643), LIT(0.06219954479), LIT(0.01535556598), LIT(0.001550077057)},
          {LIT(0.02155310298), LIT(0.004170831599), LIT(0.001292462449), LIT(6.976601077e-05)}},
         {{LIT(-0.05980841433), LIT(0.008441916499), LIT(0.01085618172), LIT(0.002319392565)},
          {LIT(0.02911413504), LIT(0.01400243777), LIT(-0.002721828457), LIT(0.000751041181)},
          {LIT(0.005900514878), LIT(-0.001936558688), LIT(-9.495446725e-06), LIT(5.385558597e-05)}}},
        {{{LIT(1.0), LIT(-0.02924021934), LIT(-0.04438342661), LIT(0.007285809825)},
          {LIT(0.6357567472), LIT(-0.3473456711), LIT(0.05454656494), LIT(-0.002407477521)},
          {LIT(-0.03301322327), LIT(0.004845219414), LIT(0.00231480583), LIT(-0.0002307248149)}},
         {{LIT(0.5925320577), LIT(-0.1757678135), LIT(0.01505928619), LIT(0.000564515273)},
          {LIT(0.1014815858), LIT(-0.06589186703), LIT(0.01272886114), LIT(-0.0007316646956)},
          {LIT(-0.007258481865), LIT(0.001096195486), LIT(0.0003934994223), LIT(-4.12701925e-05)}},
         {{LIT(0.06469649321), LIT(-0.0236701437), LIT(0.002902096474), LIT(-5.896963079e-05)},
          {LIT(0.001925008108), LIT(-0.002869809258), LIT(0.0008000589141), LIT(-6.063713228e-05)},
          {LIT(-0.0003477407336), LIT(6.959756487e-05), LIT(1.097287507e-05), LIT(-1.650964693e-06)}}},
    };
    R u = m_log(x);
    R a = m_log(alpha) - u;
    R b = m_log(total) - a;
    R pow_u[3] = {R(1), u, u * u};
    R pow_a[3] = {R(1), a, a * a};
    R pp = R(0), qq = R(0);
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            R ua = pow_u[i] * pow_a[j];
            pp += ua * (c[0][i][j][0] + b * (c[0][i][j][1] + b * (c[0][i][j][2] + b * c[0][i][j][3])));
            qq += ua * (c[1][i][j][0] + b * (c[1][i][j][1] + b * (c[1][i][j][2] + b * c[1][i][j][3])));
        }
    }
    R approx = x * (digamma_one(total) - digamma_one(alpha)) / beta;
    return pp / qq * approx;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    for (uint i = id.x; i < p.n; i += p.stride)
    {
        Stream s = { i, 0u };
        R v;
        switch (OP)
        {
        case SAMPLE_GAMMA:
        {
            v = sample_gamma(p.a[i], s);
#if defined(T_f64)
            R tiny = asdouble(0u, 0x00100000u);
#else
            R tiny = asfloat(0x00800000u);
#endif
            v = isnan(v) || !(v < tiny) ? v : tiny;  // (as PyTorch: never 0; NaN stays)
            break;
        }
        case SAMPLE_LOG_GAMMA: v = sample_log_gamma(p.a[i], s); break;
        case SAMPLE_POISSON: v = sample_poisson(p.a[i], s); break;
        case SAMPLE_BINOMIAL: v = sample_binomial(p.a[i], p.b[i], s); break;
        case SAMPLE_GAMMA_GRAD: v = standard_gamma_grad_one(p.a[i], p.b[i]); break;
        default: v = dirichlet_grad_one(p.a[i], p.b[i], p.c[i]); break;  // SAMPLE_DIRICHLET_GRAD
        }
        p.out[i] = v;
    }
}
