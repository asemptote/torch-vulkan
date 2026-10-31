// float64 elementary functions.
//
// SPIR-V's GLSL.std.450 only provides exp/log/sin/cos/tanh/pow for 16/32-bit
// floats, so float64 versions are implemented here: range reduction plus
// Taylor/atanh series carried far enough that truncation error is far below
// one ULP. Expect results within a few ULP of a correctly rounded libm.
//
// sin/cos/tan reduce arguments with a 3-part Cody-Waite pi/2 for |x| < 1e6
// (exact up to ~1.6e6), and by Payne-Hanek beyond.
//
// Literals carry the `l` suffix: unsuffixed literals are float32 in Slang.

static const double LN2 = 0.6931471805599453l;
static const double LN2_HI = 6.93147180369123816490e-01l;  // first 32 bits of ln 2
static const double LN2_LO = 1.90821492927058770002e-10l;  // ln 2 - LN2_HI
static const double INV_LN2 = 1.4426950408889634l;
static const double SQRT2 = 1.4142135623730951l;
static const double TWO_OVER_PI = 0.6366197723675814l;
static const double PIO2_1 = 1.57079632673412561417e+00l;  // first 33 bits of pi/2
static const double PIO2_2 = 6.07710050630396597660e-11l;  // next 33 bits
static const double PIO2_3 = 2.02226624871116645580e-21l;  // next 53 bits

double f64_inf() { return bit_cast<double>(0x7ff0000000000000ull); }
double f64_nan() { return bit_cast<double>(0x7ff8000000000000ull); }

// x * 2^n for any int n (scaling in steps through the subnormal/overflow range).
double ldexp64(double x, int n)
{
    double two_1023 = bit_cast<double>(uint64_t(0x7fe) << 52);         // 2^1023
    double two_m969 = bit_cast<double>(uint64_t(1023 - 969) << 52);    // 2^-969
    if (n > 1023)
    {
        x *= two_1023;
        n -= 1023;
        if (n > 1023)
        {
            x *= two_1023;
            n = min(n - 1023, 1023);
        }
    }
    else if (n < -1022)
    {
        x *= two_m969;
        n += 969;
        if (n < -1022)
        {
            x *= two_m969;
            n = max(n + 969, -1022);
        }
    }
    return x * bit_cast<double>(uint64_t(n + 1023) << 52);
}

// e^r - 1 for |r| <= ln(2)/2 (Taylor through r^13; next term < 4e-18).
double expm1_poly(double r)
{
    double p = 1.6059043836821613e-10l;
    p = p * r + 2.08767569878681e-09l;
    p = p * r + 2.505210838544172e-08l;
    p = p * r + 2.755731922398589e-07l;
    p = p * r + 2.7557319223985893e-06l;
    p = p * r + 2.48015873015873e-05l;
    p = p * r + 0.0001984126984126984l;
    p = p * r + 0.001388888888888889l;
    p = p * r + 0.008333333333333333l;
    p = p * r + 0.041666666666666664l;
    p = p * r + 0.16666666666666666l;
    p = p * r + 0.5l;
    p = p * r + 1.0l;
    return p * r;
}

double exp64(double x)
{
    if (isnan(x)) return x;
    if (x > 709.782712893383973096l) return f64_inf();
    if (x < -745.13321910194110842l) return 0.0l;
    double k = round(x * INV_LN2);
    double r = (x - k * LN2_HI) - k * LN2_LO;  // |r| <= ln(2)/2
    return ldexp64(1.0l + expm1_poly(r), int(k));
}

double exp2_64(double x)
{
    if (isnan(x)) return x;
    if (x >= 1024.0l) return f64_inf();
    if (x < -1075.0l) return 0.0l;
    double k = round(x);
    return ldexp64(1.0l + expm1_poly((x - k) * LN2), int(k));
}

double expm1_64(double x)
{
    if (abs(x) <= 0.34657359027997264l) return expm1_poly(x);
    return exp64(x) - 1.0l;  // |result| >= 0.29 here, so the subtraction loses < 2 ULP
}

// Splits finite x > 0 into x = m * 2^e with m in [sqrt(1/2), sqrt(2)) and
// returns log(m) (atanh series through s^21; next term < 1e-18).
double log_mantissa(double x, out int e)
{
    uint64_t bits = bit_cast<uint64_t>(x);
    int biased = int((bits >> 52) & 0x7ff);
    e = 0;
    if (biased == 0)  // subnormal: scale into the normal range first
    {
        bits = bit_cast<uint64_t>(x * bit_cast<double>(uint64_t(1023 + 54) << 52));
        biased = int((bits >> 52) & 0x7ff);
        e = -54;
    }
    e += biased - 1023;
    double m = bit_cast<double>((bits & 0x000fffffffffffffull) | 0x3ff0000000000000ull);
    if (m > SQRT2)
    {
        m *= 0.5l;
        e += 1;
    }
    double f = m - 1.0l;
    double s = f / (2.0l + f);
    double z = s * s;
    double p = 0.047619047619047616l;
    p = p * z + 0.05263157894736842l;
    p = p * z + 0.058823529411764705l;
    p = p * z + 0.06666666666666667l;
    p = p * z + 0.07692307692307693l;
    p = p * z + 0.09090909090909091l;
    p = p * z + 0.1111111111111111l;
    p = p * z + 0.14285714285714285l;
    p = p * z + 0.2l;
    p = p * z + 0.3333333333333333l;
    return 2.0l * s + 2.0l * s * (z * p);
}

double log64(double x)
{
    if (isnan(x)) return x;
    if (x < 0.0l) return f64_nan();
    if (x == 0.0l) return -f64_inf();
    if (isinf(x)) return x;
    int e;
    double lm = log_mantissa(x, e);
    return double(e) * LN2_HI + (lm + double(e) * LN2_LO);
}

double log2_64(double x)
{
    if (isnan(x)) return x;
    if (x < 0.0l) return f64_nan();
    if (x == 0.0l) return -f64_inf();
    if (isinf(x)) return x;
    int e;
    double lm = log_mantissa(x, e);
    return double(e) + lm * INV_LN2;  // exact for powers of two
}

// sin(r), cos(r) for |r| <= pi/4 (Taylor through r^17 / r^18).
double sin_poly(double r)
{
    double z = r * r;
    double p = 2.8114572543455206e-15l;
    p = p * z - 7.647163731819816e-13l;
    p = p * z + 1.6059043836821613e-10l;
    p = p * z - 2.505210838544172e-08l;
    p = p * z + 2.7557319223985893e-06l;
    p = p * z - 0.0001984126984126984l;
    p = p * z + 0.008333333333333333l;
    p = p * z - 0.16666666666666666l;
    return r + r * z * p;
}

double cos_poly(double r)
{
    double z = r * r;
    double p = 1.5619206968586225e-16l;
    p = p * z - 4.779477332387385e-14l;
    p = p * z + 1.1470745597729725e-11l;
    p = p * z - 2.08767569878681e-09l;
    p = p * z + 2.755731922398589e-07l;
    p = p * z - 2.48015873015873e-05l;
    p = p * z + 0.001388888888888889l;
    p = p * z - 0.041666666666666664l;
    p = p * z + 0.5l;
    return 1.0l - z * p;
}

// 2/pi in 64-bit words, a zero word first (checked against pi computed
// with Machin's formula).
static const uint64_t TWO_OVER_PI_64[25] = {
    0x0000000000000000ull, 0xa2f9836e4e441529ull, 0xfc2757d1f534ddc0ull, 0xdb6295993c439041ull,
    0xfe5163abdebbc561ull, 0xb7246e3a424dd2e0ull, 0x06492eea09d1921cull, 0xfe1deb1cb129a73eull,
    0xe88235f52ebb4484ull, 0xe99c7026b45f7e41ull, 0x3991d639835339f4ull, 0x9c845f8bbdf9283bull,
    0x1ff897ffde05980full, 0xef2f118b5a0a6d1full, 0x6d367ecf27cb09b7ull, 0x4f463f669e5fea2dull,
    0x7527bac7ebe5f17bull, 0x3d0739f78a5292eaull, 0x6bfb5fb11f8d5d08ull, 0x56033046fc7b6babull,
    0xf0cfbc209af4361dull, 0xa9e391615ee61b08ull, 0x6599855f14a06840ull, 0x8dffd8804d732731ull,
    0x06061556ca73a8c9ull};

// The 128-bit product a b as (hi, lo).
void mul64x64(uint64_t a, uint64_t b, out uint64_t hi, out uint64_t lo)
{
    uint64_t a0 = a & 0xffffffffull, a1 = a >> 32, b0 = b & 0xffffffffull, b1 = b >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffull) + (p10 & 0xffffffffull);
    lo = (mid << 32) | (p00 & 0xffffffffull);
    hi = a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

// 192 bits of the table from bit g (bit 0 = the first bit of word 0).
uint64_t two_over_pi_bits(uint g, uint word)
{
    uint w = g / 64u + word, s = g % 64u;
    return s == 0u ? TWO_OVER_PI_64[w] : (TWO_OVER_PI_64[w] << s) | (TWO_OVER_PI_64[w + 1] >> (64u - s));
}

// Payne-Hanek for finite |x| >= 2^19: with x = M 2^E (M the 53-bit
// significand), x 2/pi mod 4 needs only the bits of 2/pi from 2^-(E-1) on;
// M times 192 of them gives the quadrant and a 128-bit fraction, rounded to
// the nearest quadrant and scaled by pi/2 in double-double.
double reduce_pio2_large(double x, out int quadrant)
{
    uint64_t bits = bit_cast<uint64_t>(x);
    int E = int((bits >> 52) & 0x7ffull) - 1075;
    uint64_t M = (bits & 0xfffffffffffffull) | 0x10000000000000ull;
    uint g = uint(E + 62);  // the bit 2^-(E-1) of 2/pi (E >= -33 here)
    uint64_t h0, l0, h1, l1, h2, l2;
    mul64x64(M, two_over_pi_bits(g, 0u), h0, l0);
    mul64x64(M, two_over_pi_bits(g, 1u), h1, l1);
    mul64x64(M, two_over_pi_bits(g, 2u), h2, l2);
    // x 2/pi = (M V) 2^-190 mod 4: the words of M V from bit 64 and bit 128.
    uint64_t w64 = l1 + h2;
    uint64_t w128 = l0 + h1 + (w64 < l1 ? 1ull : 0ull);
    int q = int(uint(w128 >> 62));
    uint64_t fhi = (w128 << 2) | (w64 >> 62), flo = (w64 << 2) | (l2 >> 62);
    double sign = 1.0l;
    if ((fhi >> 63) != 0ull)  // fraction >= 1/2: the next quadrant, minus the complement
    {
        q += 1;
        sign = -1.0l;
        fhi = ~fhi;
        flo = ~flo + 1ull;
        if (flo == 0ull) fhi += 1ull;
    }
    // The fraction as a double-double h + l (units of 2^-64), times pi/2.
    double dh = double(fhi);
    int64_t rem = int64_t(fhi - uint64_t(dh));
    double h = dh * 5.421010862427522e-20l;  // 2^-64
    double l = (double(rem) + double(flo) * 5.421010862427522e-20l) * 5.421010862427522e-20l;
    const double P1 = 1.5707963267948966l, P2 = 6.123233995736766e-17l;
    double p = h * P1;
    double r = p + (fma(h, P1, -p) + (h * P2 + l * P1));
    if (x < 0.0l)
    {
        r = -r;
        q = -q;
    }
    quadrant = q & 3;
    return sign * r;
}

// x = k * pi/2 + r with |r| <= ~pi/4; returns r and the quadrant k mod 4.
double reduce_pio2(double x, out int quadrant)
{
    if (!(abs(x) < 1.0e6l))
        return reduce_pio2_large(x, quadrant);
    double k = round(x * TWO_OVER_PI);
    quadrant = int(int64_t(k) & 3);
    return ((x - k * PIO2_1) - k * PIO2_2) - k * PIO2_3;
}

double sin64(double x)
{
    if (isnan(x) || isinf(x)) return f64_nan();
    int q;
    double r = reduce_pio2(x, q);
    double v = (q & 1) == 0 ? sin_poly(r) : cos_poly(r);
    return (q & 2) == 0 ? v : -v;
}

double cos64(double x)
{
    if (isnan(x) || isinf(x)) return f64_nan();
    int q;
    double r = reduce_pio2(x, q);
    double v = (q & 1) == 0 ? cos_poly(r) : sin_poly(r);
    return (q == 0 || q == 3) ? v : -v;
}

double tan64(double x)
{
    if (isnan(x) || isinf(x)) return f64_nan();
    int q;
    double r = reduce_pio2(x, q);
    double s = sin_poly(r), c = cos_poly(r);
    return (q & 1) == 0 ? s / c : -c / s;
}

double tanh64(double x)
{
    if (isnan(x)) return x;
    double ax = abs(x);
    if (ax > 22.0l) return x > 0.0l ? 1.0l : -1.0l;
    double t = expm1_64(2.0l * ax);
    double r = t / (t + 2.0l);
    return x < 0.0l ? -r : r;
}

double sigmoid64(double x)
{
    return 1.0l / (1.0l + exp64(-x));
}
