"""Generates crates/vkcore/shaders/_special.slang from PyTorch's CPU special
functions (ATen/native/Math.h of the installed torch, derived from Cephes,
SciPy and Boost; BSD/Boost licensed, see the notices in that header).

The functions are translated mechanically: T/scalar_t become FT (float or
double per _fmath.slang), std:: calls become the m_* functions, floating
literals become LIT(...) so float32 modules contain no doubles, and the
Chebyshev/rational helpers that take array pointers become macros. Run it
after upgrading torch and review the diff:

    python tools/port_special.py
"""

import pathlib
import re

import torch

MATH_H = pathlib.Path(torch.__file__).parent / "include" / "ATen" / "native" / "Math.h"
OUT = pathlib.Path(__file__).resolve().parent.parent / "crates" / "vkcore" / "shaders" / "_special.slang"

# Functions taken as they are (after translation), in dependency order.
FUNCTIONS = [
    "zeta", "trigamma", "calc_ndtri", "erfcx_y100", "calc_erfcx", "calc_log_ndtr",
    "lanczos_sum_expg_scaled", "_igam_helper_fac", "_igam_helper_series", "_igamc_helper_series",
    "_igam_helper_asymptotic_series", "_igamc_helper_continued_fraction", "calc_igammac", "calc_igamma",
    "airy_ai_forward", "bessel_j0_forward", "bessel_j1_forward", "bessel_y0_forward", "bessel_y1_forward",
    "chebyshev_polynomial_t_forward", "chebyshev_polynomial_u_forward", "chebyshev_polynomial_v_forward",
    "chebyshev_polynomial_w_forward", "hermite_polynomial_h_forward", "hermite_polynomial_he_forward",
    "laguerre_polynomial_l_forward", "legendre_polynomial_p_forward", "modified_bessel_i0_forward",
    "modified_bessel_i1_forward", "modified_bessel_k0_forward", "modified_bessel_k1_forward",
    "scaled_modified_bessel_k0_forward", "scaled_modified_bessel_k1_forward",
    "shifted_chebyshev_polynomial_t_forward", "shifted_chebyshev_polynomial_u_forward",
    "shifted_chebyshev_polynomial_v_forward", "shifted_chebyshev_polynomial_w_forward",
    "spherical_bessel_j0_forward",
]
# Chebyshev coefficient tables (float and double variants where they differ).
CHEB = ["chebyshev_coefficients_i0e_A", "chebyshev_coefficients_i0e_B", "chebyshev_coefficients_i1e_A",
        "chebyshev_coefficients_i1e_B"]


def find_functions(text, name):
    """All definitions of `name` as (signature line, body text incl. braces)."""
    found = []
    for m in re.finditer(r"^[^\n;]*\b" + re.escape(name) + r"\s*\(([^)]*)\)[^;{]*\{", text, re.M):
        start = m.start()
        # Include a preceding return-type line ("inline ... T\nname(").
        prev = text.rfind("\n", 0, start - 1)
        line_before = text[prev + 1:start]
        if line_before.strip().startswith(("inline", "C10_HOST_DEVICE")) and "(" not in line_before:
            start = prev + 1
        depth, i = 0, m.end() - 1
        while True:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        found.append((text[start:m.end()], text[m.end():i + 1], m.group(1)))
    return found


FLOAT_LIT = re.compile(r"(?<![\w.])((?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)[fFlL]?(?![\w.])")

CALLS = {
    "std::exp": "m_exp", "std::log": "m_log", "std::log1p": "m_log1p", "std::expm1": "m_expm1",
    "std::sin": "m_sin", "std::cos": "m_cos", "std::tan": "m_tan", "std::sinh": "m_sinh", "std::cosh": "m_cosh",
    "std::pow": "m_pow", "::pow": "m_pow", "std::lgamma": "m_lgamma", "std::erfc": "m_erfc", "std::erf": "m_erf",
    "std::exp2": "m_exp2", "std::sqrt": "sqrt", "std::abs": "abs", "std::fabs": "abs", "::fabs": "abs",
    "std::floor": "floor", "std::isnan": "isnan", "std::isinf": "isinf", "std::copysign": "m_copysign",
    "sinf": "m_sin", "sin": "m_sin", "exp": "m_exp", "log": "m_log", "cos": "m_cos", "std::trunc": "trunc", "std::round": "round", "std::ceil": "ceil",
    "at::_isnan": "isnan", "std::acos": "m_acos", "::sqrt": "sqrt", "::log": "m_log", "::exp": "m_exp",
}


def translate(code):
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r"__ubsan_ignore_\w+__", "", code)
    code = re.sub(r"\b(C10_HOST_DEVICE|inline|JITERATOR_HOST_DEVICE)\b\s*", "", code)
    code = re.sub(r"\[\[maybe_unused\]\]\s*", "", code)
    code = re.sub(r"typename std::enable_if_t<[^,]*,\s*T>\s*", "FT ", code)
    code = re.sub(r"\bstd::numeric_limits<\w+>::infinity\(\)", "m_inf()", code)
    code = re.sub(r"\bstd::numeric_limits<\w+>::quiet_NaN\(\)", "m_nan()", code)
    code = re.sub(r"\bstd::numeric_limits<\w+>::epsilon\(\)", "M_EPS", code)
    code = re.sub(r"\bstd::numeric_limits<\w+>::max\(\)", "M_MAX", code)
    code = re.sub(r"\bstd::numeric_limits<\w+>::min\(\)", "M_TINY", code)
    code = re.sub(r"\bc10::pi<\w+>", "K_PI", code)
    code = re.sub(r"\bc10::frac_sqrt_2<\w+>", "K_SQRT1_2", code)
    code = re.sub(r"\bINFINITY\b", "m_inf()", code)
    code = re.sub(r"std::is_same_v<\s*\w+\s*,\s*double\s*>", "IS_F64", code)
    code = re.sub(r"static_cast<(\w+)>", r"\1", code)
    code = re.sub(r"using acc_t = [^;]*;", "", code)
    code = re.sub(r"getHermitianLimit<\w+>\(\)", "(IS_F64 ? 512 : 128)", code)
    code = re.sub(r"\b(scalar_t|acc_t|T)\b", "FT", code)
    code = re.sub(r"\bFT\{([^{}]*)\}", r"FT(\1)", code)
    code = re.sub(r"\buint8_t\b|\bsize_t\b", "int", code)
    code = re.sub(r"\bstatic constexpr\b", "static const", code)
    code = re.sub(r"\bconstexpr\b", "const", code)
    # "static FT X = ...;" scalars: plain constants.
    code = re.sub(r"\bstatic (FT|int) (\w+) =", r"const \1 \2 =", code)
    for k in sorted(CALLS, key=len, reverse=True):
        code = re.sub(r"(?<![\w:])" + re.escape(k) + r"\s*\(", CALLS[k] + "(", code)
    code = FLOAT_LIT.sub(lambda m: f"LIT({m.group(1)})", code)
    return code


def cheb_tables(text):
    """FT cheb_<name>(FT x): chbevl over the table, per precision."""
    out = []
    for name in CHEB:
        defs = find_functions(text, name)
        variants = {}
        for sig, body, _ in defs:
            kind = "double" if "is_same_v<double" in sig else "float" if "is_same_v<float" in sig else "both"
            coeffs = re.search(r"coeff\[\]\s*=\s*\{(.*?)\};", body, re.S).group(1)
            nums = FLOAT_LIT.findall(re.sub(r"/\*.*?\*/", "", coeffs, flags=re.S))
            signs = re.findall(r"([-+]?)\s*(?:\d+\.\d*|\.\d+|\d+[eE])", re.sub(r"/\*.*?\*/", "", coeffs, flags=re.S))
            values = [f"{s}LIT({n})" for s, n in zip(signs, nums)]
            variants[kind] = values
        short = name.replace("chebyshev_coefficients_", "")
        out.append(f"FT cheb_{short}(FT x)\n{{")
        if "both" in variants:
            out.append(f"    static const FT c[] = {{{', '.join(variants['both'])}}};")
            out.append(f"    CHBEVL(x, c, {len(variants['both'])});")
        else:
            out.append("#if defined(T_f64) || defined(FMATH_F64)")
            out.append(f"    static const FT c[] = {{{', '.join(variants['double'])}}};")
            out.append(f"    CHBEVL(x, c, {len(variants['double'])});")
            out.append("#else")
            out.append(f"    static const FT c[] = {{{', '.join(variants['float'])}}};")
            out.append(f"    CHBEVL(x, c, {len(variants['float'])});")
            out.append("#endif")
        out.append("}\n")
    return "\n".join(out)


HEADER = """// GENERATED by tools/port_special.py from PyTorch's ATen/native/Math.h
// (torch {version}); do not edit by hand. Special functions of PyTorch's
// CPU backend (derived from the Cephes Math Library, SciPy and Boost; see
// the license notes in that header), translated to FT = float or double of
// _fmath.slang, which the including file provides (with FMATH_F32/F64).

#if defined(T_f64) || defined(FMATH_F64)
static const bool IS_F64 = true;
static const FT M_EPS = 2.220446049250313e-16l;
static const FT M_MAX = 1.7976931348623157e308l;
static const FT M_TINY = 2.2250738585072014e-308l;
#else
static const bool IS_F64 = false;
static const FT M_EPS = 1.1920928955078125e-7f;
static const FT M_MAX = 3.4028234663852886e38f;
static const FT M_TINY = 1.1754943508222875e-38f;
#endif

static const FT K_PI = LIT(3.14159265358979323846);
static const FT K_SQRT1_2 = LIT(0.70710678118654752440);

FT m_copysign(FT a, FT b) {{ return m_signbit(b) ? -abs(a) : abs(a); }}

// Polynomial c[0] x^len + ... + c[len] (Cephes polevl).
FT polevl<let N : int>(FT x, FT c[N], int len)
{{
    FT r = FT(0);
    for (int i = 0; i <= len; i++)
        r = r * x + c[i];
    return r;
}}

// Chebyshev series sum_i c[i] T_i(x/2) (Cephes chbevl), c in reverse order.
#define CHBEVL(x, c, len)                       \\
    FT b0 = c[0], b1 = FT(0), b2 = FT(0);       \\
    for (int i = 1; i < len; ++i)               \\
    {{                                           \\
        b2 = b1;                                \\
        b1 = b0;                                \\
        b0 = x * b1 - b2 + c[i];                \\
    }}                                           \\
    return LIT(0.5) * (b0 - b2)

// num(x) / den(x) for coefficient arrays num[0..M], den[0..N] (SciPy ratevl).
#define RATEVL(x, num, M, den, N)                                                   \\
    FT absx = abs(x), y = absx > FT(1) ? FT(1) / x : x;                             \\
    FT na = absx > FT(1) ? num[M] : num[0], da = absx > FT(1) ? den[N] : den[0];    \\
    for (int i = 1; i <= M; i++) na = na * y + (absx > FT(1) ? num[M - i] : num[i]); \\
    for (int i = 1; i <= N; i++) da = da * y + (absx > FT(1) ? den[N - i] : den[i]); \\
    return absx > FT(1) ? m_pow(x, FT(N - M)) * na / da : na / da

"""


def tabulate_erfcx(out):
    """erfcx_y100's 100-case switch of degree-6 polynomials as a coefficient
    table: some drivers crash compiling the switch in float64."""
    start = out.index("FT erfcx_y100(FT y100)")
    end = out.index("return LIT(1.0);", start)
    end = out.index("}", end) + 1
    body = out[start:end]
    cases = re.findall(r"case (\d+): \{\s*FT t = 2\*y100 - (\d+);\s*return (.*?);\s*\}", body, re.S)
    assert len(cases) == 100, len(cases)
    rows = []
    for k, (case, off, expr) in enumerate(cases):
        assert int(case) == k and int(off) == 2 * k + 1
        coeffs = re.findall(r"LIT\(([^)]*)\)", expr)
        assert len(coeffs) == 7, (k, coeffs)
        rows.append("    {" + ", ".join(f"LIT({c})" for c in coeffs) + "},")
    table = "\n".join(rows)
    new = (
        "// Fits of erfcx on [k/100, (k+1)/100] in t = 2 y100 - (2k + 1), constant\n"
        "// term first (the CPU's 100-case switch, as a table).\n"
        f"static const FT ERFCX_Y100[100][7] = {{\n{table}\n}};\n\n"
        "FT erfcx_y100(FT y100)\n{\n"
        "    int k = int(y100);\n"
        "    if (k < 0 || k >= 100)\n        return LIT(1.0);\n"
        "    FT t = 2 * y100 - FT(2 * k + 1);\n"
        "    k = clamp(k, 0, 99);  // (calc_erfcx evaluates both of its arms: keep the row in range)\n"
        "    FT r = ERFCX_Y100[k][6];\n"
        "    for (int j = 0; j < 6; j++)\n        r = ERFCX_Y100[k][5 - j] + r * t;\n"
        "    return r;\n}"
    )
    return out[:start] + new + out[end:]


def main():
    text = MATH_H.read_text(encoding="utf-8")
    parts = [HEADER.format(version=torch.__version__), cheb_tables(text)]
    for name in FUNCTIONS:
        defs = find_functions(text, name)
        if name == "trigamma":
            defs = [d for d in defs if "double x" in d[0]]
        # The integer-n overloads of the polynomials (the FT-n ones convert).
        if name.endswith("polynomial_t_forward") or "polynomial" in name:
            defs = [d for d in defs if "int64_t n" in d[2]]
        assert defs, name
        sig, body, _ = defs[0]
        code = translate(sig + body)
        if len(re.findall(r"\b" + name + r"\(", code)) > 1:
            # Self-recursive (e.g. odd functions calling themselves on -x):
            # shaders cannot recurse, so the recursive call goes to a copy in
            # which it cannot happen (it would return NaN).
            impl = re.sub(r"\b" + name + r"\(", name + "_once(", code, count=1)
            head, rest = impl.split("{", 1)
            rest = re.sub(r"\b" + name + r"\([^;]*\)", "m_nan()", rest)
            code = head + "{" + rest + "\n" + re.sub(r"(?<=[^\w])" + name + r"\((?=-)", name + "_once(", code)
        if name == "trigamma":
            code = code.replace("double", "FT")
        # Loops whose exit depends on the arguments get an iteration cap: a GPU
        # kernel that never finishes hangs the device (no watchdog on most
        # Linux compute drivers), and driver float optimizations may treat
        # NaN/inf comparisons differently from the CPU.
        code = code.replace("while ((i < 9) || (a <= FT(LIT(9.0)))) {",
                            "while (((i < 9) || (a <= FT(LIT(9.0)))) && i < 1000000) {")
        code = code.replace("while (t > M_EPS) {", "for (int guard = 0; t > M_EPS && guard < 100000; guard++) {")
        if name == "lanczos_sum_expg_scaled":
            code = re.sub(r"return ratevl\(.*?\);", "RATEVL(x, lanczos_sum_expg_scaled_num, 12, "
                          "lanczos_sum_expg_scaled_denom, 12);", code, flags=re.S)
        parts.append(code.strip() + "\n")
    out = "\n".join(parts)
    out = tabulate_erfcx(out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    OUT.write_text(out, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({out.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
