// Special functions for device "vk" (M14): Bessel functions, incomplete
// gamma, zeta and polygamma, erfcx/ndtri/log_ndtr, the Airy function and the
// orthogonal polynomials, all in shaders/special.slang (a port of the CPU
// implementations, so results agree with the CPU). Integer inputs compute
// in the default float dtype, as on the CPU.

#include "vk.h"

#include "opcodes.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <iterator>
#include <string>
#include <utility>

namespace vk {
namespace {

bool float_kernel(at::ScalarType t) {
  return t == at::kFloat || (t == at::kDouble && has_feature(kFeatureFloat64));
}

// A vk operand: vk tensors as they are, 0-dim CPU tensors (Python numbers)
// copied over.
std::optional<at::Tensor> operand(const at::Tensor& t, const at::Device& device) {
  if (is_vk(t)) return t;
  if (t.dim() == 0 && t.device().is_cpu()) return t.to(device);
  return std::nullopt;
}

// The module (shaders/special.slang's axis G) holding function `code`.
const char* group_of(uint32_t code) {
  switch (code) {
    case op::SPECIAL_I0: case op::SPECIAL_I0E: case op::SPECIAL_I1: case op::SPECIAL_I1E:
    case op::SPECIAL_BESSEL_J0: case op::SPECIAL_BESSEL_J1: case op::SPECIAL_BESSEL_Y0: case op::SPECIAL_BESSEL_Y1:
      return "bes";
    case op::SPECIAL_MODIFIED_I0: case op::SPECIAL_MODIFIED_I1: case op::SPECIAL_MODIFIED_K0:
    case op::SPECIAL_MODIFIED_K1: case op::SPECIAL_SCALED_K0: case op::SPECIAL_SCALED_K1:
    case op::SPECIAL_SPHERICAL_J0: case op::SPECIAL_AIRY_AI:
      return "mod";
    case op::SPECIAL_NDTRI: case op::SPECIAL_POLYGAMMA: case op::SPECIAL_ZETA: case op::SPECIAL_IGAMMA:
    case op::SPECIAL_IGAMMAC:
      return "gam";
    case op::SPECIAL_ERFCX: case op::SPECIAL_LOG_NDTR:
      return "erf";
    default:
      return "poly";
  }
}

// The (computation) dtypes the CPU's kernel of special function `code`
// rejects: complex for all; half precision for all but i0, i0e, i1, i1e,
// polygamma, igamma and igammac; integers and bools for igamma / igammac.
void reject_like_cpu(uint32_t code, at::ScalarType t) {
  bool half_ok = code == op::SPECIAL_I0 || code == op::SPECIAL_I0E || code == op::SPECIAL_I1 ||
                 code == op::SPECIAL_I1E || code == op::SPECIAL_POLYGAMMA || code == op::SPECIAL_IGAMMA ||
                 code == op::SPECIAL_IGAMMAC;
  bool ints_ok = code != op::SPECIAL_IGAMMA && code != op::SPECIAL_IGAMMAC;
  reject_dtype("special function", t,
               at::isComplexType(t) || (is_half(t) && !half_ok) || (is_integral_or_bool(t) && !ints_ok));
}

// out = special function `code` of a (and b), with parameter `param`.
bool special_gpu(uint32_t code, const at::Tensor& a, const at::Tensor* b, double param, const at::Tensor& out) {
  reject_like_cpu(code, b != nullptr ? at::result_type(a, *b) : a.scalar_type());
  if (!is_vk(out)) {
    return false;
  }
  std::vector<int64_t> shape = b != nullptr ? at::infer_size(a.sizes(), b->sizes()) : a.sizes().vec();
  check_inplace_shape(out, shape, {&a, b});
  at::native::resize_output(out, shape);
  at::ScalarType dtype = widened(out.scalar_type());
  if (!float_kernel(dtype)) {
    return false;
  }
  auto x = operand(a, out.device());
  std::optional<at::Tensor> y;
  if (b != nullptr) {
    y = operand(*b, out.device());
    if (!y) return false;
  }
  if (!x) {
    return false;
  }
  at::Tensor xa = to_dtype(*x, dtype);
  at::Tensor yb = y ? to_dtype(*y, dtype) : at::Tensor();
  std::string kernel = std::string("special_") + dtype_label(dtype) + "_" + group_of(code);
  with_output(out, dtype, [&](const at::Tensor& target) {
    launch_strided(kernel.c_str(), {code}, target, &xa, y ? &yb : nullptr, {param});
  });
  return true;
}

// The dtype of a functional result: floats keep theirs, integers and bools
// become the default float dtype.
at::ScalarType float_result(at::ScalarType t) {
  return at::isFloatingType(t) ? t : at::typeMetaToScalarType(c10::get_default_dtype());
}

struct Special {
  const char* name;  // aten op (functional; "out" is the out= overload)
  uint32_t code;
};

constexpr Special kUnary[] = {
    {"aten::i0", op::SPECIAL_I0},
    {"aten::special_i0e", op::SPECIAL_I0E},
    {"aten::special_i1", op::SPECIAL_I1},
    {"aten::special_i1e", op::SPECIAL_I1E},
    {"aten::special_ndtri", op::SPECIAL_NDTRI},
    {"aten::special_erfcx", op::SPECIAL_ERFCX},
    {"aten::special_log_ndtr", op::SPECIAL_LOG_NDTR},
    {"aten::special_bessel_j0", op::SPECIAL_BESSEL_J0},
    {"aten::special_bessel_j1", op::SPECIAL_BESSEL_J1},
    {"aten::special_bessel_y0", op::SPECIAL_BESSEL_Y0},
    {"aten::special_bessel_y1", op::SPECIAL_BESSEL_Y1},
    {"aten::special_modified_bessel_i0", op::SPECIAL_MODIFIED_I0},
    {"aten::special_modified_bessel_i1", op::SPECIAL_MODIFIED_I1},
    {"aten::special_modified_bessel_k0", op::SPECIAL_MODIFIED_K0},
    {"aten::special_modified_bessel_k1", op::SPECIAL_MODIFIED_K1},
    {"aten::special_scaled_modified_bessel_k0", op::SPECIAL_SCALED_K0},
    {"aten::special_scaled_modified_bessel_k1", op::SPECIAL_SCALED_K1},
    {"aten::special_spherical_bessel_j0", op::SPECIAL_SPHERICAL_J0},
    {"aten::special_airy_ai", op::SPECIAL_AIRY_AI},
};

constexpr Special kBinary[] = {
    {"aten::special_zeta", op::SPECIAL_ZETA},
    {"aten::igamma", op::SPECIAL_IGAMMA},
    {"aten::igammac", op::SPECIAL_IGAMMAC},
    {"aten::special_chebyshev_polynomial_t", op::SPECIAL_CHEBYSHEV_T},
    {"aten::special_chebyshev_polynomial_u", op::SPECIAL_CHEBYSHEV_U},
    {"aten::special_chebyshev_polynomial_v", op::SPECIAL_CHEBYSHEV_V},
    {"aten::special_chebyshev_polynomial_w", op::SPECIAL_CHEBYSHEV_W},
    {"aten::special_shifted_chebyshev_polynomial_t", op::SPECIAL_SHIFTED_T},
    {"aten::special_shifted_chebyshev_polynomial_u", op::SPECIAL_SHIFTED_U},
    {"aten::special_shifted_chebyshev_polynomial_v", op::SPECIAL_SHIFTED_V},
    {"aten::special_shifted_chebyshev_polynomial_w", op::SPECIAL_SHIFTED_W},
    {"aten::special_hermite_polynomial_h", op::SPECIAL_HERMITE_H},
    {"aten::special_hermite_polynomial_he", op::SPECIAL_HERMITE_HE},
    {"aten::special_laguerre_polynomial_l", op::SPECIAL_LAGUERRE_L},
    {"aten::special_legendre_polynomial_p", op::SPECIAL_LEGENDRE_P},
};

template <size_t I>
at::Tensor& unary_out(const at::Tensor& self, at::Tensor& out) {
  if (!special_gpu(kUnary[I].code, self, nullptr, 0.0, out)) {
    on_cpu(kUnary[I].name, "out", self, out);
  }
  return out;
}

template <size_t I>
at::Tensor unary(const at::Tensor& self) {
  at::Tensor out = at::empty({0}, self.options().dtype(float_result(self.scalar_type())));
  return unary_out<I>(self, out);
}

template <size_t I>
at::Tensor& binary_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  if (!special_gpu(kBinary[I].code, self, &other, 0.0, out)) {
    on_cpu(kBinary[I].name, "out", self, other, out);
  }
  return out;
}

template <size_t I>
at::Tensor binary(const at::Tensor& self, const at::Tensor& other) {
  at::Tensor like = is_vk(self) ? self : other;
  at::Tensor out = at::empty({0}, like.options().dtype(float_result(at::result_type(self, other))));
  return binary_out<I>(self, other, out);
}

at::Tensor& polygamma_out(int64_t n, const at::Tensor& self, at::Tensor& out) {
  TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
  if (!special_gpu(op::SPECIAL_POLYGAMMA, self, nullptr, static_cast<double>(n), out)) {
    on_cpu("aten::polygamma", "out", n, self, out);
  }
  return out;
}

at::Tensor polygamma(int64_t n, const at::Tensor& self) {
  at::Tensor out = at::empty({0}, self.options().dtype(float_result(self.scalar_type())));
  return polygamma_out(n, self, out);
}

at::Tensor& i0_(at::Tensor& self) {
  return unary_out<0>(self, self);
}

at::Tensor& igamma_(at::Tensor& self, const at::Tensor& other) {
  return binary_out<1>(self, other, self);
}

at::Tensor& igammac_(at::Tensor& self, const at::Tensor& other) {
  return binary_out<2>(self, other, self);
}

std::string strip(const char* name) {
  return std::string(name).substr(6);  // "aten::"
}

template <size_t... I>
void register_unary(torch::Library& m, std::index_sequence<I...>) {
  (m.impl(strip(kUnary[I].name).c_str(), &unary<I>), ...);
  (m.impl((strip(kUnary[I].name) + ".out").c_str(), &unary_out<I>), ...);
}

template <size_t... I>
void register_binary(torch::Library& m, std::index_sequence<I...>) {
  (m.impl(strip(kBinary[I].name).c_str(), &binary<I>), ...);
  (m.impl((strip(kBinary[I].name) + ".out").c_str(), &binary_out<I>), ...);
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  register_unary(m, std::make_index_sequence<std::size(kUnary)>());
  register_binary(m, std::make_index_sequence<std::size(kBinary)>());
  m.impl("polygamma", &vk::polygamma);
  m.impl("polygamma.out", &polygamma_out);
  m.impl("i0_", &vk::i0_);
  m.impl("igamma_", &vk::igamma_);
  m.impl("igammac_", &vk::igammac_);
}
