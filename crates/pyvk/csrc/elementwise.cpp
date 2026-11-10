// More element-wise aten kernels for device "vk" (M8): parameterized
// activations, clamp, predicates, logical ops, where/masked_fill, lerp,
// addcmul/addcdiv, and compositions of existing kernels (glu, complex/polar,
// angle/sgn, pow with a scalar base, div with a rounding mode).
//
// Kernels: shaders/unary.slang (parameterized codes), compare.slang
// (predicates, logical ops), where.slang and ternary.slang (four-operand
// layout, shaders/_strided4.slang).

#include "opcodes.h"
#include "vk.h"

#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <cmath>
#include <limits>

namespace vk {

namespace {

bool is_float(at::ScalarType t) {
  return t == at::kFloat || (t == at::kDouble && vkt_has_float64());
}

bool is_int(at::ScalarType t) {
  return t == at::kInt || t == at::kLong;
}

// A 0-dim CPU tensor (e.g. a wrapped Python number) used as a scalar operand.
bool cpu_scalar(const at::Tensor& t) {
  return t.dim() == 0 && t.device().is_cpu();
}

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------- parameterized unary

// out = unary op `code` (shaders/unary.slang) of self, computed in out's
// float dtype, with parameters alpha, beta, gamma.
bool param_unary_gpu(uint32_t code, const at::Tensor& self, const at::Tensor& out, double alpha, double beta,
                     double gamma = 0.0, bool check_aliasing = true) {
  resize_elementwise_output(out, self.sizes(), &self, check_aliasing);
  at::ScalarType dtype = out.scalar_type();
  if (!is_vk(self) || !is_vk(out) || !is_float(dtype) || at::isComplexType(self.scalar_type())) {
    return false;
  }
  at::Tensor input = to_dtype(self, dtype);
  launch_strided((std::string("unary_") + dtype_label(dtype)).c_str(), {code}, out, &input, nullptr,
                 {alpha, beta, gamma});
  return true;
}

at::Tensor& gelu_out(const at::Tensor& self, c10::string_view approximate, at::Tensor& out) {
  TORCH_CHECK(approximate == "none" || approximate == "tanh", "approximate argument must be either none or tanh.");
  if (!param_unary_gpu(approximate == "tanh" ? op::UNARY_GELU_TANH : op::UNARY_GELU, self, out, 0, 0)) {
    on_cpu("aten::gelu", "out", self, approximate, out);
  }
  return out;
}

at::Tensor& softplus_out(const at::Tensor& self, const at::Scalar& beta, const at::Scalar& threshold, at::Tensor& out) {
  if (!param_unary_gpu(op::UNARY_SOFTPLUS, self, out, beta.toDouble(), threshold.toDouble())) {
    on_cpu("aten::softplus", "out", self, beta, threshold, out);
  }
  return out;
}

at::Tensor& elu_out(const at::Tensor& self, const at::Scalar& alpha, const at::Scalar& scale,
                    const at::Scalar& input_scale, at::Tensor& out) {
  if (!param_unary_gpu(op::UNARY_ELU, self, out, alpha.toDouble(), scale.toDouble(), input_scale.toDouble())) {
    on_cpu("aten::elu", "out", self, alpha, scale, input_scale, out);
  }
  return out;
}

at::Tensor& hardtanh_out(const at::Tensor& self, const at::Scalar& min, const at::Scalar& max, at::Tensor& out) {
  if (is_int(self.scalar_type()) && is_vk(self) && is_vk(out)) {
    // (as the CPU: integer bounds, so the result keeps the integer dtype)
    return at::clamp_out(out, self, at::Scalar(min.toLong()), at::Scalar(max.toLong()));
  }
  if (!param_unary_gpu(op::UNARY_HARDTANH, self, out, min.toDouble(), max.toDouble())) {
    on_cpu("aten::hardtanh", "out", self, min, max, out);
  }
  return out;
}

at::Tensor& leaky_relu_out(const at::Tensor& self, const at::Scalar& slope, at::Tensor& out) {
  if (!param_unary_gpu(op::UNARY_LEAKY_RELU, self, out, slope.toDouble(), 0)) {
    on_cpu("aten::leaky_relu", "out", self, slope, out);
  }
  return out;
}

// A 16-bit vk self and out of its dtype: f on a float32 copy, rounded once
// into out. False otherwise.
template <typename F>
bool half_unary(const at::Tensor& self, at::Tensor& out, F f) {
  if (!is_vk(self) || !is_vk(out) || !at::isReducedFloatingType(self.scalar_type()) ||
      out.scalar_type() != self.scalar_type()) {
    return false;
  }
  at::Tensor r = f(self.to(at::kFloat));
  resize_elementwise_output(out, self.sizes(), &self);
  out.copy_(r);
  return true;
}

// v rounded to the 16-bit dtype t (the CPU compares 16-bit values with
// shrink thresholds of their own dtype).
double round_to(double v, at::ScalarType t) {
  return t == at::kHalf ? static_cast<double>(static_cast<float>(c10::Half(static_cast<float>(v))))
                        : static_cast<double>(static_cast<float>(c10::BFloat16(static_cast<float>(v))));
}

at::Tensor& softshrink_out(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out) {
  double l = lambd.toDouble();
  if (half_unary(self, out, [&](const at::Tensor& x) { return at::softshrink(x, round_to(l, self.scalar_type())); })) {
    return out;
  }
  if (!param_unary_gpu(op::UNARY_SOFTSHRINK, self, out, lambd.toDouble(), 0)) {
    on_cpu("aten::softshrink", "out", self, lambd, out);
  }
  return out;
}

at::Tensor& hardshrink_out(const at::Tensor& self, const at::Scalar& lambd, at::Tensor& out) {
  double l = lambd.toDouble();
  if (half_unary(self, out, [&](const at::Tensor& x) { return at::hardshrink(x, round_to(l, self.scalar_type())); })) {
    return out;
  }
  if (!param_unary_gpu(op::UNARY_HARDSHRINK, self, out, lambd.toDouble(), 0)) {
    on_cpu("aten::hardshrink", "out", self, lambd, out);
  }
  return out;
}

at::Tensor& threshold_out(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value,
                          at::Tensor& out) {
  if (is_int(self.scalar_type()) && is_vk(self) && is_vk(out)) {
    resize_elementwise_output(out, self.sizes(), &self);
    out.copy_(at::where(at::gt(self, threshold), self, at::scalar_tensor(value, self.options())));
    return out;
  }
  // (in place on a tensor that overlaps itself is fine: threshold is idempotent, and the CPU allows it)
  if (!param_unary_gpu(op::UNARY_THRESHOLD, self, out, threshold.toDouble(), value.toDouble(), 0.0, false)) {
    on_cpu("aten::threshold", "out", self, threshold, value, out);
  }
  return out;
}

at::Tensor& logit_out(const at::Tensor& self, std::optional<double> eps, at::Tensor& out) {
  if (!param_unary_gpu(op::UNARY_LOGIT, self, out, eps.value_or(0.0), 0, eps.has_value() ? 1.0 : 0.0)) {
    on_cpu("aten::logit", "out", self, eps, out);
  }
  return out;
}

at::Tensor& nan_to_num_out(const at::Tensor& self, std::optional<double> nan, std::optional<double> posinf,
                           std::optional<double> neginf, at::Tensor& out) {
  if (!at::isFloatingType(self.scalar_type()) && !at::isComplexType(self.scalar_type()) && is_vk(self) &&
      is_vk(out)) {
    resize_elementwise_output(out, self.sizes(), &self);
    out.copy_(self);  // integers and bools have nothing to replace
    return out;
  }
  if (at::isComplexType(self.scalar_type()) && is_vk(self) && is_vk(out) && out.scalar_type() == self.scalar_type()) {
    // the real and imaginary parts separately, as the CPU's
    at::Tensor parts = at::nan_to_num(at::view_as_real(self.resolve_conj()), nan, posinf, neginf);
    resize_elementwise_output(out, self.sizes(), &self);
    out.copy_(at::view_as_complex(parts));
    return out;
  }
  if (at::isReducedFloatingType(self.scalar_type())) {
    // (the default replacements are the 16-bit dtype's extremes, not float32's)
    double top = self.scalar_type() == at::kHalf ? 65504.0 : static_cast<double>(std::numeric_limits<c10::BFloat16>::max());
    if (half_unary(self, out, [&](const at::Tensor& x) {
          return at::nan_to_num(x, nan.value_or(0.0), posinf.value_or(top), neginf.value_or(-top));
        })) {
      return out;
    }
  }
  bool f32 = out.scalar_type() == at::kFloat;
  double big = f32 ? std::numeric_limits<float>::max() : std::numeric_limits<double>::max();
  if (!param_unary_gpu(op::UNARY_NAN_TO_NUM, self, out, nan.value_or(0.0), posinf.value_or(big), neginf.value_or(-big))) {
    on_cpu("aten::nan_to_num", "out", self, nan, posinf, neginf, out);
  }
  return out;
}

std::tuple<at::Tensor&, at::Tensor&> log_sigmoid_forward_out(const at::Tensor& self, at::Tensor& output,
                                                             at::Tensor& buffer) {
  if (!param_unary_gpu(op::UNARY_LOG_SIGMOID, self, output, 0, 0)) {
    on_cpu("aten::log_sigmoid_forward", "output", self, output, buffer);
    return {output, buffer};
  }
  // The CPU kernel's buffer, e^-|x| (shaped as PyTorch's meta kernel says; our
  // backward, PyTorch's decomposition, does not read it).
  at::native::resize_output(buffer, self.sizes());
  buffer.copy_(at::exp(at::neg(at::abs(self))));
  return {output, buffer};
}

// ---------------------------------------------------------------- clamp

// Scalar bounds: UNARY_CLAMP with the bounds in alpha and beta.
bool clamp_scalar_gpu(const at::Tensor& self, const std::optional<at::Scalar>& lo,
                      const std::optional<at::Scalar>& hi, const at::Tensor& out) {
  at::ScalarType result = self.scalar_type();
  for (const auto* bound : {&lo, &hi}) {
    if (bound->has_value()) result = at::promote_types(result, at::result_type(self, **bound));
  }
  check_out_cast(result, out);
  resize_elementwise_output(out, self.sizes(), &self);
  at::ScalarType dtype = out.scalar_type();
  if (!is_vk(self) || !is_vk(out) || !(is_float(dtype) || is_int(dtype))) {
    return false;
  }
  at::Tensor input = to_dtype(self, dtype);
  std::string kernel = std::string("unary_") + dtype_label(dtype);
  if (is_float(dtype)) {
    double a = lo ? lo->toDouble() : -kInf, b = hi ? hi->toDouble() : kInf;
    if ((lo && std::isnan(a)) || (hi && std::isnan(b))) {
      out.fill_(std::numeric_limits<double>::quiet_NaN());  // a NaN bound makes every result NaN
      return true;
    }
    launch_strided(kernel.c_str(), {op::UNARY_CLAMP}, out, &input, nullptr, {a, b});
  } else {
    int64_t a = lo ? lo->toLong() : std::numeric_limits<int64_t>::min();
    int64_t b = hi ? hi->toLong() : std::numeric_limits<int64_t>::max();
    launch_strided(kernel.c_str(), {op::UNARY_CLAMP}, out, &input, nullptr, {a, b});
  }
  return true;
}

at::Tensor& clamp_out(const at::Tensor& self, const std::optional<at::Scalar>& min,
                      const std::optional<at::Scalar>& max, at::Tensor& out) {
  if (!clamp_scalar_gpu(self, min, max, out)) {
    on_cpu("aten::clamp", "out", self, min, max, out);
  }
  return out;
}

at::Tensor& clamp_min_out(const at::Tensor& self, const at::Scalar& min, at::Tensor& out) {
  if (!clamp_scalar_gpu(self, min, std::nullopt, out)) {
    on_cpu("aten::clamp_min", "out", self, min, out);
  }
  return out;
}

at::Tensor& clamp_max_out(const at::Tensor& self, const at::Scalar& max, at::Tensor& out) {
  if (!clamp_scalar_gpu(self, std::nullopt, max, out)) {
    on_cpu("aten::clamp_max", "out", self, max, out);
  }
  return out;
}

// Tensor bounds: TERNARY_CLAMP.
bool clamp_tensor_gpu(const at::Tensor& self, const std::optional<at::Tensor>& lo,
                      const std::optional<at::Tensor>& hi, const at::Tensor& out) {
  at::ScalarType result = self.scalar_type();
  if (lo.has_value()) result = at::promote_types(result, at::result_type(self, *lo));
  if (hi.has_value()) result = at::promote_types(result, at::result_type(self, *hi));
  check_out_cast(result, out);
  std::vector<int64_t> shape = self.sizes().vec();
  for (const auto* t : {&lo, &hi}) {
    if (t->has_value()) shape = at::infer_size(shape, (*t)->sizes());
  }
  at::native::resize_output(out, shape);
  at::ScalarType dtype = out.scalar_type();
  bool vk_bounds = (!lo || is_vk(*lo)) && (!hi || is_vk(*hi));
  if (dtype == at::kBool && is_vk(self) && is_vk(out) && vk_bounds) {
    // On bools: clamp_min is "or", clamp_max is "and" (computed as int32).
    at::Tensor wide = at::empty(shape, out.options().dtype(at::kInt));
    std::optional<at::Tensor> wlo = lo ? std::optional<at::Tensor>(lo->to(at::kInt)) : std::nullopt;
    std::optional<at::Tensor> whi = hi ? std::optional<at::Tensor>(hi->to(at::kInt)) : std::nullopt;
    if (!clamp_tensor_gpu(self.to(at::kInt), wlo, whi, wide)) return false;
    out.copy_(wide);
    return true;
  }
  if (!is_vk(self) || !is_vk(out) || !vk_bounds || !(is_float(dtype) || is_int(dtype))) {
    return false;
  }
  at::Tensor a = to_dtype(self, dtype);
  at::Tensor b = lo ? to_dtype(*lo, dtype) : at::Tensor();
  at::Tensor c = hi ? to_dtype(*hi, dtype) : at::Tensor();
  launch_strided4((std::string("ternary_") + dtype_label(dtype)).c_str(),
                  {op::TERNARY_CLAMP, 0, lo ? 1u : 0u, hi ? 1u : 0u}, out, &a, lo ? &b : nullptr, hi ? &c : nullptr);
  return true;
}

at::Tensor& clamp_tensor_out(const at::Tensor& self, const std::optional<at::Tensor>& min,
                             const std::optional<at::Tensor>& max, at::Tensor& out) {
  // (the CPU's clamp kernel has no bool version; with one bound it uses clamp_min / clamp_max's, which do)
  reject_dtype("clamp", out.scalar_type(), out.scalar_type() == at::kBool && min.has_value() && max.has_value());
  if (!clamp_tensor_gpu(self, min, max, out)) {
    on_cpu("aten::clamp", "Tensor_out", self, min, max, out);
  }
  return out;
}

at::Tensor& clamp_min_tensor_out(const at::Tensor& self, const at::Tensor& min, at::Tensor& out) {
  if (!clamp_tensor_gpu(self, min, std::nullopt, out)) {
    on_cpu("aten::clamp_min", "Tensor_out", self, min, out);
  }
  return out;
}

at::Tensor& clamp_max_tensor_out(const at::Tensor& self, const at::Tensor& max, at::Tensor& out) {
  if (!clamp_tensor_gpu(self, std::nullopt, max, out)) {
    on_cpu("aten::clamp_max", "Tensor_out", self, max, out);
  }
  return out;
}

// ---------------------------------------------------------------- predicates and logical ops

// out (bool) = predicate `code` (a COMPARE_* code) of self.
bool predicate_gpu(uint32_t code, const at::Tensor& self, const at::Tensor& out) {
  resize_elementwise_output(out, self.sizes(), &self);
  const char* label = dtype_label(self.scalar_type());
  if (!is_vk(self) || !is_vk(out) || label == nullptr) {
    return false;
  }
  with_output(out, at::kBool, [&](const at::Tensor& target) {
    launch_strided((std::string("compare_") + label).c_str(), {code, 0}, target, &self, nullptr);
  });
  return true;
}

at::Tensor isnan(const at::Tensor& self) {
  // (empty, so the output takes self's dim order when resized)
  at::Tensor out = at::empty({0}, self.options().dtype(at::kBool));
  if (!predicate_gpu(op::COMPARE_ISNAN, self, out)) {
    return on_cpu("aten::isnan", "", self)[0].toTensor();
  }
  return out;
}

template <uint32_t Code>
at::Tensor& predicate_out(const at::Tensor& self, at::Tensor& out) {
  if (Code == op::COMPARE_ISPOSINF || Code == op::COMPARE_ISNEGINF) {
    const char* name = Code == op::COMPARE_ISPOSINF ? "isposinf" : "isneginf";
    TORCH_CHECK(!self.is_complex(), name, " does not support complex inputs.");
    TORCH_CHECK(out.scalar_type() == at::kBool, name, " does not support non-boolean outputs.");
  }
  if (!predicate_gpu(Code, self, out)) {
    const char* name = Code == op::COMPARE_ISPOSINF   ? "aten::isposinf"
                       : Code == op::COMPARE_ISNEGINF ? "aten::isneginf"
                       : Code == op::COMPARE_SIGNBIT  ? "aten::signbit"
                                                      : "aten::logical_not";
    on_cpu(name, "out", self, out);
  }
  return out;
}

template <uint32_t Code>  // COMPARE_LOGICAL_AND, _OR, _XOR
at::Tensor& logical_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  at::native::resize_output(out, at::infer_size(self.sizes(), other.sizes()));
  // (any storable dtypes, mixed too: both sides are cast to bool first)
  bool ok = is_vk(self) && is_vk(out) && (is_vk(other) || cpu_scalar(other)) &&
            storage_label(self.scalar_type()) && storage_label(other.scalar_type());
  if (!ok) {
    const char* name = Code == op::COMPARE_LOGICAL_AND  ? "aten::logical_and"
                       : Code == op::COMPARE_LOGICAL_OR ? "aten::logical_or"
                                                        : "aten::logical_xor";
    on_cpu(name, "out", self, other, out);
    return out;
  }
  at::Tensor a = to_dtype(self, at::kBool);
  at::Tensor b = to_dtype(is_vk(other) ? other : other.to(self.device()), at::kBool);
  with_output(out, at::kBool, [&](const at::Tensor& target) {
    launch_strided("compare_b8", {Code, 0}, target, &a, &b);
  });
  return out;
}

// ---------------------------------------------------------------- where / masked_fill

// out = cond ? a : b, where a or b may be a real scalar.
bool where_gpu(const at::Tensor& cond, const at::Tensor* a, const std::optional<at::Scalar>& a_scalar,
               const at::Tensor* b, const std::optional<at::Scalar>& b_scalar, at::ScalarType dtype,
               const at::Tensor& out) {
  std::vector<int64_t> shape = cond.sizes().vec();
  if (a) shape = at::infer_size(shape, a->sizes());
  if (b) shape = at::infer_size(shape, b->sizes());
  at::native::resize_output(out, shape);
  const char* label = dtype_label(dtype);
  bool scalars_ok = !((a_scalar || b_scalar) && at::isComplexType(dtype));
  if (!is_vk(cond) || !is_vk(out) || (a && !is_vk(*a)) || (b && !is_vk(*b)) || label == nullptr || !scalars_ok) {
    return false;
  }
  at::Tensor c = to_dtype(cond, at::kBool);
  at::Tensor ta = a ? to_dtype(*a, dtype) : at::Tensor();
  at::Tensor tb = b ? to_dtype(*b, dtype) : at::Tensor();
  with_output(out, dtype, [&](const at::Tensor& target) {
    launch_strided4((std::string("where_") + label).c_str(), {a ? 0u : 1u, b ? 0u : 1u}, target,
                    a ? &ta : nullptr, b ? &tb : nullptr, &c, {a_scalar.value_or(0), b_scalar.value_or(0)});
  });
  return true;
}

// A tensor operand of where: vk tensor, or a CPU 0-dim scalar (Python number).
void where_operand(const at::Tensor& t, const at::Tensor*& tensor, std::optional<at::Scalar>& scalar) {
  if (cpu_scalar(t) && !at::isComplexType(t.scalar_type())) {
    tensor = nullptr;
    scalar = t.item();
  } else {
    tensor = &t;
  }
}

at::Tensor& where_out(const at::Tensor& cond, const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  check_out_cast(at::result_type(self, other), out);
  const at::Tensor *a = nullptr, *b = nullptr;
  std::optional<at::Scalar> sa, sb;
  where_operand(self, a, sa);
  where_operand(other, b, sb);
  at::ScalarType dtype = at::result_type(self, other);
  if (!where_gpu(cond, a, sa, b, sb, dtype, out)) {
    on_cpu("aten::where", "self_out", cond, self, other, out);
  }
  return out;
}

at::Tensor where_self(const at::Tensor& cond, const at::Tensor& self, const at::Tensor& other) {
  at::ScalarType dtype = at::result_type(self, other);
  at::Tensor out = at::empty({0}, (is_vk(self) ? self : other).options().dtype(dtype));
  return where_out(cond, self, other, out);
}

at::Tensor& masked_fill_scalar(at::Tensor& self, const at::Tensor& mask, const at::Scalar& value) {
  if (at::has_internal_overlap(self) == at::MemOverlap::Yes) {  // (as the CPU)
    TORCH_WARN("Use of masked_fill_ on expanded tensors is deprecated. Please clone() the tensor before performing "
               "this operation. This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  at::assert_no_partial_overlap(self, mask);
  if (at::isComplexType(self.scalar_type()) && is_vk(self) && is_vk(mask) && !self.is_conj() &&
      dtype_label(self.scalar_type()) != nullptr) {
    // Fill the real and imaginary parts.
    c10::complex<double> v = value.toComplexDouble();
    at::Tensor parts = at::view_as_real(self);
    parts.select(-1, 0).masked_fill_(mask, v.real());
    parts.select(-1, 1).masked_fill_(mask, v.imag());
    return self;
  }
  bool ok = !value.isComplex() && is_vk(self) && is_vk(mask);
  if (ok) {
    at::Tensor cond = mask;
    ok = where_gpu(cond.expand(self.sizes()), nullptr, value, &self, std::nullopt, self.scalar_type(), self);
  }
  if (!ok) {
    on_cpu("aten::masked_fill_", "Scalar", self, mask, value);
  }
  return self;
}

at::Tensor& masked_fill_tensor(at::Tensor& self, const at::Tensor& mask, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
              value.dim(), " dimension(s).");
  return masked_fill_scalar(self, mask, value.item());
}

// ---------------------------------------------------------------- lerp, addcmul, addcdiv

bool ternary_gpu(uint32_t code, const at::Tensor& a, const at::Tensor& b, const at::Tensor* c,
                 const std::optional<at::Scalar>& c_scalar, const at::Scalar& alpha, const at::Tensor& out) {
  std::vector<int64_t> shape = at::infer_size(a.sizes(), b.sizes());
  if (c) shape = at::infer_size(shape, c->sizes());
  at::native::resize_output(out, shape);
  at::ScalarType dtype = out.scalar_type();
  bool float_only = code == op::TERNARY_LERP || code == op::TERNARY_ADDCDIV;
  if (!is_vk(a) || !is_vk(b) || (c && !is_vk(*c)) || !is_vk(out) || !(is_float(dtype) || (!float_only && is_int(dtype)))) {
    return false;
  }
  at::Tensor ta = to_dtype(a, dtype), tb = to_dtype(b, dtype);
  at::Tensor tc = c ? to_dtype(*c, dtype) : at::Tensor();
  launch_strided4((std::string("ternary_") + dtype_label(dtype)).c_str(), {code, c ? 0u : 1u, 1, 1}, out, &ta, &tb,
                  c ? &tc : nullptr, {c_scalar.value_or(alpha)});
  return true;
}

// Complex lerp, addcmul, addcdiv from complex element-wise ops.
bool complex_ternary(const at::Tensor& self, const at::Tensor& a, const at::Tensor& out) {
  return at::isComplexType(out.scalar_type()) && dtype_label(out.scalar_type()) != nullptr && is_vk(self) &&
         is_vk(a) && is_vk(out);
}

// The CPU's lerp: self + w (end - self) for |w| < 0.5, else end - (end - self) (1 - w).
at::Tensor complex_lerp(const at::Tensor& self, const at::Tensor& end, const at::Tensor& w) {
  at::Tensor diff = at::sub(end, self);
  return at::where(at::lt(at::abs(w), 0.5), at::add(self, at::mul(w, diff)),
                   at::sub(end, at::mul(diff, at::add(at::neg(w), 1))));
}

at::Tensor& lerp_scalar_out(const at::Tensor& self, const at::Tensor& end, const at::Scalar& weight, at::Tensor& out) {
  if (complex_ternary(self, end, out)) {
    at::native::resize_output(out, at::infer_size(self.sizes(), end.sizes()));
    out.copy_(complex_lerp(self, end, at::scalar_tensor(weight, out.options())));
    return out;
  }
  if (weight.isComplex() || !ternary_gpu(op::TERNARY_LERP, self, end, nullptr, weight, 0, out)) {
    on_cpu("aten::lerp", "Scalar_out", self, end, weight, out);
  }
  return out;
}

at::Tensor& lerp_tensor_out(const at::Tensor& self, const at::Tensor& end, const at::Tensor& weight, at::Tensor& out) {
  if (complex_ternary(self, end, out) && is_vk(weight)) {
    at::native::resize_output(out, at::infer_size(at::infer_size(self.sizes(), end.sizes()), weight.sizes()));
    out.copy_(complex_lerp(self, end, weight));
    return out;
  }
  if (!ternary_gpu(op::TERNARY_LERP, self, end, &weight, std::nullopt, 0, out)) {
    on_cpu("aten::lerp", "Tensor_out", self, end, weight, out);
  }
  return out;
}

at::Tensor& addcmul_out(const at::Tensor& self, const at::Tensor& t1, const at::Tensor& t2, const at::Scalar& value,
                        at::Tensor& out) {
  check_out_cast(at::promote_types(at::result_type(self, t1), at::result_type(t1, t2)), out);
  if (complex_ternary(self, t1, out) && is_vk(t2)) {
    at::native::resize_output(out, at::infer_size(at::infer_size(self.sizes(), t1.sizes()), t2.sizes()));
    // (operands in out's dtype first, then self + value * t1 * t2 left to right, as the CPU)
    at::ScalarType ct = out.scalar_type();
    out.copy_(at::add(self.to(ct), at::mul(at::mul(t1.to(ct), value), t2.to(ct))));
    return out;
  }
  if (value.isComplex() || !ternary_gpu(op::TERNARY_ADDCMUL, self, t1, &t2, std::nullopt, value, out)) {
    on_cpu("aten::addcmul", "out", self, t1, t2, value, out);
  }
  return out;
}

at::Tensor& addcdiv_out(const at::Tensor& self, const at::Tensor& t1, const at::Tensor& t2, const at::Scalar& value,
                        at::Tensor& out) {
  if (complex_ternary(self, t1, out) && is_vk(t2)) {
    at::native::resize_output(out, at::infer_size(at::infer_size(self.sizes(), t1.sizes()), t2.sizes()));
    // (operands in out's dtype first, then self + value * t1 / t2 left to right, as the CPU)
    at::ScalarType ct = out.scalar_type();
    out.copy_(at::add(self.to(ct), at::div(at::mul(t1.to(ct), value), t2.to(ct))));
    return out;
  }
  if (value.isComplex() || !ternary_gpu(op::TERNARY_ADDCDIV, self, t1, &t2, std::nullopt, value, out)) {
    on_cpu("aten::addcdiv", "out", self, t1, t2, value, out);
  }
  return out;
}

// ---------------------------------------------------------------- compositions of existing kernels

at::Tensor& glu_out(const at::Tensor& self, int64_t dim, at::Tensor& out) {
  int64_t d = at::maybe_wrap_dim(dim, self.dim());
  int64_t half = self.size(d) / 2;
  TORCH_CHECK(self.size(d) % 2 == 0, "Halving dimension must be even, but dimension ", d, " is size ",
              self.size(d));
  at::Tensor a = self.narrow(d, 0, half), b = self.narrow(d, half, half);
  at::native::resize_output(out, a.sizes());
  at::mul_out(out, a, at::sigmoid(b));
  return out;
}

at::Tensor& complex_out(const at::Tensor& real, const at::Tensor& imag, at::Tensor& out) {
  // (the CPU's checks)
  TORCH_CHECK(real.scalar_type() == imag.scalar_type(), "Expected object of scalar type ", real.scalar_type(),
              " but got scalar type ", imag.scalar_type(), " for second argument");
  TORCH_CHECK(out.scalar_type() == c10::toComplexType(real.scalar_type()), "Expected object of scalar type ",
              c10::toComplexType(real.scalar_type()), " but got scalar type ", out.scalar_type(), " for argument 'out'");
  at::native::resize_output(out, at::infer_size(real.sizes(), imag.sizes()));
  // (complex32: written through its float16 parts)
  if (!is_vk(out) || !is_vk(real) || !is_vk(imag) ||
      (dtype_label(out.scalar_type()) == nullptr && out.scalar_type() != at::kComplexHalf)) {
    on_cpu("aten::complex", "out", real, imag, out);
    return out;
  }
  at::Tensor parts = at::view_as_real(out);
  parts.select(-1, 0).copy_(real);
  parts.select(-1, 1).copy_(imag);
  return out;
}

at::Tensor& polar_out(const at::Tensor& abs, const at::Tensor& angle, at::Tensor& out) {
  TORCH_CHECK(abs.scalar_type() == angle.scalar_type(), "Expected object of scalar type ", abs.scalar_type(),
              " but got scalar type ", angle.scalar_type(), " for second argument");
  if (at::isReducedFloatingType(abs.scalar_type()) && is_vk(abs) && is_vk(angle)) {
    // (float16 parts computed in float32 and rounded once, as on the CPU)
    at::Tensor a = abs.to(at::kFloat), t = angle.to(at::kFloat);
    return complex_out((a * at::cos(t)).to(abs.scalar_type()), (a * at::sin(t)).to(abs.scalar_type()), out);
  }
  return complex_out(abs * at::cos(angle), abs * at::sin(angle), out);
}

at::Tensor& angle_out(const at::Tensor& self, at::Tensor& out) {
  check_out_cast(at::isComplexType(self.scalar_type()) ? c10::toRealValueType(self.scalar_type())
                 : at::isFloatingType(self.scalar_type()) ? self.scalar_type()
                                                          : c10::typeMetaToScalarType(c10::get_default_dtype()),
                 out);
  if (!is_vk(self) || !is_vk(out)) {
    on_cpu("aten::angle", "out", self, out);
    return out;
  }
  resize_elementwise_output(out, self.sizes(), &self);
  if (at::isComplexType(self.scalar_type())) {
    at::atan2_out(out, at::imag(self), at::real(self));
  } else {
    // pi for negative values, 0 otherwise, NaN stays NaN.
    at::Tensor x = to_dtype(self, out.scalar_type());
    at::Tensor pi = at::where(at::lt(x, 0), 3.141592653589793, 0.0).to(out.scalar_type());
    out.copy_(at::where(at::isnan(x), x, pi));
  }
  return out;
}

at::Tensor& sgn_out(const at::Tensor& self, at::Tensor& out) {
  if (!at::isComplexType(self.scalar_type())) {
    return at::sign_out(out, self);
  }
  if (!is_vk(self) || !is_vk(out)) {
    on_cpu("aten::sgn", "out", self, out);
    return out;
  }
  resize_elementwise_output(out, self.sizes(), &self);
  if (self.scalar_type() == at::kComplexHalf) {  // (the CPU has a complex32 sgn, used by abs's backward)
    out.copy_(at::sgn(self.to(at::kComplexFloat)));
    return out;
  }
  at::Tensor magnitude = at::abs(self);
  out.copy_(at::where(at::eq(magnitude, 0), at::zeros_like(self), self / magnitude));
  return out;
}

at::Tensor& pow_scalar_base_out(const at::Scalar& self, const at::Tensor& exponent, at::Tensor& out) {
  check_overlap(out, {&exponent});
  at::native::resize_output(out, exponent.sizes());
  bool ok = is_vk(exponent) && is_vk(out);
  if (ok) {
    // In the promoted dtype, then cast (as the CPU: 2 ** uint8(17) is 0 even
    // into an int64 out). A complex base too: through the same kernel as a
    // tensor base, so both agree.
    at::ScalarType compute = at::result_type(exponent, self);
    check_out_cast(compute, out);
    at::Tensor base = at::full(exponent.sizes(), self, exponent.options().dtype(compute));
    if (compute == out.scalar_type()) {
      ok = binary_op_gpu(op::BINARY_POW, base, &exponent, nullptr, out);
    } else {
      at::Tensor result = at::empty(exponent.sizes(), base.options());
      ok = binary_op_gpu(op::BINARY_POW, base, &exponent, nullptr, result);
      if (ok) out.copy_(result);
    }
  }
  if (!ok) {
    on_cpu("aten::pow", "Scalar_out", self, exponent, out);
  }
  return out;
}

at::Tensor& div_mode_out(const at::Tensor& self, const at::Tensor& other, std::optional<c10::string_view> mode,
                         at::Tensor& out) {
  bool ok;
  if (!mode.has_value()) {
    ok = binary_op_gpu(op::BINARY_DIV, self, &other, nullptr, out);
  } else if (*mode == "floor") {
    ok = binary_op_gpu(op::BINARY_FLOOR_DIVIDE, self, &other, nullptr, out);
  } else if (is_half(at::result_type(self, other)) && is_vk(self) && is_vk(other) && is_vk(out)) {
    // As the CPU: the quotient rounded to half precision, then truncated.
    at::Tensor q = at::div(self, other);
    check_out_cast(q.scalar_type(), out);
    at::native::resize_output(out, q.sizes());
    out.copy_(at::trunc(q));
    ok = true;
  } else {
    ok = binary_op_gpu(op::BINARY_DIV_TRUNC, self, &other, nullptr, out);
  }
  if (!ok) {
    on_cpu("aten::div", "out_mode", self, other, mode, out);
  }
  return out;
}

// ---------------------------------------------------------------- functional and in-place forms

// Ops that are not structured in PyTorch reach a backend's .out kernel only
// from their .out overload; their functional and in-place overloads need
// kernels of their own, which allocate (or reuse self) and call the .out op.

at::ScalarType float_result(const at::Tensor& self) {
  return at::isFloatingType(self.scalar_type()) ? self.scalar_type()
                                                : c10::typeMetaToScalarType(at::get_default_dtype());
}

at::Tensor angle(const at::Tensor& self) {
  at::ScalarType dtype =
      at::isComplexType(self.scalar_type()) ? c10::toRealValueType(self.scalar_type()) : float_result(self);
  at::Tensor out = at::empty({0}, self.options().dtype(dtype));
  return angle_out(self, out);
}

at::Tensor floor_divide(const at::Tensor& self, const at::Tensor& other) {
  // On the vk operand's device (self may be a CPU scalar: floor_divide(2.5, x)).
  const at::Tensor& like = is_vk(self) ? self : other;
  at::Tensor out = at::empty({0}, like.options().dtype(at::result_type(self, other)));
  return at::floor_divide_out(out, self, other);
}

at::Tensor& floor_divide_(at::Tensor& self, const at::Tensor& other) {
  return at::floor_divide_out(self, self, other);
}

at::Tensor hardswish(const at::Tensor& self) {
  at::Tensor out = at::empty({0}, self.options());
  return at::hardswish_out(out, self);
}

at::Tensor& hardswish_(at::Tensor& self) {
  return at::hardswish_out(self, self);
}

at::Tensor hardtanh(const at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  at::Tensor out = at::empty({0}, self.options());
  return hardtanh_out(self, min, max, out);
}

at::Tensor& hardtanh_(at::Tensor& self, const at::Scalar& min, const at::Scalar& max) {
  return hardtanh_out(self, min, max, self);
}

at::Tensor logit(const at::Tensor& self, std::optional<double> eps) {
  at::Tensor out = at::empty({0}, self.options().dtype(float_result(self)));
  return logit_out(self, eps, out);
}

at::Tensor& logit_(at::Tensor& self, std::optional<double> eps) {
  return logit_out(self, eps, self);
}

std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward(const at::Tensor& self) {
  at::Tensor output = at::empty({0}, self.options()), buffer = at::empty({0}, self.options());
  log_sigmoid_forward_out(self, output, buffer);
  return {output, buffer};
}

at::Tensor remainder_scalar_tensor(const at::Scalar& self, const at::Tensor& other) {
  at::ScalarType dtype = at::result_type(other, self);
  return at::remainder(at::scalar_tensor(self, other.options().dtype(dtype)), other);
}

// round(x * 10^d) / 10^d (or round(x / 10^-d) * 10^-d), as PyTorch's CPU kernel.
at::Tensor& round_decimals_out(const at::Tensor& self, int64_t decimals, at::Tensor& out) {
  resize_elementwise_output(out, self.sizes(), &self);
  if (!is_vk(self) || !is_vk(out) || !at::isFloatingType(out.scalar_type())) {
    on_cpu("aten::round", "decimals_out", self, decimals, out);
    return out;
  }
  if (decimals == 0) {
    return at::round_out(out, to_dtype(self, out.scalar_type()));
  }
  // As the CPU's kernel: in the op math type (float32 for 16-bit floats),
  // nearbyint(x * 10^d) / 10^d with a true division (a 0-dim "vk" scale,
  // not a Python number, whose division becomes a multiplication).
  at::ScalarType math = is_half(out.scalar_type()) ? at::kFloat : out.scalar_type();
  at::Tensor x = to_dtype(self, math);
  at::Tensor scale = at::full({}, std::pow(10.0, static_cast<double>(std::abs(decimals))), x.options());
  out.copy_(decimals > 0 ? at::div(at::round(x * scale), scale) : at::mul(at::round(x / scale), scale));
  return out;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("gelu.out", &gelu_out);
  m.impl("softplus.out", &softplus_out);
  m.impl("elu.out", &elu_out);
  m.impl("hardtanh.out", &hardtanh_out);
  m.impl("leaky_relu.out", &leaky_relu_out);
  m.impl("softshrink.out", &softshrink_out);
  m.impl("hardshrink.out", &hardshrink_out);
  m.impl("threshold.out", &threshold_out);
  m.impl("logit.out", &logit_out);
  m.impl("nan_to_num.out", &nan_to_num_out);
  m.impl("log_sigmoid_forward.output", &log_sigmoid_forward_out);
  m.impl("clamp.out", &clamp_out);
  m.impl("clamp_min.out", &clamp_min_out);
  m.impl("clamp_max.out", &clamp_max_out);
  m.impl("clamp.Tensor_out", &clamp_tensor_out);
  m.impl("clamp_min.Tensor_out", &clamp_min_tensor_out);
  m.impl("clamp_max.Tensor_out", &clamp_max_tensor_out);
  m.impl("isnan", &vk::isnan);
  m.impl("isposinf.out", &predicate_out<op::COMPARE_ISPOSINF>);
  m.impl("isneginf.out", &predicate_out<op::COMPARE_ISNEGINF>);
  m.impl("signbit.out", &predicate_out<op::COMPARE_SIGNBIT>);
  m.impl("logical_not.out", &predicate_out<op::COMPARE_LOGICAL_NOT>);
  m.impl("logical_and.out", &logical_out<op::COMPARE_LOGICAL_AND>);
  m.impl("logical_or.out", &logical_out<op::COMPARE_LOGICAL_OR>);
  m.impl("logical_xor.out", &logical_out<op::COMPARE_LOGICAL_XOR>);
  m.impl("where.self_out", &where_out);
  m.impl("where.self", &where_self);
  m.impl("masked_fill_.Scalar", &masked_fill_scalar);
  m.impl("masked_fill_.Tensor", &masked_fill_tensor);
  m.impl("lerp.Scalar_out", &lerp_scalar_out);
  m.impl("lerp.Tensor_out", &lerp_tensor_out);
  m.impl("addcmul.out", &addcmul_out);
  m.impl("addcdiv.out", &addcdiv_out);
  m.impl("glu.out", &glu_out);
  m.impl("complex.out", &complex_out);
  m.impl("polar.out", &polar_out);
  m.impl("angle.out", &angle_out);
  m.impl("sgn.out", &sgn_out);
  m.impl("pow.Scalar_out", &pow_scalar_base_out);
  m.impl("div.out_mode", &div_mode_out);
  m.impl("angle", &vk::angle);
  m.impl("floor_divide", &vk::floor_divide);
  m.impl("floor_divide_.Tensor", &vk::floor_divide_);
  m.impl("hardswish", &vk::hardswish);
  m.impl("hardswish_", &vk::hardswish_);
  m.impl("hardtanh", &vk::hardtanh);
  m.impl("hardtanh_", &vk::hardtanh_);
  m.impl("logit", &vk::logit);
  m.impl("logit_", &vk::logit_);
  m.impl("log_sigmoid_forward", &vk::log_sigmoid_forward);
  m.impl("remainder.Scalar_Tensor", &remainder_scalar_tensor);
  m.impl("round.decimals_out", &round_decimals_out);
}
