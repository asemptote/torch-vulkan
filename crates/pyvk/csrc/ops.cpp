// Element-wise aten kernels for device "vk": strided launches of the Slang
// templates in crates/vkcore/shaders (unary, binary, compare, cast, fill).
//
// Structured ops are registered through their `.out` overloads; PyTorch's
// generated functional and in-place variants allocate/resize `out` and call
// these. Type promotion happens here: inputs are cast to the computation
// dtype first, and results are cast into `out` if its dtype differs. Cases
// without a kernel (e.g. float64 exp) run the same overload on the CPU via
// PyTorch's boxed cpu_fallback.

#include "opcodes.h"
#include "vk.h"

#include <ATen/ExpandUtils.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <utility>

namespace vk {

// ---------------------------------------------------------------- helpers

namespace {

enum class Kind { Numeric, Float, Float32, Integer };

bool kind_allows(Kind kind, at::ScalarType t) {
  switch (kind) {
    case Kind::Numeric: return t == at::kFloat || t == at::kDouble || t == at::kInt || t == at::kLong;
    case Kind::Float: return t == at::kFloat || t == at::kDouble;
    case Kind::Float32: return t == at::kFloat;
    case Kind::Integer: return t == at::kInt || t == at::kLong;
  }
  return false;
}

// ---------------------------------------------------------------- unary

struct UnaryOp {
  const char* name;
  uint32_t code;
  Kind kind;
};

constexpr UnaryOp kUnary[] = {
    {"aten::neg", op::UNARY_NEG, Kind::Numeric},
    {"aten::abs", op::UNARY_ABS, Kind::Numeric},
    {"aten::sign", op::UNARY_SIGN, Kind::Numeric},
    {"aten::reciprocal", op::UNARY_RECIPROCAL, Kind::Float},
    {"aten::sqrt", op::UNARY_SQRT, Kind::Float},
    {"aten::rsqrt", op::UNARY_RSQRT, Kind::Float},
    {"aten::floor", op::UNARY_FLOOR, Kind::Float},
    {"aten::ceil", op::UNARY_CEIL, Kind::Float},
    {"aten::trunc", op::UNARY_TRUNC, Kind::Float},
    {"aten::round", op::UNARY_ROUND, Kind::Float},
    {"aten::exp", op::UNARY_EXP, Kind::Float},
    {"aten::log", op::UNARY_LOG, Kind::Float},
    {"aten::sin", op::UNARY_SIN, Kind::Float},
    {"aten::cos", op::UNARY_COS, Kind::Float},
    {"aten::tanh", op::UNARY_TANH, Kind::Float},
    {"aten::sigmoid", op::UNARY_SIGMOID, Kind::Float},
    {"aten::exp2", op::UNARY_EXP2, Kind::Float},
    {"aten::log2", op::UNARY_LOG2, Kind::Float},
    {"aten::tan", op::UNARY_TAN, Kind::Float},
    {"aten::asin", op::UNARY_ASIN, Kind::Float},
    {"aten::acos", op::UNARY_ACOS, Kind::Float},
    {"aten::atan", op::UNARY_ATAN, Kind::Float},
    {"aten::sinh", op::UNARY_SINH, Kind::Float},
    {"aten::cosh", op::UNARY_COSH, Kind::Float},
    {"aten::asinh", op::UNARY_ASINH, Kind::Float},
    {"aten::acosh", op::UNARY_ACOSH, Kind::Float},
    {"aten::atanh", op::UNARY_ATANH, Kind::Float},
    {"aten::log10", op::UNARY_LOG10, Kind::Float},
    {"aten::log1p", op::UNARY_LOG1P, Kind::Float},
    {"aten::expm1", op::UNARY_EXPM1, Kind::Float},
    {"aten::erf", op::UNARY_ERF, Kind::Float},
    {"aten::erfc", op::UNARY_ERFC, Kind::Float},
    {"aten::erfinv", op::UNARY_ERFINV, Kind::Float},
    {"aten::lgamma", op::UNARY_LGAMMA, Kind::Float},
    {"aten::digamma", op::UNARY_DIGAMMA, Kind::Float},
    {"aten::sinc", op::UNARY_SINC, Kind::Float},
    {"aten::frac", op::UNARY_FRAC, Kind::Float},
    {"aten::silu", op::UNARY_SILU, Kind::Float},
    {"aten::mish", op::UNARY_MISH, Kind::Float},
    {"aten::hardswish", op::UNARY_HARDSWISH, Kind::Float},
    {"aten::hardsigmoid", op::UNARY_HARDSIGMOID, Kind::Float},
    {"aten::bitwise_not", op::UNARY_BITWISE_NOT, Kind::Integer},
};

// Complex unary ops: neg and abs (real output).
// Unary codes shaders/cunary.slang implements for complex inputs (with the
// same code values), besides abs (shaders/cabs.slang).
bool complex_unary_supported(uint32_t code) {
  switch (code) {
    case op::UNARY_NEG: case op::UNARY_SQUARE: case op::UNARY_RECIPROCAL: case op::UNARY_SQRT:
    case op::UNARY_RSQRT: case op::UNARY_EXP: case op::UNARY_LOG: case op::UNARY_SIN: case op::UNARY_COS:
    case op::UNARY_TANH: case op::UNARY_SIGMOID: case op::UNARY_EXP2: case op::UNARY_LOG2: case op::UNARY_TAN:
    case op::UNARY_ASIN: case op::UNARY_ACOS: case op::UNARY_ATAN: case op::UNARY_SINH: case op::UNARY_COSH:
    case op::UNARY_ASINH: case op::UNARY_ACOSH: case op::UNARY_ATANH: case op::UNARY_LOG10: case op::UNARY_LOG1P:
    case op::UNARY_EXPM1: return true;
    default: return false;
  }
}
static_assert(op::CUNARY_EXP == op::UNARY_EXP && op::CUNARY_EXPM1 == op::UNARY_EXPM1 && op::CUNARY_NEG == op::UNARY_NEG,
              "complex unary codes mirror the real ones");

bool complex_unary_gpu(uint32_t code, const at::Tensor& self, const at::Tensor& out) {
  const char* label = dtype_label(self.scalar_type());
  if (label != nullptr && is_vk(self) && is_vk(out) && (code == op::UNARY_SILU || code == op::UNARY_SINC)) {
    if (code == op::UNARY_SILU) {
      out.copy_(at::mul(self, at::sigmoid(self)));
    } else {  // sin(pi z) / (pi z), 1 at 0
      at::Tensor pz = at::mul(self, 3.141592653589793);
      out.copy_(at::where(at::eq(self, 0), at::ones_like(self), at::div(at::sin(pz), pz)));
    }
    return true;
  }
  bool abs = code == op::UNARY_ABS;
  if (label == nullptr || !is_vk(self) || !is_vk(out) || !(abs || complex_unary_supported(code))) {
    return false;
  }
  at::ScalarType want = abs ? c10::toRealValueType(self.scalar_type()) : self.scalar_type();
  with_output(out, want, [&](const at::Tensor& target) {
    if (abs) {
      launch_strided((std::string("cabs_") + label).c_str(), {}, target, &self, nullptr);
    } else {
      launch_strided((std::string("cunary_") + label).c_str(), {code}, target, &self, nullptr);
    }
  });
  return true;
}

bool unary_gpu(uint32_t code, Kind kind, const at::Tensor& self, const at::Tensor& out) {
  resize_elementwise_output(out, self.sizes(), &self);
  if (at::isComplexType(self.scalar_type())) {
    return complex_unary_gpu(code, self, out);
  }
  if (self.scalar_type() == at::kBool && code == op::UNARY_SIGN && is_vk(self) && is_vk(out)) {
    out.copy_(self);  // sign of a bool is itself
    return true;
  }
  if (self.scalar_type() == at::kBool && code == op::UNARY_BITWISE_NOT && is_vk(self) && is_vk(out) &&
      out.scalar_type() == at::kBool) {
    launch_strided("compare_b8", {op::COMPARE_LOGICAL_NOT, 0}, out, &self, nullptr);  // logical_not on bools
    return true;
  }
  bool rounding = code == op::UNARY_FLOOR || code == op::UNARY_CEIL || code == op::UNARY_TRUNC ||
                  code == op::UNARY_ROUND;
  if (!at::isFloatingType(self.scalar_type()) && rounding && is_vk(self) && is_vk(out)) {
    out.copy_(self);  // floor/ceil/trunc/round of integers: the integers themselves
    return true;
  }
  at::ScalarType dtype = widened(out.scalar_type());
  const char* label = dtype_label(dtype);
  if (!kind_allows(kind, dtype) || label == nullptr || !is_vk(self) || !is_vk(out)) {
    return false;
  }
  if (!at::isFloatingType(self.scalar_type()) && code == op::UNARY_FRAC) {
    return false;
  }
  at::Tensor input = to_dtype(self, dtype);  // e.g. sqrt(int) computes in float
  std::string kernel = std::string("unary_") + label;
  with_output(out, dtype, [&](const at::Tensor& target) {
    launch_strided(kernel.c_str(), {code}, target, &input, nullptr);
  });
  return true;
}

// The dtype PyTorch computes unary op `code` in for self.
at::ScalarType unary_result(uint32_t code, Kind kind, const at::Tensor& self) {
  at::ScalarType t = self.scalar_type();
  if (at::isComplexType(t)) {
    return code == op::UNARY_ABS ? c10::toRealValueType(t) : t;
  }
  bool rounding = code == op::UNARY_FLOOR || code == op::UNARY_CEIL || code == op::UNARY_TRUNC ||
                  code == op::UNARY_ROUND;
  if (kind == Kind::Float && !at::isFloatingType(t) && !rounding) {
    return c10::typeMetaToScalarType(at::get_default_dtype());
  }
  return t;
}

template <size_t I>
at::Tensor& unary_out(const at::Tensor& self, at::Tensor& out) {
  uint32_t code = kUnary[I].code;
  bool rounding = code == op::UNARY_FLOOR || code == op::UNARY_CEIL || code == op::UNARY_TRUNC ||
                  code == op::UNARY_ROUND;
  reject_dtype(kUnary[I].name + 6, self.scalar_type(), rounding && self.scalar_type() == at::kBool);
  // These keep their input's dtype, and the CPU's kernels (TensorIterator's
  // unary_op) want it in out= as well.
  bool same_dtype = rounding || code == op::UNARY_NEG || code == op::UNARY_SIGN || code == op::UNARY_FRAC ||
                    code == op::UNARY_BITWISE_NOT;
  TORCH_CHECK(!same_dtype || out.scalar_type() == self.scalar_type(), "Found dtype ", out.scalar_type(),
              " but expected ", self.scalar_type());
  check_out_cast(unary_result(kUnary[I].code, kUnary[I].kind, self), out);
  if (!unary_gpu(kUnary[I].code, kUnary[I].kind, self, out)) {
    on_cpu(kUnary[I].name, "out", self, out);
  }
  return out;
}

at::Tensor relu(const at::Tensor& self) {
  at::Tensor out = at::empty_like(self);
  if (!unary_gpu(op::UNARY_RELU, Kind::Numeric, self, out)) {
    return on_cpu("aten::relu", "", self)[0].toTensor();
  }
  return out;
}

at::Tensor& relu_(at::Tensor& self) {
  if (!unary_gpu(op::UNARY_RELU, Kind::Numeric, self, self)) {
    on_cpu("aten::relu_", "", self);
  }
  return self;
}

at::Tensor& conj_physical_out(const at::Tensor& self, at::Tensor& out) {
  check_out_device(self, out);
  check_out_cast(self.scalar_type(), out);
  resize_elementwise_output(out, self.sizes(), &self);
  const char* label = dtype_label(self.scalar_type());
  if (!at::isComplexType(self.scalar_type())) {
    out.copy_(self);  // conj of a real tensor is itself
  } else if (label != nullptr && is_vk(self) && is_vk(out) && out.scalar_type() == self.scalar_type()) {
    launch_strided((std::string("cunary_") + label).c_str(), {op::CUNARY_CONJ}, out, &self, nullptr);
  } else {
    on_cpu("aten::conj_physical", "out", self, out);
  }
  return out;
}

// (PyTorch's composite conj_physical_ calls the CPU/CUDA kernel stub directly)
at::Tensor& conj_physical_inplace(at::Tensor& self) {
  return at::isComplexType(self.scalar_type()) ? vk::conj_physical_out(static_cast<const at::Tensor&>(self), self) : self;
}

// ---------------------------------------------------------------- binary

struct BinaryOp {
  const char* name;
  const char* overload;  // Tensor-other out overload
  uint32_t code;
  Kind kind;
  bool true_div;    // integer inputs compute in the default float dtype
  bool commutes;    // a scalar first operand can be swapped to second
};

constexpr BinaryOp kBinary[] = {
    {"aten::add", "out", op::BINARY_ADD, Kind::Numeric, false, true},
    {"aten::sub", "out", op::BINARY_SUB, Kind::Numeric, false, false},
    {"aten::mul", "out", op::BINARY_MUL, Kind::Numeric, false, true},
    {"aten::maximum", "out", op::BINARY_MAXIMUM, Kind::Numeric, false, true},
    {"aten::minimum", "out", op::BINARY_MINIMUM, Kind::Numeric, false, true},
    {"aten::div", "out", op::BINARY_DIV, Kind::Float, true, false},
    {"aten::pow", "Tensor_Tensor_out", op::BINARY_POW, Kind::Numeric, false, false},
    {"aten::atan2", "out", op::BINARY_ATAN2, Kind::Float, true, false},
    {"aten::bitwise_and", "Tensor_out", op::BINARY_BITWISE_AND, Kind::Integer, false, true},
    {"aten::bitwise_or", "Tensor_out", op::BINARY_BITWISE_OR, Kind::Integer, false, true},
    {"aten::bitwise_xor", "Tensor_out", op::BINARY_BITWISE_XOR, Kind::Integer, false, true},
    {"aten::remainder", "Tensor_out", op::BINARY_REMAINDER, Kind::Numeric, false, false},
    {"aten::fmod", "Tensor_out", op::BINARY_FMOD, Kind::Numeric, false, false},
    {"aten::floor_divide", "out", op::BINARY_FLOOR_DIVIDE, Kind::Numeric, false, false},
    {"aten::copysign", "out", op::BINARY_COPYSIGN, Kind::Float, true, false},
    {"aten::hypot", "out", op::BINARY_HYPOT, Kind::Float, false, true},
    {"aten::nextafter", "out", op::BINARY_NEXTAFTER, Kind::Float, false, false},
    {"aten::logaddexp", "out", op::BINARY_LOGADDEXP, Kind::Float, false, true},
    {"aten::logaddexp2", "out", op::BINARY_LOGADDEXP2, Kind::Float, false, true},
    {"aten::xlogy", "OutTensor", op::BINARY_XLOGY, Kind::Float, true, false},
    {"aten::special_xlog1py", "out", op::BINARY_XLOG1PY, Kind::Float, true, false},
    {"aten::heaviside", "out", op::BINARY_HEAVISIDE, Kind::Float, false, false},
    {"aten::fmax", "out", op::BINARY_FMAX, Kind::Float, false, true},
    {"aten::fmin", "out", op::BINARY_FMIN, Kind::Float, false, true},
    {"aten::bitwise_left_shift", "Tensor_out", op::BINARY_SHIFT_LEFT, Kind::Integer, false, false},
    {"aten::bitwise_right_shift", "Tensor_out", op::BINARY_SHIFT_RIGHT, Kind::Integer, false, false},
    {"aten::gcd", "out", op::BINARY_GCD, Kind::Integer, false, true},
    {"aten::lcm", "out", op::BINARY_LCM, Kind::Integer, false, true},
};
constexpr size_t kSub = 1, kPow = 6;

at::ScalarType compute_dtype(const BinaryOp& op, at::ScalarType promoted) {
  if (op.true_div && !at::isFloatingType(promoted) && !at::isComplexType(promoted)) {
    return c10::typeMetaToScalarType(at::get_default_dtype());
  }
  return promoted;
}

// Complex add/sub (real alpha), mul and div; shaders/cbinary.slang.
bool complex_binary_gpu(const BinaryOp& op, const at::Tensor& lhs, const at::Tensor* other,
                        const std::optional<at::Scalar>& scalar, const at::Scalar& alpha, const at::Tensor& out,
                        at::ScalarType dtype) {
  uint32_t code;
  switch (op.code) {
    case op::BINARY_ADD: code = op::CBINARY_ADD; break;
    case op::BINARY_SUB: code = op::CBINARY_SUB; break;
    case op::BINARY_MUL: code = op::CBINARY_MUL; break;
    case op::BINARY_DIV: code = op::CBINARY_DIV; break;
    case op::BINARY_POW: code = op::CBINARY_POW; break;
    default: return false;
  }
  const char* label = dtype_label(dtype);
  if (label == nullptr || alpha.isComplex()) {
    return false;
  }
  bool uses_alpha = code == op::CBINARY_ADD || code == op::CBINARY_SUB;
  at::Tensor a = to_dtype(lhs, dtype);
  std::string kernel = std::string("cbinary_") + label;
  with_output(out, dtype, [&](const at::Tensor& target) {
    if (scalar) {
      c10::complex<double> v = scalar->toComplexDouble() * (uses_alpha ? alpha.toDouble() : 1.0);
      launch_strided(kernel.c_str(), {code, 1}, target, &a, nullptr, {v.real(), v.imag()});
    } else {
      at::Tensor b = to_dtype(*other, dtype);
      launch_strided(kernel.c_str(), {code, 0}, target, &a, &b, {alpha});
    }
  });
  return true;
}

// out = op(self, other) with other a vk tensor, a 0-dim CPU tensor, or a scalar.
bool binary_gpu(const BinaryOp& op, const at::Tensor& self, const at::Tensor* other, const at::Scalar* other_scalar,
                const at::Scalar& alpha, const at::Tensor& out) {
  if (!is_vk(out)) {
    return false;
  }
  check_overlap(out, {&self, other});
  {
    std::vector<int64_t> shape =
        other != nullptr ? at::infer_size(self.sizes(), other->sizes()) : self.sizes().vec();
    const at::Tensor* like = self.sizes() == at::IntArrayRef(shape) ? &self
                             : other != nullptr && other->sizes() == at::IntArrayRef(shape) ? other : nullptr;
    check_inplace_shape(out, shape, {&self, other});
    resize_elementwise_output(out, shape, like);
  }
  const at::Tensor* lhs = &self;
  std::optional<at::Scalar> scalar;
  if (other_scalar != nullptr) {
    scalar = *other_scalar;
  } else if (other->dim() == 0 && other->device().is_cpu()) {
    scalar = other->item();
  } else if (!is_vk(*other)) {
    return false;
  }
  if (!is_vk(self)) {
    // CPU scalar first operand (e.g. a 0-dim CPU tensor + vk tensor).
    if (self.dim() != 0 || other == nullptr || scalar) {
      return false;
    }
    if (&op == &kBinary[kSub]) {
      // s - alpha * b  ==  (b * -alpha) + s
      at::ScalarType dtype = at::result_type(self, *other);
      const char* label = dtype_label(dtype);
      if (!kind_allows(op.kind, dtype) || label == nullptr) {
        return false;  // (complex dtypes fail kind_allows and fall back)
      }
      at::Tensor b = to_dtype(*other, dtype);
      std::string kernel = std::string("binary_") + label;
      at::Scalar s = self.item();
      with_output(out, dtype, [&](const at::Tensor& target) {
        launch_strided(kernel.c_str(), {op::BINARY_MUL, 1}, target, &b, nullptr, {-alpha});
        launch_strided(kernel.c_str(), {op::BINARY_ADD, 1}, target, &target, nullptr, {s});
      });
      return true;
    }
    if (!(op.commutes && alpha.equal(1))) {
      // Any other op: the scalar as a 0-dim vk tensor, still marked as a
      // wrapped number so type promotion treats it as the Python number it is.
      at::Tensor moved = self.to(other->device());
      moved.unsafeGetTensorImpl()->set_wrapped_number(self.unsafeGetTensorImpl()->is_wrapped_number());
      return binary_gpu(op, moved, other, nullptr, alpha, out);
    }
    scalar = self.item();
    lhs = other;
  }
  at::ScalarType promoted = scalar ? at::result_type(*lhs, *scalar) : at::result_type(*lhs, *other);
  check_out_cast(compute_dtype(op, promoted), out);
  at::ScalarType result_dtype = compute_dtype(op, promoted);
  if (out.scalar_type() != result_dtype && !(at::isComplexType(result_dtype) && at::isComplexType(out.scalar_type()))) {
    // Computed in the promoted dtype, then cast to out= (as TensorIterator:
    // uint8 5 - 8 is 253 and float16 33000 + 33000 inf, even into float64).
    at::Tensor result = at::empty(out.sizes(), out.options().dtype(result_dtype));
    if (!binary_gpu(op, self, other, other_scalar, alpha, result)) {
      return false;
    }
    out.copy_(result);
    return true;
  }
  if (!at::isFloatingType(promoted) && !at::isComplexType(promoted) &&
      (op.code == op::BINARY_FMAX || op.code == op::BINARY_FMIN)) {
    // No NaNs to ignore: fmax and fmin are maximum and minimum.
    return binary_gpu(kBinary[op.code == op::BINARY_FMAX ? 3 : 4], *lhs, scalar ? nullptr : other,
                      scalar ? &*scalar : nullptr, alpha, out);
  }
  if (!at::isFloatingType(promoted) && !at::isComplexType(promoted) && op.code == op::BINARY_HEAVISIDE &&
      lhs == &self && !scalar) {
    // heaviside(x, v) = v where x == 0, else x > 0.
    at::Tensor x = to_dtype(self, promoted), v = to_dtype(*other, promoted);
    out.copy_(at::where(at::eq(x, 0), v, at::gt(x, 0).to(promoted)));
    return true;
  }
  if (at::isComplexType(promoted) && op.code == op::BINARY_LOGADDEXP && lhs == &self && !scalar &&
      dtype_label(promoted) != nullptr) {
    // log(e^a + e^b) = max + log1p(e^(min - max)), max and min by real part (as the CPU).
    at::Tensor a = to_dtype(self, promoted), b = to_dtype(*other, promoted);
    at::Tensor a_big = at::ge(at::real(a), at::real(b));
    at::Tensor hi = at::where(a_big, a, b), lo = at::where(a_big, b, a);
    // (lo + -hi, not lo - hi: complex subtraction makes the other part NaN by an infinite part, as the CPU's)
    at::Tensor r = at::add(at::log1p(at::exp(at::add(lo, at::neg(hi)))), hi);
    // The CPU's special cases: NaNs spread; equal infinite real parts give min
    // (-inf) or log(e^a + e^b) (+inf), where the formula above would give NaN.
    at::Tensor lr = at::real(lo), hr = at::real(hi);
    at::Tensor both_inf = at::logical_and(at::isinf(lr), at::eq(lr, hr));
    r = at::where(at::logical_and(both_inf, at::lt(lr, 0)), lo, r);
    r = at::where(at::logical_and(both_inf, at::gt(lr, 0)), at::log(at::add(at::exp(lo), at::exp(hi))), r);
    at::Tensor nan = at::logical_or(at::isnan(a), at::isnan(b));
    out.copy_(at::where(nan, at::full({}, c10::complex<double>(NAN, NAN), r.options()), r));
    return true;
  }
  at::ScalarType dtype = compute_dtype(op, promoted);
  const char* label = dtype_label(dtype);
  if (dtype == at::kBool && !scalar && alpha.equal(1)) {
    // On bools: add, maximum, bitwise_or are logical or; mul, minimum, bitwise_and are logical and.
    uint32_t logical;
    switch (op.code) {
      case op::BINARY_ADD: case op::BINARY_MAXIMUM: case op::BINARY_BITWISE_OR: logical = op::COMPARE_LOGICAL_OR; break;
      case op::BINARY_MUL: case op::BINARY_MINIMUM: case op::BINARY_BITWISE_AND: logical = op::COMPARE_LOGICAL_AND; break;
      case op::BINARY_BITWISE_XOR: logical = op::COMPARE_LOGICAL_XOR; break;
      default: return false;
    }
    at::Tensor a = to_dtype(*lhs, at::kBool), b = to_dtype(*other, at::kBool);
    with_output(out, at::kBool, [&](const at::Tensor& target) {
      launch_strided("compare_b8", {logical, 0}, target, &a, &b);
    });
    return true;
  }
  if (at::isComplexType(dtype)) {
    return complex_binary_gpu(op, *lhs, scalar ? nullptr : other, scalar, alpha, out, dtype);
  }
  bool opmath_scalar = op.code == op::BINARY_ADD || op.code == op::BINARY_SUB || op.code == op::BINARY_MUL ||
                       op.code == op::BINARY_DIV || op.code == op::BINARY_POW;
  if (scalar && is_half(dtype) && !opmath_scalar) {
    // As the CPU: a Python number joins a half-precision op as that dtype
    // (fmod(x, 2.3) uses 2.30078125), though the kernel computes in float32;
    // add, sub, mul, div and pow take it in float32 (their kernels read the
    // original scalar).
    scalar = at::scalar_tensor(*scalar, at::TensorOptions().dtype(dtype)).item();
  }
  dtype = widened(dtype);
  label = dtype_label(dtype);
  if (!kind_allows(op.kind, dtype) || label == nullptr) {
    return false;
  }
  bool floating = at::isFloatingType(dtype);
  bool uses_alpha = op.code == op::BINARY_ADD || op.code == op::BINARY_SUB;
  at::Tensor a = to_dtype(*lhs, dtype);
  std::string kernel = std::string("binary_") + label;
  with_output(out, dtype, [&](const at::Tensor& target) {
    if (scalar) {
      // Fold alpha into the scalar: a + alpha * s.
      at::Scalar v = !uses_alpha ? *scalar
                     : floating  ? at::Scalar(scalar->toDouble() * alpha.toDouble())
                                 : at::Scalar(scalar->toLong() * alpha.toLong());
      uint32_t code = op.code;
      if (code == op::BINARY_DIV && floating) {
        // a / s as a * (1 / s), with 1 / s rounded in the compute type, as
        // PyTorch's CPU and CUDA kernels (bitwise equal results).
        double d = scalar->toDouble();
        code = op::BINARY_MUL;
        v = dtype == at::kDouble ? at::Scalar(1.0 / d) : at::Scalar(static_cast<double>(1.0f / static_cast<float>(d)));
      }
      launch_strided(kernel.c_str(), {code, 1}, target, &a, nullptr, {v});
    } else {
      at::Tensor b = to_dtype(*other, dtype);
      launch_strided(kernel.c_str(), {op.code, 0}, target, &a, &b, {alpha});
    }
  });
  return true;
}

}  // namespace

bool binary_op_gpu(int64_t code, const at::Tensor& self, const at::Tensor* other, const at::Scalar* other_scalar,
                   const at::Tensor& out) {
  for (const BinaryOp& op : kBinary) {
    if (op.code == static_cast<uint32_t>(code)) {
      return binary_gpu(op, self, other, other_scalar, 1, out);
    }
  }
  // Codes without an aten op of their own (e.g. BINARY_DIV_TRUNC).
  BinaryOp extra{"", "", static_cast<uint32_t>(code), Kind::Numeric, false, false};
  return binary_gpu(extra, self, other, other_scalar, 1, out);
}

namespace {

template <size_t I>
at::Tensor& binary_alpha_out(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha, at::Tensor& out) {
  if (!binary_gpu(kBinary[I], self, &other, nullptr, alpha, out)) {
    on_cpu(kBinary[I].name, kBinary[I].overload, self, other, alpha, out);
  }
  return out;
}

template <size_t I>
at::Tensor& binary_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  if (!binary_gpu(kBinary[I], self, &other, nullptr, 1, out)) {
    on_cpu(kBinary[I].name, kBinary[I].overload, self, other, out);
  }
  return out;
}

// nextafter in half precision steps through that format's values, by their
// bits (the kernel computes in float32, whose neighbour rounds back to x).
at::Tensor& nextafter_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  at::ScalarType t = at::result_type(self, other);
  if (!(t == at::kHalf || t == at::kBFloat16) || !is_vk(self) || !is_vk(other) || !is_vk(out) ||
      out.scalar_type() != t) {
    return binary_out<16>(self, other, out);
  }
  auto shape = at::infer_size(self.sizes(), other.sizes());
  at::Tensor x = self.to(t).expand(shape).contiguous(), y = other.to(t).expand(shape).contiguous();
  at::Tensor xf = x.to(at::kFloat), yf = y.to(at::kFloat);
  at::Tensor up = at::eq(at::gt(yf, xf), at::gt(xf, 0));  // away from zero
  // (the bits as int32: vk has no int16 arithmetic; the int16 bits are
  // sign-magnitude-like, so +1 steps away from zero for either sign)
  auto bits_of = [&](const at::Tensor& v) { return v.view(at::kShort).to(at::kInt); };
  at::Tensor bits = bits_of(x) + at::where(up, 1, -1).to(at::kInt);
  at::Tensor tiny = at::where(at::lt(yf, 0), -32767, 1).to(at::kInt);  // +-smallest subnormal
  bits = at::where(at::eq(xf, 0), tiny, bits);
  bits = at::where(at::eq(xf, yf), bits_of(y), bits);
  bits = at::where(at::logical_or(at::isnan(xf), at::isnan(yf)), bits_of((xf + yf).to(t)), bits);
  at::native::resize_output(out, shape);
  out.copy_(bits.to(at::kShort).view(t));
  return out;
}

at::Tensor& heaviside_out(const at::Tensor& self, const at::Tensor& values, at::Tensor& out) {
  TORCH_CHECK(!self.is_complex() && !values.is_complex() && !out.is_complex(),
              "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(self.scalar_type() == values.scalar_type(),
              "heaviside is not yet implemented for tensors with different dtypes.");
  return binary_out<21>(self, values, out);
}

at::Tensor& pow_scalar_out(const at::Tensor& self, const at::Scalar& exponent, at::Tensor& out) {
  TORCH_CHECK(!(at::isIntegralType(self.scalar_type(), true) && exponent.isIntegral(true) && exponent.toLong() < 0),
              "Integers to negative integer powers are not allowed.");  // (as the CPU)
  if (!binary_gpu(kBinary[kPow], self, nullptr, &exponent, 1, out)) {
    on_cpu("aten::pow", "Tensor_Scalar_out", self, exponent, out);
  }
  return out;
}

// ---------------------------------------------------------------- Python's << and >>

template <bool Left>
at::Tensor shift_scalar(const at::Tensor& self, const at::Scalar& other) {
  return Left ? at::bitwise_left_shift(self, other) : at::bitwise_right_shift(self, other);
}

template <bool Left>
at::Tensor shift_tensor(const at::Tensor& self, const at::Tensor& other) {
  return Left ? at::bitwise_left_shift(self, other) : at::bitwise_right_shift(self, other);
}

template <bool Left>
at::Tensor& shift_scalar_(at::Tensor& self, const at::Scalar& other) {
  return Left ? self.bitwise_left_shift_(other) : self.bitwise_right_shift_(other);
}

template <bool Left>
at::Tensor& shift_tensor_(at::Tensor& self, const at::Tensor& other) {
  return Left ? self.bitwise_left_shift_(other) : self.bitwise_right_shift_(other);
}

// ---------------------------------------------------------------- comparisons

// Indexed by COMPARE_EQ ... COMPARE_GE.
constexpr const char* kCompare[] = {"aten::eq", "aten::ne", "aten::lt", "aten::le", "aten::gt", "aten::ge"};
static_assert(op::COMPARE_EQ == 0 && op::COMPARE_GE == 5);

bool compare_gpu(uint32_t code, const at::Tensor& self, const at::Tensor* other, const at::Scalar* other_scalar,
                 const at::Tensor& out) {
  if (!is_vk(self) || !is_vk(out)) {
    return false;
  }
  check_overlap(out, {&self, other});
  {
    std::vector<int64_t> shape =
        other != nullptr ? at::infer_size(self.sizes(), other->sizes()) : self.sizes().vec();
    const at::Tensor* like = self.sizes() == at::IntArrayRef(shape) ? &self
                             : other != nullptr && other->sizes() == at::IntArrayRef(shape) ? other : nullptr;
    resize_elementwise_output(out, shape, like);
  }
  std::optional<at::Scalar> scalar;
  if (other_scalar != nullptr) {
    scalar = *other_scalar;
  } else if (other->dim() == 0 && other->device().is_cpu()) {
    scalar = other->item();
  } else if (!is_vk(*other)) {
    return false;
  }
  at::ScalarType dtype = widened(scalar ? at::result_type(self, *scalar) : at::result_type(self, *other));
  const char* label = dtype_label(dtype);
  bool complex = at::isComplexType(dtype);
  bool complex_ok = complex && (code == op::COMPARE_EQ || code == op::COMPARE_NE);
  if (label == nullptr || !(kind_allows(Kind::Numeric, dtype) || dtype == at::kBool || complex_ok)) {
    return false;
  }
  at::Tensor a = to_dtype(self, dtype);
  std::string kernel = std::string("compare_") + label;
  with_output(out, at::kBool, [&](const at::Tensor& target) {
    if (scalar && complex) {
      c10::complex<double> c = scalar->toComplexDouble();
      launch_strided(kernel.c_str(), {code, 1}, target, &a, nullptr, {c.real(), c.imag()});
    } else if (scalar) {
      launch_strided(kernel.c_str(), {code, 1}, target, &a, nullptr, {*scalar});
    } else {
      at::Tensor b = to_dtype(*other, dtype);
      launch_strided(kernel.c_str(), {code, 0}, target, &a, &b);
    }
  });
  return true;
}

template <size_t I>
at::Tensor& compare_out(const at::Tensor& self, const at::Tensor& other, at::Tensor& out) {
  if (!compare_gpu(I, self, &other, nullptr, out)) {
    on_cpu(kCompare[I], "Tensor_out", self, other, out);
  }
  return out;
}

template <size_t I>
at::Tensor& compare_scalar_out(const at::Tensor& self, const at::Scalar& other, at::Tensor& out) {
  if (!compare_gpu(I, self, nullptr, &other, out)) {
    on_cpu(kCompare[I], "Scalar_out", self, other, out);
  }
  return out;
}

// ---------------------------------------------------------------- registration

std::string out_name(const char* aten_name, const char* overload) {
  return std::string(aten_name).substr(6) + "." + overload;  // strip "aten::"
}

template <size_t... I>
void register_unary(torch::Library& m, std::index_sequence<I...>) {
  (m.impl(out_name(kUnary[I].name, "out").c_str(), &unary_out<I>), ...);
}

template <size_t... I>
void register_compare(torch::Library& m, std::index_sequence<I...>) {
  (m.impl(out_name(kCompare[I], "Tensor_out").c_str(), &compare_out<I>), ...);
  (m.impl(out_name(kCompare[I], "Scalar_out").c_str(), &compare_scalar_out<I>), ...);
}

}  // namespace

}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  register_unary(m, std::make_index_sequence<std::size(kUnary)>());
  m.impl("relu", &relu);
  m.impl("relu_", &relu_);
  m.impl("conj_physical.out", &conj_physical_out);
  m.impl("conj_physical_", &conj_physical_inplace);

  m.impl("add.out", &binary_alpha_out<0>);
  m.impl("sub.out", &binary_alpha_out<1>);
  m.impl("mul.out", &binary_out<2>);
  m.impl("maximum.out", &binary_out<3>);
  m.impl("minimum.out", &binary_out<4>);
  m.impl("div.out", &binary_out<5>);
  m.impl("pow.Tensor_Tensor_out", &binary_out<6>);
  m.impl("atan2.out", &binary_out<7>);
  m.impl("bitwise_and.Tensor_out", &binary_out<8>);
  m.impl("bitwise_or.Tensor_out", &binary_out<9>);
  m.impl("bitwise_xor.Tensor_out", &binary_out<10>);
  m.impl("pow.Tensor_Scalar_out", &pow_scalar_out);
  m.impl("remainder.Tensor_out", &binary_out<11>);
  m.impl("fmod.Tensor_out", &binary_out<12>);
  m.impl("floor_divide.out", &binary_out<13>);

  m.impl("copysign.out", &binary_out<14>);
  m.impl("hypot.out", &binary_out<15>);
  m.impl("nextafter.out", &nextafter_out);
  m.impl("logaddexp.out", &binary_out<17>);
  m.impl("logaddexp2.out", &binary_out<18>);
  m.impl("xlogy.OutTensor", &binary_out<19>);
  m.impl("special_xlog1py.out", &binary_out<20>);
  m.impl("heaviside.out", &heaviside_out);
  m.impl("fmax.out", &binary_out<22>);
  m.impl("fmin.out", &binary_out<23>);
  m.impl("bitwise_left_shift.Tensor_out", &binary_out<24>);
  m.impl("bitwise_right_shift.Tensor_out", &binary_out<25>);
  m.impl("gcd.out", &binary_out<26>);
  m.impl("lcm.out", &binary_out<27>);

  register_compare(m, std::make_index_sequence<std::size(kCompare)>());
  m.impl("__lshift__.Scalar", &shift_scalar<true>);
  m.impl("__rshift__.Scalar", &shift_scalar<false>);
  m.impl("__lshift__.Tensor", &shift_tensor<true>);
  m.impl("__rshift__.Tensor", &shift_tensor<false>);
  m.impl("__ilshift__.Scalar", &shift_scalar_<true>);
  m.impl("__irshift__.Scalar", &shift_scalar_<false>);
  m.impl("__ilshift__.Tensor", &shift_tensor_<true>);
  m.impl("__irshift__.Tensor", &shift_tensor_<false>);
}
