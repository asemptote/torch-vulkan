// Element-wise aten kernels for device "vk": strided launches of the Slang
// templates in crates/vkcore/shaders (unary, binary, compare, cast, fill).
//
// Structured ops are registered through their `.out` overloads; PyTorch's
// generated functional and in-place variants allocate/resize `out` and call
// these. Type promotion happens here: inputs are cast to the computation
// dtype first, and results are cast into `out` if its dtype differs. Cases
// without a kernel (e.g. float64 exp) run the same overload on the CPU via
// PyTorch's boxed cpu_fallback.

#include "vk.h"

#include <ATen/ExpandUtils.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <array>
#include <cstddef>
#include <utility>

namespace vk {

// ---------------------------------------------------------------- strided launch

namespace {

// Must match the Params struct in shaders/_strided.slang.
struct Params {
  uint64_t out, a, b;
  double alpha;
  int64_t alpha_i;
  uint32_t n, ndim, stride, pad0;
  uint64_t pad1;
  uint32_t size[4];
  int32_t so[4], sa[4], sb[4];
};
static_assert(sizeof(Params) == 128, "push-constant block must be 128 bytes");
static_assert(offsetof(Params, size) == 64, "size[] must start at byte 64");

constexpr size_t kMaxDims = 4;

// Element strides of `t` broadcast to `shape` (0 where it broadcasts).
std::vector<int64_t> broadcast_strides(const at::Tensor& t, at::IntArrayRef shape) {
  int64_t pad = static_cast<int64_t>(shape.size()) - t.dim();
  TORCH_CHECK(pad >= 0, "torch_vulkan: operand has more dims than the output");
  std::vector<int64_t> out(shape.size(), 0);
  for (int64_t d = 0; d < t.dim(); ++d) {
    int64_t size = t.size(d), target = shape[pad + d];
    TORCH_CHECK(size == target || size == 1, "torch_vulkan: cannot broadcast ", t.sizes(), " to ", shape);
    out[pad + d] = (size == target && size != 1) ? t.stride(d) : 0;
  }
  return out;
}

struct Dim {
  int64_t size;
  std::array<int64_t, 3> stride;  // out, a, b
};

}  // namespace

void launch_strided(const char* kernel, std::vector<uint32_t> spec, const at::Tensor& out, const at::Tensor* a,
                    const at::Tensor* b, double alpha, int64_t alpha_i) {
  int64_t n = out.numel();
  if (n == 0) {
    return;
  }
  at::IntArrayRef shape = out.sizes();
  std::array<std::vector<int64_t>, 3> strides = {
      out.strides().vec(),
      a ? broadcast_strides(*a, shape) : std::vector<int64_t>(shape.size(), 0),
      b ? broadcast_strides(*b, shape) : std::vector<int64_t>(shape.size(), 0),
  };

  // Merge dims that are contiguous in every operand, innermost first.
  std::vector<Dim> dims;
  for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
    if (shape[d] == 1) {
      continue;
    }
    Dim dim{shape[d], {strides[0][d], strides[1][d], strides[2][d]}};
    if (!dims.empty()) {
      Dim& prev = dims.back();
      bool mergeable = true;
      for (size_t k = 0; k < 3; ++k) {
        mergeable = mergeable && dim.stride[k] == prev.stride[k] * prev.size;
      }
      if (mergeable) {
        prev.size *= dim.size;
        continue;
      }
    }
    dims.push_back(dim);
  }
  if (dims.empty()) {
    dims.push_back({1, {0, 0, 0}});
  }

  // The innermost (up to) 4 dims go to the kernel; the rest are looped here.
  size_t inner = std::min(dims.size(), kMaxDims);
  int64_t inner_n = 1;
  std::array<int64_t, 3> max_offset = {0, 0, 0};
  for (size_t i = 0; i < inner; ++i) {
    inner_n *= dims[i].size;
    for (size_t k = 0; k < 3; ++k) {
      max_offset[k] += (dims[i].size - 1) * dims[i].stride[k];
    }
  }
  TORCH_CHECK(inner_n < (int64_t(1) << 32), "torch_vulkan: more than 2^32 elements per launch");
  for (size_t k = 0; k < 3; ++k) {
    TORCH_CHECK(max_offset[k] < (int64_t(1) << 31), "torch_vulkan: element offsets beyond 2^31");
  }

  std::array<uint64_t, 3> base = {gpu_address(out), a ? gpu_address(*a) : 0, b ? gpu_address(*b) : 0};
  std::array<int64_t, 3> itemsize = {out.itemsize(), a ? a->itemsize() : 0, b ? b->itemsize() : 0};

  Params p{};
  p.alpha = alpha;
  p.alpha_i = alpha_i;
  p.n = static_cast<uint32_t>(inner_n);
  p.ndim = static_cast<uint32_t>(inner);
  uint32_t groups = 0;
  check(vkt_grid_1d(static_cast<uint64_t>(inner_n), &groups, &p.stride));
  for (size_t i = 0; i < kMaxDims; ++i) {
    bool used = i < inner;
    p.size[i] = used ? static_cast<uint32_t>(dims[i].size) : 1;
    p.so[i] = used ? static_cast<int32_t>(dims[i].stride[0]) : 0;
    p.sa[i] = used ? static_cast<int32_t>(dims[i].stride[1]) : 0;
    p.sb[i] = used ? static_cast<int32_t>(dims[i].stride[2]) : 0;
  }

  // One dispatch per index of the outer dims (odometer order).
  std::vector<int64_t> index(dims.size() - inner, 0);
  while (true) {
    std::array<int64_t, 3> offset = {0, 0, 0};
    for (size_t j = 0; j < index.size(); ++j) {
      for (size_t k = 0; k < 3; ++k) {
        offset[k] += index[j] * dims[inner + j].stride[k];
      }
    }
    p.out = base[0] + offset[0] * itemsize[0];
    p.a = a ? base[1] + offset[1] * itemsize[1] : 0;
    p.b = b ? base[2] + offset[2] * itemsize[2] : 0;
    check(vkt_dispatch(kernel, spec.data(), spec.size(), reinterpret_cast<const uint8_t*>(&p), sizeof(p), groups, 1, 1));

    size_t j = 0;
    for (; j < index.size(); ++j) {
      if (++index[j] < dims[inner + j].size) {
        break;
      }
      index[j] = 0;
    }
    if (j == index.size()) {
      break;
    }
  }
}

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

at::Tensor to_dtype(const at::Tensor& t, at::ScalarType dtype) {
  return t.scalar_type() == dtype ? t : t.to(dtype);
}

// Runs `body(target)` writing into `out` directly when its dtype is `dtype`,
// otherwise into a temporary that is then cast into `out`.
template <typename F>
void with_output(const at::Tensor& out, at::ScalarType dtype, F&& body) {
  if (out.scalar_type() == dtype) {
    body(out);
  } else {
    at::Tensor tmp = at::empty(out.sizes(), out.options().dtype(dtype));
    body(tmp);
    out.copy_(tmp);
  }
}

// ---------------------------------------------------------------- unary

// Codes must match shaders/unary.slang.
struct UnaryOp {
  const char* name;
  uint32_t code;
  Kind kind;
};

constexpr UnaryOp kUnary[] = {
    {"aten::neg", 0, Kind::Numeric},       {"aten::abs", 1, Kind::Numeric},     {"aten::sign", 4, Kind::Numeric},
    {"aten::reciprocal", 10, Kind::Float}, {"aten::sqrt", 11, Kind::Float},     {"aten::rsqrt", 12, Kind::Float},
    {"aten::floor", 13, Kind::Float},      {"aten::ceil", 14, Kind::Float},     {"aten::trunc", 15, Kind::Float},
    {"aten::round", 16, Kind::Float},      {"aten::exp", 20, Kind::Float},      {"aten::log", 21, Kind::Float},
    {"aten::sin", 22, Kind::Float},        {"aten::cos", 23, Kind::Float},      {"aten::tanh", 24, Kind::Float},
    {"aten::sigmoid", 25, Kind::Float},    {"aten::exp2", 26, Kind::Float},     {"aten::log2", 27, Kind::Float},
    {"aten::tan", 28, Kind::Float},
};

constexpr uint32_t kRelu = 2;

bool unary_gpu(uint32_t code, Kind kind, const at::Tensor& self, const at::Tensor& out) {
  at::native::resize_output(out, self.sizes());
  at::ScalarType dtype = out.scalar_type();
  const char* label = dtype_label(dtype);
  if (!kind_allows(kind, dtype) || label == nullptr || !is_vk(self) || !is_vk(out)) {
    return false;
  }
  if (!at::isFloatingType(self.scalar_type()) && code >= 13 && code <= 16) {
    return false;  // floor/ceil/trunc/round of integers keep the integer dtype
  }
  at::Tensor input = to_dtype(self, dtype);  // e.g. sqrt(int) computes in float
  std::string kernel = std::string("unary_") + label;
  launch_strided(kernel.c_str(), {code}, out, &input, nullptr, 0.0, 0);
  return true;
}

template <size_t I>
at::Tensor& unary_out(const at::Tensor& self, at::Tensor& out) {
  if (!unary_gpu(kUnary[I].code, kUnary[I].kind, self, out)) {
    on_cpu(kUnary[I].name, "out", self, out);
  }
  return out;
}

at::Tensor relu(const at::Tensor& self) {
  at::Tensor out = at::empty_like(self);
  if (!unary_gpu(kRelu, Kind::Numeric, self, out)) {
    return on_cpu("aten::relu", "", self)[0].toTensor();
  }
  return out;
}

at::Tensor& relu_(at::Tensor& self) {
  if (!unary_gpu(kRelu, Kind::Numeric, self, self)) {
    on_cpu("aten::relu_", "", self);
  }
  return self;
}

// ---------------------------------------------------------------- binary

// Codes must match shaders/binary.slang.
struct BinaryOp {
  const char* name;
  const char* overload;  // Tensor-other out overload
  uint32_t code;
  Kind kind;
  bool true_div;    // integer inputs compute in the default float dtype
  bool commutes;    // a scalar first operand can be swapped to second
};

constexpr BinaryOp kBinary[] = {
    {"aten::add", "out", 0, Kind::Numeric, false, true},
    {"aten::sub", "out", 1, Kind::Numeric, false, false},
    {"aten::mul", "out", 2, Kind::Numeric, false, true},
    {"aten::maximum", "out", 3, Kind::Numeric, false, true},
    {"aten::minimum", "out", 4, Kind::Numeric, false, true},
    {"aten::div", "out", 10, Kind::Float, true, false},
    {"aten::pow", "Tensor_Tensor_out", 20, Kind::Float32, false, false},
    {"aten::atan2", "out", 21, Kind::Float32, false, false},
    {"aten::bitwise_and", "Tensor_out", 30, Kind::Integer, false, true},
    {"aten::bitwise_or", "Tensor_out", 31, Kind::Integer, false, true},
    {"aten::bitwise_xor", "Tensor_out", 32, Kind::Integer, false, true},
};
constexpr size_t kAdd = 0, kSub = 1, kPow = 6;

at::ScalarType compute_dtype(const BinaryOp& op, at::ScalarType promoted) {
  if (op.true_div && !at::isFloatingType(promoted)) {
    return c10::typeMetaToScalarType(at::get_default_dtype());
  }
  return promoted;
}

// out = op(self, other) with other a vk tensor, a 0-dim CPU tensor, or a scalar.
bool binary_gpu(const BinaryOp& op, const at::Tensor& self, const at::Tensor* other, const at::Scalar* other_scalar,
                const at::Scalar& alpha, const at::Tensor& out) {
  if (!is_vk(out)) {
    return false;
  }
  at::native::resize_output(out, other != nullptr ? at::infer_size(self.sizes(), other->sizes()) : self.sizes().vec());
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
        return false;
      }
      bool floating = at::isFloatingType(dtype);
      at::Tensor b = to_dtype(*other, dtype);
      std::string kernel = std::string("binary_") + label;
      at::Scalar s = self.item();
      with_output(out, dtype, [&](const at::Tensor& target) {
        launch_strided(kernel.c_str(), {kBinary[2].code, 1}, target, &b, nullptr, floating ? -alpha.toDouble() : 0.0,
                       floating ? 0 : -alpha.toLong());
        launch_strided(kernel.c_str(), {kBinary[kAdd].code, 1}, target, &target, nullptr, floating ? s.toDouble() : 0.0,
                       floating ? 0 : s.toLong());
      });
      return true;
    }
    if (!(op.commutes && alpha.equal(1))) {
      return false;
    }
    scalar = self.item();
    lhs = other;
  }
  at::ScalarType promoted = scalar ? at::result_type(*lhs, *scalar) : at::result_type(*lhs, *other);
  at::ScalarType dtype = compute_dtype(op, promoted);
  const char* label = dtype_label(dtype);
  if (!kind_allows(op.kind, dtype) || label == nullptr) {
    return false;
  }
  bool floating = at::isFloatingType(dtype);
  bool uses_alpha = &op == &kBinary[kAdd] || &op == &kBinary[kSub];
  at::Tensor a = to_dtype(*lhs, dtype);
  std::string kernel = std::string("binary_") + label;
  with_output(out, dtype, [&](const at::Tensor& target) {
    if (scalar) {
      // Fold alpha into the scalar: a + alpha * s.
      double v = scalar->toDouble() * (uses_alpha ? alpha.toDouble() : 1.0);
      int64_t vi = floating ? 0 : scalar->toLong() * (uses_alpha ? alpha.toLong() : 1);
      launch_strided(kernel.c_str(), {op.code, 1}, target, &a, nullptr, floating ? v : 0.0, vi);
    } else {
      at::Tensor b = to_dtype(*other, dtype);
      launch_strided(kernel.c_str(), {op.code, 0}, target, &a, &b, floating ? alpha.toDouble() : 0.0,
                     floating ? 0 : alpha.toLong());
    }
  });
  return true;
}

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

at::Tensor& pow_scalar_out(const at::Tensor& self, const at::Scalar& exponent, at::Tensor& out) {
  if (!binary_gpu(kBinary[kPow], self, nullptr, &exponent, 1, out)) {
    on_cpu("aten::pow", "Tensor_Scalar_out", self, exponent, out);
  }
  return out;
}

// ---------------------------------------------------------------- comparisons

// Codes must match shaders/compare.slang.
constexpr const char* kCompare[] = {"aten::eq", "aten::ne", "aten::lt", "aten::le", "aten::gt", "aten::ge"};

bool compare_gpu(uint32_t code, const at::Tensor& self, const at::Tensor* other, const at::Scalar* other_scalar,
                 const at::Tensor& out) {
  if (!is_vk(self) || !is_vk(out)) {
    return false;
  }
  at::native::resize_output(out, other != nullptr ? at::infer_size(self.sizes(), other->sizes()) : self.sizes().vec());
  std::optional<at::Scalar> scalar;
  if (other_scalar != nullptr) {
    scalar = *other_scalar;
  } else if (other->dim() == 0 && other->device().is_cpu()) {
    scalar = other->item();
  } else if (!is_vk(*other)) {
    return false;
  }
  at::ScalarType dtype = scalar ? at::result_type(self, *scalar) : at::result_type(self, *other);
  const char* label = dtype_label(dtype);
  if (!kind_allows(Kind::Numeric, dtype) || label == nullptr) {
    return false;
  }
  bool floating = at::isFloatingType(dtype);
  at::Tensor a = to_dtype(self, dtype);
  std::string kernel = std::string("compare_") + label;
  with_output(out, at::kBool, [&](const at::Tensor& target) {
    if (scalar) {
      launch_strided(kernel.c_str(), {code, 1}, target, &a, nullptr, floating ? scalar->toDouble() : 0.0,
                     floating ? 0 : scalar->toLong());
    } else {
      at::Tensor b = to_dtype(*other, dtype);
      launch_strided(kernel.c_str(), {code, 0}, target, &a, &b, 0.0, 0);
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

void register_ops() {
  // Registration happens in the TORCH_LIBRARY_IMPL block below when this
  // object file is loaded; this function exists so the linker keeps it.
}

}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  register_unary(m, std::make_index_sequence<std::size(kUnary)>());
  m.impl("relu", &relu);
  m.impl("relu_", &relu_);

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

  register_compare(m, std::make_index_sequence<std::size(kCompare)>());
}
