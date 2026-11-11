// Quantized tensors on device "vk" (PyTorch's legacy per-tensor affine type:
// quint8, qint8, qint32), as far as PyTorch's tests use them here:
// construction (quantize_per_tensor, _empty_affine_quantized,
// empty_quantized), the getters (q_scale, q_zero_point, qscheme, int_repr),
// dequantize, views, clone, copies and flip. A quantized tensor's storage is
// an ordinary vk buffer of its underlying integers, so the kernels are dense
// ops on a view of it in that dtype, with the quantizer carried alongside.
// The rounding is the CPU's (FBGEMM's Quantize, or ATen's quantize_val where
// PyTorch is built without FBGEMM, and their dequantization):
//   q = clamp(nearbyint(x * (1.0f / scale)) + zero_point, qmin, qmax)
//   x = scale * (q - zero_point)                           (float32)

#include "vk.h"

#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <torch/library.h>

#include <ATen/Context.h>

#include <algorithm>
#include <limits>

namespace vk {
namespace {

at::ScalarType underlying(at::ScalarType q) {
  switch (q) {
    case at::kQUInt8:
      return at::kByte;
    case at::kQInt8:
      return at::kChar;
    case at::kQInt32:
      return at::kInt;
    default:
      TORCH_CHECK_NOT_IMPLEMENTED(false, "torch_vulkan: quantized dtype ", q,
                                  " is not supported (quint8, qint8 and qint32 are)");
  }
}

void check_per_tensor_affine(const at::Tensor& q) {
  TORCH_CHECK_NOT_IMPLEMENTED(q.qscheme() == at::kPerTensorAffine,
                              "torch_vulkan: only per-tensor affine quantized tensors are supported, got ", q.qscheme());
}

// The underlying integers of quantized tensor q, as a dense vk tensor sharing
// its storage (same sizes, strides and offset).
at::Tensor raw(const at::Tensor& q) {
  at::Tensor r = at::empty({0}, q.options().dtype(underlying(q.scalar_type())));
  r.set_(q.storage(), q.storage_offset(), q.sizes(), q.strides());
  return r;
}

// A new quantized vk tensor with q's quantizer, sizes and (dense) layout.
at::Tensor like(const at::Tensor& q, at::IntArrayRef sizes, at::MemoryFormat mf = at::MemoryFormat::Contiguous) {
  return at::new_qtensor(sizes, q.options().memory_format(mf), q.quantizer());
}

// Whether PyTorch's CPU kernels quantize through FBGEMM (its x86 builds;
// not its Windows ones, which use ATen's quantize_val): they differ for
// values out of range (inf, NaN, beyond int32), so vk follows the CPU it is
// installed with.
bool built_with_fbgemm() {
  static const bool fbgemm = [] {
    const auto& engines = at::globalContext().supportedQEngines();
    return std::find(engines.begin(), engines.end(), at::QEngine::FBGEMM) != engines.end();
  }();
  return fbgemm;
}

at::Tensor quantize_per_tensor(const at::Tensor& self, double scale, int64_t zero_point, at::ScalarType dtype) {
  TORCH_CHECK(self.scalar_type() == at::kFloat, "Quantize only works on Float Tensor, got ", self.scalar_type());
  at::ScalarType ut = underlying(dtype);
  int64_t qmin = ut == at::kByte ? 0 : ut == at::kChar ? -128 : std::numeric_limits<int32_t>::min();
  int64_t qmax = ut == at::kByte ? 255 : ut == at::kChar ? 127 : std::numeric_limits<int32_t>::max();
  auto quantizer = at::make_per_tensor_affine_quantizer(scale, zero_point, dtype);
  at::Tensor q = at::new_qtensor(self.sizes(), self.options().dtype(dtype).memory_format(self.suggest_memory_format()), quantizer);
  float inv = 1.0f / static_cast<float>(scale);  // (rounded once, as the CPU's)
  at::Tensor t = at::mul(self, at::scalar_tensor(inv, self.options()));
  auto full = [&](double v) { return at::full({}, v, t.options()); };
  at::Tensor out;
  if (built_with_fbgemm() && ut != at::kInt) {
    // FBGEMM's vector Quantize: t clipped at the largest float below 2^31
    // (a NaN too: _mm256_min_ps returns that bound), rounded to int32 (below
    // -2^31: INT32_MIN), plus the zero point in int32 (wrapping), clamped
    constexpr double kTop = 2147483520.0;
    at::Tensor tc = at::where(at::isnan(t), full(kTop), at::minimum(t, full(kTop)));
    at::Tensor r = at::round(tc);
    at::Tensor iv = at::where(at::lt(r, -2147483648.0), full(-2147483648.0), r).to(at::kLong) + zero_point;
    out = at::clamp(iv.to(at::kInt), qmin, qmax);
  } else if (built_with_fbgemm()) {
    // FBGEMM's scalar Quantize (qint32): float(zero_point + nearbyint(t))
    // clamped in double; a NaN converts to INT32_MIN
    at::Tensor r = at::round(t) + static_cast<double>(zero_point);
    at::Tensor c = at::clamp(at::where(at::isnan(r), full(static_cast<double>(qmin)), r), static_cast<double>(qmin),
                             static_cast<double>(qmax));
    out = at::clamp(c.to(at::kLong), qmin, qmax);  // (qmax is not a float32: again in int64)
  } else {
    // ATen's quantize_val (builds without FBGEMM): int64(zero_point +
    // nearbyint(t)) in float32 before the clamp, a value beyond int64 (inf,
    // NaN) converting to INT64_MIN on x86, so qmin
    at::Tensor r = at::round(t) + static_cast<double>(zero_point);
    at::Tensor c = at::where(at::lt(at::abs(r), 9.223372036854775808e18),
                             at::clamp(r, static_cast<double>(qmin), static_cast<double>(qmax)),
                             full(static_cast<double>(qmin)));
    out = at::clamp(c.to(at::kLong), qmin, qmax);
  }
  raw(q).copy_(out);
  return q;
}

at::Tensor quantize_per_tensor_tensor_qparams(const at::Tensor& self, const at::Tensor& scale,
                                              const at::Tensor& zero_point, at::ScalarType dtype) {
  return vk::quantize_per_tensor(self, scale.item<double>(), zero_point.item<int64_t>(), dtype);
}

at::Tensor dequantize(const at::Tensor& self) {
  check_per_tensor_affine(self);
  float scale = static_cast<float>(self.q_scale());
  // (q - zero_point in int32, as the CPU's int arithmetic, wrapping for qint32's extremes)
  at::Tensor diff = raw(self).to(at::kInt) - at::scalar_tensor(static_cast<int32_t>(self.q_zero_point()),
                                                               self.options().dtype(at::kInt));
  return at::mul(diff.to(at::kFloat), at::scalar_tensor(scale, self.options().dtype(at::kFloat)));
}

at::Tensor int_repr(const at::Tensor& self) {
  return raw(self).clone(at::MemoryFormat::Preserve);
}

at::Tensor clone(const at::Tensor& self, std::optional<at::MemoryFormat> memory_format) {
  at::MemoryFormat mf = memory_format.value_or(at::MemoryFormat::Preserve);
  if (mf == at::MemoryFormat::Preserve) mf = self.suggest_memory_format();
  at::Tensor out = like(self, self.sizes(), mf);
  raw(out).copy_(raw(self));
  return out;
}

at::Tensor& copy_(at::Tensor& self, const at::Tensor& src, bool /*non_blocking*/) {
  TORCH_CHECK(self.is_quantized() && src.is_quantized(),
              "torch_vulkan: copies between quantized and other tensors are not supported");
  TORCH_CHECK(self.scalar_type() == src.scalar_type(), "Quantized Copy only works with same qtype");
  TORCH_CHECK(self.qscheme() == src.qscheme(), "Quantized Copy only works with same qscheme");
  at::set_quantizer_(self, src.quantizer());
  if (is_vk(self) && is_vk(src)) {
    raw(self).copy_(raw(src));
  } else {
    // (to or from the CPU: the integers, through a dense tensor of their dtype)
    at::Tensor s = src.is_cpu() ? src.int_repr() : raw(src);
    at::Tensor d = self.is_cpu() ? at::empty({0}, s.options().device(at::kCPU)) : raw(self);
    if (self.is_cpu()) {
      d.set_(self.storage(), self.storage_offset(), self.sizes(), self.strides());
    }
    d.copy_(s);
  }
  return self;
}

at::Tensor flip(const at::Tensor& self, at::IntArrayRef dims) {
  at::Tensor out = like(self, self.sizes());
  raw(out).copy_(at::flip(raw(self), dims));
  return out;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("quantize_per_tensor", TORCH_FN(vk::quantize_per_tensor));
  m.impl("quantize_per_tensor.tensor_qparams", TORCH_FN(vk::quantize_per_tensor_tensor_qparams));
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("_empty_affine_quantized", TORCH_FN(at::native::empty_affine_quantized));
  m.impl("empty_quantized", TORCH_FN(at::native::empty_quantized));
  m.impl("empty.memory_format", TORCH_FN(at::native::empty_unknown_quantized));
  m.impl("empty_strided", TORCH_FN(at::native::empty_strided_unknown_quantized));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  m.impl("as_strided", TORCH_FN(at::native::as_strided_qtensorimpl));
  m.impl("q_scale", TORCH_FN(at::native::q_scale_quant));
  m.impl("q_zero_point", TORCH_FN(at::native::q_zero_point_quant));
  m.impl("qscheme", TORCH_FN(at::native::qscheme_quant));
  m.impl("dequantize.self", TORCH_FN(vk::dequantize));
  m.impl("int_repr", TORCH_FN(vk::int_repr));
  m.impl("clone", TORCH_FN(vk::clone));
  m.impl("copy_", TORCH_FN(vk::copy_));
  m.impl("flip", TORCH_FN(vk::flip));
}
