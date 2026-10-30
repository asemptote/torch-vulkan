// Data movement and indexing aten kernels for device "vk" (M9): gather and
// scatter-style indexing (index, index_put_, index_select, gather, scatter,
// index_copy, index_fill_, take, put_, masked_select, masked_scatter_),
// nonzero, cat, flip, roll, padding, and index sequences (arange, linspace,
// logspace, eye).
//
// Every indexing op reduces to the same two steps:
//  1. element offsets (int64, the result's shape) = sum over index tensors of
//     wrap(index) * stride, computed by shaders/index.slang, which also
//     checks bounds (an out-of-range index sets the device error word, raised
//     as IndexError at the next synchronize, like CUDA's device asserts);
//  2. shaders/move.slang copies elements by size (any dtype): GATHER reads
//     the source restrided to the result's shape (indexed dims at stride 0)
//     at those offsets, SCATTER writes the destination likewise.
// Ops that accumulate (index_put_ with accumulate=True, put_ with accumulate,
// scatter with a reduction) need sorting or atomics and run on the CPU until
// M10.

#include "opcodes.h"
#include "vk.h"

#include <ATen/ExpandUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace vk {

namespace {

// Size and element stride of dim `d`, treating a 0-dim tensor as one element.
int64_t size_at(const at::Tensor& t, int64_t d) {
  return t.dim() == 0 ? 1 : t.size(d);
}

int64_t stride_at(const at::Tensor& t, int64_t d) {
  return t.dim() == 0 ? 0 : t.stride(d);
}

// One index tensor's contribution to element offsets: wrap(index) * stride,
// with index broadcast to the result's shape and checked against `size`.
struct IndexTerm {
  at::Tensor index;
  int64_t size;
  int64_t stride;
};

bool index_dtype_ok(const at::Tensor& index) {
  return index.scalar_type() == at::kLong || index.scalar_type() == at::kInt;
}

// Element offsets (int64 vk tensor of `shape`) = sum of the terms. With
// `wrap`, negative indices count from the end (index, take, put_,
// index_fill_); without it they are out of range (gather, index_select,
// scatter, index_copy), as on the CPU.
at::Tensor index_offsets(at::IntArrayRef shape, const std::vector<IndexTerm>& terms, const at::TensorOptions& opts,
                         bool wrap = true) {
  at::Tensor offsets = at::empty(shape, opts.dtype(at::kLong));
  if (offsets.numel() == 0) {
    return offsets;
  }
  if (terms.empty()) {
    return offsets.zero_();
  }
  int64_t error = static_cast<int64_t>(device_error_address());
  for (size_t k = 0; k < terms.size(); ++k) {
    const IndexTerm& t = terms[k];
    TORCH_CHECK_INDEX(t.size > 0, "index is out of bounds for dimension with size 0");
    at::Tensor index = index_dtype_ok(t.index) ? t.index : t.index.to(at::kLong);
    std::string kernel = index.scalar_type() == at::kLong ? "index_i64" : "index_i32";
    launch_strided(kernel.c_str(), {k == 0 ? 0u : 1u, wrap ? 1u : 0u}, offsets, k == 0 ? nullptr : &offsets, &index,
                   {t.stride, t.size, error});
  }
  return offsets;
}

// out = src_view[offsets], src_view being the source restrided to out's shape.
void gather(const at::Tensor& out, const at::Tensor& src_view, const at::Tensor& offsets) {
  std::string kernel = move_kernel(out.itemsize());
  launch_strided(kernel.c_str(), {op::MOVE_GATHER, 0}, out, &src_view, &offsets);
}

// dst_view[offsets] = values (broadcast to dst_view's shape).
void scatter(const at::Tensor& dst_view, const at::Tensor& values, const at::Tensor& offsets) {
  std::string kernel = move_kernel(dst_view.itemsize());
  launch_strided(kernel.c_str(), {op::MOVE_SCATTER, 0}, dst_view, &values, &offsets);
}

// dst_view[offsets] = value.
void scatter_fill(const at::Tensor& dst_view, const at::Scalar& value, const at::Tensor& offsets) {
  std::string kernel = move_kernel(dst_view.itemsize());
  std::array<uint64_t, 2> bits = fill_bits(value, dst_view.scalar_type());
  launch_strided_bits(kernel.c_str(), {op::MOVE_SCATTER, 1}, dst_view, nullptr, &offsets, bits);
}

// `t` restrided to `shape` (same rank) with dim `dim` at stride 0.
at::Tensor restride_without(const at::Tensor& t, at::IntArrayRef shape, int64_t dim) {
  if (t.dim() == 0) {
    return t.as_strided(shape, std::vector<int64_t>(shape.size(), 0), t.storage_offset());
  }
  std::vector<int64_t> strides = t.strides().vec();
  strides[dim] = 0;
  return t.as_strided(shape, strides, t.storage_offset());
}

// A 1-D (or 0-D) index viewed along dim `dim` of a rank-`ndim` shape.
at::Tensor index_along(const at::Tensor& index, int64_t dim, int64_t ndim) {
  std::vector<int64_t> shape(std::max<int64_t>(ndim, 1), 1);
  shape[ndim == 0 ? 0 : dim] = index.numel();
  at::Tensor flat = index.reshape({-1});
  return ndim == 0 ? flat.reshape({}) : flat.reshape(shape);
}

bool movable(const at::Tensor& t) {
  return is_vk(t) && !move_kernel(t.itemsize()).empty();
}

// Linear (row-major) indices into `t`, already wrapped and checked, to
// element offsets in t's layout.
at::Tensor linear_to_offsets(const at::Tensor& linear, const at::Tensor& t) {
  if (t.is_contiguous()) {
    return linear;
  }
  at::Tensor offsets = at::zeros_like(linear);
  at::Tensor rem = linear;
  for (int64_t d = t.dim() - 1; d >= 0; --d) {
    at::Tensor coord = at::remainder(rem, t.size(d));
    offsets.add_(coord, t.stride(d));
    rem = at::div(rem, t.size(d), "floor");
  }
  return offsets;
}

// ---------------------------------------------------------------- index_select, gather, take

at::Tensor& index_select_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& out) {
  dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  std::vector<int64_t> shape = self.sizes().vec();
  if (self.dim() > 0) {
    shape[dim] = index.numel();
  }
  at::native::resize_output(out, shape);
  if (!movable(self) || !is_vk(index) || !is_vk(out) || out.scalar_type() != self.scalar_type() ||
      !index_dtype_ok(index)) {
    on_cpu("aten::index_select", "out", self, dim, index, out);
    return out;
  }
  at::Tensor src_view = restride_without(self, shape, dim);
  at::Tensor offsets = index_offsets(
      shape, {{index_along(index, dim, self.dim()), size_at(self, dim), stride_at(self, dim)}}, self.options(),
      /*wrap=*/false);
  gather(out, src_view, offsets);
  return out;
}

at::Tensor index_select(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  at::Tensor out = at::empty({0}, self.options());
  return index_select_out(self, dim, index, out);
}

at::Tensor& gather_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, bool sparse_grad,
                       at::Tensor& out) {
  dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
  at::native::resize_output(out, index.sizes());
  if (!movable(self) || !is_vk(index) || !is_vk(out) || out.scalar_type() != self.scalar_type() ||
      !index_dtype_ok(index) || std::max<int64_t>(index.dim(), 1) != std::max<int64_t>(self.dim(), 1)) {
    on_cpu("aten::gather", "out", self, dim, index, sparse_grad, out);
    return out;
  }
  at::Tensor src_view = restride_without(self, index.sizes(), dim);
  at::Tensor offsets =
      index_offsets(index.sizes(), {{index, size_at(self, dim), stride_at(self, dim)}}, self.options(), /*wrap=*/false);
  gather(out, src_view, offsets);
  return out;
}

// Wrapped, checked linear indices of `index` into a tensor of `numel` elements.
at::Tensor checked_linear(const at::Tensor& index, int64_t numel) {
  return index_offsets(index.sizes(), {{index, numel, 1}}, index.options());
}

at::Tensor& take_out(const at::Tensor& self, const at::Tensor& index, at::Tensor& out) {
  at::native::resize_output(out, index.sizes());
  if (!movable(self) || !is_vk(index) || !is_vk(out) || out.scalar_type() != self.scalar_type() ||
      index.scalar_type() != at::kLong) {
    on_cpu("aten::take", "out", self, index, out);
    return out;
  }
  if (index.numel() == 0) {
    return out;
  }
  at::Tensor offsets = linear_to_offsets(checked_linear(index, self.numel()), self);
  at::Tensor src_view = self.as_strided(index.sizes(), std::vector<int64_t>(index.dim(), 0), self.storage_offset());
  gather(out, src_view, offsets);
  return out;
}

at::Tensor take(const at::Tensor& self, const at::Tensor& index) {
  at::Tensor out = at::empty({0}, self.options());
  return take_out(self, index, out);
}

// ---------------------------------------------------------------- advanced indexing

// What at::native::make_info computes (its AdvancedIndex constructor is not
// exported): `src` restrided to the result's shape with the indexed dims
// replaced by the broadcast index shape at stride 0, and one offset term per
// index tensor.
struct Advanced {
  at::Tensor src;
  std::vector<IndexTerm> terms;
};

Advanced make_advanced(at::Tensor self, const c10::List<std::optional<at::Tensor>>& orig) {
  at::native::checkIndexTensorTypes(orig, /*allow_int=*/true);
  std::vector<at::Tensor> indices = at::native::expandTensors(self, orig, /*ensure_same_device=*/true);
  try {
    indices = at::expand_outplace(indices);
  } catch (std::exception&) {
    TORCH_CHECK_INDEX(false, "shape mismatch: indexing tensors could not be broadcast together");
  }
  while (indices.size() < static_cast<size_t>(self.dim())) {
    indices.emplace_back();
  }
  if (!at::native::hasContiguousSubspace(indices)) {
    std::tie(self, indices) = at::native::transposeToFront(self, indices);
  }
  int64_t before = 0, after = 0, indexed = 0;
  at::IntArrayRef replacement;
  std::vector<int64_t> sizes, strides;
  for (size_t d = 0; d < indices.size(); ++d) {
    if (!indices[d].defined()) {
      (indexed == 0 ? before : after)++;
    } else {
      indexed++;
      replacement = indices[d].sizes();
      sizes.push_back(self.size(d));
      strides.push_back(self.stride(d));
    }
  }
  std::vector<int64_t> shape, src_strides;
  for (int64_t d = 0; d < before; ++d) {
    shape.push_back(self.size(d));
    src_strides.push_back(self.stride(d));
  }
  for (int64_t s : replacement) {
    shape.push_back(s);
    src_strides.push_back(0);
  }
  for (int64_t d = before + indexed; d < self.dim(); ++d) {
    shape.push_back(self.size(d));
    src_strides.push_back(self.stride(d));
  }
  Advanced info;
  info.src = self.as_strided(shape, src_strides, self.storage_offset());
  size_t k = 0;
  for (const at::Tensor& index : indices) {
    if (!index.defined()) {
      continue;
    }
    std::vector<int64_t> view(before, 1);
    view.insert(view.end(), index.sizes().begin(), index.sizes().end());
    view.insert(view.end(), after, 1);
    info.terms.push_back({index.reshape(view), sizes[k], strides[k]});
    ++k;
  }
  return info;
}

bool indices_on_vk(const c10::List<std::optional<at::Tensor>>& indices) {
  for (const std::optional<at::Tensor>& t : indices) {
    if (t.has_value() && t->defined() && !is_vk(*t) && !(t->dim() == 0 && t->device().is_cpu())) {
      return false;
    }
  }
  return true;
}

at::Tensor& index_out(const at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices, at::Tensor& out) {
  if (!movable(self) || !is_vk(out) || out.scalar_type() != self.scalar_type() || !indices_on_vk(indices)) {
    on_cpu("aten::index", "Tensor_out", self, indices, out);
    return out;
  }
  Advanced info = make_advanced(self, indices);
  at::native::resize_output(out, info.src.sizes());
  gather(out, info.src, index_offsets(info.src.sizes(), info.terms, self.options()));
  return out;
}

at::Tensor index(const at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices) {
  at::Tensor out = at::empty({0}, self.options());
  return index_out(self, indices, out);
}

// Defined with the accumulating ops below.
bool accum_supported(at::ScalarType t, bool sum_only);
void accumulate_into(const at::Tensor& dst_view, const at::Tensor& offsets, const at::Tensor& values);

at::Tensor& index_put_impl(at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
                           const at::Tensor& values, bool accumulate, bool unsafe) {
  bool values_ok = is_vk(values) || (values.dim() == 0 && values.device().is_cpu());
  if (!movable(self) || !values_ok || !indices_on_vk(indices) ||
      (accumulate && !accum_supported(self.scalar_type(), true))) {
    on_cpu("aten::_index_put_impl_", "", self, indices, values, accumulate, unsafe);
    return self;
  }
  Advanced info = make_advanced(self, indices);
  if (info.src.numel() == 0) {
    return self;
  }
  at::Tensor v = values.to(self.device(), self.scalar_type());
  at::Tensor offsets = index_offsets(info.src.sizes(), info.terms, self.options());
  if (accumulate) {
    accumulate_into(info.src, offsets, v);
  } else {
    scatter(info.src, v, offsets);
  }
  return self;
}

// ---------------------------------------------------------------- scatter, index_copy, index_fill_, put_


// The destination restrided to the index's shape (index sizes may be
// smaller than the destination's in every dim but `dim`).
bool scatter_ok(const at::Tensor& self, const at::Tensor& index, int64_t dim) {
  return movable(self) && is_vk(index) && index_dtype_ok(index) &&
         std::max<int64_t>(index.dim(), 1) == std::max<int64_t>(self.dim(), 1);
}

void scatter_into(const at::Tensor& out, int64_t dim, const at::Tensor& index, const at::Tensor* src,
                  const at::Scalar* value) {
  if (index.numel() == 0) {
    return;
  }
  at::Tensor dst_view = restride_without(out, index.sizes(), dim);
  at::Tensor offsets =
      index_offsets(index.sizes(), {{index, size_at(out, dim), stride_at(out, dim)}}, out.options(), /*wrap=*/false);
  if (src != nullptr) {
    at::Tensor values = src->dim() == 0 ? *src : src->as_strided(index.sizes(), src->strides(), src->storage_offset());
    scatter(dst_view, values.to(out.scalar_type()), offsets);
  } else {
    scatter_fill(dst_view, *value, offsets);
  }
}

at::Tensor& scatter_src_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src,
                            at::Tensor& out) {
  dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
  if (!scatter_ok(self, index, dim) || !is_vk(src) || !is_vk(out)) {
    on_cpu("aten::scatter", "src_out", self, dim, index, src, out);
    return out;
  }
  if (!out.is_same(self)) {
    at::native::resize_output(out, self.sizes());
    out.copy_(self);
  }
  scatter_into(out, dim, index, &src, nullptr);
  return out;
}

at::Tensor& scatter_value_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Scalar& value,
                              at::Tensor& out) {
  dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
  if (!scatter_ok(self, index, dim) || !is_vk(out)) {
    on_cpu("aten::scatter", "value_out", self, dim, index, value, out);
    return out;
  }
  if (!out.is_same(self)) {
    at::native::resize_output(out, self.sizes());
    out.copy_(self);
  }
  scatter_into(out, dim, index, nullptr, &value);
  return out;
}

at::Tensor& index_copy_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& source,
                           at::Tensor& out) {
  dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
  if (!movable(self) || !is_vk(index) || !is_vk(source) || !is_vk(out) || index.scalar_type() != at::kLong ||
      index.dim() > 1) {
    on_cpu("aten::index_copy", "out", self, dim, index, source, out);
    return out;
  }
  if (!out.is_same(self)) {
    at::native::resize_output(out, self.sizes());
    out.copy_(self);
  }
  if (source.numel() == 0) {
    return out;
  }
  at::Tensor dst_view = restride_without(out, source.sizes(), dim);
  at::Tensor offsets = index_offsets(
      source.sizes(), {{index_along(index, dim, source.dim()), size_at(out, dim), stride_at(out, dim)}}, out.options(),
      /*wrap=*/false);
  scatter(dst_view, source.to(out.scalar_type()), offsets);
  return out;
}

at::Tensor& index_fill_scalar(at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Scalar& value) {
  dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
  if (!movable(self) || !is_vk(index) || index.scalar_type() != at::kLong || index.dim() > 1) {
    on_cpu("aten::index_fill_", "int_Scalar", self, dim, index, value);
    return self;
  }
  std::vector<int64_t> shape = self.sizes().vec();
  if (self.dim() > 0) {
    shape[dim] = index.numel();
  }
  at::Tensor dst_view = restride_without(self, shape, dim);
  if (dst_view.numel() == 0) {
    return self;
  }
  at::Tensor offsets =
      index_offsets(shape, {{index_along(index, dim, self.dim()), size_at(self, dim), stride_at(self, dim)}}, self.options());
  scatter_fill(dst_view, value, offsets);
  return self;
}

at::Tensor& index_fill_tensor(at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor with ",
              value.dim(), " dimension(s).");
  return index_fill_scalar(self, dim, index, value.item());
}

at::Tensor& put_(at::Tensor& self, const at::Tensor& index, const at::Tensor& source, bool accumulate) {
  if (!movable(self) || !is_vk(index) || !is_vk(source) || index.scalar_type() != at::kLong ||
      (accumulate && !accum_supported(self.scalar_type(), true))) {
    on_cpu("aten::put_", "", self, index, source, accumulate);
    return self;
  }
  TORCH_CHECK(index.numel() == source.numel(), "put_(): Expected source and index to have the same number of elements");
  if (index.numel() == 0) {
    return self;
  }
  at::Tensor offsets = linear_to_offsets(checked_linear(index, self.numel()), self);
  at::Tensor dst_view = self.as_strided(index.sizes(), std::vector<int64_t>(index.dim(), 0), self.storage_offset());
  at::Tensor values = source.reshape(index.sizes()).to(self.scalar_type());
  if (accumulate) {
    accumulate_into(dst_view, offsets, values);
  } else {
    scatter(dst_view, values, offsets);
  }
  return self;
}

// ---------------------------------------------------------------- accumulation into repeated indices

// No atomics: the source elements are sorted (stably) by destination
// offset, and shaders/segreduce.slang folds each run of equal offsets in
// source order, like the CPU's sequential loop (so float sums agree).

struct SegReduceParams {
  uint64_t dst, offs, vals, order;
  uint32_t n, stride;
};
static_assert(sizeof(SegReduceParams) == 40, "must match shaders/segreduce.slang");

const char* accum_label(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kDouble: return has_feature(kFeatureFloat64) ? "f64" : nullptr;
    case at::kInt: return "i32";
    case at::kLong: return "i64";
    default: return nullptr;
  }
}

// Also bool, the small types (through their widened type) and, for sums,
// complex (through the real and imaginary parts).
bool accum_supported(at::ScalarType t, bool sum_only) {
  if (accum_label(t) != nullptr) return true;
  if (t == at::kBool || widened(t) != t) return accum_label(t == at::kBool ? at::kInt : widened(t)) != nullptr;
  if (t == at::kComplexFloat) return sum_only;
  if (t == at::kComplexDouble) return sum_only && has_feature(kFeatureFloat64);
  return false;
}

bool sum_like(uint32_t op) {
  return op == op::SEGREDUCE_SUM || op == op::SEGREDUCE_MEAN;
}

uint32_t reduce_code(c10::string_view reduce) {
  if (reduce == "sum" || reduce == "add") return op::SEGREDUCE_SUM;
  if (reduce == "prod" || reduce == "multiply") return op::SEGREDUCE_PROD;
  if (reduce == "amax") return op::SEGREDUCE_MAX;
  if (reduce == "amin") return op::SEGREDUCE_MIN;
  if (reduce == "mean") return op::SEGREDUCE_MEAN;
  TORCH_CHECK(false, "reduce argument must be either sum, prod, mean, amax or amin, got ", reduce);
}

// Element offsets of every position of `view` (its own strides; stride-0
// dims contribute nothing), as an int64 tensor of its shape.
at::Tensor view_positions(const at::Tensor& view) {
  std::vector<IndexTerm> terms;
  for (int64_t d = 0; d < view.dim(); ++d) {
    if (view.stride(d) != 0 && view.size(d) > 1) {
      at::Tensor i = at::arange(view.size(d), view.options().dtype(at::kLong));
      terms.push_back({index_along(i, d, view.dim()), view.size(d), view.stride(d)});
    }
  }
  return index_offsets(view.sizes(), terms, view.options());
}

// dst_view[offsets] = op(dst_view[offsets] (if include_self), values...),
// values broadcast to the offsets' shape. As for MOVE_SCATTER, offsets are
// added to each element's own position in dst_view.
void accumulate(const at::Tensor& dst_view, const at::Tensor& offsets, const at::Tensor& values, uint32_t op,
                bool include_self) {
  int64_t n = offsets.numel();
  if (n == 0) {
    return;
  }
  at::ScalarType t = dst_view.scalar_type();
  if (accum_label(t) == nullptr) {
    // The whole storage as a flat tensor, so that dst_view's strides and
    // offset (and the offsets) carry over to a widened or real copy.
    int64_t itemsize = static_cast<int64_t>(dst_view.element_size());
    int64_t total = static_cast<int64_t>(dst_view.storage().nbytes()) / itemsize;
    at::Tensor flat = at::empty({0}, dst_view.options()).set_(dst_view.storage(), 0, {total}, {1});
    if (at::isComplexType(t)) {
      // Sums of complex values are sums of their parts.
      at::Tensor parts = at::view_as_real(flat);  // [total, 2]
      at::Tensor vals = values.to(t).expand(offsets.sizes());
      std::vector<int64_t> strides;
      for (int64_t st : dst_view.strides()) strides.push_back(2 * st);
      for (int64_t k = 0; k < 2; ++k) {
        at::Tensor part = parts.as_strided(dst_view.sizes(), strides, 2 * dst_view.storage_offset() + k);
        at::Tensor v = k == 0 ? at::real(vals) : at::imag(vals);
        accumulate(part, offsets * 2, v, op, include_self);
      }
      return;
    }
    // bool (as int32: sums become "or", products "and") and the small types.
    at::ScalarType wide = t == at::kBool ? at::kInt : widened(t);
    at::Tensor wflat = flat.to(wide);
    accumulate(wflat.as_strided(dst_view.sizes(), dst_view.strides(), dst_view.storage_offset()), offsets,
               values.to(wide), op, include_self);
    flat.copy_(wflat);
    return;
  }
  TORCH_CHECK(n < (int64_t(1) << 31), "torch_vulkan: accumulation of more than 2^31 elements");
  at::Tensor offs = (offsets + view_positions(dst_view)).reshape({1, n}).contiguous();
  at::Tensor vals = values.to(dst_view.scalar_type()).expand(offsets.sizes()).contiguous();
  at::Tensor order = sort_rows(offs, false);
  SegReduceParams p{};
  p.dst = gpu_address(dst_view);
  p.offs = gpu_address(offs);
  p.vals = gpu_address(vals);
  p.order = gpu_address(order);
  p.n = static_cast<uint32_t>(n);
  uint32_t groups = 0;
  check(vkt_grid_1d(p.n, &groups, &p.stride));
  dispatch(std::string("segreduce_") + accum_label(dst_view.scalar_type()), {op, include_self ? 1u : 0u}, p, groups);
}

// out = self, then out[index along dim] op= src (scatter_add, scatter_reduce, scatter(reduce=...)).
bool scatter_accumulate(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src,
                        uint32_t op, bool include_self, at::Tensor& out) {
  dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
  if (!scatter_ok(self, index, dim) || !is_vk(src) || !is_vk(out) ||
      !accum_supported(self.scalar_type(), sum_like(op)) || out.scalar_type() != self.scalar_type()) {
    return false;
  }
  if (!out.is_same(self)) {
    at::native::resize_output(out, self.sizes());
    out.copy_(self);
  }
  if (index.numel() == 0) {
    return true;
  }
  at::Tensor dst_view = restride_without(out, index.sizes(), dim);
  at::Tensor offsets =
      index_offsets(index.sizes(), {{index, size_at(out, dim), stride_at(out, dim)}}, out.options(), /*wrap=*/false);
  at::Tensor values = src.dim() == 0 ? src : src.as_strided(index.sizes(), src.strides(), src.storage_offset());
  accumulate(dst_view, offsets, values, op, include_self);
  return true;
}

// out = self, then out[index[i] along dim] op= source[i along dim] (index_add, index_reduce).
bool index_accumulate(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& source,
                      uint32_t op, bool include_self, at::Tensor& out) {
  dim = at::maybe_wrap_dim(dim, std::max<int64_t>(self.dim(), 1));
  if (!movable(self) || !is_vk(index) || !is_vk(source) || !is_vk(out) || index.dim() > 1 ||
      !index_dtype_ok(index) || !accum_supported(self.scalar_type(), sum_like(op)) ||
      out.scalar_type() != self.scalar_type()) {
    return false;
  }
  if (!out.is_same(self)) {
    at::native::resize_output(out, self.sizes());
    out.copy_(self);
  }
  if (source.numel() == 0) {
    return true;
  }
  at::Tensor dst_view = restride_without(out, source.sizes(), dim);
  at::Tensor offsets = index_offsets(
      source.sizes(), {{index_along(index, dim, source.dim()), size_at(out, dim), stride_at(out, dim)}}, out.options(),
      /*wrap=*/false);
  accumulate(dst_view, offsets, source, op, include_self);
  return true;
}

void accumulate_into(const at::Tensor& dst_view, const at::Tensor& offsets, const at::Tensor& values) {
  accumulate(dst_view, offsets, values, op::SEGREDUCE_SUM, true);
}

at::Tensor& scatter_add_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src,
                            at::Tensor& out) {
  if (!scatter_accumulate(self, dim, index, src, op::SEGREDUCE_SUM, true, out)) {
    on_cpu("aten::scatter_add", "out", self, dim, index, src, out);
  }
  return out;
}

at::Tensor& scatter_reduce_two_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                                   const at::Tensor& src, c10::string_view reduce, bool include_self,
                                   at::Tensor& out) {
  if (!scatter_accumulate(self, dim, index, src, reduce_code(reduce), include_self, out)) {
    on_cpu("aten::scatter_reduce", "two_out", self, dim, index, src, reduce, include_self, out);
  }
  return out;
}

at::Tensor& scatter_reduce_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& src,
                               c10::string_view reduce, at::Tensor& out) {
  if (!scatter_accumulate(self, dim, index, src, reduce_code(reduce), true, out)) {
    on_cpu("aten::scatter", "reduce_out", self, dim, index, src, reduce, out);
  }
  return out;
}

at::Tensor& scatter_value_reduce_out(const at::Tensor& self, int64_t dim, const at::Tensor& index,
                                     const at::Scalar& value, c10::string_view reduce, at::Tensor& out) {
  at::Tensor src = at::scalar_tensor(value, self.options());
  if (!scatter_accumulate(self, dim, index, src, reduce_code(reduce), true, out)) {
    on_cpu("aten::scatter", "value_reduce_out", self, dim, index, value, reduce, out);
  }
  return out;
}

at::Tensor& index_add_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& source,
                          const at::Scalar& alpha, at::Tensor& out) {
  bool unit = alpha.equal(1) || (source.scalar_type() == at::kBool && alpha.toBool());
  at::Tensor scaled = unit ? source : source.scalar_type() == at::kBool ? at::zeros_like(source) : source * alpha;
  if (!index_accumulate(self, dim, index, scaled, op::SEGREDUCE_SUM, true, out)) {
    on_cpu("aten::index_add", "out", self, dim, index, source, alpha, out);
  }
  return out;
}

at::Tensor& index_reduce_out(const at::Tensor& self, int64_t dim, const at::Tensor& index, const at::Tensor& source,
                             c10::string_view reduce, bool include_self, at::Tensor& out) {
  TORCH_CHECK(reduce == "prod" || reduce == "mean" || reduce == "amax" || reduce == "amin",
              "index_reduce(): Expected reduce to be one of prod, mean, amax or amin but got ", reduce, ".");
  if (!index_accumulate(self, dim, index, source, reduce_code(reduce), include_self, out)) {
    on_cpu("aten::index_reduce", "out", self, dim, index, source, reduce, include_self, out);
  }
  return out;
}

at::Tensor bincount(const at::Tensor& self, const std::optional<at::Tensor>& weights, int64_t minlength) {
  TORCH_CHECK(minlength >= 0, "minlength should be >= 0");
  TORCH_CHECK(self.dim() == 1 && !at::isFloatingType(self.scalar_type()) && self.scalar_type() != at::kBool,
              "bincount only supports 1-d non-negative integral inputs.");
  bool has_w = weights.has_value() && weights->defined();
  at::ScalarType dtype = !has_w ? at::kLong : at::isFloatingType(weights->scalar_type()) ? weights->scalar_type()
                                                                                         : at::kDouble;
  if (!is_vk(self) || (has_w && !is_vk(*weights)) || accum_label(dtype) == nullptr) {
    return on_cpu("aten::bincount", "", self, weights, minlength)[0].toTensor();
  }
  TORCH_CHECK(!has_w || weights->sizes() == self.sizes(), "weights should be 1-d and have the same length as input");
  if (self.numel() == 0) {
    return at::zeros({minlength}, self.options().dtype(dtype));
  }
  at::Tensor bounds = at::stack({self.min(), self.max()}).to(at::kLong).cpu();
  TORCH_CHECK(bounds[0].item<int64_t>() >= 0, "bincount only supports 1-d non-negative integral inputs.");
  int64_t size = std::max(bounds[1].item<int64_t>() + 1, minlength);
  at::Tensor out = at::zeros({size}, self.options().dtype(dtype));
  at::Tensor values = has_w ? *weights : at::ones({self.numel()}, self.options().dtype(dtype));
  at::Tensor dst_view = out.as_strided({self.numel()}, {0});
  accumulate(dst_view, self.to(at::kLong), values, op::SEGREDUCE_SUM, true);
  return out;
}

at::Tensor& histc_out(const at::Tensor& self, int64_t bins, const at::Scalar& min, const at::Scalar& max,
                      at::Tensor& out) {
  TORCH_CHECK(bins > 0, "torch.histc(): bins must be > 0, but got ", bins);
  at::ScalarType t = self.scalar_type();
  if (!is_vk(self) || !is_vk(out) || !at::isFloatingType(t) || accum_label(t) == nullptr) {
    on_cpu("aten::histc", "out", self, bins, min, max, out);
    return out;
  }
  double lo = min.toDouble(), hi = max.toDouble();
  if (lo == hi && self.numel() > 0) {
    at::Tensor bounds = at::stack({self.min(), self.max()}).to(at::kDouble).cpu();
    lo = bounds[0].item<double>();
    hi = bounds[1].item<double>();
  }
  if (lo == hi) {
    lo -= 1;
    hi += 1;
  }
  TORCH_CHECK(std::isfinite(lo) && std::isfinite(hi), "torch.histc(): range of [", lo, ", ", hi, "] is not finite");
  TORCH_CHECK(lo < hi, "torch.histc(): max must be larger than min");
  at::native::resize_output(out, {bins});
  out.zero_();
  if (self.numel() == 0) {
    return out;
  }
  // Bin b holds edges[b] <= x < edges[b + 1]; x == max goes in the last bin.
  at::Tensor edges = at::linspace(lo, hi, bins + 1, self.options());
  at::Tensor x = self.reshape({-1});
  at::Tensor last = edges.narrow(0, bins, 1), first = edges.narrow(0, 0, 1);
  at::Tensor bucket = at::searchsorted(edges, x, false, /*right=*/true) - 1;
  bucket = at::where(at::eq(x, last), at::full_like(bucket, bins - 1), bucket);
  at::Tensor keep = at::logical_and(at::ge(x, first), at::le(x, last));
  at::Tensor offsets = at::masked_select(bucket, keep);
  at::Tensor dst_view = out.as_strided({offsets.numel()}, {0});
  accumulate(dst_view, offsets, at::ones({offsets.numel()}, out.options()), op::SEGREDUCE_SUM, true);
  return out;
}

at::Tensor histc(const at::Tensor& self, int64_t bins, const at::Scalar& min, const at::Scalar& max) {
  at::Tensor out = at::empty({0}, self.options());
  return histc_out(self, bins, min, max, out);
}


// ---------------------------------------------------------------- nonzero, masked_select, masked_scatter_

struct NonzeroParams {
  uint64_t mask, csum, out;
  uint32_t n, ndim, stride, pad;
  uint32_t size[8];
};
static_assert(sizeof(NonzeroParams) == 72, "must match shaders/nonzero.slang");

at::Tensor& nonzero_out(const at::Tensor& self, at::Tensor& out) {
  if (!is_vk(self) || !is_vk(out) || self.dim() > 8 || dtype_label(self.scalar_type()) == nullptr ||
      out.scalar_type() != at::kLong) {
    on_cpu("aten::nonzero", "out", self, out);
    return out;
  }
  int64_t ndim = self.dim();
  at::Tensor mask = (self.scalar_type() == at::kBool ? self : at::ne(self, 0)).contiguous().reshape({-1});
  int64_t count = 0;
  at::Tensor csum;
  if (mask.numel() > 0) {
    csum = at::cumsum(mask, 0, at::kLong);
    count = csum[-1].item<int64_t>();
  }
  at::native::resize_output(out, {count, ndim});
  if (count == 0 || ndim == 0) {
    return out;
  }
  // The kernel writes rows contiguously; use a temporary if out is strided.
  at::Tensor target = out.is_contiguous() ? out : at::empty({count, ndim}, out.options());
  NonzeroParams p{};
  p.mask = gpu_address(mask);
  p.csum = gpu_address(csum);
  p.out = gpu_address(target);
  TORCH_CHECK(mask.numel() < (int64_t(1) << 32), "torch_vulkan: nonzero of more than 2^32 elements");
  p.n = static_cast<uint32_t>(mask.numel());
  p.ndim = static_cast<uint32_t>(ndim);
  for (int64_t d = 0; d < ndim; ++d) {
    p.size[d] = static_cast<uint32_t>(self.size(d));
  }
  uint32_t groups = 0;
  check(vkt_grid_1d(p.n, &groups, &p.stride));
  dispatch("nonzero", {}, p, groups);
  if (!target.is_same(out)) {
    out.copy_(target);
  }
  return out;
}

at::Tensor nonzero_vk(const at::Tensor& self) {
  at::Tensor out = at::empty({0}, self.options().dtype(at::kLong));
  return nonzero_out(self, out);
}

at::Tensor nonzero_static(const at::Tensor& self, int64_t size, int64_t fill_value) {
  TORCH_CHECK(size >= 0, "nonzero_static: 'size' must be an non-negative integer");
  at::Tensor all = nonzero_vk(self);
  at::Tensor out = at::full({size, self.dim()}, fill_value, self.options().dtype(at::kLong));
  int64_t n = std::min(size, all.size(0));
  if (n > 0) {
    out.narrow(0, 0, n).copy_(all.narrow(0, 0, n));
  }
  return out;
}

at::Tensor count_nonzero(const at::Tensor& self, at::IntArrayRef dims) {
  return at::sum(at::ne(self, 0), dims, false, at::kLong);
}

at::Tensor& masked_select_out(const at::Tensor& self, const at::Tensor& mask, at::Tensor& out) {
  bool ok = movable(self) && is_vk(mask) && is_vk(out) && out.scalar_type() == self.scalar_type() &&
            (mask.scalar_type() == at::kBool || mask.scalar_type() == at::kByte);
  if (!ok) {
    on_cpu("aten::masked_select", "out", self, mask, out);
    return out;
  }
  std::vector<int64_t> shape = at::infer_size(self.sizes(), mask.sizes());
  at::Tensor positions = nonzero_vk(mask.expand(shape).reshape({-1})).reshape({-1});
  return index_select_out(self.expand(shape).reshape({-1}), 0, positions, out);
}

at::Tensor masked_select(const at::Tensor& self, const at::Tensor& mask) {
  at::Tensor out = at::empty({0}, self.options());
  return masked_select_out(self, mask, out);
}

at::Tensor& masked_scatter_(at::Tensor& self, const at::Tensor& mask, const at::Tensor& source) {
  bool ok = movable(self) && is_vk(mask) && is_vk(source) && mask.scalar_type() == at::kBool &&
            source.scalar_type() == self.scalar_type();
  if (!ok) {
    on_cpu("aten::masked_scatter_", "", self, mask, source);
    return self;
  }
  at::Tensor positions = nonzero_vk(mask.expand(self.sizes()).reshape({-1})).reshape({-1});
  int64_t count = positions.numel();
  TORCH_CHECK(source.numel() >= count, "masked_scatter: expected source to have at least ", count,
              " elements, but it has ", source.numel());
  if (count == 0) {
    return self;
  }
  at::Tensor offsets = linear_to_offsets(positions, self);
  at::Tensor dst_view = self.as_strided({count}, {0}, self.storage_offset());
  scatter(dst_view, source.reshape({-1}).narrow(0, 0, count), offsets);
  return self;
}

// ---------------------------------------------------------------- cat, flip, roll, padding

at::Tensor& cat_out(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out) {
  std::vector<at::Tensor> inputs;
  for (const at::Tensor& t : tensors) {
    if (!(t.dim() == 1 && t.size(0) == 0)) {  // legacy: 1-D empty tensors are skipped
      inputs.push_back(t);
    }
  }
  bool ok = is_vk(out);
  for (const at::Tensor& t : inputs) {
    ok = ok && is_vk(t);
  }
  if (!ok) {
    std::vector<at::Tensor> all(tensors.begin(), tensors.end());
    on_cpu("aten::cat", "out", all, dim, out);
    return out;
  }
  if (inputs.empty()) {
    at::native::resize_output(out, {0});
    return out;
  }
  dim = at::maybe_wrap_dim(dim, inputs[0].dim());
  std::vector<int64_t> shape = inputs[0].sizes().vec();
  int64_t total = 0;
  for (const at::Tensor& t : inputs) {
    TORCH_CHECK(t.dim() == inputs[0].dim(), "Tensors must have same number of dimensions");
    for (int64_t d = 0; d < t.dim(); ++d) {
      TORCH_CHECK(d == dim || t.size(d) == shape[d], "Sizes of tensors must match except in dimension ", dim);
    }
    total += t.size(dim);
  }
  shape[dim] = total;
  at::native::resize_output(out, shape);
  int64_t offset = 0;
  for (const at::Tensor& t : inputs) {
    if (t.numel() > 0) {
      out.narrow(dim, offset, t.size(dim)).copy_(t);
    }
    offset += t.size(dim);
  }
  return out;
}

// Indices 0..n-1 (int64, on the device of `like`).
at::Tensor iota(int64_t n, const at::Tensor& like) {
  return at::arange(n, like.options().dtype(at::kLong));
}

// out = self gathered with index tensors along `dims` (each 1-D, the size of
// that dim, viewed along it).
at::Tensor gather_dims(const at::Tensor& self, at::IntArrayRef out_shape, const std::vector<int64_t>& dims,
                       const std::vector<at::Tensor>& indices) {
  at::Tensor out = at::empty(out_shape, self.options());
  std::vector<int64_t> strides = self.strides().vec();
  std::vector<IndexTerm> terms;
  for (size_t k = 0; k < dims.size(); ++k) {
    strides[dims[k]] = 0;
    terms.push_back({index_along(indices[k], dims[k], self.dim()), self.size(dims[k]), self.stride(dims[k])});
  }
  at::Tensor src_view = self.as_strided(out_shape, strides, self.storage_offset());
  gather(out, src_view, index_offsets(out_shape, terms, self.options()));
  return out;
}

at::Tensor flip(const at::Tensor& self, at::IntArrayRef dims) {
  if (!movable(self)) {
    return on_cpu("aten::flip", "", self, dims)[0].toTensor();
  }
  std::vector<int64_t> flipped;
  std::vector<at::Tensor> indices;
  for (int64_t d : dims) {
    int64_t w = at::maybe_wrap_dim(d, self.dim());
    if (self.size(w) > 1) {
      flipped.push_back(w);
      indices.push_back(at::arange(self.size(w) - 1, -1, -1, self.options().dtype(at::kLong)));
    }
  }
  if (flipped.empty() || self.numel() == 0) {
    return self.clone(at::MemoryFormat::Contiguous);
  }
  return gather_dims(self, self.sizes(), flipped, indices);
}

at::Tensor roll_vk(const at::Tensor& self, at::IntArrayRef shifts, at::IntArrayRef dims) {
  if (!movable(self)) {
    return on_cpu("aten::roll", "", self, shifts, dims)[0].toTensor();
  }
  if (dims.empty()) {
    TORCH_CHECK(shifts.size() == 1, "`shifts` required");
    return roll_vk(self.reshape({-1}), shifts, {0}).view(self.sizes());
  }
  TORCH_CHECK(shifts.size() == dims.size(), "shifts and dimensions must align. shifts: ", shifts.size(),
              ", dims:", dims.size());
  if (self.numel() == 0) {
    return self.clone(at::MemoryFormat::Contiguous);
  }
  // Repeated dims compose: accumulate each dim's total shift.
  std::vector<int64_t> rolled, amount;
  for (size_t k = 0; k < dims.size(); ++k) {
    int64_t w = at::maybe_wrap_dim(dims[k], self.dim());
    auto it = std::find(rolled.begin(), rolled.end(), w);
    if (it == rolled.end()) {
      rolled.push_back(w);
      amount.push_back(shifts[k]);
    } else {
      amount[it - rolled.begin()] += shifts[k];
    }
  }
  std::vector<at::Tensor> indices;
  for (size_t k = 0; k < rolled.size(); ++k) {
    int64_t n = self.size(rolled[k]);
    int64_t s = ((amount[k] % n) + n) % n;
    // out[i] = self[(i - s) mod n] = self[(i + n - s) mod n]
    indices.push_back(at::remainder(at::arange(n - s, 2 * n - s, self.options().dtype(at::kLong)), n));
  }
  return gather_dims(self, self.sizes(), rolled, indices);
}

// Padding of the last pad.size() / 2 dims (pad pairs start at the last dim)
// by reflection or replication, as one gather.
at::Tensor& pad_out(const at::Tensor& self, at::IntArrayRef pad, bool reflect, const char* name, at::Tensor& out) {
  int64_t k = static_cast<int64_t>(pad.size()) / 2;
  TORCH_CHECK(self.dim() == k + 1 || self.dim() == k + 2, name, ": expected ", k + 1, "D or ", k + 2,
              "D input, got ", self.dim(), "D");
  std::vector<int64_t> shape = self.sizes().vec();
  std::vector<int64_t> dims;
  std::vector<at::Tensor> indices;
  for (int64_t j = 0; j < k; ++j) {
    int64_t d = self.dim() - 1 - j;
    int64_t left = pad[2 * j], right = pad[2 * j + 1], n = self.size(d);
    if (reflect) {
      TORCH_CHECK(left < n && right < n, name, ": padding size should be less than the corresponding input dimension");
    } else {
      TORCH_CHECK(n > 0, name, ": input dimension must be nonzero");
    }
    int64_t len = n + left + right;
    TORCH_CHECK(len > 0, name, ": output size is too small");
    at::Tensor i = at::arange(-left, n + right, self.options().dtype(at::kLong));
    if (reflect) {
      i = (n - 1) - at::abs((n - 1) - at::abs(i));  // reflect at both ends (pad < n)
    } else {
      i = at::clamp(i, 0, n - 1);
    }
    shape[d] = len;
    dims.push_back(d);
    indices.push_back(i);
  }
  at::Tensor result = gather_dims(self, shape, dims, indices);
  at::native::resize_output(out, shape);
  out.copy_(result);
  return out;
}

template <size_t K, bool Reflect>
at::Tensor& pad_kernel_out(const at::Tensor& self, at::IntArrayRef pad, at::Tensor& out) {
  static const std::string names[] = {"reflection_pad1d", "reflection_pad2d", "reflection_pad3d",
                                      "replication_pad1d", "replication_pad2d", "replication_pad3d"};
  const std::string& name = names[(Reflect ? 0 : 3) + K - 1];
  TORCH_CHECK(pad.size() == 2 * K, name, ": padding size is expected to be ", 2 * K);
  if (!movable(self) || !is_vk(out) || out.scalar_type() != self.scalar_type()) {
    on_cpu(("aten::" + name).c_str(), "out", self, pad, out);
    return out;
  }
  return pad_out(self, pad, Reflect, name.c_str(), out);
}

// reflection_pad2d is not structured, so its functional form needs a kernel too.
at::Tensor reflection_pad2d(const at::Tensor& self, at::IntArrayRef pad) {
  at::Tensor out = at::empty({0}, self.options());
  return pad_kernel_out<2, true>(self, pad, out);
}

at::Tensor channel_shuffle(const at::Tensor& self, int64_t groups) {
  TORCH_CHECK(self.dim() > 2 && groups > 0 && self.size(1) % groups == 0,
              "channel_shuffle expects input with > 2 dims and channels divisible by groups");
  std::vector<int64_t> shape = self.sizes().vec();
  std::vector<int64_t> split = {shape[0], groups, shape[1] / groups};
  split.insert(split.end(), shape.begin() + 2, shape.end());
  return self.reshape(split).transpose(1, 2).contiguous().reshape(shape);
}

// ---------------------------------------------------------------- arange, linspace, logspace, eye

// The dtype the iota kernel computes `dtype` in (small types go through a cast).
at::ScalarType iota_dtype(at::ScalarType dtype) {
  if (dtype == at::kHalf && has_feature(kFeatureFloat16)) {
    return at::kHalf;
  }
  if (dtype == at::kFloat || dtype == at::kDouble || dtype == at::kInt || dtype == at::kLong) {
    return dtype == at::kDouble && !has_feature(kFeatureFloat64) ? at::kFloat : dtype;
  }
  return at::isFloatingType(dtype) ? at::kFloat : at::kLong;
}

at::Tensor& arange_out(const at::Scalar& start, const at::Scalar& end, const at::Scalar& step, at::Tensor& out) {
  at::ScalarType dtype = out.scalar_type();
  bool floating = at::isFloatingType(dtype);
  if (!is_vk(out) || at::isComplexType(dtype) || dtype == at::kBool) {
    on_cpu("aten::arange", "start_out", start, end, step, out);
    return out;
  }
  double xstart = start.toDouble(), xend = end.toDouble(), xstep = step.toDouble();
  TORCH_CHECK(xstep != 0, "step must be nonzero");
  TORCH_CHECK(std::isfinite(xstart) && std::isfinite(xend), "unsupported range: ", xstart, " -> ", xend);
  TORCH_CHECK((xstep > 0 && xend >= xstart) || (xstep < 0 && xend <= xstart),
              "upper bound and lower bound inconsistent with step sign");
  double size_d = floating ? std::ceil((xend - xstart) / xstep)
                           : std::ceil(static_cast<double>(end.toLong() - start.toLong()) / step.toLong());
  TORCH_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
              "invalid size, possible overflow?");
  int64_t size = static_cast<int64_t>(size_d);
  if (out.numel() != size) {
    at::native::resize_output(out, {size});
  }
  at::ScalarType compute = iota_dtype(dtype);
  std::string kernel = std::string("iota_") + storage_label(compute);
  with_output(out, compute, [&](const at::Tensor& target) {
    launch_strided(kernel.c_str(), {op::IOTA_ARANGE}, target, nullptr, nullptr, {start, step});
  });
  return out;
}

at::Tensor& linspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps, at::Tensor& out) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");
  at::ScalarType dtype = out.scalar_type();
  if (!is_vk(out) || at::isComplexType(dtype) || dtype == at::kBool) {
    on_cpu("aten::linspace", "out", start, end, steps, out);
    return out;
  }
  if (out.numel() != steps) {
    at::native::resize_output(out, {steps});
  }
  if (steps == 0) {
    return out;
  }
  if (steps == 1) {
    out.fill_(start);
    return out;
  }
  // Floating types compute in their own precision, integers in double, as on the CPU.
  at::ScalarType compute = at::isFloatingType(dtype) ? iota_dtype(dtype) : iota_dtype(at::kDouble);
  double s = start.toDouble(), e = end.toDouble(), step;
  if (!at::isFloatingType(dtype)) {
    // Integer types convert the endpoints first (uint8: -3 -> 253), as on the CPU.
    s = at::scalar_tensor(start, at::TensorOptions().dtype(dtype)).item<double>();
    e = at::scalar_tensor(end, at::TensorOptions().dtype(dtype)).item<double>();
  }
  if (compute == at::kHalf) {
    // In c10::Half arithmetic, as PyTorch's CPU kernel.
    c10::Half hs(static_cast<float>(s)), he(static_cast<float>(e));
    c10::Half hstep = (he - hs) / c10::Half(static_cast<float>(steps - 1));
    s = static_cast<float>(hs);
    e = static_cast<float>(he);
    step = static_cast<float>(hstep);
  } else if (compute == at::kFloat) {
    float fs = static_cast<float>(s), fe = static_cast<float>(e);
    step = (fe - fs) / static_cast<float>(steps - 1);
  } else {
    step = (e - s) / static_cast<double>(steps - 1);
  }
  std::string kernel = std::string("iota_") + storage_label(compute);
  with_output(out, compute, [&](const at::Tensor& target) {
    launch_strided(kernel.c_str(), {op::IOTA_LINSPACE}, target, nullptr, nullptr, {s, e, step});
  });
  return out;
}

at::Tensor& logspace_out(const at::Scalar& start, const at::Scalar& end, int64_t steps, double base, at::Tensor& out) {
  at::ScalarType dtype = out.scalar_type();
  if (!is_vk(out) || at::isComplexType(dtype) || dtype == at::kBool) {
    on_cpu("aten::logspace", "out", start, end, steps, base, out);
    return out;
  }
  // Exponents in float32 for the 16-bit types (pow has no 16-bit kernel).
  at::ScalarType compute = at::isFloatingType(dtype) && at::elementSize(dtype) > 2 ? iota_dtype(dtype)
                           : at::isFloatingType(dtype)                         ? at::kFloat
                                                                               : iota_dtype(at::kDouble);
  at::Tensor exponents = at::empty({0}, out.options().dtype(compute));
  linspace_out(start, end, steps, exponents);
  at::native::resize_output(out, {steps});
  out.copy_(at::pow(base, exponents));
  return out;
}

at::Tensor& eye_m_out(int64_t n, int64_t m, at::Tensor& out) {
  TORCH_CHECK(n >= 0 && m >= 0, "n and m must be non-negative");
  if (!movable(out)) {
    on_cpu("aten::eye", "m_out", n, m, out);
    return out;
  }
  at::native::resize_output(out, {n, m});
  out.zero_();
  if (std::min(n, m) > 0) {
    out.diagonal().fill_(1);
  }
  return out;
}

at::Tensor& eye_out(int64_t n, at::Tensor& out) {
  return eye_m_out(n, n, out);
}

// ---------------------------------------------------------------- equal, isin

bool equal(const at::Tensor& self, const at::Tensor& other) {
  if (!is_vk(self) || !is_vk(other) || dtype_label(self.scalar_type()) == nullptr ||
      dtype_label(other.scalar_type()) == nullptr) {
    return on_cpu("aten::equal", "", self, other)[0].toBool();
  }
  if (self.sizes() != other.sizes()) {
    return false;
  }
  return self.numel() == 0 || at::eq(self, other).all().item<bool>();
}

at::Tensor& isin_out(const at::Tensor& elements, const at::Tensor& test, bool assume_unique, bool invert,
                     at::Tensor& out) {
  // Broadcast comparison: fine for moderate sizes (a sort-based method comes with M10).
  bool small = elements.numel() * std::max<int64_t>(test.numel(), 1) <= (int64_t(1) << 24);
  if (!is_vk(elements) || !is_vk(test) || !is_vk(out) || !small || dtype_label(elements.scalar_type()) == nullptr ||
      dtype_label(test.scalar_type()) == nullptr) {
    on_cpu("aten::isin", "Tensor_Tensor_out", elements, test, assume_unique, invert, out);
    return out;
  }
  at::native::resize_output(out, elements.sizes());
  at::Tensor hit = at::eq(elements.unsqueeze(-1), test.reshape({-1})).any(-1);
  out.copy_(invert ? at::logical_not(hit) : hit);
  return out;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("index_select", &vk::index_select);
  m.impl("index_select.out", &index_select_out);
  m.impl("gather.out", &gather_out);
  m.impl("take", &vk::take);
  m.impl("take.out", &take_out);
  m.impl("index.Tensor", &vk::index);
  m.impl("index.Tensor_out", &index_out);
  m.impl("_index_put_impl_", &index_put_impl);
  m.impl("scatter.src_out", &scatter_src_out);
  m.impl("scatter.value_out", &scatter_value_out);
  m.impl("index_copy.out", &index_copy_out);
  m.impl("index_fill_.int_Scalar", &index_fill_scalar);
  m.impl("index_fill_.int_Tensor", &index_fill_tensor);
  m.impl("put_", &vk::put_);
  m.impl("scatter_add.out", &scatter_add_out);
  m.impl("scatter_reduce.two_out", &scatter_reduce_two_out);
  m.impl("scatter.reduce_out", &scatter_reduce_out);
  m.impl("scatter.value_reduce_out", &scatter_value_reduce_out);
  m.impl("index_add.out", &index_add_out);
  m.impl("index_reduce.out", &index_reduce_out);
  m.impl("bincount", &vk::bincount);
  m.impl("histc", &vk::histc);
  m.impl("histc.out", &histc_out);
  m.impl("nonzero", &vk::nonzero_vk);
  m.impl("nonzero.out", &nonzero_out);
  m.impl("nonzero_static", &vk::nonzero_static);
  m.impl("count_nonzero.dim_IntList", &vk::count_nonzero);
  m.impl("masked_select", &vk::masked_select);
  m.impl("masked_select.out", &masked_select_out);
  m.impl("masked_scatter_", &vk::masked_scatter_);
  m.impl("cat.out", &cat_out);
  m.impl("flip", &vk::flip);
  m.impl("roll", &vk::roll_vk);
  m.impl("reflection_pad1d.out", &pad_kernel_out<1, true>);
  m.impl("reflection_pad2d.out", &pad_kernel_out<2, true>);
  m.impl("reflection_pad2d", &vk::reflection_pad2d);
  m.impl("reflection_pad3d.out", &pad_kernel_out<3, true>);
  m.impl("replication_pad1d.out", &pad_kernel_out<1, false>);
  m.impl("replication_pad2d.out", &pad_kernel_out<2, false>);
  m.impl("replication_pad3d.out", &pad_kernel_out<3, false>);
  m.impl("channel_shuffle", &vk::channel_shuffle);
  m.impl("native_channel_shuffle", &vk::channel_shuffle);
  m.impl("arange.start_out", &arange_out);
  m.impl("linspace.out", &linspace_out);
  m.impl("logspace.out", &logspace_out);
  m.impl("eye.m_out", &eye_m_out);
  m.impl("eye.out", &eye_out);
  m.impl("equal", &vk::equal);
  m.impl("isin.Tensor_Tensor_out", &isin_out);
}
