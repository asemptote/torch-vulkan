// Strided-iteration planning shared by every element-wise launcher and the
// reductions: broadcasting, merging dims, and splitting work into dispatches.
#pragma once

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <vector>

namespace vk {

// Element strides of `t` broadcast to `shape` (0 where it broadcasts).
inline std::vector<int64_t> broadcast_strides(const at::TensorBase& t, at::IntArrayRef shape) {
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

// K operands over one index space, innermost dim first: per dim its size
// and each operand's element stride. Size-1 dims are dropped and dims that
// are contiguous in every operand merged, so e.g. any contiguous layout
// becomes a single dim.
template <size_t K>
struct StridedDims {
  std::vector<int64_t> size;
  std::vector<std::array<int64_t, K>> stride;

  size_t ndim() const { return size.size(); }
};

// `shape` and `strides[k]` are outermost-first, as in PyTorch.
template <size_t K>
StridedDims<K> merge_dims(at::IntArrayRef shape, const std::array<std::vector<int64_t>, K>& strides) {
  StridedDims<K> d;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] == 1) {
      continue;
    }
    std::array<int64_t, K> st;
    for (size_t k = 0; k < K; ++k) st[k] = strides[k][i];
    if (!d.size.empty()) {
      bool mergeable = true;
      for (size_t k = 0; k < K; ++k) mergeable = mergeable && st[k] == d.stride.back()[k] * d.size.back();
      if (mergeable) {
        d.size.back() *= shape[i];
        continue;
      }
    }
    d.size.push_back(shape[i]);
    d.stride.push_back(st);
  }
  return d;
}

// Plan for `out` (operand 0) and inputs broadcast to its shape; a null
// input contributes zero strides.
template <size_t K>
StridedDims<K> plan_operands(const at::Tensor& out, const std::array<const at::Tensor*, K>& operands) {
  at::IntArrayRef shape = out.sizes();
  std::array<std::vector<int64_t>, K> strides;
  for (size_t k = 0; k < K; ++k) {
    strides[k] = k == 0 ? out.strides().vec()
                        : (operands[k] ? broadcast_strides(*operands[k], shape) : std::vector<int64_t>(shape.size(), 0));
  }
  return merge_dims<K>(shape, strides);
}

// Splits the work into dispatches: the innermost `max_inner` dims go to the
// kernel, the rest are looped over here. For each dispatch calls
// emit(inner_count, offsets) where offsets[k] is operand k's element offset.
// Checks the kernel-side limits (32-bit element offsets and counts).
template <size_t K, typename F>
void for_each_dispatch(const StridedDims<K>& d, size_t max_inner, F&& emit) {
  size_t inner = std::min(d.ndim(), max_inner);
  int64_t inner_count = 1;
  std::array<int64_t, K> span{};
  for (size_t i = 0; i < inner; ++i) {
    inner_count *= d.size[i];
    for (size_t k = 0; k < K; ++k) span[k] += (d.size[i] - 1) * d.stride[i][k];
  }
  TORCH_CHECK(inner_count < (int64_t(1) << 32), "torch_vulkan: more than 2^32 elements per dispatch");
  for (size_t k = 0; k < K; ++k) {
    TORCH_CHECK(span[k] < (int64_t(1) << 31), "torch_vulkan: element offsets beyond 2^31");
  }
  std::vector<int64_t> index(d.ndim() - inner, 0);
  while (true) {
    std::array<int64_t, K> offsets{};
    for (size_t j = 0; j < index.size(); ++j) {
      for (size_t k = 0; k < K; ++k) offsets[k] += index[j] * d.stride[inner + j][k];
    }
    emit(inner_count, offsets);
    size_t j = 0;
    for (; j < index.size(); ++j) {
      if (++index[j] < d.size[inner + j]) break;
      index[j] = 0;
    }
    if (j == index.size()) break;
  }
}

}  // namespace vk
