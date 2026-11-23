// Strided-iteration planning shared by every element-wise launcher and the
// reductions: broadcasting, merging dims, and splitting work into dispatches.
#pragma once

#include <ATen/ATen.h>

#include <array>
#include <cstddef>
#include <cstdlib>
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
// kernel, the rest are looped over here. Kernels count elements and offset
// operands in 32 bits (docs/log/M56: below 2^31, so that a grid-stride
// loop's counter cannot wrap either); where the kernel's dims pass that,
// the outermost of them is cut into pieces (fewer dims go to the kernel
// where even one element of it does not fit). For each dispatch calls
// emit(count, offsets, inner, last): offsets[k] is operand k's element
// offset, `inner` the dims the kernel takes and `last` the length of the
// outermost of them (index inner - 1) in that dispatch.
template <size_t K, typename F>
void for_each_dispatch(const StridedDims<K>& d, size_t max_inner, F&& emit) {
  constexpr int64_t kLimit = int64_t(1) << 31;
  size_t inner = std::min(d.ndim(), max_inner);
  // (the dims below the outermost inner one: their count and spans)
  int64_t below = 1;
  std::array<int64_t, K> span{};
  int64_t piece = 0;  // elements of the outermost inner dim per dispatch
  while (true) {
    below = 1;
    span = {};
    for (size_t i = 0; i + 1 < inner; ++i) {
      below *= d.size[i];
      for (size_t k = 0; k < K; ++k) span[k] += (d.size[i] - 1) * std::abs(d.stride[i][k]);
    }
    if (inner == 0) {
      piece = 1;
      break;
    }
    // the longest piece of dim inner - 1 that keeps count and spans below the limit
    piece = (kLimit - 1) / below;
    bool negative = false;
    for (size_t k = 0; k < K; ++k) {
      int64_t st = std::abs(d.stride[inner - 1][k]);
      negative = negative || d.stride[inner - 1][k] < 0;
      if (span[k] >= kLimit) piece = 0;
      else if (st > 0) piece = std::min(piece, (kLimit - 1 - span[k]) / st + 1);
    }
    piece = std::min(piece, d.size[inner - 1]);
    // (a reversed dim only whole: copy_reversed's offsets assume it)
    if (negative && piece < d.size[inner - 1]) piece = 0;
    if (piece >= 1 || inner == 1) break;
    --inner;  // (not even one element of it fits: it goes to the host's loop)
  }
  TORCH_CHECK(piece >= 1, "torch_vulkan: element offsets beyond 2^31 in one dim");
  std::vector<int64_t> index(d.ndim() - inner, 0);
  int64_t last = inner > 0 ? d.size[inner - 1] : 1;
  while (true) {
    std::array<int64_t, K> offsets{};
    for (size_t j = 0; j < index.size(); ++j) {
      for (size_t k = 0; k < K; ++k) offsets[k] += index[j] * d.stride[inner + j][k];
    }
    for (int64_t start = 0; start < last; start += piece) {
      int64_t len = std::min(piece, last - start);
      std::array<int64_t, K> at = offsets;
      if (inner > 0) {
        for (size_t k = 0; k < K; ++k) at[k] += start * d.stride[inner - 1][k];
      }
      emit(below * len, at, inner, len);
    }
    size_t j = 0;
    for (; j < index.size(); ++j) {
      if (++index[j] < d.size[inner + j]) break;
      index[j] = 0;
    }
    if (j == index.size()) break;
  }
}

}  // namespace vk
