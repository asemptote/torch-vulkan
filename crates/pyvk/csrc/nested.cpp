// Nested (jagged) tensors for device "vk": the conversions between a jagged
// tensor's packed values and a padded dense tensor that torch.nested's
// jagged layout uses for most ops (_jagged_to_padded_dense_forward,
// _padded_dense_to_jagged_forward), as one gather or scatter of rows with
// each row's batch found by a sorted search in the offsets, where the CPU
// loops over the batch.

#include "vk.h"

#include <torch/library.h>

#include <cmath>
#include <limits>
#include <vector>

namespace vk {
namespace {

// The CPU's _get_padding_value: an infinite padding for an integer dtype
// is the dtype's largest or lowest value.
at::Scalar padding_scalar(double padding_value, at::ScalarType t) {
  if (!std::isinf(padding_value) || at::isFloatingType(t)) return padding_value;
  at::Scalar s;
  AT_DISPATCH_ALL_TYPES_AND3(at::kBool, at::kHalf, at::kBFloat16, t, "padding_value", [&] {
    s = padding_value > 0 ? std::numeric_limits<scalar_t>::max() : std::numeric_limits<scalar_t>::lowest();
  });
  return s;
}

// Each packed row's batch (offsets[b] <= row < offsets[b + 1]; -1 or
// batches outside them) and position in it.
std::pair<at::Tensor, at::Tensor> rows_in_batches(const at::Tensor& offsets, int64_t rows) {
  at::Tensor o = offsets.to(at::kLong);
  at::Tensor r = at::arange(rows, o.options());
  at::Tensor b = at::searchsorted(o, r, /*out_int32=*/false, /*right=*/true) - 1;
  int64_t batches = o.size(0) - 1;
  at::Tensor pos = r - o.index_select(0, at::clamp(b, 0, std::max<int64_t>(batches, 0)));
  return {b, pos};
}

at::Tensor jagged_to_padded_dense(const at::Tensor& values, at::TensorList offsets_list, at::IntArrayRef max_lengths,
                                  double padding_value) {
  bool ok = offsets_list.size() == 1 && max_lengths.size() == 1 && offsets_list[0].dim() == 1 &&
            offsets_list[0].size(0) >= 1 && values.dim() >= 1 && is_vk(values) && is_vk(offsets_list[0]) &&
            !at::isComplexType(values.scalar_type());
  if (!ok) {
    return on_cpu("aten::_jagged_to_padded_dense_forward", "", values, offsets_list, max_lengths, padding_value)[0]
        .toTensor();
  }
  const at::Tensor& offsets = offsets_list[0];
  int64_t batches = offsets.size(0) - 1, length = max_lengths[0], total = values.size(0);
  std::vector<int64_t> shape = {batches, length};
  shape.insert(shape.end(), values.sizes().begin() + 1, values.sizes().end());
  int64_t inner = 1;
  for (size_t d = 2; d < shape.size(); ++d) inner *= shape[d];
  at::Scalar fill = padding_scalar(padding_value, values.scalar_type());
  if (total == 0 || batches == 0 || length == 0 || inner == 0) return at::full(shape, fill, values.options());
  // Rows past their batch's max length (or outside every batch) go to a
  // spare row, dropped after.
  auto [b, pos] = rows_in_batches(offsets, total);
  at::Tensor keep = at::logical_and(at::logical_and(at::ge(b, 0), at::lt(b, batches)), at::lt(pos, length));
  at::Tensor dest = at::where(keep, b * length + pos, batches * length);
  at::Tensor out = at::full({batches * length + 1, inner}, fill, values.options());
  out.index_copy_(0, dest, values.reshape({total, inner}));
  return out.narrow(0, 0, batches * length).view(shape);
}

at::Tensor padded_dense_to_jagged(const at::Tensor& padded, at::TensorList offsets_list, std::optional<int64_t> total_L) {
  bool ok = offsets_list.size() == 1 && offsets_list[0].dim() == 1 && offsets_list[0].size(0) >= 1 &&
            padded.dim() >= 2 && is_vk(padded) && is_vk(offsets_list[0]) &&
            offsets_list[0].size(0) - 1 <= padded.size(0);
  const at::Tensor& offsets = offsets_list[0];
  int64_t batches = ok ? offsets.size(0) - 1 : 0, width = ok ? padded.size(1) : 0;
  int64_t final_offset = 0;
  if (ok) {
    // (the CPU's checks, from one transfer: the final offset and the longest batch)
    at::Tensor o = offsets.to(at::kLong);
    at::Tensor stats = batches > 0 ? at::stack({o[-1], (o.narrow(0, 1, batches) - o.narrow(0, 0, batches)).max()})
                                   : o.narrow(0, 0, 1);
    at::Tensor host = stats.cpu();
    final_offset = host[0].item<int64_t>();
    int64_t longest = batches > 0 ? host[1].item<int64_t>() : 0;
    ok = longest <= width && (!total_L.has_value() || *total_L == final_offset) && final_offset >= 0;
  }
  if (!ok) {
    return on_cpu("aten::_padded_dense_to_jagged_forward", "", padded, offsets_list, total_L)[0].toTensor();
  }
  int64_t total = total_L.value_or(final_offset);
  std::vector<int64_t> shape = {total};
  shape.insert(shape.end(), padded.sizes().begin() + 2, padded.sizes().end());
  int64_t inner = 1;
  for (size_t d = 1; d < shape.size(); ++d) inner *= shape[d];
  if (total == 0 || inner == 0 || batches == 0 || width == 0) return at::empty(shape, padded.options());
  auto [b, pos] = rows_in_batches(offsets, total);
  at::Tensor src = at::clamp(b, 0, batches - 1) * width + at::clamp(pos, 0, width - 1);
  return padded.narrow(0, 0, batches).reshape({batches * width, inner}).index_select(0, src).view(shape);
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_jagged_to_padded_dense_forward", TORCH_FN(vk::jagged_to_padded_dense));
  m.impl("_padded_dense_to_jagged_forward", TORCH_FN(vk::padded_dense_to_jagged));
}
