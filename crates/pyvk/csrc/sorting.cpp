// Sorting, searching and order statistics for device "vk" (M10): sort,
// topk, kthvalue, median, nanmedian, mode, unique, unique_consecutive,
// searchsorted, bucketize, cummax/cummin, logcumsumexp, and small reductions
// built from existing kernels (aminmax, var_mean, std_mean, nansum, trace).
//
// The primitive is sort_rows (vk.h): a bitonic sort of (key, index) pairs
// (shaders/sort_keys.slang, shaders/sort.slang). Keys are the values mapped
// to unsigned integers in PyTorch's order (NaN largest, -0 equal to +0), and
// ties are broken by index, so the result equals a stable sort; topk,
// kthvalue, median and mode then pick the same tie indices as the CPU.

#include "opcodes.h"
#include "vk.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <limits>

namespace vk {

namespace {

struct SortKeysParams {
  uint64_t x, keys, idx;
  uint32_t n, P, total, rows, stride, pad;
};
static_assert(sizeof(SortKeysParams) == 48, "must match shaders/sort_keys.slang");

struct SortParams {
  uint64_t keys, idx;
  uint32_t total, P, j, k, stride, pad;
};
static_assert(sizeof(SortParams) == 40, "must match shaders/sort.slang");

constexpr uint32_t kSortLocalFull = 0, kSortGlobal = 1, kSortLocalMerge = 2;
constexpr int64_t kSortBlock = 1024;

bool sortable(const at::Tensor& t) {
  return is_vk(t) && !at::isComplexType(t.scalar_type()) && storage_label(t.scalar_type()) != nullptr;
}

}  // namespace

at::Tensor sort_rows(const at::Tensor& x, bool descending) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "torch_vulkan: sort_rows expects a contiguous 2-D tensor");
  int64_t rows = x.size(0), n = x.size(1);
  at::TensorOptions opts = x.options();
  if (rows == 0 || n == 0) {
    return at::empty({rows, n}, opts.dtype(at::kInt));
  }
  int64_t P = 1;
  while (P < n) P <<= 1;
  int64_t total = (rows * P + kSortBlock - 1) / kSortBlock * kSortBlock;
  TORCH_CHECK(total < (int64_t(1) << 31), "torch_vulkan: sort of more than 2^31 (padded) elements");
  bool wide = x.itemsize() == 8;
  at::Tensor keys = at::empty({total}, opts.dtype(wide ? at::kLong : at::kInt));
  at::Tensor idx = at::empty({total}, opts.dtype(at::kInt));

  SortKeysParams kp{};
  kp.x = gpu_address(x);
  kp.keys = gpu_address(keys);
  kp.idx = gpu_address(idx);
  kp.n = static_cast<uint32_t>(n);
  kp.P = static_cast<uint32_t>(P);
  kp.total = static_cast<uint32_t>(total);
  kp.rows = static_cast<uint32_t>(rows);
  uint32_t groups = 0;
  check(vkt_grid_1d(kp.total, &groups, &kp.stride));
  dispatch(std::string("sort_keys_") + storage_label(x.scalar_type()), {descending ? 1u : 0u}, kp, groups);

  std::string kernel = wide ? "sort_k64" : "sort_k32";
  SortParams p{};
  p.keys = kp.keys;
  p.idx = kp.idx;
  p.total = kp.total;
  p.P = kp.P;
  uint32_t blocks = static_cast<uint32_t>(total / kSortBlock);
  uint32_t global_groups = std::min<uint32_t>(blocks, 65535);
  p.stride = global_groups * 512;
  dispatch(kernel, {kSortLocalFull}, p, blocks);
  for (int64_t k = 2 * kSortBlock; k <= P; k <<= 1) {
    p.k = static_cast<uint32_t>(k);
    for (int64_t j = k / 2; j >= kSortBlock; j >>= 1) {
      p.j = static_cast<uint32_t>(j);
      dispatch(kernel, {kSortGlobal}, p, global_groups);
    }
    dispatch(kernel, {kSortLocalMerge}, p, blocks);
  }
  return idx.narrow(0, 0, rows * P).view({rows, P}).narrow(1, 0, n).contiguous();
}

namespace {

// `self` with `dim` moved last, as a contiguous [rows, n] tensor.
at::Tensor as_rows(const at::Tensor& self, int64_t dim) {
  int64_t n = self.dim() == 0 ? 1 : self.size(dim);
  at::Tensor moved = self.dim() == 0 ? self.reshape({1}) : self.movedim(dim, -1);
  int64_t rows = 1;
  for (int64_t d = 0; d + 1 < moved.dim(); ++d) {
    rows *= moved.size(d);
  }
  return moved.contiguous().reshape({rows, n});
}

// [rows, n] results back to self's layout with `dim` restored.
at::Tensor from_rows(const at::Tensor& r, const at::Tensor& self, int64_t dim) {
  if (self.dim() == 0) {
    return r.reshape({});
  }
  std::vector<int64_t> moved = self.movedim(dim, -1).sizes().vec();
  moved.back() = r.size(1);
  return r.reshape(moved).movedim(-1, dim);
}

// Stable sort along `dim`: (values, int64 indices) in self's layout.
std::tuple<at::Tensor, at::Tensor> sort_along(const at::Tensor& self, int64_t dim, bool descending) {
  at::Tensor x = as_rows(self, dim);
  at::Tensor idx = sort_rows(x, descending).to(at::kLong);
  at::Tensor values = at::gather(x, 1, idx);
  return {from_rows(values, self, dim), from_rows(idx, self, dim)};
}

void assign_out(at::Tensor& out, const at::Tensor& result) {
  check_out_device(result, out);
  if (out.scalar_type() != at::kBool) {  // logical results (any, all, ...) accumulate in other dtypes
    check_out_cast(result.scalar_type(), out);
  }
  at::native::resize_output(out, result.sizes());
  out.copy_(result);
}

// ---------------------------------------------------------------- sort, topk, kthvalue

std::tuple<at::Tensor&, at::Tensor&> sort_out(const at::Tensor& self, std::optional<bool> stable, int64_t dim,
                                              bool descending, at::Tensor& values, at::Tensor& indices) {
  if (!sortable(self) || !is_vk(values) || !is_vk(indices)) {
    on_cpu("aten::sort", "values_stable", self, stable, dim, descending, values, indices);
    return {values, indices};
  }
  dim = at::maybe_wrap_dim(dim, self.dim());
  auto [v, i] = sort_along(self, dim, descending);
  assign_out(values, v);
  assign_out(indices, i);
  return {values, indices};
}

std::tuple<at::Tensor&, at::Tensor&> sort_values(const at::Tensor& self, int64_t dim, bool descending,
                                                 at::Tensor& values, at::Tensor& indices) {
  return sort_out(self, std::nullopt, dim, descending, values, indices);
}

std::tuple<at::Tensor, at::Tensor> sort_stable(const at::Tensor& self, std::optional<bool> stable, int64_t dim,
                                               bool descending) {
  // Outputs laid out like self (as the CPU's and PyTorch's meta kernel).
  at::Tensor values = at::empty_like(self), indices = at::empty_like(self, self.options().dtype(at::kLong));
  sort_out(self, stable, dim, descending, values, indices);
  return {values, indices};
}

std::tuple<at::Tensor&, at::Tensor&> topk_out(const at::Tensor& self, int64_t k, int64_t dim, bool largest,
                                              bool sorted, at::Tensor& values, at::Tensor& indices) {
  if (!sortable(self) || !is_vk(values) || !is_vk(indices)) {
    on_cpu("aten::topk", "values", self, k, dim, largest, sorted, values, indices);
    return {values, indices};
  }
  dim = at::maybe_wrap_dim(dim, self.dim());
  int64_t n = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK(k >= 0 && k <= n, "selected index k out of range");
  auto [v, i] = sort_along(self, dim, largest);
  int64_t d = self.dim() == 0 ? 0 : dim;
  assign_out(values, self.dim() == 0 ? v : v.narrow(d, 0, k));
  assign_out(indices, self.dim() == 0 ? i : i.narrow(d, 0, k));
  return {values, indices};
}

// Picks position pos[row] (int64, [rows]) of each sorted row; keepdim as reductions.
std::tuple<at::Tensor, at::Tensor> pick(const at::Tensor& self, int64_t dim, bool keepdim, const at::Tensor& sorted_v,
                                        const at::Tensor& sorted_i, const at::Tensor& pos) {
  at::Tensor p = pos.reshape({-1, 1});
  at::Tensor v = at::gather(sorted_v, 1, p), i = at::gather(sorted_i, 1, p);
  std::vector<int64_t> shape = self.sizes().vec();
  if (self.dim() > 0) {
    shape[dim] = 1;
  }
  v = v.reshape(shape);
  i = i.reshape(shape);
  if (!keepdim && self.dim() > 0) {
    v = v.squeeze(dim);
    i = i.squeeze(dim);
  }
  return {v, i};
}

// Rows of self sorted ascending: (values, indices) as [rows, n].
std::tuple<at::Tensor, at::Tensor> sorted_rows(const at::Tensor& self, int64_t dim) {
  at::Tensor x = as_rows(self, dim);
  at::Tensor idx = sort_rows(x, false).to(at::kLong);
  return {at::gather(x, 1, idx), idx};
}

std::tuple<at::Tensor&, at::Tensor&> kthvalue_out(const at::Tensor& self, int64_t k, int64_t dim, bool keepdim,
                                                  at::Tensor& values, at::Tensor& indices) {
  reject_dtype("kthvalue", self.scalar_type(), self.scalar_type() == at::kBool);
  at::assert_no_overlap(self, values);  // (as the CPU)
  at::assert_no_overlap(self, indices);
  if (!sortable(self) || !is_vk(values) || !is_vk(indices)) {
    on_cpu("aten::kthvalue", "values", self, k, dim, keepdim, values, indices);
    return {values, indices};
  }
  dim = at::maybe_wrap_dim(dim, self.dim());
  int64_t n = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK(k >= 1 && k <= n, "kthvalue(): selected number k out of range for dimension ", dim);
  auto [sv, si] = sorted_rows(self, dim);
  at::Tensor pos = at::full({sv.size(0)}, k - 1, sv.options().dtype(at::kLong));
  auto [v, i] = pick(self, dim, keepdim, sv, si, pos);
  assign_out(values, v);
  assign_out(indices, i);
  return {values, indices};
}

// ---------------------------------------------------------------- median, nanmedian, mode

// Position of the (nan)median in each sorted row: the lower median; for
// median, the first NaN if the row has one (NaNs sort last).
at::Tensor median_positions(const at::Tensor& sorted_v, bool ignore_nan) {
  int64_t n = sorted_v.size(1);
  at::Tensor nans = at::isnan(sorted_v).sum(1);  // int64
  if (ignore_nan) {
    at::Tensor valid = n - nans;
    return at::clamp_min(at::div(valid - 1, 2, "floor"), 0);
  }
  at::Tensor lower = at::full_like(nans, (n - 1) / 2);
  return at::where(nans > 0, n - nans, lower);
}

std::tuple<at::Tensor, at::Tensor> median_dim(const at::Tensor& self, int64_t dim, bool keepdim, bool ignore_nan) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  TORCH_CHECK(self.dim() == 0 || self.size(dim) > 0, "median() cannot compute median for a dimension of size 0");
  auto [sv, si] = sorted_rows(self, dim);
  return pick(self, dim, keepdim, sv, si, median_positions(sv, ignore_nan));
}

template <bool IgnoreNan>
std::tuple<at::Tensor&, at::Tensor&> median_dim_out(const at::Tensor& self, int64_t dim, bool keepdim,
                                                    at::Tensor& values, at::Tensor& indices) {
  reject_dtype(IgnoreNan ? "nanmedian" : "median", self.scalar_type(), self.scalar_type() == at::kBool);
  if (!sortable(self) || !is_vk(values) || !is_vk(indices)) {
    on_cpu(IgnoreNan ? "aten::nanmedian" : "aten::median", "dim_values", self, dim, keepdim, values, indices);
    return {values, indices};
  }
  auto [v, i] = median_dim(self, dim, keepdim, IgnoreNan);
  assign_out(values, v);
  assign_out(indices, i);
  return {values, indices};
}

template <bool IgnoreNan>
at::Tensor median_all(const at::Tensor& self) {
  reject_dtype(IgnoreNan ? "nanmedian" : "median", self.scalar_type(), self.scalar_type() == at::kBool);
  if (!sortable(self)) {
    return on_cpu(IgnoreNan ? "aten::nanmedian" : "aten::median", "", self)[0].toTensor();
  }
  if (self.numel() == 0) {
    return at::full({}, std::numeric_limits<double>::quiet_NaN(), self.options());
  }
  return std::get<0>(median_dim(self.reshape({-1}), 0, false, IgnoreNan));
}

struct RowScanParams {
  uint64_t x, out, idx, out_idx;
  uint32_t rows, n, stride, pad;
};
static_assert(sizeof(RowScanParams) == 48, "must match shaders/rowscan.slang");

const char* rowscan_label(at::ScalarType t) {
  switch (t) {
    case at::kFloat:
    case at::kInt:
    case at::kLong:
    case at::kBool: return dtype_label(t);
    case at::kDouble: return has_feature(kFeatureFloat64) ? "f64" : nullptr;
    default: return nullptr;
  }
}

void rowscan(uint32_t op, const at::Tensor& x, const at::Tensor& out, const at::Tensor* idx,
             const at::Tensor* out_idx = nullptr) {
  RowScanParams p{};
  p.x = gpu_address(x);
  p.out = gpu_address(out);
  p.idx = idx ? gpu_address(*idx) : 0;
  p.out_idx = out_idx ? gpu_address(*out_idx) : 0;
  p.rows = static_cast<uint32_t>(x.size(0));
  p.n = static_cast<uint32_t>(x.size(1));
  uint32_t groups = 0;
  check(vkt_grid_1d(p.rows, &groups, &p.stride));
  dispatch(std::string("rowscan_") + rowscan_label(x.scalar_type()), {op}, p, groups);
}

std::tuple<at::Tensor, at::Tensor> mode(const at::Tensor& self, int64_t dim, bool keepdim) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  if (!sortable(self) || rowscan_label(self.scalar_type()) == nullptr || self.numel() == 0) {
    auto stack = on_cpu("aten::mode", "", self, dim, keepdim);
    return {stack[0].toTensor(), stack[1].toTensor()};
  }
  auto [sv, si] = sorted_rows(self, dim);
  sv = sv.contiguous();
  si = si.contiguous();
  int64_t rows = sv.size(0);
  at::Tensor v = at::empty({rows}, sv.options()), i = at::empty({rows}, si.options());
  rowscan(op::ROWSCAN_MODE, sv, v, &si, &i);
  std::vector<int64_t> shape = self.sizes().vec();
  if (self.dim() > 0) {
    shape[dim] = 1;
  }
  v = v.reshape(shape);
  i = i.reshape(shape);
  if (!keepdim && self.dim() > 0) {
    v = v.squeeze(dim);
    i = i.squeeze(dim);
  }
  return {v, i};
}

// (PyTorch's composite mode.values allows only CPU, CUDA and XPU)
std::tuple<at::Tensor&, at::Tensor&> mode_out(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& values,
                                              at::Tensor& indices) {
  auto [v, i] = at::mode(self, dim, keepdim);  // (our functional kernel, via the dispatcher)
  assign_out(values, v);
  assign_out(indices, i);
  return {values, indices};
}

// ---------------------------------------------------------------- cummax, cummin, logcumsumexp

template <uint32_t Op>
void cum_helper(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim) {
  if (!is_vk(self) || rowscan_label(self.scalar_type()) == nullptr || !is_vk(values) || !is_vk(indices)) {
    on_cpu(Op == op::ROWSCAN_CUMMAX ? "aten::_cummax_helper" : "aten::_cummin_helper", "", self, values, indices,
           dim);
    return;
  }
  dim = at::maybe_wrap_dim(dim, self.dim());
  if (self.numel() == 0) {
    return;
  }
  at::Tensor x = as_rows(self, dim);
  at::Tensor v = at::empty_like(x), i = at::empty(x.sizes(), x.options().dtype(at::kLong));
  rowscan(Op, x, v, &i);
  values.copy_(from_rows(v, self, dim));
  indices.copy_(from_rows(i, self, dim));
}

at::Tensor logcumsumexp_vk(const at::Tensor& self, int64_t dim) {
  at::ScalarType t = self.scalar_type();
  if (is_vk(self) && at::isComplexType(t) && dtype_label(t) != nullptr && self.dim() == 0) {
    at::maybe_wrap_dim(dim, 1);
    return self.clone();  // of one element: the element
  }
  if (is_vk(self) && at::isComplexType(t) && dtype_label(t) != nullptr && self.dim() > 0) {
    // Complex: a Hillis-Steele scan with logaddexp (complex logaddexp runs on the GPU).
    dim = at::maybe_wrap_dim(dim, self.dim());
    at::Tensor x = self.movedim(dim, -1).resolve_conj().contiguous();
    int64_t n = x.size(-1);
    for (int64_t offset = 1; offset < n; offset *= 2) {
      at::Tensor next = x.clone();
      next.narrow(-1, offset, n - offset).copy_(at::logaddexp(x.narrow(-1, offset, n - offset),
                                                              x.narrow(-1, 0, n - offset)));
      x = next;
    }
    return x.movedim(-1, dim).contiguous();
  }
  if (!is_vk(self) || !(t == at::kFloat || (t == at::kDouble && has_feature(kFeatureFloat64)))) {
    return on_cpu("aten::_logcumsumexp", "", self, dim)[0].toTensor();
  }
  dim = at::maybe_wrap_dim(dim, self.dim());
  if (self.numel() == 0) {
    return at::empty_like(self);
  }
  at::Tensor x = as_rows(self, dim);
  at::Tensor v = at::empty_like(x);
  rowscan(op::ROWSCAN_LOGCUMSUMEXP, x, v, nullptr);
  return from_rows(v, self, dim).contiguous();
}

at::Tensor& logcumsumexp_out(const at::Tensor& self, int64_t dim, at::Tensor& out) {
  assign_out(out, logcumsumexp_vk(self, dim));
  return out;
}

// ---------------------------------------------------------------- searchsorted, bucketize

struct SearchParams {
  uint64_t seq, vals, out;
  uint32_t m, q, seq_row, total, stride, pad;
};
static_assert(sizeof(SearchParams) == 48, "must match shaders/searchsorted.slang");

at::Tensor& searchsorted_impl(const at::Tensor& sorted, const at::Tensor& self, bool out_int32, bool right,
                              const std::optional<at::Tensor>& sorter, at::Tensor& out) {
  at::ScalarType dtype = at::result_type(sorted, self);
  const char* label = rowscan_label(dtype);
  bool ok = is_vk(sorted) && is_vk(self) && is_vk(out) && label != nullptr && dtype != at::kBool &&
            (!sorter || is_vk(*sorter)) && (sorted.dim() == 1 || sorted.dim() == self.dim());
  if (!ok) {
    on_cpu("aten::searchsorted", "Tensor_out", sorted, self, out_int32, right, std::nullopt, sorter, out);
    return out;
  }
  at::native::resize_output(out, self.sizes());
  if (self.numel() == 0) {
    return out;
  }
  at::Tensor seq = sorted;
  if (sorter) {
    seq = at::gather(sorted, -1, *sorter);
  }
  int64_t m = seq.size(-1);
  seq = seq.to(dtype).contiguous().reshape({-1, m});
  at::Tensor vals = self.to(dtype).contiguous();
  int64_t q = self.dim() == 0 ? 1 : self.size(-1);
  at::Tensor result = at::empty(self.sizes(), self.options().dtype(out_int32 ? at::kInt : at::kLong));
  SearchParams p{};
  p.seq = gpu_address(seq);
  p.vals = gpu_address(vals);
  p.out = gpu_address(result);
  p.m = static_cast<uint32_t>(m);
  p.q = static_cast<uint32_t>(sorted.dim() == 1 ? self.numel() : q);
  p.seq_row = sorted.dim() == 1 ? 0 : static_cast<uint32_t>(m);
  p.total = static_cast<uint32_t>(self.numel());
  uint32_t groups = 0;
  check(vkt_grid_1d(p.total, &groups, &p.stride));
  if (m > 0) {
    dispatch(std::string("searchsorted_") + label, {right ? 1u : 0u, out_int32 ? 1u : 0u}, p, groups);
  } else {
    result.zero_();
  }
  out.copy_(result);
  return out;
}

bool right_side(bool right, std::optional<c10::string_view> side) {
  if (side.has_value()) {
    TORCH_CHECK(*side == "left" || *side == "right", "side can only be 'left' or 'right'");
    TORCH_CHECK(!right || *side == "right", "side and right can't be set to opposites");
    return *side == "right";
  }
  return right;
}

at::Tensor& searchsorted_out(const at::Tensor& sorted, const at::Tensor& self, bool out_int32, bool right,
                             std::optional<c10::string_view> side, const std::optional<at::Tensor>& sorter,
                             at::Tensor& out) {
  return searchsorted_impl(sorted, self, out_int32, right_side(right, side), sorter, out);
}

at::Tensor searchsorted(const at::Tensor& sorted, const at::Tensor& self, bool out_int32, bool right,
                        std::optional<c10::string_view> side, const std::optional<at::Tensor>& sorter) {
  at::Tensor out = at::empty({0}, self.options().dtype(out_int32 ? at::kInt : at::kLong));
  return searchsorted_out(sorted, self, out_int32, right, side, sorter, out);
}

at::Tensor& bucketize_out(const at::Tensor& self, const at::Tensor& boundaries, bool out_int32, bool right,
                          at::Tensor& out) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  return searchsorted_impl(boundaries, self, out_int32, right, std::nullopt, out);
}

at::Tensor bucketize(const at::Tensor& self, const at::Tensor& boundaries, bool out_int32, bool right) {
  at::Tensor out = at::empty({0}, self.options().dtype(out_int32 ? at::kInt : at::kLong));
  return bucketize_out(self, boundaries, out_int32, right, out);
}

// ---------------------------------------------------------------- unique

// (values, inverse, counts) of `flat` (1-D, already in the order to
// deduplicate) given the permutation `order` that produced it (or undefined).
std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_runs(const at::Tensor& s, const at::Tensor& order,
                                                           at::IntArrayRef inverse_shape, bool return_inverse,
                                                           bool return_counts) {
  int64_t n = s.numel();
  at::TensorOptions lopts = s.options().dtype(at::kLong);
  at::Tensor head = at::ones({n}, s.options().dtype(at::kBool));
  if (n > 1) {
    head.narrow(0, 1, n - 1).copy_(at::ne(s.narrow(0, 1, n - 1), s.narrow(0, 0, n - 1)));
  }
  at::Tensor values = at::masked_select(s, head);
  at::Tensor inverse = at::empty({0}, lopts), counts = at::empty({0}, lopts);
  if (return_inverse) {
    at::Tensor group = at::cumsum(head, 0, at::kLong) - 1;
    if (order.defined()) {
      inverse = at::empty({n}, lopts);
      inverse.index_put_({order}, group);
    } else {
      inverse = group;
    }
    inverse = inverse.reshape(inverse_shape);
  }
  if (return_counts) {
    at::Tensor starts = at::nonzero(head).reshape({-1});
    at::Tensor ends = at::cat({starts.narrow(0, 1, starts.numel() - 1), at::full({1}, n, lopts)});
    counts = ends - starts;
  }
  return {values, inverse, counts};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique2(const at::Tensor& self, bool sorted, bool return_inverse,
                                                       bool return_counts) {
  if (!sortable(self)) {
    auto stack = on_cpu("aten::_unique2", "", self, sorted, return_inverse, return_counts);
    return {stack[0].toTensor(), stack[1].toTensor(), stack[2].toTensor()};
  }
  at::TensorOptions lopts = self.options().dtype(at::kLong);
  if (self.numel() == 0) {
    return {at::empty({0}, self.options()), at::empty(return_inverse ? self.sizes() : at::IntArrayRef{0}, lopts),
            at::empty({0}, lopts)};
  }
  // Always sorted (the CPU returns sorted values for sorted=False too).
  at::Tensor flat = self.reshape({1, -1}).contiguous();
  at::Tensor order = sort_rows(flat, false).to(at::kLong).reshape({-1});
  at::Tensor s = flat.reshape({-1}).index_select(0, order);
  return unique_runs(s, order, self.sizes(), return_inverse, return_counts);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_consecutive(const at::Tensor& self, bool return_inverse,
                                                                  bool return_counts, std::optional<int64_t> dim) {
  if (dim.has_value() || !sortable(self)) {
    auto stack = on_cpu("aten::unique_consecutive", "", self, return_inverse, return_counts, dim);
    return {stack[0].toTensor(), stack[1].toTensor(), stack[2].toTensor()};
  }
  at::TensorOptions lopts = self.options().dtype(at::kLong);
  if (self.numel() == 0) {
    return {at::empty({0}, self.options()), at::empty(return_inverse ? self.sizes() : at::IntArrayRef{0}, lopts),
            at::empty({0}, lopts)};
  }
  return unique_runs(self.reshape({-1}), at::Tensor(), self.sizes(), return_inverse, return_counts);
}

// ---------------------------------------------------------------- small reductions from existing kernels

std::tuple<at::Tensor&, at::Tensor&> aminmax_out(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim,
                                                 at::Tensor& min, at::Tensor& max) {
  for (const at::Tensor* out : {&min, &max}) {
    TORCH_CHECK(out->scalar_type() == self.scalar_type(), "Expected out tensor to have dtype ", self.dtype(),
                ", but got ", out->dtype(), " instead");
  }
  if (dim.has_value()) {
    assign_out(min, at::amin(self, *dim, keepdim));
    assign_out(max, at::amax(self, *dim, keepdim));
  } else {
    TORCH_CHECK(self.numel() > 0, "aminmax(): cannot compute aminmax over an empty dimension as the operation has no "
                                  "identity.");
    at::Tensor flat = self.reshape({-1});
    at::Tensor lo = at::amin(flat, 0), hi = at::amax(flat, 0);
    if (keepdim && self.dim() > 0) {
      std::vector<int64_t> ones(self.dim(), 1);
      lo = lo.reshape(ones);
      hi = hi.reshape(ones);
    }
    assign_out(min, lo);
    assign_out(max, hi);
  }
  return {min, max};
}

std::tuple<at::Tensor, at::Tensor> var_mean(const at::Tensor& self, at::OptionalIntArrayRef dim,
                                            const std::optional<at::Scalar>& correction, bool keepdim) {
  return {at::var(self, dim, correction, keepdim), at::mean(self, dim, keepdim)};
}

std::tuple<at::Tensor, at::Tensor> std_mean(const at::Tensor& self, at::OptionalIntArrayRef dim,
                                            const std::optional<at::Scalar>& correction, bool keepdim) {
  return {at::std(self, dim, correction, keepdim), at::mean(self, dim, keepdim)};
}

at::Tensor nansum_vk(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                  std::optional<at::ScalarType> dtype) {
  reject_dtype("nansum", self.scalar_type(), at::isComplexType(self.scalar_type()));
  at::Tensor x = at::isFloatingType(self.scalar_type()) ? at::where(at::isnan(self), at::zeros_like(self), self) : self;
  return at::sum(x, dim, keepdim, dtype);
}

at::Tensor& nansum_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                       std::optional<at::ScalarType> dtype, at::Tensor& out) {
  assign_out(out, nansum_vk(self, dim, keepdim, dtype));
  return out;
}

at::Tensor trace(const at::Tensor& self) {
  reject_dtype("trace", self.scalar_type(), self.scalar_type() == at::kBool);
  TORCH_CHECK(self.dim() == 2, "trace: expected a matrix, but got tensor with dim ", self.dim());
  return self.diagonal().sum();
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("sort.values_stable", &sort_out);
  m.impl("sort.values", &sort_values);
  m.impl("sort.stable", &sort_stable);
  m.impl("topk.values", &topk_out);
  m.impl("kthvalue.values", &kthvalue_out);
  m.impl("median.dim_values", &median_dim_out<false>);
  m.impl("nanmedian.dim_values", &median_dim_out<true>);
  m.impl("median", &median_all<false>);
  m.impl("nanmedian", &median_all<true>);
  m.impl("mode", &vk::mode);
  m.impl("mode.values", &mode_out);
  m.impl("_cummax_helper", &cum_helper<op::ROWSCAN_CUMMAX>);
  m.impl("_cummin_helper", &cum_helper<op::ROWSCAN_CUMMIN>);
  m.impl("_logcumsumexp", &vk::logcumsumexp_vk);
  m.impl("_logcumsumexp.out", &logcumsumexp_out);
  m.impl("searchsorted.Tensor", &vk::searchsorted);
  m.impl("searchsorted.Tensor_out", &searchsorted_out);
  m.impl("bucketize.Tensor", &vk::bucketize);
  m.impl("bucketize.Tensor_out", &bucketize_out);
  m.impl("_unique2", &unique2);
  m.impl("unique_consecutive", &vk::unique_consecutive);
  m.impl("aminmax.out", &aminmax_out);
  m.impl("var_mean.correction", &vk::var_mean);
  m.impl("std_mean.correction", &vk::std_mean);
  m.impl("nansum", &vk::nansum_vk);
  m.impl("nansum.out", &nansum_out);
  m.impl("trace", &vk::trace);
}
