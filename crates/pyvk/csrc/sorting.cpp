// Sorting, searching and order statistics for device "vk" (M10): sort,
// topk, kthvalue, median, nanmedian, mode, unique, unique_consecutive (also
// along a dim),
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
#include <mutex>
#include <unordered_map>

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
  dispatch(kernel, {kSortLocalFull}, p, global_groups);
  for (int64_t k = 2 * kSortBlock; k <= P; k <<= 1) {
    p.k = static_cast<uint32_t>(k);
    for (int64_t j = k / 2; j >= kSortBlock; j >>= 1) {
      p.j = static_cast<uint32_t>(j);
      dispatch(kernel, {kSortGlobal}, p, global_groups);
    }
    dispatch(kernel, {kSortLocalMerge}, p, global_groups);
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
  if (at::isComplexType(self.scalar_type()) && is_vk(self)) {
    at::maybe_wrap_dim(dim, self.dim());  // (as sort's meta function, first)
    TORCH_CHECK(false, " Sort does not support complex dtypes on CPU");  // (the CPU's message)
  }
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

// (the CPU's zero_numel_check_dims)
void check_reduction_dim(const at::Tensor& self, int64_t dim, const char* fn) {
  if (self.dim() == 0) {
    TORCH_CHECK_INDEX(dim == 0 || dim == -1, fn, ": Expected reduction dim -1 or 0 for scalar but got ", dim);
  } else {
    TORCH_CHECK_INDEX(self.size(dim) != 0, fn, ": Expected reduction dim ", dim, " to have non-zero size.");
  }
}

// (values, indices) of a reduction over `dim` of an empty tensor.
std::tuple<at::Tensor, at::Tensor> empty_reduction(const at::Tensor& self, int64_t dim, bool keepdim) {
  std::vector<int64_t> shape = self.sizes().vec();
  if (self.dim() > 0) {
    if (keepdim) {
      shape[dim] = 1;
    } else {
      shape.erase(shape.begin() + dim);
    }
  }
  return {at::empty(shape, self.options()), at::empty(shape, self.options().dtype(at::kLong))};
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
  check_reduction_dim(self, dim, "kthvalue()");
  int64_t n = self.dim() == 0 ? 1 : self.size(dim);
  TORCH_CHECK(k >= 1 && k <= n, "kthvalue(): selected number k out of range for dimension ", dim);
  if (self.numel() == 0) {
    auto [v, i] = empty_reduction(self, dim, keepdim);
    assign_out(values, v);
    assign_out(indices, i);
    return {values, indices};
  }
  auto [sv, si] = sorted_rows(self, dim);
  at::Tensor pos = at::full({sv.size(0)}, k - 1, sv.options().dtype(at::kLong));
  auto [v, i] = pick(self, dim, keepdim, sv, si, pos);
  assign_out(values, v);
  assign_out(indices, i);
  return {values, indices};
}

// ---------------------------------------------------------------- median, nanmedian, mode

// The index the CPU's nanmedian gives a row of n NaNs. It runs
// std::nth_element with a comparator that is false for every pair of NaNs,
// so the element left first is the standard library's: index 0 with MSVC's,
// with libstdc++'s introselect another one that depends on n alone (7 for
// n = 9). Asked of the CPU once per n.
int64_t all_nan_nanmedian_index(int64_t n) {
  static std::mutex mutex;
  static std::unordered_map<int64_t, int64_t> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(n);
  if (it != cache.end()) return it->second;
  at::Tensor row = at::full({1, n}, std::numeric_limits<float>::quiet_NaN(), at::TensorOptions().dtype(at::kFloat));
  int64_t index = std::get<1>(at::nanmedian(row, 1)).item<int64_t>();
  cache.emplace(n, index);
  return index;
}

// Position of the (nan)median in each sorted row: the lower median; for
// median, the first NaN if the row has one (NaNs sort last, in index order).
// For nanmedian, a row of NaNs takes the CPU's choice of NaN.
at::Tensor median_positions(const at::Tensor& sorted_v, bool ignore_nan) {
  int64_t n = sorted_v.size(1);
  at::Tensor nans = at::isnan(sorted_v).sum(1);  // int64
  if (ignore_nan) {
    at::Tensor valid = n - nans;
    at::Tensor pos = at::clamp_min(at::div(valid - 1, 2, "floor"), 0);
    if (!at::isFloatingType(sorted_v.scalar_type()) && !at::isComplexType(sorted_v.scalar_type())) return pos;
    int64_t c = all_nan_nanmedian_index(n);
    return c == 0 ? pos : at::where(valid == 0, c, pos);
  }
  at::Tensor lower = at::full_like(nans, (n - 1) / 2);
  return at::where(nans > 0, n - nans, lower);
}

std::tuple<at::Tensor, at::Tensor> median_dim(const at::Tensor& self, int64_t dim, bool keepdim, bool ignore_nan) {
  dim = at::maybe_wrap_dim(dim, self.dim());
  check_reduction_dim(self, dim, ignore_nan ? "nanmedian()" : "median()");
  if (self.numel() == 0) {
    return empty_reduction(self, dim, keepdim);
  }
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
  p.out = gpu_address_out(out);
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
  // (the CPU's messages)
  TORCH_CHECK(self.device() == values.device(), "mode: expected device ", self.device(), " but got ", values.device(),
              " for values output");
  TORCH_CHECK(self.device() == indices.device(), "mode: expected device ", self.device(), " but got ",
              indices.device(), " for indices output");
  TORCH_CHECK(self.scalar_type() == values.scalar_type(), "mode: expected scalar type '", self.scalar_type(),
              "' but got '", values.scalar_type(), "' for values output");
  TORCH_CHECK(indices.scalar_type() == at::kLong, "mode: expected scalar type '", at::kLong, "' but got '",
              indices.scalar_type(), "' for indices output");
  auto [v, i] = at::mode(self, dim, keepdim);  // (our functional kernel, via the dispatcher)
  assign_out(values, v);
  assign_out(indices, i);
  return {values, indices};
}

// ---------------------------------------------------------------- cummax, cummin, logcumsumexp

template <uint32_t Op>
void cum_helper(const at::Tensor& self, at::Tensor& values, at::Tensor& indices, int64_t dim) {
  at::ScalarType t = self.scalar_type();
  bool small_int = t == at::kChar || t == at::kByte || t == at::kShort;
  if (is_vk(self) && (at::isReducedFloatingType(t) || small_int) && storage_label(t) != nullptr && is_vk(values) &&
      is_vk(indices)) {
    // (16-bit floats as float32, small integers as int32: the values selected are exact)
    at::ScalarType wide = small_int ? at::kInt : at::kFloat;
    at::Tensor v = at::empty(values.sizes(), values.options().dtype(wide));
    cum_helper<Op>(self.to(wide), v, indices, dim);
    values.copy_(v);
    return;
  }
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
  if (is_vk(self) && at::isReducedFloatingType(t)) {
    return logcumsumexp_vk(self.to(at::kFloat), dim).to(t);  // (in float32, rounded once)
  }
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
  if (sorter.has_value() && sorter->defined()) {  // (the CPU's searchsorted_pre_check)
    TORCH_CHECK(sorter->sizes() == sorted.sizes(), "torch.searchsorted(): boundary and sorter must have the same "
                "size, but got boundary tensor ", sorted.sizes(), "and got sorter tensor ", sorter->sizes());
    TORCH_CHECK(sorter->scalar_type() == at::kLong, "torch.searchsorted(): sorter must be a tensor of long dtype but "
                "got dtype ", sorter->scalar_type());
    if (sorter->numel() > 0) {
      auto [lo, hi] = at::aminmax(*sorter);
      TORCH_CHECK(lo.item<int64_t>() >= 0 && hi.item<int64_t>() < sorter->size(-1),
                  "torch.searchsorted(): sorter index out of range");
    }
  }
  TORCH_CHECK(out.scalar_type() == (out_int32 ? at::kInt : at::kLong), "torch.searchsorted(): output tensor's dtype "
              "is wrong, it can only be Int(int32) or Long(int64) depending on whether out_int32 flag is True, but we "
              "got output tensor's dtype ", out.scalar_type(), " and out_int32 flag is ", out_int32 ? "True" : "False");
  // (the CPU's searchsorted_pre_check of the shapes)
  if (self.dim() == 0) {
    TORCH_CHECK(sorted.dim() == 1, "torch.searchsorted(): input value can be a scalar only when boundaries tensor "
                "dimension is 1, but we got boundaries tensor dim(", sorted.dim(), ") and input value's dim(", self.dim(),
                ") numel(", self.numel(), ")");
  } else {
    TORCH_CHECK(sorted.dim() == 1 || (sorted.dim() == self.dim() &&
                                      sorted.sizes().slice(0, sorted.dim() - 1).equals(self.sizes().slice(0, self.dim() - 1))),
                "torch.searchsorted(): boundaries tensor should be 1 dimension or the first N-1 dimensions of boundaries "
                "tensor and input value tensor must match, but we got boundaries tensor ", sorted.sizes(),
                " and input value tensor ", self.sizes());
  }
  at::ScalarType dtype = at::result_type(sorted, self);
  const char* label = rowscan_label(dtype);
  bool ok = is_vk(sorted) && is_vk(self) && is_vk(out) && label != nullptr && dtype != at::kBool &&
            (!sorter || is_vk(*sorter)) && (sorted.dim() == 1 || sorted.dim() == self.dim());
  if (!ok) {
    on_cpu("aten::searchsorted", "Tensor_out", sorted, self, out_int32, right, std::nullopt, sorter, out);
    return out;
  }
  // (the CPU's warnings, from searchsorted_maybe_trim_input_tensors)
  if (!self.is_contiguous()) {
    TORCH_WARN_ONCE("torch.searchsorted(): input value tensor is non-contiguous, this will lower the performance due "
                    "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous "
                    "input value tensor if possible. This message will only appear once per program.");
  }
  if (!sorted.is_contiguous()) {
    TORCH_WARN_ONCE("torch.searchsorted(): boundary tensor is non-contiguous, this will lower the performance due "
                    "to extra data copy when converting non-contiguous tensor to contiguous, please use contiguous "
                    "boundary tensor if possible. This message will only appear once per program.");
  }
  at::native::resize_output(out, self.sizes());
  if (self.numel() == 0) {
    return out;
  }
  if (sorted.size(-1) == 0) {  // nothing to search: every value goes first, as the CPU
    return out.zero_();
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
  p.out = gpu_address_out(result);
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
  at::ScalarType st = sorted.scalar_type();
  if ((st == at::kBool || at::isComplexType(st)) && self.scalar_type() == st && is_vk(sorted) && is_vk(self) &&
      is_vk(out)) {
    // (the CPU has no bool or complex kernel, but searches no values without one)
    TORCH_CHECK_NOT_IMPLEMENTED(self.numel() == 0, "\"searchsorted_out_cpu\" not implemented for '", st, "'");
    at::Tensor r = at::empty(self.sizes(), self.options().dtype(out_int32 ? at::kInt : at::kLong));
    TORCH_CHECK(out.scalar_type() == r.scalar_type(), "torch.searchsorted(): output tensor's dtype is wrong, it can only "
                "be Int(int32) or Long(int64) depending on whether out_int32 flag is True, but we got output tensor's "
                "dtype ", out.scalar_type(), " and out_int32 flag is ", out_int32 ? "True" : "False");
    at::native::resize_output(out, r.sizes());
    return out;
  }
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
  at::ScalarType st = self.scalar_type();
  if (is_vk(self) && (st == at::kUInt16 || st == at::kUInt32 || st == at::kUInt64)) {
    // in an order-preserving signed dtype: the values widened (uint16/32), or
    // uint64's bits with the top one flipped
    at::Tensor key = st == at::kUInt64 ? at::bitwise_xor(self.view(at::kLong), INT64_MIN)
                                       : self.to(st == at::kUInt16 ? at::kInt : at::kLong);
    auto [values, inverse, counts] = unique2(key, sorted, return_inverse, return_counts);
    values = st == at::kUInt64 ? at::bitwise_xor(values, INT64_MIN).view(at::kUInt64) : values.to(st);
    return {values, inverse, counts};
  }
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

// ---------------------------------------------------------------- unique along a dim

// Dense lexicographic ranks of the rows of x2 [n, m] (equal rows, under ==,
// share a rank; NaNs are unequal to everything): every column ranked at once
// by sorting the columns, then adjacent column pairs combined into one key
// (rank * n + rank, below n^2) and re-ranked, halving the columns each round.
at::Tensor row_ranks(const at::Tensor& x2) {
  int64_t n = x2.size(0);
  at::TensorOptions lopts = x2.options().dtype(at::kLong);
  // [m, n]: column c's values in row order; rank each row of it.
  auto rank_rows = [&](const at::Tensor& cols) {
    at::Tensor order = sort_rows(cols.contiguous(), false).to(at::kLong);
    at::Tensor sorted = cols.gather(1, order);
    at::Tensor step = at::zeros(cols.sizes(), lopts);
    if (n > 1) {
      step.narrow(1, 1, n - 1).copy_(at::ne(sorted.narrow(1, 1, n - 1), sorted.narrow(1, 0, n - 1)));
    }
    return at::empty(cols.sizes(), lopts).scatter_(1, order, at::cumsum(step, 1));
  };
  at::Tensor ranks = rank_rows(x2.t());
  while (ranks.size(0) > 1) {
    int64_t m = ranks.size(0), half = m / 2;
    at::Tensor pairs = ranks.narrow(0, 0, 2 * half).reshape({half, 2, n});
    at::Tensor keys = pairs.select(1, 0) * n + pairs.select(1, 1);
    if (m % 2 == 1) {
      keys = at::cat({keys, ranks.narrow(0, m - 1, 1)});
    }
    ranks = rank_rows(keys);
  }
  return ranks.reshape({n});
}

// The checks and empty results the CPU gives for unique along `dim`; true
// if `self` is empty (outputs set).
bool unique_dim_empty(const at::Tensor& self, int64_t dim, std::tuple<at::Tensor, at::Tensor, at::Tensor>& out) {
  if (self.numel() != 0) return false;
  int64_t zero_dims = 0;
  for (int64_t s : self.sizes()) zero_dims += s == 0;
  TORCH_CHECK(zero_dims <= 1, "Number of zero sized dimensions is more than one, so unique cannot be applied ");
  TORCH_CHECK(self.size(dim) == 0, "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");
  at::TensorOptions lopts = self.options().dtype(at::kLong);
  out = {at::empty(self.sizes(), self.options()), at::empty({0}, lopts), at::empty({0}, lopts)};
  return true;
}

// Rows `keep` (int64 indices along dim 0 of x2) as a tensor shaped like self
// with `dim` resized.
at::Tensor rows_along(const at::Tensor& self, int64_t dim, const at::Tensor& x2, const at::Tensor& keep) {
  std::vector<int64_t> moved(self.sizes().begin(), self.sizes().end());
  moved.erase(moved.begin() + dim);
  moved.insert(moved.begin(), keep.numel());
  return x2.index_select(0, keep).reshape(moved).movedim(0, dim);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim(const at::Tensor& self, int64_t dim, bool sorted,
                                                          bool return_inverse, bool return_counts) {
  at::ScalarType st = self.scalar_type();
  if (is_vk(self) && self.dim() > 0 && (st == at::kUInt16 || st == at::kUInt32 || st == at::kUInt64)) {
    // (as unique2: in an order-preserving signed dtype)
    at::Tensor key = st == at::kUInt64 ? at::bitwise_xor(self.view(at::kLong), INT64_MIN)
                                       : self.to(st == at::kUInt16 ? at::kInt : at::kLong);
    auto [values, inverse, counts] = vk::unique_dim(key, dim, sorted, return_inverse, return_counts);
    values = st == at::kUInt64 ? at::bitwise_xor(values, INT64_MIN).view(at::kUInt64) : values.to(st);
    return {values, inverse, counts};
  }
  if (!sortable(self) || self.dim() == 0) {
    auto stack = on_cpu("aten::unique_dim", "", self, dim, sorted, return_inverse, return_counts);
    return {stack[0].toTensor(), stack[1].toTensor(), stack[2].toTensor()};
  }
  dim = c10::maybe_wrap_dim(dim, self.dim());
  std::tuple<at::Tensor, at::Tensor, at::Tensor> out;
  if (unique_dim_empty(self, dim, out)) return out;
  if (at::isFloatingType(self.scalar_type()) && at::isnan(self).any().item<bool>()) {
    // With NaNs the comparison sort has no order: slices with a NaN stay where
    // they are, each unique (NaN != NaN), and the slices between them sort
    // among themselves, as PyTorch's tests expect ([nan, 0, 0, nan, nan, 1]
    // gives [nan, 0, nan, nan, 1]). Each slice gets a leading key: 2k for the
    // k-th NaN slice, 2k + 1 for the slices after it, so one sort keeps that.
    int64_t n = self.size(dim);
    at::ScalarType ft = self.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;  // (holds the keys exactly)
    if (ft == at::kFloat && 2 * n + 1 >= (int64_t(1) << 24)) {
      auto stack = on_cpu("aten::unique_dim", "", self, dim, sorted, return_inverse, return_counts);
      return {stack[0].toTensor(), stack[1].toTensor(), stack[2].toTensor()};
    }
    at::Tensor moved = self.movedim(dim, 0);
    at::Tensor x = moved.reshape({n, -1}).to(ft);
    at::Tensor nanrow = at::isnan(x).any(1);
    at::Tensor c = at::cumsum(nanrow.to(at::kLong), 0) * 2;
    at::Tensor key = at::where(nanrow, c, c + 1).to(ft);
    // (a NaN slice is alone with its key: its NaNs, compared with nothing, are
    // zeros for the sort; the values come from the slices themselves)
    at::Tensor aug = at::cat({key.unsqueeze(1), at::nan_to_num(x, 0.0, 0.0, 0.0)}, 1);
    auto [u, inverse, counts] = vk::unique_dim(aug, 0, sorted, true, return_counts);
    int64_t groups = u.size(0);
    at::Tensor first = at::full({groups}, n, inverse.options())
                           .scatter_reduce_(0, inverse, at::arange(n, inverse.options()), "amin");
    std::vector<int64_t> shape(moved.sizes().begin(), moved.sizes().end());
    shape[0] = groups;
    at::Tensor values = moved.reshape({n, -1}).index_select(0, first).reshape(shape).movedim(0, dim);
    if (!return_inverse) inverse = at::empty({0}, inverse.options());
    return {values, inverse, counts};
  }
  int64_t n = self.size(dim);
  at::TensorOptions lopts = self.options().dtype(at::kLong);
  at::Tensor x2 = self.movedim(dim, 0).reshape({n, -1});
  at::Tensor key = row_ranks(x2);  // the rows' places among the distinct rows, in order
  int64_t u = key.max().item<int64_t>() + 1;
  at::Tensor first = at::full({u}, n, lopts).scatter_reduce_(0, key, at::arange(n, lopts), "amin");
  // (inverse and counts whether asked for or not, as the CPU)
  at::Tensor counts = at::zeros({u}, lopts).index_add_(0, key, at::ones({n}, lopts));
  return {rows_along(self, dim, x2, first), key, counts};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_dim_consecutive(const at::Tensor& self, int64_t dim,
                                                                      bool return_inverse, bool return_counts) {
  if (!sortable(self) || self.dim() == 0) {
    auto stack = on_cpu("aten::unique_dim_consecutive", "", self, dim, return_inverse, return_counts);
    return {stack[0].toTensor(), stack[1].toTensor(), stack[2].toTensor()};
  }
  dim = c10::maybe_wrap_dim(dim, self.dim());
  std::tuple<at::Tensor, at::Tensor, at::Tensor> out;
  if (unique_dim_empty(self, dim, out)) return out;
  int64_t n = self.size(dim);
  at::TensorOptions lopts = self.options().dtype(at::kLong);
  at::Tensor x2 = self.movedim(dim, 0).reshape({n, -1});
  at::Tensor head = at::ones({n}, self.options().dtype(at::kBool));
  if (n > 1) {
    head.narrow(0, 1, n - 1).copy_(at::ne(x2.narrow(0, 1, n - 1), x2.narrow(0, 0, n - 1)).any(1));
  }
  at::Tensor starts = at::nonzero(head).reshape({-1});
  // (inverse and counts whether asked for or not, as the CPU)
  at::Tensor inverse = at::cumsum(head, 0, at::kLong) - 1;
  at::Tensor ends = at::cat({starts.narrow(0, 1, starts.numel() - 1), at::full({1}, n, lopts)});
  return {rows_along(self, dim, x2, starts), inverse, ends - starts};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique_consecutive(const at::Tensor& self, bool return_inverse,
                                                                  bool return_counts, std::optional<int64_t> dim) {
  if (dim.has_value() && sortable(self) && self.dim() > 0) {
    return vk::unique_dim_consecutive(self, *dim, return_inverse, return_counts);
  }
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

// searchsorted with a scalar value: as a wrapped-number tensor, as the CPU
// does (its promotion rules are a binary op's).
at::Tensor scalar_value(const at::Scalar& value, const at::Device& device) {
  at::Tensor t = c10::scalar_to_tensor(value, device);
  t.unsafeGetTensorImpl()->set_wrapped_number(true);
  return t;
}

at::Tensor searchsorted_scalar(const at::Tensor& sorted, const at::Scalar& self, bool out_int32, bool right,
                               std::optional<c10::string_view> side, const std::optional<at::Tensor>& sorter) {
  return vk::searchsorted(sorted, scalar_value(self, sorted.device()), out_int32, right, side, sorter);
}

at::Tensor& searchsorted_scalar_out(const at::Tensor& sorted, const at::Scalar& self, bool out_int32, bool right,
                                    std::optional<c10::string_view> side, const std::optional<at::Tensor>& sorter,
                                    at::Tensor& out) {
  return vk::searchsorted_out(sorted, scalar_value(self, sorted.device()), out_int32, right, side, sorter, out);
}

// _unique (no counts): _unique2's values and inverse.
std::tuple<at::Tensor, at::Tensor> unique_values(const at::Tensor& self, bool sorted, bool return_inverse) {
  auto [values, inverse, counts] = unique2(self, sorted, return_inverse, false);
  return {values, inverse};
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
  m.impl("searchsorted.Scalar", &searchsorted_scalar);
  m.impl("searchsorted.Scalar_out", &searchsorted_scalar_out);
  m.impl("_unique", &unique_values);
  m.impl("bucketize.Tensor", &vk::bucketize);
  m.impl("bucketize.Tensor_out", &bucketize_out);
  m.impl("_unique2", &unique2);
  m.impl("unique_consecutive", &vk::unique_consecutive);
  m.impl("unique_dim", &vk::unique_dim);
  m.impl("unique_dim_consecutive", &vk::unique_dim_consecutive);
  m.impl("aminmax.out", &aminmax_out);
  m.impl("var_mean.correction", &vk::var_mean);
  m.impl("std_mean.correction", &vk::std_mean);
  m.impl("nansum", &vk::nansum_vk);
  m.impl("nansum.out", &nansum_out);
  m.impl("trace", &vk::trace);
}
