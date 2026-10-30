// Reductions and scans for device "vk": sum, mean, prod, amax/amin, max/min
// (full and along a dim, with indices), argmax/argmin, var/std, vector
// norms (ord 1, 2, +-inf), any/all, cumsum/cumprod.
//
// Kernels: shaders/reduce.slang (tree reduction in chunks, partials reduced
// again by further passes), argreduce.slang ((value, index) pairs) and
// scan.slang (chunked scan + recursive scan of chunk totals). Integer and
// bool inputs accumulate in int64, as PyTorch's sums do.

#include "vk.h"

#include "opcodes.h"
#include "strided.h"

#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vk {
namespace {

// Op codes; must match shaders/reduce.slang.
// Shorthands for the generated codes (shaders/opcodes.txt).
enum ROp : uint32_t {
  kSum = op::REDUCE_SUM, kProd = op::REDUCE_PROD, kMax = op::REDUCE_MAX, kMin = op::REDUCE_MIN,
  kSumAbs = op::REDUCE_SUMABS, kSumSq = op::REDUCE_SUMSQ, kMaxAbs = op::REDUCE_MAXABS, kMinAbs = op::REDUCE_MINABS,
  kSqDev = op::REDUCE_SQDEV, kAny = op::REDUCE_ANY, kAll = op::REDUCE_ALL,
};
enum Final : uint32_t { kNone = op::FINAL_NONE, kScale = op::FINAL_SCALE, kSqrtScale = op::FINAL_SQRT_SCALE };

struct ReduceParams {
  uint64_t in, out, mean;
  uint32_t ond, rnd;
  double alpha;
  uint32_t n_out, n_red, chunk, nchunks, ngroups, pad;
  uint32_t osz[4];
  int32_t ost[4];
  uint32_t rsz[4];
  int32_t rst[4];
};
static_assert(sizeof(ReduceParams) == 128 && offsetof(ReduceParams, alpha) == 32 && offsetof(ReduceParams, osz) == 64);

struct ArgParams {
  uint64_t in, out, out_idx, in_idx;
  uint32_t ond, rnd, n_out, n_red, chunk, nchunks, ngroups, pad;
  uint32_t osz[4];
  int32_t ost[4];
  uint32_t rsz[4];
  int32_t rst[4];
};
static_assert(sizeof(ArgParams) == 128 && offsetof(ArgParams, osz) == 64);

struct ScanParams {
  uint64_t in, out, totals;
  uint32_t ond;
  int32_t dstride;
  uint32_t rows, n, nchunks, ngroups, pad[4];
  uint32_t osz[4];
  int32_t ost[4];
  uint32_t pad2[8];
};
static_assert(sizeof(ScanParams) == 128 && offsetof(ScanParams, rows) == 32 && offsetof(ScanParams, osz) == 64);

constexpr uint32_t kMaxGroups = 65535;

// Up to 4 dims, innermost first.
struct Dims {
  uint32_t n = 0;
  std::array<uint32_t, 4> size = {1, 1, 1, 1};
  std::array<int32_t, 4> stride = {0, 0, 0, 0};
};

// Merges dims given outermost-first into innermost-first Dims (strided.h);
// false if more than 4 remain.
bool coalesce(const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, Dims& out) {
  StridedDims<1> d = merge_dims<1>(sizes, {strides});
  if (d.ndim() > 4) {
    return false;
  }
  out = Dims{};
  out.n = static_cast<uint32_t>(d.ndim());
  for (size_t i = 0; i < d.ndim(); ++i) {
    out.size[i] = static_cast<uint32_t>(d.size[i]);
    out.stride[i] = static_cast<int32_t>(d.stride[i][0]);
  }
  return true;
}

Dims single_dim(int64_t size, int64_t stride) {
  Dims d;
  d.n = 1;
  d.size[0] = static_cast<uint32_t>(size);
  d.stride[0] = static_cast<int32_t>(stride);
  return d;
}

at::ScalarType accumulator(at::ScalarType t) {
  return at::isFloatingType(t) ? t : at::kLong;
}

// Which input elements reduce into which output elements.
struct Plan {
  Dims outer, red;
  int64_t n_out = 1, n_red = 1;
  std::vector<int64_t> out_shape;
};

// Plans reducing `self` over `dims` (none/empty = all dims). False if the
// kernels cannot address it (more than 4 non-mergeable kept or reduced dims,
// or offsets beyond 32 bits).
bool plan_reduce(const at::Tensor& self, at::OptionalIntArrayRef dims, bool keepdim, Plan& plan) {
  int64_t ndim = self.dim();
  std::vector<bool> reduced(ndim, !dims.has_value() || dims->empty());
  if (dims.has_value()) {
    for (int64_t d : *dims) {
      int64_t w = at::maybe_wrap_dim(d, ndim);  // validates; a 0-dim tensor accepts dim 0 / -1
      if (ndim > 0) {
        reduced[w] = true;
      }
    }
  }
  std::vector<int64_t> kept_size, kept_stride, red_size, red_stride;
  plan = Plan{};
  int64_t span = 0;
  for (int64_t d = 0; d < ndim; ++d) {
    int64_t size = self.size(d), stride = self.stride(d);
    span += (size - 1) * stride;
    if (reduced[d]) {
      red_size.push_back(size);
      red_stride.push_back(stride);
      plan.n_red *= size;
      if (keepdim) {
        plan.out_shape.push_back(1);
      }
    } else {
      kept_size.push_back(size);
      kept_stride.push_back(stride);
      plan.n_out *= size;
      plan.out_shape.push_back(size);
    }
  }
  if (span >= (int64_t(1) << 31) || plan.n_out * std::max<int64_t>(plan.n_red, 1) >= (int64_t(1) << 32)) {
    return false;
  }
  return coalesce(kept_size, kept_stride, plan.outer) && coalesce(red_size, red_stride, plan.red);
}

// Workgroup chunks per output: enough work items to fill the GPU, each
// chunk at least ~2048 elements.
int64_t choose_chunks(int64_t n_out, int64_t n_red) {
  constexpr int64_t kTargetGroups = 1024;
  if (n_out >= kTargetGroups || n_red <= 4096) {
    return 1;
  }
  return std::max<int64_t>(1, std::min((kTargetGroups + n_out - 1) / n_out, (n_red + 2047) / 2048));
}

template <typename P>
void set_dims(P& p, const Dims& outer, const Dims& red) {
  p.ond = outer.n;
  p.rnd = red.n;
  for (int i = 0; i < 4; ++i) {
    p.osz[i] = outer.size[i];
    p.ost[i] = outer.stride[i];
    p.rsz[i] = red.size[i];
    p.rst[i] = red.stride[i];
  }
}

// Reduces `self` per `plan`; returns a contiguous [n_out] tensor of the
// accumulator dtype. `mean` (OP kSqDev) holds one mean per output.
at::Tensor run_reduce(const at::Tensor& self, const Plan& plan, ROp op, Final final, double alpha,
                      const at::Tensor* mean = nullptr) {
  at::ScalarType acc = accumulator(self.scalar_type());
  at::Tensor in = self;
  Dims outer = plan.outer, red = plan.red;
  int64_t n_red = plan.n_red;
  bool level0 = true;
  while (true) {
    // Strided reduced dims with many adjacent outputs (e.g. column sums): one
    // thread per (output, chunk) so neighbouring threads read neighbouring
    // addresses. Otherwise one workgroup per (output, chunk).
    bool per_thread = level0 && red.n > 0 && red.stride[0] != 1 && outer.n > 0 && outer.stride[0] == 1 &&
                      plan.n_out >= 256;
    int64_t nchunks = per_thread ? std::max<int64_t>(1, std::min((int64_t(1) << 16) / plan.n_out, n_red / 64))
                                 : choose_chunks(plan.n_out, n_red);
    int64_t chunk = (n_red + nchunks - 1) / nchunks;
    at::Tensor dst = at::empty({plan.n_out * nchunks}, self.options().dtype(acc));
    ReduceParams p{};
    p.in = gpu_address(in);
    p.out = gpu_address(dst);
    p.mean = (mean != nullptr && level0) ? gpu_address(*mean) : 0;
    p.alpha = alpha;
    p.n_out = static_cast<uint32_t>(plan.n_out);
    p.n_red = static_cast<uint32_t>(n_red);
    p.chunk = static_cast<uint32_t>(chunk);
    p.nchunks = static_cast<uint32_t>(nchunks);
    int64_t work = plan.n_out * nchunks;
    uint32_t groups = static_cast<uint32_t>(std::min<int64_t>(per_thread ? (work + 255) / 256 : work, kMaxGroups));
    p.ngroups = groups;
    set_dims(p, outer, red);
    std::string kernel = std::string("reduce_") + dtype_label(in.scalar_type());
    dispatch(kernel, {op, level0 ? 1u : 0u, nchunks == 1 ? static_cast<uint32_t>(final) : 0u, per_thread ? 1u : 0u}, p,
             groups);
    if (nchunks == 1) {
      return dst;
    }
    // Reduce the partials [n_out, nchunks] in another pass.
    in = dst;
    outer = single_dim(plan.n_out, nchunks);
    red = single_dim(nchunks, 1);
    n_red = nchunks;
    level0 = false;
  }
}

// (values in self's dtype, int64 indices), each contiguous [n_out]; OP 0 max, 1 min.
std::pair<at::Tensor, at::Tensor> run_argreduce(const at::Tensor& self, const Plan& plan, uint32_t op) {
  at::Tensor in = self, in_idx;
  Dims outer = plan.outer, red = plan.red;
  int64_t n_red = plan.n_red;
  bool level0 = true;
  std::string kernel = std::string("argreduce_") + dtype_label(self.scalar_type());
  while (true) {
    int64_t nchunks = choose_chunks(plan.n_out, n_red);
    int64_t chunk = (n_red + nchunks - 1) / nchunks;
    at::Tensor values = at::empty({plan.n_out * nchunks}, self.options());
    at::Tensor indices = at::empty({plan.n_out * nchunks}, self.options().dtype(at::kLong));
    ArgParams p{};
    p.in = gpu_address(in);
    p.out = gpu_address(values);
    p.out_idx = gpu_address(indices);
    p.in_idx = level0 ? 0 : gpu_address(in_idx);
    p.n_out = static_cast<uint32_t>(plan.n_out);
    p.n_red = static_cast<uint32_t>(n_red);
    p.chunk = static_cast<uint32_t>(chunk);
    p.nchunks = static_cast<uint32_t>(nchunks);
    uint32_t groups = static_cast<uint32_t>(std::min<int64_t>(plan.n_out * nchunks, kMaxGroups));
    p.ngroups = groups;
    set_dims(p, outer, red);
    dispatch(kernel, {op, level0 ? 1u : 0u}, p, groups);
    if (nchunks == 1) {
      return {values, indices};
    }
    in = values;
    in_idx = indices;
    outer = single_dim(plan.n_out, nchunks);
    red = single_dim(nchunks, 1);
    n_red = nchunks;
    level0 = false;
  }
}

bool reducible(const at::Tensor& self, bool allow_bool = true) {
  const char* label = dtype_label(self.scalar_type());
  return is_vk(self) && label != nullptr && !at::isComplexType(self.scalar_type()) &&
         (allow_bool || self.scalar_type() != at::kBool);
}

void store(const at::Tensor& out, const at::Tensor& flat, const std::vector<int64_t>& shape) {
  at::native::resize_output(out, shape);
  out.copy_(flat.view(shape));
}

// ---------------------------------------------------------------- complex inputs

// Complex reductions from real ones: sum and mean act on the real view (parts
// as a last dim); products are log-depth trees of complex multiplications.

bool complex_vk(const at::Tensor& t) {
  return is_vk(t) && at::isComplexType(t.scalar_type()) && dtype_label(t.scalar_type()) != nullptr;
}

std::vector<int64_t> reduce_dims_of(const at::Tensor& self, at::OptionalIntArrayRef dim) {
  std::vector<int64_t> dims;
  if (!dim.has_value() || dim->empty()) {
    for (int64_t d = 0; d < self.dim(); ++d) dims.push_back(d);
  } else {
    for (int64_t d : *dim) dims.push_back(at::maybe_wrap_dim(d, std::max<int64_t>(self.dim(), 1)));
  }
  return dims;
}

// sum or mean of complex `self` over dims, through its real view.
at::Tensor complex_sum(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, bool mean) {
  at::Tensor real = at::view_as_real(self.resolve_conj());
  std::vector<int64_t> dims = self.dim() == 0 ? std::vector<int64_t>{} : reduce_dims_of(self, dim);
  at::Tensor r = dims.empty() ? real.clone() : (mean ? real.mean(dims, keepdim) : real.sum(dims, keepdim));
  return at::view_as_complex(r.contiguous());
}

// Product along the last dim of x [..., n] by pairwise multiplication.
at::Tensor complex_prod_last(at::Tensor x) {
  if (x.size(-1) == 0) {
    return at::ones(x.sizes().slice(0, x.dim() - 1), x.options());
  }
  while (x.size(-1) > 1) {
    int64_t n = x.size(-1);
    if (n % 2 == 1) {
      std::vector<int64_t> shape = x.sizes().vec();
      shape.back() = 1;
      x = at::cat({x, at::ones(shape, x.options())}, -1);
      n += 1;
    }
    x = at::mul(x.slice(-1, 0, n, 2), x.slice(-1, 1, n, 2));
  }
  return x.squeeze(-1);
}

// Inclusive scan with complex multiplication along the last dim (Hillis-Steele).
at::Tensor complex_cumprod_last(at::Tensor x) {
  int64_t n = x.size(-1);
  for (int64_t offset = 1; offset < n; offset *= 2) {
    at::Tensor next = x.clone();
    next.narrow(-1, offset, n - offset).copy_(at::mul(x.narrow(-1, offset, n - offset), x.narrow(-1, 0, n - offset)));
    x = next;
  }
  return x;
}

// ---------------------------------------------------------------- sum / mean / prod / any / all

at::Tensor& sum_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                    std::optional<at::ScalarType> dtype, at::Tensor& out) {
  at::Tensor input = dtype.has_value() && is_vk(self) ? self.to(*dtype) : self;
  if (complex_vk(input) && is_vk(out)) {
    at::Tensor r = complex_sum(input, dim, keepdim, false);
    store(out, r, r.sizes().vec());
    return out;
  }
  Plan plan;
  if (!reducible(input) || !is_vk(out) || !plan_reduce(input, dim, keepdim, plan)) {
    on_cpu("aten::sum", "IntList_out", self, dim, keepdim, dtype, out);
    return out;
  }
  store(out, run_reduce(input, plan, kSum, kNone, 0.0), plan.out_shape);
  return out;
}

at::Tensor& mean_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                     std::optional<at::ScalarType> dtype, at::Tensor& out) {
  at::Tensor input = dtype.has_value() && is_vk(self) ? self.to(*dtype) : self;
  if (complex_vk(input) && is_vk(out)) {
    at::Tensor r = complex_sum(input, dim, keepdim, true);
    store(out, r, r.sizes().vec());
    return out;
  }
  Plan plan;
  if (!reducible(input) || !at::isFloatingType(input.scalar_type()) || !is_vk(out) ||
      !plan_reduce(input, dim, keepdim, plan)) {
    on_cpu("aten::mean", "out", self, dim, keepdim, dtype, out);
    return out;
  }
  double alpha = 1.0 / static_cast<double>(plan.n_red);  // inf for empty: mean of nothing is NaN
  store(out, run_reduce(input, plan, kSum, kScale, alpha), plan.out_shape);
  return out;
}

at::Tensor& prod_out(const at::Tensor& self, int64_t dim, bool keepdim, std::optional<at::ScalarType> dtype,
                     at::Tensor& out) {
  at::Tensor input = dtype.has_value() && is_vk(self) ? self.to(*dtype) : self;
  if (complex_vk(input) && is_vk(out)) {
    int64_t d = at::maybe_wrap_dim(dim, std::max<int64_t>(input.dim(), 1));
    at::Tensor r = input.dim() == 0 ? input.clone() : complex_prod_last(input.movedim(d, -1).resolve_conj());
    if (keepdim && input.dim() > 0) r = r.unsqueeze(d);
    store(out, r, r.sizes().vec());
    return out;
  }
  Plan plan;
  int64_t dims[] = {dim};
  if (!reducible(input) || !is_vk(out) || !plan_reduce(input, dims, keepdim, plan)) {
    on_cpu("aten::prod", "int_out", self, dim, keepdim, dtype, out);
    return out;
  }
  store(out, run_reduce(input, plan, kProd, kNone, 0.0), plan.out_shape);
  return out;
}

at::Tensor prod(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  at::Tensor input = dtype.has_value() ? self.to(*dtype) : self;
  if (complex_vk(input)) {
    return complex_prod_last(input.resolve_conj().reshape({-1}));
  }
  Plan plan;
  if (!reducible(input) || !plan_reduce(input, std::nullopt, false, plan)) {
    return on_cpu("aten::prod", "", self, dtype)[0].toTensor();
  }
  at::ScalarType result = at::isIntegralType(input.scalar_type(), true) ? at::kLong : input.scalar_type();
  return run_reduce(input, plan, kProd, kNone, 0.0).view(at::IntArrayRef{}).to(result);
}

template <ROp Op>
at::Tensor& any_all_dims_out(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, at::Tensor& out) {
  if (complex_vk(self) && is_vk(out)) {
    return any_all_dims_out<Op>(at::ne(self, 0), dim, keepdim, out);
  }
  Plan plan;
  if (!reducible(self) || !is_vk(out) || !plan_reduce(self, dim, keepdim, plan)) {
    on_cpu(Op == kAny ? "aten::any" : "aten::all", "dims_out", self, dim, keepdim, out);
    return out;
  }
  store(out, run_reduce(self, plan, Op, kNone, 0.0), plan.out_shape);
  return out;
}

template <ROp Op>
at::Tensor& any_all_out(const at::Tensor& self, at::Tensor& out) {
  return any_all_dims_out<Op>(self, std::nullopt, false, out);
}

template <ROp Op>
at::Tensor& any_all_dim_out(const at::Tensor& self, int64_t dim, bool keepdim, at::Tensor& out) {
  int64_t dims[] = {dim};
  return any_all_dims_out<Op>(self, dims, keepdim, out);
}

// ---------------------------------------------------------------- max / min / argmax / argmin

template <ROp Op>
at::Tensor& amax_amin_out(const at::Tensor& self, at::IntArrayRef dim, bool keepdim, at::Tensor& out) {
  Plan plan;
  if (!reducible(self) || !is_vk(out) || !plan_reduce(self, dim, keepdim, plan)) {
    on_cpu(Op == kMax ? "aten::amax" : "aten::amin", "out", self, dim, keepdim, out);
    return out;
  }
  TORCH_CHECK(plan.n_red > 0 || self.numel() > 0 || plan.n_out == 0,
              Op == kMax ? "amax" : "amin", "(): Expected reduction dim to have non-zero size.");
  store(out, run_reduce(self, plan, Op, kNone, 0.0), plan.out_shape);
  return out;
}

template <ROp Op>
at::Tensor full_max_min(const at::Tensor& self) {
  Plan plan;
  if (!reducible(self) || !plan_reduce(self, std::nullopt, false, plan)) {
    return on_cpu(Op == kMax ? "aten::max" : "aten::min", "", self)[0].toTensor();
  }
  TORCH_CHECK(self.numel() > 0, Op == kMax ? "max" : "min",
              "(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the "
              "'dim' argument.");
  return run_reduce(self, plan, Op, kNone, 0.0).view(at::IntArrayRef{}).to(self.scalar_type());
}

template <ROp Op>
at::Tensor& full_max_min_out(const at::Tensor& self, at::Tensor& out) {
  at::Tensor r = full_max_min<Op>(self);
  at::native::resize_output(out, {});
  out.copy_(r);
  return out;
}

bool arg_reducible(const at::Tensor& self) {
  return reducible(self, /*allow_bool=*/false);
}

template <uint32_t Op>  // 0 max, 1 min
std::tuple<at::Tensor&, at::Tensor&> max_min_dim_out(const at::Tensor& self, int64_t dim, bool keepdim,
                                                     at::Tensor& values, at::Tensor& indices) {
  Plan plan;
  int64_t dims[] = {dim};
  if (self.dim() == 0 && is_vk(self) && is_vk(values) && is_vk(indices)) {
    at::maybe_wrap_dim(dim, 1);  // checks the dim
    at::native::resize_output(values, {});
    at::native::resize_output(indices, {});
    values.copy_(self);
    indices.zero_();
    return {values, indices};
  }
  if (self.scalar_type() == at::kBool && is_vk(self) && is_vk(values) && is_vk(indices)) {
    // max is "any", min is "all"; through int32.
    at::Tensor wide = at::empty({0}, values.options().dtype(at::kInt));
    max_min_dim_out<Op>(self.to(at::kInt), dim, keepdim, wide, indices);
    at::native::resize_output(values, wide.sizes());
    values.copy_(wide);
    return {values, indices};
  }
  if (!arg_reducible(self) || !is_vk(values) || !is_vk(indices) || self.dim() == 0 ||
      !plan_reduce(self, dims, keepdim, plan)) {
    on_cpu(Op == 0 ? "aten::max" : "aten::min", Op == 0 ? "dim_max" : "dim_min", self, dim, keepdim, values, indices);
    return {values, indices};
  }
  TORCH_CHECK(plan.n_red > 0, Op == 0 ? "max" : "min", "(): Expected reduction dim ", dim, " to have non-zero size.");
  auto [v, i] = run_argreduce(self, plan, Op);
  store(values, v, plan.out_shape);
  store(indices, i, plan.out_shape);
  return {values, indices};
}

template <uint32_t Op>
at::Tensor& argmax_argmin_out(const at::Tensor& self, std::optional<int64_t> dim, bool keepdim, at::Tensor& out) {
  Plan plan;
  std::vector<int64_t> dims;
  if (dim.has_value()) {
    dims.push_back(*dim);
  }
  bool ok = arg_reducible(self) && is_vk(out) &&
            plan_reduce(self, dim.has_value() ? at::OptionalIntArrayRef(dims) : std::nullopt, keepdim, plan);
  if (!ok || (!dim.has_value() && keepdim)) {
    on_cpu(Op == 0 ? "aten::argmax" : "aten::argmin", "out", self, dim, keepdim, out);
    return out;
  }
  TORCH_CHECK(plan.n_red > 0, Op == 0 ? "argmax" : "argmin", "(): Expected reduction dim to have non-zero size.");
  store(out, run_argreduce(self, plan, Op).second, plan.out_shape);
  return out;
}

// ---------------------------------------------------------------- var / std / norms

template <bool Std>
at::Tensor& var_std_out(const at::Tensor& self, at::OptionalIntArrayRef dim, const std::optional<at::Scalar>& correction,
                        bool keepdim, at::Tensor& out) {
  if (complex_vk(self) && is_vk(out)) {
    // var(z) = var(Re z) + var(Im z)
    at::Tensor real = at::view_as_real(self.resolve_conj());
    at::Tensor v = at::var(real.select(-1, 0), dim, correction, keepdim) +
                   at::var(real.select(-1, 1), dim, correction, keepdim);
    at::Tensor r = Std ? v.sqrt() : v;
    store(out, r, r.sizes().vec());
    return out;
  }
  Plan plan;
  if (!reducible(self) || !at::isFloatingType(self.scalar_type()) || !is_vk(out) ||
      !plan_reduce(self, dim, keepdim, plan)) {
    on_cpu(Std ? "aten::std" : "aten::var", "correction_out", self, dim, correction, keepdim, out);
    return out;
  }
  double n = static_cast<double>(plan.n_red);
  double ddof = correction.has_value() ? correction->toDouble() : 1.0;
  at::Tensor mean = run_reduce(self, plan, kSum, kScale, 1.0 / n);
  double alpha = 1.0 / std::max(n - ddof, 0.0);  // inf -> nan/inf results, as on the CPU
  store(out, run_reduce(self, plan, kSqDev, Std ? kSqrtScale : kScale, alpha, &mean), plan.out_shape);
  return out;
}

template <bool Std>
at::Tensor var_std(const at::Tensor& self, at::OptionalIntArrayRef dim, const std::optional<at::Scalar>& correction,
                   bool keepdim) {
  // var/std of complex input are real.
  at::Tensor out = at::empty({0}, self.options().dtype(c10::toRealValueType(self.scalar_type())));
  return var_std_out<Std>(self, dim, correction, keepdim, out);
}

at::Tensor& vector_norm_out(const at::Tensor& self, const at::Scalar& ord, at::OptionalIntArrayRef dim, bool keepdim,
                            std::optional<at::ScalarType> dtype, at::Tensor& out) {
  at::Tensor input = dtype.has_value() && is_vk(self) ? self.to(*dtype) : self;
  if (complex_vk(input) && is_vk(out)) {  // the norm of |z|
    at::Tensor r = at::linalg_vector_norm(at::abs(input), ord, dim, keepdim);
    store(out, r, r.sizes().vec());
    return out;
  }
  double p = ord.toDouble();
  ROp op = kSum;
  Final final = kNone;
  if (p == 2.0) {
    op = kSumSq;
    final = kSqrtScale;
  } else if (p == 1.0) {
    op = kSumAbs;
  } else if (p == std::numeric_limits<double>::infinity()) {
    op = kMaxAbs;
  } else if (p == -std::numeric_limits<double>::infinity()) {
    op = kMinAbs;
  }
  if (op == kSum && at::isFloatingType(input.scalar_type()) && is_vk(input) && is_vk(out)) {
    // Other orders: (sum |x|^p)^(1/p); p = 0 counts the nonzeros.
    std::vector<int64_t> dims = reduce_dims_of(input, dim);
    at::Tensor a = at::abs(input);
    at::Tensor r = p == 0.0 ? at::ne(a, 0).to(input.scalar_type()).sum(dims, keepdim)
                            : at::pow(at::pow(a, p).sum(dims, keepdim), 1.0 / p);
    store(out, r, r.sizes().vec());
    return out;
  }
  Plan plan;
  if (op == kSum || !reducible(input) || !at::isFloatingType(input.scalar_type()) || !is_vk(out) ||
      !plan_reduce(input, dim, keepdim, plan) || plan.n_red == 0) {
    on_cpu("aten::linalg_vector_norm", "out", self, ord, dim, keepdim, dtype, out);
    return out;
  }
  store(out, run_reduce(input, plan, op, final, 1.0), plan.out_shape);
  return out;
}

// ---------------------------------------------------------------- cumsum / cumprod

// Scans `in` along its dim (`rows` x `n`, row dims `rowdims`, `dstride`
// between elements) into contiguous `out` [rows, n] of the accumulator dtype.
void run_scan(const at::Tensor& in, const Dims& rowdims, int64_t rows, int64_t n, int64_t dstride, uint32_t op,
              const at::Tensor& out) {
  constexpr int64_t kChunk = 1024;  // 256 threads x 4 items (scan.slang)
  int64_t nchunks = (n + kChunk - 1) / kChunk;
  at::ScalarType acc = out.scalar_type();
  at::Tensor totals = at::empty({rows * nchunks}, out.options());
  ScanParams p{};
  p.in = gpu_address(in);
  p.out = gpu_address(out);
  p.totals = gpu_address(totals);
  p.ond = rowdims.n;
  p.dstride = static_cast<int32_t>(dstride);
  p.rows = static_cast<uint32_t>(rows);
  p.n = static_cast<uint32_t>(n);
  p.nchunks = static_cast<uint32_t>(nchunks);
  uint32_t groups = static_cast<uint32_t>(std::min<int64_t>(rows * nchunks, kMaxGroups));
  p.ngroups = groups;
  for (int i = 0; i < 4; ++i) {
    p.osz[i] = rowdims.size[i];
    p.ost[i] = rowdims.stride[i];
  }
  dispatch(std::string("scan_") + dtype_label(in.scalar_type()), {op, 0}, p, groups);
  if (nchunks == 1) {
    return;
  }
  // Scan the chunk totals of every row, then add them to the later chunks.
  at::Tensor scanned = at::empty({rows * nchunks}, out.options());
  run_scan(totals, single_dim(rows, nchunks), rows, nchunks, 1, op, scanned);
  p.totals = gpu_address(scanned);
  dispatch(std::string("scan_") + dtype_label(acc), {op, 1}, p, groups);
}

template <uint32_t Op>  // 0 sum, 1 prod
at::Tensor& cumulative_out(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype, at::Tensor& out) {
  const char* name = Op == 0 ? "aten::cumsum" : "aten::cumprod";
  at::Tensor input = dtype.has_value() && is_vk(self) ? self.to(*dtype) : self;
  if (complex_vk(input) && is_vk(out) && input.dim() > 0) {
    int64_t d = at::maybe_wrap_dim(dim, input.dim());
    at::Tensor r = Op == 0 ? at::view_as_complex(at::view_as_real(input.resolve_conj()).cumsum(d).contiguous())
                           : complex_cumprod_last(input.movedim(d, -1).resolve_conj().contiguous()).movedim(-1, d);
    at::native::resize_output(out, input.sizes());
    out.copy_(r);
    return out;
  }
  if (input.dim() == 0 && is_vk(input) && is_vk(out)) {
    at::maybe_wrap_dim(dim, 1);
    at::native::resize_output(out, {});
    out.copy_(input);
    return out;
  }
  if (!reducible(input) || !is_vk(out) || input.dim() == 0) {
    on_cpu(name, "out", self, dim, dtype, out);
    return out;
  }
  dim = at::maybe_wrap_dim(dim, input.dim());
  int64_t n = input.size(dim);
  std::vector<int64_t> row_size, row_stride, moved_shape;
  int64_t rows = 1, span = 0;
  for (int64_t d = 0; d < input.dim(); ++d) {
    span += (input.size(d) - 1) * input.stride(d);
    if (d != dim) {
      row_size.push_back(input.size(d));
      row_stride.push_back(input.stride(d));
      moved_shape.push_back(input.size(d));
      rows *= input.size(d);
    }
  }
  moved_shape.push_back(n);
  Dims rowdims;
  if (!coalesce(row_size, row_stride, rowdims) || span >= (int64_t(1) << 31) || rows * n >= (int64_t(1) << 32)) {
    on_cpu(name, "out", self, dim, dtype, out);
    return out;
  }
  at::native::resize_output(out, input.sizes());
  if (input.numel() == 0) {
    return out;
  }
  at::Tensor flat = at::empty({rows * n}, input.options().dtype(accumulator(input.scalar_type())));
  run_scan(input, rowdims, rows, n, input.stride(dim), Op, flat);
  out.copy_(flat.view(moved_shape).movedim(-1, dim));
  return out;
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("sum.IntList_out", &sum_out);
  m.impl("mean.out", &mean_out);
  m.impl("prod.int_out", &prod_out);
  m.impl("prod", &prod);
  m.impl("any.dims_out", &any_all_dims_out<kAny>);
  m.impl("all.dims_out", &any_all_dims_out<kAll>);
  m.impl("any.all_out", &any_all_out<kAny>);
  m.impl("all.all_out", &any_all_out<kAll>);
  m.impl("any.out", &any_all_dim_out<kAny>);
  m.impl("all.out", &any_all_dim_out<kAll>);
  m.impl("amax.out", &amax_amin_out<kMax>);
  m.impl("amin.out", &amax_amin_out<kMin>);
  m.impl("max", &full_max_min<kMax>);
  m.impl("min", &full_max_min<kMin>);
  m.impl("max.unary_out", &full_max_min_out<kMax>);
  m.impl("min.unary_out", &full_max_min_out<kMin>);
  m.impl("max.dim_max", &max_min_dim_out<0>);
  m.impl("min.dim_min", &max_min_dim_out<1>);
  m.impl("argmax.out", &argmax_argmin_out<0>);
  m.impl("argmin.out", &argmax_argmin_out<1>);
  m.impl("var.correction_out", &var_std_out<false>);
  m.impl("std.correction_out", &var_std_out<true>);
  m.impl("var.correction", &var_std<false>);
  m.impl("std.correction", &var_std<true>);
  m.impl("linalg_vector_norm.out", &vector_norm_out);
  m.impl("cumsum.out", &cumulative_out<0>);
  m.impl("cumprod.out", &cumulative_out<1>);
}
