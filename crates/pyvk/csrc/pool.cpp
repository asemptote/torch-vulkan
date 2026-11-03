// Pooling, resampling and grid-sampling kernels (shaders/pool.slang,
// resample.slang, gridsample.slang) as custom ops torch_vulkan::
// max_pool_forward / max_pool_backward / avg_pool_forward / avg_pool_backward
// / upsample_linear / grid_sampler / grid_sampler_backward. The aten kernels are
// Python (torch_vulkan/_kernels.py): they run the CPU's argument checks,
// compute the output size and lay the planes out, then call these.

#include "opcodes.h"
#include "vk.h"

#include <torch/library.h>

#include <algorithm>
#include <cstddef>

namespace vk {
namespace {

struct PoolParams {
  uint64_t a, out, idx;
  uint32_t total, ngroups;
  uint32_t isz[4], osz[4], ks[4], st[4];
  int32_t pd[4];
  uint32_t dl[4];
};
static_assert(sizeof(PoolParams) == 128 && offsetof(PoolParams, isz) == 32 && offsetof(PoolParams, dl) == 112,
              "must match shaders/pool.slang");

constexpr int64_t kGroup = 256;
constexpr int64_t kMaxGroups = 65535;

// Window settings per spatial dim: [D, H, W] (3 values each).
void set_window(PoolParams& p, at::IntArrayRef in, at::IntArrayRef out, at::IntArrayRef kernel,
                at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  TORCH_CHECK(in.size() == 3 && out.size() == 3 && kernel.size() == 3 && stride.size() == 3 && padding.size() == 3 &&
                  dilation.size() == 3,
              "max_pool: expected 3 sizes per window setting");
  for (int d = 0; d < 3; ++d) {
    p.isz[d] = static_cast<uint32_t>(in[d]);
    p.osz[d] = static_cast<uint32_t>(out[d]);
    p.ks[d] = static_cast<uint32_t>(kernel[d]);
    p.st[d] = static_cast<uint32_t>(stride[d]);
    p.pd[d] = static_cast<int32_t>(padding[d]);
    p.dl[d] = static_cast<uint32_t>(dilation[d]);
  }
}

void launch(uint32_t op, at::ScalarType t, PoolParams& p, int64_t total) {
  TORCH_CHECK(total < (int64_t(1) << 32), "max_pool: more elements than 32-bit offsets address");
  p.total = static_cast<uint32_t>(total);
  int64_t groups = std::min<int64_t>((total + kGroup - 1) / kGroup, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  dispatch(std::string("pool_") + dtype_label(t), {op}, p, static_cast<uint32_t>(groups));
}

// x [planes, D, H, W] -> (values, indices) [planes, *output_size].
std::tuple<at::Tensor, at::Tensor> max_pool_forward(const at::Tensor& x, at::IntArrayRef output_size,
                                                    at::IntArrayRef kernel_size, at::IntArrayRef stride,
                                                    at::IntArrayRef padding, at::IntArrayRef dilation) {
  TORCH_CHECK(x.dim() == 4 && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble ||
                                x.scalar_type() == at::kLong),
              "max_pool_forward: expected a float32/float64/int64 [planes, D, H, W] tensor");
  at::Tensor in = x.contiguous();
  std::vector<int64_t> shape = {x.size(0), output_size[0], output_size[1], output_size[2]};
  at::Tensor out = at::empty(shape, x.options());
  at::Tensor idx = at::empty(shape, x.options().dtype(at::kLong));
  if (out.numel() == 0) {
    return {out, idx};
  }
  PoolParams p{};
  p.a = gpu_address(in);
  p.out = gpu_address_out(out);
  p.idx = gpu_address_out(idx);
  set_window(p, in.sizes().slice(1), output_size, kernel_size, stride, padding, dilation);
  launch(op::POOL_MAX_FORWARD, x.scalar_type(), p, out.numel());
  return {out, idx};
}

// grad_output, indices [planes, *output] -> grad_input [planes, *input_size].
at::Tensor max_pool_backward(const at::Tensor& grad_output, const at::Tensor& indices, at::IntArrayRef input_size,
                             at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding,
                             at::IntArrayRef dilation) {
  TORCH_CHECK(grad_output.dim() == 4 && indices.sizes() == grad_output.sizes() && indices.scalar_type() == at::kLong &&
                  (grad_output.scalar_type() == at::kFloat || grad_output.scalar_type() == at::kDouble),
              "max_pool_backward: expected float32/float64 grad_output and int64 indices [planes, D, H, W]");
  at::Tensor g = grad_output.contiguous(), i = indices.contiguous();
  at::Tensor gin = at::empty({g.size(0), input_size[0], input_size[1], input_size[2]}, g.options());
  if (gin.numel() == 0) {
    return gin;
  }
  PoolParams p{};
  p.a = gpu_address(g);
  p.out = gpu_address_out(gin);
  p.idx = gpu_address(i);
  set_window(p, input_size, g.sizes().slice(1), kernel_size, stride, padding, dilation);
  launch(op::POOL_MAX_BACKWARD, g.scalar_type(), p, gin.numel());
  return gin;
}

// Average pooling: dilation slots carry (divisor_override, count_include_pad, truncate).
void set_avg(PoolParams& p, int64_t divisor_override, bool count_include_pad, bool truncate) {
  p.dl[0] = static_cast<uint32_t>(divisor_override);
  p.dl[1] = count_include_pad ? 1u : 0u;
  p.dl[2] = truncate ? 1u : 0u;
}

// x [planes, D, H, W] -> [planes, *output_size]; integer inputs as int64,
// divided with truncation as the CPU's kernel.
at::Tensor avg_pool_forward(const at::Tensor& x, at::IntArrayRef output_size, at::IntArrayRef kernel_size,
                            at::IntArrayRef stride, at::IntArrayRef padding, int64_t divisor_override,
                            bool count_include_pad, bool truncate) {
  TORCH_CHECK(x.dim() == 4 && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble ||
                                x.scalar_type() == at::kLong),
              "avg_pool_forward: expected a float32/float64/int64 [planes, D, H, W] tensor");
  at::Tensor in = x.contiguous();
  at::Tensor out = at::empty({x.size(0), output_size[0], output_size[1], output_size[2]}, x.options());
  if (out.numel() == 0) {
    return out;
  }
  PoolParams p{};
  p.a = gpu_address(in);
  p.out = gpu_address_out(out);
  int64_t ones[] = {1, 1, 1};
  set_window(p, in.sizes().slice(1), output_size, kernel_size, stride, padding, ones);
  set_avg(p, divisor_override, count_include_pad, truncate);
  launch(op::POOL_AVG_FORWARD, x.scalar_type(), p, out.numel());
  return out;
}

// grad_output [planes, *output] -> grad_input [planes, *input_size].
at::Tensor avg_pool_backward(const at::Tensor& grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size,
                             at::IntArrayRef stride, at::IntArrayRef padding, int64_t divisor_override,
                             bool count_include_pad) {
  TORCH_CHECK(grad_output.dim() == 4 &&
                  (grad_output.scalar_type() == at::kFloat || grad_output.scalar_type() == at::kDouble),
              "avg_pool_backward: expected a float32/float64 [planes, D, H, W] grad_output");
  at::Tensor g = grad_output.contiguous();
  at::Tensor gin = at::empty({g.size(0), input_size[0], input_size[1], input_size[2]}, g.options());
  if (gin.numel() == 0) {
    return gin;
  }
  PoolParams p{};
  p.a = gpu_address(g);
  p.out = gpu_address_out(gin);
  int64_t ones[] = {1, 1, 1};
  set_window(p, input_size, g.sizes().slice(1), kernel_size, stride, padding, ones);
  set_avg(p, divisor_override, count_include_pad, false);
  launch(op::POOL_AVG_BACKWARD, g.scalar_type(), p, gin.numel());
  return gin;
}

struct ResampleParams {
  uint64_t in, out;
  uint32_t total, ngroups, align, ndim;
  uint32_t isz[4], osz[4];
  uint64_t scale[3];
  uint64_t pad[2];
};
static_assert(sizeof(ResampleParams) == 104 && offsetof(ResampleParams, scale) == 64,
              "must match shaders/resample.slang");

// x [planes, D, H, W] -> [planes, *output_size], linear interpolation per
// dim with the given scales (computed by the caller in x's dtype, as the CPU
// does); ndim: the real spatial dims (the last ndim of the three).
at::Tensor upsample_linear(const at::Tensor& x, at::IntArrayRef output_size, bool align_corners,
                           at::ArrayRef<double> scales, int64_t ndim) {
  TORCH_CHECK(x.dim() == 4 && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble) &&
                  output_size.size() == 3 && scales.size() == 3 && ndim >= 1 && ndim <= 3,
              "upsample_linear: expected a float32/float64 [planes, D, H, W] tensor and 3 sizes and scales");
  at::Tensor in = x.contiguous();
  at::Tensor out = at::empty({x.size(0), output_size[0], output_size[1], output_size[2]}, x.options());
  if (out.numel() == 0 || in.numel() == 0) {
    return out;
  }
  int64_t total = out.numel();
  TORCH_CHECK(total < (int64_t(1) << 32) && in.numel() < (int64_t(1) << 32),
              "upsample_linear: more elements than 32-bit offsets address");
  ResampleParams p{};
  p.in = gpu_address(in);
  p.out = gpu_address_out(out);
  p.total = static_cast<uint32_t>(total);
  int64_t groups = std::min<int64_t>((total + kGroup - 1) / kGroup, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  p.align = align_corners ? 1u : 0u;
  p.ndim = static_cast<uint32_t>(ndim);
  for (int d = 0; d < 3; ++d) {
    p.isz[d] = static_cast<uint32_t>(in.size(1 + d));
    p.osz[d] = static_cast<uint32_t>(output_size[d]);
    p.scale[d] = pack_param(scales[d], x.scalar_type());
  }
  dispatch(std::string("resample_") + dtype_label(x.scalar_type()), {op::RESAMPLE_FORWARD}, p,
           static_cast<uint32_t>(groups));
  return out;
}

// Backward of linear interpolation along one dim: g [A, n_out, B] -> [A,
// n_in, B], with the forward's scale (shaders/resample.slang, RESAMPLE_BACKWARD).
at::Tensor upsample_linear_backward_dim(const at::Tensor& g, int64_t n_in, bool align_corners, double scale) {
  TORCH_CHECK(g.dim() == 3 && (g.scalar_type() == at::kFloat || g.scalar_type() == at::kDouble) && n_in > 0,
              "upsample_linear_backward_dim: expected a float32/float64 [A, n_out, B] tensor");
  at::Tensor in = g.contiguous();
  at::Tensor out = at::empty({g.size(0), n_in, g.size(2)}, g.options());
  if (out.numel() == 0) {
    return out;
  }
  if (g.size(1) == 0) {
    return out.zero_();
  }
  TORCH_CHECK(out.numel() < (int64_t(1) << 32) && in.numel() < (int64_t(1) << 32),
              "upsample_linear_backward_dim: more elements than 32-bit offsets address");
  ResampleParams p{};
  p.in = gpu_address(in);
  p.out = gpu_address_out(out);
  p.total = static_cast<uint32_t>(out.numel());
  int64_t groups = std::min<int64_t>((out.numel() + kGroup - 1) / kGroup, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  p.align = align_corners ? 1u : 0u;
  p.ndim = 1;
  p.isz[0] = static_cast<uint32_t>(g.size(0));
  p.isz[1] = static_cast<uint32_t>(n_in);
  p.isz[2] = static_cast<uint32_t>(g.size(2));
  p.osz[1] = static_cast<uint32_t>(g.size(1));
  p.scale[0] = pack_param(scale, g.scalar_type());
  dispatch(std::string("resample_") + dtype_label(g.scalar_type()), {op::RESAMPLE_BACKWARD}, p,
           static_cast<uint32_t>(groups));
  return out;
}

struct GridParams {
  uint64_t input, grid, out, gout, ggrid, idx, wgt;
  uint32_t n, c, p, k, padding, align, ngroups, want_grid, pad[2];
  uint32_t isz[4];
};
static_assert(sizeof(GridParams) == 112 && offsetof(GridParams, isz) == 96, "must match shaders/gridsample.slang");

// input [N, C, *S] and grid [N, *P, k] (k = 2 or 3) of one float dtype:
// the parameters common to both directions; P the output points per batch.
GridParams grid_params(const at::Tensor& input, const at::Tensor& grid, int64_t padding, bool align) {
  int64_t k = grid.size(-1);
  TORCH_CHECK((k == 2 || k == 3) && input.dim() == k + 2 && grid.dim() == k + 2 &&
                  (input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble) &&
                  grid.scalar_type() == input.scalar_type(),
              "grid_sampler: expected float32/float64 [N, C, *S] input and [N, *P, k] grid of its dtype");
  GridParams p{};
  p.n = static_cast<uint32_t>(input.size(0));
  p.c = static_cast<uint32_t>(input.size(1));
  p.p = static_cast<uint32_t>(grid.numel() / std::max<int64_t>(grid.size(0) * k, 1));
  p.k = static_cast<uint32_t>(k);
  p.padding = static_cast<uint32_t>(padding);
  p.align = align ? 1u : 0u;
  for (int64_t d = 0; d < 3; ++d) {  // (by grid coordinate: W, H, D)
    p.isz[d] = d < k ? static_cast<uint32_t>(input.size(input.dim() - 1 - d)) : 1u;
  }
  TORCH_CHECK(input.numel() < (int64_t(1) << 31) && grid.numel() < (int64_t(1) << 31) &&
                  int64_t(p.c) * p.n * p.p < (int64_t(1) << 31),
              "grid_sampler: more elements than 32-bit offsets address");
  return p;
}

void launch_grid(uint32_t op, int64_t mode, GridParams& p, at::ScalarType t) {
  int64_t threads = int64_t(p.n) * p.p;
  int64_t groups = std::min<int64_t>((threads + 63) / 64, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(std::max<int64_t>(groups, 1));
  dispatch(std::string("gridsample_") + dtype_label(t), {op, static_cast<uint32_t>(mode)}, p, p.ngroups);
}

// [N, C, *P]: bilinear (mode 0) or nearest (1) samples of input at grid.
at::Tensor grid_sampler(const at::Tensor& input, const at::Tensor& grid, int64_t mode, int64_t padding,
                        bool align_corners) {
  at::Tensor in = input.contiguous(), g = grid.contiguous();
  GridParams p = grid_params(in, g, padding, align_corners);
  std::vector<int64_t> shape = {in.size(0), in.size(1)};
  shape.insert(shape.end(), g.sizes().begin() + 1, g.sizes().end() - 1);
  at::Tensor out = at::empty(shape, in.options());
  if (out.numel() == 0) {
    return out;
  }
  p.input = gpu_address(in);
  p.grid = gpu_address(g);
  p.out = gpu_address_out(out);
  launch_grid(op::GRID_FORWARD, mode, p, in.scalar_type());
  return out;
}

// (grad_grid [N, *P, k] (empty unless want_grid), corner offsets [N, P, 2^k]
// (-1: outside the input), corner weights [N, P, 2^k]).
std::tuple<at::Tensor, at::Tensor, at::Tensor> grid_sampler_backward(const at::Tensor& grad_output,
                                                                     const at::Tensor& input, const at::Tensor& grid,
                                                                     int64_t mode, int64_t padding,
                                                                     bool align_corners, bool want_grid) {
  at::Tensor in = input.contiguous(), g = grid.contiguous(), go = grad_output.contiguous();
  GridParams p = grid_params(in, g, padding, align_corners);
  int64_t corners = int64_t(1) << p.k;
  at::Tensor ggrid = want_grid ? at::empty(g.sizes(), g.options()) : at::empty({0}, g.options());
  at::Tensor idx = at::empty({int64_t(p.n), int64_t(p.p), corners}, g.options().dtype(at::kLong));
  at::Tensor wgt = at::empty({int64_t(p.n), int64_t(p.p), corners}, g.options());
  if (idx.numel() == 0) {
    return {ggrid.zero_(), idx, wgt};
  }
  p.input = gpu_address(in);
  p.grid = gpu_address(g);
  p.gout = gpu_address(go);
  p.ggrid = want_grid ? gpu_address_out(ggrid) : 0;
  p.idx = gpu_address_out(idx);
  p.wgt = gpu_address_out(wgt);
  p.want_grid = want_grid ? 1u : 0u;
  launch_grid(op::GRID_BACKWARD, mode, p, in.scalar_type());
  return {ggrid, idx, wgt};
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY(torch_vulkan, m) {
  m.def("max_pool_forward(Tensor x, int[] output_size, int[] kernel_size, int[] stride, int[] padding, "
        "int[] dilation) -> (Tensor, Tensor)");
  m.def("max_pool_backward(Tensor grad_output, Tensor indices, int[] input_size, int[] kernel_size, int[] stride, "
        "int[] padding, int[] dilation) -> Tensor");
  m.def("avg_pool_forward(Tensor x, int[] output_size, int[] kernel_size, int[] stride, int[] padding, "
        "int divisor_override, bool count_include_pad, bool truncate) -> Tensor");
  m.def("avg_pool_backward(Tensor grad_output, int[] input_size, int[] kernel_size, int[] stride, int[] padding, "
        "int divisor_override, bool count_include_pad) -> Tensor");
  m.def("upsample_linear(Tensor x, int[] output_size, bool align_corners, float[] scales, int ndim) -> Tensor");
  m.def("upsample_linear_backward_dim(Tensor g, int n_in, bool align_corners, float scale) -> Tensor");
  m.def("grid_sampler(Tensor input, Tensor grid, int mode, int padding, bool align_corners) -> Tensor");
  m.def("grid_sampler_backward(Tensor grad_output, Tensor input, Tensor grid, int mode, int padding, "
        "bool align_corners, bool want_grid) -> (Tensor, Tensor, Tensor)");
  // (indexing.cpp)
  m.def("sum_rows_by_key(Tensor keys, Tensor vals, int num_rows) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_vulkan, PrivateUse1, m) {
  m.impl("max_pool_forward", &vk::max_pool_forward);
  m.impl("max_pool_backward", &vk::max_pool_backward);
  m.impl("avg_pool_forward", &vk::avg_pool_forward);
  m.impl("avg_pool_backward", &vk::avg_pool_backward);
  m.impl("upsample_linear", &vk::upsample_linear);
  m.impl("upsample_linear_backward_dim", &vk::upsample_linear_backward_dim);
  m.impl("grid_sampler", &vk::grid_sampler);
  m.impl("grid_sampler_backward", &vk::grid_sampler_backward);
}
