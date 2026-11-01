// Pooling kernels (shaders/pool.slang) as custom ops torch_vulkan::
// max_pool_forward / max_pool_backward / avg_pool_forward / avg_pool_backward.
// The aten kernels are
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
  TORCH_CHECK(x.dim() == 4 && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
              "max_pool_forward: expected a float32/float64 [planes, D, H, W] tensor");
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

// x [planes, D, H, W] -> [planes, *output_size]; truncate: integer inputs
// (passed as float64), whose CPU kernel divides with truncation.
at::Tensor avg_pool_forward(const at::Tensor& x, at::IntArrayRef output_size, at::IntArrayRef kernel_size,
                            at::IntArrayRef stride, at::IntArrayRef padding, int64_t divisor_override,
                            bool count_include_pad, bool truncate) {
  TORCH_CHECK(x.dim() == 4 && (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
              "avg_pool_forward: expected a float32/float64 [planes, D, H, W] tensor");
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
}

TORCH_LIBRARY_IMPL(torch_vulkan, PrivateUse1, m) {
  m.impl("max_pool_forward", &vk::max_pool_forward);
  m.impl("max_pool_backward", &vk::max_pool_backward);
  m.impl("avg_pool_forward", &vk::avg_pool_forward);
  m.impl("avg_pool_backward", &vk::avg_pool_backward);
}
