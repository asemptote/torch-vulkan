// Batch norm and group norm backward for device "vk" (shaders/normgrad.slang):
// one kernel for the per-slice sums of grad_output and grad_output * x, one
// element-wise kernel for the input gradient, and per-channel coefficients
// on the host side (small tensors), where PyTorch's decompositions took ~40
// kernels. The arithmetic is the decompositions' (which match the CPU):
// float32 for float32 and 16-bit inputs, float64 for float64; results in
// the input's dtype (grad_input) and the weight's (grad_weight, grad_bias).
// Unusual arguments (shape mismatches, missing statistics, empty inputs)
// go to the CPU, for its checks and messages.

#include "opcodes.h"
#include "vk.h"

#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cstddef>

namespace vk {
namespace {

struct NormGradParams {
  uint64_t x, g, center, a, b, c, out, sum_g, sum_gx;
  uint32_t outer, slices, inner, ngroups;
};
static_assert(sizeof(NormGradParams) == 88 && offsetof(NormGradParams, outer) == 72,
              "must match shaders/normgrad.slang");

constexpr int64_t kMaxGroups = 65535;

bool defined(const std::optional<at::Tensor>& t) {
  return t.has_value() && t->defined();
}

// float64 for float64, float32 for float32 and the 16-bit types; Undefined
// when the kernels cannot take x.
at::ScalarType compute_type(const at::Tensor& x) {
  at::ScalarType t = x.scalar_type();
  if (t == at::kDouble) return vkt_has_float64() ? at::kDouble : at::ScalarType::Undefined;
  if (t == at::kFloat || t == at::kHalf || t == at::kBFloat16) return at::kFloat;
  return at::ScalarType::Undefined;
}

bool all_vk(std::initializer_list<const std::optional<at::Tensor>*> ts) {
  for (const auto* t : ts) {
    if (defined(*t) && (!is_vk(**t) || !at::isFloatingType((*t)->scalar_type()))) return false;
  }
  return true;
}

// x, g: [K, O, L] contiguous of the compute type. (sum_g, sum_gx) per slice.
std::pair<at::Tensor, at::Tensor> stats(const at::Tensor& x, const at::Tensor& g, const at::Tensor& center,
                                        int64_t K, int64_t O, int64_t L) {
  at::Tensor sg = at::empty({O}, x.options()), sgx = at::empty({O}, x.options());
  NormGradParams p{};
  p.x = gpu_address(x);
  p.g = gpu_address(g);
  p.center = center.defined() ? gpu_address(center) : 0;
  p.sum_g = gpu_address_out(sg);
  p.sum_gx = gpu_address_out(sgx);
  p.outer = static_cast<uint32_t>(K);
  p.slices = static_cast<uint32_t>(O);
  p.inner = static_cast<uint32_t>(L);
  int64_t groups = std::min<int64_t>(O, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  dispatch(std::string("normgrad_") + dtype_label(x.scalar_type()), {op::NORMGRAD_STATS, center.defined() ? 1u : 0u, 1u},
           p, static_cast<uint32_t>(groups));
  return {sg, sgx};
}

at::Tensor apply(uint32_t op, const at::Tensor& x, const at::Tensor& g, const at::Tensor& center, const at::Tensor& a,
                 const at::Tensor& b, const at::Tensor& c, bool train, int64_t K, int64_t O, int64_t L) {
  at::Tensor out = at::empty(x.sizes(), x.options());
  NormGradParams p{};
  p.x = gpu_address(x);
  p.g = gpu_address(g);
  p.center = center.defined() ? gpu_address(center) : 0;
  p.a = gpu_address(a);
  p.b = b.defined() ? gpu_address(b) : 0;
  p.c = c.defined() ? gpu_address(c) : 0;
  p.out = gpu_address_out(out);
  p.outer = static_cast<uint32_t>(K);
  p.slices = static_cast<uint32_t>(O);
  p.inner = static_cast<uint32_t>(L);
  int64_t groups = std::clamp<int64_t>((K * O * L + 255) / 256, 1, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  dispatch(std::string("normgrad_") + dtype_label(x.scalar_type()),
           {op, center.defined() ? 1u : 0u, train ? 1u : 0u}, p, static_cast<uint32_t>(groups));
  return out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& running_mean, const std::optional<at::Tensor>& running_var,
    const std::optional<at::Tensor>& save_mean, const std::optional<at::Tensor>& save_invstd, bool train, double eps,
    std::array<bool, 3> mask) {
  at::ScalarType t = compute_type(input);
  int64_t C = input.dim() >= 2 ? input.size(1) : 0;
  bool stats_given = train ? defined(save_mean) && defined(save_invstd) : defined(running_mean) && defined(running_var);
  std::optional<at::Tensor> g_opt = grad_out, x_opt = input;
  bool kernel = t != at::ScalarType::Undefined && input.dim() >= 2 && input.numel() > 0 &&
                input.numel() < (int64_t(1) << 31) && grad_out.sizes().equals(input.sizes()) && stats_given &&
                all_vk({&g_opt, &x_opt, &weight, &running_mean, &running_var, &save_mean, &save_invstd});
  for (const auto* s : {&weight, &running_mean, &running_var, &save_mean, &save_invstd}) {
    kernel = kernel && (!defined(*s) || (*s)->numel() == C);
  }
  if (!kernel) {
    auto s = on_cpu("aten::native_batch_norm_backward", "", grad_out, input, weight, running_mean, running_var,
                    save_mean, save_invstd, train, eps, mask);
    return {s[0].toTensor(), s[1].toTensor(), s[2].toTensor()};
  }
  int64_t K = input.size(0), L = input.numel() / (K * C);
  at::Tensor x = input.to(t).contiguous(), g = grad_out.to(t).contiguous();
  at::Tensor mean, invstd;
  if (train) {
    mean = save_mean->to(t).contiguous().view({C});
    invstd = save_invstd->to(t).contiguous().view({C});
  } else {
    mean = running_mean->to(t).contiguous().view({C});
    invstd = at::rsqrt(running_var->to(t).view({C}) + eps);
  }
  at::Tensor sum_g, dot_p;
  if (mask[1] || mask[2] || (mask[0] && train)) {
    std::tie(sum_g, dot_p) = stats(x, g, mean, K, C, L);
  }
  at::Tensor gx, gw, gb;
  if (mask[0]) {
    double norm = 1.0 / static_cast<double>(K * L);
    at::Tensor scale = defined(weight) ? invstd * weight->to(t).view({C}) : invstd;
    at::Tensor proj, gmean;
    if (train) {
      proj = (dot_p * norm) * (invstd * invstd);
      gmean = sum_g * norm;
    }
    gx = apply(op::NORMGRAD_APPLY_BN, x, g, mean, scale.contiguous(), proj, gmean, train, K, C, L)
             .to(input.scalar_type())
             .contiguous(input.suggest_memory_format());  // (the input's memory format, as the CPU's)
  }
  // (the parameters' dtype: float32 for a 16-bit input with float32 parameters)
  at::ScalarType wt = input.scalar_type();
  for (const auto* s : {&save_invstd, &save_mean, &running_var, &running_mean, &weight}) {
    if (defined(*s)) wt = (*s)->scalar_type();
  }
  if (mask[1]) gw = (dot_p * invstd).to(wt);
  if (mask[2]) gb = sum_g.to(wt);
  return {gx, gw, gb};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean, const at::Tensor& rstd,
    const std::optional<at::Tensor>& gamma, int64_t N, int64_t C, int64_t HxW, int64_t group,
    std::array<bool, 3> mask) {
  at::ScalarType t = compute_type(input);
  std::optional<at::Tensor> g_opt = grad_out, x_opt = input, m_opt = mean, r_opt = rstd;
  bool kernel = t != at::ScalarType::Undefined && group > 0 && C % group == 0 && N > 0 && C > 0 && HxW > 0 &&
                input.numel() == N * C * HxW && input.numel() < (int64_t(1) << 31) &&
                grad_out.sizes().equals(input.sizes()) && mean.numel() == N * group && rstd.numel() == N * group &&
                (!defined(gamma) || gamma->numel() == C) && all_vk({&g_opt, &x_opt, &m_opt, &r_opt, &gamma});
  if (!kernel && N == 0 && t != at::ScalarType::Undefined && group > 0 && C > 0 && HxW > 0 && C % group == 0 &&
      input.numel() == 0 && all_vk({&g_opt, &x_opt, &m_opt, &r_opt, &gamma})) {
    // (no samples: the parameter gradients are zeros, as the CPU's)
    auto popts = input.options().dtype(defined(gamma) ? gamma->scalar_type() : input.scalar_type());
    return {mask[0] ? at::empty(input.sizes(), input.options()) : at::Tensor(),
            mask[1] ? at::zeros({C}, popts) : at::Tensor(), mask[2] ? at::zeros({C}, popts) : at::Tensor()};
  }
  if (!kernel) {
    auto s = on_cpu("aten::native_group_norm_backward", "", grad_out, input, mean, rstd, gamma, N, C, HxW, group,
                    mask);
    return {s[0].toTensor(), s[1].toTensor(), s[2].toTensor()};
  }
  int64_t cpg = C / group;
  at::Tensor x = input.to(t).contiguous(), g = grad_out.to(t).contiguous();
  auto [db, ds] = stats(x, g, at::Tensor(), 1, N * C, HxW);  // [N C] each
  at::Tensor mu = mean.to(t).reshape({N, group}), rs = rstd.to(t).reshape({N, group});
  at::Tensor gm = defined(gamma) ? gamma->to(t).reshape({1, C}) : at::Tensor();
  at::Tensor gx, dgamma, dbias;
  if (mask[0]) {
    double s = 1.0 / static_cast<double>(HxW * cpg);
    at::Tensor dsv = (gm.defined() ? ds.view({N, C}) * gm : ds.view({N, C})).view({N, group, cpg}).sum(2);
    at::Tensor dbv = (gm.defined() ? db.view({N, C}) * gm : db.view({N, C})).view({N, group, cpg}).sum(2);
    at::Tensor c1 = gm.defined() ? rs.unsqueeze(-1) * gm.view({1, group, cpg}) : rs.unsqueeze(-1).expand({N, group, cpg});
    at::Tensor c2 = at::addcmul(-dsv, dbv, mu) * rs * rs * rs * s;
    at::Tensor c3 = -c2 * mu - dbv * rs * s;
    gx = apply(op::NORMGRAD_APPLY_GN, x, g, at::Tensor(), c1.contiguous().view({N * C}),
               c2.unsqueeze(-1).expand({N, group, cpg}).contiguous().view({N * C}),
               c3.unsqueeze(-1).expand({N, group, cpg}).contiguous().view({N * C}), true, 1, N * C, HxW)
             .to(input.scalar_type())
             .view(input.sizes())
             .contiguous(input.suggest_memory_format());  // (the input's memory format, as the CPU's)
  }
  at::ScalarType wt = defined(gamma) ? gamma->scalar_type() : input.scalar_type();
  if (mask[1]) {
    dgamma = ((ds.view({N, group, cpg}) - db.view({N, group, cpg}) * mu.unsqueeze(-1)) * rs.unsqueeze(-1))
                 .sum(0)
                 .reshape({C})
                 .to(wt);
  }
  if (mask[2]) dbias = db.view({N, C}).sum(0).to(wt);
  return {gx, dgamma, dbias};
}

}  // namespace

std::pair<at::Tensor, at::Tensor> slice_moments(const at::Tensor& x3);
at::Tensor slice_affine(const at::Tensor& x3, const at::Tensor& a, const at::Tensor& c);

namespace {

// Group norm forward: each (n, group)'s mean and variance (slice_moments),
// then y = x a + c per (n, c) with a = rstd w[c] and c = b[c] - mean a, as the
// CPU's kernel. Inputs the CPU would refuse go to it, for its messages.
std::tuple<at::Tensor, at::Tensor, at::Tensor> native_group_norm(const at::Tensor& input,
                                                                 const std::optional<at::Tensor>& weight,
                                                                 const std::optional<at::Tensor>& bias, int64_t N,
                                                                 int64_t C, int64_t HxW, int64_t group, double eps) {
  at::ScalarType t = compute_type(input);
  bool reduced = at::isReducedFloatingType(input.scalar_type());
  auto param_ok = [&](const std::optional<at::Tensor>& p) {
    return !defined(p) || (is_vk(*p) && p->dim() == 1 && p->numel() == C &&
                           (p->scalar_type() == input.scalar_type() || (reduced && p->scalar_type() == at::kFloat)));
  };
  bool kernel = t != at::ScalarType::Undefined && is_vk(input) && group > 0 && N > 0 && C > 0 && HxW > 0 &&
                C % group == 0 && input.numel() == N * C * HxW && input.numel() < (int64_t(1) << 31) &&
                param_ok(weight) && param_ok(bias);
  if (!kernel && N == 0 && is_vk(input) && t != at::ScalarType::Undefined && group > 0 && C > 0 && HxW > 0 &&
      C % group == 0 && input.numel() == 0 && param_ok(weight) && param_ok(bias)) {
    // (no samples: empty results, as the CPU's, its statistics in float32 only
    // for 16-bit inputs with float32 weight or bias)
    bool f32 = (defined(weight) && weight->scalar_type() == at::kFloat) || (defined(bias) && bias->scalar_type() == at::kFloat);
    at::Tensor stats = at::empty({0, group}, input.options().dtype(reduced && !f32 ? input.scalar_type() : t));
    return {at::empty(input.sizes(), input.options()), stats, stats.clone()};
  }
  if (!kernel && is_vk(input) && !at::isFloatingType(input.scalar_type())) {
    // (as the CPU's _check_group_norm_inputs, then GroupNormKernelImpl's dtype dispatch)
    TORCH_CHECK(group > 0, "Expected num groups to be greater than 0, got ", group);
    TORCH_CHECK(C % group == 0, "Expected number of channels in input to be divisible by num_groups, but got input of shape ",
                input.sizes(), " and num_groups=", group);
    for (const auto* p : {&weight, &bias}) {
      TORCH_CHECK(!defined(*p) || ((*p)->dim() == 1 && (*p)->numel() == C), "Expected ",
                  p == &weight ? "weight" : "bias", " to be a vector of size equal to the number of channels in input, but got ",
                  p == &weight ? "weight" : "bias", " of shape ", (*p)->sizes(), " and input of shape ", input.sizes());
    }
    TORCH_CHECK_NOT_IMPLEMENTED(false, "\"GroupNormKernelImpl\" not implemented for '", input.scalar_type(), "'");
  }
  if (!kernel) {
    auto s = on_cpu("aten::native_group_norm", "", input, weight, bias, N, C, HxW, group, eps);
    return {s[0].toTensor(), s[1].toTensor(), s[2].toTensor()};
  }
  bool mixed = (defined(weight) && weight->scalar_type() != input.scalar_type()) ||
               (defined(bias) && bias->scalar_type() != input.scalar_type());
  int64_t cpg = C / group;
  at::Tensor x = input.to(t).contiguous();
  auto [mean, var] = slice_moments(x.view({1, N * group, cpg * HxW}));
  at::Tensor rstd = at::sqrt(var + eps).reciprocal();  // (1 / sqrt, as the CPU)
  at::Tensor r3 = rstd.view({N, group, 1}), m3 = mean.view({N, group, 1});
  at::Tensor a = defined(weight) ? r3 * weight->to(t).view({1, group, cpg}) : r3.expand({N, group, cpg});
  at::Tensor shift = -m3 * a;
  if (defined(bias)) {
    shift = shift + bias->to(t).view({1, group, cpg});
  }
  at::Tensor y = slice_affine(x.view({1, N * C, HxW}), a.reshape({N * C}), shift.reshape({N * C}))
                     .view(input.sizes())
                     .to(input.scalar_type());
  at::MemoryFormat fmt = input.suggest_memory_format();
  if (fmt != at::MemoryFormat::Contiguous) {
    y = y.contiguous(fmt);
  }
  at::ScalarType stat = mixed ? at::kFloat : input.scalar_type();
  return {y, mean.view({N, group}).to(stat), rstd.view({N, group}).to(stat)};
}

}  // namespace

std::pair<at::Tensor, at::Tensor> slice_moments(const at::Tensor& x3) {
  int64_t K = x3.size(0), O = x3.size(1), L = x3.size(2);
  at::Tensor mean = at::empty({O}, x3.options()), var = at::empty({O}, x3.options());
  NormGradParams p{};
  p.x = gpu_address(x3);
  p.sum_g = gpu_address_out(mean);
  p.sum_gx = gpu_address_out(var);
  p.outer = static_cast<uint32_t>(K);
  p.slices = static_cast<uint32_t>(O);
  p.inner = static_cast<uint32_t>(L);
  int64_t groups = std::min<int64_t>(O, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  dispatch(std::string("normgrad_") + dtype_label(x3.scalar_type()), {op::NORMGRAD_MOMENTS, 0u, 1u}, p,
           static_cast<uint32_t>(groups));
  return {mean, var};
}

at::Tensor slice_affine(const at::Tensor& x3, const at::Tensor& a, const at::Tensor& c) {
  int64_t K = x3.size(0), O = x3.size(1), L = x3.size(2);
  at::Tensor ac = a.contiguous(), cc = c.contiguous();
  at::Tensor out = at::empty(x3.sizes(), x3.options());
  NormGradParams p{};
  p.x = gpu_address(x3);
  p.a = gpu_address(ac);
  p.c = gpu_address(cc);
  p.out = gpu_address_out(out);
  p.outer = static_cast<uint32_t>(K);
  p.slices = static_cast<uint32_t>(O);
  p.inner = static_cast<uint32_t>(L);
  int64_t groups = std::clamp<int64_t>((K * O * L + 255) / 256, 1, kMaxGroups);
  p.ngroups = static_cast<uint32_t>(groups);
  dispatch(std::string("normgrad_") + dtype_label(x3.scalar_type()), {op::NORMGRAD_AFFINE, 0u, 1u}, p,
           static_cast<uint32_t>(groups));
  return out;
}

}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_batch_norm_backward", &vk::native_batch_norm_backward);
  m.impl("native_group_norm_backward", &vk::native_group_norm_backward);
  m.impl("native_group_norm", &vk::native_group_norm);
}
