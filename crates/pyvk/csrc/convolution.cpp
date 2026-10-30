// Convolution for device "vk" (M12): convolution_overrideable and
// convolution_backward_overrideable (what aten::convolution dispatches to on
// a PrivateUse1 backend), for 1-3 spatial dims, groups, stride, padding,
// dilation and transposed convolution.
//
// Built from existing kernels:
//  - forward: im2col as a strided view of the padded input (one copy kernel)
//    and a batched GEMM against the weight;
//  - transposed: an ordinary convolution of the zero-inserted input with the
//    flipped, transposed weight (no accumulation into overlapping windows);
//  - backward: grad_input is a transposed convolution, grad_weight a GEMM of
//    grad_output against the same columns, grad_bias a sum.

#include "vk.h"

#include <torch/library.h>

namespace vk {

namespace {

// Output length of one spatial dim of a (non-transposed) convolution.
int64_t conv_out_size(int64_t in, int64_t k, int64_t stride, int64_t pad_lo, int64_t pad_hi, int64_t dilation) {
  return (in + pad_lo + pad_hi - dilation * (k - 1) - 1) / stride + 1;
}

// im2col: [N, prod(O), C * prod(K)] columns of x (already padded) for kernel
// sizes k, stride, dilation, as one copy of a strided view.
at::Tensor im2col_view(const at::Tensor& xp, at::IntArrayRef k, at::IntArrayRef stride, at::IntArrayRef dilation,
                       at::IntArrayRef out) {
  int64_t s = static_cast<int64_t>(k.size());
  std::vector<int64_t> sizes = {xp.size(0)}, strides = {xp.stride(0)};
  for (int64_t d = 0; d < s; ++d) {
    sizes.push_back(out[d]);
    strides.push_back(xp.stride(2 + d) * stride[d]);
  }
  sizes.push_back(xp.size(1));
  strides.push_back(xp.stride(1));
  for (int64_t d = 0; d < s; ++d) {
    sizes.push_back(k[d]);
    strides.push_back(xp.stride(2 + d) * dilation[d]);
  }
  at::Tensor windows = xp.as_strided(sizes, strides, xp.storage_offset());  // [N, *O, C, *K]
  int64_t L = 1, CK = xp.size(1);
  for (int64_t d = 0; d < s; ++d) {
    L *= out[d];
    CK *= k[d];
  }
  return windows.reshape({xp.size(0), L, CK});
}

// Zero-pads the spatial dims of x: pad[d] = (lo, hi), negative values crop.
at::Tensor pad_spatial(const at::Tensor& x, const std::vector<std::pair<int64_t, int64_t>>& pad) {
  std::vector<int64_t> p;  // constant_pad_nd order: last dim first
  bool any = false;
  for (int64_t d = static_cast<int64_t>(pad.size()) - 1; d >= 0; --d) {
    p.push_back(pad[d].first);
    p.push_back(pad[d].second);
    any = any || pad[d].first != 0 || pad[d].second != 0;
  }
  return any ? at::constant_pad_nd(x, p, 0) : x;
}

// Plain convolution: x [N, C, *S], w [Co, C/G, *K], symmetric or asymmetric
// zero padding pad[d] = (lo, hi).
at::Tensor conv_forward(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                        at::IntArrayRef stride, const std::vector<std::pair<int64_t, int64_t>>& pad,
                        at::IntArrayRef dilation, int64_t groups) {
  int64_t s = x.dim() - 2, N = x.size(0), C = x.size(1), Co = w.size(0);
  std::vector<int64_t> k(w.sizes().begin() + 2, w.sizes().end()), out;
  for (int64_t d = 0; d < s; ++d) {
    out.push_back(conv_out_size(x.size(2 + d), k[d], stride[d], pad[d].first, pad[d].second, dilation[d]));
  }
  std::vector<int64_t> out_shape = {N, Co};
  out_shape.insert(out_shape.end(), out.begin(), out.end());
  int64_t L = 1;
  for (int64_t o : out) L *= o;
  if (N == 0 || Co == 0 || L == 0) {
    return at::zeros(out_shape, x.options());
  }
  at::Tensor xp = pad_spatial(x, pad);
  int64_t Kp = 1;
  for (int64_t v : k) Kp *= v;
  int64_t Cg = C / groups, Cog = Co / groups;
  at::Tensor cols = im2col_view(xp, k, stride, dilation, out);       // [N, L, C*K] (c-major, then k)
  cols = cols.view({N, L, groups, Cg * Kp}).permute({2, 0, 1, 3});   // [G, N, L, Cg*K]
  cols = cols.reshape({groups, N * L, Cg * Kp});
  at::Tensor wg = w.reshape({groups, Cog, Cg * Kp}).transpose(1, 2);  // [G, Cg*K, Cog]
  at::Tensor y = at::bmm(cols, wg);                                   // [G, N*L, Cog]
  y = y.view({groups, N, L, Cog}).permute({1, 0, 3, 2}).reshape({N, Co, L});
  if (bias.has_value() && bias->defined()) {
    y = y + bias->view({1, Co, 1});
  }
  return y.reshape(out_shape);
}

// Transposed convolution: x [N, Ci, *S], w [Ci, Co/G, *K].
at::Tensor conv_transpose_forward(const at::Tensor& x, const at::Tensor& w, const std::optional<at::Tensor>& bias,
                                  at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding,
                                  at::IntArrayRef dilation, int64_t groups) {
  int64_t s = x.dim() - 2, N = x.size(0), Ci = x.size(1), Cog = w.size(1), Co = Cog * groups;
  // Insert stride - 1 zeros between input elements.
  std::vector<int64_t> dil_shape = {N, Ci};
  for (int64_t d = 0; d < s; ++d) {
    dil_shape.push_back(x.size(2 + d) == 0 ? 0 : (x.size(2 + d) - 1) * stride[d] + 1);
  }
  at::Tensor xd = x;
  bool strided = false;
  for (int64_t d = 0; d < s; ++d) strided = strided || stride[d] > 1;
  if (strided) {
    xd = at::zeros(dil_shape, x.options());
    at::Tensor view = xd;
    for (int64_t d = 0; d < s; ++d) {
      view = view.slice(2 + d, 0, dil_shape[2 + d], stride[d]);
    }
    view.copy_(x);
  }
  // The equivalent convolution: padding d (k - 1) - p (output_padding more at the end), flipped kernel.
  std::vector<std::pair<int64_t, int64_t>> pad;
  std::vector<int64_t> flip_dims, ones(s, 1);
  for (int64_t d = 0; d < s; ++d) {
    int64_t full = dilation[d] * (w.size(2 + d) - 1);
    pad.push_back({full - padding[d], full - padding[d] + output_padding[d]});
    flip_dims.push_back(2 + d);
  }
  // [Ci, Co/G, *K] -> per group [Co/G, Ci/G, *K], flipped.
  int64_t Cig = Ci / groups;
  std::vector<int64_t> gshape = {groups, Cig, Cog};
  gshape.insert(gshape.end(), w.sizes().begin() + 2, w.sizes().end());
  at::Tensor wt = w.reshape(gshape).transpose(1, 2);
  std::vector<int64_t> wshape = {Co, Cig};
  wshape.insert(wshape.end(), w.sizes().begin() + 2, w.sizes().end());
  wt = wt.reshape(wshape).flip(flip_dims);
  return conv_forward(xd, wt, bias, ones, pad, dilation, groups);
}

std::vector<std::pair<int64_t, int64_t>> symmetric(at::IntArrayRef padding) {
  std::vector<std::pair<int64_t, int64_t>> pad;
  for (int64_t p : padding) pad.push_back({p, p});
  return pad;
}

// Expands 1-element parameter lists to one per spatial dim.
std::vector<int64_t> per_dim(at::IntArrayRef v, int64_t s) {
  if (static_cast<int64_t>(v.size()) == s) return v.vec();
  TORCH_CHECK(v.size() == 1, "torch_vulkan: convolution parameter has ", v.size(), " values for ", s, " dims");
  return std::vector<int64_t>(s, v[0]);
}

bool conv_ok(const at::Tensor& x, const at::Tensor& w) {
  at::ScalarType t = x.scalar_type();
  return is_vk(x) && is_vk(w) && x.dim() >= 3 && x.dim() <= 5 && w.dim() == x.dim() &&
         (t == at::kFloat || (t == at::kDouble && has_feature(kFeatureFloat64))) && w.scalar_type() == t;
}

at::Tensor convolution(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias,
                       at::IntArrayRef stride_, at::IntArrayRef padding_, at::IntArrayRef dilation_, bool transposed,
                       at::IntArrayRef output_padding_, int64_t groups) {
  if (!conv_ok(input, weight)) {
    // (convolution_overrideable has no CPU kernel; aten::convolution does.)
    return on_cpu("aten::convolution", "", input, weight, bias, stride_, padding_, dilation_, transposed,
                  output_padding_, groups)[0]
        .toTensor();
  }
  int64_t s = input.dim() - 2;
  auto stride = per_dim(stride_, s), padding = per_dim(padding_, s), dilation = per_dim(dilation_, s);
  auto output_padding = per_dim(output_padding_, s);
  if (transposed) {
    return conv_transpose_forward(input, weight, bias, stride, padding, output_padding, dilation, groups);
  }
  return conv_forward(input, weight, bias, stride, symmetric(padding), dilation, groups);
}

// grad_weight of a plain convolution: [Co, C/G, *K] = sum over N, L of
// grad_output x columns.
at::Tensor conv_weight_grad(const at::Tensor& x, const at::Tensor& g, at::IntArrayRef wsize, at::IntArrayRef stride,
                            const std::vector<std::pair<int64_t, int64_t>>& pad, at::IntArrayRef dilation,
                            int64_t groups) {
  int64_t s = x.dim() - 2, N = x.size(0), C = x.size(1), Co = g.size(1);
  std::vector<int64_t> k(wsize.begin() + 2, wsize.end()), out(g.sizes().begin() + 2, g.sizes().end());
  int64_t L = 1, Kp = 1;
  for (int64_t o : out) L *= o;
  for (int64_t v : k) Kp *= v;
  int64_t Cg = C / groups, Cog = Co / groups;
  if (N == 0 || L == 0) {
    return at::zeros(wsize, x.options());
  }
  at::Tensor cols = im2col_view(pad_spatial(x, pad), k, stride, dilation, out);      // [N, L, C*K]
  cols = cols.view({N * L, groups, Cg * Kp}).permute({1, 0, 2});                     // [G, N*L, Cg*K]
  at::Tensor gg = g.reshape({N, groups, Cog, L}).permute({1, 2, 0, 3}).reshape({groups, Cog, N * L});
  return at::bmm(gg, cols).reshape(wsize);                                            // [G, Cog, Cg*K]
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> convolution_backward(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, at::IntArrayRef stride_,
    at::IntArrayRef padding_, at::IntArrayRef dilation_, bool transposed, at::IntArrayRef output_padding_,
    int64_t groups, std::array<bool, 3> mask) {
  if (!conv_ok(input, weight) || !is_vk(grad_output)) {
    std::vector<int64_t> bias_sizes = {transposed ? weight.size(1) * groups : weight.size(0)};
    auto stack = on_cpu("aten::convolution_backward", "", grad_output, input, weight, bias_sizes, stride_, padding_,
                        dilation_, transposed, output_padding_, groups, mask);
    return {stack[0].toTensor(), stack[1].toTensor(), stack[2].toTensor()};
  }
  int64_t s = input.dim() - 2;
  auto stride = per_dim(stride_, s), padding = per_dim(padding_, s), dilation = per_dim(dilation_, s);
  auto output_padding = per_dim(output_padding_, s);
  at::Tensor g = grad_output.contiguous();
  at::Tensor grad_input, grad_weight, grad_bias;
  if (!transposed) {
    if (mask[0]) {
      // The adjoint of a strided convolution is the transposed convolution;
      // output_padding recovers input sizes the stride rounded away.
      std::vector<int64_t> op;
      for (int64_t d = 0; d < s; ++d) {
        int64_t full = (g.size(2 + d) - 1) * stride[d] - 2 * padding[d] + dilation[d] * (weight.size(2 + d) - 1) + 1;
        op.push_back(input.size(2 + d) - full);
      }
      grad_input = conv_transpose_forward(g, weight, std::nullopt, stride, padding, op, dilation, groups);
    }
    if (mask[1]) {
      grad_weight = conv_weight_grad(input, g, weight.sizes(), stride, symmetric(padding), dilation, groups);
    }
  } else {
    if (mask[0]) {
      // The adjoint of a transposed convolution is the plain convolution.
      grad_input = conv_forward(g, weight, std::nullopt, stride, symmetric(padding), dilation, groups);
    }
    if (mask[1]) {
      // y = A_W^T x for the plain convolution A_W using W [Ci, Co/G, *K] as
      // is (Co channels in, Ci out), so dL/dW is A_W's weight gradient with
      // grad_output as its input and `input` as its output gradient.
      grad_weight = conv_weight_grad(g, input, weight.sizes(), stride, symmetric(padding), dilation, groups);
    }
  }
  if (mask[2]) {
    std::vector<int64_t> dims = {0};
    for (int64_t d = 0; d < s; ++d) dims.push_back(2 + d);
    grad_bias = g.sum(dims);
  }
  return {grad_input, grad_weight, grad_bias};
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("convolution_overrideable", &vk::convolution);
  m.impl("convolution_backward_overrideable", &vk::convolution_backward);
}
