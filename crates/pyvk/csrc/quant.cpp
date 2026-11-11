// Quantised matrix products for device "vk", computed as the CPU computes
// them, on the GPU:
//
//   _weight_int8pack_mm       float @ (int8 weights * per-channel scales).T
//                             (shaders/qmm8.slang: dequantised as staged)
//   _dyn_quant_pack_4bit_weight, _dyn_quant_matmul_4bit
//                             int4 weights against activations quantised
//                             per row to int8 (shaders/qmm4.slang, bit for
//                             bit as the CPU's reference kernels)
//   _scaled_mm (and .out)     float8 operands, tensor-wise or row-wise
//                             scales, as the CPU's emulated path: upcast to
//                             float32, scale, float32 product, bias, cast
//
// Arguments are checked as the CPU's kernels check them, with their messages.

#include "vk.h"

#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace vk {
namespace {

constexpr int64_t kMaxGroups = 65535;

struct Q8Params {
  uint64_t a, b, scales, c;
  uint32_t m, n, k;
  int32_t a_sm, a_sk;
  uint32_t pad;
};
static_assert(sizeof(Q8Params) == 56, "must match shaders/qmm8.slang");

struct Q4Params {
  uint64_t inp, q, rows, packed, out;
  uint32_t m, n, k, bs, has_bias, pad;
};
static_assert(sizeof(Q4Params) == 64, "must match shaders/qmm4.slang");

// Kernel suffix for the float dtypes these kernels take, or nullptr where
// the device cannot store them.
const char* float_label(at::ScalarType t) {
  if (t == at::kFloat) return "f32";
  if (t == at::kHalf || t == at::kBFloat16) return storage_label(t);
  return nullptr;
}

bool all_vk(std::initializer_list<const at::Tensor*> ts) {
  return std::all_of(ts.begin(), ts.end(), [](const at::Tensor* t) { return t == nullptr || is_vk(*t); });
}

// ------------------------------------------------------------ int8 weights

at::Tensor weight_int8pack_mm(const at::Tensor& A, const at::Tensor& B, const at::Tensor& scales) {
  // (ATen/native/LinearAlgebra.cpp _weight_int8pack_mm_cpu)
  static constexpr const char* func = "_weight_int8pack_mm_cpu";
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);
  TORCH_CHECK(A.dtype() == at::kBFloat16 || A.dtype() == at::kHalf || A.dtype() == at::kFloat, func,
              " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.dim() == 2, func, " : expect A to be 2D tensor.");
  TORCH_CHECK(A.stride(1) == 1, func, " : A must be contiguous on the last dimension.");
  TORCH_CHECK(B.dtype() == at::kChar, func, " : expect B to be int8 tensor.");
  TORCH_CHECK(B.is_contiguous(), func, " : expect B to be contiguous.");
  TORCH_CHECK(B.size(1) == K, func, " : expect B.size(1) == ", K);
  TORCH_CHECK(scales.dim() == 1 && scales.size(0) == N, func, " : expect scales to be 1d tensor with size ", N);

  const char* label = float_label(A.scalar_type());
  if (!all_vk({&A, &B, &scales}) || label == nullptr || storage_label(at::kChar) == nullptr ||
      A.stride(0) >= (int64_t(1) << 31) || B.numel() >= (int64_t(1) << 31) || M * N >= (int64_t(1) << 31)) {
    return on_cpu("aten::_weight_int8pack_mm", "", A, B, scales)[0].toTensor();
  }
  at::Tensor C = at::empty({M, N}, A.options());
  if (C.numel() == 0) {
    return C;
  }
  // (the CPU reads the scales as A's dtype)
  at::Tensor s = scales.to(A.scalar_type()).contiguous();
  int64_t item = A.element_size();
  if (M <= 8) {
    // (qmm8.slang ROWS: matrix-vector products, 8 output columns per workgroup)
    TORCH_CHECK((N + 7) / 8 <= kMaxGroups, "_weight_int8pack_mm: too many output columns for one dispatch: ", N);
    Q8Params p{};
    p.a = gpu_address(A);
    p.b = gpu_address(B);
    p.scales = gpu_address(s);
    p.c = gpu_address_out(C);
    p.m = static_cast<uint32_t>(M);
    p.n = static_cast<uint32_t>(N);
    p.k = static_cast<uint32_t>(K);
    p.a_sm = static_cast<int32_t>(A.stride(0));
    p.a_sk = static_cast<int32_t>(A.stride(1));
    dispatch(std::string("qmm8_") + label, {static_cast<uint32_t>(M)}, p, static_cast<uint32_t>((N + 7) / 8));
    return C;
  }
  int64_t gx = (N + 63) / 64;
  TORCH_CHECK(gx <= kMaxGroups, "_weight_int8pack_mm: too many output columns for one dispatch: ", N);
  for (int64_t r0 = 0; r0 < M; r0 += kMaxGroups * 64) {  // (rows in chunks of the grid's limit)
    int64_t rows = std::min<int64_t>(M - r0, kMaxGroups * 64);
    Q8Params p{};
    p.a = gpu_address(A) + static_cast<uint64_t>(r0 * A.stride(0) * item);
    p.b = gpu_address(B);
    p.scales = gpu_address(s);
    p.c = gpu_address_out(C) + static_cast<uint64_t>(r0 * N * item);
    p.m = static_cast<uint32_t>(rows);
    p.n = static_cast<uint32_t>(N);
    p.k = static_cast<uint32_t>(K);
    p.a_sm = static_cast<int32_t>(A.stride(0));
    p.a_sk = static_cast<int32_t>(A.stride(1));
    dispatch(std::string("qmm8_") + label, {}, p, static_cast<uint32_t>(gx), static_cast<uint32_t>((rows + 63) / 64));
  }
  return C;
}

// ------------------------------------------------------------ int4 weights, int8 activations

// The CPU's reference layout (ATen/native/cpu/int4mm_kernel.cpp
// dyn_quant_pack_4bit_weight_kernel without KleidiAI): a float32 vector of
// the weights' bytes, the scales and the bias, so that the CPU's packing
// works with this matmul and the other way round.
at::Tensor dyn_quant_pack_4bit_weight(const at::Tensor& weights, const at::Tensor& scales_zeros,
                                      const std::optional<at::Tensor>& bias, int64_t block_size,
                                      int64_t in_features, int64_t out_features) {
  TORCH_CHECK(weights.dtype() == at::kByte, "_dyn_quant_pack_4bit_weight_cpu", " : expect weight to be kByte.");
  TORCH_CHECK(block_size == in_features || (!(block_size % 32) && !(in_features % block_size)),
              "_dyn_quant_pack_4bit_weight_cpu", ": Group size should be multiple of 32, in_features [", in_features,
              "]. Provided ", block_size);
  if (!all_vk({&weights, &scales_zeros, bias.has_value() ? &*bias : nullptr})) {
    return on_cpu("aten::_dyn_quant_pack_4bit_weight", "", weights, scales_zeros, bias, block_size, in_features,
                  out_features)[0]
        .toTensor();
  }
  std::vector<at::Tensor> parts = {weights.reshape({-1}).to(at::kFloat), scales_zeros.reshape({-1}).to(at::kFloat)};
  if (bias.has_value()) {
    parts.push_back(bias->view({-1}).to(at::kFloat));
  }
  return at::cat(parts, 0);
}

at::Tensor dyn_quant_matmul_4bit(const at::Tensor& inp, const at::Tensor& packed_weights, int64_t block_size,
                                 int64_t in_features, int64_t out_features) {
  // (ATen/native/LinearAlgebra.cpp _dyn_quant_matmul_4bit_cpu)
  auto M = inp.size(0);
  TORCH_CHECK(inp.dtype() == at::kFloat || (inp.dtype() == at::kBFloat16 && block_size == in_features),
              "_dyn_quant_matmul_4bit_cpu", " : expect input to be float32 or bfloat16 tensor.");
  TORCH_CHECK(block_size == in_features || (!(block_size % 32) && !(in_features % block_size)),
              "_dyn_quant_matmul_4bit_cpu", ": Group size should be multiple of 32, in_features [", in_features,
              "]. Provided ", block_size);
  // (The CPU reads the input as contiguous rows of in_features values.)
  TORCH_CHECK(inp.dim() == 2 && inp.size(1) == in_features, "_dyn_quant_matmul_4bit: expected input of shape [M, ",
              in_features, "], got ", inp.sizes());
  const char* label = inp.scalar_type() == at::kFloat ? "f32" : storage_label(at::kBFloat16);
  if (!all_vk({&inp, &packed_weights}) || label == nullptr ||
      M * std::max<int64_t>(in_features, out_features) >= (int64_t(1) << 31) ||
      packed_weights.numel() >= (int64_t(1) << 31)) {
    return on_cpu("aten::_dyn_quant_matmul_4bit", "", inp, packed_weights, block_size, in_features, out_features)[0]
        .toTensor();
  }
  int64_t N = out_features, K = in_features;
  at::Tensor output = at::empty({M, N}, inp.options());
  // (int4mm_kernel.cpp dyn_quant_matmul_4bit_kernel's checks)
  const int64_t weights_elements = N * K / 2;
  const int64_t scale_elements = N * (K / block_size);
  TORCH_CHECK(packed_weights.numel() >= (weights_elements + scale_elements), "Invalid packed weight tensor size");
  int64_t bias_elements = packed_weights.numel() - (weights_elements + scale_elements);
  if (bias_elements) {
    TORCH_CHECK(bias_elements == N, "Expected bias length to match output dimension");
  }
  if (output.numel() == 0) {
    return output;
  }
  at::Tensor x = inp.contiguous();
  at::Tensor packed = packed_weights.to(at::kFloat).contiguous();
  at::Tensor q = at::empty({M, K}, inp.options().dtype(at::kInt));
  at::Tensor rows = at::empty({M, 2}, inp.options().dtype(at::kFloat));
  uint32_t groupwise = block_size != K ? 1u : 0u;
  std::string kernel = std::string("qmm4_") + label;
  int64_t gx = (N + 31) / 32;
  TORCH_CHECK(gx <= kMaxGroups, "_dyn_quant_matmul_4bit: too many output features for one dispatch: ", N);
  int64_t item = x.element_size();
  for (int64_t r0 = 0; r0 < M; r0 += kMaxGroups) {  // (rows in chunks of the grid's limit)
    int64_t rows_here = std::min<int64_t>(M - r0, kMaxGroups);
    Q4Params p{};
    p.inp = gpu_address(x) + static_cast<uint64_t>(r0 * K * item);
    p.q = gpu_address(q) + static_cast<uint64_t>(r0 * K * 4);
    p.rows = gpu_address(rows) + static_cast<uint64_t>(r0 * 2 * 4);
    p.packed = gpu_address(packed);
    p.out = gpu_address_out(output) + static_cast<uint64_t>(r0 * N * item);
    p.m = static_cast<uint32_t>(rows_here);
    p.n = static_cast<uint32_t>(N);
    p.k = static_cast<uint32_t>(K);
    p.bs = static_cast<uint32_t>(block_size);
    p.has_bias = bias_elements != 0 ? 1u : 0u;
    dispatch(kernel, {0u, groupwise}, p, static_cast<uint32_t>(rows_here));
    dispatch(kernel, {1u, groupwise}, p, static_cast<uint32_t>(gx), static_cast<uint32_t>((rows_here + 7) / 8));
  }
  return output;
}

// ------------------------------------------------------------ float8 (_scaled_mm)

// (ATen/native/ScaledBlas.cpp: invalid_scaling_config, get_joint_scaling)
[[noreturn]] void invalid_scaling_config(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& scale_a,
                                         const at::Tensor& scale_b) {
  std::stringstream ss;
  ss << "Invalid scaling configuration.\n";
  ss << "- For TensorWise scaling, a and b should be float8, scales should be float and singletons.\n";
  ss << "- For RowWise scaling, a and b should be float8, scales should be float, scale_a should be (";
  ss << mat_a.size(0) << ", 1) and scale_b should be (1, " << mat_b.size(1) << "), and both should be contiguous.\n";
  ss << "Got mat_a.dtype()=" << mat_a.scalar_type();
  ss << ", scale_a.dtype()=" << scale_a.scalar_type() << ", scale_a.size()=" << scale_a.sizes()
     << ", scale_a.stride()=" << scale_a.strides();
  ss << ", mat_b.dtype()=" << mat_b.scalar_type();
  ss << ", scale_b.dtype()=" << scale_b.scalar_type() << ", scale_b.size()=" << scale_b.sizes()
     << " and scale_b.stride()=" << scale_b.strides();
  TORCH_CHECK_VALUE(false, ss.str());
}

bool is_tensorwise(const at::Tensor& t, const at::Tensor& scale) {
  return at::isFloat8Type(t.scalar_type()) && scale.scalar_type() == at::kFloat && scale.numel() == 1;
}

bool is_rowwise(const at::Tensor& t, const at::Tensor& scale) {
  return at::isFloat8Type(t.scalar_type()) && scale.scalar_type() == at::kFloat && scale.dim() == 2 &&
         scale.size(0) == t.size(0) && scale.size(1) == 1 && scale.is_contiguous();
}

at::Tensor& scaled_mm_out(const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& scale_a,
                          const at::Tensor& scale_b, const std::optional<at::Tensor>& bias,
                          const std::optional<at::Tensor>& scale_result, std::optional<c10::ScalarType> out_dtype,
                          bool use_fast_accum, at::Tensor& out) {
  if (!all_vk({&mat1, &mat2, &out})) {
    on_cpu("aten::_scaled_mm", "out", mat1, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
    return out;
  }
  // (ScaledBlas.cpp _scaled_mm_out_cpu, without its oneDNN path for AMX CPUs)
  TORCH_CHECK_VALUE(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
  bool tensorwise = false;
  if (is_tensorwise(mat1, scale_a) && is_tensorwise(mat2.t(), scale_b.t())) {
    tensorwise = true;
  } else if (!(is_rowwise(mat1, scale_a) && is_rowwise(mat2.t(), scale_b.t()))) {
    invalid_scaling_config(mat1, mat2, scale_a, scale_b);
  }
  // (_scaled_mm_out_cpu_emulated)
  TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
  TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
  TORCH_CHECK(mat1.sizes()[1] == mat2.sizes()[0], "mat_a and mat_b shapes cannot be multiplied (", mat1.sizes()[0],
              "x", mat1.sizes()[1], " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
  if (tensorwise) {
    TORCH_CHECK_VALUE(at::isFloat8Type(mat1.scalar_type()) && at::isFloat8Type(mat2.scalar_type()),
                      "mat1 and mat2 must be fp8 types, got: ", mat1.scalar_type(), mat2.scalar_type());
    TORCH_CHECK_VALUE(scale_a.numel() == 1 && scale_a.scalar_type() == at::kFloat, "scale_a must have 1 Float element");
    TORCH_CHECK_VALUE(scale_b.numel() == 1 && scale_b.scalar_type() == at::kFloat, "scale_b must have 1 Float element");
  } else {
    TORCH_CHECK_VALUE(at::isFloat8Type(mat1.scalar_type()) && at::isFloat8Type(mat2.scalar_type()),
                      "mat1 and mat_b must be fp8 types, got: ", mat1.scalar_type(), mat2.scalar_type());
    TORCH_CHECK_VALUE(scale_a.size(0) == mat1.size(0) && scale_a.size(1) == 1, "scale_a must have shape [",
                      mat1.size(0), ", 1], got [", scale_a.sizes(), "]");
    TORCH_CHECK_VALUE(scale_a.numel() == mat1.size(0) && scale_a.scalar_type() == at::kFloat, "scale_a must have ",
                      mat1.size(0), " Float elements, got ", scale_a.numel());
    TORCH_CHECK_VALUE(scale_b.numel() == mat2.size(1) && scale_b.scalar_type() == at::kFloat, "scale_b must have ",
                      mat2.size(1), " Float elements, got ", scale_b.numel());
    TORCH_CHECK_VALUE(scale_a.stride(1) == 1, "expected scale_a.stride(1) to be 1, but got ", scale_a.stride(1));
    TORCH_CHECK_VALUE(scale_b.stride(1) == 1, "expected scale_b.stride(1) to be 1, but got ", scale_b.stride(1));
  }
  TORCH_CHECK(!scale_result || (scale_result->numel() == 1 && scale_result->scalar_type() == at::kFloat),
              "scale_result must be a float scalar");
  TORCH_CHECK(!bias || bias->numel() == mat2.sizes()[1], "Bias must be size ", mat2.sizes()[1], " but got ",
              bias->numel());
  TORCH_CHECK(at::isFloat8Type(mat1.scalar_type()), "Expected mat1 to be Float8 matrix got ", mat1.scalar_type());
  TORCH_CHECK(at::isFloat8Type(mat2.scalar_type()), "Expected mat2 to be Float8 matrix got ", mat2.scalar_type());

  at::native::resize_output(out, {mat1.size(0), mat2.size(1)});
  const auto out_dtype_ = out_dtype.value_or(c10::ScalarType::BFloat16);
  // Upcasts are exact; scale factors multiply in float32 as the CPU's
  // at::mul(t, scale.item<float>()) does (a 0-dim tensor: no host read).
  at::Tensor a = mat1.to(at::kFloat), b = mat2.to(at::kFloat);
  if (tensorwise) {
    a = at::mul(a, scale_a.reshape({}));
    b = at::mul(b, scale_b.reshape({}));
  } else {
    a = at::mul(a, scale_a);
    b = at::mul(b, scale_b);
  }
  at::Tensor out_tmp = at::matmul(a, b);
  if (bias) {
    out_tmp.add_(*bias);
  }
  if (out_dtype_ == at::ScalarType::Float8_e4m3fn || out_dtype_ == at::ScalarType::Float8_e5m2) {
    float output_scale = scale_result.has_value() ? scale_result->item<float>() : 1.0f;
    out_tmp = at::mul(out_tmp, 1 / output_scale);
  }
  out.copy_(out_tmp.to(out.scalar_type()));
  return out;
}

at::Tensor scaled_mm(const at::Tensor& mat_a, const at::Tensor& mat_b, const at::Tensor& scale_a,
                     const at::Tensor& scale_b, const std::optional<at::Tensor>& bias,
                     const std::optional<at::Tensor>& scale_result, std::optional<c10::ScalarType> out_dtype,
                     bool use_fast_accum) {
  const auto out_dtype_ = out_dtype.value_or(mat_a.scalar_type());
  at::Tensor out = at::empty({0}, mat_a.options().dtype(out_dtype_));
  return scaled_mm_out(mat_a, mat_b, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("_weight_int8pack_mm", &weight_int8pack_mm);
  m.impl("_dyn_quant_pack_4bit_weight", &dyn_quant_pack_4bit_weight);
  m.impl("_dyn_quant_matmul_4bit", &dyn_quant_matmul_4bit);
  m.impl("_scaled_mm", &scaled_mm);
  m.impl("_scaled_mm.out", &scaled_mm_out);
}
