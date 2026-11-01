// Shared declarations for the C++ PyTorch backend of torch_vulkan.
//
// The Rust runtime (crates/pyvk/src/ffi.rs) owns the Vulkan device, memory
// pool, kernel cache and command stream; this side implements PyTorch's
// backend interfaces (allocator, device guard, hooks) and aten kernels on the
// PrivateUse1 dispatch key, renamed to "vk" by the Python package.
#pragma once

#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>

#include <array>
#include <initializer_list>
#include <cstddef>
#include <cstdint>
#include <string>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>

#include <algorithm>
#include <vector>

extern "C" {
struct VktAlloc {
  void* handle;
  uint8_t* host;
  uint64_t address;
  uint64_t size;
};
const char* vkt_last_error();
int vkt_alloc(uint64_t nbytes, VktAlloc* out);
void vkt_free(void* handle);
int vkt_synchronize();
int vkt_dispatch(const char* name, const uint32_t* spec, size_t nspec, const uint8_t* push, size_t npush, uint32_t gx,
                 uint32_t gy, uint32_t gz);
int vkt_grid_1d(uint64_t n, uint32_t* groups, uint32_t* stride);
int vkt_has_float64();
uint32_t vkt_capabilities();
}

namespace vk {

constexpr c10::DeviceType kDevice = c10::DeviceType::PrivateUse1;

inline void check(int rc) {
  TORCH_CHECK(rc == 0, "torch_vulkan: ", vkt_last_error());
}

// Waits for queued GPU work; raises IndexError if an indexing kernel met an
// out-of-range index since the last check (backend.cpp).
void synchronize();

// GPU address of the 4-byte device error word that indexing kernels set on
// an out-of-range index (backend.cpp).
uint64_t device_error_address();

// The bytes of `value` converted to `dtype`, as kernel parameter slots
// (backend.cpp).
std::array<uint64_t, 2> fill_bits(const at::Scalar& value, at::ScalarType dtype);

// Optional device features (bits of vkt_capabilities).
enum Feature : uint32_t { kFeatureFloat64 = 1, kFeatureInt16 = 2, kFeatureFloat16 = 4, kFeatureInt8 = 8 };

inline bool has_feature(Feature f) {
  static const uint32_t caps = vkt_capabilities();
  return (caps & f) != 0;
}

// Per-allocation context stored in each DataPtr: data() is `host + offset`
// for views, so a tensor's GPU address is `address + (data - host)`.
struct AllocCtx {
  void* handle;
  uint8_t* host;
  uint64_t address;
  uint64_t size;
};

// GPU address of the first element of `t` (0 for empty storage).
uint64_t gpu_address(const at::TensorBase& t);
uint64_t gpu_address(const at::TensorBase& t, const void* data);
// The same for a tensor a kernel writes: materializes copy-on-write storage
// first, so the write doesn't reach the tensors sharing it.
uint64_t gpu_address_out(const at::TensorBase& t);

c10::Allocator* allocator();

// Kernel name suffix for a dtype with compute kernels ("f32", "f64", "i32",
// "i64", "b8", "c64", "c128"), or nullptr.
const char* dtype_label(at::ScalarType t);

// Kernel name suffix for a dtype that cast kernels can read and write: the
// dtype_label types plus "u8", "i8", "i16", "f16", "bf16" where the device
// supports them; nullptr otherwise.
const char* storage_label(at::ScalarType t);

// The dtype element-wise kernels compute `t` in: 8- and 16-bit integers in
// int32, float16 and bfloat16 in float32 (then cast back, which wraps
// integers and rounds to nearest even, as on the CPU); others unchanged.
at::ScalarType widened(at::ScalarType t);

// The data-movement kernel (shaders/move.slang) for elements of `itemsize`
// bytes, or "" if the device cannot address elements of that size.
std::string move_kernel(int64_t itemsize);

// Records that `op` ran on the CPU instead of the GPU (warns once per op).
void record_fallback(const std::string& op);

// Before running an op on the CPU for "vk" tensors, the device checks of
// PyTorch's own kernels (the fallback would otherwise compute on the CPU
// and succeed): out= tensors must be on "vk" when an input is, and "vk" and
// CPU tensors do not mix (except 0-dim CPU tensors, i.e. scalars, and the
// indices of indexing ops).
void check_out_devices(const c10::FunctionSchema& schema, const torch::jit::Stack& stack);

// out (vk) = result computed on the CPU by `compute`, recorded as a fallback.
template <typename F>
void cpu_out(const char* op, const at::Tensor& out, F&& compute) {
  record_fallback(op);
  at::Tensor result = compute();
  out.copy_(result);
}

// Strided element-wise launch (shaders/_strided.slang): `out` plus up to two
// inputs (nullptr = absent) broadcast to out's shape. Dims that are
// contiguous in every operand are merged; dims beyond 4 are looped over on
// the host, one dispatch per outer index. `params` fill the kernel's alpha,
// beta, gamma slots, converted to its element type T (a's dtype, else
// out's): int64 for integer/bool kernels, double for float64/complex128,
// float otherwise. launch.cpp.
void launch_strided(const char* kernel, std::vector<uint32_t> spec, const at::Tensor& out, const at::Tensor* a,
                    const at::Tensor* b, c10::ArrayRef<c10::Scalar> params = {});

// Four-operand variant (shaders/_strided4.slang): out, a, b, c (any may be
// nullptr except out), up to 3 merged dims per dispatch, parameters alpha
// and beta converted to out's dtype. launch.cpp.
void launch_strided4(const char* kernel, std::vector<uint32_t> spec, const at::Tensor& out, const at::Tensor* a,
                     const at::Tensor* b, const at::Tensor* c, c10::ArrayRef<c10::Scalar> params = {});

// The device's default random generator and fresh generators (Philox;
// random.cpp).
const at::Generator& default_generator();
at::Generator new_generator();

// Stable sort of each row of `x` (contiguous [rows, n], any real dtype with
// a storage_label): the int32 indices [rows, n] that order each row
// ascending (descending: largest first), ties in original order. sorting.cpp.
at::Tensor sort_rows(const at::Tensor& x, bool descending);

// A parameter slot for a kernel whose element type T is `t`: an int64 for
// integer and bool kernels, double bits for float64/complex128, float bits
// (in the low word) otherwise, so that only float64 kernels need a double
// (GPUs without shaderFloat64 run the rest). launch.cpp.
uint64_t pack_param(const c10::Scalar& s, at::ScalarType t);

// launch_strided with the parameter slots given as raw bits (e.g. a fill
// value's bytes).
void launch_strided_bits(const char* kernel, std::vector<uint32_t> spec, const at::Tensor& out, const at::Tensor* a,
                         const at::Tensor* b, c10::ArrayRef<uint64_t> bits);

// Runs binary op `code` (shaders/binary.slang) with PyTorch's type promotion;
// false if it cannot run on the GPU. ops.cpp.
bool binary_op_gpu(int64_t code, const at::Tensor& self, const at::Tensor* other, const at::Scalar* other_scalar,
                   const at::Tensor& out);

inline at::Tensor to_dtype(const at::Tensor& t, at::ScalarType dtype) {
  return t.scalar_type() == dtype ? t : t.to(dtype);
}

// Output layouts as PyTorch's own kernels give them: functional kernels here
// ask the op's meta kernel (shapes and strides only, no data) and lay their
// results out the same way, so strides agree with the CPU and with
// FakeTensor (e.g. column-major linalg outputs, FFT outputs that keep the
// input's dim order).

// A meta tensor with t's sizes, strides and dtype.
inline at::Tensor meta_of(const at::Tensor& t) {
  return at::empty_strided(t.sizes(), t.strides(), t.options().device(at::kMeta));
}

inline std::optional<at::Tensor> meta_of(const std::optional<at::Tensor>& t) {
  return t.has_value() && t->defined() ? std::optional<at::Tensor>(meta_of(*t)) : std::nullopt;
}

// An empty vk tensor shaped and strided like the meta tensor `layout`.
inline at::Tensor empty_like_meta(const at::Tensor& layout, const at::TensorOptions& vk_options) {
  return at::empty_strided(layout.sizes(), layout.strides(), vk_options.dtype(layout.scalar_type()));
}

// `result` laid out like the meta tensor `layout` (result itself if it already is).
inline at::Tensor laid_out_like(const at::Tensor& result, const at::Tensor& layout) {
  if (result.sizes() != layout.sizes() || result.strides() == layout.strides()) {
    return result;
  }
  at::Tensor out = empty_like_meta(layout, result.options());
  out.copy_(result);
  return out;
}

// Resizes an element-wise op's output as PyTorch's TensorIterator does: an
// output that must be resized (e.g. a fresh empty one) takes the dim order
// of `like` (an input of the output's shape, if any), contiguous in that
// order, so ops on transposed or channels-last inputs keep their layout.
// The CPU's (TensorIterator's) aliasing checks for an op writing out from
// inputs: no element of out written twice, no input partly overlapping it
// (an input identical to out is fine).
inline void check_overlap(const at::Tensor& out, std::initializer_list<const at::Tensor*> inputs = {}) {
  at::assert_no_internal_overlap(out);
  for (const at::Tensor* input : inputs) {
    if (input != nullptr && input->defined()) {
      at::assert_no_partial_overlap(out, *input);
    }
  }
}

// An output that is also an input (in place) cannot grow to the broadcast
// shape, as in PyTorch's TensorIterator.
inline void check_inplace_shape(const at::Tensor& out, at::IntArrayRef shape,
                                std::initializer_list<const at::Tensor*> inputs) {
  if (out.sizes() == shape) return;
  for (const at::Tensor* input : inputs) {
    TORCH_CHECK(input == nullptr || !input->defined() || !out.is_same(*input), "output with shape ", out.sizes(),
                " doesn't match the broadcast shape ", shape);
  }
}

inline void resize_elementwise_output(const at::Tensor& out, at::IntArrayRef shape, const at::Tensor* like,
                                      bool check_aliasing = true) {
  if (check_aliasing) {
    check_overlap(out, {like});
  }
  if (out.sizes() == shape) {
    return;
  }
  at::native::resize_output(out, shape);
  if (like == nullptr || like->sizes() != shape || like->is_contiguous() || shape.size() < 2) {
    return;
  }
  int64_t n = static_cast<int64_t>(shape.size());
  std::vector<int64_t> order(n);  // dims from outermost (largest stride) to innermost
  for (int64_t d = 0; d < n; ++d) order[d] = d;
  std::stable_sort(order.begin(), order.end(),
                   [&](int64_t a, int64_t b) { return like->stride(a) > like->stride(b); });
  std::vector<int64_t> strides(n);
  int64_t step = 1;
  for (int64_t i = n - 1; i >= 0; --i) {
    strides[order[i]] = step;
    step *= std::max<int64_t>(shape[order[i]], 1);
  }
  out.as_strided_(shape, strides);
}

// "vk" rejects the dtypes PyTorch's CPU kernel of an op rejects (with the
// CPU's kind of message), so code behaves the same on both devices.
inline void reject_dtype(const char* op, at::ScalarType t, bool rejected) {
  TORCH_CHECK(!rejected, "\"", op, "\" not implemented for '", c10::toString(t), "'");
}

inline bool is_half(at::ScalarType t) {
  return t == at::kHalf || t == at::kBFloat16;
}

inline bool is_integral_or_bool(at::ScalarType t) {
  return at::isIntegralType(t, /*includeBool=*/true);
}

// PyTorch's rule for out= arguments: the result dtype must cast safely to
// out's (no float -> integer, complex -> real, ...).
inline void check_out_cast(at::ScalarType result, const at::Tensor& out) {
  TORCH_CHECK(at::canCast(result, out.scalar_type()), "result type ", result,
              " can't be cast to the desired output type ", out.scalar_type());
}

// An out= tensor must be on "vk" when the result is (copy_ would quietly
// write it anywhere): PyTorch's own kernels check the device.
inline void check_out_device(const at::Tensor& result, const at::Tensor& out) {
  TORCH_CHECK(result.device().type() != kDevice || out.device().type() == kDevice,
              "Expected all tensors to be on the same device, but found at least two devices, ", result.device(),
              " and ", out.device(), "!");
}

// Runs `body(target)` writing into `out` directly when its dtype is `dtype`,
// otherwise into a temporary that is then cast into `out`.
template <typename F>
void with_output(const at::Tensor& out, at::ScalarType dtype, F&& body) {
  if (out.scalar_type() == dtype) {
    body(out);
  } else {
    at::Tensor tmp = at::empty(out.sizes(), out.options().dtype(dtype));
    body(tmp);
    out.copy_(tmp);
  }
}

inline bool is_vk(const at::Tensor& t) {
  return t.device().type() == kDevice;
}

// Runs `op` on `stack` where "vk" has no kernel for it: float16/bfloat16 vk
// tensors in float32 on the GPU when that applies (half_via_float in
// backend.cpp), otherwise on the CPU (vk tensors copied over, mutated/out
// arguments copied back), recorded as a fallback.
void run_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool error_on_views);

// Runs aten op `name.overload` without a Vulkan kernel (see run_fallback) with
// these arguments. Returns the stack.
template <typename... Args>
torch::jit::Stack on_cpu(const char* name, const char* overload, const Args&... args) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, overload);
  torch::jit::Stack stack;
  (stack.emplace_back(args), ...);
  run_fallback(op, &stack, /*error_on_views=*/false);
  return stack;
}

// Queues `kernel` with a push-constant struct (at most 128 bytes) over a
// gx x gy x gz grid of workgroups.
template <typename P>
void dispatch(const std::string& kernel, const std::vector<uint32_t>& spec, const P& params, uint32_t gx,
              uint32_t gy = 1, uint32_t gz = 1) {
  static_assert(sizeof(P) <= 128, "push constants are limited to 128 bytes");
  check(vkt_dispatch(kernel.c_str(), spec.data(), spec.size(), reinterpret_cast<const uint8_t*>(&params), sizeof(P), gx,
                     gy, gz));
}

}  // namespace vk
