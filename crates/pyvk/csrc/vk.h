// Shared declarations for the C++ PyTorch backend of torch_vulkan.
//
// The Rust runtime (crates/pyvk/src/ffi.rs) owns the Vulkan device, memory
// pool, kernel cache and command stream; this side implements PyTorch's
// backend interfaces (allocator, device guard, hooks) and aten kernels on the
// PrivateUse1 dispatch key, renamed to "vk" by the Python package.
#pragma once

#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>

#include <cstddef>
#include <cstdint>
#include <string>
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
}

namespace vk {

constexpr c10::DeviceType kDevice = c10::DeviceType::PrivateUse1;

inline void check(int rc) {
  TORCH_CHECK(rc == 0, "torch_vulkan: ", vkt_last_error());
}

inline void synchronize() {
  check(vkt_synchronize());
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

c10::Allocator* allocator();

// Kernel name suffix for a dtype ("f32", "f64", "i32", "i64", "b8"), or
// nullptr if no kernels exist for it.
const char* dtype_label(at::ScalarType t);

// Records that `op` ran on the CPU instead of the GPU (warns once per op).
void record_fallback(const std::string& op);

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

// Runs binary op `code` (shaders/binary.slang) with PyTorch's type promotion;
// false if it cannot run on the GPU. ops.cpp.
bool binary_op_gpu(int64_t code, const at::Tensor& self, const at::Tensor* other, const at::Scalar* other_scalar,
                   const at::Tensor& out);

inline at::Tensor to_dtype(const at::Tensor& t, at::ScalarType dtype) {
  return t.scalar_type() == dtype ? t : t.to(dtype);
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

// Runs aten op `name.overload` on the CPU with these arguments (vk tensors
// are copied over and mutated/out arguments copied back). Returns the stack.
template <typename... Args>
torch::jit::Stack on_cpu(const char* name, const char* overload, const Args&... args) {
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name, overload);
  record_fallback(*overload ? std::string(name) + "." + overload : std::string(name));
  torch::jit::Stack stack;
  (stack.emplace_back(args), ...);
  at::native::cpu_fallback(op, &stack);
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
