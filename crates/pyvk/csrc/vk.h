// Shared declarations for the C++ PyTorch backend of torch_vulkan.
//
// The Rust runtime (crates/pyvk/src/ffi.rs) owns the Vulkan device, memory
// pool, kernel cache and command stream; this side implements PyTorch's
// backend interfaces (allocator, device guard, hooks) and aten kernels on the
// PrivateUse1 dispatch key, renamed to "vk" by the Python package.
#pragma once

#include <ATen/ATen.h>

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
int vkt_dispatch(const char* name, const uint32_t* spec, size_t nspec, const uint8_t* push, size_t npush, uint32_t groups);
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
// the host, one dispatch per outer index.
void launch_strided(const char* kernel, std::vector<uint32_t> spec, const at::Tensor& out, const at::Tensor* a,
                    const at::Tensor* b, double alpha, int64_t alpha_i);

void register_ops();  // ops.cpp

}  // namespace vk
