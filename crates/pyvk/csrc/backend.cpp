// Backend plumbing for device "vk" (PrivateUse1): allocator, device guard,
// hooks, tensor creation/views/resizing, copies, and the CPU fallback.

#include "opcodes.h"
#include "vk.h"

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/is_set_to_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include <c10/util/safe_numerics.h>

#include <cstring>
#include <map>
#include <mutex>

namespace vk {

// ---------------------------------------------------------------- allocator

namespace {

// Live allocations by host address. The DataPtr's context is its data
// pointer (as PyTorch's copy-on-write storages require: they wrap simple
// DataPtrs only), so the allocation behind a pointer is looked up here.
std::mutex g_allocations_mutex;
std::map<const uint8_t*, AllocCtx> g_allocations;

void delete_allocation(void* p) {
  if (p == nullptr) {
    return;
  }
  void* handle = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_allocations_mutex);
    auto it = g_allocations.find(static_cast<const uint8_t*>(p));
    TORCH_INTERNAL_ASSERT(it != g_allocations.end(), "torch_vulkan: freeing an unknown allocation");
    handle = it->second.handle;
    g_allocations.erase(it);
  }
  vkt_free(handle);
}

struct VkAllocator final : c10::Allocator {
  c10::DataPtr allocate(size_t nbytes) override {
    c10::Device device(kDevice, 0);
    if (nbytes == 0) {
      return {nullptr, nullptr, &delete_allocation, device};
    }
    VktAlloc a{};
    check(vkt_alloc(nbytes, &a));
    {
      std::lock_guard<std::mutex> lock(g_allocations_mutex);
      g_allocations[a.host] = AllocCtx{a.handle, a.host, a.address, a.size};
    }
    return {a.host, a.host, &delete_allocation, device};
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &delete_allocation;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const override {
    synchronize();
    std::memcpy(dest, src, count);
  }
};

VkAllocator g_allocator;

}  // namespace

c10::Allocator* allocator() {
  return &g_allocator;
}

uint64_t gpu_address(const at::TensorBase& t, const void* data) {
  if (data == nullptr) {
    return 0;
  }
  // The allocation containing `data` (a view may point anywhere inside it).
  const auto* p = static_cast<const uint8_t*>(data);
  std::lock_guard<std::mutex> lock(g_allocations_mutex);
  auto it = g_allocations.upper_bound(p);
  TORCH_CHECK(it != g_allocations.begin(), "torch_vulkan: tensor storage was not allocated by the vk allocator");
  --it;
  const AllocCtx& a = it->second;
  TORCH_CHECK(p < a.host + a.size || (p == a.host + a.size && a.size > 0),
              "torch_vulkan: tensor storage was not allocated by the vk allocator");
  return a.address + static_cast<uint64_t>(p - a.host);
}

uint64_t gpu_address(const at::TensorBase& t) {
  return gpu_address(t, t.const_data_ptr());
}

uint64_t gpu_address_out(const at::TensorBase& t) {
  return gpu_address(t, t.mutable_data_ptr());
}

const char* dtype_label(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kDouble: return vkt_has_float64() ? "f64" : nullptr;
    case at::kInt: return "i32";
    case at::kLong: return "i64";
    case at::kBool: return "b8";
    case at::kComplexFloat: return "c64";
    case at::kComplexDouble: return vkt_has_float64() ? "c128" : nullptr;
    default: return nullptr;
  }
}

const char* storage_label(at::ScalarType t) {
  switch (t) {
    case at::kByte: return has_feature(kFeatureInt8) ? "u8" : nullptr;
    case at::kChar: return has_feature(kFeatureInt8) ? "i8" : nullptr;
    case at::kShort: return has_feature(kFeatureInt16) ? "i16" : nullptr;
    case at::kHalf: return has_feature(kFeatureFloat16) ? "f16" : nullptr;
    case at::kBFloat16: return has_feature(kFeatureInt16) ? "bf16" : nullptr;
    default: return dtype_label(t);
  }
}

at::ScalarType widened(at::ScalarType t) {
  switch (t) {
    case at::kByte:
    case at::kChar:
    case at::kShort: return storage_label(t) != nullptr ? at::kInt : t;
    case at::kHalf:
    case at::kBFloat16: return storage_label(t) != nullptr ? at::kFloat : t;
    default: return t;
  }
}

// ---------------------------------------------------------------- device error word

// Allocated on first use, released by vkt_release_device_error_word at
// shutdown (the runtime that owns the memory goes away).
VktAlloc* g_error_word = nullptr;

namespace {

VktAlloc& error_word() {
  if (g_error_word == nullptr) {
    auto* a = new VktAlloc{};
    check(vkt_alloc(16, a));
    // Pool memory is reused in stream order: kernels still queued may write
    // its previous contents, so wait before the host clears it.
    check(vkt_synchronize());
    std::memset(a->host, 0, 16);
    g_error_word = a;
  }
  return *g_error_word;
}

}  // namespace

// The bytes of `value` converted to `dtype`, as parameter slots for a fill
// (declared in vk.h).
std::array<uint64_t, 2> fill_bits(const at::Scalar& value, at::ScalarType dtype) {
  at::Tensor v = at::scalar_tensor(value, at::TensorOptions().dtype(dtype));
  std::array<uint64_t, 2> bits{};
  std::memcpy(bits.data(), v.const_data_ptr(), v.itemsize());
  return bits;
}

uint64_t device_error_address() {
  return error_word().address;
}

// 1 (and clears it) if an indexing kernel has flagged an out-of-range index.
// Call after synchronizing.
int take_device_error() {
  if (g_error_word == nullptr) {
    return 0;
  }
  auto* word = reinterpret_cast<uint32_t*>(g_error_word->host);
  int flagged = *word != 0;
  *word = 0;
  return flagged;
}

void synchronize() {
  check(vkt_synchronize());
  TORCH_CHECK_INDEX(!take_device_error(), "torch_vulkan: index out of range in an earlier indexing kernel");
}

// ---------------------------------------------------------------- device guard and hooks

namespace {

struct VkGuardImpl final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return kDevice; }
  c10::Device exchangeDevice(c10::Device) const override { return {kDevice, 0}; }
  c10::Device getDevice() const override { return {kDevice, 0}; }
  void setDevice(c10::Device d) const override {
    TORCH_CHECK(d.index() <= 0, "torch_vulkan: only device vk:0 exists");
  }
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device) const override { return c10::Stream(c10::Stream::DEFAULT, {kDevice, 0}); }
  c10::Stream exchangeStream(c10::Stream) const override { return c10::Stream(c10::Stream::DEFAULT, {kDevice, 0}); }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  // One in-order stream: events are trivially complete once recorded.
  void record(void**, const c10::Stream&, const c10::DeviceIndex, const c10::EventFlag) const override {}
  void block(void*, const c10::Stream&) const override {}
  bool queryEvent(void*) const override { return true; }
  void destroyEvent(void*, const c10::DeviceIndex) const noexcept override {}
  bool queryStream(const c10::Stream&) const override { return true; }
  void synchronizeStream(const c10::Stream&) const override { synchronize(); }
  void synchronizeEvent(void*) const override { synchronize(); }
  void synchronizeDevice(const c10::DeviceIndex) const override { synchronize(); }
};

C10_REGISTER_GUARD_IMPL(PrivateUse1, VkGuardImpl);

void resize_storage(const c10::Storage& storage, size_t nbytes);
void resize_storage_bytes(const c10::Storage& storage, size_t nbytes);

struct VkHooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  c10::DeviceIndex deviceCount() const override { return 1; }
  c10::DeviceIndex getCurrentDevice() const override { return 0; }
  const at::Generator& getDefaultGenerator(c10::DeviceIndex) const override {
    return default_generator();
  }
  at::Generator getNewGenerator(c10::DeviceIndex) const override {
    return new_generator();
  }
  at::Device getDeviceFromPtr(void*) const override { return {kDevice, 0}; }
  bool isPinnedPtr(const void*) const override { return false; }
  c10::Allocator* getPinnedMemoryAllocator() const override { return at::getCPUAllocator(); }
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {
    resize_storage_bytes(storage, newsize);
  }
};

// ---------------------------------------------------------------- creation, views, resize

// torch.use_deterministic_algorithms(True) (with fill_uninitialized_memory, the
// default): new memory is filled with NaN or the largest integer, as the CPU.
bool fill_uninitialized() {
  return at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory();
}

at::Tensor empty_memory_format(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                               std::optional<at::Device>, std::optional<bool>,
                               std::optional<at::MemoryFormat> memory_format) {
  at::Tensor t = at::detail::empty_generic(size, allocator(), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                           c10::dtype_or_default(dtype), memory_format);
  if (fill_uninitialized()) at::native::fill_empty_deterministic_(t);
  return t;
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout>, std::optional<at::Device>, std::optional<bool>) {
  at::Tensor t = at::detail::empty_strided_generic(size, stride, allocator(),
                                                   c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                                   c10::dtype_or_default(dtype));
  if (fill_uninitialized()) at::native::fill_empty_deterministic_(t);
  return t;
}

// Grows (never shrinks) a storage, keeping its contents.
void resize_storage(const c10::Storage& storage, size_t nbytes) {
  if (nbytes <= storage.nbytes()) {
    return;
  }
  c10::DataPtr fresh = allocator()->allocate(nbytes);
  if (storage.nbytes() > 0 && storage.data() != nullptr) {
    synchronize();  // queued kernels may still write the old memory
    std::memcpy(fresh.get(), storage.data(), storage.nbytes());
  }
  storage.unsafeGetStorageImpl()->set_data_ptr_noswap(std::move(fresh));
  storage.unsafeGetStorageImpl()->set_nbytes(nbytes);
}

// untyped_storage().resize_(n): exactly n bytes, shrinking too, as the CPU's
// resize_bytes_cpu (the first min(old, n) bytes kept).
void resize_storage_bytes(const c10::Storage& storage, size_t nbytes) {
  TORCH_CHECK(storage.resizable(), "Trying to resize storage that is not resizable");
  if (nbytes == storage.nbytes()) {
    return;
  }
  c10::DataPtr fresh = allocator()->allocate(nbytes);
  size_t keep = std::min(nbytes, storage.nbytes());
  if (keep > 0 && storage.data() != nullptr) {
    synchronize();  // queued kernels may still write the old memory
    std::memcpy(fresh.get(), storage.data(), keep);
  }
  storage.unsafeGetStorageImpl()->set_data_ptr_noswap(std::move(fresh));
  storage.unsafeGetStorageImpl()->set_nbytes(nbytes);
}

const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size, std::optional<at::MemoryFormat> memory_format) {
  auto* impl = self.unsafeGetTensorImpl();
  if (self.sizes() == size && !memory_format.has_value()) {
    return self;  // (as PyTorch: the same sizes keep their strides, e.g. a strided out=)
  }
  {  // (the CPU's overflow checks)
    int64_t stride = 1;
    for (int64_t d = static_cast<int64_t>(size.size()) - 1; d > 0; --d) {
      TORCH_CHECK(!c10::mul_overflows(stride, std::max<int64_t>(size[d], 1), &stride), "Stride calculation overflowed");
    }
    at::detail::computeStorageNbytesContiguous(size, self.itemsize(), self.storage_offset());
  }
  impl->set_sizes_contiguous(size);
  if (memory_format.has_value() && *memory_format != at::MemoryFormat::Preserve) {
    impl->empty_tensor_restride(*memory_format);
  }
  size_t needed = (impl->storage_offset() + impl->numel()) * self.itemsize();
  if (impl->numel() > 0) {
    int64_t old_nbytes = static_cast<int64_t>(self.storage().nbytes());
    resize_storage(self.storage(), needed);
    if (fill_uninitialized()) at::native::fill_resize_deterministic_(self, old_nbytes);
  }
  return self;
}

// ---------------------------------------------------------------- copies

// dst (vk) <- src (vk), with broadcasting and dtype conversion.
void copy_vk_to_vk(const at::Tensor& dst, const at::Tensor& src) {
  bool chalf_src = src.scalar_type() == at::kComplexHalf, chalf_dst = dst.scalar_type() == at::kComplexHalf;
  if (chalf_src != chalf_dst && storage_label(at::kHalf) != nullptr) {
    // complex32 holds two halves: convert its parts.
    at::Tensor s = src.expand(dst.sizes());
    if (chalf_dst) {
      at::Tensor parts = at::view_as_real(dst);
      parts.select(-1, 0).copy_(s.is_complex() ? at::real(s) : s);
      if (s.is_complex()) {
        parts.select(-1, 1).copy_(at::imag(s));
      } else {
        parts.select(-1, 1).zero_();
      }
    } else {
      at::Tensor parts = at::view_as_real(s);
      if (dst.is_complex()) {
        at::view_as_real(dst).copy_(parts);
      } else {
        dst.copy_(parts.select(-1, 0));
      }
    }
    return;
  }
  if (src.scalar_type() == dst.scalar_type()) {
    std::string kernel = move_kernel(src.itemsize());
    if (!kernel.empty()) {
      launch_strided(kernel.c_str(), {op::MOVE_COPY}, dst, &src, nullptr);
      return;
    }
  }
  const char* from = storage_label(src.scalar_type());
  const char* to = storage_label(dst.scalar_type());
  if (from == nullptr || to == nullptr || src.scalar_type() == dst.scalar_type()) {
    // No kernel for this dtype: round-trip through the CPU.
    record_fallback("aten::copy_ (dtype " + std::string(c10::toString(src.scalar_type())) + ")");
    dst.copy_(src.cpu());
    return;
  }
  std::string kernel = std::string("cast_") + from + "_" + to;
  launch_strided(kernel.c_str(), {}, dst, &src, nullptr);
}

at::Tensor host_view(const at::Tensor& vk_tensor);
at::Tensor host_view_const(const at::Tensor& vk_tensor);

// dst (vk) <- src (cpu), with broadcasting and dtype conversion.
void copy_cpu_to_vk(const at::Tensor& dst, const at::Tensor& src) {
  at::Tensor host = src.to(dst.scalar_type()).expand(dst.sizes());
  if (dst.is_contiguous() && host.is_contiguous()) {
    if (dst.numel() > 0) {
      synchronize();  // the destination may still be in use by queued kernels
      std::memcpy(dst.data_ptr(), host.const_data_ptr(), dst.nbytes());
    }
    return;
  }
  if (!move_kernel(dst.itemsize()).empty()) {
    // Upload contiguously, then scatter with a strided copy kernel.
    at::Tensor staging = empty_memory_format(dst.sizes(), dst.scalar_type(), {}, {}, {}, {});
    copy_cpu_to_vk(staging, host.contiguous());
    copy_vk_to_vk(dst, staging);
    return;
  }
  synchronize();
  host_view(dst).copy_(host);
}

// A CPU tensor aliasing the mapped memory of `t` (valid until queued GPU work
// touches it again; callers synchronize first).
at::Tensor host_view(const at::Tensor& t) {
  return at::from_blob(t.data_ptr(), t.sizes(), t.strides(), at::TensorOptions().dtype(t.scalar_type()));
}

// A read-only host view: does not materialize a copy-on-write storage (as
// asking for the mutable data pointer would).
at::Tensor host_view_const(const at::Tensor& t) {
  return at::from_blob(const_cast<void*>(t.const_data_ptr()), t.sizes(), t.strides(),
                       at::TensorOptions().dtype(t.scalar_type()));
}

// Applies conj (complex) and/or negation in place to vk tensor `t`'s data.
void fix_up_in_place(const at::Tensor& t, bool conj, bool neg) {
  const char* label = dtype_label(t.scalar_type());
  bool complex = at::isComplexType(t.scalar_type());
  conj = conj && complex;
  if (!conj && !neg) {
    return;
  }
  if (label == nullptr) {
    record_fallback("aten::copy_ (conj/neg bit, dtype " + std::string(c10::toString(t.scalar_type())) + ")");
    at::Tensor host = t.cpu();
    if (conj) host = host.conj_physical();
    if (neg) host = host.neg();
    t.copy_(host);
    return;
  }
  if (complex) {
    std::string kernel = std::string("cunary_") + label;
    if (conj) launch_strided(kernel.c_str(), {op::CUNARY_CONJ}, t, &t, nullptr);
    if (neg) launch_strided(kernel.c_str(), {op::CUNARY_NEG}, t, &t, nullptr);
  } else {
    launch_strided((std::string("unary_") + label).c_str(), {op::UNARY_NEG}, t, &t, nullptr);
  }
}

// Copies are conj/neg-aware (registered as fallthrough for the Conjugate and
// Negative keys): data is copied as stored, then the difference between the
// source's and destination's lazy conj/neg bits is applied.
at::Tensor copy_from(const at::Tensor& self, const at::Tensor& dst, bool /*non_blocking*/) {
  if (self.numel() == 0 && dst.numel() == 0) {
    return dst;
  }
  check_overlap(dst, {&self});  // (as the CPU's copy_)
  bool conj = self.is_conj() != dst.is_conj();
  bool neg = self.is_neg() != dst.is_neg();
  bool src_vk = self.device().type() == kDevice;
  bool dst_vk = dst.device().type() == kDevice;
  if (src_vk && dst_vk) {
    copy_vk_to_vk(dst, self);
    fix_up_in_place(dst, conj, neg);
  } else if (dst_vk) {
    copy_cpu_to_vk(dst, self.resolve_conj().resolve_neg());  // cheap on the CPU
    // The source is resolved; dst's own lazy bits mean its stored data is their inverse.
    fix_up_in_place(dst, dst.is_conj(), dst.is_neg());
  } else {
    TORCH_CHECK(src_vk && dst.is_cpu(), "torch_vulkan: unsupported copy from ", self.device(), " to ", dst.device());
    synchronize();
    at::Tensor host = host_view_const(self);
    if (self.is_conj()) host = host.conj();
    if (self.is_neg()) host = host._neg_view();
    dst.copy_(host);
  }
  return dst;
}

at::Tensor copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  // Used by the CPU fallback to write results into vk tensors: the vk tensor
  // takes the result's shape (this is where `out=` fallbacks must resize).
  if (dst.sizes() != self.sizes()) {
    resize_(dst, self.sizes(), std::nullopt);
  }
  return copy_from(self, dst, false);
}

at::Scalar local_scalar_dense(const at::Tensor& self) {
  TORCH_CHECK(self.numel() > 0, "_local_scalar_dense: Empty tensor not supported");
  synchronize();
  return host_view_const(self).item();
}

at::Tensor& fill_scalar(at::Tensor& self, const at::Scalar& value) {
  std::string kernel = move_kernel(self.itemsize());
  if (kernel.empty()) {
    cpu_out("aten::fill_.Scalar", self, [&] { return at::full(self.sizes(), value, self.options().device(at::kCPU)); });
    return self;
  }
  std::array<uint64_t, 2> bits = fill_bits(value, self.scalar_type());
  launch_strided_bits(kernel.c_str(), {op::MOVE_FILL}, self, nullptr, nullptr, bits);
  return self;
}

at::Tensor& fill_tensor(at::Tensor& self, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(),
              " dimensions.");
  return fill_scalar(self, value.item());
}

at::Tensor& zero_(at::Tensor& self) {
  return fill_scalar(self, 0);
}

// ---------------------------------------------------------------- CPU fallback

std::mutex g_fallback_mutex;
std::map<std::string, int64_t> g_fallbacks;

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // A view op run on the CPU would return a copy, silently breaking aliasing
  // (e.g. `x.real += 1`); error out instead so a missing view kernel is caught.
  run_fallback(op, stack, /*error_on_views=*/true);
}

}  // namespace

// TORCH_VK_STRICT=1: every op must run on Vulkan; a CPU fallback raises.
bool strict_mode() {
  static const bool strict = [] {
    const char* v = std::getenv("TORCH_VK_STRICT");
    return v != nullptr && *v != '\0' && std::string(v) != "0";
  }();
  return strict;
}

void check_out_devices(const c10::FunctionSchema& schema, const torch::jit::Stack& stack) {
  const auto& arguments = schema.arguments();
  if (stack.size() < arguments.size()) {
    return;
  }
  size_t first = stack.size() - arguments.size();
  auto on_vk = [](const c10::IValue& v) {
    if (v.isTensor()) {
      return v.toTensor().defined() && v.toTensor().device().type() == kDevice;
    }
    if (v.isTensorList()) {
      for (const at::Tensor& t : v.toTensorList().vec()) {
        if (t.defined() && t.device().type() == kDevice) return true;
      }
    }
    return false;
  };
  auto is_out = [](const c10::Argument& a) { return a.kwarg_only() && a.alias_info() && a.alias_info()->isWrite(); };
  // The indices of indexing ops (Tensor?[]) may be on the CPU, and index_fill's index.
  bool index_fill = schema.name().find("index_fill") != std::string::npos;
  auto indices = [&](const c10::Argument& a) {
    return a.type()->str() == "Tensor?[]" || (index_fill && a.name() == "index");
  };
  auto cpu_tensor = [](const c10::IValue& v) {  // a CPU tensor that is not a scalar
    if (v.isTensor()) {
      const at::Tensor& t = v.toTensor();
      return t.defined() && t.device().is_cpu() && t.dim() > 0;
    }
    if (v.isTensorList()) {
      for (const at::Tensor& t : v.toTensorList().vec()) {
        if (t.defined() && t.device().is_cpu() && t.dim() > 0) return true;
      }
    }
    return false;
  };
  bool vk_input = false, any_vk = false, any_cpu = false;
  for (size_t i = 0; i < arguments.size(); ++i) {
    const c10::IValue& v = stack[first + i];
    vk_input = vk_input || (!is_out(arguments[i]) && on_vk(v));
    any_vk = any_vk || on_vk(v);
    any_cpu = any_cpu || (!indices(arguments[i]) && cpu_tensor(v));
  }
  bool copy = schema.name().find("copy") != std::string::npos;
  TORCH_CHECK(!any_vk || !any_cpu || copy, "Expected all tensors to be on the same device, but found at least two "
              "devices, ", kDevice, " and cpu!");
  if (!vk_input) {
    return;
  }
  for (size_t i = 0; i < arguments.size(); ++i) {
    const c10::IValue& v = stack[first + i];
    if (is_out(arguments[i]) && v.isTensor() && v.toTensor().defined()) {
      const at::Device device = v.toTensor().device();
      TORCH_CHECK(device.type() == kDevice, "Expected all tensors to be on the same device, but found at least two "
                  "devices, ", kDevice, " and ", device, "!");
    }
  }
}

// ---------------------------------------------------------------- 16-bit floats through float32

namespace {

bool is_half_vk(const at::Tensor& t) {
  return t.defined() && is_vk(t) && (t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16) &&
         storage_label(t.scalar_type()) != nullptr;
}

at::Tensor widen_half_arg(const at::Tensor& t) {
  return is_half_vk(t) ? at::_to_copy(t, t.options().dtype(at::kFloat)) : t;
}

at::Tensor meta_like(const at::Tensor& t) {
  return t.defined() && is_vk(t) ? at::empty_strided(t.sizes(), t.strides(), t.options().device(at::kMeta)) : t;
}

// Runs `op` with its float16/bfloat16 "vk" tensors widened to float32, on the
// GPU, instead of on the CPU: results are rounded once from float32, as CUDA
// computes most 16-bit ops. The op first runs on meta tensors of the original
// dtypes, which applies PyTorch's dtype checks and gives the result dtypes.
// False (stack untouched) where this does not apply: no 16-bit vk tensor,
// floating vk tensors of more than one dtype (mixed-dtype errors, which not
// every meta kernel raises, e.g. mm's, stay the CPU's), views, mutated tensor
// lists, or the meta call failing (the CPU then decides).
bool half_via_float(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  for (const auto& r : schema.returns()) {
    if (r.alias_info() != nullptr && !r.alias_info()->isWrite()) {
      return false;
    }
  }
  size_t n = schema.arguments().size();
  size_t base = stack->size() - n;
  bool any = false;
  std::optional<at::ScalarType> floating;  // the one floating dtype of the vk tensors
  bool mixed = false;
  auto note = [&](const at::Tensor& t) {
    if (t.defined() && is_vk(t) && (at::isFloatingType(t.scalar_type()) || at::isComplexType(t.scalar_type()))) {
      mixed = mixed || (floating.has_value() && *floating != t.scalar_type());
      floating = t.scalar_type();
    }
  };
  for (size_t i = 0; i < n; ++i) {
    const c10::IValue& v = (*stack)[base + i];
    const auto* ai = schema.arguments()[i].alias_info();
    bool write = ai != nullptr && ai->isWrite();
    if (v.isTensor()) {
      any = any || is_half_vk(v.toTensor());
      note(v.toTensor());
    } else if (v.isTensorList()) {
      for (const at::Tensor& t : v.toTensorList()) {
        note(t);
        if (is_half_vk(t)) {
          if (write) {
            return false;
          }
          any = true;
        }
      }
    } else if (v.isList()) {
      for (const c10::IValue& e : v.toListRef()) {
        if (e.isTensor() && is_half_vk(e.toTensor())) {
          return false;  // (lists of optional tensors: left to the CPU)
        }
      }
    }
  }
  if (!any || mixed) {
    return false;
  }
  torch::jit::Stack meta(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end());
  for (c10::IValue& v : meta) {
    if (v.isTensor()) {
      v = meta_like(v.toTensor());
    } else if (v.isTensorList()) {
      c10::List<at::Tensor> l;
      for (const at::Tensor& t : v.toTensorList()) {
        l.push_back(meta_like(t));
      }
      v = l;
    }
  }
  try {
    op.callBoxed(&meta);
  } catch (const std::exception&) {
    return false;
  }
  std::vector<std::pair<at::Tensor, at::Tensor>> written;  // (original, widened)
  torch::jit::Stack saved(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end());
  for (size_t i = 0; i < n; ++i) {
    c10::IValue& v = (*stack)[base + i];
    const auto* ai = schema.arguments()[i].alias_info();
    if (v.isTensor() && is_half_vk(v.toTensor())) {
      at::Tensor orig = v.toTensor();
      at::Tensor w = widen_half_arg(orig);
      if (ai != nullptr && ai->isWrite()) {
        written.emplace_back(orig, w);
      }
      v = w;
    } else if (v.isTensorList()) {
      c10::List<at::Tensor> l;
      for (const at::Tensor& t : v.toTensorList()) {
        l.push_back(widen_half_arg(t));
      }
      v = l;
    }
  }
  try {
    op.callBoxed(stack);
  } catch (const std::exception&) {
    // (e.g. a dtype= argument the widened tensors contradict: the CPU decides;
    // the original tensors are only written after success)
    stack->resize(base);
    stack->insert(stack->end(), saved.begin(), saved.end());
    return false;
  }
  for (auto& [orig, w] : written) {
    if (!orig.sizes().equals(w.sizes())) {
      orig.resize_(w.sizes());  // (an out= the op resized; it warned on the widened copy)
    }
    orig.copy_(w);
  }
  size_t nr = schema.returns().size();
  for (size_t j = 0; j < nr; ++j) {
    c10::IValue& r = (*stack)[stack->size() - nr + j];
    const c10::IValue& m = meta[meta.size() - nr + j];
    if (r.isTensor()) {
      at::Tensor t = r.toTensor();
      for (auto& [orig, w] : written) {
        if (t.is_same(w)) {
          t = orig;
        }
      }
      if (t.defined() && m.isTensor() && t.scalar_type() != m.toTensor().scalar_type()) {
        t = t.to(m.toTensor().scalar_type());
      }
      r = t;
    } else if (r.isTensorList() && m.isTensorList()) {
      c10::List<at::Tensor> l;
      auto ml = m.toTensorList();
      auto rl = r.toTensorList();
      for (size_t k = 0; k < rl.size(); ++k) {
        at::Tensor t = rl.get(k);
        at::ScalarType want = k < ml.size() ? ml.get(k).scalar_type() : t.scalar_type();
        l.push_back(t.defined() && t.scalar_type() != want ? t.to(want) : t);
      }
      r = l;
    }
  }
  return true;
}

}  // namespace

void run_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool error_on_views) {
  if (half_via_float(op, stack)) {
    return;
  }
  const auto& name = op.schema().operator_name();
  record_fallback(name.overload_name.empty() ? name.name : name.name + "." + name.overload_name);
  check_out_devices(op.schema(), *stack);
  at::native::cpu_fallback(op, stack, error_on_views);
}

void record_fallback(const std::string& op) {
  TORCH_CHECK_NOT_IMPLEMENTED(!strict_mode(), "torch_vulkan: ", op,
                              " has no Vulkan kernel (TORCH_VK_STRICT forbids running it on the CPU)");
  std::lock_guard<std::mutex> lock(g_fallback_mutex);
  if (g_fallbacks[op]++ == 0) {
    TORCH_WARN("torch_vulkan: ", op, " has no Vulkan kernel yet; running it on the CPU");
  }
}

}  // namespace vk

// For Python-registered kernels (torch_vulkan/_decomp.py) that hand an op
// to the CPU: counted and warned about like the C++ fallbacks. Returns 1 in
// strict mode (TORCH_VK_STRICT), where the caller raises instead.
extern "C" int vkt_record_fallback(const char* op) {
  try {
    vk::record_fallback(op);
    return 0;
  } catch (const std::exception&) {
    return 1;
  }
}

extern "C" int vkt_take_device_error() {
  return vk::take_device_error();
}

// Frees the device error word (called before the runtime shuts down).
extern "C" void vkt_release_device_error_word() {
  if (vk::g_error_word != nullptr) {
    vkt_free(vk::g_error_word->handle);
    delete vk::g_error_word;
    vk::g_error_word = nullptr;
  }
}

extern "C" size_t vkt_fallback_counts(char* buf, size_t len) {
  // "op=count\n" lines; returns the length needed (write only if it fits).
  std::string text;
  {
    std::lock_guard<std::mutex> lock(vk::g_fallback_mutex);
    for (const auto& [op, count] : vk::g_fallbacks) {
      text += op + "=" + std::to_string(count) + "\n";
    }
  }
  if (buf != nullptr && text.size() < len) {
    std::memcpy(buf, text.c_str(), text.size() + 1);
  }
  return text.size() + 1;
}

// Called once from the Python module's init: installs the allocator and
// hooks. The aten kernels register themselves through the
// TORCH_LIBRARY_IMPL blocks of each file (the library is linked whole, so
// none are dropped).
extern "C" void vkt_register_backend() {
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  c10::SetAllocator(vk::kDevice, vk::allocator());
  at::RegisterPrivateUse1HooksInterface(new vk::VkHooks());
}

namespace vk {
namespace {

// set_ overloads besides (storage, offset, size, stride): the whole storage
// as a 1-D tensor, another tensor's view, or a fresh empty storage. (Storage
// copies, e.g. untyped_storage().clone(), go through these.)
at::Tensor& set_storage(at::Tensor& self, c10::Storage source) {
  int64_t numel = static_cast<int64_t>(source.nbytes() / self.dtype().itemsize());
  return at::native::set_storage_cpu_(self, std::move(source), 0, {numel}, {});
}

at::Tensor& set_tensor(at::Tensor& self, const at::Tensor& source) {
  if (self.unsafeGetTensorImpl() == source.unsafeGetTensorImpl()) {
    return self;
  }
  return at::native::set_storage_cpu_(self, source.storage(), source.storage_offset(), source.sizes(),
                                      source.strides());
}

at::Tensor& set_empty(at::Tensor& self) {
  c10::Storage empty(c10::Storage::use_byte_size_t(), 0, allocator(), /*resizable=*/true);
  return at::native::set_storage_cpu_(self, std::move(empty), 0, {0}, {});
}

}  // namespace
}  // namespace vk

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("set_.source_Storage", vk::set_storage);
  m.impl("set_.source_Tensor", vk::set_tensor);
  m.impl("set_", vk::set_empty);
  m.impl("is_set_to", at::native::is_set_to);
  m.impl("empty.memory_format", vk::empty_memory_format);
  m.impl("empty_strided", vk::empty_strided);
  m.impl("view", at::native::view);
  m.impl("as_strided", at::native::as_strided_tensorimpl);
  m.impl("_reshape_alias", at::native::_reshape_alias);
  m.impl("view_as_real", at::native::view_as_real);
  m.impl("view_as_complex", at::native::view_as_complex);
  m.impl("unfold", at::native::unfold);
  m.impl("resize_", vk::resize_);
  m.impl("set_.source_Storage_storage_offset", at::native::set_storage_cpu_);
  m.impl("_copy_from", vk::copy_from);
  m.impl("_copy_from_and_resize", vk::copy_from_and_resize);
  m.impl("_local_scalar_dense", vk::local_scalar_dense);
  m.impl("fill_.Scalar", vk::fill_scalar);
  m.impl("fill_.Tensor", vk::fill_tensor);
  m.impl("zero_", vk::zero_);
}

// The Conjugate/Negative fallbacks would materialize lazy bits by copying,
// which reaches _copy_from again; copy_from handles the bits itself.
TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("_copy_from", torch::CppFunction::makeFallthrough());
  m.impl("_copy_from_and_resize", torch::CppFunction::makeFallthrough());
}
TORCH_LIBRARY_IMPL(aten, Negative, m) {
  m.impl("_copy_from", torch::CppFunction::makeFallthrough());
  m.impl("_copy_from_and_resize", torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&vk::cpu_fallback>());
}

// Autocast on "vk" changes no dtypes (no lower-precision policy for this
// backend yet): every op passes straight through the AutocastPrivateUse1 key.
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
