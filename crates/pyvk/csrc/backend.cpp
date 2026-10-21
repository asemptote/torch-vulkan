// Backend plumbing for device "vk" (PrivateUse1): allocator, device guard,
// hooks, tensor creation/views/resizing, copies, and the CPU fallback.

#include "vk.h"

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include <cstring>
#include <map>
#include <mutex>

namespace vk {

// ---------------------------------------------------------------- allocator

namespace {

void delete_ctx(void* p) {
  auto* ctx = static_cast<AllocCtx*>(p);
  vkt_free(ctx->handle);
  delete ctx;
}

struct VkAllocator final : c10::Allocator {
  c10::DataPtr allocate(size_t nbytes) override {
    c10::Device device(kDevice, 0);
    if (nbytes == 0) {
      return {nullptr, nullptr, &delete_ctx, device};
    }
    VktAlloc a{};
    check(vkt_alloc(nbytes, &a));
    auto* ctx = new AllocCtx{a.handle, a.host, a.address, a.size};
    return {a.host, ctx, &delete_ctx, device};
  }

  // data() and context differ, so raw_allocate/raw_deallocate are unsupported.
  c10::DeleterFnPtr raw_deleter() const override {
    return nullptr;
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
  auto* ctx = static_cast<AllocCtx*>(t.storage().data_ptr().get_context());
  TORCH_CHECK(ctx != nullptr, "torch_vulkan: tensor storage was not allocated by the vk allocator");
  return ctx->address + (static_cast<const uint8_t*>(data) - ctx->host);
}

uint64_t gpu_address(const at::TensorBase& t) {
  return gpu_address(t, t.const_data_ptr());
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

struct VkHooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  c10::DeviceIndex deviceCount() const override { return 1; }
  c10::DeviceIndex getCurrentDevice() const override { return 0; }
  // Random numbers are generated on the CPU for now.
  const at::Generator& getDefaultGenerator(c10::DeviceIndex) const override {
    return at::detail::getDefaultCPUGenerator();
  }
  at::Device getDeviceFromPtr(void*) const override { return {kDevice, 0}; }
  bool isPinnedPtr(const void*) const override { return false; }
  c10::Allocator* getPinnedMemoryAllocator() const override { return at::getCPUAllocator(); }
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {
    resize_storage(storage, newsize);
  }
};

// ---------------------------------------------------------------- creation, views, resize

at::Tensor empty_memory_format(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                               std::optional<at::Device>, std::optional<bool>,
                               std::optional<at::MemoryFormat> memory_format) {
  return at::detail::empty_generic(size, allocator(), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                   c10::dtype_or_default(dtype), memory_format);
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout>, std::optional<at::Device>, std::optional<bool>) {
  return at::detail::empty_strided_generic(size, stride, allocator(), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                           c10::dtype_or_default(dtype));
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

const at::Tensor& resize_(const at::Tensor& self, at::IntArrayRef size, std::optional<at::MemoryFormat> memory_format) {
  auto* impl = self.unsafeGetTensorImpl();
  impl->set_sizes_contiguous(size);
  if (memory_format.has_value() && *memory_format != at::MemoryFormat::Preserve) {
    impl->empty_tensor_restride(*memory_format);
  }
  size_t needed = (impl->storage_offset() + impl->numel()) * self.itemsize();
  if (impl->numel() > 0) {
    resize_storage(self.storage(), needed);
  }
  return self;
}

// ---------------------------------------------------------------- copies

// dst (vk) <- src (vk), with broadcasting and dtype conversion.
void copy_vk_to_vk(const at::Tensor& dst, const at::Tensor& src) {
  const char* from = dtype_label(src.scalar_type());
  const char* to = dtype_label(dst.scalar_type());
  if (from == nullptr || to == nullptr) {
    // No kernel for this dtype: round-trip through the CPU.
    record_fallback("aten::copy_ (dtype " + std::string(c10::toString(src.scalar_type())) + ")");
    dst.copy_(src.cpu());
    return;
  }
  std::string kernel = std::string("cast_") + from + "_" + to;
  launch_strided(kernel.c_str(), {}, dst, &src, nullptr, 0.0, 0);
}

at::Tensor host_view(const at::Tensor& vk_tensor);

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
  if (dtype_label(dst.scalar_type()) != nullptr) {
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
    if (conj) launch_strided(kernel.c_str(), {1}, t, &t, nullptr, 0.0, 0);
    if (neg) launch_strided(kernel.c_str(), {0}, t, &t, nullptr, 0.0, 0);
  } else {
    launch_strided((std::string("unary_") + label).c_str(), {0}, t, &t, nullptr, 0.0, 0);
  }
}

// Copies are conj/neg-aware (registered as fallthrough for the Conjugate and
// Negative keys): data is copied as stored, then the difference between the
// source's and destination's lazy conj/neg bits is applied.
at::Tensor copy_from(const at::Tensor& self, const at::Tensor& dst, bool /*non_blocking*/) {
  if (self.numel() == 0 && dst.numel() == 0) {
    return dst;
  }
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
    at::Tensor host = host_view(self);
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
  synchronize();
  return host_view(self).item();
}

at::Tensor& fill_scalar(at::Tensor& self, const at::Scalar& value) {
  const char* label = dtype_label(self.scalar_type());
  if (label == nullptr) {
    cpu_out("aten::fill_.Scalar", self, [&] { return at::full(self.sizes(), value, self.options().device(at::kCPU)); });
    return self;
  }
  std::string kernel = std::string("fill_") + label;
  if (at::isComplexType(self.scalar_type())) {
    // Real part in alpha, the bits of the (double) imaginary part in alpha_i.
    c10::complex<double> c = value.toComplexDouble();
    double imag = c.imag();
    int64_t imag_bits;
    std::memcpy(&imag_bits, &imag, sizeof(imag_bits));
    launch_strided(kernel.c_str(), {}, self, nullptr, nullptr, c.real(), imag_bits);
    return self;
  }
  bool floating = at::isFloatingType(self.scalar_type());
  launch_strided(kernel.c_str(), {}, self, nullptr, nullptr, floating ? value.toDouble() : 0.0,
                 floating ? 0 : value.toLong());
  return self;
}

at::Tensor& zero_(at::Tensor& self) {
  return fill_scalar(self, 0);
}

// ---------------------------------------------------------------- CPU fallback

std::mutex g_fallback_mutex;
std::map<std::string, int64_t> g_fallbacks;

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& name = op.schema().operator_name();
  record_fallback(name.overload_name.empty() ? name.name : name.name + "." + name.overload_name);
  // A view op run on the CPU would return a copy, silently breaking aliasing
  // (e.g. `x.real += 1`); error out instead so a missing view kernel is caught.
  at::native::cpu_fallback(op, stack, /*error_on_views=*/true);
}

}  // namespace

void record_fallback(const std::string& op) {
  std::lock_guard<std::mutex> lock(g_fallback_mutex);
  if (g_fallbacks[op]++ == 0) {
    TORCH_WARN("torch_vulkan: ", op, " has no Vulkan kernel yet; running it on the CPU");
  }
}

}  // namespace vk

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

// Called once from the Python module's init: registers everything. Kept as
// an explicit call (rather than only static initializers) so the linker
// cannot drop this object file from the extension.
extern "C" void vkt_reduce_linked();
extern "C" void vkt_linalg_linked();
extern "C" void vkt_fft_linked();

extern "C" void vkt_register_backend() {
  // Keep reduce.cpp and linalg.cpp (and their kernel registrations) in the link.
  vkt_reduce_linked();
  vkt_linalg_linked();
  vkt_fft_linked();
  static bool done = false;
  if (done) {
    return;
  }
  done = true;
  c10::SetAllocator(vk::kDevice, vk::allocator());
  at::RegisterPrivateUse1HooksInterface(new vk::VkHooks());
  vk::register_ops();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
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
