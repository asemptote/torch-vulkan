// Backend plumbing for device "vk" (PrivateUse1): allocator, device guard,
// hooks, tensor creation/views/resizing, copies, and the CPU fallback.

#include "opcodes.h"
#include "vk.h"

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/ResizeCommon.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/ops/_has_compatible_shallow_copy_type_native.h>
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

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <map>
#include <set>
#include <mutex>

namespace vk {

#if defined(__ANDROID__)
// ---------------------------------------------------------------- typed errors on Android (vk.h)

}  // namespace vk

#include <dlfcn.h>

#include <new>
#include <typeinfo>

extern "C" void* __cxa_allocate_exception(size_t) noexcept;
extern "C" [[noreturn]] void __cxa_throw(void*, std::type_info*, void (*)(void*));

namespace vk {
namespace {

// The type_info `symbol` names as libtorch_python's catch clauses see it:
// the globally visible definition if there is one, else libtorch_python's
// own (each library loaded locally binds to the first definition in its
// own dependency order, libtorch_python's being itself).
std::type_info* python_type_info(const char* symbol) {
  void* p = dlsym(RTLD_DEFAULT, symbol);
  if (p == nullptr) {
    void* lib = dlopen("libtorch_python.so", RTLD_NOW | RTLD_NOLOAD);
    if (lib != nullptr) p = dlsym(lib, symbol);
  }
  return static_cast<std::type_info*>(p);
}

template <typename E>
[[noreturn]] void throw_with(const char* symbol, const char* func, const char* file, uint32_t line, std::string msg) {
  static std::type_info* info = python_type_info(symbol);
  c10::SourceLocation where{func, file, line};
  if (info == nullptr) throw E(where, std::move(msg));  // (as before: RuntimeError in Python)
  void* object = __cxa_allocate_exception(sizeof(E));
  new (object) E(where, std::move(msg));
  __cxa_throw(object, info, [](void* p) { static_cast<E*>(p)->~E(); });
}

}  // namespace

void throw_typed(TypedError kind, const char* func, const char* file, uint32_t line, std::string msg) {
  switch (kind) {
    case TypedError::Index: throw_with<c10::IndexError>("_ZTIN3c1010IndexErrorE", func, file, line, std::move(msg));
    case TypedError::Type: throw_with<c10::TypeError>("_ZTIN3c109TypeErrorE", func, file, line, std::move(msg));
    case TypedError::Value: throw_with<c10::ValueError>("_ZTIN3c1010ValueErrorE", func, file, line, std::move(msg));
    case TypedError::NotImplemented:
      throw_with<c10::NotImplementedError>("_ZTIN3c1019NotImplementedErrorE", func, file, line, std::move(msg));
  }
  throw c10::Error(c10::SourceLocation{func, file, line}, std::move(msg));
}
#endif

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

// A c10::DeviceAllocator, so torch.accelerator's memory functions
// (empty_cache, memory_allocated, ...) work for "vk" as for CUDA.
struct VkAllocator final : c10::DeviceAllocator {
  bool initialized() override { return true; }

  void emptyCache(c10::MempoolId_t /*mempool_id*/) override { check(vkt_empty_cache()); }

  void recordStream(const c10::DataPtr& /*ptr*/, c10::Stream /*stream*/) override {}  // (one queue)

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex /*device*/) override {
    uint64_t allocated = 0, reserved = 0;
    vkt_memory_stats(&allocated, &reserved);
    c10::CachingDeviceAllocator::DeviceStats stats;
    for (auto& s : stats.allocated_bytes) s.current = static_cast<int64_t>(allocated);
    for (auto& s : stats.reserved_bytes) s.current = static_cast<int64_t>(reserved);
    return stats;
  }

  void resetAccumulatedStats(c10::DeviceIndex /*device*/) override {}

  void resetPeakStats(c10::DeviceIndex /*device*/) override {}

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

// Pinned (page-locked) host memory for "vk": vk's own buffers are host-visible
// already, so pinning gives ordinary CPU memory, recorded so is_pinned()
// answers for it (torch.save's use_pinned_memory_for_d2h, Tensor.pin_memory).
std::mutex g_pinned_mutex;
std::map<const uint8_t*, size_t> g_pinned;  // start -> size

void free_pinned(void* p) {
  {
    std::lock_guard<std::mutex> lock(g_pinned_mutex);
    g_pinned.erase(static_cast<const uint8_t*>(p));
  }
  c10::free_cpu(p);
}

struct PinnedAllocator final : c10::Allocator {
  c10::DataPtr allocate(size_t nbytes) override {
    void* p = nbytes == 0 ? nullptr : c10::alloc_cpu(nbytes);
    if (p != nullptr) {
      std::lock_guard<std::mutex> lock(g_pinned_mutex);
      g_pinned[static_cast<const uint8_t*>(p)] = nbytes;
    }
    return {p, p, &free_pinned, c10::Device(c10::kCPU)};
  }
  c10::DeleterFnPtr raw_deleter() const override { return &free_pinned; }
  void copy_data(void* dest, const void* src, std::size_t count) const override { std::memcpy(dest, src, count); }
};

PinnedAllocator g_pinned_allocator;

bool is_pinned_ptr(const void* data) {
  const auto* p = static_cast<const uint8_t*>(data);
  std::lock_guard<std::mutex> lock(g_pinned_mutex);
  auto it = g_pinned.upper_bound(p);
  if (it == g_pinned.begin()) return false;
  --it;
  return p < it->first + it->second;
}

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
  bool isPinnedPtr(const void* data) const override { return is_pinned_ptr(data); }
  c10::Allocator* getPinnedMemoryAllocator() const override { return &g_pinned_allocator; }
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

// GPUs without float64 arithmetic (shaderFloat64: Mali, most phones) hold no
// float64 or complex128 tensors, as PyTorch's MPS backend: creating one is a
// TypeError rather than a silent CPU fallback or a slow emulation.
void check_dtype_supported(at::ScalarType t) {
  if ((t == at::kDouble || t == at::kComplexDouble) && !vkt_has_float64()) {
    bool complex = t == at::kComplexDouble;
    TORCH_CHECK_TYPE(false, "Cannot create a vk tensor of ", complex ? "complex128" : "float64",
                     " dtype: this GPU has no float64 arithmetic (shaderFloat64). Please use ",
                     complex ? "complex64" : "float32", " instead.");
  }
}

at::Tensor empty_memory_format(at::IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                               std::optional<at::Device>, std::optional<bool>,
                               std::optional<at::MemoryFormat> memory_format) {
  check_dtype_supported(c10::dtype_or_default(dtype));
  at::Tensor t = at::detail::empty_generic(size, allocator(), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                           c10::dtype_or_default(dtype), memory_format);
  if (fill_uninitialized()) at::native::fill_empty_deterministic_(t);
  return t;
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride, std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout>, std::optional<at::Device>, std::optional<bool>) {
  check_dtype_supported(c10::dtype_or_default(dtype));
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

// ---------------------------------------------------------------- float8

bool is_float8(at::ScalarType s) {
  return s == at::kFloat8_e4m3fn || s == at::kFloat8_e5m2 || s == at::kFloat8_e4m3fnuz || s == at::kFloat8_e5m2fnuz;
}

bool fnuz(at::ScalarType s) {
  return s == at::kFloat8_e4m3fnuz || s == at::kFloat8_e5m2fnuz;
}

// A float8 format's conversions, as tables built once from the CPU's own
// (c10's) conversions, so they are the CPU's bit for bit: `values` [256]
// float32, each code's value; `thresholds` [K] int64, sorted, and `codes`
// [K] uint8: a float32 of magnitude bits m (sign cleared, NaN included)
// converts to codes[k] for the last k with thresholds[k] <= m (the
// conversion's magnitude code is monotone in m), the sign then set as c10
// does (fnuz formats: none on zero, NaN 0x80).
struct Float8Tables {
  at::Tensor values, thresholds, codes;
};

const Float8Tables& float8_tables(at::ScalarType t) {
  static std::mutex mutex;
  static std::map<at::ScalarType, Float8Tables> tables;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = tables.find(t);
  if (it != tables.end()) return it->second;
  auto cpu = at::TensorOptions().device(at::kCPU);
  at::Tensor all = at::arange(256, cpu.dtype(at::kInt)).to(at::kByte);
  at::Tensor values = all.view(t).to(at::kFloat);
  // the magnitude code of float32 magnitude bits m: (conversion & 0x7f), or
  // 0x80 for an fnuz NaN (above every other code)
  auto code_of = [&](const at::Tensor& m) {
    at::Tensor c = m.to(at::kInt).view(at::kFloat).to(t).view(at::kByte).to(at::kInt);
    return fnuz(t) ? at::where(at::eq(c, 0x80), 0x80, at::bitwise_and(c, 0x7f)) : at::bitwise_and(c, 0x7f);
  };
  // each code's first magnitude, by bisection over [0, 2^31)
  std::vector<int64_t> starts = {0};
  std::vector<int64_t> codes = {code_of(at::zeros({1}, cpu.dtype(at::kLong))).item<int64_t>()};
  int64_t m = 0;
  const int64_t top = (int64_t(1) << 31) - 1;
  int64_t current = codes[0];
  int64_t last = code_of(at::full({1}, top, cpu.dtype(at::kLong))).item<int64_t>();
  while (current != last) {
    // the first magnitude whose code exceeds `current`
    int64_t lo = m, hi = top;
    while (lo + 1 < hi) {
      int64_t mid = lo + (hi - lo) / 2;
      if (code_of(at::full({1}, mid, cpu.dtype(at::kLong))).item<int64_t>() > current) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    m = hi;
    current = code_of(at::full({1}, m, cpu.dtype(at::kLong))).item<int64_t>();
    starts.push_back(m);
    codes.push_back(current);
  }
  Float8Tables f;
  auto vk = at::TensorOptions().device(at::Device(kDevice, 0));
  f.values = values.to(vk);
  f.thresholds = at::tensor(starts, cpu.dtype(at::kLong)).to(vk);
  f.codes = at::tensor(codes, cpu.dtype(at::kLong)).to(vk);
  return tables.emplace(t, f).first->second;
}

// dst <- src where either is float8 (vk tensors): through float32 and the
// tables. False if neither is.
bool copy_float8(const at::Tensor& dst, const at::Tensor& src) {
  at::ScalarType st = src.scalar_type(), dt = dst.scalar_type();
  if (!is_float8(st) && !is_float8(dt)) return false;
  if (st == dt) return false;  // (a same-dtype copy moves bytes)
  if (is_float8(st) && !at::isFloatingType(dt) && !at::isComplexType(dt)) {
    // (to an integer or bool dtype: each code's value as the CPU converts
    // it directly, NaN included, which a cast through float32 would not
    // give: an fnuz NaN is 0 there)
    static std::mutex mutex;
    static std::map<std::pair<at::ScalarType, at::ScalarType>, at::Tensor> tables;
    at::Tensor table;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto& t = tables[{st, dt}];
      if (!t.defined()) {
        at::Tensor codes = at::arange(256, at::TensorOptions().dtype(at::kInt)).to(at::kByte).view(st);
        t = codes.to(dt).to(at::Device(kDevice, 0));
      }
      table = t;
    }
    at::Tensor r = table.index_select(0, src.view(at::kByte).reshape({-1}).to(at::kLong)).view(src.sizes());
    dst.copy_(r.expand(dst.sizes()));
    return true;
  }
  at::Tensor f;  // the source's values as float32 (as c10: float8 <-> float)
  if (is_float8(st)) {
    f = float8_tables(st).values.index_select(0, src.view(at::kByte).reshape({-1}).to(at::kLong)).view(src.sizes());
  } else {
    f = src.to(at::kFloat);
  }
  if (!is_float8(dt)) {
    dst.copy_(f.expand(dst.sizes()));
    return true;
  }
  const Float8Tables& tb = float8_tables(dt);
  at::Tensor bits = f.contiguous().view(at::kInt).to(at::kLong);
  at::Tensor negative = at::lt(bits, 0);
  at::Tensor mag = at::bitwise_and(bits, 0x7fffffff);
  at::Tensor k = at::searchsorted(tb.thresholds, mag.reshape({-1}), false, true) - 1;
  at::Tensor code = tb.codes.index_select(0, k).view(f.sizes());
  if (fnuz(dt)) {
    code = at::where(at::logical_and(negative, at::logical_and(at::ne(code, 0), at::ne(code, 0x80))),
                     at::bitwise_or(code, 0x80), code);
  } else {
    code = at::where(negative, at::bitwise_or(code, 0x80), code);
  }
  dst.view(at::kByte).copy_(code.to(at::kByte).expand(dst.sizes()));
  return true;
}

// uint16/32/64 have no cast kernels of their own: their bits are those of
// int16/32/64, so conversions go through the signed twins, as the CPU's
// static_casts on x86 (integers keep their low bits; floats truncate through
// int32 for uint16, through int64 for uint32, and from 2^63 up with the top
// bit set for uint64). Returns false if neither side is one of them.
at::ScalarType signed_twin(at::ScalarType s) {
  switch (s) {
    case at::kUInt16: return at::kShort;
    case at::kUInt32: return at::kInt;
    case at::kUInt64: return at::kLong;
    default: return s;
  }
}

bool copy_unsigned(const at::Tensor& dst, const at::Tensor& src) {
  at::ScalarType st = src.scalar_type(), dt = dst.scalar_type();
  if (signed_twin(st) == st && signed_twin(dt) == dt) return false;
  // the source's values in a dtype that holds them (uint64: its int64 bits)
  at::Tensor v = src;
  if (st == at::kUInt16) v = at::bitwise_and(src.view(at::kShort).to(at::kInt), 0xFFFF);
  if (st == at::kUInt32) v = at::bitwise_and(src.view(at::kInt).to(at::kLong), 0xFFFFFFFFLL);
  if (st == at::kUInt64) {
    at::Tensor bits = src.view(at::kLong);
    if (at::isFloatingType(dt) || at::isComplexType(dt)) {  // 2^63 and up: the sign bit as +2^63
      at::ScalarType ft = at::isComplexType(dt) ? at::toRealValueType(dt) : dt;
      at::Tensor low = at::bitwise_and(bits, INT64_MAX).to(ft);
      v = at::where(at::lt(bits, 0), low + 9.223372036854775808e18, low);
    } else {
      v = bits;
    }
  }
  at::ScalarType target = signed_twin(dt);
  if (target != dt && at::isFloatingType(v.scalar_type())) {
    if (dt == at::kUInt16) v = v.to(at::kInt);
    if (dt == at::kUInt32) v = v.to(at::kLong);
    if (dt == at::kUInt64) {
      at::Tensor big = at::ge(v, 9.223372036854775808e18);
      v = at::where(big, at::bitwise_or((v - 9.223372036854775808e18).to(at::kLong), INT64_MIN), v.to(at::kLong));
    }
  }
  at::Tensor out = target != dt ? dst.view(target) : dst;
  out.copy_(v.expand(dst.sizes()));
  return true;
}

// dst (vk) <- src (vk), with broadcasting and dtype conversion.
void copy_vk_to_vk(const at::Tensor& dst, const at::Tensor& src) {
  // (complex32 and bcomplex32: pairs of float16 or bfloat16)
  auto pair16 = [](at::ScalarType t) { return t == at::kComplexHalf || t == at::ScalarType::BComplex32; };
  bool chalf_src = pair16(src.scalar_type()), chalf_dst = pair16(dst.scalar_type());
  if ((chalf_src || chalf_dst) && src.scalar_type() != dst.scalar_type() &&
      (!chalf_src || storage_label(c10::toRealValueType(src.scalar_type())) != nullptr) &&
      (!chalf_dst || storage_label(c10::toRealValueType(dst.scalar_type())) != nullptr)) {
    // complex32 holds two halves (bcomplex32 two bfloat16s): convert its parts.
    at::Tensor s = src.expand(dst.sizes());
    if (chalf_dst) {
      at::Tensor parts = at::view_as_real(dst);
      if (chalf_src) {  // (both pairs: part by part)
        parts.copy_(at::view_as_real(s));
        return;
      }
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
      if (!copy_many_dims(dst, src, kernel.c_str() + kernel.find('_') + 1)) {
        launch_strided(kernel.c_str(), {op::MOVE_COPY}, dst, &src, nullptr);
      }
      return;
    }
  }
  if (copy_unsigned(dst, src)) return;
  if (copy_float8(dst, src)) return;
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
  at::ScalarType st = t.scalar_type();
  if (label == nullptr && (st == at::kComplexHalf || st == at::ScalarType::BComplex32 || st == at::kHalf ||
                           st == at::kBFloat16)) {
    // (no 16-bit kernel: in complex64/float32 on the GPU, through an alias of t
    // without its lazy bits)
    at::Tensor raw = at::empty({0}, t.options()).set_(t.storage(), t.storage_offset(), t.sizes(), t.strides());
    at::Tensor w = raw.to(at::isComplexType(st) ? at::kComplexFloat : at::kFloat);
    if (conj) w = w.conj_physical();
    if (neg) w = w.neg();
    raw.copy_(w);
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
  // (a tensor on a third device, meta say, as FakeTensor's cross-checks pass:
  // the CPU would raise the same, after copying the rest over)
  for (size_t i = 0; i < arguments.size() && any_vk && !copy; ++i) {
    const c10::IValue& v = stack[first + i];
    std::vector<at::Tensor> ts = v.isTensor() ? std::vector<at::Tensor>{v.toTensor()}
                                 : v.isTensorList() ? v.toTensorList().vec() : std::vector<at::Tensor>{};
    for (const at::Tensor& t : ts) {
      if (!t.defined()) continue;
      at::DeviceType d = t.device().type();
      TORCH_CHECK(d == kDevice || d == at::kCPU, "Expected all tensors to be on the same device, but found at least two "
                  "devices, ", at::Device(kDevice, 0), " and ", t.device(), "!");
    }
  }
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

// ---------------------------------------------------------------- 16-bit floats through float32, small ints through int32

namespace {

bool is_small_int(at::ScalarType s) {
  return s == at::kChar || s == at::kByte || s == at::kShort;
}

// Whether the op half_via_float is handling takes complex32 through
// complex64 too (complex32_ops).
thread_local bool g_complex_half = false;

// The ops the CPU computes for complex32: through complex64 on the GPU, the
// result rounded back. (Others raise on the CPU, and still do: their
// complex32 tensors stay the CPU's.)
bool complex32_op(const std::string& name) {
  static const std::set<std::string> ops = {
      "aten::add", "aten::sub", "aten::mul", "aten::eq", "aten::ne", "aten::abs", "aten::isnan", "aten::isinf",
      "aten::isfinite", "aten::neg", "aten::conj_physical", "aten::where", "aten::tril", "aten::triu",
      "aten::masked_fill", "aten::masked_fill_", "aten::index_add", "aten::fill_", "aten::zero_", "aten::sgn",
      "aten::index_select", "aten::roll", "aten::nonzero", "aten::addcmul", "aten::isclose", "aten::normal_",
      "aten::uniform_", "aten::masked_select", "aten::index_put_", "aten::_index_put_impl_", "aten::repeat"};
  return ops.count(name) > 0;
}

// A float16/bfloat16 (widened to float32), int8/uint8/int16 (to int32) or,
// for complex32_op ops, complex32 (to complex64) vk tensor.
bool is_half_vk(const at::Tensor& t) {
  if (!t.defined() || !is_vk(t)) return false;
  at::ScalarType s = t.scalar_type();
  if (s == at::kComplexHalf || s == at::ScalarType::BComplex32) {
    return g_complex_half && storage_label(c10::toRealValueType(s)) != nullptr;
  }
  if (storage_label(s) == nullptr) return false;
  return s == at::kHalf || s == at::kBFloat16 || is_small_int(s);
}

at::ScalarType widened(at::ScalarType s) {
  return is_small_int(s) ? at::kInt : (at::isComplexType(s) ? at::kComplexFloat : at::kFloat);
}

at::Tensor widen_half_arg(const at::Tensor& t) {
  if (!is_half_vk(t)) return t;
  return at::_to_copy(t, t.options().dtype(widened(t.scalar_type())));
}

// Ops whose integer results depend on the dtype's width beyond wrapping
// arithmetic: their small-int tensors stay the CPU's.
bool small_int_on_cpu(const std::string& name) {
  std::string base = name.rfind("aten::", 0) == 0 ? name.substr(6) : name;
  return base.rfind("random", 0) == 0;
}

// Whether the op's integer Scalar arguments (and 0-dim CPU integer tensors)
// fit every small integer dtype among its vk tensors: wrapping arithmetic
// then gives the narrow results from int32 ones, while an out-of-range
// scalar, which the CPU first converts to the narrow dtype (clamp,
// threshold, comparisons), would not.
bool scalars_fit(const c10::FunctionSchema& schema, const torch::jit::Stack& args,
                 const std::vector<at::ScalarType>& narrow) {
  int64_t lo = std::numeric_limits<int64_t>::min(), hi = std::numeric_limits<int64_t>::max();
  for (at::ScalarType s : narrow) {
    lo = std::max<int64_t>(lo, s == at::kChar ? -128 : s == at::kByte ? 0 : -32768);
    hi = std::min<int64_t>(hi, s == at::kChar ? 127 : s == at::kByte ? 255 : 32767);
  }
  auto fits = [&](const c10::IValue& v) {
    if (v.isInt()) return v.toInt() >= lo && v.toInt() <= hi;
    if (v.isTensor()) {
      const at::Tensor& t = v.toTensor();
      if (t.defined() && t.dim() == 0 && t.device().is_cpu() && at::isIntegralType(t.scalar_type(), false)) {
        int64_t x = t.item<int64_t>();
        return x >= lo && x <= hi;
      }
    }
    return true;
  };
  for (size_t i = 0; i < args.size() && i < schema.arguments().size(); ++i) {
    const c10::IValue& v = args[i];
    c10::TypePtr type = schema.arguments()[i].type();
    if (auto opt = type->cast<c10::OptionalType>()) type = opt->getElementType();
    bool scalar = type->kind() == c10::TypeKind::NumberType;  // (Scalar)
    if ((scalar || v.isTensor()) && !fits(v)) return false;
  }
  return true;
}

// Ops left to the CPU even with 16-bit floats (keep in step with _decomp.py's
// _HALF_ON_CPU): linear algebra and FFTs, which the CPU has no 16-bit
// kernels for (widening would make them work where PyTorch says they do
// not); nan_to_num, whose default replacements are the dtype's extremes;
// softshrink/hardshrink, whose threshold the CPU rounds to the dtype.
bool half_on_cpu(const std::string& name) {
  static const char* prefixes[] = {"linalg_", "_linalg_", "cholesky", "lu", "_lu", "triangular_solve", "svd", "_svd",
                                   "geqrf", "ormqr", "orgqr", "qr", "householder", "eig", "_eig", "lstsq", "ldl",
                                   "inverse", "det", "logdet", "slogdet", "matrix_exp", "_fft", "fft_", "stft",
                                   "istft", "nan_to_num", "softshrink", "hardshrink"};
  std::string base = name.rfind("aten::", 0) == 0 ? name.substr(6) : name;
  for (const char* p : prefixes) {
    if (base.rfind(p, 0) == 0) {
      // ("lu" and "eig" prefixes, not substrings: relu, gelu, glu stay)
      return true;
    }
  }
  return false;
}

at::Tensor meta_like(const at::Tensor& t) {
  return t.defined() && is_vk(t) ? at::empty_strided(t.sizes(), t.strides(), t.options().device(at::kMeta)) : t;
}

// Runs `op` with its float16/bfloat16 "vk" tensors widened to float32 (and
// int8/uint8/int16 ones to int32), on the GPU, instead of on the CPU: results
// are rounded once from float32, as CUDA computes most 16-bit ops (small
// integers wrap as they would in their own width). The op first runs on meta tensors of the original
// dtypes, which applies PyTorch's dtype checks and gives the result dtypes.
// False (stack untouched) where this does not apply: no 16-bit vk tensor,
// floating vk tensors of more than one dtype (mixed-dtype errors, which not
// every meta kernel raises, e.g. mm's, stay the CPU's; 0-dim ones do not
// count, as type promotion lets the other operand's dtype win), views, mutated tensor
// lists, or the meta call failing (the CPU then decides).
bool half_via_float(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  if (half_on_cpu(schema.name())) {
    return false;
  }
  g_complex_half = complex32_op(schema.name());
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
  std::vector<at::ScalarType> small_ints;
  bool too_big = false;  // (a widened int32 copy beyond 2 GiB, more than some devices' largest buffer)
  auto note = [&](const at::Tensor& t) {
    if (is_half_vk(t) && is_small_int(t.scalar_type())) {
      small_ints.push_back(t.scalar_type());
      too_big = too_big || t.numel() >= (int64_t(1) << 29);
    }
    if (t.defined() && is_vk(t) && t.dim() > 0 &&
        (at::isFloatingType(t.scalar_type()) || at::isComplexType(t.scalar_type()))) {
      // (complex32 with float16, its parts' type, counts as one: abs of complex32 is float16)
      auto same = [&](at::ScalarType a, at::ScalarType b) {
        return a == b || (g_complex_half && c10::toRealValueType(a) == c10::toRealValueType(b));
      };
      mixed = mixed || (floating.has_value() && !same(*floating, t.scalar_type()));
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
  if (!small_ints.empty() &&
      (too_big || small_int_on_cpu(schema.name()) ||
       !scalars_fit(schema, torch::jit::Stack(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end()),
                    small_ints))) {
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
    if (schema.arguments()[i].name() == "dtype" && v.isInt()) {
      // (a dtype= of a narrow type: the widened one, the result cast back below)
      auto d = static_cast<at::ScalarType>(v.toInt());
      if (is_small_int(d)) v = static_cast<int64_t>(at::kInt);
      if (d == at::kHalf || d == at::kBFloat16) v = static_cast<int64_t>(at::kFloat);
      if ((d == at::kComplexHalf || d == at::ScalarType::BComplex32) && g_complex_half) {
        v = static_cast<int64_t>(at::kComplexFloat);
      }
      continue;
    }
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

// With TORCH_VK_FALLBACK_DETAIL set, fallbacks are counted per argument
// signature ("op [f32 vk 2d, i64 cpu 0d, 3]"), to find the cases a kernel
// declines (tools/pytorch_tests.py records them as usual).
bool fallback_detail() {
  static const bool detail = [] {
    const char* v = std::getenv("TORCH_VK_FALLBACK_DETAIL");
    return v != nullptr && *v != '\0' && std::string(v) != "0";
  }();
  return detail;
}

// TORCH_VK_FALLBACK_TRACE=1 (a debugging aid): each fallback's arguments,
// sizes, strides and (small tensors) values, printed to stderr.
void trace_fallback(const std::string& key, const c10::FunctionSchema& schema, const torch::jit::Stack& stack) {
  static const bool on = [] {
    const char* v = std::getenv("TORCH_VK_FALLBACK_TRACE");
    return v != nullptr && *v != '\0' && std::string(v) != "0";
  }();
  size_t n = schema.arguments().size();
  if (!on || stack.size() < n) return;
  std::ostringstream o;
  o << "torch_vulkan fallback: " << key << "\n";
  for (size_t i = 0; i < n; ++i) {
    const c10::IValue& v = stack[stack.size() - n + i];
    o << "  " << schema.arguments()[i].name() << ": ";
    if (v.isTensor() && v.toTensor().defined()) {
      const at::Tensor& t = v.toTensor();
      o << t.scalar_type() << " " << t.device() << " sizes " << t.sizes();
      if (t.layout() == at::kStrided) o << " strides " << t.strides() << " offset " << t.storage_offset();
      if (t.layout() == at::kStrided && t.numel() <= 16 && t.numel() > 0) o << " values " << t.cpu().reshape({-1});
    } else {
      o << v;
    }
    o << "\n";
  }
  std::cerr << o.str() << std::flush;
}

std::string describe_args(const c10::FunctionSchema& schema, const torch::jit::Stack& stack) {
  size_t n = schema.arguments().size();
  if (stack.size() < n) return "";
  auto tensor = [](const at::Tensor& t) {
    if (!t.defined()) return std::string("undef");
    std::string s = c10::toString(t.scalar_type());
    s += t.is_cpu() ? " cpu " : (t.is_nested() ? " nested " : " ");
    s += std::to_string(t.dim()) + "d";
    if (t.numel() == 0) s += " empty";
    if (t.layout() != at::kStrided) s += " sparse";
    return s;
  };
  std::string out = " [";
  for (size_t i = 0; i < n; ++i) {
    const c10::IValue& v = stack[stack.size() - n + i];
    if (i) out += ", ";
    if (v.isTensor()) {
      out += tensor(v.toTensor());
    } else if (v.isTensorList()) {
      out += "list(";
      auto l = v.toTensorList();
      out += l.size() ? tensor(l.get(0)) : std::string("");
      out += ")";
    } else if (v.isNone()) {
      out += "None";
    } else if (v.isScalar()) {
      out += v.isDouble() ? "float" : v.isComplexDouble() ? "complex" : v.isBool() ? "bool" : "int";
    } else if (v.isString()) {
      out += std::string(v.toStringRef());
    } else {
      out += v.tagKind();
    }
  }
  return out + "]";
}

// The key of cpu_dtype_errors.inc: the op and its arguments' dtypes (tensors,
// "empty" marked), "list" (tensor lists) and kinds (scalars), as
// tools/dev/cpu_dtype_errors.py.
std::string dtype_signature(const std::string& op, const c10::FunctionSchema& schema, const torch::jit::Stack& stack) {
  size_t n = schema.arguments().size();
  std::string out = op + " [";
  for (size_t i = 0; i < n && stack.size() >= n; ++i) {
    const c10::IValue& v = stack[stack.size() - n + i];
    if (i) out += ", ";
    if (v.isTensor()) {
      const at::Tensor& t = v.toTensor();
      out += t.defined() ? std::string(c10::toString(t.scalar_type())) + (t.numel() == 0 ? " empty" : "") : "undef";
    } else if (v.isTensorList()) {
      out += "list";
    } else if (v.isNone()) {
      out += "None";
    } else if (v.isScalar()) {
      out += v.isDouble() ? "float" : v.isComplexDouble() ? "complex" : v.isBool() ? "bool" : "int";
    } else if (v.isString()) {
      out += std::string(v.toStringRef());
    } else {
      out += v.tagKind();
    }
  }
  return out + "]";
}

struct CpuDtypeError {
  const char* signature;
  bool not_implemented;  // NotImplementedError, else RuntimeError
  const char* message;
};

// The CPU's dtype errors (dispatch errors such as '"lerp_kernel_scalar" not
// implemented for 'Int''), by signature: raised here, so the op does not go
// to the CPU only to fail there.
void raise_cpu_dtype_error(const std::string& signature) {
  static const CpuDtypeError table[] = {
#include "cpu_dtype_errors.inc"
      {nullptr, false, nullptr}};
  static const std::map<std::string, const CpuDtypeError*> index = [] {
    std::map<std::string, const CpuDtypeError*> m;
    for (const CpuDtypeError& e : table) {
      if (e.signature != nullptr) m.emplace(e.signature, &e);
    }
    return m;
  }();
  auto it = index.find(signature);
  if (it == index.end()) return;
  // Messages that name a dtype's C++ type ("but got long int instead") hold
  // {c10:Long} in the table: filled in with this platform's name for it (as
  // c10's TypeMeta gives it: __int64 with MSVC), so they read as the CPU's here.
  std::string msg = it->second->message;
  for (size_t at = msg.find("{c10:"); at != std::string::npos; at = msg.find("{c10:", at)) {
    size_t end = msg.find('}', at);
    if (end == std::string::npos) break;
    std::string name = msg.substr(at + 5, end - at - 5), cpp = name;
    for (int i = 0; i < static_cast<int>(at::ScalarType::NumOptions); ++i) {
      auto st = static_cast<at::ScalarType>(i);
      if (name == c10::toString(st)) {
        cpp = std::string(c10::scalarTypeToTypeMeta(st).name());
        break;
      }
    }
    msg.replace(at, end - at + 1, cpp);
    at += cpp.size();
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!it->second->not_implemented, msg);
  TORCH_CHECK(false, msg);
}

// ---------------------------------------------------------------- before the CPU: other ways to run an op

thread_local int g_rescue_depth = 0;  // (each rescue below runs the op again: once)

struct RescueGuard {
  RescueGuard() { ++g_rescue_depth; }
  ~RescueGuard() { --g_rescue_depth; }
};

bool is_written(const c10::Argument& a) {
  return a.alias_info() != nullptr && a.alias_info()->isWrite();
}

// A CPU 0-dim tensor among vk tensors (a Python number, wrapped): moved to the
// device, keeping its wrapped-number flag (so type promotion is unchanged),
// and the op run again, so kernels that take only vk tensors run.
bool rescue_cpu_scalars(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& args = op.schema().arguments();
  size_t n = args.size();
  if (stack->size() < n) return false;
  size_t base = stack->size() - n;
  bool any_vk = false, moved = false;
  for (size_t i = 0; i < n; ++i) {
    const c10::IValue& v = (*stack)[base + i];
    if (v.isTensor() && v.toTensor().defined() && is_vk(v.toTensor())) any_vk = true;
  }
  if (!any_vk) return false;
  torch::jit::Stack retry(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end());
  for (size_t i = 0; i < n; ++i) {
    c10::IValue& v = retry[i];
    if (!is_written(args[i]) && v.isTensor() && v.toTensor().defined() && v.toTensor().is_cpu() &&
        v.toTensor().dim() == 0) {
      const at::Tensor& t = v.toTensor();
      at::Tensor d = t.to(at::Device(kDevice, 0));
      if (t.unsafeGetTensorImpl()->is_wrapped_number()) d.unsafeGetTensorImpl()->set_wrapped_number(true);
      v = d;
      moved = true;
    }
  }
  if (!moved) return false;
  RescueGuard guard;
  op.callBoxed(&retry);
  stack->resize(base);
  stack->insert(stack->end(), retry.begin(), retry.end());
  return true;
}

bool raise_meta_error(const c10::OperatorHandle& op, const torch::jit::Stack& stack);

// An out= op whose functional overload has a vk kernel: that overload, its
// results copied into the out tensors, as TensorIterator's out= casts (an
// out of another dtype the result can be cast to, resized if needed).
bool rescue_out(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto& args = schema.arguments();
  size_t n = args.size();
  if (stack->size() < n) return false;
  size_t base = stack->size() - n;
  size_t first_out = n;
  for (size_t i = 0; i < n; ++i) {
    if (args[i].kwarg_only() && is_written(args[i])) {
      first_out = std::min(first_out, i);
    } else if (first_out < n) {
      return false;  // (outs last)
    }
  }
  size_t nout = n - first_out;
  if (nout == 0) return false;
  std::string overload = schema.overload_name();
  std::string functional;
  if (overload == "out" || overload == "grad_input" || overload == "output") {
    functional = "";
  } else if (overload.size() > 4 && overload.compare(overload.size() - 4, 4, "_out") == 0) {
    functional = overload.substr(0, overload.size() - 4);
  } else {
    return false;
  }
  auto f = c10::Dispatcher::singleton().findOp({schema.name(), functional});
  // (a functional overload with a vk kernel, or a structured one, whose
  // kernel calls this out= overload with an out of the result's dtype)
  bool has_kernel = f.has_value() && (f->hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1) ||
                                      f->hasKernelForDispatchKey(c10::DispatchKey::CompositeExplicitAutogradNonFunctional));
  if (!has_kernel || f->schema().arguments().size() != first_out || f->schema().returns().size() != nout) {
    return false;
  }
  for (size_t i = first_out; i < n; ++i) {
    if (!(*stack)[base + i].isTensor() || !is_vk((*stack)[base + i].toTensor())) return false;
  }
  torch::jit::Stack call(stack->begin() + static_cast<std::ptrdiff_t>(base),
                         stack->begin() + static_cast<std::ptrdiff_t>(base + first_out));
  RescueGuard guard;
  f->callBoxed(&call);
  if (call.size() != nout) return false;
  for (size_t j = 0; j < nout; ++j) {
    if (!call[j].isTensor()) return false;
  }
  // An out of another dtype than the result: cast into only where PyTorch's
  // own checks (the op on meta tensors) take it, with their error otherwise;
  // without a conclusive meta kernel (nonzero, say), the CPU decides.
  bool other_dtype = false;
  for (size_t j = 0; j < nout; ++j) {
    other_dtype = other_dtype ||
                  call[j].toTensor().scalar_type() != (*stack)[base + first_out + j].toTensor().scalar_type();
  }
  if (other_dtype && !raise_meta_error(op, *stack)) return false;
  torch::jit::Stack results;
  for (size_t j = 0; j < nout; ++j) {
    at::Tensor out = (*stack)[base + first_out + j].toTensor();
    const at::Tensor& r = call[j].toTensor();
    TORCH_CHECK(at::canCast(r.scalar_type(), out.scalar_type()), "result type ", r.scalar_type(),
                " can't be cast to the desired output type ", out.scalar_type());
    at::native::resize_output(out, r.sizes());
    out.copy_(r);
    results.push_back(out);
  }
  stack->resize(base);
  stack->insert(stack->end(), results.begin(), results.end());
  return true;
}

// Every tensor input empty: no element to compute. The op on meta tensors
// gives the results' shapes and dtypes (and PyTorch's argument checks); the
// results are empty vk tensors (the CPU's kernels return early on empty
// tensors, whatever their dtype). Written arguments (in place, out=) are
// resized as the meta op resized theirs.
bool rescue_empty(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  size_t n = schema.arguments().size();
  if (stack->size() < n) return false;
  size_t base = stack->size() - n;
  bool any_vk = false;
  for (size_t i = 0; i < n; ++i) {
    const c10::IValue& v = (*stack)[base + i];
    if (v.isTensor() && v.toTensor().defined()) {
      const at::Tensor& t = v.toTensor();
      if (t.layout() != at::kStrided || t.is_nested()) return false;
      if (!is_written(schema.arguments()[i]) && t.numel() != 0) return false;
      any_vk = any_vk || is_vk(t);
    } else if (v.isTensorList()) {
      for (const at::Tensor& t : v.toTensorList()) {
        if (t.defined() && t.numel() != 0) return false;
      }
    }
  }
  if (!any_vk) return false;
  torch::jit::Stack meta(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end());
  for (c10::IValue& v : meta) {
    if (v.isTensor() && v.toTensor().defined()) {
      const at::Tensor& t = v.toTensor();
      v = at::empty_strided(t.sizes(), t.strides(), t.options().device(at::kMeta));
    }
  }
  try {
    op.callBoxed(&meta);
  } catch (const std::exception&) {
    // (an in-place op on an empty tensor, which the meta kernel refuses for
    // its dtype, as normal_ an int tensor: the CPU's kernel returns at once)
    const auto* alias = schema.returns().size() == 1 ? schema.returns()[0].alias_info() : nullptr;
    const auto* self_alias = schema.arguments()[0].alias_info();
    if (alias != nullptr && alias->isWrite() && self_alias != nullptr && self_alias->isWrite() &&
        (*stack)[base].isTensor() && (*stack)[base].toTensor().numel() == 0 && schema.name().back() == '_') {
      at::Tensor self = (*stack)[base].toTensor();
      stack->resize(base);
      stack->push_back(self);
      return true;
    }
    return false;  // (the CPU decides: its error, or a result)
  }
  size_t nr = schema.returns().size();
  if (meta.size() != nr) return false;
  torch::jit::Stack results;
  for (size_t j = 0; j < nr; ++j) {
    const c10::IValue& r = meta[j];
    const auto* alias = schema.returns()[j].alias_info();
    if (!r.isTensor()) return false;
    const at::Tensor& m = r.toTensor();
    if (m.numel() != 0) return false;  // (a result with elements from empty inputs, e.g. a sum: the CPU's)
    if (alias != nullptr && alias->isWrite()) {
      bool found = false;
      for (size_t i = 0; i < n && !found; ++i) {
        const auto* ai = schema.arguments()[i].alias_info();
        if (ai != nullptr && ai->isWrite() && ai->beforeSets() == alias->beforeSets()) {
          at::Tensor t = (*stack)[base + i].toTensor();
          if (!t.sizes().equals(m.sizes())) t.resize_(m.sizes());
          results.push_back(t);
          found = true;
        }
      }
      if (!found) return false;
    } else {
      results.push_back(at::empty_strided(m.sizes(), m.strides(), m.options().device(at::Device(kDevice, 0))));
    }
  }
  stack->resize(base);
  stack->insert(stack->end(), results.begin(), results.end());
  return true;
}

// PyTorch's own argument checks (shapes, out= dtypes, dims...), by running the
// op on meta tensors: an error there is raised as is, the CPU not asked. Errors
// that only say the meta kernel is missing or cannot run are ignored (the CPU
// then decides). Returns whether the meta op ran.
bool raise_meta_error(const c10::OperatorHandle& op, const torch::jit::Stack& stack) {
  const auto& schema = op.schema();
  size_t n = schema.arguments().size();
  if (stack.size() < n) return false;
  torch::jit::Stack meta(stack.end() - static_cast<std::ptrdiff_t>(n), stack.end());
  auto to_meta = [](const at::Tensor& t) {
    if (!t.defined() || (t.is_cpu() && t.dim() == 0)) return t;  // (CPU scalars stay: meta takes them)
    if (t.layout() != at::kStrided || t.is_nested() || t.is_quantized()) throw std::invalid_argument("layout");
    at::Tensor m = at::empty_strided(t.sizes(), t.strides(), t.options().device(at::kMeta));
    if (t.is_conj()) m = m.conj();
    if (t.is_neg()) m._set_neg(true);
    return m;
  };
  try {
    for (c10::IValue& v : meta) {
      if (v.isTensor()) {
        v = to_meta(v.toTensor());
      } else if (v.isTensorList()) {
        c10::List<at::Tensor> l;
        for (const at::Tensor& t : v.toTensorList()) l.push_back(to_meta(t));
        v = l;
      } else if (v.isList()) {
        const auto& elems = v.toListRef();
        for (const c10::IValue& e : elems) {
          if (e.isTensor()) return false;  // (lists of optional tensors: left to the CPU)
        }
      }
    }
  } catch (const std::invalid_argument&) {
    return false;
  }
  try {
    op.callBoxed(&meta);
  } catch (const c10::Error& e) {
    std::string msg = e.what_without_backtrace();
    for (const char* skip : {"Meta", "meta", "Could not run", "data-dependent", "data dependent", "fake", "symbolic",
                             "SymInt", "not implemented for", "not supported", "unimplemented"}) {
      if (msg.find(skip) != std::string::npos) return false;
    }
    throw;
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

// Every input empty (or a 0-d tensor whose dtype float32 holds exactly), in a
// dtype without a vk kernel for the op (16-bit floats, integers, bools for
// linalg, say): the op in float32 (complex64 for complex dtypes), as nothing
// is computed from the inputs' values; its results of that dtype converted
// back (an identity permutation, zeros), written arguments copied into.
bool rescue_empty_cast(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const auto& args = schema.arguments();
  size_t n = args.size();
  if (stack->size() < n) return false;
  size_t base = stack->size() - n;
  at::ScalarType main = at::ScalarType::Undefined;
  for (size_t i = 0; i < n && main == at::ScalarType::Undefined; ++i) {
    const c10::IValue& v = (*stack)[base + i];
    if (v.isTensor() && v.toTensor().defined()) main = v.toTensor().scalar_type();
  }
  if (main == at::ScalarType::Undefined || main == at::kFloat || main == at::kComplexFloat) return false;
  if (!at::isFloatingType(main) && !at::isComplexType(main)) {
    // (integers and bools: only ops the CPU computes for them, whatever their emptiness)
    const std::string& op_name = schema.name();
    bool known = false;
    for (const char* p : {"aten::linalg_lu", "aten::lu_unpack", "aten::mm", "aten::bmm", "aten::baddbmm",
                          "aten::addmm", "aten::addbmm", "aten::mv", "aten::dot", "aten::vdot"}) {
      known = known || op_name == p || (op_name.rfind(p, 0) == 0 && op_name.size() > strlen(p) && op_name[strlen(p)] == '_');
    }
    if (!known) return false;
  }
  at::ScalarType target = at::isComplexType(main) ? at::kComplexFloat : at::kFloat;
  bool exact = main == at::kBool || main == at::kByte || main == at::kChar || main == at::kShort ||
               main == at::kHalf || main == at::kBFloat16 || main == at::kComplexHalf;
  // Written arguments of the op's dtype: all in float32 (P, L, U), else only the
  // first (LU, where the pivots and info also have the integer dtype).
  torch::jit::Stack retry;
  bool ran = false;
  for (int variant = 0; variant < 2 && !ran; ++variant) {
    retry.assign(stack->begin() + static_cast<std::ptrdiff_t>(base), stack->end());
    bool any_vk = false, first_written = true;
    for (size_t i = 0; i < n; ++i) {
      c10::IValue& v = retry[i];
      if (v.isTensorList()) {
        for (const at::Tensor& t : v.toTensorList()) {
          if (t.defined() && t.numel() != 0) return false;
        }
        continue;
      }
      if (!v.isTensor() || !v.toTensor().defined()) continue;
      at::Tensor t = v.toTensor();
      if (t.layout() != at::kStrided || t.is_nested()) return false;
      any_vk = any_vk || is_vk(t);
      if (is_written(args[i])) {
        if (t.scalar_type() == main && (variant == 0 || first_written)) {
          v = at::empty(t.sizes(), t.options().dtype(target));
        }
        first_written = false;
        continue;
      }
      if (t.numel() != 0 && !(t.dim() == 0 && exact)) return false;
      if (t.scalar_type() == main) v = t.to(target);
    }
    if (!any_vk) return false;
    RescueGuard guard;
    try {
      op.callBoxed(&retry);
      ran = true;
    } catch (const c10::Error&) {
      // (the other variant, or the CPU decides)
    }
  }
  if (!ran) return false;
  const auto& returns = schema.returns();
  if (retry.size() != returns.size()) return false;
  torch::jit::Stack results;
  for (size_t j = 0; j < returns.size(); ++j) {
    const auto* alias = returns[j].alias_info();
    const c10::IValue& rv = retry[j];
    if (alias != nullptr && alias->isWrite()) {
      bool found = false;
      for (size_t i = 0; i < n && !found; ++i) {
        const auto* ai = args[i].alias_info();
        if (ai == nullptr || !ai->isWrite() || ai->beforeSets() != alias->beforeSets()) continue;
        at::Tensor orig = (*stack)[base + i].toTensor();
        const at::Tensor& r = rv.toTensor();
        if (!orig.is_same(r)) {
          if (!orig.sizes().equals(r.sizes())) orig.resize_(r.sizes());
          orig.copy_(r);
        }
        results.push_back(orig);
        found = true;
      }
      if (!found) return false;
    } else if (rv.isTensor() && rv.toTensor().defined() && rv.toTensor().scalar_type() == target) {
      results.push_back(rv.toTensor().to(main));
    } else {
      results.push_back(rv);
    }
  }
  stack->resize(base);
  stack->insert(stack->end(), results.begin(), results.end());
  return true;
}

void run_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool error_on_views) {
  const auto& name = op.schema().operator_name();
  std::string key = name.overload_name.empty() ? name.name : name.name + "." + name.overload_name;
  // (first: where the CPU raises, no other route may compute a result)
  raise_cpu_dtype_error(dtype_signature(key, op.schema(), *stack));  // (keyed with emptiness too)
  check_out_devices(op.schema(), *stack);  // (the device errors raised here: no CPU involved, not a fallback)
  // (a meta tensor that reached a vk kernel, as FakeTensor's cross-checks run
  // kernels on: the error the CPU route would give, copying it over)
  for (const c10::IValue& v : torch::jit::Stack(stack->end() - static_cast<std::ptrdiff_t>(
                                                                std::min(stack->size(), op.schema().arguments().size())),
                                                stack->end())) {
    TORCH_CHECK_NOT_IMPLEMENTED(!(v.isTensor() && v.toTensor().defined() && v.toTensor().is_meta()),
                                "Cannot copy out of meta tensor; no data!");
  }
  if (half_via_float(op, stack)) {
    return;
  }
  // (rescue_out's functional overload may call this op again: it is tried once; the
  // others cannot recur, as their retry has no CPU scalar, runs on meta, or in float32)
  if ((g_rescue_depth == 0 && (rescue_cpu_scalars(op, stack) || rescue_out(op, stack))) ||
      (g_rescue_depth < 4 && (rescue_empty(op, stack) || rescue_empty_cast(op, stack)))) {
    return;
  }
  if (fallback_detail()) key += describe_args(op.schema(), *stack);
  raise_meta_error(op, *stack);
  trace_fallback(key, op.schema(), *stack);
  TORCH_CHECK_NOT_IMPLEMENTED(op.hasComputedKernelForDispatchKey(c10::DispatchKey::CPU), "Could not run '",
                              op.schema().name(), "' with arguments from the 'vk' backend: it has no vk or CPU kernel.");
  // Counted and warned about only once the CPU has computed something: where
  // it only raises (a dtype or argument check, say), the call is counted
  // apart as "op (raised)", silently, and its error is the user's. Under
  // TORCH_VK_STRICT the CPU runs on copies first, so that its own error
  // still reaches the user; only a call it would compute raises
  // NotImplementedError.
  if (strict_mode()) {
    strict_probe(op, *stack, key);
  }
  try {
    at::native::cpu_fallback(op, stack, error_on_views);
  } catch (...) {
    count_fallback(key + " (raised)", false);
    throw;
  }
  count_fallback(key, true);
}

void count_fallback(const std::string& op, bool warn) {
  std::lock_guard<std::mutex> lock(g_fallback_mutex);
  if (g_fallbacks[op]++ == 0 && warn) {
    TORCH_WARN("torch_vulkan: ", op, " has no Vulkan kernel yet; running it on the CPU");
  }
}

void record_fallback(const std::string& op) {
  TORCH_CHECK_NOT_IMPLEMENTED(!strict_mode(), "torch_vulkan: ", op,
                              " has no Vulkan kernel (TORCH_VK_STRICT forbids running it on the CPU)");
  count_fallback(op, true);
}

// TORCH_VK_STRICT: the op on CPU copies of the vk tensors (so nothing the
// caller holds changes): its error if it raises, else NotImplementedError.
void strict_probe(const c10::OperatorHandle& op, const torch::jit::Stack& stack, const std::string& key) {
  size_t n = std::min(stack.size(), op.schema().arguments().size());
  auto host = [](const c10::IValue& v) -> c10::IValue {
    if (v.isTensor()) {
      const at::Tensor& t = v.toTensor();
      return t.defined() && is_vk(t) ? c10::IValue(t.cpu()) : v;
    }
    if (v.isTensorList()) {
      std::vector<at::Tensor> ts;
      for (const at::Tensor& t : v.toTensorVector()) ts.push_back(t.defined() && is_vk(t) ? t.cpu() : t);
      return c10::IValue(ts);
    }
    if (v.isList()) {
      c10::impl::GenericList out(v.toList().elementType());
      for (const c10::IValue& e : v.toList()) {
        out.push_back(e.isTensor() && e.toTensor().defined() && is_vk(e.toTensor()) ? c10::IValue(e.toTensor().cpu()) : e);
      }
      return c10::IValue(out);
    }
    return v;
  };
  torch::jit::Stack cpu;
  for (size_t i = stack.size() - n; i < stack.size(); ++i) cpu.push_back(host(stack[i]));
  try {
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), &cpu);
  } catch (...) {
    count_fallback(key + " (raised)", false);
    throw;
  }
  TORCH_CHECK_NOT_IMPLEMENTED(false, "torch_vulkan: ", key,
                              " has no Vulkan kernel (TORCH_VK_STRICT forbids running it on the CPU)");
}

}  // namespace vk

// For Python-registered kernels (torch_vulkan/_decomp.py) that hand an op
// to the CPU: counted and warned about like the C++ fallbacks. Returns 1 in
// strict mode (TORCH_VK_STRICT), where the caller raises instead.
extern "C" int vkt_record_fallback(const char* op) {
  try {
    std::string name(op);
    const std::string raised = " (raised)";
    if (name.size() > raised.size() && name.compare(name.size() - raised.size(), raised.size(), raised) == 0) {
      vk::count_fallback(name, false);  // (the CPU only raised: counted, silently, strict or not)
      return 0;
    }
    vk::record_fallback(name);
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

// Whether `from`'s data can be shallow-copied into `self` (param.data = x,
// as module.to("vk") does): for two dense tensors, a vk and a CPU (or vk)
// one, as for CUDA (TensorImpl's own rule lists only the built-in dense
// backends; PyTorch asks PrivateUse1 backends to override this op).
bool compatible_shallow_copy_type(const at::Tensor& self, const at::Tensor& from) {
  auto dense = [](const at::Tensor& t) {
    return t.layout() == at::kStrided && (t.device().is_cpu() || t.device().type() == kDevice);
  };
  if (dense(self) && dense(from)) return true;
  return at::native::_has_compatible_shallow_copy_type(self, from);
}

// A zero tensor (autograd's efficient zeros): no storage, the ZeroTensor
// key, as the CPU's and CUDA's.
at::Tensor efficient_zero_tensor(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                                 std::optional<at::Layout> /*layout*/, std::optional<at::Device> device,
                                 std::optional<bool> /*pin_memory*/) {
  auto allocator = at::native::ZeroTensorAllocator(device.value_or(at::Device(kDevice, 0)));
  auto keys = at::DispatchKeySet(c10::DispatchKey::PrivateUse1) | at::DispatchKeySet(c10::DispatchKey::ZeroTensor);
  return at::detail::empty_generic(size, &allocator, keys, dtype.value_or(c10::get_default_dtype_as_scalartype()),
                                   std::nullopt);
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
  m.impl("_efficientzerotensor", vk::efficient_zero_tensor);
  m.impl("_has_compatible_shallow_copy_type", vk::compatible_shallow_copy_type);
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
