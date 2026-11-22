// Strided element-wise launches: the push-constant layouts of
// shaders/_strided.slang (out, a, b) and shaders/_strided4.slang (out, a,
// b, c), planned with strided.h.

#include "opcodes.h"
#include "strided.h"
#include "vk.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <string_view>

namespace vk {

namespace {

// K element pointers, 6 - K raw parameter slots, then up to D = 12 / K
// merged dims: sizes and each operand's element strides (st[k][d]).
template <size_t K>
struct StridedLayout {
  static constexpr size_t D = 12 / K;
  uint64_t ptr[K];
  uint64_t param[6 - K];
  uint32_t n, ndim, stride, pad;
  uint32_t size[4];
  int32_t st[K][D];
};
static_assert(sizeof(StridedLayout<3>) == 128 && offsetof(StridedLayout<3>, size) == 64,
              "must match shaders/_strided.slang");
static_assert(sizeof(StridedLayout<4>) == 128 && offsetof(StridedLayout<4>, size) == 64,
              "must match shaders/_strided4.slang");

}  // namespace

uint64_t pack_param(const c10::Scalar& s, at::ScalarType t) {
  if (at::isIntegralType(t, /*includeBool=*/true)) {
    return static_cast<uint64_t>(s.toLong());
  }
  if (t == at::kDouble || t == at::kComplexDouble) {
    double v = s.toDouble();
    uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
  }
  float v = static_cast<float>(s.toDouble());
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

namespace {

// Kernels on the _strided layouts that touch each operand only at its own
// strided offsets (out[o.o] written; a[o.a], b[o.b], c[o.c] read; checked in
// each shader): for these, launch() declares to the stream exactly the bytes
// each operand spans in each dispatch, so independent element-wise ops, and
// the dispatches of the >4-dim loop, need no barrier between them. move's
// gather and scatter index with data and are left to whole allocations.
bool exact_operands(const char* kernel, const std::vector<uint32_t>& spec) {
  std::string_view name(kernel);
  std::string_view family = name.substr(0, name.find('_'));
  if (family == "move") {
    return !spec.empty() && (spec[0] == op::MOVE_COPY || spec[0] == op::MOVE_FILL);
  }
  for (std::string_view f : {"unary", "binary", "cunary", "cbinary", "cabs", "cast", "compare", "special", "ternary",
                             "where", "iota", "index"}) {
    if (family == f) return true;
  }
  return false;
}

// The bytes a strided operand spans: elements at ptr + itemsize * sum_d(i_d *
// stride[d]) for 0 <= i_d < size[d].
VktRange strided_range(uint64_t ptr, int64_t itemsize, size_t ndim, const uint32_t* size, const int32_t* stride,
                       bool write) {
  int64_t lo = 0, hi = 0;
  for (size_t d = 0; d < ndim; ++d) {
    int64_t reach = (static_cast<int64_t>(size[d]) - 1) * stride[d];
    (reach < 0 ? lo : hi) += reach;
  }
  return VktRange{ptr, ptr + static_cast<uint64_t>(lo * itemsize), ptr + static_cast<uint64_t>((hi + 1) * itemsize),
                  write ? 1u : 0u, 0};
}

// Launches `kernel` over `dims` (planned by strided.h), operand k at GPU
// address base[k] (0: absent) with elements of itemsize[k] bytes.
template <size_t K>
void launch_dims(const char* kernel, const std::vector<uint32_t>& spec, const StridedDims<K>& dims,
                 const std::array<uint64_t, K>& base, const std::array<int64_t, K>& itemsize,
                 c10::ArrayRef<uint64_t> bits) {
  StridedLayout<K> p{};
  constexpr size_t D = StridedLayout<K>::D;
  TORCH_CHECK(bits.size() <= std::size(p.param), "torch_vulkan: too many kernel parameters for ", kernel);
  std::copy(bits.begin(), bits.end(), p.param);
  size_t inner = std::min(dims.ndim(), D);
  p.ndim = static_cast<uint32_t>(inner);
  for (size_t d = 0; d < D; ++d) {
    p.size[d] = d < inner ? static_cast<uint32_t>(dims.size[d]) : 1;
    for (size_t k = 0; k < K; ++k) {
      p.st[k][d] = d < inner ? static_cast<int32_t>(dims.stride[d][k]) : 0;
    }
  }
  bool exact = exact_operands(kernel, spec);
  std::vector<VktRange> ranges;
  for_each_dispatch<K>(dims, D, [&](int64_t count, const std::array<int64_t, K>& offsets) {
    p.n = static_cast<uint32_t>(count);
    uint32_t groups = 0;
    check(vkt_grid_1d(static_cast<uint64_t>(count), &groups, &p.stride));
    ranges.clear();
    for (size_t k = 0; k < K; ++k) {
      p.ptr[k] = base[k] != 0 ? base[k] + offsets[k] * itemsize[k] : 0;
      if (exact && p.ptr[k] != 0) {
        ranges.push_back(strided_range(p.ptr[k], itemsize[k], inner, p.size, p.st[k], k == 0));
      }
    }
    dispatch_declared(kernel, spec, p, ranges, groups);
  });
}

template <size_t K>
void launch(const char* kernel, const std::vector<uint32_t>& spec, const std::array<const at::Tensor*, K>& ops,
            c10::ArrayRef<uint64_t> bits) {
  const at::Tensor& out = *ops[0];
  if (out.numel() == 0) {
    return;
  }
  std::array<uint64_t, K> base{};
  std::array<int64_t, K> itemsize{};
  for (size_t k = 0; k < K; ++k) {
    base[k] = !ops[k] ? 0 : k == 0 ? gpu_address_out(*ops[k]) : gpu_address(*ops[k]);
    itemsize[k] = ops[k] ? ops[k]->itemsize() : 0;
  }
  launch_dims<K>(kernel, spec, plan_operands<K>(out, ops), base, itemsize, bits);
}

struct Copy8Layout {
  uint64_t out, a;
  uint32_t n, ndim, stride, pad;
  uint32_t size[8];
  int32_t so[8], sa[8];
};
static_assert(sizeof(Copy8Layout) == 128 && offsetof(Copy8Layout, size) == 32 && offsetof(Copy8Layout, sa) == 96,
              "must match shaders/copy8.slang");

std::vector<uint64_t> pack_params(c10::ArrayRef<c10::Scalar> params, at::ScalarType t) {
  std::vector<uint64_t> bits;
  for (const c10::Scalar& s : params) {
    bits.push_back(pack_param(s, t));
  }
  return bits;
}

}  // namespace

void launch_strided(const char* kernel, std::vector<uint32_t> spec, const at::Tensor& out, const at::Tensor* a,
                    const at::Tensor* b, c10::ArrayRef<c10::Scalar> params) {
  launch<3>(kernel, spec, {&out, a, b}, pack_params(params, (a ? *a : out).scalar_type()));
}

namespace {

// dst = src over `dims` (planned by strided.h; strides may be negative) in
// one dispatch of shaders/copy8.slang, operands at GPU addresses out and a;
// false where it does not apply (not 5 to 8 dims, or past 32-bit offsets).
bool copy8_dims(const StridedDims<2>& dims, uint64_t out, uint64_t a, int64_t itemsize, const char* size_label) {
  if (dims.ndim() <= 4 || dims.ndim() > 8) {
    return false;
  }
  int64_t count = 1;
  std::array<int64_t, 2> span{};
  for (size_t d = 0; d < dims.ndim(); ++d) {
    count *= dims.size[d];
    for (size_t k = 0; k < 2; ++k) span[k] += (dims.size[d] - 1) * std::abs(dims.stride[d][k]);
  }
  if (count >= (int64_t(1) << 32) || span[0] >= (int64_t(1) << 31) || span[1] >= (int64_t(1) << 31)) {
    return false;
  }
  Copy8Layout p{};
  p.out = out;
  p.a = a;
  p.n = static_cast<uint32_t>(count);
  p.ndim = static_cast<uint32_t>(dims.ndim());
  for (size_t d = 0; d < 8; ++d) {
    bool used = d < dims.ndim();
    p.size[d] = used ? static_cast<uint32_t>(dims.size[d]) : 1;
    p.so[d] = used ? static_cast<int32_t>(dims.stride[d][0]) : 0;
    p.sa[d] = used ? static_cast<int32_t>(dims.stride[d][1]) : 0;
  }
  uint32_t groups = 0;
  check(vkt_grid_1d(static_cast<uint64_t>(count), &groups, &p.stride));
  // (copy8.slang writes out[o] and reads a[a] only: exact ranges, as launch())
  std::vector<VktRange> ranges = {strided_range(p.out, itemsize, dims.ndim(), p.size, p.so, true),
                                  strided_range(p.a, itemsize, dims.ndim(), p.size, p.sa, false)};
  dispatch_declared(std::string("copy8_") + size_label, {}, p, ranges, groups);
  return true;
}

}  // namespace

bool copy_many_dims(const at::Tensor& dst, const at::Tensor& src, const char* size_label) {
  if (dst.numel() == 0) {
    return false;
  }
  StridedDims<2> dims = plan_operands<2>(dst, {&dst, &src});
  return copy8_dims(dims, gpu_address_out(dst), gpu_address(src), dst.itemsize(), size_label);
}

void copy_reversed(const at::Tensor& dst, const at::Tensor& src, const std::vector<int64_t>& dims) {
  TORCH_CHECK(dst.sizes() == src.sizes() && dst.itemsize() == src.itemsize(),
              "torch_vulkan: copy_reversed needs operands of one shape and element size");
  if (dst.numel() == 0) {
    return;
  }
  std::string kernel = move_kernel(src.itemsize());
  TORCH_CHECK(!kernel.empty(), "torch_vulkan: no copy kernel for elements of ", src.itemsize(), " bytes");
  std::vector<int64_t> strides = src.strides().vec();
  int64_t start = 0;  // (src's element offset of the first element read)
  for (int64_t d : dims) {
    start += (src.size(d) - 1) * strides[d];
    strides[d] = -strides[d];
  }
  int64_t itemsize = src.itemsize();
  uint64_t out = gpu_address_out(dst), a = gpu_address(src) + start * itemsize;
  if (!copy8_dims(merge_dims<2>(dst.sizes(), {dst.strides().vec(), strides}), out, a, itemsize,
                  kernel.c_str() + kernel.find('_') + 1)) {
    // (move.slang's layout is _strided.slang's three operands: b absent)
    std::vector<int64_t> none(dst.dim(), 0);
    launch_dims<3>(kernel.c_str(), {op::MOVE_COPY}, merge_dims<3>(dst.sizes(), {dst.strides().vec(), strides, none}),
                   {out, a, 0}, {itemsize, itemsize, 0}, {});
  }
}

void launch_strided_bits(const char* kernel, std::vector<uint32_t> spec, const at::Tensor& out, const at::Tensor* a,
                         const at::Tensor* b, c10::ArrayRef<uint64_t> bits) {
  launch<3>(kernel, spec, {&out, a, b}, bits);
}

void launch_strided4(const char* kernel, std::vector<uint32_t> spec, const at::Tensor& out, const at::Tensor* a,
                     const at::Tensor* b, const at::Tensor* c, c10::ArrayRef<c10::Scalar> params) {
  launch<4>(kernel, spec, {&out, a, b, c}, pack_params(params, out.scalar_type()));
}

std::string move_kernel(int64_t itemsize) {
  switch (itemsize) {
    case 1: return has_feature(kFeatureInt8) ? "move_b1" : "";
    case 2: return has_feature(kFeatureInt16) ? "move_b2" : "";
    case 4: return "move_b4";
    case 8: return "move_b8";
    case 16: return "move_b16";
    default: return "";
  }
}

}  // namespace vk
