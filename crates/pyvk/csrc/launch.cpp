// Strided element-wise launches: the push-constant layouts of
// shaders/_strided.slang (out, a, b) and shaders/_strided4.slang (out, a,
// b, c), planned with strided.h.

#include "strided.h"
#include "vk.h"

#include <array>
#include <cstddef>
#include <cstring>

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

template <size_t K>
void launch(const char* kernel, const std::vector<uint32_t>& spec, const std::array<const at::Tensor*, K>& ops,
            c10::ArrayRef<uint64_t> bits) {
  const at::Tensor& out = *ops[0];
  if (out.numel() == 0) {
    return;
  }
  StridedLayout<K> p{};
  constexpr size_t D = StridedLayout<K>::D;
  TORCH_CHECK(bits.size() <= std::size(p.param), "torch_vulkan: too many kernel parameters for ", kernel);
  std::copy(bits.begin(), bits.end(), p.param);
  StridedDims<K> dims = plan_operands<K>(out, ops);
  size_t inner = std::min(dims.ndim(), D);
  p.ndim = static_cast<uint32_t>(inner);
  for (size_t d = 0; d < D; ++d) {
    p.size[d] = d < inner ? static_cast<uint32_t>(dims.size[d]) : 1;
    for (size_t k = 0; k < K; ++k) {
      p.st[k][d] = d < inner ? static_cast<int32_t>(dims.stride[d][k]) : 0;
    }
  }
  std::array<uint64_t, K> base{};
  std::array<int64_t, K> itemsize{};
  for (size_t k = 0; k < K; ++k) {
    base[k] = !ops[k] ? 0 : k == 0 ? gpu_address_out(*ops[k]) : gpu_address(*ops[k]);
    itemsize[k] = ops[k] ? ops[k]->itemsize() : 0;
  }
  for_each_dispatch<K>(dims, D, [&](int64_t count, const std::array<int64_t, K>& offsets) {
    p.n = static_cast<uint32_t>(count);
    uint32_t groups = 0;
    check(vkt_grid_1d(static_cast<uint64_t>(count), &groups, &p.stride));
    for (size_t k = 0; k < K; ++k) {
      p.ptr[k] = ops[k] ? base[k] + offsets[k] * itemsize[k] : 0;
    }
    dispatch(kernel, spec, p, groups);
  });
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

bool copy_many_dims(const at::Tensor& dst, const at::Tensor& src, const char* size_label) {
  StridedDims<2> dims = plan_operands<2>(dst, {&dst, &src});
  if (dims.ndim() <= 4 || dims.ndim() > 8 || dst.numel() == 0) {
    return false;
  }
  int64_t count = 1;
  std::array<int64_t, 2> span{};
  for (size_t d = 0; d < dims.ndim(); ++d) {
    count *= dims.size[d];
    for (size_t k = 0; k < 2; ++k) {
      if (dims.stride[d][k] < 0) {
        return false;
      }
      span[k] += (dims.size[d] - 1) * dims.stride[d][k];
    }
  }
  if (count >= (int64_t(1) << 32) || span[0] >= (int64_t(1) << 31) || span[1] >= (int64_t(1) << 31)) {
    return false;
  }
  Copy8Layout p{};
  p.out = gpu_address_out(dst);
  p.a = gpu_address(src);
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
  dispatch(std::string("copy8_") + size_label, {}, p, groups);
  return true;
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
