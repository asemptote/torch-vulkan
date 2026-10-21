// FFTs for device "vk": _fft_c2c, _fft_r2c, _fft_c2r (everything in
// torch.fft reduces to these), complex64 and complex128.
//
// A multi-dim transform is a sequence of 1-D transforms along each dim; each
// works on contiguous rows [batch, n] (the dim moved last):
// - n a power of two: radix-2 Stockham passes (shaders/fft.slang, PHASE 0)
//   with twiddles exp(-2 pi i t / n) computed on the host in float64;
// - any other n: Bluestein's algorithm, a convolution with the chirp
//   w_k = exp(-+ i pi k^2 / n) done with power-of-two FFTs of length >= 2n - 1.
// r2c transforms the input as complex and keeps n/2 + 1 outputs; c2r rebuilds
// the Hermitian-symmetric spectrum (fft.slang PHASE 1), inverts it and keeps
// the real part. Twiddle and chirp tables are cached per (n, dtype).

#include "vk.h"

#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <tuple>

namespace vk {
namespace {

struct FftParams {
  uint64_t in, out, table;
  uint32_t n, ns, batch, m, ngroups, pad;
};
static_assert(sizeof(FftParams) == 48, "must match shaders/fft.slang");

constexpr double kPi = 3.14159265358979323846;

bool is_pow2(int64_t n) {
  return n > 0 && (n & (n - 1)) == 0;
}

uint32_t groups_for(int64_t threads) {
  return static_cast<uint32_t>(std::clamp<int64_t>((threads + 255) / 256, 1, 65535));
}

std::mutex g_cache_mutex;
std::map<std::pair<int64_t, at::ScalarType>, at::Tensor> g_twiddles;
std::map<std::tuple<int64_t, at::ScalarType, bool>, std::pair<at::Tensor, at::Tensor>> g_bluestein;

at::Tensor to_device(const at::Tensor& host, at::ScalarType dtype) {
  return host.to(dtype).to(at::Device(kDevice, 0));
}

// exp(-2 pi i t / n) for t < n/2, on the device.
at::Tensor twiddles(int64_t n, at::ScalarType dtype) {
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  auto key = std::make_pair(n, dtype);
  auto it = g_twiddles.find(key);
  if (it != g_twiddles.end()) {
    return it->second;
  }
  at::Tensor host = at::empty({std::max<int64_t>(n / 2, 1)}, at::kComplexDouble);
  auto* w = host.data_ptr<c10::complex<double>>();
  for (int64_t t = 0; t < host.numel(); ++t) {
    double angle = -2.0 * kPi * static_cast<double>(t) / static_cast<double>(n);
    w[t] = {std::cos(angle), std::sin(angle)};
  }
  at::Tensor table = to_device(host, dtype);
  g_twiddles.emplace(key, table);
  return table;
}

// Unnormalized FFT of contiguous rows x [batch, n], n a power of two.
at::Tensor fft_pow2(const at::Tensor& x, int64_t batch, int64_t n, bool inverse) {
  if (n == 1) {
    return x.clone();
  }
  at::Tensor table = twiddles(n, x.scalar_type());
  std::string kernel = std::string("fft_") + dtype_label(x.scalar_type());
  // Rows that fit in shared memory (fft.slang LOCAL_MAX) take one pass.
  int64_t local_max = x.scalar_type() == at::kComplexFloat ? 2048 : 1024;
  if (n <= local_max) {
    at::Tensor out = at::empty_like(x);
    FftParams p{};
    p.in = gpu_address(x);
    p.out = gpu_address(out);
    p.table = gpu_address(table);
    p.n = static_cast<uint32_t>(n);
    p.batch = static_cast<uint32_t>(batch);
    p.ngroups = static_cast<uint32_t>(std::min<int64_t>(batch, 65535));
    dispatch(kernel, {2, inverse ? 1u : 0u}, p, p.ngroups);
    return out;
  }
  at::Tensor src = x, buffers[2] = {at::empty_like(x), at::empty_like(x)};
  int which = 0;
  for (int64_t ns = 1; ns < n; ns *= 2) {
    at::Tensor& dst = buffers[which];
    FftParams p{};
    p.in = gpu_address(src);
    p.out = gpu_address(dst);
    p.table = gpu_address(table);
    p.n = static_cast<uint32_t>(n);
    p.ns = static_cast<uint32_t>(ns);
    p.batch = static_cast<uint32_t>(batch);
    p.ngroups = groups_for(batch * n / 2);
    dispatch(kernel, {0, inverse ? 1u : 0u}, p, p.ngroups);
    src = dst;
    which ^= 1;
  }
  return src;
}

// (chirp w [n], FFT of the convolution kernel b [M]) for Bluestein of size n.
std::pair<at::Tensor, at::Tensor> bluestein_tables(int64_t n, int64_t m, at::ScalarType dtype, bool inverse) {
  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    auto it = g_bluestein.find({n, dtype, inverse});
    if (it != g_bluestein.end()) {
      return it->second;
    }
  }
  at::Tensor w_host = at::empty({n}, at::kComplexDouble);
  at::Tensor b_host = at::zeros({m}, at::kComplexDouble);
  auto* w = w_host.data_ptr<c10::complex<double>>();
  auto* b = b_host.data_ptr<c10::complex<double>>();
  double sign = inverse ? 1.0 : -1.0;
  for (int64_t k = 0; k < n; ++k) {
    // k^2 mod 2n keeps the angle exact for large k.
    int64_t k2 = static_cast<int64_t>((static_cast<uint64_t>(k) * static_cast<uint64_t>(k)) % (2 * static_cast<uint64_t>(n)));
    double angle = sign * kPi * static_cast<double>(k2) / static_cast<double>(n);
    w[k] = {std::cos(angle), std::sin(angle)};
    b[k] = std::conj(w[k]);
    if (k > 0) {
      b[m - k] = std::conj(w[k]);
    }
  }
  at::Tensor chirp = to_device(w_host, dtype);
  at::Tensor kernel = fft_pow2(to_device(b_host, dtype).view({1, m}), 1, m, false).view({m});
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  auto entry = std::make_pair(chirp, kernel);
  g_bluestein.emplace(std::make_tuple(n, dtype, inverse), entry);
  return entry;
}

// Unnormalized FFT of contiguous rows x [batch, n].
at::Tensor fft_rows(const at::Tensor& x, bool inverse) {
  int64_t batch = x.size(0), n = x.size(1);
  if (is_pow2(n)) {
    return fft_pow2(x, batch, n, inverse);
  }
  int64_t m = 1;
  while (m < 2 * n - 1) {
    m *= 2;
  }
  auto [chirp, kernel] = bluestein_tables(n, m, x.scalar_type(), inverse);
  at::Tensor a = at::zeros({batch, m}, x.options());
  a.narrow(1, 0, n).copy_(x * chirp);
  at::Tensor spectrum = fft_pow2(a, batch, m, false);
  spectrum.mul_(kernel);
  at::Tensor conv = fft_pow2(spectrum, batch, m, true);
  return (conv.narrow(1, 0, n) * chirp).mul_(1.0 / static_cast<double>(m));
}

// Unnormalized FFT of `x` along `dim`.
at::Tensor fft_dim(const at::Tensor& x, int64_t dim, bool inverse) {
  at::Tensor moved = x.movedim(dim, -1).contiguous();
  int64_t n = moved.size(-1);
  if (moved.numel() == 0 || n == 0) {
    return moved.movedim(-1, dim);
  }
  at::Tensor rows = moved.view({-1, n});
  return fft_rows(rows, inverse).view(moved.sizes()).movedim(-1, dim);
}

// fft_norm_mode: 0 none, 1 1/sqrt(n), 2 1/n.
at::Tensor& normalize(at::Tensor& t, int64_t mode, int64_t n) {
  if (mode == 1) {
    t.mul_(1.0 / std::sqrt(static_cast<double>(n)));
  } else if (mode == 2) {
    t.mul_(1.0 / static_cast<double>(n));
  }
  return t;
}

bool fft_dtype(const at::Tensor& t) {
  return is_vk(t) && dtype_label(t.scalar_type()) != nullptr;
}

at::Tensor fft_c2c(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool forward) {
  if (!fft_dtype(self) || !at::isComplexType(self.scalar_type())) {
    return on_cpu("aten::_fft_c2c", "", self, dim, normalization, forward)[0].toTensor();
  }
  at::Tensor out = self;
  int64_t n = 1;
  for (int64_t d : dim) {
    n *= self.size(d);
    out = fft_dim(out, d, !forward);
  }
  out = dim.empty() ? self.clone() : out.contiguous();
  return normalize(out, normalization, n);
}

at::Tensor fft_r2c(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool onesided) {
  if (!fft_dtype(self) || !at::isFloatingType(self.scalar_type()) || dim.empty()) {
    return on_cpu("aten::_fft_r2c", "", self, dim, normalization, onesided)[0].toTensor();
  }
  at::Tensor out = self.to(c10::toComplexType(self.scalar_type()));
  int64_t n = 1;
  for (int64_t d : dim) {
    n *= self.size(d);
    out = fft_dim(out, d, false);
  }
  if (onesided) {
    int64_t last = dim.back();
    out = out.narrow(last, 0, out.size(last) / 2 + 1);
  }
  out = out.contiguous();
  return normalize(out, normalization, n);
}

at::Tensor fft_c2r(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  int64_t last = dim.empty() ? 0 : at::maybe_wrap_dim(dim.back(), self.dim());
  if (!fft_dtype(self) || !at::isComplexType(self.scalar_type()) || dim.empty() ||
      self.size(last) < last_dim_size / 2 + 1 || last_dim_size < 1) {
    return on_cpu("aten::_fft_c2r", "", self, dim, normalization, last_dim_size)[0].toTensor();
  }
  at::Tensor x = self;
  int64_t n = last_dim_size;
  for (size_t i = 0; i + 1 < dim.size(); ++i) {
    n *= self.size(dim[i]);
    x = fft_dim(x, dim[i], true);
  }
  // Rebuild the full Hermitian spectrum along the last dim, then invert it.
  at::Tensor moved = x.movedim(last, -1).contiguous();
  int64_t m = moved.size(-1);
  at::Tensor rows = moved.view({-1, m});
  int64_t batch = rows.size(0);
  at::Tensor full = at::empty({batch, n == 0 ? 0 : last_dim_size}, rows.options());
  if (batch > 0) {
    FftParams p{};
    p.in = gpu_address(rows);
    p.out = gpu_address(full);
    p.n = static_cast<uint32_t>(last_dim_size);
    p.m = static_cast<uint32_t>(m);
    p.batch = static_cast<uint32_t>(batch);
    p.ngroups = groups_for(batch * last_dim_size);
    dispatch(std::string("fft_") + dtype_label(rows.scalar_type()), {1, 0}, p, p.ngroups);
  }
  at::Tensor spectrum = fft_rows(full, true);
  std::vector<int64_t> shape = moved.sizes().vec();
  shape.back() = last_dim_size;
  at::Tensor out = at::real(spectrum).view(shape).movedim(-1, last).contiguous();
  return normalize(out, normalization, n);
}

}  // namespace
}  // namespace vk

// Releases cached device tables (called before the runtime shuts down).
extern "C" void vkt_clear_fft_caches() {
  std::lock_guard<std::mutex> lock(vk::g_cache_mutex);
  vk::g_twiddles.clear();
  vk::g_bluestein.clear();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_fft_c2c", &vk::fft_c2c);
  m.impl("_fft_r2c", &vk::fft_r2c);
  m.impl("_fft_c2r", &vk::fft_c2r);
}

// Referenced from backend.cpp so the linker keeps this object file.
extern "C" void vkt_fft_linked() {}
