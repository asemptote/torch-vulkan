// FFTs for device "vk": _fft_c2c, _fft_r2c, _fft_c2r (everything in
// torch.fft reduces to these), complex64 and complex128.
//
// A multi-dim transform is a sequence of 1-D transforms along each dim; each
// works on contiguous rows [batch, n] (the dim moved last):
// - n a power of two: Stockham FFTs, the whole row in shared memory up to
//   fft.slang's LOCAL_MAX, radix-r passes (shaders/fft_radix.slang, r <= 128)
//   beyond, with twiddles exp(-2 pi i t / n) computed on the host in float64;
// - any other n: Bluestein's algorithm, a convolution with the chirp
//   w_k = exp(-+ i pi k^2 / n) done with power-of-two FFTs of length >= 2n - 1.
// r2c and c2r of even length run at half length (fft.slang FFT_R2C /
// FFT_C2R: the real row read as complex pairs); of odd length, r2c
// transforms the input as complex and keeps n/2 + 1 outputs, and c2r rebuilds
// the Hermitian-symmetric spectrum (fft.slang PHASE 1), inverts it and keeps
// the real part. Twiddle and chirp tables are cached per (n, dtype).

#include "vk.h"

#include "opcodes.h"

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

struct RadixParams {
  uint64_t in, out, table;
  uint32_t n, ns, batch, r, g, ngroups;
};
static_assert(sizeof(RadixParams) == 48, "must match shaders/fft_radix.slang");

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
  // fft_radix.slang's workgroups hold `budget` values (1024 complex64 or 512
  // complex128, 8 KiB): rows up to that long take one radix-n pass, budget / n
  // rows per workgroup. Rows up to twice that (fft.slang LOCAL_MAX, 32 KiB)
  // take its radix-2 FFT_LOCAL pass, one row per workgroup (faster there
  // than two radix passes).
  int64_t budget = x.scalar_type() == at::kComplexFloat ? 1024 : 512;
  if (n > budget && n <= 2 * budget) {
    at::Tensor out = at::empty_like(x);
    FftParams p{};
    p.in = gpu_address(x);
    p.out = gpu_address_out(out);
    p.table = gpu_address(table);
    p.n = static_cast<uint32_t>(n);
    p.batch = static_cast<uint32_t>(batch);
    p.ngroups = static_cast<uint32_t>(std::min<int64_t>(batch, 65535));
    dispatch(kernel, {op::FFT_LOCAL, inverse ? 1u : 0u}, p, p.ngroups);
    return out;
  }
  // Longer rows: radix-r Stockham passes, r <= 128, as few passes as that
  // allows with the bits of n split evenly between them, and g = budget / r
  // groups per workgroup so that every workgroup is full.
  int64_t bits = 0;
  while ((int64_t(1) << bits) < n) ++bits;
  int64_t passes = n <= budget ? 1 : (bits + 6) / 7;
  std::string radix = std::string("fft_radix_") + dtype_label(x.scalar_type());
  at::Tensor src = x, buffers[2] = {at::empty_like(x), at::empty_like(x)};
  int which = 0;
  for (int64_t pass = 0, ns = 1; pass < passes; ++pass) {
    int64_t r = int64_t(1) << (bits / passes + (pass < bits % passes ? 1 : 0));
    int64_t g = n <= budget ? budget / n : budget / r;
    at::Tensor& dst = buffers[which];
    RadixParams p{};
    p.in = gpu_address(src);
    p.out = gpu_address_out(dst);
    p.table = gpu_address(table);
    p.n = static_cast<uint32_t>(n);
    p.ns = static_cast<uint32_t>(ns);
    p.batch = static_cast<uint32_t>(batch);
    p.r = static_cast<uint32_t>(r);
    p.g = static_cast<uint32_t>(g);
    p.ngroups = static_cast<uint32_t>(std::min<int64_t>((batch * (n / r) + g - 1) / g, 65535));
    dispatch(radix, {inverse ? 1u : 0u}, p, p.ngroups);
    src = dst;
    which ^= 1;
    ns *= r;
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

// float16 and complex32 (which the CPU does not transform; CUDA does): in
// float32 / complex64, the result rounded back, any size.
bool half_fft(const at::Tensor& t) {
  return is_vk(t) && (t.scalar_type() == at::kHalf || t.scalar_type() == at::kComplexHalf);
}

at::Tensor widen_fft(const at::Tensor& t) {
  return t.to(t.scalar_type() == at::kHalf ? at::kFloat : at::kComplexFloat);
}

at::Tensor narrow_fft(const at::Tensor& t) {
  return t.to(t.is_complex() ? at::kComplexHalf : at::kHalf);
}

// Runs fft.slang's elementwise `phase` (FFT_R2C, FFT_C2R, FFT_HERMITIAN)
// over rows in -> out for transform length n (in rows of m elements).
void fft_phase(uint32_t phase, const at::Tensor& in, const at::Tensor& out, int64_t n, int64_t m, int64_t batch,
               int64_t threads) {
  if (batch == 0 || threads == 0) {
    return;
  }
  FftParams p{};
  p.in = gpu_address(in);
  p.out = gpu_address_out(out);
  p.table = phase == op::FFT_HERMITIAN ? 0 : gpu_address(twiddles(n, in.scalar_type()));
  p.n = static_cast<uint32_t>(n);
  p.m = static_cast<uint32_t>(m);
  p.batch = static_cast<uint32_t>(batch);
  p.ngroups = groups_for(threads);
  dispatch(std::string("fft_") + dtype_label(in.scalar_type()), {phase, 0}, p, p.ngroups);
}

// The onesided spectrum [.., n/2 + 1] of real `x` along its last dim, n even:
// a complex FFT of half length (the row read as complex pairs) and
// fft.slang's FFT_R2C to separate the even and odd samples' spectra.
at::Tensor rfft_last_even(const at::Tensor& x) {
  at::Tensor moved = x.contiguous();
  if (moved.storage_offset() % 2 != 0) {
    moved = moved.clone();  // (complex pairs must start at an even offset)
  }
  int64_t n = moved.size(-1), h = n / 2, batch = moved.numel() / std::max<int64_t>(n, 1);
  at::Tensor z = at::view_as_complex(moved.view({batch, h, 2}));
  at::Tensor spectrum = fft_rows(z, false);
  at::Tensor out = at::empty({batch, h + 1}, spectrum.options());
  fft_phase(op::FFT_R2C, spectrum, out, n, h, batch, batch * (h + 1));
  std::vector<int64_t> shape = moved.sizes().vec();
  shape.back() = h + 1;
  return out.view(shape);
}

at::Tensor fft_c2c_compute(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool forward) {
  if (half_fft(self) && self.is_complex()) {
    return narrow_fft(fft_c2c_compute(widen_fft(self), dim, normalization, forward));
  }
  if (!fft_dtype(self) || !at::isComplexType(self.scalar_type())) {
    return on_cpu("aten::_fft_c2c", "", self, dim, normalization, forward)[0].toTensor();
  }
  at::Tensor out = self;
  int64_t n = 1;
  for (int64_t d : dim) {
    n *= self.size(d);
    out = fft_dim(out, d, !forward);
  }
  if (dim.empty()) {
    out = self.clone();
  }
  return normalize(out, normalization, n);
}

at::Tensor fft_r2c_compute(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool onesided) {
  if (half_fft(self) && !self.is_complex() && !dim.empty()) {
    return narrow_fft(fft_r2c_compute(widen_fft(self), dim, normalization, onesided));
  }
  if (!fft_dtype(self) || !at::isFloatingType(self.scalar_type()) || dim.empty()) {
    return on_cpu("aten::_fft_r2c", "", self, dim, normalization, onesided)[0].toTensor();
  }
  int64_t n = 1;
  for (int64_t d : dim) n *= self.size(d);
  int64_t last = at::maybe_wrap_dim(dim.back(), self.dim()), len = self.size(last);
  at::Tensor out;
  if (len % 2 == 0 && len > 0 && self.numel() > 0) {
    // Even length: the last dim's real transform at half length first, then
    // the other dims' complex transforms (the CPU's order too).
    out = rfft_last_even(self.movedim(last, -1)).movedim(-1, last);
    if (!onesided) {
      // The full spectrum, from Hermitian symmetry along the last dim.
      at::Tensor half = out.movedim(last, -1).contiguous();
      int64_t batch = half.numel() / half.size(-1);
      at::Tensor full = at::empty({batch, len}, half.options());
      fft_phase(op::FFT_HERMITIAN, half.view({batch, half.size(-1)}), full, len, half.size(-1), batch, batch * len);
      std::vector<int64_t> shape = half.sizes().vec();
      shape.back() = len;
      out = full.view(shape).movedim(-1, last);
    }
    for (size_t i = 0; i + 1 < dim.size(); ++i) {
      out = fft_dim(out, dim[i], false);
    }
  } else {
    out = self.to(c10::toComplexType(self.scalar_type()));
    for (int64_t d : dim) {
      out = fft_dim(out, d, false);
    }
    if (onesided) {
      out = out.narrow(last, 0, out.size(last) / 2 + 1);
    }
  }
  return normalize(out, normalization, n);
}

at::Tensor fft_c2r_compute(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  if (half_fft(self) && self.is_complex() && !dim.empty()) {
    return narrow_fft(fft_c2r_compute(widen_fft(self), dim, normalization, last_dim_size));
  }
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
  at::Tensor moved = x.movedim(last, -1).contiguous();
  int64_t m = moved.size(-1);
  at::Tensor rows = moved.view({-1, m});
  int64_t batch = rows.size(0);
  std::vector<int64_t> shape = moved.sizes().vec();
  shape.back() = last_dim_size;
  at::Tensor out;
  if (last_dim_size % 2 == 0 && batch > 0) {
    // Even length: fft.slang's FFT_C2R folds the half spectrum into one of
    // half length, whose inverse FFT, read as reals, is the output (times n).
    int64_t h = last_dim_size / 2;
    at::Tensor folded = at::empty({batch, h}, rows.options());
    fft_phase(op::FFT_C2R, rows, folded, last_dim_size, m, batch, batch * h);
    at::Tensor z = fft_rows(folded, true);
    out = at::view_as_real(z).view(shape).movedim(-1, last);
  } else {
    // Rebuild the full Hermitian spectrum along the last dim, then invert it.
    at::Tensor full = at::empty({batch, n == 0 ? 0 : last_dim_size}, rows.options());
    fft_phase(op::FFT_HERMITIAN, rows, full, last_dim_size, m, batch, batch * last_dim_size);
    at::Tensor spectrum = fft_rows(full, true);
    out = at::real(spectrum).view(shape).movedim(-1, last);
  }
  return normalize(out, normalization, n);
}

// ---------------------------------------------------------------- output layouts

// The CPU's FFT output layouts (MKL's _exec_fft, which PyTorch's meta kernels
// reproduce for every non-CUDA device, so FakeTensor expects them too): the
// untransformed dims ordered by the input's strides (largest first), then
// the transformed dims in `dims` order, contiguous in that permuted order.
std::vector<int64_t> exec_fft_strides(at::IntArrayRef in_strides, at::IntArrayRef out_sizes,
                                      const std::vector<int64_t>& dims) {
  int64_t ndim = static_cast<int64_t>(out_sizes.size());
  std::vector<bool> transformed(ndim, false);
  for (int64_t d : dims) transformed[d] = true;
  std::vector<int64_t> batch;
  for (int64_t d = 0; d < ndim; ++d) {
    if (!transformed[d]) batch.push_back(d);
  }
  std::stable_sort(batch.begin(), batch.end(), [&](int64_t a, int64_t b) { return in_strides[a] > in_strides[b]; });
  std::vector<int64_t> strides(ndim);
  int64_t step = 1;  // contiguous over [batch..., dims...] (sizes of 0 count as 1)
  for (auto it = dims.rbegin(); it != dims.rend(); ++it) {
    strides[*it] = step;
    step *= std::max<int64_t>(out_sizes[*it], 1);
  }
  for (auto it = batch.rbegin(); it != batch.rend(); ++it) {
    strides[*it] = step;
    step *= std::max<int64_t>(out_sizes[*it], 1);
  }
  return strides;
}

// dims sorted by the input's strides, largest first (all but the last with exclude_last).
std::vector<int64_t> sort_dims(at::IntArrayRef in_strides, at::IntArrayRef dim, bool exclude_last) {
  std::vector<int64_t> sorted(dim.begin(), dim.end());
  auto end = sorted.end() - (exclude_last && !sorted.empty() ? 1 : 0);
  std::stable_sort(sorted.begin(), end, [&](int64_t a, int64_t b) { return in_strides[a] > in_strides[b]; });
  return sorted;
}

at::Tensor with_strides(const at::Tensor& result, std::vector<int64_t> strides) {
  // The transforms above leave each result in the layout of its last pass
  // (e.g. a column transform's, transposed), often the CPU's own: copied
  // only when it differs. Without MKL the CPU's outputs are contiguous.
  if (!at::hasMKL()) {
    int64_t step = 1;
    for (int64_t d = result.dim() - 1; d >= 0; --d) {
      strides[d] = step;
      step *= std::max<int64_t>(result.size(d), 1);
    }
  }
  if (result.strides() != at::IntArrayRef(strides)) {
    at::Tensor out = at::empty_strided(result.sizes(), strides, result.options());
    out.copy_(result);
    return out;
  }
  if (!result.is_view()) {
    return result;
  }
  // An op's output must not be a view: a view covering all of its storage
  // (a permutation of a working buffer) takes the storage over, any other
  // (e.g. the onesided half of a full spectrum) is copied.
  if (result.storage_offset() == 0 && result.is_non_overlapping_and_dense() &&
      result.storage().nbytes() == static_cast<size_t>(result.numel()) * result.element_size()) {
    return at::empty({0}, result.options()).set_(result.storage(), 0, result.sizes(), result.strides());
  }
  return result.clone();
}

// The registered kernels: results in the CPU's layouts.
// bfloat16 (and bcomplex32): no FFT on the CPU or CUDA either; their error.
void reject_bfloat16_fft(const at::Tensor& t) {
  at::ScalarType st = t.scalar_type();
  TORCH_CHECK_NOT_IMPLEMENTED(!is_vk(t) || (st != at::kBFloat16 && st != at::ScalarType::BComplex32),
                              "Unsupported dtype ", st);
}

at::Tensor fft_c2c(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool forward) {
  reject_bfloat16_fft(self);
  at::Tensor result = fft_c2c_compute(self, dim, normalization, forward);
  if (dim.empty() || !is_vk(result)) {
    return result;
  }
  return with_strides(result, exec_fft_strides(self.strides(), result.sizes(), sort_dims(self.strides(), dim, false)));
}

at::Tensor fft_r2c(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool onesided) {
  reject_bfloat16_fft(self);
  TORCH_CHECK_TYPE(!self.is_complex() || !is_vk(self), "Only supports floating-point dtypes, but found: ",
                   self.scalar_type());  // (the CPU's error, a TypeError)
  at::Tensor result = fft_r2c_compute(self, dim, normalization, onesided);
  if (dim.empty() || !is_vk(result)) {
    return result;
  }
  return with_strides(result, exec_fft_strides(self.strides(), result.sizes(), sort_dims(self.strides(), dim, true)));
}

at::Tensor fft_c2r(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size) {
  reject_bfloat16_fft(self);
  at::Tensor result = fft_c2r_compute(self, dim, normalization, last_dim_size);
  if (dim.empty() || !is_vk(result)) {
    return result;
  }
  // The CPU first does the c2c transforms of all but the last dim, then the c2r of the last.
  std::vector<int64_t> in_strides(self.strides().begin(), self.strides().end());
  if (dim.size() > 1) {
    at::IntArrayRef c2c_dims = dim.slice(0, dim.size() - 1);
    in_strides = exec_fft_strides(self.strides(), self.sizes(), sort_dims(self.strides(), c2c_dims, false));
  }
  return with_strides(result, exec_fft_strides(in_strides, result.sizes(), {dim.back()}));
}

// out= forms: the functional result in out (resized and cast as PyTorch's
// out= rules; otherwise they would run on the CPU and differ slightly).
at::Tensor& fft_out(const at::Tensor& result, at::Tensor& out) {
  check_out_device(result, out);
  check_out_cast(result.scalar_type(), out);
  at::native::resize_output(out, result.sizes());
  out.copy_(result);
  return out;
}

at::Tensor& fft_c2c_out(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool forward,
                        at::Tensor& out) {
  return fft_out(fft_c2c(self, dim, normalization, forward), out);
}

at::Tensor& fft_r2c_out(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, bool onesided,
                        at::Tensor& out) {
  return fft_out(fft_r2c(self, dim, normalization, onesided), out);
}

at::Tensor& fft_c2r_out(const at::Tensor& self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size,
                        at::Tensor& out) {
  return fft_out(fft_c2r(self, dim, normalization, last_dim_size), out);
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
  m.impl("_fft_c2c.out", &vk::fft_c2c_out);
  m.impl("_fft_r2c.out", &vk::fft_r2c_out);
  m.impl("_fft_c2r.out", &vk::fft_c2r_out);
}
