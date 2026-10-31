// Random number generation for device "vk" (M11): a Philox generator
// registered as the device's default and new generators, and the random
// fills (uniform_, normal_, random_, bernoulli_, exponential_, geometric_,
// cauchy_, log_normal_) plus ops built on them (normal with tensor
// parameters, multinomial, randperm, native_dropout).
//
// shaders/rng.slang draws element i's bits from Philox4x32-10 with counter
// (i, offset) and key seed. The generator hands out one offset per call, so
// results depend only on (seed, sequence of calls). They differ from the
// CPU's (a Mersenne Twister) for the same seed, as CUDA's do.

#include "opcodes.h"
#include "vk.h"

#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>

namespace vk {

class VkGenerator final : public c10::GeneratorImpl {
 public:
  explicit VkGenerator(uint64_t seed = c10::default_rng_seed_val)
      : c10::GeneratorImpl(c10::Device(kDevice, 0), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
        seed_(seed) {}

  static c10::DeviceType device_type() { return kDevice; }

  void set_current_seed(uint64_t seed) override {
    seed_ = seed;
    offset_ = 0;
  }
  uint64_t current_seed() const override { return seed_; }
  uint64_t seed() override {
    uint64_t s = c10::detail::getNonDeterministicRandom();
    set_current_seed(s);
    return s;
  }
  void set_offset(uint64_t offset) override { offset_ = offset; }
  uint64_t get_offset() const override { return offset_; }

  // State: 16 bytes, the seed then the offset.
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override {
    at::Tensor state = at::empty({16}, at::TensorOptions().dtype(at::kByte));
    uint64_t words[2] = {seed_, offset_};
    std::memcpy(state.data_ptr(), words, sizeof(words));
    return state.getIntrusivePtr();
  }
  void set_state(const c10::TensorImpl& new_state) override {
    TORCH_CHECK(new_state.numel() == 16 && new_state.dtype() == caffe2::TypeMeta::Make<uint8_t>(),
                "torch_vulkan: RNG state must be a uint8 tensor of 16 bytes");
    uint64_t words[2];
    std::memcpy(words, new_state.data(), sizeof(words));
    seed_ = words[0];
    offset_ = words[1];
  }

  // (seed, offset) for one call, advancing the offset (under the
  // generator's mutex, GeneratorImpl::mutex_).
  std::pair<uint64_t, uint64_t> next_call() {
    std::lock_guard<std::mutex> lock(mutex_);
    return {seed_, offset_++};
  }

 private:
  VkGenerator* clone_impl() const override {
    auto* g = new VkGenerator(seed_);
    g->offset_ = offset_;
    return g;
  }

  uint64_t seed_;
  uint64_t offset_ = 0;
};

const at::Generator& default_generator() {
  static at::Generator gen = at::make_generator<VkGenerator>(c10::detail::getNonDeterministicRandom());
  return gen;
}

at::Generator new_generator() {
  return at::make_generator<VkGenerator>();
}

namespace {

VkGenerator* generator(const std::optional<at::Generator>& gen) {
  return at::get_generator_or_default<VkGenerator>(gen, default_generator());
}

struct RngParams {
  uint64_t out, seed, offset, a, b;
  uint32_t n, stride;
};
static_assert(sizeof(RngParams) == 48, "must match shaders/rng.slang");

// The raw slot for a float parameter of a kernel computing in `t`.
uint64_t float_bits(double v, at::ScalarType t) {
  uint64_t bits = 0;
  if (t == at::kDouble) {
    std::memcpy(&bits, &v, sizeof(v));
  } else {
    float f = static_cast<float>(v);
    uint32_t b32;
    std::memcpy(&b32, &f, sizeof(f));
    bits = b32;
  }
  return bits;
}

// The dtype rng.slang computes `t` in: float types in float32/float64,
// integers in int32/int64.
at::ScalarType rng_dtype(at::ScalarType t) {
  if (t == at::kDouble) {
    return has_feature(kFeatureFloat64) ? at::kDouble : at::kFloat;
  }
  if (at::isFloatingType(t)) {
    return at::kFloat;
  }
  return t == at::kInt ? at::kInt : at::kLong;
}

// Fills `self` by distribution `op` with parameter slots a, b (computed for
// the kernel's dtype by `bits(kernel_dtype)`).
template <typename Bits>
at::Tensor& fill_random(at::Tensor& self, uint32_t op, at::ScalarType compute, Bits bits,
                        const std::optional<at::Generator>& gen) {
  if (self.numel() == 0) {
    return self;
  }
  TORCH_CHECK(self.numel() < (int64_t(1) << 32), "torch_vulkan: random fill of more than 2^32 elements");
  bool direct = self.scalar_type() == compute && self.is_contiguous();
  at::Tensor target = direct ? self : at::empty(self.sizes(), self.options().dtype(compute));
  auto [seed, offset] = generator(gen)->next_call();
  auto [a, b] = bits(compute);
  RngParams p{};
  p.out = gpu_address_out(target);
  p.seed = seed;
  p.offset = offset;
  p.a = a;
  p.b = b;
  p.n = static_cast<uint32_t>(target.numel());
  uint32_t groups = 0;
  check(vkt_grid_1d(p.n, &groups, &p.stride));
  dispatch(std::string("rng_") + dtype_label(compute), {op}, p, groups);
  if (!direct) {
    self.copy_(target);
  }
  return self;
}

bool float_fill_ok(const at::Tensor& self) {
  return is_vk(self) && at::isFloatingType(self.scalar_type()) && storage_label(self.scalar_type()) != nullptr;
}

bool complex_fill_ok(const at::Tensor& self) {
  return is_vk(self) && at::isComplexType(self.scalar_type()) && dtype_label(self.scalar_type()) != nullptr;
}

at::Tensor& float_fill(at::Tensor& self, uint32_t op, double a, double b, const std::optional<at::Generator>& gen) {
  at::ScalarType compute = rng_dtype(self.scalar_type());
  return fill_random(self, op, compute,
                     [&](at::ScalarType t) { return std::pair<uint64_t, uint64_t>{float_bits(a, t), float_bits(b, t)}; },
                     gen);
}

at::Tensor& uniform_(at::Tensor& self, double from, double to, std::optional<at::Generator> gen) {
  if (complex_fill_ok(self)) {  // both parts uniform, as PyTorch does
    at::Tensor parts = at::view_as_real(self);
    uniform_(parts, from, to, gen);
    return self;
  }
  if (!float_fill_ok(self)) {
    on_cpu("aten::uniform_", "", self, from, to, gen);
    return self;
  }
  // (the CPU's checks: bounds of the dtype, and to - from representable)
  double max = self.scalar_type() == at::kHalf       ? 65504.0
               : self.scalar_type() == at::kBFloat16 ? 3.3895313892515355e38
               : self.scalar_type() == at::kFloat    ? double(std::numeric_limits<float>::max())
                                                     : std::numeric_limits<double>::max();
  TORCH_CHECK(from >= -max && from <= max, "from is out of bounds for ", self.dtype());
  TORCH_CHECK(to >= -max && to <= max, "to is out of bounds for ", self.dtype());
  TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
  TORCH_CHECK(to - from <= max, "uniform_ expects to-from <= std::numeric_limits<", toString(self.scalar_type()),
              ">::max(), but found to=", to, " and from=", from, " which result in to-from to exceed the limit");
  return float_fill(self, op::RNG_UNIFORM, from, to, gen);
}

at::Tensor& normal_(at::Tensor& self, double mean, double std, std::optional<at::Generator> gen) {
  if (complex_fill_ok(self)) {  // each part with std / sqrt(2), as PyTorch does
    TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);
    at::Tensor parts = at::view_as_real(self);
    normal_(parts, mean, std / std::sqrt(2.0), gen);
    return self;
  }
  if (!float_fill_ok(self)) {
    on_cpu("aten::normal_", "", self, mean, std, gen);
    return self;
  }
  TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);
  return float_fill(self, op::RNG_NORMAL, mean, std, gen);
}

at::Tensor& exponential_(at::Tensor& self, double lambd, std::optional<at::Generator> gen) {
  if (!float_fill_ok(self)) {
    on_cpu("aten::exponential_", "", self, lambd, gen);
    return self;
  }
  TORCH_CHECK(lambd > 0.0, "exponential_ expects lambda > 0.0, but found lambda=", lambd);
  return float_fill(self, op::RNG_EXPONENTIAL, lambd, 0, gen);
}

at::Tensor& cauchy_(at::Tensor& self, double median, double sigma, std::optional<at::Generator> gen) {
  if (!float_fill_ok(self)) {
    on_cpu("aten::cauchy_", "", self, median, sigma, gen);
    return self;
  }
  TORCH_CHECK(sigma > 0.0, "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
  return float_fill(self, op::RNG_CAUCHY, median, sigma, gen);
}

at::Tensor& log_normal_(at::Tensor& self, double mean, double std, std::optional<at::Generator> gen) {
  if (!float_fill_ok(self)) {
    on_cpu("aten::log_normal_", "", self, mean, std, gen);
    return self;
  }
  TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
  return float_fill(self, op::RNG_LOG_NORMAL, mean, std, gen);
}

at::Tensor& geometric_(at::Tensor& self, double prob, std::optional<at::Generator> gen) {
  bool ok = is_vk(self) && storage_label(self.scalar_type()) != nullptr && self.scalar_type() != at::kBool &&
            !at::isComplexType(self.scalar_type());
  if (!ok) {
    on_cpu("aten::geometric_", "", self, prob, gen);
    return self;
  }
  TORCH_CHECK(prob > 0 && prob < 1, "geometric_ expects p to be in (0, 1), but got p=", prob);
  // Integer outputs are computed in float and cast.
  at::ScalarType compute = at::isFloatingType(self.scalar_type()) ? rng_dtype(self.scalar_type()) : at::kFloat;
  return fill_random(self, op::RNG_GEOMETRIC, compute,
                     [&](at::ScalarType t) { return std::pair<uint64_t, uint64_t>{float_bits(prob, t), 0}; }, gen);
}

// ---------------------------------------------------------------- random_

// Largest value random_() draws for dtype t: 2^digits for floats, the type's
// maximum for integers.
int64_t random_max(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return int64_t(1) << 24;
    case at::kDouble: return int64_t(1) << 53;
    case at::kHalf: return int64_t(1) << 11;
    case at::kBFloat16: return int64_t(1) << 8;
    case at::kBool: return 1;
    case at::kByte: return 255;
    case at::kChar: return 127;
    case at::kShort: return 32767;
    case at::kInt: return std::numeric_limits<int32_t>::max();
    default: return std::numeric_limits<int64_t>::max();
  }
}

// self = from + uniform integer in [0, range); range == 0 means 2^64.
at::Tensor& random_range(at::Tensor& self, int64_t from, uint64_t range, std::optional<at::Generator> gen) {
  at::ScalarType t = self.scalar_type();
  at::ScalarType compute = at::isFloatingType(t) ? rng_dtype(t) : (t == at::kInt ? at::kInt : at::kLong);
  return fill_random(self, op::RNG_RANDOM, compute,
                     [&](at::ScalarType) { return std::pair<uint64_t, uint64_t>{static_cast<uint64_t>(from), range}; },
                     gen);
}

bool random_ok(const at::Tensor& self) {
  return is_vk(self) && storage_label(self.scalar_type()) != nullptr && !at::isComplexType(self.scalar_type());
}

at::Tensor& random_from(at::Tensor& self, int64_t from, std::optional<int64_t> to, std::optional<at::Generator> gen) {
  if (!random_ok(self)) {
    on_cpu("aten::random_", "from", self, from, to, gen);
    return self;
  }
  int64_t hi = to.has_value() ? *to : random_max(self.scalar_type());
  if (to.has_value()) {
    TORCH_CHECK(from < hi, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", hi);
    at::native::templates::check_from_to_in_range(from, hi - 1, self.dtype());
    return random_range(self, from, static_cast<uint64_t>(hi) - static_cast<uint64_t>(from), gen);
  }
  TORCH_CHECK(from <= hi, "random_ expects 'from' casted to dtype to be less than or equal to 'to'");
  at::native::templates::check_from_to_in_range(from, hi, self.dtype());
  // Inclusive of the maximum: [from, hi]; the full int64 range wraps to 0 (= 2^64).
  return random_range(self, from, static_cast<uint64_t>(hi) - static_cast<uint64_t>(from) + 1, gen);
}

at::Tensor& random_to(at::Tensor& self, int64_t to, std::optional<at::Generator> gen) {
  if (!random_ok(self)) {
    on_cpu("aten::random_", "to", self, to, gen);
    return self;
  }
  TORCH_CHECK(to > 0, "random_ expects 'from' to be less than 'to', but got from=0 >= to=", to);
  at::native::templates::check_from_to_in_range(0, to - 1, self.dtype());
  return random_range(self, 0, static_cast<uint64_t>(to), gen);
}

at::Tensor& random_full(at::Tensor& self, std::optional<at::Generator> gen) {
  if (!random_ok(self)) {
    on_cpu("aten::random_", "", self, gen);
    return self;
  }
  return random_range(self, 0, static_cast<uint64_t>(random_max(self.scalar_type())) + 1, gen);
}

// ---------------------------------------------------------------- bernoulli

// u < p in the float precision of `p`, cast into self.
at::Tensor& bernoulli_tensor_(at::Tensor& self, const at::Tensor& p, std::optional<at::Generator> gen) {
  at::assert_no_internal_overlap(self);  // (as the CPU)
  if (!random_ok(self) || !is_vk(p) || !at::isFloatingType(p.scalar_type())) {
    on_cpu("aten::bernoulli_", "Tensor", self, p, gen);
    return self;
  }
  at::ScalarType ft = rng_dtype(p.scalar_type());
  at::Tensor u = at::empty(self.sizes(), self.options().dtype(ft));
  float_fill(u, op::RNG_UNIFORM, 0, 1, gen);
  self.copy_(at::lt(u, p.to(ft)));
  return self;
}

at::Tensor& bernoulli_float_(at::Tensor& self, double p, std::optional<at::Generator> gen) {
  at::assert_no_internal_overlap(self);
  if (!random_ok(self)) {
    on_cpu("aten::bernoulli_", "float", self, p, gen);
    return self;
  }
  TORCH_CHECK(p >= 0 && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
  at::Tensor u = at::empty(self.sizes(), self.options().dtype(rng_dtype(at::kDouble)));
  float_fill(u, op::RNG_UNIFORM, 0, 1, gen);
  self.copy_(at::lt(u, p));
  return self;
}

at::Tensor bernoulli(const at::Tensor& self, std::optional<at::Generator> gen) {
  at::Tensor out = at::empty_like(self, at::MemoryFormat::Contiguous);
  return bernoulli_tensor_(out, self, gen);
}

at::Tensor bernoulli_p(const at::Tensor& self, double p, std::optional<at::Generator> gen) {
  at::Tensor out = at::empty_like(self, at::MemoryFormat::Contiguous);
  return bernoulli_float_(out, p, gen);
}

at::Tensor& bernoulli_out(const at::Tensor& self, std::optional<at::Generator> gen, at::Tensor& out) {
  at::native::resize_output(out, self.sizes());
  return bernoulli_tensor_(out, self, gen);
}

// ---------------------------------------------------------------- normal with tensor parameters

void check_std(const at::Tensor& std) {
  TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
  TORCH_CHECK(std.numel() == 0 || at::ge(std, 0).all().item<bool>(), "normal expects all elements of std >= 0.0");
}

at::Tensor normal_tt(const at::Tensor& mean, const at::Tensor& std, std::optional<at::Generator> gen) {
  check_std(std);
  std::vector<int64_t> shape = at::infer_size(mean.sizes(), std.sizes());
  at::Tensor z = at::empty(shape, mean.options().dtype(at::result_type(mean, std)));
  normal_(z, 0, 1, gen);
  return z * std + mean;
}

at::Tensor normal_tf(const at::Tensor& mean, double std, std::optional<at::Generator> gen) {
  TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);
  at::Tensor z = at::empty(mean.sizes(), mean.options());
  normal_(z, 0, 1, gen);
  return z * std + mean;
}

at::Tensor normal_ft(double mean, const at::Tensor& std, std::optional<at::Generator> gen) {
  check_std(std);
  at::Tensor z = at::empty(std.sizes(), std.options());
  normal_(z, 0, 1, gen);
  return z * std + mean;
}

at::Tensor& normal_tt_out(const at::Tensor& mean, const at::Tensor& std, std::optional<at::Generator> gen,
                          at::Tensor& out) {
  at::Tensor r = normal_tt(mean, std, gen);
  at::native::resize_output(out, r.sizes());
  return out.copy_(r);
}

at::Tensor& normal_tf_out(const at::Tensor& mean, double std, std::optional<at::Generator> gen, at::Tensor& out) {
  at::Tensor r = normal_tf(mean, std, gen);
  at::native::resize_output(out, r.sizes());
  return out.copy_(r);
}

at::Tensor& normal_ft_out(double mean, const at::Tensor& std, std::optional<at::Generator> gen, at::Tensor& out) {
  at::Tensor r = normal_ft(mean, std, gen);
  at::native::resize_output(out, r.sizes());
  return out.copy_(r);
}

// ---------------------------------------------------------------- multinomial, randperm, dropout

// Sampling with replacement: invert each row's running sum at total * u.
// Without: Gumbel-top-k (the num_samples largest log p - log(-log u)).
at::Tensor& multinomial_out(const at::Tensor& self, int64_t n, bool replacement, std::optional<at::Generator> gen,
                            at::Tensor& out) {
  TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "multinomial only supports floating-point dtypes for input");
  TORCH_CHECK(n > 0, "cannot sample n_sample <= 0 samples");
  TORCH_CHECK(out.scalar_type() == at::kLong, "multinomial expects Long tensor out, got: ", out.scalar_type());
  TORCH_CHECK(self.size(-1) <= (int64_t(1) << 24), "number of categories cannot exceed 2^24");
  int64_t ncat = self.size(-1);
  TORCH_CHECK(replacement || n <= ncat, "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
  if (!is_vk(self) || !is_vk(out)) {
    on_cpu("aten::multinomial", "out", self, n, replacement, gen, out);
    return out;
  }
  at::Tensor probs = (self.dim() == 1 ? self.unsqueeze(0) : self).to(rng_dtype(self.scalar_type()));
  int64_t rows = probs.size(0);
  at::Tensor bad = at::logical_or(at::logical_or(at::lt(probs, 0), at::isinf(probs)), at::isnan(probs)).any();
  TORCH_CHECK(!bad.item<bool>(), "probability tensor contains either `inf`, `nan` or element < 0");
  at::Tensor positive = at::gt(probs, 0).sum(1);
  TORCH_CHECK(at::gt(positive, 0).all().item<bool>(), "invalid multinomial distribution (sum of probabilities <= 0)");
  if (!replacement) {
    TORCH_CHECK(at::ge(positive, n).all().item<bool>(),
                "invalid multinomial distribution (with replacement=False, not enough non-negative category to sample)");
  }
  at::Tensor result;
  if (replacement) {
    at::Tensor cdf = at::cumsum(probs, 1);
    at::Tensor u = at::empty({rows, n}, probs.options());
    uniform_(u, 0, 1, gen);
    at::Tensor targets = u * cdf.narrow(1, ncat - 1, 1);
    // First category whose running sum exceeds the target, skipping zero-probability ones.
    result = at::clamp_max(at::searchsorted(cdf, targets, false, /*right=*/true), ncat - 1);
  } else {
    at::Tensor u = at::empty(probs.sizes(), probs.options());
    uniform_(u, 0, 1, gen);
    at::Tensor gumbel = -at::log(-at::log1p(-u));  // u in [0, 1): log1p(-u) in (-inf, 0]
    at::Tensor keys = at::where(at::gt(probs, 0), at::log(probs) + gumbel, at::full_like(probs, -INFINITY));
    result = std::get<1>(at::topk(keys, n, 1));
  }
  if (self.dim() == 1) {
    result = result.squeeze(0);
  }
  at::native::resize_output(out, result.sizes());
  return out.copy_(result);
}

at::Tensor multinomial(const at::Tensor& self, int64_t n, bool replacement, std::optional<at::Generator> gen) {
  at::Tensor out = at::empty({0}, self.options().dtype(at::kLong));
  return multinomial_out(self, n, replacement, gen, out);
}

// A uniformly random permutation: sort random 64-bit keys (ties are
// vanishingly rare and broken by position).
at::Tensor& randperm_out(int64_t n, std::optional<at::Generator> gen, at::Tensor& out) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  // (as the CPU: every value must be representable in out's dtype)
  at::ScalarType t = out.scalar_type();
  if (at::isIntegralType(t, /*includeBool=*/true)) {
    int64_t top = t == at::kByte ? 255 : t == at::kChar ? 127 : t == at::kShort ? 32767 : t == at::kInt ? 2147483647
                : t == at::kBool ? 1 : std::numeric_limits<int64_t>::max();
    TORCH_CHECK(n <= 1 || n - 1 <= top, "n is too large for result tensor type: '", out.toString(), "'");
  }
  TORCH_CHECK(t != at::kHalf || n <= (int64_t(1) << 11) + 1, "n cannot be greater than 2049 for Half type.");
  TORCH_CHECK(t != at::kFloat || n <= (int64_t(1) << 24) + 1, "n cannot be greater than 2^24+1 for Float type.");
  TORCH_CHECK(t != at::kDouble || n <= (int64_t(1) << 53) + 1, "n cannot be greater than 2^53+1 for Double type.");
  if (!is_vk(out) || storage_label(out.scalar_type()) == nullptr || at::isComplexType(out.scalar_type())) {
    on_cpu("aten::randperm", "generator_out", n, gen, out);
    return out;
  }
  at::native::resize_output(out, {n});
  if (n == 0) {
    return out;
  }
  at::Tensor keys = at::empty({1, n}, out.options().dtype(at::kLong));
  random_range(keys, std::numeric_limits<int64_t>::min(), 0, gen);
  at::Tensor perm = sort_rows(keys, false).reshape({n});
  out.copy_(perm);
  return out;
}

std::tuple<at::Tensor, at::Tensor> native_dropout(const at::Tensor& input, double p, std::optional<bool> train) {
  if (!train.has_value() || *train) {
    reject_dtype("native_dropout", input.scalar_type(), at::isIntegralType(input.scalar_type(), /*includeBool=*/true));
    TORCH_CHECK(p >= 0 && p <= 1, "dropout probability has to be between 0 and 1, but got ", p);
    at::Tensor mask = at::empty(input.sizes(), input.options().dtype(at::kBool));
    bernoulli_float_(mask, 1 - p, std::nullopt);
    double scale = p == 1 ? 0.0 : 1.0 / (1.0 - p);
    return {input * mask * scale, mask};
  }
  return {input.clone(), at::ones(input.sizes(), input.options().dtype(at::kBool))};
}

at::Tensor native_dropout_backward(const at::Tensor& grad, const at::Tensor& mask, double scale) {
  return grad * mask * scale;
}

}  // namespace
}  // namespace vk

// Python-facing seeding of the default generator (torch.vk.manual_seed, ...).
extern "C" void vkt_manual_seed(uint64_t seed) {
  auto gen = vk::default_generator();
  std::lock_guard<std::mutex> lock(gen.mutex());
  gen.set_current_seed(seed);
}

extern "C" uint64_t vkt_initial_seed() {
  return vk::default_generator().current_seed();
}

extern "C" uint64_t vkt_seed() {
  auto gen = vk::default_generator();
  std::lock_guard<std::mutex> lock(gen.mutex());
  return gen.seed();
}

extern "C" void vkt_get_rng_state(uint64_t* seed, uint64_t* offset) {
  auto gen = vk::default_generator();
  *seed = gen.current_seed();
  *offset = gen.get_offset();
}

extern "C" void vkt_set_rng_state(uint64_t seed, uint64_t offset) {
  auto gen = vk::default_generator();
  std::lock_guard<std::mutex> lock(gen.mutex());
  gen.set_current_seed(seed);
  gen.set_offset(offset);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  using namespace vk;
  m.impl("uniform_", &uniform_);
  m.impl("normal_", &normal_);
  m.impl("exponential_", &exponential_);
  m.impl("cauchy_", &cauchy_);
  m.impl("log_normal_", &log_normal_);
  m.impl("geometric_", &geometric_);
  m.impl("random_.from", &random_from);
  m.impl("random_.to", &random_to);
  m.impl("random_", &random_full);
  m.impl("bernoulli_.Tensor", &bernoulli_tensor_);
  m.impl("bernoulli_.float", &bernoulli_float_);
  m.impl("bernoulli", &vk::bernoulli);
  m.impl("bernoulli.p", &bernoulli_p);
  m.impl("bernoulli.out", &bernoulli_out);
  m.impl("normal.Tensor_Tensor", &normal_tt);
  m.impl("normal.Tensor_float", &normal_tf);
  m.impl("normal.float_Tensor", &normal_ft);
  m.impl("normal.Tensor_Tensor_out", &normal_tt_out);
  m.impl("normal.Tensor_float_out", &normal_tf_out);
  m.impl("normal.float_Tensor_out", &normal_ft_out);
  m.impl("multinomial", &vk::multinomial);
  m.impl("multinomial.out", &multinomial_out);
  m.impl("randperm.generator_out", &randperm_out);
  m.impl("native_dropout", &vk::native_dropout);
  m.impl("native_dropout_backward", &vk::native_dropout_backward);
}
