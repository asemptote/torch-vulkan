# Element-wise ops (M8)

Every dense element-wise aten op on float32/float64/int32/int64/bool now has a
Vulkan kernel: the remaining transcendental and special functions (`asin` ...
`atanh`, `log10`, `log1p`, `expm1`, `erf`, `erfc`, `erfinv`, `lgamma`,
`digamma`, `sinc`, `frac`), binary math (`pow`, `atan2`, `fmod`, `remainder`,
`floor_divide`, `logaddexp`, `xlogy`, `hypot`, `copysign`, `nextafter`,
`heaviside`, `fmax`/`fmin`, shifts, `gcd`/`lcm`), activations (`gelu`, `silu`,
`mish`, `softplus`, `elu`/`selu`, `logsigmoid`, `logit`, `hardtanh`,
`hardswish`, `leaky_relu`, shrinks, `threshold`), `clamp` (scalar and tensor
bounds), predicates (`isnan`, `isinf`, `signbit`, ...), logical ops, `where`,
`masked_fill`, `lerp`, `addcmul`, `addcdiv`, `nan_to_num`, `round(decimals)`.

Math precision, by dtype:

- **float32** uses the GPU's GLSL built-ins (`exp`, `log`, `sin`, `pow`,
  `atan2`, ...), so its accuracy follows Vulkan's precision rules, plus
  `shaders/_math32.slang` for functions with no built-in (`erf`, `erfinv`,
  `lgamma`, `digamma`, `log1p`, `expm1`, `sinpi`) and short wrappers where a
  built-in mishandles edge cases or is too inaccurate on Intel (`tanh`
  overflow, `sinh`/`asinh` cancellation, `acos` and `tan` accuracy, C99
  special values of `pow`/`atan2`). float32 kernels contain no float64 code,
  so they run on GPUs without `shaderFloat64` (a Rust test checks the SPIR-V).
- **float64** uses the software library in `_math64.slang` and
  `_math64_ext.slang` (SPIR-V has no 64-bit transcendentals), agreeing with
  the CPU to a few ULP.
- `fmod`/`remainder`/`floor_divide` are exact for both.

`shaders/_fmath.slang` gives both under one set of names (`m_exp`, `m_pow`,
...), so `unary.slang` and `binary.slang` are each one kernel per dtype.

Infrastructure cleanup in the same step:

- Operation codes live in `shaders/opcodes.txt`; the build generates
  `opcodes.slang` for the shaders and `opcodes.h` (`vk::op::UNARY_EXP`, ...)
  for C++, so the two cannot disagree.
- One strided launcher (`csrc/launch.cpp` + `csrc/strided.h`) packs scalar
  parameters per kernel dtype (int64 / double / float) for the three- and
  four-operand layouts; push-constant sizes and strides are 4-vectors, since
  Intel's driver misreads dynamically indexed push-constant arrays.
- Kernels are compiled with scalar block layout and denormal preservation
  (stripped at load time on devices that cannot preserve denormals); loading
  a kernel that needs float64/int64/int8 on a device without it is a clear
  error. Shader variants are rebuilt only when their inputs change.
- The C++ static library is linked whole, so every file's kernel
  registrations are kept without link-anchor functions.
- `opinfo_coverage.py` runs OpInfo samples in parallel worker processes:
  272 of 659 ops now run fully on the GPU (170 at M7).
