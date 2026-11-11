# Random numbers (M11)

- A Philox4x32-10 generator per device (`csrc/random.cpp`), registered with
  PyTorch: `torch.manual_seed`, `torch.Generator(device="vk")`,
  `torch.vk.get_rng_state`/`set_rng_state` and `torch.random.fork_rng`
  work as for CUDA.
- `shaders/rng.slang` draws element i's bits from counter (i, offset) under
  the seed, so results depend only on the seed and the sequence of calls,
  not on how the work is split across GPU threads.
- On the GPU: `uniform_` (`rand`), `normal_` (`randn`, Box-Muller, also with
  tensor mean/std), `random_` (`randint`, every integer and float dtype),
  `exponential_`, `geometric_`, `cauchy_`, `log_normal_`, `bernoulli`
  (scalar or tensor probabilities), `multinomial` (with replacement by
  inverting the running sum, without by Gumbel top-k), `randperm` (sorting
  random keys) and dropout.
- The numbers differ from the CPU's for the same seed (the CPU uses a
  Mersenne Twister), as CUDA's do; tests check reproducibility, ranges and
  distribution statistics instead. The OpInfo sweep does not compare samples
  whose CPU result depends on the seed (dropout inside attention, fractional
  pooling); it still records whether they ran on the GPU.

OpInfo sweep: 371 of 689 ops fully on the GPU (346 after M10; the sweep now
lists 689 OpInfo entries), no mismatches.
