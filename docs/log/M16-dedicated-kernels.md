# Dedicated kernels (M16)

Ops that ran as chains of generic kernels (PyTorch's decompositions), chosen
by profiling (`benchmarks/composed.py`, which counts each op's launches and
CPU fallbacks) and by the slowest tests of the full runs on lavapipe:

- softmax / log_softmax, layer norm and RMS norm (`_fused_rms_norm`),
  forward and backward: one workgroup per row (`softmax.slang`,
  `layernorm.slang`);
- batch norm and group norm, forward and backward: per-channel (or per-group)
  moments and gradient sums, then one element-wise pass (`normgrad.slang`);
- max / average pooling, linear upsampling (a gather per dim for the
  backward) and grid sampling, forward and backward (`pool.slang`,
  `resample.slang`, `gridsample.slang`);
- nll_loss (so cross_entropy) forward and backward (`loss.slang`);
  smooth-L1 and Huber loss backward as one binary kernel each;
- embedding's backward and grid sampling's grad_input as rows summed by key:
  one sort of the keys, then whole rows folded in index order, as the CPU's
  loops (`segreduce.slang`'s row mode);
- the activations' backward (gelu, silu, mish, hardswish, hardsigmoid,
  log_sigmoid, sigmoid, tanh, softplus, elu/selu/celu, hardtanh, leaky_relu,
  logit) as one binary kernel each;
- reductions and arg-reductions of a few elements per output: one thread per
  output instead of a workgroup;
- the one-sided Jacobi SVD: column-major storage, and no rotations for
  column pairs already orthogonal to the sweep's tolerance or for columns
  negligible beside |A| (as LAPACK's gesvj), so rank-deficient matrices
  converge in a third of the sweeps.

Not done: rms_norm stays composed (its fused op `_fused_rms_norm` has no CPU
backward kernel, and PyTorch only takes that path on CUDA-like devices);
the smooth-L1/Huber losses' backward still take a few generic kernels;
convolution and attention are matrix products already.
