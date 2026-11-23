# Fused attention (M52)

`scaled_dot_product_attention` on vk always took PyTorch's math path:
`_fused_sdp_choice` returned 0 (math), so every call built the full
[B, H, L, S] attention matrix, its softmax, and (training) the dropout
mask, each a separate tensor. The model benchmarks (#128) found the cost:
DINOv2's ViT-B/14 at 518 x 518 (1,370 tokens) in batches of 128 failed to
allocate on the L40S (#132), where CUDA's fused kernels peak at 7.6 GB.

## How PyTorch lets a PrivateUse1 device choose

`scaled_dot_product_attention` is composite: it calls the
`_fused_sdp_choice_stub` DispatchStub (not the `aten::_fused_sdp_choice`
op, which vk had registered, to no effect on the choice) and switches on
the backend it returns. PyTorch 2.15 keeps one backend for out-of-tree
devices, `SDPBackend.overrideable`, which calls
`_scaled_dot_product_fused_attention_overrideable` with the boolean mask
already turned additive; its autograd formula calls
`_scaled_dot_product_fused_attention_overrideable_backward` with what the
forward returned (output, logsumexp, the Philox seed and offset). So vk
registers the stub with `REGISTER_PRIVATEUSE1_DISPATCH` and implements
both ops (`csrc/attention.cpp`); `torch.nn.attention.sdpa_kernel` enables
and disables the two backends as it does on CUDA (`SDPBackend.MATH`,
`SDPBackend.OVERRIDEABLE`).

## shaders/attention.slang

FlashAttention 2's algorithm with float32 throughout. A workgroup of 256
threads takes 64 query rows and runs over the keys in blocks of 64: the
64 x 64 scores are a product of 16-deep slices of q and k staged in shared
memory (each thread 4 x 4 scores), the row maxima come from the 16 threads
sharing a row, and the block's weights exp(s - m) go to shared memory for
the product with v, read in 2048-element chunks of rows (each thread 4 rows
x EV / 16 columns of the output in registers). The running maximum and sum
rescale the output as each block arrives; the end divides by the sum and
writes logsumexp. Nothing grows with L x S: the working memory is the
output and logsumexp.

The backward recomputes the weights from logsumexp (the scores are never
stored) in two kernels, as most FlashAttention ports without float atomics:

- grad_q: per query block, over the key blocks, dS = P (dP - delta) with
  dP = grad_out v^T, grad_q = scale dS k;
- grad_k and grad_v: per key block of a key head, over the query blocks of
  every query head that reads it (GQA without atomics or a sum afterwards),
  grad_v = P^T grad_out, grad_k = scale dS^T q.

delta = rowsum(grad_out out) and logsumexp go into one stats buffer with
the dropout's parameters (`attention_aux.slang`), since the kernels' push
constants (128 bytes) are full with four operands' strides.

What it takes:

- float32, float16 and bfloat16 (16-bit inputs widened as they are loaded,
  float32 sums, results rounded once, as the math path, which computes
  16-bit attention in float32 too); head dims up to 256 for q and k and for
  v separately (variants for 64, 128 and 256; the exact dims are
  specialisation constants);
- q, k and v with any strides along batch, heads and sequence (ViT's
  views of one qkv projection are read in place), the head dim contiguous
  (else copied); the output is [B, L, H, EV] in memory, as CUDA's flash
  kernel's, so the usual `transpose(1, 2).reshape(B, L, H EV)` is a view;
- additive and boolean masks broadcast to [B, H, L, S] (read with their own
  strides; a stride other than 0 or 1 along keys is copied first); a row
  every key of which is masked gives zeros, as the math path's
  `_safe_softmax`, and a zero gradient;
- `is_causal` (keys after the query hidden, the math path's `tril`; key
  blocks wholly after a query block are skipped);
- grouped-query attention (`enable_gqa`, or one key head for all query
  heads), any `scale` including negative ones;
- dropout: each (b, h, query, key) draws from Philox4x32-10 with counter
  (key / 4, query, b H + h, offset) under the generator's seed, so the
  backward draws the same mask again from the seed and offset the forward
  returns. The mask differs from the math path's (`at::dropout`'s
  Bernoulli draws), as CUDA's fused kernels' do;
  `_scaled_dot_product_fused_attention_overrideable(..., return_debug_mask=True)`
  returns it, and the tests compare against the math path given that mask.

The math path remains for everything else: float64, nested tensors, head
dims past 256, sizes past 32-bit offsets or the launch limits, a mask that
needs a gradient (the fused backward has no gradient for it), forward-mode
gradients, devices with less than 32 KiB of shared memory. The math path's
gradient of the gradient also stays available that way only for float64,
which is what `gradgradcheck` uses.

## Choosing

MEASURED

`TORCH_VK_SDPA=math` prefers the math path (and `flash` the fused kernel)
when `sdpa_kernel` leaves both enabled; with only one enabled, that one
runs, and the fused one raises PyTorch's "No available kernel" where it
does not apply. `tests/python/test_attention.py` forces each route through
`sdpa_kernel` and checks `TORCH_VK_SDPA=math` in a subprocess.

## Peak memory

The vk allocator now records the most bytes its live tensors held at once
(`Pool::peak`), so `torch.accelerator.max_memory_allocated()` and
`reset_peak_memory_stats()` work for vk; `benchmarks/attention.py` reports
it per call.
