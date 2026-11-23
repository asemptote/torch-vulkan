# Fused attention (M54)

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

The fused kernel runs wherever it applies, with no size threshold: it was
faster than the math path in every case measured, from 64 sequences of 128
tokens to one of 8,192, on both devices measured (below), and its memory
does not grow with L x S. `TORCH_VK_SDPA=math` prefers the math path (and
`flash` the fused kernel) when `sdpa_kernel` leaves both enabled; with only one enabled, that one
runs, and the fused one raises PyTorch's "No available kernel" where it
does not apply. `tests/python/test_attention.py` forces each route through
`sdpa_kernel` and checks `TORCH_VK_SDPA=math` in a subprocess.

## Peak memory

The vk allocator now records the most bytes its live tensors held at once
(`Pool::peak`), so `torch.accelerator.max_memory_allocated()` and
`reset_peak_memory_stats()` work for vk; `benchmarks/attention.py` reports
it per call.

## Measured

`benchmarks/attention.py`, best of 3, milliseconds and the peak memory of
the call's tensors beyond its inputs (MiB). "fails": the math path's
batched product passes 65,535 workgroups along one grid dimension (#131).

H100 NVL, forward:

| case | dtype | math ms | fused ms | speed-up | math MiB | fused MiB |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/14 (DINOv2, 1,370 tokens) b8, 12 heads | float32 | 8.38 | 5.20 | 1.61x | 1580 | 33 |
| | float16 | 9.24 | 4.85 | 1.90x | 1676 | 17 |
| | bfloat16 | 9.07 | 5.06 | 1.79x | 1676 | 17 |
| ViT-B/14 b128 | float32 | fails | 82.6 | | fails | 522 |
| | float16 | fails | 76.2 | | fails | 265 |
| GPT-2 causal b4 L1024 | float32 | 2.92 | 0.93 | 3.14x | 448 | 12 |
| | float16 | 3.23 | 0.83 | 3.88x | 484 | 6 |
| | bfloat16 | 3.21 | 0.86 | 3.73x | 484 | 6 |
| Llama GQA 32/8 heads, L2048, d128, causal | float32 | 8.80 | 3.09 | 2.85x | 1265 | 32 |
| | float16 | 9.32 | 10.20 | 0.91x | 1313 | 16 |
| | bfloat16 | 9.25 | 3.00 | 3.08x | 1313 | 16 |
| L8192, 16 heads | float32 | 46.1 | 26.6 | 1.73x | 9249 | 32 |
| | float16 | 50.5 | 25.1 | 2.02x | 9345 | 16 |
| | bfloat16 | 49.5 | 26.5 | 1.87x | 9345 | 16 |
| b64 L128, 8 heads | float32 | 1.00 | 0.31 | 3.25x | 89 | 16 |
| | float16 | 1.25 | 0.27 | 4.58x | 137 | 8 |

(The Llama float16 rows were measured with float16 loaded as `half`;
the kernel now reads float16 as its bits, as bfloat16, whose rows are 3x
faster. To be measured again.)

H100 NVL, forward and backward:

| case | dtype | math ms | fused ms | speed-up | math MiB | fused MiB |
|---|---|---:|---:|---:|---:|---:|
| ViT-B/14 b8 | float32 | 19.8 | 18.9 | 1.05x | 2190 | 162 |
| | float16 | 27.2 | 17.8 | 1.53x | 2174 | 82 |
| GPT-2 causal b4 L1024 | float32 | 6.84 | 3.30 | 2.07x | 624 | 61 |
| | float16 | 7.39 | 3.14 | 2.35x | 618 | 31 |
| Llama GQA, as above | float32 | 21.3 | 14.4 | 1.48x | 1664 | 113 |
| | float16 | 22.2 | 21.3 | 1.04x | 1648 | 57 |
| L8192, 16 heads | float32 | 114 | 99 | 1.15x | 12416 | 162 |
| | float16 | 119 | 96 | 1.23x | 12400 | 82 |
| b64 L128 | float32 | 2.55 | 1.16 | 2.20x | 160 | 81 |
| | float16 | 3.12 | 1.13 | 2.76x | 153 | 41 |

The forward reaches about 9 TFLOP/s in float32 on the H100 (ViT-B/14:
46 GFLOP in 5.2 ms): plain float32 multiply-adds from shared memory, no
tensor cores, about a sixth of the GPU's float32 peak. The backward does
about seven products of the forward's size (grad_q recomputes the scores
and grad_out v^T that grad_k and grad_v compute too) against the math
path's four, so its gain is smaller. Cooperative-matrix tiles for 16-bit
inputs, and keeping the query tile in shared memory where the device has
more than 32 KiB, are the obvious next steps; neither changes the memory.

lavapipe (llvmpipe, 256-bit, 4 threads), `--small`, float32:

| case | | math ms | fused ms | speed-up | math MiB | fused MiB |
|---|---|---:|---:|---:|---:|---:|
| ViT b1 L257, 12 heads | forward | 106 | 38 | 2.78x | 8 | 1 |
| causal b2 L512, 8 heads | forward | 427 | 66 | 6.43x | 39 | 2 |
| ViT b1 L257 | forward and backward | 171 | 132 | 1.29x | | |
| causal b2 L512 | forward and backward | 566 | 259 | 2.19x | | |

The whole model: TIMM's `vit_base_patch14_dinov2` at 518 x 518, float32
inference, batch 128, on the H100 NVL: 4.08 s per batch and a peak of
7.23 GiB of tensors (0.71 GiB of it the input and weights) with the fused
kernel, where CUDA peaks at 7.6 GB; on the math path the batch fails
(#131's product limit; past that, each layer's scores alone are 11.5 GB).
The L40S, where #132 was found, was not measured here (its queue was
taken by the model benchmarks).

`tests/python/test_attention.py` compares the fused kernel with the math
path on the CPU in float64, forward and backward: float32, float16 and
bfloat16; head dims 8 to 256 (q, k and v's differing); lengths 1 to 130
around the 64-row blocks; GQA and a shared key head; bool, additive,
broadcast, sliced and key-broadcast masks with a fully masked row; causal;
negative scale; ViT's strided views; dropout against the math path given
the kernel's mask (forward and backward); a 2,048-token sequence's peak
memory against the math path's.
