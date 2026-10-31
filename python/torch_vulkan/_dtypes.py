"""The dtypes PyTorch's CPU kernels reject, rejected on "vk" too.

A kernel composed from other ops (_kernels.py, PyTorch's decompositions
registered by _decomp.py) often works for more dtypes than the CPU kernel
of the same op, e.g. integer pooling or half-precision losses. "vk" follows
the CPU: those dtypes raise the CPU's kind of error ('"op" not implemented
for 'Int''), so code behaves the same on both devices and PyTorch's own
supported-dtype tests agree. The table lists, per aten op, the dtypes of its
first tensor argument the CPU rejects (from PyTorch's OpInfo database and
test_ops.py's test_dtypes).
"""

import functools

import torch

INTS = (torch.int8, torch.int16, torch.int32, torch.int64)
UINT8 = (torch.uint8,)
BOOL = (torch.bool,)
HALF = (torch.float16, torch.bfloat16)
COMPLEX = (torch.complex64, torch.complex128)

REJECTED = {
    "segment_reduce": INTS + UINT8 + BOOL,
    "_softmax_backward_data": INTS + UINT8,
    "_upsample_bilinear2d_aa": INTS + BOOL + HALF,
    "_upsample_bilinear2d_aa_backward": HALF,
    "_upsample_bicubic2d_aa": INTS + BOOL + HALF,
    "_upsample_bicubic2d_aa_backward": HALF,
    "grid_sampler_2d": INTS + UINT8 + BOOL,
    "grid_sampler_3d": INTS + UINT8 + BOOL,
    "histogram": INTS + HALF,
    "_histogramdd_bin_edges": INTS + HALF,
    "_histogramdd_from_bin_cts": INTS + HALF,
    "_histogramdd_from_bin_tensors": INTS + HALF,
    "max_pool2d_with_indices": BOOL,
    "max_pool3d_with_indices": BOOL,
    "max_pool2d_with_indices_backward": INTS + UINT8 + BOOL,
    "max_pool3d_with_indices_backward": INTS + UINT8 + BOOL,
    "adaptive_max_pool2d": INTS + UINT8 + BOOL,
    "adaptive_max_pool3d": INTS + UINT8 + BOOL,
    "max_unpool2d": INTS + UINT8 + BOOL,
    "max_unpool3d": INTS + UINT8 + BOOL,
    "huber_loss": INTS + UINT8,
    "nll_loss_forward": INTS + UINT8,
    "nll_loss2d_forward": INTS + UINT8,
    "multi_margin_loss": INTS + UINT8 + HALF,
    "multilabel_margin_loss_forward": INTS + UINT8 + HALF,
    "im2col": INTS + UINT8,
    "avg_pool2d": (torch.int32,),
    "avg_pool3d": (torch.int32,),
    "_cdist_forward": HALF,
    "_cdist_backward": HALF,
    "_pdist_forward": HALF,
    "_pdist_backward": HALF,
    "_ctc_loss": HALF,
    "_ctc_loss_backward": HALF,
    "rrelu_with_noise": (torch.float16,),
    "vdot": BOOL,
    "hash_tensor": COMPLEX,
    "unfold_backward": (torch.complex32,),
    "upsample_nearest1d": INTS + BOOL,
    "upsample_nearest2d": INTS + BOOL,
    "upsample_nearest3d": INTS + BOOL,
    "_upsample_nearest_exact1d": INTS + BOOL,
    "_upsample_nearest_exact2d": INTS + BOOL,
    "_upsample_nearest_exact3d": INTS + BOOL,
    "upsample_linear1d": INTS + UINT8 + BOOL,
    "upsample_bilinear2d": INTS + BOOL,
    "upsample_trilinear3d": INTS + UINT8 + BOOL,
    "upsample_bicubic2d": INTS + BOOL,
}

# c10's names for dtypes in error messages.
NAMES = {torch.int8: "Char", torch.int16: "Short", torch.int32: "Int", torch.int64: "Long", torch.uint8: "Byte",
         torch.bool: "Bool", torch.float16: "Half", torch.bfloat16: "BFloat16", torch.complex64: "ComplexFloat",
         torch.complex128: "ComplexDouble"}


def _first_tensor(args, kwargs):
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value
    return None


def with_cpu_dtypes(name, kernel):
    """`kernel` for aten op `name` (overload packet name), raising for the dtypes the CPU rejects."""
    rejected = REJECTED.get(name)
    if not rejected:
        return kernel

    @functools.wraps(kernel)
    def checked(*args, **kwargs):
        first = _first_tensor(args, kwargs)
        if first is not None and first.dtype in rejected:
            raise RuntimeError(f'"{name}" not implemented for \'{NAMES.get(first.dtype, str(first.dtype))}\'')
        return kernel(*args, **kwargs)

    return checked
