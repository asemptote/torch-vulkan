"""Python kernels for "vk" built from Vulkan ops, for ops that have neither a
C++ kernel nor a PyTorch decomposition: average pooling, adaptive pooling,
grid sampling backward (and 3-D forward), and the backward passes of
upsampling.

Interpolation and adaptive average pooling are separable linear maps: one
small matrix per spatial dim (out x in), applied with matmul, and its
transpose for the backward pass. Interpolation matrices come from running
PyTorch's own decomposition of the forward op on an identity input, so the
backward pass is exactly the adjoint of our forward pass. Average pooling is
a depthwise convolution with a ones kernel, divided by the window sizes.
"""

import functools
import math
import struct

import torch
import torch.nn.functional as F
from torch._decomp import decomposition_table

from . import _checks, _decomp, _dtypes, _layout, _native

aten = torch.ops.aten
_lib = None


# ---------------------------------------------------------------- separable linear maps

def _apply_dims(x, mats):
    """x [..., *S] with mats[d] (out_d x in_d) applied along the last len(mats) dims."""
    k = len(mats)
    for d, m in enumerate(mats):
        dim = x.dim() - k + d
        x = torch.matmul(m.to(x.dtype), x.movedim(dim, -2)).movedim(-2, dim)
    return x


def _interp_matrix(op, in_size, out_size, like, *extra):
    """The out x in matrix of 1-D interpolation by `op` (a 1-D upsample
    decomposition), from interpolating an identity input [N=1, C=in, L=in]."""
    eye = torch.eye(in_size, device=like.device, dtype=like.dtype)
    return op(eye.unsqueeze(0), out_size, *extra)[0].transpose(0, 1)


def _upsample_matrices(name, input_size, output_size, align_corners, scales, like):
    """Per-dim interpolation matrices for upsample op `name`."""
    k = len(output_size)
    mats = []
    for d in range(k):
        n_in, n_out, scale = input_size[2 + d], output_size[d], scales[d] if scales else None
        if name.startswith("nearest"):
            exact = name.startswith("nearest_exact")
            fn = decomposition_table[aten._upsample_nearest_exact1d.default if exact else aten.upsample_nearest1d.default]
            m = _interp_matrix(lambda x, o, s: fn(x, [o], s), n_in, n_out, like, scale)
        elif name == "bicubic2d":
            fn = decomposition_table[aten.upsample_bicubic2d.default]
            eye = torch.eye(n_in, device=like.device, dtype=like.dtype).view(1, n_in, n_in, 1)
            m = fn(eye, [n_out, 1], align_corners, scale, None)[0, :, :, 0].transpose(0, 1)
        else:  # linear1d, bilinear2d, trilinear3d: separable linear
            m = _linear_matrix(n_in, n_out, align_corners, scale, like)
        mats.append(m)
    return mats


def _linear_scale(n_in, n_out, align_corners, scale, dtype):
    """The CPU's area_pixel_compute_scale in opmath (float32 for float32 and
    16-bit floats, float64 for float64), as a Python float."""
    opmath = torch.float64 if dtype == torch.float64 else torch.float32
    if align_corners:
        if n_out <= 1:
            return 0.0
        return (torch.tensor(n_in - 1, dtype=opmath) / torch.tensor(n_out - 1, dtype=opmath)).item()
    if scale is not None and scale > 0:
        return torch.tensor(1.0 / scale, dtype=opmath).item()
    return (torch.tensor(n_in, dtype=opmath) / torch.tensor(n_out, dtype=opmath)).item()


def _linear_matrix(n_in, n_out, align_corners, scale, like):
    """The out x in matrix of 1-D linear interpolation, built on the host with
    the weights shaders/resample.slang uses (as the CPU), then moved to like's
    device and dtype."""
    opmath = torch.float64 if like.dtype == torch.float64 else torch.float32
    sc = torch.tensor(_linear_scale(n_in, n_out, align_corners, scale, like.dtype), dtype=opmath)
    o = torch.arange(n_out, dtype=opmath)
    if align_corners:
        src = sc * o if n_out > 1 else torch.zeros(n_out, dtype=opmath)
    else:
        src = (sc * (o + 0.5) - 0.5).clamp_min(0)
    i0 = src.floor().clamp(max=n_in - 1).long()
    i1 = i0 + (i0 < n_in - 1).long()
    w1 = (src - i0).clamp(0, 1)
    rows = torch.arange(n_out)
    m = torch.zeros(n_out, n_in, dtype=opmath)
    m.index_put_((rows, i0), 1 - w1, accumulate=True)
    m.index_put_((rows, i1), w1, accumulate=True)
    return m.to(device=like.device, dtype=like.dtype)


def _make_upsample_linear(k):
    """upsample_linear1d / bilinear2d / trilinear3d: one thread per output
    (shaders/resample.slang), after the CPU's checks with its messages. (Not
    the meta kernel: it is PyTorch's Python decomposition, 65 ms a call where
    the kernel takes 3; the result is the input's leading dims and
    output_size.)"""
    op = [aten.upsample_linear1d, aten.upsample_bilinear2d, aten.upsample_trilinear3d][k - 1].default

    def forward(x, output_size, align_corners, *scales):
        # (the CPU's upsample checks, which the meta kernel lacks)
        if len(output_size) != k:
            raise RuntimeError(f"It is expected output_size equals to {k}, but got size {len(output_size)}")
        if x.dim() != k + 2:
            raise RuntimeError(f"It is expected input_size equals to {k + 2}, but got size {x.dim()}")
        names = ["D", "H", "W"][3 - k:]
        if min(x.shape[-k:]) <= 0 or min(output_size) <= 0:
            def dims(v):
                return ", ".join(f"{n}: {s}" for n, s in zip(names, v))
            raise RuntimeError(f"Input and output sizes should be greater than 0, but got input ({dims(x.shape[-k:])}) "
                               f"output ({dims(output_size)})")
        if x.numel() == 0 and math.prod(x.shape[1:]) == 0:
            raise RuntimeError(f"Non-empty {k + 2}D data tensor expected but got a tensor with sizes {list(x.shape)}")
        out_shape = (*x.shape[:-k], *output_size)
        if x.dtype == torch.uint8 and k == 2:  # (the CPU's one integer dtype: its fixed-point kernel)
            return _u8_resize(x, output_size, align_corners, scales, "bilinear", False)
        if not x.is_floating_point():
            return _on_cpu(op._schema.name.split("::")[1], x, output_size, align_corners, *scales)
        spatial, out = list(x.shape[-k:]), list(out_shape[-k:])
        sc = [_linear_scale(spatial[d], out[d], align_corners, scales[d] if d < len(scales) else None, x.dtype)
              for d in range(k)]
        planes = x.reshape(math.prod(x.shape[:-k]), *_as_3d(spatial, 1)).to(_pool_dtype(x.dtype))
        y = torch.ops.torch_vulkan.upsample_linear(planes, _as_3d(out, 1), align_corners, _as_3d(sc, 0.0), k)
        return y.reshape(out_shape).to(x.dtype).contiguous(memory_format=_pool_memory_format(x, k))

    return forward


def _make_upsample_backward(name, has_align):
    def backward(grad_output, output_size, input_size, *rest):
        if has_align:
            align_corners, scales = rest[0], list(rest[1:])
        else:
            align_corners, scales = None, list(rest)
        if not any(s is not None for s in scales):
            scales = None
        if name in ("linear1d", "bilinear2d", "trilinear3d") and grad_output.is_floating_point():
            # One gather pass per dim (shaders/resample.slang's backward),
            # where the dense interpolation matrices multiplied every input
            # by every output.
            k = len(output_size)
            dt = _pool_dtype(grad_output.dtype)
            g = grad_output.to(dt)
            for d in reversed(range(k)):
                n_in = input_size[2 + d]
                scale = _linear_scale(n_in, output_size[d], align_corners, scales[d] if scales else None, dt)
                lead = math.prod(g.shape[:g.dim() - k + d])
                trail = math.prod(g.shape[g.dim() - k + d + 1:])
                shape = list(g.shape)
                shape[g.dim() - k + d] = n_in
                g = torch.ops.torch_vulkan.upsample_linear_backward_dim(
                    g.reshape(lead, output_size[d], trail), n_in, align_corners, scale).reshape(shape)
            return g.to(grad_output.dtype).contiguous()
        mats = _upsample_matrices(name, input_size, output_size, align_corners, scales, grad_output)
        return _apply_dims(grad_output, [m.t() for m in mats]).contiguous()

    return backward


# ---------------------------------------------------------------- average pooling

def _pool_out_size(n, k, s, p, ceil_mode):
    out = (n + 2 * p - k + (s - 1 if ceil_mode else 0)) // s + 1
    if ceil_mode and (out - 1) * s >= n + p:
        out -= 1
    return out


def _avg_pool_setup(x, kernel_size, stride, padding, ceil_mode, k):
    ks = list(kernel_size) * (k // len(kernel_size)) if len(kernel_size) < k else list(kernel_size)
    st = list(stride) if stride else ks
    st = st * (k // len(st)) if len(st) < k else st
    pd = list(padding) * (k // len(padding)) if len(padding) < k else list(padding)
    spatial = x.shape[-k:]
    out = [_pool_out_size(spatial[d], ks[d], st[d], pd[d], ceil_mode) for d in range(k)]
    # Pad by p in front and enough at the end for the last (ceil-mode) window.
    pad = []
    for d in reversed(range(k)):
        pad += [pd[d], max(0, (out[d] - 1) * st[d] + ks[d] - spatial[d] - pd[d])]
    return ks, st, pd, out, pad


def _int32(v):
    if not -2 ** 31 <= v < 2 ** 31:
        raise RuntimeError("value cannot be converted to type int without overflow")
    return v


def _pooling_output_shape(n, kernel, pad, stride, ceil_mode):
    """The CPU's pooling_output_shape, with its checks."""
    if stride == 0:
        raise RuntimeError("stride should not be zero")
    if pad < 0:
        raise RuntimeError(f"pad must be non-negative, but got pad: {pad}")
    if pad > kernel // 2:
        raise RuntimeError(f"pad should be at most half of effective kernel size, but got pad={pad}, "
                           f"kernel_size={kernel} and dilation=1")
    return _pool_out_size(n, kernel, stride, pad, ceil_mode)


def _check_avg_pool(x, kernel_size, stride, padding, ceil_mode, divisor_override, k):
    """The CPU's avg_pool2d/avg_pool3d argument checks, with its messages."""
    if k == 2:
        counts = "either be a single int, or a tuple of two ints"
        stride_counts = "either be omitted, a single int, or a tuple of two ints"
    else:
        counts = "be a single int, or a tuple of three ints"
        stride_counts = "be omitted, a single int, or a tuple of three ints"
    if len(kernel_size) not in (1, k):
        raise RuntimeError(f"avg_pool{k}d: kernel_size must {counts}")
    if len(stride) not in (0, 1, k):
        raise RuntimeError(f"avg_pool{k}d: stride must {stride_counts}")
    if len(padding) not in (1, k):
        raise RuntimeError(f"avg_pool{k}d: padding must {counts}")
    ks = [_int32(v) for v in (list(kernel_size) * k)[:k]]
    st = [_int32(v) for v in (list(stride) * k)[:k]] if len(stride) else ks
    pd = [_int32(v) for v in (list(padding) * k)[:k]]
    if divisor_override is not None and divisor_override == 0:
        raise RuntimeError("divisor must be not zero")
    spatial = x.shape[-k:]
    out = [_pooling_output_shape(spatial[d], ks[d], pd[d], st[d], ceil_mode) for d in range(k)]
    names = ["T", "H", "W"][-k:]
    if min(ks) <= 0:
        raise RuntimeError("kernel size should be greater than zero, but got " +
                           " ".join(f"k{n}: {v}" for n, v in zip(names, ks)))
    if min(st) <= 0:
        raise RuntimeError("stride should be greater than zero, but got " +
                           " ".join(f"d{n}: {v}" for n, v in zip(names, st)))
    planes = x.shape[-k - 1]
    if k == 3 and any(n < kk for n, kk in zip(spatial, ks)):
        raise RuntimeError(f"input image (T: {spatial[0]} H: {spatial[1]} W: {spatial[2]}) smaller than kernel size "
                           f"(kT: {ks[0]} kH: {ks[1]} kW: {ks[2]})")
    if min(out) < 1:
        raise RuntimeError(f"Given input size: ({'x'.join(map(str, (planes, *spatial)))}). Calculated output size: "
                           f"({'x'.join(map(str, (planes, *out)))}). Output size is too small")


def _make_avg_pool(k):
    """avg_poolNd and its backward: one thread per output / per input element
    (shaders/pool.slang), with the CPU's divisors and integer truncation."""
    def forward(x, kernel_size, stride=(), padding=0, ceil_mode=False, count_include_pad=True,
                divisor_override=None):
        padding = [padding] if isinstance(padding, int) else padding
        _check_avg_pool(x, kernel_size, stride, padding, ceil_mode, divisor_override, k)
        ks, st, pd, out, _ = _avg_pool_setup(x, kernel_size, stride, padding, ceil_mode, k)
        head = list(x.shape[:-k])
        planes = x.reshape(math.prod(head), *_as_3d(list(x.shape[-k:]), 1)).to(_pool_dtype(x.dtype))
        y = torch.ops.torch_vulkan.avg_pool_forward(
            planes, _as_3d(out, 1), _as_3d(ks, 1), _as_3d(st, 1), _as_3d(pd, 0), divisor_override or 0,
            count_include_pad, not x.is_floating_point())
        return y.reshape(*head, *out).to(x.dtype).contiguous(memory_format=_pool_memory_format(x, k))

    def backward(grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
        ks, st, pd, out, _ = _avg_pool_setup(x, kernel_size, stride, padding, ceil_mode, k)
        planes = math.prod(x.shape[:-k])
        g = grad_output.reshape(planes, *_as_3d(list(grad_output.shape[-k:]), 1)).to(_pool_dtype(grad_output.dtype))
        gx = torch.ops.torch_vulkan.avg_pool_backward(
            g, _as_3d(list(x.shape[-k:]), 1), _as_3d(ks, 1), _as_3d(st, 1), _as_3d(pd, 0), divisor_override or 0,
            count_include_pad)
        return gx.reshape(x.shape).to(grad_output.dtype).contiguous(memory_format=_pool_memory_format(x, k))

    return forward, backward


# ---------------------------------------------------------------- adaptive pooling

def _adaptive_matrix(n_in, n_out, device):
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    for o in range(n_out):
        start, end = (o * n_in) // n_out, -((-(o + 1) * n_in) // n_out)
        m[o, start:end] = 1.0 / (end - start)
    # (rounded to the input's precision by _apply_dims; float32 on GPUs without float64)
    return m.to(device, torch.float64 if device.type != "vk" or _native_has_float64() else torch.float32)


def _make_adaptive_avg(k):
    def forward(x, output_size):
        mats = [_adaptive_matrix(x.shape[-k + d], output_size[d], x.device) for d in range(k)]
        return _apply_dims(x, mats).contiguous()

    def backward(grad_output, x):
        mats = [_adaptive_matrix(x.shape[-k + d], grad_output.shape[-k + d], x.device) for d in range(k)]
        return _apply_dims(grad_output, [m.t() for m in mats]).contiguous()

    return forward, backward


def _make_adaptive_max_backward(k):
    def backward(grad_output, x, indices):
        head = x.dim() - k
        g = grad_output.reshape(*grad_output.shape[:head], math.prod(grad_output.shape[head:]))
        i = indices.reshape(*indices.shape[:head], math.prod(indices.shape[head:]))
        gx = torch.zeros((*x.shape[:head], math.prod(x.shape[head:])), device=x.device, dtype=x.dtype)
        return gx.scatter_add(-1, i, g).reshape(x.shape)

    return backward


def _per_dim(v, k):
    """A pooling setting as k values (an int or a 1-element list repeats)."""
    v = [v] if isinstance(v, int) else list(v)
    return v * k if len(v) == 1 else v


def _pool_window(kernel_size, stride, padding, dilation, k):
    ks = _per_dim(kernel_size, k)
    return ks, (_per_dim(stride, k) if len(stride) else ks), _per_dim(padding, k), _per_dim(dilation, k)


def _as_3d(v, fill):
    """k spatial values as the kernels' 3 (D, H, W), leading dims `fill`."""
    return [fill] * (3 - len(v)) + list(v)


def _pool_dtype(t):
    """The dtype shaders/pool.slang computes `t` in: 16-bit floats in float32,
    integers in int64 (exact, divided with truncation as the CPU's)."""
    if t in (torch.float32, torch.float64):
        return t
    if t in (torch.float16, torch.bfloat16):
        return torch.float32
    return torch.int64


def _pool_memory_format(x, k):
    """Channels-last results for a channels-last input, as the CPU kernels."""
    if x.dim() == k + 2 and not x.is_contiguous():
        fmt = torch.channels_last if k == 2 else torch.channels_last_3d
        if x.is_contiguous(memory_format=fmt):
            return fmt
    return torch.contiguous_format


def _make_max_pool(k):
    """max_poolNd_with_indices: one thread per output (shaders/pool.slang)."""
    def forward(x, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
        ks, st, pd, dl = _pool_window(kernel_size, stride, padding, dilation, k)
        spatial = list(x.shape[-k:])
        out = [_checks._pooling_output_shape(spatial[d], ks[d], pd[d], st[d], dl[d], ceil_mode) for d in range(k)]
        head = list(x.shape[:-k])
        planes = x.reshape(math.prod(head), *_as_3d(spatial, 1)).to(_pool_dtype(x.dtype))
        values, indices = torch.ops.torch_vulkan.max_pool_forward(
            planes, _as_3d(out, 1), _as_3d(ks, 1), _as_3d(st, 1), _as_3d(pd, 0), _as_3d(dl, 1))
        fmt = _pool_memory_format(x, k)
        values = values.reshape(*head, *out).to(x.dtype).contiguous(memory_format=fmt)
        return values, indices.reshape(*head, *out).contiguous(memory_format=fmt)

    return forward


def _make_max_pool_backward(k):
    """max_poolNd_with_indices_backward: a gather per input element over the
    outputs whose windows cover it (shaders/pool.slang), in output order."""
    def backward(grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices):
        ks, st, pd, dl = _pool_window(kernel_size, stride, padding, dilation, k)
        planes = math.prod(x.shape[:-k])
        out = list(grad_output.shape[-k:])
        g = grad_output.reshape(planes, *_as_3d(out, 1)).to(_pool_dtype(grad_output.dtype))
        i = indices.reshape(planes, *_as_3d(out, 1))
        gx = torch.ops.torch_vulkan.max_pool_backward(
            g, i, _as_3d(list(x.shape[-k:]), 1), _as_3d(ks, 1), _as_3d(st, 1), _as_3d(pd, 0), _as_3d(dl, 1))
        return gx.reshape(x.shape).to(grad_output.dtype).contiguous(memory_format=_pool_memory_format(x, k))

    return backward


# ---------------------------------------------------------------- adaptive max pooling

def _adaptive_windows(n_in, n_out):
    """[n_out, w] window indices (repeating each window's last index, which
    leaves a max unchanged) and the window width w."""
    starts = [(o * n_in) // n_out for o in range(n_out)]
    ends = [-((-(o + 1) * n_in) // n_out) for o in range(n_out)]
    w = max(e - st for st, e in zip(starts, ends))
    return [[min(st + j, e - 1) for j in range(w)] for st, e in zip(starts, ends)], w


def _make_adaptive_max(k):
    def forward(x, output_size):
        head = x.dim() - k
        spatial = x.shape[head:]
        idx, flat = [], None
        for d in range(k):
            windows, w = _adaptive_windows(spatial[d], output_size[d])
            shape = [1] * (2 * k)
            shape[d], shape[k + d] = output_size[d], w
            i = torch.tensor(windows, device=x.device).view(shape)
            idx.append(i)
            stride = math.prod(spatial[d + 1:])
            flat = i * stride if flat is None else flat + i * stride
        windows = x[(Ellipsis, *idx)]  # [..., *out, *w]
        values, arg = windows.flatten(-k).max(-1)
        flat = flat.expand(*output_size, *flat.shape[k:]).flatten(-k)
        indices = torch.gather(flat.expand(*x.shape[:head], *flat.shape), -1, arg.unsqueeze(-1)).squeeze(-1)
        return values, indices

    return forward


# ---------------------------------------------------------------- grid sampling

def _grid_dtype(t):
    """The dtype shaders/gridsample.slang computes `t` in (16-bit floats in float32)."""
    return torch.float64 if t == torch.float64 else torch.float32


def _grid_sampler_forward(input, grid, mode, padding_mode, align_corners):
    _checks.grid_sampler(input, grid, mode, 3)
    if mode not in (0, 1):  # bicubic
        return _on_cpu("grid_sampler_3d", input, grid, mode, padding_mode, align_corners)
    dt = _grid_dtype(input.dtype)
    out = torch.ops.torch_vulkan.grid_sampler(input.to(dt), grid.to(dt), mode, padding_mode, align_corners)
    return out.to(input.dtype)


def _bicubic_grid_sampler_backward(grad_output, input, grid, padding_mode, align_corners, output_mask):
    """grid_sampler_2d's bicubic backward as autograd of its decomposition (Vulkan
    gathers forward, scatter-adds backward): the gradients flow through the
    cubic coefficients and the sampled values, as the CPU's (none through the
    tap indices). Autograd is re-enabled here, since an op's backward kernel
    runs below it."""
    decomp = decomposition_table[torch.ops.aten.grid_sampler_2d.default]
    tls_exclude = torch._C._dispatch_tls_local_exclude_set()
    for key in (torch._C.DispatchKey.AutogradPrivateUse1, torch._C.DispatchKey.ADInplaceOrView):
        tls_exclude = tls_exclude.remove(key)
    with torch._C._ForceDispatchKeyGuard(torch._C._dispatch_tls_local_include_set(), tls_exclude), torch.enable_grad():
        x = input.detach().requires_grad_(bool(output_mask[0]))
        g = grid.detach().requires_grad_(bool(output_mask[1]))
        wrt = [t for t in (x, g) if t.requires_grad]
        if not wrt:
            return None, None
        out = decomp(x, g, 2, padding_mode, align_corners)
        grads = list(torch.autograd.grad(out, wrt, grad_output, allow_unused=True))
    gi = grads.pop(0) if output_mask[0] else None
    gg = grads.pop(0) if output_mask[1] else None
    if output_mask[0] and gi is None:
        gi = torch.zeros_like(input)
    if output_mask[1] and gg is None:
        gg = torch.zeros_like(grid)
    return gi, gg


def _make_grid_sampler_backward(name):
    def backward(grad_output, input, grid, mode, padding_mode, align_corners, output_mask):
        _checks.grid_sampler(input, grid, mode, 2 if "2d" in name else 3)
        expected = [input.shape[0], input.shape[1], *grid.shape[1:-1]]
        if list(grad_output.shape) != expected:
            raise ValueError(f"{name}(): expected grad_output to have sizes {expected}, but got grad_output with "
                             f"sizes {list(grad_output.shape)}")
        if mode not in (0, 1):  # bicubic (2-D only): differentiate PyTorch's decomposition
            return _bicubic_grid_sampler_backward(grad_output, input, grid, padding_mode, align_corners, output_mask)
        # (shaders/gridsample.slang: grad_grid, and each output point's corners
        # with their weights, which one scatter-add turns into grad_input)
        dt = _grid_dtype(input.dtype)
        gg, idx, wgt = torch.ops.torch_vulkan.grid_sampler_backward(
            grad_output.to(dt), input.to(dt), grid.to(dt), mode, padding_mode, align_corners, output_mask[1])
        gi = None
        if output_mask[0]:
            # Rows (n, point, corner) of grad_output[n, :, point] * weight,
            # summed by input position in (point, corner) order, as the CPU:
            # one sort of the n * points * corners keys, all channels at once.
            n, c = input.shape[:2]
            points, corners = idx.shape[1], idx.shape[2]
            plane = math.prod(input.shape[2:])
            go = grad_output.to(dt).reshape(n, c, points).transpose(1, 2)  # [n, points, c]
            rows = (go.unsqueeze(2) * wgt.reshape(n, points, corners, 1)).reshape(n * points * corners, c)
            base = (torch.arange(n, device=idx.device) * plane).reshape(n, 1, 1)
            keys = torch.where(idx >= 0, idx + base, -1).reshape(-1)
            gi = torch.ops.torch_vulkan.sum_rows_by_key(keys, rows, n * plane)  # [n * plane, c]
            gi = gi.reshape(n, plane, c).transpose(1, 2).reshape(input.shape).to(input.dtype)
        return gi, (gg.to(grid.dtype) if output_mask[1] else None)

    return backward


def _on_cpu(name, *args):
    op = getattr(torch.ops.aten, name).default
    return _decomp._cpu_fallback(op, op.name(), args, {})


# ---------------------------------------------------------------- backward kernels with no decomposition

def _shrink_backward(name):
    """softshrink_backward / hardshrink_backward: the gradient where |x| > lambd
    (the CPU's test, lambd in x's dtype), else 0."""
    def kernel(grad_output, x, lambd):
        if not (x.is_floating_point() and grad_output.dtype == x.dtype):
            return _on_cpu(name, grad_output, x, lambd)
        lam = torch.tensor(lambd, dtype=x.dtype)
        inside = (x >= (-lam).to(x.device)) & (x <= lam.to(x.device))
        return torch.where(inside, torch.zeros((), dtype=x.dtype, device=x.device), grad_output)

    return kernel


def _replication_pad_backward(dims):
    """replication_pad{1,2,3}d_backward: each padded dim's gradient summed into
    the input position it replicates (the clamped index), a dim at a time."""
    name = f"replication_pad{dims}d_backward"

    def kernel(grad_output, x, padding):
        ok = len(padding) == 2 * dims and grad_output.dim() == x.dim() and x.dim() >= dims + 1
        for i in range(dims if ok else 0):
            d = x.dim() - 1 - i
            ok = ok and x.shape[d] > 0 and grad_output.shape[d] == x.shape[d] + padding[2 * i] + padding[2 * i + 1]
        if not ok:
            return _on_cpu(name, grad_output, x, padding)
        g = grad_output
        for i in range(dims):
            d = x.dim() - 1 - i
            idx = (torch.arange(g.shape[d], device=g.device) - padding[2 * i]).clamp(0, x.shape[d] - 1)
            shape = list(g.shape)
            shape[d] = x.shape[d]
            g = torch.zeros(shape, dtype=g.dtype, device=g.device).index_add_(d, idx, g)
        return g

    return kernel


def _targets_in_range(target, low, high):
    """True if every target is in [low, high) (one host transfer)."""
    if target.numel() == 0:
        return True
    lo, hi = torch.stack([target.min(), target.max()]).cpu().tolist()
    return lo >= low and hi < high


def _margin_scale(grad_input, grad_output, reduction):
    # (the CPU's: a scalar grad_output for reduced losses, else one per sample)
    if reduction != 0 or grad_output.dim() == 0:
        return grad_input * grad_output.reshape(())
    return grad_input * grad_output.reshape(-1, 1)


def _multi_margin_loss_backward(grad_output, x, target, p, margin, weight=None, reduction=1):
    """The CPU's multi_margin_loss_backward: for each class d other than the
    target y with z = margin - x[y] + x[d] > 0, h = g (p = 1) or 2 g z (times
    weight[y]) at d and -sum h at y; g = 1 / C (1 / (N C) for mean)."""
    args = (grad_output, x, target, p, margin, weight, reduction)
    if not (x.is_floating_point() and target.dtype == torch.int64 and p in (1, 2) and 1 <= x.dim() <= 2) or             (weight is not None and (weight.dim() != 1 or weight.numel() != x.shape[-1])):
        return _on_cpu("multi_margin_loss_backward", *args)
    xs = x.reshape(-1, x.shape[-1])
    n, c = xs.shape
    t = target.reshape(-1)
    if t.numel() != n or c == 0 or not _targets_in_range(t, 0, c):
        return _on_cpu("multi_margin_loss_backward", *args)
    g = 1.0 / (n * c) if reduction == 1 else 1.0 / c
    g = torch.tensor(g, dtype=x.dtype).item()  # (in x's dtype, as the CPU's scalar_t)
    ty = t.unsqueeze(1)
    z = margin - xs.gather(1, ty) + xs
    h = z.new_full((), g) if p == 1 else 2 * g * z
    hit = (z > 0) & (torch.arange(c, device=x.device) != ty)
    h = torch.where(hit, h, torch.zeros((), dtype=x.dtype, device=x.device))
    if weight is not None:
        h = h * weight.to(x.dtype)[t].unsqueeze(1)
    gi = h.scatter(1, ty, -h.sum(1, keepdim=True))
    return _margin_scale(gi, grad_output, reduction).reshape(x.shape)


def _multilabel_margin_loss_backward(grad_output, x, target, reduction, is_target):
    """The CPU's multilabel_margin_loss_backward: for each target j (up to the
    first -1) and each non-target d with z = 1 - x[j] + x[d] > 0, -g at j and
    +g at d; g = 1 / C (1 / (N C) for mean)."""
    args = (grad_output, x, target, reduction, is_target)
    if not (x.is_floating_point() and target.dtype == torch.int64 and 1 <= x.dim() <= 2 and
            target.shape == x.shape and is_target.shape == x.shape):
        return _on_cpu("multilabel_margin_loss_backward", *args)
    xs = x.reshape(-1, x.shape[-1])
    n, c = xs.shape
    tg = target.reshape(n, c)
    if c == 0 or not _targets_in_range(tg, -1, c):
        return _on_cpu("multilabel_margin_loss_backward", *args)
    g = 1.0 / (n * c) if reduction == 1 else 1.0 / c
    g = torch.tensor(g, dtype=x.dtype).item()
    valid = torch.cumprod((tg >= 0).to(torch.int64), 1).bool()  # (targets before the first -1)
    tc = tg.clamp(min=0)
    z = 1 - xs.gather(1, tc).unsqueeze(2) + xs.unsqueeze(1)  # [N, target j, class d]
    hit = (z > 0) & valid.unsqueeze(2) & (is_target.reshape(n, c) == 0).unsqueeze(1)
    gi = g * hit.sum(1).to(x.dtype)
    gi = gi.scatter_add(1, tc, -g * hit.sum(2).to(x.dtype) * valid.to(x.dtype))
    return _margin_scale(gi, grad_output, reduction).reshape(x.shape)


# ---------------------------------------------------------------- distances

def _pairwise_diff(x1, x2):
    return x1.unsqueeze(-2) - x2.unsqueeze(-3)  # [..., P, R, M]


def _cdist_forward(x1, x2, p, compute_mode=None):
    if x1.shape[-1] == 0:
        return torch.zeros((*x1.shape[:-1], x2.shape[-2]), device=x1.device, dtype=x1.dtype)
    diff = _pairwise_diff(x1, x2)
    if p == 0:
        # (as the CPU's min(ceil(|d|), 1): a NaN difference, e.g. inf - inf, propagates)
        return torch.where(diff.isnan(), diff, (diff != 0).to(diff.dtype)).sum(-1)
    if p == math.inf:
        return diff.abs().amax(-1)
    if p == 1:
        return diff.abs().sum(-1)
    if p == 2:
        return diff.square().sum(-1).sqrt()
    return diff.abs().pow(p).sum(-1).pow(1.0 / p)


def _cdist_backward(grad, x1, x2, p, dist):
    """Gradient w.r.t. x1 of cdist(x1, x2), shaped as PyTorch's meta kernel:
    x1 broadcast to both batch shapes, or x1's own shape when anything is empty."""
    batch = list(torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2]))
    if 0 in (x1.shape[-2], x2.shape[-2], x1.shape[-1], math.prod(batch)):
        return torch.zeros_like(x1)
    r = _cdist_backward_impl(grad, x1.expand(*batch, *x1.shape[-2:]), x2, p, dist)
    return r.contiguous()


def _cdist_backward_impl(grad, x1, x2, p, dist):
    if p == 0 or x1.shape[-1] == 0:
        return torch.zeros_like(x1)
    diff = _pairwise_diff(x1, x2)
    d = dist.unsqueeze(-1)
    if p == 1:
        dd = torch.sign(diff)
    elif p == 2:
        dd = torch.where(d == 0, torch.zeros_like(diff), diff / d)
    elif p == math.inf:
        dd = torch.sign(diff) * (diff.abs() == d).to(diff.dtype)
    else:  # (zero where the difference is: |0|^(p - 1) is inf for p < 1)
        dd = torch.where((d == 0) | (diff == 0), torch.zeros_like(diff),
                         torch.sign(diff) * diff.abs().pow(p - 1) / d.pow(p - 1))
    return (grad.unsqueeze(-1) * dd).sum(-2)


# ---------------------------------------------------------------- repeat_interleave

def _repeat_interleave_tensor(repeats, output_size=None):
    """i repeated repeats[i] times, for each i: each output position's i is
    the number of cumulative repeat counts at or below it."""
    if repeats.dim() != 1:
        raise RuntimeError("repeat_interleave only accept 1D vector as repeat")
    if repeats.dtype not in (torch.int32, torch.int64):
        op = torch.ops.aten.repeat_interleave.Tensor  # (the CPU's error)
        return _decomp._cpu_fallback(op, op.name(), (repeats,), {"output_size": output_size})
    if repeats.numel() == 0:
        return torch.empty(0, dtype=repeats.dtype, device=repeats.device)
    ends = repeats.to(torch.int64).cumsum(0)
    total, smallest = torch.stack([ends[-1], repeats.min().to(torch.int64)]).tolist()
    if smallest < 0:
        raise RuntimeError("repeats can not be negative")
    if output_size is not None and output_size != total:
        raise RuntimeError("allocated size does not match required size")
    positions = torch.arange(total, device=repeats.device)
    return torch.searchsorted(ends, positions, right=True).to(repeats.dtype)


# ---------------------------------------------------------------- triu_indices / tril_indices

def _tri_indices(upper, row, col, offset=0, *, dtype=None, layout=None, device=None, pin_memory=None):
    """Row-major (row, col) indices of the upper / lower triangle, in integer
    arithmetic (PyTorch's decomposition computes them with float64 square
    roots, which GPUs without float64 lack)."""
    if row < 0 or col < 0:
        raise RuntimeError(f"{'triu' if upper else 'tril'}_indices expects row and col to be non-negative, got {row} and {col}")
    dtype = torch.int64 if dtype is None else dtype
    if dtype not in (torch.int32, torch.int64):
        from . import _checks
        raise NotImplementedError(f'"{"triu" if upper else "tril"}_indices" not implemented for '
                                  f"'{_checks._C10_NAMES[str(dtype)[6:]]}'")
    first = [max(0, r + offset) if upper else 0 for r in range(row)]
    count = [max(0, col - f) if upper else min(col, max(0, r + offset + 1)) for r, f in enumerate(first)]
    n = sum(count)
    if n == 0:
        return torch.empty(2, 0, dtype=dtype, device=device)
    r = torch.arange(row, device=device)
    first_t = (r + offset).clamp(min=0) if upper else torch.zeros_like(r)
    count_t = (col - first_t).clamp(min=0) if upper else (r + offset + 1).clamp(0, col)
    end = count_t.cumsum(0)
    i = torch.arange(n, device=device)
    rows = torch.searchsorted(end, i, right=True)
    cols = first_t[rows] + i - (end - count_t)[rows]
    return torch.stack([rows, cols]).to(dtype)


def _pdist_forward(x, p=2):
    n = x.shape[0]
    i, j = torch.triu_indices(n, n, 1, device=x.device)
    return _cdist_forward(x, x, p)[i, j]


def _pdist_backward(grad, x, p, pdist):
    n = x.shape[0]
    i, j = torch.triu_indices(n, n, 1, device=x.device)
    full = torch.zeros((n, n), device=x.device, dtype=x.dtype).index_put_((i, j), grad)
    dist = torch.zeros((n, n), device=x.device, dtype=x.dtype).index_put_((i, j), pdist)
    full, dist = full + full.t(), dist + dist.t()
    return _cdist_backward(full, x, x, p, dist)


# ---------------------------------------------------------------- fractional max pooling

def _make_fractional_max(k):
    # random_samples[..., s] drives dim d: 2-D uses (W, H) = (0, 1), 3-D (T, H, W) = (0, 1, 2).
    sample_of = [1, 0] if k == 2 else [0, 1, 2]

    def forward(x, kernel_size, output_size, random_samples):
        batched = x.dim() == k + 2
        xb = x if batched else x.unsqueeze(0)
        n, c = xb.shape[:2]
        spatial = xb.shape[2:]
        dtype = xb.dtype
        idx, flat = [], None
        for d in range(k):
            n_in, n_out, pool = spatial[d], output_size[d], kernel_size[d]
            u = random_samples[:n, :c, sample_of[d]].to(dtype)  # [n, c] (the CPU reads those; samples may have more)
            if n_out > 1:
                alpha = torch.tensor(float(n_in - pool), dtype=dtype) / (n_out - 1)
                steps = torch.arange(n_out, device=x.device, dtype=dtype)
                seq = (torch.floor((steps + u.unsqueeze(-1)) * alpha) - torch.floor(u * alpha).unsqueeze(-1)).long()
                seq[..., -1] = n_in - pool
            else:
                seq = torch.full((n, c, 1), n_in - pool, device=x.device, dtype=torch.long)
            pos = seq.unsqueeze(-1) + torch.arange(pool, device=x.device)  # [n, c, out_d, pool_d]
            shape = [n, c] + [1] * (2 * k)
            shape[2 + d], shape[2 + k + d] = n_out, pool
            pos = pos.reshape(shape)
            idx.append(pos)
            stride = math.prod(spatial[d + 1:])
            flat = pos * stride if flat is None else flat + pos * stride
        bn = torch.arange(n, device=x.device).reshape([n, 1] + [1] * (2 * k))
        bc = torch.arange(c, device=x.device).reshape([1, c] + [1] * (2 * k))
        windows = xb[(bn, bc, *idx)]  # [n, c, *out, *pool]
        values, arg = windows.flatten(-k).max(-1)
        flat = flat.expand(windows.shape).flatten(-k)
        indices = torch.gather(flat, -1, arg.unsqueeze(-1)).squeeze(-1)
        if not batched:
            values, indices = values.squeeze(0), indices.squeeze(0)
        return values, indices

    def backward(grad_output, x, kernel_size, output_size, indices):
        head = x.dim() - k
        g = grad_output.reshape(*grad_output.shape[:head], math.prod(grad_output.shape[head:]))
        i = indices.reshape(*indices.shape[:head], math.prod(indices.shape[head:]))
        gx = torch.zeros((*x.shape[:head], math.prod(x.shape[head:])), device=x.device, dtype=x.dtype)
        return gx.scatter_add(-1, i, g).reshape(x.shape)

    return forward, backward


# ---------------------------------------------------------------- margin losses

# PyTorch's decompositions, with the CPU's output shape without reduction:
# 0-dim for an unbatched target (where they return [1]).

def _multi_margin_loss(x, target, p=1, margin=1, weight=None, reduction=1):
    r = decomposition_table[aten.multi_margin_loss.default](x, target, p, margin, weight, reduction)
    return r.reshape(()) if target.dim() == 0 and reduction == 0 else r


def _multilabel_margin_loss_forward(x, target, reduction):
    r, is_target = decomposition_table[aten.multilabel_margin_loss_forward.default](x, target, reduction)
    return (r.reshape(()) if target.dim() <= 1 and reduction == 0 else r), is_target


# ---------------------------------------------------------------- bucketize

def _bucketize_scalar(x, boundaries, out_int32=False, right=False):
    # (the CPU's scalar_to_tensor: a 0-dim double or int64 on the boundaries' device)
    # (single precision on GPUs without float64)
    f64 = _native_has_float64()
    dtype = ((torch.float64 if f64 else torch.float32) if isinstance(x, float) else
             (torch.complex128 if f64 else torch.complex64) if isinstance(x, complex) else torch.int64)
    return torch.bucketize(torch.tensor(x, dtype=dtype, device=boundaries.device), boundaries,
                           out_int32=out_int32, right=right)


# ---------------------------------------------------------------- ldexp

def _ldexp(x, other):
    # PyTorch's x * pow(2, other), with 2 in x's dtype (float32 for integers),
    # but a half-precision power computed in float32: 2^20 overflows float16
    # while 2^-10 * 2^20 does not.
    two_dtype = x.dtype if x.is_floating_point() or x.is_complex() else torch.float32
    power_dtype = torch.result_type(torch.empty((), dtype=two_dtype), other)
    if power_dtype in (torch.float16, torch.bfloat16):
        dtype = torch.result_type(x, torch.empty((), dtype=power_dtype))
        return (x.float() * torch.pow(x.new_full((), 2.0, dtype=torch.float32), other)).to(dtype)
    return x * torch.pow(x.new_full((), 2.0, dtype=two_dtype), other)


# ---------------------------------------------------------------- rrelu

def _rrelu_with_noise(x, noise, lower=1.0 / 8, upper=1.0 / 3, training=False, generator=None):
    if not training:
        return F.leaky_relu(x, (lower + upper) / 2)
    slopes = torch.empty_like(x).uniform_(lower, upper, generator=generator)
    negative = x < 0
    noise.copy_(torch.where(negative, slopes, torch.ones_like(x)))
    return torch.where(negative, x * slopes, x)


def _rrelu_with_noise_(x, noise, lower=1.0 / 8, upper=1.0 / 3, training=False, generator=None):
    # (the same random numbers as the functional form, from the "vk" generator)
    return x.copy_(_rrelu_with_noise(x, noise, lower, upper, training, generator))


# ---------------------------------------------------------------- antialiased upsampling

def _aa_filter(kind, x):
    x = abs(x)
    if kind == "bilinear":
        return max(0.0, 1.0 - x)
    a = -0.5  # bicubic with antialiasing uses a = -0.5
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def _aa_matrix(kind, n_in, n_out, align_corners, scale_opt, like):
    """out x in weights of antialiased interpolation, as PyTorch's
    _compute_index_ranges_weights / _compute_weights_aa (UpSampleKernel.cpp)."""
    if align_corners:
        scale = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
    else:
        scale = 1.0 / scale_opt if scale_opt is not None and scale_opt > 0 else n_in / n_out
    interp = 2 if kind == "bilinear" else 4
    support = interp * 0.5 * scale if scale >= 1.0 else interp * 0.5
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    for i in range(n_out):
        center = scale * (i + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xsize = min(int(center + support + 0.5), n_in) - xmin
        w = [_aa_filter(kind, (j + xmin - center + 0.5) * invscale) for j in range(xsize)]
        total = sum(w)
        for j, v in enumerate(w):
            m[i, xmin + j] = v / total if total != 0 else v
    return m.to(like.device, like.dtype)


# ---------------------------------------------------------------- uint8 bilinear / bicubic, as the CPU's

def _u8_filter(kind, antialias, x):
    """HelperInterpLinear / HelperInterpCubic::aa_filter (a = -0.5 with
    antialiasing, -0.75 without), in double."""
    x = abs(x)
    if kind == "bilinear":
        return 1.0 - x if x < 1.0 else 0.0
    a = -0.5 if antialias else -0.75
    if x < 1.0:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2.0:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return 0.0


def _c_trunc(x):
    return int(x)  # (C++'s static_cast<int64_t> of a double: toward zero, as Python's int)


@functools.lru_cache(maxsize=256)
def _u8_taps(kind, antialias, n_in, n_out, align_corners, scale_opt):
    """The CPU's fixed-point taps for one dim of a uint8 resize
    (_compute_index_ranges_int16_weights): per output its first input
    index, tap count and int16 weights, and the weights' precision (bits)."""
    interp = 2 if kind == "bilinear" else 4
    if align_corners:
        scale = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
    else:
        scale = 1.0 / scale_opt if scale_opt is not None and scale_opt > 0 else n_in / n_out
    taps = []
    wt_max = 0.0
    if antialias:
        support = interp * 0.5 * scale if scale >= 1.0 else interp * 0.5
        max_size = math.ceil(support) * 2 + 1
        invscale = 1.0 / scale if scale >= 1.0 else 1.0
        for i in range(n_out):
            center = scale * (i + 0.5)
            xmin = max(_c_trunc(center - support + 0.5), 0)
            xsize = min(max(min(_c_trunc(center + support + 0.5), n_in) - xmin, 0), max_size)
            w = [_u8_filter(kind, True, (j + xmin - center + 0.5) * invscale) for j in range(xsize)]
            total = sum(w)  # (accumulated in order, as the CPU's loop)
            if total != 0.0:
                w = [v / total for v in w]
                wt_max = max([wt_max] + w)
            taps.append((xmin, xsize, w))
    else:
        max_size = interp
        cubic = max_size > 2
        support = int(max_size * 0.5)
        for i in range(n_out):
            if align_corners:
                real = scale * i
            else:
                real = scale * (i + 0.5) - 0.5
                if not cubic and real < 0.0:
                    real = 0.0
            # guard_index_and_lambda: floorf (a float32 floor) of the double index
            index = min(int(math.floor(_f32(real))), n_in - 1)
            lam = min(max(real - index, 0.0), 1.0)
            umin = index - support + 1
            xmin = max(umin, 0)
            xsize = min(max(min(index + support + 1, n_in) - xmin, 0), max_size)
            wt = [0.0] * max_size
            w_index = 0
            for j in range(max_size):
                wt[j] = 0.0
                w = _u8_filter(kind, False, (j + 1 - support) - lam)
                if umin + j <= 0:
                    w_index = 0
                elif umin + j >= n_in - 1:
                    w_index = xsize - 1
                wt[w_index] += w
                wt_max = max(wt_max, wt[w_index])
                w_index += 1
            taps.append((xmin, xsize, wt[:xsize]))
    precision = 0
    while precision < 22 and int(0.5 + wt_max * (1 << (precision + 1))) < (1 << 15):
        precision += 1
    width = max([t[1] for t in taps] + [1])
    idx, wts = [], []
    for xmin, xsize, w in taps:
        ints = [int(-0.5 + v * (1 << precision)) if v * (1 << precision) < 0 else int(0.5 + v * (1 << precision))
                for v in w]
        idx.append([min(xmin + j, n_in - 1) if j < xsize else xmin for j in range(width)])
        wts.append(ints + [0] * (width - xsize))
    return idx, wts, precision


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _u8_resize(x, output_size, align_corners, scales, kind, antialias):
    """uint8 bilinear/bicubic resize of N, C, H, W as the CPU's separable
    fixed-point kernel: the width pass, then the height pass, each output
    the int16-weighted sum of its taps plus half, shifted by the weights'
    precision and clamped to [0, 255] (the intermediate kept as uint8)."""
    out_hw = list(output_size)
    if list(x.shape[-2:]) == out_hw:
        return x.clone()
    y = x.contiguous()
    for d in (3, 2):  # (W first, as the CPU)
        n_in, n_out = y.shape[d], out_hw[d - 2]
        if n_in == n_out:
            continue
        s = scales[d - 2] if scales is not None and d - 2 < len(scales) else None
        idx, wts, precision = _u8_taps(kind, antialias, n_in, n_out, bool(align_corners), s)
        idx_t = torch.tensor(idx, dtype=torch.long, device=y.device)
        w_t = torch.tensor(wts, dtype=torch.int32, device=y.device)
        moved = y.movedim(d, -1).to(torch.int32)
        g = moved.index_select(-1, idx_t.reshape(-1)).reshape(*moved.shape[:-1], n_out, idx_t.shape[1])
        acc = (g * w_t).sum(-1, dtype=torch.int32) + (1 << (precision - 1))
        y = torch.bitwise_right_shift(acc, precision).clamp(0, 255).to(torch.uint8).movedim(-1, d).contiguous()
    return y


def _upsample_bicubic2d(x, output_size, align_corners, scales_h=None, scales_w=None):
    """uint8 as the CPU's fixed-point kernel; other dtypes PyTorch's
    decomposition (as before this registration)."""
    if x.dtype == torch.uint8 and x.dim() == 4 and len(output_size) == 2 and min(x.shape[-2:]) > 0             and min(output_size) > 0:
        return _u8_resize(x, output_size, align_corners, (scales_h, scales_w), "bicubic", False)
    return decomposition_table[aten.upsample_bicubic2d.default](x, output_size, align_corners, scales_h, scales_w)


def _make_aa(kind):
    def matrices(input_size, output_size, align_corners, scales_h, scales_w, like):
        return [_aa_matrix(kind, input_size[2 + d], output_size[d], align_corners, (scales_h, scales_w)[d], like)
                for d in range(2)]

    def forward(x, output_size, align_corners, scales_h=None, scales_w=None):
        if x.dtype == torch.uint8:  # (the only integer dtype the CPU takes)
            return _u8_resize(x, output_size, align_corners, (scales_h, scales_w), kind, True)
        mats = matrices(x.shape, output_size, align_corners, scales_h, scales_w, x)
        return _apply_dims(x, mats).contiguous()

    def backward(grad_output, output_size, input_size, align_corners, scales_h=None, scales_w=None):
        mats = matrices(input_size, output_size, align_corners, scales_h, scales_w, grad_output)
        return _apply_dims(grad_output, [m.t() for m in mats]).contiguous()

    return forward, backward


# ---------------------------------------------------------------- embedding_bag

def _embedding_bag(weight, indices, offsets, scale_grad_by_freq=False, mode=0, sparse=False,
                   per_sample_weights=None, include_last_offset=False, padding_idx=-1, forward_only=False):
    n = indices.numel()
    nb = offsets.numel() - (1 if include_last_offset else 0)
    dim = weight.shape[1]
    device, dtype = weight.device, weight.dtype
    starts = offsets[:nb]
    positions = torch.arange(n, device=device)
    offset2bag = torch.searchsorted(starts, positions, right=True) - 1 if n else positions
    valid = indices != padding_idx if padding_idx >= 0 else torch.ones(n, dtype=torch.bool, device=device)
    validf = valid.to(dtype)
    bag_size = torch.zeros(nb, dtype=torch.long, device=device).index_add(0, offset2bag, valid.long()) if n \
        else torch.zeros(nb, dtype=torch.long, device=device)
    rows = weight.index_select(0, indices) if n else weight.new_zeros((0, dim))
    if mode in (0, 1):
        # (16-bit weights: products and sums in float32, rounded once, as the CPU)
        acc = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        scale = validf.to(acc) if per_sample_weights is None else validf.to(acc) * per_sample_weights.to(acc)
        out = torch.zeros((nb, dim), device=device, dtype=acc).index_add(0, offset2bag, rows.to(acc) * scale.unsqueeze(1))
        if mode == 1:
            out = out / bag_size.clamp(min=1).unsqueeze(1).to(acc)
            max_indices = bag_size.clone()
        else:
            max_indices = torch.zeros(nb, dtype=torch.long, device=device)
        out = out.to(dtype)
        # The auxiliary outputs as PyTorch predicts them for a non-CPU device
        # (as CUDA's): bag_size has one entry per offset, max_indices is
        # empty except in max mode.
        if include_last_offset:
            bag_size = torch.cat([bag_size, bag_size.new_zeros(1)])
        return out, offset2bag, bag_size, max_indices.new_zeros(0)
    # max: per (bag, feature) the first row reaching the maximum.
    neg = torch.finfo(dtype).min
    vals = torch.where(valid.unsqueeze(1), rows, torch.full_like(rows, neg))
    out = torch.full((nb, dim), neg, device=device, dtype=dtype).scatter_reduce(
        0, offset2bag.unsqueeze(1).expand(n, dim), vals, "amax", include_self=True)
    empty = (bag_size == 0).unsqueeze(1)
    out = torch.where(empty, torch.zeros_like(out), out)
    hit = (vals == out.index_select(0, offset2bag)) & valid.unsqueeze(1)
    pos = torch.where(hit, positions.unsqueeze(1).expand(n, dim), torch.full((n, dim), n, device=device))
    first = torch.full((nb, dim), n, device=device, dtype=torch.long).scatter_reduce(
        0, offset2bag.unsqueeze(1).expand(n, dim), pos, "amin", include_self=True)
    max_indices = torch.where(first < n, indices.index_select(0, first.clamp(max=max(n - 1, 0)).flatten()).view(nb, dim)
                              if n else first, torch.zeros_like(first))
    if include_last_offset:
        bag_size = torch.cat([bag_size, bag_size.new_zeros(1)])
    return out, offset2bag, bag_size, max_indices


def _embedding_bag_forward_only(*args):
    return _embedding_bag(*args, forward_only=True)


def _embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights,
                                  scale_grad_by_freq, mode, per_sample_weights, padding_idx=-1):
    dim = grad.shape[1]
    gw = torch.zeros((num_weights, dim), device=grad.device, dtype=grad.dtype)
    bag_size = bag_size[:grad.shape[0]]  # (one entry per offset, with include_last_offset one too many)
    if mode == 2:
        valid = bag_size > 0
        rows = maximum_indices[valid]
        return gw.index_put_((rows.flatten(), torch.arange(dim, device=grad.device).repeat(rows.shape[0])),
                             grad[valid].flatten(), accumulate=True)
    g = grad.index_select(0, offset2bag)
    if mode == 1:
        g = g / bag_size.index_select(0, offset2bag).clamp(min=1).unsqueeze(1).to(grad.dtype)
    if per_sample_weights is not None:
        g = g * per_sample_weights.unsqueeze(1)
    if padding_idx >= 0:
        g = g * (indices != padding_idx).unsqueeze(1).to(grad.dtype)
    if scale_grad_by_freq:
        counts = torch.zeros(num_weights, device=grad.device, dtype=grad.dtype).index_add(
            0, indices, torch.ones_like(indices, dtype=grad.dtype))
        g = g / counts.index_select(0, indices).unsqueeze(1)
    return gw.index_add(0, indices, g)


def _embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights,
                            scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx=-1):
    if sparse:
        return _on_cpu("_embedding_bag_backward", grad, indices, offsets, offset2bag, bag_size, maximum_indices,
                       num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx)
    return _embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights,
                                         scale_grad_by_freq, mode, per_sample_weights, padding_idx)


def _embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, offset2bag, mode, padding_idx=-1):
    rows = weight.index_select(0, indices)
    out = (grad.index_select(0, offset2bag) * rows).sum(1)
    if padding_idx >= 0:
        out = out * (indices != padding_idx).to(out.dtype)
    return out


# ---------------------------------------------------------------- matrix exponential

# Higham (2005), "The scaling and squaring method for the matrix exponential
# revisited": degree-13 Pade approximant after scaling by 2^-s.
def _native_has_float64():
    return bool(_native.capabilities() & 1)


_PADE13 = (64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
           129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0,
           40840800.0, 960960.0, 16380.0, 182.0, 1.0)
_THETA13 = 5.371920351148152


def _matrix_exp(a):
    if not (a.is_floating_point() or a.is_complex()) or a.shape[-1] == 0:
        return _on_cpu("linalg_matrix_exp", a)
    if a.dtype in (torch.float16, torch.bfloat16):
        return _matrix_exp(a.float()).to(a.dtype)  # (as the CPU: computed in single precision)
    if a.dtype in (torch.float32, torch.complex64) and _native_has_float64():
        # The solve and the squarings lose ~4 bits in single precision; double
        # keeps the result as accurate as single-precision storage allows.
        wide = torch.float64 if a.dtype == torch.float32 else torch.complex128
        return _matrix_exp(a.to(wide)).to(a.dtype)
    shape = a.shape
    x = a.reshape(-1, shape[-2], shape[-1])
    norm = x.abs().sum(-2).amax(-1)  # 1-norm of each matrix
    s = torch.clamp(torch.ceil(torch.log2(norm / _THETA13)), min=0)
    s = torch.where(torch.isfinite(s), s, torch.zeros_like(s))
    x = x / torch.pow(2.0, s).to(x.dtype).view(-1, 1, 1)
    b = _PADE13
    eye = torch.eye(shape[-1], device=x.device, dtype=x.dtype).expand_as(x)
    x2 = x @ x
    x4 = x2 @ x2
    x6 = x4 @ x2
    u = x @ (x6 @ (b[13] * x6 + b[11] * x4 + b[9] * x2) + b[7] * x6 + b[5] * x4 + b[3] * x2 + b[1] * eye)
    v = x6 @ (b[12] * x6 + b[10] * x4 + b[8] * x2) + b[6] * x6 + b[4] * x4 + b[2] * x2 + b[0] * eye
    r = torch.linalg.solve(v - u, v + u)
    steps = int(s.max().item()) if s.numel() else 0
    for i in range(steps):
        r = torch.where((s > i).view(-1, 1, 1), r @ r, r)
    return r.reshape(shape)


# ---------------------------------------------------------------- hashing, frexp

def _xor_last(x):
    """XOR of x [..., n] (int64) over its last dim, as a log-depth tree."""
    if x.shape[-1] == 0:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    while x.shape[-1] > 1:
        if x.shape[-1] % 2:
            x = torch.cat([x, torch.zeros_like(x[..., :1])], -1)
        x = torch.bitwise_xor(x[..., 0::2], x[..., 1::2])
    return x[..., 0]


def _hash_tensor(self, dim=(), *, keepdim=False, mode=0):
    if mode != 0:
        return _on_cpu("hash_tensor", self, dim, keepdim=keepdim, mode=mode)
    # The CPU's checks of the reduced dims.
    if self.dim() > 64:
        raise RuntimeError("only tensors with up to 64 dims are supported")
    n = max(self.dim(), 1)
    for d in dim:
        if not -n <= d < n:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-n}, {n - 1}], but got {d})")
    wrapped = [d % n for d in dim]
    for d in wrapped:
        if wrapped.count(d) > 1:
            raise RuntimeError(f"dim {d} appears multiple times in the list of dims")
    for d in wrapped if dim else range(self.dim()):
        if self.dim() > 0 and self.shape[d] == 0:
            raise IndexError(f"hash_tensor: Expected reduction dim {d} to have non-zero size.")
    # Elements as 64-bit floats or integers, their bits XOR-reduced.
    if not self.is_floating_point():
        bits = self.to(torch.int64)
    elif self.dtype == torch.float64 or _native_has_float64():
        bits = self.to(torch.float64).view(torch.int64)
    else:
        bits = _float_bits_as_double(self)
    dims = sorted(d % max(self.dim(), 1) for d in dim) if dim else list(range(self.dim()))
    if self.dim() == 0:
        return bits.clone().view(torch.uint64)
    keep = [d for d in range(self.dim()) if d not in dims]
    reduced = math.prod(self.shape[d] for d in dims)  # (not -1: ambiguous with 0 elements)
    moved = bits.permute(keep + dims).reshape([self.shape[d] for d in keep] + [reduced])
    out = _xor_last(moved)
    if keepdim:
        for d in dims:
            out = out.unsqueeze(d)
    return out.view(torch.uint64)


def _float_bits_as_double(x):
    """The bits of x (float32 or narrower) converted to float64, as int64,
    in integer arithmetic (for GPUs without float64): exact, with NaNs
    quieted as the CPU converts them."""
    b = x.to(torch.float32).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign, e, m = b >> 31, (b >> 23) & 0xFF, b & 0x7FFFFF
    # Subnormals: m 2^-149 with m's leading bit p (frexp is exact for m < 2^24).
    p = torch.frexp(m.to(torch.float32)).exponent.to(torch.int64) - 1
    sub = (e == 0) & (m != 0)
    e64 = torch.where(e == 0xFF, 0x7FF, torch.where(e == 0, torch.where(sub, p - 149 + 1023, 0), e + 896))
    m64 = torch.where(sub, (m << (52 - p.clamp(0, 22))) & ((1 << 52) - 1), m << 29)
    m64 = torch.where((e == 0xFF) & (m != 0), m64 | (1 << 51), m64)
    return (sign << 63) | (e64 << 52) | m64


def _hash_tensor_out(self, dim=(), *, keepdim=False, mode=0, out):
    r = _hash_tensor(self, dim, keepdim=keepdim, mode=mode)
    out.resize_(r.shape)
    return out.copy_(r)


# Per float dtype: integer view, mantissa bits, exponent mask, exponent of
# [0.5, 1), and the power of two that makes subnormals normal.
_FREXP = {torch.float32: (torch.int32, 23, 0xFF, 126, 24), torch.float64: (torch.int64, 52, 0x7FF, 1022, 54)}


def _frexp_nonfinite_exponent():
    # C leaves frexp's exponent for inf/NaN unspecified (0 with glibc, -1 with
    # the MSVC runtime): use whatever the CPU kernel gives on this platform.
    global _FREXP_NONFINITE
    if _FREXP_NONFINITE is None:
        _FREXP_NONFINITE = int(torch.frexp(torch.tensor(float("inf"))).exponent)
    return _FREXP_NONFINITE


_FREXP_NONFINITE = None


def _frexp_out(self, *, mantissa, exponent):
    if not self.is_floating_point():
        raise RuntimeError("torch.frexp() only supports floating-point dtypes")
    if mantissa.dtype != self.dtype:
        raise RuntimeError(f"torch.frexp() expects mantissa to have dtype {self.dtype} but got {mantissa.dtype}")
    if exponent.dtype != torch.int32:
        raise RuntimeError(f"torch.frexp() expects exponent to have int dtype but got {exponent.dtype}")
    x = self
    if x.dtype not in _FREXP:
        m, e = _frexp_out(x.float(), mantissa=torch.empty(0, device=x.device), exponent=exponent)
        mantissa.resize_(x.shape)
        mantissa.copy_(m)
        return mantissa, exponent
    itype, mbits, mask, bias, k = _FREXP[x.dtype]
    field = (x.view(itype) >> mbits) & mask
    sub = (field == 0) & (x != 0)
    a = torch.where(sub, x * 2.0 ** k, x)
    bits = a.view(itype)
    e = ((bits >> mbits) & mask) - bias - torch.where(sub, k, 0)
    m = ((bits & ~(mask << mbits)) | (bias << mbits)).view(x.dtype)
    nonfinite = ~torch.isfinite(x)
    special = (x == 0) | nonfinite
    from torch._prims_common.wrappers import _maybe_resize_out
    _maybe_resize_out(mantissa, x.shape)  # (warns on resizing a non-empty out, as the CPU)
    _maybe_resize_out(exponent, x.shape)
    mantissa.copy_(torch.where(special, x, m))
    exponent.copy_(torch.where(nonfinite, _frexp_nonfinite_exponent(), torch.where(special, 0, e)))
    return mantissa, exponent


# ---------------------------------------------------------------- histograms

def _outer_edges(points, ranges):
    """(left, right) per dim of points [N, D], as the CPU picks them."""
    D = points.shape[-1]
    if ranges is not None:
        lr = [(float(ranges[2 * d]), float(ranges[2 * d + 1])) for d in range(D)]
    elif points.shape[0] == 0:
        lr = [(0.0, 0.0)] * D
    else:
        lo, hi = points.amin(0).tolist(), points.amax(0).tolist()
        lr = list(zip(lo, hi))
    out = []
    for left, right in lr:
        if left == right:  # widen an empty range, as the CPU does
            left, right = left - 0.5, right + 0.5
        out.append((left, right))
    return out


def _bin_index(x, edges):
    """Bin of each x along edges (None-like: len(edges) - 1 for outside)."""
    nbins = edges.shape[0] - 1
    idx = torch.searchsorted(edges, x.contiguous(), right=True) - 1
    idx = torch.where(x == edges[-1], nbins - 1, idx)
    valid = (x >= edges[0]) & (x <= edges[-1])
    return idx, valid


def _histogramdd_counts(points, edges, weight, density):
    """Histogram [*bins] of points [N, D] with per-dim bin edges."""
    nb = [e.shape[0] - 1 for e in edges]
    flat = torch.zeros((), dtype=torch.int64, device=points.device).expand(points.shape[0])
    valid = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    stride = 1
    for d in reversed(range(len(edges))):
        idx, ok = _bin_index(points[:, d], edges[d])
        flat = flat + idx * stride
        valid = valid & ok
        stride *= nb[d]
    total = stride
    flat = torch.where(valid, flat, total)  # an extra bin for outside points, dropped below
    w = weight.reshape(-1).to(points.dtype) if weight is not None else torch.ones(points.shape[0], dtype=points.dtype,
                                                                                    device=points.device)
    hist = torch.zeros(total + 1, dtype=points.dtype, device=points.device).index_add(0, flat, w)[:total]
    hist = hist.reshape(nb)
    if density:
        widths = None
        for d in range(len(edges)):
            w_d = (edges[d][1:] - edges[d][:-1]).reshape([-1 if i == d else 1 for i in range(len(edges))])
            widths = w_d if widths is None else widths * w_d
        hist = hist / (hist.sum() * widths)
    return hist


def _points(self):
    return self.reshape(-1, self.shape[-1])


def _histogramdd_bin_edges(self, bins, *, range=None, weight=None, density=False):
    if self.dim() > 0 and len(bins) != self.shape[-1]:
        raise RuntimeError("histogramdd: The size of bins must be equal to the innermost dimension of the input.")
    pts = _points(self)
    return [torch.linspace(left, right, b + 1, dtype=self.dtype, device=self.device)
            for (left, right), b in zip(_outer_edges(pts, range), bins)]


def _histogramdd_from_bin_cts(self, bins, *, range=None, weight=None, density=False):
    edges = _histogramdd_bin_edges(self, bins, range=range, weight=weight, density=density)
    return _histogramdd_counts(_points(self), edges, weight, density)


def _histogramdd_from_bin_tensors(self, bins, *, weight=None, density=False):
    return _histogramdd_counts(_points(self), [b.to(self.dtype) for b in bins], weight, density)


def _histogram_bin_ct(self, bins=100, *, range=None, weight=None, density=False):
    pts = self.reshape(-1, 1)
    edges = _histogramdd_bin_edges(pts, [bins], range=range)
    return _histogramdd_counts(pts, edges, weight, density), edges[0]


def _histogram_bins_tensor(self, bins, *, weight=None, density=False):
    pts = self.reshape(-1, 1)
    return _histogramdd_counts(pts, [bins.to(self.dtype)], weight, density), bins.clone()


# ---------------------------------------------------------------- segment_reduce

def _segment_reduce(data, reduce, *, lengths=None, indices=None, offsets=None, axis=0, unsafe=False, initial=None):
    if indices is not None or reduce not in ("sum", "mean", "max", "min", "prod"):
        return _on_cpu("segment_reduce", data, reduce, lengths=lengths, indices=indices, offsets=offsets, axis=axis,
                       unsafe=unsafe, initial=initial)
    axis = axis % data.dim()
    if offsets is None:
        offsets = torch.cat([torch.zeros_like(lengths[..., :1]), torch.cumsum(lengths, -1)], -1)
    offsets = offsets.to(torch.int64)
    S = offsets.shape[-1] - 1
    L = data.shape[axis]
    lead = data.shape[:axis]
    # Segment of each position along axis (S for positions after the last segment).
    pos = torch.arange(L, device=data.device).expand(*offsets.shape[:-1], L).contiguous()
    seg = torch.searchsorted(offsets[..., 1:].contiguous(), pos, right=True)  # [*lead, L]
    seg = seg.reshape(*seg.shape, *([1] * (data.dim() - axis - 1))).expand(data.shape)
    shape = list(data.shape)
    shape[axis] = S + 1
    if reduce == "mean":
        sums = torch.zeros(shape, dtype=data.dtype, device=data.device).scatter_add(axis, seg, data)
        counts = torch.zeros(shape, dtype=data.dtype, device=data.device).scatter_add(axis, seg, torch.ones_like(data))
        if initial is not None:
            sums = sums + initial
        out = torch.where(counts > 0, sums / counts.clamp(min=1), sums if initial is not None else float("nan"))
    else:
        empty = {"sum": 0.0, "prod": 1.0, "max": -math.inf, "min": math.inf}[reduce]
        start = torch.full(shape, empty if initial is None else initial, dtype=data.dtype, device=data.device)
        out = start.scatter_reduce(axis, seg, data, {"sum": "sum", "prod": "prod", "max": "amax", "min": "amin"}[reduce],
                                   include_self=True)
    return out.narrow(axis, 0, S)


# ---------------------------------------------------------------- CTC loss

def _ctc_setup(log_probs, targets, input_lengths, target_lengths, blank):
    unbatched = log_probs.dim() == 2
    lp = log_probs.unsqueeze(1) if unbatched else log_probs  # [T, N, C]
    T, N, C = lp.shape
    dev = lp.device
    il = torch.as_tensor(input_lengths, device="cpu").reshape(-1).tolist()
    tl = torch.as_tensor(target_lengths, device="cpu").reshape(-1).tolist()
    # (the CPU's checks, with its messages)
    torch._check(log_probs.numel() > 0, lambda: "log_probs tensor must not be empty")
    torch._check(0 <= blank < C, lambda: "blank must be in label range")
    torch._check(len(il) == N, lambda: "input_lengths must be of size batch_size")
    torch._check(len(tl) == N, lambda: "target_lengths must be of size batch_size")
    where = "(while checking arguments for ctc_loss_cpu)"
    for n in tl:
        torch._check(n >= 0, lambda: f"Expected target_lengths to have value at least 0, but got value {n} {where}")
    S = max(tl) if tl else 0
    if targets.dim() != 1 and not unbatched:
        torch._check(targets.shape[-1] >= S, lambda: f"Expected tensor to have size at least {S} at dimension 1, but "
                     f"got size {targets.shape[-1]} for argument #2 'targets' {where}")
    for n in il:
        torch._check(n >= 0, lambda: f"Expected input_lengths to have value at least 0, but got value {n} {where}")
        torch._check(n <= T, lambda: f"Expected input_lengths to have value at most {T}, but got value {n} {where}")
    if targets.dim() == 1 and not unbatched or (unbatched and targets.dim() == 1 and N > 1):
        starts = [0]
        for n in tl[:-1]:
            starts.append(starts[-1] + n)
        tg = torch.full((N, max(S, 1)), blank, dtype=torch.int64, device=dev)
        for n in range(N):
            if tl[n]:
                tg[n, :tl[n]] = targets[starts[n]:starts[n] + tl[n]]
    else:
        tg = targets.reshape(N, -1).to(torch.int64)
        if tg.shape[1] < max(S, 1):
            tg = torch.cat([tg, torch.full((N, max(S, 1) - tg.shape[1]), blank, dtype=torch.int64, device=dev)], 1)
    tg = tg[:, :max(S, 1)]
    # Extended labels l' = (blank, t1, blank, t2, ..., blank).
    W = 2 * S + 1
    ext = torch.full((N, W), blank, dtype=torch.int64, device=dev)
    if S:
        ext[:, 1::2] = tg[:, :S]
    # Where the skip transition s-2 -> s is allowed.
    skip = torch.zeros((N, W), dtype=torch.bool, device=dev)
    if W > 2:
        skip[:, 2:] = (ext[:, 2:] != blank) & (ext[:, 2:] != ext[:, :-2])
    width = torch.tensor([2 * n + 1 for n in tl], device=dev).view(N, 1)
    inside = torch.arange(W, device=dev).view(1, W) < width  # s within this sample's labels
    return lp, ext, skip, inside, il, tl, T, N, W, unbatched


def _lse3(a, b, c):
    m = torch.maximum(torch.maximum(a, b), c)
    m = torch.where(m == -math.inf, 0.0, m)
    return torch.log(torch.exp(a - m) + torch.exp(b - m) + torch.exp(c - m)) + m


def _shift(x, k):
    """x [N, W] shifted right by k along W, filled with -inf."""
    pad = torch.full_like(x[:, :k], -math.inf)
    return torch.cat([pad, x[:, :-k]], 1) if x.shape[1] > k else torch.full_like(x, -math.inf)


def _ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False):
    lp, ext, skip, inside, il, tl, T, N, W, unbatched = _ctc_setup(log_probs, targets, input_lengths, target_lengths,
                                                                   blank)
    ninf = -math.inf
    # CPU layout: zeros where never computed; the t = 0 row is -inf beyond the start.
    log_alpha = torch.zeros((N, T, W), dtype=lp.dtype, device=lp.device)
    emit = lp.gather(2, ext.unsqueeze(0).expand(T, N, W))  # [T, N, W]
    la = torch.full((N, W), ninf, dtype=lp.dtype, device=lp.device)
    la[:, 0] = emit[0, :, 0]
    if W > 1:
        la[:, 1] = torch.where(torch.tensor([n > 0 for n in tl], device=lp.device), emit[0, :, 1], ninf)
    if T:
        log_alpha[:, 0] = la
    il_t = torch.tensor(il, device=lp.device).view(N, 1)
    for t in range(1, T):
        nxt = _lse3(la, _shift(la, 1), torch.where(skip, _shift(la, 2), ninf)) + emit[t]
        nxt = torch.where(inside, nxt, ninf)
        la = nxt
        log_alpha[:, t] = torch.where(inside & (t < il_t), la, 0.0)
    # -log p = -logsumexp of the last two positions at each sample's last step.
    last = torch.tensor([max(n - 1, 0) for n in il], device=lp.device)
    final = log_alpha[torch.arange(N, device=lp.device), last]  # [N, W]
    ends = torch.tensor([2 * n for n in tl], device=lp.device)
    a = final.gather(1, ends.view(N, 1)).squeeze(1)
    b = torch.where(ends > 0, final.gather(1, (ends - 1).clamp(min=0).view(N, 1)).squeeze(1), ninf)
    m = torch.maximum(a, b)
    m = torch.where(m == ninf, 0.0, m)
    nll = -(torch.log(torch.exp(a - m) + torch.exp(b - m)) + m)
    nll = torch.where(torch.tensor(il, device=lp.device) == 0, torch.where(ends == 0, 0.0, math.inf), nll)
    # (zero_infinity zeroes infinite losses in ctc_loss and their gradients in the backward, not here)
    if unbatched:
        return nll.squeeze(0), log_alpha.squeeze(0)
    return nll, log_alpha


def _ctc_loss_tensor(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False):
    return _ctc_loss(log_probs, targets, input_lengths.tolist(), target_lengths.tolist(), blank, zero_infinity)


def _ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank,
                       zero_infinity=False):
    lp, ext, skip, inside, il, tl, T, N, W, unbatched = _ctc_setup(log_probs, targets, input_lengths, target_lengths,
                                                                   blank)
    ninf = -math.inf
    la_all = (log_alpha.unsqueeze(0) if unbatched else log_alpha).transpose(0, 1)  # [T, N, W]
    nll = neg_log_likelihood.reshape(N)
    emit = lp.gather(2, ext.unsqueeze(0).expand(T, N, W))
    il_t = torch.tensor(il, device=lp.device).view(N, 1)
    ends = torch.tensor([2 * n for n in tl], device=lp.device).view(N, 1)
    pos = torch.arange(W, device=lp.device).view(1, W)
    # skip into s from s + 2 (backward): allowed where skip[s + 2].
    skip_back = torch.cat([skip[:, 2:], torch.zeros_like(skip[:, :2])], 1) if W > 2 else torch.zeros_like(skip)
    lb = torch.full((N, W), ninf, dtype=lp.dtype, device=lp.device)
    lbs = [None] * T
    for t in reversed(range(T)):
        start = (pos == ends) | ((pos == ends - 1) & (ends > 0))
        init = torch.where(start, emit[t], ninf)
        up1 = torch.cat([lb[:, 1:], torch.full_like(lb[:, :1], ninf)], 1)
        up2 = torch.cat([lb[:, 2:], torch.full_like(lb[:, :2], ninf)], 1) if W > 2 else torch.full_like(lb, ninf)
        rec = _lse3(lb, up1, torch.where(skip_back, up2, ninf)) + emit[t]
        lb = torch.where(t == il_t - 1, init, torch.where(t < il_t - 1, rec, ninf))
        lb = torch.where(inside, lb, ninf)
        lbs[t] = lb
    lbeta = torch.stack(lbs) if T else torch.empty((0, N, W), dtype=lp.dtype, device=lp.device)
    valid_t = (torch.arange(T, device=lp.device).view(T, 1) < il_t.view(1, N)).unsqueeze(2)  # [T, N, 1]
    lab = torch.where(valid_t & inside.unsqueeze(0), la_all + lbeta, ninf)
    m = lab.amax(2, keepdim=True)
    m = torch.where(m == ninf, 0.0, m)
    acc = torch.zeros_like(lp).scatter_add(2, ext.unsqueeze(0).expand(T, N, W), torch.exp(lab - m))
    lcab = torch.log(acc) + m
    g = torch.exp(lp) - torch.exp(lcab + nll.view(1, N, 1) - lp)
    g = g * grad.reshape(1, N, 1)
    g = torch.where(valid_t, g, 0.0)
    if zero_infinity:
        g = torch.where(torch.isinf(nll).view(1, N, 1), 0.0, g)
    return g.squeeze(1) if unbatched else g


def _ctc_loss_backward_tensor(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha,
                              blank, zero_infinity=False):
    return _ctc_loss_backward(grad, log_probs, targets, input_lengths.tolist(), target_lengths.tolist(),
                              neg_log_likelihood, log_alpha, blank, zero_infinity)


# ---------------------------------------------------------------- matrix functions

def _matrix_sqrth(a):
    # Square root of a Hermitian matrix: V sqrt(max(w, 0)) V^H.
    w, v = torch.linalg.eigh(a)
    return (v * w.clamp(min=0).sqrt().unsqueeze(-2).to(v.dtype)) @ v.mH


def _linalg_polar(a):
    if a.dim() < 2:
        raise RuntimeError(f"linalg.polar: A must have at least 2 dimensions, but got a {a.dim()}-D tensor")
    if a.shape[-2] < a.shape[-1]:
        raise RuntimeError("linalg.polar: A must have at least as many rows as columns")
    # A = U H from A = W S V^H: U = W V^H, H = V S V^H.
    w, s, vh = torch.linalg.svd(a, full_matrices=False)
    u = w @ vh
    h = (vh.mH * s.unsqueeze(-2).to(vh.dtype)) @ vh
    return u, h


# ---------------------------------------------------------------- fused RNN cells

# PyTorch's LSTM/GRU cells call these on every device but the CPU (as on
# CUDA): the gates' nonlinearities in one step, and their backward. The
# workspace (saved for backward) holds the activated gates (LSTM: i, f, g, o)
# or r, z, n, h and the hidden part of n's pre-activation (GRU).

def _thnn_fused_lstm_cell(input_gates, hidden_gates, cx, input_bias=None, hidden_bias=None):
    gates = input_gates + hidden_gates
    if input_bias is not None:
        gates = gates + input_bias + hidden_bias
    i, f, g, o = gates.chunk(4, 1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    cy = f * cx + i * g
    hy = o * cy.tanh()
    return hy, cy, torch.cat([i, f, g, o], 1)


def _thnn_fused_lstm_cell_backward_impl(grad_hy, grad_cy, cx, cy, workspace, has_bias):
    i, f, g, o = workspace.chunk(4, 1)
    tcy = cy.tanh()
    gcy = grad_cy if grad_cy is not None else torch.zeros_like(cy)
    go = torch.zeros_like(o)
    if grad_hy is not None:
        gcy = gcy + grad_hy * o * (1 - tcy * tcy)
        go = grad_hy * tcy * o * (1 - o)
    grad_gates = torch.cat([gcy * g * i * (1 - i), gcy * cx * f * (1 - f), gcy * i * (1 - g * g), go], 1)
    grad_bias = grad_gates.sum(0) if has_bias else grad_gates.new_empty(0)
    return grad_gates, gcy * f, grad_bias


def _thnn_fused_gru_cell(input_gates, hidden_gates, hx, input_bias=None, hidden_bias=None):
    if input_bias is not None:
        input_gates = input_gates + input_bias
        hidden_gates = hidden_gates + hidden_bias
    ir, iz, in_ = input_gates.chunk(3, 1)
    hr, hz, hn = hidden_gates.chunk(3, 1)
    r = (ir + hr).sigmoid()
    z = (iz + hz).sigmoid()
    n = (in_ + r * hn).tanh()
    hy = n + z * (hx - n)
    return hy, torch.cat([r, z, n, hx, hn], 1)


def _thnn_fused_gru_cell_backward(grad_hy, workspace, has_bias):
    r, z, n, hx, hn = workspace.chunk(5, 1)
    gn = grad_hy * (1 - z) * (1 - n * n)
    gz = grad_hy * (hx - n) * z * (1 - z)
    gr = gn * hn * r * (1 - r)
    grad_input = torch.cat([gr, gz, gn], 1)
    grad_hidden = torch.cat([gr, gz, gn * r], 1)
    empty = grad_input.new_empty(0)
    return (grad_input, grad_hidden, grad_hy * z, grad_input.sum(0) if has_bias else empty,
            grad_hidden.sum(0) if has_bias else empty)


# ---------------------------------------------------------------- registration

def _kernels():
    avg2, avg2_b = _make_avg_pool(2)
    avg3, avg3_b = _make_avg_pool(3)
    ad2, ad2_b = _make_adaptive_avg(2)
    ad3, ad3_b = _make_adaptive_avg(3)
    fmp2, fmp2_b = _make_fractional_max(2)
    fmp3, fmp3_b = _make_fractional_max(3)
    aa_bilinear, aa_bilinear_b = _make_aa("bilinear")
    aa_bicubic, aa_bicubic_b = _make_aa("bicubic")
    upsample = {
        "upsample_nearest1d_backward": _make_upsample_backward("nearest1d", False),
        "upsample_nearest2d_backward": _make_upsample_backward("nearest2d", False),
        "upsample_nearest3d_backward": _make_upsample_backward("nearest3d", False),
        "_upsample_nearest_exact1d_backward": _make_upsample_backward("nearest_exact1d", False),
        "_upsample_nearest_exact2d_backward": _make_upsample_backward("nearest_exact2d", False),
        "_upsample_nearest_exact3d_backward": _make_upsample_backward("nearest_exact3d", False),
        "upsample_linear1d": _make_upsample_linear(1),
        "upsample_bilinear2d": _make_upsample_linear(2),
        "upsample_trilinear3d": _make_upsample_linear(3),
        "upsample_linear1d_backward": _make_upsample_backward("linear1d", True),
        "upsample_bilinear2d_backward": _make_upsample_backward("bilinear2d", True),
        "upsample_trilinear3d_backward": _make_upsample_backward("trilinear3d", True),
        "upsample_bicubic2d_backward": _make_upsample_backward("bicubic2d", True),
        "upsample_bicubic2d": _upsample_bicubic2d,
    }
    return {
        "avg_pool2d": avg2,
        "avg_pool2d_backward": avg2_b,
        "avg_pool3d": avg3,
        "avg_pool3d_backward": avg3_b,
        "_adaptive_avg_pool2d_backward": ad2_b,
        "_adaptive_avg_pool3d": ad3,
        "_adaptive_avg_pool3d_backward": ad3_b,
        "adaptive_max_pool2d_backward": _make_adaptive_max_backward(2),
        "adaptive_max_pool3d_backward": _make_adaptive_max_backward(3),
        "max_pool2d_with_indices": _make_max_pool(2),
        "max_pool3d_with_indices": _make_max_pool(3),
        "max_pool2d_with_indices_backward": _make_max_pool_backward(2),
        "max_pool3d_with_indices_backward": _make_max_pool_backward(3),
        "adaptive_max_pool2d": _make_adaptive_max(2),
        "adaptive_max_pool3d": _make_adaptive_max(3),
        "grid_sampler_3d": _grid_sampler_forward,
        "grid_sampler_2d_backward": _make_grid_sampler_backward("grid_sampler_2d_backward"),
        "grid_sampler_3d_backward": _make_grid_sampler_backward("grid_sampler_3d_backward"),
        "ldexp.Tensor": _ldexp,
        "bucketize.Scalar": _bucketize_scalar,
        "multi_margin_loss": _multi_margin_loss,
        "multilabel_margin_loss_forward": _multilabel_margin_loss_forward,
        "_cdist_forward": _cdist_forward,
        "_cdist_backward": _cdist_backward,
        "_pdist_forward": _pdist_forward,
        "repeat_interleave.Tensor": _repeat_interleave_tensor,
        "triu_indices": functools.partial(_tri_indices, True),
        "tril_indices": functools.partial(_tri_indices, False),
        "_pdist_backward": _pdist_backward,
        "fractional_max_pool2d": fmp2,
        "fractional_max_pool2d_backward": fmp2_b,
        "fractional_max_pool3d": fmp3,
        "fractional_max_pool3d_backward": fmp3_b,
        "rrelu_with_noise": _rrelu_with_noise,
        "rrelu_with_noise_": _rrelu_with_noise_,
        "addbmm_": lambda x, b1, b2, beta=1, alpha=1: x.copy_(torch.addbmm(x, b1, b2, beta=beta, alpha=alpha)),
        "_upsample_bilinear2d_aa": aa_bilinear,
        "_upsample_bilinear2d_aa_backward": aa_bilinear_b,
        "_upsample_bicubic2d_aa": aa_bicubic,
        "_upsample_bicubic2d_aa_backward": aa_bicubic_b,
        "_embedding_bag": _embedding_bag,
        "_embedding_bag_forward_only": _embedding_bag_forward_only,
        "_embedding_bag_dense_backward": _embedding_bag_dense_backward,
        "_embedding_bag_backward": _embedding_bag_backward,
        "linalg_matrix_exp": _matrix_exp,
        "_embedding_bag_per_sample_weights_backward": _embedding_bag_per_sample_weights_backward,
        "hash_tensor": _hash_tensor,
        "hash_tensor.out": _hash_tensor_out,
        "frexp.Tensor_out": _frexp_out,
        "histogram.bin_ct": _histogram_bin_ct,
        "histogram.bins_tensor": _histogram_bins_tensor,
        "_histogramdd_bin_edges": _histogramdd_bin_edges,
        "_histogramdd_from_bin_cts": _histogramdd_from_bin_cts,
        "_histogramdd_from_bin_tensors": _histogramdd_from_bin_tensors,
        "segment_reduce": _segment_reduce,
        "_ctc_loss": _ctc_loss,
        "_ctc_loss.Tensor": _ctc_loss_tensor,
        "_ctc_loss_backward": _ctc_loss_backward,
        "_ctc_loss_backward.Tensor": _ctc_loss_backward_tensor,
        "linalg_matrix_sqrth": _matrix_sqrth,
        "linalg_polar": _linalg_polar,
        "_thnn_fused_lstm_cell": _thnn_fused_lstm_cell,
        "_thnn_fused_lstm_cell_backward_impl": _thnn_fused_lstm_cell_backward_impl,
        "_thnn_fused_gru_cell": _thnn_fused_gru_cell,
        "_thnn_fused_gru_cell_backward": _thnn_fused_gru_cell_backward,
        "softshrink_backward": _shrink_backward("softshrink_backward"),
        "hardshrink_backward": _shrink_backward("hardshrink_backward"),
        "replication_pad1d_backward": _replication_pad_backward(1),
        "replication_pad2d_backward": _replication_pad_backward(2),
        "replication_pad3d_backward": _replication_pad_backward(3),
        "multi_margin_loss_backward": _multi_margin_loss_backward,
        "multilabel_margin_loss_backward": _multilabel_margin_loss_backward,
        **upsample,
    }


def _out_overload(op):
    """The out= overload of functional OpOverload `op` (same inputs, keyword-only
    outputs), or None."""
    inputs = [a.name for a in op._schema.arguments]
    packet = getattr(torch.ops.aten, op._schema.name.split("::", 1)[1])
    for name in packet.overloads():
        other = getattr(packet, name)
        args = other._schema.arguments
        outs = [a for a in args if a.kwarg_only and a.alias_info and a.alias_info.is_write]
        if outs and [a.name for a in args if a not in outs] == inputs and len(outs) == len(op._schema.returns):
            return other, [a.name for a in outs]
    return None


def _make_out_kernel(functional, out_names, exact_dtype):
    """An out= kernel from a functional one: its results resized into and
    copied to the out= tensors (PyTorch's rules for both, as its own
    decompositions use; linalg ops want out='s dtype exact)."""
    from torch._prims_common.wrappers import _maybe_resize_out, _safe_copy_out

    def out_kernel(*args, **kwargs):
        outs = [kwargs.pop(name) for name in out_names]
        results = functional(*args, **kwargs)
        results = results if isinstance(results, (tuple, list)) else (results,)
        for out, result in zip(outs, results):
            if out.shape != result.shape and any(a is out for a in args):  # in place: no resizing
                raise RuntimeError(f"output with shape {list(out.shape)} doesn't match the broadcast shape "
                                   f"{list(result.shape)}")
            _maybe_resize_out(out, result.shape)
            _safe_copy_out(copy_from=result, copy_to=out, exact_dtype=exact_dtype)
        return outs[0] if len(outs) == 1 else tuple(outs)

    return out_kernel


def register():
    """Registers the kernels (functional overloads, and their out= overloads
    when nothing else implements those); returns their names."""
    global _lib
    if _lib is not None:
        return []
    _lib = torch.library.Library("aten", "IMPL")
    names = []
    for name, fn in _kernels().items():
        if torch._C._dispatch_has_kernel_for_dispatch_key("aten::" + name, "PrivateUse1"):
            continue  # a C++ or decomposed kernel exists
        base, _, overload = name.partition(".")
        op = getattr(getattr(torch.ops.aten, base), overload or "default")
        checked = _checks.with_cpu_checks(op, _dtypes.with_cpu_dtypes(base, fn))
        _lib.impl(name, _layout.with_layout(op, checked), "PrivateUse1")
        names.append("aten::" + name)
        found = _out_overload(op) if not op._schema.is_mutable else None
        if found is not None:
            out_op, out_names = found
            if not torch._C._dispatch_has_kernel_for_dispatch_key(out_op.name(), "PrivateUse1"):
                out_fn = _checks.with_cpu_checks(out_op, _make_out_kernel(checked, out_names, base.startswith("linalg_")))
                _lib.impl(out_op._schema.name.split("::", 1)[1] + "." + out_op._overloadname,
                          _layout.with_layout(out_op, out_fn), "PrivateUse1")
                names.append(out_op.name())
    return names
