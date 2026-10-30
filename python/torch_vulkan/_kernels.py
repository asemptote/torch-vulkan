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

import math

import torch
import torch.nn.functional as F
from torch._decomp import decomposition_table

from . import _decomp

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
            fn = decomposition_table[aten.upsample_linear1d.default]
            m = _interp_matrix(lambda x, o, ac, s: fn(x, [o], ac, s), n_in, n_out, like, align_corners, scale)
        mats.append(m)
    return mats


def _make_upsample_backward(name, has_align):
    def backward(grad_output, output_size, input_size, *rest):
        if has_align:
            align_corners, scales = rest[0], list(rest[1:])
        else:
            align_corners, scales = None, list(rest)
        if not any(s is not None for s in scales):
            scales = None
        mats = _upsample_matrices(name, input_size, output_size, align_corners, scales, grad_output)
        return _apply_dims(grad_output, [m.t() for m in mats]).contiguous()

    return backward


# ---------------------------------------------------------------- average pooling

def _pool_out_size(n, k, s, p, ceil_mode):
    out = (n + 2 * p - k + (s - 1 if ceil_mode else 0)) // s + 1
    if ceil_mode and (out - 1) * s >= n + p:
        out -= 1
    return out


def _window_counts(n, k, s, p, out, count_include_pad, device):
    starts = torch.arange(out, dtype=torch.float64) * s - p
    ends = torch.clamp(starts + k, max=n + p)
    if count_include_pad:
        counts = ends - starts
    else:
        counts = torch.clamp(ends, max=n) - torch.clamp(starts, min=0)
    return counts.to(device)


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


def _divisor(spatial, ks, st, pd, out, count_include_pad, divisor_override, device, dtype):
    if divisor_override:
        return torch.tensor(float(divisor_override), device=device, dtype=dtype)
    div = None
    for d in range(len(ks)):
        c = _window_counts(spatial[d], ks[d], st[d], pd[d], out[d], count_include_pad, device).to(dtype)
        shape = [1] * len(ks)
        shape[d] = out[d]
        c = c.view(shape)
        div = c if div is None else div * c
    return div


def _make_avg_pool(k):
    conv = [F.conv1d, F.conv2d, F.conv3d][k - 1]

    def forward(x, kernel_size, stride=(), padding=0, ceil_mode=False, count_include_pad=True,
                divisor_override=None):
        if not x.is_floating_point():  # integer pooling: the convolution below is float-only
            return _on_cpu(f"avg_pool{k}d", x, kernel_size, stride, padding, ceil_mode, count_include_pad,
                           divisor_override)
        padding = [padding] if isinstance(padding, int) else padding
        ks, st, pd, out, pad = _avg_pool_setup(x, kernel_size, stride, padding, ceil_mode, k)
        batched = x.dim() == k + 2
        xb = x if batched else x.unsqueeze(0)
        c = xb.shape[1]
        xp = F.pad(xb, pad)
        w = torch.ones((c, 1, *ks), device=x.device, dtype=x.dtype)
        sums = conv(xp, w, None, st, 0, 1, c)
        y = sums / _divisor(x.shape[-k:], ks, st, pd, out, count_include_pad, divisor_override, x.device, x.dtype)
        return y if batched else y.squeeze(0)

    def backward(grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
        ks, st, pd, out, pad = _avg_pool_setup(x, kernel_size, stride, padding, ceil_mode, k)
        batched = x.dim() == k + 2
        xb = x if batched else x.unsqueeze(0)
        gb = grad_output if batched else grad_output.unsqueeze(0)
        c = xb.shape[1]
        g = gb / _divisor(x.shape[-k:], ks, st, pd, out, count_include_pad, divisor_override, x.device, x.dtype)
        xp = F.pad(xb, pad)
        w = torch.ones((c, 1, *ks), device=x.device, dtype=x.dtype)
        gxp = aten.convolution_backward(g, xp, w, None, st, [0] * k, [1] * k, False, [0] * k, c,
                                        [True, False, False])[0]
        # Drop the padding again.
        index = [slice(None), slice(None)]
        for d in range(k):
            index.append(slice(pd[d], pd[d] + x.shape[-k + d]))
        gx = gxp[tuple(index)].contiguous()
        return gx if batched else gx.squeeze(0)

    return forward, backward


# ---------------------------------------------------------------- adaptive pooling

def _adaptive_matrix(n_in, n_out, device):
    m = torch.zeros(n_out, n_in, dtype=torch.float64)
    for o in range(n_out):
        start, end = (o * n_in) // n_out, -((-(o + 1) * n_in) // n_out)
        m[o, start:end] = 1.0 / (end - start)
    return m.to(device)


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
        g = grad_output.reshape(*grad_output.shape[:head], -1)
        i = indices.reshape(*indices.shape[:head], -1)
        gx = torch.zeros((*x.shape[:head], math.prod(x.shape[head:])), device=x.device, dtype=x.dtype)
        return gx.scatter_add(-1, i, g).reshape(x.shape)

    return backward


def _make_max_pool_backward(k):
    def backward(grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices):
        head = x.dim() - k
        g = grad_output.reshape(*grad_output.shape[:head], -1)
        i = indices.reshape(*indices.shape[:head], -1)
        gx = torch.zeros((*x.shape[:head], math.prod(x.shape[head:])), device=x.device, dtype=x.dtype)
        return gx.scatter_add(-1, i, g).reshape(x.shape)

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

def _unnormalize(coord, size, align_corners):
    if align_corners:
        return (coord + 1) / 2 * (size - 1), (size - 1) / 2
    return ((coord + 1) * size - 1) / 2, size / 2


def _reflect(coord, twice_low, twice_high):
    """PyTorch's reflect_coordinates and its derivative."""
    if twice_low == twice_high:
        return torch.zeros_like(coord), torch.zeros_like(coord)
    low, span = twice_low / 2, (twice_high - twice_low) / 2
    sign = torch.where(coord < low, -1.0, 1.0).to(coord.dtype)
    c = (coord - low).abs()
    extra = torch.fmod(c, span)
    flips = torch.floor(c / span)
    even = torch.remainder(flips, 2) == 0
    out = torch.where(even, extra + low, span - extra + low)
    return out, torch.where(even, sign, -sign)


def _source_coord(coord, size, padding_mode, align_corners):
    """Grid coordinate -> input coordinate, and d(input coord)/d(grid coord)."""
    c, mult = _unnormalize(coord, size, align_corners)
    grad = torch.full_like(c, mult)
    if padding_mode == 2:  # reflection
        if align_corners:
            c, g = _reflect(c, 0, 2 * (size - 1))
        else:
            c, g = _reflect(c, -1, 2 * size - 1)
        grad = grad * g
    if padding_mode in (1, 2):  # border (and reflection): clip into [0, size - 1]
        inside = (c > 0) & (c < size - 1)
        grad = grad * inside.to(c.dtype)
        c = c.clamp(0, size - 1)
    return c, grad


def _grid_sample(input, grid, mode, padding_mode, align_corners, grad_output=None, mask=(True, True)):
    """Forward (grad_output None) or backward of grid_sampler_{2,3}d for
    bilinear (mode 0) and nearest (mode 1) interpolation."""
    k = grid.shape[-1]
    n, c = input.shape[:2]
    spatial = input.shape[2:]  # (D,) H, W
    out_spatial = grid.shape[1:-1]
    # grid[..., 0] indexes W (the last dim), grid[..., 1] H, grid[..., 2] D.
    coords, dcoord = [], []
    for d in range(k):
        cd, gd = _source_coord(grid[..., d], spatial[k - 1 - d], padding_mode, align_corners)
        coords.append(cd)
        dcoord.append(gd)
    flat_in = input.reshape(n, c, -1)
    strides = [math.prod(spatial[k - d:]) for d in range(k)]  # element stride of coordinate d's dim
    count = math.prod(out_spatial)

    def corner(idx):
        """(flat index [n, 1, P], in-bounds mask [n, P]) for integer coords idx."""
        valid, flat = None, None
        for d in range(k):
            size = spatial[k - 1 - d]
            v = (idx[d] >= 0) & (idx[d] < size)
            valid = v if valid is None else valid & v
            term = idx[d].clamp(0, size - 1) * strides[d]
            flat = term if flat is None else flat + term
        return flat.reshape(n, 1, -1), valid.reshape(n, -1)

    if mode == 1:  # nearest (round half to even, as nearbyint)
        flat, valid = corner([torch.round(cd).long() for cd in coords])
        index = flat.expand(n, c, count)
        if grad_output is None:
            vals = torch.gather(flat_in, 2, index) * valid.unsqueeze(1).to(input.dtype)
            return vals.reshape(n, c, *out_spatial)
        g = grad_output.reshape(n, c, -1) * valid.unsqueeze(1).to(input.dtype)
        gi = torch.zeros_like(flat_in).scatter_add(2, index, g).reshape(input.shape)
        return (gi if mask[0] else None), (torch.zeros_like(grid) if mask[1] else None)

    lo = [torch.floor(cd) for cd in coords]
    frac = [(cd - l).reshape(n, count) for cd, l in zip(coords, lo)]
    lo = [l.long() for l in lo]
    out = None
    backward = grad_output is not None
    gi = torch.zeros_like(flat_in) if backward and mask[0] else None
    g = grad_output.reshape(n, c, -1) if backward else None
    gcoord = [torch.zeros(n, count, device=input.device, dtype=input.dtype) for _ in range(k)]
    for bits in range(1 << k):
        b = [(bits >> d) & 1 for d in range(k)]
        flat, valid = corner([lo[d] + b[d] for d in range(k)])
        w_parts = [frac[d] if b[d] else 1 - frac[d] for d in range(k)]
        validf = valid.to(input.dtype)
        weight = math.prod(w_parts) * validf
        index = flat.expand(n, c, count)
        if not backward:
            term = torch.gather(flat_in, 2, index) * weight.unsqueeze(1)
            out = term if out is None else out + term
            continue
        if gi is not None:
            gi = gi.scatter_add(2, index, g * weight.unsqueeze(1))
        if mask[1]:
            dot = (torch.gather(flat_in, 2, index) * g).sum(1) * validf  # [n, count]
            for d in range(k):
                others = math.prod(w_parts[e] for e in range(k) if e != d) if k > 1 else 1.0
                gcoord[d] = gcoord[d] + dot * others * (1.0 if b[d] else -1.0)
    if not backward:
        return out.reshape(n, c, *out_spatial)
    gg = None
    if mask[1]:
        gg = torch.stack([gcoord[d].reshape(n, *out_spatial) * dcoord[d] for d in range(k)], -1)
    return (gi.reshape(input.shape) if gi is not None else None), gg


def _grid_sampler_forward(input, grid, mode, padding_mode, align_corners):
    if mode not in (0, 1):  # bicubic
        return _on_cpu("grid_sampler_3d", input, grid, mode, padding_mode, align_corners)
    return _grid_sample(input, grid, mode, padding_mode, align_corners)


def _make_grid_sampler_backward(name):
    def backward(grad_output, input, grid, mode, padding_mode, align_corners, output_mask):
        if mode not in (0, 1):  # bicubic
            return _on_cpu(name, grad_output, input, grid, mode, padding_mode, align_corners, output_mask)
        return _grid_sample(input, grid, mode, padding_mode, align_corners, grad_output, output_mask)

    return backward


def _on_cpu(name, *args):
    op = getattr(torch.ops.aten, name).default
    return _decomp._cpu_fallback(op, op.name(), args, {})


# ---------------------------------------------------------------- distances

def _pairwise_diff(x1, x2):
    return x1.unsqueeze(-2) - x2.unsqueeze(-3)  # [..., P, R, M]


def _cdist_forward(x1, x2, p, compute_mode=None):
    if x1.shape[-1] == 0:
        return torch.zeros((*x1.shape[:-1], x2.shape[-2]), device=x1.device, dtype=x1.dtype)
    diff = _pairwise_diff(x1, x2)
    if p == 0:
        return (diff != 0).sum(-1).to(x1.dtype)
    if p == math.inf:
        return diff.abs().amax(-1)
    if p == 1:
        return diff.abs().sum(-1)
    if p == 2:
        return diff.square().sum(-1).sqrt()
    return diff.abs().pow(p).sum(-1).pow(1.0 / p)


def _cdist_backward(grad, x1, x2, p, dist):
    """Gradient w.r.t. x1 of cdist(x1, x2)."""
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
    else:
        dd = torch.where(d == 0, torch.zeros_like(diff),
                         torch.sign(diff) * diff.abs().pow(p - 1) / d.pow(p - 1))
    return (grad.unsqueeze(-1) * dd).sum(-2)


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
            u = random_samples[..., sample_of[d]].to(dtype)  # [n, c]
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
        g = grad_output.reshape(*grad_output.shape[:head], -1)
        i = indices.reshape(*indices.shape[:head], -1)
        gx = torch.zeros((*x.shape[:head], math.prod(x.shape[head:])), device=x.device, dtype=x.dtype)
        return gx.scatter_add(-1, i, g).reshape(x.shape)

    return forward, backward


# ---------------------------------------------------------------- rrelu

def _rrelu_with_noise(x, noise, lower=1.0 / 8, upper=1.0 / 3, training=False, generator=None):
    if not training:
        return F.leaky_relu(x, (lower + upper) / 2)
    slopes = torch.empty_like(x).uniform_(lower, upper, generator=generator)
    negative = x < 0
    noise.copy_(torch.where(negative, slopes, torch.ones_like(x)))
    return torch.where(negative, x * slopes, x)


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


def _make_aa(kind):
    def matrices(input_size, output_size, align_corners, scales_h, scales_w, like):
        return [_aa_matrix(kind, input_size[2 + d], output_size[d], align_corners, (scales_h, scales_w)[d], like)
                for d in range(2)]

    def forward(x, output_size, align_corners, scales_h=None, scales_w=None):
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
        scale = validf if per_sample_weights is None else validf * per_sample_weights
        out = torch.zeros((nb, dim), device=device, dtype=dtype).index_add(0, offset2bag, rows * scale.unsqueeze(1))
        if mode == 1:
            out = out / bag_size.clamp(min=1).unsqueeze(1).to(dtype)
            max_indices = bag_size.clone()
        else:
            max_indices = torch.zeros(nb, dtype=torch.long, device=device)
            if forward_only:
                offset2bag = offset2bag.new_zeros(0)
                bag_size = torch.zeros(nb, dtype=torch.long, device=device)
        return out, offset2bag, bag_size, max_indices
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
    return out, offset2bag, bag_size, max_indices


def _embedding_bag_forward_only(*args):
    return _embedding_bag(*args, forward_only=True)


def _embedding_bag_dense_backward(grad, indices, offset2bag, bag_size, maximum_indices, num_weights,
                                  scale_grad_by_freq, mode, per_sample_weights, padding_idx=-1):
    dim = grad.shape[1]
    gw = torch.zeros((num_weights, dim), device=grad.device, dtype=grad.dtype)
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
        "upsample_linear1d_backward": _make_upsample_backward("linear1d", True),
        "upsample_bilinear2d_backward": _make_upsample_backward("bilinear2d", True),
        "upsample_trilinear3d_backward": _make_upsample_backward("trilinear3d", True),
        "upsample_bicubic2d_backward": _make_upsample_backward("bicubic2d", True),
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
        "max_pool3d_with_indices_backward": _make_max_pool_backward(3),
        "adaptive_max_pool2d": _make_adaptive_max(2),
        "adaptive_max_pool3d": _make_adaptive_max(3),
        "grid_sampler_3d": _grid_sampler_forward,
        "grid_sampler_2d_backward": _make_grid_sampler_backward("grid_sampler_2d_backward"),
        "grid_sampler_3d_backward": _make_grid_sampler_backward("grid_sampler_3d_backward"),
        "_cdist_forward": _cdist_forward,
        "_cdist_backward": _cdist_backward,
        "_pdist_forward": _pdist_forward,
        "_pdist_backward": _pdist_backward,
        "fractional_max_pool2d": fmp2,
        "fractional_max_pool2d_backward": fmp2_b,
        "fractional_max_pool3d": fmp3,
        "fractional_max_pool3d_backward": fmp3_b,
        "rrelu_with_noise": _rrelu_with_noise,
        "_upsample_bilinear2d_aa": aa_bilinear,
        "_upsample_bilinear2d_aa_backward": aa_bilinear_b,
        "_upsample_bicubic2d_aa": aa_bicubic,
        "_upsample_bicubic2d_aa_backward": aa_bicubic_b,
        "_embedding_bag": _embedding_bag,
        "_embedding_bag_forward_only": _embedding_bag_forward_only,
        "_embedding_bag_dense_backward": _embedding_bag_dense_backward,
        "_embedding_bag_backward": _embedding_bag_backward,
        "_embedding_bag_per_sample_weights_backward": _embedding_bag_per_sample_weights_backward,
        **upsample,
    }


def register():
    """Registers the kernels (functional overloads only); returns their names."""
    global _lib
    if _lib is not None:
        return []
    _lib = torch.library.Library("aten", "IMPL")
    names = []
    for name, fn in _kernels().items():
        if torch._C._dispatch_has_kernel_for_dispatch_key("aten::" + name, "PrivateUse1"):
            continue  # a C++ or decomposed kernel exists
        _lib.impl(name, fn, "PrivateUse1")
        names.append("aten::" + name)
    return names
