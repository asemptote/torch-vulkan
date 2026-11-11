"""Ops whose PyTorch implementations make float64 temporaries on the input's
device, for float32 / complex64 inputs: on GPUs without float64 arithmetic
(where a vk float64 tensor is a TypeError, as on MPS) they would fail. These
versions keep every temporary at the input's precision, with the same
results; on GPUs with float64 they compute as PyTorch's implementations.

quantile and nanquantile compute their ranks q (n - 1) in float64 on the CPU;
for a float32 q and n - 1 < 2^31 that product is exact in 55-bit integer
arithmetic, so the integer part and the (once rounded) fraction are the
CPU's. pinv and matrix_rank compare singular values with a float64
tolerance; without float64 the tolerance is float32 (a different decision
only for a singular value within an ulp of it).

Registered on the AutogradPrivateUse1 key (these are composite ops, so
autograd differentiates the ops they call), except linalg_pinv.atol_rtol_tensor:
it has its own derivative (pinv_backward), so its kernel is a backend
(PrivateUse1) one, and pinv's other overloads call it with their tolerances
as tensors of the right dtype. (Differentiating through the SVD instead hits
svd_backward's check, an allclose, which vmap has no batching rule for, and
the SVD's forward-mode derivative, which fails on a view.)"""

import torch

from . import _kernels

_lib = None
aten = torch.ops.aten


def _has_float64():
    return _kernels._native_has_float64()


# ---------------------------------------------------------------- quantile

_MODES = ("linear", "lower", "higher", "midpoint", "nearest")


def _pow2_neg(s):
    """2^-s (int64 tensor s in [0, 149]) as float32, exactly."""
    normal = (127 - s).clamp(min=1) << 23
    sub = 1 << (149 - s).clamp(0, 22)
    return torch.where(s <= 126, normal, sub).to(torch.int32).view(torch.float32)


def _exact_ranks(q, last):
    """q * last (q: float32 in [0, 1], last: int64 >= 0) as (floor: int64,
    fraction: float32 rounded once), without float64."""
    bits = q.view(torch.int32).to(torch.int64) & 0x7FFFFFFF
    e = bits >> 23
    mant = torch.where(e == 0, bits & 0x7FFFFF, (bits & 0x7FFFFF) | 0x800000)
    s = torch.where(e == 0, 149, 150 - e)  # q = mant 2^-s, s >= 23
    prod = mant * last
    big = s >= 56  # (prod < 2^55: the integer part is 0)
    sc = s.clamp(max=55)
    below = torch.where(big, 0, prod >> sc)
    rem = prod - (below << sc)
    frac = rem.to(torch.float32) * _pow2_neg(s)
    return below, frac


def _quantile_impl(nan, self, q, dim, keepdim, interpolation):
    name = "nanquantile" if nan else "quantile"
    scalar_q = not isinstance(q, torch.Tensor)
    if self.numel() == 0:
        raise RuntimeError(f"{name}() input tensor must be non-empty")
    if self.dtype not in (torch.float32, torch.float64):
        raise RuntimeError(f"{name}() input tensor must be either float or double dtype")
    if interpolation not in _MODES:
        raise RuntimeError(f"{name}() interpolation must be one of linear, lower, higher, midpoint or nearest. "
                           f"Got {interpolation}")
    if scalar_q:
        if not 0 <= q <= 1:
            raise RuntimeError(f"{name}() q must be in the range [0, 1] but got {q}")
        qt = torch.full((), q, dtype=self.dtype, device=self.device)
    else:
        if q.dim() > 1:
            raise RuntimeError(f"{name}() q must be a scalar or 1D tensor")
        if q.dtype != self.dtype:
            raise RuntimeError(f"{name}() q tensor must be same dtype as the input tensor")
        if q.device != self.device:
            raise RuntimeError(f"{name}() q tensor must be on the same device as the input tensor")
        if q.numel() and not bool(((q >= 0) & (q <= 1)).all()):
            raise RuntimeError(f"{name}() q values must be in the range [0, 1]")
        qt = q
    ndim = self.dim()
    if dim is None:
        x, d = self.reshape(-1), 0
    else:
        d = dim % max(ndim, 1)
        x = self if ndim else self.reshape(1)
    srt = x.movedim(d, -1).sort(-1).values  # [..., n], NaNs last
    n = srt.shape[-1]
    qf = qt.reshape(-1)  # [Q]
    nans = srt.isnan()
    if nan:
        last = (~nans).sum(-1, keepdim=True) - 1
        empty = last < 0  # all NaN: rank 0 (a NaN)
        below, frac = _exact_ranks(qf, last.clamp(min=0))
        below = torch.where(empty, 0, below)
        frac = torch.where(empty, 0.0, frac)
    else:
        last = torch.full(srt.shape[:-1] + (1,), n - 1, dtype=torch.int64, device=srt.device)
        below, frac = _exact_ranks(qf, last)
        has_nan = nans.any(-1, keepdim=True)  # rank n - 1 (a NaN)
        below = torch.where(has_nan, n - 1, below)
        frac = torch.where(has_nan, 0.0, frac)
    up = (frac > 0).to(torch.int64)
    if interpolation == "lower":
        r = srt.gather(-1, below)
    elif interpolation == "higher":
        r = srt.gather(-1, below + up)
    elif interpolation == "nearest":  # (round half to even, as the CPU's round)
        half_up = (frac > 0.5) | ((frac == 0.5) & (below % 2 == 1))
        r = srt.gather(-1, below + half_up.to(torch.int64))
    else:
        lo, hi = srt.gather(-1, below), srt.gather(-1, below + up)
        w = frac if interpolation == "linear" else torch.full_like(frac, 0.5)
        r = lo.lerp(hi, w)
    r = r.movedim(-1, 0)  # [Q, ...]
    if dim is None:
        r = r.reshape((r.shape[0],) + ((1,) * ndim if keepdim else ()))
    elif keepdim and ndim:
        r = r.unsqueeze(d + 1)
    if scalar_q or qt.dim() == 0:
        r = r[0]
    return r


def _make_quantile(nan, original):
    def kernel(self, q, dim=None, keepdim=False, *, interpolation="linear"):
        if _has_float64():
            return original.decompose(self, q, dim, keepdim, interpolation=interpolation)
        return _quantile_impl(nan, self, q, dim, keepdim, interpolation)

    return kernel


# ---------------------------------------------------------------- pinv / matrix_rank

_COMPLEX_NAMES = {torch.complex32: "ComplexHalf", torch.complex64: "ComplexFloat", torch.complex128: "ComplexDouble"}


def _check_real(t, name, what):
    """PyTorch's check (and message) that a tolerance tensor is not complex."""
    if isinstance(t, torch.Tensor) and t.is_complex():
        raise RuntimeError(f"{name}: {what} tensor of complex type is not supported. Got {_COMPLEX_NAMES[t.dtype]}")


def _tolerance(input, atol, rtol, name):
    """(atol, rtol) as tensors with PyTorch's defaults: float64 as PyTorch's
    where the GPU has it, else input's real dtype."""
    real = input.real.dtype if input.is_complex() else input.dtype
    eps = torch.finfo(real).eps
    if _has_float64():
        real = torch.float64
    for t, what in ((atol, "atol"), (rtol, "rtol")):
        _check_real(t, name, what)
    as_t = lambda v: v.to(real) if isinstance(v, torch.Tensor) else torch.full((), v, dtype=real, device=input.device)
    a = as_t(0.0 if atol is None else atol)
    if rtol is not None:
        r = as_t(rtol)
    else:
        default = eps * max(input.shape[-1], input.shape[-2])
        r = as_t(default) if atol is None else torch.where(a > 0, 0.0, default).to(real)
    return a, r


def _pinv_impl(input, atol, rtol, hermitian):
    if input.dim() < 2 or not (input.is_floating_point() or input.is_complex()):
        # (PyTorch's checks, with its messages, on a meta tensor of the shape)
        torch.linalg.pinv(torch.empty(input.shape, dtype=input.dtype, device="meta"), hermitian=hermitian)
    if input.numel() == 0:
        shape = list(input.shape)
        shape[-1], shape[-2] = shape[-2], shape[-1]
        return input.new_zeros(shape) + input.sum() * 0  # (zeros that keep the autograd graph, as PyTorch's)
    a, r = _tolerance(input, atol, rtol, "torch.linalg.pinv")
    if not hermitian:
        U, S, Vh = torch.linalg.svd(input, full_matrices=False)
        tol = torch.maximum(a.unsqueeze(-1), r.unsqueeze(-1) * S.narrow(-1, 0, 1))
        S_inv = torch.where(S > tol, S.reciprocal(), 0.0).to(input.dtype)
        return torch.matmul(Vh.mH * S_inv.unsqueeze(-2), U.mH)
    S, U = torch.linalg.eigh(input)
    S_abs = S.abs()
    tol = torch.maximum(a.unsqueeze(-1), r.unsqueeze(-1) * S_abs.amax(-1, keepdim=True))
    S_inv = torch.where(S_abs > tol, S.reciprocal(), 0.0).to(input.dtype)
    return torch.matmul(U * S_inv.unsqueeze(-2), U.mH)


def _matrix_rank_impl(input, atol, rtol, hermitian):
    if input.dim() < 2:
        raise RuntimeError(f"linalg.matrix_rank: The input tensor input must have at least 2 dimensions.")
    if input.numel() == 0:
        return torch.zeros(input.shape[:-2], dtype=torch.int64, device=input.device)
    a, r = _tolerance(input, atol, rtol, "torch.linalg.matrix_rank")
    if not hermitian:
        S = torch.linalg.svdvals(input)
        max_S = S.narrow(-1, 0, 1)
    else:
        S = torch.linalg.eigvalsh(input).abs()
        max_S = S.amax(-1, keepdim=True)
    tol = torch.maximum(a.unsqueeze(-1), r.unsqueeze(-1) * max_S)
    return (S > tol).sum(-1)


def _overloads(fn_name, impl, original_packet):
    """Kernels for the overloads of linalg_pinv / linalg_matrix_rank."""
    def pick(overload, convert):
        def kernel(*args, **kwargs):
            input, atol, rtol, hermitian = convert(*args, **kwargs)
            return impl(input, atol, rtol, hermitian)

        return kernel

    if fn_name == "linalg_pinv":
        def via_tensor(input, atol, rtol, hermitian):
            # (pinv.atol_rtol_tensor, whose derivative autograd knows, with
            # numbers as tensors of the tolerance's dtype)
            real = input.real.dtype if input.is_complex() else input.dtype
            real = torch.float64 if _has_float64() else real
            as_t = lambda v: v if v is None or isinstance(v, torch.Tensor) else torch.full(
                (), v, dtype=real, device=input.device)
            return aten.linalg_pinv.atol_rtol_tensor(input, atol=as_t(atol), rtol=as_t(rtol), hermitian=hermitian)

        def pick_composite(convert):
            return lambda *args, **kwargs: via_tensor(*convert(*args, **kwargs))

        return {
            "linalg_pinv.atol_rtol_float": pick_composite(lambda x, *, atol=None, rtol=None, hermitian=False:
                                                          (x, atol, rtol, hermitian)),
            "linalg_pinv": pick_composite(lambda x, rcond, hermitian=False: (x, 0.0, rcond, hermitian)),
            "linalg_pinv.rcond_tensor": pick_composite(lambda x, rcond, hermitian=False: (
                _check_real(rcond, "torch.linalg.pinv", "rcond"), (x, 0.0, rcond, hermitian))[1]),
        }
    return {
        "linalg_matrix_rank.atol_rtol_tensor": pick("atol_rtol_tensor", lambda x, *, atol=None, rtol=None,
                                                    hermitian=False: (x, atol, rtol, hermitian)),
        "linalg_matrix_rank.atol_rtol_float": pick("atol_rtol_float", lambda x, *, atol=None, rtol=None,
                                                   hermitian=False: (x, atol, rtol, hermitian)),
        "linalg_matrix_rank": pick("default", lambda x, tol, hermitian=False: (x, tol, 0.0, hermitian)),
        "linalg_matrix_rank.tol_tensor": pick("tol_tensor", lambda x, tol, hermitian=False: (x, tol, 0.0, hermitian)),
    }


# ---------------------------------------------------------------- zeta with a Scalar

def _scalar_operand(value, like):
    """A Python number as a 0-dim tensor on like's device, of the dtype the
    CPU computes in (its wrapped number promotes no further than this)."""
    dtype = like.dtype if like.is_floating_point() else torch.get_default_dtype()
    return torch.full((), value, dtype=dtype, device=like.device)


def _zeta_other_scalar(self, other):
    return torch.special.zeta(self, _scalar_operand(other, self))


def _zeta_self_scalar(self, other):
    return torch.special.zeta(_scalar_operand(self, other), other)


# ---------------------------------------------------------------- batch norm of an empty input

def _batch_norm_impl_index(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    """PyTorch's, except that an empty input's (uninitialized) saved
    statistics are float32 without float64, not its accumulate type double."""
    if input.numel() != 0 or _has_float64():
        return aten._batch_norm_impl_index.default.decompose(input, weight, bias, running_mean, running_var, training,
                                                             momentum, eps, cudnn_enabled)
    c = input.shape[1] if input.dim() > 1 else 0
    out = input.clone()
    if weight is not None:
        out = out * weight[0]
    if bias is not None:
        out = out + bias[0]
    stats = torch.empty(c, dtype=torch.float32, device=input.device)
    return out, stats, stats.clone(), torch.empty(0, dtype=torch.uint8, device=input.device), 0


def register():
    """Registers the kernels on AutogradPrivateUse1; returns their names."""
    global _lib
    if _lib is not None:
        return []
    _lib = torch.library.Library("aten", "IMPL")
    kernels = {
        "quantile": _make_quantile(False, aten.quantile.default),
        "quantile.scalar": _make_quantile(False, aten.quantile.scalar),
        "nanquantile": _make_quantile(True, aten.nanquantile.default),
        "nanquantile.scalar": _make_quantile(True, aten.nanquantile.scalar),
        **_overloads("linalg_pinv", _pinv_impl, aten.linalg_pinv),
        **_overloads("linalg_matrix_rank", _matrix_rank_impl, aten.linalg_matrix_rank),
        "_batch_norm_impl_index": _batch_norm_impl_index,
    }
    for name, fn in kernels.items():
        _lib.impl(name, fn, "AutogradPrivateUse1")
    _lib.impl("linalg_pinv.atol_rtol_tensor", lambda x, *, atol=None, rtol=None, hermitian=False:
              _pinv_impl(x, atol, rtol, hermitian), "PrivateUse1")
    # (CompositeExplicitAutograd ops: a backend kernel; zeta has no gradient)
    _lib.impl("special_zeta.other_scalar", _zeta_other_scalar, "PrivateUse1")
    _lib.impl("special_zeta.self_scalar", _zeta_self_scalar, "PrivateUse1")
    return ["aten::" + n for n in kernels] + ["aten::linalg_pinv.atol_rtol_tensor", "aten::special_zeta.other_scalar",
                                              "aten::special_zeta.self_scalar"]
