"""Finds the vk ops whose results differ from the CPU's inside a PyTorch test.

    python tools/dev/optrace.py test_nn.py -k "linear_cross_entropy and bias_True_float16"

Runs the test on "vk" under a dispatch mode that re-runs every op touching a
vk tensor on CPU copies of its inputs, and prints the ops with the largest
difference in ULPs of the result's dtype (at the value's power of two, so
results near zero or subnormal are exaggerated). Ops implemented in Python
(torch.library custom ops) are traced inside. In-place ops are compared on
their first argument; out= ops on what they return. Random ops are skipped.
DUMP=SUBSTRING DUMPFILE=PATH saves the inputs of the worst matching op
(torch.save) for replaying it alone.
"""
import collections
import os
import sys

os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "vk")
import pytest  # noqa: E402
import torch  # noqa: E402
import torch_vulkan  # noqa: E402,F401
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402
from torch.utils._pytree import tree_flatten, tree_map  # noqa: E402

SKIP = ("normal", "uniform", "random", "bernoulli", "exponential", "rand", "empty", "set_", "dropout", "multinomial")
worst = collections.defaultdict(float)
where = {}


def ulps(a, b):
    a = a.detach().to("cpu")
    if not a.is_floating_point() or a.numel() == 0 or a.shape != b.shape:
        return 0.0
    ad, bd = a.double(), b.detach().double()
    ok = torch.isfinite(bd)
    if (torch.isfinite(ad) != ok).any():
        return float("inf")
    if not ok.any():
        return 0.0
    fi = torch.finfo(a.dtype)
    spacing = fi.eps * torch.exp2(torch.floor(torch.log2(bd.abs().clamp_min(fi.tiny))))
    return ((ad - bd).abs()[ok] / spacing[ok]).max().item()


def to_cpu(x):
    if isinstance(x, torch.Tensor) and x.device.type == "vk":
        return x.detach().to("cpu").clone()
    if isinstance(x, torch.device) and x.type == "vk":
        return torch.device("cpu")
    return x


class Trace(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        flat, _ = tree_flatten((args, kwargs))
        name = str(func)
        on_vk = any(isinstance(t, torch.Tensor) and t.device.type == "vk" for t in flat)
        if not on_vk or any(s in name for s in SKIP):
            return func(*args, **kwargs)
        if name.startswith("torch_nn."):  # a Python custom op: trace inside it
            from torch._library.custom_ops import _maybe_get_opdef
            with Trace():
                return _maybe_get_opdef(func)._init_fn(*args, **kwargs)
        cargs, ckw = tree_map(to_cpu, args), tree_map(to_cpu, kwargs)
        saved = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, (cargs, ckw))
        r = func(*args, **kwargs)
        try:
            cr = func(*cargs, **ckw)
        except Exception:
            return r
        pairs = list(zip(tree_flatten(r)[0], tree_flatten(cr)[0]))
        if func._schema.name.endswith("_") and args and isinstance(args[0], torch.Tensor):  # in place
            pairs.append((args[0], cargs[0]))
        for a, b in pairs:
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                u = ulps(a, b)
                if u > worst[name]:
                    worst[name] = u
                    where[name] = [f"{t.dtype}{tuple(t.shape)}" for t in flat if isinstance(t, torch.Tensor)]
                    if os.environ.get("DUMP") and os.environ["DUMP"] in name:
                        torch.save(saved, os.environ["DUMPFILE"])
        return r


os.chdir(os.environ.get("PYTORCH_TEST_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                       "pytorch-src", "test")))
sys.path.insert(0, os.getcwd())
with Trace():
    pytest.main(["-q", "-p", "no:cacheprovider", "--no-header", "--tb=line", "-W", "ignore"] + sys.argv[1:])
for name, u in sorted(worst.items(), key=lambda kv: -kv[1])[:25]:
    if u > 0:
        print(f"{u:12.1f}  {name}  {where.get(name)}")
