"""Barriers only where a dependency needs one (docs/log/M37).

The stream leaves out the barrier between commands that touch disjoint
memory, so a dependency the stream misses is a silent race. These tests run
long random programs of independent and dependent ops on aliased views of
one allocation (overlapping, interleaved, transposed, more than four dims)
and separate tensors, in exact integer arithmetic in float32, so the CPU's
results must match bit for bit, at host reads in between and at the end.

TORCH_VK_STRESS_SEEDS (default 6) and TORCH_VK_STRESS_STEPS (default 250)
scale the random programs up for runs on big GPUs.
"""
import os
import random

import pytest
import torch

import torch_vulkan as tv
from torch_vulkan import _native

SEEDS = int(os.environ.get("TORCH_VK_STRESS_SEEDS", "6"))
STEPS = int(os.environ.get("TORCH_VK_STRESS_STEPS", "250"))
MOD = 97  # values stay integers in [0, MOD) after every op that grows them


def barriers():
    return dict(_native.barrier_stats())["barriers"]


def commands():
    return dict(_native.barrier_stats())["commands"]


def tracked():
    return _native.barrier_mode() in ("tracked", "check")


class World:
    """The same tensors on one device: views of one storage S, and separate
    tensors, all holding small integers."""

    def __init__(self, device, gen):
        n = 8192
        self.S = torch.randint(0, MOD, (n,), generator=gen).float().to(device)
        S = self.S
        self.mats = {
            "v0": S[0:4096].view(64, 64),
            "v1": S[2048:6144].view(64, 64),  # overlaps v0 and v2 by half
            "v2": S[4096:8192].view(64, 64),
            "v3": S[0:8192:2].view(64, 64),  # interleaved with v4: same span, no common element
            "v4": S[1:8192:2].view(64, 64),
            "v5": S[0:4096].view(64, 64).t(),  # v0 transposed
            "v6": S[64:4160].view(64, 64)[:, :32].t(),  # rows of v0 shifted, half the columns: 32 x 64
        }
        for name in "ABC":
            self.mats[name] = torch.randint(0, MOD, (64, 64), generator=gen).float().to(device)
        self.mats["D"] = torch.randint(0, MOD, (32, 64), generator=gen).float().to(device)
        # More than four dims that do not merge: the strided kernels loop over
        # the outer ones on the host, one dispatch per index.
        self.bigs = {
            "g0": S[0:4096].view(2, 2, 2, 2, 2, 2, 64).permute(6, 0, 2, 4, 1, 3, 5),
            "g1": S[4096:8192].view(2, 2, 2, 2, 2, 2, 64).permute(6, 1, 3, 5, 0, 2, 4),
            "g2": S[2048:6144].view(2, 2, 2, 2, 2, 2, 64).permute(6, 5, 4, 3, 2, 1, 0),
            "G": torch.randint(0, MOD, (64, 2, 2, 2, 2, 2, 2), generator=gen).float().to(device),
        }


def elements(t):
    """The storage offsets of t's elements (a CPU tensor)."""
    idx = torch.zeros((), dtype=torch.long)
    for size, stride in zip(t.shape, t.stride()):
        idx = idx.unsqueeze(-1) + torch.arange(size) * stride
    return idx.flatten() + t.storage_offset()


def overlap(a, b):
    if a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr():
        return False
    return bool(torch.isin(elements(a), elements(b)).any())


def program(seed, steps):
    """A list of ops, each f(world) on either device, chosen for the CPU
    world `ref`. Ops whose result depends on the order the elements are
    computed in (an output overlapping an input other than itself, which
    PyTorch does not always detect) are not chosen."""
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    ref = World("cpu", gen)
    ops = []

    def pick(group, shape=None):
        names = [k for k, v in group.items() if shape is None or tuple(v.shape) == shape]
        return rng.choice(names)

    for _ in range(steps):
        kind = rng.randrange(16)
        M, G = ref.mats, ref.bigs
        if kind <= 3:  # element-wise, in place or out=, on matrices or 7-dim views
            group = rng.choice(["mats", "mats", "bigs"])
            grp = getattr(ref, group)
            o = pick(grp)
            shape = tuple(grp[o].shape)
            x, y = pick(grp, shape), pick(grp, shape)
            if any(i != o and overlap(grp[o], grp[i]) for i in (x, y)):
                continue  # (PyTorch cannot always tell, and the result depends on the order)
            k = rng.randrange(1, 5)
            choice = rng.randrange(5)

            def op(w, group=group, o=o, x=x, y=y, k=k, choice=choice):
                g = getattr(w, group)
                if choice == 0:
                    g[o].add_(g[x], alpha=k).remainder_(MOD)
                elif choice == 1:
                    torch.add(g[x], g[y], out=g[o])
                    g[o].remainder_(MOD)
                elif choice == 2:
                    torch.mul(g[x], g[y], out=g[o])
                    g[o].remainder_(MOD)
                elif choice == 3:
                    g[o].mul_(k).sub_(g[x]).remainder_(MOD)
                else:
                    torch.where(g[x] > MOD // 2, g[x], g[y], out=g[o])

        elif kind == 4:  # copies, also transposed and strided
            o = pick(M)
            x = pick(M, tuple(M[o].shape))
            t = rng.random() < 0.5 and M[o].shape[0] == M[o].shape[1]
            if overlap(M[o], M[x]) and (x != o or t):
                continue

            def op(w, o=o, x=x, t=t):
                (w.mats[o].t() if t else w.mats[o]).copy_(w.mats[x])

        elif kind == 5:  # fills of a view's rows
            o = pick(M)
            r0 = rng.randrange(M[o].shape[0])
            r1 = rng.randrange(r0, M[o].shape[0]) + 1
            v = float(rng.randrange(MOD))

            def op(w, o=o, r0=r0, r1=r1, v=v):
                w.mats[o][r0:r1].fill_(v)

        elif kind == 6:  # a matrix product into a matrix that overlaps neither input
            o = pick(M, (64, 64))
            x, y = pick(M, (64, 64)), pick(M, (64, 64))
            if overlap(M[o], M[x]) or overlap(M[o], M[y]):
                continue

            def op(w, o=o, x=x, y=y):
                torch.mm(w.mats[x], w.mats[y], out=w.mats[o])
                w.mats[o].remainder_(MOD)

        elif kind == 7:  # a reduction into one row of a view
            o, x = pick(M), pick(M, (64, 64))
            r = rng.randrange(M[o].shape[0])
            if M[o].shape[1] != 64 or overlap(M[o][r], M[x]):
                continue

            def op(w, o=o, x=x, r=r):
                torch.sum(w.mats[x], 0, out=w.mats[o][r])
                w.mats[o][r].remainder_(MOD)

        elif kind == 8:  # index_add_ with distinct indices (deterministic)
            o = pick(M, (64, 64))
            x = pick(M)
            n = M[x].shape[0]
            if overlap(M[o], M[x]):
                continue
            idx = torch.tensor(rng.sample(range(64), n))

            def op(w, o=o, x=x, idx=idx):
                w.mats[o].index_add_(0, idx.to(w.S.device), w.mats[x]).remainder_(MOD)

        elif kind == 9:  # temporaries (freed and reused from the pool at once)
            o = pick(M)
            x = pick(M, tuple(M[o].shape))

            def op(w, o=o, x=x):
                t = (w.mats[x] * 3 + 1).remainder(MOD)
                s = torch.sort(t, 1).values
                w.mats[o].copy_((s + t).remainder(MOD))

        elif kind == 10:  # cumulative sums out= into a matrix that does not overlap the input
            o = pick(M)
            x = pick(M, tuple(M[o].shape))
            if overlap(M[o], M[x]):
                continue

            def op(w, o=o, x=x):
                torch.cumsum(w.mats[x], 1, out=w.mats[o])
                w.mats[o].remainder_(MOD)

        elif kind == 11:  # gather and scatter through index_select / index_put_
            o = pick(M, (64, 64))
            x = pick(M, (64, 64))
            if overlap(M[o], M[x]):
                continue
            idx = torch.tensor([rng.randrange(64) for _ in range(64)])
            perm = torch.tensor(rng.sample(range(64), 64))

            def op(w, o=o, x=x, idx=idx, perm=perm):
                d = w.S.device
                sel = torch.index_select(w.mats[x], 1, idx.to(d))
                w.mats[o].index_put_((perm.to(d),), sel)

        elif kind == 12:  # the 7-dim views copied into each other
            o = pick(G)
            x = pick(G)

            def op(w, o=o, x=x):
                w.bigs[o].copy_(w.bigs[x].flip(1))

        elif kind == 13:  # concatenation into a matrix (copies into its slices)
            o = pick(M, (64, 64))
            x, y = pick(M, (64, 64)), pick(M, (64, 64))
            if overlap(M[o], M[x]) or overlap(M[o], M[y]):
                continue

            def op(w, o=o, x=x, y=y):
                torch.cat([w.mats[x][:24], w.mats[y][24:]], 0, out=w.mats[o])

        else:  # a host read in between
            x = pick(M)

            def op(w, x=x):
                return w.mats[x].cpu() if w.S.device.type != "cpu" else w.mats[x].clone()

        try:
            op(ref)
        except RuntimeError:
            continue  # (e.g. an in-place op on partly overlapping operands: the CPU raises too)
        ops.append(op)
    return ops


def run(ops, device, seed):
    w = World(device, torch.Generator().manual_seed(seed))
    reads = []
    for op in ops:
        r = op(w)
        if r is not None:
            reads.append(r)
    return w, reads


@pytest.mark.parametrize("seed", range(SEEDS))
def test_random_programs_on_aliased_views(seed):
    ops = program(seed, STEPS)
    ref, ref_reads = run(ops, "cpu", seed)
    b0, c0 = barriers(), commands()
    got, got_reads = run(ops, "vk", seed)
    tv.synchronize()
    elided = (commands() - c0) - (barriers() - b0)
    for i, (a, b) in enumerate(zip(ref_reads, got_reads)):
        assert torch.equal(a, b), f"host read {i} differs"
    assert torch.equal(got.S.cpu(), ref.S)
    for name in "ABCD":
        assert torch.equal(got.mats[name].cpu(), ref.mats[name]), name
    assert torch.equal(got.bigs["G"].cpu(), ref.bigs["G"])
    if tracked():
        assert elided > 0  # (some barriers were left out: the mechanism is on)


def test_independent_ops_need_no_barrier_dependent_ones_do():
    a = [torch.full((256,), float(i), device="vk") for i in range(10)]
    o = [torch.empty(256, device="vk") for _ in range(10)]
    tv.synchronize()
    b0 = barriers()
    for x, y in zip(a, o):
        torch.add(x, 1.0, out=y)
    independent = barriers() - b0
    b0 = barriers()
    for _ in range(10):
        a[0].add_(1.0)
    dependent = barriers() - b0
    tv.synchronize()
    assert [y[0].item() for y in o] == [i + 1.0 for i in range(10)]
    assert a[0][0].item() == 10.0
    assert dependent >= 10 if tracked() else dependent == 10
    if tracked():
        assert independent <= 1


def test_disjoint_slices_then_a_reduction():
    # Writes into disjoint rows of one tensor run without barriers between
    # them; the reduction reading all rows must still wait for every write.
    x = torch.zeros(200, 64, device="vk")
    for it in range(5):
        for i in range(200):
            x[i].add_(float(i + it))
        s = x.sum(1).cpu()
        expected = torch.tensor([sum(float(i + k) for k in range(it + 1)) * 64 for i in range(200)])
        assert torch.equal(s, expected), it


def test_write_after_read():
    # Many readers of x, then x overwritten: no reader may see the new values.
    x = torch.arange(4096.0, device="vk")
    ref = torch.arange(4096.0)
    results = []
    for it in range(20):
        results += [(x * 1.0 + float(k), ref + k) for k in range(8)]
        x.fill_(float(-it))
        ref = torch.full((4096,), float(-it))
    for i, (y, expected) in enumerate(results):
        assert torch.equal(y.cpu(), expected), i


def test_pool_reuse_write_after_read():
    # A freed temporary's memory goes straight to the next allocation while
    # kernels reading it may still be queued.
    acc = torch.zeros(1024, device="vk")
    expected = torch.zeros(1024)
    for i in range(300):
        t = torch.full((1024,), float(i % 7), device="vk")
        acc.add_(t)
        expected += float(i % 7)
        del t
    assert torch.equal(acc.cpu(), expected)


def test_host_reads_see_every_write():
    xs = [torch.zeros(64, device="vk") for _ in range(50)]
    for r in range(10):
        for i, x in enumerate(xs):
            x.add_(float(i))
        assert [x[0].item() for x in xs] == [float(i * (r + 1)) for i in range(50)]


def test_split_k_reduction_reads_the_pieces():
    # Matrix products split along k write a workspace that a second kernel
    # sums: a read after write within one op, followed by independent work.
    torch.manual_seed(0)
    a = torch.randint(-3, 4, (64, 4096)).float()
    b = torch.randint(-3, 4, (4096, 64)).float()
    va, vb = a.to("vk"), b.to("vk")
    outs = []
    for _ in range(8):
        outs.append(va @ vb)
        va.add_(1.0)
        a += 1.0
    expected = []
    a0 = a - 8.0
    for i in range(8):
        expected.append((a0 + i) @ b)
    for i, (o, e) in enumerate(zip(outs, expected)):
        assert torch.equal(o.cpu(), e), i


def test_mode_switch_orders_earlier_commands():
    x = torch.zeros(1 << 16, device="vk")
    mode = _native.barrier_mode()
    try:
        for m in ["every", "tracked", "full", "tracked"]:
            _native.set_barrier_mode(m)
            x.add_(1.0)
        assert x.sum().item() == 4.0 * (1 << 16)
    finally:
        _native.set_barrier_mode(mode)
