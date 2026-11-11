"""Builds crates/pyvk/csrc/cpu_dtype_errors.inc: the CPU's dtype errors for
the argument signatures that a run saw fall back to the CPU only to raise
there, so vk raises them itself (backend.cpp, run_fallback) and nothing
reaches the CPU.

    python tools/dev/cpu_dtype_errors.py RUN_DIR... [--out crates/pyvk/csrc/cpu_dtype_errors.inc]
    python tools/dev/cpu_dtype_errors.py --check [--all]     (CI: the table still matches the CPU)

RUN_DIR: tools/pytorch_tests.py results recorded with TORCH_VK_FALLBACK_DETAIL=1
(their cpu_ops entries name each fallback's signature, "aten::lerp.Scalar_out
[Int 1d, Int 1d, float, Int 1d] (raised)"). Each raised signature is rebuilt
from small CPU tensors of those dtypes and dims and run on the CPU; where the
CPU raises a dtype dispatch error ('"lerp_kernel_scalar" not implemented for
'Int''), the signature and message go in the table. Signatures whose error
depends on more than the dtypes (shapes, values) are left out: vk sends those
to the CPU, which raises.
"""

import argparse
import json
import pathlib
import re

import torch

DTYPES = {str(d).removeprefix("torch."): d for d in vars(torch).values() if isinstance(d, torch.dtype)}
# c10's names (c10::toString) for torch's dtypes
C10_NAMES = {
    "Float": torch.float32, "Double": torch.float64, "Half": torch.float16, "BFloat16": torch.bfloat16,
    "ComplexHalf": torch.complex32, "ComplexFloat": torch.complex64, "ComplexDouble": torch.complex128,
    "Bool": torch.bool, "Byte": torch.uint8, "Char": torch.int8, "Short": torch.int16, "Int": torch.int32,
    "Long": torch.int64, "UInt16": torch.uint16, "UInt32": torch.uint32, "UInt64": torch.uint64,
    "Float8_e4m3fn": torch.float8_e4m3fn, "Float8_e5m2": torch.float8_e5m2,
    "Float8_e4m3fnuz": torch.float8_e4m3fnuz, "Float8_e5m2fnuz": torch.float8_e5m2fnuz,
}
if hasattr(torch, "bcomplex32"):
    C10_NAMES["BComplex32"] = torch.bcomplex32
SIGNATURE = re.compile(r"^(aten::[\w.]+) \[(.*)\] \(raised\)$")
# CPU errors that depend on the argument dtypes (and emptiness) alone
DTYPE_ERROR = re.compile("|".join([
    r"not implemented for '\w+'",
    r"value cannot be converted to type \w+ without overflow",
    r"only support floating point and complex dtypes",
    r"dtype must be a floating point but you specified",
    r"is not supported for complex",
    r"inputs not supported for",
    r"Negation, the `-` operator, on a bool tensor",
    r"softmax with half to float conversion is not supported on CPU",
    r"Integer division with addcdiv is no longer supported",
    r"input dtype should be either floating point or complex",
    r"Expected a floating point or complex tensor as input",
    r"result type \w+ can't be cast to the desired output type",
    r"Expected out tensor to have dtype",
    r"expected to have the same dtype",
    r"must have the same scalar type",
    r"expected scalar type \w+ but found",
    r"Found dtype \w+ but expected",
]))


def make_list(arg):
    """A plausible value for a list argument described only as "GenericList"."""
    t = str(arg.type) if arg is not None else ""
    t = re.sub(r"^Optional\[(.*)\]$", r"\1", t)
    m = re.match(r"^(?:(?:Sym)?[Ii]nt\[(\d*)\]|List\[(?:Sym)?[Ii]nt\])", t)
    if m:
        n = int(m.group(1) or 2)
        if "dim" in arg.name:
            return [0]
        return [0] * n if "padding" in arg.name else [1] * n
    if t.startswith("float[") or t == "List[float]":
        return [1.0]
    raise ValueError(t)


def make_arg(text, arg=None, device=None):
    """A value for one described argument (CPU tensors, or on `device` where the
    description has no " cpu"), or raise ValueError. `arg`: its schema argument."""
    text = text.strip()
    if text in ("None", "undef"):
        return None
    if text in ("int", "float", "bool", "complex"):
        return {"int": 1, "float": 0.5, "bool": True, "complex": 1j}[text]
    if text == "GenericList":
        return make_list(arg)
    if re.match(r"^\[[-\d, ]*\]$", text):  # (an int list, described by value)
        return [int(v) for v in text[1:-1].split(",") if v.strip()]
    if text.startswith("list(") and text.endswith(")"):
        inner = text[5:-1]
        return [make_arg(inner, None, device)] if inner else []
    m = re.match(r"^(\w+)( cpu| nested)? (\d+)d( empty)?( sparse)?$", text)
    if m and m.group(1) in C10_NAMES and m.group(2) != " nested" and not m.group(5):
        dims = int(m.group(3))
        dt = C10_NAMES[m.group(1)]
        try:
            t = torch.ones([0 if m.group(4) else 2] * dims, dtype=dt)
        except (RuntimeError, NotImplementedError):  # (no CPU fill for this dtype: through complex64)
            t = torch.ones([0 if m.group(4) else 2] * dims, dtype=torch.complex64).to(dt)
        return t.to(device) if device is not None and m.group(2) != " cpu" else t
    if arg is not None and str(arg.type).removeprefix("Optional[").startswith("str") and re.match(r"^\w+$", text):
        return text  # (a string argument: "amax", "floor", ...)
    raise ValueError(text)


def dtype_signature(name, described):
    """The lookup key: the op and its arguments' dtypes and kinds (no dims)."""
    parts = []
    for text in described:
        text = text.strip()
        m = re.match(r"^(\w+)( cpu| nested)? \d+d( empty)?", text)
        if m:
            parts.append(m.group(1) + (" empty" if m.group(3) else ""))
        elif text.startswith("list("):
            parts.append("list")
        elif re.match(r"^\[[-\d, ]*\]$", text):
            parts.append("IntList")  # (an int list's values, as Python's fallback describes it: not in the key)
        else:
            parts.append(text)
    return f"{name} [{', '.join(parts)}]"


def split_args(body):
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def probe(name, described):
    """(kind, message) of the CPU's error for this signature, or None."""
    qualified = name.removeprefix("aten::")
    packet, _, overload = qualified.partition(".")
    op = getattr(getattr(torch.ops.aten, packet), overload or "default")
    schema = op._schema
    try:
        values = [make_arg(t, a) for t, a in zip(described, schema.arguments)]
    except ValueError:
        return None
    # (shape lists, layer_norm's normalized_shape: the next tensor's dims, of size 2)
    for i, a in enumerate(schema.arguments):
        if i < len(described) and described[i].strip() == "GenericList" and "shape" in a.name:
            later = [v for v in values[i + 1:] if isinstance(v, torch.Tensor)]
            values[i] = [2] * (later[0].dim() if later else 1)
    # Scalar arguments (histc's min and max, clamp's...) in several values: the
    # error must be the same for each, or it depends on them (min > max raises
    # first on the CPU), and vk leaves the signature to the CPU's checks.
    scalars = [i for i, (t, a) in enumerate(zip(described, schema.arguments))
               if t.strip() in ("int", "float") and str(a.type).removeprefix("Optional[").startswith("number")]
    found = set()
    for variant in range(len(SCALAR_VARIANTS) if scalars else 1):
        vals = list(values)
        for k, i in enumerate(scalars):
            vals[i] = SCALAR_VARIANTS[(variant + k) % len(SCALAR_VARIANTS)]
        found.add(run_probe(op, schema, vals))
    return found.pop() if len(found) == 1 else None


SCALAR_VARIANTS = [0.5, -1.0, 2.0, 0.0]


_CPP_NAMES = None


def cpp_names():
    """{c10 name: this platform's C++ type name} (c10's TypeMeta names, as messages
    such as "but got long int instead" print them), from the CPU's own out= dtype
    error on 1x1 tensors."""
    global _CPP_NAMES
    if _CPP_NAMES is None:
        _CPP_NAMES = {}
        for c10, dtype in C10_NAMES.items():
            src = torch.float64 if dtype == torch.float32 else torch.float32
            try:
                torch.mm(torch.ones(1, 1, dtype=src), torch.ones(1, 1, dtype=src), out=torch.empty(1, 1, dtype=dtype))
            except RuntimeError as e:
                m = re.search(r"but got (.*) instead", str(e))
                if m:
                    _CPP_NAMES[c10] = m.group(1)
    return _CPP_NAMES


# Messages that name dtypes by their C++ types: those names become {c10:Name}.
CPP_NAMED = [re.compile(r"(have dtype )(.+?)(, but got )(.+?)( instead)"), re.compile(r"(you specified )(.+)$")]


def templatize(msg):
    """msg with C++ type names, where it prints them, as {c10:Name} (the table
    is the same on every platform; backend.cpp and _decomp.py fill them in)."""
    back = {v: k for k, v in cpp_names().items()}
    for pat in CPP_NAMED:
        m = pat.search(msg)
        if m:
            parts = list(m.groups())
            for i in range(1, len(parts), 2):
                if parts[i] in back:
                    parts[i] = "{c10:" + back[parts[i]] + "}"
            msg = msg[:m.start()] + "".join(parts) + msg[m.end():]
    return msg


def render(msg):
    """templatize's inverse, with this platform's names."""
    return re.sub(r"\{c10:(\w+)\}", lambda m: cpp_names().get(m.group(1), m.group(1)), msg)


def run_probe(op, schema, values):
    args, kwargs = [], {}
    for a, v in zip(schema.arguments, values):
        if a.kwarg_only:
            kwargs[a.name] = v
        else:
            args.append(v)
    try:
        op(*args, **kwargs)
    except NotImplementedError as e:
        msg = str(e).splitlines()[0]
        return ("NotImplementedError", templatize(msg)) if DTYPE_ERROR.search(msg) else None
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        return ("RuntimeError", templatize(msg)) if DTYPE_ERROR.search(msg) else None
    except Exception:
        return None
    return None


def raised_signatures(dirs):
    found = set()
    for d in dirs:
        for p in pathlib.Path(d).glob("*.jsonl"):
            if p.name.endswith(".cpu.jsonl"):
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                for key in entry.get("cpu_ops", {}):
                    m = SIGNATURE.match(key)
                    if m:
                        found.add((m.group(1), m.group(2)))
    return sorted(found)


def c_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


EXAMPLES = pathlib.Path(__file__).with_name("cpu_dtype_errors_examples.json")
PY_TABLE = pathlib.Path(__file__).resolve().parents[2] / "python/torch_vulkan/_cpu_dtype_errors.py"


def check(count, everything):
    """Rebuilds the table's signatures (a deterministic sample of `count`, or
    all) on the CPU from their recorded examples; returns the mismatches."""
    ns = {}
    exec(PY_TABLE.read_text(encoding="utf-8"), ns)
    table = ns["TABLE"]
    examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))
    keys = sorted(table)
    if not everything and len(keys) > count:
        step = len(keys) / count
        keys = [keys[int(i * step)] for i in range(count)]
    bad = []
    for key in keys:
        name, body = key.split(" [", 1)[0], examples.get(key)
        want = ("NotImplementedError" if table[key][0] else "RuntimeError", table[key][1])
        got = probe(name, split_args(body)) if body is not None else None
        if got != want:
            bad.append((key, want, got))
    return len(keys), bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[2] / "crates/pyvk/csrc/cpu_dtype_errors.inc"))
    ap.add_argument("--check", action="store_true",
                    help="rebuild a sample of the table's signatures on the CPU (from their recorded examples) and "
                         "fail if the CPU's exception type or message no longer matches (CI, after a PyTorch upgrade)")
    ap.add_argument("--all", action="store_true", help="with --check: every signature, not a sample")
    ap.add_argument("--sample", type=int, default=100, help="with --check: how many signatures")
    args = ap.parse_args()
    if args.check:
        n, bad = check(args.sample, args.all)
        for key, want, got in bad:
            print(f"MISMATCH {key}\n  table: {want}\n  CPU:   {got}")
        print(f"{n} signatures checked, {len(bad)} mismatched")
        raise SystemExit(1 if bad else 0)
    if not args.dirs:
        ap.error("RUN_DIR... needed (or --check)")
    table, examples = {}, {}
    skipped = 0
    for name, body in raised_signatures(args.dirs):
        described = split_args(body)
        result = probe(name, described)
        if result is None:
            skipped += 1
            continue
        key = dtype_signature(name, described)
        table.setdefault(key, result)
        examples.setdefault(key, body)
    lines = [
        "// Generated by tools/dev/cpu_dtype_errors.py from a run's raised fallbacks: the CPU's dtype",
        "// errors by op and argument dtypes (backend.cpp raises them for vk). Do not edit by hand.",
    ]
    for key, (kind, msg) in sorted(table.items()):
        lines.append(f"{{{c_string(key)}, {'true' if kind == 'NotImplementedError' else 'false'}, {c_string(msg)}}},")
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    # The same table for the Python kernels' route to the CPU (_decomp.py).
    rows = ["# Generated by tools/dev/cpu_dtype_errors.py, as crates/pyvk/csrc/cpu_dtype_errors.inc: the CPU's",
            "# dtype errors by op and argument dtypes (_decomp.py raises them for vk). Do not edit by hand.",
            "TABLE = {"]
    for key, (kind, msg) in sorted(table.items()):
        rows.append(f"    {key!r}: ({kind == 'NotImplementedError'}, {msg!r}),")
    rows.append("}")
    PY_TABLE.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    # One argument description per signature, for --check.
    EXAMPLES.write_text(json.dumps(dict(sorted(examples.items())), indent=0) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(table)} signatures written to {args.out}; {skipped} raised signatures not dtype errors")


if __name__ == "__main__":
    main()
