"""Builds crates/pyvk/csrc/cpu_dtype_errors.inc: the CPU's dtype errors for
the argument signatures that a run saw fall back to the CPU only to raise
there, so vk raises them itself (backend.cpp, run_fallback) and nothing
reaches the CPU.

    python tools/dev/cpu_dtype_errors.py RUN_DIR... [--out crates/pyvk/csrc/cpu_dtype_errors.inc]

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
    r"result type \w+ can't be cast to the desired output type",
    r"Expected out tensor to have dtype",
    r"expected to have the same dtype",
    r"must have the same scalar type",
    r"expected scalar type \w+ but found",
    r"Found dtype \w+ but expected",
]))


def make_arg(text):
    """A CPU value for one described argument, or raise ValueError."""
    text = text.strip()
    if text == "None":
        return None
    if text in ("int", "float", "bool", "complex"):
        return {"int": 1, "float": 0.5, "bool": True, "complex": 1j}[text]
    if text.startswith("list(") and text.endswith(")"):
        inner = text[5:-1]
        return [make_arg(inner)] if inner else []
    m = re.match(r"^(\w+)( cpu| nested)? (\d+)d( empty)?( sparse)?$", text)
    if not m or m.group(1) not in C10_NAMES or m.group(2) == " nested" or m.group(5):
        raise ValueError(text)
    dims = int(m.group(3))
    shape = [0 if m.group(4) else 2] * dims
    return torch.ones(shape, dtype=C10_NAMES[m.group(1)])


def dtype_signature(name, described):
    """The lookup key: the op and its arguments' dtypes and kinds (no dims)."""
    parts = []
    for text in described:
        text = text.strip()
        m = re.match(r"^(\w+)( cpu| nested)? \d+d( empty)?", text)
        parts.append(m.group(1) + (" empty" if m.group(3) else "") if m else "list" if text.startswith("list(") else text)
    return f"{name} [{', '.join(parts)}]"


def split_args(body):
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
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
    try:
        values = [make_arg(t) for t in described]
    except ValueError:
        return None
    schema = op._schema
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
        return ("NotImplementedError", msg) if DTYPE_ERROR.search(msg) else None
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        return ("RuntimeError", msg) if DTYPE_ERROR.search(msg) else None
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[2] / "crates/pyvk/csrc/cpu_dtype_errors.inc"))
    args = ap.parse_args()
    table = {}
    skipped = 0
    for name, body in raised_signatures(args.dirs):
        described = split_args(body)
        result = probe(name, described)
        if result is None:
            skipped += 1
            continue
        table.setdefault(dtype_signature(name, described), result)
    lines = [
        "// Generated by tools/dev/cpu_dtype_errors.py from a run's raised fallbacks: the CPU's dtype",
        "// errors by op and argument dtypes (backend.cpp raises them for vk). Do not edit by hand.",
    ]
    for key, (kind, msg) in sorted(table.items()):
        lines.append(f"{{{c_string(key)}, {'true' if kind == 'NotImplementedError' else 'false'}, {c_string(msg)}}},")
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(table)} signatures written to {args.out}; {skipped} raised signatures not dtype errors")


if __name__ == "__main__":
    main()
