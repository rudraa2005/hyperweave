import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "kernel" / "infer.c"
TIMING_BATCH_ROWS = 256


def compile_kernel(output):
    output = Path(output)
    compiler = shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("C compiler required: install cc or gcc")
    if os.name == "nt":
        output = output.with_suffix(".exe")
    subprocess.run([compiler, "-O3", "-std=c11", "-D_GNU_SOURCE", str(SOURCE), "-o", str(output)], check=True)
    return output


def write_inputs(path, inputs_q):
    x = np.ascontiguousarray(inputs_q, dtype=np.int8)
    if x.ndim != 2:
        raise ValueError("expected two-dimensional batch")
    with Path(path).open("wb") as out:
        out.write(struct.pack("<I", len(x)))
        out.write(x.tobytes())


def infer(binary, model_path, input_path, count):
    result = subprocess.run([str(binary), "infer", str(model_path), str(input_path)],
                            check=True, capture_output=True)
    logits = np.frombuffer(result.stdout, dtype="<i4").copy()
    if len(logits) != count:
        raise ValueError("kernel returned wrong number of logits")
    return logits


def bench(binary, model_path, input_path, *, warmup=100, iterations=1000):
    result = subprocess.run([str(binary), "bench", str(model_path), str(input_path),
                             str(warmup), str(iterations)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def enforce_latency_budget(measurement, *, median_ns, p99_ns):
    """Deployment gate: only a pinned-core measurement may satisfy a budget."""
    if not measurement.get("pinned"):
        raise RuntimeError("line-rate budget requires a pinned-core measurement")
    if median_ns <= 0 or p99_ns <= 0:
        raise ValueError("latency budgets must be positive")
    if measurement.get("median_ns_per_flow") is None or measurement.get("p99_ns_per_flow") is None:
        raise ValueError("measurement is missing median or p99")
    if measurement["median_ns_per_flow"] > median_ns or measurement["p99_ns_per_flow"] > p99_ns:
        raise RuntimeError("line-rate latency budget exceeded")
    return True
