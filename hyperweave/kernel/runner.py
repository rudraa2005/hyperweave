import json
import struct
import subprocess
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[2] / "kernel" / "infer.c"


def compile_kernel(output):
    output = Path(output)
    subprocess.run(["cc", "-O3", "-std=c11", "-D_GNU_SOURCE", str(SOURCE), "-o", str(output)], check=True)
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
