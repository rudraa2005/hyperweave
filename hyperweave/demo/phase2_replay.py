"""Offline synthetic replay of the actual Phase 1 -> Phase 2 model pipeline."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hyperweave.baselines.phase2 import run as run_phase2
from hyperweave.baselines.run import run as run_phase1
from hyperweave.registry import Registry


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    features = [f"f{i}" for i in range(43)]
    blocks = []
    for campaign in range(4):
        x = rng.normal(size=(300, 43))
        frame = pd.DataFrame(x, columns=features)
        frame["Label"] = (x[:, 0] + .3 * x[:, 1] > 0).astype(int)
        frame["campaign"] = f"debug_campaign_{campaign}"
        frame["timestamp"] = campaign * 300 + np.arange(300)
        blocks.append(frame)
    csv = output / "synthetic_flows.csv"
    pd.concat(blocks, ignore_index=True).to_csv(csv, index=False)
    (output / "features.txt").write_text("\n".join(features) + "\n")
    p1 = output / "phase1"
    run_phase1(csv, features, "nf_unsw_nb15_v2", "campaign", "timestamp", p1,
               epochs=1, target_fpr=.01)
    p2 = output / "phase2"
    report = run_phase2(csv, features, "nf_unsw_nb15_v2", "campaign", "timestamp",
                        p1 / "registry.sqlite", "nf_unsw_nb15_v2_b2_seed11", p2, epochs=1)
    registry = Registry(p2 / "registry.sqlite")
    print("SYNTHETIC OFFLINE REPLAY — debug FPR target 1%; research target remains 0.1%")
    print("Measured on the same batch; latency is kernel-only and excludes input quantization.")
    print("run ID                              detection  FPR      PR-AUC   MACs    bytes   median ns/flow  pinned")
    for run_id in [report["fixed_run_id"]] + report["b3_run_ids"]:
        m = registry.read(run_id)["metrics"]
        f = m["float"]
        print(f"{run_id:35} {f['detection_rate']:.4f}     {f['fpr']:.4f}   {f['pr_auc']:.4f}   "
              f"{m['macs']:6}  {m['int8_model_bytes']:6}  {m['latency']['median_ns_per_flow']:14.1f}  "
              f"{m['latency']['pinned']}")
    print(report["timing_note"])
    print(f"Registry: {p2 / 'registry.sqlite'}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Fresh directory for immutable replay artifacts")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
