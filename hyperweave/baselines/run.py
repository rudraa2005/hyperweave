"""Run B1/B2 on a user-provided NF-v2 CSV and ordered 43-feature manifest.

Example: python3 -m hyperweave.baselines.run --csv flows.csv --features features.txt
  --dataset nf_unsw_nb15_v2 --campaign-col campaign --time-col timestamp
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from hyperweave.config import Config
from hyperweave.data import deduplicate, load_nf_csv, split_by_campaign, validate_disjoint, validate_chronological
from hyperweave.data.flows import dataset_hash
from hyperweave.eval import score_binary, threshold_at_fpr
from hyperweave.models import AffineMLP, FunctionalNet, match_affine_width
from hyperweave.registry import Registry
from hyperweave.seed import seed_everything


def _git_commit():
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unversioned"


def _arrays(frame, features, center, scale):
    x = (frame[features].to_numpy(dtype=float) - center) / scale
    return x, frame["Label"].to_numpy(dtype=int)


def run(csv_path, features, dataset, campaign_col, time_col, output, *, seed=11, neurons=16, h=4, epochs=20, target_fpr=.001):
    seed_everything(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = deduplicate(load_nf_csv(csv_path, dataset, features, campaign_col=campaign_col, time_col=time_col), features)
    splits = split_by_campaign(df, campaign_col, time_col)
    validate_disjoint(splits, features, campaign_col)
    validate_chronological(splits, time_col)
    center = splits["a_train"][features].mean().to_numpy(dtype=float)
    scale = splits["a_train"][features].std().fillna(0).to_numpy(dtype=float, copy=True)
    scale[scale == 0] = 1
    train_x, train_y = _arrays(splits["a_train"], features, center, scale)
    val_x, val_y = _arrays(splits["a_val"], features, center, scale)
    b_x, b_y = _arrays(splits["b"], features, center, scale)
    if not all(np.unique(y).size == 2 for y in (train_y, val_y, b_y)):
        raise ValueError("train, A-val and B must each contain both classes")
    n_benign = int((val_y == 0).sum())
    if n_benign < int(np.ceil(1 / target_fpr)):
        raise ValueError("A-val has insufficient benign flows to resolve target FPR")
    d = len(features)
    models = {"b1": AffineMLP(d, match_affine_width(d, neurons, h), seed),
              "b2": FunctionalNet(d, neurons, h, seed)}
    baseline = LogisticRegression(max_iter=1000, random_state=seed).fit(train_x, train_y)
    baseline_threshold = threshold_at_fpr(baseline.decision_function(val_x)[val_y == 0], target_fpr)
    baseline_b = score_binary(b_y, baseline.decision_function(b_x), baseline_threshold)
    registry = Registry(output / "registry.sqlite")
    run_ids = [f"{dataset}_{name}_seed{seed}" for name in models]
    if any(registry.read(run_id) is not None or (output / f"{run_id}.npz").exists() for run_id in run_ids):
        raise FileExistsError("run ID or checkpoint already exists")
    records = []
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(train_x, train_y, epochs=epochs, seed=seed)
        training_s = time.perf_counter() - t0
        val_scores = model.logits(val_x)
        threshold = threshold_at_fpr(val_scores[val_y == 0], target_fpr)
        t0 = time.perf_counter_ns()
        b_scores = model.logits(b_x)
        float_ns_per_flow = (time.perf_counter_ns() - t0) / len(b_x)
        metrics = score_binary(b_y, b_scores, threshold)
        metrics.update(train_seconds=training_s, float_ns_per_flow=float_ns_per_flow,
                       macs=model.macs, params=model.params, threshold=threshold,
                       threshold_source="a_val", logistic_b_detection=baseline_b["detection_rate"],
                       beats_logistic_on_b=metrics["detection_rate"] > baseline_b["detection_rate"])
        checkpoint = output / f"{dataset}_{name}_seed{seed}.npz"
        np.savez(checkpoint, **{k: v for k, v in vars(model).items() if isinstance(v, (np.ndarray, float))},
                 feature_center=center, feature_scale=scale)
        config = Config(seed=seed, dataset=dataset, model=name, input_dim=d,
                        width=neurons, functional_hidden=h, epochs=epochs, target_fpr=target_fpr)
        record = {"run_id": f"{dataset}_{name}_seed{seed}", "config": config.as_dict(), "seed": seed,
                  "dataset_hash": dataset_hash(csv_path), "git_commit": _git_commit(),
                  "metrics": metrics, "checkpoint_path": str(checkpoint)}
        registry.write(record)
        records.append(record)
    (output / "phase1_summary.json").write_text(json.dumps(records, indent=2, sort_keys=True))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features", required=True, help="Text file with 43 ordered NF-v2 feature names")
    parser.add_argument("--dataset", required=True, choices=["nf_unsw_nb15_v2", "nf_cse_cic_ids2018_v2"])
    parser.add_argument("--campaign-col", required=True)
    parser.add_argument("--time-col", required=True)
    parser.add_argument("--output", default="runs/phase1")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    features = Path(args.features).read_text().splitlines()
    records = run(args.csv, features, args.dataset, args.campaign_col, args.time_col,
                  args.output, seed=args.seed, epochs=args.epochs)
    print(json.dumps([{"run_id": r["run_id"], "metrics": r["metrics"]} for r in records], indent=2))


if __name__ == "__main__":
    main()
