"""Streaming real NF-v2 CSV audit and training-only smoke check.

This command never invents campaign/time metadata and never reports detection
performance without a valid split.
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from hyperweave.config import Config
from hyperweave.data.flows import csv_header, iter_nf_csv_chunks, validate_nf_header
from hyperweave.models import AffineMLP, FunctionalNet
from hyperweave.registry import Registry
from hyperweave.seed import seed_everything


def _hashes(path):
    sha256, sha1 = hashlib.sha256(), hashlib.sha1()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(chunk)
            sha1.update(chunk)
    return sha256.hexdigest(), sha1.hexdigest()


def _git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def audit(path, dataset, features, output, *, chunksize=100000, smoke_train=False,
          sample_per_class=500, duplicate_check_limit=500000):
    path, output = Path(path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh audit output directory")
    output.mkdir(parents=True, exist_ok=True)
    header = validate_nf_header(path, dataset, features)
    labels, attacks = Counter(), Counter()
    rows = 0
    nonfinite = 0
    label_attack_conflicts = 0
    seen_hashes = set()
    duplicate_hashes = 0
    dedup_checked = 0
    samples = {0: [], 1: []}
    sample_counts = {0: 0, 1: 0}
    for chunk in iter_nf_csv_chunks(path, dataset, features, chunksize=chunksize):
        rows += len(chunk)
        values = chunk[features].to_numpy(dtype=float)
        nonfinite += int((~np.isfinite(values)).sum())
        labels.update({int(k): int(v) for k, v in chunk["Label"].value_counts().items()})
        if "Attack" in chunk:
            attacks.update({str(k): int(v) for k, v in chunk["Attack"].value_counts(dropna=False).items()})
            benign_name = chunk["Attack"].astype(str).str.lower().eq("benign")
            label_attack_conflicts += int((benign_name != chunk["Label"].eq(0)).sum())
        if dedup_checked < duplicate_check_limit:
            part = chunk.iloc[:duplicate_check_limit - dedup_checked]
            hashes = pd.util.hash_pandas_object(part[features], index=False).to_numpy(dtype=np.uint64)
            for value in hashes:
                number = int(value)
                duplicate_hashes += number in seen_hashes
                seen_hashes.add(number)
            dedup_checked += len(part)
        if smoke_train:
            for label in (0, 1):
                needed = sample_per_class - sample_counts[label]
                if needed > 0:
                    selected = chunk.loc[chunk["Label"] == label].head(needed)
                    if len(selected):
                        samples[label].append(selected[features].to_numpy(dtype=float))
                        sample_counts[label] += len(selected)
    digest, sha1 = _hashes(path)
    campaign_columns = [name for name in header if name.lower() in {"campaign", "campaign_id"}]
    time_columns = [name for name in header if name.lower() in {
        "timestamp", "start_time", "flow_start", "capture_time"}]
    split_columns = campaign_columns + time_columns
    report = {"dataset": dataset, "path": str(path), "size_bytes": path.stat().st_size,
              "sha256": digest, "sha1": sha1, "rows": rows, "feature_count": len(features),
              "feature_order": features, "csv_header_order_matches_manifest": header[:43] == features,
              "labels": dict(labels), "attack_categories": dict(attacks),
              "nonfinite_feature_values": nonfinite, "label_attack_conflicts": label_attack_conflicts,
              "duplicate_feature_hashes_in_checked_rows": duplicate_hashes,
              "dedup_checked_rows": dedup_checked, "split_metadata_columns": split_columns,
              "campaign_time_split_ready": bool(campaign_columns and time_columns),
              "performance_evaluation": "not_run_audit_only"}
    if smoke_train:
        if min(sample_counts.values()) < 2 or nonfinite:
            raise ValueError("real-data smoke training needs both classes and finite features")
        seed_everything(11)
        x = np.vstack(samples[0] + samples[1])
        y = np.r_[np.zeros(sample_counts[0], dtype=int), np.ones(sample_counts[1], dtype=int)]
        center = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale == 0] = 1
        x = (x - center) / scale
        registry = Registry(output / "registry.sqlite")
        smoke_results = []
        for name, model in (("b1", AffineMLP(len(features), 8, seed=11)),
                            ("b2", FunctionalNet(len(features), 8, 2, seed=11))):
            model.fit(x, y, epochs=1, batch_size=64, seed=11)
            logits = model.logits(x)
            checkpoint = output / f"{dataset}_{name}_smoke_seed11.npz"
            np.savez(checkpoint, **{k: v for k, v in vars(model).items() if isinstance(v, (np.ndarray, float))},
                     feature_center=center, feature_scale=scale)
            metrics = {"training_smoke_only": True, "train_rows": len(x),
                       "finite_logits": bool(np.isfinite(logits).all()),
                       "macs": model.macs, "params": model.params,
                       "detection_rate": None, "pr_auc": None, "fpr": None,
                       "reason": "no verified campaign/time split; no validation or test metrics"}
            record = {"run_id": f"{dataset}_{name}_real_smoke_seed11", "config": Config(
                dataset=dataset, model=name, width=8, functional_hidden=2, epochs=1,
                batch_size=64).as_dict(), "seed": 11, "dataset_hash": digest,
                "git_commit": _git_commit(), "metrics": metrics,
                "checkpoint_path": str(checkpoint), "status": "smoke_only"}
            registry.write(record)
            smoke_results.append({"run_id": record["run_id"], "finite_logits": metrics["finite_logits"],
                                  "checkpoint_path": str(checkpoint)})
        report["training_smoke"] = smoke_results
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset", choices=("nf_unsw_nb15_v2", "nf_cse_cic_ids2018_v2"), required=True)
    parser.add_argument("--features", default="configs/nf_v2_features.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-train", action="store_true")
    parser.add_argument("--chunksize", type=int, default=100000)
    args = parser.parse_args()
    features = Path(args.features).read_text().splitlines()
    report = audit(args.csv, args.dataset, features, args.output,
                   chunksize=args.chunksize, smoke_train=args.smoke_train)
    print(json.dumps({key: report[key] for key in ("dataset", "rows", "labels", "attack_categories",
                                                    "nonfinite_feature_values", "label_attack_conflicts",
                                                    "duplicate_feature_hashes_in_checked_rows",
                                                    "dedup_checked_rows", "campaign_time_split_ready",
                                                    "performance_evaluation")}, indent=2))
    if args.smoke_train:
        print(json.dumps(report["training_smoke"], indent=2))


if __name__ == "__main__":
    main()
