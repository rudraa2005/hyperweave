"""Checkpoint -> hypernetwork -> baked int8 comparison on one flow batch."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from hyperweave.config import Config
from hyperweave.data import deduplicate, load_nf_csv, split_by_campaign, validate_chronological, validate_disjoint
from hyperweave.data.flows import dataset_hash
from hyperweave.data.shifts import apply_shift
from hyperweave.eval import hypervolume, normalize_objectives, score_binary, threshold_at_fpr
from hyperweave.kernel.runner import bench, compile_kernel, infer, write_inputs
from hyperweave.models import (BakedModel, FunctionalNet, HyperNet, NeuronSpec, bake,
                               quantize, save_hypernet, write_int8)
from hyperweave.registry import Registry
from hyperweave.seed import seed_everything


PREFERENCES = ((.8, .1, .1), (.28, .12, .6), (.25, .10, .65))


def _commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def _dirty():
    result = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=False)
    return result.returncode != 0 or bool(result.stdout.strip())


def _fixed_baked(checkpoint):
    with np.load(checkpoint) as data:
        w1, b1, w2, b2, out = (data[key].copy() for key in ("w1", "b1", "w2", "b2", "out"))
        center, scale = data["feature_center"].copy(), data["feature_scale"].copy()
    n = len(out)
    return BakedModel(w1, b1, w2, b2, out, np.ones(n, bool), np.arange(n),
                      np.array([1., 0., 0.]), (0, 0)), center, scale


def _measure(baked, train_x, val_x, val_y, b_x, b_y, target_fpr, binary, path, manifest):
    int8 = quantize(baked, train_x[:min(1024, len(train_x))])
    sidecar = write_int8(int8, path, manifest)
    input_path = path.with_suffix(".i8")
    write_inputs(input_path, int8.quantize_inputs(b_x))
    expected = int8.logits_int(int8.quantize_inputs(b_x))
    actual = infer(binary, path, input_path, len(b_x))
    if not np.array_equal(actual, expected):
        raise ValueError("C kernel disagrees with Python int8 reference")
    threshold = threshold_at_fpr(baked.logits(val_x)[val_y == 0], target_fpr)
    float_metrics = score_binary(b_y, baked.logits(b_x), threshold)
    int8_metrics = score_binary(b_y, int8.logits(b_x), threshold)
    timing = bench(binary, path, input_path)
    return {"float": float_metrics, "int8": int8_metrics, "threshold": threshold,
            "threshold_source": "a_val", "int8_detection_gap": int8_metrics["detection_rate"] - float_metrics["detection_rate"],
            "macs": baked.macs, "params": baked.params, "float_model_bytes": baked.params * 8,
            "int8_model_bytes": path.stat().st_size, "latency": timing,
            "gate_mask": baked.gate_mask.tolist(), "artifact": str(path), "artifact_manifest": str(sidecar)}


def run(csv_path, features, dataset, campaign_col, time_col, phase1_registry, fixed_run_id, output,
        *, epochs=3, seed=11, preferences=PREFERENCES, context=(0, 0), train_b2b=True,
        shift_columns=(0,), loss_mode="linear", concentration=(1., 1., 1.)):
    seed_everything(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Registry(phase1_registry).read(fixed_run_id)
    if source is None or source["config"]["model"] != "b2":
        raise ValueError("fixed_run_id must identify a Phase 1 B2 checkpoint")
    if source["seed"] != seed or source["dataset_hash"] != dataset_hash(csv_path):
        raise ValueError("seed or dataset hash differs from Phase 1 checkpoint")
    fixed, center, scale = _fixed_baked(source["checkpoint_path"])
    if fixed.w1.shape[-1] != len(features):
        raise ValueError("feature manifest differs from checkpoint input dimension")
    df = deduplicate(load_nf_csv(csv_path, dataset, features, campaign_col=campaign_col, time_col=time_col), features)
    splits = split_by_campaign(df, campaign_col, time_col)
    validate_disjoint(splits, features, campaign_col)
    validate_chronological(splits, time_col)
    def arrays(name):
        frame = splits[name]
        return (frame[features].to_numpy(dtype=float) - center) / scale, frame["Label"].to_numpy(dtype=int)
    train_x, train_y = arrays("a_train")
    val_x, val_y = arrays("a_val")
    b_x, b_y = arrays("b")
    target_fpr = source["config"]["target_fpr"]
    if (val_y == 0).sum() < int(np.ceil(1 / target_fpr)):
        raise ValueError("A-val insufficient for FPR calibration")
    registry = Registry(output / "registry.sqlite")
    fixed_comparison_id = f"{dataset}_fixed_b2_seed{seed}"
    run_ids = [fixed_comparison_id] + [f"{dataset}_b3_seed{seed}_lambda{i}" for i in range(len(preferences))]
    if train_b2b:
        run_ids += [f"{dataset}_b2b_seed{seed}_lambda{i}" for i in range(len(preferences))]
    planned_artifacts = [output / f"{run_id}.hwi" for run_id in run_ids]
    planned_artifacts += [output / f"{dataset}_hypernet_seed{seed}.npz", output / "comparison.json"]
    if train_b2b:
        planned_artifacts += [output / f"{run_id}.npz" for run_id in run_ids if "_b2b_" in run_id]
    if any(registry.read(run_id) for run_id in run_ids) or any(path.exists() for path in planned_artifacts):
        raise FileExistsError("Phase 2 run ID or artifact already exists")
    binary = compile_kernel(output / "infer")
    git_commit = _commit()
    git_dirty = _dirty()
    ds_hash = source["dataset_hash"]
    fixed_metrics = _measure(fixed, train_x, val_x, val_y, b_x, b_y, target_fpr, binary,
                             output / f"{fixed_comparison_id}.hwi", {"run_id": fixed_comparison_id,
                                "source_run_id": fixed_run_id, "dataset_hash": ds_hash,
                                "seed": seed, "git_commit": git_commit, "git_dirty": git_dirty, "kind": "fixed_b2"})
    fixed_record = {"run_id": fixed_comparison_id, "config": dict(source["config"], model="b2"),
                    "seed": seed, "dataset_hash": ds_hash, "git_commit": git_commit,
                    "git_dirty": git_dirty, "metrics": fixed_metrics,
                    "checkpoint_path": source["checkpoint_path"], "source_run_id": fixed_run_id}
    registry.write(fixed_record)
    model = HyperNet(NeuronSpec(len(features), fixed.w1.shape[0], fixed.w1.shape[1]), seed=seed)
    start = time.perf_counter()
    def shift_transform(batch, family, columns, shift_seed):
        raw = batch * scale + center
        shifted = apply_shift(raw, family, columns, 1, seed=shift_seed)
        return (shifted - center) / scale
    history = model.fit(train_x, train_y, epochs=epochs, seed=seed, mode=loss_mode,
                        concentration=concentration, shift_columns=shift_columns,
                        shift_transform=shift_transform)
    training_seconds = time.perf_counter() - start
    hyper_checkpoint = output / f"{dataset}_hypernet_seed{seed}.npz"
    save_hypernet(model, hyper_checkpoint)
    records = []
    for i, preference in enumerate(preferences):
        run_id = f"{dataset}_b3_seed{seed}_lambda{i}"
        baked = bake(model, preference, context)
        path = output / f"{run_id}.hwi"
        config = Config(**dict(source["config"], model="b3", epochs=epochs,
                               preference=list(preference), context=list(context),
                               shift_columns=list(shift_columns), loss_mode=loss_mode,
                               dirichlet_concentration=list(concentration))).as_dict()
        manifest = {"run_id": run_id, "config": config, "seed": seed, "dataset_hash": ds_hash,
                    "git_commit": git_commit, "git_dirty": git_dirty,
                    "hypernet_checkpoint": str(hyper_checkpoint),
                    "source_run_id": fixed_run_id}
        metrics = _measure(baked, train_x, val_x, val_y, b_x, b_y, target_fpr, binary, path, manifest)
        metrics["training_seconds"] = training_seconds
        record = {"run_id": run_id, "config": config, "seed": seed, "dataset_hash": ds_hash,
                  "git_commit": git_commit, "git_dirty": git_dirty,
                  "metrics": metrics, "checkpoint_path": str(hyper_checkpoint),
                  "export_path": str(path), "source_run_id": fixed_run_id}
        registry.write(record)
        records.append(record)
    b2b_records = []
    if train_b2b:
        for i, preference in enumerate(preferences):
            width = max(1, round(fixed.w1.shape[0] * (1 - .8 * preference[2])))
            competitor = FunctionalNet(len(features), width, fixed.w1.shape[1], seed=seed)
            start = time.perf_counter()
            competitor.fit(train_x, train_y, epochs=epochs, seed=seed)
            seconds = time.perf_counter() - start
            fixed_competitor = BakedModel(competitor.w1, competitor.b1, competitor.w2, competitor.b2,
                                        competitor.out, np.ones(width, bool), np.arange(width),
                                        np.asarray(preference), tuple(context))
            run_id = f"{dataset}_b2b_seed{seed}_lambda{i}"
            checkpoint = output / f"{run_id}.npz"
            np.savez(checkpoint, **{k: v for k, v in vars(competitor).items() if isinstance(v, (np.ndarray, float))})
            config = Config(**dict(source["config"], model="b2b", width=width,
                                   preference=list(preference), context=list(context), epochs=epochs)).as_dict()
            metrics = _measure(fixed_competitor, train_x, val_x, val_y, b_x, b_y, target_fpr, binary,
                               output / f"{run_id}.hwi", {"run_id": run_id, "config": config,
                                                            "seed": seed, "dataset_hash": ds_hash,
                                                            "git_commit": git_commit, "checkpoint": str(checkpoint)})
            metrics["training_seconds"] = seconds
            record = {"run_id": run_id, "config": config, "seed": seed, "dataset_hash": ds_hash,
                      "git_commit": git_commit, "git_dirty": git_dirty,
                      "metrics": metrics, "checkpoint_path": str(checkpoint)}
            registry.write(record)
            b2b_records.append(record)
    def hv(rows):
        points = [normalize_objectives(r["metrics"]["float"]["detection_rate"], r["metrics"]["macs"],
                                       r["metrics"]["latency"]["median_ns_per_flow"]) for r in rows]
        return hypervolume(points, (1, 1, 1))
    timing_comparable = bool(fixed_metrics["latency"]["pinned"] and all(
        r["metrics"]["latency"]["pinned"] for r in records + b2b_records))
    report = {"kind": "phase2_debug_comparison", "data_source": str(csv_path),
              "fixed_run_id": fixed_comparison_id, "source_run_id": fixed_run_id,
              "fixed": fixed_metrics, "b3_run_ids": [r["run_id"] for r in records],
              "b2b_run_ids": [r["run_id"] for r in b2b_records],
              "b3_hv": hv(records) if timing_comparable else None,
              "b2b_hv": hv(b2b_records) if timing_comparable and b2b_records else None,
              "timing_comparable": timing_comparable,
              "timing_note": "Pinned-core research timing" if timing_comparable else "Unpinned debug timing; HV withheld"}
    (output / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser()
    for name in ("csv", "features", "dataset", "campaign-col", "time-col", "phase1-registry", "fixed-run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", default="runs/phase2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--preferences", help="JSON list of three-objective simplex vectors")
    parser.add_argument("--context", nargs=2, type=int, default=(0, 0))
    parser.add_argument("--shift-columns", nargs="+", type=int, default=(0,),
                        help="Indices of mutable count/timing features used for training shifts")
    parser.add_argument("--loss-mode", choices=("linear", "phn_hvi"), default="linear")
    parser.add_argument("--dirichlet-concentration", nargs=3, type=float, default=(1., 1., 1.))
    args = parser.parse_args()
    preferences = json.loads(args.preferences) if args.preferences else PREFERENCES
    report = run(args.csv, Path(args.features).read_text().splitlines(), args.dataset,
                 args.campaign_col, args.time_col, args.phase1_registry, args.fixed_run_id,
                 args.output, epochs=args.epochs, seed=args.seed,
                 preferences=preferences, context=tuple(args.context),
                 shift_columns=tuple(args.shift_columns), loss_mode=args.loss_mode,
                 concentration=tuple(args.dirichlet_concentration))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
