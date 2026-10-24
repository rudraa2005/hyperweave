"""Bounded real NF-v2 endpoint-group study and once-only sealed evaluation."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from hyperweave.agents.budget import BudgetLedger
from hyperweave.agents.experiment import ExperimentHarness
from hyperweave.baselines.nsga2 import NSGA2Search
from hyperweave.baselines.phase3 import run_condition
from hyperweave.config import Config, SEARCH_OPTIONS
from hyperweave.data.normalization import FlowNormalizer
from hyperweave.eval.metrics import score_binary, score_with_recalibration, threshold_at_fpr
from hyperweave.eval.robustness import robustness_report
from hyperweave.eval.sealed import SealedValidator
from hyperweave.kernel.runner import bench, compile_kernel, infer, write_inputs
from hyperweave.models import (AffineMLP, BakedModel, FunctionalNet, match_affine_width,
                               load_hypernet, bake, quantize, read_int8, write_int8)
from hyperweave.models.noise_envelope import NoiseEnvelope
from hyperweave.registry import Registry


SEEDS = (11, 23, 37)
PREFERENCES = ((.8, .1, .1), (.55, .25, .2), (.28, .12, .6), (.25, .1, .65))
TARGET_FPR = .001


def _commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unversioned"


def _load_public(split_dir, name):
    with np.load(Path(split_dir) / "public" / f"{name}.npz", allow_pickle=False) as data:
        return data["x"].copy(), data["y"].copy()


def _train_subset(x, y, per_class=2000):
    rng = np.random.default_rng(20261024)
    ids = np.concatenate([rng.choice(np.flatnonzero(y == label), min(per_class, int(np.sum(y == label))),
                                     replace=False) for label in (0, 1)])
    return x[ids], y[ids]


def _functional_int8_metrics(model, train_x, b_x, b_y, threshold, artifacts, run_id, binary,
                             dataset_hash):
    n = len(model.out)
    baked = BakedModel(model.w1, model.b1, model.w2, model.b2, model.out,
                       np.ones(n, bool), np.arange(n), np.array([1., 0., 0.]), (0, 0))
    int8 = quantize(baked, train_x[:1024])
    int8.bias = int(round(model.bias / (int8.scales[4] * int8.scales[5])))
    export = artifacts / f"{run_id}.hwi"
    write_int8(int8, export, {"run_id": run_id, "dataset_hash": dataset_hash,
                             "float_bias_folded_into_int8": True})
    inputs = int8.quantize_inputs(b_x)
    input_path = artifacts / f"{run_id}.i8"
    write_inputs(input_path, inputs)
    if not np.array_equal(infer(binary, export, input_path, len(b_x)), int8.logits_int(inputs)):
        raise ValueError("fixed functional model disagrees with C int8 kernel")
    int8_metrics = score_binary(b_y, int8.logits(b_x), threshold)
    timing = bench(binary, export, input_path)
    return {"int8": int8_metrics, "int8_detection_gap": int8_metrics["detection_rate"] -
            float(np.mean(model.logits(b_x)[b_y == 1] > threshold)),
            "int8_model_bytes": export.stat().st_size, "latency": timing,
            "kernel_bit_exact": True, "export_path": str(export)}


def _fixed_model_records(registry, artifacts, train, val, b, manifest_hash, *, features,
                         implementation_hash):
    train_x, train_y = train
    val_x, val_y = val
    b_x, b_y = b
    records = []
    binary = compile_kernel(artifacts / "fixed_infer")
    for seed in SEEDS:
        for width in (4, 8, 16):
            for name in ("b1", "b2"):
                model = (AffineMLP(len(features), match_affine_width(len(features), width, 2), seed)
                         if name == "b1" else FunctionalNet(len(features), width, 2, seed))
                started = time.perf_counter()
                model.fit(train_x, train_y, epochs=1, batch_size=256, seed=seed)
                seconds = time.perf_counter() - started
                threshold = threshold_at_fpr(model.logits(val_x)[val_y == 0], TARGET_FPR)
                metrics = score_binary(b_y, model.logits(b_x), threshold)
                metrics.update(macs=model.macs, params=model.params, threshold=threshold,
                               threshold_source="a_val", training_seconds=seconds,
                               traffic_processed=len(b_y),
                               b_recalibrated=score_with_recalibration(b_y, model.logits(b_x), threshold,
                                                                      TARGET_FPR)["recalibrated"])
                run_id = f"group_{name}_w{width}_seed{seed}"
                if name == "b2":
                    metrics.update(_functional_int8_metrics(model, train_x, b_x, b_y, threshold,
                                                             artifacts, run_id, binary, manifest_hash))
                checkpoint = artifacts / f"{run_id}.npz"
                np.savez(checkpoint, **{key: value for key, value in vars(model).items()
                                        if isinstance(value, (np.ndarray, float))})
                metrics["model_size_bytes"] = checkpoint.stat().st_size
                record = {"run_id": run_id, "config": Config(model=name, seed=seed,
                           width=(model.w1.shape[1] if name == "b1" else width),
                           functional_hidden=2, epochs=1,
                           shift_columns=SEARCH_OPTIONS["shift_columns"][1]).as_dict(),
                           "seed": seed, "dataset_hash": manifest_hash, "git_commit": _commit(),
                           "metrics": metrics, "checkpoint_path": str(checkpoint),
                           "implementation_hash": implementation_hash,
                           "status": "completed", "dataset_version": "endpoint_pair_group_disjoint_v1"}
                registry.write(record)
                records.append(record)
        for i, preference in enumerate(PREFERENCES):
            width = max(1, round(16 * (1 - .8 * preference[2])))
            model = FunctionalNet(len(features), width, 2, seed)
            started = time.perf_counter()
            model.fit(train_x, train_y, epochs=1, batch_size=256, seed=seed)
            seconds = time.perf_counter() - started
            threshold = threshold_at_fpr(model.logits(val_x)[val_y == 0], TARGET_FPR)
            metrics = score_binary(b_y, model.logits(b_x), threshold)
            metrics.update(macs=model.macs, params=model.params, threshold=threshold,
                           threshold_source="a_val", training_seconds=seconds,
                           traffic_processed=len(b_y))
            run_id = f"group_b2b_lambda{i}_seed{seed}"
            metrics.update(_functional_int8_metrics(model, train_x, b_x, b_y, threshold,
                                                     artifacts, run_id, binary, manifest_hash))
            checkpoint = artifacts / f"{run_id}.npz"
            np.savez(checkpoint, **{key: value for key, value in vars(model).items()
                                    if isinstance(value, (np.ndarray, float))})
            metrics["model_size_bytes"] = checkpoint.stat().st_size
            record = {"run_id": run_id, "config": Config(model="b2b", seed=seed, width=width,
                       functional_hidden=2, epochs=1, preference=preference,
                       shift_columns=SEARCH_OPTIONS["shift_columns"][1]).as_dict(),
                       "seed": seed, "dataset_hash": manifest_hash, "git_commit": _commit(),
                       "metrics": metrics, "checkpoint_path": str(checkpoint),
                       "implementation_hash": implementation_hash,
                       "status": "completed", "dataset_version": "endpoint_pair_group_disjoint_v1"}
            registry.write(record)
            records.append(record)
    return records


def _run_b3(registry, experiment):
    ledger = BudgetLedger(registry, limits={"b3": 12})
    rows = []
    for seed in SEEDS:
        for i, preference in enumerate(PREFERENCES):
            run_id = f"b3_seed{seed}_lambda{i}"
            proposal_id = f"scheduled_{run_id}"
            config = Config(model="b3", seed=seed, width=8, functional_hidden=2, epochs=1,
                            batch_size=256, target_fpr=TARGET_FPR, preference=preference,
                            shift_columns=SEARCH_OPTIONS["shift_columns"][1])
            if not ledger.reserve("b3", proposal_id, 1, approved=True, actor="predeclared_research_protocol"):
                raise RuntimeError("B3 budget exhausted before scheduled run")
            ledger.start(proposal_id)
            try:
                record = experiment(run_id, config, proposal={"method": "fixed_schedule"})
            except RuntimeError:
                record = registry.read(run_id)
                ledger.finish(proposal_id, success=False)
            else:
                ledger.finish(proposal_id, success=True)
            rows.append(record)
    return rows, ledger.remaining("b3")


def _selected(registry, condition):
    rows = [r for r in registry.all_runs() if r["config"]["model"] == condition
            and r.get("status") == "completed"]
    if not rows:
        return None
    def rank(record):
        metrics = record["metrics"]
        deployed = metrics.get("int8", metrics)
        return (-deployed["detection_rate"], deployed["fpr"], metrics["macs"], record["run_id"])
    return min(rows, key=rank)


def final_sealed_evaluation(registry, split_dir, output, normalizer, envelope=None):
    """Open sealed partition exactly once after selection; write one event."""
    if registry.events("sealed_open_attempt") or registry.events("sealed_final"):
        raise RuntimeError("sealed final evaluation already performed")
    selected = {condition: _selected(registry, condition) for condition in
                ("b1", "b2", "b2b", "b3", "b4", "b5", "b5b")}
    registry.append_event("sealed_open_attempt", {"split": "endpoint_pair_group_disjoint_v1",
                                                  "selected_run_ids": {key: None if row is None else row["run_id"]
                                                                       for key, row in selected.items()}})
    with np.load(Path(split_dir) / "private" / "sealed.npz", allow_pickle=False) as sealed:
        x = normalizer.transform(sealed["x"])
        y = sealed["y"].copy()
    results = {}
    if envelope is not None:
        results["autoencoder"] = {"threshold_source": "a_val_benign",
                                  "device": envelope.device,
                                  "offline_only": True,
                                  "original": score_binary(y, envelope.reconstruction_error(x), envelope.threshold)}
    for condition, row in selected.items():
        if row is None:
            results[condition] = None
            continue
        cfg = row["config"]
        if condition in {"b1", "b2", "b2b"}:
            with np.load(row["checkpoint_path"], allow_pickle=False) as data:
                if condition == "b1":
                    model = AffineMLP(len(normalizer.features), len(data["b1"]), row["seed"])
                    for key in ("w1", "b1", "w2", "b2"):
                        setattr(model, key, data[key].copy() if key != "b2" else float(data[key]))
                else:
                    model = FunctionalNet(len(normalizer.features), len(data["out"]),
                                          data["w1"].shape[1], row["seed"])
                    for key in ("w1", "b1", "w2", "b2", "out", "bias"):
                        setattr(model, key, data[key].copy() if key != "bias" else float(data[key]))
            scores = model.logits(x)
        else:
            checkpoint = Path(output) / row["checkpoint_path"]
            model = load_hypernet(checkpoint)
            scores = bake(model, cfg["preference"], cfg["context"]).logits(x)
        threshold = row["metrics"]["threshold"]
        metrics = score_with_recalibration(y, scores, threshold, TARGET_FPR)
        result = {"run_id": row["run_id"], "threshold_source": "a_val",
                  "original": metrics["original"], "recalibrated_diagnostic": metrics["recalibrated"],
                  "traffic_processed": len(y)}
        if condition in {"b2", "b2b", "b3", "b4", "b5", "b5b"}:
            artifact = Path(output) / "artifacts" / f"{row['run_id']}.hwi"
            int8 = read_int8(artifact)
            result["int8_original"] = score_binary(y, int8.logits(x), threshold)
        results[condition] = result
    registry.append_event("sealed_final", {"split": "endpoint_pair_group_disjoint_v1",
                                           "results": results, "rows": int(len(y)),
                                           "attack_rows": int(np.sum(y == 1))})
    return results


def run(split_dir, output):
    split_dir, output = Path(split_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh Phase 4 output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    if manifest["kind"] != "endpoint_pair_group_disjoint_v1":
        raise ValueError("unexpected split protocol")
    features = manifest["features"]
    train_raw, train_y = _train_subset(*_load_public(split_dir, "a_train"))
    val_raw, val_y = _load_public(split_dir, "a_val")
    b_raw, b_y = _load_public(split_dir, "b")
    normalizer = FlowNormalizer(features).fit(train_raw)
    normalizer.save(output / "normalizer.npz")
    train = normalizer.transform(train_raw), train_y
    val = normalizer.transform(val_raw), val_y
    b = normalizer.transform(b_raw), b_y
    source_hash = hashlib.sha256((json.dumps(manifest, sort_keys=True) +
                                  "|train_subsample_seed=20261024|per_class=2000").encode()).hexdigest()
    repo = Path(__file__).resolve().parents[2]
    code_files = sorted((repo / "hyperweave").rglob("*.py")) + [repo / "kernel" / "infer.c",
                  repo / "configs" / "nf_v2_features.txt", repo / "docs" / "group_split_protocol.md"]
    code_digest = hashlib.sha256()
    for path in code_files:
        code_digest.update(str(path.relative_to(repo)).encode())
        code_digest.update(path.read_bytes())
    implementation_hash = code_digest.hexdigest()
    registry = Registry(output / "registry.sqlite")
    artifacts = output / "artifacts"
    artifacts.mkdir()
    _fixed_model_records(registry, artifacts, train, val, b, source_hash, features=features,
                         implementation_hash=implementation_hash)
    experiment = ExperimentHarness(registry, artifacts, train, val, b,
                                   dataset_version="endpoint_pair_group_disjoint_v1",
                                   normalizer=normalizer, dataset_hash_override=source_hash,
                                   implementation_hash=implementation_hash)
    envelope = NoiseEnvelope(len(features), seed=11).fit(train[0][train_y == 0], epochs=10, batch_size=256)
    envelope.calibrate(val[0][val_y == 0], TARGET_FPR)
    envelope.save(output / "noise_envelope.npz")
    ae_metrics = score_binary(b_y, envelope.reconstruction_error(b[0]), envelope.threshold)
    registry.append_event("noise_envelope", {"artifact": str(output / "noise_envelope.npz"),
                       "device": envelope.device, "threshold_source": "a_val_benign",
                       "converged": envelope.converged,
                       "training_rows": int(np.sum(train_y == 0)), "b_metrics": ae_metrics,
                       "deployment_path": "offline_only"})
    b3, b3_remaining = _run_b3(registry, experiment)
    base_b4 = Config(model="b4", seed=11, epochs=1, batch_size=256,
                     target_fpr=TARGET_FPR, shift_columns=SEARCH_OPTIONS["shift_columns"][1])
    b4_search = NSGA2Search(registry, experiment, base_b4, objective_mode="debug_proxy")
    b4 = b4_search.run()
    baseline_record = _selected(registry, "b2")
    baseline = baseline_record["metrics"]["int8"]["detection_rate"]
    validator = SealedValidator(output / "private" / "b_validator.sqlite",
                                lambda rid: max(-1., min(1., registry.read(rid)["metrics"]["int8"]["detection_rate"] - baseline)),
                                query_limit=10)
    b5, b5_remaining = run_condition(registry, experiment, validator, "b5", target_fpr=TARGET_FPR,
                                     shift_columns=SEARCH_OPTIONS["shift_columns"][1])
    b5b, b5b_remaining = run_condition(registry, experiment, validator, "b5b", target_fpr=TARGET_FPR,
                                       shift_columns=SEARCH_OPTIONS["shift_columns"][1])
    if any(v != 0 for v in (b3_remaining, b4_search.budget.remaining("b4"), b5_remaining, b5b_remaining)):
        raise RuntimeError("B3/B4/B5/B5b budgets were not equally consumed")
    for condition in ("b3", "b4", "b5", "b5b"):
        chosen = _selected(registry, condition)
        if chosen is None:
            continue
        detector = read_int8(artifacts / f"{chosen['run_id']}.hwi")
        robustness = robustness_report(detector, features, train_raw, train_y, b_raw, b_y,
                                       normalizer.center, normalizer.scale,
                                       chosen["metrics"]["threshold"], seed=11,
                                       transform=normalizer.transform)
        registry.append_event("robustness_selected", {"condition": condition,
                              "run_id": chosen["run_id"], "metrics": robustness,
                              "backend": "int8_python_reference_bit_exact_with_C_kernel"})
    sealed = final_sealed_evaluation(registry, split_dir, output, normalizer, envelope)
    from hyperweave.eval.phase4_report import build_report
    report = build_report(registry, manifest, sealed, output)
    report["implementation_hash"] = implementation_hash
    report["normalizer_sha256"] = hashlib.sha256((output / "normalizer.npz").read_bytes()).hexdigest()
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args.split, args.output)
    print(json.dumps({k: report[k] for k in ("kind", "budget", "selected_runs", "claim_verdicts")}, indent=2))


if __name__ == "__main__":
    main()
