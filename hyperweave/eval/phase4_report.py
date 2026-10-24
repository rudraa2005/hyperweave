"""Registry-derived Phase 4 outcome report; no hand-entered scores."""
import numpy as np

from hyperweave.eval.hypervolume import hypervolume, nondominated


CONDITIONS = ("b1", "b2", "b2b", "b3", "b4", "b5", "b5b")


def bootstrap_seed_interval(seed_values, *, resamples=1000, seed=20261024):
    values = np.asarray(seed_values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or resamples < 1000:
        raise ValueError("need finite seed-level values and at least 1000 resamples")
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(len(values), size=(resamples, len(values)))].mean(axis=1)
    return [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]


def build_report(registry, manifest, sealed, output):
    rows = registry.all_runs()
    if len(registry.events("sealed_final")) != 1:
        raise ValueError("final sealed evaluation must occur exactly once")
    if any(row["dataset_hash"] != rows[0]["dataset_hash"] for row in rows):
        raise ValueError("dataset hash differs across runs")
    code_hashes = {row.get("implementation_hash") for row in rows}
    if len(code_hashes) != 1:
        raise ValueError("implementation hash differs across runs")
    scoreboard, selected = {}, {}
    for condition in CONDITIONS:
        group = [row for row in rows if row["config"]["model"] == condition]
        completed = [row for row in group if row.get("status") == "completed"]
        per_seed = []
        for seed in (11, 23, 37):
            subset = [row["metrics"].get("int8", row["metrics"])["detection_rate"]
                      for row in completed if row["seed"] == seed]
            if subset:
                per_seed.append(float(np.mean(subset)))
        points = [(1 - row["metrics"].get("int8", row["metrics"])["detection_rate"],
                   row["metrics"]["macs"] / 100000)
                  for row in completed]
        frontier = nondominated(points).tolist() if points else []
        measured_latency = [row["metrics"].get("latency", {}).get("median_ns_per_flow")
                            for row in completed]
        measured_latency = [value for value in measured_latency if value is not None]
        int8_bytes = [row["metrics"].get("int8_model_bytes") for row in completed]
        int8_bytes = [value for value in int8_bytes if value is not None]
        quant_gaps = [row["metrics"].get("int8_detection_gap") for row in completed]
        quant_gaps = [value for value in quant_gaps if value is not None]
        scoreboard[condition] = {
            "attempts": len(group), "completed": len(completed), "failed": len(group) - len(completed),
            "seeds": sorted(set(row["seed"] for row in group)),
            "mean_b_detection_rate": float(np.mean([r["metrics"]["detection_rate"] for r in completed])) if completed else None,
            "mean_b_fpr": float(np.mean([r["metrics"]["fpr"] for r in completed])) if completed else None,
            "mean_b_pr_auc": float(np.mean([r["metrics"]["pr_auc"] for r in completed])) if completed else None,
            "mean_b_deployed_detection_rate": float(np.mean([r["metrics"].get("int8", r["metrics"])["detection_rate"]
                                                        for r in completed])) if completed else None,
            "mean_b_deployed_fpr": float(np.mean([r["metrics"].get("int8", r["metrics"])["fpr"]
                                         for r in completed])) if completed else None,
            "mean_b_deployed_pr_auc": float(np.mean([r["metrics"].get("int8", r["metrics"])["pr_auc"]
                                            for r in completed])) if completed else None,
            "mean_macs": float(np.mean([r["metrics"]["macs"] for r in completed])) if completed else None,
            "mean_int8_model_bytes": float(np.mean(int8_bytes)) if int8_bytes else None,
            "mean_unpinned_kernel_ns_per_flow": float(np.mean(measured_latency)) if measured_latency else None,
            "mean_int8_detection_gap": float(np.mean(quant_gaps)) if quant_gaps else None,
            "seed_bootstrap_detection_95": bootstrap_seed_interval(per_seed) if len(per_seed) >= 2 else None,
            "b_frontier_2d": frontier,
            "b_hv_2d_debug_proxy": hypervolume(points, (1, 1)) if points else None,
            "research_hv_3d": None,
            "sealed": sealed[condition],
        }
        selected[condition] = None if sealed[condition] is None else sealed[condition]["run_id"]
    spent = {condition: sum(event["payload"]["condition"] == condition
                            for event in registry.events("budget_reservation"))
             for condition in ("b3", "b4", "b5", "b5b")}
    if set(spent.values()) != {12}:
        raise ValueError("B3/B4/B5/B5b did not spend equal 12-run budgets")
    if any(scoreboard[name]["seeds"] != [11, 23, 37] for name in ("b3", "b4", "b5", "b5b")):
        raise ValueError("search conditions did not share seeds")
    by_id = {row["run_id"]: row for row in rows}
    matched_mac = []
    for seed in (11, 23, 37):
        for width in (4, 8, 16):
            affine = by_id[f"group_b1_w{width}_seed{seed}"]
            functional = by_id[f"group_b2_w{width}_seed{seed}"]
            a, b = affine["metrics"], functional["metrics"]
            matched_mac.append({"seed": seed, "functional_width": width,
                                "b1_macs": a["macs"], "b2_macs": b["macs"],
                                "mac_relative_gap": abs(a["macs"] - b["macs"]) / b["macs"],
                                "b1_b_detection": a["detection_rate"],
                                "b2_b_detection": b["detection_rate"],
                                "b2_int8_b_detection": b.get("int8", b)["detection_rate"]})
    if any(pair["mac_relative_gap"] > .02 for pair in matched_mac):
        raise ValueError("C1 baselines not matched within 2% MACs")
    training_time = {}
    for name in CONDITIONS:
        completed = [row for row in rows if row["config"]["model"] == name and row.get("status") == "completed"]
        training_time[name] = (sum(row["metrics"]["training_seconds"] for row in completed)
                               if completed and all("training_seconds" in row["metrics"] for row in completed)
                               else None)
    checkpoint_bytes = {}
    for name in CONDITIONS:
        sizes = [row["metrics"].get("model_size_bytes") for row in rows
                 if row["config"]["model"] == name and row.get("status") == "completed"]
        checkpoint_bytes[name] = float(np.mean(sizes)) if sizes and all(v is not None for v in sizes) else None
    proxy_hv_per_training_second = {
        name: (scoreboard[name]["b_hv_2d_debug_proxy"] / training_time[name]
               if training_time[name] and scoreboard[name]["b_hv_2d_debug_proxy"] is not None else None)
        for name in ("b4", "b5", "b5b")}
    return {
        "kind": "real_nf_v2_endpoint_group_alternate_study",
        "protocol": "docs/group_split_protocol.md",
        "dataset_hash": rows[0]["dataset_hash"],
        "source_sha256": manifest["source_sha256"],
        "partitions": manifest["partitions"],
        "target_fpr": .001,
        "a_val_empirical_fpr_resolution": 1 / manifest["partitions"]["a_val"]["benign"],
        "budget": spent,
        "selected_runs": selected,
        "scoreboard": scoreboard,
        "analysis": {
            "c1_matched_mac_pairs": matched_mac,
            "c2_b3_vs_b2b_2d_proxy_hv": {name: scoreboard[name]["b_hv_2d_debug_proxy"] for name in ("b3", "b2b")},
            "c3_b4_b5_b5b_2d_proxy_hv": {name: scoreboard[name]["b_hv_2d_debug_proxy"] for name in ("b4", "b5", "b5b")},
            "training_seconds_observed": training_time,
            "mean_checkpoint_bytes": checkpoint_bytes,
            "c3_2d_proxy_hv_per_training_second": proxy_hv_per_training_second,
            "c4_autopilot": "not_evaluable_without_temporal_order_and_equal_detection_pair",
        },
        "offline_autoencoder": {"b": registry.events("noise_envelope")[-1]["payload"] if registry.events("noise_envelope") else None,
                                "sealed": sealed.get("autoencoder")},
        "robustness_selected": {event["payload"]["condition"]: event["payload"]
                                for event in registry.events("robustness_selected")},
        "claim_verdicts": {f"C{i}": "inconclusive_missing_preregistered_threshold" for i in range(1, 5)},
        "agent_execution": "scripted_offline_replay",
        "approval_actor": "scripted_human_approval",
        "validator": "coarse_B_slice_only; sealed_opened_once_after_selection",
        "latency_note": "Unpinned debug timing only; 3D research hypervolume withheld",
        "sealed_opened_once": True,
        "registry_path": str(output / "registry.sqlite"),
    }
