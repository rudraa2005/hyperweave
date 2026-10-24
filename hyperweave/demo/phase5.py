"""Local-only Phase 5 dashboard backed by the real group-split registry."""
import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from hyperweave.agents.budget import HumanDecision
from hyperweave.agents.cycle import AgentCycle, ScriptedClient
from hyperweave.baselines.phase3 import scripted_responses
from hyperweave.config import Config, SEARCH_OPTIONS
from hyperweave.data.evasion import constrained_nf_evasion, nf_mutable_mask
from hyperweave.data.group_split import file_sha256, study_dataset_hash
from hyperweave.data.normalization import FlowNormalizer
from hyperweave.demo.live_capture import LiveCapture
from hyperweave.eval.metrics import score_binary, threshold_at_fpr
from hyperweave.kernel.runner import TIMING_BATCH_ROWS, bench, compile_kernel, infer, write_inputs
from hyperweave.models import bake, load_hypernet, quantize, write_int8
from hyperweave.registry import Registry


STATIC = Path(__file__).with_name("static")


class DemoState:
    def __init__(self, study, split):
        self.study, self.split = Path(study), Path(split)
        self.report = json.loads((self.study / "report.json").read_text())
        if self.report["kind"] != "real_nf_v2_endpoint_group_alternate_study":
            raise ValueError("demo needs the real endpoint-group study")
        manifest = json.loads((self.split / "manifest.json").read_text())
        if manifest.get("kind") != "endpoint_pair_group_disjoint_v1" or \
                study_dataset_hash(manifest) != self.report["dataset_hash"]:
            raise ValueError("demo split does not match the study report")
        for name in ("a_train", "a_val", "b"):
            expected_hash = manifest.get("partitions", {}).get(name, {}).get("sha256")
            if expected_hash is not None and file_sha256(self.split / "public" / f"{name}.npz") != expected_hash:
                raise ValueError(f"demo public partition hash mismatch: {name}")
        self.registry = Registry(self.study / "registry.sqlite")
        self.normalizer = FlowNormalizer.load(self.study / "normalizer.npz")
        self.train_raw = self._public("a_train")
        self.val_raw = self._public("a_val")
        self.b_raw = self._public("b")
        self.train_x = self.normalizer.transform(self.train_raw[0])
        self.val_x = self.normalizer.transform(self.val_raw[0])
        self.b_x = self.normalizer.transform(self.b_raw[0])
        selected = self.report["selected_runs"].get("b3")
        if not selected:
            raise ValueError("no completed B3 candidate available for slider")
        self.source = self.registry.read(selected)
        if self.source is None or self.source["dataset_hash"] != self.report["dataset_hash"]:
            raise ValueError("selected B3 run does not match the study report")
        self.hypernet = load_hypernet(self.study / self.source["checkpoint_path"])
        self.binary = compile_kernel(self.study / "demo_infer")
        self.generated_dir = self.study / "demo_generated"
        self.generated_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.current = None
        self.demo_registry = Registry(self.study / "demo_actions.sqlite")
        self.action_number = len(self.demo_registry.events("cycle_outcome"))
        self.generation_number = len(self.demo_registry.events("model_generation"))
        self.live_capture = LiveCapture(self)

    def start_live_capture(self, interface):
        if self.current is None:
            self.generate(.1)
        return self.live_capture.start(interface)

    def _public(self, name):
        with np.load(self.split / "public" / f"{name}.npz", allow_pickle=False) as data:
            return data["x"].copy(), data["y"].copy()

    def overview(self):
        proposal_id = "b5_seed11_trial0"
        run_id = f"b5_{proposal_id}"
        registry_events = self.registry.events()
        events = [event for event in registry_events
                  if event["payload"].get("proposal_id") == proposal_id or
                  (event["kind"] == "validator_verdict" and event["payload"].get("run_id") == run_id)]
        unlinked_calls = [{"event_id": event["event_id"], **event["payload"]}
                          for event in registry_events if event["kind"] == "agent_call" and
                          not event["payload"].get("proposal_id")]
        validator_events = self.registry.events("validator_verdict")
        return {"report": self.report, "agent_events": events,
                "unlinked_agent_calls": unlinked_calls,
                "validator_rejections": sum(event["payload"]["verdict"] == "reject"
                                            for event in validator_events),
                "agent_mode": "recorded scripted offline cycle; no live LLM",
                "data_mode": "real NF-v2 group-disjoint alternate study",
                "limits": ["No verified campaign or timestamp field; chronological generalization untested",
                           "B4 search used a debug cost proxy; research HV is shown only where timing is pinned",
                           "C1–C4 claim thresholds absent; verdicts inconclusive",
                           "Feature-space evasion is a proxy for packet-level manipulation",
                           "No DPU or production traffic blocking measured"]}

    def generate(self, cost, load=0, alerts=0):
        if not 0 <= cost <= 1 or load not in (0, 1, 2) or alerts not in (0, 1, 2):
            raise ValueError("cost must be in [0,1] and context bins in {0,1,2}")
        with self.lock:
            preference = ((1 - cost) * .8, (1 - cost) * .2, cost)
            baked = bake(self.hypernet, preference, (load, alerts))
            quantized = quantize(baked, self.train_x[:1024])
            float_threshold = threshold_at_fpr(baked.logits(self.val_x)[self.val_raw[1] == 0], .001)
            threshold = threshold_at_fpr(quantized.logits(self.val_x)[self.val_raw[1] == 0], .001)
            self.generation_number += 1
            run_key = f"generation{self.generation_number:06d}_cost{round(cost * 1000):04d}_load{load}_alerts{alerts}"
            export = self.generated_dir / f"{run_key}.hwi"
            inputs = self.generated_dir / f"{run_key}.i8"
            manifest = {"source_run_id": self.source["run_id"],
                                           "source_checkpoint": self.source["checkpoint_path"],
                                           "preference": preference, "context": [load, alerts],
                                           "dataset_hash": self.report["dataset_hash"],
                                           "mode": "offline_generation_no_retraining"}
            manifest_path = write_int8(quantized, export, manifest)
            input_q = quantized.quantize_inputs(self.b_x)
            write_inputs(inputs, input_q)
            c_logits = infer(self.binary, export, inputs, len(self.b_x))
            python_logits = quantized.logits_int(input_q)
            exact_match = np.array_equal(c_logits, python_logits)
            if not exact_match:
                raise ValueError("generated model disagrees with C kernel")
            float_metrics = score_binary(self.b_raw[1], baked.logits(self.b_x), float_threshold)
            int8_metrics = score_binary(self.b_raw[1], quantized.logits(self.b_x), threshold)
            float_alerts = baked.logits(self.b_x) > float_threshold
            int8_alerts = quantized.logits(self.b_x) > threshold
            timing_inputs = self.generated_dir / f"{run_key}.bench.i8"
            timing_rows = min(TIMING_BATCH_ROWS, len(input_q))
            write_inputs(timing_inputs, input_q[:timing_rows])
            timing = bench(self.binary, export, timing_inputs, warmup=20, iterations=500)
            timing["batch_rows"] = timing_rows
            result = {"source_run_id": self.source["run_id"], "preference": preference,
                    "context": [load, alerts], "float": float_metrics, "int8": int8_metrics,
                    "int8_detection_gap": int8_metrics["detection_rate"] - float_metrics["detection_rate"],
                    "alert_decision_agreement": float(np.mean(float_alerts == int8_alerts)),
                    "float_threshold": float(float_threshold), "int8_threshold": float(threshold),
                    "evaluation_rows": int(len(self.b_x)),
                    "input_mode": "recorded_nf_v2_flow_rows",
                    "macs": baked.macs, "int8_model_bytes": export.stat().st_size,
                    "gate_mask": baked.gate_mask.tolist(), "latency": timing,
                    "export": {"filename": export.name, "manifest_filename": manifest_path.name,
                        "sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
                        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        "active_neurons": int(np.count_nonzero(baked.gate_mask)),
                        "pruned_neurons": int(len(baked.gate_mask) - np.count_nonzero(baked.gate_mask)),
                        "total_neurons": int(len(baked.gate_mask)),
                        "dimensions": {"neurons_after_pruning": int(quantized.w1.shape[0]),
                            "hidden_per_neuron": int(quantized.w1.shape[1]),
                            "input_features": int(quantized.w1.shape[2])},
                        "quantization": {"dtype": "int8 weights / int32 biases",
                            "scales": dict(zip(("input", "w1", "activation", "w2", "feature", "output"),
                                                map(float, quantized.scales))),
                            "weight_values_at_int8_limit": int(sum(
                                np.count_nonzero(np.abs(array.astype(np.int16)) == 127)
                                for array in (quantized.w1, quantized.w2, quantized.out))),
                            "weight_value_count": int(quantized.w1.size + quantized.w2.size + quantized.out.size)}},
                    "kernel_verification": {"verified": bool(exact_match), "comparison": "C int32 logits == Python int32 logits",
                        "rows_compared": int(len(c_logits)), "mismatched_rows": int(np.count_nonzero(c_logits != python_logits)),
                        "maximum_absolute_logit_difference": int(np.max(np.abs(c_logits.astype(np.int64) - python_logits.astype(np.int64)))) if len(c_logits) else 0,
                        "sample_integer_logits": [{"row": int(i + 1), "c": int(c_logits[i]), "python": int(python_logits[i])}
                                                   for i in range(min(8, len(c_logits)))]},
                    "optimizer_steps_during_generation": 0,
                    "timing_note": "Unpinned kernel-only debug measurement",
                    "model_mode": "offline_checkpoint_generation_without_retraining"}
            event = self.demo_registry.append_event("model_generation", {
                "generation_id": run_key, "source_run_id": self.source["run_id"],
                "dataset_hash": self.report["dataset_hash"],
                "artifact": str(export), "artifact_sha256": hashlib.sha256(export.read_bytes()).hexdigest(),
                    "threshold": threshold, "threshold_source": "a_val_int8_calibration", "result": result})
            result["registry_event_id"] = event
            self.current = {"int8": quantized, "threshold": threshold, "baked": baked,
                            "preference": preference, "generation_event_id": event}
            return result

    def domain_analysis(self, load=None, alerts=None, feature="IN_BYTES"):
        if self.current is None:
            self.generate(.1)
        if load is not None and (not isinstance(load, int) or load not in (0, 1, 2)):
            raise ValueError("load context must be 0, 1, or 2")
        if alerts is not None and (not isinstance(alerts, int) or alerts not in (0, 1, 2)):
            raise ValueError("alert context must be 0, 1, or 2")
        if feature not in self.normalizer.features:
            raise ValueError("feature is not in the trained NF-v2 schema")
        preference = self.current["preference"]
        train_volume = np.log1p(self.train_raw[0][:, 6] + self.train_raw[0][:, 7])
        volume_cutoffs = np.quantile(train_volume, [1 / 3, 2 / 3])
        reference_context = (1, 1)
        reference_baked = bake(self.hypernet, preference, reference_context)
        reference_int8 = quantize(reference_baked, self.train_x[:1024])
        reference_threshold = threshold_at_fpr(
            reference_int8.logits(self.val_x)[self.val_raw[1] == 0], .001)

        def profile(name, raw, normalized):
            volume = np.log1p(raw[:, 6] + raw[:, 7])
            load_bin = int(np.searchsorted(volume_cutoffs, np.mean(volume)))
            alert_rate = float(np.mean(reference_int8.logits(normalized) > reference_threshold))
            alert_bin = 0 if alert_rate < .01 else 1 if alert_rate < .1 else 2
            return {"name": name, "rows": int(len(raw)),
                    "mean_log_byte_volume": float(np.mean(volume)),
                    "predicted_alert_rate": alert_rate,
                    "measured_context": [load_bin, alert_bin]}

        source_domain = profile("UNSW-NB15 A-train", self.train_raw[0], self.train_x)
        target_domain = profile("CSE-CIC-IDS2018 B-evaluation", self.b_raw[0], self.b_x)
        source_context = tuple(source_domain["measured_context"])
        target_context = ((load if load is not None else target_domain["measured_context"][0]),
                          (alerts if alerts is not None else target_domain["measured_context"][1]))
        source_raw = self.hypernet.generate(preference, source_context)
        target_raw = self.hypernet.generate(preference, target_context)
        source_baked = bake(self.hypernet, preference, source_context)
        target_baked = bake(self.hypernet, preference, target_context)
        source_int8 = quantize(source_baked, self.train_x[:1024])
        target_int8 = quantize(target_baked, self.train_x[:1024])
        source_threshold = threshold_at_fpr(
            source_int8.logits(self.val_x)[self.val_raw[1] == 0], .001)
        target_threshold = threshold_at_fpr(
            target_int8.logits(self.val_x)[self.val_raw[1] == 0], .001)
        parameter_parts = ("w1", "b1", "w2", "b2", "out")
        per_neuron_delta = np.sqrt(sum(np.sum((target_raw[key] - source_raw[key]) ** 2,
                                                    axis=tuple(range(1, source_raw[key].ndim)))
                                       if source_raw[key].ndim > 1 else
                                       (target_raw[key] - source_raw[key]) ** 2
                                       for key in parameter_parts))
        denominator = np.sqrt(sum(float(np.sum(source_raw[key] ** 2)) for key in parameter_parts))
        parameter_delta = np.sqrt(sum(float(np.sum((target_raw[key] - source_raw[key]) ** 2))
                                      for key in parameter_parts))
        changed_order = np.argsort(per_neuron_delta)[::-1][:min(12, len(per_neuron_delta))]
        feature_index = self.normalizer.features.index(feature)
        feature_points = np.quantile(self.b_raw[0][:, feature_index], np.linspace(0, 1, 41))
        curve_raw = np.tile(np.median(self.b_raw[0], axis=0), (len(feature_points), 1))
        curve_raw[:, feature_index] = feature_points
        curve_x = self.normalizer.transform(curve_raw)

        def neuron_outputs(generated):
            preactivation = np.einsum("bd,mhd->bmh", curve_x, generated["w1"]) + generated["b1"]
            hidden = np.maximum(preactivation, 0)
            values = np.einsum("bmh,mh->bm", hidden, generated["w2"]) + generated["b2"]
            return values * generated["out"] * generated["gate"]

        source_neuron_outputs = neuron_outputs(source_raw)
        target_neuron_outputs = neuron_outputs(target_raw)
        neuron_rows = [{"neuron": int(index + 1),
                        "source_gate": float(source_raw["gate"][index]),
                        "target_gate": float(target_raw["gate"][index]),
                        "contribution_delta_l2": float(np.linalg.norm(
                            target_neuron_outputs[:, index] - source_neuron_outputs[:, index])),
                        "source_output_weight": float(source_raw["out"][index]),
                        "target_output_weight": float(target_raw["out"][index])}
                       for index in changed_order]
        source_logits = source_neuron_outputs.sum(axis=1)
        target_logits = target_neuron_outputs.sum(axis=1)
        source_metrics = score_binary(self.b_raw[1], source_int8.logits(self.b_x), source_threshold)
        target_metrics = score_binary(self.b_raw[1], target_int8.logits(self.b_x), target_threshold)
        result = {"source_run_id": self.source["run_id"],
                  "preference": [float(value) for value in preference],
                  "source_domain": source_domain, "target_domain": target_domain,
                  "source_context": list(source_context), "target_context": list(target_context),
                  "context_rules": {"load": "mean(log1p(IN_BYTES + OUT_BYTES)) compared with A-train one-third/two-thirds quantiles",
                                    "alerts": "reference int8 predicted-alert rate: <1% low, <10% medium, otherwise high",
                                    "reference_context": list(reference_context)},
                  "target_context_mode": "measured_domain" if load is None and alerts is None else "user_selected_context_scenario",
                  "same_context": source_context == target_context,
                  "parameter_delta_l2": float(parameter_delta),
                  "parameter_delta_relative": float(parameter_delta / max(denominator, 1e-12)),
                  "gate_delta_l2": float(np.linalg.norm(target_raw["gate"] - source_raw["gate"])),
                  "changed_gate_count": int(np.count_nonzero(
                      (target_raw["gate"] > 0) != (source_raw["gate"] > 0))),
                  "source_neurons": int(np.count_nonzero(source_baked.gate_mask)),
                  "target_neurons": int(np.count_nonzero(target_baked.gate_mask)),
                  "source_macs": int(source_baked.macs), "target_macs": int(target_baked.macs),
                  "source_threshold": float(source_threshold), "target_threshold": float(target_threshold),
                  "gate_constants": {"gamma": self.hypernet.gamma, "zeta": self.hypernet.zeta,
                                     "temperature": self.hypernet.temperature},
                  "source_b_metrics": source_metrics, "target_b_metrics": target_metrics,
                  "feature": feature, "feature_points": feature_points.tolist(),
                  "source_logit_curve": source_logits.tolist(), "target_logit_curve": target_logits.tolist(),
                  "neurons": neuron_rows,
                  "math": {"condition": "z_j = [e_j, lambda_1, lambda_2, lambda_3, c_load/2, c_alert/2, 1]",
                           "generator": "r_j = z_j @ P; split r_j into W1_j, b1_j, w2_j, b2_j, o_j, s_j",
                           "gate": "g_j = clip(sigmoid(s_j / tau) * (zeta - gamma) + gamma, 0, 1)",
                           "neuron": "h_j(x) = w2_j^T ReLU(W1_j x + b1_j) + b2_j",
                           "network": "f(x) = sum_j o_j * g_j * h_j(x)"}}
        event = self.demo_registry.append_event("domain_context_comparison", {
            key: value for key, value in result.items()
            if key not in {"feature_points", "source_logit_curve", "target_logit_curve", "neurons", "math"}})
        result["registry_event_id"] = event
        return result

    def evasion(self, kind, epsilon):
        if self.current is None:
            self.generate(.1)
        with self.lock:
            attack = self.b_raw[0][self.b_raw[1] == 1]
            if not len(attack):
                raise ValueError("B evaluation partition contains no attack flows")
            train_benign = self.train_raw[0][self.train_raw[1] == 0]
            if not len(train_benign):
                raise ValueError("training partition contains no benign flows")
            reference = np.median(train_benign, axis=0)
            candidate = constrained_nf_evasion(attack, reference, self.normalizer.features,
                                               epsilon, kind, seed=20261024)
            model, threshold = self.current["int8"], self.current["threshold"]
            original_scores = model.logits(self.normalizer.transform(attack))
            candidate_scores = model.logits(self.normalizer.transform(candidate))
            original_alerts = original_scores > threshold
            candidate_alerts = candidate_scores > threshold
            delta = candidate - attack
            immutable = nf_mutable_mask(self.normalizer.features)
            changed = np.any(delta != 0, axis=0)
            feature_changes = []
            for index in np.flatnonzero(changed):
                values = delta[:, index]
                feature_changes.append({"feature": self.normalizer.features[index],
                    "changed_flows": int(np.count_nonzero(values)),
                    "mean_absolute_delta": float(np.mean(np.abs(values))),
                    "median_absolute_delta": float(np.median(np.abs(values))),
                    "mean_relative_delta_pct": float(100 * np.mean(
                        np.abs(values) / np.maximum(np.abs(attack[:, index]), 1)))})
            feature_changes.sort(key=lambda row: row["mean_absolute_delta"], reverse=True)
            affected_rows = np.flatnonzero(np.any(delta != 0, axis=1))
            examples = [{"flow": int(i + 1), "original_score": float(original_scores[i]),
                         "candidate_score": float(candidate_scores[i]),
                         "original_alert": bool(original_alerts[i]),
                         "candidate_alert": bool(candidate_alerts[i]),
                         "changed_features": int(np.count_nonzero(delta[i]))}
                        for i in affected_rows[:12]]
            bound = float(epsilon) * np.maximum(np.abs(attack), 1)
            tolerance = 4 * np.spacing(np.maximum(np.abs(attack), 1)) + 1e-8
            constraint_checks = {"immutable_fields_preserved": bool(
                    np.array_equal(candidate[:, ~immutable], attack[:, ~immutable])),
                "epsilon_bound_preserved": bool(np.all(np.abs(delta) <= bound + tolerance)),
                "nonnegative_features": bool(np.all(candidate >= 0)),
                "all_changes_within_mutable_features": bool(not np.any(delta[:, ~immutable]))}
            result = {"kind": kind, "epsilon": float(epsilon), "attack_rows": len(attack),
                "threshold": float(threshold), "threshold_source": "A-val fixed operating point",
                "original_detection_rate": float(np.mean(original_alerts)),
                "evasion_detection_rate": float(np.mean(candidate_alerts)),
                "original_detected": int(np.count_nonzero(original_alerts)),
                "candidate_detected": int(np.count_nonzero(candidate_alerts)),
                "detected_to_missed": int(np.count_nonzero(original_alerts & ~candidate_alerts)),
                "missed_to_detected": int(np.count_nonzero(~original_alerts & candidate_alerts)),
                "changed_flows": int(len(affected_rows)), "changed_feature_count": int(np.count_nonzero(changed)),
                "feature_changes": feature_changes[:12], "examples": examples,
                "constraint_checks": constraint_checks,
                "immutable_fields_preserved": constraint_checks["immutable_fields_preserved"],
                "feature_space_proxy": True, "evaluation_partition": "B (CSE)",
                "scoring_model": "current generated int8 candidate",
                "perturbation_seed": 20261024,
                "scope_note": "Constrained feature-space transformation of recorded flow rows; no packets were generated or modified."}
            result["registry_event_id"] = self.demo_registry.append_event("evasion_measurement", {
                "generation_event_id": self.current["generation_event_id"], "result": result})
            return result

    def traffic(self, count=20):
        if self.current is None:
            self.generate(.1)
        count = min(max(int(count), 1), 100)
        with self.lock:
            model, threshold = self.current["int8"], self.current["threshold"]
            scores = model.logits(self.b_x[:count])
            result = {"partition": "B (CSE)", "total_rows": len(self.b_x),
                    "sample": [{"row": i, "label": int(self.b_raw[1][i]),
                                "score": float(scores[i]), "alert": bool(scores[i] > threshold)}
                               for i in range(count)], "threshold_source": "A-val"}
            result["registry_event_id"] = self.demo_registry.append_event("traffic_replay", {
                "generation_event_id": self.current["generation_event_id"], "result": result})
            return result

    def autopilot(self, window=128, detection_tolerance=.02):
        """Order-free B replay: switch baked context once per aggregate window."""
        if self.current is None:
            self.generate(.1)
        if window < 2 or not 0 <= detection_tolerance <= 1:
            raise ValueError("invalid window or tolerance")
        with self.lock:
            current = self.current
            baseline = current["int8"].logits(self.b_x) > current["threshold"]
            dynamic = np.zeros(len(self.b_x), dtype=bool)
            train_volume = np.log1p(self.train_raw[0][:, 6] + self.train_raw[0][:, 7])
            cutoffs = np.quantile(train_volume, [1/3, 2/3])
            cache = {}
            contexts = []
            weighted_macs = 0
            for start in range(0, len(self.b_x), window):
                stop = min(start + window, len(self.b_x))
                raw = self.b_raw[0][start:stop]
                load = int(np.searchsorted(cutoffs, np.mean(np.log1p(raw[:, 6] + raw[:, 7]))))
                alert_rate = float(np.mean(baseline[start:stop]))
                alerts = 0 if alert_rate < .01 else 1 if alert_rate < .1 else 2
                context = (load, alerts)
                contexts.append(context)
                if context not in cache:
                    baked = bake(self.hypernet, current["preference"], context)
                    quantized = quantize(baked, self.train_x[:1024])
                    threshold = threshold_at_fpr(quantized.logits(self.val_x)[self.val_raw[1] == 0], .001)
                    cache[context] = quantized, threshold, baked.macs
                model, threshold, macs = cache[context]
                dynamic[start:stop] = model.logits(self.b_x[start:stop]) > threshold
                weighted_macs += (stop - start) * macs
            labels = self.b_raw[1]
            attack, benign = labels == 1, labels == 0
            static_detection = float(np.mean(baseline[attack]))
            dynamic_detection = float(np.mean(dynamic[attack]))
            same_detection = abs(static_detection - dynamic_detection) <= detection_tolerance
            static_macs = current["baked"].macs
            dynamic_macs = weighted_macs / len(self.b_x)
            result = {"windows": len(contexts), "context_changes": sum(a != b for a, b in zip(contexts, contexts[1:])),
                    "static_detection_rate": static_detection, "dynamic_detection_rate": dynamic_detection,
                    "static_fpr": float(np.mean(baseline[benign])),
                    "dynamic_fpr": float(np.mean(dynamic[benign])),
                    "static_macs_per_flow": static_macs, "dynamic_macs_per_flow": dynamic_macs,
                    "equal_detection_within_tolerance": same_detection,
                    "cost_reduction_at_equal_detection": (static_macs - dynamic_macs) / max(static_macs, 1)
                        if same_detection else None,
                    "detection_tolerance": detection_tolerance,
                    "temporal_order_verified": False,
                    "note": "Order-free B group-sample replay; no C4 research claim"}
            result["registry_event_id"] = self.demo_registry.append_event("autopilot_measurement", {
                "generation_event_id": self.current["generation_event_id"], "result": result})
            return result

    def decide(self, approved):
        if not isinstance(approved, bool):
            raise TypeError("approval must be boolean")
        with self.lock:
            self.action_number += 1
            proposal_id = f"demo_{self.action_number}"
            responses = scripted_responses(count=1, validation_calls=0)
            responses["Experimental Scientist"][0].update(
                action="simulate_response", hypothesis="A safe response simulation can be reviewed",
                rationale="Demonstrate the approval gate without touching production traffic")
            client = ScriptedClient(responses)
            class Context:
                dataset_hash = self.report["dataset_hash"]
                git_commit = self.source["git_commit"]
            cycle = AgentCycle(self.demo_registry, Context(), None, client, condition="b5", budget_limit=2)
            base = Config(model="b5", seed=11, epochs=1, target_fpr=.001,
                          shift_columns=SEARCH_OPTIONS["shift_columns"][1])
            result = cycle.run(proposal_id, base, HumanDecision(approved, "demo_operator"),
                               request_validation=False)
            return {"outcome": result, "action_kind": "controlled_response_simulation",
                    "production_action": False, "registry": str(self.demo_registry.path)}


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, code=200):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            url = urlparse(self.path)
            args = parse_qs(url.query)
            try:
                if url.path == "/api/overview":
                    return self._json(state.overview())
                if url.path == "/api/artifact":
                    name = args.get("name", [""])[0]
                    if not name or Path(name).name != name or not name.endswith((".hwi", ".hwi.json")):
                        return self._json({"error": "invalid generated artifact name"}, 400)
                    artifact = state.generated_dir / name
                    if not artifact.is_file() or artifact.resolve().parent != state.generated_dir.resolve():
                        return self._json({"error": "generated artifact not found"}, 404)
                    payload = artifact.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if url.path == "/api/generate":
                    return self._json(state.generate(float(args.get("cost", ["0.1"])[0]),
                                                     int(args.get("load", ["0"])[0]),
                                                     int(args.get("alerts", ["0"])[0])))
                if url.path == "/api/evasion":
                    return self._json(state.evasion(args.get("kind", ["noise"])[0],
                                                    float(args.get("epsilon", ["0.1"])[0])))
                if url.path == "/api/traffic":
                    return self._json(state.traffic(int(args.get("count", ["20"])[0])))
                if url.path == "/api/autopilot":
                    return self._json(state.autopilot(int(args.get("window", ["128"])[0])))
                if url.path == "/api/live/interfaces":
                    return self._json(state.live_capture.interfaces())
                if url.path == "/api/live/status":
                    return self._json(state.live_capture.status())
                if url.path == "/api/live/events":
                    return self._json(state.live_capture.snapshot(int(args.get("after", ["0"])[0])))
                if url.path == "/api/domain/compare":
                    load = int(args["load"][0]) if "load" in args else None
                    alerts = int(args["alerts"][0]) if "alerts" in args else None
                    return self._json(state.domain_analysis(load, alerts,
                                                           args.get("feature", ["IN_BYTES"])[0]))
                filename = "index.html" if url.path == "/" else url.path.removeprefix("/")
                if filename not in {"index.html", "app.js", "style.css"}:
                    return self._json({"error": "not found"}, 404)
                path = STATIC / filename
                payload = path.read_bytes()
                kind = "text/html" if filename.endswith("html") else "text/javascript" if filename.endswith("js") else "text/css"
                self.send_response(200)
                self.send_header("Content-Type", kind + "; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                self._json({"error": str(exc)}, 400)

        def do_POST(self):
            if self.path in {"/api/live/start", "/api/live/stop"}:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length > 1024:
                        raise ValueError("request too large")
                    body = json.loads(self.rfile.read(length) or b"{}")
                    result = (state.start_live_capture(body.get("interface")) if self.path.endswith("start")
                              else state.live_capture.stop())
                    return self._json(result)
                except (ValueError, TypeError, RuntimeError, OSError) as exc:
                    return self._json({"error": str(exc)}, 400)
            if self.path != "/api/decision":
                return self._json({"error": "not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024:
                    raise ValueError("request too large")
                body = json.loads(self.rfile.read(length))
                return self._json(state.decide(body["approved"]))
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                self._json({"error": str(exc)}, 400)

        def log_message(self, *_):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    state = DemoState(args.study, args.split)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print(f"Offline HyperWeave demo: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
