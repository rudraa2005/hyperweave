"""Local-only Phase 5 dashboard backed by the real group-split registry."""
import argparse
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
from hyperweave.data.evasion import constrained_nf_evasion
from hyperweave.data.normalization import FlowNormalizer
from hyperweave.eval.metrics import score_binary, threshold_at_fpr
from hyperweave.kernel.runner import bench, compile_kernel, infer, write_inputs
from hyperweave.models import bake, load_hypernet, quantize, write_int8
from hyperweave.registry import Registry


STATIC = Path(__file__).with_name("static")


class DemoState:
    def __init__(self, study, split):
        self.study, self.split = Path(study), Path(split)
        self.report = json.loads((self.study / "report.json").read_text())
        if self.report["kind"] != "real_nf_v2_endpoint_group_alternate_study":
            raise ValueError("demo needs the real endpoint-group study")
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
        self.hypernet = load_hypernet(self.study / self.source["checkpoint_path"])
        self.binary = compile_kernel(self.study / "demo_infer")
        self.generated_dir = self.study / "demo_generated"
        self.generated_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.current = None
        self.demo_registry = Registry(self.study / "demo_actions.sqlite")
        self.action_number = len(self.demo_registry.events("cycle_outcome"))

    def _public(self, name):
        with np.load(self.split / "public" / f"{name}.npz", allow_pickle=False) as data:
            return data["x"].copy(), data["y"].copy()

    def overview(self):
        proposal_id = "b5_seed11_trial0"
        events = [event for event in self.registry.events()
                  if proposal_id in json.dumps(event["payload"], sort_keys=True) and
                  event["kind"] in {"proposal", "human_decision", "action_started", "action_status",
                                    "validator_verdict", "cycle_outcome"}]
        validator_events = self.registry.events("validator_verdict")
        return {"report": self.report, "agent_events": events,
                "validator_rejections": sum(event["payload"]["verdict"] == "reject"
                                            for event in validator_events),
                "agent_mode": "recorded scripted offline cycle; no live LLM",
                "data_mode": "real NF-v2 group-disjoint alternate study",
                "limits": ["No verified campaign or timestamp field; chronological generalization untested",
                           "B4 search used a 2D debug cost proxy because CPU core pinning is unavailable",
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
            threshold = threshold_at_fpr(baked.logits(self.val_x)[self.val_raw[1] == 0], .001)
            run_key = f"cost{round(cost * 1000):04d}_load{load}_alerts{alerts}"
            export = self.generated_dir / f"{run_key}.hwi"
            inputs = self.generated_dir / f"{run_key}.i8"
            write_int8(quantized, export, {"source_run_id": self.source["run_id"],
                                           "source_checkpoint": self.source["checkpoint_path"],
                                           "preference": preference, "context": [load, alerts],
                                           "dataset_hash": self.report["dataset_hash"],
                                           "mode": "offline_generation_no_retraining"})
            input_q = quantized.quantize_inputs(self.b_x)
            write_inputs(inputs, input_q)
            if not np.array_equal(infer(self.binary, export, inputs, len(self.b_x)),
                                  quantized.logits_int(input_q)):
                raise ValueError("generated model disagrees with C kernel")
            float_metrics = score_binary(self.b_raw[1], baked.logits(self.b_x), threshold)
            int8_metrics = score_binary(self.b_raw[1], quantized.logits(self.b_x), threshold)
            timing = bench(self.binary, export, inputs, warmup=20, iterations=500)
            self.current = {"int8": quantized, "threshold": threshold, "baked": baked,
                            "preference": preference}
            return {"source_run_id": self.source["run_id"], "preference": preference,
                    "context": [load, alerts], "float": float_metrics, "int8": int8_metrics,
                    "int8_detection_gap": int8_metrics["detection_rate"] - float_metrics["detection_rate"],
                    "macs": baked.macs, "int8_model_bytes": export.stat().st_size,
                    "gate_mask": baked.gate_mask.tolist(), "latency": timing,
                    "optimizer_steps_during_generation": 0,
                    "timing_note": "Unpinned kernel-only debug measurement"}

    def evasion(self, kind, epsilon):
        if self.current is None:
            self.generate(.1)
        with self.lock:
            attack = self.b_raw[0][self.b_raw[1] == 1]
            train_benign = self.train_raw[0][self.train_raw[1] == 0]
            reference = np.median(train_benign, axis=0)
            candidate = constrained_nf_evasion(attack, reference, self.normalizer.features,
                                               epsilon, kind, seed=20261024)
            model, threshold = self.current["int8"], self.current["threshold"]
            original_rate = float(np.mean(model.logits(self.normalizer.transform(attack)) > threshold))
            rate = float(np.mean(model.logits(self.normalizer.transform(candidate)) > threshold))
            return {"kind": kind, "epsilon": epsilon, "attack_rows": len(attack),
                    "original_detection_rate": original_rate, "evasion_detection_rate": rate,
                    "immutable_fields_preserved": True, "feature_space_proxy": True}

    def traffic(self, count=20):
        if self.current is None:
            self.generate(.1)
        count = min(max(int(count), 1), 100)
        with self.lock:
            model, threshold = self.current["int8"], self.current["threshold"]
            scores = model.logits(self.b_x[:count])
            return {"partition": "B (CSE)", "total_rows": len(self.b_x),
                    "sample": [{"row": i, "label": int(self.b_raw[1][i]),
                                "score": float(scores[i]), "alert": bool(scores[i] > threshold)}
                               for i in range(count)], "threshold_source": "A-val"}

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
                    threshold = threshold_at_fpr(baked.logits(self.val_x)[self.val_raw[1] == 0], .001)
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
            return {"windows": len(contexts), "context_changes": sum(a != b for a, b in zip(contexts, contexts[1:])),
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
            except (ValueError, TypeError, RuntimeError) as exc:
                self._json({"error": str(exc)}, 400)

        def do_POST(self):
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
