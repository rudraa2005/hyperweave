# HyperWeave-FN

HyperWeave-FN is a research prototype for studying compact, preference-aware classifiers for network-flow data. It combines neural baselines, a hypernetwork that generates classifiers for different cost preferences, an integer inference path, and experiment tooling for comparing models and search strategies.

The repository contains code and reports for Phases 1–5. It is intended for research and reproducible experiments, not production network monitoring or automated incident response. The included agent cycle is scripted and offline; it does not call a live language model or block network traffic.

## Get started

Requires Python 3.10 or newer. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[test]'
```

Run the test gates with:

```sh
make verify-phase5
```

Each `verify-phaseN` target runs that phase and all earlier phase tests, plus the global gate.

## Try a demo

The quickest way to see the model pipeline is the deterministic replay below. It generates synthetic network flows and exercises the B2 and hypernetwork paths. Its output demonstrates the pipeline; it is not a real-data performance result.

```sh
python3 -m hyperweave.demo.phase2_replay --output runs/replay_01
```

The offline Phase 3 replay exercises the scripted proposal cycle, approval gate, and equal B4/B5/B5b run budgets:

```sh
python3 -m hyperweave.baselines.phase3 --output runs/agent_replay_01
```

This replay uses no live model API. More detail is in the [Phase 2](docs/phase2_status.md) and [Phase 3](docs/phase3_status.md) reports.

## Run the NF-v2 study and dashboard

The real-data workflow uses the official NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 CSV files. Put them at `data/raw/` using these names:

```text
data/raw/NF-UNSW-NB15-v2.csv
data/raw/NF-CSE-CIC-IDS2018-v2.csv
```

Then create a group-disjoint split, run the Phase 4 study, and start the local Phase 5 dashboard:

```sh
python3 -m hyperweave.data.group_split \
  --unsw data/raw/NF-UNSW-NB15-v2.csv \
  --cse data/raw/NF-CSE-CIC-IDS2018-v2.csv \
  --output runs/group_split_01

python3 -m hyperweave.baselines.phase4 \
  --split runs/group_split_01 \
  --output runs/phase4_study_01

python3 -m hyperweave.demo.phase5 \
  --study runs/phase4_study_01 \
  --split runs/group_split_01 \
  --port 8765
```

Open <http://127.0.0.1:8765/>. The dashboard binds to localhost, uses local assets, and records its actions in the study's `demo_actions.sqlite`. Keep each study's output directory together with its split artifacts. Use new output directory names for reruns.

The CSVs and generated run outputs are excluded from version control. Download sources, preparation instructions, audits, and artifact details are in the [data and artifact guide](docs/data_and_artifacts.md). To check which inputs your local checkout can find, run:

```sh
python3 -m hyperweave.data.prepare
```

Fashion-MNIST is optional and used only for the Phase 1 image-debug path; it is not needed for the network-flow study.

## How the pipeline fits together

```text
NF-v2 flow data
      │
      ├── deterministic split and normalization
      ├── baseline classifiers and hypernetwork training
      ├── preference-conditioned model generation and int8 baking
      ├── inference, robustness checks, and experiment registry
      └── reports and local study dashboard
```

The Python package is organized by this workflow:

| Directory | Purpose |
| --- | --- |
| `hyperweave/data/` | Loaders, audits, splits, normalization, and evasion transforms |
| `hyperweave/models/` | Baselines, hypernetwork, model baking, and noise handling |
| `hyperweave/kernel/` and `kernel/` | Inference runner and native C kernel |
| `hyperweave/agents/` | Proposal cycle, tools, budget controls, and surrogate logic |
| `hyperweave/baselines/` | Phase-specific experiment runners |
| `hyperweave/registry/` | SQLite experiment and event records |
| `hyperweave/eval/` | Metrics, robustness evaluation, and reports |
| `hyperweave/demo/` | Offline replays and local dashboard |
| `configs/`, `tests/`, `docs/` | Configuration, tests, and research documentation |

## What the results do and do not establish

The real-data study uses an endpoint-group split because the available NF-v2 CSVs do not provide verified campaign or timestamp metadata. It therefore does not establish results under the original chronological protocol. The C1–C4 research verdicts remain inconclusive without the preregistered thresholds, and comparable 3D latency results require measurements on a pinned CPU core. Treat local timings as debugging measurements, not line-rate guarantees.

For the phase-by-phase implementation and evidence, see the [implementation audit](docs/implementation_audit.md), [Phase 4 status](docs/phase4_status.md), [detailed Phase 4 report](docs/phase4_detailed_report.md), [claims and open gates](docs/claims.md), and [Phase 5 demo guide](docs/phase5_demo.md).

## License

This repository does not currently include a license file. Contact the repository owner about reuse or redistribution.
