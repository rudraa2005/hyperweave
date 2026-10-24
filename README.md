# HyperWeave-FN

Phase 1–5 research prototype with a runner for a real NF-v2 endpoint-group study and an offline local dashboard. Install with `python3 -m pip install -e '.[test]'` and run `make verify-phase5`.

For a phase-by-phase account of what this checkout can verify and what still
requires datasets or pinned hardware, see the [implementation audit](docs/implementation_audit.md).
For official dataset sources, checksum checks, and the exact run artifacts,
see the [data and artifact handoff](docs/data_and_artifacts.md).
Run `python -m hyperweave.data.prepare` to see which inputs are present.
The completed real-data alternate study is in
[`runs/phase4_group_20261024_agent_audit/report.json`](runs/phase4_group_20261024_agent_audit/report.json).
The [detailed performance and multiagent report](docs/phase4_detailed_report.md)
explains its measurements, workflow trace, and limits; see also [Phase 4 status](docs/phase4_status.md).

For a deterministic offline model comparison using the actual B2 and hypernetwork paths, run `python3 -m hyperweave.demo.phase2_replay --output runs/debug_01`. This replay uses synthetic flows and is a pipeline demonstration, not a network-defense performance result. See [Phase 2 status](docs/phase2_status.md).

For the offline scripted agent/search cycle, run `python3 -m hyperweave.baselines.phase3 --output runs/phase3_debug_01`. It exercises the approval gate and equal B4/B5/B5b run budgets without a live API; see [Phase 3 status](docs/phase3_status.md).

Both downloaded NF-v2 CSVs have been audited. For the preregistered alternate group-disjoint study, run:

```sh
python3 -m hyperweave.data.group_split --unsw data/raw/NF-UNSW-NB15-v2.csv --cse data/raw/NF-CSE-CIC-IDS2018-v2.csv --output runs/group_split_01
python3 -m hyperweave.baselines.phase4 --split runs/group_split_01 --output runs/phase4_group_final
python3 -m hyperweave.demo.phase5 --study runs/phase4_group_final --split runs/group_split_01 --port 8765
```

Open `http://127.0.0.1:8765/` for the five-view dashboard. The CLI creates immutable run records and a registry-derived report. Use fresh output directories for a rerun. The dashboard only binds to localhost and works without external network access. See [group split protocol](docs/group_split_protocol.md), [Phase 4 status](docs/phase4_status.md), and [Phase 5 guide](docs/phase5_demo.md).

The two NF-v2 CSV datasets and Fashion-MNIST files are **not bundled**. The two
NF-v2 CSVs are sufficient for the network-defense study; Fashion-MNIST is only
the separate Phase 1 image debug tier. The endpoint-group study is an
alternative to the original chronological protocol because neither CSV has
verified campaign/time metadata. C1–C4 verdicts remain inconclusive without
the original proposal thresholds, and 3D research hypervolume remains
unmeasured without pinned-core latency. See [claims](docs/claims.md) for the
research commitments.
