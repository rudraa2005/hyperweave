# Phase 4 real-data status (2026-10-24)

The two downloaded UQ NF-v2 CSVs were found under `hyperweave/data/raw/` and
hard-linked into the documented top-level `data/raw/` paths without duplicating
their storage. Full streaming audits produced
[`unsw_audit_20261024/audit.json`](../runs/unsw_audit_20261024/audit.json) and
[`cse_audit_20261024/audit.json`](../runs/cse_audit_20261024/audit.json).
Their row counts were 2,390,275 and 18,893,708; both had zero nonfinite
feature values and zero label/category conflicts. Both CSV SHA-1 values match
their local UQ BagIt manifests. The audits are data checks, not performance
measurements.

The approved [endpoint-group alternate split](group_split_protocol.md) completed
at [`runs/group_split_20261024`](../runs/group_split_20261024/manifest.json):
A-train 11,762 rows, A-val 11,743, B 12,000, sealed 6,857. A-val contains
6,000 benign flows, giving an empirical FPR resolution of 1/6,000. The split
passed its class, deduplication and group-overlap checks. The original
campaign/time split remains untested because neither CSV has verified campaign
or timestamp fields.

The [registry-derived Phase 4 report](../runs/phase4_group_20261024_agent_audit/report.json)
and [registry](../runs/phase4_group_20261024_agent_audit/registry.sqlite) now exist. All
78 planned records completed: 9 B1, 9 B2, 12 B2b and 12 each for B3/B4/B5/B5b.
The four search conditions consumed identical 12-run budgets over seeds
`11, 23, 37`. The once-only sealed evaluation was recorded after selection.
The report contains the full scoreboard, seed bootstrap intervals, matched-MAC
pairs, float/int8 gaps, B robustness sweeps and offline autoencoder results.
Use the JSON report as the numeric source of truth rather than copying rounded
values into another scoreboard.
The [detailed performance and multiagent report](phase4_detailed_report.md)
is generated from the JSON report, split manifest and registry with
`python -m scripts.generate_detailed_report --study runs/phase4_group_20261024_agent_audit --split runs/group_split_20261024 --output docs/phase4_detailed_report.md`.
The audited run uses implementation hash
`306004d9f3138d26bb26e0462753f50acb7083f6b87508639cda58f80913f95b`.
Earlier completed runs are retained under `runs/phase4_group_20261024` and
`runs/phase4_group_20261024_final`.

The audited multiagent cycle recorded 126 role calls across 24 completed
B5/B5b cycles. Calls have proposal and condition IDs. The B5 surrogate reads
only B5 history; B5b exposes no observed metric fields through agent tools.
Architecture/training edits and Coordinator rejection are implemented and
covered by focused tests, but the default script exercised neither branch.
The run made six coarse validator queries and one final sealed evaluation.
It used a scripted client and scripted approval actor, not a live LLM or
live human review.

The result is a **negative or inconclusive research outcome**, not a superiority
claim. B detection at the fixed A-val threshold is weak for several conditions,
and B false-positive rates exceed the 0.1% target under dataset shift. The
2D hypervolume is a debug proxy. Windows kernel timing is unpinned, so the
preregistered 3D research hypervolume and line-rate claim remain unmeasured.
C1–C4 verdicts are locked as inconclusive because the original supported-if
and refuted-if thresholds were not supplied before the runs. The agent cycle
is a recorded scripted replay, not a real-LLM run.

The five-view Phase 5 dashboard was exercised against this audited study:
generation, 12,000-row traffic replay, constrained evasion on 6,000 attacks,
94-window autopilot, rejection and approval all returned registry-backed
outcomes. Approval ran a local
controlled response **simulation**, not production blocking. A timed manual
three-minute presentation and clean-machine reproduction remain unverified.

## Test gate

`python -m pytest -q --tb=short` passed **87 tests**, with **1 skipped** Linux
core-pinning test, after the full-data run. Phase 4 and Phase 5 focused tests
also passed after the bounded timing-batch change. The real-data CLI commands
and HTTP routes above completed. The supplied implementation guide ends at
Phase 5; it defines no Phase 6.
