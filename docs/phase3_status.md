# Phase 3 status

Run `make verify-phase3` for the Phase 1–3 and global gate. Run `python3 -m hyperweave.baselines.phase3 --output runs/phase3_debug_01` for a fresh, offline scripted cycle. The output contains `registry.sqlite`, immutable checkpoints, the private mock-validator store, and `report.json`.

The scripted cycle exercises all six roles, the JSON proposal contract, a separately supplied approval decision, transactional run budgets, actual hypernetwork training and baking, coarse validation, and outcome reporting. B4 uses NSGA-II. B4, B5 and B5b each run 12 attempts over seeds 11, 23 and 37. B5b receives no observed metric fields through agent tools. `ReplayClient` reuses logged responses and tool results without a model API. The test suite checks replay event and registry equality.

The six-role workflow now records proposal and condition IDs on each agent
call. The Neuron Architect and Training Strategist can apply validated edits
from the shared search space; the Optimization Scientist supplies a ranked
frontier; the Coordinator can reject before budget reservation. B5's surrogate
uses only B5 history and B5b requires blind mode. Focused tests exercise the
edit, rejection, isolation and blind-mode controls. The [real-data workflow
audit](phase4_detailed_report.md) records which branches the default script
actually used.

This is a **synthetic debug workflow**. Its approval actor is `scripted_human_approval`, its validator effect is scripted, and its B4 search uses a model-size proxy because this macOS host cannot pin a CPU core. The research objective mode refuses unpinned latency. The validator API supports separate general, shift, evasion, latency and regression checks, but trusted real-data check functions have not yet been supplied. Agents only receive an allowlisted read-only filesystem tool and coarse validator replies; the debug process itself is not an OS-isolated security boundary.

The Phase 3 scripted exit gate is met. The [Phase 4 alternate study](phase4_status.md)
has now performed once-only real-data sealed evaluation. Physical-core timing,
real LLM cycles, the original chronological protocol and C1–C4 verdicts remain
open. The earlier macOS timing description above refers to the original debug
run; the current real-data study ran on Windows with unpinned timings.
