# HyperWeave-FN

Phase 1–3 implementation. Install with `python3 -m pip install -e '.[test]'` and run `make verify-phase3`.

For a deterministic offline model comparison using the actual B2 and hypernetwork paths, run `python3 -m hyperweave.demo.phase2_replay --output runs/debug_01`. This replay uses synthetic flows and is a pipeline demonstration, not a network-defense performance result. See [Phase 2 status](docs/phase2_status.md).

For the offline scripted agent/search cycle, run `python3 -m hyperweave.baselines.phase3 --output runs/phase3_debug_01`. It exercises the approval gate and equal B4/B5/B5b run budgets without a live API; see [Phase 3 status](docs/phase3_status.md).

The downloaded NF-UNSW-NB15-v2 CSV has been audited and passed training-only model/kernel smoke checks. See [real-data status](docs/real_data_status.md) for results and the missing campaign/time metadata that blocks research evaluation.

The two NF-v2 CSV datasets and Fashion-MNIST files are **not bundled**. Real-data baselines require those files and campaign/time metadata. Synthetic fixtures exercise invariants, not performance. See `docs/claims.md` for commitments and unresolved proposal thresholds.
