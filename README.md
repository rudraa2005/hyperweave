# HyperWeave-FN

Phase 1 and Phase 2 implementation. Install with `python3 -m pip install -e '.[test]'` and run `make verify-phase2`.

For a deterministic offline model comparison using the actual B2 and hypernetwork paths, run `python3 -m hyperweave.demo.phase2_replay --output runs/debug_01`. This replay uses synthetic flows and is a pipeline demonstration, not a network-defense performance result. See [Phase 2 status](docs/phase2_status.md).

The two NF-v2 CSV datasets and Fashion-MNIST files are **not bundled**. Real-data baselines require those files and campaign/time metadata. Synthetic fixtures exercise invariants, not performance. See `docs/claims.md` for commitments and unresolved proposal thresholds.
