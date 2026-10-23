# Phase 1 status

The implementation includes deterministic Fashion-MNIST IDX image loading, partitions and corruptions; B1/B2 NumPy training; analytic MAC accounting and matching; NF-v2 CSV validation; deduplication before chronological campaign splits; shift transforms; constrained feature-space evasion; A-val threshold calibration; and immutable SQLite records with checkpoints.

The test gate passed on synthetic fixtures. This is **not** the Phase 1 exit gate: neither NF-v2 dataset nor the Fashion-MNIST files were present locally. Real-data detection, PR-AUC, float latency, and the B1 versus logistic sanity check remain unverified. The runner refuses an A-val set too small to resolve the target FPR.

The official [UQ dataset catalog](https://www.cyber.uq.edu.au/node/824) describes the NF-v2 collections as 43-feature datasets. The CSV runner requires an ordered 43-feature manifest plus campaign and time columns. It currently requires numeric features and rejects missing/categorical features; a source-specific encoder and verified campaign/time metadata are still needed for the published CSVs. This avoids guessing the schema or performing a random flow split.

Run `make verify-phase1` for the local test gate. For real data, run `python3 -m hyperweave.baselines.run --csv PATH --features FEATURES_TXT --dataset nf_unsw_nb15_v2 --campaign-col COLUMN --time-col COLUMN --output runs/phase1`; repeat for `nf_cse_cic_ids2018_v2` and seeds 11, 23, 37. Each input must contain at least four nonoverlapping chronological campaigns, with both labels represented in A-train, A-val and B.
