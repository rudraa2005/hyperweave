# Phase 1 status

The implementation includes deterministic Fashion-MNIST IDX image loading, partitions and corruptions; B1/B2 NumPy training; analytic MAC accounting and matching; NF-v2 CSV validation; deduplication before chronological campaign splits; shift transforms; constrained feature-space evasion; A-val threshold calibration; and immutable SQLite records with checkpoints.

The test gate passed on synthetic fixtures. Both NF-v2 CSVs are now present and
have been fully audited; B1/B2 detection and PR-AUC were measured under the
approved endpoint-group alternate split in [Phase 4](phase4_status.md).
Fashion-MNIST files, the original chronological/campaign split, and the B1
versus logistic sanity check remain unverified. The split builder refuses an
A-val set too small to resolve the target FPR.

The official [UQ dataset catalog](https://www.cyber.uq.edu.au/node/824)
describes the NF-v2 collections as 43-feature datasets. The loader converts
IPv4 strings deterministically and aligns the published columns with the
ordered feature manifest. The original chronological runner still requires
verified campaign and time columns that these CSVs do not provide; the
preregistered endpoint-group study is the approved alternate evaluation.

Run `make verify-phase1` for the local test gate. For real data, run `python3 -m hyperweave.baselines.run --csv PATH --features FEATURES_TXT --dataset nf_unsw_nb15_v2 --campaign-col COLUMN --time-col COLUMN --output runs/phase1`; repeat for `nf_cse_cic_ids2018_v2` and seeds 11, 23, 37. Each input must contain at least four nonoverlapping chronological campaigns, with both labels represented in A-train, A-val and B.
