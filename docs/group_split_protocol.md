# Preregistered alternate NF-v2 evaluation split

Declared before reading model outcomes. The downloaded CSVs contain no verified
campaign or timestamp field. The user authorized a group-disjoint alternative.
This protocol evaluates endpoint and dataset shift; it does not test chronology.

- Canonical 43-feature manifest: `configs/nf_v2_features.txt`.
- Group key: unordered pair of unsigned 32-bit source/destination IPv4 values.
  Reverse traffic uses the same key. Mix the pair with fixed 64-bit constants
  and salt `20261024`, then take modulo 100. Code: `hyperweave/data/group_split.py`.
- UNSW buckets 0–69 supply A-train; UNSW buckets 70–89 supply A-val.
- CSE buckets 90–94 supply B shift validation; CSE buckets 95–99 supply the
  once-only sealed evaluation. The same hash partition applies to both files,
  so an endpoint pair cannot cross partitions even if present in both datasets.
- Select at most 6,000 benign and 6,000 attack flows per partition using the
  lowest deterministic row-rank hashes over the **entire** CSV. Deduplicate
  selected feature vectors; refuse conflicting labels, group overlap, or a
  partition lacking either class. No flow-level random split or row-order
  cutoff is used. Checkpoint the selected rows and exact source SHA-256 hashes.
- For the bounded training budget, draw up to 2,000 flows per class from
  A-train with RNG seed `20261024`. This is training subsampling inside the
  already assigned group partition; A-val, B and sealed retain their selected
  rows. Fit the log/standardization transform on this training subset only.
  The run dataset hash covers the split manifest and this subsampling rule;
  every registry record also carries a SHA-256 hash of the training/evaluation
  Python sources, C kernel and canonical feature manifest.
- Seeds `[11, 23, 37]`, FPR target `0.001`, threshold calibrated on A-val
  benign scores only. Report empirical FPR resolution. B selects proposals;
  sealed is read by the final evaluator once and never exposed to agents.
- B3/B4/B5/B5b: 12 run attempts each, four per seed. Failures count against
  budget. Objective normalization and reference follow `docs/claims.md`.
- Candidate selection, B4 search feedback and B-slice agent diagnostics use
  int8 detection/FPR where an int8 export exists. Float scores and the
  float-to-int8 gap are reported separately. B1 has no int8 kernel export.
- Research latency and 3D hypervolume require a physically pinned core. On an
  unpinned host, report debug timings separately and leave research HV blank.
- C1–C4 verdicts remain inconclusive because the original supported/refuted
  thresholds were not supplied. Do not choose them after seeing outcomes.

The hash bucket is a group assignment, not a claim that these datasets have
independent attack campaigns. The sampling cap makes this a bounded study;
full-file performance must not be inferred from it.
