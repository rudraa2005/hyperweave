# Phase 2 status

`make verify-phase2` runs the Phase 1, Phase 2, and global tests. An offline end-to-end debug replay is available with:

```bash
python3 -m hyperweave.demo.phase2_replay --output runs/debug_01
```

Use a fresh output directory on each run because registry IDs and artifacts are immutable. The replay generates **synthetic** 43-column flow data and uses a **1% debug FPR target** to keep A-val large enough. It runs the actual Phase 1 B2 trainer, loads its checkpoint, trains one hypernetwork, generates three preferences, bakes float and int8 models, runs the C kernel, trains B2b candidates, and writes a registry-backed comparison. The real NF-v2 CSVs are now present; the [Phase 4 alternate study](phase4_status.md) reports their B2b/B3 outcomes.

For supplied numeric NF-v2 CSVs, use `python3 -m hyperweave.baselines.phase2 --help` after a Phase 1 B2 run. Pass the same CSV and ordered feature manifest, the Phase 1 registry and B2 run ID, and the indices of real count/timing columns through `--shift-columns`. `--preferences`, `--context`, `--loss-mode`, and `--dirichlet-concentration` select the generation/training settings. The runner checks seed and dataset hash against the source checkpoint.

The hypernetwork is a linear preference/context conditioner over learned neuron embeddings. Its gate has a differentiable expected MAC cost, an L0-style expected-open probability, and exact zero clipping at export. The `phn_hvi` ablation generates three preferences per batch and differentiates the joint 3D hypervolume numerically with respect to their objective vectors, using a fixed training reference `(2, 2, 1.1)` for `(L_task, L_shift, normalized_cost)`. It follows the multi-sample idea in the [PHN-HVI paper](https://ojs.aaai.org/index.php/AAAI/article/view/25953), though numerical gradients make it a small experimental implementation. Neither it nor the debug replay establishes C1–C4.

The C kernel produces bit-exact logits against the Python int8 reference. Timing includes warmup, 1000 iterations and five repeats. On macOS the harness cannot pin a physical core, reports `pinned: false`, and withholds the research HV comparison. On Linux it pins to the first allowed CPU. Latency is kernel-only, excluding input quantization, file loading and model generation. Never compare its values with a run on another machine.

The alternate study now includes B2b/B3 over three shared seeds and the fixed
2D debug reference. The original Phase 2 research exit gate remains open for
physically pinned timing and the preregistered 3D hypervolume, as well as the
original chronological evaluation. Windows timings in the current report are
explicitly unpinned. B4/B5/B5b equal-budget results are in Phase 4.
