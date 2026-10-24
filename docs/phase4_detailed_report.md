# HyperWeave real-data and multiagent workflow report

Generated from the Phase 4 `report.json`, split manifest and SQLite registry by `python -m scripts.generate_detailed_report`. Percentages and counts below are derived from those artifacts; edit the sources and regenerate this file to change them.

## Provenance and evaluation contract

- Study: `runs/phase4_group_20261024_agent_audit`; split: `runs/group_split_20261024`.
- Dataset hash: `7633e17889028702373842cb7355b703a905d6b77d45a81e6f1aba97c0200309`. Source SHA-256 values: nf_cse_cic_ids2018_v2 `16a545a3e139b2e85d08b1b7d9d40328c3689a49d0deaafce6dcf8d7b63e1257`, nf_unsw_nb15_v2 `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`.
- Implementation hash: `306004d9f3138d26bb26e0462753f50acb7083f6b87508639cda58f80913f95b`; current source matches that hash.
- Shared seeds: 11, 23, 37; A-val FPR target: 0.10%; empirical FPR resolution: 0.02%.
- Objective normalization: `[1.0, 100000.0, 1000000.0]`; reference: `[1.0, 1.0, 1.0]`; protocol SHA-256: `db62fb9ddd3b4416e86ef1aafee3ca2b21c1e83f0e117b485591bc450baea9c5`.
- Timing: Unpinned debug timing only; 3D research hypervolume withheld. Kernel timing uses a common bounded B batch; full B is used for detection and bit-exact checks.

| Partition | Rows | Benign | Attack | Groups |
|---|---:|---:|---:|---:|
| a_train | 11,762 | 6,000 | 5,762 | 97 |
| a_val | 11,743 | 6,000 | 5,743 | 37 |
| b | 12,000 | 6,000 | 6,000 | 2,432 |
| sealed | 6,857 | 6,000 | 857 | 2,650 |

## Multiagent workflow audit

The research run used a **scripted offline six-role client**. The role calls, proposal decisions and training actions are real code paths, but no live LLM was called. Approval actor `scripted_human_approval` is a replay label, not evidence of live human review.

| Role | Recorded calls | Wired responsibility |
|---|---:|---|
| Neuron Architect | 24 | Profiles the model; can apply validated architecture edits. |
| Training Strategist | 24 | Reviews prior diagnostics; can apply validated training edits and launches approved runs. |
| Optimization Scientist | 24 | Computes frontier and HV contribution ranking for proposal context. |
| Experimental Scientist | 24 | Emits the typed proposal that selects the experiment. |
| Coordinator | 24 | Can reject a proposal before budget reservation; otherwise requests explicit approval. |
| Adversarial Validator | 6 | Reviews only coarse B-slice validator replies. |

- Agent calls: 126; proposal events: 24; cycle outcomes: 24. Each call carries its proposal ID, condition, prompt, tool result, response, model version and temperature. Recorded client settings: `[('scripted-v1', 0.0)]`.
- Cycle outcomes: `{'completed': 24}`. Applied role config edits: 0; Coordinator rejections: 0. The default script supplied no architecture/training edits and requested review for every valid proposal.
- Budget reservations: `{'b3': 12, 'b4': 12, 'b5': 12, 'b5b': 12}`. Approval actors: `{'predeclared_research_protocol': 24, 'scripted_human_approval': 24}`. B5 and B5b spent 12 runs each; the same seed list appears in the registry.
- Validator queries: 6 of 10 allowed; coarse verdicts: `{'reject/worse': 6}`. The raw effect stays in the private validator store, while agents see only verdict and bucket. These are candidate checks, not C1–C4 claim verdicts.
- B5b blind-tool audit: 63 calls checked; no forbidden metric fields in tool results. B5's surrogate is scoped to past B5 runs, so B3/B4 outcomes cannot seed its predictor.
- Final sealed evaluations: 1; the sealed partition was opened after selection and was not used to tune the original A-val threshold.

### Recorded B5 proposal example

- Hypothesis: Changing neuron width may improve the debug cost-quality proxy
- Proposed config diff: `{'functional_hidden': 2, 'width': 4}`; predicted debug-proxy effect: `0.01`; estimated run cost: `1`.
- Success criterion: B detection and MACs logged under the fixed A-val threshold
- Rationale: Compare a bounded architecture change
- Validation plan: Use at most one coarse validator query per seed

## Held-out B performance

Means are over completed run points. Detection and FPR use each run's fixed A-val threshold; `deployed` means int8 when available and float for B1. The bootstrap interval resamples seed means (at least 1,000 resamples). The B sample is class balanced. Kernel latency is unpinned debug timing on this host.

| Condition | Runs | B detection | 95% seed interval | B FPR | B PR-AUC | Mean MACs | Int8 bytes | Kernel ns/flow | 2D proxy HV | 3D research HV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 9/9 | 3.06% | 0.09%–8.69% | 3.88% | 0.446 | 821.3 | not measured | not measured | 0.2489 | not measured |
| B2 | 9/9 | 9.86% | 0.08%–29.41% | 5.90% | 0.456 | 830.7 | 995 | 1,202.2 | 0.8695 | not measured |
| B2B | 12/12 | 7.47% | 0.01%–22.14% | 3.98% | 0.467 | 979.0 | 1,163 | 1,303.0 | 0.8606 | not measured |
| B3 | 12/12 | 0.04% | 0.00%–0.09% | 1.98% | 0.507 | 526.6 | 650 | 650.1 | 0.0023 | not measured |
| B4 | 12/12 | 2.37% | 0.01%–3.89% | 2.57% | 0.494 | 490.6 | 599 | 634.3 | 0.1540 | not measured |
| B5 | 12/12 | 0.29% | 0.23%–0.35% | 1.12% | 0.501 | 1,242.0 | 1,446 | 1,738.7 | 0.0110 | not measured |
| B5B | 12/12 | 0.29% | 0.23%–0.35% | 1.12% | 0.501 | 1,242.0 | 1,446 | 1,545.4 | 0.0110 | not measured |

B false-positive rates exceed the 0.1% A-val target under dataset shift. Several detector conditions have very low B detection; the measured results do not support a superiority claim. The 2D HV column is a debug proxy, not the preregistered 3D research comparison.

In this scripted run, B5's 2D proxy HV was 0.0110 versus B4's 0.1540. B5 and B5b produced the same B detection/FPR and proxy HV; their unpinned latency measurements differ and are not research-comparable.

### Compute, size and quantization diagnostics

Training seconds are summed over completed runs. Detection gap is int8 minus float in percentage points. Proxy HV per second is a debug ratio using training time only, not full wall-clock compute.

| Condition | Total training seconds | Mean checkpoint bytes | Mean int8 detection gap | 2D proxy HV / training second |
|---|---:|---:|---:|---:|
| B1 | 0.1 | 7,694 | not measured | not measured |
| B2 | 0.3 | 8,321 | -0.011 pp | not measured |
| B2B | 0.2 | 9,548 | -1.758 pp | not measured |
| B3 | 1.2 | 18,448 | -0.007 pp | not measured |
| B4 | 15.3 | 30,988 | -4.524 pp | 0.010043 |
| B5 | 2.1 | 24,644 | -0.064 pp | 0.005206 |
| B5B | 1.0 | 24,644 | -0.064 pp | 0.011080 |

## Once-only sealed results for selected runs

The original threshold still comes from A-val. Recalibration on sealed benign scores is diagnostic only and did not select a model.

| Condition | Selected run | Sealed float detection | Sealed float FPR | Sealed int8 detection | Sealed int8 FPR | Recalibrated detection (diagnostic) |
|---|---|---:|---:|---:|---:|---:|
| B1 | `group_b1_w8_seed37` | 16.69% | 4.22% | not measured | not measured | 0.58% |
| B2 | `group_b2_w16_seed37` | 66.98% | 27.18% | 64.41% | 25.85% | 0.12% |
| B2B | `group_b2b_lambda0_seed11` | 62.54% | 24.62% | 59.28% | 23.97% | 0.70% |
| B3 | `b3_seed11_lambda0` | 20.30% | 6.77% | 15.17% | 5.22% | 0.23% |
| B4 | `b4_seed37_trial1` | 36.06% | 17.15% | 36.17% | 16.77% | 0.70% |
| B5 | `b5_b5_seed11_trial0` | 5.13% | 1.47% | 5.25% | 1.60% | 0.00% |
| B5B | `b5b_b5b_seed11_trial0` | 5.13% | 1.47% | 5.25% | 1.60% | 0.00% |

## Robustness and offline reconstruction

- Offline benign-only autoencoder: device `cuda`, A-train benign rows 2,000; B attack flag rate 17.57%, B benign flag rate 17.28%. It is not in the int8 inference path.
- Sealed autoencoder attack flag rate 64.18%; benign flag rate 15.13%. High benign flag rates are a measured limitation, not an incident alert claim.
- Selected B3/B4/B5/B5b robustness sweeps are in `report.json` under `robustness_selected`. They use the fixed A-val threshold and constrained feature-space evasion. These tests do not establish packet-level safety.

## Remaining research and product limits

- No live LLM client was used in this run. Only the Experimental Scientist's scripted proposal changed the default B5/B5b configuration; other roles supplied logged analysis and gate decisions without a nonempty edit in this script.
- `scripted_human_approval` is replay metadata. The Phase 5 demo separately verified rejection and a labelled local response simulation; it did not block production traffic.
- Agent validator queries use the held-out B slice and return only a coarse verdict. The final sealed partition is evaluated once after selection; it is not an agent feedback channel.
- The NF-v2 CSVs have no verified campaign/time fields. The endpoint group split tests dataset/endpoint shift, not chronological generalization.
- Windows timing is unpinned. The 3D research HV and line-rate cost claim remain unmeasured; latencies should not be compared across hosts.
- Original proposal thresholds for C1–C4 were unavailable before these runs. All four claim verdicts remain inconclusive; setting thresholds now would require a new preregistered study.

## Verification performed in this checkout

- `python -m pytest -q --tb=short`: 87 passed, 1 skipped (Linux core-pinning test on Windows).
- `git diff --check`: passed. Report generation cross-checks the dataset hash, run counts, agent-call lineage, B5b tool fields and once-only sealed event against the registry.
- Phase 5 smoke on this study: generation, 12,000-row B traffic replay, constrained noise evasion on 6,000 attacks, 94-window autopilot, explicit human rejection and approved controlled response simulation completed. The replay has no verified temporal order.
