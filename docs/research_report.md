# HyperWeave-FN: Research and Implementation Report

**Study scope:** network-defense prototype, real NF-v2 flow evaluation, preference/context-conditioned functional neurons, scripted research agents, and a live passive-capture demonstration path.  
**Report date:** 2026-10-24.  
**Evidence source:** `runs/phase4_group_final/report.json`, its split manifest and registry, source tests, and the current implementation. The benchmark JSON was produced with implementation hash `822fb254f084634918cf4197d5543f4fcf457d5d5c3d38740cd0d7a55febf763`; later demo and domain-explorer changes are not part of that historical hash.

## Executive summary

HyperWeave implements a complete research pipeline for training compact detectors, generating preference-conditioned functional-neuron models, pruning and quantizing candidates, running int8 inference, coordinating budgeted experiment agents, and recording evaluations in a SQLite registry. The real-data study processes NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 flow records under an endpoint-group split.

The research result is **negative or inconclusive**. On held-out CSE flows, the selected B3 model has very low detection at the A-val operating threshold; false-positive rates under shift remain well above the 0.1% target. The preregistered 3D hypervolume is unavailable because timing is unpinned, and C1–C4 cannot be adjudicated because the proposal did not provide supported-if/refuted-if thresholds. No superiority claim is supported by this run.

The dashboard now includes passive `tcpdump` capture, packet-derived approximate flow scoring, live-rate visualization, and six rule-based analysis agents. This capture path is implemented and packet parsing is unit-tested, but a full live capture session was not verified in this sandbox. Its features are approximations of NF-v2, so live scores are demonstrations rather than validated attack detections. The domain explorer computes actual generated parameter tensors and response curves; it does not substitute a domain-name embedding that the trained model never learned.

## Research question and model

The central question is whether a hypernetwork can generate compact functional-neuron detectors that trade task loss, shift loss, and inference cost as a preference vector changes, while context bins adapt the generated model to network operating conditions.

For neuron `j`, the implemented conditioner is:

```text
z_j = [e_j, λ_task, λ_shift, λ_cost, c_load / 2, c_alert / 2, 1]
r_j = z_j @ P
(W1_j, b1_j, w2_j, b2_j, o_j, s_j) = split(r_j)
g_j = clip(sigmoid(s_j / τ) * (ζ - γ) + γ, 0, 1)
h_j(x) = w2_jᵀ ReLU(W1_j x + b1_j) + b2_j
f(x) = Σ_j o_j g_j h_j(x)
```

Here `e_j` is a learned neuron embedding; `λ` is a three-objective simplex over task, shift, and normalized cost; `c_load` and `c_alert` are discrete bins in `{0,1,2}`. The implementation uses `γ=-0.1`, `ζ=1.1`, and `τ=2/3`. During export, neurons whose clipped gate is exactly zero are removed. Remaining weights are quantized to int8. The C inference kernel is checked against the Python int8 implementation for exact integer logits.

The context is **load and alert rate**, not a network-domain identifier. A domain change only changes parameters when its measured or selected conditions map to different context bins. If two domains map to the same bins, identical generated parameters are the expected behavior.

## Data and evaluation design

Both NF-v2 sources provide 43 flow features and binary benign/attack labels. The available real-data evaluation uses an endpoint-pair group-disjoint alternate protocol because verified campaign and timestamp metadata are unavailable. It does not establish chronological generalization.

| Partition | Source | Rows | Benign | Attack | Groups |
|---|---|---:|---:|---:|---:|
| A-train | UNSW-NB15-v2 | 11,762 | 6,000 | 5,762 | 97 |
| A-val | UNSW-NB15-v2 | 11,743 | 6,000 | 5,743 | 37 |
| B | CSE-CIC-IDS2018-v2 | 12,000 | 6,000 | 6,000 | 2,432 |
| Sealed | CSE-CIC-IDS2018-v2 | 6,857 | 6,000 | 857 | 2,650 |

The nominal operating point is FPR `0.1%`, with thresholds calibrated using A-val benign rows. Because A-val has 6,000 benign rows, empirical FPR resolution is approximately `0.0167%`. Sealed evaluation is kept separate and performed once after selection. Candidate validator feedback is coarse and comes from B; it is not a claim verdict.

The Phase 4 study contains 78 completed records: B1 9, B2 9, and B2b/B3/B4/B5/B5b 12 each. The four search conditions B3/B4/B5/B5b use 12-run budgets and shared seeds 11, 23, and 37. Bootstrap intervals resample the shared seed results.

## Benchmark results

The following values are means from the current `phase4_group_final` report. B detection, FPR, and PR-AUC are the report's float metrics; quantization detection gaps are shown separately. Kernel times are unpinned debug measurements and must not be compared across hosts. The 2D hypervolume is a debug proxy; 3D research hypervolume is withheld.

| Condition | Completed | B detection | B FPR | B PR-AUC | MACs | Int8 bytes | Unpinned ns/flow | 2D proxy HV | 3D research HV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | 9/9 | 2.68% | 3.43% | 0.445 | 821 | — | — | 0.2152 | — |
| B2 | 9/9 | 9.87% | 6.03% | 0.407 | 831 | 995 | 142.2 | 0.8698 | — |
| B2b | 12/12 | 8.37% | 4.71% | 0.464 | 979 | 1,163 | 167.8 | 0.8696 | — |
| B3 | 12/12 | 0.076% | 2.82% | 0.478 | 527 | 650 | 88.5 | 0.0041 | — |
| B4 | 12/12 | 7.59% | 2.98% | 0.474 | 505 | 615 | 78.4 | 0.8784 | — |
| B5 | 12/12 | 1.14% | 2.52% | 0.490 | 1,242 | 1,446 | 201.6 | 0.1008 | — |
| B5b | 12/12 | 1.14% | 2.52% | 0.490 | 1,242 | 1,446 | 201.7 | 0.1008 | — |

The B false-positive rates range from 2.52% to 6.03%, all materially above the A-val target. B3's smaller model is not a successful detector on B: mean detection is 0.076%. B4 has the largest 2D proxy HV in this report, but that proxy is not the preregistered 3D comparison. B5 and B5b have identical reported B metrics and proxy HV in this run; their scripted search did not demonstrate an agent advantage.

For the selected sealed models, the original-threshold int8 B3 result is 14.59% detection at 5.23% FPR; B2 is 66.28% at 26.85% FPR; B4 is 28.35% at 15.13% FPR; and B5/B5b are 0% at 0.117% FPR. These selected-run outcomes vary widely and do not establish robust generalization. Sealed recalibration is diagnostic only and cannot be used to claim improved detection.

The offline benign-only noise-envelope diagnostic reports B detection 88.25% at 20.55% FPR and did not converge. It is not part of the int8 packet path. Selected robustness tests are feature-space shift/evasion proxies, not packet-level adversarial validation.

## Domain-conditioned function demonstration

The domain explorer uses actual A-train and B-evaluation feature distributions to estimate mean log byte volume and alert rate. In the current checkpoint run, UNSW A-train maps to context `[1,1]` (mean log byte volume `7.942`, predicted alert rate `7.50%`) and CSE B-evaluation also maps to `[1,1]` (mean `6.902`, predicted alert rate `2.51%`). With equal context input, the hypernetwork should produce no parameter difference. The interface reports this instead of manufacturing a domain effect.

The explorer also allows a clearly labelled operating-context scenario. For preference approximately `[0.72,0.18,0.10]`, moving from measured source context `[1,1]` to selected high-load context `[2,1]` produced:

- Generated parameter L2 change `0.08796`, or `2.01%` relative to source parameter norm.
- Gate-vector L2 change `0.03469`; active neurons remained `8 → 8`, so MACs remained `712 → 712`.
- On labeled CSE B evaluation, detection was `0.10% → 0.10%`; FPR was `4.92% → 4.83%`; PR-AUC was `0.3528 → 0.3520`, with each int8 model's threshold calibrated on A-val.

These measurements show that generated weights can change while gate count and MACs stay fixed. They do not show that changing the network domain improves performance. The plotted function slice varies one selected feature over real B percentiles while holding the other features at B medians; per-neuron gate and contribution changes come from the generated tensors returned by the backend. The domain comparison is recorded in the demo action registry.

## Implemented system

| Area | Implemented behavior | Evidence / status |
|---|---|---|
| Data | Fashion-MNIST debug loader; 43-feature NF-v2 validation, encoding, deduplication, group/time split utilities, shift and constrained feature-space evasion | Unit tests; two NF-v2 sources audited. Original chronological protocol remains unavailable. |
| Models | B1 affine, B2 functional neurons, hypernetwork B3, B2b per-preference models, hard-concrete gates, expected MAC cost, preference/context sampling, PHN-HVI ablation | Phase 1/2 tests and registry-backed real-data runs. |
| Deployment | Bake/prune, int8 quantization/export, C kernel and bit-exact Python comparison | Tests and report artifacts. Timing remains unpinned. |
| Agents | Six typed roles, schema validation, run budgets, human approval, coarse validator, B5 surrogate, blind B5b tools, deterministic replay | Scripted calls and tests; not live LLM behavior. Current live demo routines are local, deterministic rule-based analyses. |
| Registry/reporting | Immutable run records, dataset/implementation provenance, validator events, sealed evaluation, report generation from registry | SQLite study artifacts and report scripts. |
| Dashboard | Recorded benchmark, preference generation, autopilot proxy, constrained evasion, agent replay, scoreboard, limits, live capture display, domain function comparison | Offline routes and tests; packet decode and domain analysis tests pass. Live interface capture and localhost server operation were not verified in the sandbox. |

The passive capture adapter parses Ethernet, VLAN, IPv4/IPv6, TCP, and UDP packet headers from `tcpdump`'s pcap stream, aggregates bidirectional five-tuples, and scores updates with the selected int8 weights. It derives counts, IP byte volume, duration, ports, transport protocol, flags/window, TTL, and packet-size statistics. It cannot derive all NF-v2 application features; unknown values are explicitly zero-filled. IPv6 addresses map to zero in IPv4-only training fields. Live packet scores therefore have no validated calibration or attack/benign label. The network-rate graph reports observed captured bytes, not a line-rate guarantee.

The domain explorer compares the trained B3 checkpoint under measured or user-selected context bins. It evaluates generated gates, parameter deltas, MACs, A-val-calibrated int8 thresholds, CSE B metrics, feature response curves, and per-neuron contributions. The live packet path currently uses the selected model context; automatic per-subnet/context model switching during packet capture is future work.

## Multi-agent status

The research workflow uses a scripted client. Role prompts, tool results, responses, decisions, and outcomes are logged, but the research runs did not call a live LLM. The live capture display runs six bounded local routines over actual captured flow updates: architecture profile, normalized feature drift check, MAC/cost report, score/threshold comparison, coarse review status, and coordinator triage priority. This is a real event-driven analysis path, but it is not a language model and does not autonomously block traffic. The coordinator is read-only with respect to network actions.

## Claims and limitations

All C1–C4 verdicts remain inconclusive because supported-if and refuted-if criteria were not supplied before observing outcomes. The 3D hypervolume remains unavailable because core-pinned latency was not produced. Dataset campaigns/timestamps are not verified, so chronological robustness is unknown. The capture adapter has not been feature-parity validated against a reference flow exporter; live classification quality is unknown. The study and demo are research software, not a production IDS or response system.

Other important limitations include class/domain shift, elevated false-positive rates, no measured DPU deployment, no clean-machine timed presentation, no live LLM provider, and no packet-level evasion test. The B5/B5b scripted comparison is not evidence that multi-agent reasoning improves search.

## Future research and engineering plan

1. **Validate flow feature parity.** Compare packet-derived rows against a recognized flow exporter on the same labeled PCAPs. Implement DNS, retransmission, bidirectional direction, and missing-field behavior or restrict the trained model to fields available online. Version the feature adapter and recalibrate only on permitted training/validation data.
2. **Test live detection in an isolated lab.** Use benign and labeled attack PCAP replays or a controlled test network. Report per-class detection, FPR, calibration, confusion counts, drift, throughput, packet loss, and end-to-end latency. Keep the default capture action passive.
3. **Learn context selection.** Validate load/alert bin definitions on development data, then preregister how live domains map to bins. Add per-subnet context tracking and cache-safe model switching only after confirming no label leakage and equal-threshold operating behavior.
4. **Study domain-conditioned adaptation.** Collect multiple labeled network domains for training or a principled unlabeled context estimator. Compare a shared static model, the current two-bin conditioner, and explicit domain-conditioned alternatives using held-out domains. Do not train on the sealed partition.
5. **Evaluate generated neural functions.** Report per-neuron parameter and activation changes, gate sparsity, MACs, model size, output calibration, and paired predictions across a fixed common batch for each preference/context. Add invariance tests: equal conditions yield identical parameters; changed conditions produce the backend-reported function differences.
6. **Run the preregistered study gates.** Recover and record C1–C4 thresholds before new runs, obtain verified temporal metadata or define a new split protocol, pin timing hardware, rerun the full equal-budget comparison, and generate the report from immutable registry artifacts.
7. **Make agent claims testable.** If live LLM agents are required, configure a provider and model explicitly, validate structured outputs, retain tool allowlists/budgets/approval gates, log model and prompt provenance, and compare against scripted and non-agent baselines under equal compute budgets. Until then, describe the current agents as rule-based or scripted according to their actual mode.
8. **Reproduce the demo cleanly.** Verify capture permissions, run a timed live packet demonstration on a clean machine, validate UI/API lifecycle and resource cleanup, and record environment, interface, capture duration, packet counts, and any dropped packets.

## Reproducibility

- Run `make verify-phase5` for the cumulative Phase 1–5 and global gates.
- Launch the dashboard with the study and split paths in `README.md`; select a capture interface only when ready to observe that host's traffic.
- Use the domain explorer's measured-domain comparison first. Then select a context scenario and inspect the returned parameter/gate deltas and response curve. A user-selected scenario is not an observed domain result.
- Treat `runs/phase4_group_final/report.json` as the numeric source of truth for the benchmark table. The test suite validates parameter invariance for equal contexts and nonzero generated changes for a changed context.

## Conclusion

HyperWeave provides a complete, auditable research implementation of conditional functional-neuron generation and a live demonstration path. Its clearest verified contribution is the end-to-end implementation and measurement machinery, not improved network-defense performance. The current real-data results show substantial shift-related degradation. The domain explorer now exposes the actual conditioning math and generated function response, while the passive-capture UI is deliberately identified as approximate and unvalidated until packet-to-feature parity and labeled live testing are completed.
