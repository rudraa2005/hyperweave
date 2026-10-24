# HyperWeave-FN (Network Defense Edition): Implementation Guide

A phase-by-phase build plan with a concrete test gate for each phase. Total budget: 8 hours. Prioritize a complete, auditable loop over a large model zoo. Nothing here presumes results; every performance claim is a hypothesis.

**Revision contract:** Phase 1 below is preserved verbatim because it is already implemented. Phases 2–5 retain the original research deliverables, baselines, claims, tests and exit gates; additive MVP integration gates make the IT Infrastructure business workflow demonstrable without reducing the research ambition. Treat the MVP as a reliable path through the research system, not a replacement for it.

## 0. Ground rules (before writing code)

**Write down before running anything**
- Compute budget for B3/B4/B5: fixed number of runs OR fixed wall-clock minutes (same for all three).
- Seeds: 3 or more, shared across all conditions.
- Target FPR for the operating point (default 0.1%).
- Hypervolume reference point and objective normalization (documented, fixed).
- Claim thresholds C1 to C4 (supported-if / refuted-if), copied from the proposal into `docs/claims.md`.
- Validator query budget (e.g., 10 queries total) and the verdict format (accept / reject / inconclusive + coarse effect bucket).

**Repo layout**
```
hyperweave/
  data/            # loaders, dedup, splits, shift families, evasion
  models/          # affine MLP, functional neuron, hypernet, gates, bake
  kernel/          # int8 inference kernel (Go or C) + timing harness
  agents/          # schemas, tools, six agents, coordinator
  baselines/       # B1, B2, B2b, B3, B4 (NSGA-II / TPE), B5b
  registry/        # SQLite or JSON experiment registry
  eval/            # metrics, hypervolume, sealed validator
  demo/            # web UI + replay
  tests/           # one folder per phase
  docs/claims.md
  configs/
```

**Global conventions**
- One `config.yaml` schema shared by every condition (this is the config space B4 and the agents search).
- Every run writes an immutable record: config, seed, dataset version hash, git commit, metrics, checkpoint path.
- Fixed seeds via a single `seed_everything()`; CPU determinism flags on.
- Pin the timing machine and core; never compare latency across machines.

**Global tests (run in CI from day one)**
- `test_determinism`: same config + seed gives identical metrics.
- `test_registry_immutable`: writing to an existing run ID fails.
- `test_no_test_leak`: sealed-environment paths are not importable from `agents/` (static check).

---

## Phase 1: Establish baselines (hours 0 to 2)

### Deliverables
1. **Tier 1 (debug):** Fashion-MNIST loaders with fixed train/val/test partitions and deterministic corruptions (Gaussian noise, blur, brightness, occlusion at 3 severities).
2. **Standard MLP (B1)** with a MAC counter.
3. **Functional-neuron layer:** each neuron is `g(x; θ) = w2ᵀ σ(W1 x + b1) + b2`, tiny MLP with hidden size `h`, ReLU σ. Fixed (non-generated) parameters = B2.
4. **Tier 2 (application) pipeline:**
   - Loaders for NF-UNSW-NB15-v2 and NF-CSE-CIC-IDS2018-v2 on the shared 43-feature schema.
   - Binary labels (attack vs benign).
   - Deduplication and label sanity checks.
   - Campaign/time-aware splits (never random by flow).
   - Environments: A (train + val), B (shift-validation slice for agents), sealed (third dataset if one exists, else untouched B remainder split by time/campaign).
   - Shift families (rescaling byte/packet counts, timing jitter, feature dropout), split into train-families and eval-families.
   - Constrained evasion harness: mutable-feature mask, immutable fields fixed, cross-feature consistency, drift-toward-benign sweep, projected white-box gradient attack.
   - Training augmentation must use stochastic, seeded count rescaling, timing jitter and optional derived-feature dropout on raw NF-v2 feature groups. Keep evaluation shift family names and parameter ranges distinct, and verify endpoint/protocol immutability and byte/packet relations.
   - Feed bounded noise, jitter, padding and drift-toward-benign attack examples into the functional-neuron/hypernetwork shift objective. Evaluate each evasion type separately; feature-space tests are a proxy for packet-level attacks.
5. **Metrics module:** detection rate at fixed FPR (threshold set on A-val), PR-AUC, recalibrated-threshold variant.
6. **MAC matching utility:** given a target MAC budget, search width of B1 so deployed MACs match B2 (neuron with `d` inputs costs `d*h + h` MACs).

### Tests
| Test | Pass criterion |
|---|---|
| `test_fmnist_corruptions_deterministic` | Same corruption + severity gives byte-identical arrays across runs |
| `test_mac_counter_affine` | Counter equals analytic `sum(d_in * d_out)` for B1 |
| `test_mac_counter_functional` | Counter equals `sum(d*h + h)` per neuron for B2 |
| `test_mac_matching` | B1 and B2 deployed MACs within 2% at 3+ budgets |
| `test_split_no_campaign_overlap` | No attack campaign/time window appears in both train and eval splits |
| `test_dedup` | No duplicate flows within or across train/eval splits |
| `test_schema_alignment` | Both datasets have identical column names and order |
| `test_threshold_calibration` | On A-val, the chosen threshold yields FPR within tolerance of the target (e.g., 0.1% ± 0.02) |
| `test_shift_family_disjoint` | Train-time and eval-time shift families share no member |
| `test_evasion_constraints` | Immutable features unchanged after attack; cross-feature relations (e.g., bytes ≥ packets × min size) still satisfied |
| `test_evasion_monotone` | Attack at ε=0 leaves predictions unchanged; detection rate is non-increasing as ε grows (on a trained toy model) |
| `test_affine_is_not_functional` | Functional neuron with `h` hidden units is NOT reducible to an affine+sigmoid (rank/linearity probe) |

### Exit gate
- B1 and B2 train reproducibly on both tiers, with checkpoints and a registry entry.
- Reported: detection rate @ FPR, PR-AUC, MACs, params, float latency (placeholder OK until Phase 2).
- Sanity: B1 beats a trivial baseline (majority class / logistic regression) on tier 2.

---

## Phase 2: Add hypernetwork, gates, baking, int8 kernel (hours 2 to 3.5)

### Deliverables
1. **Hypernetwork `H_φ(e_j, λ, c)`** outputting `[θ_j, s_j]` per neuron; learned neuron embeddings `e_j`.
2. **Fan-in-variance initialization** of the last hypernetwork layer.
3. **Hard-concrete (L0) gates** producing exact zeros; pruned neuron removes its outgoing connections.
4. **λ and c fixed per batch;** discrete context bins (link load × alert rate) with generation once per bin.
5. **Objectives:** `[L_task, L_shift, C_infer]`, with differentiable `C_infer` = expected MACs over gate probabilities; Dirichlet λ sampling per batch.
6. **Loss modes:** linear scalarization (default) and PHN-HVI as a switchable ablation.
7. **Bake step:** generate params once for (λ, c), drop pruned neurons, quantize to int8, export a plain network file.
8. **Int8 inference kernel** (Go or C): loads the baked file, runs a batch of flows; timing harness pins one core, warms up, reports median and p99 ns/flow.
   - Keep shift sampling, evasion generation and reconstruction diagnostics offline. The deployment kernel accepts only a baked int8 model and its fixed input normalization; do not add a per-packet autoencoder or diagnostic branch.
9. **B2b:** K separately trained functional models, one per preference.
10. **B3:** hypernetwork-generated model on a fixed schedule.

### Tests
| Test | Pass criterion |
|---|---|
| `test_hypernet_init_variance` | Generated weights at init have variance ≈ 1/fan_in (within 20%) |
| `test_hypernet_shapes` | Output param count matches neuron spec for every layer |
| `test_gate_exact_zeros` | After training with L0 pressure, some gates are exactly 0 in eval mode |
| `test_gate_prunes_outgoing` | Pruned neuron's outgoing contributions are removed from baked graph |
| `test_lambda_changes_cost` | Higher cost weight in λ yields strictly lower mean expected MACs (averaged over seeds); this is the core "cost responds to λ" check |
| `test_context_changes_params` | Different bins `c` produce different generated parameters |
| `test_generation_once_per_batch` | Hypernet forward count per batch equals 1 (per bin), not per sample |
| `test_bake_equivalence_float` | Baked float model outputs match the generated model within 1e-5 |
| `test_bake_drops_neurons` | Baked MACs equal true count over surviving neurons |
| `test_int8_gap` | Detection-rate delta (float vs int8) is measured and logged per frontier point; fail only if the pipeline can't compute it |
| `test_kernel_vs_python` | Int8 kernel logits match a Python int8 reference exactly (bit-exact) |
| `test_kernel_timing_protocol` | Harness reports median and p99 over ≥ N iterations after warmup on a pinned core; variance across 5 repeats is reported |
| `test_phn_hvi_runs` | PHN-HVI mode trains for a few steps without NaNs |
| `test_hypervolume_correct` | HV of a hand-built 2D and 3D point set matches the known analytic value |
| `test_hypernet_no_divergence` | Loss finite and bounded over a short run for 3 seeds |
| `test_heldout_lambda_split` | Held-out preference vectors are absent from training samples |

### Exit gate
- One hypernetwork produces ≥ 3 distinct baked models at different λ with measurably different MACs and ns/flow.
- Float→int8 gap and ns/flow measured for each.
- B2b and B3 both produce a frontier (even if poor) with hypervolume under the fixed reference point.


### MVP integration requirement (additive; research scope unchanged)
The MVP must exercise the actual Phase 1 detector and the actual Phase 2 model-generation path where implemented. It must expose one reproducible comparison between a fixed model and a hypernetwork-generated/baked candidate at two or more preference settings (λ), using the same input batch and recording detection metrics, MACs, model size, and measured latency where available. Do not present a mocked model or unmeasured latency as a real result; clearly label any temporary placeholder. The UI may use a small, deterministic replay dataset to keep the demo fast, while the full research datasets and evaluation protocol remain unchanged.

### MVP test gate
- A single command or documented sequence loads a known checkpoint, generates a candidate for a selected (λ, c), and runs inference.
- The generated/baked model is traceable to its config, seed, dataset version and checkpoint/export artifact.
- The interface can show the measured accuracy/detection-versus-cost comparison without claiming superiority unless the results support it.


---

## Phase 3: Implement agent tools (hours 3.5 to 5)

### Deliverables
1. **Registry:** SQLite or JSON; stores hypotheses, configs, seeds, dataset versions, metrics, frontier snapshots, decisions, prompts, model versions, sampling settings.
2. **Typed proposal schema** (JSON Schema): hypothesis, config diff, numeric predicted effect, cost estimate, success criterion. Invalid proposals rejected with no budget spent.
3. **Typed tools** per agent:
   - Coordinator: approval, budget allocator, run-state query, rollback.
   - Neuron Architect: architecture generator, parameter counter, module profiler (true MACs, gate sparsity, int8 size).
   - Training Strategist: config editor, launcher, diagnostics (can switch to PHN-HVI, set Dirichlet concentration, choose shift families).
   - Optimization Scientist: Pareto analyzer, hypervolume evaluator, candidate ranker (by HV contribution).
   - Experimental Scientist: planner, queue, comparator, budget estimator, **surrogate predictor** of HV gain per cost (agent may overrule with stated reason).
   - Adversarial Validator: sealed evaluator with query budget, shift/evasion tests, latency profiler, regression checker.
4. **Sealed environment:** replies accept / reject / inconclusive + coarse effect bucket only; raw numbers never returned; query counter enforced.
5. **Replay mode:** logged prompts + tool results can be replayed deterministically without calling the model API.
6. **B4:** NSGA-II or multi-objective TPE over the same config space and budget.
7. **B5b:** evidence-blind agents (same tools and budget, results withheld).

### Tests
| Test | Pass criterion |
|---|---|
| `test_proposal_schema_valid` | Valid proposals pass; each of the required fields missing yields rejection |
| `test_invalid_proposal_costs_nothing` | Budget counter unchanged after a rejected proposal |
| `test_budget_enforced` | Coordinator refuses an approval that exceeds remaining budget |
| `test_validator_query_budget` | (N+1)th sealed query is refused |
| `test_validator_coarse_output` | Response contains only verdict + bucket; no raw metric floats (regex/schema check) |
| `test_agents_cannot_read_sealed` | Agent tool whitelist excludes sealed data paths; filesystem permission test |
| `test_tool_typing` | Each tool rejects wrong-typed arguments |
| `test_registry_logs_prompt_and_model` | Every agent call records prompt, model version, temperature, tool results |
| `test_replay_deterministic` | Replaying a recorded cycle reproduces identical decisions and registry state with the API disabled |
| `test_surrogate_trains` | Surrogate fits past runs and returns a prediction with an uncertainty or fallback when data is sparse |
| `test_prediction_logged_for_calibration` | Every approved proposal has a stored numeric prediction; calibration score computable after the run |
| `test_b4_same_config_space` | B4's sampled configs validate against the same schema the agents use |
| `test_b5b_blind` | B5b agents receive no metric fields in tool outputs (mock check) |
| `test_agent_decision_recorded` | A scripted mock-LLM run produces ≥ 1 recorded experiment decision caused by agent output |

### Exit gate
- With a mock/deterministic LLM, a full proposal → select → approve → run → validate cycle completes and is logged.
- B4 and B5b run end to end under the same budget accounting as B5.


### Business workflow and MVP integration (additive; six-agent research design unchanged)
Position the application in **IT Infrastructure / enterprise network defense**. The user-facing workflow must make the business process explicit:

1. **Data ingestion:** load a dataset or replay of network-flow records through the existing Phase 1 pipeline.
2. **Intelligent decision:** the detection model scores flows; agents inspect recorded metrics/context and propose a model/configuration experiment or a safe incident-response recommendation.
3. **Human oversight:** the Coordinator presents the proposed action, rationale, predicted effect, compute cost, success criterion and validation plan. Approval/rejection is explicit and recorded. The execution tool must refuse unapproved consequential actions and over-budget runs.
4. **Action execution:** after approval, execute the actual permitted tool action (for example, launch a training/evaluation run, bake a model, or run a controlled response simulation). Do not claim production traffic blocking or remediation unless genuinely integrated and tested. A simulated response must be visibly labelled as simulated.
5. **Outcome reporting:** store the action status, validator verdict, measured metrics and registry/run ID; show the outcome in the demo.

The six research agents and their roles remain as originally specified. This is an integration requirement, not a replacement architecture or a reduction in the research claims.

### MVP test gate
- An end-to-end proposal → validation/schema check → human approval or rejection → execution gate → action/run → validator verdict → registry/report cycle works at least once.
- Rejection prevents execution; invalid proposals spend no budget; over-budget actions are refused.
- The registry records the proposal, agent/tool decisions, approval status, action result, metrics and model/version identifiers.
- If a live LLM/API is unavailable, replay a previously recorded cycle; label replay clearly and do not describe it as a live agent run.


---

## Phase 4: Close the loop (hours 5 to 6.5)

### Deliverables
1. Run the full seven-step cycle with real LLM agents, one or more cycles, under the pre-declared budget.
2. Run B3, B4, B5, B5b under identical budget, splits, seeds, hardware.
3. At least one Validator verdict recorded, including (ideally) a rejected claim with reason.
4. Experiments for the claims:
   - **C1:** matched-MAC curve, B1 vs B2 across several MAC budgets.
   - **C2:** B3 vs B2b on hypervolume, training time, bytes per frontier point; held-out λ interpolation.
   - **C3:** B5 vs B4 and B5b on hypervolume and hypervolume per compute second.
   - **C4:** autopilot stream replay, static vs context-conditioned switching at equal detection rate.
   - Quantization gap per frontier point.
   - Offline benign-only autoencoder: train on A-train benign flows with GPU acceleration when available, calibrate reconstruction-error threshold on A-val benign flows, and report B anomaly flags without feeding B labels or sealed data into training. Report device, artifact and failure cases; test whether rigid but manifold-close records escape reconstruction detection.
   - Offline B robustness sweep: held-out shift families and bounded noise/jitter/padding/drift evasion at the A-val detector threshold, with immutable fields and physical constraints checked. Store measured effects in the registry without adding work to the int8 kernel.
5. Bootstrap intervals; label any single-seed result preliminary.

### Tests
| Test | Pass criterion |
|---|---|
| `test_budget_equal_across_conditions` | B3, B4, B5, B5b consumed identical declared budget (runs or wall-clock) within tolerance |
| `test_seed_sharing` | All conditions use the same seed list |
| `test_frontier_nondominated` | Reported frontier contains no dominated point |
| `test_hv_fixed_reference` | All conditions' HV computed with the same reference point and normalization (hash check) |
| `test_bootstrap_intervals` | Intervals computed with ≥ 1000 resamples; deterministic given seed |
| `test_registry_complete` | Every run, including failures, has a registry entry; no gaps in run IDs |
| `test_no_cherry_pick` | Final report is generated from the registry by script, not hand-edited numbers |
| `test_sealed_opened_once` | Final sealed evaluation executes exactly once and is flagged in the registry |
| `test_threshold_not_tuned_on_sealed` | Threshold provenance points to A-val only |
| `test_autopilot_equal_detection` | Cost comparison only reported where detection rates match within a declared tolerance |
| `test_heldout_env_scored` | Detection at original and re-calibrated thresholds both reported on held-out environment |
| `test_claim_verdicts_scripted` | C1 to C4 verdicts computed from the pre-written thresholds in `docs/claims.md`, not rewritten after seeing results |
| `test_shift_evasion_offline` | Training and evaluation shift families are disjoint; evasion stays within its feature/epsilon constraints and uses the fixed A-val threshold |
| `test_autoencoder_provenance` | AE trains only on A-train benign flows, calibrates only on A-val benign flows, and does not appear in the int8 kernel |

### Exit gate
- Scoreboard table for B1, B2, B2b, B3, B4, B5, B5b with intervals; claims marked supported / refuted / inconclusive per the pre-declared rules.
- Negative results kept and reported.


### MVP evidence and business outcome (additive; research evaluation unchanged)
The MVP should report a compact, registry-derived outcome for the chosen run: traffic processed, detection rate at the declared FPR, PR-AUC where computed, false-positive rate, model MACs/size, measured latency, selected action, human approval status and validator verdict. Only show metrics actually produced by the pipeline; use “not measured” rather than invented values. Keep the full B1/B2/B2b/B3/B4/B5/B5b comparisons, fixed budgets/seeds/splits, hypervolume analysis, bootstrap intervals and C1–C4 claim-verdict process as the research track.

### MVP test gate
- At least one completed workflow has an auditable record from input through decision and execution to outcome.
- A rejected or inconclusive validation is shown honestly if it occurs; do not manufacture a rejection merely for presentation.
- The report is generated from the registry, not manually typed metrics. Research claims remain supported, refuted or inconclusive under the predeclared criteria.


---

## Phase 5: Demonstrate (hours 6.5 to 8)

### Deliverables
1. Five views: traffic console, posture slider + autopilot, evasion test button, agent notebook, scoreboard.
2. Pre-recorded agent run replayed from logs; optional single live cycle only if the API is reliable.
3. Limits slide / panel: what was not tested, research prototype, feature-space evasion is a proxy, no DPU measured.
4. Model diagram and the 3-minute script.

### Tests
| Test | Pass criterion |
|---|---|
| `test_slider_generates_without_retrain` | Moving λ triggers generation only; no optimizer step called |
| `test_slider_monotone_cost` | MACs and ns/flow counters decrease (on average) as cost weight rises |
| `test_pruned_neurons_greyed` | UI graph state matches baked gate mask |
| `test_evasion_button_constrained` | Button invokes the constrained attack (not unconstrained) |
| `test_replay_matches_log` | Demo replay output equals logged trace byte for byte |
| `test_demo_offline` | Whole demo runs with network disabled |
| `test_notebook_has_rejection` | Agent notebook includes ≥ 1 rejected claim with reason (or states none occurred) |
| `test_scoreboard_from_registry` | Displayed numbers equal registry-derived values |
| `test_limits_visible` | Limits panel is reachable in one click |
| Manual dry run | Full 3-minute script timed (≤ 3:00) with no live dependency failure |

### Exit gate
- End-to-end demo runs from replay on a clean machine.
- Every number on screen traceable to a registry record.


### MVP-first presentation contract (additive; all five views retained)
The five original views remain: **traffic console, posture slider + autopilot, evasion test button, agent notebook, scoreboard**. Organize them so a judge can follow one clear business scenario without removing research depth:

1. **Business problem (15–20 seconds):** an IT team needs to detect network attacks while balancing detection quality and inference cost under changing conditions.
2. **Ingest and detect:** replay/load network flows and show the model's predictions and operating-point metrics.
3. **Research novelty:** change λ/context, generate or select a model without retraining where the implementation supports this, and show measured MAC/latency/quality changes.
4. **Agentic decision:** show the actual proposal, predicted effect/cost, tool calls and rationale from the six-agent workflow.
5. **Human approval and execution:** approve or reject; demonstrate the enforced gate and an actual run or explicitly labelled safe simulation.
6. **Validate and report:** show the validator verdict, registry-backed outcome, and limitations.

Use a recorded deterministic replay as the reliable default and an optional live cycle only when reliable. A replay demonstrates reproducibility, not a live API interaction. Preserve the model diagram, full research scoreboard, limits panel and original three-minute time cap.

### MVP acceptance gate
- A clean-machine/offline replay demonstrates data ingestion → intelligent decision → human approval → action execution → outcome reporting.
- The demo shows at least one actual model/agent result from the implementation; any mocks or simulated actions are explicitly labelled.
- The first screen and narration identify the IT Infrastructure business problem and explain the measurable outcome.
- Every displayed metric traces to a registry record; limitations and untested claims remain visible.


---

## Suggested test tooling
- `pytest` with markers `@phase1 ... @phase5` so you can run `pytest -m phase2`.
- `hypothesis` for property tests (gate monotonicity, evasion constraints).
- A `make verify-phaseN` target per phase that runs that phase's tests plus the global tests.
- A mock LLM client (fixed scripted outputs) for all agent tests; real model only in Phase 4.

## Risk checklist (check at each phase exit)
- [ ] Is the hypernetwork's last layer initialized for fan-in variance?
- [ ] Does cost actually change with λ (gates working), not just accuracy?
- [ ] Are baselines matched on deployed MACs, not just params?
- [ ] Any split by random flow? (should be none)
- [ ] Any agent path to sealed data? (should be none)
- [ ] Latency measured on the same machine, pinned core, with warmup?
- [ ] Are claim thresholds still the ones written before the runs?
- [ ] Are failed and negative runs in the registry?

## If time runs short (cut order)
1. Drop PHN-HVI ablation.
2. Drop the third environment; use time/campaign-split remainder of B as sealed.
3. Reduce seeds to 3 and subsample large datasets in the agent loop.
4. Replay-only demo.
Never cut: matched-MAC baselines, sealed validator, B4, B5b, registry, limits statement.
