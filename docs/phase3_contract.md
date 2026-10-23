# Phase 3 scripted integration contract

This contract is for offline debugging. It does **not** supply missing C1–C4 claim thresholds or authorize a research claim.

- B4, B5 and B5b each spend 12 runs: seeds 11, 23 and 37 with four trials per seed. Each attempted run costs one unit, including failures.
- The shared `Config` dataclass and its `SEARCH_OPTIONS` are the sole sampled configuration space for B4/B5/B5b; B4 uses NSGA-II selection, while B5/B5b use scripted six-role agent calls. B5b receives no observed metric fields in agent tool outputs.
- Approval in the scripted replay is labelled `scripted_human_approval`. Live consequential actions require a separately supplied explicit `HumanDecision`.
- Sealed validator maximum: 10 queries total. Phase 3 debug maps a mock detection-rate delta above +0.02 to `accept/better`, below -0.02 to `reject/worse`, and the middle to `inconclusive/similar`; no result maps to `inconclusive/unknown`. These coarse buckets are test controls, not C1–C4 verdicts.
- The Phase 3 synthetic runner uses the actual hypernetwork trainer and bake path for **all** B4/B5/B5b conditions, a **1% debug FPR target**, A-val threshold calibration, and an unpinned CPU. B4's debug NSGA-II objective uses detection, MACs and model bytes; its research mode requires pinned latency. The surrogate's `proxy_hv_gain_per_cost` is a deterministic detector/cost proxy. Neither debug value is the declared 3D research hypervolume.
- No live model API, production traffic blocking, or actual sealed research dataset is used by the scripted integration run.
