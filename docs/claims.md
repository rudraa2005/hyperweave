# Pre-run commitments

Written before any experiment or test run. No measured performance is asserted here.

- B3, B4, B5, and B5b: **12 training runs each**, counting failures. Each uses seeds `[11, 23, 37]` with four config trials per seed. The same CPU machine and core must be used for timing.
- Operating point: FPR `0.001` (0.1%), threshold calibrated on A-val benign samples only. Empirical FPR resolution and unattainable targets must be reported.
- Hypervolume objectives: minimize `(1 - detection_rate_at_fixed_threshold, normalized_MACs, normalized_latency_ns_per_flow)`. Fixed normalization denominators `(1, 100000, 1000000)` and reference `(1, 1, 1)`; points outside the reference contribute zero. These values are commitments, not observations.
- Sealed validator: **10 total queries**. Output schema: `{verdict: accept|reject|inconclusive, bucket: worse|similar|better|unknown}`. Final sealed evaluation is separate and occurs once.
- Claim C1: functional neurons improve detection at matched deployed MACs. **Thresholds unavailable:** the referenced proposal was not supplied. No supported/refuted verdict can be assigned until its exact supported-if and refuted-if rules are copied here before results are examined.
- Claim C2: generated frontier improves hypervolume/time/storage against B2b. **Thresholds unavailable; verdict locked as inconclusive.**
- Claim C3: agent search improves hypervolume per compute against B4 and B5b. **Thresholds unavailable; verdict locked as inconclusive.**
- Claim C4: context switching lowers cost at equal detection. **Thresholds unavailable; verdict locked as inconclusive.**

Do not fill missing claim thresholds after looking at outcomes. Obtain the proposal and start a new preregistered study if needed.

