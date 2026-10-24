# Implementation guide audit (2026-10-24)

This audit compares `implementation (1).md` with the files in this checkout.
Passing a local test gate is distinct from satisfying a research exit gate.

| Phase | Implemented and locally testable | Remaining exit-gate evidence |
|---|---|---|
| 1 | Deterministic loaders, B1/B2 models, MAC matching, split and evasion checks, registry tests; both real NF-v2 CSVs audited and B1/B2 included in the alternate study | Fashion-MNIST debug files, B1 versus logistic sanity, and original chronological/campaign evaluation remain open. |
| 2 | Hypernetwork, gates, baking, int8 C kernel, B2b/B3 replay, bit-exact tests; real-data B2b/B3 points now in the Phase 4 report | Physically pinned latency and original-protocol research frontier remain unavailable on this Windows host. |
| 3 | Six-role scripted cycle with traceable calls, validated Architect/Strategist edits, Coordinator rejection, approval and budget gate, condition-scoped B5 surrogate, B5b blind tools, coarse validator and deterministic replay | The real-data run did not exercise role edits or rejection; the scripted cycle is not a real LLM run or production response. |
| 4 | Both CSVs audited; alternate endpoint-group split and 78-run real-data study completed with offline robustness/autoencoder, once-only sealed evaluation, registry-derived report and [detailed metrics](phase4_detailed_report.md) | The guide's real-LLM cycle, original chronological split, research comparison on a pinned core, and preregistered C1–C4 verdicts remain open. |
| 5 | Offline research dashboard plus passive `tcpdump` capture, packet-derived approximate flow scoring, live traffic-rate chart, recent classifications, and six bounded local rule-based analysis agents | This sandbox cannot bind the demo server or validate interface capture end to end. Packet-derived NF-v2 fields are approximate and unlabeled; no LLM agents or production response are integrated. Clean-machine reproduction and timed dry run remain unverified. |

The supplied guide ends at Phase 5; it defines no Phase 6.

## Checks added in this audit

- The split manifest records SHA-256 for each partition. Phase 4 verifies public
  partition artifacts before writing outputs, and Phase 5 verifies the selected
  run, manifest, and public partition hashes before displaying scores.
- The Phase 4 report checks the registry against the manifest, sealed event,
  12-run budget and seed quotas. It records the fixed objective protocol hash.
  It computes 3D hypervolume only for complete pinned timing on one host/core;
  otherwise the field is `null`.
- The dashboard shows 2D proxy and 3D research hypervolume separately. Its
  on-demand model, traffic, evasion, and autopilot measurements have local
  registry events; generated int8 artifacts have unique names and SHA-256 hashes.
- The C kernel builds and writes binary logits correctly with MinGW GCC on
  Windows. Windows timing remains unpinned and is labelled as debug timing.

See [`phase4_status.md`](phase4_status.md) for this checkout's test result and
the exact commands required to generate a real study report.
