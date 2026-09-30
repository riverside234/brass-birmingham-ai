# Recovery implementation checks

30 September 2026, Windows CPU runtime. This records implementation evidence for the [learning recovery](../ai-research/TRAINING_RECOVERY_PLAN.md) and [throughput recovery](../ai-research/TRAINING_THROUGHPUT_PLAN.md) revisions. Commands for the lab are in [TRAINING_RECOVERY_LAB.md](TRAINING_RECOVERY_LAB.md).

## Focused and regression checks

| Suite | Passed | Coverage |
| --- | ---: | --- |
| Recovery | 13 | Full-legal BC gradient; cached supervised update equivalence; effective PPO steps; microbatch gradient parity; KL stopping; exact curriculum quotas/resume; encoder parity; privacy/model rejection; shared batching/reply routing; propagated search deadline; BC early stop/resume/hash integrity; timing-only RPC byte/return parity |
| Step 5 | 35 | Existing collection/update/evaluation, distribution hashes, frozen references, support/query adaptation, context and Pottery contracts |
| R4/R5 | 16 | Four-seat backup, chance metadata, history/recall separation, complete paired rotations, evidence gates, cache behavior, combined prior/value reuse with private-card and choice-order invalidation |
| R2 | 24 | Original model/features/checkpoint/export compatibility, legal-choice boundaries, all four seats, privacy, replay integrity and losses |
| **Total** | **88** | No strength tournament or lab CUDA qualification implied |

`git diff --check` passes. Qualification-report verification passes for all six admitted policies with **no repaired reports and unchanged fingerprints**. The pinned R2 model/feature code, R3 adapters/runner/bank sources, frozen policies, qualification reports and runtime binaries are unchanged.

## Complete four-player execution

`scripts/verify-recovery.py --integration` completed ten games:

- Two current-v3 demonstration games in distinct train/validation setup partitions, then an exact observation/choice/transition/terminal-result audit.
- One short actual imitation epoch with a compatible fresh small-model initialization.
- Four curriculum games over two training invocations, including a mixed v3 table. Resume retained game/iteration counters and produced a different saved generation.
- Four development evaluation games, rotating the candidate through every seat. No missing games, no promotion, and the source bank was unchanged.

The two PPO rounds performed **22 and 14 optimizer steps**, rather than one step per full epoch. The run's raw record is `artifacts/recovery/exit-1dff0533d9/exit.json`, with source identity `39a827b20c650c99f78c9e629b9abb0ae2cfd219720cb55d40693fc6c61b8d72`. This complete-game run preceded the final telemetry, cached-BC/checkpoint and deadline refinements; those refinements were subsequently verified by the final focused suites and search smoke. Raw evidence retains its original source fingerprint and is not rewritten as a qualification for another source version.

## Final shared-inference search smoke

The current `searchlab-experiment/1.2.0` preset ran one decision each for S2, R4 and S3 on three spawned CPU workers against the same initial four-player setup. The source checkpoint stayed unchanged and all three live selected choices were accepted. The final tiny smoke serviced nine requests with a maximum batch of one and two deadline-expired requests, without invented teacher labels. The focused overlapping-request check verifies a same-model batch of two. Larger workloads are needed to measure useful batch filling. Every search worker also reported its host RPC and pipe-wait timers.

Record: `artifacts/recovery/search-smoke-final.json`. The one-second smoke searches were incomplete; actual elapsed times were approximately 1.22 s for S2, 1.11 s for R4 and 1.22 s for S3. Host/IPC calls cannot be interrupted mid-call. This is wiring evidence, **not an equal-budget advantage**, and no positive-confidence teacher or promotion was produced.

## Remaining measured acceptance

The RTX GPU, Linux/Mono execution, useful batched throughput at 32 live games and improved playing strength require lab runs. The existing R4/R5 device/source gate is deliberately stale after this revision; rerun its integration verifier before longer comparisons or S3 teacher collection. Parallel screening cannot qualify search promotion. No new network is installed into Unity.

Exact per-tile era-end waste/scored-VP attribution is not claimed by the current economic diagnostics; last-observed unflipped tiles are a labeled proxy. DAgger-style state querying, compiled execution, transfer overlap and combined host RPCs remain conditional follow-ups requiring distribution-shift or profiler evidence. The core imitation, revised PPO/curriculum, shared actor/inference pipeline, evaluation and additional R4/R5 paths are implemented.
