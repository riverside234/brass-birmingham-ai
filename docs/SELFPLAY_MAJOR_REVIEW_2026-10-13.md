# Remaining self-play bugs — 13 October 2026

Reviewed the active Step 5 pipeline from actor collection through replay admission, PPO and utility calibration, teacher search/distillation, checkpoint continuation, population retention and champion evaluation. Five confirmed bug classes were reproduced and repaired. This follows the [performance work](STEP5_PERFORMANCE_2026-10-13.md); its changes and the pre-existing P0–P6 work were preserved.

All Step 5 configurations remain unchanged. Teacher phase clocks remain **12/12/16/24 seconds**, and searched v4 opponents retain **one second per whole turn**. Rewards, losses, model/feature contracts, schedules, worker counts, inference precision, champion thresholds and workload settings are unchanged.

## Confirmed findings and fixes

All five findings are **P2 (medium priority)**: confirmed correctness or failure-recovery defects, with bounded reproductions below.

| Finding | Reproduction and consequence | Repair |
| --- | --- | --- |
| Replay integrity missed sequence and filtered observations | Checked games accepted reordered/duplicated records. Changing a filtered opponent's debt observation removed a learner's causal insolvency cost without invalidating its checked features. | `recovery/replay.py` binds ordered `(step, seat)` membership. `recovery/updates.py` verifies existing row seals for filtered observations contributing to causal costs. No additional schema traversal is introduced. |
| Initial calibration failure changed optimizer behavior | An exception in the first utility forward pass left LR at `0.001` instead of `0.00003` and froze non-head parameters. | `recovery/utility.py` includes the initial measurement in the restoration block, restoring learning rates and gradient flags on failure. |
| Failed forest submission left workers running | A later submit failure returned while an earlier worker still owned reply resources. A real `Thread.start` failure also left enqueued work without a returned future. | `recovery/parallel.py` shuts down the failed executor, cancels queued work, waits for retirement and drains returned futures before raising the original error. Normal search clocks and evidence handling are unchanged. |
| Actor caches omitted actual execution inputs | Unchanged provenance reused old games after teacher model, rollout opponents, focus, sampling, rotation or execution settings changed. Stale model labels can then fail distillation admission on restart. | `recovery/execution.py` versions/hashes the effective jobs and execution configuration, persists the recipe and verifies it before reuse. Output paths are excluded and equivalent defaults normalize. Healthy partial reuse is preserved. |
| Population refresh could accept changed cache bytes | Full history replaced the recorded slim-checkpoint receipt; refresh then accepted the checksum of changed cache bytes. Champion use had the same gap. | `recovery/retention.py` preserves/verifies committed population and champion receipts. An uncommitted cache must exactly match the trusted source's slim payload, including weights, dtypes, calibration, lineage and RNG. Champion callers provide existing receipts. Mismatches are rejected without overwrite; normal reuse needs no checkpoint deserialization. |

These are correctness and failure-recovery fixes, not evidence of stronger play or faster training.

## Evidence and validation

Evidence is in [`artifacts/selfplay-major-review-20261013`](../artifacts/selfplay-major-review-20261013). Starting source bytes and checksums separate this review from existing workspace changes.

- Learning: four failing assertions before; 17 focused tests passed after. Logs: `learning/before.log`, `learning/focused-complete.log`.
- Forest: both submission failures reproduced before their respective fixes; final 38 concurrency, clock, cancellation and teacher checks passed. Logs: `search/submission-before.log`, `search/submission-thread-start-before.log`, `search/submission-thread-start-after.log`.
- Actor recipe: nine failing assertions across eight tests before; final 17 cache/lifecycle/boundary checks passed. Logs: `runtime/actor-recipe-before.log`, `runtime/actor-recipe-final.log`.
- Population: five failing assertions before; four new integrity/reuse tests pass, including unchanged bytes/RNG and an older champion's identical weights re-entering the population. Logs: `population-before-repro.log`, `population-complete.log`.
- Separate runtime review: 26 focused tests passed for spawned inference routing, private pipes/shared slots, batching, failed replies, shutdown, CPU-admission wakeup and partial replay reuse. Log: `runtime/focused.log`.

The full sequential Recovery sweep completed **352 tests in 1,154.1 seconds: 346 passed, two errored and four skipped**. The skips cover two CUDA tests, an opt-in four-game native screen and a POSIX-only unlinked-directory test. One error was an outdated champion test double; its signature was corrected for the added optional receipt. The other was a Windows `WinError 5` during atomic replacement of a retention journal. That retention test passed on its second appearance in the same sweep. The final focused rerun of both affected areas completed **12 tests, all passed**, in 71.5 seconds. Logs: `full-recovery.log`, `full-recovery-result.json`, `final-rerun.log`. The complete suite was not repeated after the test-double correction.

A bounded native check then completed **two ordinary games / 619 decisions**, followed by an exact two-game cache retry using the real spawned actor, native host, inference service, replay writer and reader. Both cached games retained their original IDs, paths and raw bytes; 154 seat-0 learner rows prepared successfully using the complete causal replay. The trajectory digest exactly matches the earlier fixed-model performance fixture: `2a6c208a941296a66aab54e7590779a908a99cb4cd561ec2cbe3151375a4a161`. This uses authored small CPU weights and one actor/host, with no search or optimizer updates. Total smoke time was 57.0 seconds; this is not a comparative throughput measurement. Evidence: `native-cache/report.json`, `native-cache.log`, `native_cache_smoke.py`.

Intermediate logs retain an initial sandbox temporary-directory failure, champion mocks needing the new optional receipt argument, and transition tests run before final receipts existed. These are not reported as product defects or successful complete runs. The initial population batch's two remaining errors were missing source receipts; its four new population checks passed.

Validation uses Windows CPU and authored fixture checkpoints/replays. GPU routing uses CPU-backed replicas where applicable; actual CUDA and Linux/Mono lab behavior require that hardware. No production training campaign or strength tournament was run.

**Remaining observation:** the intermittent Windows journal-access error has no established root cause. The restore implementation is byte-for-byte unchanged from the starting source, and subsequent executions passed; no retry loop or validation bypass was added to conceal it. An equivalent failure was also recorded in the preceding performance audit. See `retention-io-observation.json`. Passing reruns do not certify that the filesystem issue is resolved.

## Continuation and preservation

- Previous: `aa41d2a962ded63fb83f89ec4b31fddc47a550ecab0ef8be8654d56262911f46`.
- Repaired: `40ef5e2adf1903f4ab6e473599c8c9a50a35c5cfaf5320d6e70fe2c353660a52`.
- Unchanged learning compatibility: `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`.

The `selfplay-major-review/1.0.0` receipt permits exact same-config continuation from the preceding source and carries forward its 99 approved incoming routes with their existing migration definitions. All 834 prior self-play and 23 prior IL entries remain intact; 100 continuation entries and one preceding-source IL receipt are appended. Unknown sources and unreviewed setting changes remain rejected.

Preservation checks cover **2,427 starting workspace files**: only the seven intended production modules, two registries, four test fixtures and lab-documentation link changed. All **24 Step 5 configurations** and **1,359 training assets** in that snapshot are unchanged. All **170 installed managed files** also match the independent snapshot from the preceding performance task. No starting file disappeared. Registry prefixes compare exactly to the archived originals. See `preservation.json` and the task-only `review-source.diff`.

Continuation tests cover all **13 active training presets**; the **24 Step 5 JSON files** also include evaluation and archived configurations. Real fixture checkpoints check weights, nonempty Adam state, RNG, counters, population, champion and scheduling debt. Old partial iterations and old-source teacher labels remain rejected. Fresh utility calibration after a reviewed source transition remains required.

Finish a committed iteration with the old source, then update the repaired source and both review registries together. Repeat the existing training command with the same run, config and IL origin, adding `--resume-reviewed-source`. An unfinished old-source iteration still requires recovery with its matching source before upgrading.

The stronger replay validator invalidates old derived-feature receipts; authoritative replay remains readable and is revalidated/re-encoded. Old provenance-only actor caches remain untouched and readable but cannot prove the complete execution recipe. New-format exact partial collections reuse normally. No dataset or checkpoint is deleted to bypass these checks.
