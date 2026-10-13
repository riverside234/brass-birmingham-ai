# Step 5 performance improvement — 13 October 2026

Implemented a measured CPU optimization in `src/brass_ai/recovery/inputs.py`: unchanged industry mats now reuse their validated content when a player's cash, hand count or other scalar changes. Previously, a changed player caused the validator to walk all its tiles again. The original player constraints, mat array bounds, tile schemas, unknown-field rejection and content hashing remain enforced. Any changed mat is revalidated; caches remain bounded by the existing limits.

**All current Step 5 configuration files are byte-for-byte unchanged.** Teacher clocks remain **12/12/16/24 seconds** across the four phases; v4 opponents retain **one second per whole turn**. Workers, shared CPU admission, finalization rules, search frequency, five-game collections, disabled cumulative wall-fraction restriction, inference flush/batch limits, PPO settings, rewards and champion rules are unchanged.

## Why this change

The generation-149 export records 517.6 aggregate actor-seconds validating inputs, versus 119.2 encoding them. These are overlapping per-actor phase sums, not iteration wall time. The export's search/runtime settings predate the current implementation, so its 792-second iteration cannot serve as a before/after benchmark.

A local profile of 311 actual native-game observations found player validation responsible for about 0.99 seconds of a 1.89-second boundary pass. Players' industry stacks change less often than money/hand counters. Splitting their validation at that boundary removes repeated work without changing the model, action distribution or training workload. A dedicated regression checks that a cash-only update does not revisit tile schemas; other regressions compare malformed nested inputs against the original schema.

PPO itself took only 4.28 seconds in the archived generation-149 report. Increasing optimizer batches or changing precision would target a small measured component while introducing different numerical behavior. No such change was made. Current batching, transport, persistent actors, overlapped readers and single-GPU routing were already implemented.

## Measurements

Local evidence is under [`artifacts/step5-performance-20261013`](../artifacts/step5-performance-20261013). It uses Windows CPU, existing dependencies and authored fixture weights, not the lab checkpoint. No installed binaries were changed and no training campaign was run.

An alternating, same-process comparison across the 311 recorded observations measured median boundary-validation time of **3.366 seconds before / 1.834 seconds after**, a **45.5% reduction**. Each pass used a fresh cache and the original implementation saved at task start. See `boundary-ab.json`; initial profiling is in `profile-before.log` and the `.prof` files.

The separate native benchmark uses the real persistent Step 5 actor/inference/replay pipeline, one actor/host, CPU float32 small utility model, two fixed setup seeds and unchanged sampling. Only the validation implementation differs. Each run completed both full games and **619 decisions**. All four runs have the exact same trajectory digest, including permitted views, choices, selected actions, behavior probabilities, context and final results:

`2a6c208a941296a66aab54e7590779a908a99cb4cd561ec2cbe3151375a4a161`

| Order | Implementation | Collection wall time | Actor validation time | Actor Python CPU time |
| --- | --- | ---: | ---: | ---: |
| 1 | Before | 47.602 s | 6.031 s | 12.328 s |
| 2 | After | 43.270 s | 3.609 s | 10.125 s |
| 3 | After | 38.530 s | 2.598 s | 7.000 s |
| 4 | Before | 33.539 s | 4.049 s | 7.969 s |

Validation was **35.8–40.2% lower** and actor Python CPU time **12.2–17.9% lower** in the two comparisons. The initial pair overlapped some regression work; the reverse pair ran without another benchmark/test command. Native apply time also changed substantially between runs. Therefore **no reliable end-to-end training speedup is established**: the first pair improved wall time and the reverse pair did not. The verified result is lower validation/actor CPU cost with identical ordinary-game behavior. Detailed reports, fixture runner and input hashes are retained in the evidence directory; `measurements.json` consolidates the results.

These native measurements used source `45aadaaba1f499add015dbe2c7f7888170c3ef1eaf5745bc1f96d048710aa18a`. The final source below also includes the continuation repair found during regression testing. The measured validator is unchanged; that repair runs when checking checkpoint continuation, outside the timed game-collection path.

Search keeps its existing clocks. Reduced CPU work can free capacity and permit more completed search work within those clocks, but neither effect has been measured on the lab GPU/server yet. No inference about strength or search-label quality follows from these ordinary-game checks.

## Validation and continuation

The optimization is source-bound as:

- Previous: `c875dde1baa158108f2a23d05cf0b60b82b0592462e3c85b33d3d5510a8e6f73`
- Current: `aa41d2a962ded63fb83f89ec4b31fddc47a550ecab0ef8be8654d56262911f46`
- Learning compatibility remains `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`.

The `step5-boundary-cache/1.0.0` review adds an exact same-configuration continuation receipt and carries forward the 98 existing approved routes to the preceding source, retaining their existing configuration-migration definitions. All 735 earlier receipts remain intact. Completed IL from the immediately preceding source is added to the existing compatibility list; its 22 earlier entries remain intact. No broad whitelist or validation bypass was added.

Regression testing also exposed a pre-existing continuation bug in `recovery/resume.py`: after finding an exact approved configuration, the checker continued into obsolete migration alternatives, which could raise an error for today's disabled wall-fraction cap. It now stops when the configuration already matches. The reviewed source/learning identity checks, committed-generation requirement and guards on changed configurations remain in place; no clock or other saved setting is altered by this fix.

New tests cover nested malformed fields/types, changing valid tiles, empty stacks, cache eviction, unchanged original schemas, unnecessary tile traversal, exact current settings, rejected clock/learning changes, and checkpoint/Adam/RNG/counter preservation. Existing replay, feature, PPO, inference, resume and time-budget tests provide broader coverage.

- The three final cache regressions pass. Running the counter-update regression against the saved original implementation produces the expected failure: it revisits unchanged tile schemas. Logs: `performance-regression-after.log` and `performance-regression-before.log`.
- The full Recovery sweep ran **332 tests**: it reported two failures and nine error entries across **five test methods**, with four hardware/OS skips. All five failing methods reproduced against the task's starting source (`preexisting-failures.log`). The fixes address exact-config continuation, historical throughput fixtures that incorrectly borrowed today's phase-clock shape, and a Windows virtual-environment redirector holding the launch directory open. The directory probe now uses the base interpreter for its standard-library-only child; the production launcher is unchanged.
- The corrected focused rerun completed **16 tests: 15 passed, one POSIX-only skip** (`final-targeted-after.log`). The final sweep across **21 affected modules completed 155 tests: 154 passed, one POSIX-only skip**, in 82.0 seconds (`final-resume-regressions-after.log`). It covers continuation, rejected configuration changes, Adam/RNG/counters, champion/evaluation history, time budgets, replay/PPO, cache validation and spawned runtime behavior. The temporary runner includes the required Windows spawn guard; its earlier unguarded attempt is retained separately. The entire 332-test sweep was not repeated after these repairs.
- An initial 17-test batch had one Windows `WinError 5` while replacing a retention cleanup receipt. That unrelated test subsequently passed twice within the broader run; no retention implementation was changed for this optimization. The initial result remains in `boundary-tests.log`.
- Preservation checks compare all **2,946** snapshotted files. The expected existing-file changes are the validator, the two-line resume repair, two continuation registries, two test fixtures and lab documentation. All **24 Step 5 configs**, **170 installed managed files**, and other snapshotted assets are unchanged. Reconstructing the original registry prefixes reproduces their starting byte hashes, confirming all earlier entries were preserved. See `preservation.json` and `verify_preservation.py`.

Because the validator source changed, old derived-feature caches are ignored and authoritative replays are revalidated/re-encoded when loaded. Readers do not rewrite source cache files automatically, so the extra work can recur when an old cache is loaded again; newly generated shards use the current validator receipt. Do not delete or regenerate authoritative datasets to bypass this check. The existing fresh utility-calibration check after a reviewed source transition also remains in force.

Stop the old command after a committed iteration, preserve the run, and update source plus both review registries together. Repeat the existing training command with the **same config, run and IL origin**, adding `--resume-reviewed-source`. Do not switch presets for this optimization. For the currently documented generation-150 gate72 setup, the bounded continuation check is:

```bash
conda activate ai-training
brass_root=/data/not_backed_up/yxu209/ai-training
cd "$brass_root"
python scripts/joint7_preflight.py --root "$brass_root" \
  --run artifacts/step5/v4-selfplay-joint \
  --evidence results/v4-selfplay-joint-g0150-20261013T173011108713Z \
  --config configs/step5/joint7-g150-gate72.json

python scripts/step5.py --root "$brass_root" \
  --config configs/step5/joint7-g150-gate72.json --device cuda \
  train --run artifacts/step5/v4-selfplay-joint \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source --iterations 1
```

Use the actual existing config if different. The exported generation-150 preflight applies to that stopped generation; later generations require evidence matching their state. An uncommitted old-source iteration must be finished/recovered with its matching source before upgrading. No lab command above was executed by this task.

## Next performance measurement

Run matched bounded comparisons on the lab's existing single-GPU allocation and fixed frozen checkpoint/setup list, preserving the current clocks and all workload settings. Compare ordinary games/second, actor validation/CPU time, completed teacher trajectories and qualified rows per phase, inference queue delay, reader overlap and complete iteration wall time. Account separately for first-use cache revalidation and process startup. Use several alternating runs; do not use GPU-memory occupancy as the success metric.

Only pursue another optimization after that measurement identifies the remaining limiting phase. The local profile supports validation caching; it does not justify changing actor counts, batching delays, numerical precision, search support gates or training schedules.
