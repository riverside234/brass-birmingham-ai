# Second g1190 restart: insolvency charge kept, 256 teacher rows

Date: 2026-11-10. The first g1190 restart ([G1190_RESTART.md](G1190_RESTART.md), run `artifacts/step5/joint7-v11-2-g1190-restart`) ran g1491 to g1500. This restart goes back to g1491 again, in a new run folder, `artifacts/step5/joint7-v11-2-g1190-restart-2`. The first run's folder is not touched.

It starts from the same place as the first one: the exact committed g1491 state with the retained g1190 snapshot (model `d872ea92…`) as the learner, counted as generation 1491.

## Changes against the first g1190 restart

| Setting | First attempt | Second attempt |
| --- | --- | --- |
| Teacher rows per PPO step | 72 | **256** |
| Insolvency charge | none (event cost 0) | **kept: event cost 0.0025, max cost 0.025** |

Everything else is the same: placement 0.1, win 1, VP 0.001 (cap 180) with the canal/final VP timing; 32 teacher tables (+4 held out); long-search labels only; the 4:3:2:1 buffer; two PPO epochs; n=96/K=8 short and n=640/K=16 long searches; one shadow iteration that re-measures beta with the PPO LR held; exploration and cotton; the fresh 256-wide search head (Stage A 50 iterations with the held-out early stop, Stage B 50, Stage C 10, PPG beta floor 0.5 after Stage A); LR multiplier 1.25 carried.

## What the kept charge touches

- **Reward:** `post_transfer_reward` gains `charged(config)`: the restored-placement timed reward with the old charge. It counts as a timed reward (`is_revised`), so the projection back to the reviewed shadow recipe and the timed teacher core selection treat it like the zero-charge one. `keep_charge()` only starts from the restored-placement, zero-charge recipe.
- **Utility:** g1190's utility (placement 0.1, charge 0.0025, no timing) moves to the same reward with the timing added. All tensors are kept and calibration is cleared. The model id changes with the specification. `--skip-utility-qualification` allows beta measurement immediately without waiting for a utility refit. Later PPG phases with zero retained updates preserve the existing utility qualification; a retained update still records its measured result.
- **Entropy clock:** the g1491 state records the applied charge as zero. Its plan already ends at 0.0025, so the restart stamps 0.0025 as applied (only when the clock is complete and at that last step). There is no new reward step.
- **Teacher games:** the timed zero-insolvency teacher core accepts any event cost of zero or more (cap above zero, at most 0.16), so the same qualified build serves these games. The CPU path needs no change.
- **Value rows:** insolvency targets are computed from each game's recorded events with the job's reward, so they follow the charge.

## What 256 rows touches

`training-revision-g1491` gains a second restart variant, `g1190-branch-restart-256-rows`: the g1490 restart values with `rows_per_step=256`. The row count must equal the variant's own value (256 is within the 2048 effective batch). Beta is measured in the shadow iteration at the same 256 rows, so the fixed beta matches. The marker can't change inside a run.

## Commands

Stop any trainer, pull the code on the lab, then:

```bash
python scripts/prepare-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --share-target 0.05
```

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 1491 --iterations 120 --share-target 0.05 --skip-utility-qualification
```

Pass the same `--share-target` to both. The current run uses 0.05. With the skip flag, the first iteration collects the full 32 training + 4 held-out teacher tables and measures beta with 256 teacher rows per PPO step. The PPO LR controller holds and teacher gradients are measured without being applied. Teacher training switches on only after measured beta qualification passes. The flag persists in the run configuration, including across subsequent resumes; reports mark utility qualification as skipped. It does not fabricate a measured utility fit or bypass beta qualification.

The requested fresh attempt uses a new run folder (`...-restart-3`) with the same second-restart recipe. Preparation loads the original retained **g1190 weights again**, and starts the generation counter at **1491**. It preserves the partial run in `...-restart-2`. Beta is measured during g1491→g1492, then teaching switches on if qualification passes. The 120 iterations include beta measurement and finish at g1611. If beta needs another measured iteration, that also counts toward the total. Without the skip flag in either the command or saved configuration, utility readiness remains required.

## Verification (laptop)

- `tests/Recovery/test_g1190_restart.py` (28 tests): the second restart's recipe differs from the first only in the charge (reward, search and teacher search rewards, `insolvency`) and the 256-row variant; its projection, entropy validation, job planning with the kept charge, the shadow→on transition, and refusal of other charges, other row counts and a switch between the two restarts. A run tree built from the real g1491 state for each restart prepares and loads (utility spec, entropy clock charge, rows, 256-wide head). Added numerical checks preserve a nonzero learned utility residual and exact predictions across the seed load and g1491 checkpoint save/load, including rotated seats and different remaining insolvency budgets. Old g1190 and restarted critic targets match in both eras, including capped charges; paid canal VP is subtracted only in the future-return coordinate.
- `tests/Recovery/test_skip_utility_qualification.py`: real teacher planning and native job validation with missing/failed utility calibration, persistence across resumes, full measured beta still required, and the command sequence from g1491 (one beta-measurement iteration, then 119 teaching iterations if qualified). Focused PPG tests reproduce zero retained steps after held-out rejection, preserve true/false/missing calibration appropriately, and check checkpoint round trips without adding model/optimizer copies.
- Broader checks encountered seven existing failures; all seven reproduce with the original HEAD source and review metadata: two missing g660 export fixtures, two historical source-transition expectations, two rollback-manifest path checks, and one outdated error-message expectation. No rules were loosened for those failures.
- The first-restart and g1490 tests still pass.
- No Rust change and no native test run for this change; the zero-insolvency core's charge check was read, not executed with this reward. Not run on the lab.

The pasted g1495 log's `historical:-0.368vs0.423[insufficient_support]` compares historical-table EV with self-play EV. It has five historical held-out games from one contributing iteration and zero retained PPG updates. It is measured after PPO (46 steps in that iteration), not at the instant the g1190 weights are loaded. The restart preserves the utility-head tensors and prediction coordinate; these values alone do not establish a loading or reward bug. The exact lab g1190 checkpoint and failed-run replay are not present in this laptop checkout, so the numerical regression uses an authored checkpoint with a nonzero utility head; the full restart fixture verifies tensor preservation through preparation and resume.
