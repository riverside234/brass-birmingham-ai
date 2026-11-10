# g1294 review and training revision

Date: 2026-11-09. Evidence: `results/joint7-v11-2-g1290-trained-restart-g1294-20261110T000540895274Z`. Both exported iterations (1292, 1293) are the first iteration of a fresh process (different PIDs). Both ran overlapped collection with Gumbel c_scale 1.

## 1. What the profile shows

| Phase (s) | g1292 | g1293 | Note |
| --- | ---: | ---: | --- |
| Planning before the pool | 44.0 | 40.2 | Bug, fixed below |
| Ordinary pool | 48.4 | 50.5 | Started only after planning |
| Teacher tables | 54.5–62.4 | 52.6–65.3 | 28 tables beside ordinary games |
| Teacher wait and admission after the pool | 3.7 | 7.3 | |
| Replay join | 3.8 | 3.3 | Was 12 s before streaming seals |
| PPO | 36.1 | 46.2 | g1292 stopped at 12 steps on its KL target |
| PPG | 6.7 | 8.1 | No retained steps in either |
| Iteration | 146.6 | 165.6 | Fresh-process iterations |

### Planning bug (fixed)

Planning re-installed and re-verified the local-scoring host once for each of the 320 games: about 230 path-containment checks each, 73,000 in total. Replaying `plan_collection` on the exported state under a profiler put 254 s of 256 s there (laptop, with profiler overhead); on the lab it was 40 s. Every game carried the identical receipt. The plan now verifies the host once and copies the receipt into each game, so the job bytes are unchanged.

While planning ran, the teacher tables already held 28 CPUs and the other 36 sat idle. With the fix, ordinary games start about 2 s after the tables, so collection should end with the longest teacher table (about 60–65 s, plus about 5 s of admission) instead of after 91 s plus admission. That is an estimate; the next export will show it.

### Early PPG cache write (removed)

Last session moved the PPG cache write onto a thread beside PPO. The writer is a Python loop, so the GIL serialized it with PPO's own staging: `ppo.freeze_values` went from 4.7–8.1 s (g1288–g1291) to 9.1–13.6 s, about the 5.9 s that PPG saved. It gained nothing and added a thread, so PPG writes the cache itself again, as before.

### Remaining costs, not changed

- Teacher tables are now the critical path of collection. Beside ordinary games they take about 58 s median, against 45–51 s when they ran alone, most likely because the clock drops when all cores are busy.
- PPO staging is 15 s of main-thread Python (per-row writes of about 8 GB). A single joined write was slower in a benchmark, so this was left alone.
- PPO forward/backward is 15.5 s for 46 steps; batch sizes and precision are numerics and stay as they are.

## 2. Finding: PPG almost never keeps a step

In 10 of the 12 iterations since g1278, the PPG phase kept zero steps. In every one, the first 16-step interval raised the pooled held-out loss slightly (for example 1.1185 → 1.1234 at g1292), and the strict rule `loss_after > loss_before` rejected it. The critic is therefore fitted almost only by PPO's preservation term. Changing this gate is a training decision, so it was left unchanged. The KL-cap change below does not affect it.

## 3. Training revision (implemented)

Applied once, at the next committed boundary at g1294 or later, by `continue-rust-teacher.py`; older boundaries such as the g1284 restore keep their recipe. It is part of the signed recipe (`training_revision_g1294.py`). The reviewed-transition checks project it away, so every older gate still compares the exact earlier recipe. It cannot be undone or re-timed afterwards. Weights, both Adams, RNG, replay, the fixed teacher beta and the stage counters are kept.

1. **Entropy:** PPO entropy rises linearly from 0.001 to 0.005, one step per committed iteration: 0.00104 at g1295, 0.003 at g1344, 0.005 from g1394. The completed anneal stays complete; the rise counts every committed iteration, healthy or not. Collection receipts and PPO check the same coefficients.
2. **PPG KL cap** (`ppg-value-phase/26.0.0`): a hit stops further steps and keeps that update, with no rollback, even if its held-out loss rose. The held-out loss gate still rejects intervals that did not hit the cap. Beta feedback is unchanged: an early hit is judged on the steps before the kept interval, so it still raises beta ×1.5 as before.
3. **Search head:** at most 64 steps per iteration (was 32). Stages A and B run 64 head-only steps; Stage C runs up to four 16-step trunk intervals, each still checked against its KL cap of 0.003 and its held-out loss. The head was paused for the return-buffer rebuild (7 of 8 ages at g1293), so it should resume training at the next iteration.

## 4. Verification

- 28 new tests (`tests/Recovery/test_training_revision_g1294.py`) on the exported g1294 state and config: recipe scope and exact projection, one-time adoption from g1294, the entropy ramp and restart, PPG keep-at-cap against loss rollback and retention, beta feedback, Adam/replay preservation, 64-step head and trunk passes, and a single host install per plan.
- Related suites (PPG, entropy, search head, teacher stages and activation, overlap, startup, share target): 186 tests run. The 7 errors all predate this change: 3 also fail on the unmodified code, 1 is the known source-review failure, and 3 need pruned g0418/g0660 exports. The source-review failures also occur in the entropy, g570 and g611 suites, again on the unmodified code too.
- Also fixed: the B05 smoke (`lab_b05_smoke.task_teacher_step`) required an actor-gradient norm on every step, which last session's one-in-four sampling broke. It now checks the sampled steps.
- Not run on the lab. The CPU worker build is unaffected, so `prepare-rust-cpu-inference.py` is not needed.

## 5. Continue

Stop the trainer at an iteration boundary, commit and pull, then use the committed generation (1294 if nothing ran since):

```bash
python scripts/continue-rust-teacher.py --expected-generation 1294 --iterations 10 --share-target 0.1 --inference cpu
```

The revised recipe is written to its own file, `configs/step5/<run>-rust-teacher-on-stages-v2-no-insolvency-entropy-raise-g1295-share01.json`.
