# g1190 restart: placement 0.1, a fresh search head, Stage A/B 50

Date: 2026-11-10. Source run: `artifacts/step5/joint7-v11-2-g1290-restart-g1290` (read only). New run: `artifacts/step5/joint7-v11-2-g1190-restart`. It replaces the g1490 restart ([G1490_RESTART.md](G1490_RESTART.md)); that prepared run folder is left as it is.

## What it starts from

- **Weights:** the retained g1190 snapshot, `population/weights-d872ea92…-aux-dd8dcbfaf5350001.pt`, bound by the g1491 state's own rating receipt (generation 1190, model `d872ea92…`). It is generation 1190 of the `g1290-trained-restart` lineage, not of `grouped-lr-critic` (whose g1190 is a different model).
- **State:** the same exact committed g1491 state as the g1490 restart, from `results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z`. Its opponent pool (members up to g1490), ratings and counters continue; the learner is counted as generation 1491.

g1190 was trained before the teacher phase, the Rust backend and the post-transfer reward:

- **Reward:** placement 0.1, insolvency charge 0.0025, no canal VP timing.
- **Search head:** none.

The restart keeps every tensor and changes only these:

- **Utility:** the utility specification moves to the restart reward, and its calibration is cleared. The model id changes with it, because the specification is part of the id.
- **Search head:** none is installed at restart. A fresh one is installed at Stage A on the first teaching iteration, the same path the original adoption used.
- **Companion heads:** the local-scoring companion heads (and their Adam) move to generation 1491 with no fitting recorded.
- **Run-level state:** exploration, IL-KL limit, startup receipts and the backend come from the g1491 state. The g1190 policy-anchor payload is dropped, as the run retired it.

The fresh head is 256 wide instead of 32: `Linear(548→256)`, GELU, then `Linear(256→4)`, with a zero final layer as before. That adds about 124k parameters (the model goes from about 5.93M to about 6.05M, under the 6.5M limit). A head of any width other than 32 is recorded in the checkpoint as `search-value-correction/1.1.0` with its width, so it loads back at that width. Earlier 32-wide heads keep their original marker. The width is valid only with the restart schedule, and a run can't change it.

## Recipe

The same as the g1490 restart, with these changes:

| Setting | g1490 restart | g1190 restart |
| --- | --- | --- |
| Reward | win 1, placement 0.05, VP 0.001 (cap 180), canal/final VP timing, insolvency 0 | **placement 0.1**, otherwise the same |
| Search head | Stage C kept | **fresh, Stage A** |
| Search head width | 32 | **256** |
| Stage A | – | **50 iterations**, each fitting with the Stage C held-out early stop |
| Stage B | 20 iterations (w +0.05) | **50 iterations (w +0.02)** |
| Stage C | 10 iterations | 10 iterations |
| PPG beta floor | 0.25 | 0.25 in Stage A, **0.5 from Stage B on** |

Unchanged from the g1490 restart:
- **Teacher:** 32 teacher tables (+4 held out), long-search labels only, 72 teacher rows per PPO step, the 4-iteration 4:3:2:1 buffer, two PPO epochs.
- **Calibration:** one shadow iteration fixes beta, with the PPO LR held.
- **Exploration:** visit-count and cotton exploration.
- **Search:** n=96/K=8 short and n=640/K=16 long searches on 64 CPU workers.
- **Optimizers:** new PPO/PPG optimizers. The warmup restarts and g1491's LR multiplier of 1.25 is carried.

In code:
- **Reward:** `post-transfer-reward` gains a restored-placement variant, reached only from the post-transfer reward and adopted only by this restart.
- **Stage schedule:** the restart schedule (`restart_schedule()`) is a second reviewed one; the earlier schedules are unchanged.
- **In-run changes:** a run can't change its schedule's Stage B length, Stage A fit or floor.

## Iterations

1. **g1491→g1492, shadow:** the teacher needs utility readiness under the current reward. This iteration's value phase re-fits it, so it plans no teacher tables. The LR is held.
2. **g1492→g1493, shadow:** all 32+4 tables run and beta is fixed. The LR is held.
3. **From g1493:** teaching is on. The fresh head is installed after that iteration's PPG and Stage A counts start.
   - **Stage A (50 iterations):** eight whole games are held out from the value buffer, and the head fits in up to four 16-step intervals. An interval is kept only if the held-out loss does not rise; otherwise it is undone and fitting stops.
   - **Stage B (50 iterations):** the blend rises 0.02 per collection, and the PPG beta floor is 0.5 (raised at once if beta is lower). Head fitting in Stage B stays the fixed-step pass, as before.
   - **Stage C (10 iterations):** the trunk ramp, with its own held-out early stop.

Stage A ends around g1543, Stage B around g1593, and Stage C around g1603.

The 0.02 blend steps needed one small CPU-teacher change. The teacher's job check and the Python binding now accept a 1/100 grid of `w` instead of 1/20, and every earlier value is still valid. The overlay fingerprint changes, so the continue command rebuilds the CPU teacher once (focused regressions only) before training.

## Commands

Stop any trainer, pull the code on the lab, then:

```bash
python scripts/prepare-rust-g1190-restart.py --share-target 0.75
```

```bash
python scripts/continue-rust-g1190-restart.py --expected-generation 1491 --iterations 120 --share-target 0.75
```

- **Share target:** pass the same value to both commands.
- **Defaults:** the second command defaults to `--inference cpu --workers 64 --teacher-fast 96 --teacher-fast-k 8 --teacher-full 640`.
- **Length:** `--iterations 120` runs to g1611; any count works.
- **After a stop:** rerun the second command with the last committed generation.
- **Summary:** prepare prints its summary and also saves it as `branch-restart.json` in the new run. It lists the learner and seed model ids, the reward migration, compacted rating slots, the LR multiplier and the schedule.

## Verification (laptop)

- **New tests:** 20 in `tests/Recovery/test_g1190_restart.py`:
  - **Recipe and transitions:** the recipe applied to the real g1491 state changes exactly the listed fields. Other schedules and rewards are refused, and the recipe can't be adopted inside a run. The shadow→on switch is reviewed, and the schedule can't change after it.
  - **Stage schedule:** 50+50 stages with distinct 0.02 steps; the first Stage B collection uses 0.02.
  - **Stage A early stop:** a held-out rise stops fitting or undoes the interval; no rise runs all intervals; a real fit doesn't raise the held-out loss. The fresh head is 256 wide; without the restart schedule it stays 32 and uses fixed steps.
  - **PPG floor:** 0.5 past Stage A only.
  - **Build:** a run tree built from the real g1491 state with a g1190-style seed prepares and loads:
    - same tensors, new utility specification, no head;
    - optimizers with the 1.25 multiplier;
    - companion heads at 1491;
    - teacher not ready until a current-reward receipt, then 32+4 jobs;
    - a fresh Stage A machine with 50/50;
    - an adopted 256-wide head within the parameter limit that saves and loads at its own width.
    The continue command counts two calibration iterations. Wrong weights and an unreviewed seed reward are refused.
- **Native:** a CPU teacher rebuilt locally from the current overlay passes its 26 crate tests. A real table with a 0.02 blend is accepted and blends exactly like the GPU path. The 0.45-blend, g1490 long-only and g1491 short-label tables still pass, and so do the overlay tests.
- **Existing suites:** 89 suites touching the changed code (1,022 tests). 120 fail, and all 120 fail identically on the unmodified code (old fixtures, removed exports, "Backend rollback manifest changed", a stale local CPU binary).
- Not run on the lab.
