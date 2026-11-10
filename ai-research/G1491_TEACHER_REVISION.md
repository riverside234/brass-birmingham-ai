# g1491: long+short teacher labels, 48 tables, one PPO epoch, visit-count exploration

Date: 2026-11-10. Run: `artifacts/step5/joint7-v11-2-g1290-restart-g1290`, committed at g1491 (export `results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z`). At g1491 teaching is on with share target 0.4 and the fixed beta 0.0397 (qualified at g1294); the search head is in Stage C (w=1, trunk scale 1).

## What changes

| Setting | g1491 | From the next iteration |
| --- | --- | --- |
| Teacher tables | 28 train (+4 held out when due) | 48 train (+4 held out when due) |
| CPU workers | 64 | 80 (`--workers 80`) |
| Teacher labels | long searches only | long and short searches |
| Teacher loss | mean over long labels | `0.6 mean(L_long) + 0.4 mean(L_short)` |
| Short search | n=96, K=8 | n=128, K=8 (`--teacher-fast 128`) |
| Long search | n=640, K=16 | unchanged |
| Teacher rows per PPO step | 64 | 2048 (1229 long + 819 short) |
| Label buffer | 4 iterations, weights 4:3:2:1 | 2 iterations, weights 4:3 |
| PPO epochs | 2 | 1 |
| Played action in teacher games | the search's choice | sampled from the root visit counts (see below) |

Everything else is kept: weights, both Adams, RNG streams, the opponent pool, PPG, the search-head stage and its buffer, share target 0.4, the 4.0e-6 LR floor.

## Re-calibrating beta (first two iterations)

The new loss mixes two label sets over a 32× larger batch, so the old beta no longer gives a 0.4 share. The revision retires the old labels and their beta together:

1. **Two shadow iterations** (g1491→g1492→g1493): the teacher gradient is measured on every PPO step and discarded, so there are no teacher updates. The search head waits as in any shadow iteration. The four held-out tables run on both, as the qualification requires.
2. **PPO LR held:** the controller makes no change in those two iterations, raise or cut, and their KL stays out of its three-iteration window. Raises can't follow straight from the shadow KLs; the window refills from the first teaching iterations. One epoch already resets that window, because epochs are part of its data signature.
3. **Fixed beta:** after two consecutive full 48+4 iterations, beta = 0.4 / median(measured share).
4. **Teaching on from the third iteration** through the reviewed switch, with that beta. The search head resumes in Stage C.

## Exploration

After each search the played action is sampled from the decision root's visit counts (temperature 1: probability = visits / total), instead of playing the search's own choice. This applies to:

- each seat's **first 7 actions of each era**, on short and long searches. Canal round 1 has one action, so in the canal era that is rounds 1–4; in the rail era it is rounds 1–3 plus the first action of round 4.
- **every long-search action**, at any point in the game.

Only action decisions are sampled; the following payments and sub-choices are searched as usual. When the sampled action isn't the one the search planned, the session replans that action's next decision with the same turn budget. That costs one more search per changed action, which is the main new search cost. Labels are unchanged: the target is still the improved policy from the search, and `played_action` records the sampled move with `explored: true`. Sampling uses its own deterministic stream (job seed and decision index), so a rerun of a table plays the same game.

## Costs to expect

- **Labels:** about 1,500 long and 4,600 short labels per iteration (now 870 long). Each short label is validated and encoded in Python like a long one (about 5–20 ms of encoding per row on this laptop). Short-label turns check model-logit parity once per table instead of once per turn, so they don't queue on the shared model lock the way long-label turns do (0.45 s median wait at g1490).
- **Teacher term:** 2048 rows per step in PPO-sized chunks (512 rows or 131,072 candidate cells), with exact gradient sums. With one epoch there are about 23 steps instead of 46.
- **Search:** 48 tables, n=128 short searches and the replans after changed actions.
- **CPUs:** the lab process sees 64 logical CPUs (affinity 0–63 in the g1490 report). With 80 workers, 16 CPUs host two pinned lanes each; `--no-pinning` hands placement to the OS instead.

The next export shows the actual collection, verification and PPO times.

## Commands

Stop the trainer, then commit and pull on the lab. The CPU teacher changed, so rebuild it first; the build runs the native tests, including a full revised table:

```bash
python scripts/prepare-rust-cpu-inference.py --run artifacts/step5/joint7-v11-2-g1290-restart-g1290 --expected-generation 1491
```

```bash
python scripts/qualify-rust-cpu-inference.py --run artifacts/step5/joint7-v11-2-g1290-restart-g1290 --expected-generation 1491 --workers 80
```

```bash
python scripts/continue-rust-g1491.py --expected-generation 1491 --iterations 100 --share-target 0.4
```

The last command defaults to `--inference cpu --workers 80 --teacher-fast 128 --teacher-fast-k 8 --teacher-full 640`. It runs the two calibration iterations, then teaching for the rest (here up to g1591). If the Ctrl-C landed after g1491's teacher tables were sealed, it first finishes that iteration under the old recipe and budget, and the revision starts at g1492. After any later stop, rerun it with `--expected-generation` set to the last committed generation; it resumes whichever step is unfinished.

## Verification (laptop)

- Rust: all 46 tests of the patched CPU crate pass, 6 of them new (revision parsing, per-seat era counting, visit-count sampling, exploration gating, label marking, and a real search session that replans after an explored action).
- Native end to end: a real all-seat table on the rebuilt CPU teacher with the revision. Long and short labels both appear and all pass Python validation and the turn checks. Every long-search action is sampled and short ones only in each seat's opening. The worker echoes the revision, the deferred logit check covers every short row, a rerun plays the identical game, and a job without the revision produces the old rows byte for byte.
- Python: the revision applied to the real g1491 recipe changes exactly the listed fields. It passes the reviewed on→shadow and shadow→on transitions one way only. It retires the old labels and beta, and plans 48+4 jobs carrying the revision. Two synthetic 48+4 shadow iterations qualify a beta that authorizes teaching at Stage C. The teacher term matches the weighted formula's gradient for any chunking. The LR controller holds and keeps those KLs out of its window.
- Existing suites: the full native module (48 tests) and the B04/B045/B07 sealing, Stage A, stage-schedule, turn-ack, overlap, qualification, collector, branch-restart and g1294 suites pass. The 28 tests that still fail in the teacher, B07 and LR suites fail identically on the unmodified code (old fixtures, removed exports, "Backend rollback manifest changed").
- Not run on the lab. Iteration timing with 48 tables and short labels is an estimate until the next export.
