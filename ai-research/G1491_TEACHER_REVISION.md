# g1491: long+short teacher labels, 48 tables, one PPO epoch, visit-count exploration

> Superseded on 2026-11-10 after g1500: training restarts from the g1490 weights in a new run with long-search labels only. See [G1490_RESTART.md](G1490_RESTART.md).

Date: 2026-11-10. Run: `artifacts/step5/joint7-v11-2-g1290-restart-g1290`, committed at g1491 (export `results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z`). At g1491 teaching is on with share target 0.4 and the fixed beta 0.0397 (qualified at g1294); the search head is in Stage C (w=1, trunk scale 1).

## What changes

| Setting | g1491 | From the next iteration |
| --- | --- | --- |
| Teacher tables | 28 train (+4 held out when due) | 48 train (+4 held out when due) |
| CPU workers | 64 | 64 (unchanged; the lab process sees 64 CPUs) |
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

### Cotton-directed exploration (2026-11-10)

New adoption through `continue-rust-g1491.py` also enables
`teacher-cotton-exploration/1.0.0` on **training teacher tables only**. Held-out,
rating, ordinary self-play and previously committed recipes keep their behavior.

- Eligibility: a legal cotton build, the same opening-seven/long-search window,
  fewer than **3 targeted builds per seat per era**, and no own unflipped cotton mill.
  Three is a ceiling, not a quota. Natural cotton choices do not spend it.
- Before a fresh root search, reserve up to two distinct cotton building locations
  on long searches, one on short searches, **inside the existing K**. Prefer the
  highest raw prior for each location; discard-card variants do not consume the
  distinct-location slots. Keep an ordinary comparator where legal. The original
  priors, Gumbel values, simulation quota, backups and improved-policy formula stay
  unchanged. The detached CPU teacher core records its own execution fingerprint.
- After search, **80%** uses the existing visit-count sampler; **20%** samples
  visited cotton candidates in proportion to their counts. The actual behavior
  probability is `0.8*N(a)/N + 0.2*I_cotton(a)*N(a)/N_cotton`. This is not a 20%
  cotton share of all actions. An already supported cached continuation is reused;
  if it has no cotton visits, record the fallback and use the original sampler.
  Coverage never launches another search just to fill the cotton slots.
- Only successful action commits spend an allowance. The two eras/four seats have
  independent counters. Job/decision-keyed random streams reproduce the mixture
  on retry without consuming search RNG. Stale commits are rejected; cancelled
  tables remain unpublished and retry from their frozen seed. The normal session
  identity checks still replan payments/second actions when the played move differs.
- The reward and **improved policy label calculation are unchanged**. A targeted
  played move can have a low teacher target. Small `cotton_exploration` label
  metadata records the branch, exact behavior probability, pre-commit allowance
  and search coverage; existing admission recomputes these from the legal choices
  and visit counts. No cotton bonus, prior boost, one-hot target or Q gate is added.

`report.teacher_phase.collection.cotton_exploration` aggregates long/short/total counts once per
iteration: eligible decisions, blockers, covered decisions/locations, cotton and
total visits, missing evidence/zero cotton visits, triggers, targeted/actual/natural
builds, coverage and trigger rates, and summed base/mixed cotton probability mass.
Each table's `cpu_inference.revision.cotton` also retains the 2×4 quota/build counts.
Counters are fixed-size Rust accumulators; only the existing table receipt and
eligible label metadata cross the wire. No per-action logging, timer, file write,
NN forward or full-history copy is added for these measurements.

Candidate selection is at most two linear scans plus work on K≤16 elements.
The changed moves can still change subsequent game/search work; unchanged wall
time is **not** claimed without lab measurement. The added implementation itself
does not raise K or the frozen search quota. Watch actual coverage and completed
builds, not trigger rate alone, before deciding whether to lower 20% later.

## Costs to expect

- **Labels:** about 1,500 long and 4,600 short labels per iteration (now 870 long). Each short label is validated and encoded in Python like a long one (about 5–20 ms of encoding per row on this laptop). Every labelled row of a g1491 table (long and short) checks model-logit parity once, in a single batch as the table finishes, instead of once per turn under the shared model lock (0.47 s median and 8.5 s worst wait at g1490). The check still runs before anything is published.
- **Teacher term:** 2048 rows per step in PPO-sized chunks (512 rows or 131,072 candidate cells), with exact gradient sums. With one epoch there are about 23 steps instead of 46.
- **Search:** 48 tables, n=128 short searches and the replans after changed actions.
- **CPUs:** the lab process sees 64 logical CPUs (`nproc` 64; affinity 0–63 in the g1490 report), so the worker cap stays at 64: 52 tables (48+4) hold 52 of the pinned CPUs and ordinary lanes fill the rest as tables finish.

The next export shows the actual collection, verification and PPO times.

## What the g1490 export and the first qualification show

- **Tables spend about 40% of their time waiting.** At g1490 a table took 76 s (median, 28 tables) but its own searches summed to 43 s (30 s long + 13 s short). The same searches in the 16-table qualification took 41 s per table with about 2 s of overhead. The wait is the verification of labelled turns: a worker acks a fixed-count turn at once and searches on, but the event queue holds two events, so a turn whose verification takes longer than the next turns' search stalls the table. The sampled verification of a labelled turn took 0.66 s median and 3.3 s at p90, almost all of it waiting for the shared policy lock (0.47 s median, 8.5 s worst), because the reserved long turns of all 28 tables reach the lock together at the phase boundaries. Python preparation itself was 0.12 s median.
- **What changed because of it:** a revised table runs its model-logit parity once, as it finishes (`TurnVerifier.settle`), so no turn takes the shared lock during collection. Each table takes the lock once, at staggered end times. The check is the same one and still precedes publication; jobs without the revision are unchanged.
- **What stays a risk:** with 52 tables and four times the labelled turns, the main process does about 3,300 turn verifications per iteration (encoding about 1.5 ms per early-game row on the laptop, more for late-game boards), all under one GIL. If that exceeds the collection window the tables stretch to match it. The qualification teacher stage now runs 48 tables at once for that reason and records `search_seconds`, `stall_seconds` (table time beyond its own searches; the g1490 figure to compare with is about 33 s), the labels per table and a verification profile.
- **First qualification (`--workers 80`):** failed only `enough_cpus`. Everything else passed: 41 regression tests, bundles, parity, 16 teacher tables (n=96, K=8; 0 deadline hits), 128 ordinary games, overlap, rating and startup (all 80 lanes ready within 3.8 s). The ordinary and overlap stages report 136 s and 122 s of wall time around only 22 s and 20 s of pool collection; the rest is the stage re-reading every game with full validation, which the training run does not do. The stage now times the pool start, the pool close and that re-read separately (`outside_collection`).
- **CPU count:** `nproc` on the lab shell is 64, which agrees with the g1490 report (`cpu_count` 64, affinity 0–63). `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are what the run already used (recorded as `omp_threads` and `mkl_threads` at g1490); the trainer sets its own tensor threads (4) and each lane is one pinned thread, so leave them. `OMP_NUM_THREADS=1` doesn't change `nproc` (it only sets a minimum).

## Commands

Stop the trainer and pull the updated code on the lab. The continuation command
below detects a stale CPU build and rebuilds it once, with focused regressions.
No full qualification, benchmark or complete game is required by that path.
To prepare it separately:

```bash
python scripts/prepare-rust-cpu-inference.py --run artifacts/step5/joint7-v11-2-g1290-restart-g1290 --expected-generation 1491 --focused-regressions
```

```bash
python scripts/qualify-rust-cpu-inference.py --run artifacts/step5/joint7-v11-2-g1290-restart-g1290 --expected-generation 1491 --workers 64
```

The full qualification above is optional and is **not needed again** for this
cotton change. It is read only. From g1491 it plans its teacher tables under the
g1491 recipe on a copy of the committed one (`--revision auto`) at n=128 for short
searches (`--teacher-fast`). Existing successful unrelated qualifications remain valid.

```bash
python scripts/continue-rust-g1491.py --expected-generation 1491 --iterations 100 --share-target 0.4
```

The last command defaults to `--inference cpu --workers 64 --teacher-fast 128 --teacher-fast-k 8 --teacher-full 640`. It runs the two calibration iterations, then teaching for the rest (here up to g1591). If the Ctrl-C landed after g1491's teacher tables were sealed, it first finishes that iteration under the old recipe and budget, and the revision starts at g1492. After any later stop, rerun it with `--expected-generation` set to the last committed generation; it resumes whichever step is unfinished.

The cotton recipe uses a distinct `-g1491-revision-cotton20-share04.json` config
suffix. An older prepared g1491 config is retained for recovery. Existing
committed g1491 recipes without the cotton marker remain loadable and unchanged;
the new marker is adopted with the initial g1491 recipe transition.

Cotton change verification: 42 focused Python regressions and 28 Rust unit
regressions passed locally (including three tiny 16-trajectory root decisions,
metadata/commit integration, stale root binding, caps, isolation and old behavior).
No complete game, training iteration or performance benchmark was run for this
change. Lab throughput and the resulting policy's playing strength remain to be measured.

## Verification (laptop)

- Rust: all 46 tests of the patched CPU crate pass, 6 of them new (revision parsing, per-seat era counting, visit-count sampling, exploration gating, label marking, and a real search session that replans after an explored action).
- Native end to end: a real all-seat table on the rebuilt CPU teacher with the revision. Long and short labels both appear and all pass Python validation and the turn checks. Every long-search action is sampled and short ones only in each seat's opening. The worker echoes the revision, the deferred logit check covers every short row, a rerun plays the identical game, and a job without the revision produces the old rows byte for byte.
- Python: the revision applied to the real g1491 recipe changes exactly the listed fields. It passes the reviewed on→shadow and shadow→on transitions one way only. It retires the old labels and beta, and plans 48+4 jobs carrying the revision. Two synthetic 48+4 shadow iterations qualify a beta that authorizes teaching at Stage C. The teacher term matches the weighted formula's gradient for any chunking. The LR controller holds and keeps those KLs out of its window.
- Existing suites: the full native module (48 tests) and the B04/B045/B07 sealing, Stage A, stage-schedule, turn-ack, overlap, qualification, collector, branch-restart and g1294 suites pass. The 28 tests that still fail in the teacher, B07 and LR suites fail identically on the unmodified code (old fixtures, removed exports, "Backend rollback manifest changed").
- Qualification script: the fixture teacher stage runs under the revision (long and short labels, exploration, the one batched logit check per table) and the revised-label checks fail by name when a label kind, the exploration or the worker's counts are missing. The qualification and g1491 modules pass (73 tests). The native module's 48 tests pass; its revised-table test was adjusted to the single batched check and re-run on its own.
- Not run on the lab. Iteration timing with 48 tables and short labels is an estimate until the next export.
