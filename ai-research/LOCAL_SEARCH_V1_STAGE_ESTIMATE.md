# Local search_v1 estimate by era stage: release 1.4.63 (installed)

The user's request of 2026-11-21:
- make the local search's startup estimate depend on the position, either by era stage (early Canal, late Canal,
  early Rail, late Rail) or by round, choosing the better of the two;
- retire the Alpha Test and IL models, so that only two models need rebuilding and testing.

**Result:** Beta and Beta-Plus now size each turn's search from its stage's own estimate. The game is installed as
`1.4.63-rust-local`. Alpha Test and IL are retired, and their packages are in the recovery backup.

| stage (rounds) | 1.4.62 | 1.4.63 | simulations per 4 s search (median, qualification) |
| --- | --- | --- | --- |
| canal early (1–4) | 6.36 ms | 7.34 ms | 934 → 810 (0.87×) |
| canal late (5–8) | 6.36 ms | 4.63 ms | 937 → 1,285 (1.37×) |
| rail early (1–4) | 6.36 ms | 4.74 ms | 937 → 1,257 (1.34×) |
| rail late (5–8) | 6.36 ms | 4.34 ms | 938 → 1,374 (1.46×) |

Every candidate search completed its quota inside its time: 96 of 96 in qualification, using at most 97% of the
available time.

## One engine, two deployments

The engine is shared, so a core improvement reaches both training and the local game in one pass:
- `native/brass-ai-core` holds the rules, the search and the engine speedups, such as decision reuse.
- The training CPU teacher copies it (`lab_cpu_overlay.TEACHER_CORE_FILES`).
- From 1.4.63 the local packages are built from the same repository `native/` tree
  (`scripts/local-game/search-stage/build-search-stage.py binaries`). Up to 1.4.62 they were built from their own
  archived `native-source.zip`.

What differs is the deployment layer, and each keeps its own:

| | training (lab CPU teacher) | local game (`native/brass-ai-worker`, `local.rs`) |
| --- | --- | --- |
| budget | fixed simulation counts (`Mode::Count`) | time-based (`Mode::Timed`): quota = available time / estimate |
| searches | long and short searches mixed, from the recipe | one search per turn, no long/short split |
| what is calibrated | nothing timing-related; counts come from the recipe | the startup estimate (`startup_ns`, now `startup_ns_by_stage`) |

So a faster core immediately makes local searches finish sooner. Turning that into more simulations is a local
calibration step, the one done here.

The training execution identity (`ca3ea6e9…`) and the lab teacher's fingerprint do not cover `brass-ai-worker`, and
both are unchanged.

## Era stage or round: era stage

The local game's own captured positions (the L06 capture) have action roots only at Canal and Rail rounds 1, 3 and
6: 20 of each, from 20 local-rules games. Calibration (`calibrate-search-stage.py measure`) timed every searching
root on both models: action, continuation and payment roots, 276 searches.

| bucket | searches | maximum | median |
| --- | --- | --- | --- |
| Canal round 1 | 46 | 7.34 ms | — |
| Canal round 3 | 46 | 4.95 ms | — |
| Canal round 6 | 52 | 4.63 ms | — |
| Rail rounds 1, 3, 6 | 44, 40, 40 | 4.74, 4.38, 4.34 ms | — |
| Canal rounds 5, 7, 8; Rail round 8 | 2 each | — | — |
| **canal early / late** | 92 / 58 | 7.34 / 4.63 ms | 4.33 / 3.02 ms |
| **rail early / late** | 84 / 42 | 4.74 / 4.34 ms | 3.08 / 2.69 ms |

Each grouping was fitted on 10 games and checked on the other 10 (276 held-out searches):

| grouping | simulations vs 1.4.62 | held-out overruns |
| --- | --- | --- |
| one estimate | 0.91× | 4 |
| **by era stage (4)** | **1.29×** | **11** |
| by round (10 seen) | 1.36× | 11 |

By round gains a little more on the rounds that were measured. But rounds 2, 4, 5, 7 and 8 have no local action
roots, so their estimates would rest on no measurement or a single game. By era stage covers every round with 42–92
searches each, so it is the one built. Most of the possible round gain lies inside Canal early: round 1 costs up to
7.34 ms per simulation, against 4.95 ms for round 3.

## Release 1.4.63

**Worker.** `native/brass-ai-worker/src/local_startup.rs`, plus four lines in `local.rs` and `lib.rs`:
- The optional policy field `startup_ns_by_stage` holds canal_early, canal_late, rail_early and rail_late.
- Rounds 1–4 are early and 5–8 are late; NN seats play four-player games, so each era has eight rounds.
- A turn's first search uses its stage's estimate. `startup_ns` stays the fallback, and the table may not exceed it.
- The ready message reports the table, and each quota names its stage in the estimate's source.
- New unit tests cover stage boundaries, value selection with fallback, and table validation.

**Packages.**
- Only `beta-g1290` (`local-beta`) and `beta2-g1190` (`local-beta2`) were rebuilt from `native/`. Their six policy
  files each carry the table, with `startup_ns` set to 7.34 ms.
- The models, warm-up roots, K mappings, temperatures and the managed patch are unchanged.
- `alpha-g890` and `il-v11-2` were removed from `BrassNative` (440 files), and the installer kept them in its
  recovery backup.

**Checks.**
- The core's unit tests passed: 7, including the three decision-reuse tests.
- The worker's local tests passed for both models: 11 each, including the 3 new ones.
- Qualification compared against the installed 1.4.62 on 6 action roots per stage from different games, both
  models, K 16 and K 8, interleaved.
- The installed audit started both packages from the game folder; both reported the table and completed their
  quota at every stage.

**Calibration inputs.** The roots keep their replayed settled rounds, with no budget spent. The worker checks the
remaining insolvency budget against its own arithmetic for exact equality. A value such as 0.015000000000000001
does not survive its JSON float parsing, and a simulation's cost does not depend on the insolvency penalty.

**Verification.** The full verification sweep at the new table was stopped at the user's request after about 100
roots, so its rows were not recorded. The qualification is the check.

## What the user will see

- **More simulations after the opening.** Late Canal and all of Rail run 1.34–1.46× the simulations in the same
  thinking time.
- **Fewer, but complete, in the opening.** Canal rounds 1–4 run 0.87× (7.34 ms against 6.36 ms). At 6.36 ms, 6 of
  92 early-Canal calibration searches ran out of time: three round-1 roots, each on both models. The new value lets
  those finish. 1.4.62 therefore already overran on some round-1 turns, and stopped them part-way through.
- **Alpha Test and IL** show as "Model runtime is not installed" and cannot be chosen. A saved game with an Alpha
  Test or IL seat cannot continue until they are restored by rollback.

To undo it (this restores 1.4.62 with all four models):

```
& "scripts/local-game/search-stage/install-search-stage.ps1" -Action Rollback
```

## Possible next steps (not done)

- **Round 1 of Canal as its own stage.** This is most of the remaining gain in the opening: round 3 costs about
  4.95 ms against round 1's 7.34 ms. It needs local Canal round 2 and 4 action roots to calibrate the rest of the
  stage, which the existing captures do not have.
- **Canal early at 6.36 ms.** Keeping 1.4.62's value for canal early would trade back the opening's simulations for
  overruns on some round-1 turns (about 7% of early-Canal searches in calibration).
