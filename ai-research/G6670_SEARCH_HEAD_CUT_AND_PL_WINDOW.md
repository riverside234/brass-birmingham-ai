# g6670 revision: search head cut, PL window, PPG floor .25, PPG clip 3.0

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, committed at g6670 in the newest export
(`joint7-v11-2-g4970-rail025-soft8-beta0.02-g6670-20261128T033135117029Z`). The revision is adopted once with
`--g6670-revision` at a committed boundary and then kept (`src/brass_ai/recovery/training_revision_g6670.py`, block
`g6670_revision`).

## Request (2026-11-28)

1. Cut the search head when training continues.
2. The PL score stores and uses only the latest 64 history checkpoints, and older ones are deleted (about 500 are
   stored now; clean them when training continues). The PL fit uses only the latest 40,960 games (320 rounds).
3. PPG beta floor 0.25.
4. PPG `grad_clip` 3.0, with the Adam state rescaled.
5. ~~PPO `grad_clip` 5.8, with the Adam state rescaled.~~ Withdrawn by the user: PPO keeps its current clip (6.5).

## Changes

| | before (g6230 to g6669) | after |
| --- | --- | --- |
| Search-value head | in the model, paused since g4475 (no training, no predictions), with its own Adam | removed at the adoption iteration's PPG boundary; the next checkpoint has neither the head nor its Adam |
| PL checkpoints | 492 rated checkpoints kept, every one a possible opponent | the latest 64 (plus the generation-0, v4 and g1290 references); the older ones are retired |
| PL fit | every rated game (62,464 at g6670) | the latest 40,960 rated games (320 slots of 128) |
| PPG beta floor | .125 (g3658) | .25; beta is raised to .25 once if it is below |
| PPG clip | 2.5 (g6230) | 3.0, with a one-time PPG Adam rescale |
| PPO clip | 6.5 | 6.5 (unchanged) |

The following do not change:
- the teacher recipe, its labels, buffers and beta;
- the soft head (weight 4);
- the era heads and their search rows;
- PPO and PPG rates;
- the rating protocol (32 setups x 4 seats, every 20 iterations).

### 1. Search head cut

The head had nothing left to do:
- since g4475 it neither trains nor predicts;
- the teacher tables already bind `w = 0` (`teacher_search_head_weight` 0, `value_source` main);
- the CPU bundles at `w = 0` already leave it out.

Its stage record says C with `w = 1`, frozen since g4474, but the teacher weight override already beats it.

On the adoption iteration, the committed g6670 checkpoint is loaded with the head attached, and collection uses that
model unchanged. At the PPG boundary, `SearchValueIteration.finish_ppg` runs `cut_head`, which removes
`model.search_value_head` and its companion Adam before the g6671 checkpoint is written. That checkpoint has no
head marker. Its model id is the head-less model's, and the ordinary weights are untouched.
`report.search_value.cut` records:
- the width (32);
- the parameter count;
- the companion's steps;
- the stage.

From g6671, the search-value iteration requires that the head and its Adam are both absent. `plan_jobs` also
requires `w = 0` for every teacher job.

The following stay:
- the stage record, as history;
- the search tables' value shards and their retirement on schedule;
- the era heads' search rows (search-aux 2.0.0).

### 2. PL rating: 64 checkpoints, 40,960 games

**Checkpoints.** `ratings.retire_checkpoints` keeps the latest 64 rated checkpoints in `state.ratings.agents`. Older
ones move to `state.ratings.retired_agents` with only their generation and model id, so no weights path remains.
Retention pins weights through the paths in `state.json`, so the retired files lose their pins. The trim runs:
- at the trainer's startup rating pass, which is the cleaning on continuation;
- after every new slot, when one goes in and the oldest goes out.

Opponents are chosen only among the kept checkpoints.

Effect on the g6670 state (dry run):
- 495 agents: 492 rated checkpoints plus 3 references;
- 64 + 3 are kept;
- 428 are retired, the oldest kept being generation-005410;
- 21 of the retired weights files are in this run, and the next scheduled cleanup (due at g6675, every 5
  iterations) deletes them;
- 407 are in the parent run `joint7-v11-2-g1190-restart-3`, which this run branched from at g4970. The parent
  stopped at g5177 and copied its registry into this run. This run's retention never deletes another run's files,
  and the parent's own `state.json` still pins them, so they are removed by the parent cleanup below.

**Parent cleanup (`rating_parent_cleanup.py`).** This runs once, after the first committed iteration (g6671), in
`league.py` before the scheduled cleanup. For each other run that stores retired checkpoints:
1. **Plan.** It picks the retired names whose entry in that run's own registry is the same checkpoint (same name,
   generation and model id, stored in that run). The plan, with each path, is written first to this run's
   `.retention/parent-rating-cleanup-g6670.json`. That receipt is not a control file and pins nothing, so an
   interrupted pass resumes with the same files.
2. **Retire.** Under the parent's run lock it backs up the parent's `state.json` once, to
   `configuration-history/g6670-rating-cleanup/state-before.json`. It then retires the planned names from the
   parent's registry (`ratings.retire_named`), keeping generation, model id and `retired_by` but no path. If the
   parent run is active, the pass stays pending and is retried after the next commit.
3. **Delete.** It runs a fresh reference scan of every control file in every run (`retention.references`: all
   `state.json`, unfinished or formal specs, pins and the manifests they name), and deletes only the planned files
   that nothing pins. A pinned file stays, with its reason in the receipt.

Nothing else in the parent changes:
- its 10 checkpoints rated after the branch (g4990-g5170), its references and its pool;
- its checkpoints, replays and rating evidence.

If a registry has retired agents but no game window (the parent's case), its own rating code still fits every game
and names retired agents.

Dry run on the g6670 child state and the g5177 parent state, with placeholder files:
- 407 planned, **396 removed**, 11 kept;
- 7 of the kept files are opponents named by the parent's g5170 rating-slot spec;
- 4 are in the parent's own opponent pool (g3830, g4550, g4870, g4910);
- the parent's registry goes from 420 to 13 agents;
- the pass took 11 s, and a second call does nothing.

The parent's four other unarchived slot specs (g5090-g5150) were not exported. On the lab they may pin a few more
opponents. Those specs, and the parent's pool, are the parent's own evidence and training state. They are left
alone, and nothing in the parent advances to archive them.

**Games.** The fit reads the latest 40,960 rated games (`ratings.latest_games`). On the lab that is the 320 slots
from g1730 (g1740 from the next slot). An agent that appears in those games stays a fit parameter after its
checkpoint is retired. The report, plot and alerts name it through `known_agents`. The game records and the compact
archive are kept as they are.

Dry run on the export's rating history (four recent slot files are not in the export, so its window starts at
g1690):
- 376 agents in the window;
- the full fit took 14.2 s instead of 20.5 s on the laptop (the lab's full fit was 6.9 s, so about 4.8 s
  expected);
- **scores move up**: g6670 goes from 185.9 to 197.5 PL points. The 61 kept-era checkpoints move up by +8.5 to
  +14.6 (median +9.6) against the g1290 baseline. The baseline now meets them only through recent games. Earlier
  slot reports were fit on every game, so the progress curve steps up about 10 points at the first windowed slot
  (g6690).

### 3. PPG beta floor .25

`value_phase.beta_clone_min` goes from .125 to .25. g3658 had lowered it from .25; `g3658.applied` stays true for
this lineage, so its transition check and recipe stage name are unchanged. At the first resume,
`ppg_g210.initialize` raises the restored beta to .25 if it is below (migration
`ppg-beta-floor-quarter-g6670/1.0.0`). It was 0.1875 at g6669. The KL trip history, both Adams, RNG, replay and
cadence are kept. From then on a clean phase halves beta only down to .25.

The head schedule's Stage C floor (.125) no longer binds, because the effective floor is the larger of the two.

### 4. PPG clip 3.0

The PPG Adam rescale uses the same method as g5466, g5685 and g6230: a normal fit of each iteration's logged mean
and P98 (`value_phase.gradient_clipping`), pooled by steps.

| PPG window g6640-g6669 (clip 2.5) | |
| --- | --- |
| attempted steps | 954 |
| pre-clip mean | 1.46 |
| steps above 2.5 | 5.1% logged, 5.6% from the fit |
| predicted above 3.0 | 1.4% |
| exp_avg / exp_avg_sq | x1.0106 / x1.0339 |

The rescale runs once, on the adoption generation's checkpoint (`ppg_g210.initialize` -> `rescale_ppg`). Its
receipt is stored in the PPG Adam group beside the g5466, g5685 and g6230 receipts.

## Cost

- **Search head:** no compute change, since it was already idle. The checkpoint loses the head (32 wide) and its
  Adam. Cutting it cannot be undone without a fresh head.
- **PL rating:**
  - the fit is about 30% faster (about 2 s of a 61 s slot on the lab);
  - opponents come only from the latest 64 checkpoints (about 1,260 generations);
  - this run's 21 old weights files are deleted, and about 396 in the parent run;
  - the parent's rating registry loses the 407 entries (its `state.json` is backed up first), so its rating
    slots can no longer pick them as opponents;
  - scores shift by about +10 points once (see above);
  - games older than g1730 no longer inform the fit.
- **PPG floor:** PPG clones more strongly whenever beta would have gone below .25. That means smaller value-phase
  policy drift, and possibly less value-fit progress per phase on clean phases.
- **PPG clip:** the 5% of PPG steps above 2.5 become up to 20% larger. The predicted clip rate drops to about 1.4%.

## Checks after adoption

- The log prints the g6670 summary line and `training-revision-g6670: PPG clip 2.5 -> 3` with factors about
  1.0106 / 1.0339.
- `value_phase.beta_floor.minimum` is .25. The migration `ppg-beta-floor-quarter-g6670/1.0.0` appears in
  `adoption`, with `beta_after` .25.
- `value_phase.gradient_clipping.max_norm` is 3.0, with `clipped_percent` near 1-2%.
- `report.search_value.cut` is present only in the g6670 iteration's report. From g6671, the checkpoint has no
  `search_value_head` and `search_value.companion` is null.
- `state.ratings.agents` has 67 entries and `retired_agents` has 428. The g6675 cleanup report lists the retired
  population weights, and the first windowed slot (g6690) has `timing.fit_window` with 40,960 games.
- The g6671 iteration log prints `g6670 parent rating cleanup artifacts/step5/joint7-v11-2-g1190-restart-3: done`
  with the removed and kept counts. Its report has `parent_rating_cleanup`, and the full list is in
  `.retention/parent-rating-cleanup-g6670.json`.

## Tests

`tests/Recovery/test_g6670_revision.py` (12 tests):
- the recipe: one block plus the floor, PPO clip unchanged, refusals, forward-only transitions;
- the PPG fit on the logged g6640-g6669 norms, its norm reader, and the Adam rescaled once after the earlier
  receipts;
- the floor: beta raised to .25 once (and kept above), trips kept, the clip rescale at the same resume, halving
  stopping at .25;
- the search head: the cut at the PPG boundary, ordinary weights unchanged, the next checkpoint saved and loaded
  without the head;
- the rating: 64 kept with references, retired entries without paths but with the run that stores them, one in
  and one out per slot, and the windowed fit keeping old opponents of recent games;
- the parent cleanup on a two-run tree:
  - unpinned files removed;
  - a file in the parent's pool kept;
  - a same-name different checkpoint skipped;
  - the parent's own later checkpoints untouched;
  - the backup written;
  - an active parent left pending;
  - an interrupted pass resumed from its plan.

16 modules were run first (171 tests):
- this file;
- g6230, g5685, the g3658 floor, g4475, the strength rating, rating cadence, rating retention, `test_ppg_g210` and
  the g1190 restart;
- the g6547 soft weight and value speed, g3423 soft, g3528, g3471 and g3270.

The only errors were the 2 `test_ppg_g210.Restored` g660 tests already listed in
`docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md` (missing g0660 export).

After the parent cleanup was added, 11 modules were run (95 tests), and all pass:
- this file, the strength rating, rating cadence, rating retention, the rating plot and audit retention;
- g6230, g5685, the g3658 floor, g4475 and the g6547 soft weight.

Dry run on the g6670 export:
- only `g6670_revision` and `value_phase.beta_clone_min` change;
- the transition is forward True, back False, same True;
- `previous` round-trips.

## Execution identity

`12294b942491b953219a6fb9b4114113ce5fefb5b3bc2acf9caefb38cb751f1a`, with predecessor `6e2152b1...`.
`d8f8834d...` was the version without the parent cleanup and was never run. The entry is
`g6670_revision` in `configs/learning/b07-replay-location-review.json`.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 6670 --iterations 10 --share-target 0.065 --g6670-revision
```

The soft weight (4) is kept from the run. The earlier revision flags are already adopted and are not needed again.
