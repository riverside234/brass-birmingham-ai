# g3270: teacher build steps 2 and 3, and a 32-game search-value hold-out

This implements build steps 2 and 3 of [TRAINING_IMPROVEMENT_PLAN_G2910.md](TRAINING_IMPROVEMENT_PLAN_G2910.md) (section 8), with the user's changes of 2026-11-18:

- Short-search labels stay off for policy training. Long-search labels rebuild their target from the current policy.
- 128 label rows per PPO step, with no row repeated inside the PPO update and each step's rows of similar legal width.
- 32 dedicated held-out search games for the search-value phase. 4 new ones arrive each iteration, and the 4 oldest move to the search-value training buffer. The set first fills from the 16 already held out.
- A 16-iteration search-value buffer and at most 32 search-value steps.

The recipe part is adopted once with `--teacher-revision` and then kept. Four collection changes from step 2 are code-only, on by default, and each has a switch back.

## Where the run stands (g3171–g3270)

**PL rating (128 games per slot, every 20 iterations).** The run is flat to slightly rising. No regression alert fired.

| Generation | 3170 | 3190 | 3210 | 3230 | 3250 | 3270 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PL points vs g1290 | 154.3 | 125.8 | 174.0 | 156.6 | 174.2 | 182.6 (95% 140–229) |

**Teacher tables still wait, and the wait is the trainer's, not the table's.** In g3264–g3269 a table took 54–65 s. Of that, 42 s was search and 10–22 s was blocked after a search, writing its turn to the trainer.

The long waits are simultaneous. In g3268 all 32 tables stopped together for 15.5 s (22–38 s into collection) and again for 5.6 s. In g3269 they stopped four times, for 5.1, 3.4, 7.1 and 6.9 s. Each time, every table resumed within half a second of the others.

The trainer process stops draining every teacher pipe at once. A per-table verification queue cannot cause that; a pause of the whole process can.

The likeliest cause is CPython's full garbage collection. It stops every thread, and the process keeps large long-lived trees: frozen label views and 3,128-float value states held as Python lists.

The cost on the iteration:
- **Collection is teacher-bound.** The teacher finishes 4–5 s after the ordinary pool (`teacher_collection_seconds` 3.9–5.4 s).
- **Admission waits too.** Admission runs only after the ordinary pool is done.

**The search-value head barely trains.** Its early stop reads the latest 16 dedicated held-out games (4 every fifth iteration). In g3264–g3269 it kept 0 steps four times and 16 steps twice, with `heldout_loss_increased` every time.

## What changes

### Recipe (`--teacher-revision`, `teacher_revision` block, `teacher-revision-g3270/1.0.0`)

The qualified `teacher_phase` recipe stays exactly as reviewed: 32+4 tables, the 4:3:2:1 buffer and 96 rows. Every validator, the gradient-qualification context and recipe recognition still read it, and `is_shadow` peels the new key.

The trainer runs this overlay (`training_revision_g3270`):

| Part | Before | Now |
| --- | --- | --- |
| Tables | 32 train; 4 held-out every fifth iteration | 36 train + 4 validation, every iteration |
| Validation labels | never train | measured before and after the update, then train from the next iteration at age 0 |
| Label window | 4 iterations, weights 4:3:2:1 | 8 iterations, weights 8, 7, …, 1 |
| Labels for policy training | long searches | long searches only (short labels stay off) |
| Target | fixed at collection: old policy plus search bonus | rebuilt at each use: softmax(current logits + 0.20 × (50 + most visits) × completed value) |
| Rows per PPO step | 96, drawn independently each step | 128, drawn by age weight without repeating a row inside the PPO update, grouped by legal width |
| Staleness check | a forward pass over every row | a fixed 256-row sample of each age |
| Logit parity check | every new label (~31 per table) | 8 rows per table, two per phase |
| Search-value buffer | 8 iterations | 16 iterations |
| Search-value steps | at most 64 (4 × 16) | at most 32 (2 × 16) |
| Search-value early stop | latest 16 held-out games | latest 32 (at least 16 to run) |

**Targets.** A label stores its search statistics: prior logits, visit counts, mean returns and node value. The bonus is computed from them as in `improved_policy`. At the label's own C_scale (0.20 for long searches), softmax(stored logits + bonus) is the exported target exactly. A test checks this.

Each step adds the bonus to the current logits and holds the result fixed for the gradient. An old label then adds only the search's preference and never pulls the policy back toward the policy that collected it.

**Batches.**
- **Plan.** PPO tells the term its planned step count (46 today) before its first step. The term draws that many batches with exponential keys divided by the age weight: weighted sampling without replacement, fixed by seed and generation.
- **Grouping.** The drawn rows are sorted by legal width, cut into batches of 128, and the batches are shuffled.
- **Small pool.** A pool smaller than the demand (the first iteration after adoption: about 5,200 rows against 46 × 128 = 5,888) gives every row once, in batches of ceil(pool / steps).
- **Staging.** Only the drawn rows are staged on the GPU.

**Validation tables.** The 4 validation tables of an iteration play the role of both kinds of held-out table:
- **Policy.** Their labels give the held-out teacher KL before and after the update, now with the policy's own movement on the same rows (`heldout.movement`) and `heldout.pull` = movement − KL change. From the next iteration they train, their age counted from then.
- **Search value.** Their games join the held-out set.

**Search-value hold-out** (`search_value_retention.rotating_holdout`):
- **Filling.** No game leaves the set before it holds 32: 20, 24, 28 and then 32 games from g3270 to g3273, starting from the 16 already held out.
- **Leaving.** After that, the 4 oldest leave each iteration. A game that leaves trains the head while it is younger than 16 iterations, its age counted from its collection, so in steady state it trains at ages 8–15.
- **The old games.** g3254's games age out of the 20-iteration hold-out window first. g3259's leave at age 16 and do not train; g3264's leave at 12 and train 4 iterations; g3269's leave at 8 and train 8.

**Beta** is re-measured once, with teaching on, in the adoption iteration. Rows per step, the pool and the target all change the teacher gradient's size. The term runs with the current beta (0.0278), measures the raw share, then sets beta to 0.1 / measured median and restarts the adaptive averages. Labels, the beta's qualification, the search head and both buffers carry over: the label-buffer identity does not change.

### Collection: search workers never wait (code-only, step 2)

| Change | Why | Switch back |
| --- | --- | --- |
| Each streamed (count-mode CPU) teacher table writes its output to an unlinked file in `/dev/shm` that a reader thread tails. The event queue is unbounded. | A regular file never blocks its writer. Through a 64 KB pipe, every pause of the trainer blocked every search worker. `lab-native/` is unchanged; only the Python side of the stream changes. | `BRASS_TEACHER_SPOOL=0` |
| The collection window moves the existing heap to the permanent GC generation (`gc.freeze`) and returns it afterwards. Every collector pause inside the window is timed in the new `collection_gc` report section. | A full pass then walks only objects made during collection. Reference counting frees garbage as usual. | `BRASS_COLLECTION_GC_FREEZE=0` keeps the timing only |
| Teacher tables are admitted on the teacher thread as soon as they are collected, while the last ordinary games finish. | Admission took 3.5–5 s after the ordinary pool. | `BRASS_TEACHER_TAIL_ADMISSION=0` |
| The label cache keeps each row's label and encoding; the view stays in the sealed shard. Value rows cache their state as a float32 array. | A frozen view costs about 260 KB and a 3,128-float list 99 KB of memory and GC traversal. The array costs 12 KB, and training reads the same float32 values. | none needed |

## Evidence (Windows laptop; no games played)

**Memory and GC, measured on the captured production-like game** (`tmp/step10-publisher/gcheap.py`):
- **Today's size.** 4,000 label views plus 18,432 value rows (today's windows) are 3.85 M tracked objects. A full GC over them takes 0.82 s on the laptop; after `gc.freeze()` it takes about 0.
- **At the new windows, kept as before.** About 10,000 label rows and 37,000 value rows would have taken about 4 GB more memory and about twice that pass time.
- **At the new windows, as stored now.** The label cache stays at about today's size, about 1.7 GB, and the value cache falls from about 1.9 GB to about 0.6 GB.

**Tests** (`tests/Recovery/test_teacher_revision_g3270.py`, 18 tests):
- **Recipe.**
  - The overlay leaves the qualified recipe and the buffer identity unchanged, is adopted once and is recognized as teacher-on.
  - The transition passes forward only, and it changes the collection signature, so pending work is archived at adoption.
  - Beta is re-measured once, and 36+4 jobs are planned every iteration.
- **Label window.** It keeps 8 ages, and validation labels start training at age 0 one iteration later.
- **Targets and loss.**
  - The bonus at the stored scale rebuilds the exported target.
  - The loss and its gradient equal the KL from softmax(current logits + bonus), with the target detached.
- **Batches.**
  - A real `ppo_update` plans exactly its optimizer steps, with no row repeated.
  - At most one batch spans two widths.
  - A small pool gives every row once.
- **Search value.**
  - The held-out set goes 20 → 32 → rotating, with the retirement ages above.
  - Cached states are float32 and equal to the published lists.
- **Collection.**
  - A spooled table's worker finished writing 3 MB in milliseconds while the test held the GIL in one long `sorted()` call; through the old pipe it would have waited.
  - The pipe path and truncated output still work.
  - The GC window freezes, unfreezes and times passes.
  - The parity sample covers every phase.

**Existing suites.**
- **Passing.** These pass after updating three assertions about the old representations: the recipe and revision suites (116 tests), the teacher, search-value and collection lab suites (188 tests) and the b07 resume and sealing suites.
- **Already failing.** `test_teacher_schedule`, `test_teacher_attribution_audit`, `test_teacher_phase_revision` and `test_b07_completion` fail identically with my edits reverted (4 failures, 12 errors). The known `test_b07` subtests also fail.

**Identity.** Re-recorded in `configs/learning/b07-replay-location-review.json`. `f93f6a05…`, the identity the lab ran g3180–g3270 under, is added as a predecessor so the g3270 checkpoint resumes.

## Expected effect on the lab (*estimates*)

- **Teacher tables.**
  - With the waiting gone, a table takes about 43–45 s instead of 54–65 s. The 40 tables then finish well inside the ordinary collection window.
  - Collection becomes ordinary-bound again, and teacher admission runs inside it, saving about 4–5 s per iteration.
  - If the trainer's pauses were GC, ordinary lanes and publishers stop pausing as well.
- **Labels.**
  - About 1,240 labels per iteration (36 + 4 tables) instead of about 990. The pool grows by that each iteration, to about 9,900 at 8 ages.
  - From the second revised iteration each update uses 5,888 distinct rows, about 60% of a full pool, mostly the younger ages. Today it uses 4,416 presentations of about 2,530 distinct rows.
- **Time.**
  - The parity check reads about 320 rows instead of about 1,100.
  - The staleness check reads at most 2,048 rows instead of all of them.
  - Staging moves from about 4,000 to about 5,900 rows.
- **Search value.** The early stop reads up to 2,048 rows of 32 games instead of 1,024 rows of 16. A one-epoch pass covers 32 batches of 512 rows from a buffer of up to about 37,000 rows.

## What to check in the next reports

- **Collection.**
  - `teacher_phase.collection.tables` is `{'train': 36, 'heldout': 4}` every iteration, and `heldout_validation.reason` is `every-iteration`.
  - Teacher stalls: in `collection_metrics[*].turns`, almost no `post_search_ns` over 1 s, and table `elapsed_ns` close to the search time.
  - `collection_gc`: `frozen_objects`, `total_pause_seconds` and `longest_pause_seconds`. A large pause with the freeze on means the stall has another cause.
  - `teacher_phase.collection.admitted_during_ordinary_tail` is true, and `timings.teacher_collection_seconds` is near 0.
  - `finalization` / `deferred_logit_check.rows` is 8 per table.
- **Teacher term.**
  - `teacher_phase.term.revision.plan`: `drawn_rows` (about 5,200 at g3270, then 5,888), `repeated_rows` 0, `padding_efficiency` and `inclusion_by_age`.
  - `teacher_phase.term.beta_remeasurement` is accepted at g3270, and `share.median` is near 0.1 afterwards.
  - `teacher_phase.heldout.movement` and `pull` on the 4 validation tables (also in the log line as `heldout_move`/`heldout_pull`).
- **Search value.**
  - `search_value.heldout_rotation.available_games` goes 20, 24, 28, 32, 32 …, and `retired_training_games` is 0 until g3276.
  - `search_value.update.max_steps` is 32.

## Code

| File | Change |
| --- | --- |
| `src/brass_ai/recovery/training_revision_g3270.py` (new) | The revision block, adoption, recipe projection and the settings the trainer reads; beta re-measurement at adoption |
| `src/brass_ai/recovery/teacher_term.py` | `search_bonus`, rebuilt targets, `plan()` (no-repeat width-grouped batches), sampled staleness, `movement` |
| `src/brass_ai/recovery/teacher_buffer.py` | 8-iteration window, validation labels from the next iteration, slim label cache |
| `src/brass_ai/recovery/teacher_phase.py`, `search_value_schedule.py` | Overlay options, 36+4 jobs, validation every iteration, validation movement |
| `src/brass_ai/recovery/updates.py` | PPO hands its planned step count to the teacher term |
| `src/brass_ai/recovery/search_value_*.py` | 16-iteration buffer, float32 state cache, 32-step options, rotating 32-game hold-out |
| `src/brass_ai/recovery/collection_gc.py` (new), `league.py` | Collection GC window and its report; log line with validation movement |
| `src/brass_ai/rust_backend/lab_teacher.py` | `Spool`: file-backed streamed output and unbounded event queue |
| `src/brass_ai/rust_backend/lab_shadow.py` | Teacher admission during the ordinary tail; parity sample wiring |
| `src/brass_ai/rust_backend/lab_teacher_finalization.py` | `parity_sample` |
| `src/brass_ai/rust_backend/lab_shadow_config.py`, `recovery/config.py` | Recipe recognition and transition; the key's validation |
| `scripts/continue-rust-g1491.py`, `continue-rust-shadow.py`, `continue-rust-g1190-restart-2.py` | `--teacher-revision`, journaled adoption, config filename tag, summary line |
| `configs/learning/b07-replay-location-review.json` | Execution identity re-recorded; f93f6a05… as predecessor; review notes |

## Continue command

20 iterations end on the g3290 rating:

```bash
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3270 --iterations 20 --share-target 0.1 --teacher-revision
```

`--teacher-revision` is needed only once; passing it again later is harmless. The value revision and the rating cadence persist from earlier adoptions. No Rust source changed, so the CPU workers are not rebuilt.
