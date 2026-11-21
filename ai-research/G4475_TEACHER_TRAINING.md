# g4475: search-value head paused, 160 teacher rows per PPO step, after-PPO soft check, faster local AI

Four requests from 2026-11-21, made together at g4475:

1. Pause the search-value head's training completely, so it costs as little as possible. The search games must
   still train the auxiliary (era) heads.
2. Raise the search games' rows per PPO step from 128 to 160.
3. Add one more soft held-out measurement, after the PPO update and before PPG.
4. Install the core engine speedup (`ENGINE_SEARCH_SPEED.md`) in the local game.

Items 1 and 2 form one recipe revision, `--teacher-training-revision` (module `recovery/training_revision_g4475.py`,
top-level block `teacher_training_revision`). It is adopted once at a committed boundary and then kept. It changes
only the trainer: the collection, the CPU teacher, its jobs and every table output stay the same.

Item 3 is code-only: a measurement with no flag. Item 4 is a local game patch, release 1.4.61, already installed.

Execution identity: `configs/learning/b07-replay-location-review.json` (`g4475_teacher_training`, `ca3ea6e9…`,
predecessor `535226da…`).

## Continue command

From the newest export (g4475) to the g4490 rating:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 4475 --iterations 15 --share-target 0.1 --teacher-training-revision
```

The flag is needed once. Later continuations keep the revision without it.

## 1. Search-value head paused

**Before.** After PPG, the search-value phase ran its own pass every iteration:
- 32 steps on the head and the trunk (Stage C, scale 1.0);
- guarded joint-KL, clone and ordinary probes;
- held-out predictions and a stage advance.

That pass cost 2.5–3.0 s an iteration (`value_phase.timings.post_ppg_auxiliary_seconds`, g4469–g4474). In those
six iterations it kept steps only once (g4473). In the other five its first interval was rejected.

**Now**, once the revision is adopted:
- PPG gets no post-fit hook (`SearchValueIteration.prepare_ppg`). There is no pass, so no head, trunk or Adam step,
  no probe, no held-out prediction and no stage advance.
- The head, its companion Adam, the Stage C state and λ .95 are kept exactly as committed, so the head can be
  resumed later.
- The PPG beta floor still reads the kept stage, so it stays .125.

**Unchanged:**
- Every teacher table still publishes its search-value shard: roots, the g3471 extra rows and the era-end scoring
  maps.
- Shards that age out of their window still retire at the same point, by the same rule (`ValueBuffer.retire`,
  which `buffer.rows` now calls), so the buffer does not grow.
- **The era heads keep training on the search games** (search-aux 2.0.0). The pool is still the last four
  iterations' train tables plus the held-out tables that left the latest 32. The held-out set is still the latest
  32, pooled into the ownership stop's owner-CE criterion.

**Report.** `search_value.paused` holds the retired and kept shard counts, with `updated: false` and the kept
stage.

**Saving:** about 2.5–3 s per iteration, and the trunk no longer gets the head's steps.

## 2. 160 teacher rows per PPO step

`g3270.phase_options` takes the revision's 160 in place of the g3270 revision's 128.
- **Unchanged:** the draw (by age weight, never repeating a row inside the PPO update, batches grouped by legal
  width) and the 8-iteration label buffer with its identity and long labels.
- **More rows per update:** at 46 PPO steps the update draws 7,360 of about 11,900 buffered labels, against 5,888
  before.
- **The soft head** trains on the same 160 rows.
- **Beta:** the adaptive teacher beta carries over, because its basis (share target, qualification, re-measurement)
  does not include rows per step. The teacher gradient is a mean over rows, so with more rows its norm falls
  slightly (less noise). The controller (ρ .9, η .1) re-tracks the 0.1 share over roughly 10 iterations, with no
  re-measurement.
- **Cost:** the teacher term took 1.7 s per update at 128 rows, so expect about +0.4–0.5 s, plus about 25% more
  soft-head time (0.34 s now).

## 3. Soft held-out check after PPO

Until now the soft held-out check ran twice on the iteration's 4 validation tables:
- before the PPO update (`prepare`);
- after PPG and every other phase (`finish`).

`TeacherPhase.after_ppo` now runs it once more, right after the PPO update and before PPG. It is called from the
iteration loop before `ppo_finished`, so its cost is counted in `post_ppo_diagnostics_seconds`.

It uses the same rows and the same forwards: no gradients, RNG forked, train mode restored. It therefore changes
nothing that trains. The test checks this: RNG state, weights, soft head state and every other report value are
identical with and without it.

New fields in `teacher_phase.heldout`:
- `soft.after_ppo`: the soft KL, with rows and phase breakdown;
- `soft.ppo_delta`: after PPO minus before;
- `soft.ppg_delta`: after minus after PPO;
- `soft.after_ppo_seconds`;
- `after_ppo` and `ppo_delta`: the policy's own teacher KL from the same forward, at no extra cost.

`ppo_delta` and `ppg_delta` add up to the existing `delta`. The log line reads
`soft_heldout=before->after_ppo(ppo)->after`.

The soft head trains only inside PPO; PPG moves only the shared trunk under it. So `ppg_delta` shows how much the
value phase's trunk movement costs the soft head.

Cost: one forward of about 124 rows, roughly 0.1–0.3 s.

## 4. Local game: engine-speed release 1.4.61 (installed)

The four local AI packages (alpha-g890, beta-g1290, beta2-g1190, il-v11-2) were rebuilt with the teacher's faster
search core and installed. Scripts are in `scripts/local-game/engine-speed/`:
- `build-engine-speed.py` builds the release;
- `qualify-engine-speed.py qualify` and `qualify-engine-speed.py audit` check it before and after installation;
- `install-engine-speed.ps1` installs it and also supports `-Action Rollback` and `-Action Recover`.

**What changed.**
- The installed native source is the core from before the engine change (commit `6739595e`). The build checks
  this, up to formatting and line endings.
- The build replaced exactly the three core files that carry the exact decision reuse: `search/brass.rs`,
  `search/reward.rs` and `state/observation.rs`.
- Every other source file, the models, the policies and the managed patch are unchanged.
- The rollout touches native files only: the `brass-ai-local.exe`, `native-source.zip` and `build.json` of each
  package.
- The receipt is now version `1.4.61-rust-local`, with an `engineSpeed` record that points to the recovery backup
  (`artifacts/patch-recovery/engine-speed-f2f3…`).

**Checks.**
- In the local tree, the core's unit tests passed, including the three decision-reuse tests:
  - every reused decision equals a rebuilt one;
  - searches with reuse on and off give identical reports.
- The worker's local tests passed.

**How the local AI uses the speed.** The local search does not run until its deadline. Each lane gets a
simulation quota from its time and a calibrated 7.26 ms per simulation (about 410 per lane at a 4 s allowance).
The search answers as soon as the quota is done.

So the faster engine does two things:
1. It answers sooner.
2. It completes its quota in positions where the old engine ran out of time. In early runs on a loaded laptop, the
   installed engine finished only 609–774 of about 814 simulations on beta's captured positions; the new one
   finished all of them.

It does not run more simulations, because the quota is unchanged.

Qualification: 4 packages × 4 positions (the warm-up position and captured positions 0, 8 and 17), installed and
candidate interleaved.

| model | reply time, installed → candidate (positions 0, 8, 17) |
| --- | --- |
| alpha-g890 | 2.30 → 1.85, 2.08 → 1.51, 2.09 → 1.38 s |
| beta-g1290 | 2.50 → 2.08, 2.51 → 1.38, 1.95 → 1.16 s |
| beta2-g1190 | 2.48 → 1.93, 2.36 → 1.58, 2.03 → 1.25 s |
| il-v11-2 | 2.65 → 2.10, 2.20 → 1.52, 2.04 → 1.32 s |

Over all 16 searches the median reply-time ratio was 0.76, and every search completed its quota. Mid-game Rail
positions gain most, about 0.6×.

After installation, the audit started each package from the game folder: warm-up passed, a search completed, and
the receipt matches the build. The game itself was not opened.

**Follow-up (done the same day).** The 7.26 ms estimate was re-calibrated to 6.36 ms in release 1.4.62, also
installed. A turn's search now runs about 14% more simulations in the same time. See
`LOCAL_SEARCH_V1_CALIBRATION.md`.

## What to check after the first iteration

- `search_value.paused` is present and `updated` is false.
- `value_phase.timings` has no `post_ppg_auxiliary_seconds`.
- `value_phase.search_auxiliary` still reports train and held-out tables.
- `teacher_phase.term.revision.plan.rows_per_step` is 160, with `demand` 7,360 at 46 steps.
- `teacher_phase.term.adaptive_beta.ema_share` stays near 0.1.
- `teacher_phase.heldout.soft` has `after_ppo`, `ppo_delta` and `ppg_delta`.
