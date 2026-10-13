# Self-play teacher phase support

The current requested settings are **16 seconds for early Rail**, **24 seconds for late Rail**, and **12 seconds for each Canal phase**. Collections use **five games**, retaining the existing eight distillation steps, quality gates, cadence and phase reservations. `search_wall_fraction` defaults to `None` in Python and `null` in all active training JSON presets.

The implementation was inspected against the existing uncommitted checkout before editing. Its champion-gate, PPO, reward, opponent-search, execution and personality changes were preserved. Across the 13 active training presets, only `teacher_work_budget`, `search_games` and `search_wall_fraction` changed. The named `joint7-g150-distill4.json` experiment retains its explicit four-step override; the lab and recommended continuation preset still use eight.

| Phase | Existing reservation range | New shared turn budget |
|---|---|---:|
| `canal_early` | Canal rounds 1-4 | 12 s |
| `canal_late` | Canal rounds 5-8 | 12 s |
| `rail_early` | Rail rounds 1-6 | 16 s |
| `rail_late` | Rail rounds 7-8 | 24 s |

Each of four seats reserves one seeded turn in each phase. Five games therefore provide 20 reserved turns per phase and 80 total, versus 16 and 64 previously. The phase definitions, seeded reservation function, 640-game warm-up, 160-game cadence and maximum two collections per iteration are unchanged in the lab recipe. Failed or insufficient collections still consume their reservation; there are no extra retries to manufacture labels.

**Clock and scheduling behavior.** The versioned `teacher-turn-time/1.1.0` contract records `seconds_per_phase`. Atomic CPU admission starts one absolute deadline. All four private search workers, the incumbent, both actions and continuation choices share that deadline. Search does not restart for a second action or continuation. Each exported label records its actual phase clock. Existing cancellation, search retirement, late-publication rejection and inference failure limits remain active. Neural inference requests inherit the remaining turn deadline; the existing 120-second request ceiling and 180-second transport failure timeout already accommodate 24 seconds. These are failure safeguards, not extra thinking time. Native host retirement and application of legal continuations can make measured turn elapsed time exceed the search deadline; any label published after the deadline is rejected.

`None` disables only cumulative teacher-versus-ordinary time admission. Reports use JSON `null` for available/remaining allowance and `wall_fraction_enabled: false`, including when utility calibration defers teaching. They continue accumulating ordinary and teacher wall time. Explicit numeric restrictions in `(0, 0.90]` remain supported by current phase-contract configurations. Invalid numbers, booleans, NaN and infinity are rejected. Historical budget contracts and their migrations remain readable under their original meanings.

**Cost.** The admission estimate now sums the four actual phase clocks, multiplies by four seats and five games, and adds ordinary play overhead:

```text
original: 4 games x 4 seats x (12 + 12 + 12 + 12) =   768 reserved seconds
previous: 5 games x 4 seats x (12 + 12 + 12 + 24) = 1,200 reserved seconds
current:  5 games x 4 seats x (12 + 12 + 16 + 24) = 1,280 reserved seconds
```

The early-Rail adjustment adds **80 reserved seconds per collection (6.67%)** over the previous phase settings. The full change from four uniform-budget games adds 66.67%: five games add 25%, and the phase mix adds 33.33% per game. At two collections per iteration, the reservation total is 2,560 seconds, versus the original 1,536. These are sums across games, not predicted elapsed training time: games can overlap under the existing shared CPU allowance. Workers do not multiply a turn's clock. Ordinary game execution, policy updates, calibration, retirement and contention also cost time. Removing cumulative admission can additionally increase collection frequency when the old allowance would have deferred a due collection. Recipe-bound measured collection costs remain reported, with their existing 25% headroom.

**Distinct labels and update limits.** Acceptance still requires diverse games and phases, sufficient completed root-action evidence, a supported selected continuation, a current model/source fingerprint and a ready utility head. Duplicate qualified positions are rejected by setup, seat and revision. Additional games get fresh setup seeds, sampling seeds and paths. The sampler still visits each qualified position once before a second shuffled pass.

The effective distillation minibatch is `min(PPO effective_batch_size, distillation.batch_size)`, currently 16. The independent step ceiling remains eight, and reuse remains at most twice per position. Microbatching changes memory usage, not that sample allowance. For a collection with 80 qualified labels, all 80 are used, 48 receive a second presentation, and eight steps consume 128 presentations. A 79-label collection uses 127 presentations because its first pass ends in a partial batch. Quality failures use zero labels. A smaller step cap or larger future collection can leave qualified labels unused; this is now explicit rather than inferred from presentation counts.

Reports and compact retention history now preserve:

- Per phase: search calls, calls with completed evidence, attempted/completed/interrupted trajectories, qualified labels, insufficient coverage, timeouts, allocated time and actual search/turn elapsed time.
- Action support: legal root actions, actions visited at least once, actions with repeat support and supported continuation plans. These count completed search visits, not the label's full legal-action probability mask.
- Per collection and iteration: qualified `teacher_rows`, `distinct_labels_used`, `unused_qualified_labels`, presentations, maximum reuse and corresponding per-phase counts. Collection summaries retain the minibatch size and presentation ceiling.
- Per collection: exact categorical KL from the pre-distillation policy to the post-distillation policy on a bounded qualified-label probe. This precedes utility recalibration. The existing iteration-level post-PPO-to-post-teaching probe remains in place. Both are diagnostics, not strength measures.

**Validation.** The initial phase-support receipts and measured smoke results are recorded below. Tests exercise shared four-worker clocks, actions and continuations, timeout/cancellation rejection, disabled and explicit numeric admission, five distinct games, label reuse/unused counts, current-source checks, calibration, old migrations, resumed checkpoint/Adam/RNG preservation, champion gating and report export. Original run artifacts and installed-game files are not changed.

The subsequent **early-Rail 16-second revision passed all 66 focused tests** in 61.336 seconds. These cover its shared worker/action clock, updated collection cost, disabled cumulative limit, exact default migration, custom phase settings, historical migration results, and checkpoint/Adam/RNG/progress/lineage preservation. Source `48831bc83887a8d6beb563fd908dff18ee6b180dd8d7da6824724f4c53d9e596` transitions to `c875dde1baa158108f2a23d05cf0b60b82b0592462e3c85b33d3d5510a8e6f73` through 98 added review paths; all 637 earlier paths remain intact. Only early Rail changed in each of the 13 active presets. See [the early-Rail revision receipt](teacher-early-rail16-evidence.json). No new native benchmark or lab training was run for this setting, so its effects on label yield and playing strength remain unmeasured. The continuation command below is unchanged.

For the initial **12/12/12/24** phase settings, **176 distinct focused tests passed**, including the corrected historical fixture and final targeted rerun. That native smoke passed in **253.875 seconds** on source `48831bc83887a8d6beb563fd908dff18ee6b180dd8d7da6824724f4c53d9e596`, using ten reserved-turn probes: two positions in each phase, plus matched 12-second baselines for the two late-Rail positions. Mode order was alternated between those two comparisons. This was a small random CPU network calibrated on two complete ordinary smoke replays, not the lab generation-150 checkpoint. These measurements predate the early-Rail increase to 16 seconds.

| Phase / budget | Search calls / calls with completed work | Trajectories attempted / completed | Qualified labels | Insufficient coverage / timeouts | Visited / legal root actions | Search / whole-turn elapsed (s) |
|---|---:|---:|---:|---:|---:|---:|
| canal_early / 12s | 2 / 2 | 32 / 16 | 1 | 0 / 1 | 9 / 781 | 24.047 / 24.531 |
| canal_late / 12s | 2 / 2 | 32 / 16 | 2 | 0 / 0 | 8 / 166 | 23.859 / 24.469 |
| rail_early / 12s | 2 / 2 | 31 / 16 | 2 | 0 / 0 | 8 / 114 | 23.781 / 24.172 |
| rail_late / 24s | 2 / 2 | 112 / 96 | 2 | 0 / 0 | 10 / 10 | 47.766 / 48.126 |
| late-Rail baseline / 12s | 2 / 2 | 60 / 45 | 2 | 0 / 0 | 10 / 10 | 23.782 / 24.110 |

Late-Rail completed trajectories increased **45 → 96**, and supported continuation plans **11 → 24**. Qualified-label yield remained **2/2 under both budgets**, action coverage remained **10/10**, and insufficient-coverage counts remained zero. The selected positions had only four and six legal root actions: this validates additional completed work, but does **not** demonstrate improved label yield on difficult late-Rail positions. The early-Canal timeout correctly rejected one late-published label. Timed results are subject to local CPU contention.

**Actual training usage:** the current probes generated seven qualified labels, and the baseline generated two. These are partial-turn probes, so all nine were deliberately excluded from training. A separate authored five-game smoke collection generated **80 qualified labels**, used **80 distinct labels**, left **zero unused**, and made **128 presentations / eight optimizer steps**, with maximum reuse two. Each phase contributed 20 generated and 20 used labels; presentations were Canal early 28, Canal late 36, Rail early 34 and Rail late 30. Its exact pre/post-distillation policy KL was **0.004028** on all 80 labels. Utility calibration was invalidated as required. This is a sample-accounting and optimizer check, not evidence of playing strength.

Full counters, per-test outcomes, input identities and limitations are in [the evidence receipt](teacher-phase-support-evidence.json). [The local full smoke report](../artifacts/recovery/teacher-phase-revision/native-smoke-24/report.json) includes per-turn receipts, decisions, inference statistics and training diagnostics. An earlier smoke used the superseded 36-second request and is kept separately; it is not the active configuration.

Initial source transition: `3db61ed932779f2cd8d8e149509ab0c1b7232af63102213f17d5666c2d338130` → `48831bc83887a8d6beb563fd908dff18ee6b180dd8d7da6824724f4c53d9e596`; learning compatibility `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2` is unchanged. That revision's 97 added pinned review paths preserve the existing historical review receipts. Real lab continuation was not executed here; optimizer and lineage preservation were checked with actual checkpoint serialization in focused fixtures, including a generation-150 fixture.

**Continuation.** Sync the reviewed source, configuration and source-review receipts to the lab checkout. Resume the existing run with the current gate preset, retaining its already implemented two-stage champion gate. The following performs one iteration so the new collection receipts can be inspected:

```bash
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/joint7_preflight.py \
  --root /data/not_backed_up/yxu209/ai-training \
  --run artifacts/step5/v4-selfplay-joint \
  --evidence artifacts/step5/v4-selfplay-joint/state.json \
  --config configs/step5/joint7-g150-gate72.json

python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training \
  --config configs/step5/joint7-g150-gate72.json \
  --device cuda --inference-devices auto \
  train --run artifacts/step5/v4-selfplay-joint \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source --iterations 1

python scripts/export-training-reports.py \
  --root /data/not_backed_up/yxu209/ai-training \
  --run artifacts/step5/v4-selfplay-joint
```

Preflight reads the actual stopped run's current state, verifies its checkpoint and lineage, and checks the pinned transition. It does not require the run to remain at generation 150. Stop if preflight fails. A partial iteration from older source must be recovered with that source; stale partial teacher targets are not reused. The reviewed transition preserves weights, Adam moments/steps, RNG, progress, population, accepted champion, IL lineage and scheduling counters. Fresh ordinary utility calibration is still required before teachers can run. No long training was launched as part of this change.

Use the exported per-phase rows to assess label yield and completed work over multiple collections. A higher qualified-label rate or more supported plans alone does not demonstrate stronger play; that requires frozen checkpoints evaluated on matched, independent setups under the existing evaluation protocol.
