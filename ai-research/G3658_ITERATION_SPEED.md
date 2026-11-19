# g3658: iteration speed (replay load and sampling, the unmeasured commit, the value phase)

The speed changes are code-only: no recipe flag and no new run state. Training data, targets, losses, every check
that decides training, RNG, checkpoints and the qualified implementation are unchanged. The same continuation also
adopts two recipe changes the user asked for at g3658 (section "Recipe changes at g3658" below). Execution identity
`7f8399ea…` (predecessor `7f8431f5…`).

Continue to the g3670 rating with both recipe changes:

```
python scripts/continue-rust-g1190-restart-2.py --run artifacts/step5/joint7-v11-2-g1190-restart-3 --expected-generation 3658 --iterations 12 --share-target 0.1 --value-revision weight=0.5,vp --ppg-beta-floor 0.125
```

Later continuations drop both flags; the settings persist.

## What the g3658 export shows (g3600–g3657, 58 iterations, mean 241 s)

| part | mean | what drives it |
| --- | --- | --- |
| replay load + sampling | 10.4 s (median 5.0) | 5.6 s normally; 29.1 s in the iteration after each cleanup (g36x1, g36x6) |
| unmeasured remainder | 19.1 s (18.2–21.4) | the commit between final diagnostics and evaluation, plus the cache trim |
| value phase | 37.3 s (22.5–57.3) | replay open 15–21 s in 20 of 58 iterations (1 s otherwise); stage 22–28 s spikes |
| cleanup (every 5) | 44.5–54.2 s | plan 37.5–39 s, move and hashes 2.7 s, history 4.3 s; then a 208 s background purge |

- **Garbage collector.** CPython 3.12 (the lab's Python) starts a full pass once a quarter more objects reach the old
  generation. The g3270 collection window froze the heap, so "old" became only the window's own survivors, and the
  ingested replay crossed that quarter again and again. The g3654–g3657 windows paused all trainer threads for
  39–57 s, 26–39 s of it in 7–18 full passes of 2–5 s each. After the window the heap was unfrozen, so the next full
  passes walked everything (11M frozen objects plus the ingested games) wherever they fell.
  - PPO + replay open is nearly constant (56 ± 9 s, correlation −0.61): the same pauses land in PPO critic
    preparation (25.9 s at g3655 against 4.4 s) or freeze values (18.2 s against 5.3 s) or the replay open.
  - The iteration after a cleanup gets more of them: its background purge parsed every `state.json` under
    `artifacts/` while the trainer allocated (g3656: 18 full passes in the window; sampling 28.4 s).
- **Cleanup.** Each due pass walks `artifacts/` for pins and hashes every candidate (7.5 GB at g3655) before moving
  it; the purge walks `artifacts/` again and rehashes the same 7.5 GB before deleting, beside the next iteration.
  Quarantine is 0 hours, so nothing is ever restored from it.
- **Commit (the unmeasured 19 s).**
  - `update_population` hashed all 24 population files twice, one after another, at every commit, although pool
    planning hashes every member (in parallel) before each collection.
  - The 16 MB report was written twice and the 20 MB state twice, all as indented JSON (pure-Python encoder).
  - The PPG and teacher caches each parsed `state.json` again to check the commit.
  - Full collector passes triggered by all of this.
- **Value phase.**
  - The gate scan (`current_validation_seconds`, 2.1 s) measured the same 31k held-out rows, with the same weights,
    as the acceptance scan: g3655–g3657 metrics are bit-identical, including g3657 after 22 head-only steps.
  - The masked-descriptor scan (1.05 s) feeds only the automatic critic-data rollout before snapshot rows switch
    on. This run's critic-data recipe is the fixed one (critic-rows/3.0.0), which never reads it.

## Changes

1. **Collector over the iteration** (`recovery/collection_gc.py`, `iteration-gc/1.0.0`).
   - Collection start: freeze the heap and turn automatic collection off. Reference counting still frees garbage at
     once; only cyclic garbage made during collection waits.
   - Collection end: freeze again (O(1), no pass), then young passes resume with full passes off. The rest of the
     iteration scans only objects made after collection.
   - Iteration end (after the commit and trim): unfreeze and run one timed full pass. `iteration_gc` reports the
     young passes after collection and the full pass with the cyclic garbage it freed, so no cycle lives longer
     than one iteration.
   - An iteration that ends without the full pass (an exception) is settled by the next window.
   - `BRASS_ITERATION_GC=0` keeps the g3270 window; `BRASS_COLLECTION_GC_FREEZE=0` still turns everything off.
2. **Scheduled cleanup** (`recovery/retention.py`). For the trainer's automatic cleanup with quarantine 0 h:
   - the move and the purge check each file's size, not its sha256 (manifest `verification: size`);
   - the purge reuses the plan's reference pins for the manifest the plan just wrote, so `artifacts/` is walked once;
   - other unsettled manifests, manual cleanups and delayed quarantine keep the hashes and the fresh scan;
   - restore and interrupted-move recovery accept size-only entries;
   - `timings.references_seconds` now shows how much of the plan is the pin scan.
3. **Commit** (`recovery/league.py`).
   - `report.json` and `state.json` are compact JSON (16.2 → 9.6 MB and 19.9 → 12.1 MB; 0.8 → 0.2 s each to
     encode on the laptop). Readers parse JSON, and nothing pins the live files' bytes.
   - A population member already in its population file keeps its receipt without rehashing
     (`retention.slim_checkpoint`); a missing file or a changed receipt still stops the commit.
   - `commit_cleanups` hands the state it just wrote to the PPG and teacher caches (`committed_state`).
   - New timings: `commit_seconds`, `trim_seconds`, `gc_full_collection_seconds`, and the breakdown
     `commit_timings` (teacher finish, report assembly and write, lineage, purge join, population, state write,
     owner cleanups, trim). They and `collection_gc`, `iteration_gc`, `previous_cleanup_purge` enter the retention
     history.
4. **Value phase** (`recovery/ppg_g210.py`, `recovery/value_phase.py`).
   - The gate before/after reuse the acceptance measurements when the acceptance rows are the current age's held-out
     games with the same rows per game (`gate_measurement: acceptance`); otherwise both are measured as before.
   - The masked-descriptor scan runs only when the critic-data rollout can read its gain
     (`critic_data.reads_descriptor_gain`; otherwise `descriptor_gain.measured: false`).
   - `value_work.prefetch` reports when the page-cache hint for the previous age returned and when the replay opened.

## Estimated effect (per iteration; nothing measured on the lab yet)

| change | saves | costs |
| --- | --- | --- |
| collector policy | the 39–57 s of pauses in collection (part of that is collection wall time), the spikes in sampling, PPO and the value phase (about 10–20 s average), and most of the unmeasured remainder's passes | one full pass at the end, about 2–5 s (reported); cyclic garbage made during collection waits up to one iteration |
| cleanup | about 2.5 s foreground per cleanup; the 208 s purge shrinks to the deletes; the after-cleanup iteration's +23 s replay and sampling and +17 s collection go with the GC change | a file changed in place with the same size would be deleted unnoticed (it is retired data either way) |
| commit | population hashes (about 1–2 s, more if the files are cold), JSON (about 2 s), two `state.json` parses (about 0.5 s) | none |
| value phase | gate 2.1 s, descriptors 1.05 s, plus the GC spikes above | none |

Overall: replay load and sampling about 10.4 → 5 s, the unmeasured remainder about 19 → 5 s (plus the timed
full pass), and the value phase about 37 → 25–30 s. Collection should also get faster. The cleanup plan
(about 38 s every fifth iteration) is the largest remaining item: the next export's `references_seconds` shows
whether the `artifacts/` pin scan is it.

## Checks after the first iteration

- `collection_gc.policy` = `iteration` and `collection_gc.collections` is empty.
- `iteration_gc.after_collection.collections` has only `0` and `1`; `iteration_gc.full_collection.seconds` is a few
  seconds; `unreachable` stays small from one iteration to the next.
- `timings.commit_seconds` plus the trim and the full pass account for nearly all of the old remainder.
- `value_phase.gate_measurement` = `acceptance`, and `heldout_before` equals `acceptance_heldout_before`.
- At the next cleanup (g3660): `cleanup.verification` = `size`, `hash_reuse.final_delete_rehash` = false,
  `timings.references_seconds`; and g3661's `previous_cleanup_purge.seconds` is a few seconds.

## Tests

- `tests/Recovery/test_iteration_speed_g3658.py` covers:
  - the GC policy: freeze through the iteration, one full pass, settle, and the switches;
  - size-only cleanup with a single reference scan, a size change stopping the purge, size-only restore, and manual
    cleanup keeping its hashes;
  - the unhashed population receipt;
  - the commit hand-over and compact JSON;
  - through `ppg_g210.fit`: the reused gate equals the separately measured one (before, after, steps, presentations),
    and skipping the descriptor scan changes nothing else; `reads_descriptor_gain` for each critic-data recipe.
- `tests/Recovery/test_teacher_revision_g3270.py` keeps its g3270 window test under `BRASS_ITERATION_GC=0`.

## Recipe changes at g3658

### `--value-revision weight=0.5,vp` (deferred since g3318), checked on the current recipe

Dry run on the committed g3658 recipe:
- era heads: weight .15 → .5 for both eras. The Rail-end heads removed at g3528 stay removed, and Rail-end
  stays at 0. The companion config moves forward (`local_scoring_v3.config_forward`).
- total-VP loss weight .1 → .5 in training; the held-out check keeps .1.
- recipe tag: `...-eraheads-w05-norailend-blend50-vp05-cestop16-searchaux-searchrows`.
- `is_on` holds before and after. The reviewed transition accepts it forward and refuses it backward.
- the search rows are unchanged, and the change is idempotent at a later generation. Moving the weight back down
  is refused.
- the era-head companion restores its g3528 checkpoint (no Rail-end outputs) under weight .5, with the same
  parameters.

Cost: no time and no data. The era heads' gradient into the shared trunk is about 3.3 times larger, and the
total-VP term five times larger in training. The early stops still decide on the unchanged checks.

### `--ppg-beta-floor 0.125` (new; `recovery/training_revision_g3658.py`)

- In g3600–g3653 the PPG beta sat at the .25 floor in all 54 summarized iterations. In 32 of them the feedback was
  `at_minimum`: the phase stayed below half the KL cap and would have halved beta. Retained probe KL averaged .0007
  (max .0017) against the .003 cap.
- Two floors held it, and both move .25 → .125:
  - `value_phase.beta_clone_min`;
  - the search-head schedule's Stage C floor, `ppg_beta_floor_after_stage_b`. Without this one,
    `search_value_schedule.beta_floor` would keep .25 in Stage C.
- Hotfix: the first resume under the new floor sets the PPG beta to .125 at once
  (`ppg-beta-floor-eighth-hotfix/1.0.0`, recorded as the PPG adoption and in `value_phase.schedule_migration`).
  The KL trip history, both Adams, RNG, replay and cadence carry over.
- Afterwards beta adapts as before. A clean phase below half the cap halves it, down to .125. A KL hit within
  16 retained steps raises it ×1.5, and a third hit in five doubles it.
- The transition check accepts only .25 → .125, forward. Recipe projections map .125 back to .25 for the reviewed
  comparisons, and the native-backend receipt is rebound with the new floor.
- Collection, labels, search and teacher job ids are unchanged.

Cost: no time and no data. The value phase's policy-cloning constraint is half as strong, so a PPG phase can move
the policy further. The hard KL cap (.003, stop and keep) still bounds each phase.

Checks after g3659:
- `value_phase.schedule_migration.contract` = `ppg-beta-floor-eighth-hotfix/1.0.0` with `beta_before` .25 and
  `beta_after` .125;
- `value_phase.beta_clone` = .125, and `beta_feedback.minimum` = .125;
- the log prints `PPG beta floor: {...}` and the value tag `...-w05-...-vp05-...`.

Tests: `tests/Recovery/test_ppg_floor_g3658.py` covers:
- the floor recipe, its transitions and refusals;
- `weight=0.5,vp` after drop-rail-end, with and without the floor;
- the PPG resume: beta .125 once, Adam kept, later resumes keep their beta, and the way back is refused;
- beta adapting at the new floor;
- the era-head companion resuming at weight .5.
