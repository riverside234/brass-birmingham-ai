# V4 gate correction: released tables go to history

Date: 2026-10-20. Reviewed checkout: `77cf9b5` (`add res`), following
`5c8bee8` (`Add V4 share gate recovery`).

The user's clarification replaces the original post-gate allocation with
**160 self-play / 154 history / 6 v4 games per 320**. Self-play remains 50%;
history becomes 48.125%, and v4 becomes 1.875%.

| Allocation | Self-play | History | V4 | Learner seat trajectories |
| --- | ---: | ---: | ---: | ---: |
| Before qualification | 160 | 144 | 16 | 960 |
| After qualification | 160 | 154 | 6 | 960 |

The qualification signal is unchanged: the latest completed rating v4 panel
must exceed 95% learner win share, strictly. It is the existing 32-game,
one-learner/three-v4 panel with official ties. No new evaluation or automatic
reversal is introduced. PPG and v4 search remain disabled.

## Generation 213 inspection

The received bundle is
`results/joint7-v11-2-grouped-lr-critic-g0213-20261020T173708767483Z`.
Its committed state records **generation 213 / 68,160 ordinary games**.
The PFSP pool agrees with that progress and contains pinned IL plus 13 scored
snapshots. The latest checkpoint reference is generation 213.

The gate was adopted at generation 210 / 67,200 games and is **inactive**.
Its latest panel is still generation 210, with win share **0.8125**. Therefore
none of the completed iterations used the mistaken 170/144/6 allocation.
The correction does not require a restart or rollback.

The two full reports in this bundle use zero-based directory indices:
`iteration-0211` describes generation 212 and `iteration-0212` describes
generation 213. Both show:

- 160/144/16 ordinary tables and the requested 8/8/2 parent-game EV holdout.
- PPG disabled, zero PPG optimizer steps.
- Actual PPO critic work: 46 and 44 optimizer steps respectively.
- GAE lambda 0.95, shared economic health passing, and no PPO early stop.
- Exact categorical PPO KL of 0.005796 and 0.007177.

Generation 213's ordinary learner mean VP is 111.60, with 5 zero-VP seats
and 28 insolvent seats out of 960. These are training diagnostics; there is
no new rating panel establishing a strength change.

The extraction is saved in
[g213-review.json](../artifacts/v4-history-g213/g213-review.json).
The received files are analysis reports, not local copies of the lab's full
checkpoints. The recovery/train commands below validate those live files on
the lab machine.

## PFSP adjustment and seat rotation

All ten transferred tables use the normal two-learner/two-history path.
Both historical opponents are independently sampled from the same frozen
per-iteration PFSP distribution as the existing history tables. Recorded
probabilities, pinned IL share and source/behavior checks remain active.

History exposure rises from **288 to 308 opponent seats** and from **576 to
616 learner/opponent pairwise comparisons** per iteration, a 6.94% increase.
The estimator consumes all actual extra outcomes. Its five-iteration decay,
S/C history, prior, hardness weighting, uniform floor, membership, admission,
capacity, and reciprocal IL share are preserved. There is no artificial
rescaling or reseeding of old evidence.

History learner-pair rotation now uses the actual historical-table ordinal.
The original quota-only helper rejected indices reassigned from v4, and
would not have rotated those additional tables correctly. The corrected
154-table plan gives each of the six learner pairs 25 or 26 games, continuing
the ordinal across iterations and resumes. The six v4 tables still cover all
six learner pairs once; the five admitted bot variants each occupy two or
three of the twelve frozen opponent seats.

`quality_pool.exposure` reports actual history tables, fraction, opponent
seats, pairwise comparisons and estimator half-life. This survives compact
history together with `v4_share_gate`. Original historical table indices
retain their opponent draws/probabilities; learner pairs include the extra
tables in their rotation.

The existing LR controller sees the changed actual table composition and
resets its measurement window when the gate activates, preserving its
multiplier and optimizer state. It does not reset merely because this
inactive gate's configuration is corrected.

## Continuation and identity

New configs use `v4-table-share-gate/2.0.0` with
`released_tables="historical"`. Existing v1 configs keep their recorded
semantics so old collection provenance is not silently reinterpreted.
The original config migration also remains reproducible.

The explicit reviewed v1-to-v2 transition preserves gate adoption origin,
activation, panel receipts and all other training state. It records the
new allocation origin at committed progress. Even if a v1 gate had already
activated, old transfers are not retroactively counted as historical games.
The actual v4 ordinal remains continuous.

For generation 213, **only `v4_share_gate` changes in the run config**.
The already-adopted 8/8/2 holdout is unchanged; GAE/PPO critic evidence is not
reset again. Model tensors, both optimizer states, RNG, counters, population,
PFSP estimates, IL/model identities and checkpoint lineage are preserved.
All 4,446 earlier source approvals remain unchanged. Learning compatibility
remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.

After pulling this revision, run from the lab repository root:

```bash
python scripts/recover-step5-iteration.py \
  --root "$PWD" --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 213 --apply && \
python scripts/joint7-v4-share.py && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-v4-history.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

The first command handles the Ctrl+C interruption under the run lock. It
verifies the committed checkpoint, optimizer and live opponents, and moves
only an unfinished next-iteration directory to `.interrupted-iterations`,
preserving its bytes. It is a no-op if there is no partial collection.
An uncommitted checkpoint/report is rejected for separate commit recovery.
It never replaces generation 213 or relabels partial games for the new source.

The helper now writes `joint7-v11-2-grouped-lr-critic-v4-history.json` so the
existing `v4-share.json` file is retained. It refuses a conflicting output.
The IL directory name is unchanged. Subsequent runs can reuse the new train
command without the recovery/helper steps. `--iterations 20` means twenty
additional iterations, through generation 233; use 17 to finish the original
twenty-iteration batch that began at generation 210.

## Validation

- **130 distinct focused tests passed**, including targeted reruns after two
  archived interruption fixtures were corrected to use appropriate source
  approvals. No production architecture guard was relaxed. Coverage includes
  strict gate threshold, receipts, old/new allocations, late migration,
  restart continuity, 154-table PFSP outcomes/probabilities and pair coverage,
  holdout quotas and branch exclusion, preserved GAE/optimizers, LR composition
  changes, source reviews, interrupt recovery and search-free v4 policies.
- The current-model isolated PPO smoke passed on CPU/two threads:
  **5,928,616 parameters**, 128 authored Features-4 rows from eight game
  identities. PPO took **1.151 seconds**, total smoke **5.172 seconds**.
  Disabled PPG took zero optimizer steps and created no cache. The next
  update after checkpoint/resume matched the uninterrupted path bitwise,
  including model, both optimizers, gate payload and RNG.
- The config helper passed against the actual generation 213 exported state
  and changed only the gate block. It did not write the run.
- The maintained PFSP simulation was rerun for both **144 and 154 history
  tables**, using 16 dynamic seeds at scored capacities 16 and 23. Both runs
  retained all deliberately forgotten PFSP members (zero evictions out of
  16 seeds per cap). At cap 16, median half/90% recovery times were 7/14
  iterations in both cases; at cap 23 they were 6/14 versus 7/14.
  This synthetic experiment does not demonstrate a stability or strength gain.

Receipts:
[validation](../artifacts/v4-history-g213/validation.json),
[resume tests](../artifacts/v4-history-g213/resume-tests.log),
[integration tests](../artifacts/v4-history-g213/integration-tests.log),
[PPO smoke](../artifacts/v4-history-g213/ppo-smoke/report.json),
[PFSP 144](../artifacts/v4-history-g213/pfsp-144.json),
[PFSP 154](../artifacts/v4-history-g213/pfsp-154.json).

The integration log contains the two initial fixture failures; the resume
test log records both passing after correction. No long training or rating
campaign was launched, and existing training checkpoints were not modified.

The gate schedules no extra games and no extra evaluations. Learner seat
trajectory count and self-play branch eligibility remain unchanged; ten
more games use frozen neural history opponents instead of v4 bots. Wall
time depends on those policies and game lengths and was not benchmarked
on a full native collection.

Changed implementation: `v4_share.py`, `ordinary.py`, `pfsp_pool.py`,
`resume.py`, the config helper, fresh template, new continuation config and
source review registries. Tests and the lab/simulation documentation were
updated alongside this report.
