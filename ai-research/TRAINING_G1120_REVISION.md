# g1120 openings and PPG continuation

Date: 2026-10-25 (America/Chicago). Export:
`results/joint7-v11-2-grouped-lr-critic-g1120-20261026T012745632742Z`.
The timestamp in the export directory is UTC; its committed generation is 1120.

## Changes

- Opening temperature **1.3** for the first **three action decisions per neural
  seat**: counters 0, 1 and 2. The fourth uses temperature 1. Payment, bonus,
  debt and continuation choices do not consume the opening window. Existing
  counting semantics for forced action decisions are preserved; forced rows
  remain excluded from the actor loss.
- Branch eligibility skips those same three decisions. The earliest eligible
  fork is counter 3, with the existing legal-choice/endgame restrictions.
  After a fork, each seat starts a fresh three-action opening window; the
  alternative fork move remains excluded from PPO and critic training.
  The probability remains **1% per eligible decision**.
- The existing decoupled PPO objective covers the same nonforced three-action
  window, including branch continuations. The ratio uses frozen old/current
  raw T1 probabilities; the detached capped weight remains
  `min(2, old_T1_probability / actual_behavior_probability)`, with no minibatch
  renormalization. Actual behavior still includes **epsilon 0.05** and T1.3;
  strict probability verification, legal masks and separate behavior/T1 KL
  safeguards remain active. Returns/GAE are still recorded-behavior returns.
- PPG loss checks return to **every 16 attempted steps**, plus short epoch
  tails, using the full two-iteration held-out set. Periodic KL checks run
  every **32**, with the **0.002** cap, epoch-tail checks and final retained-state
  verification. A step-16 loss acceptance is provisional until that specific
  checkpoint has passed KL. A safe later step-32 probe does not certify a
  different step-16 checkpoint after loss rejection.
  Beta stays at its saved value (**0.5 at g1120**), with adaptive bounds .5–32.
  This migration does not repeat the earlier one-time beta reset.

Versioned identities: opening-temperature/4.0.0, selfplay-branching/4.0.0,
opening-ppo-objective/3.0.0 and ppg-value-phase/19.0.0. Fresh config generation
uses these settings; explicit older recipes keep their original behavior.
The helper defaults to KL32. It supports direct adoption from the g1120 export's
v17 checkpoint; the earlier v18/KL16 continuation command is not a prerequisite.

Model weights, Adam/RNG, iteration/game progress, population, lineage, IL
identity and independent gate evidence are preserved. The LR controller's
existing behavior-regime change handling still clears its decision window,
without resetting its multiplier. Compatible committed MC critic caches are
retained with their original targets/provenance; PPO never trains on those old
trajectories. New collection identities bind the new opening/objective settings,
and old or partial collections cannot be relabeled. The interrupted-iteration
recovery command below archives any uncommitted collection before continuation.

Unchanged PPG settings: every iteration, two cached ages, uniform shuffled
training, two-epoch maximum, .65 × actual PPO LR peak, linear warmup to step 16,
then cosine decay to .001 × peak. Other reward, entropy, opponent, ownership-head
and training settings are preserved. No completed IL or production checkpoint
was modified locally.

## Log check and reporting fix

The g1120 report used T1.5/two decisions, PPG loss8/KL16 and beta .5. It retained
8 of 16 attempted PPG steps. Retained KL was **0.00025719**, below .002. Pooled
loss acceptance covered **15,427 rows / 44 game identities**, including branch
families, from g1119 and g1120.

Fixed an opening-only KL reporting bug: the probe mask hard-coded temperature
2, so the g1120 T1.5 report showed zero opening probe positions despite 2,792
eligible opening training rows. The diagnostic now uses the objective's actual
temperature for both T1.5 and T1.3. Overall raw-T1 and actual-behavior KL guards
were already using the complete probe; this fix changes the opening diagnostic.

## Initial loss16/KL16 validation

**41 tests passed**: eight g1120 regressions (19.6 seconds) and 33 existing
opening/PPG/compatibility regressions (92.1 seconds). Logs are
`artifacts/g1120-validation.log` and `artifacts/g1120-regressions.log`.
They exercise the
three-action boundary, independent seats, post-fork reset, actual mixed
probabilities, stale provenance rejection, decoupled selection, opening-only KL,
loss16/KL16 rollback, and preservation of beta/Adam/RNG/caches/gate state.
The isolated CPU smoke also passed in **28.4 seconds**. Three authored small
PPO/PPG iterations retained 2, 4 and 4 PPG steps; repeating the third from the
saved second checkpoint exactly reproduced weights, PPO/value/auxiliary Adam,
RNG, beta/trips and gate state. Each iteration selected 121 eligible T1.3
opening rows, including a branch-continuation opening. These short phases test
restart and integration; the focused interval regression separately exercised
loss/KL checks at steps 16 and 32, rejecting step 32 and restoring step 16 with
matching Adam state. The initial forced-row fixture assertion was corrected to
select a nonforced seat before the final passing regression run.

The source review preserves all 74,514 historical review records and adds only
the reviewed routes for this implementation and exact configuration transition.
No production checkpoints, long training runs or evaluations were launched.
The smoke is not a server performance benchmark.

## Loss16/KL32 debug review

Reviewed the frozen post-PPO raw-T1 reference, separated loss/KL cadence,
provisional acceptance, model/value-Adam/ownership rollback, retained-work
accounting, two-age holdout separation, strict actual-behavior verification,
and direct v17-to-v19 continuation. The new transition preserves saved beta,
trip history, optimizer moments, RNG, cache files and independent gate state.
It also handles restart after an already-applied v18 signature transition.

Focused checks force step 32 to fail and independently test a safe or unsafe
step 16; they also test a safe step 32 followed by loss rejection with an unsafe
step 16. The latter must return to step zero. Epoch-tail checks remain active,
and the 32-step periodic counter spans epochs. Results are recorded in
`artifacts/g1120-kl32-validation.log` and `artifacts/g1120-kl32-smoke/report.json`.
No change to the existing rollback algorithm was needed for the longer interval.
**32 tests passed**: 30 focused regression/source-review tests in 94.6 seconds
and two fresh-default checks. The isolated loss16/KL32 CPU smoke passed in
**24.1 seconds**, retaining 2, 4 and 4 steps across three authored small
iterations and reproducing the third iteration exactly after restart. It checks
weights, PPO/value/ownership Adam, RNG, beta/trips and gate state. The longer
intervals are exercised by the focused 16/32-step rollback tests, rather than
the smoke's short phases. No additional blocking training bug was found in
these reviewed paths. The prior opening-only KL diagnostic fix remains included.
This follow-up preserves all 74,561 existing source-review records and supports
the g1120 export directly; no previous continuation command needs to be run.

Changed implementation: exploration/opening-objective validation and selection,
branch execution/replay admission, opening-only KL reporting, PPG version/adoption,
fresh config generation and reviewed resume handler. Added
`scripts/configure-training-g1120.py`, the narrow migration module, focused tests
and smoke wrapper, and updated both lab guides.

Loss16 halves the periodic held-out evaluations relative to loss8; KL32 halves
periodic KL evaluations relative to KL16. Tail/retention checks can still add
work, and later KL detection can discard more attempted steps. No server
throughput or playing-strength improvement is claimed.

## Continue g1121–1140

After syncing this code, run from the lab repository root:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 1120 --apply && \
python scripts/configure-training-g1120.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 1120 --kl-check-steps 32 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-training-g1120-kl32.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-training-g1120-kl32.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1120 --iterations 20
```

The helper checks the complete generation-1120 checkpoint and hash, validates
the narrow reviewed transition and writes only the new config. It refuses to
rewind a newer committed run or overwrite a different config. Do not use a
fresh-buffer reset or repeat auxiliary-head adoption. Keep the IL folder unchanged.
