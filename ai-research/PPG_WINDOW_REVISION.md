# PPG acceptance intervals and separate GAE health window

Date: 2026-10-20. Applies to the existing v11.2 self-play run, including g185
(59,200 committed ordinary games), and fresh generated configs.

## Implemented behavior

- Remove the fixed-IL `P(pass) <= 2 * IL P(pass) + 0.05` check completely:
  no eligible-position scan, pass/fail metric or shared-health decision input.
  Old `behavior` flags are discarded when computing new health. This applies
  to auto IL-KL limits, LR health, both historical-pool samplers, GAE and
  snapshot-critic adoption/recovery. Previous sales-check removal remains.
- Preserve ordinary aggregate passes-per-seat, VP, zero-VP and insolvency
  checks, fixed-probe KL/group limits and the LR recovery latch. The old probe
  cache, manifest, reference probabilities and receipts are reused unchanged.
- PPG contract `ppg-value-phase/4.0.0` defaults to **16 optimizer steps per
  acceptance interval**, including a shorter last interval in each epoch.
  Accept an interval only when held-out loss does not increase. A rejected
  interval restores its starting model and value Adam exactly, retaining
  earlier accepted intervals; stop after the rejection.
- Keep the **0.002 cumulative PPG policy KL cap** and its per-step checks.
  The reference is frozen once after PPO; it is never reset at an acceptance
  boundary. A violating step rolls back, and any safe prefix still needs the
  held-out-loss check. Retained/attempted counts and row presentations exclude
  rejected work correctly. Value LR stays **3e-6**, batches **4096**, max epochs
  **3**, clone coefficient/trip history and all PPO settings stay unchanged.
- Stage gate `staged-gae-gate/2.0.0` retains **20 consecutive healthy
  observations** for lambda .98 → .95. Its separate rolling 20-iteration
  window must contain **at least 16 accepted PPG steps across at least 3
  healthy iterations**. These are initial experimental support thresholds:
  one standard interval of retained work, distributed across several
  collections. Zero-step iterations alone never reset the counter.
- Economic health, completed warmup, nonempty PPO, no PPO early stop, finite
  PPO categorical KL <= 1.5 times target, finite self-play held-out learner
  EV and no PPG KL trip remain required. An unhealthy iteration resets the
  streak. Insufficient accepted work blocks advancement without resetting
  an otherwise healthy streak. Work expires when it leaves the window.
- Keep the nonnegative common-fit rating slope and existing PL/v4-watch
  conditions. A complete healthy window is not sufficient without rating
  support. Keep the initial lambda 1 → .98 gate, mixed-table pooled EV window,
  snapshot-weight recovery and all their distinct support requirements.

The checkpoint contract is `actor-value-phases/7.0.0`. The new stage window
starts empty at the **current committed generation**, without reconstructing
credit from old reports. Existing lambda, stage origin, counters, streaks,
mixed-table EV evidence, snapshot state, both optimizers, RNG, beta and
checkpoint lineage are preserved. At a g185 adoption, the earliest complete
new observation window is g186–205; rating/health/support may delay advancement.

The g185 PPG cache remains compatible because targets, rows and holdout
selection do not change. Current plus previous rows are still used, and the
previous cache is deleted only after the next checkpoint/state commit.
Interrupted startup repeats the same migration; committed restarts restore
the window exactly. Fresh health configs use `auto-il-kl-limit/1.2.0`;
existing signed 1.0/1.1 configs remain readable and need no probe rebuild.

## Reporting and runtime

Iteration reports and compact history retain interval acceptance/rejection,
attempted/retained steps and rows, held-out losses, cumulative KL and the GAE
window's support, origin and blockers. Console adds `value_intervals`,
`GAE-health` and `PPG-window`; retired `pass_fail` disappears.

For g185's 54-step epoch, a fully attempted epoch now has four held-out
acceptance evaluations (16 + 16 + 16 + 6), previously one. Up to three epochs
means up to twelve instead of three. Per-step policy-probe work is unchanged;
more work may be retained or training may stop earlier. Production wall-time
impact is not established. The window adds at most 20 small scalar records,
with no extra inference or games. Do not equate accepted training, higher EV,
or passing these tests with improved stability or playing strength. Review
future, out-of-window EV and scheduled rating uncertainty.

## Continuation

Update the lab checkout, then generate the new config from its **actual saved
run recipe**. This helper writes only the config; it never edits run state,
weights or caches, and rejects unrelated/unreviewed changes:

```bash
python scripts/joint7-ppg-window.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppg-window.json

python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-window.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 185 --iterations 20
```

The included config is also exactly the exported g185 recipe plus the two
requested PPG/GAE config-block upgrades. If training has advanced, use its
actual committed generation for the optional `--expected-generation` assertion
or omit that assertion. No rewind, new IL or g110 hotfix is required.
Incompatible pending old-source collections remain rejected by freshness rules.
Keeping the previous config applies the source-only IL pass-check removal;
the new config is needed to opt into shorter PPG intervals and the GAE window.

## Validation

**219 distinct focused checks passed**, including all new interval/window
tests and g185 source/config admission. They verify a later loss rejection
retains the first 16 steps and their exact Adam state; a KL trip at step 18
can retain only the 17 safe steps after held-out acceptance; and the final
short interval is checked. Sparse accepted-work, window expiration and
economic/PPO/PPG/rating blockers are covered.

The **44.47-second**, two-thread CPU smoke used the **5,928,616-parameter
v11.2 model** and authored g185 checkpoints. Continuous and restarted branches
each ran g186/g187 with one PPO and one accepted PPG step per iteration.
Both reused **36 previous critic rows**, kept value LR **3e-6**, and produced
PPG probe KL **0.00000269** and **0.00000184**. Weights, both optimizers, RNG,
GAE/critic/IL-KL state and the next update matched exactly after restart.
The new window began at g185; the original authored checkpoint was unchanged.
This tiny smoke does not exercise a full 16-step interval; the transaction
regressions above exercise those boundaries separately.

The helper was also run against the actual exported g185 state: only
`advantage` and `value_phase` differ in the new config. The original 2,382
source approvals remain unchanged; 155 exact continuation paths were added.
Learning compatibility is unchanged. Existing model/training checkpoints
were not edited.

Three additional legacy resume checks remain incompatible with the current
project: the old v2 test expects `learning/1.0.0` to equal current
`learning/4.0.0`; two compact-audit tests combine an archived source with
current compatibility without a matching approval. The pre-task registry
already lacked those approvals. These are documented in `validation.json`;
admission was not weakened to make them pass. The broad-suite count is
therefore not an all-tests-pass claim.

Five older g55/g85 JSON fixtures were absent during later validation. The
affected checks passed using read-only `git show HEAD:<path>` fixtures,
without restoring or editing the removed result folders. An initial test
runner invocation also treated `-v` as a module; that harness parsing error
was corrected and does not represent an application failure.

Validation receipts and the isolated CPU smoke are recorded under
`artifacts/ppg-window-20261020/`. Tests cover exact model/value-Adam rollback,
cumulative KL across intervals, shorter tails, sparse accepted work, expiration,
economic/KL/rating blockers, legacy probe reuse, source/config admission and
restart continuity. The smoke uses authored v11.2 weights and rows with the
exported g185 economic baseline, not the unavailable live lab checkpoint.
No long training, native-game campaign or rating campaign is launched.
