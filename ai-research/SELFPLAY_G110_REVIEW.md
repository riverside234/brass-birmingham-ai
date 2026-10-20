# Joint7 generation 110 review and continuation repair

The subsequent [PPG refinement](PPG_REFINEMENT_REPORT.md) implements the smaller
value updates and stratified v4 holdout proposed here. Use its updated g110
continuation command to adopt both changes together with this repair.

Reviewed export: `results/joint7-v11-2-grouped-lr-critic-g0110-20261019T233943412893Z`.
The run has committed generation **110 / 35,200 ordinary games**, including its
completed 128-main-game + 64-panel-game rating. The exception occurred during
rating-history compaction, after that commit. Resume from 110; do not restore an
older iteration or repeat the completed rating campaign.

## Findings

The export contains 24 of the 25 generation 86–110 reports. Generation 108 was
outside the latest-two selection but had not entered compact history before the
interruption. Statistics below use those 24 available reports unless stated
otherwise. The exporter now includes any still-available committed report needed
to fill such gaps; it cannot reconstruct a report already deleted from the lab.

| Evidence | Result and interpretation |
| --- | --- |
| PPO exact categorical KL | Median **0.004041**, range **0.003260–0.004439**; no early stops. PPO LR remained **6e-6**, with anchor distance blocking increases. |
| Latest five-iteration self-play health | **104.43 VP/seat**, 2.589 sells/seat, 0.417 passes/seat, **0.375% zero-VP seats**, **3.1875% insolvent seats**. The initial insolvency baseline is 3.46875%; these aggregates do not show a current insolvency surge. |
| Generation 110 learner collection | **65 insolvency events**, 28 affected trajectories among 960 learner trajectories; old mean penalty 0.00033854. |
| PPG value phase | **983 of 1,971 attempted steps retained**; **15/24** phases retained zero steps. **19/24** stopped for increased held-out loss. Raising the policy-KL cap would not address this dominant rejection reason. |
| Learner held-out EV | Latest **0.4388**, below the fixed 0.4470 baseline. Snapshot weight remains **0.5**; the sustained recovery gate is working as configured. |
| Mixed-table EV | Historical pooled EV **0.3837** versus matched self-play **0.3890**, using 152 games. v4 pooled EV **0.1766** versus **0.3899**, using 17 games from 13 contributing iterations within the bounded 20-iteration window. The latest single-game v4 EV of 0.6756 does not establish sustained recovery. |
| GAE | Lambda **0.98**, with six healthy iterations in the current streak. No premature move to 0.95 is warranted. |
| PFSP | IL plus g60/g75/g85/g100/g110: six models, next pinned IL share **1/7**. No regression alarms. The small pool reflects the diversity/admission rules, not a capacity bug. |

Generation 110 scores **372.62 PL points**, with a 95% bootstrap interval of
**[316.52, 432.86]**. In the same common fit, its paired difference versus g70 is
**+110.90 [57.43, 156.13]**, supporting a directional improvement within the rating
model. Differences versus g90 (**+0.33 [-52.55, 63.22]**) and g100
(**+31.10 [-34.73, 96.30]**) do not establish further improvement.

The separate eight-setup/32-game panels report learner win shares of **0.9375**
against IL and **0.59375** against v4, with intervals **[0.8125, 1.0]** and
**[0.46875, 0.71875]** respectively. The last three v4 panel shares rose from
0.53125 to 0.5625 to 0.59375. These are diagnostic panels; deployment remains
subject to review. Collection composition stays **50% self-play / 45% history /
5% v4**, and teacher/v4 search stays disabled.

## Fixes and requested settings

1. **Rating compaction:** live GAE evidence contains tuples, while JSON stores
   lists. Comparing the live state directly to its saved JSON incorrectly
   reported concurrent advancement at the first rating expiry, g110. Compare
   the complete state in its JSON representation instead. Genuine same-generation
   edits still fail the stale-writer check. Restart reuses the committed rating.
2. **One-time IL-KL hotfix:** the reviewed helper changes only the saved limit
   from **0.152587890625 to 0.2288818359375**. The linked factor-two group limit
   becomes **0.457763671875**. Fresh initial limits, calibration cadence, health
   evidence, fixed probe and recovery latch are unchanged. A receipt binds the
   override to this exact g110 checkpoint; repeat application is idempotent.
   The helper backs up `state.json` and never rewrites the checkpoint. The next
   ordinary checkpoint records the adjusted state normally.
3. **Reward:** fresh defaults and the prepared continuation config use
   `event_cost=0.0075`, with **`max_cost=0.075` unchanged**. Ten events now reach
   the cap, compared with fifteen previously. Search reward metadata matches
   the learner, although teacher search remains disabled. Existing configs are
   not silently edited. The exact reviewed transition preserves all weights,
   both optimizer states, RNG, progress, population and checkpoint lineage.
4. **Previous PPG rows:** their saved numeric returns use the old coefficient.
   Exclude those rows for the first new-reward iteration; do not relabel them.
   Preserve the old cache until the next successful commit, then clean it up.
   Subsequent iterations resume the usual two-iteration buffer. Reports retain
   the transition and the number of excluded rows. Utility calibration is
   refreshed against ordinary data with the new reward.
5. **Report exports:** include committed full reports that have not yet entered
   compact history, even outside `--latest`. The manifest records them under
   `uncompacted_iterations`; pending collections are still excluded.

The latest measured global IL-KL is **0.315263**, already above the new limit.
Canal-action and rail-action KL are **0.481032** and **0.539772**, also above the
new group limit. This requested adjustment therefore **does not immediately
clear the recovery latch or enable LR raises**. No additional limit relaxation
has been introduced.

## Validation and scope

**169 distinct focused tests passed across the main suites and focused reruns.**
Coverage includes real exported-signature admission, unauthorized-edit rejection,
exact reward migration, Adam/RNG preservation, idempotent and tamper-resistant
hotfix application, rating expiry/restart/stale-state checks, PPG cache lifecycle,
PFSP, auto IL-KL/LR gates, GAE, snapshot-weight recovery and report exports.
Older numeric reward fixtures and an archived champion-toggle approval fixture
were updated; repaired cases were rerun. Logs and the validation receipt are in
`artifacts/selfplay110-review-20261019/`.

An isolated **13.03-second CPU smoke test** used authored mixed-table v11.2 rows
with the 5,928,616-parameter model. Two independent resumes of an old-reward
checkpoint each completed one PPO and one accepted value update. Next weights,
both optimizers, RNG and controller state matched exactly. The old checkpoint
was unchanged, old-reward rows were excluded, and their cache was removed only
after commit. See `tests/Recovery/smoke_reward_transition.py` and
`artifacts/selfplay110-review-20261019/reward-smoke-final/report.json`.

No lab checkpoint was modified here, no long training or native-game evaluation
was launched, and the smoke test says nothing about playing strength. The
changes add no collection or evaluation games. The first continuation needs
fresh utility calibration and uses fewer value rows; steady-state buffering is
unchanged. Export bundles can contain a few more reports after interrupted
cleanup.

## Lab continuation

After updating the lab checkout, run from the repository root:

```bash
python scripts/joint7-g110-hotfix.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-g110.json \
  --apply && \
python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic-g110.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 110 --iterations 20
```

Omit `--apply` to validate and preview without writing state/config. The helper
requires the actual verified checkpoint and frozen opponents on the lab. It
accepts an identical existing output config but refuses to overwrite different
settings. No IL regeneration or rename is required. For later continuations,
use the same config and the actual current expected generation, or omit that
optional assertion; do not keep asserting 110 after training advances.

## Suggested next experiments, not applied

- Compare smaller value-phase updates in an isolated continuation while keeping
  both held-out-loss and policy-KL guards. The present value LR uses base PPO LR;
  any separately tuned value LR should have an explicit, reviewed config contract.
- Consider deterministic whole-game holdout stratification by table type to
  stabilize v4 critic evidence. Preserve exclusion of every seat of a held-out
  game from critic fitting and caches; do not change the 50/45/5 mixture merely
  to make the gate easier to pass.
- Review the next rating and the IL-KL recovery trajectory before changing PPO
  LR limits, GAE lambda or snapshot weights further. More retained value steps
  or higher entropy alone would not establish improved playing strength.
