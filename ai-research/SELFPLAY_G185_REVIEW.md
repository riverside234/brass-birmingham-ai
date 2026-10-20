# Joint7 generation 185 review

**Follow-up implemented:** the IL pass-probability guard is now removed too,
and the two proposals below now have an explicit continuation upgrade:
[16-step PPG acceptance and separate 20-iteration GAE health evidence](PPG_WINDOW_REVISION.md).
Use that report's continuation command to enable them. Findings below describe
the original g185 export and the preceding sales-only repair.

Reviewed `results/joint7-v11-2-grouped-lr-critic-g0185-20261020T090519451283Z`,
committed at **generation 185 / 59,200 ordinary games**. Report directory indices
are zero-based: `iteration-0184/report.json` describes the update to g185.
This export has complete reports for g56–185, including the g108 report missing
from the earlier g110 export. Comparisons below use the newly available evidence.

## Findings

| Evidence | Result |
| --- | --- |
| PPO, g111–185 | No early stops in 75 iterations. Over g161–185, median categorical KL **0.006037**, range **0.004327–0.007908**. |
| LR controller | Raised at g145 and g164. Current multiplier **1.5625**, PPO LR **9.375e-6**. No reset or unexpected LR cut. |
| Latest five-iteration self-play health | **105.69 VP/seat**, **3.094%** insolvent seats, **0.656%** zero-VP seats, **0.511** passes with alternatives per seat. This does not show an insolvency surge. |
| Current collection | 160 self-play / 144 history / 16 v4 tables; 640 / 288 / 32 learner trajectories. The 50/45/5 table mix correctly produces roughly 2/3, 30%, 3.3% of learner trajectories. All five v4 training profiles occur; pottery-flex remains evaluation-only. |
| PPG, g111–185 | **2,623 / 6,350** attempted steps retained (**41.3%**). **42/75** phases accepted no steps. 70 phases stopped on increased held-out loss; five finished their epoch cap. Maximum attempted policy KL **0.000571**, below the unchanged **0.002** cap. |
| PPG comparison | g86–110: 15/25 zero-step phases; g111–185: 42/75. Smaller LR has not established a material sustained acceptance improvement. Rejected steps/epochs restore weights and value Adam; accepted held-out loss never increased. |
| Holdout | All 75 new-split collections contain **8 self-play / 7 history / 2 v4** held-out games. Current/previous PPG caches are reused normally; g111 is the intended one-time migration exception. |
| Mixed-table EV, g166–185 window | History **0.3876**, v4 **0.3535**, paired self-play **0.3914**. Supported by 126 history / 36 v4 / 144 paired self-play games from 18 healthy contributing iterations. Both mixed types meet the existing 0.05 tolerance. |
| Snapshot critic weight | Still **0.5**. Current learner EV **0.4043**, fixed recovery baseline **0.4470**. No five-iteration recovery streak has qualified. Do not interpret this sequential comparison as a causal measure of snapshot benefit. |
| GAE | All types use **λ=0.98**. The latest healthy streak is two; the longest since g110 is four. Accepted PPG work is currently required on every iteration in the 20-iteration stage streak. |
| IL-KL | Current distance **0.41360**, overall limit **0.44457**, group limit **0.88913**, recovery latch clear. Canal-action / rail-action KL **0.71953 / 0.72016**, both within budget. |
| Shared health failures | Only g167–168 since g110, each from one of 511 pass-probability positions. No sales-count health failures occurred in g111–185. Those pass checks remain active. |
| PFSP | Twelve retained members including IL; next pinned IL share **1/13**. Effective scored pool size **9.39**. No pool regression alarms. Current scheduled capacity is IL + 16 scored models, rising later under the existing plan; configured eventual maximum remains 24 total. |
| Rehearsal | Weight reaches zero for g181 onward, as configured from 180 committed iterations. Preparation drops from about 40 seconds to effectively zero. This is a scheduled change, not missing rehearsal data. |

The 0.002 value-policy cap is not causing the current PPG rejection pattern.
Increasing it would not address the observed held-out-loss stops. Current value
LR stays 3e-6 and acceptance/rollback safeguards remain unchanged.

The runtime checks confirm the expected game progress, mixture, learner seats,
exploration/temperature, PPG LR/cap/loss acceptance, holdout quotas,
two-iteration cache lineage, disabled teacher and normalized PFSP distributions
across all 75 iterations. These checks inspect exported evidence; raw replay
and checkpoint tensors are not included in this report bundle. Latest inference
reports zero expired/cancelled requests and no service error. Its insolvency
cost remains 0.0075 per event, capped at 0.075.

## Strength and uncertainty

The latest rating is **g180**, not g185. In its common PL fit:

- g180: **475.42 points**, setup-cluster bootstrap 95% interval **[412.15, 552.84]**.
- Versus g110: **+109.85**, paired interval **[59.34, 171.98]**.
- Versus g140: **−9.37**, interval **[−53.94, 37.89]**.
- Versus g160: **−2.90**, interval **[−66.62, 56.28]**.

This supports improvement over g110 under the rating protocol, but no clear
improvement or regression since g140. The g160/170/180 point-estimate slope is
−0.145 PL points per iteration, so the configured GAE rating check also blocks
the next lambda stage. This tiny slope is not evidence of a meaningful decline.

Separate g180 panels: 32 games / eight setups each. Learner win share is **1.0**
against IL and **0.6875** against v4; the v4 setup-bootstrap interval is
**[0.59375, 0.8125]**. An all-win IL bootstrap interval is degenerate and does not
establish a true 100% win rate. Deployment qualification remains human-reviewed.

## Implemented repair: sales are reporting only

Both IL sale-probability and aggregate sales-per-seat checks are now absent from
shared economic health. No sale count, sales threshold, sale probability, or
archived aggregate sale-failure flag can:

- tighten/grow IL-KL limits or control its recovery latch;
- permit/block PPO LR increases;
- permit/block self-play or mixed-table GAE progression;
- permit/block snapshot critic adoption or weight recovery;
- permit/block quality/PFSP population admission.

Ordinary action-family/sales statistics remain report diagnostics. VP, passes,
zero VP, insolvency, fixed-IL pass probability, and their existing evidence and
PPO/PPG guards remain active. Removing sales did not retroactively improve any
recorded metric or repair an earlier checkpoint.

Fresh config generation uses `auto-il-kl-limit/1.1.0`, without `sells_ratio`.
Existing signed 1.0 configs still load; their retired `sells_ratio` has no
decision effect. Old baseline sales fields are ignored during validation and
removed from newly committed controller evidence. All active thresholds and
baseline measurements remain exact. Sales are no longer mandatory health
inputs, so even absent or nonfinite sales metadata cannot fail shared health.
Other missing/nonfinite health inputs still fail closed.

Continuation preserves the current IL-KL limit/latch, healthy windows, iteration
and game counters, model weights, both Adam states, RNG, population, lineage,
GAE lambda and snapshot weight. It grants no retrospective health credit and
does not reset existing streaks. The existing source-transition path invalidates
utility-calibration readiness for ordinary recalibration, as before.

The fixed IL probe is reused with its exact original receipt, bytes, selection
and reference probabilities. A fresh sale-free config can reuse an otherwise
identical older probe. Every other probe config/source mismatch is rejected.
No new validation-shard load, IL run or teacher collection is needed.

Changed implementation: `recovery/il_kl_limit.py`, `il_kl_probe.py`, `config.py`,
latest config template and reviewed-source/IL-handoff registries. Focused tests,
an authored continuation smoke and this lab documentation accompany the change.

## Continue from generation 185

The following existing config is byte-for-byte equal to the export's recorded
run config. **Reuse it**, including its legacy health-contract metadata:

```bash
python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic-g110-ppg.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 185 --iterations 20
```

After updating the lab checkout, this continues with g186–205. Do not rerun the
g110-only hotfix or rewind the run. No new config is required for this source-only
repair. `--expected-generation 185` is an optional assertion, not a restore
command; use the actual committed generation if the run has advanced.
Pending old-source collections remain rejected by the existing freshness guard.
The actual lab g185 checkpoint is not present locally and was not modified.

## Suggested next experiments

1. **Shorter PPG acceptance intervals.** Current g185 epochs attempt 54 steps
   before the held-out-loss decision. Consider measuring/accepting prefixes
   every 12–16 steps, while retaining the 0.002 policy cap and exact rollback of
   rejected weights/Adam. This could retain useful work before later overshoot,
   at the cost of more held-out evaluations and greater selection pressure.
   Validate out-of-window EV; this is a proposal, not implemented here.
2. **Separate GAE health from compulsory PPG progress.** Consider the existing
   bounded EV evidence-window approach for the λ=.98→.95 stage, so a fully
   restored, loss-rejected PPG phase can count as healthy when supported critic
   evidence and recent accepted value work remain adequate. Keep PPO/pass/econ
   checks and the rating condition. Current lambda/streak rules are unchanged.
3. **Observe the post-rehearsal period before further tuning.** Rehearsal just
   reached zero at g181. Track the next scheduled rating, v4 panel, per-table EV
   and IL-KL together. Keep the mixture, reward penalty, snapshot baseline and
   LR recipe unchanged while measuring that transition.

## Validation

**173 distinct focused tests passed** across health, LR, GAE, snapshot recovery,
both history samplers, PPG/cache migration, source review, rating retention and
report export. Twelve report invariants passed across all 75 post-g110
iterations. The exact g185 signed config/source transition was admitted, with
unrelated config edits and unreviewed source changes still rejected.

The **32.36-second** smoke used the 5,928,616-parameter v11.2 model and a real
checkpoint roundtrip. Authored g186/g187 each accepted one PPO and one PPG step;
both reused 36 previous critic rows and kept value LR 3e-6. Continuous and
restarted branches matched exactly in weights, both optimizers, RNG, GAE/critic
state and IL-KL state. Original authored g185 checkpoint bytes stayed unchanged.
No long training or native-game/rating campaign was launched.

Validation logs, source receipts, report analysis and the bounded g185
continuation smoke live under `artifacts/selfplay185-review-20261020/`.
The smoke uses authored CPU model weights and rows with the actual exported
g185 IL health baseline; it is not a live checkpoint or a playing-strength test.
The validation receipt records counts, timings and the unchanged learning
compatibility hash. This repair adds no model passes, games or optimizer steps;
material training-speed or playing-strength improvement is not claimed.
