# Self-play generation 85: review and fixes

Date: 2026-10-19. Run: [generation-85 export](../results/joint7-v11-2-grouped-lr-critic-g0085-20261019T172419550588Z/state.json). Analysis: [structured metrics and integrity checks](../artifacts/selfplay85-review-20261019/analysis.json).

The recorded evaluations support progress through generation 80. The next five generations are not yet rated. PPO remains within its KL guard, PPG accepts more useful work, history sampling is no longer IL-only, and the pooled mixed-table gate has qualified both history and v4. The concerns are rising self-play insolvency, historical-table critic EV below its relative tolerance, and an LR controller still blocked by the recovering IL-distance limit. These observations do not justify silently changing the reward, LR bounds or rollback policy.

## Scope and integrity

Fetched and fast-forwarded the clean checkout from `b97a4e2` to `9fb8836`; the incoming commit adds only this export. All **51 manifest-listed files** match after Git CRLF normalization, with no other hash mismatches, missing files or export warnings. The export contains 85 committed iterations / **27,200 ordinary games**, compact reports through generation 83, full reports for 84–85 and rating slots through 80. It contains reports, not the production model/Adam checkpoints or dense probe tensors.

Generations 56–85 use implementation `997ce5bc26b8416d694f42ddac196feac8b365ee40bc412c6321ec85ae8885a5`, exercising the prior sale-check removal, bounded EV window, PPG safe-prefix fix, five-profile v4 scheduling and frozen-runtime changes. All 30 new iterations report healthy pass/economy checks; the removed IL sale-probability fields remain absent.

The recipe remains 160 self-play / 144 history / 16 v4 games per iteration: **50/45/5**, with 960 eligible learner seat-games. The latest full reports contain five distinct v4 policy identities with 6–7 opponent seats each, rather than the former two-profile pattern. The configured official v4 bank remains `bank-3dc65253a610191d66a5`; teacher search, v4 search, Monitor and Broader Evaluation remain disabled. PL rating slots remain enabled.

## Observed training behavior

These ten-iteration aggregates describe ordinary self-play behavior. Historical and v4 training opposition changed after generation 55, so their raw training win rates are not controlled strength comparisons.

| All-current-model self-play learner metric | g46–55 | g76–85 |
| --- | ---: | ---: |
| VP per seat | 99.72 | 102.62 |
| Insolvent seats | 2.67% | 3.92% |
| Insolvency events per seat | 0.0597 | 0.0897 |
| Zero-VP seats | 0.36% | 0.42% |
| Passes with an alternative per seat | 0.373 | 0.397 |
| Selected sell decisions per seat | 3.009 | 2.792 |

Insolvency increased **1.25 percentage points**, about 47% relative to the earlier seat rate; event frequency increased about 50%. It rose before the lambda change: g66–75 already had 3.48% affected seats. This does not identify a causal mechanism. At g85 no learner trajectory reached the current `max_cost=0.075`; maximum observed cost was 0.045. Raising the cap alone would therefore not change any g85 trajectory's penalty. Keep `event_cost=0.005` and `max_cost=0.075` for this review.

PPO had **zero early stops in all 85 iterations**. In the new 30, exact categorical KL ranges **0.003151–0.004149**, median **0.003642**, against target 0.01. The effective LR stays **6e-6**, multiplier 1.0. Average probe action entropy moves from 1.389 to 1.364 nats between the two ten-iteration windows; entropy divided by log legal-action count stays approximately **0.369**. This bounded ordinary nonforced probe does not suggest a sudden entropy collapse or establish adequate strategic exploration.

### PPG and critic data

PPG accepted **1,193 / 2,477 attempted steps** in g56–85, with accepted work in **13/30 iterations**. There were **zero value KL trips**: 26 phases stopped on held-out loss increases and four completed all three epochs. G76–85 accepted work in 6/10 iterations, versus 3/10 in g46–55. G85 accepted 153/153 steps, reducing held-out loss **0.87369 → 0.86408**, with retained raw policy KL **0.001108 < 0.002**. The cap is not the current bottleneck.

Snapshot critic admission was decided at g56 for subsequent iterations. At that time the historical pool had no scored snapshot. The first scored model entered the pool at g60, and **the first actual snapshot fitting occurred at g61**. Yet the old sequential EV rule reduced its future snapshot weight to **0.5 at g60**, when both PPO and accepted PPG snapshot counts were zero. This is a control bug fixed below; that EV decline cannot be evidence of harm from snapshot training that had not occurred. G85 PPO actually used **12,104 distinct snapshot rows** and 51,526 distinct learner critic rows. The PPG phase used **207,740 distinct rows**, current plus previous iteration, for 623,220 presentations. Previous/current disk caches peaked at about **3.01 GB** and the previous cache was deleted after commit.

The saved cloning coefficient stays at **8**, without any new escalation. Its earlier erroneous increases are not reset retrospectively. Mean aggregate held-out EV dipped to 0.276 in g66–75 and recovered to 0.401 in g76–85. Composition, correlated decisions and changing held-out games limit interpretation of those scalar averages.

### GAE and pooled EV

History qualified at g65 using 79 held-out games; v4 qualified at g66 using nine. Both remained at lambda 1 until the global gate reached 30 accepted-value iterations at g81. **The first collections using lambda 0.98 were g82**, for all three table types. At g85 the global 20-healthy-iteration streak is two: g83 accepted no PPG steps and reset that existing requirement. The rating slope gate is positive. No gate is bypassed or changed here.

At g85:

| Pooled learner EV evidence | History | v4 |
| --- | ---: | ---: |
| Window | g66–85 | g66–85 |
| Contributing iterations | 20 | 12 |
| Distinct held-out games | 150 | 16 |
| Mixed learner EV | 0.280 | 0.390 |
| Matched self-play learner EV | 0.366 | 0.353 |
| Difference | −0.086 | +0.038 |
| Existing tolerance of −0.05 | Below | Within |

History was also below tolerance at g79, g82 and g84. Qualification is intentionally persistent, so this does not trigger an automatic rollback. It should be visible for review, rather than hidden behind `already_enabled` and a healthy streak of 30/10.

The g85 v4 **single-iteration EV of −13.38** comes from one held-out game / 176 correlated learner rows. Its target variance is approximately **0.001247**, and residual variance approximately **0.017933**, so the reported arithmetic is correct. The bounded paired window yields 0.390 across 16 distinct games. This is neither evidence of a −13 pooled critic nor proof that the critic is accurate everywhere. Pooled results still describe online predictions from successive models, not re-evaluation of every old position with g85 weights.

### LR and population

The fixed raw IL KL is **0.26692** at final g85 weights, with largest group **0.45753**. The recovered mean/group limits are **0.097656 / 0.195313**, up from 0.05/0.10 after healthy windows at g65, g75 and g85. The recovery latch therefore correctly blocks LR increases. The controller's slightly different IL distance, 0.26792, is measured after PPO and before PPG; the final limit report includes PPG movement.

At a hypothetical constant distance, six further 1.25× limit increases would be needed for the existing 0.8 recovery rule: about **60 more healthy iterations**, reaching a mean limit of 0.37253. Actual policy drift or failed health can change this. This slow recovery follows the configured rule; it is not an LR restoration defect. Limits, baseline and latch are not reset by this revision.

The historical pool is now **4/24: IL, g60, g75 and g85**. G65, g70 and g80 failed the existing distinctness test, not health. The g85 snapshot joins only subsequent collections. The falling historical training win share after g55 is expected to be confounded by replacing repeated IL opponents with scored current-era snapshots. Do not fill the pool with near-duplicate models simply to reach 24.

## Held-out playing-strength evidence

Use the **same generation-80 PL fit**, with IL fixed at zero:

| Agent | PL points |
| --- | ---: |
| g50 | 271.62 |
| g60 | 256.77 |
| g70 | 259.50 |
| g80 | 327.04 |
| Frozen search-free v4 | 190.84 |

G80 exceeds g70 by **67.53 points**, paired setup-cluster 95% interval **[15.39, 114.69]**. G80 exceeds g50 by **55.42 [4.09, 110.80]**, and v4 by **136.20 [78.25, 200.74]**. G50–g70 differences remain uncertain. The cumulative fit includes 1,024 main games over 32 setup clusters; it converged and reports no regression/v4-watch alert. [Common fit](../results/joint7-v11-2-grouped-lr-critic-g0085-20261019T172419550588Z/ratings/slots/generation-000080/report.json).

The g80 separate panels each use 32 games / eight setups: one learner versus three v4 opponents has **0.531 win share [0.406, 0.656]**, while the two-learner versus two-IL panel has **0.875 combined learner share [0.750, 1.000]**. These are different matchups and cannot be equated with PL pairwise outranking probabilities. Deployment qualification remains pending review. [Panels](../results/joint7-v11-2-grouped-lr-critic-g0085-20261019T172419550588Z/ratings/slots/generation-000080/panels.json).

This supports improvement within the recorded evaluation pool through g80. It does not rate g85 or demonstrate a benefit from GAE: the first lambda-0.98 collection was g82, after the last rating slot. No new strength campaign was launched.

## Implemented fixes

1. **Require actual snapshot training before reducing its weight.** The rollout now consumes PPO's actual snapshot distinct-row count and PPG's accepted snapshot usage. It can reduce the weight only after retained snapshot fitting in that iteration. It holds when a route is enabled but the pool is IL-only, or when snapshot rows were only available/attempted and all updates were rejected. Reports distinguish enabled routing from actual usage and suppress the snapshot-effect delta when no snapshot fitting occurred. The g60 reproduction fails before the fix and passes afterward. Actual PPO snapshot work still counts when PPG rejects all its steps. No checkpoint schema changes or new thresholds are required. **The current run retains its already-saved weight of 0.5**; no retrospective reset to 1.0 is performed.

2. **Correct mixed-table GAE review ages.** The old report counted from EV qualification even when lambda remained 1, displaying 20/19 iterations at g85 although only four collections had used GAE. Reports now retain that qualification origin separately and derive the current effective stage origin from the later of table qualification and global lambda-stage origin. G85 correctly reports origin 81 and four collections. A subsequent lambda stage starts its own review age. Checkpoint state and GAE admission decisions are unchanged.

3. **Expose supported EV deterioration after qualification.** `gate_diagnostics` version 2.1 adds `mixed_ev_status`, including paired pooled EVs, gap, support, freshness and reasons. A qualified active table below its existing tolerance emits a review warning. The console prints `mixedEV=historical:0.280vs0.366[below_tolerance],legacy:0.390vs0.353[ok]` for the latest evidence. Missing support stays unavailable; a negative single-game scalar cannot override pooled evidence. Full and compact reports retain these fields. No threshold, automatic rollback, extra inference or training objective is introduced.

Runtime files changed: recovery `critic_data.py`, `actor_value.py`, `gae.py`, `review_metrics.py`, `league.py`. Also added `tests/Recovery/test_selfplay85_review.py`, corrected the existing snapshot-exposure test fixture, updated exact source/IL-handoff receipts and [lab instructions](../docs/MLP_V11_LAB.md). Model architecture, rewards, optimizers, losses, LR/GAE/PPG thresholds, dataset/probe, history sampling, v4 identity/search settings and retention remain unchanged.

## Validation and continuation

**143 focused regressions passed in 185.13 s** ([final log](../artifacts/selfplay85-review-20261019/final-regressions.log)). Coverage includes the real g60 no-exposure regression; actual PPO versus accepted/rejected PPG usage; preservation of saved snapshot weight across restart; exact replay of all g56–85 GAE decisions and pooled evidence with a mid-window save/load; effective review ages; sparse v4/unsupported evidence; persistent warnings in compact history; exact reviewed continuation; PPO probability/KL/LR controls; IL health; PPG rollback/buffers; history sampling and five-profile search-free v4. The original g60 regression was reproduced before the fix; the corrected exposure suite also passed independently.

The [isolated two-iteration compute smoke](../artifacts/selfplay85-review-20261019/compute-smoke-final/report.json) passed in **21.17 s** using authored mixed-table Features-4 rows, the 5,928,616-parameter v11.2 model and two CPU threads. Each iteration completed one PPO step. PPG accepted 2/2 steps in the first, with policy KL 0.00003955; the second correctly restored a held-out-loss rejection (0/2 accepted). Snapshot usage distinguishes retained PPO work from rejected PPG work. Exact resumed next-update weights, both optimizers and gate states match; the prior checkpoint is unchanged and expired cache deletion succeeds. This is not a production-checkpoint reload or a native-game strength test.

The scalar diagnostics plus console benchmark measured median **0.0169 ms**, p95 **0.0601 ms**, over 1,000 calls after 25 warmups on the g85 report. This measures reporting only, not end-to-end throughput. [Validation record](../artifacts/selfplay85-review-20261019/validation.json).

Final implementation: `fd85565852a00328236c96b4ed8b6fc9cd245627cc019bbf8c05d12721a89d52`. Learning compatibility stays `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. All **2,172** preceding exact source approvals are preserved; six explicit transitions add this revision. Intermediate development sources are not approved. The exported run, production checkpoints and completed IL are not modified.

The corrections use existing scalar counts and bounded evidence only, with no extra model forward, collection or retained cache. In the lab logs, g76–85 median iteration time is **391.5 s**, collection 194.7 s, PPO 59.4 s, rehearsal loading 38.4 s and value phase 59.7 s. Rating iterations include an additional campaign, so these medians do not imply every iteration takes that long or sum exactly. Increased PPG work helps explain its higher recent cost.

Continue the existing run and upgraded IL directory; no new config or training restart is needed:

```bash
python scripts/step5.py --config configs/step5/joint7-v11-2-grouped-lr-critic.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source \
  --iterations 20
```

This adds 20 iterations from the latest committed generation, reaching 105 if the run is still at 85. `--expected-generation 85` is optional if that exact entry is intended. Do not use `--fresh`. Preserve the existing EV window, GAE streak, both optimizers, RNG, IL validation cache, PPG cache, population and lineage. Source review is exact and unrelated config edits still fail closed.

## Potential improvements, not applied

- **Investigate insolvency by era and cash situation before changing the reward.** Its recent increase deserves attention, but the current cap was not binding in g85. A separately controlled event-cost adjustment would change more trajectories than increasing the cap; no such change or causal claim is made here.
- **Prioritize history critic calibration.** Review the new pooled-EV warning against future rating slots and the actual lambda-0.98 window. A bounded offline comparison of learner-only versus snapshot-weighted value updates or a smaller value-update chunk is more informative than relaxing EV gates. Keep whole-game exclusions and report learner-only loss. No automatic rollback or lambda acceleration is added.
- **If faster LR adaptation is desired, explicitly review the inherited recovery budget.** Rebaselining a budget tightened by the removed sale check would be a deliberate experimental controller change. The current configured recovery is slow but functioning; do not simply bypass its latch.
- **Keep the 0.002 PPG policy cap.** Recent rejection is held-out loss, not excessive policy KL. Smaller acceptance chunks could preserve useful sub-epoch work, but add validation passes and may overfit the small holdout; test on a bounded saved batch first. Do not raise the cap on this evidence.
- **Rehearsal loading is the clearest remaining fixed runtime cost.** It already reuses 16 workers but still costs about 38 s/iteration for 32 shard draws. A bounded encoded-row cache or cheaper selective shard decoding deserves a separate benchmark with identical sample IDs/probabilities and strict disk limits. That optimization is not needed for the reporting fixes.
