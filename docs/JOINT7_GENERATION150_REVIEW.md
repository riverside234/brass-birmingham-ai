# Joint-7 generation 150: review and continuation

Subsequent configuration update: [teacher phase support](TEACHER_PHASE_SUPPORT.md)
sets early Rail to 16 seconds, late Rail to 24 seconds, five games per collection and a disabled cumulative
wall-fraction restriction. That report contains the current continuation command;
the historical measurements below retain their original source/settings.

Reviewed 2026-10-13. **Continuation uses the current new code and runtime/search settings.** The user's later clarification supersedes the pasted request to preserve fixed simulation budgets. The control preserves learning settings; the experimental arm reduces distillation steps only. Neither arm has been trained on the lab by this review.

The run recovered against its fixed IL opponent, with a promising generation-140 monitor, but **generation 150 has no held-out strength result and the accepted champion remains generation 0**. Continue the latest learner and compare frozen panels before selecting a new learning default.

## Evidence and identities

Primary evidence: [export manifest](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/export-manifest.json), [state](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/state.json), [recorded configuration](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/recorded-config.json), [retained history](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/retention-history.json), and full reports for [iteration 148](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/iteration-0148/report.json) and [149](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/iteration-0149/report.json). Earlier evidence: [selfplay-joint](../results/selfplay-joint/), [generation-60 review](JOINT7_ITERATION60_REVIEW.md), and [IL comparisons](../results/joint-v7/).

Reproduce aggregation with `python scripts/review_joint7_150.py`. Outputs: [metrics and input hashes](joint7-150-evidence/metrics.json), [all windows and rolling tables](joint7-150-evidence/tables.md), [source identities](joint7-150-evidence/source-versions.json), and [verification receipt](joint7-150-evidence/verification.json).

The manifest lists 30 files, no warnings, and 20 missing paths: five old champion and five old monitor report/spec pairs. All 30 **LF-normalized** hashes match; raw local hashes differ because this Git checkout uses CRLF. Both hashes are recorded and inputs remain intact. This verifies normalized text identity, not absent checkpoint bytes. Seven completed IL ancestors have exported state/report pairs; their checkpoint/dataset files still require lab preflight.

Combining the exports recovers learner outcomes for all iterations 0–149, including iteration 58 missing from the earlier review's last window. New retained summaries cover 24–147; full reports survive for 59, 148 and 149. All 90 new iterations have the new PPO/reward/teaching diagnostics. Complete per-collection calibration receipts survive only in the latest two full reports. Older missing telemetry is not backfilled or counted as zero. Monitor-80/100 per-game evidence is unavailable.

| Identity | Record |
|---|---|
| Completion | `state.iteration=150`; completed reports 0–149; next report 150 produces generation 151 |
| Ordinary games | 48,000 total; 19,200 before continuation; 28,800 additional = 90 × 320 |
| Latest checkpoint | `artifacts/step5/v4-selfplay-joint/generation-0150.pt` |
| Checkpoint SHA-256 | `391f22af923e21e25f8b532ae68ee5027f1169400780f15a318b47dd20e4b482` |
| Latest model | `0206be1a14ee3f8e891704d2a608a5345c68cac4d8398cb66a70092bb2bfeb0b` |
| Last ordinary collector | Generation 149, model `47b4629da0b2c8d2fea4374ab5d495a946668d0cfbdf1b2005a0517e7e2f61b8` |
| Accepted incumbent | Generation 0, IL model `ea1d9e4da5820a9bfa60647570d498ecf705dacceeb58221c87a77bbcbb26a53` |
| IL parent | `artifacts/step5/v4-batch7-joint/epoch-0006.pt`; SHA `29e7577bc374321f54d99b663800c4ccf762659ef0c02cf08a2eb8c1817317cf` |
| Population/history | 64 live entries (16 recent/48 anchors), 151 history entries |
| Counters | Next seed 18,571,502; context 48,000; next teaching reservation 48,160 |
| Cumulative wall accounting | Ordinary 102,949.7726 s; teaching 64,838.0745 s |

The first 61 history receipts exactly equal the generation-60 history. Every new incoming model equals the preceding outgoing model; all 90 outgoing IDs match committed history receipts. This establishes recorded continuity. Actual tensors/Adam/RNG are checked by fixtures here and must be verified on the lab with the supplied preflight.

## What ran versus current code

| Layer | Identity and behavior |
|---|---|
| Recorded additional 90 | Implementation `9a60a40aa1e4d8193243681d68de2f4661b82d2cf213b2d86f9e78bd3a6e5541`; reproduced from 77 pinned files at Git `a4cd1a7`. Saved config equals the export snapshot. |
| Checkout before review | Git `72bd43ff01a10a8d917705617b3805f6df41e924`; implementation `f6f8028538c13fdaa5cbcd40ffde5e81fe5d9cb58b649875d8a5cd1ea8487191`. Later CPU/GPU, time-budget and transfer changes already present. |
| Reviewed implementation | `3de301188b2eb777a7929da14a3e8d3362f106e75e54129eef8776864809f8e6`; teaching/retention diagnostics and retention-resolution caching correction. |
| Learning compatibility | Unchanged `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`. |

The recorded additional 90 used shared whole-turn **32/16/16/16 attempted NN simulations** in early/late Canal/Rail and **16 v4 simulations** per opponent turn, four workers sharing the allocation, with a 120-second emergency watchdog. The archived `search.seconds=12` field was not the normal stopping rule. The original 60 used the preceding timed implementation; v4 comparisons across that boundary are not budget-matched.

Current code intentionally adopts **12-second neural whole-turn deadlines and one-second v4 whole-turn deadlines**, with cumulative teaching wall fraction .5. Per the user's clarification, continuation keeps these new settings, four search workers sharing a turn, 64 CPUs with four reserved/60 shared work slots, and runtime selection of one to six allocated GPUs. GPU count does not multiply the search allowance.

Recent code already supplies 24 actor processes × two private hosts, shared inference transport, demand routing, cached/fused encoding, CPU admission/retirement fixes, CPU storage of immutable snapshots for single-GPU inference, and portable path resolution. These changes did not generate the archived 90-iteration timings. This review additionally avoids resolving an absolute retention dependency twice for different local bases; confinement checks and per-pass refresh remain intact.

The LR-after-Adam-restore repair, exact PPO categorical probe, reward distributions and post-teacher sampled KL were **already in the recorded source**. Actual reported LR is `1e-5` throughout the new 90 and both latest teaching collections. They are not claimed as new fixes; real Adam bytes were not exported.

## Ranked findings

| Priority | Finding | Confidence and consequence |
|---|---|---|
| 1 | Strength recovery is matchup-specific; generation 150 is untested | High. Preserve IL champion; run matched endpoint panels. |
| 2 | Teaching adds substantial drift despite about 99% top-choice agreement | High for measurements; medium for excessive-distillation hypothesis. Test four versus eight updates. |
| 3 | Curriculum changes explain much of pooled win and speed increases | High. Compare within matchups and hold runtime settings equal between arms. |
| 4 | Neural-opponent economies remain fragile | High for low/zero VP; low for a specific action defect. Inspect replays before changing incentives. |
| 5 | Insolvency caps are common; optimal coefficients are unidentified | High for counts, uncertain intervention. Preserve reward and specify future separate experiments. |
| 6 | Retention loses evaluation/calibration comparability | High, reproduced. Retain bounded opponent/setup/seat/calibration evidence. |

## Progress and strength

`selfplay` is **one learner versus three frozen IL incumbents**, not four current learners. `historical` uses moving neural references; `legacy` is searched v4. These are incoming-learner outcomes, including official shared wins. Rates weight learner games. SD is across iteration means, not a game-level confidence interval.

| Iterations | Opponents | Games | Win share ± iteration SD | Rank | VP |
|---|---|---:|---:|---:|---:|
| 0–19 | Fixed IL | 960 | 30.83% ± 6.24 pp | 2.349 | 78.19 |
| 20–39 | Fixed IL | 960 | 23.02% ± 9.21 pp | 2.559 | 69.01 |
| 40–59 | Fixed IL | 960 | 19.48% ± 7.16 pp | 2.731 | 59.82 |
| 60–89 | Fixed IL | 4,080 | 28.37% ± 6.03 pp | 2.430 | 67.64 |
| 90–119 | Fixed IL | 6,420 | 36.29% ± 4.55 pp | 2.206 | 73.13 |
| 120–149 | Fixed IL | 6,720 | 34.67% ± 8.23 pp | 2.268 | 71.35 |
| 0–19 | Historical | 1,600 | 27.75% ± 5.34 pp | 2.418 | 77.60 |
| 20–39 | Historical | 1,600 | 19.14% ± 8.44 pp | 2.711 | 60.55 |
| 40–59 | Historical | 1,600 | 13.55% ± 6.14 pp | 2.842 | 37.13 |
| 60–89 | Historical | 2,400 | 19.78% ± 5.81 pp | 2.541 | 32.78 |
| 90–119 | Historical | 2,400 | 21.06% ± 4.50 pp | 2.457 | 30.64 |
| 120–149 | Historical | 2,400 | 20.69% ± 4.22 pp | 2.418 | 27.79 |
| 0–19 | Searched v4 | 3,840 | 4.92% ± 1.51 pp | 3.494 | 80.09 |
| 20–39 | Searched v4 | 3,840 | 5.94% ± 2.59 pp | 3.477 | 77.36 |
| 40–59 | Searched v4 | 3,840 | 4.17% ± 2.51 pp | 3.573 | 70.42 |
| 60–89 | v4 fixed 16 | 3,120 | 3.85% ± 2.73 pp | 3.580 | 81.43 |
| 90–119 | v4 fixed 16 | 780 | 5.64% ± 4.68 pp | 3.485 | 85.56 |
| 120–149 | v4 fixed 16 | 480 | 6.88% ± 5.67 pp | 3.471 | 83.39 |

Fixed-IL rolling ten-iteration win shares from 60–69 through 140–149: **28.50, 24.05, 32.59, 35.00, 36.32, 37.38, 27.42, 33.53, 43.05%**. Recovery includes another dip at 120–129 and a recent rebound. Historical outcomes plateau near 21% while VP falls; changing opponent identities prevent treating this as a fixed-panel strength trajectory.

Recovery boundaries use absolute game counters and sometimes fall inside a batch:

| Iterations | IL / historical / v4 games per 320 |
|---|---:|
| 60–61 | 48 / 80 / 192 |
| 62, crosses 20,000 | 96 / 80 / 144 |
| 63–92 | 144 / 80 / 96 |
| 93, crosses 30,000 | 164 / 80 / 76 |
| 94–149 | 224 / 80 / 16 |

The boost began at zero and expired at 30,000; continuation must not restart it. Baseline curriculum is now 70/25/5. Rehearsal expired at 10,000, absent from iteration 32 onward. Pooled new-window win components rise .1825→.2999→.2978 alongside the removal of difficult v4 exposure. Within-IL recovery is real in the observed games, but learning, increased IL exposure and teacher changes are confounded.

Historical selection is 25% IL, 25% recent, 50% older reservoir, with fallback for empty strata. Iteration 149's 240 historical opponent seats sampled 61 generations, including 58 IL seats; the pool had 64 generations. Diversity exists, but it is not a constant-difficulty benchmark. The unchanged incumbent also supplies all 224 direct-IL tables.

### Held-out evaluations

Champion gates use two independent seats per model, all six seat pairs for eight validation setups: 48 games, symmetric **50%** aggregate candidate share, neutral policy-only play. Acceptance requires point share ≥.55 and setup-bootstrap lower bound >.5. Every gate retained IL:

| Generation | Candidate share | 95% setup interval | Candidate / IL rank | Candidate / IL VP |
|---|---:|---:|---:|---:|
| 20 | 56.25% | 47.92–66.67% | 2.563 / 2.438 | 81.49 / 79.11 |
| 40 | 20.83% | 12.50–27.08% | 2.906 / 2.073 | 48.04 / 65.33 |
| 60 | 45.83% | 33.33–60.42% | 2.510 / 2.458 | 59.98 / 61.22 |
| 80 | 41.67% | 25.00–58.33% | 2.552 / 2.385 | 51.49 / 60.24 |
| 100 | 43.75% | 33.33–54.17% | 2.500 / 2.417 | 55.47 / 55.71 |
| 120 | 33.33% | 20.83–43.75% | 2.813 / 2.177 | 56.95 / 67.41 |
| 140 | 58.33% | 45.83–68.75% | 2.469 / 2.448 | 55.74 / 54.40 |

Gate 120 shows panel-specific regression; 140 recovers without clearing uncertainty. Setups change across gates, so these are not paired comparisons. Gate-140 seeds: 1954801, 1954809, 1954820, 1954833, 1954851, 1954854, 1954863, 1954867. Full specs/results survive for 120/140; older scalar receipts remain in state.

Monitors use one candidate versus three fixed IL opponents, no exploration/search, four seats × 16 validation setups = 64 games; baseline **25%**:

| Panel | Candidate | Share [95% setup interval] | Rank | VP |
|---|---:|---:|---:|---:|
| [Monitor 6](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/evaluations/monitor-0006/report.json) | 120 | 20.31% [12.50, 28.13] | 2.609 | 69.77 |
| [Monitor 7](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/evaluations/monitor-0007/report.json) | 140 | 40.63% [28.13, 53.13] | 2.094 | 78.58 |
| [Broader 1](../results/v4-selfplay-joint-g0150-20261013T173011108713Z/evaluations/broader-0001/report.json), historical | 80 | 28.39% [17.58, 38.80] | 1.969 | 7.94 |
| Broader 1, searched v4 | 80 | 6.25% [0.00, 14.06] | 3.547 | 83.05 |

Monitor-6 seeds range 916005–916102 and monitor-7 seeds 917022–917178; exact noncontiguous lists are retained in specs/audit JSON. They share no setups. Generation 140's lower bound above 25% is credible evidence on that selected IL panel, subject to repeated monitoring and small sample size. It does not establish generation-150 strength or a successful champion promotion.

Broader-1 uses 16 setups 911010–911176. Its historical opponents are exactly generations **78/79**, models `a77ff6bf…`/`b51a1732…`, not IL. Forty of 64 candidate neural games ended with zero VP; eight received shared positive win credit. Rank/wins can look acceptable in a weak shared economy. The v4 lineup uses fixed 16 attempts, incompatible with a clean comparison to upcoming one-second panels. All complete retained evaluations report zero failed games; missing evaluations are not counted as successes.

Iteration-149 IL seat shares: 37.50/37.50/44.64/42.86% (56 games each); historical: 17.50/22.50/11.25/26.25% (20 each); v4: 25/0/0/25% (four each). These samples cannot establish a seat defect. Era action telemetry exists, but exact era scores and rail spending do not. New retention keeps matchup-by-seat summaries.

The IL initializer's union validation loss/top-1 (2.1072/52.05%) and separate 30% win share against batch6 on 40 setups [22.5,37.5%] establish its provenance. No matching post-PPO IL validation survives: forgetting remains plausible, unmeasured. Generation 150 has no independent evaluation; next scheduled gate/monitor/broader boundary is 51,200 games, generation 160.

## PPO and teaching

Recorded AdamW LR `1e-5`, decay `.0001`, two PPO epochs, clip `.2`, sampled-KL threshold `.01`, entropy `.005`, value weight `.5`, gradient clip 5, effective batch 2048/microbatch 512/critic batch 2048, FP32 and nonforced actor decisions remain the control settings.

| Window | Sampled / exact PPO KL | Post-teacher sampled KL | Clip fraction | Entropy / normalized | Utility MSE / baseline EV | Gradient mean |
|---|---:|---:|---:|---:|---:|---:|
| 60–89 | .00582 / .00569 | .01875 | .0596 | 1.457 / .449 | .1814 / .300 | 8.64 |
| 90–119 | .00483 / .00474 | .01534 | .0513 | 1.402 / .429 | .1731 / .353 | 7.78 |
| 120–149 | .00522 / .00519 | .01716 | .0537 | 1.350 / .418 | .1682 / .363 | 6.87 |

Mean value losses: .9433/.9200/.9073. Six of 90 PPO updates stop early (3/0/3); total PPO optimizer steps 1,165. Legal-action means stay near 84–85 while normalized entropy falls. Advantage SD .358/.409/.403 and return SD .429/.509/.506 change with mixture/outcomes. Loss changes do not prove stronger play.

Iteration 149: 28,735 learner decisions, 12,432 nonforced, 16,303 forced (56.7%); 14 PPO steps, 10 clipped gradients; sampled/exact KL .00427/.00419, post-teacher .01503. Win/rank/VP losses .87792/.04025/.03736; utility MSE .15548, MAE .28622, EV .41955. Calibration and EV are in-sample ordinary-trajectory diagnostics.

Total post-teacher KL averages .01708, about 3.2× PPO-only sampled KL; 79/90 exceed .01 afterward, maximum .04720 at iteration 141. PPO's stopping rule does not constrain subsequent distillation. Subtracting total KL values is not an exact teacher-drift decomposition. This review adds **exact categorical KL(post-PPO policy || post-teaching policy)** on at most 2,048 proportional nonforced rows, restoring RNG after selection and tensorizing only selected rows. It performs no optimizer writes and remains diagnostic.

The 90 iterations include 180 teaching collections × four games = 720 teaching games, separate from ordinary game totals. Each phase window was scheduled and reached 2,880 times:

| Phase | Actual searches | Qualified labels | Labels/scheduled | Completed attempts | Fallbacks | Insufficient coverage |
|---|---:|---:|---:|---:|---:|---:|
| Early Canal | 2,880 | 2,877 | 99.90% | 92,160 | 3 | 0 |
| Late Canal | 2,833 | 2,703 | 93.85% | 45,328 | 10 | 120 |
| Early Rail | 2,880 | 2,454 | 85.21% | 46,080 | 75 | 351 |
| Late Rail | 2,722 | 1,845 | 64.06% | 43,552 | 2 | 875 |

All 227,120 allocated attempts completed; recorded interrupted rollouts, deadlines, timeouts and cancellations are zero. Whole-turn counts were exhausted, including before many continuations. Reached windows without actual searches are not watchdog failures. Late-Rail weakness principally reflects insufficient qualified support under archived counts. New timed budgets may change both cost and coverage; measure prospectively.

There are 9,879 labels and 1,432 distillation optimizer steps versus 1,165 PPO steps; their batch sizes/objectives differ, so this is not equal gradient work. Pre-fit top-choice agreement averages 98.87/98.70/99.06%. Teacher KL falls .1617→.1505, .1501→.1406, .1408→.1329 across windows; early-window teacher entropy is about .92. Soft targets can contain useful information despite top-choice agreement. Fit improvements are not gameplay benefits.

Iteration 149 teacher models `e2a3c218…`/`65b4b9c0…` differ from collector/final models; opponent generation sets [0,7,149]/[0,115,149]. They supply 57/51 labels, eight steps each, 114/102 presentations, maximum reuse two, LR `1e-5`. Both invalidate utility readiness after shared-trunk updates and refit on 28,735 fresh ordinary decisions/320 games/45 steps, MSE .13784→.13629 and .14003→.13806. Readiness is an in-sample receipt, not certified search strength.

Current checks reject stale source/model/reward teacher labels; teaching games never enter on-policy PPO. Distillation changes trunk/policy, not outcome/utility heads directly; fresh utility-head fitting restores leaf readiness. Reviewed source transitions preserve weights/Adam but require fresh ordinary calibration before teaching.

**Implemented experiment:** four versus eight maximum distillation steps, keeping LR/epochs/clip/entropy/batches/reward/schedule/opponents/current runtime equal. This targets extra drift more directly than halving `distill_weight`, whose effect under Adam need not be proportional. Evidence does not justify replacing the default yet.

## Reward audit

Unchanged contract `brass-combined-return/1.0.0`, reward identity `e9b6efa8566b60ea451000fb0cd3274cb56285ecdab9750ba7b7221627d8db29`:

`U = official shared win + .1 × pairwise placement score + .001 × final VP − future insolvency cost`.

Placement credits wins over opponents and half credit for exact official ties; VP/income/cash tiebreakers and shared wins are preserved. Insolvency costs .005 per event, capped at .025 per learner trajectory. Events are charged once, not once per liquidation choice. Return-to-go excludes already-incurred costs; remaining cap state is consistent across ordinary PPO, utility calibration, terminal values and search leaves/backups. Focused event/return/search parity tests pass. `utility_scale=1.2` is a search scale, **not a probability or ceiling**. Linear VP stays unbounded. No reward normalization, Pass penalty or artificial VP cap is added.

Pooled population moments over all 28,800 new learner trajectories, including between-iteration variation:

| Weighted component | Mean | SD |
|---|---:|---:|
| Official win | .260093 | .436879 |
| Placement | .047612 | .038763 |
| VP | .062661 | .036824 |
| Negative insolvency cost | −.005723 | .009519 |
| Combined utility | .364643 | .488798 |

| Covariance | Win | Placement | VP | Negative cost |
|---|---:|---:|---:|---:|
| Win | .190864 | .013526 | .006717 | .001161 |
| Placement | .013526 | .001503 | .000704 | .000149 |
| VP | .006717 | .000704 | .001356 | .000299 |
| Negative cost | .001161 | .000149 | .000299 | .000091 |

Win variance dominates. Secondary mean incentives total .11027 before debt, versus mean win .26009. Ten VP contributes .01 utility, equivalent to one win-probability percentage point in expectation; one untied placement step contributes .0333. The objective is not mathematically lexicographic. Positive covariances suggest alignment in observed trajectories but do not identify counterfactual sacrifices of wins for VP.

There are 48,572 insolvency events; 9,830/28,800 trajectories affected (34.13%) and 4,924 capped (17.10%). Cap frequency rises 15.88→17.06→18.35% across new windows; mean cost .00538→.00568→.00611. Iteration 149 records 629 events, 110 affected trajectories and 60 capped, but 513 liquidation choices: choices and events differ. Capping removes further marginal penalty once allowance is spent. Frequent caps warrant testing; they do not prove stronger debt penalties improve wins.

Concrete **future** one-factor comparisons, after the distillation experiment:

| Arm | Win / placement / VP | Debt event / cap | Question |
|---|---|---|---|
| Current | 1 / .1 / .001 | .005 / .025 | Control |
| VP-half | 1 / .1 / .0005 | .005 / .025 | Does weaker score incentive improve official wins? |
| Debt-double | 1 / .1 / .001 | .010 / .050 | Does stronger cost reduce collapses without harming wins? |

These coefficient changes are **not enabled**. They require a new explicit target/reward identity, signed migration, stale-teacher invalidation, utility-readiness invalidation, fresh on-policy collection and recalibration before teaching. Editing coefficients under an old checkpoint contract is not an acceptable experiment. Aggregate logs cannot identify optimal coefficients.

## Gameplay and throughput

Iteration 149: 1,205 Passes/9,920 primary action choices (12.15%); 709 had a legal non-Pass alternative, which need not be beneficial. Canal Passes are 202/4,800 (4.21%), Rail 1,003/5,120 (19.59%). Loans 1,828; builds 4,564; developments 975; networks 1,973 (1,914 Rail); scouts 100. These identify replay questions, not action penalties.

Builds: Coal 1,644, Brewery 1,576, Iron 758, Pottery 245, Manufacturer 216, Cotton 125. Coal/Brewery account for 70.55%. Last observed unflipped tiles include Coal 817/Brewery 706/Manufacturer 101/Pottery 137/Cotton 80/Iron 48. These are observation proxies, not exact end-of-era scoring. Dependence on opponents consuming supply is a plausible explanation for poor neural-table economies, but not a proven cause. Build-level counts are in the audit JSON; optimal development/tile choices require legal choices and economic context.

There are 268 sale actions: 261 starts, seven continuations; 423 decisions with feasible sales. Grouped sales prevent translating action frequency into industries sold. Beer usage is 849. Merchant bonus choices: money 18, VP 57, income 34, development 29, with 16 bonus developments and 13 skips. Exact rail cash spend, era score decomposition, complete sale groups and counterfactual alternatives require replays absent here. Capture complete representative wins/losses, zero-VP neural games, capped-insolvency games and late-Rail Passes for further gameplay analysis.

| New window | Collection wall s | Replay final join s | PPO s | Teaching s | Evaluation s, averaged | Iteration wall s |
|---|---:|---:|---:|---:|---:|---:|
| 60–89 | 978.85 | 3.85 | 3.77 | 448.61 | 32.57 | 1,486.24 |
| 90–119 | 394.18 | 4.77 | 3.99 | 466.20 | 9.99 | 897.61 |
| 120–149 | 271.82 | 5.03 | 3.93 | 451.67 | 5.09 | 755.90 |

Reduced v4 exposure drives much of the collection decline. These windows share the archived implementation and do not benchmark current performance improvements. Replay loading already overlapped collection. Latest iteration: collection 275.82 s, final replay join 5.32 s, PPO 4.28 s, teaching 488.21 s, checkpoint .018 s, cleanup 17.74 s; total 792.02 s. Scheduled evaluations distort simple averages.

Its ordinary inference delta is 47,856 positions/36,602 batches, mean batch 1.31, padding 19.9%; forward 48.61 s, summed request wait 131.43 s. Actor-summed encoding/validation 119.19/517.63 s; native CPU 1,695.09 s; IPC read-wait 1,654.73 s; CPU admission wait 7,428.15 s. They overlap across actors and phases: **do not add these as wall time**. Sixteen replay readers used 258.56 task-CPU seconds; 65/67 shards prepared before collection ended. CUDA peak allocated/reserved memory is approximately 1.45/2.83 GB; occupancy is not compute utilization.

The reported machine is Linux with 64-CPU affinity, Torch 2.8/cu128 and RTX PRO 6000 Blackwell. Allocated GPU count/utilization are not established. Benchmark current code separately on identical hardware/allocation/mixture/time limits and record attempted/completed work. Keep budgets shared across workers/GPUs; do not enlarge the model merely to consume VRAM.

## Implemented settings and tools

| Setting | Recorded run | New control | Experiment |
|---|---|---|---|
| Source | `9a60a40a…` | Reviewed current `3de30118…` | Same |
| Neural teacher | Fixed 32/16/16/16 | Current 12 s shared turn | Same |
| v4 opponent | Fixed 16 | Current 1 s shared turn | Same |
| Teaching wall fraction | .9 | Current .5, saved counters retained | Same |
| Actors | 48 × 1 | Current 24 × 2/shared transport/demand routing | Same |
| CPU/search workers | 64, reserve 4/four per turn | Same | Same |
| GPUs | Export does not establish count | Allocated visible 1–6, explicit `auto` | Same allocation |
| PPO/LR/reward | Recorded above | Unchanged | Unchanged |
| Distillation maximum | Eight steps/collection | Eight | **Four** |
| Teacher schedule/weight | Four games/160, at most two collections, .25 | Same | Same |
| Monitor/broader setups | 16 | Keep joint-7's 16 instead of generic lab's one | Same |
| Rehearsal/recovery | Expired at 10k/30k | Preserve progress and IL manifest | Same |
| Diagnostics | Total teaching drift, retention v3 | Separate exact teaching drift, retention v4 | Same |

Presets: [control](../configs/step5/joint7-g150-control.json), [distill4](../configs/step5/joint7-g150-distill4.json). Both adopt current lab execution while keeping run-specific IL lineage and 16-setup screens. Generic defaults must not erase those contracts.

Changes: [probe](../src/brass_ai/recovery/review_metrics.py), [integration](../src/brass_ai/recovery/league.py), [retention](../src/brass_ai/recovery/history.py), [champion spec](../src/brass_ai/recovery/champion.py), [path-resolution cache](../src/brass_ai/recovery/retention.py). New summaries keep teacher model/rollout provenance, per-collection LR/calibration, matchup-by-seat summaries and up to 1,024 outcomes per evaluation panel with truncation flags. Old v2/v3 summaries keep their version/missingness. New champion reports embed their spec; original exports are untouched.

[Preflight](../scripts/joint7_preflight.py) accepts `--evidence` for an exported generation, retaining its old generation-60 default. It checks checkpoint/counters/population/history, live opponent hashes, complete IL lineage, compatibility, reviewed source/config changes and restored Adam configuration before optional archival. Arbitrary setting changes still fail. **New code/settings are adopted through a reviewed continuation, without reinitializing or loading best.pt.**

[Panel comparator](../scripts/compare-training-panels.py) verifies source/configuration/opponents/setups/seats/budgets, rejects failures and missing/duplicate outcomes, and computes paired setup-bootstrap official win/rank/VP differences. It never promotes. [The exporter](../scripts/export-training-reports.py) includes new full report fields and recognizes `comparison.json`; `--extra` includes independent panels outside the run. Automatic selection includes state, retention, latest two completed iterations, available evaluations and completed IL lineage.

## Verification and limits

Focused suites cover generation-150/60 and earlier reviewed resumes, restored Adam/RNG/counters, timed contracts, teacher readiness/stale-label rejection, combined reward/insolvency remaining returns, shared work, champion selection, bounded diagnostics, single-GPU behavior, export and paired bootstrap. The verification receipt lists exact commands, corrected initial failures and final results.

The native CPU smoke completed two full games, streaming replay ingestion, two PPO updates and generation-1 checkpoint writing. Search was disabled in this small smoke; teacher correctness is covered by focused tests. This is small-model correctness evidence, not a lab-generation-150 restore, GPU benchmark or strength qualification. No long training, lab checkpoint mutation, installed-game AI change or promotion occurred.

## Safe lab continuation

Sync these reviewed files to the lab checkout. Use the existing conda environment and stopped generation-150 run with complete checkpoint/IL files. Keep scheduler-provided `CUDA_VISIBLE_DEVICES`; `--inference-devices auto` explicitly opts into allocated visible GPUs. Stop if a command fails. The backup path must be new.

```bash
set -euo pipefail
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/joint7_preflight.py \
  --root /data/not_backed_up/yxu209/ai-training/ \
  --run artifacts/step5/v4-selfplay-joint \
  --evidence results/v4-selfplay-joint-g0150-20261013T173011108713Z \
  --config configs/step5/joint7-g150-control.json \
  --backup artifacts/joint7-g150-backup

python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training/ \
  --config configs/step5/joint7-g150-control.json --device cuda preflight --record-parity

python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training/ pin \
  --run artifacts/step5/v4-selfplay-joint \
  --paths artifacts/step5/v4-selfplay-joint/generation-0150.pt \
    artifacts/step5/v4-selfplay-joint/initial.pt \
    artifacts/step5/v4-selfplay-joint/population/weights-25c9ba80fba0e2a8d3981b5926b65827ca336ee691250e238e362b71976b295d.pt \
    artifacts/step5/v4-selfplay-joint/population/weights-21e9a68dbb74f96f6c747460e9cacff2abaaaf4575b0a8f0ef1646d71a4f7db7.pt
```

One iteration of the original run, 320 ordinary games:

```bash
python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training/ \
  --config configs/step5/joint7-g150-control.json --device cuda \
  --inference-devices auto train --run artifacts/step5/v4-selfplay-joint \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source --iterations 1

python scripts/export-training-reports.py \
  --root /data/not_backed_up/yxu209/ai-training/ \
  --run artifacts/step5/v4-selfplay-joint
```

`--imitation-run` verifies completed IL provenance; an existing state resumes PPO without repeating IL/resetting Adam. Omit `--initial`/`--from-run`. Nonempty pending iteration 150 or uncommitted generation 151 from another source is rejected: recover it under its original source first. Saved search accounting/recovery/seed/context counters persist. With changed GPU topology, current code restores matching logical-device CUDA RNG streams and reports topology differences; it cannot promise exact stochastic replay across different GPU allocations.

## Controlled comparison from identical generation 150

Use the archive above, including if the original run has since completed the optional iteration. Keep relative run identity `artifacts/step5/v4-selfplay-joint` in independent project roots. These commands copy project dependencies and restore the archived run under that same relative path; they never hard-link mutable files. They may copy substantial disk data. Both destination roots must be new disjoint siblings. Stop project writers during copying. The transfer exporter correctly rejects an untransitioned old-source run under new code; these are explicit reviewed experiment copies, with no edits to checkpoint/state signatures.

Run in the same activated conda shell:

```bash
cd /data/not_backed_up/yxu209/ai-training/
export JOINT7_BASE=/data/not_backed_up/yxu209/ai-training
export JOINT7_CONTROL=/data/not_backed_up/yxu209/joint7-g150-control
export JOINT7_CANDIDATE=/data/not_backed_up/yxu209/joint7-g150-distill4

for ARM_ROOT in "$JOINT7_CONTROL" "$JOINT7_CANDIDATE"; do
  test ! -e "$ARM_ROOT"
  mkdir "$ARM_ROOT"
  rsync -a --exclude='.git' --exclude='__pycache__' \
    --exclude='.step5-relocation.json' \
    --exclude='/artifacts/step5/v4-selfplay-joint/' \
    --exclude='/artifacts/joint7-g150-backup/' \
    "$JOINT7_BASE/" "$ARM_ROOT/"
  mkdir -p "$ARM_ROOT/artifacts/step5/v4-selfplay-joint"
  cp -a --reflink=auto "$JOINT7_BASE/artifacts/joint7-g150-backup/." \
    "$ARM_ROOT/artifacts/step5/v4-selfplay-joint/"
done

# Map historical absolute paths to each independent copy without rewriting them.
PYTHONPATH="$JOINT7_BASE/src" python - <<'PY'
import json, os
from pathlib import Path
from brass_ai.bank import sha
from brass_ai.learning.paths import make_relocation, source_roots, RELOCATION_FILE
base = Path(os.environ['JOINT7_BASE']).resolve()
archive_id = sha(base/'artifacts/joint7-g150-backup/state.json')
for name in ('JOINT7_CONTROL', 'JOINT7_CANDIDATE'):
    root = Path(os.environ[name]).resolve()
    receipt = make_relocation(root, source_roots(base), archive_id)
    with (root/RELOCATION_FILE).open('x', encoding='utf-8') as out:
        json.dump(receipt, out, indent=2)
PY

for ARM in control distill4; do
  if [ "$ARM" = control ]; then ARM_ROOT="$JOINT7_CONTROL"; else ARM_ROOT="$JOINT7_CANDIDATE"; fi
  cd "$ARM_ROOT"
  python scripts/joint7_preflight.py --root "$ARM_ROOT" \
    --evidence results/v4-selfplay-joint-g0150-20261013T173011108713Z \
    --config "configs/step5/joint7-g150-$ARM.json"
  python scripts/step5.py --root "$ARM_ROOT" \
    --config "configs/step5/joint7-g150-$ARM.json" --device cuda preflight --record-parity
  python scripts/step5.py --root "$ARM_ROOT" pin --run artifacts/step5/v4-selfplay-joint \
    --paths artifacts/step5/v4-selfplay-joint/generation-0150.pt \
      artifacts/step5/v4-selfplay-joint/initial.pt \
      artifacts/step5/v4-selfplay-joint/population/weights-25c9ba80fba0e2a8d3981b5926b65827ca336ee691250e238e362b71976b295d.pt \
      artifacts/step5/v4-selfplay-joint/population/weights-21e9a68dbb74f96f6c747460e9cacff2abaaaf4575b0a8f0ef1646d71a4f7db7.pt
  python scripts/step5.py --root "$ARM_ROOT" \
    --config "configs/step5/joint7-g150-$ARM.json" --device cuda --inference-devices auto \
    train --run artifacts/step5/v4-selfplay-joint \
    --imitation-run artifacts/step5/v4-batch7-joint --resume-reviewed-source --iterations 4
done
```

Both arms stop at generation **154**, 49,280 ordinary games, before the 51,200 selection boundary. No curriculum/rehearsal boundary is crossed. The sole learning-setting difference is maximum distillation steps. Shared current runtime/code changes are unavoidable relative to the archived generator and controlled between arms. Timed search produces variable completed work even with identical starting RNG/counters: record attempts, qualified labels, deferrals and actual updates rather than assuming equal work. Run arms sequentially on the same CPU/GPU allocation. This four-iteration screen is diagnostic, not a full strength campaign.

Use one common evaluation config, frozen IL plus generations 120/140, identical held-out setups/seats and hardware. `recent-neural` is IL-only; `mixed-neural` uses fixed IL/120/140; `mixed-v3` uses the fixed admitted bank with current one-second limits. Baseline/endpoint are policy-only, avoiding reliance on old checkpoint search readiness:

```bash
for ARM_ROOT in "$JOINT7_CONTROL" "$JOINT7_CANDIDATE"; do
  cd "$ARM_ROOT"
  for GEN in 0150 0154; do
    python scripts/step5.py --root "$ARM_ROOT" \
      --config configs/step5/joint7-g150-control.json --device cuda --inference-devices auto \
      evaluate --candidate "artifacts/step5/v4-selfplay-joint/generation-$GEN.pt" \
      --references artifacts/step5/v4-selfplay-joint/initial.pt \
        artifacts/step5/v4-selfplay-joint/population/weights-25c9ba80fba0e2a8d3981b5926b65827ca336ee691250e238e362b71976b295d.pt \
        artifacts/step5/v4-selfplay-joint/population/weights-21e9a68dbb74f96f6c747460e9cacff2abaaaf4575b0a8f0ef1646d71a4f7db7.pt \
      --lineups recent-neural mixed-neural mixed-v3 --setups 16 --seed 29761500 \
      --partition validation --no-search \
      --output "artifacts/joint7-g150-panel/g$GEN"
  done
done

cd "$JOINT7_BASE"
python scripts/compare-training-panels.py \
  --control "$JOINT7_CONTROL/artifacts/joint7-g150-panel/g0154/report.json" \
  --candidate "$JOINT7_CANDIDATE/artifacts/joint7-g150-panel/g0154/report.json" \
  --output artifacts/joint7-g150-comparison/comparison.json

for ARM_ROOT in "$JOINT7_CONTROL" "$JOINT7_CANDIDATE"; do
  cd "$ARM_ROOT"
  python scripts/compare-training-panels.py \
    --control artifacts/joint7-g150-panel/g0150/report.json \
    --candidate artifacts/joint7-g150-panel/g0154/report.json \
    --output artifacts/joint7-g150-panel/endpoint-vs-start/comparison.json
  python scripts/export-training-reports.py --root "$ARM_ROOT" \
    --run artifacts/step5/v4-selfplay-joint --extra artifacts/joint7-g150-panel
done

cd "$JOINT7_BASE"
python scripts/export-training-reports.py --root "$JOINT7_BASE" \
  --run artifacts/step5/v4-selfplay-joint --extra artifacts/joint7-g150-comparison
```

Each model screen is 192 games, uncertainty clustered over 16 setups. Independent generation-150 runs use identical weights but can expose timed-search variation. Compare each endpoint with its start and with the other endpoint. Drift, support, utility fit and collapse frequency are supporting diagnostics, not replacements for official wins.

Before viewing results, designate **mixed-neural official win-share delta** as primary. A promising screen has positive effect, no infrastructure failures and no concerning IL/v4 regression; wide intervals are inconclusive. Confirm a selected candidate against the control on **64 new test setups**, seed `30761500`, `--partition test`, fresh output directories, identical three lineups/budgets. Predeclare primary point improvement ≥2 percentage points with paired 95% lower bound >0; require IL/v4 lower delta bounds above −3 percentage points and inspect rank/VP, zero-VP games and seats for regressions. If bounds are too wide, report insufficient evidence. These are operational margins, not universally optimal thresholds; repeated screening requires fresh confirmation data.

Repeat success from another matched training replicate before generalizing beyond these endpoints. If deployment uses search, also compare search-on freshly calibrated endpoints under the same new clocks: policy-only gains do not establish searched strength. Do not tune on test results or pool these with archived fixed-count-v4 panels. Champion selection remains separate; this comparator cannot promote or deploy.

**Implemented and tested:** current-setting continuation receipts/configs, preserved-state checks, separate teaching drift, retained comparison evidence, matched-panel analysis and native smoke. **Promising but unproven:** reduced distillation and the generation-140 IL recovery. **Demonstrated stronger play:** a positive result on generation 140's specific IL monitor; no established general strength gain for generation 150 or these new changes.
