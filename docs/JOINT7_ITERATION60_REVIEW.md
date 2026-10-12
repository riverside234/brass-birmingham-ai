# Joint-7: first 60 self-play iterations

Review date: 11 October 2026 (America/Chicago). Lab project root: `/data/not_backed_up/yxu209/ai-training/`.

**The run has not established an improvement over its completed IL initializer.** Recorded outcomes rose early, regressed substantially around generations 30–50, and recovered partly near generation 60. The accepted best remains generation 0. Continue the learner from generation 60, preserve the IL incumbent, and measure fixed matchups before promoting anything.

This review used the local export at Git `179b839`. It did not access the lab GPU machine or the actual generation-60 checkpoint. Source/configuration changes and CPU verification are local; the commands below verify actual lab files before continuation. No installed game AI, supplied results, frozen opponent bank, or original checkpoint was changed.

## Evidence and limits

The request's `results/joint-v7` directory contains **IL comparisons**. The self-play evidence is in `results/selfplay-joint`.

| Source | What it establishes | What it cannot establish |
|---|---|---|
| [Self-play state](../results/selfplay-joint/state.json) | Committed generation 60, exact configuration/source signature, counters, 61 history/population entries, IL origin, three champion receipts, two recent training summaries | Actual checkpoint tensor/Adam contents; bytes are absent here |
| [Retention history](../results/selfplay-joint/retention-history.json) | Compact iteration 0–57 metrics, monitor 1, champion 1 | Discarded per-game trajectories, most optimizer/teacher distributions and economics |
| [Iteration 59](../results/selfplay-joint/iter59.json) | Full latest collection/PPO/search/economy/host report, producing generation 60 | Individual ordinary-game terminal scores or replay decisions: `execution` contains receipts/timings, not those replays |
| [Champion report](../results/selfplay-joint/champion.json) | Generation-60 gate, 48 outcomes with candidate seat pairs across eight setups | Individual candidate-seat outcomes; each row pools its two candidate seats |
| [Monitor report](../results/selfplay-joint/monitor.json) | Generation-60 policy-only evaluation, four candidate seats on one setup | A meaningful setup confidence interval or general seat advantage |
| [IL validation](../results/joint-v7/validation-comparison.json), [IL panel comparison](../results/joint-v7/comparison.json), [IL vs batch6](../results/joint-v7/summary.json), [raw head-to-head report](../results/joint-v7/report.json) | Initializer's validation and separate pre-RL gameplay comparisons | Post-PPO forgetting or matched generation-60 results against batch6 |

All 60 iterations have PPO/search summaries: 0–57 from retention and 58–59 from `state.recent_training`. **Iteration 58 has no composition outcomes or timings in this export.** Outcome/timing windows labelled 40–59 therefore contain 19 measured iterations. Full reports for 0–58, replay shards, complete intermediate evaluation reports/specifications, CUDA profiler traces, and lab model files are missing. Monitor 2's state receipt records its existence but does not preserve its results. No broader panel was yet due: its first threshold is 25,600 ordinary games.

[Reproducer](../scripts/review_joint7.py), [generated tables](joint7-60-evidence/tables.md), and [machine-readable measurements](joint7-60-evidence/metrics.json) retain source hashes, per-field coverage, all available iteration summaries, descriptive variability, and ten-iteration windows. SD in the tables is **between iteration means**, not per-game variability or a confidence interval. Teacher fields have unequal retention coverage; `phase_field_iterations` records it. In particular, early fallback/interruption counts must not be filled with zero.

The exported monitor/champion files use Windows line endings. Their LF-normalized SHA-256 hashes exactly match the generation-60 receipts in state (`5679651b…` and `5de4f326…`). The evidence JSON records both raw export and LF hashes.

## What actually ran

The recorded implementation is `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`. Recomputing the pinned source fingerprint from Git matches the learning/search trees at `e55dbb1` and `7f1d04a`; the actual lab invocation's full Git commit was not recorded. The checkout before this review used `fff450449f8fbceaa8c57cffcf826da006d53d44351c2fce4d985fc973d0df7c`, which already contains later teacher-readiness, fixed-work, and throughput repairs. Those repairs did **not** run in these 60 iterations. This review's source fingerprint is `9a60a40aa1e4d8193243681d68de2f4661b82d2cf213b2d86f9e78bd3a6e5541`.

Recorded contracts: `step5-training/2.2.0`, `nonforced-actor-ppo/2.1.0`, `four-seat-learning/1.0.0`, `brass-four-seat-mlp-resnet/1.0.0`, rules `steam-brass-patch-1.1.0`, combined reward `brass-combined-return/1.0.0`, search `masked-ismcts/1.2.0`. The learning compatibility digest remains `c349e816f2e301ea2a3b438902059cffe868f48e131f7c120691d37cd1c409d2`.

| Setting | Recorded run |
|---|---|
| Initializer | `artifacts/step5/v4-batch7-joint/epoch-0006.pt`, SHA `29e7577bc374321f54d99b663800c4ccf762659ef0c02cf08a2eb8c1817317cf` |
| IL model / pooled manifest | Model `ea1d9e4d…`; manifest `artifacts/il-replay/v4-batch7-joint/manifest.json`, SHA `7da77202…`; completed IL handoff, new PPO optimizer |
| Model size | 1,840,109 IL parameters; 1,850,129 after deterministic utility-head extension |
| Ordinary collection | 320 games/iteration; 48 processes × one host; 24 queued heavy tables; 16 replay readers; exploration 0.05 |
| CPU / device | 64 CPUs, four reserved, one shared 60-slot admission gate; four workers/search; Torch 2.8.0+cu128, CUDA 12.8, RTX PRO 6000 Blackwell Server Edition, learner device `cuda`; GPU count not recorded |
| PPO | LR `1e-5`, AdamW decay `1e-4`; two epochs, effective batch 2048, microbatch 512, critic batch 2048, 131072 candidate cells, FP32 |
| Objective/optimization | Clip 0.2, sampled KL threshold 0.01, entropy weight 0.005, value weight 0.5, gradient clip 5; normalize current-batch nonforced advantages |
| Rehearsal | Audited joint IL training manifest; weight `0.1 * max(0, 1 - games_before/10000)`, absent from iteration 32 onward |
| Actual table mixture | 48 learner-vs-incumbent, 80 historical, 192 searched-v4 games each iteration: 15% / 25% / 60% |
| Historical sampling | 25% IL anchor, 25% recent, 50% older reservoir per opponent selection, with fallback for unavailable strata; 16 recent/48 anchor capacity |
| Frozen v4 | `bank-3dc65253a610191d66a5`; five training profiles, held-out Pottery Flex identity; internal `v3-*` names are compatibility aliases |
| Search teaching | Warm-up 640 ordinary games; every 160 thereafter; four games/collection, at most two collections/iteration; four reserved phase turns per NN seat |
| Distillation | Weight 0.25; at most eight updates/collection, 16 rows/batch, two presentations/row; minimum 16 labels, two setups, two labels/phase; policy-only loss |
| Recorded search limits | NN teacher **12-second whole-turn clock**, ceiling 100000 simulations; v4 **one-second** clock with four workers. These records have no fixed-work contracts |
| Selection | Champion every 6400 games: eight setups × six seat pairs, policy-only; candidate's two-seat win sum ≥0.55 and setup-bootstrap lower bound >0.5 |
| Other evaluation | Policy-only monitor every 6400 games, one setup × four candidate seats; broader every 25600, one setup/lineup |

The 30,000-game recovery boost started at game 0. It overrides the nominal curriculum and the `search_free_until_games: 10000` default. All retained collections use 60% **searched** v4 tables. At game 20,000 it changes to 45% incumbent / 25% historical / 30% v4, then at 30,000 to 70% / 25% / 5%. Do not attribute the observed generation-40 regression to a recorded opponent-mixture change: that change had not occurred.

The committed state is 19,200 ordinary games, next seed 18,534,700, context counter 19,200, next teacher threshold 19,360. There were 117 teacher collections = 468 teacher games, separate from the ordinary counter, plus 144 champion and 12 monitor games recorded by the evaluation schedule.

| Role | Actual identity |
|---|---|
| Latest learner / resume checkpoint | `artifacts/step5/v4-selfplay-joint/generation-0060.pt`, SHA `fa1f90736ed0eb3dd92ca0c7bef3ebd9ced396f93a0a693a5f6b0034a9d1f963`, model `1ddc910f…` |
| Accepted best | Generation 0, model `ea1d9e4d…`, weights-only population copy SHA `2bd4bccf…`; `best.pt` is a convenience copy, not the Adam resume checkpoint |
| Ordinary iteration-59 learner | Frozen incoming generation 59, model `56f516e3…`; its results are not post-update generation-60 results |
| “selfplay” opponents | Three copies of the accepted generation-0 incumbent; only one current learner seat supplies PPO rows |
| Historical opponents | 61 stored population entries at commit 60; latest collection samples 58 distinct generations, including 61/240 opponent seats from generation 0 |
| NN teachers | Fresh post-PPO learner, then updated learner after each distillation; latest rollout reference generations `[0,34,59]` and `[0,35,59]`, with the current learner also included |

The initial utility extension changes model identity (`06b0dc03…` for the first incoming collector) while preserving compatible IL tensors. It is not evidence of a different IL parent. The original IL source optimizer was intentionally not imported; the **PPO** Adam state must now be retained.

## Ranked findings

| Priority | Finding | Confidence / response |
|---|---|---|
| 1 | No accepted strength gain; marked middle-run regression and only partial recovery | High for recorded matchups. Keep the IL incumbent; continue generation 60 as learner and improve evaluation |
| 2 | Run source allowed a pre-distillation utility-fit receipt to remain ready after the shared trunk changed | High code evidence; causal impact on strength unmeasured. Later checkout already invalidates/refits the head after teaching and on reviewed resume; preserve these repairs |
| 3 | Adam restoration silently overwrites a requested LR change | High, reproduced with actual Adam state. Fixed by applying validated LR/decay after restore; no evidence the recorded `1e-5` itself was wrong |
| 4 | Retention drops actor loss, LR, row counts, gradients, and teacher information needed for this audit | High, direct field-name mismatch and whitelist loss. Fixed with bounded summary v3; old v2 missingness preserved |
| 5 | Insolvency rose while mean cost remained small and saturated after five events | High association, medium hypothesis of weak incentive. Keep current coefficients for continuation; compare alternatives under an explicit reward migration later |
| 6 | Teaching mostly agrees with current top choices in the latest qualified set | High for latest sample; unknown across the entire run and unknown decision benefit. Keep quotas; log drift after teaching and evaluate searched decisions separately |
| 7 | Host/search work dominates elapsed time; extra GPU memory/model size is not a demonstrated solution | High for timings, no GPU-utilization measurement. Preserve fixed CPU allowance and flexible GPU routing |

## Learning progress and evaluation

The following uses only `incoming_policy_id` outcomes. It does not pool all four seats. In symmetric four-player self-play, pooling everyone's win shares would mechanically yield 25%; here the named selfplay matchup is **one learner versus three frozen IL policies**, so learner win share is informative for that matchup.

| Iterations | Matchup | Learner games | Win share ± between-iteration SD | Mean rank | Final VP ± SD |
|---|---|---:|---:|---:|---:|
| 0–19 | Fixed IL incumbent | 960 | 30.83% ± 6.24 pp | 2.349 | 78.19 ± 3.49 |
| 20–39 | Fixed IL incumbent | 960 | 23.02% ± 9.21 pp | 2.559 | 69.01 ± 7.67 |
| 40–59, excluding 58 | Fixed IL incumbent | 912 | 18.86% ± 6.81 pp | 2.751 | 59.26 ± 4.73 |
| 0–19 | Historical mix | 1600 | 27.75% ± 5.34 pp | 2.418 | 77.60 ± 2.92 |
| 20–39 | Historical mix | 1600 | 19.14% ± 8.44 pp | 2.711 | 60.55 ± 12.50 |
| 40–59, excluding 58 | Historical mix | 1520 | 13.32% ± 6.21 pp | 2.852 | 37.22 ± 3.83 |
| 0–19 | Searched v4 mix | 3840 | 4.92% ± 1.51 pp | 3.494 | 80.09 ± 3.02 |
| 20–39 | Searched v4 mix | 3840 | 5.94% ± 2.59 pp | 3.477 | 77.36 ± 5.80 |
| 40–59, excluding 58 | Searched v4 mix | 3648 | 4.08% ± 2.55 pp | 3.584 | 70.04 ± 4.09 |

Ten-iteration fixed-incumbent win shares are **28.96%, 32.71%, 29.58%, 16.46%, 16.04%, 21.99%** (last window has nine observations). Latest incoming generation 59 scores 35.42%/2.354/68.67 VP against IL, 19.06%/2.588/39.13 against history, and 8.33%/3.349/75.88 against v4. A good final collection is a rebound, not proof that the last checkpoint is stronger.

Historical-model mixtures change as the population grows. Their trend is not a fixed-opponent strength curve. Incumbent identities remain fixed, but collection uses training setups, 5% exploration, and mixed learner intent (half neutral, half industry focus); neutral held-out evaluation is distinct. Nominal v4 clock/worker settings were stable, but actual completed search work is hardware/content dependent.

| Candidate generation | Gate games / setups | Candidate two-seat win sum | Setup-bootstrap 95% | Candidate / incumbent rank | Candidate / incumbent VP | Accepted |
|---|---:|---:|---|---|---|---|
| 20 | 48 / 8 | 56.25% | 47.92–66.67% | 2.563 / 2.438 | 81.49 / 79.11 | No |
| 40 | 48 / 8 | 20.83% | 12.50–27.08% | 2.906 / 2.073 | 48.04 / 65.33 | No |
| 60 | 48 / 8 | 45.83% | 33.33–60.42% | 2.510 / 2.458 | 59.98 / 61.22 | No |

The symmetric reference in this **two-versus-two population comparison is 50%**, not 25%. Each seat still plays independently. Generation 20 passed the point threshold but failed the confidence condition. Generation 40 is convincing regression on its panel. Generation 60 is inconclusive versus IL. Different gate seeds prevent a paired early/middle/latest comparison, and repeated selection screens are not formal promotion tests.

Monitor 1: 50% win share, rank 2.0, VP 105.25 on one setup. Monitor 3: 50%, rank 2.5, VP 62.5 on a different setup. Monitor 3's seats score `(win, VP)` = `(0,0), (1,89), (1,105), (0,56)`. Four rotations of one setup cannot estimate setup variability. Generation-60 gate pair win sums range 25–75% across six candidate pairs, with only eight games per pair; they cannot identify individual-seat skill. Latest collection seats are balanced at 12/20/48 per seat in incumbent/history/v4 panels. Earlier per-seat results and era-specific VP/scoring outcomes are unavailable.

The IL initializer had union-validation loss **2.1072**, meaningful top-1 **52.05%**, versus batch7's 2.2869/47.71% and batch6's 2.4903/44.79%. Its separate one-versus-three gameplay shares were 32.5% against batch7, 30% against batch6 (95% setup interval 22.5–37.5%), and 5% against searched v4. These support its provenance, not post-RL superiority. There is no matching post-PPO IL-validation measurement; **forgetting is a hypothesis**, especially as rehearsal fades near the regression, not a measured diagnosis.

## Reward audit and coefficient comparison

Implemented utility is

`U_t = W + 0.10 P + 0.001 final_VP − future_capped_insolvency_cost_t`.

`W` preserves official VP, then income, then cash tiebreakers and shared wins. `P` is mean pairwise result against the other three seats, with half credit for exact official ties; it is not blindly `(4 − competition_rank)/3` in tied games. VP is linear and uncapped. Each seat pays 0.005 per insolvency event, at most 0.025/game.

| Iterations | Mean W ± SD | Mean 0.1P ± SD | Mean 0.001VP ± SD | Mean cost ± SD | Mean total utility |
|---|---:|---:|---:|---:|---:|
| 0–19 | .14516 ± .01897 | .03155 ± .00282 | .07918 ± .00250 | .00105 ± .00029 | .25484 |
| 20–39 | .11801 ± .04472 | .02828 ± .00583 | .07191 ± .00749 | .00260 ± .00112 | .21559 |
| 40–59 | .08810 ± .03304 | .02401 ± .00421 | .06051 ± .00308 | .00466 ± .00051 | .16796 |

These are moments of **iteration-average learner trajectories**, not component variances across individual games. Individual VP/placement/cost covariance and reward-induced action advantages cannot be reconstructed. The new telemetry records per-trajectory distributions and covariance for the next run.

The secondary mean bonus is substantial relative to rare learner wins: about .111 early and .0845 recently, versus mean W .145 and .088. Marginally, ten VP equals one percentage point of win probability; one untied placement step equals .0333 utility (3.33 win-probability points or 33.3 VP). A full win's +1 dominates ordinary observed score differences, but the unbounded linear VP term is not mathematically lexicographic with winning. The logs do not show counterfactual action values, so they cannot prove that the agent deliberately sacrifices wins for VP/placement.

Insolvency affected 649/6400 (10.1%), 1285/6400 (20.1%), and 2002/6400 (31.3%) learner trajectories in the three windows; events total 1618, 4523, and 8338. Latest: **421 events, 102 affected games, 104 Canal / 317 Rail events**, total cost 1.51, mean .004719. This cost corresponds to 302 charged-event equivalents, compared with 421 observed events: saturation is material. The fraction of trajectories reaching the cap is missing. Each event costs five VP-equivalents; the entire cap costs 25. Increasing the penalty could help, or could induce excessive loans/development avoidance; neither is established here.

Code checked: [reward accounting](../src/brass_ai/learning/reward.py), [insolvency extraction](../src/brass_ai/recovery/insolvency.py), [PPO targets](../src/brass_ai/recovery/updates.py), [utility model](../src/brass_ai/learning/utility_model.py), [search backup](../src/brass_ai/search/ismcts.py), and [head calibration](../src/brass_ai/recovery/utility.py).

- Events are settled once per seat/round, including automatic no-asset debt. Multiple liquidation choices do not create repeated charges. Final Rail has no income/debt settlement.
- At a debt observation the event has already occurred: replay returns subtract only subsequent charges. Live `CostLedger` and replay remaining budgets agree in focused tests. PPO does not subtract insolvency again after using the combined target/baseline.
- Search leaves predict remaining utility; backups subtract costs between the node and leaf, while terminal utility has no future cost. Cache/information identities include remaining budgets and reward IDs. No already-incurred cost is charged again.
- The official win head is a probability; the utility head is an unbounded regression head. Search divides Q by fixed **1.2** for exploration scale; **1.2 is not an upper bound**. Combined values and first-play estimates are not clipped to [0,1]. Utility uses VP in VP units, not twice-normalized VP. Official heads remain separate.
- The old run's teacher readiness was stale after distillation moved the shared encoder. The later checkout's invalidation and bounded head-only recalibration repair this certification gap. This review retains them. It is an implementation defect with an unmeasured effect on playing strength.

**Continue with unchanged reward coefficients.** A subsequent small coefficient experiment should compare the following with the same LR, frozen evaluations, work quotas, seed schedule, and starting generation-60 state:

| Arm | W | P | VP | Event / cap | Question |
|---|---:|---:|---:|---|---|
| R0 control | 1 | .10 | .001 | .005 / .025 | Recorded utility |
| R1 VP | 1 | .10 | .0005 | .005 / .025 | Does halving VP competition improve official wins? |
| R2 debt | 1 | .10 | .001 | .010 / .050 | Does stronger cost reduce harmful insolvency while preserving the five-event cap structure? |

Placement stays fixed in this first comparison. These arms are **recommendations, not implemented reward migrations or optimal settings**. Before R1/R2, add an explicit reviewed reward migration: new target/reward identity, compatible old weights/Adam/counters preserved, utility-fit invalidated, incompatible teacher data excluded, fresh complete PPO rollouts collected, and head recalibrated before any search teacher. Current guards deliberately reject changing an existing utility model's coefficients by editing configuration alone. No Pass penalty, action-frequency bonus, VP ceiling, running reward normalization, or “cannot win” switch is introduced.

## PPO optimization and teaching

| Iterations | PPO steps mean | Final sampled KL | Clip fraction | Nonforced entropy | KL stops | Utility MSE |
|---|---:|---:|---:|---:|---:|---:|
| 0–19 | 10.40 | .00948 | .081 | 1.629 | 12/20 | .1654 |
| 20–39 | 11.45 | .00777 | .070 | 1.575 | 5/20 | .1570 |
| 40–59 | 12.70 | .00626 | .059 | 1.553 | 2/20 | .1495 |

There are 691 PPO optimizer steps overall. Value loss decreases .988 → .942 → .903 approximately while strength regresses. It is not a strength signal. Utility MSE is accumulated during critic minibatch optimization over four heads, not a held-out final prediction error. Explained variance, return/advantage quantiles, and legal-action-normalized entropy did not survive for most iterations.

Iteration 59 has 28,645 learner decisions, **12,484 nonforced and 16,161 forced** (56.4%). Forced choices contribute critic data but no policy loss or actor entropy. Both epochs see 12,484 actor and 14,336 sampled critic positions; the critic is not a full pass over all rows each epoch. Final losses: actor −.00271, official value .94486 (win .93051, rank .04090, VP .04121), utility MSE .15526 / MAE .29665. Raw actor advantage mean/std .04419/.33239. Final sampled full/probe KL .004386/.004138; last preclip gradient norm 8.39; 12/14 steps clip at five. Effective LR is explicitly **1e-5** in iterations 58/59 and distillation receipts. Earlier effective-LR logs were discarded; an unchanged config is not direct observation of every optimizer group.

`post_update_kl` measures `((ratio − 1) − log ratio)` on recorded chosen actions across all nonforced rows; `probe_kl` uses 2048 proportional era/phase rows. Neither was exact categorical KL. The new categorical probe computes `KL(old behavior || new behavior)` across all legal probabilities on those same existing probe rows, without extra RNG draws. It also reports entropy/log(number of legal choices), raw advantage/remaining-return distributions, baseline explained variance, and all-step gradient norm distributions. Existing clipping and early stopping remain unchanged.

The restored-LR regression is now fixed in [resume settings](../src/brass_ai/recovery/resume.py): `optimizer.load_state_dict` restores parameter-group LR/decay as well as Adam tensors, so constructing AdamW with a new LR was insufficient. Step 5 now applies its **already signature-validated** LR/decay afterward. Tests preserve moments, steps, tensors, and RNG. The optional `joint7-lr-half.json` compares `5e-6` with `1e-5`; it is not the new default. This changes the shared PPO/distillation Adam LR, while head-only calibration retains its explicit temporary 1e-3 and restores the configured LR afterward. Epochs, clip, entropy/value coefficients, and batches remain fixed.

Teacher totals are 117 collections, **5934 qualified rows and 864 optimizer steps**. Sparse policy distillation can therefore have substantial influence despite its 0.25 coefficient and sample reuse ≤2. It is not extra on-policy PPO data.

| Phase | Scheduled turns | Actual searches | Qualified labels | Completed rollouts | Timeouts | Insufficient coverage |
|---|---:|---:|---:|---:|---:|---:|
| Early Canal | 1872 | 1872 | 1756 (93.8%) | 33143 | 94 | 9 |
| Late Canal | 1872 | 1865 | 1468 (78.4%) | 21147 | 117 | 217 |
| Early Rail | 1872 | 1872 | 1302 (69.6%) | 16630 | 149 | 297 |
| Late Rail | 1872 | 1833 | 1408 (75.2%) | 29923 | 120 | 249 |

These common fields cover all 59 teaching iterations. Attempt counts were not instrumented in this run, so completed + interrupted is not presented as exact attempts. Detailed fallback/interruption telemetry survives only recently. In iteration 59: 127 searches across 128 reservations, 107 labels, 1759 completed/831 interrupted rollouts, seven timeouts, one fallback, twelve insufficient-coverage failures. One reserved turn had no search. These are **old clock expirations**, not measured new-work watchdog rates.

Latest teacher fit goes from weighted KL .25356 to .23462 and entropy 1.02338; all 107 qualified choices already matched the policy's global top choice before fitting, and top-1 ends at 98.13%. Visit distributions contain information beyond top-1, but the observed loss reduction is in-sample fit, not better decisions. No matched search-on/search-off strength evaluation or decision counterfactual survives. Teacher outcomes never train the value head; current-source/current-model/reward identity and minimum support are checked before distillation. Post-teacher sampled KL now measures total drift from the ordinary collection policy, which PPO-only KL previously missed.

## Gameplay and throughput

Latest learner diagnostics show 4795 builds: 1838 CoalMine, 1685 Brewery, 542 IronWorks, 365 Manufacturer, 258 Pottery, 107 CottonMill. Coal/brewery account for 73.5%. There are 984 development selections, 858 beer payments, merchant beer uses by bonus 21 Money / 57 VP / 50 Income / 40 Develop, with 12 bonus-development and 28 skip choices. Pottery is legal and used; the held-out policy identity is not a banned industry.

There are 737 Pass selections across 9920 primary action decisions. Historical tables account for 402/2480 (16.2%), versus 94/1488 (6.3%) against IL and 241/5952 (4.0%) against v4. Historical-table observed income averages −2.48 versus +2.38 against v4. This supports investigating economic fragility against neural opponents, possibly supply industries depending on opponents' demand. It does **not** prove that a particular Pass, development, rail spend, or sale was avoidable.

There are 362 selected `sell` choices and 908 decisions with a feasible sale; initial and continuation sales were not separated. Efficient grouped sales cannot be inferred from those counts. “Last-seen unflipped” counts (733 CoalMine, 689 Brewery, etc.) are public-view proxies, not exact unsold/wasted tiles at era scoring. The 385 liquidation choices are not the 421 insolvency events. Industry-level trajectories, grouped-sale sizes, exact end-era VP/waste, and action-attributed rail cash spending are missing. New diagnostics separate sale starts/continuations, count Passes with legal non-Pass alternatives (not necessarily beneficial alternatives), retain era/action and public next-tile build levels, and preserve learner seat outcome sums. Exact rail-spending attribution still requires action traces; it is not fabricated from frequency.

Iteration-59 wall time: collection **862.13 s**, replay ingestion **20.04 s**, PPO **3.89 s**, search/distillation **402.77 s**, evaluation **99.39 s**, cleanup **19.85 s**, total **1408.70 s**. Collection means are 848.16, 855.33, and 859.99 seconds in the early/middle/recent measured windows; PPO about four seconds throughout. Cumulative search is 23922.58 s versus 52844.29 ordinary s, a 31.2% share of their sum, well below the 90% admission allowance. The allowance is not a target to consume.

| Latest measured component | Evidence and interpretation |
|---|---|
| Simulator / IPC | 2590.40 s summed actor stdout wait; `host.apply` 1560.34, choices 355.93, observe 987.48 overlap this RPC work; do not add them as independent wall costs |
| CPU admission / v4 | 14658.64 s summed queue wait, 9897.56 s external-policy elapsed; overlapping actors/search workers, not machine elapsed time |
| Validation / encoding | 727.54 / 209.89 s summed actor work |
| Inference | Ordinary-collection delta 27616 positions / 26369 forward calls = **1.047** positions/batch; 41.53 s forward host elapsed, 57.83 s summed queue wait, 3.42% padding |
| Replay | 88 shards, 16 readers, 238.61 s summed read/validate, 14.64 s preparation, 20.04 s join wall; about 12.72 reader cores during the old nonstreaming ingestion interval |
| Optimization | Forward/backward 1.516 s, optimizer .032 s, policy measurement 1.578 s, tensor preparation .532 s; no case for larger model/batches solely to occupy VRAM |
| GPU memory | About .576 GB allocated / 3.437 GB reserved in the learner report; not GPU compute utilization |

The report's 2,434,447 inference batches are cumulative for the invocation and include other work; they must not be divided by this iteration's wall time. Expired-request counts there likewise cannot be interpreted as the latest ordinary failure rate. Native CPU time and kernel utilization were not recorded. Later checkout improvements (fused native calls, shared memory, ready-leaf batching, streaming replay, demand-based GPU replicas) need new matched lab measurement; this review makes no speedup claim.

Keep 1–6 allocated RTX 6000 Pro GPUs selectable at runtime. The learner remains on logical `cuda:0`; frozen inference replicas are routed on demand. The review presets preserve the recorded 48 × 1 actor layout and 64/4 CPU budget, rather than changing scheduling and learning simultaneously. Current transport defaults still apply. More GPUs do not multiply games/search allocations or the shared 60 CPU slots. Do not increase the model size to fill memory.

## Implemented changes and configuration

| Item | Before | Review continuation / optional comparison | Rationale |
|---|---|---|---|
| Adam LR after resume | Saved group could override constructor setting | Apply verified LR/decay after restoration; log before/after | Makes an approved LR comparison real while preserving state |
| Reward | W + .1P + .001VP − costs .005/.025 | Unchanged, same reward identity | No coefficient optimum established |
| Main LR | 1e-5 | 1e-5; separate half-rate arm 5e-6 | Avoid assuming an unstable LR from diminishing KL alone |
| Epochs / clip / entropy / value / batches | 2 / .2 / .005 / .5 / 2048 | Unchanged | No isolated evidence supporting simultaneous tuning |
| Monitor / broader setups | 1 / 1 | 16 / 16 in review and half-rate presets | One setup cannot support setup uncertainty; schedule frequency unchanged |
| Champion gate | 8 setups, .55 point threshold + lower bound >.5 | Unchanged | Existing rule correctly declined promotion |
| Work limits | Export used 12 s NN / 1 s v4 | Current whole-turn NN 32/16/16/16 attempts, v4 16, four workers, 120 s stall watchdog | Preserve the already implemented fixed-work revision; do not relabel old measurements |
| Teacher targets | Old source could retain stale utility readiness | Current source invalidates/refits utility after teaching; reviewed transition recalibrates on fresh ordinary data | Existing repair carried forward; no old teacher reuse |
| Diagnostics / retention | Summary v2 dropped actual actor/LR/row fields; weak distribution/drift coverage | Bounded summary v3, reward moments/covariance, exact categorical probe, cumulative post-teacher drift, gradients/legal counts, seat/era/economy summaries | Make subsequent tuning assessable without retaining bulk replay |
| Architecture / population / CPU | Existing utility model, 16 recent +48 anchors, shared 60 slots | Unchanged | Preserve continuation and avoid confounded throughput changes |

Presets: [control](../configs/step5/joint7-control.json), [review continuation](../configs/step5/joint7-review.json), [half LR](../configs/step5/joint7-lr-half.json). All were derived from the **saved configuration**, not today's `lab.json`. Control preserves original numerical learning/curriculum/evaluation settings with required current fixed-work contracts. Review changes only evaluation sample counts relative to that control; half-rate changes only shared LR relative to review. Current source repairs/telemetry are common to all arms. This is not a claim that any arm exactly reproduces the old time-limited implementation.

Code changes: [optimizer restoration and guards](../src/brass_ai/recovery/resume.py), [PPO metrics](../src/brass_ai/recovery/updates.py), [bounded statistics/probe](../src/brass_ai/recovery/review_metrics.py), [post-teacher drift](../src/brass_ai/recovery/league.py), [retention](../src/brass_ai/recovery/history.py), [gameplay diagnostics](../src/brass_ai/recovery/diagnostics.py). [Pinned source receipts](../configs/learning/selfplay-resume-sources.json) retain earlier transitions and add exact new ones; unapproved reward/config edits still fail. [Preflight/backup helper](../scripts/joint7_preflight.py) verifies actual checkpoint/Adam/IL files without training, then optionally archives the inactive run.

## Verification

Verification passed **72 focused tests**: the 71-test suite and an additional generation-60 resume fixture. They cover official tiebreakers/shared wins, unbounded linear VP, capped once-only causal insolvency, live/offline remaining-budget parity, terminal/leaf utility, stale PPO/teacher rejection, source/config guards, Adam/RNG/tensor preservation, fixed work across workers/subchoices, and teacher readiness/recalibration. New tests check the LR overwrite regression, exact categorical KL, RNG-neutral telemetry, reward covariance, retention missingness, and learner seat attribution. See [focused test log](../artifacts/recovery/joint7-review-20261011/focused-tests.txt) and [generation-60 fixture log](../artifacts/recovery/joint7-review-20261011/generation60-test.txt).

The [two-game native CPU smoke](../artifacts/recovery/joint7-review-20261011/native-smoke/report.json) completed both games, committed generation 1 with two PPO updates, calibrated the utility head, used streaming replay/shared inference, and emitted the new metrics. It used a fresh small fixture model and the smoke LR 3e-5; it did not load joint-7, run a search-teacher campaign, measure CUDA throughput, or establish playing strength. Search-budget/readiness behavior was covered by focused tests. No long training campaign was launched.

The [verification receipt](joint7-60-evidence/verification.json) preserves log hashes and scope. After the smoke, a retention-only edge case was corrected to keep the last eight epoch summaries; all seven review tests passed again. The receipt records both source fingerprints. All nine supplied result files remain unchanged. Documented CLI arguments parse, report links resolve locally, and the lab helper correctly rejects absent model files without creating a run.

Reproduce the analysis without Torch or game dependencies:

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python scripts/review_joint7.py
```

Run the focused checks in the lab environment (no training campaign):

```sh
PYTHONPATH=src:tests/Recovery:tests/R2 python -m unittest \
  test_joint7_review test_combined_reward test_insolvency_recovery \
  test_teacher_readiness_review test_teacher_work test_reviewed_resume \
  test_teacher_work_resume test_throughput_resume -v
```

## Exact safe continuation commands

Use the updated project source/configuration on the lab and the existing conda environment. Keep the scheduler's `CUDA_VISIBLE_DEVICES`; `auto` uses the allocated visible GPUs, whether one or up to six. These commands verify and archive the committed run, pin generation 60 against future retention, run runtime preflight, and request **one** continuation iteration (320 ordinary games).

Run these commands in order and stop if any fails. The checkpoint/lineage verification must succeed before training.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/joint7_preflight.py \
  --root /data/not_backed_up/yxu209/ai-training/ \
  --config configs/step5/joint7-review.json \
  --backup artifacts/joint7-review/iteration-0060-original

python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training/ \
  --config configs/step5/joint7-review.json pin \
  --run artifacts/step5/v4-selfplay-joint \
  --paths artifacts/step5/v4-selfplay-joint/generation-0060.pt

python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training/ \
  --config configs/step5/joint7-review.json --device cuda preflight

python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training/ \
  --config configs/step5/joint7-review.json --device cuda \
  --inference-devices auto train \
  --run artifacts/step5/v4-selfplay-joint \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source --iterations 1
```

The backup must be a new directory; it copies existing run files without overwriting sources. After successful archival, repeat preflight without `--backup` if needed. Keep the IL parent runs/manifests in their original locations: the helper checks completed IL ancestry, not just a loose `best.pt`. Repeating `--imitation-run` supplies the recorded signature/origin; an existing self-play state follows the resume path and does not run IL again. Do not use `best.pt`, `--initial`, or `--from-run` as generation-60 replacements.

The source transition preserves checkpoint bytes, compatible tensors/Adam steps, counters, population, accepted best, and recovery-window progress. It rejects nonempty `iteration-0060` or an uncommitted generation 61 from another source. Fresh source-bound PPO is required; old teacher labels cannot be relabelled. Utility readiness is invalidated and recalibrated before teaching. No reward-semantic change is included in these commands.

An independent matched policy panel for the frozen generation-60 baseline can be run after runtime preflight, using a new output directory:

```sh
python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training/ \
  --config configs/step5/joint7-review.json --device cuda \
  --inference-devices auto evaluate \
  --candidate artifacts/step5/v4-selfplay-joint/generation-0060.pt \
  --references artifacts/step5/v4-selfplay-joint/initial.pt \
  --lineups recent-neural mixed-v3 --setups 16 --seed 29761000 \
  --partition validation --no-search \
  --output artifacts/joint7-review/evaluation-g0060-fixed16
```

This is 64 games per lineup, 128 total; v4 opponents still use fixed 16-attempt search while the candidate is policy-only. It is a new baseline, **not comparable as a clean search-budget match to the archived one-second v4 panels**. Search-on candidate evaluation should use a checkpoint freshly calibrated by the repaired pipeline; do not certify the old generation-60 readiness receipt as such.

## Small controlled comparison

1. Preserve the generation-60 archive before any arm. Use isolated copies of the lab project/run, keeping the relative run identity `artifacts/step5/v4-selfplay-joint`, IL ancestry, frozen models, Adam state, and counters unchanged. Do not point two writers at the same run, or change `fresh_run` metadata to make a copied checkpoint load. Each project copy stays under the lab experiment area and has its own root argument. Run arms sequentially under the same CPU/GPU allocation.
2. For the measurement-only change, compare `joint7-control.json` with `joint7-review.json`. The learner settings and budgets are identical; the first two continued iterations should differ only in observational telemetry/evaluation coverage when a scheduled panel occurs. No scheduled gate is due before 20,000 games. Use the external matched panel above for both.
3. Primary learning comparison: review LR **1e-5** versus `joint7-lr-half.json` LR **5e-6**, two iterations each from the exact generation-60 state. Both stop at 19,840 ordinary games, before the 20,000-game mixture boundary. Same fixed work, search frequency, seed cursor, reward, batches, epochs, and opponents. This short comparison diagnoses update/drift sensitivity, not a definitive strength ranking. Inspect effective LR, exact/sampled KL before/after teaching, utility prediction, reward distributions/cap rate, insolvency, and fixed-panel official outcomes.
4. Evaluate generation 60 and both endpoints on the same 16 validation setups and four candidate seats, separately against IL, the same fixed historical reference set, and fixed-work v4. Report paired setup differences/intervals, individual seat outcomes, rank/VP, failures, and cost. Do not evaluate each arm against a different moving historical panel. Keep final test setups untouched for a later confirmation; 16 setups still give wide intervals for rare v4 wins.
5. Only after this measurement consider the three reward arms above, with explicit target migration and fresh calibration, holding the selected LR fixed. If cumulative post-teacher drift is excessive despite modest PPO KL, a later single-factor `distill_weight` comparison is preferable to simultaneously changing LR, epochs, entropy, curriculum, and search work.

Remaining uncertainties are actual checkpoint/Adam integrity on the lab, missing per-game economic/return distributions, post-PPO IL forgetting, the causal role of old stale utility readiness, strength effects of fixed versus timed search, teacher decision benefit, and real 1–6 GPU throughput. The added telemetry and matched evaluations address these; the present logs cannot resolve them retrospectively.
