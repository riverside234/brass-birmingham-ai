# Fresh v11.1 self-play: quality pool and strength ratings

Date: 2026-10-18. Implements the supplied *Self-Play Revision and Strength Rating* plan, with the user's two clarifications: keep the 50/45/5 mixture fixed and issue a v4 warning for review; report panels but require manual deployment qualification. All thresholds remain experimental.

The later [search-free v4 follow-up](#search-free-v4-follow-up) below supersedes the original ordinary-v4 search setting. Earlier validation evidence retains its original implementation identity.

## Effective recipe and compatibility

| Table type | Share | Tables per 320 | Learner seats/table | Learner trajectories |
|---|---:|---:|---:|---:|
| Current-model self-play | 50% | 160 | 4 | 640 |
| History | 45% | 144 | 2 | 288 |
| v4 | 5% | 16 | 2 | 32 |
| Total | 100% | 320 | | 960 |

Recipe `latest-model-selfplay/3.0.0` replaces the default 60/30/10 recipe. Each mixed table type cycles through all six learner seat pairs using committed game progress; ordinary table rotation is retained. History uses each pair 24 times per 320-game iteration. Six equal counts are impossible for 16 v4 games: counts are two or three and the cycle continues across iterations. This avoids permanently starving a pair.

The capacity is **24 training-pool members**, including pinned IL. The compatibility `population` declaration is 2 + 22; quality-pool membership replaces reservoir selection for recipe 3. Rated checkpoints and recent full resume checkpoints can make the retained checkpoint count larger than 24.

Compared with the pre-task template, only `selfplay_recipe`, `curriculum`, `population` and the new `table_plan`, `opponent_pool`, `rating` blocks change. Model, features, completed IL, PPO settings and categorical stopping, reward, optimizer/LR controller, calibrated IL-KL limits, rehearsal floor, ordinary v4 search and disabled teacher settings are preserved. Learner exploration remains 0.08; frozen historical policies remain 0.05. Teacher budgets remain 9/12/12/12 seconds with `search_wall_fraction=null`. The learning compatibility digest remains `b3b944dbc492bc1483c41bdb5a9d566f13e2b2a8b7988dbeec8fcb9ebb372674`.

Completed v11.1 IL is reused unchanged, with an exact preceding-source handoff receipt. Existing PPO configurations/runs are not converted or edited. Older signed recipe constants remain valid for their original semantics, but no new PPO source migration approval is added. New blocks are part of the run signature. Matching-source continuation restores model/Adam/RNG, counters, pool scores, health/admission windows and pending rating progress.

## Quality-score pool

- Independent history-seat draws: 20% IL, otherwise `p_i = 0.8 softmax(q)_i + 0.2/N`. Both seats may draw the same member. Before any scored admission, IL fills every history seat.
- After collection, update every learner/scored-opponent seat pair with `q_i -= 0.01 * outcome/(N*p_i)`, where outcome is 1/0.5/0 after official tiebreaks. The denominator uses the **conditional scored-pool probability frozen during planning**, not its unconditional 80% mixture weight. Subtract max(q) afterward. IL results are diagnostic only.
- Consider admission every five committed iterations. Require five recorded healthy iterations under the existing fixed-probe health/behavior/evidence checks, and minimum `KL(member || candidate) >= 0.02` over all members. New q starts at max(q). The health window is not retroactively filled; the first admission may occur later than the plan's approximate generation 15.
- At capacity, protect IL, the two newest members and admissions younger than 20 iterations. Evict the lowest-q eligible member; exact q ties use minimum KL to another member. Skip admission if all members are protected.
- State includes q, planned p/N, member/checkpoint/probe receipts, health window, admission health and membership events. A digest binds it to the generation checkpoint. Derived weights-only paths may change without altering this digest; scores, identities and probe receipts may not.
- Reports and compact history retain q before/after, planning p, next distribution, seats, comparisons, raw outrank rates, five-iteration comparison-weighted rates, capacity and membership decisions. Cache/retention tests preserve active members and rated models even after training-pool eviction.

## Ratings and review

Every ten generations, evaluate 128 main games: 32 fixed validation setups and four candidate-seat rotations. One reference seat is IL for half the setups and v4 for half. Two seats come from the eight nearest rated checkpoints, using the latest rated checkpoint as the new candidate's initial estimate. Startup uses IL + v4; when only one rated checkpoint exists, it fills both extra slots. Evaluation uses exploration zero, neutral focus and no search. The pinned v4 reference is Balanced from the existing training bank; bank admission `evaluation` is reserved for the Pottery holdout. This distinction does not change the games' held-out validation split or evaluation provenance.

Two separate default diagnostic panels add 64 games: eight setups × four rotations for one learner versus three v4 bots, and eight × four for two learners versus two IL policies. Existing all-learner health evidence is attached separately. Panel games are excluded from the PL likelihood and all evaluation games are excluded from training.

PL fits agent theta plus zero-sum seat effects with the requested Gaussian priors; IL theta is fixed at zero. Official residual ties use a local seeded shuffle. Analytic gradients drive L-BFGS-B. Every slot refits all main games in canonical order and performs 200 bootstrap refits, resampling setup IDs once across all slots/rotations. Stored results include 173.7×theta points, 95% intervals, probabilities of outranking IL/v4, seat effects and paired difference intervals. Directional strength flags require the relevant difference interval to exclude zero.

Immutable per-slot specifications, compact outcome JSONL, panels, fit reports and bootstrap samples are committed before rating state advances. Interrupted retries validate/reuse their evidence; committed slots do not repeat. A new upper interval more than 50 points below the best point estimate raises a regression alert. Three strictly declining v4 panel win shares raise a review warning. Neither changes the curriculum, promotes a model or invents deployment pass criteria. The highest lower-bound candidate is shown only as a review aid; deployment remains pending manual review.

## Validation evidence

All checks use isolated artifact directories. No production training checkpoint was modified and no full native rating campaign or long training run was launched.

- **110 focused regressions passed**: the 109-test suite plus one additional offline-refit command regression. All 11 rating tests were rerun after the final bank-role clarification. Coverage includes exact weighted win/tie/loss arithmetic, duplicate opponent draws, sampler floor and recovery, health/KL gates, protected eviction, 24-member capacity, pair rotation, 960 learner trajectories, score/checkpoint continuity, rated-checkpoint retention, PL gradients/known-parameter recovery, paired bootstrap, immutable-slot retries, warnings/manual qualification, and report exports. The source review, PPO guard/LR, calibrated IL-KL and retention regressions are included.
- A native three-game PPO smoke from the existing completed diagnostic v11.1 IL collected two self-play tables and one history table: ten learner trajectories, 748 learner decisions, 337 actor-eligible decisions, and one PPO step. Teacher rows/steps stayed zero. Weights handed off exactly, and Adam/RNG/controller plus pool digest restored exactly. The source IL files were unchanged. The final-source iteration took 98.8 seconds (54.8 collection, 14.5 PPO, 28.4 retention); full command/startup/probe/resume checks took 230.3 seconds during concurrent checks. This small IL dataset supplies only 144 fixed-probe positions, so admission/LR-increase evidence remains insufficient; no thresholds were relaxed.
- A separate native three-game rating smoke exercised a main lineup, one-vs-three v4 and two-vs-two IL. It finished in 91.30 seconds and left the source training artifacts and RNG unchanged. One setup per panel cannot support uncertainty or strength claims. These are integration diagnostics, not deployment evidence.
- Three **synthetic** slots fitted 128/256/384 cumulative main games with 200 bootstrap refits each. Measured fitting times were 1.26/0.84/1.37 seconds on this machine. Known-parameter and analytic-gradient regressions pass; IL remains exactly zero.
- Forty independent synthetic startup-protocol trials gave coverage of 37/40 (92.5%) for both the candidate and v4 intervals, and 36/40 (90%) for their paired difference. The median candidate interval half-width was **47.56 points**. This small calibration study is compatible with sampling variation around nominal coverage; it does not establish universal 95% coverage, ±40-point precision, or suitability of the PL assumptions for real gameplay.
- The progress curve was rendered and visually checked. Original-byte report export tests cover compact JSONL/panels/SVG and exclude dense tensors/full replays; altered committed evidence is rejected. An actual export of the final smoke copied six original files with matching hashes and no warnings. The absent first-generation compact archive was correctly listed as missing. The bundle is preserved under `artifacts/quality-rating-20261018/report-bundle/`.

Reproducible evidence lives under `artifacts/quality-rating-20261018/`: `before.json`, `final-tests.log`, `release-smoke/`, `native-ratings/`, `synthetic/` and the generated-config audit. The original completed diagnostic IL is `artifacts/feature-engineering/v11-1-20261016/smoke/imitation`. Final learning implementation: `43bd782b1a1d8915a26011a1bf5db7722d1f3b711fa043e720264aa4bc78a214`.

## Runtime and storage implications

192 held-out games per 3,200 ordinary games add **6% game volume**, plus existing monitors and fitting/I/O. This is not a 6% wall-time prediction: v4 ratings have no search, ordinary v4 table count falls from 32 to 16, ordinary learner trajectories fall from 1,024 to 960, and hardware/concurrency affect cost. Full rating refits grow with accumulated evidence; the small synthetic timings are not a long-run guarantee.

Healthy admission checks add one current-model forward pass over the fixed probe every five iterations and compare cached member probabilities. Pool capacity increases from 16 to 24; every tenth generation also retains a separate rated snapshot indefinitely for future comparisons. Compact outcomes and rating fits stay permanent, while completed bulk rating replays follow diagnostic-sample retention. Report bundles include outcomes/panels/curves but omit large probability/bootstrap arrays and model files.

## Files and commands

Core additions: `recovery/quality_config.py`, `quality_pool.py`, `plackett_luce.py`, `ratings.py`. Integration: `config.py`, `ordinary.py`, `league.py`, `il_kl_probe.py`, `retention.py`, `history.py`. Defaults/admission: `configs/step5/joint7-latest.json`, the completed-IL source review, and `requirements-step5.txt`. Tooling: `scripts/strength-ratings.py`, `scripts/export-training-reports.py`. Focused tests and isolated smoke/benchmark scripts accompany the changes. Pre-existing categorical-KL guard and other unrelated edits are preserved.

Use the [updated lab commands](../docs/MLP_V11_LAB.md#fresh-self-play-from-completed-v111-il) to install dependencies, bind a **new** config to the completed IL and start a **new** self-play directory. If `joint7-v11-1-grouped-lr` already exists, use a new suffix such as `joint7-v11-1-grouped-lr-quality` consistently for config/run/export; do not rename or redo the completed IL.

Continue the newly created run with its generated config and original IL path, omitting `--fresh`; `--iterations 20` means twenty additional iterations. `--expected-generation` is optional. Ratings run automatically at the configured slots. To refit stored ratings without collecting games, use `python scripts/strength-ratings.py --run <new-run> --output <new-review-directory>`. Export analysis with `python scripts/export-training-reports.py --run <new-run>`; see [report contents](../docs/EXPORT_TRAINING_REPORTS.md).

Verified behavior is implementation and integration correctness. Increased stability or playing strength remains unproven and needs the fresh run's held-out evidence.

## Search-free v4 follow-up

The fresh generator now writes the strictly boolean `v4_search_enabled=false`. All ordinary v4 seats use the existing incumbent-only policy at every game count. Scheduled mixed-v4 and held-out Pottery panels inherit the same flag and run on ordinary actor workers; rating/reference panels were already search-free. Neural evaluation search remains a separate setting. The stored one-second v4 MCTS budget is inactive while this flag is false; existing cancellation, native incumbent work limits and failure watchdogs remain in force. The model, completed IL, teacher settings and the rest of the 50/45/5 quality recipe are unchanged.

Old signed configs without the flag retain their prior behavior. The new flag and actual search-free policy identities enter the run/evaluation/collection signatures, so searched replay caches cannot be relabelled. The frozen demonstration-teacher identities remain unchanged for completed IL admission. An exact source receipt admits completed v11.1 IL from the immediately preceding implementation; no PPO-run migration was added.

`git fetch origin` and `git pull --ff-only` confirmed `main` was already current at `e69fe3cc090b1c60b9fb75c8f07764c1714478d8` (`add new config`). Per the user's final instruction, the existing `configs/step5/joint7-v11-1-grouped-lr.json` is byte-for-byte unchanged. The lab's generation, training, continuation and report-export commands now consistently use **joint7-v11-1-grouped-lr-quality**, retaining the completed **features4-v11-1-batch7-il** name. Generate the new config on the machine holding that IL; do not reuse the older config for the new recipe.

Validation evidence is under `artifacts/v4-search-off-20261018/`. Eight new regressions cover strict/default flags, old-config semantics, every game age and legacy boost, the full 320-table plan, learner/opponent exploration, both evaluation banks and worker routing, rating references, cache identity and unchanged IL binding. In total **88 focused regressions pass**; the archived-controller fixture was corrected to omit the newly introduced flag, and its 19-test module was rerun successfully (`regressions.log`, `controller-regressions.log`). The actual `feature-engineering.py configure` command succeeds against the completed local diagnostic v11.1 IL. The production IL files are on the lab machine and were not available for a local hash check; the generator and training startup retain their exact completion/hash/lineage checks.

The isolated native smoke completed **five real games: one ordinary 2-learner/2-v4 table and four 1-learner/3-v4 evaluation rotations**. All **14 v4 seat records** reported zero allocated, attempted and completed MCTS simulations while recording actual turns. Both collections used ordinary actors. Source checkpoint/run files and caller RNG were unchanged; no optimizer updates ran. The bounded command took **364.56 seconds**, including startup, collection, validation and I/O, with some concurrent regression activity. Evidence: `native-smoke/report.json`, `records.json`, the ordinary replay and the scheduled evaluation report. This is integration evidence, not a before/after speed comparison or a strength estimate.

Only `v4_search_enabled` changed in the current default template. Learning compatibility remains `b3b944dbc492bc1483c41bdb5a9d566f13e2b2a8b7988dbeec8fcb9ebb372674`; this follow-up's implementation is `3a3a5d230b62f31451664bedeacf83e1d8a87d4c15e924b5b6fe2d1d35fe2c71`. Files changed for this follow-up: `recovery/config.py`, `curriculum.py`, `league.py`, `evaluation.py`, `step5/evaluation.py`, the default config, completed-IL source receipt, lab instructions, this report and focused test/smoke files. The archived controller test fixture preserves its historical search setting explicitly.

Removing v4 MCTS removes its simulation work and search-worker reservations. Incumbent planning, model inference, replay validation and rating games still have a cost. No before/after throughput campaign or playing-strength claim is implied.

## Monitor and Broader Evaluation disabled

The subsequent user request disables both automatic development panels in the Python default, current template and fresh-config generator through `evaluation_schedule.enabled=false`. The existing scheduler returns before selecting opponents, creating output or collecting games, both at startup and after commits. Intervals and setup counts are retained as inactive settings. Independent Plackett–Luce ratings and their diagnostic panels remain enabled every ten iterations. Existing signed configs/checkpoints and the completed IL binding are unchanged; the lab generation command produces the new setting.

Ten focused regressions pass. A separate scheduler check at 0, 6,400, 25,600 and 1,000,000 committed games confirms zero evaluation calls, zero state writes and unchanged state. The actual config-generation command also succeeds against completed diagnostic v11.1 IL. No gameplay or optimizer updates were needed for this scheduling switch. Evidence: `artifacts/monitor-broader-off-20261018/`. Implementation: `aefd86e35f9c6a3e2d71bec0e51240235c6da1ce6ade4ebfd34e18523d80442d`; learning compatibility remains unchanged.
