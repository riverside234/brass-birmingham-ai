# Stronger four-player legacy Hard AI and a tactical opponent bank

**30 September implementation update:** M0–M5 of [Masked ISMCTS](MASKED_ISMCTS_REVISION_PLAN.md#10-implementation-order-and-component-map) are implemented. Current training uses a fresh 1,840,109-parameter MLP-ResNet, CPU/GPU batching, separate legal/padding/expansion masks and completed root-visit teachers. Package 1.4.9 switches all six v3 styles to the shared C# backend, preserving 2/5/12 and old-save 10 seconds plus resource/UI management. The new lab bank uses one second directly. Historical S1/S2/S3/R4 variants and checkpoint-import commands below are archival. Use the [current lab guide](../docs/TRAINING_RECOVERY_LAB.md) and [exit record](../docs/MASKED_ISMCTS_EXIT_CHECKS.md).

**30 September selected revision, plan only:** [Masked ISMCTS](MASKED_ISMCTS_REVISION_PLAN.md) will replace the active v3 beam/reply planners for all six styles. Preserve the existing cash/completion/merchant-beer scoring and persistent style helpers. The planned game patch keeps 2/5/12-second choices; the new training bank uses 1s directly with no clock-calibration tournament. H0–H5 and old S1/S2 bank evidence below remain historical records.

**V3 follow-up:** [industry completion and merchant beer](../docs/LOCAL_AI_V3_CONVERSION.md) adds current-era deadlines, viable merchant routes, resource-demand estimates, complete beer-source alternatives and cash protection to all six S1 styles. A new immutable four-player bank has separate qualification; the earlier H4/S2 study remains historical.

27 September 2026. Design and implementation record, based on the original AI inspection, shared rules patch, official rules and BoardGameGeek discussions below. **H0–H5 code is implemented locally, including all six styles and required S2 reply search.** [R3 exit checks](../docs/R3_EXIT_CHECKS.md) distinguish verified behavior, frozen-bank admission and remaining strength/deployment gates; implementation does not imply calibrated difficulty or human enjoyment.

## 1. Recommendation and scope

Build a family of conventional C# heuristic opponents from the legacy Hard AI's useful tactical ideas. Replace its single winning-rule decision with scored, explicitly specified legal candidates, then add complete-turn planning. No neural network or GPU is required. These opponents can provide an earlier playable improvement, reproducible benchmarks and diverse training partners for the later Daiyu-based model.

**Four-player optimization is the scope for all six profiles.** Tune evaluator weights, tactical preferences, candidate limits and thinking budgets using four-player games only. Optimize for three independently motivated rivals: contested industry sites and beer, shared resource demand, spending-based order and the root's actual next turn. Use the host's legal four-player setup and remaining-action counts; do not import two-player assumptions or treat the other three seats as a coalition.

The deployment target is one human plus three AI seats; automated tuning and training-opponent evaluation use four AI seats. Every bank configuration and registry entry declares `player_count: 4`, and bank runners reject other counts. The game adapter enables these profiles only in four-player games; two-/three-player games retain their existing AI path. Shared adapter changes still need compatibility smoke checks, but strategic optimization and strength claims for those player counts are outside this delivery.

Keep the existing [search design](SEARCH_DESIGN.md): S0 fast decisions, S1 both current actions together, S2 sampled independent opponent responses, and optional deeper search. The legacy family supplies a handwritten policy and value evaluator to these interfaces. It is part of the [implementation plan](IMPLEMENTATION_PLAN.md), not a separate simulator or a substitute for the longer-term model work.

Preserve three distinct identities: untouched historical Hard, the original strategic policy running with declared rules/safety fixes, and each strategically improved variant. The [local patch report](../docs/LOCAL_PATCH.md) records patch 1.1.0, including B01–B16 and R01–R03. Its AI guards and resource corrections are existing compatibility work; do not count them as a newly trained or strategically improved opponent. Confirm exact binaries before each benchmark and retain that rules version in every result.

## 2. Strategy sources and how to use them

Both user-supplied threads were read directly in the browser, including substantive text replies. The additional Birmingham tactics thread was also inspected. Search results for older Lancashire guides were excluded. Player advice supplies testable heuristics; it does not establish optimal play or a universal winning score.

| Source inspected | Useful theme and qualification |
| --- | --- |
| [Dave C / PROJ, Strategy primer and openings (4p)][primer], original post 2 January 2021, later discussion and November update | Evaluate industry sequences and market saturation. Heavy cotton/goods plans depend on access and competing demand. Replies and the author's update expose differing assumptions about shared beer and canals; preserve alternatives rather than one mandatory opening. |
| [IM Spoon, Guide to score 5VP/action][efficiency], 15 June 2023 and subsequent discussion | Evaluate the whole investment sequence, including financing and setup. Cash, income and turn order have timing-dependent value. Treat the suggested points-per-action yardstick as a comparison aid, not a legality rule or a score guarantee. |
| [milos majstorovic / mmajstor, Advance Strategy and Tactics – Hints][tactics], 16 March 2021 | Connect era preparation, public resource demand, useful rail links and spending-based turn order. Test the advice against current opportunities; fixed quotas for loans, scouting, development or any industry are not universal requirements. |
| [Roxley official rulebook][rules], with the local rules audit and patch report | Rules authority for resources, actions, scoring and transitions. Resolve legality through the declared host contract; a forum recommendation never overrides it. |

The mechanisms and tests below are our proposed implementation of these themes. Recompute costs and rewards from game data. Guide examples sometimes include future Rail points in their discussion of Canal value; they are not the actual score displayed at the Canal boundary. Store concise source notes and links, not copied articles, screenshots or rigid transcriptions of their openings.

## 3. What the existing code suggests improving first

Evidence paths below are relative to the local `ai-research/decompiled/` inspection tree, which remains excluded from Git. These findings identify test targets; they do not quantify their individual contribution to weak play.

| Observed implementation | Improvement |
| --- | --- |
| `BrassApplication.AI.RuleBasedSystem.RuleSelectors/PrioritySelector.cs:17–26` inserts integer priorities into a dictionary and selects the first applicable rule in descending order. | Score multiple concrete proposals in common units. Use a stable list/tie-breaker so equal scores or profile weights cannot cause duplicate-key errors. Retain original behavior only in the frozen stock wrapper. |
| `BrassApplication.AI.RuleBasedSystem.Rules/PlaceLinkDouble.cs:35` requires £29, income 11, one action remaining and Rail round 4 or later for this particular rule. | Compare affordable single/double rails in every legal Rail round. These are heuristic thresholds, not rules. Other network paths can already add a second link, so do not claim stock never double-rails early. |
| `BrassApplication.AI.RuleBasedSystem.Rules/DevelopCanalEraIndustries.cs:69–121` follows a fixed industry order and includes cotton only in its late-Canal branch. | Generate development targets from a feasible follow-up investment and available hand, rather than one global order. Other development rules remain separate sources of proposals. |
| `BrassApplication.AI.RuleBasedSystem.Rules/BuildIndustry.cs:75–94` ranks tiles by printed building/link points and accepts the first suitable slot. | Compare tile, slot, card, resource sources, sale feasibility and remaining actions together. Separate potential points from points likely to be realized. |
| `BrassApplication.AI.RuleBasedSystem.Rules/TakeLoan.cs:34–45` has fixed cash triggers and disables this loan rule at income 15 or above. Other loan-before-build/sell rules exist. | Compare borrowing now, borrowing later and no borrowing using exact income-track changes, deadlines and useful follow-ups. Avoid a universal cash or income cutoff. |
| `BrassApplication.AI.AIHelpers/BoardHelper.cs:113` uses discard heuristics; `Find2ndLink` and the network helper select continuations internally. | Preserve cards needed by the intended second action; explicitly choose whether and where to place another rail. A helper must not silently change the selected plan. |
| `BrassApplication.AI.AIHelpers/SellActionAIHelper.cs:60` favors income in sale ordering and later uses fixed merchant-development preferences. | Compare sale subsets, order, chosen merchants and beer allocation, including immediate selling versus waiting. Reuse the current patch's corrected sale legality. |
| `BrassApplication.AI.RuleBasedSystem.Rules/BuildIndustryForSale.cs:87` calls `StartBuildAction` during candidate calculation; `TurnFlow.StartBuildAction` changes `CurrentAction`. | Make proposal generation and explanation read-only. Repeated scoring, rejected candidates and diagnostic logging must leave the live state unchanged. |

The current patch replaces network and sale helper methods, as well as the B14/R03 null guards. Inspect the patched dispatch when implementing these improvements; do not reintroduce the original nearest-coal or multi-merchant-beer defects. `src/BrassPatch/Resources/ResourcesFixes.cs` still delegates second-link selection to `BoardHelper.Find2ndLink`, so explicit plan execution remains useful work.

## 4. Shared decision engine

### Legal proposals and execution

Use the common `PlayerObservation`, `LegalChoiceProvider`, typed action plan and executor from the main plan. Translate useful legacy rules into **proposal generators** that return several candidates and their rationale. Supplement them with coverage of all legal action types, resources, merchants, card choices and continuations. An upstream rule's applicability is a strategic filter, not the definition of legality.

The pure evaluator receives public state, that seat's hand/legitimate history and persistent intent. It never reads real opponent hands, future deck order, private setup discards or game RNG. Legality and execution stay in the trusted rules host. Isolate simulations and bind helper state correctly; existing shallow copies are not sufficient. Profile configuration cannot relax any rule.

Return a complete valid incumbent promptly. S1 considers both actions, retaining enabling loans, developments, scouting and connections. Commit one action, refresh the permitted observation, then validate its continuation with the remaining deadline. Preserve the first Canal round's one-action exception and the ordinary end-turn draw boundary. No silent pass or partially applied fallback.

### A common, explainable evaluator

Calculate the change in estimated final outcome between candidate afterstates. Keep a component ledger so improvements can be audited:

| Component | Calculation and boundary |
| --- | --- |
| Industry points | Already earned points plus expected **remaining** scoring opportunities for surviving industries; discount unsold/unflipped tiles by realistic completion likelihood. Overbuilding removes future entitlement, not previously earned income or VP. |
| Link points | Current scoring under the rules, plus a separately discounted estimate of later adjacent flips. Account for ownership of links that our buildings would improve. Unflipped neighbors do not guarantee future link points. |
| Financing and liquidity | One cash-flow forecast covering current money, exact income-track steps, remaining collections, spending and possible loans. Value money through investments it can enable; do not separately count the same income twice. Terminal cash/income matter through official tie handling, not invented VP conversion. |
| Access and hand flexibility | Feasible future builds/sales opened by a link, surviving industry or retained card. Count the future actions and costs needed to realize them; access is not itself a free completed investment. |
| Completion and resource risk | Exposure to contested slots, merchant beer, opponents consuming resources, depleted stocks and too few actions remaining. Include a viable self-supplied alternative where practical. |
| Opportunity cost | Compare complete feasible sequences, including loans, development, connections, scouting, builds and sales. Charge supporting actions once; amortize shared actions without counting the same future reward twice. |

Use approximate VP-equivalent weights for nonterminal comparison, fitted on development matches and frozen before evaluation. They are heuristic estimates, not calibrated win probabilities. At game end use the actual four-seat result and tiebreak order. Keep raw components in logs so a high score cannot hide an impossible sale or exaggerated income value.

For L3, produce four seat estimates from the root's permitted view, using public information for opponents rather than their sampled private hands. Back up/select the actual decision owner's component where the algorithm requires it; there is no alternating sign or combined reward for the three bots. Keep objective and value-scale settings explicit when comparing heuristic and neural search.

Five VP per action is a useful diagnostic from [the efficiency guide][efficiency], not a hard threshold. A zero-point loan can enable a profitable continuation; a nominally valuable build can require several extra actions and scarce beer. Compare incremental alternatives from the current state, while showing full sequence cost in explanations. Do not reject necessary recovery moves merely because their average is below five.

For example, a nine-point industry flipped in Canal can have one Canal award and a later Rail award if it survives. Once the first nine are in the score, only the future nine belong in its remaining-value estimate. Do not credit eighteen now and add another nine at game end.

### Tactical priorities to implement

1. **Sale timing and packing:** compare build-then-sell with sell-now and further accumulation. Reserve beer across the selected set of sales, enforce each industry's chosen merchant, and evaluate the free-development target against the current plan. Trigger earlier selling when delay risks beer, cash flow or era-end completion; never mandate one sale per era.
2. **Development with a purpose:** pair each proposed development with reachable industries and a cash/resource/action budget. Preserve an enabling first action even when its immediate value is low. Evaluate a second development tile independently; do not consume extra iron merely to fill both slots of the action.
3. **Loan sequencing:** calculate loan-before-flip versus flip-before-loan on the actual income track. Include public round spending and income obligations. A low-spend move changes the next order only according to the real scheduler and the remaining players' spending; it cannot guarantee a last-to-first sequence by itself.
4. **Rail opportunities:** price complete one-/two-rail actions, including coal per link and legal brewery beer. Compare both link orders and stopping after one. Favor valuable access and points that justify total cost; fixed location preferences only propose candidates. Merchant beer cannot fund a double rail.
5. **Resource demand:** use visible market gaps, connected demand and likely depletion before the scoring boundary. Respect nearest-coal rules and iron's different reachability. Compare selling to the market, facilitating one's own follow-up, and competition from existing supply. Never assume opponents will consume a resource just because it was built.
6. **Own and rival overbuilding:** use the exact rules predicate, including the board-and-market exhaustion condition for rival coal/iron. Price sacrificed future scoring and resources; keep previously banked points. Rival benefit/denial is player-specific, not a two-player swing automatically counted against all three opponents.
7. **Cards and scouting:** preserve scarce cards for the remaining turn and stated intent. Scout only when improved access justifies its action and three-card exchange; permit it whenever genuinely useful rather than enforcing a quota. Follow current wild-card, ownership and exchange checks.
8. **Era transition and endgame:** favor achievable completions and useful surviving presence; price Rail entry cash and resources against Canal opportunities. In the last decisions, drop value for income with no collections left and for plans that cannot finish. Forced liquidation selects legal assets with the least lost future utility and stops once debt is covered.

## 5. Six tactical profiles

All six are implemented as conditional preferences and proposal patterns over the common evaluator, with S1 and S2 settings. H5 admission remains version-specific. Each specialist retains a Balanced fallback and records why it abandoned a plan. Tactical benchmark variants need not have identical strength; player-facing personality/difficulty calibration remains separate.

| Versioned profile family | Strategic preference | Entry conditions and recovery |
| --- | --- | --- |
| `legacy-hard-balanced-v3` | Best available complete sequence with conservative completion estimates | Default comparator; switch focus when the hand, demand or remaining actions changes |
| `legacy-hard-rail-tempo-v3` | Beer/iron economy and useful early Rail combinations; spending/order opportunities | Require feasible resource access and worthwhile links; pivot to sales or supply when rail value or beer availability falls |
| `legacy-hard-cotton-v3` | Cotton development and efficient grouped sales | Check available slots, relevant cards, merchants and contestable beer before committing; scale down development or add an affordable own-beer contingency when access fails |
| `legacy-hard-goods-beer-v3` | Flexible manufactured-goods sequences with brewery use | Read each tile's actual development, beer, cost and link properties; use high-level goods only when the complete sequence fits, otherwise prefer shorter viable combinations |
| `legacy-hard-supplier-v3` | Demand-led coal/iron, useful flips and selective overbuilding | Enter unsaturated profitable supply opportunities; avoid marooned stock, excess inventory and helping a rival more than the investment warrants |
| `legacy-hard-pottery-flex-v3` | Opportunistic pottery plus compatible financing, sales and links | Require the right cards, legal development path, sites and sufficient capital; the pottery preference remains weak when those conditions are absent |

Implement strategy intent as a small per-seat record: current focus, target opportunities, required cards/resources, next milestones, remaining-action/cash budget and abandonment conditions. Refresh each turn without forgetting a good multi-turn plan. No shared intent or private information between AI seats; mutually beneficial resource use is ordinary self-interested play.

Use profile preferences only among credible alternatives, and measure when each specialist actually behaves differently. Avoid six labels that all choose the same move or six rigid policies that collapse when their preferred opening is blocked. Include crowded and uncooperative tables in evaluation, not just favorable examples from a guide.

## 6. Depth, speed and reuse

| Operating mode | Purpose | Dependencies |
| --- | --- | --- |
| L1 / S0: scored rules | Fast standalone opponent and rollout policy | Pure proposals, legal complete actions and the common evaluator |
| L2 / S1: complete turns | First intended optimized legacy release | Validated reconstruction, complete legal interface, enabling-action coverage and one turn deadline |
| L3 / S2: sampled replies | Required implementation; promotion requires measured benefit | Fair beliefs, independent L1 opponent policies and the actual next-root-turn horizon |

Start CPU-only. Measure L1 throughput, then compare proposed L2/L3 whole-turn caps such as 0.25, 1 and 3 seconds; these are experiments, not latency promises or difficulty labels. Keep deterministic fixed-work mode for regressions and real wall-clock mode for laptop acceptance. Include proposal generation, restore, execution validation and IPC in timing.

L3 uses the main plan's matched scenarios, fresh finalist checks and zero-to-six intervening ordinary opponent turns. Simulated opponents use L1 policies; do not recursively launch a full search for every reply. [S3 and OpenSpiel](../docs/R4_R5_SEARCH_LAB.md) are now required additional neural-search experiments; they do not modify this frozen legacy bank. Reuse the host, candidate representation, scheduler, telemetry and tests with the Daiyu-based learner, replacing only policy/value evaluation where appropriate.

## 7. Store and freeze opponents reproducibly

Implemented project-owned locations:

- `src/BrassLegacyAI/`: pure proposal generators, evaluator, intent and optional game adapter.
- `configs/legacy-ai/`: versioned profiles, source IDs, weights, proposal limits and supported operating modes.
- `training/opponents/registry.json`: immutable opponent IDs and verified compatible rules/observation/action schemas.
- `tests/legacy-ai/`: legal tactical positions, plan recovery, purity, hidden-information and regression fixtures.
- `artifacts/legacy-ai/`: ignored local traces, benchmark reports and compiled outputs; publish only curated authored summaries if requested.

Each registry entry must pin `player_count: 4`, policy-code revision/hash, profile/config hash, operating mode, deterministic work limit or time cap, tie-break/seed policy, observation/action schema and exact rules/patch fingerprints. Hash and validate the complete dependency set; a profile name alone is insufficient to reproduce a result. Freeze released versions and add a new version rather than changing a benchmark in place.

Reserve `stock-hard-original` for the untouched policy/rules combination, isolated with its original binaries. Reserve a separate versioned `stock-hard-guarded` identity for original strategic priorities with the selected shared repairs. Run improved candidates and that guarded comparator under the same rules for the principal strategic comparison. Historical original-game results remain separately labeled; crashes and illegal moves are reliability findings, not strategic wins. No game assemblies or decompiled files enter Git.

For save/load, retain profile version, intent and decision randomness in the agreed sidecar/game configuration, and reconstruct the right behavior through the new/load factory hooks. Keep the existing rules patch's safety guards intact. Strategy upgrades should be an optional module/configuration, not silent changes to the frozen stock definition.

## 8. Implementation milestones

These milestones expand **delivery-order item 4, R3/H0–H5**, in the [main plan](IMPLEMENTATION_PLAN.md#delivery-order-and-immediate-next-milestone). That item builds the first four-player legacy bank across the main plan's detailed sections 4 (game bridge) and 5 (heuristics/search); it is not a claim that all this work belongs to detailed section 4 alone. H0 starts with R1 baseline work. H5 freezes and qualifies each usable L1/L2 bank version before benchmark/training use; H4 is explicitly required by the user. Both S1 and S2 versions must pass their declared H5 gates before pool admission; S2 is not presumed stronger.

| Milestone | Deliverable | Completion criterion |
| --- | --- | --- |
| H0: baseline and diagnosis | Version/fingerprint manifest; four-player trace runner recording selected rule, candidate alternatives, timings and outcomes | Repeatable stock/guarded baselines; explanation and candidate collection do not mutate decisions; existing patch regressions remain intact |
| H1: shared scored actions | Pure proposal API, evaluator ledger, explicit cards/resources/merchants and valid fallback | L1 Balanced completes both eras with the shared legality and information tests; current helpers cannot override a chosen continuation |
| H2: complete-turn legacy | S1 integration, scenario fixtures and Balanced/Rail Tempo profiles | Correct setup combinations and measured improvement over guarded stock at declared time caps |
| H3: tactical bank | Remaining four profiles, intent persistence and failure recovery | Distinct behavior in qualifying positions; each profile's global or niche strength is reported rather than assumed |
| H4: required reply search | Shared S2 with L1 rollout opponents | Benefit over L2 at equal total turn time, with privacy and interruption checks passing |
| H5: freeze and integrate | Versioned registry, tournament reports and training-pool selection | Repeatable results, held-out tests and approved deployment settings; no benchmark version changes during training |

**H0–H5 local implementation:** `src/BrassLegacyAI` includes scored-action/S1 planning, six conditional styles, persistent intent journals and S2 masked-opponent replies. `src/brass_ai/bank.py` freezes sources, profiles, contracts and runtime fingerprints; `qualification.py` runs prespecified tournaments and gates disjoint pools. The R2 adapter records external behavior, styles, intents and complete-game outcomes. [Implementation details](../docs/R3_TACTICAL_BANK.md) and [exit evidence](../docs/R3_EXIT_CHECKS.md) distinguish functionality, niche admission, equal-time S2 benefit and general/human deployment gates. The original H5 implementation did not change the installed game; the separately versioned v3 follow-up is linked above.

**Qualification outcome:** all 156 prescribed games completed with zero match/rollout failures. Five S1 profiles qualify for initial training and Pottery Flex for held-out evaluation. H4 does **not** meet its strength exit criterion: Balanced S2's first-place share is 56.25%, versus 100% for S1 at the same 10-second allowance across 16 paired games. Required S2 is implemented and tested, but stays diagnostic; the qualified S1 bank is the default for initial training. General improved-Hard, live deployment, human enjoyment and Linux/lab acceptance remain open. Future S2 changes require a new frozen identity and fresh evaluation groups.

H0 can start from the existing patched engine and probes. H1/H2 depend on the legal interface and restoration work already required by main-plan stages 1–4; do that work once. The immediate target is a measurable four-player L2 Balanced/Rail Tempo improvement, followed by the other four profiles. R2's neural scaffold can proceed alongside this work, but its completion, GPU training and the OpenSpiel adapter are not prerequisites for the heuristic release. No delivery-time or strength estimate is justified until candidate enumeration and reconstruction are profiled.

Instrument stock selection once and log the captured result. Do not rerun `GetNextAIMove`, `PrintNextAIMove` or side-effecting applicability code merely to explain a historical stock decision. Evaluate new alternatives through the pure proposal API or an isolated, verified reconstruction so diagnostic work cannot change the benchmark.

## 9. Tests and promotion criteria

Use deliberately small legal four-player fixtures where consequences are known, plus full four-player games. All profile tuning, strength comparisons and promotion decisions use four-player results. Keep any two-/three-player compatibility smoke results separate; they are not optimization data or evidence of strategic strength. Target these failures:

1. Equal candidate scores, repeated evaluation and logging: stable results without dictionary exceptions, state changes or altered RNG.
2. Loan/develop/scout-then-build: the enabling first action survives, and its required second-action card is retained.
3. An affordable early-Rail double link competes with a single link; both resource payments and the second-link beer source remain legal.
4. Multiple sales share limited beer correctly; changing the merchant or sale order changes eligibility and bonuses as the rules require.
5. Selling now beats waiting when a necessary resource is threatened; waiting can win when a genuinely feasible grouped sale saves an action.
6. Extra income is priced through actual marker bands and remaining collections; a final-turn loan is not rejected or rewarded mechanically.
7. Existing supply suppresses speculative new supply; own/rival overbuilding handles spent resources and remaining scoring correctly.
8. Canal-to-Rail survival, income, scoring and turn order match the selected engine contract; no doubled counting of future points.
9. Cotton/goods/pottery openings abandon or repair their plan when beer, a site or a merchant becomes unavailable; cheap recovery remains an option.
10. Each seat passes hidden-state invariance and save/load tests; Play Now, deadlines and failures preserve a complete valid action and one confirmation.

First run broad inexpensive pilots and fix traceable failures. Then freeze code/configuration and evaluate on separate setup groups, for example 200 fresh setups with four rotations for one candidate versus three guarded stock bots. Report first-place share with ties handled explicitly, average rank, score margins, failures and latency. Use uncertainty intervals grouped by setup because rotated games are correlated. More games may be needed; predeclare evaluation/stopping rules and do not repeatedly tune on final-test results.

For a general improved-Hard label, require a meaningful advantage over guarded stock, with the lower uncertainty bound above the symmetric 25% first-place reference, plus corroborating rank/score and mixed-table results. Mixed tables must include different clockwise opponent arrangements as well as seat rotations. Evaluate identical-profile tables, diverse legacy tables, later checkpoints and at least one held-out tactical configuration/family. Stock-only success is insufficient.

A specialist that is useful only in certain conditions may remain a labeled niche challenge or training partner; do not advertise it as universally stronger. Track diagnostics such as unfinished investments, wasted resources, unnecessary setup, spending/order mistakes, endgame cash without useful deployment and repeated openings. These diagnose play and are not extra rewards to game. Report feasible recoveries and human enjoyment, not just VP totals.

The neural project's requested new-Easy-above-original-Hard criterion retains its original meaning. Report results against the stronger legacy bank separately rather than silently changing what “original Hard” means. Likewise, copying a player's suggested 155-point target does not prove strength across different merchant layouts or opponents.

## 10. Role in later training

Use the tactical bank to supply varied competent opponents before strong neural checkpoints exist. It can also bootstrap proposals or provide optional short imitation experiments; its choices are never labeled optimal merely because they follow a guide. Keep outcomes and search-quality metadata, including losses and plan failures, for the appropriate learning targets.

Once a neural population is viable, an initial experiment for opponent-seat sampling is 50% current/recent models, 25% historical models, 20% varied improved legacy profiles and 5% guarded stock. These weights are configurable hypotheses; evaluate against mixtures with less or no stock, and retain coverage of weaknesses rather than raising old-bot exposure just because it is cheap. During bootstrap, use the available legal heuristic pool and record that different distribution explicitly.

Freeze each sampled opponent's profile/checkpoint within a match, rotate seats and track the exact mixture. For occasional L3 teachers or benchmarks, preserve fair information and total-budget accounting. Use L1 variants inside rollouts to keep throughput practical. Keep a separate frozen evaluation pool; training against a profile and evaluating against the same one does not establish generalization to human play.

[primer]: https://boardgamegeek.com/thread/2571272/brass-birmingham-strategy-primer-and-openings-4p
[efficiency]: https://boardgamegeek.com/thread/3100883/brass-strategy-guide-to-score-5vpaction-and-win-th
[tactics]: https://boardgamegeek.com/thread/2622371/advance-strategy-and-tactics-hints
[rules]: https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281
