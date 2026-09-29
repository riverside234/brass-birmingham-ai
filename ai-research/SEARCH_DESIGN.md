# Search design for four-player Brass

Revision: 27 September 2026. Proposed architecture and experiments; no search implementation, trained checkpoint, or laptop search benchmark is claimed here. This specifies the thinking-time recommendation in [DESIGN.md](DESIGN.md) and the search milestones in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

Implementation references are now selected: **Daiyu AI for the Brass policy/search/training scaffold, with OpenSpiel for multiplayer backup, information-state interfaces and algorithm comparisons.** [REFERENCE_INTEGRATION.md](REFERENCE_INTEGRATION.md) pins the source revisions and maps the required adaptations.

This document remains the planner's behavioral specification. The [unified requirements checklist](IMPLEMENTATION_PLAN.md#unified-requirements-from-the-earlier-search-design) maps its earlier recommendations to delivery stages and acceptance checks. The optional OpenSpiel comparison and the POMCP-inspired S3 experiment are independent branches; both retain the complete-turn, information, personality and time-budget requirements below.

## Decision

Start with a compact policy/value model plus a planner that compares complete current-turn plans across sampled scenarios. Simulate independently motivated opponent policies until the root player next gets to act, then evaluate the resulting position from that player's permitted information. Add deeper search over future observations only after this simpler system demonstrates value at a fixed laptop time budget.

The first implementation is not unrestricted four-player minimax, perfect-information search on sampled hands, or a full AlphaZero port. Those choices would introduce the wrong opponent assumptions or additional information-handling and compute problems before we have a useful baseline.

Approximately two million parameters remains the first neural experiment. More thinking time should buy better comparisons, additional plausible scenarios and selective depth. It must not merely repeat the same narrow plan against the same imagined opponents. Whether this beats a larger network without search remains an experimental question.

Daiyu already supports four players, but some search values are approximated from a root scalar; replace those paths with explicit four-seat predictions. Adapt its candidate scorer and batched evaluation loop to our complete-turn planner and planned rules host, subject to the host's verification gates. OpenSpiel is a companion reference, not a replacement rules engine. Its ordinary n-player MCTS supplies backup mechanics; its ISMCTS supplies information-state/resampling hooks for a separately tested comparison branch. S2 remains the first practical target, and a full OpenSpiel adapter is optional laboratory work rather than a deployment dependency.

## Why the search should follow turns and observations

The inspected `TurnFlow.EndTurn` replenishes cards after the player's actions. `MaxActionsCount` gives one action in the first Canal round and two otherwise. Within an ordinary current turn, choices and their effects are predominantly determined by the public board and the acting hand. Scout supplies specified wild-card types; it does not allow planning the second action with an imagined ordinary end-turn draw. These properties make explicit current-turn planning an attractive first layer.

`PlayersTrack.GetNewPlayersOrder` orders players by money spent in the round, and `GameFlow.EndRound` applies the transition. A forecast that gives each opponent exactly one turn is wrong at some round boundaries. In an ordinary four-player game, there can be zero through six intervening opponent turns before the root player's next turn: for example, last-to-first versus first-to-last across a round boundary. This is a consequence of the inspected scheduling code, not a measured strength improvement.

The first planning horizon is therefore **the root player's next ordinary decision**, or game end if earlier. Resolve draws, income, debt choices, changed order and era transitions along the way. If the root has an intervening special decision, such as liquidation, give its permitted observation to its continuation policy. Never evaluate an unfinished resource-payment phase as though it were an ordinary settled position.

Before relying on the simulator, address the relevant defects in the [rules audit](../rules-audit/BUG_FIX_PLAN.md). In particular, legality/resource selection, scoring, identity collisions and reconstruction can change search conclusions. Stronger search can actively discover an engine exploit. Training, search and live execution must agree on one explicit rules version; correcting only the training environment would create a different game from the one the user plays. Any shared rules fixes need their own implementation and regression checks. This document installs none.

## 1. Represent actions, turns and future decisions separately

| Level | Meaning | Example |
| --- | --- | --- |
| Choice | One required decision within an action | Which card, coal source, merchant, or development tile |
| Action plan | Enough ordered choices to finish one legal action | A network action including its optional second rail and all resource payments |
| Turn plan | The player's remaining one or two ordinary actions | Loan then build; develop then build; network then sell |
| Continuation policy | Decisions made later from observations actually available then | Change investment after an opponent uses the last useful beer |

Generate turns through the complete phase-aware legal interface. After the first action, generate the second from the actual resulting state, remaining cards and resources. A fixed menu of action-type pairs alone is not enough.

Keep a fast, fully validated incumbent turn plan available before expensive comparison. A simple legal heuristic or policy supplies this initial plan. At execution time commit one action, refresh the observation and validate the continuation. Resource choices within that action do not consume a second action slot.

### Avoid pruning the first half of a good plan

A loan or development may look weak in isolation while enabling an excellent second action. The candidate generator should reserve coverage for legal enabling actions, supplement policy proposals with several domain-informed combinations, and sample some alternatives outside the initial top scores. These are proposal mechanisms, not a fixed strategic script.

Use a bounded beam initially and progressively widen it when time remains. Explore both action orderings when legal and strategically distinct. Preserve discard opportunity cost, resource ownership, merchant choice, spending and changed turn order. Do not deduplicate plans merely because the visible buildings match: remaining cards, resources, history and later opportunities may differ.

All generated choices must be legal, but a bounded search cannot examine all legal turns. Report candidate coverage separately from rules completeness. Compare with exhaustive enumeration on deliberately small fixtures to find important combinations the proposal mechanism misses.

## 2. Compare complete plans under uncertainty

For each search request, construct a belief from that seat's permitted history and observation. Here a belief means a distribution over plausible unseen cards and opponent behavior, not access to their actual private state.

For each sampled scenario:

1. Construct a hypothetical rules state consistent with the root observation and public constraints.
2. Sample a plausible opponent-policy identity for each other seat; keep it stable during this rollout.
3. Apply a candidate complete root turn, including its exact cards, costs and source choices.
4. Advance using the real turn scheduler. Each opponent selects actions through its own masked observation and sampled policy. Do not run another full root search recursively for every opponent action.
5. Stop at the root's next ordinary decision or at game end. At a nonterminal leaf, evaluate the root's observation with the value network. At game end, use the declared terminal utility.

Average across scenarios to estimate the plan's utility. Compare plans against matched initial scenarios where possible, so one candidate is not favored merely because it drew easier opponents or better future cards. Coupling must preserve each candidate's correct marginal distribution: different draw counts or action sequences are still resolved by the rules. Use a separate stream of fresh scenarios to check the finalists and reduce selection of a plan that was lucky on the search samples.

The root cannot choose a different current second action for every sampled hidden hand. Its whole current-turn plan must be selected using information available now. In compact form, compare `max_plan mean_scenario utility(plan, scenario)`, not `mean_scenario max_plan utility(plan, scenario)`. Future decisions may differ after genuinely different observations, such as newly drawn cards or public opponent moves.

### Root allocation and stopping

Begin with equal pilot evaluation of a diverse candidate set. Give promising or uncertain candidates further samples, while retaining a proposal/exploration quota. Compare uniform allocation with an anytime sequential-halving-style scheduler; do not assume a particular bandit formula is best. An interruptible selection method is useful because the player may press Play Now. [Anytime Sequential Halving](https://arxiv.org/abs/2411.07171) is a relevant algorithm reference, not a Brass performance result.

When candidates are introduced later, evaluate them on fresh matched batches before comparing them with heavily sampled incumbents. Log uncertainty from scenarios separately from disagreement between opponent models and value-model error. More samples reduce sampling noise; they do not repair a biased value function or systematically wrong opponent assumptions.

Early exit is permitted when there is only one distinct plan, a leader remains stable across batches, or several well-tested plans are close enough for the chosen personality/difficulty band. These are calibrated practical stopping rules. Correlated scenarios, adaptive candidate selection and neural bias mean ordinary standard-error bars do not certify optimality.

## 3. Keep hidden information hidden throughout search

The boundary must cover the whole planner, not only the network input. The initial face-down discard is an explicit trap: the engine stores its identity in a player's discard collection. Keep that identity unknown, including for its nominal owner, unless later rules legitimately reveal it. Model those cards as an unknown allocation until the era transition; do not silently remove their known identities from the unseen-card pool. The publisher's [rulebook](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281), also retained locally for the audit, describes the face-down setup and later reshuffle.

The sampler must account for card multiplicities, own visible hand, observable discards, hidden allocations, deck and hand counts, wild-card supply, and era-specific reshuffling. If public history is unavailable after loading, retain uncertainty rather than reconstructing it from private events. Do not carry old per-opponent hand assignments across a shuffle as though they remain valid.

Start with a distribution satisfying hard information constraints. Treat an opponent's failure to take an apparent opportunity as soft evidence at most: human and weaker AI players can overlook it. Learned hand inference or Bayesian weighting by behavior is a later experiment. Retain a broad component to avoid overconfidence; if a belief update loses all support, rebuild from valid hard constraints and report the approximation.

Use independent search randomness, never the actual game seed or future deck. A live full-state checkpoint belongs to the trusted validation/recovery layer. Search workers receive a player view and build hypothetical worlds; they do not receive the real private checkpoint.

### A masked policy alone does not make a search fair

An opponent tree can learn from values that depend on the root's private hand even when the opponent's neural policy is masked. The first planner avoids this by sampling opponent actions from fixed observation-based policies, without optimizing their branches using root-private knowledge.

Each rollout policy must be invariant to changes in information its actor cannot know. The leaf evaluator uses the root's permitted observation at that leaf, including its own cards if they have been legitimately drawn by that point. It must not use the actual simulated opponent hands as extra value inputs. The four-seat value vector is an estimate from one observer's information, not four private views combined.

The information-set MCTS literature describes problems with solving each hidden-state sample as a perfect-information game, including choosing different actions before the player can distinguish those states. [Cowling, Powley and Whitehouse](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) provide the foundational discussion. Work on [re-determinizing search in Hanabi](https://arxiv.org/abs/1902.06075) also demonstrates opponent-model information leakage. Hanabi is cooperative and has different visibility rules; its algorithm should not be transplanted without analysis.

**Required tests:** changing hidden engine assignments while preserving the actor's full allowed history cannot change its observation, root candidate set, priors or fixed-work search result. Also test each simulated opponent independently: preserve that opponent's view, change the root's hidden hand, and require unchanged action probabilities. Include caches, candidate ordering, opaque IDs, face-down discards, save/load and era changes in these tests. Parallel/GPU numerical nondeterminism requires controlled deterministic test settings or a declared tolerance; wall-clock cutoffs are not suitable for exact invariance tests.

## 4. Model three independent opponents

Use a pool of fast observation-based policies and styles to model plausible replies. Do not grant the planner the hidden policy checkpoint or internal random state used by another seat. A displayed personality may be treated as public; otherwise infer broad preferences only from permitted history. Freeze the mixture used within a root comparison so candidate estimates remain comparable. Resample behavioral uncertainty across scenarios rather than switching a sampled player's personality on every action.

Public actions can later update beliefs about opponent tendencies, but a few moves are insufficient evidence for a precise model. Begin with a conservative mixture, then test history-based adaptation against both familiar and unfamiliar opponents. Do not assume the human makes the same choices as the training population.

Each seat has its own terminal utility. A useful initial comparison is final placement with an explicit first-place bonus versus first-place probability alone. Auxiliary final-VP prediction can support learning; do not automatically reward building, borrowing or supplying resources regardless of eventual benefit. Tie handling must follow the selected rules version. Keep utility weights fixed and recorded within an experiment.

Predict outcomes for all four seats but select actions for the acting seat. There is no alternating two-player value sign and no sum of the three bots' rewards. The baseline criterion averages over plausible opponent responses. A minimum over everything the opponents could jointly do would create an unnecessarily pessimistic adversarial team model. Tail-risk preferences may vary modestly by personality and must be evaluated for both strength and enjoyable play.

Use stable-seat vectors of predicted win share, placement and auxiliary VP, learned from one permitted observer view. A root win share of 0.40 does not imply that every other player has 0.20. Upgrade the model, training labels, inference protocol and search backup together. OpenSpiel's [n-player MCTS implementation](https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/mcts.py) is the actor-specific backup reference: a decision by player `p` uses that player's component of the outcome, including when the same player acts again. Verify this on four-seat fixtures before integrating hidden information.

## 5. Add depth only when its benefit is measurable

| Search level | Mechanism | What it establishes |
| --- | --- | --- |
| S0 | Fast legal policy/heuristic, action by action | Reliable incumbent and no-search baseline |
| S1 | Generate current-turn combinations and evaluate their settled afterstates without reading hypothetical hidden cards | Value of planning both actions together |
| S2 — first target | Compare complete turns through masked-policy rollouts to the root's next turn | Value of resource competition, changed order and plausible responses |
| S3 | Selective search over the root's future action-observation histories; opponents remain sampled policies | Whether additional contingent planning is worth its cost |

For an early non-neural implementation, use the evaluator and tactical profiles in [LEGACY_HARD_AI_PLAN.md](LEGACY_HARD_AI_PLAN.md). Its L1/L2/L3 modes map to S0/S1/S2. Keep the same observation restrictions, complete-turn candidates and deadline accounting; L3 rollout opponents use the fast L1 policies. Freeze heuristic/profile versions for comparisons and report historical stock, guarded stock and strategic upgrades separately.

S2 is the first intended practical search system. S3 is an experiment with a release gate, not a prerequisite for a useful opponent.

Add an OpenSpiel ISMCTS comparison alongside this progression, after the common host and information contracts work. Implement stable legal-action mapping, isolated clones, four-seat returns, legitimate information histories, belief resampling and explicit chance boundaries before claiming full-game compatibility. Its evaluator and bot callbacks need masking; information-set keys alone do not ensure that opponent tree statistics ignore the root's private cards. Run the counterfactual opponent-policy tests against the entire selection procedure. A variant that fails remains a diagnostic and is excluded from fair strength comparisons. The source-to-adapter checklist is in [REFERENCE_INTEGRATION.md](REFERENCE_INTEGRATION.md).

For S3, represent root decision nodes by the root's allowed action-observation history, not by a complete sampled world. New root draws and public moves create branches only when observed. Sample hidden worlds at the root and advance opponent policies between root decisions. This is a POMCP-inspired treatment of opponents as part of a sampled environment; [Silver and Veness](https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf) provide the planning reference. Any benefit here is conditional on our approximate belief and opponent models, not a multiplayer equilibrium guarantee.

Use policy priors and progressive expansion. First compare reaching the next root turn against selectively reaching a second future root turn. Extend important near-term era/scoring/debt transitions when feasible, rather than truncating immediately before their consequences. Under a strict deadline, evaluate only settled states with a phase-aware value model and record the actual horizon. Never assume more depth is automatically more accurate: weak opponent policies or a biased value model can make deeper search worse.

Large branching from hands and public action histories can make deeper trees sparse. Start with exact information-history keys for correctness; benchmark any later observation abstraction separately. Do not reuse arbitrary full-state transposition tables, merge different private observations, or combine values across seats/model versions just to increase cache hits.

Full opponent lookahead, equilibrium solvers, learned simulators and a second large decision network remain outside the initial scope. The existing rules simulator, masked rollout policies and one compact model are sufficient to test the proposed benefit.

## 6. Allocate time to a whole turn

Keep one deadline across candidate generation, scenario construction, inference, comparison, validation and the second-action continuation. Start warm worker processes and load model weights once. Do not create a process or reload the model for each hypothetical branch.

Most strategic comparison occurs before the first committed action. Retain its best legal second action, revalidate against the actual result, and spend only the remaining budget refining it. Reserve measured time for validation/execution and a second-action fallback; do not spend the complete allowance on planning the first action.

| Situation | Preferred use of additional compute |
| --- | --- |
| Only one useful legal continuation | Return early |
| Good first actions require setup | Broaden two-action combinations before adding depth |
| Candidate ranking changes across hidden hands | Sample additional consistent worlds |
| Candidate ranking changes across opponent styles | Broaden plausible opponent replies and test model calibration |
| Few credible plans, stable scenario results | Compare selective additional root-turn depth |
| Imminent resource race, new order, era scoring or debt | Resolve the relevant transition before relying on a coarse leaf value |
| Several plans remain close and well evaluated | Select using personality and stop; extra delay may add little enjoyment |

Retain Easy 1–2 s, Medium 3–5 s and Hard 8–12 s as initial whole-turn caps. Strength and thinking speed remain separate settings. Also benchmark an optional 20–30 s cap for users who prefer long deliberation; do not ship it as stronger unless it wins the comparison. Three 30-second AI turns can mean 90 seconds of thinking between human turns, plus animations. Profile actual wait distributions rather than assuming every turn consumes its cap.

Expose Play Now once a complete incumbent is available. Cancellation must stop scheduling work and reject late responses; it must not cause a partial action, double confirmation, or another full budget for action two. Background pondering during the human's turn is deferred: it complicates resource use, observation changes and honest budget comparisons.

### Laptop execution

Use a small persistent CPU worker pool, each with one active rules context and correctly bound static references. Several queued search scenarios can run sequentially in each worker. Measure whether IPC or reconstruction dominates before adding workers. Reserve resources for Unity and avoid nested CPU thread pools that oversubscribe the laptop.

For scale, 24 candidate turns times 16 scenarios times six intervening opponent turns times two ordinary actions is 4,608 simulated opponent actions, before compound choices, root generation or leaf evaluation. This is an illustrative workload, not a measured throughput target. It explains why pilot sampling, selective allocation and fast rollout policies matter more than a blanket promise to examine every combination.

Batch candidate scoring and ready value evaluations with bounded queue waits. Compare batch limits such as 1, 8 and 32 and worker counts such as 1, 2 and 4 as experimental settings, not throughput promises. CPU inference may win for small batches; Intel GPU inference may help when useful batches form. All variants use the same information boundary and complete-turn latency measurement.

Reuse canonical actor observations and valid deterministic prefixes where proven equivalent. Cache keys include observer, private observation/history, phase, style/intent, relevant opponent-mixture version, rules/schema/model version and budget/search settings where necessary. Public board equality alone is not enough. Do not share one seat's private search tree with another; shared model weights are fine.

## 7. Make planning support personality

Search should help a Supplier pursue a sensible supply plan and a Networker exploit useful connections, while still evaluating their alternatives. Preserve an explicit soft intent across turns, such as an industry sequence or network area. Train with that context and save it per seat. Reconsider it when conditions change; a personality does not justify continuing a clearly failed investment.

Separate estimated game quality from bounded style preferences. After adequate comparison, form a set of credible near-best turns, then select within it using persistent style, measured risk preference and modest seeded variation. Define the acceptable estimated utility gap separately for each difficulty and calibrate it through games; there is no universal numerical threshold before the value model is calibrated.

Independent seeds provide variety, but random blunders are not the main mechanism. Maintain exploration during training and stylistic variation during play. Require style diversity to survive additional search: if all Hard personalities converge to indistinguishable choices, adjust their proposals, training data or near-best selection instead of simply adding more randomness.

The three bots must not share hands, coordinate resource denial, target the human by identity or use a team reward. Normal self-interested competition and mutually beneficial resource use remain valid Brass behavior.

## 8. Train the model that search actually needs

Train the policy to propose useful action continuations and complete turns, including setup moves whose benefit is delayed. Train the value network on root-observable settled states near the rollout horizons used in deployment, with four-seat outcome/score targets. Include first-Canal-turn, era boundaries, exhausted decks and scarce-resource positions. Record rule and feature versions so incompatible samples are not mixed.

Start the model/training scaffold from Daiyu's separate encoders, candidate scorer and batched search interface; initialize new weights under our schema. Override two-player example configurations and replace scalar value aggregation throughout. Keep valid losing and unconventional games in the population learner. Upstream strategy filters such as excluding an early network or repeated loan must not become hard constraints on our personalities or training distribution.

Search can generate stronger examples on the lab server, then distill them into the compact laptop policy. Keep lab teachers subject to the same information restrictions. Occasionally use longer teacher search, but retain many diverse complete games rather than spending all training compute optimizing a few positions. [Expert Iteration](https://arxiv.org/abs/1705.08439) motivates learning from search; these Brass-specific choices remain experiments.

Root comparison targets need care. An elimination scheduler's raw visit counts reflect sample allocation as well as quality, so they are not automatically the desired policy distribution. Initially use finalist recommendations or value-based targets from fresh confirmation samples, with explicit confidence/coverage metadata. Do not label unexamined legal alternatives as bad. Group targets by first action without rewarding an action merely for having more enumerated completions; also train its selected second-action continuation.

Record permitted observation/history, style/intent, candidate turns, legal choice masks, selection and confirmation sample counts, means, uncertainty diagnostics, horizon, opponent-mixture ID, search time and terminal outcome. Keep raw private environment state in separate trusted debugging records. Split whole games/setup seeds between training and evaluation, rather than splitting correlated positions from one game.

Continue population training against varied styles and historical agents. Evaluate both the fast policy used for rollout and the complete policy-plus-search system. A rollout policy that consistently predicts unrealistic replies can limit the planner even if its own isolated win rate seems adequate.

## 9. Experiments and release gates

Run experiments in stages, not one enormous parameter sweep. Freeze rules, utility, training-data exposure, opponent mixture and personality settings for each comparison unless that is the variable under study. Record training compute and learning progress as well as deployment cost, so an undertrained larger model is not mistaken for evidence against larger models generally.

| Question | Controlled comparison | Evidence to collect |
| --- | --- | --- |
| Does planning both actions help? | S0 versus S1 with the same model | Setup/completion fixtures and held-out four-player results |
| Do opponent forecasts help? | S1 versus S2 | Rank/score/win changes, resource-race errors and order-boundary decisions |
| Is depth worth its cost? | S2 versus S3 at equal whole-turn caps | Match results, searched horizons, value error and actual wait |
| Does the OpenSpiel comparison help? | Validated ISMCTS adapter versus S2, with the same model, rules, utility and time cap | Four-player performance, information-invariance results and adapter overhead; exclude variants failing fairness |
| Would a larger model be better? | About 0.5M, 2M and 8M parameters, with and without search | Equal-budget strength plus parameter/inference memory and latency |
| Where should extra time go? | More candidate turns, more scenarios, broader opponent models or more depth | Individual ablations before combined tuning |
| Does the hardware backend help? | CPU versus Intel GPU, then bounded batching/workers | End-to-end latency while Unity runs, sustained behavior and responsiveness |
| Is diversity preserved? | Same difficulty, varied styles; short versus long search | Distinct viable plans/openings, strength bands and human preference |

First compare complete-turn caps around 1, 3 and 10 seconds on the laptop; investigate 30 seconds only after the useful shorter configuration is established. Use a richer pool of opponents than stock Hard, rotate all four seats, and use independent held-out seed groups. Record cold-start separately but include required planning and validation work in normal-turn timing. A server-only advantage does not establish a laptop advantage.

Report uncertainty by independent game/setup groups. Game-level performance intervals and approximate within-search uncertainty answer different questions. Long-search decisions are not an oracle; do not claim a shorter policy is wrong solely because it disagrees with one longer run.

The release gates remain: no unresolved illegal choices/state corruption/information leakage, every shipped Easy personality measurably above stock Hard on the declared rules benchmark, calibrated aggregate difficulty ordering, and enjoyable human sessions. Record stock crashes or illegal legacy moves explicitly; do not count a crashed opponent as a clean strategic victory. Where shared rule fixes alter stock behavior, label that benchmark separately from an unmodified-game comparison.

## 10. Implementation packages and targeted tests

Proposed modules, not existing files:

| Module | Responsibility |
| --- | --- |
| `TurnPlanGenerator` | Phase-aware candidates, enabling combinations, widening and complete incumbents |
| `PlayerObservation` / `BeliefSampler` | Allowed history, hidden-card constraints, sampled worlds and era reset |
| `OpponentPolicyPool` | Masked policies, stable scenario identities and conservative mixtures |
| `ScenarioEvaluator` | Real scheduler, settled horizons, per-observer value inputs and four-seat utility |
| `AnytimeSearchScheduler` | Pilot batches, comparison, fresh confirmation and one turn deadline |
| `DecisionCache` / `SearchReport` | Correct context keys, telemetry, reproducibility and stale-result rejection |
| `BrassSpielGame` / `BrassSpielState` / masked bot wrapper | Optional laboratory adapter over the same rules host; chance, information-state, action-ID and four-seat return contracts |

Targeted checks in addition to the legal-interface and game-bridge tests:

1. An initially unattractive loan/development survives proposal pruning when it enables a good legal second action.
2. First Canal round has one action; a double rail action still consumes one action; ordinary hand replenishment occurs only after the turn.
3. The same current-turn plan is evaluated across hidden scenarios; unrevealed hands cannot select different second actions.
4. Last-to-first and first-to-last schedules reach the correct next root decision, including repeated opponent turns.
5. Root and every rollout opponent pass hidden-state invariance, including face-down setup discards and cache reuse.
6. A simulated new hand affects future choices only after its draw; beliefs reset correctly across the era shuffle.
7. Near-era/endgame/debt horizons produce the same settled consequences as a full verified trace.
8. Play Now, deadline expiry and worker failure always return a legal incumbent or a defined safe stop; no second budget or duplicate confirmation.
9. Repeated comparisons with fixed scenario streams reproduce results in deterministic test mode; new candidates receive adequate evaluation.
10. Personality/seat changes cannot reuse an incompatible private tree; style remains measurable at longer search budgets.
11. Reordering players preserves stable seat identity, correctly maps all four value outputs, and never introduces a two-player value sign flip.
12. Daiyu-derived model/loss/replay/export paths agree on four-seat targets; no equal redistribution of non-root win mass or two-player example configuration survives in production training.
13. OpenSpiel callbacks never disclose private chance outcomes; clones, semantic action IDs and resampled information states satisfy the shared contract before full-game comparisons.

Deliver S0/S1 and instrumentation first, S2 next, and S3 only after its fixed-budget comparison succeeds. This makes the extra thinking time an explicitly testable feature rather than an assumption about model size or tree depth.
