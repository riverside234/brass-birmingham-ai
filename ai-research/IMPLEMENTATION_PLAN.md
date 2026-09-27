# Four-player AI implementation plan

27 September 2026. Proposed work, grounded in the local assembly inspection and the headless feasibility tests. This document does not report a trained model or an installed replacement AI.

## Recommendation and scope

Build a new decision policy, reuse the installed rules engine, and train through search and a diverse population of four-player opponents. The original Hard AI is a handwritten priority system: there are no neural weights to fine-tune. Keep it as a benchmark and occasional training opponent. Do not restrict the new policy to choosing between the old AI's rule outputs.

The starting model remains a shared policy/value network of approximately two million parameters, with a state encoder, candidate-choice encoder, four-seat outcome estimates, and a personality input. Compare smaller and larger models at the same laptop thinking budget after the environment is reliable. All training and evaluation games for this project use four players.

The deployment target is offline play on the Core Ultra 5 135U laptop. CPU rules/search workers and optional Intel GPU inference are separate from RTX training on the lab server. The product goal is varied, coherent, fair opponents with Easy, Medium, and Hard settings. New Easy beating stock Hard is a release criterion to demonstrate, not a promised consequence of using machine learning.

## What the code already establishes

| Finding | Consequence for implementation |
| --- | --- |
| `IAiBehavior.GetNextAIMove(BrassApplicationGame)` returns an executable action; the factory constructs behaviors for new and loaded games. | There is a concrete integration point. |
| `PlayAiTurnUseCase.PlayAITurn` synchronously calls the behavior, executes it, and confirms each action. `PlayGamePresenter.StartAITurn` calls this loop after displaying the AI view. | A factory replacement is enough for a quick prototype, but longer search also needs an asynchronous turn coordinator to keep the interface responsive. Exactly one component must confirm an action. |
| Fifteen stock-AI games completed in the existing probe, including five four-player games. | The installed rules engine can run outside Unity on the tested laptop runtime. This is not full rule coverage or a learning benchmark. |
| Build/network helpers automatically choose resource sources; the network helper can automatically choose a second link. | Existing AI helpers cannot serve as an exhaustive legal-move interface. |
| The rules-game copy constructor is empty; the application-game copy constructor shares its underlying game. | Neither is a usable simulation clone. |
| `BoardHelper` and `DevelopMerchantBonus.Instance` have mutable static state. | Concurrent games and search branches require isolation and deliberate context binding. |
| `CancelCurrentAction` resets action bookkeeping; it does not reverse previously applied events. | Never use cancellation as transaction rollback or attempt a fallback after a partially applied move. |
| Debt handling initializes dictionaries outside its event handlers. | Replaying raw events alone is not yet a complete restore solution, particularly during special phases. |
| The engine exposes all hands and the actual deck. | The decision interface must enforce a separate, explicit information boundary. |

These observations come from local files under `decompiled`; see `REVIEW.md` for the original evidence and executable smoke results. No debt phase occurred in those smoke matches, so debt tests are a specific outstanding requirement.

## 1. Establish the environment and state restoration

**Deliverable:** a small C# rules host using the installed assembly, with a versioned protocol for training and evaluation. Python controls matches and model training; C# applies the game's rules. Do not rewrite the game engine initially.

Proposed operations:

| Operation | Contract |
| --- | --- |
| `ResetFourPlayer(seed, setup)` | Construct a legal four-player game and advance to the first decision. The seed stays inside the trusted environment. |
| `GetDecisionContext()` | Return the actual decision-maker, phase, actions remaining, and revision. A debt decision may belong to a different player from the normal turn owner. |
| `Observe(player)` | Return only that player's permitted information. |
| `LegalChoices(context)` | Return legal next choices, including continuation/finish choices where appropriate. |
| `ApplyChoice(choice)` | Advance a decision inside an isolated simulation; expose whether more choices are required. |
| `ValidateAndApplyAction(plan)` | Validate a complete ordinary action and apply it once at the appropriate boundary. |
| `AdvanceAutomaticPhases()` | Handle draws, turn/round transitions, income, era changes and scoring until another decision or game end. |
| `Checkpoint / Restore / Replay` | Reproduce trusted engine state for validation, debugging and simulation. |
| `GetResult()` | Return final scores, official ordering/tiebreak results, and per-seat learning targets. |

Keep two representations separate: a private **engine checkpoint**, containing the full state needed to reproduce the environment, and a **player observation**, suitable for the policy and search. A full checkpoint must never become the model input or the runtime search request.

Start with one active game per worker process. Build reconstruction from typed commands/events and explicit phase state, including current action progress, pending resource requirements, merchant bonuses and debt dictionaries. Copy event payloads instead of sharing mutable event objects. Audit all relevant statics, then bind them to the active reconstructed game. Use separate worker processes for simultaneous branches until an alternative is verified. Do not create search clones inside the live Unity process while static references can interfere with the live game.

Correctness comes before fast copying. Once reconstruction works, measure replay and snapshot costs; add checkpoints or a compact simulation representation only if profiling justifies it. Search also needs a constructor for **sampled hypothetical worlds** from permitted player information, independently of the real hidden state.

**Exit checks:** replay the same four-player command trace to identical canonical states; restoring and applying a branch leaves the source untouched; isolated workers do not affect each other; restore works at action, bonus, debt and era boundaries. Compare representative complete traces with Unity's actual runtime, because the successful .NET headless probe does not establish Mono/.NET or lab-OS equivalence. Include explicit setup/card data in reproducibility records rather than assuming a numeric seed means identical shuffling on every runtime.

## 2. Implement every legal decision, including action continuations

**Deliverable:** `LegalChoiceProvider`, typed action plans and an executor, shared by training, search and the game adapter.

Represent an action as a sequence of legal decisions. This avoids constructing the full Cartesian product of cards, locations, resources and continuation choices before the policy can make progress. A complete plan contains all decisions required to finish one ordinary action. Search may generate complete candidate plans lazily from that decision tree. This refines the earlier complete-candidate model proposal; the same state/choice scorer can guide both partial choices and completed plans.

| Action or phase | Choices and edge cases to cover |
| --- | --- |
| Build | Card from the acting hand; city/slot; industry and available tile; overbuilding; exact coal/iron sources and resource consumption order; market access and affordability. |
| Network | First link; resource payment; optional second rail link or stopping; coal and beer sources for each required payment; connectivity changes caused by the first link. |
| Sell | Eligible industries and their order; reachable merchants; beer sources; merchant rewards; legal development bonus and completion; when to stop selling. |
| Develop | One or two eligible tiles, chosen in sequence; iron sources/costs; stopping after one; unavailable or undevelopable tiles. |
| Loan and pass | Legal card discard and the action's own restrictions and effects. Passing is an explicit legal move, never a silent response to an adapter error. |
| Scout | Three distinct cards from the original hand. The engine splits this into the normal one-card discard and a two-card exchange; also check wild-card restrictions and supply. |
| Forced liquidation | Correct player and ownership; eligible industries; remaining debt; completion with no industries left; resulting VP loss. |
| Turn/round/era transitions | First-round one-action rule, later two-action turns, card replenishment at the correct point, spending-based turn order, income, scoring, removal and era setup. |

Use the engine's action APIs and relevant UI preconditions, but do not treat exceptions as a complete legality oracle: some lower-level methods assume the caller already checked ownership, hand membership or a valid action phase. Inspect and enforce those preconditions in the adapter. Do not call an old helper that silently picks another resource or link after the model has selected a plan.

Give board elements stable semantic identifiers. Keep transport handles separate from model features. Equivalent duplicate cards or choices may be grouped only after confirming that doing so preserves every strategically distinct resulting state.

The legal provider must represent every legal alternative. Search is allowed to examine only a subset within its time budget, but that pruning belongs to search and must not be confused with the rules interface. Use progressive expansion to control large branching factors, retaining a way to explore alternatives outside the initial preferred candidates.

**Exit checks:** hand-constructed rule fixtures for every row above; exhaustive enumeration on small constrained states; legal alternatives cross-checked with the human UI and rule preconditions; randomized four-player games checking progress and state invariants. Check cards, cash, resources, ownership, action counts and terminal scoring. Invalid plans must be rejected without changing the source. Detect dead ends and duplicate confirmations explicitly instead of masking them with passes.

## 3. Enforce the information boundary

**Deliverable:** an allowlisted observation schema and a hidden-state sampler with automated information-leak tests.

Include public board/layout, links, industries and resources, public player tracks and available tiles, permitted public history, turn order, era, decision phase, actions remaining, visible counts, and the acting player's hand. Verify public fields against what a human is allowed to see in this client. Retain card/action history only when it is legitimately visible or inferable.

Exclude other hands, exact future deck order, private setup randomness, the real RNG seed/state and hidden event payloads. Do not encode engine object IDs, allocation order or internal cache state as model features. Legal-choice ordering and metadata must also avoid leaking hidden state.

Runtime search receives the permitted observation and public rules context. It samples plausible unseen cards and future draws consistent with that information, including card conservation, visible discards, setup exclusions and hand counts. It must not start from the real opponent hands or the real future deck and merely hide them from the policy.

During a rollout, each simulated player decides from its own permitted observation of that sampled world. When evaluating future choices for the root player, do not let different unseen card assignments create access to information that player has not learned. Group search statistics by the player's information state; reveal sampled draws only at the real rules boundary. Begin with simple rollout evaluation, then add information-set tree search once its behavior can be tested. [Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) provides a reference for this distinction; it does not establish strength for Brass.

**Exit checks:** construct two engine states with the same public situation and acting hand but different opponents' cards/deck orders. Observations, root legal choices and policy outputs must match. With an independent fixed search seed and fixed simulation count, the search result must also match. Inspect serialized requests and cache keys. Repeat for all seats and special phases. Environment rewards may reflect the eventual real outcome, but observations and action targets must come from permitted information.

## 4. Deliver a playable baseline and reliable game bridge

**Deliverable:** a locally playable replacement using the complete decision interface, initially with a simple new heuristic policy; no neural training is required to test integration.

Implement the `IAiBehavior` adapter and intercept the behavior factory. Verify runtime patching compatibility before relying on a particular loader. For a first controlled build, configuration can map the existing difficulty selection to the replacement; menu and personality controls can follow after the bridge works.

Longer thinking cannot be a blocking call inside the existing synchronous loop. Add an asynchronous turn coordinator: copy the permitted decision data on the game thread, request a decision from a local worker, keep the UI responsive, then validate and apply the result on the game thread. Do not let a background thread mutate Unity objects or the live rules engine.

Messages carry protocol/model/schema versions, game identity, revision, actor, phase, request ID and deadline. Discard stale responses after loading, restarting or advancing the game. A time limit returns the best complete validated plan found so far; provide a Play Now control once available. Preserve exactly one confirmation per ordinary action and the game's separate end-turn/end-round lifecycle. Forced liquidation needs its own decision hook.

Search may plan both ordinary actions together, but commit one complete action at a time. Refresh the real observation and revalidate or revise the second action using the remaining whole-turn budget. Keep decisions within a compound action in their proper phase; do not accidentally count resource selection or a second rail link as another ordinary action.

Validate a complete plan on an isolated reconstruction before live execution. Capture a tested recovery checkpoint at a safe boundary. If the worker times out or fails before live mutation, use a cached valid choice or validated fallback. If live application unexpectedly fails partway through, restore through the tested game-history/state path and synchronize the UI, or stop that turn visibly; never continue by pretending cancellation undid the move. Exercise injected failures before accepting recovery as reliable.

Keep personality/configuration and reproducible random state in a versioned sidecar keyed by game and seat if the vendor serializer cannot safely persist them. Loaded games must reconstruct the right behavior without attempting to deserialize an unsupported custom AI class. Include clean disable/uninstall behavior.

**Exit checks:** complete four-player matches with one human and three replacements; both eras; save/load between turns and at supported decision boundaries; restart/menu return during thinking; worker crash/timeout; duplicate/stale response rejection; repeated confirmation protection; bonus and debt flows. UI responsiveness must remain acceptable during the longest configured thinking time.

## 5. Establish baselines before expensive training

**Deliverable:** an automated four-player tournament runner and several independently useful opponents.

Keep stock Hard unchanged as a benchmark. Add a legal random policy for coverage testing, a straightforward new heuristic using the complete move interface, and that heuristic with bounded hidden-information-aware search. The new heuristic can favor estimated future points, affordability, resource access and useful development without being constrained to stock rule priorities. Search should evaluate two-action combinations such as loan-then-build and develop-then-build, then consider opponent responses and later turns within budget.

These baselines separate improvements from a better move interface, better search and a trained model. If a compact heuristic/search opponent already meets some play goals, retain it as an early usable opponent and a diverse training partner.

**Exit checks:** measure legal-decision throughput, snapshot cost, simulations per second, move latency and match results. Do not extrapolate neural-training time from the old bot's roughly one-second headless games. Inspect the lab OS, CPU capacity and actual training runtime when server setup begins; the RTX GPU accelerates model work, while rules generation/search may be CPU-bound.

## 6. Train new weights with search and a diverse opponent population

**Deliverable:** a reproducible four-player trainer, checkpoint pool, replay dataset and evaluation reports.

Initialize a new model. The default learning loop is:

1. Sample four seats from a mixture of recent agents, older checkpoints, distinct styles and heuristic opponents; include some stock-Hard games. Rotate seats and use varied valid setups. All-new-agent and mixed tables both matter.
2. Run games with legal masks, the correct information boundary and bounded search. Search produces an improved action distribution; the eventual game result provides value targets.
3. Train the policy to predict those search choices and the value heads to estimate outcomes for all four seats. Use extra final-score prediction as an auxiliary target if it improves learning.
4. Evaluate a frozen candidate against a held-out opponent pool and fresh seeds. Promote useful candidates without deleting diverse older ones merely because their average rating is lower.
5. Rebalance opponents toward weaknesses discovered in the matchup results, while retaining variety and a stable evaluation set.

This combines the search/learning loop of [Expert Iteration](https://arxiv.org/abs/1705.08439) with a practical population of opponents. Research on [responses to mixtures of policies](https://arxiv.org/abs/1711.00832) motivates using diverse partners to reduce overfitting. We are proposing these ideas for Brass, not claiming a proven optimum or a full game-theoretic equilibrium solver.

At the beginning, simple heuristic evaluation can bootstrap search before the new value network is informative. Optional short imitation pretraining is an experiment; prefer improved search examples and compare any stock-imitation warm start against fresh initialization under the same budget. Long training that simply copies stock Hard is not the objective. Pure random-start self-play is also an experiment, not a requirement to rediscover every useful economic concept unaided.

A masked-PPO baseline is a useful lower-complexity learning comparison if search is initially expensive. Maintain separate per-seat trajectories or a correct multi-agent formulation. Do not feed four alternating players' observations and rewards into one ordinary single-agent trajectory. During a learner's episode, freeze sampled opponent versions and retain the learner's own action/reward attribution.

Use terminal placement/win outcomes as the principal task signal, with official tie handling and score prediction for extra learning signal. Decide the final balance between winning and improving placement through evaluation. Avoid simplistic rewards that make borrowing, developing, or supplying opponents inherently wrong: these can be sound Brass decisions. At search nodes use the acting player's utility; there is no two-player sign-flip rule and no shared objective for the three bots facing a human.

Replay records contain observation/choice-schema versions, seat, permitted observation, legal candidates, search targets, policy version, style, chosen action and terminal outcomes. Keep full debug state separately in the trusted harness. Separate training, validation and final test seeds/opponents; do not train on the final acceptance suite.

**Exit checks:** learning outperforms the non-neural baselines at matched inference budgets, generalizes to unseen opponent mixtures and seeds, and avoids information leaks, invalid actions and reproducible policy collapse. Plot a four-player matchup matrix and rank/score results; a single Elo number or a stock-only win rate can hide style-specific weaknesses and nontransitive matchups.

## 7. Add personalities and calibrate difficulty

**Deliverable:** several stable styles at each difficulty, with measured strength and human play feedback.

Use one shared network plus a style input or small adapters. Initial styles: Industrialist, Networker, Supplier and Investor. Train with those preferences present, not only as a last-minute random alteration to a competitive model. Keep preferences soft, preserve plans over several actions, and allow adaptation when a plan becomes unprofitable.

Choose among several credible moves using calibrated variation. Personality preferences can break close choices, but do not require a player to make a clearly bad investment just to display its theme. A predicted quality threshold is approximate, so validate its effect in actual matches. Independent seeds and observations for each seat prevent synchronized behavior; shared weights do not imply shared objectives.

Difficulty combines a suitable policy/checkpoint, search effort and breadth of acceptable alternatives. Starting whole-turn budgets are Easy 1–2 seconds, Medium 3–5 seconds and Hard 8–12 seconds; they are proposed configurable limits, not measurements. Three Hard turns can therefore add 24–36 seconds before the human plays again, plus animations. Return early when a decision is clear and offer a separate speed cap.

Match styles into similar strength bands through tournaments rather than asserting they are equal by design. Calibrate every style's Easy above stock Hard, and confirm Medium/Hard improve across a representative mixture. Human playtests assess varied openings, coherent multi-turn plans, recovery from setbacks, plausible use of shared resources, excessive disruption, repetitive actions and acceptable waiting. Optional human move examples can later help imitate human preferences; self-play alone does not establish human-like enjoyment.

**Exit checks:** each shipped style passes the strength gate; styles show meaningful behavioral differences; diversity comes from sensible alternatives; no bot uses a team reward or targets the human simply for being human. No concealed difficulty changes are needed to meet this design.

## 8. Optimize the laptop package and run acceptance tests

**Deliverable:** a versioned offline release with model, adapter, local worker, configuration, diagnostics and rollback instructions.

Export the validated policy/value network to an appropriate local inference format. Compare CPU and Intel GPU using real game/search batches while Unity is running. Test approximately 0.5M, 2M and 8M parameters at identical wall-clock budgets; measure whole-turn strength, not just raw inference rate. The proposed 2M model is about 8 MB of float32 weights or 4 MB of float16 weights, excluding runtime, activations and search memory. Verify export and precision conversion against the training model before judging strength.

Record median and slow-tail thinking times, peak memory, batch throughput, UI smoothness and sustained laptop behavior. Limit CPU parallelism so the interface stays responsive. Keep a CPU backend even when GPU inference wins. Resolve model/schema incompatibility before starting a match, not after an AI has partly acted.

Acceptance suite:

| Area | Required evidence |
| --- | --- |
| Rules and lifecycle | Complete action/phase fixtures, deterministic replay where intended, long four-player runs, no unresolved illegal moves or deadlocks, both eras and terminal scoring. |
| Fairness | Counterfactual hidden-state invariance tests pass for observations, legal choices, policy and fixed-budget search. |
| New Easy vs stock Hard | One new Easy plus three stock Hard opponents, across held-out seeds and every seat; first-place share exceeds the symmetric 25% reference with uncertainty bounds, corroborated by rank and score margins. Repeat for every personality. |
| Broader skill | Mixed tables, new-vs-new opponents, older checkpoints and independent heuristics; difficulty ordering holds on aggregate, with matchup exceptions documented. |
| Reliability | Save/load, restarts, worker faults, late responses, safe fallbacks and recovery; no changes to original rules/gameplay data from the AI installation. |
| Play experience | Human sessions find opponents coherent and varied; waiting times fit the selected budget; no coordinated behavior against the human. |

Begin formal strength evaluation with, for example, 200 distinct held-out setups and four seat rotations per configuration, expanding if uncertainty remains too large. Analyze confidence intervals by independent setup/seed groups because seat rotations are correlated; split tied wins according to the declared metric. A 25% reference is a symmetry baseline, not a guarantee every matchup behaves transitively. Passing the stock test alone does not establish enjoyable or human-level play.

## Delivery order and immediate next milestone

1. Reliable four-player host and state restoration.
2. Complete legal decisions and strict player observations, with their tests.
3. Playable heuristic replacement and asynchronous game bridge.
4. Search baseline, measurement and lab worker setup.
5. First trained model and diverse-opponent evaluation.
6. Personality/difficulty calibration, laptop optimization and human playtests.

The first milestone is a four-player environment that can represent and replay every action, including special phases, without exposing hidden information. The first playable release can use the new heuristic while model training proceeds. Training duration and final strength cannot be estimated responsibly until legal-move generation, reconstruction, search and batched inference have been measured.

Existing background: `REVIEW.md` contains the feasibility evidence; `DESIGN.md` covers model size, Intel graphics and inspected open-source references. This plan refines their training recommendation and the representation of multi-step actions. All milestones above remain proposed work unless separately reported as implemented and tested.
