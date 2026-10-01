# Daiyu and OpenSpiel integration plan

**30 September implementation update:** M0–M5 of [Masked ISMCTS](MASKED_ISMCTS_REVISION_PLAN.md#10-implementation-order-and-component-map) are implemented. Current training uses a fresh 1,840,109-parameter MLP-ResNet, CPU/GPU batching, separate legal/padding/expansion masks and completed root-visit teachers. Package 1.4.9 switches all six v3 styles to the shared C# backend, preserving 2/5/12 and old-save 10 seconds plus resource/UI management. The new lab bank uses one second directly. Historical S1/S2/S3/R4 variants and checkpoint-import commands below are archival. Use the [current lab guide](../docs/TRAINING_RECOVERY_LAB.md) and [exit record](../docs/MASKED_ISMCTS_EXIT_CHECKS.md).

**30 September selected search revision:** [Masked ISMCTS](MASKED_ISMCTS_REVISION_PLAN.md) replaces the active S1/S2/S3 and separate R4/R5 roadmap. Daiyu remains the model/batching reference; OpenSpiel ISMCTS becomes the main search reference. Add a new immutable component/reference manifest during implementation and preserve the R0/R2 historical pins below. The new component map is `configs/references/masked-ismcts.lock.json`; historical R0 pins remain unchanged.

**Updated architecture/reference boundary:** the new main model is a freshly initialized MLP-ResNet ≤2M total parameters; the earlier dense R2 checkpoint is not imported. Later goal/context additions may reach 2.10M. The selected plan adds primary residual-learning, batched MCTS, invalid-action masking, progressive-unpruning and CPU-runtime references. Preserve Daiyu feature/scorer provenance where code is reused, and add explicit component records for required CPU batching, dynamic masks and progressive policy top-K.

27 September 2026. Selected references for the four-player AI, following the user's preference. This is a source-level implementation plan; no upstream package, model or replacement AI is installed by this document. Read with [DESIGN.md](DESIGN.md), [SEARCH_DESIGN.md](SEARCH_DESIGN.md) and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

R0 implementation is available in [R0_CONTRACTS.md](../docs/R0_CONTRACTS.md): machine-readable commit/file pins, license provenance, ten component mappings, contract `0.1.0` and four-seat semantic fixtures. Its [exit checks passed](../docs/R0_EXIT_CHECKS.md). R1 supplies runtime `0.2.0`. [R2 implements the model/training scaffold](../docs/R2_TRAINING.md), with a separate [adaptation record](../configs/references/r2-components.json) and [Daiyu license/notice](../src/brass_ai/learning/NOTICE.md). R0's historical manifests remain unchanged. R3 and Step 5 supply the legacy/neural S1/S2 interfaces; [R4/R5](../docs/R4_R5_SEARCH_LAB.md) now implement both required additional search experiments alongside current training, with independent correctness and promotion gates.

The earlier complete-turn search design remains the behavior to implement. The [unified requirements checklist](IMPLEMENTATION_PLAN.md#unified-requirements-from-the-earlier-search-design) maps every earlier recommendation to its implementation stage and required evidence. This reference plan explains how Daiyu and OpenSpiel support those requirements.

The complementary [legacy Hard optimization plan](LEGACY_HARD_AI_PLAN.md) supplies conventional scored-rule opponents and a frozen tactical bank using BGG strategy references. It shares the same rules/observation/action interfaces and S1/S2 planner. Daiyu remains the neural model/training reference; the legacy family provides early playable policies, rollout policies and diverse benchmarks.

## 1. Decisions and reference versions

**Daiyu is the primary Brass-specific starting reference. OpenSpiel supplies complementary multiplayer search mechanics, information-state contracts and a laboratory comparison harness.** Adapt selected components behind the planned Steam-game interface. Start with fresh weights, a roughly 2M-parameter model and four-player training; architecture reuse does not require copying an existing policy's behavior.

Daiyu already supports two through four players. The work is to verify and improve its four-player assumptions, rather than convert an exclusively two-player game. Its model/search includes scalar outcome estimates and approximate redistribution of non-root win probability. OpenSpiel provides n-player search examples, but its standard Python AlphaZero trainer explicitly rejects games with more than two players. Neither is a turnkey fair four-player Brass trainer. Sources: [Daiyu search][d-search], [Daiyu model][d-model], [OpenSpiel AlphaZero][o-az].

| Reference | Pinned revision inspected | Role |
| --- | --- | --- |
| [Yuiffy/brass-birmingham-daiyu-ai][d-readme] | `387840dd7a784f2c05fe561cb381e4f7d7bc590b` | First implementation reference for Brass features, candidate scoring, batched search and training workflow |
| [google-deepmind/open_spiel][o-readme] | `48401890ee9857e611678302371378175a8e4c6b` | Multiplayer backup semantics, information-set interfaces and independent algorithm comparisons |

Use these revisions for the initial adaptation; review later upstream changes deliberately. Record source path, revision, modifications and license for each reused component. The pinned licenses are [GNU AGPL v3 for Daiyu][d-license] and [Apache 2.0 for OpenSpiel][o-license]; preserve applicable notices and source requirements in any derived distribution. No third-party code is vendored in this planning update.

## 2. What to adapt, and what to replace

| Inspected source | Useful starting material | Required project adaptation |
| --- | --- | --- |
| Daiyu [`training/model.py`][d-model] | Separate state and sparse candidate encoders, shared candidate scorer, policy/value outputs | Our allowed-observation features, phase-aware choices, style/intent input, four-seat value targets and exportable operators; remeasure parameter count |
| Daiyu [`training/neural_search.py`][d-batch] | Request/evaluate/submit batching around a search session | Versioned messages to our C# host, cancellation and one whole-turn deadline; its Rust methods are not our environment API |
| Daiyu [`training/schema.py`][d-schema] | Feature dimensions and schema compatibility checks | A new project schema/signature covering observations, actions, values and rules; reject incompatible checkpoints and shards |
| Daiyu [`src/game/search.rs`][d-search] | Candidate expansion, search control and neural evaluation plumbing | Complete-turn S1/S2 planner first; replace root-relative multiplayer approximations and two-player aggregation paths |
| Daiyu [`src/game/hidden_information.rs`][d-hidden] | Card conservation and publicly acquired wild-card handling | Construct beliefs from permitted information, including unseen setup discards; do not copy the true hands/deck as our sampler's input |
| Daiyu [`docs/training-methodology.md`][d-method] | Separate data/evaluation seeds, staged experiments and explicit promotion records | Four-seat rotations and mixed tables; keep valid losing games and alternative strategies instead of importing all upstream quality filters |
| OpenSpiel [`mcts.py`][o-mcts] | N-player reward vectors and max-n-style selection/backup | Use actual acting-seat utility, handle repeated turns and chance; ordinary full-state MCTS alone is not a hidden-information solution |
| OpenSpiel [`ismcts.py`][o-ismcts] | Information-state keys, resampling callback and actor-specific returns | Brass observation/history and belief adapters; audit evaluator inputs and opponent search statistics before any fair-play claim |
| OpenSpiel [`evaluate_bots.py`][o-eval] | Terminal-return match loop with a bot per seat | Our masked bot wrapper, private-event filtering, four-seat evaluation and game-level statistics |

Consult additional Daiyu feature, legal-action and self-play modules while building each adapter, but cross-check their semantics against our rules fixtures. Reuse useful algorithms and scaffolding; do not import its web application or assume its action IDs, data files or checkpoints match the Steam version.

Daiyu's documented producer filters include rejecting games for non-positive final cash/income, a network before the first build, or immediately repeated loans. These are strategy preferences, not general rules of legality. For our population learner, reject corrupt/illegal records, retain valid weak and losing trajectories for value learning, and annotate any selectively weighted expert data. Test proposed strategy filters separately for strength and diversity. [Upstream methodology][d-method].

## 3. Keep one authoritative rules contract

The initial C# host over the installed game assembly and shared fixes is implemented and validated by R1 on Windows. R2's Python/PyTorch scaffold consumes that `0.2.0` contract and pins its qualified host build. Search and the eventual OpenSpiel adapter will use the same legal choices and isolated reconstructed states. This avoids silently training under a second set of Brass rules. Linux host and lab GPU qualification remain outstanding.

Daiyu's Rust engine is an optional later acceleration backend, not a prerequisite for the first model. Before using it for production training, map states/actions and compare legal choices, resource payments, turn order, both eras, hidden setup, final scoring and ties against independent fixtures and the selected live-game contract. Compare explicit setups and action traces, not equal integer seeds across different shuffle implementations. Document disagreements and resolve them before mixing datasets. A Rust backend must also pass restoration and information-boundary tests.

The laboratory may run OpenSpiel as a reference implementation; the laptop need only ship the validated compact model, required rules/search worker and game bridge. Do not require a Python-to-C++-to-Rust-to-C# call chain for every simulated choice. Measure IPC and simulation cost before choosing a second engine or moving hot loops. Any later backend change creates a new benchmark and compatibility gate.

## 4. Replace multiplayer value handling end to end

Use one observation of seat `observer_id` to produce:

- A score for each legal candidate choice, plus the existing plan-generation context.
- Four predicted first-place shares, indexed by stable seat ID. Train against official winners, splitting a tied win if the metric calls for it; a softmax head can enforce a total predicted share of one.
- Four expected final placements and four auxiliary final-VP estimates. Handle unresolved ties consistently with the declared metric. Score prediction is initially auxiliary, not an automatic reward for maximizing raw VP irrespective of placement.

Construct the acting seat's search utility from an explicitly versioned win/placement objective; compare first-place-only with a win/placement mixture. For example, knowing seat 0 has a 0.40 predicted win share does not justify assigning 0.20 to every other seat: the model might instead predict `[0.40, 0.35, 0.20, 0.05]`. At a decision owned by seat 1, use seat 1's utility. Consecutive actions by the same seat do not change its value sign. Three AI seats never have a combined team utility.

Upgrade the model heads, loss/targets, replay schema, inference response, search accumulation, exported runtime and reports together. Reject scalar-value checkpoints under the new signature rather than silently expanding a scalar into four values. If a relative seat encoding is used internally, map it explicitly to stable IDs before backup and test all seat rotations and spending-based reorderings.

Predicting four outcomes does not permit reading four private hands. Every prediction comes from one allowed observer view. S2 evaluates settled leaves from the root observer; opponent rollout policies use their own allowed observations. Terminal targets come from actual final results in the trusted environment. This is our proposed adaptation of the separate encoder/scorer model, not an upstream feature claim.

## 5. Use OpenSpiel for multiplayer mechanics and fair-search comparisons

The inspected [MCTS implementation][o-mcts] backs up the relevant player's component of the return vector and describes max-n-style play. This supplies a useful reference for selfish multiplayer decisions; it does not imply that opponents jointly minimize our utility. Test these mechanics in small four-seat fixtures with known payoffs, consecutive decisions and chance outcomes before connecting Brass.

Keep the search progression already selected:

| Stage | Role of the references | Gate |
| --- | --- | --- |
| S0/S1 | Daiyu-inspired candidate scorer and batching; legal policy, then both actions of the current turn | Reliable complete plans and useful enabling combinations |
| S2, first practical target | Complete-turn comparisons over beliefs and independent masked opponent policies; four-seat outcomes | Correct scheduler and information use; improvement at equal laptop time |
| OpenSpiel comparison branch | ISMCTS resampling/information-state hooks plus actor-specific value backup, using the same rules and model | Pass dedicated hidden-information tests before inclusion in fair match results |
| S3 | Earlier POMCP-inspired root action-observation tree with sampled opponent policies; independent of the OpenSpiel adapter | Outperform S2 without sacrificing fairness, personality or responsiveness |

The comparison branch is not automatically the shipping algorithm. An ISMCTS tree that conditions opponent search statistics on the root's known hand can still create an information advantage even if its neural inputs are masked. Test that an opponent's action distribution is invariant when only someone else's private cards change. Treat a failing search variant as a diagnostic, exclude it from fair strength claims, and retain S2's independent opponent policies.

Information-state grouping also prevents choosing a different current-turn continuation for each unrevealed sampled hand. Choose a single complete current-turn plan across samples; future policies may branch only after a new observation. Keep the real scheduler, fresh finalist checks, current-turn deadline and private-cache rules from [SEARCH_DESIGN.md](SEARCH_DESIGN.md).

When adapting the search loop, retain enabling first actions, the actual next-root-turn horizon, matched scenarios and adaptive allocation between breadth, sampling and depth. The horizon may contain zero to six intervening ordinary opponent turns. Confirm finalists before choosing among near-best plans using persistent personality, then execute one action and revalidate the continuation within the same deadline. These requirements apply to Daiyu-derived search and any OpenSpiel experiment.

Retain the initial approximately 2M-parameter model and the 0.5M/2M/8M comparisons with and without search at roughly 1/3/10-second whole-turn caps. Investigate 20–30 seconds separately, measuring the combined wait for all three opponents. S3 keeps the earlier POMCP-inspired future-observation approach; the OpenSpiel comparison may inform it but is not a prerequisite. Rules corrections must agree between training and live execution before any search-generated data is trusted.

## 6. Laboratory OpenSpiel adapter contract

Build a thin `BrassSpielGame` / `BrassSpielState` adapter after the host's legal choices, restoration and observations pass their gates. Start with four-seat synthetic fixtures, then frozen Brass scenarios, then complete four-player games. A partial adapter is not evidence of full-game compatibility.

| Interface | Project requirement |
| --- | --- |
| Game metadata | Exactly four players, sequential decisions, imperfect information and terminal rewards. Declare utility bounds and constant-sum/general-sum status from the configured terminal objective; do not inherit two-player zero-sum defaults. |
| `current_player()` | Actual decision owner, including liquidation; support two ordinary actions by one seat and variable order between rounds. |
| `legal_actions()` / `apply_action()` | Stable bounded semantic IDs for typed phase choices, reversible mapping to host commands and identical legality. Dense neural batch indices remain separate. No hidden-state-dependent enumeration or automatic unselected resource choices. |
| `clone()` | An isolated trusted host context with complete phase state and correctly bound statics; never the empty/shallow game copy constructors. |
| Observation / information-state methods | Current allowed view plus the specified legitimate history for the queried seat. Information-state keys preserve relevant private observations and recall; a public-board hash is insufficient. |
| `resample_from_infostate(player, rng)` | A world consistent with that player's information, produced by our belief sampler; preserve public and own known cards and allocate every unseen bucket, including face-down setup cards. |
| Chance transitions | For the full comparison adapter, expose setup/deal/reshuffle randomness incrementally, with normalized probabilities and private reveals delivered only to entitled observers. Add host pause/resume support at these boundaries; never enumerate all deck permutations or substitute the actual future deck. |
| `returns()` | Four fixed-seat terminal utilities calculated from the declared standings/tie rules. Nonterminal evaluations use the matching four-seat schema and a permitted observer. |
| Bot/evaluator boundary | The adapter may hold trusted state internally; decision code receives only `PlayerObservation`, legal choices and permitted events. Filter all callbacks, including reset and chance notifications. |

OpenSpiel's [resampling contract][o-resample] and [information-state contract][o-info] define the framework entry points. Its evaluation helper passes state objects and chance notifications to bots, so using that helper does not itself enforce privacy. Wrap it or adapt its loop; never forward the original state or a private card identity to every bot. [Evaluation helper][o-eval].

The production S2 host may keep automatic draws inside sampled-world advancement. Explicit chance nodes are extra work for the comparison adapter and must not block the first playable S1/S2 release. If that work is incomplete, label the OpenSpiel experiment as fixture-only. Do not report complete-game algorithm comparisons from a wrapper that skips chance events or misdeclares them deterministic.

## 7. Implementation sequence and acceptance

| Work package | Concrete output | Dependency / acceptance |
| --- | --- | --- |
| R0: pin and map — complete locally | Reference lock, code-origin notes, ten-component assumption map and contract `0.1.0` | Sixteen source files verified; 12 message examples and 37 unit tests pass; see the R0 exit record |
| R1: shared host contracts — complete locally for Windows | Runtime contract `0.2.0`, four-player host, legal decisions, observations, sampled worlds, results and restoration | Main plan stages 0–3 validated in .NET Framework/Unity Mono; [exit evidence and remaining platform limits](../docs/R1_EXIT_CHECKS.md). R2/R3 must target `0.2.0`; R0's `0.1.0` stays immutable |
| R2: Daiyu-based model/training scaffold — implemented locally | Candidate scorer, four-seat heads/losses, batch protocol, replay/checkpoint validation, four-player runner and lab-root CLI | Fresh weights; fixed four-player configurations; shape, stable-seat, privacy, learning/resume and CPU export checks. [Exit evidence](../docs/R2_EXIT_CHECKS.md); no strength or lab/GPU qualification claim |
| R3: first useful planner | Four-player legacy bank H0–H5 with heuristic S1 and required S2; share R1 and the agreed batching interface with R2 | Complete turns, valid deadlines, masked opponents and held-out four-player equal-time results; H5 freezes each released bank version. H4 is implemented and required; measured benefit and H5 admission remain separate gates |
| R4: OpenSpiel comparison harness — implemented additional experiment | Full four-player adapter, explicit chance worker, masked ISMCTS comparison and per-seat privacy regressions; executable dependency `open_spiel==2.0.2` separately pinned from R0 | [Exit evidence](../docs/R4_R5_EXIT_CHECKS.md); correctness before matches and equal-budget benefit before promotion; existing training is unchanged |
| R5: learning and selection — S3 implemented as additional experiment | Population training and S2 from Step 5; independent POMCP-inspired S3, one/two-future-turn comparisons and isolated S3 distillation | [Lab commands](../docs/R4_R5_SEARCH_LAB.md); held-out whole games, rotated seats, mixed opponents, calibrated difficulty and human feedback remain release requirements |

Specific regression checks: asymmetric four-seat values, all seat mappings, two actions by one player, round reordering, terminal ties, replay/export shape parity, stable action-ID mapping, independent clones, complete legal choices, sampled-card conservation, private chance callbacks and counterfactual hidden-state invariance for every seat. Test long-search teachers as well as deployment policies. Keep upstream-inspired code behind the same project tests.

Train all actual policies on four-player games. Use new/old checkpoints, heuristics and distinct styles; stock Hard remains one reference opponent. Preserve Industrialist, Networker, Supplier and Investor intent and near-best selection when adapting search. Evaluate S0/S1/S2 first with the same model; add the OpenSpiel and S3 branches after correctness gates. Hold rules, information access, utility, opponent pool and whole-turn time constant in algorithm comparisons; report training exposure and sample uncertainty.

Release criteria remain unchanged: every shipped Easy personality demonstrably above stock Hard, useful Medium/Hard ordering across mixed tables, no hidden information advantage, credible stylistic diversity and acceptable waiting on the laptop. Upstream functionality, framework popularity and model size do not establish these outcomes. R0/R1/R2 now supply the local foundation. R3 H0–H5 supplies search/legacy interfaces and evidence-gated frozen opponents; substantial training and calibrated releases follow. See the [R3 exit record](../docs/R3_EXIT_CHECKS.md).

[d-readme]: https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/README.md
[d-model]: https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/training/model.py
[d-batch]: https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/training/neural_search.py
[d-schema]: https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/training/schema.py
[d-search]: https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/src/game/search.rs
[d-hidden]: https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/src/game/hidden_information.rs
[d-method]: https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/docs/training-methodology.md
[d-license]: https://github.com/Yuiffy/brass-birmingham-daiyu-ai/blob/387840dd7a784f2c05fe561cb381e4f7d7bc590b/LICENSE
[o-readme]: https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/README.md
[o-mcts]: https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/mcts.py
[o-ismcts]: https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/ismcts.py
[o-az]: https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/alpha_zero/alpha_zero.py
[o-eval]: https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/open_spiel/python/algorithms/evaluate_bots.py
[o-resample]: https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/docs/api_reference/state_resample_from_infostate.md
[o-info]: https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/docs/api_reference/state_information_state_string.md
[o-license]: https://github.com/google-deepmind/open_spiel/blob/48401890ee9857e611678302371378175a8e4c6b/LICENSE
