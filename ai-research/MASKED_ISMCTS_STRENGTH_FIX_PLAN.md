# Improving Masked ISMCTS opponents in four player Brass

30 September 2026 investigation; implemented 1 October 2026. **Status: F0–F6 implemented and checked; game package 1.4.11 installed.** The investigation initially left 1.4.9 unchanged. The implementation now uses new profiles, bank and search contracts; the earlier frozen evidence is preserved.

The priority is to make the search compare a few credible **complete turns repeatedly**, rather than sample many unrelated first actions once. Rule-based v3 should use narrower candidates, a short settled rollout and regularized Q normalization. The MLP-ResNet branch should retain policy top-K and CPU/GPU batching, use its bounded neural values directly, and evaluate a learned-value frontier without inheriting v3's shallow rollout cap. Both remain Masked ISMCTS and share information, legality, deadline and completion contracts.

This supplements [the implemented M0–M5 revision](MASKED_ISMCTS_REVISION_PLAN.md) and [the main implementation plan](IMPLEMENTATION_PLAN.md). Its numerical settings are starting proposals, not measured optima or a promise of human-level play.

## 1. What the current code and diagnostics show

The review covers `src/BrassMaskedSearch/MaskedSearch.cs`, both C# planners, `src/brass_ai/search/ismcts.py`, `search/common.py`, and the lab legacy forest adapter. The shared C# source is the installed v3 backend; the isolated diagnostics use its frozen research worker. Native checks from the previous release establish legality/lifecycle correctness, not playing strength.

| Finding | Evidence and implication |
| --- | --- |
| Unseen actions take precedence over exploitation | C# `Decide` and Python `trajectory` choose an unseen eligible edge whenever one exists. PUCT/Q comparison runs only when that pool is empty. A high-value visited candidate cannot compete with an unseen one at that point. |
| Expansion grows before alternatives are understood | Root/interior K is 16/8, with `2*sqrt(visits)` widening, protected families and additional tail admissions. Hundreds of card/location variants remain legal. Breadth can outrun the available completed simulations. |
| The first action receives credit for different second actions | C# final selection orders complete `Outcome` plans by their first edge's visit count. The lab forest similarly assigns a first edge's statistics to each plan starting with it. Those are not evidence for a particular continuation. Python stores the last completed trace for a first action; it is not necessarily the most supported continuation. |
| A path must reach the next ordinary root turn | Current C# already searches the root's current turn, not an unrestricted whole-game tree. However, evaluating only at the root's actual next turn requires up to six intervening opponent turns. The 512-choice guard is an abort guard, not an effective strategic cutoff. |
| Heuristic values are already bounded | `LeafValues` softmaxes relative heuristic scores at temperature 20 into four shares. These are heuristic proxies, not calibrated win probabilities. Adding normalization cannot help an edge-selection branch that never consults Q. |
| Repeated work consumes the clock | Root ranking repeatedly sorts the same legal list; plan scoring rebuilds/replays a world; `LeafValues` calls `Evaluator.Evaluate` for four seats, recomputing forecasts/seat ledgers. Profiling must establish each cost before optimization. |
| Replicas do not guarantee different warm-up candidates | The C# partition index changes world seeds. The current Masked selection loop does not allocate disjoint root candidate warm-ups. Four short replicas can repeat preparation and collect little useful common evidence. |

Three stored four-player self-play positions were evaluated with the unchanged **one-second** frozen lab profile on Windows CPU: Canal round 2, Canal round 8 and Rail round 5. All three completed **zero** search paths and returned their legal incumbent. Their legal choice counts were 369, 29 and 70. This is a small diagnostic, not a lab-server performance measurement or a match-strength test. [Decision receipt](../artifacts/masked-strength/diagnosis.json).

On the same early Canal position, detached clock overrides reproduced the existing game allowances:

| Allowance | Completed simulations | Visited first actions | Revisited first actions | Root Q range |
| --- | ---: | ---: | ---: | --- |
| 2 seconds | 2 | 2 | 0 | 0.18290–0.18503 |
| 5 seconds | 7 | 7 | 0 | 0.18290–0.21686 |
| 12 seconds | 22 | 22 | 0 | 0.18290–0.32526 |

The clock overrides changed only detached requests; the lab profile remains 1,000 ms. [Game-clock receipt](../artifacts/masked-strength/game-clock-diagnosis.json). These results support prioritizing repeat visits and path cost. They do not identify every cause of the user's game losses or prove that any proposed fix improves win rate.

## 2. Relevant research and implementations

| Primary source | Relevant result or mechanism | Application and limit |
| --- | --- | --- |
| Cowling, Powley and Whitehouse, [Information Set Monte Carlo Tree Search, 2012](https://pure.york.ac.uk/portal/en/publications/information-set-monte-carlo-tree-search/) | Searches information sets to address hidden information and uncertainty. | Preserve root information sets and independently permitted opponents. Normalization/pruning must not admit private-state keys. |
| Chaslot et al., [Progressive Strategies for Monte Carlo Tree Search, 2008](https://dke.maastrichtuniversity.nl/m.winands/documents/pMCTS.pdf) | Restricts branching initially and progressively reopens candidates; reports improvement in its Go domain. | Supports a small initial shortlist and slower reopening. Brass K and visit quotas below are proposed adaptations. |
| Lanctot et al., [MCTS with Heuristic Evaluations using Implicit Minimax Backups, 2014](https://arxiv.org/abs/1406.0486) | Studies separate heuristic/simulation evidence and fixed/dynamic early playout termination. | Supports testing heuristic cutoffs and separating evidence. Do not transplant two-player minimax/sign inversion into independently motivated four-player Brass. |
| DeepMind [Mctx Q transformations](https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py) | Implements fixed-range and parent/sibling Q normalization, plus completion of unvisited Q estimates. | Reference for guarded normalization and missing-value treatment; Mctx is a neural-search example, so neural search is not universally exempt from normalization. |
| Leela Chess Zero [AlphaZero primer](https://draft.lczero.org/dev/lc0/search/alphazero/) | Describes PUCT and First Play Urgency, estimating an unvisited edge from its parent rather than forcing every edge first. | Reference for a shared unseen-edge selection repair. Chess values, alternating signs and tuned constants do not transfer to four-seat Brass. |

The literature supports these mechanisms, not a guarantee that the combination is stronger in Brass. The recommendation below is an inference from those mechanisms and this repository's diagnostics. Pin selected code commits/file hashes and preserve notices when implementing adaptations. No new search-algorithm menu is required.

## 3. Separate heuristic and neural settings

Use explicit evaluator and leaf-policy identities, rather than one default parameter bundle for both backends. Proposed contract fields include `evaluator_kind`, `q_transform`, `leaf_policy`, `opponent_turn_limit`, candidate policy, and FPU policy. A neural profile must reject heuristic-only normalization/cutoff fields. An evaluator or horizon change invalidates cached statistics.

| Setting | Masked ISMCTS with rule-based v3 | Masked ISMCTS with MLP-ResNet |
| --- | --- | --- |
| Priors | Cheap complete-turn feasibility and existing tactical/economic scoring | Masked policy head over the full legal list |
| Q normalization | Regularized node-local sibling normalization for selection | **None by default**; use raw four-seat win-share means |
| Initial root K | Lab/Easy 4; Mid 6; Hard 8 | Start 16; tune separately at equal NN budgets |
| Initial interior K | Lab/Easy 2; Mid 3; Hard 4 | Start 8 |
| Expansion | Slow reopening based on completed evidence; explicit tail quota | Policy-guided progressive reopening with bounded pending paths |
| Current turn | Plan both available actions and their resource/beer/merchant subchoices | Same complete-turn commitment and legal incumbent requirement |
| Heuristic horizon | Current turn only, plus a small number of complete opponent turns | **No v3 strategic-depth cap**; learned-value frontier |
| Starting reply limit | Lab 0 voluntary replies; Easy 1; Mid/Hard 2 | No inherited 0/1/2 reply limit; follow the actual scheduler to a supported NN leaf |
| Safety limits | Deadline, tree/memory/choice guards, exact phase completion | Deadline, tree/memory/choice guards and pending/batch limits still required |
| Parallelism | Laptop initially one worker; lab up to four private workers | Required CPU batching; laptop one worker with pending lanes, lab private roots plus central inference |
| Clocks | Lab exactly 1 second; live 2/5/12; old saves 10 | Independent NN budgets; current lab 3-second and laptop presets remain separate |

Protected candidates can make the eligible set larger than K. K restricts ordinary expansion, never legality. Avoid interpreting the table as a requirement to visit every protected candidate before exploiting a strong one.

Bounded neural values justify starting without an extra transform in this project; they do not prove the value head is calibrated. Monitor held-out value calibration. If targets later become unbounded scores/returns, revisit the NN selection scale independently. The base MLP-ResNet remains 1,840,109 parameters and the fresh-training requirement remains in force.

## 4. Repair selection and continuation credit first

Replace unconditional unseen-edge expansion with selection over **both visited and eligible unvisited edges**. Give an unseen edge a parent-based FPU estimate and its prior/exploration term. Its first expansion occurs when it wins that competition or a small explicit exploration quota, not merely because it exists. Pending reservations reduce duplicate batch work but never become completed visits or Q returns.

Starting FPU proposal for the heuristic's existing bounded units:

`FPU = clip(parent_Q - 0.02 * sqrt(visited_prior_mass), 0, 1)`.

Use the same units/transform for FPU and visited Q. Bootstrap from a permitted parent evaluation when no completed parent evidence exists. The 0.02 coefficient is a tuning proposal. The NN branch uses its own parent neural value and separately tuned FPU, with no heuristic Q transform.

For v3, maintain statistics for **each complete turn signature**, including card/resource/beer/bonus choices. First-action statistics can guide tree selection, but cannot select an arbitrary second action. Sample the incumbent and credible challengers under matched belief worlds/opponent policy assignments where practical. Use deterministic scenario IDs and independent random streams to reduce avoidable comparison noise without sharing private state between workers.

Publish a searched complete plan only when its own evidence supports it. Start with at least two completed evaluations for the plan and comparable incumbent evidence; below that, retain the improved incumbent. Raw heuristic/terminal scores and uncertainty remain visible separately. Apply personality only among credible plans after cash, conversion and official terminal outcomes are respected.

For NN, keep semantic-choice visit targets compatible with the existing policy head. Build a continuation from supported child edges at matching permitted information sets, rather than overwriting it with the last trace under the same first action. If a supported second-action trace is unavailable, construct a legal incumbent and replan against the actual observation within the remaining original deadline. Current choices cannot depend on hypothetical cards that would only be observed later.

The lab root forest must merge raw completed plan/edge sums and counts, then make the continuation decision. Do not average worker-normalized Q values: each worker may have a different observed range. Allocate different warm-up challengers across replicas while evaluating an incumbent on common scenarios where useful. Retain private trees, hosts, RNG streams and channels.

## 5. Normalize heuristic Q with safeguards

Keep the current raw bounded four-seat returns for backup and reporting initially. For the root seat `r`, a completed edge has `q(a) = W_r(a) / N(a)`. Normalize **across sibling actions for that seat**, not across the four players and not across unrelated nodes, styles, horizons or requests.

Starting selection transform:

```text
L = min(parent_Q, supported sibling q values)
U = max(parent_Q, supported sibling q values)
q_select(a) = clip((q(a) - L) / max(U - L, 0.05), 0, 1)
```

Enable range estimation after at least two siblings have two completed visits each. Earlier selection uses the bounded raw value. The span floor avoids expanding tiny one-sample differences into a full 0–1 range; compare floors 0.02/0.05/0.10 only if the initial diagnosis warrants it. Recompute from raw completed means when the range changes, instead of modifying already backed-up returns. Normalize FPU through the same transform.

Tune exploration in these selection units: **0.6** is a starting heuristic proposal, with a small 0.3/0.6/1.0 screen if needed. Do not assume the present 1.4 remains appropriate after rescaling. Store raw Q range, normalized range, exploration contribution, support counts and warm-up status in telemetry. Exact proven final outcomes retain priority; normalization must not turn a known losing complete turn into a winning one.

Normalization cannot correct a biased evaluation function or create visits when no path finishes. Its contribution must be measured after selection and rollout-cost repairs. The softmax temperature 20 is a separate evaluator-scale decision; avoid changing it simultaneously with sibling normalization in the first comparison.

## 6. Prune candidates without losing useful turns

Rank strategically distinct first actions, rather than allowing equivalent-looking card variants to consume the whole shortlist. Treat action kind, industry, site/route and economically different consequences as candidate families. Cards, merchant bonuses and resource ownership can change future opportunities, so merge exact afterstate-equivalent variants only after equivalence is demonstrated. Otherwise defer lower-ranked variants through expansion and retain every legal ID.

Protect viable loan-to-build, develop-to-build, network-to-sale, build-to-sale and beer-to-sale combinations. A low standalone proposal score cannot discard their setup action. Use a small, cached completion preview to improve the prior; the planner remains ISMCTS. Pottery and every other industry retain eligible representatives and positive tail support.

For both evaluators:

- Keep the full legal mask and policy denominator unchanged; use a distinct expansion mask.
- Give feasible cash recovery, imminent scoring/conversion and meaningful merchant beer options priority. Compare merchant bonus effects and other beer ownership/consumption effects.
- Keep existing completed edges. Reopen one ordinary candidate only after leading candidates have useful completed support; start with a four-visit support target.
- Reserve approximately **5%** of heuristic admissions for deferred candidates, rather than forcing a tail move on the very first simulation. Maintain eventual reopening as the budget grows. NN tail allocation is separately tuned.
- In batched NN search, count completed support separately from pending paths. Avoid flooding all lanes with low-prior new edges while a promising candidate waits for repeat evidence.

Test a smaller K against full expansion on a few saved states. Search coverage must remain auditable: legal count, eligible count, families, deferred prior mass and selected continuations. Pruning quality is about stronger decisions at the same clock, not simply fewer nodes.

## 7. Shorten heuristic rollouts at settled boundaries

Depth means **completed game actions or turns**, not JSON/RPC count, resource clicks or mandatory liquidation steps. Complete both current actions when available. Then settle their costs, card accounting, income/debt and any due scoring/era transition according to the engine before evaluating.

Starting heuristic horizons are those in the table: the one-second lab worker evaluates the settled current-turn afterstate; Easy allows one complete voluntary opponent turn; Mid/Hard allow two. The opponent prefix follows the actual scheduler. If the root next acts sooner, stop there. A full next-root horizon remains an ablation, not the default requirement for every v3 simulation.

The lab's zero-reply proposal is deliberately cheap and could miss resource competition. Use public threat/opportunity features at the leaf, then compare zero versus one reply on the same one-second budget. Keep the better qualified variant without extending that clock. This is a heuristic accuracy/throughput tradeoff, not a claim that shorter is always stronger.

Never cut during coal/iron/beer selection, a sale bonus, incomplete payment or mandatory liquidation. If the budget prevents reaching a valid cutoff, discard the path. Predicted future voluntary decisions may be replaced by the leaf evaluator; unresolved mandatory mechanics may not. Choose one horizon policy for the request so differing depths do not silently compete in incompatible units. Tag teachers/statistics with evaluator and leaf-policy versions.

The NN branch uses a separate frontier rule. Preserve current-turn completion, explicit chance and independently masked fixed opponents. At a new, valid root information-set leaf supported by the NN's training distribution, request the policy/value batch and back up its four-seat win shares. When that leaf is already expanded, later simulations may deepen it. Replace the current hard `future_turns=1` cutoff with this frontier policy when implementing the separation; do not attach the v3 reply cap. Avoid feeding an off-turn observation to a value head trained only on acting-player states. Deadline, choice and memory guards remain finite even without a fixed strategic-depth cap.

## 8. Improve the heuristic and reduce repeated work

An efficient search around a poor evaluator remains a poor opponent. Audit the most influential v3 ledger terms on saved decisions: unflipped industries versus reachable sales, shared beer/action capacity, merchant bonuses, loan timing, development payback and resource depletion before era end. Completion forecasts must not promise the same scarce beer/action to several incompatible sales. Respect genuine batched-sale opportunities rather than charging one action per tile.

Strengthen the fast incumbent with a few feasible setup/completion combinations and imminent conversion. Separate certainly unaffordable payments from speculative future financing penalties, so temporary borrowing-floor risk does not automatically veto a demonstrably recoverable complete turn. Keep the known-payment protection and exact endgame scoring.

Profile and then cache:

1. Legal ranking and protected-family membership once per compatible information node; eliminate repeated sorts inside the unseen-candidate loop.
2. Root permitted keys and static proposal features once per request; never use sampled hidden contents as an information cache key.
3. A shared board conversion forecast and four public seat ledgers once per evaluated boundary, instead of four full `Evaluate` recomputations.
4. Complete-turn public scores by validated turn signature, while keeping sampled opponent outcomes separate.
5. Private simulator restore/reuse and incremental evaluation where the host supports exact isolation; avoid reconstructing an extra world merely to score the already completed root turn.

Measure incumbent, world setup, legal generation, priors, hashing, root completion, replies and leaf evaluation separately. Stop preparing expensive new paths when there is too little time to finish them. A short reply horizon should increase completed comparisons, not just move time into a larger seed generator.

This work leaves the [1.4.7 resource lifecycle](../docs/AI_EXECUTION_RESOURCES_1.4.7.md) intact: 30 FPS computation/retirement rendering cap, original VSync restoration, single live job, cancellation distinct from completion, device retirement before buffer reuse, optional-work restoration and shared weight ownership. Laptop and lab CPU allowances remain separate. Increasing private lab workers does not increase the game clock or authorize overlapping live turn jobs.

## 9. Implementation order and acceptance

| Stage | Main components | Required exit |
| --- | --- | --- |
| F0 Diagnosis and contracts | Shared C# telemetry; Python search statistics; new versioned evaluator/leaf/selection contract | Reproduce current low completion/no-revisit case; record incumbent/candidate/reply/leaf timing; distinguish heuristic and NN settings |
| F1 Selection and continuation credit | `BrassMaskedSearch/MaskedSearch.cs`, `search/ismcts.py`, `recovery/legacy_search.py`, `recovery/parallel.py` | Visited edges compete with unseen edges; pending work adds no visits; returned continuation has its own support; raw forest merge agrees with serial arithmetic |
| F2 Heuristic cost, horizon and candidates | Shared C# planner/Evaluator/Conversion and both C# profile adapters | Complete legal two-action incumbents; settled short leaves; cached scoring/ranks; protected setup/industry/sale options; more completed repeat comparisons at the unchanged clock |
| F3 Heuristic Q transform | Shared C# selection and heuristic profile contracts | Warm-up/constant-range/terminal/seat tests; no NaN; transform affects selection only; independent contribution measured |
| F4 Neural separation | `search/common.py`, `search/ismcts.py`, batching/inference and active NN configs | NN Q transform off; no inherited heuristic depth/reply cap; supported NN frontier, complete turns, private opponents, CPU batching and valid visit teachers |
| F5 Strength screen | Current frozen v3, improved incumbent, shortlisted fixes and saved-position panel | Correctness gates pass; useful repeats; equal-budget match and economic evidence; no claim based solely on self-play scores |
| F6 Freeze and package | Six profiles, immutable bank, authored worker transfer, game catalog/build/install, lab guides | New source/binary/config fingerprints; clocks/saves/resources preserved; current frozen evidence remains intact; promotion linked to its own evidence |

F1 is shared, but F2/F3 are heuristic changes and must not silently alter NN values/horizons. F4 preserves the 1.84M residual model, CPU/GPU batching, fresh initialization, self-play curriculum, audited heuristic data, separate search distillation and storage retention. Existing source-bound runs cannot silently resume after learning/search implementation changes; retain them as evidence and follow the existing fresh-run contract when starting the revised recipe.

Create new versioned heuristic profiles and a new immutable bank; do not overwrite the current frozen bank or relabel its old results. Keep the six player-facing v3 styles, Pottery Flex identity holdout and unrestricted learner Pottery actions. Generate new teacher data with provenance for the new bank. Choose the next free game package version after inspecting the then-installed patches. Preserve their UI/rules/statistics binaries and old saved-profile hash validation.

## 10. Compact evaluation plan

Use **12–16 saved four-player positions** spanning early/late Canal, early/late Rail, money distress, sale/merchant beer, setup combinations and contested resources. Fixed seeds and matched scenarios permit a small screen of repaired selection, smaller K, shorter horizon and Q transform one change at a time. Add a combined variant only after the individual behavior is understood.

Engineering checks should verify masking/hidden-world invariance, stable four-seat returns, two-action completion, repeated root support, continuation credit, cutoff phase safety, terminal outcomes, cancellation and batch retirement. Require useful repeat evidence on non-forced decisions; a zero-completion run must report fallback, not claim a search improvement. Use the existing native compact matrix/resource checks for packaging rather than a large new synthetic suite.

For strength, start with **20 four-player games**: five independent setups, four seat rotations, one candidate, one frozen current v3 control and two rotating frozen styles. Compare with the improved incumbent without search as well as current 1.4.9; stock Hard alone is too weak a qualification target. Use the same one-second per-turn lab budget and CPU allowance for candidate and controls. All-candidate self-play measures economics/diversity but cannot establish superiority.

Report matched candidate/control outcomes, VP, rank, cash distress, unused level-one tiles, unflipped endgame industries, merchant beer use, industry opportunity/chosen counts, completed/revisited plans and fallback rate. Treat setup blocks, not every move, as independent evaluation units. Twenty games is a screening panel: report uncertainty and extend only an ambiguous decisive comparison. Validate selected decision fixtures at the laptop's 2/5/12 clocks; lab timing does not establish laptop performance.

Promotion requires improved practical decisions and an equal-budget advantage over current v3, with no cash/conversion/Pottery/resource regression. Raw simulations per second or lower loss alone are insufficient. Select a qualified configuration for each evaluator; subsequent difficulty/personality calibration remains a separate step after adequate neural training.

## 11. Implemented F0–F6

See [1.4.11 release notes](../docs/MASKED_ISMCTS_STRENGTH_1.4.11.md), [exit checks](../docs/MASKED_ISMCTS_STRENGTH_EXIT_CHECKS.md) and the [bank receipt](../training/opponents/evaluations/masked-strength-1.1.0/report.json). The final defaults are heuristic K 4/2 and zero voluntary replies at the lab's unchanged one second; game Easy/Mid/Hard use 4/2, 6/3, 8/4 and one/two/two reply turns. Heuristic sibling normalization uses minimum span 0.05 after two siblings each reach two completed visits. NN K 16/8, raw Q, supported information-set frontier and batching are separate.

The final 16-position panel averaged 4.625 completed simulations versus 0.4375 for the control; only supported complete turns can replace the incumbent. The 20-game screen averaged 47.60 versus 36.15 VP and 2.15 versus 2.90 rank. Five setup blocks gave a descriptive VP-difference bootstrap interval of 0.9–22.5. This supports repair promotion, not human-level or per-style strength claims. Q's isolated advantage was not established by the four-position ablation. Selected laptop-clock decisions, native resources/saves, fresh training/resume, batched NN teachers, distillation, ONNX and retention checks passed.

The active immutable bank is `bank-981744ddae4aaf07f1cf`; five training identities and the evaluation-only Pottery Flex identity use `*-v3-masked-strength`. All learner industries remain legal. [Updated conda instructions](../docs/TRAINING_RECOVERY_LAB.md) require a new run directory and the new private runtime transfer. No existing neural checkpoint is imported.


## Era economics and four-worker implementation (1 October 2026)

[Implementation, diagnosis and matched screens](../docs/MASKED_V3_ERA_REPAIR.md) supersede the previous active heuristic-bank/default concurrency references. Rail charges and complete investment alternatives are priced without an industry/rail quota. Development must improve a feasible replacement; loans consider actual funding gaps. Industry and link valuation share one flip forecast. Heuristic v3 and NN keep separate Q/horizon settings. Each decision uses four private roots; CPU heuristic coalescing and central NN batching do not share mutable trees. NN pending paths use selection-only virtual loss. All owners retire before release.

The main lab uses 24 queued heavy tables and a shared 60-slot admission gate in the 64-core allocation; ordinary turns use one slot and searched turns four, before their clocks start. Imitation/evaluation v3 remains searched at one second. Early RL uses distinct search-free incumbent opponents; at 10,000 games/5% v3 tables, those switch to searched v3. Fresh MLP-ResNet training, unrestricted Pottery, separate distillation, evaluation exclusion and retention remain. A game update requires a fresh equal-budget win and correctness/package gates; previous banks/results remain immutable.

### Final game/lab worker allowance (2026-10-01)

The user's final preference supersedes the preceding four-worker game allowance: installed patch 1.4.16 uses **two game search workers**, while imitation, late searched v3 RL opponents and NN distillation retain **four lab workers** with **24 queued tables**. The active frozen strategy is `bank-3f7221cadcd472aafdb9`; early search-free RL opponents use that same bank. The eight-game four-worker strength screen tied four wins each (78.75 versus 76.0 mean VP); installation was explicitly requested after the tie. Game clocks remain 2/5/12 seconds. See [the final release](../docs/MASKED_V3_1.4.16.md) and [the conda lab guide](../docs/TRAINING_RECOVERY_LAB.md).
