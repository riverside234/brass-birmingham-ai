# search_v2 implementation and evaluation plan

Date: **2026-10-28**, design revision **2**, V205 follow-up **1**. Status: **cache-only search_v2 installed at the user's request; V205 disabled; installed 4s/K8 smoke passed**. See [installation report](SEARCH_V2_INSTALLATION_REPORT.md), [original evaluation](SEARCH_V2_EVALUATION_REPORT.md), and [V205 follow-up](SEARCH_V205_EVALUATION_REPORT.md).

Deployment revision **1.4.39-rust-local** selects `search_v2_focus_only`. The actual installed player passed 745 checks in one short 4s/K8 turn, committing both actions through one request in 3,849.85 ms, then saving/loading the completed-turn checkpoint. Existing 4/8/10/12-second settings and K8/K16 mapping remain unchanged. Exact rollback was tested and retained. Earlier study receipts describing an unchanged search_v1 installation are historical, pre-deployment evidence; their hashes and contents remain preserved.

The original implementation/evaluation uses the simpler `search_v2_focus_only` option below. It performs optional internal second-action refinement with exact NN-cache reuse and **zero inherited tree visits**. Its completed native comparison remains 48 trials plus two no-switch trials, totaling 328 seconds of admitted search allowances, in independent short batches; those measurements must not be relabeled as subtree reuse.

**Current user-authorized V205 scope:** compare statistical descendant reuse (`search_v2`) against cache-only refinement **only at 4s/K8**. Use six short coverage pairs and two predeclared diagnostic second-seed pairs (16 timed trials, 64 seconds of admitted allowances, independent 60-second pair watchdogs), plus deterministic correctness and bounded isolated delivery checks. No 8s/10s/12s timing, long sessions, lab deployment or production switch. The new [runner](../scripts/benchmark-search-v205.py) freezes the protocol before timing. Keep or reject reuse based on measured usefulness versus cache-only, without treating inherited edge counts as new simulations or assuming deeper search proves stronger play.

V205 implementation uses the existing `slotmap` arena with a bounded first-expansion parent forest and a completed-round undo journal. It moves each owner's arena, restores the eligible watermark, and prunes unrelated/partial branches after acceptance. Full prefix/semantic equality, each owner's sampler certificate and node-relative ledger are required; any miss uses the existing cache-only path. The promoted root resets N/W, ledger and tournament; compatible descendant N/W may survive. The independent [scalar reuse reference](../tests/RustBackend/gumbel_reuse_reference.py) and its pinned [fixture](../tests/GumbelISMCTS/fixtures/reuse-v1.json) specify the rollback/count contract. The forest deliberately misses descendants whose first-expansion ancestry cannot prove the exact prefix; it does not infer reachability from hash equality.

**V205 disposition:** all 16 timed plans were independently legal; 16/16 V205 owners hit retained descendants. The six-case primary median was -0.27% new completions, +19.2 ms native delivery, unchanged process CPU and +13.9 MiB peak memory. All eight pairs returned identical plans. One Canal latency gain repeated, but the actual Unity pair was essentially unchanged (3,344.8 vs 3,345.4 ms). Keep cache-only for 4s/K8; leave V205 available only as a disabled opt-in experiment. Native correctness, cancellation, between-action load and completed-turn save checks passed. [Recorded decision](../artifacts/search-v2/20261028/v205/decision.json)

The frozen protocol and the authored multi-sale fixture correction are separate receipts: [original study](../configs/search/search-v2-study.json), [corrected fixture protocol](../configs/search/search-v2-fixture-correction-v2.json). The originally selected L02 fixture failed strict inventory admission; the correction uses the existing independently captured L04 repaired fixture. Both rejected variants remain in the evidence directory. No gate, cost, seed or model tuning followed held-out results.

Retirement detail: owners can acknowledge a switch at a drained trajectory wave while retaining only the **last completed round's** recommendation/support watermark. Later partial-round work is excluded from the switch evidence and discarded only after acceptance. A failed preparation resumes that exact tournament, including its partial work and draw stream. This avoids imposing a blocking two-owner barrier after every round.

Live-boundary detail: Rust revisions count staged semantic choices; Unity revisions count game events. The actual-successor comparison translates only the new revision/recall stamps before comparing the entire permitted state and context. The real Unity revision, original request identity, actor, cancellation and detached validation receipt remain authoritative at commit. A projected revision never acknowledges a live action.

Define the current local NN search scheduling method as **`search_v1`**. The revised **`search_v2`** is **adaptive whole-turn planning with optional second-action refinement inside Rust, one shared deadline, and strictly compatible subtree reuse**. It retains a complete legal plan as fallback and normally returns both actions in one response.

**Design change:** logically choose the complete first action before focusing on action two, but perform both planning phases inside the same native request, before applying either action in Unity. Replace the earlier mandatory live handoff and provisional 60/40 split with a conditional switch at completed Gumbel rounds. If the switch is unsuitable, continue the original-root search. This is a proposed engineering improvement, not a demonstrated strength gain.

This is a local Windows/CPU follow-up to the [Rust backend migration plan](RUST_AI_BACKEND_MIGRATION_PLAN.md). Both methods continue to use Rust Gumbel IS-MCTS, the Rust game simulator and ONNX Runtime CPU inference. These names identify scheduling/reuse variants, not NN model versions or a replacement of Gumbel IS-MCTS. The proposed reuse is an extension to the [existing mathematical contract](GUMBEL_IS_MCTS_REFERENCE.md), requiring its own correctness evidence.

The documentation baseline was inspected at Git revision `04ba51072d432f389842496d849b299a7e085f86`, with local release 1.4.38 installed. Freeze the actual source and binary hashes again before implementation/evaluation. Writing this plan does not change the installed game or its configuration.

## 1. Objective and hypotheses

Determine whether `search_v2` makes better use of a short whole-turn allowance, particularly when action two has many meaningful alternatives. The first action must still be evaluated through simulations that include action two and subsequent opponents' play. Focusing on the selected continuation must not become greedy one-action lookahead.

The earlier design paid for an extra Unity snapshot/validation/IPC handoff during the search window and assumed a useful time split could be predicted before observing search progress. It also made the first action irreversible earlier. For ordinary boundaries whose permitted successor can be reconstructed deterministically, the Rust simulator can prepare the second-action root directly. This removes the routine extra host handoff and lets difficult first decisions retain the full budget.

Test these hypotheses separately:

1. Switching only after a stable complete first-action recommendation and a useful remaining work window improves second-action decisions without sacrificing important first-action combinations.
2. Reusing compatible descendants reduces repeated expansion and NN evaluation enough to justify retention/matching overhead.
3. Keeping the refinement inside one native request makes the gain survive snapshot, validation and IPC costs in the installed-game execution path.

More simulations do not establish stronger play. The first choice becomes fixed within the candidate search after switching, so premature commitment can still lose a better combination even though Unity has not applied it. The switch heuristic must be evaluated on such cases. Earlier first-action presentation is no longer a primary goal; no latency improvement is assumed merely because planning has two phases.

| Design option | Decision for this revision |
| --- | --- |
| Fixed split and live first-action commit before searching action two | Defer; requires a host handoff and a split predictor, and is unnecessary for certified deterministic ordinary boundaries |
| Adaptive in-process refinement with a complete-plan fallback | Preferred `search_v2` candidate; preserves one request and uses measured search progress |
| Same adaptive refinement with a fresh logical tree and existing NN cache | Required simpler ablation; it may be sufficient if statistical reuse adds little value |
| Import old visits into the new root tournament, or repeatedly switch between several first-action plans | Defer; substantially more statistical and lifecycle complexity |

## 2. Exact method definitions

| Behavior | `search_v1`: frozen baseline | `search_v2`: candidate |
| --- | --- | --- |
| Whole-turn clock | One allowance shared by both actions, staged choices and retries | Same admission boundary, allowance and monotonic clock |
| First root | Searches using the available remaining work allowance | Starts the same original-root search; may switch at a qualified completed-round checkpoint |
| Plan extraction | Extracts one or two supported complete actions from the search; bounded completion handles missing support | Retains a legal incumbent plan, optionally improves its second action, and returns the final plan in one response |
| Second action | Uses a valid returned continuation; searches a changed root if a continuation is absent or rejected and time remains | Optionally searches a certified projected successor inside Rust; actual Unity application still validates the second action against the real state |
| Search tree across roots | A changed root requiring search creates a fresh tree | Retains each owner's eligible descendants within the current request; no persistent cross-turn tree reuse |
| New-root Gumbel statistics | Fresh tournament and counts | Fresh tournament and root counts; compatible descendant statistics may survive |
| Live application | Detached action validation and guarded main-thread commit | Same validation and commit protections, applied to each actual state |

Current implementation anchors: [session and continuation handling](../native/brass-ai-core/src/search/session/mod.rs), [supported complete-action plans](../native/brass-ai-core/src/search/session/plans.rs), [whole-turn clock](../native/brass-ai-core/src/search/session/clock.rs), [two-session coordinator](../native/brass-ai-worker/src/pair.rs), and [Unity bridge](../src/BrassLocalAI/Unity/LocalTurnBridge.cs).

`search_v1` must remain available as a frozen control. Do not silently improve its scheduler, fallback, batching or validation while comparing it with `search_v2`.

The ordinary path is: **original-root search → optional internal second-action search → one complete-plan response → validate/commit action one → validate/commit action two**. Unsupported projected boundaries use the existing continuation/replan path with the original remaining allowance.

## 3. Fixed experimental conditions

Run both methods at all three requested settings:

| Configuration ID | Whole-turn allowance | Requested root K | Search ownership |
| --- | --- | --- | --- |
| `4s_k8` | 4,000 ms | 8 | Two independent sessions, one compute thread each |
| `8s_k8` | 8,000 ms | 8 | Two independent sessions, one compute thread each |
| `8s_k16` | 8,000 ms | 16 | Two independent sessions, one compute thread each |

- Use the same pinned temporary g890 model/export, feature schema, reward/utility contract, rules, CPU provider, compiler profile and host for every arm. The test NN is not the final model. Preserve the existing model-specific acceptance of its utility head despite a null calibration receipt.
- Keep ORT intra-op and inter-op settings at one and its execution sequential. Each session performs simulation, encoding, inference and backup on its own compute thread. No extra inference pool, shared mutable tree or third search thread.
- Keep batching, tree limits, legal-choice enumeration, root candidate rules and two-session recommendation policy fixed except for explicitly versioned changes necessary for root epochs/reuse.
- Requested K stays fixed for both actions of a turn. Report effective K separately for each session/action when legal-count or quota limits reduce it. Never choose requested K from remaining time.
- **`8s_k8` is an experimental override.** The installed mapping currently uses K=16 at 8 seconds. Supply the override through an isolated benchmark manifest/package and pin it for both methods; do not edit the installed `policy.json` or the user-facing difficulty mapping.
- Compare methods within the same configuration first. The `4s_k8` versus `8s_k8` comparison measures extra time; `8s_k8` versus `8s_k16` measures candidate width. Do not attribute either difference solely to scheduling.

## 4. Whole-turn timing and adaptive focus

### 4.1 One authority for time

Keep the current caller-owned stopwatch and native deadline narrowing. Record the initial snapshot/model-startup exclusions exactly as in `search_v1`; measure them separately in end-to-end latency. Once the turn clock starts, do not pause it between actions.

For allowance `T`, retain the current finalization reserve `R = min(1.2 seconds, 0.10 * T)`: 400 ms at 4 seconds and 800 ms at 8 seconds. The turn work deadline is `t0 + T - R`; the delivery deadline is `t0 + T`.

An internal search checkpoint or root switch must not shorten the shared deadline through `Budget::narrow`, create a new admission clock, or set the turn-cancellation flag. Search, checkpoint messaging, projected-root construction, matching, retirement and finalization all consume the original allowance. The second planning phase receives only the time actually remaining.

There is no ordinary Unity handoff between planning phases and no speculative search concurrent with live mutation. Any exceptional live replan still charges snapshot preparation, IPC, validation and presentation waits against the existing clock. Measure initial preparation, final delivery, both live validations/commits and total observed turn latency separately, using the same definitions for both methods.

### 4.2 Switch policy: no mandatory percentage split

Start the first root with `search_v1`'s quota estimation, candidate selection and frozen Gumbel draws. Preserve that tournament's state across completed-round checkpoints; do not restart it to obtain another recommendation. Root selection must still evaluate future actions and opponents.

The coordinator may request **at most one successful switch per turn**, only when all these conditions hold:

1. Two complete ordinary actions remain at the initial boundary. A one-action turn, debt, a terminal boundary or an unsupported information transition keeps the baseline path.
2. The pooled recommendation has selected the **same complete first-action path** at two consecutive qualified coordinator checkpoints. Both owners must have advanced their completed-round watermark between those observations. Matching only the action family, initial edge or repeated read of one snapshot does not satisfy this condition. Use the existing minimum complete-plan support of two; do not invent probability/confidence bounds from correlated MCTS samples.
3. The selected full first action has a certified permitted successor, with the same actor and turn, and a legal full-turn fallback has been retained. A search-supported second action is preferred; a separately labeled bounded legal completion is acceptable. Never switch from a better validated incumbent to an incomplete prefix.
4. The second root has at least two legal candidates and sufficient estimated work for a useful halving tournament after preparation/retirement cost. Both owners must satisfy the affordability check. A cache/tree hit is helpful but is not required; this keeps the same switch policy usable by the no-reuse ablation.

Recommendation stability is a heuristic, not statistical confidence or proof of optimality. All gate outcomes, including no-switch turns, must be reported. Gate tuning uses only the development roots and must be frozen before held-out cases; the initial policy is shared across all three test settings.

### 4.3 Affordability and bounded preparation

Let `B` be time remaining to the turn work deadline and `H_native` a pinned conservative estimate for checkpoint retirement, projected-root preparation and matching. Obtain a per-trajectory cost estimate `c` for each owner from the maximum of the admitted startup estimate and its last up to eight completed trajectory costs. Use the startup estimate alone before samples exist. This conservative estimator is for the switch gate; do not silently replace `search_v1`'s quota estimator or claim it is a measured population tail bound.

Compute an affordability quota `floor(max(0, B - H_native) / c)`. Apply the existing legal/quota-based effective-K rule and require effective K at least two with enough work for its complete halving schedule. In the current implementation, retaining effective K requires at least `K * log2(K)` trajectories. Report when requested K=8 or 16 is reduced. Recheck after actual preparation; estimates do not extend deadlines.

Make at most one bounded projected-root preparation attempt after the cheap stability/affordability checks pass. Preview/replay uses a separate deterministic completion stream, not the original tournament's sampling stream. A failed certificate or insufficient time resumes the same original tournament if possible; it does not reset its quota, draws or statistics. Include this failed preparation cost in results.

If all checks pass, quiesce both owners at safe round boundaries, recheck the selected full path against their final eligible snapshots, and retire every pending trajectory before switching. Cancel the switch if the recommendation changed. The second root freezes its own quota from the actual remaining time. There is no return to competing first-action branches in this revision.

### 4.4 Round semantics and checkpoint overhead

Keep round quotas fixed inside each tournament. The original tournament may stop early at a completed-round snapshot; label it as a policy-driven early stop, not as full-quota completion. Never publish a partial halving round or convert old root work into the new root's quota.

Use bounded progress messages containing eligible snapshots and complete-plan evidence, never copies of the tree. Owners may continue while the coordinator observes snapshots; a switch uses an explicit retirement/acknowledgment handshake. Distinguish stale observations, internal root epochs and external cancellation. Avoid a compulsory blocking barrier at every round, and measure checkpoint/handshake idle time.

Extract progress only at eligible round boundaries. Reuse immutable proofs and path signatures where possible; do not scan/copy the full plan ledger on every trajectory. Profile this added extraction cost on dense roots as part of the no-switch control.

If no switch qualifies, finish the original tournament/normal cutoff. A fake-clock, recorded-draw no-switch run must agree with the baseline's decisions and backup trace. Real-time overhead may still reduce completed work and must be measured rather than declared free.

## 5. Subtree retention and statistical semantics

### 5.1 Projected root and authoritative state

Construct the second-action root by replaying the exact selected complete first action through the Rust simulator on an isolated permitted reconstruction. Certify that this prefix reaches the same acting player's next ordinary decision and that its permitted successor is determined by known information and the action path. Chance, new unknown private observations, another actor's decision or a changed turn boundary prevents this optimization unless explicitly qualified later.

The certificate must cover the visible board/resources, own hand and recall, ordered legal choices, causal reward ledger and expected revision. Establish the permitted-transition property from the supported rules plus tractable exhaustive tests; agreement between a few sampled hidden worlds alone is insufficient. Construct new sampling inputs from the permitted successor; never expose a sampled opponent hand or deck order as known information.

Separate **projected search state** from **acknowledged live state**. The external request ticket/root hash/revision remains bound to the original live request throughout both phases. A projected future revision is an internal hypothesis, not authority to commit. Do not advance the session's acknowledged recall/revision or fake a Unity acknowledgment to satisfy `Session::accept`.

The final response contains the original ticket, combined complete semantic actions, and the expected intermediate preconditions. Unity must capture the actual state after action one and validate action two again. If the state differs, reject that continuation and follow the original remaining-budget/recovery path.

### 5.2 Retention and matching

Reuse the existing generational arena, semantic-key lookup, buffers and crates. Give resumable search state ownership of the arena, root handle, draws, tournament, support ledger and pending work. Retain these across checkpoints and the optional internal focus; avoid serializing or cloning entire trees. This revision does not require cross-turn or save/load tree persistence.

The current arena is indexed by information-state keys rather than explicit persistent parent/child links. Add bounded successor/reachability metadata, or another verified way to restrict reuse to compatible descendants of the selected complete first action. A hash match alone is insufficient; retain full semantic equality and owner/generation checks.

Reuse requires all of the following:

- Same live game instance, turn, seat, model/export, rules, feature/reward/profile identities and permitted policy assumptions.
- Exact complete action path and certified projected observation, including own hand, recall, phase, expected revision, legal semantic IDs and ordering; actual live application is verified separately afterward.
- Compatible hidden-information sampling distribution. Initially admit only ordinary same-player action boundaries for which conditional sampler compatibility is established. A new private observation, chance event or changed belief assumption invalidates statistical reuse unless separately qualified.
- Compatible utility ledger, remaining cost context, settlement state, value perspective and evaluator/sampling semantics. The current `max_depth` is a diagnostic failure guard, not a truncated-value horizon: preserve its meaning and test depth accounting after reroot. If a future evaluator introduces a horizon or discount change, version it and discard incompatible statistics.

Resample hidden worlds from the new permitted root for new trajectories. Never retain a particular determinized opponent hand as the real state. On any mismatch, safely rebuild the root using only the remaining budget; record a precise miss reason.

Keep each session's arena private. Both sessions follow the same pooled first-action prefix, even when only one session supplied the selected plan. The other session may have no matching subtree and must handle that independently. Do not copy the winning owner's statistics into the other owner. Limit retained memory, release unrelated branches only after the switch is accepted, and include retention/pruning time in the measurement.

### 5.3 Conservative root-count contract

For the first `search_v2` implementation:

1. Retain compatible descendant structure, NN evaluations and node-relative completed statistics.
2. At the promoted root, reuse compatible NN priors/value, but start a **fresh root epoch with root `N/W = 0`** and a fresh halving tournament. Old counts at this node are not inherited as new-root simulation credit.
3. Preserve compatible statistics below the new root. Every new trajectory contributes exactly once to the new root and once to each visited descendant, respecting the existing node-relative utility accounting.
4. Generate fresh root perturbations shared by the two sessions; keep their world/chance streams independent. Bind draw identity to the new decision/root epoch and common semantic root, not to a session's retained-node count. Do not force the old second-action fallback into the new Gumbel candidate set or redraw noise until it is selected.
5. Rebuild the current-root plan-support ledger and completed-round watermark from new completed trajectories. Preserve an old complete plan separately with its original provenance; it does not acquire new search support simply by being reused as fallback.
6. Pool only compatible current-epoch root snapshots. Descendant counts are neither added to both owners nor counted again as newly completed root work.

This deliberately keeps the existing root-count/halving invariants checkable. Importing old promoted-root counts as weighted evidence is a separate possible extension, outside this first comparison. The [paper's Algorithm 1](https://arxiv.org/html/2405.00243v1#S6) starts a fresh search with zero counts and distinguishes root halving from internal-node selection. Neither its improvement argument nor the existing reference proves strength for this adaptive stopping/reuse extension with approximate Brass beliefs.

Implement the scheduler first with a fresh logical second-root tree and the existing exact NN cache. Add retained descendant statistics only after the sampler/value invariants pass. If only cache reuse can be qualified or it already captures the useful gain, report that simpler variant explicitly instead of labeling it full statistical subtree reuse.

### 5.4 Complete-plan incumbent and evidence composition

Preserve `search_v1`'s supported plan extraction and bounded legal completion. Before switching, retain an immutable complete legal plan for the selected first action, with exact preconditions and provenance. Building or validating this incumbent consumes the same work window. No incumbent can cross an unknown draw or unsupported information boundary.

During focused search, keep the incumbent intact until a complete second action has qualified under the new root's current-epoch selection/support rules. Replay the composed plan from the original input before replacing the incumbent. A partial search, timeout, missing support or failed candidate validation leaves the incumbent available for finalization; cancellation invalidates all delivery.

Do not compare a first-root full-turn value directly against a second-root value to claim monotonic improvement. Their causal costs, sampled evidence and Gumbel draws differ. The incumbent guarantees a known legal alternative, not a mathematically nondecreasing playing value. Replacement quality is an experimental question.

The existing `SupportedAction.support` is support for an entire prefix from one root. A first action supported in epoch one and a second action supported in epoch two **do not form a jointly sampled prefix with either count**. Introduce a versioned composite-plan record with per-action source root/epoch, owner, completed-round watermark, conditional support and expected boundary identity. Keep the existing `search_v1` record semantics unchanged. A fallback keeps its old provenance and never receives fabricated new support.

The final plan must validate from the original live root, then each action must pass the existing live validation/commit guards. If no switch occurs, retain the baseline completion behavior. If no complete legal result can be delivered, preserve safe-stop/checkpoint recovery; never reset time or execute stale evidence. Report incomplete plans, safe stops and fallback use in both arms.

## 6. Implementation sequence

V201–V204 and V206–V207 have cache-only implementation/evidence. V205 is the separate follow-up below. Build candidates outside the installed runtime; this plan authorizes no production switch by itself.

| Step | Work | Completion evidence |
| --- | --- | --- |
| V201 — Freeze controls | Pin `search_v1` sources/binaries/model/configuration and current receipts. Add explicit method/configuration IDs in the isolated harness, including 8s/K=8. Freeze fixture IDs, seeds and run order before timing. | Reproducible baseline manifest; exact old recorded-draw traces still pass |
| V202 — Resumable reference and no-switch control | Extend the scalar reference with fake time and completed-round checkpoints. Refactor native owner-local search into resumable state while keeping the switch disabled. | Recorded-draw decisions/backups match `search_v1`; no-switch overhead, cancellation and pending-work ownership are measured |
| V203 — In-process refinement with cache only | Implement the stability/affordability gate, certified projected successor and immutable incumbent. Search the second root with fresh statistics and existing exact NN-cache reuse. | Fake-clock gate/no-gate cases; combined plan validates from the original input; no extra ordinary Unity request or acknowledgment |
| V204 — Integrate both owners and assess the simpler candidate | Coordinate snapshots and switch retirement; align shared draws and version composite per-action evidence. Run the scheduled cache-only/no-switch diagnostics before investing in statistical reuse. | Stable/changed recommendation during retirement, mixed-epoch rejection, no fabricated joint support, and recorded evidence that useful focus windows exist |
| V205 — Qualify statistical reuse | **Implemented and evaluated; reject for selected 4s/K8 candidate.** Opt-in `search_v2`, policy `descendant-reuse/1.0.0`, remains available for reproducibility; cache-only stays selected. | 88 native test executions; scalar/count parity, hit/miss/value/depth/sampler/collision/cancellation checks; 16 timed trials and 32 independently validated actions; four short Unity smokes; no overall delivery benefit and higher memory. See [report](SEARCH_V205_EVALUATION_REPORT.md). |
| V206 — Verify local delivery | Return one final plan under the original request identity. Preserve actual-state validation for action two, exceptional replanning, cancellation, presentation and checkpoints. | Short isolated Unity checks verify predicted/actual state equality, mismatch rejection, fallback, save/load and original deadline |
| V207 — Run short evaluation | Execute the three-setting paired matrix and small component ablations below. Keep all failures and raw rows. | Per-configuration report with measured benefits, regressions, limits and a retain/revise/reject recommendation |

Use the scheduled two development-root diagnostics as an early feasibility check. If the gate has no useful windows, projection is too expensive or cache-only focus clearly damages first-action combinations, revise or stop before implementing full statistical reuse. If the simpler method already works well, retain it as a viable outcome. These diagnostics use the planned study rows, not an additional parameter sweep; reuse a measurement only when its tested binary/configuration and execution path still match the final comparison.

Primary implementation locations are `native/brass-ai-core/src/search/session/{mod,clock,plans,tournament,cooperative,merge}.rs`, `native/brass-ai-core/src/search/{tree,brass,reward}.rs`, `native/brass-ai-worker/src/{pair,local}.rs`, and `src/BrassLocalAI/{Native/NativeProcess,Native/NativeGameState,Unity/LocalTurnBridge}.cs`.

Extend the existing [Python session reference](../tests/RustBackend/gumbel_session_reference.py), [native session tests](../native/brass-ai-core/tests/sessions.rs), [local protocol tests](../tests/RustBackendLocal/test_local.py) and short L14/Unity harnesses. Reuse existing crates and infrastructure. Add versioned fields for method/policy revision, common turn deadline, internal phase/root epoch, gate reason, projected boundary, per-action evidence, fallback provenance and reuse metrics. The external request identity and `remaining_ms` continue to describe the original live turn. Version the coordinator/composite-plan contract explicitly; preserve `search_v1` parsing and evidence.

A future protocol field must be validated against the admitted configuration, not accepted as an arbitrary client extension of time or K. User-facing difficulty choices remain a separate deployment decision.

## 7. Correctness and lifecycle coverage

Use deterministic/fake-clock checks for most cases, then a few short real-runtime checks:

- Both planning phases share one clock; retries, cache misses, forced subchoices and root rebuilds cannot replenish it. Checkpoint pause/switch differs from turn cancellation. No NN work or search backup starts/publishes after its permitted cutoff.
- Loan/build combinations, development/payment/bonus substeps, multi-sale ordering and Rail double-link completion remain complete and legal. First-action search still considers action-two consequences.
- Only a stable complete first-action path satisfies the gate. Repeated snapshots, stable initial edges with different payment paths, an owner that has not advanced, and a recommendation changed during retirement do not. No-switch execution preserves baseline count/draw semantics.
- Equivalent complete paths project the right root; changed card, cash/resource availability, hand recall, actor, revision, legal ordering, reward context or model rejects reuse. Exercise deliberate hash collisions and expired/foreign arena handles. Projected revisions never authorize live commits.
- Verify conditional hidden-state sampling and the deterministic permitted transition on tractable enumerated fixtures. Incompatible beliefs or value semantics reset statistics; unsupported successor boundaries keep the baseline path. Unknown future cards never enter a stored continuation proof.
- Cancel during traversal, inference, checkpoint messaging, projection, backup, pruning, validation, between actions and immediately before commit. Both owners retire before reroot/new requests; late results mutate neither arena epochs nor the live game. Retained statistics must agree with the final eligible watermark, without carrying unbalanced work from a later partial round.
- Test zero balanced rounds, only one eligible owner, insufficient remaining time, failed projection, no reusable subtree, missing second-action support and stale fallback. A failed focused result cannot erase the incumbent. An old fallback cannot gain current-epoch support; independently supported actions cannot masquerade as a jointly sampled prefix.
- Action two receives a fresh detached snapshot after action one's actual commit. Preserve final instance/generation/request/model/profile identity, revision, actor/phase, cancellation and necessary legality checks on the main thread.
- Load during action-one search, between actions and during action-two validation. Loading invalidates all old trees, clocks and validation receipts and starts from the latest completed-turn checkpoint. Same-live-turn retries retain their original clock.
- Preserve one checkpoint after a human clicks End Turn and one after each AI completes its whole turn, including the next-turn boundary. No saves during search or individual actions. Include the city-click/Cancel regression and mixed per-seat difficulty smoke coverage using isolated saves.

Correctness failures block promotion. Performance gains never justify accepting stale data or reducing these checks.

## 8. Short evaluation protocol

### 8.1 Fixtures and bounded run size

Choose **eight ordinary two-action roots**, frozen before timing, from the existing native/local fixture corpus. Use four development roots and four held-out roots, with coverage across:

1. Early Canal, an obvious first action and several second-action alternatives.
2. A loan-then-build or similarly dependent two-action combination.
3. Dense Canal with competing resource/payment choices.
4. Development with meaningful follow-up construction choices.
5. Rail double-link construction and resource/beer constraints.
6. Multiple sales, merchant beer and bonus choices.
7. Dense late Rail with high branching and expensive legal generation.
8. A first-action ambiguity or near-era/endgame case where premature commitment can hurt.

These are coverage requirements, not invented fixture IDs. Record actual fixture paths/hashes and expected conditions in the protocol; add a small authored fixture only if the existing corpus lacks a required case. Include anticipated switch and no-switch roots before observing results; never remove roots merely because they fail to activate the optimization. Keep one-action opening, forced choices, debt and chance/information-change cases in the separate correctness smoke suite, not in the ordinary two-action throughput aggregate.

The main matrix is **8 roots × 3 configurations × 2 methods × 1 repetition = 48 whole-turn trials**. Their combined search allowances total **320 seconds**, excluding warmup/handoff overhead. Split them into small independently recorded batches with a **60-second watchdog per batch**, normally one or two A/B pairs. Do not run a long game session or a tournament.

Alternate A/B order across fixtures/configurations; predeclare one seed pair per fixture. Reuse the same initial permitted input and seed policy for paired methods, but do not claim identical trajectories once decisions diverge. Warm both models equivalently and reset per-turn trees/caches consistently between arms; preserve reuse within a candidate turn. No simultaneous A/B execution that would introduce CPU contention.

At most one predeclared second seed/order pass on two diagnostic roots may resolve an inconclusive result. Do not keep adding trials until a preferred result appears. Freeze the gate, cost estimates and reuse policy after development cases, before opening held-out results.

### 8.2 Component ablations

Use two preselected development roots and the same three settings for a small additional arm, **`search_v2_focus_only`**: identical adaptive gate, internal successor construction and incumbent handling, but start a fresh logical tree at the second root while keeping the same permitted NN-cache policy. The gate's affordability estimate must not depend on a reuse hit, so this arm differs only after switching.

- `search_v1` versus `search_v2_focus_only` diagnoses adaptive second-root refinement and its early choice of a first action.
- `search_v2_focus_only` versus full `search_v2` diagnoses statistical/structural retention under the same gate.
- Record the switch checkpoint, selected first path, remaining time and any different decisions. Timed runs can diverge; do not report different successor states or different gate outcomes as a pure reuse speedup.

This adds **six whole-turn trials, 40 seconds of combined search allowance**, reusing already captured control rows. If the switch points/decisions differ, use a tiny fixed-work continuation microcheck with the same recorded first-phase checkpoint, projected root, fallback and draw tape to isolate avoided work. Label it separately; it is not a whole-turn result and cannot receive an unreported fresh allowance. Count old visits separately from new work.

Measure the checkpoint refactor with switching disabled on the same two development roots at 4s/K=8: **two additional trials, eight seconds of combined search allowance**, paired with their existing `search_v1` rows. This distinguishes message/retention overhead from the benefit of changing decisions. The main matrix plus these ablations totals **56 trials and 368 seconds of search allowance**, split into short batches, excluding warmup, correctness/Unity checks and any explicitly labeled fixed-work microcheck.

Also run a bounded isolated Unity delivery smoke for each configuration, covering snapshot/validation/IPC overhead absent from a native harness and asserting that ordinary focused plans still use one search request. Use existing recorded correctness evidence for unaffected code. Stop each batch on correctness failures and retain its evidence rather than launching long recovery games.

### 8.3 Measurements

| Area | Required fields and interpretation |
| --- | --- |
| Identity | Method, gate/reuse/coordinator/protocol versions, source/build/model hashes, fixture, seed, setting, owner, internal root epoch and action ordinal |
| Budget and gate | Full allowance, requested/effective K by phase, checkpoint watermarks, stable full path, switch/no-switch reason, remaining time before/after projection, preparation estimate/actual, work cutoff and delivery deadline |
| Work | Newly completed trajectories, completed balanced rounds, recommendation-eligible work and action-two work; inherited visits reported separately |
| Reuse | Match/hit/miss reason per owner, retained/reused/discarded nodes, reused descendant evidence, pruning/reroot time and retained/peak memory |
| Inference | NN forwards, unique leaf evaluations, cache hits, repeated equivalent inputs, batch sizes and inference/encoding time |
| Decisions | Both complete semantic action paths, per-action source-root support/provenance, composite-plan validation, fallback availability/use, rejected continuations, safe stops and legal completion rate |
| Latency | Initial snapshot/startup separately; checkpoint/owner idle time; projection/reroot/focused search; one final response; first and second actual commits; exceptional replan requests; total observed turn latency |
| Host overhead | Snapshot preparation, detached full-action validation, any duplicate full-action validation, IPC, final main-thread checks/commit and presentation waits separately |
| Resources | Per-session CPU time, process CPU, compute thread count, peak memory and pending-work retirement time |

Record action two as zero newly searched trajectories when `search_v1` executes a previously supported continuation; do not imply that it had no prior search evidence. Deduplicate reused evidence so accumulated old visits cannot inflate `search_v2` throughput. Report all roots, including no-switch cases, failed preparation, zero-work roots and effective-K reductions. Give both overall results and the predeclared coverage categories; a switched-only subset is descriptive and must not be presented as an unbiased treatment effect. Use raw paired rows, medians and observed maxima; a tiny smoke set cannot establish population p95 or statistical noninferiority.

### 8.4 Decision-quality assessment

Freeze tactical expectations before observing which arm chooses what. Use small authored cases with an independent exact/reduced-game oracle where tractable, and existing rule/outcome checks for resource feasibility, known dominated actions and required two-step combinations.

For complex roots, compare complete two-action outcomes and retain a blinded, labeled tactical review or a small, fixed common-continuation diagnostic when needed. Label model-value estimates and short rollout proxies explicitly; evaluating both arms with the same g890 head is not independent proof of strength. If the first actions differ, compare the resulting whole-turn plans, not just second-action values at different states.

Short tests can establish correctness, work reuse and latency behavior, and identify tactical improvements/regressions. They cannot establish a win-rate improvement or broad playing-strength ranking. If quality evidence remains ambiguous, the conclusion is inconclusive rather than a manufactured pass.

## 9. Decision rules and deliverables

Evaluate each of `4s_k8`, `8s_k8` and `8s_k16` separately. A gain at 8 seconds does not establish a gain at 4 seconds, and a gain at K=8 does not establish one at K=16.

Recommend retaining the candidate for further local qualification when correctness passes and paired evidence shows useful improvement: tactically sound second-action improvements or less repeated work without an observed loss of critical first-action combinations or an unacceptable increase in whole-turn latency, fallback or memory. Require the gain to survive the complete native request and Unity delivery path. Confirm an apparent single-root win with the bounded diagnostic repeat before treating it as a reason to promote.

Do not require every fixture to improve by a rigid percentage. Retain beneficial optimizations with explained, bounded tradeoffs, consistent with the user's prior preference. Choose the simplest successful variant: if cache-only refinement gives the same useful benefit, do not require statistical reuse merely to complete the more complex design. Reject/revise if early focus causes clear tactical regressions, the gate rarely finds useful opportunities, matching seldom succeeds, retained memory dominates, or checkpoint/projection overhead consumes the gain. If benefits differ by configuration, recommend a configuration-specific outcome rather than a universal switch.

Proposed outputs, to be created during implementation/evaluation:

- `configs/search/search-v2-study.json`: design revision 2, frozen method/ablation IDs, three settings, gate/cost/reuse policy, fixtures, seeds, run order and batch limits.
- `scripts/benchmark-search-v2.py`: short native/host study runner using the existing pinned package and fixture infrastructure.
- `tests/RustBackend/test_search_v2.py` plus native/local harness extensions: resumable reference, timing, projected-state certification, reuse, composite evidence and lifecycle checks.
- `artifacts/search-v2/<run-id>/`: immutable protocol and source/model receipts, correctness results, raw per-turn/per-action metrics, ablations and bounded failure reproductions.
- `ai-research/SEARCH_V2_EVALUATION_REPORT.md`: per-configuration retain/revise/reject/inconclusive recommendation, measured improvements and limits.

During the study, installed `search_v1` remained the production control. The subsequent user-authorized cache-only promotion completed scoped installation/rollback verification and an actual installed-player smoke; see the [deployment record](SEARCH_V2_INSTALLATION_REPORT.md). AI setup persistence, the V4/NN table restriction, independent per-seat difficulty and completed-turn saves are preserved. Lab training and the final model require separate admission. No long evaluation sessions or lab rollout are part of this study or deployment.
