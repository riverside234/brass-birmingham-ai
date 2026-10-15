# Feature Engineering Compaction and Regrouping Plan

**Date:** 15 October 2026  
**Status:** Implemented FC0–FC3 as compact grouped v8 (Features-4 / layout-4). Two focused checks and one isolated lifecycle smoke passed; see [implementation receipt](FEATURE_ENGINEERING_COMPACTION_REPORT.md).  
**Scope:** Review all state groups and candidate-action inputs, remove selected redundancy, revise projection widths, and perform only focused debugging and one small isolated smoke.

## 1. Target and boundaries

Use the latest active implementation and this proposed target. Preserve unrelated concurrent updates. Before implementation, recheck the active feature manifest and relevant source; do not overwrite newer changes to force the counts in this document.

This plan supersedes the input layout and width targets in [GROUPED_PROJECTION_REVISION_PLAN.md](GROUPED_PROJECTION_REVISION_PLAN.md) for the next revision. Preserve the semantic goals in [FEATURE_ENGINEERING_REVISION_PLAN.md](FEATURE_ENGINEERING_REVISION_PLAN.md): finance, candidate consequences, realization, order, legal opportunities, and development into useful higher tiers.

The recommended target is:

- State inputs: **3,128** components.
- Candidate-action inputs: **327** components.
- Six learned state projection widths: **32 / 64 / 48 / 160 / 192 / 48**.
- Policy context: **12** components, passed through.
- Joint fusion: **556 -> 304**.
- Backbone: **six residual blocks at width 304**.
- Candidate encoder: **327 -> 128**, retaining its existing activation and normalization.

These are expected counts, conditional on the rule-invariance checks below. If a proposed omission loses a supported distinction, retain it and regenerate the dimension and parameter report. Never force the target using zero-filling or silently discarded facts.

The reductions combine three different decisions:

1. Lossless representation compaction: stop repeating exact values and their bounds.
2. Omission of quantities reconstructible from retained facts under the pinned rules.
3. A limited information-selection decision: remove five lower-priority history counters. Their strategic value is unproven, not known to be zero.

Projection widths are an engineering starting point, not an established optimum. Reducing redundant input components does not establish that useful latent capacity should fall in the same proportion.

No previous-model benchmark, historical checkpoint reconstruction, architecture A/B campaign, long training, or strength evaluation is required. Existing production artifacts must remain untouched.

## 2. Regroup the state by the relationships learned before compression

Keep six learned groups. Rearrange a small number of fields so closely related quantities meet before each group's first projection.

| Group | Expected input | Projection | Intended contents |
| --- | ---: | ---: | --- |
| turn_global | **31** | **32** | Era, phase, round/turn/actions, turn owner, remaining own-action opportunities and shared settlement applicability. |
| cards_history | **218** | **64** | Own hand, public discards, known wildcards, selected strategic history counters, deck and wildcard supplies. |
| player_economy | **109** | **48** | Current order, public player statistics, financial exposure, provisional order and spending differences. |
| industry_development | **1,314** | **160** | All public industry mats, current unlocked status and development targets for both relevant horizons. |
| board_resources | **1,247** | **192** | Compact slots, links, merchants, resource markets, sale feasibility, secured VP and Canal-end exposure. |
| current_action | **197** | **48** | Pending-action identity and obligations, plus phase-aware legal-opportunity counts. |
| policy_context | **12** | **12, pass through** | Existing style/intent tail, unchanged. |
| **Total** | **3,128** | **556 after concatenation** | **544 projected values + 12 context values.** |

### Required moves

- Move coal_market and iron_market from turn_global to board_resources.
- Move deck_count, wild_industry_count and wild_location_count from turn_global to cards_history.
- Move the eight board-related realization facts into board_resources: saleable_count, potential_sale_vp, beer_needed, individually_saleable_count, individually_saleable_vp, unflipped_level_one_count, unflipped_level_one_vp and secured_industry_vp.
- Move realization.own_action_slots, realization.card_supported_actions and realization.current_action_in_progress into turn_global.
- Replace the four identical finance settlement indicators with one shared settlement fact in turn_global.
- Rename industry_realization to **industry_development** to reflect its remaining contents.
- Keep current order and provisional order together with spending in player_economy.

This puts sellability beside tiles, connections, merchants and beer; draw/scout supplies beside cards; and timing beside remaining actions. It avoids making the industry-mat projection also carry unrelated map-realization summaries.

Assign each emitted component to exactly one group. Do not duplicate moved fields. Keep joint fusion after concatenation: the network must still learn cross-group relationships such as cards plus location, cash plus build cost, and remaining actions plus development.

### Width rationale

Keep **160** for industry_development and **192** for board_resources. Both retain substantial decision information; many removed inputs were repeated masks or static facts, so aggressive proportional narrowing would be unjustified. The map group also retains spatial ownership, connectivity, resources and sale relationships.

Use **64** for cards_history and **48** each for player_economy and current_action after their respective reductions. Keep turn_global at **32**: its dimensional expansion comes from useful timing facts, and this projection is inexpensive. These allocations prioritize distinct information and interaction needs, not raw input counts alone.

Do not add a seventh group, attention, gating, extra depth or a larger fusion layer in this revision.

## 3. Use field-specific encodings for state and action facts

The current seven-component Measure representation is:

    value, known, applicable, lower, lower_known, upper, upper_known

For an exact fact, the same value occurs three times. For fields whose source logic permits only exact, unknown or inapplicable values, emit:

    normalized_value, known, applicable

| Meaning | Compact representation |
| --- | --- |
| Exact value v | [v / scale, 1, 1] |
| Exact zero | [0, 1, 1] |
| Applicable but unknown | [0, 0, 1] |
| Inapplicable | [0, 0, 0] |

A compact field must reject a genuine interval or a known bound without an exact value. Do not silently erase uncertainty. Preserve existing scales, signs and exact/unknown/inapplicable semantics.

Keep the internal Measure abstraction and full semantic facts where other calculations need them. Separate the emitted neural-field manifest from internal rule facts. Use explicit per-field encoding metadata; replace uniform-length assumptions with sums of declared component lengths.

Use the full seven-component encoding for:

- State: realization.card_supported_actions.
- Actions: portion_cash_spent, portion_cash_proceeds, portion_cash_after, portion_projected_cash and portion_shortfall.

These fields can have real uncertainty or one-sided bounds. A small sample containing only exact values is not grounds for compacting them.

The expected retained state-derived block has **194 compact facts and one bounded fact**, totaling **589 components**. The expected action-derived block has **36 compact facts and five bounded facts**, totaling **143 components**.

Do not add running normalization or remove every mask merely because it is constant in a sample.

## 4. Review and reduce every state group

### 4.1 turn_global: remove repeated global context, add timing

Retain the existing 15 components for era, phase, round/turn/actions and turn owner.

Move its five market/supply components to the groups in section 2. Add:

- own_action_slots: three components.
- card_supported_actions: seven components.
- current_action_in_progress: three components.
- One shared settlement fact: three components.

Expected size: **15 + 3 + 7 + 3 + 3 = 31**.

Derive settlement using the same current rule function as the financial projections. Keep debt, final-round and era-boundary behavior consistent. Do not reinterpret a payment subchoice as an additional main action.

### 4.2 cards_history: keep card knowledge; trim bookkeeping counters

Keep:

- Own hand counts: 27 components.
- All four players' public discard counts: 108.
- Known wildcard information: eight.
- Deck and wildcard supply counts moved from turn_global: three.

Reduce neural history counters from 14 to nine kinds. Keep:

    build, develop, sell, loan, network, pass, scout, bonus_develop, liquidate

Omit only the neural counters for:

    coal, iron, beer, finish, bonus_skip

This removes **five kinds x four players x two eras = 40 components**. Payment-resource and UI-completion frequencies are lower-priority historical summaries than main decisions, development and liquidation. This is a deliberate information tradeoff: resource-use history can contain strategic information, and no importance study proves it irrelevant.

Retain the full permitted raw history for replay, audit and existing belief/knowledge calculations. Do not remove public discards, known wildcards, opponent public hand counts or distinct card alternatives.

Expected size: **27 + 108 + 8 + 72 + 3 = 218**.

### 4.3 player_economy: remove repeated masks and duplicated quantities

Keep the current order's 16 components.

For raw player statistics, omit only hidden_discards: under the inspected observation producer it is the same era-derived indicator for every player. Keep the raw observation field itself. Verify this rule remains true for the supported profile before omitting its four neural columns.

Retain nine raw statistics per player: cash, income_marker, income, vp, bonus_vp, link_tiles, hand_count, spent and debt. That is **36 components**. Keep both income_marker and actual income: their nonlinear relationship is useful context, despite being derivable.

For derived finance:

- Collapse four identical settlement facts into the one shared turn_global fact.
- Omit four public_debt facts that duplicate the retained raw debt.
- Keep negative_payment, projected_cash and shortfall for every player: **12 compact facts = 36 components**.

For provisional order:

- Keep all four provisional positions.
- Keep three opponent spending differences.
- Omit order.0.spending_difference, which is always observer spending minus itself.
- Encode the retained seven facts as triplets: **21 components**.

Expected size: **16 + 36 + 36 + 21 = 109**.

Do not remove shortfall or negative-payment summaries simply because a network could derive them. Preserve useful financial relationships, pending-debt unknowns and final-Rail order inapplicability.

### 4.4 industry_development: remove duplication, retain development value

Keep all four players' raw industry mats, including level counts and top-two tile attributes: **864 components**.

For each industry, omit these eight derived current-head fields:

    present, level, cash_cost, vp, beer,
    canal_eligible, rail_eligible, survives_canal

The observing player's retained raw mat head supplies their source information. Check normalization and absence semantics explicitly. **Keep current.already_unlocked**, which is nonconstant and useful.

For both current_target and next_era_target, omit five fields:

| Field | Reason under the current target rule |
| --- | --- |
| already_unlocked | The selected target is strictly higher than the stack head; its index cannot be zero. |
| blocked | Reconstruct from target presence and reachability. |
| survives_canal | A present strictly higher target has level greater than one. |
| delta_canal_eligible | Reconstruct from retained target eligibility and current mat-head flags. |
| delta_rail_eligible | Reconstruct from retained target eligibility and current mat-head flags. |

Remove **48 current-head facts and 60 target facts**. Retain **150 development facts**, encoded as triplets: **450 components**.

Keep target presence, reachability, removal distance, level, cost, VP, beer, both era eligibility flags, delta_cost, delta_vp and delta_beer. Preserve current and next-era horizons, missing targets, blockers and Rail next-era inapplicability.

Move all eleven realization facts as specified in section 2; do not delete them.

Expected size: **864 + 450 = 1,314**.

Do not further prune opponents' mats, higher-tier investment relationships or repeated-level information based solely on sparsity.

### 4.5 board_resources: remove fixed context and selected placed-tile properties

For the supported fixed board/rules profile, omit from neural inputs:

- Slot allowed-industry sets: **49 x 6 = 294 components**.
- Merchant bonus identities: **9 x 4 = 36 components**.

Their contribution is constant only if their identities are fixed by position under the supported rules. Verify this from board/rule construction; sample constancy is insufficient. Pin the supported static contract at admission. Keep the complete raw facts for rules and action effects. Retain a field if supported setups can vary it.

Keep merchant accepted-industry types and available beer, which can vary.

Omit the standalone slot.present bit: **49 components**. Occupied slots must have exactly one retained relative-owner bit; empty slots must have four zero owner bits. Keep all four owner bits and six industry-type bits.

For placed tiles only, omit:

    coal, iron, developable, canal_only, rail_only

This removes **49 x 5 = 245 components**. Confirm the retained type/level and pinned catalogue determine these properties, or establish that no omitted live distinction is needed. Check same-type/level variants and the relevant pending-payment, overbuilding and era-transition rules. If that proof fails, keep the necessary distinction.

Keep these eight placed-tile properties:

    level, flipped, resource, vp, link_vp, income, cost, beer

**Keep cost:** current liquidation proceeds use tile.cost // 2. Keep current resource stock, sale beer requirements, ownership, scoring and income. Preserve full raw tiles and pending resource obligations.

Give raw mat tiles and placed board tiles separate encoding lists; changing a shared tile-field list must not accidentally prune unbuilt mat information.

Expected size:

| Component | Dimensions |
| --- | ---: |
| 49 slots x (4 owners + 6 industry types + 8 tile properties) | 882 |
| 69 links x 4 owners | 276 |
| Nine merchants x (6 accepted types + beer) | 63 |
| Eight compact realization facts | 24 |
| Coal and iron markets | 2 |
| **Total** | **1,247** |

Do not collapse spatial information to global totals.

### 4.6 current_action: retain pending identity and deduplicated opportunities

Keep the existing pending-action block: **155 components**, comprising kind, target, merchant and four part/resource counts.

For the opportunity summary:

- Omit main_action_available, already determined by the retained phase.
- Omit each kind's available fact; its value is determined by targets > 0 when applicable and known.
- Keep all 14 target-count facts as triplets: **42 components**.

Expected size: **155 + 42 = 197**.

Preserve applicability and unknowns when reconstructing availability. No main-action candidates during a resource subchoice must not become evidence of no future main-action opportunity. Keep the existing opportunity deduplication semantics and all actual legal policy rows.

Do not remove rare pending categories just because a small corpus does not exercise them.

## 5. Candidate-action input analysis and reductions

### 5.1 Inventory and retained categorical information

The inspected candidate input is **548 components**:

| Block | Current components | Decision |
| --- | ---: | --- |
| Action kind | 14 | Keep all kinds. |
| Target identity | 128 | Keep slots, links, merchant resource sources and market distinct. |
| Selected sale merchant | 9 | Keep; it has a different role from a resource-source target. |
| Selected industry | 6 | Keep. |
| Spent/discarded card counts | 27 | Keep physical multiplicity, including Scout's three cards. |
| 52 derived action facts x seven components | 364 | Use selective omission and per-field encoding below. |
| **Total** | **548** | **184 categorical components remain unchanged.** |

Do not merge targets, remove card alternatives, deduplicate legal actions or change choice IDs to reduce dimensionality. Categories identify different decisions even when immediate numeric consequences happen to match.

### 5.2 Omit eleven reconstructible derived facts

| Facts to omit | Count | Reconstruction and scope |
| --- | ---: | --- |
| coal_used, iron_used, beer_used | 3 | Each is the indicator for that resource-choice kind. Keep the action-kind encoding and all portion resource commitments. |
| effects.1.immediate_vp, effects.2.immediate_vp, effects.3.immediate_vp | 3 | Always zero for a single chosen portion in the inspected rule implementation; direct merchant VP goes to the actor. Automatic turn/era advancement is outside this accounting basis. Preserve other players' flip, income and secured-era-VP effects. |
| develop.removed_tiles | 1 | One guaranteed removal for Develop or bonus_develop; otherwise inapplicable. Recover from the retained kind. |
| develop.free | 1 | Determined by bonus_develop versus ordinary Develop, with development applicability recovered from kind. |
| develop.result.survives_canal | 1 | Recover from retained resulting-head level > 1 and its applicability. |
| develop.removals_before, develop.next_era_removals_before | 2 | A known strict-higher target requires at least one removal; before = retained after + 1. Unknown/inapplicable status is copied by the current source logic. |
| **Total** | **11** | **Keep internal semantic facts if callers still use them.** |

For removal distance, retain the existing meaning: remaining removals to the original selected target after one guaranteed removal. Do not recompute it against a newly selected higher target after the action, and do not assume an unselected second development happened.

The opponent immediate-VP omission is pinned to the current incremental-effect rules. A future rule/profile allowing direct VP awards to another player requires a new contract, not silent zeroing.

### 5.3 Retain financially and strategically useful relationships

Keep:

- cash_spent and cash_proceeds as separate gross quantities.
- cash_after, projected_cash, shortfall and debt_after.
- portion_coal, portion_iron and portion_beer.
- All five portion financial fields and their full bounds.
- merchant_free_develop as an explicit direct opportunity.
- All four players' flips, income_marker_delta, income_delta and secured_era_vp_delta.
- The observer's immediate_vp.
- Both retained removal-after distances, immediately_unlocks, and resulting-head level, cost, VP, beer and era eligibility.

Some are mathematically derivable from state plus other action inputs. They remain useful candidate-specific relationships, and removing all such features would undo the purpose of the original feature revision.

In particular, do not merge incremental-choice and determined-portion accounting. A build root, a resource payment, a second railway link and an unresolved merchant bonus have different already-paid and still-outstanding amounts. Keep immediate VP separate from secured era VP.

### 5.4 Compact encoding and candidate width

Of 52 derived facts, retain **41**:

- **36** exact/unknown/inapplicable facts x three components = **108**.
- **Five** genuinely bounded portion-finance facts x seven components = **35**.

Expected action dimension: **184 + 108 + 35 = 327**.

Keep the candidate encoder width at **128**. The reduced input still combines target and card identities with several accounting bases and four-seat effects. There is no evidence that another simultaneous bottleneck reduction is worthwhile. Use one ordinary candidate projection; do not introduce a second grouped architecture for actions in this revision.

Preserve the existing context-local effect cache and distinct policy rows for equivalent card alternatives. Do not cache dynamic costs by choice text alone.

### Planning-time evidence and limits

Read-only source inspection and a pure-rule audit covered **14 saved permitted positions and 1,014 candidate actions**, including all 14 kinds. The eleven proposed omissions satisfied the relationships above in that corpus.

The corpus's observed non-exact bounds were limited to portion_shortfall. Source logic and existing focused cases also allow genuine intervals for the other four portion-finance fields. Keep all five full representations regardless of sample frequency.

This audit is not a feature-importance study or an implementation test of the proposed encoder. It does not establish playing-strength preservation or speedup.

## 6. Model architecture and expected budget

Use the existing grouped architecture with updated dimensions:

    state [B, 3128]
      -> one static gather and six group splits
      -> Linear(group_input, group_width) -> GELU for each group
      -> concatenate 544 projected values + 12 policy-context values
      -> Linear(556, 304) -> GELU
      -> six residual blocks at width 304
      -> final LayerNorm(304)

    candidates [B, A, 327]
      -> Linear(327, 128) -> GELU -> LayerNorm(128)

    policy scorer:
      concatenate state context and candidate embedding
      -> Linear(432, 128) -> GELU -> Linear(128, 1)
      -> existing legal mask

Residual block:

    x + Linear(304, 304)(
          GELU(Linear(304, 304)(LayerNorm(304)(x))))

Preserve zero initialization of the last Linear in each residual branch. Keep official outcome heads, utility and personality integration, activation/normalization behavior and observer-relative mappings. Encode the state once per observation/model, not once per candidate.

Expected medium-model counts if the specified omissions are admitted:

| Component/model | Parameters |
| --- | ---: |
| Six group projections | **479,840** |
| Joint fusion | **169,328** |
| Six residual blocks | **1,116,288** |
| Final state LayerNorm | **608** |
| State encoder | **1,766,064** |
| Base model | **1,904,445** |
| Utility model | **1,914,465** |
| Personality model | **1,980,298** |

Expected base dense work is **1,797,760 MACs per position + 97,280 per padded candidate**, multiplied by batch size. This excludes encoding, gathering, activation, normalization, queueing and search.

Generate exact counts from the implemented manifest rather than copying these numbers into validators. Keep current explicit resource limits; do not impose a new 2M limit or enlarge other layers just to consume saved parameters.

Tensor compaction can reduce memory movement and neural work without removing Python rule derivations. Avoid a promised laptop speedup. Profiling is needed only if focused debugging reveals a concrete performance problem.

## 7. Contract, data and downstream integration

Relevant source touchpoints:

| Area | Files |
| --- | --- |
| Field metadata and Measure encoding | src/brass_ai/feature_engineering/contracts.py |
| Finance, order and opportunities | src/brass_ai/feature_engineering/finance.py |
| Development and realization | src/brass_ai/feature_engineering/industry.py |
| Incremental and determined-portion effects | src/brass_ai/feature_engineering/effects.py |
| Rule facts and context/caches | src/brass_ai/feature_engineering/rules.py and context.py |
| Reference and optimized encoders | src/brass_ai/learning/features.py; src/brass_ai/recovery/encoding.py |
| Layout, projections and model contract | src/brass_ai/learning/state_layout.py and model.py |
| Raw public observation production | src/BrassHost/Observation.cs |
| Utility/personality consumers | src/brass_ai/learning/utility_model.py; src/brass_ai/personality/model.py |

Implementation requirements:

- Assign unused feature, derived-encoding, layout and architecture identities as required. Do not overwrite identities already used by another feature variant.
- Make one declarative manifest authoritative for retained fields, component lengths, scales, group membership and dimensions.
- Update both encoder paths, hardcoded shape checks, action dimensions, batching/shared memory, model loading, export and runtime metadata.
- Keep strict coverage checks for semantic facts and emitted fields; do not use arbitrary filtering to hide missing or unexpected facts.
- Bind layout identity to exact group membership as well as shape. Equal shapes do not guarantee equal semantics.
- Preserve raw observation and legal-choice wire contracts, candidate order/IDs, labels, seat mappings, policy context and privacy.
- Preserve deadlines, cancellation, cache bounds and complete context/source identities.
- Keep rules and action-effect calculations that need fields omitted from the neural tensor.

Use a fresh compatible IL model and a separate self-play output directory unless an exactly target-compatible checkpoint already exists. Reject incompatible checkpoints and encoded caches clearly. Do not slice weights, mislabel existing behavior probabilities, reuse incompatible teacher targets, reset an existing run automatically or pretend this is ordinary checkpoint continuation.

Re-encode compatible raw data only through the existing explicit audit/admission workflow and into new output paths. Same-target continuation must preserve optimizer/RNG, progress, population, curriculum and lineage.

Carry the schema through IL, utility calibration, ordinary PPO, teacher/inference workers, evaluation, personality and production export. Preserve the latest effective training recipe, including learner exploration, rewards, curriculum, teacher collection settings, distillation and quality/freshness safeguards. Do not copy stale settings from earlier plans.

Keep search_wall_fraction as **None in Python / null in JSON**. Only the cumulative wall-fraction restriction is disabled; per-turn budgets, cancellation and failure timeouts remain active.

## 8. Minimal required debugging

The purpose is to find mistakes in the changed layout, accounting and integration. Do not create a broad new validation campaign.

### A. One focused semantic/encoder check

Reuse a small selection of existing fixtures, approximately 6–10 positions covering the changed branches. Combine the checks in one pass:

- Exact/zero/unknown/inapplicable encoding, plus an actual bounded cost/proceeds case and the card-supported-action interval.
- Omitted-field reconstruction and the source-proven static board/tile assumptions.
- Reference/optimized encoder parity; complete, non-overlapping group coverage; correct state/action dimensions and legal-row identities.
- A few meaningful boundary cases: repeated-tier development, an unresolved payment/merchant choice, debt or liquidation, and an era boundary.

Use source/rule reasoning for universal invariants and existing fixtures for regressions. Do not substitute many random seeds for a missing proof. Keep the checks at the changed boundary; do not add tests that merely restate the manifest.

### B. One tiny model/debug batch

Use a small batch to exercise finite forward/backward behavior, candidate padding/legal masks and gradient flow through the changed projections. Check computed parameter counts and output mappings.

Exercise the applicable utility/personality entry points with minimal construction/forward checks, not separate training runs. Use synthetic mismatched metadata to confirm incompatible artifacts are rejected; historical neural checkpoints are unnecessary.

### C. One isolated lifecycle smoke

Use one separate scratch output directory and the latest target only:

- A tiny IL/update fixture and the affected handoff path.
- At most one ordinary-game collection and one PPO update if needed by the existing harness; use actual probabilities from the same target policy.
- One matching checkpoint save/load with optimizer/RNG and progress checks.
- One schema/leaf-inference check for the teacher worker; no timed teacher collection is required.
- One export/load parity check for the affected production path, with one or two small shapes sufficient to catch padding/dynamic-shape bugs.

Reuse records and setup across these steps. If an existing harness would require substantial training, replace that portion with a narrow fixture-level check and state what was not exercised. Do not launch multiple self-play iterations merely to complete the smoke.

### Stop rule

Stop when these focused checks and the single smoke pass. Broaden testing only to diagnose an observed failure or verify a necessary fix.

Do not require the full repository suite, seed sweeps, per-phase timed searches, teacher-label collection, performance benchmarking, strength games or old-model comparisons. Do not rerun passing checks after documentation-only changes.

Record pass/fail results, elapsed smoke time, actual update/game counts and unexercised paths. Profile only a specific observed slowdown, allocation problem or timeout. Preserve normal runtime validation and label-quality guards; reducing development test scope does not remove production safeguards.

## 9. Implementation sequence and deliverables

| Stage | Work | Sufficient completion evidence |
| --- | --- | --- |
| FC0 — Confirm the field decisions | Reinspect the active code; verify removal dependencies and static assumptions. | Concise keep/remove/move/compact ledger with source reasons; retain exceptions. |
| FC1 — Implement the new contract | Generate the manifest, new identities, group assignment and both encoders; update model consumers. | Consistent dimensions, layouts and parameter counts. |
| FC2 — Debug the affected paths | Run the focused checks and one isolated smoke from section 8; fix observed bugs. | A short debugging receipt and explicit unexercised paths. |
| FC3 — Update instructions | Align active grouped-projection/lab documentation and new-run commands with the final manifest. | One authoritative target and exact commands; existing production artifacts preserved. |

The final implementation report should state:

- Final state/action dimensions, group membership, projection tuple and model counts.
- Actual retained/omitted fields, including any departure from these conditional targets.
- Focused debugging and smoke results, with their small measurement scope.
- Required new-data/new-model bindings and same-target continuation instructions.
- Any remaining correctness uncertainty or unexercised integration path.

Do not claim better playing strength, preserved strength, optimal widths or a measured deployment speedup from this feature audit or smoke. The objective is a simpler, coherent representation with explicit semantics and a working latest-model pipeline.
