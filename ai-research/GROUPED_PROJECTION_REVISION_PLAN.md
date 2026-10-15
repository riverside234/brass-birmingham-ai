# Grouped State Projection Revision Plan

> Historical v7 document. The active target is now [compact grouped v8](GROUPED_PROJECTION_CONTRACT.md), implemented from [the compaction plan](FEATURE_ENGINEERING_COMPACTION_PLAN.md). Earlier evidence below remains historical.

**Date:** 15 October 2026  
**Status:** Implemented and locally validated as architecture v7 / layout 3.0.0. The target-only tests, isolated smoke, exports and absolute profile are complete; see the linked implementation report.

Use **32 / 80 / 64 / 256 / 256 / 64** for the six learned state groups, plus the unchanged **12-dimensional policy-context tail**. The complete state pathway uses **764 -> 304 joint fusion** and **six residual blocks at width 304**, followed by the existing action and output pathways.

This plan covers only the latest model and the allocation below. Implement and validate this target directly. Do not prepare, restore, train, export or benchmark earlier model versions for comparison, and do not require historical model artifacts to complete the task.

The larger industry and board projections are an explicit capacity choice. Large input dimensions motivate retaining more information before fusion, but do not by themselves establish that a particular width is sufficient or that playing strength will improve. Editing this plan does not modify model code, configurations or training runs.

## 1. Inspect the latest implementation and preserve scope

Before implementation, inspect model.py, state_layout.py, the active frozen layout manifest, applicable generated configurations and current inference/export code. Record the latest source and configuration identities for reproducibility. Preserve concurrent unrelated updates and reuse completed grouped-projection infrastructure.

If the target is already implemented, verify it and complete only missing integration or validation.

Preserve:

- Features-2 state/action dimensions **5,489 / 548**, feature values, masks, normalization, index order and observer-relative semantics.
- Existing six semantic groups and the final style[4]/intent[8] tail.
- Backbone width **304**, residual depth **6**, existing residual initialization and final LayerNorm.
- Action encoding, policy scoring, official outcome semantics, utility calibration and personality/goal integration.
- Current PPO, teacher, evaluation, curriculum, opponent-search and deployment settings.
- Existing datasets, checkpoints, source-review receipts and run directories.

The architecture change is limited to the specified group widths, resulting fusion input size and necessary integration. Do not add hand matching, residual depth, an additional dense intermediate layer, attention, GNNs, Dropout or new normalization. Do not modify the installed game AI.

## 2. Required width allocation

Use this exact ordered allocation:

| Group | Input dimensions | Target width | Projection parameters, including bias | Purpose |
| --- | ---: | ---: | ---: | --- |
| turn_global | 20 | **32** | 672 | Modest nonlinear expansion for era, phase, turn state and global counters. |
| cards_history | 255 | **80** | 20,480 | Represent own hand, public discards, known wildcards and action history before fusion. |
| player_economy | 252 | **64** | 16,192 | Represent public finance, debt and order, including correlated derived facts. |
| industry_realization | 2,747 | **256** | 703,488 | Provide substantial capacity for industry mats, development targets and realization information before cross-group fusion. |
| board_resources | 1,845 | **256** | 472,576 | Provide substantial capacity for distributed slot, ownership, link, merchant and resource combinations. |
| current_action | 358 | **64** | 22,976 | Represent pending steps and legal-opportunity summaries alongside the separate candidate-action encoder. |
| policy_context | 12 | **12, pass through** | 0 | Preserve all style and intent values without a separate learned projection. |

The six learned groups produce **752 components**. Adding the **12** context values gives **764 fusion inputs**. The main backbone remains **304** wide.

Feature composition to preserve:

- cards_history: 27 own-hand components, 116 public discard/wildcard components and 112 action-history components.
- industry_realization: 864 industry-mat components, 1,806 development components and 77 realization components.
- Development and realization include 269 derived facts, each encoded as value, known, applicable, lower, lower_known, upper and upper_known. Keep these components together.
- board_resources: 1,470 slot components, 276 link components and 99 merchant components.
- current_action: 155 pending-action components and 203 opportunity components.

Input components are not all independent facts; the allocation is a chosen representation budget, not a dimension-based proof of adequacy. Implement this single target without a width sweep or alternative architecture comparison.

## 3. Complete network specification

### State pathway

Retain the static grouping implementation:

    states [B, 5489]
      -> one registered int64 permutation / batched gather
      -> seven splits, preserving within-group field order

    turn_global:          Linear(20,   32) -> GELU
    cards_history:        Linear(255,  80) -> GELU
    player_economy:        Linear(252,  64) -> GELU
    industry_realization: Linear(2747,256) -> GELU
    board_resources:      Linear(1845,256) -> GELU
    current_action:       Linear(358,  64) -> GELU
    policy_context:       12 existing values, unchanged

      -> concatenate [B, 764]
      -> Linear(764, 304) -> GELU
      -> six residual blocks at width 304
      -> LayerNorm(304)
      -> shared state embedding [B, 304]

Each residual block remains:

    x -> LayerNorm(304) -> Linear(304, 304) -> GELU
      -> Linear(304, 304)
      -> add the original x

Preserve zero initialization of the final Linear weight and bias in each residual branch for fresh models. Six blocks contain twelve residual-branch Linear layers. Equal width applies to the 304-dimensional backbone after fusion; the concatenated representation does not determine residual width.

Joint fusion is mandatory. It combines retained group information, including hand/map and money/development relationships. The larger group representations still pass through a 304-dimensional shared representation. Widening the projections does not eliminate that final bottleneck or guarantee preservation of every input distinction.

### Candidate and output pathways

Keep these dimensions, activations and semantics:

- Action encoder, once per candidate: Linear(548, 128) -> GELU -> LayerNorm(128).
- Policy scorer: concatenate state[304] and action[128], then Linear(432, 128) -> GELU -> Linear(128, 1), followed by existing legal-action masking.
- Official outcomes: Linear(304, 128) -> GELU -> Linear(128, 12), retaining four-seat win, placement and VP transformations and seat mapping.
- Utility residual: existing remaining-budget inputs, Linear(308, 32) -> GELU -> Linear(32, 4), initialization, prior and calibration.
- Personality model: existing context injection, 128-wide goal trunk, goal/confidence heads and parent-import semantics.

Candidate processing repeats over the legal-action list. Its widths stay unchanged in this state-projection revision.

### Runtime implementation constraints

- Preserve state_encoder(states) -> [B, 304] across base, utility and personality models.
- Compute the state embedding once per observation/model, not once per candidate.
- Retain registered static indices, efficient batched gather/split operations and dynamic batch/candidate support.
- Do not reconstruct features in Python inside forward or add new feature-engineering work.
- Preserve cache correctness and update any affected model/source provenance.
- Preserve PyTorch and ONNX compatibility and current deployment precision.

## 4. Target parameter and compute budget

Retain the current **3M base/utility** and **3.1M personality** limits, including the existing personality extension limit. The selected target fits these limits; there is no requirement to reduce its parameter count to match another model.

Expected medium-model counts with the specified heads:

| Component or model | Target parameters |
| --- | ---: |
| Six group projections | **1,236,384** |
| Fusion layer | **232,560** |
| Six residual blocks | **1,116,288** |
| Final state LayerNorm | **608** |
| State encoder total | **2,585,840** |
| Base model total | **2,752,509** |
| Utility model total | **2,762,529** |
| Personality model total | **2,828,362** |

Calculate exact counts from constructed models and reconcile them with these expectations. Do not change unrelated heads merely to force a matching total. If a supported small preset is retained, verify its counts and contract separately using the same explicit group widths; the intended deployment model remains width 304.

Expected medium dense Linear work:

| Model | Fixed MACs per position | MACs per padded candidate |
| --- | ---: | ---: |
| Base | **2,617,328** | **125,568** |
| Utility | **2,627,312** | **125,568** |
| Personality | **2,692,960** | **125,568** |

For a padded batch, dense work is B * (fixed + N * per_candidate). One MAC is two FLOPs. Counts exclude gather, activations, normalization, masking, softmax, feature encoding, queueing and search. Confirm the counts using the latest implementations.

Report absolute parameter counts, work, memory and latency for the target. Do not add parameter-reduction tables, relative speedup claims or comparisons with historical networks.

## 5. Layout identity, checkpoint admission and data

Reuse the current semantic index mapping. Do not regroup inputs or alter encoded features to implement the widths.

- Every state index appears exactly once, without omissions or duplicates.
- Preserve group membership, within-group ordering, ownership and all seven components of each derived fact.
- Keep style/intent as the final 12 raw values.
- Generate the manifest from shared layout metadata, not handwritten offsets.
- Record input sizes, output widths, group order, feature identity and a deterministic layout hash.

Assign fresh, unused architecture and layout identities if the latest registered identities do not already describe this exact target. Bind the identities, widths and hash consistently in model configs, checkpoints, policy identities, exports and runtime/handoff packages. Never overwrite a published identity with different tensor shapes or layout semantics.

Keep permitted-features/2.0.0 when feature arrays and semantics remain unchanged.

Admission and continuation:

- Accept only checkpoints compatible with the target's architecture, width tuple, layout and source requirements.
- Reject corrupted index buffers, stale manifests and incorrect width metadata, including fixtures with matching total dimensions.
- Do not pad, slice, reshape or relabel weights, optimizer state or provenance to force compatibility.
- Do not use source-review flags to bypass architecture checks.
- Start the target with fresh IL and a separate self-play run unless a genuinely compatible target checkpoint already exists.
- Same-target continuation must preserve its weights, optimizer/RNG, counters, completed games, population, curriculum and lineage.
- Preserve existing on-disk runs and receipts. No backward-compatible loader, checkpoint migration or historical-model recovery work is part of this plan.

Checkpoint rejection tests can use synthetic altered metadata or tensors; they do not require loading or constructing a previous model.

Unchanged encoded features can support existing complete heuristic raw demonstrations. Follow the established explicit audit/re-cache process when required, writing separate outputs. Do not regenerate demonstration games solely because widths changed. Neural PPO behavior and teacher targets must still satisfy their own provenance/freshness checks and cannot use the heuristic-data exception.

## 6. Downstream integration and preserved training recipe

Update all architecture-dependent paths together:

1. Model/layout factories, preset serialization, strict validation and parameter budgets.
2. Generated configs, checkpoint save/load and architecture/layout/source identities.
3. IL dataset admission, completion and target self-play handoff.
4. Target self-play initialization, same-architecture resume and inference workers.
5. Teacher model loading, utility calibration, source freshness and policy/cache identities.
6. Personality parent import, named outputs, ONNX export and packaged layout metadata.
7. Focused regressions, target layout contract, architecture-budget report and English lab instructions.

Generate fresh-run configuration from the actual current latest-model recipe, currently configs/step5/joint7-latest.json. Record the resolved configuration and verify that only architecture and required target-run/checkpoint/dataset bindings change.

At inspection, preserve:

- Ordinary learner exploration **0.08**, temperature **1.0**, family/industry exploration distribution and probabilistic sampling.
- Entropy/value weights **0.005 / 0.5** and other PPO/optimizer settings.
- Latest-model four-seat collection, learner-only collection against frozen opponents, current curriculum and disabled champion gate.
- Reward coefficients and insolvency event_cost **0.005**, max_cost **0.075**.
- Teacher games per collection **5**, cadence, phase reservations, distillation steps **8**, minibatch/reuse limits and label-quality safeguards.
- Teacher clocks **12 / 12 / 16 / 24 seconds** for early Canal / late Canal / early Rail / late Rail.
- search_wall_fraction **None in Python / null in JSON**: disable only the cumulative teacher-versus-ordinary allowance; retain shared per-turn deadlines, cancellation and failure timeouts.
- Teacher source freshness, utility calibration and sample-use reporting.
- Current opponent budgets, monitor/evaluation settings, retention and population rules.
- Deployment two-worker search and **2 / 5 / 12-second** difficulty budgets, distinct from training worker settings.

If the effective recipe has subsequently changed, preserve and report that latest recipe rather than reverting it to the recorded settings. Keep defaults, validation and runtime reporting aligned.

Bind initialization and rehearsal to the selected target-compatible IL checkpoint/dataset through the existing generator. Production curriculum and opponent rules remain outside this architecture change; they do not create a requirement to benchmark an earlier model.

## 7. Focused tests and bounded integration smoke

### Required tests

- Exact tuple (32, 80, 64, 256, 256, 64), input split sizes, 752 learned outputs, 12 passthrough values, 764 fusion inputs and 304 state outputs.
- Six residual blocks with unchanged internals, initialization and final normalization.
- Complete/disjoint/deterministic grouping and parity between the reference and optimized feature encoders.
- Unchanged legal masks, choice order/IDs, privacy, seat mappings, style/intent and unavailable-fact handling.
- Shape-correct forwards and gradient flow through every group projection and joint fusion.
- Exact base, utility and personality counts and output-head behavior.
- Variable batches/candidate lists, padding, legal masking and one state encoding per observation/model.
- Rejection of wrong width metadata, altered indices/order, incompatible identities and stale manifests using target-model fixtures.
- Target checkpoint save/load and same-architecture optimizer/RNG/progress resume.
- PyTorch/ONNX parity for the same latest checkpoint, all named outputs, dynamic batch/candidate dimensions and legal masks.
- Existing PPO behavior-probability, forced-action filtering, source/cache admission, teacher-clock and retention safeguards.

Use relevant focused tests and existing integration coverage. Do not create an earlier architecture merely to obtain a test fixture.

### Isolated smoke

Use a new output directory and the latest target model only. Keep all reduced smoke settings separate from production defaults.

1. Audit/reuse a tiny complete heuristic train/validation dataset through the established mechanism.
2. Train a bounded fresh target IL model and complete its normal handoff.
3. Run two small PPO iterations with save/resume between them, using latest-model self-play for the smoke; verify optimizer/RNG and committed progress at the resume boundary.
4. Run up to four additional functional self-play games with the latest model, reporting game completion, legality and failures.
5. Probe neural teacher inference in each phase using the target checkpoint and unchanged shared turn budgets.
6. Export the latest checkpoint and verify CPU runtime parity, including utility and personality integration.

This smoke verifies pipeline execution and does not rank the model against another architecture. Report any production curriculum paths not exercised by the latest-only smoke.

Report seeds, dataset/split scope, games, iterations, runtime, errors and smoke-only batch/epoch changes.

For teacher probes, report per-phase attempts/completions, qualified labels, insufficient coverage, timeouts/overruns and elapsed time. Report generated labels separately from distinct labels trained, presentations/reuse and unused labels. If no distillation update runs, report zero training usage and teacher-induced policy drift and explicitly mark distillation training as unexercised.

Retain all deadline, freshness and label-quality requirements. Distinguish a quality/deadline rejection from an implementation failure. Do not launch a long training run, full teacher campaign or expensive playing-strength evaluation.

## 8. Profile only the latest target model

Measure the latest model's absolute performance. Use only checkpoints, exports and runtime packages matching this target's identities. Numerical parity between PyTorch and ONNX must use the same checkpoint.

Measure separately:

- Feature encoding cold/miss/hit paths.
- Pure PyTorch and ORT forward latency, including state gather, projection and fusion.
- Complete inference requests with queueing, batching and candidate processing.
- Model load/cold startup and warm execution.
- Memory where available, distinguishing serialized weights, Python allocations and process memory.
- Completed search work and elapsed time under the configured whole-turn budgets.

Use a fixed permitted-position corpus spanning both eras, small/large legal sets and realistic batch sizes. Preserve candidate order and record seeds. Repeat measurements with consistent warmup and report medians, tail latency, variability and sample counts.

For laptop deployment prioritize **ORT CPU with two threads**, recording actual hardware and runtime versions, while retaining the configured two search workers. Keep deployment and training settings distinct. Report completed simulations per turn and per second, incomplete work, cache conditions and deadline behavior.

There are no cross-version A/B arms, historical performance tables, earlier-model artifact requirements or comparative speedup acceptance criteria. Do not copy measurements from another architecture into this report.

An untrained-model timing probe establishes latency only. A functional smoke, parameter count or absolute search-completion count does not demonstrate improved playing strength. Describe only measured target behavior and its limits; no relative strength claim is required for this task.

## 9. Implementation sequence, deliverables and acceptance

Implement in this order:

1. Inspect the latest source/configuration and record their identities.
2. Implement the specified widths and publish matching immutable layout/architecture identities.
3. Construct the latest model variants and verify dimensions, parameters and dense-work counts.
4. Complete IL/self-play/teacher/personality/export integration and admission checks.
5. Run focused tests and the isolated latest-only smoke.
6. Profile the latest model's latency, memory and search work.
7. Update the target contract, report and English lab commands.

Deliver:

- The latest allocation and full network diagram with exact architecture/layout/feature identities.
- Changed-file list and resolved configuration diff showing recipe preservation.
- Absolute parameter and MAC counts for supported target presets and model variants.
- Target-only test, smoke and performance receipts with commands, seeds, environments and output paths.
- Lab commands for data audit/re-cache if required, fresh IL, handoff, new self-play, export and same-architecture continuation.
- The target's observed runtime characteristics and any untested paths.

Implementation is complete only after required checks pass. A blocked or omitted required check remains pending. Deployment readiness requires the relevant runtime and integration evidence; playing-strength improvement remains unproven without separate evidence.

## 10. Latest-target documentation and evidence status

During implementation, update the target-facing [layout contract](GROUPED_PROJECTION_CONTRACT.md), [implementation report](GROUPED_PROJECTION_REPORT.md) and [English lab instructions](../docs/FEATURE_ENGINEERING_LAB.md) to describe this exact allocation and its actual validation results. Do not import earlier model results into the target's acceptance evidence.

Store new receipts in a unique output directory identified by date and target identity. Preserve existing files without using them as comparison artifacts or requiring historical model reconstruction.

**Completed for this target:** exact allocation and immutable identity/manifest; all-variant budgets; downstream IL/self-play/personality integration; 291 passing focused tests; bounded latest-only IL/PPO/resume/functional-game/teacher smoke; same-checkpoint ONNX parity; current-target CPU latency, memory and 12 whole-turn search probes; and updated English lab instructions. Receipts are in `artifacts/feature-engineering/grouped-v7-20261015/`. Full scheduled distillation, GPU throughput, production opponent-table execution in the smoke, aggregate search-process memory, full deployment qualification and playing-strength improvement are not established by this local validation.
