# Feature engineering revision: fresh IL and self-play

**Date:** 14 October 2026  
**Status:** FE0–FE6 complete; all six groups are enabled in features 2, with bounded validation and fresh-IL lab instructions.  
**Scope:** Six required feature groups, a versioned model/data integration, and focused verification.  
**Initialization:** Fresh imitation learning followed by fresh self-play; no old neural weights or optimizer migration required.

## 1. Objective and boundaries

Help the four-player AI understand financial exposure, investment costs, industry realization, turn order, legal opportunities and development into higher-tier industries. A specific priority is learning **development -> higher-tier construction -> timely flipping/scoring** in both eras.

Implement all six groups in section 3. Higher industry levels alone are not a success criterion: affordability, beer, access, remaining actions and actual realization still matter. Do not introduce a high-tier reward, prescribed opening, handcrafted strategic score or additional action-frequency penalty.

The user will restart IL. Feature dimensions and input/model layers may change, with a new compatibility contract. Preserve the small shared MLP-ResNet approach. **User update during FE0–FE3: exceeding 2M parameters is authorized; retain width 304 as the intended new-model design.** Report backbone and full-model counts, including utility/goal extensions. FE4 must introduce an explicit versioned limit matching the final dimensions/counts; do not silently bypass an existing parameter-limit check. Existing active models/limits remain unchanged during FE0–FE3.

Preserve the currently approved reward/head semantics, search deadlines, worker/resource controls, opponent schedules, PPO settings, evaluation cadence and single-GPU performance paths. Snapshot the resolved active settings at implementation time. **Do not copy historical settings from earlier prompts.** At this document's creation, the lab guide already describes a newer `joint7-latest` recipe; its exact settings are not redefined here.

Do not modify/install the game AI, start a long training campaign, overwrite existing datasets/runs, or refactor unrelated personality/deployment code. An implementation request for one FE stage authorizes that stage and its focused checks, not a full training campaign.

## 2. Starting points and implementation hazards

Inspect these components before editing; filenames are a map, not a requirement to change every file.

| Area | Existing components | Required attention |
| --- | --- | --- |
| Reference encoding and model | `src/brass_ai/learning/features.py`, `model.py`, `utility_model.py`; `configs/learning/features-1.0.0.json` | Raw financial/order fields and top-two tile properties already exist; add relationships without replacing useful raw information. |
| Optimized encoding | `src/brass_ai/recovery/encoding.py` | `PreparedEncoder` has its own encode path. Both paths must agree exactly on the new contract. |
| Action caching | `PreparedEncoder.choice_features` | Its current cache is keyed by categorical choice semantics. That is insufficient for state-dependent costs/effects. Cache static and dynamic portions separately or use a complete permitted-context identity. |
| Observation boundary | `src/brass_ai/runtime_contracts.py`; `configs/contracts/v0.2.0/`; `src/BrassTrainingHost/` | Use permitted observations and legal choices. Avoid editing pinned wire contracts in place. |
| Data and IL | `src/brass_ai/recovery/demonstrations.py`, `replay.py`, `updates.py`; `src/brass_ai/learning/replay.py`, `training.py` | Preserve choice/label alignment and raw-format compatibility; invalidate encoded caches. Trace the active IL entry point rather than assuming one path. |
| Inference and search | `src/brass_ai/recovery/inference.py`, `execution.py`, `shared_transport.py`; `src/brass_ai/search/` | Preserve dynamic batching, tensor dimensions, observer privacy, admitted shared deadlines and revision-safe caches. |
| Training/evaluation | `src/brass_ai/recovery/league.py`, `ordinary.py`, `teacher_collection.py`, `evaluation.py`; `scripts/step5.py` | Same encoding everywhere; preserve reward, scheduling and behavior-policy provenance. |
| Exports/personality consumers | `src/brass_ai/learning/export.py`; `src/brass_ai/personality/` | Check dimension/version assumptions and require compatible artifacts. No automatic requalification of old rosters or packages. |
| Tests/docs | `tests/R2/`, `tests/Recovery/`, `tests/Step5/`, `tests/MaskedISMCTS/`; `docs/TRAINING_RECOVERY_LAB.md` | Add focused regressions and fresh-training instructions. |

The current raw features include cash, actual income, income marker, debt, spending, order, tile-level counts and top-two tile attributes. Derived features should explain relationships among them. Inspect the full permitted tile stacks before assuming that additional wire data is needed.

## 3. Required feature contracts

For every new field record: name, owner/seat convention, source, units, normalization, applicability, exact/bounded/unknown meaning, and ordering in the tensor. Use finite values plus validity/applicability masks. Unknown, inapplicable and a genuine zero must not be confused. Use fixed documented scales; preserve signed values. No running normalization or feature changes to the reward.

### A. Financial safety — state

For all four players, in the established observer-relative order:

- Whether a further relevant income settlement will occur.
- Payment due from negative income at that settlement.
- Signed projected cash after that settlement, holding current cash/income fixed.
- Cash shortfall if the projection is negative.

For an applicable ordinary settlement, the basic projection is `cash + income`, payment is `max(0, -income)`, and shortfall is `max(0, -projected_cash)`. Bind these to the actual engine phase: pending debt, already processed settlements and final/era boundaries require explicit handling. Do not charge a past settlement again.

This is a current-position projection, not a guaranteed future balance or recommended reserve. Do not encode an arbitrary rule such as always retaining a fixed amount of cash. If an unresolved debt phase prevents an exact projection, retain the public debt information and mark the affected projection appropriately.

### B. Candidate costs and direct effects — action

For each existing legal candidate, expose applicable information:

- Cash expenditure and immediate cash proceeds separately.
- Coal, iron and beer consumption.
- Resulting cash and financial-safety projection where determinable.
- Guaranteed flips, income-marker/actual-income changes and immediate bonuses.
- Immediate VP awards separately from industry VP that would be scored at an era boundary.

Define the accounting basis: incremental choice, determined action portion, or remaining complete action. Never mix bases in the same field. Payments/resources already committed by earlier subchoices must not be charged twice.

For unresolved resource sources, second links, merchants or other subchoices, use exact values only when justified. Otherwise provide valid bounds and known/bounded masks. An unresolved upper bound is unknown, not an optimistic estimate. A complete action's base price must not masquerade as its total cost.

Use rule-derived calculations shared across candidates. Do not introduce full searches, exhaustive action completion or simulator clones per candidate merely to populate this block. Engine transitions are the test oracle; permitted observations are the production input.

### C. Industry realization and era-end exposure — state

For the observing player:

- Number of unflipped saleable industries, their potential flip/era VP and beer needs.
- Number currently individually saleable through a compatible merchant with legal beer access.
- Number and potential VP of unflipped level-one industries removed at Canal end.
- Remaining own action opportunities before the relevant era ends, with a precise definition for the current partially completed turn.

Separate sales industries from coal, iron and breweries: their flip conditions differ. Distinguish potential VP, currently secured era score and already recorded cumulative VP. Do not sum the same points twice.

Individual feasibility does not establish joint/group-sale feasibility. Several industries may share the same beer. Full group-sale enumeration is not required; reuse a verified existing joint summary only if cheap. Do not mark an industry unsaleable solely because the current UI subphase contains no Sell candidate. Obtain the bounded current-state fact or explicitly mark it unavailable.

Use the actual turn/era rules for action counts, including opening turns, remaining cards and pending subchoices. A resource selection is not another main action. Future resource availability must not be treated as guaranteed.

### D. Provisional next-round order — state

- Each player's provisional next-round position using spending so far.
- Observer-to-other-player spending differences with a documented sign convention.

Apply official tie resolution and current-order rules. Label this as provisional; opponents can still spend. Handle final-round and era-boundary applicability explicitly. Current order is already encoded and need not be duplicated.

### E. Legal-opportunity summary — state

- Availability of each relevant candidate/action category.
- Deduplicated effective opportunity/target counts by category.

Aggregate the already generated legal list. Document category-specific deduplication keys: equivalent representations/card multiplicity must not inflate opportunity counts, while materially different opportunities remain distinct. This summary must not remove or alter the policy's actual legal candidates or labels.

Keep the block phase-aware. No Build candidates during coal payment does not mean no building opportunities at the main-action level. Do not generate another full legal list just for this summary or compute opponents' private legal choices.

### F. Development progress and higher-tier investment — state and action

This is mandatory and central to the objective, not a later optional experiment.

For each of the observing player's six industry types, use the actual public stack order and era rules to describe:

- The current next-build tile under stack/era rules.
- The next strictly higher-tier target reachable through legal development and usable in the current era.
- Where relevant, a strictly higher-tier next-era target for Canal-to-Rail preparation.
- Minimum tile removals needed to expose each target; already unlocked, reachable and blocked/missing status.
- Target level, construction cash cost, VP, beer requirement and era eligibility.
- Changes in cost, VP, beer requirement and era eligibility relative to the current next-build tile, where that reference exists.
- Whether the target survives Canal-end industry removal under the rules.

Choose targets deterministically by stack/level order, not a heuristic profitability score. Bound the set to at most one current-era and one next-era target per industry. Do not enumerate every long-term build/development sequence. Reuse raw attributes already present where that preserves a clear contract.

Distinguish **stack/era unlock** from having cash, cards, resources and a vacant location for construction. Do not invent opponents' hand-based opportunities. Multiple copies at one level count individually; undevelopable blockers cannot be skipped by assumption. A target may be unreachable through development alone even if building intervening tiles could eventually expose it.

For Develop and merchant-bonus development candidates, encode:

- Which stack advances and remaining removals before/after the guaranteed choice.
- Whether that choice immediately unlocks a higher tier.
- The resulting next-build tile's cost, VP, beer need and era eligibility.
- Determinable iron/cash cost under group B's conventions.

Do not assume an unselected second development has occurred. Distinguish development tile count from development action count, including free merchant development. Productive progress toward a higher tier must remain visible even when a single choice does not unlock it yet.

The model should learn when this investment pays off; a high-level build without time or beer to flip is not automatically desirable. Preserve profitable lower-tier options and all original legal actions.

## 4. Shared implementation rules

### Privacy and correctness

- Derive features from the acting observer's permitted view, public rule constants and admitted legal choices only.
- Do not accept an unrestricted simulator/checkpoint object at the feature boundary.
- An opponent node in search uses that actor's permitted observation; do not reuse root-private features across seats.
- Keep stable seat mapping, action IDs, labels, masks, official predictions and remaining-utility semantics intact.
- Cash, resource, income, spending and scoring calculations must match the project's declared rules, not duplicated approximations.

### Computation and caching

- Build one bounded derived context per observation; share financial, connectivity, industry-stack and market calculations across candidates.
- Reuse existing revision-safe graph/rule facts. Do not run an extra search to create an input feature.
- Keep immutable categorical action caching separate from state-dependent consequences. Identical choice text on different boards can have different costs/effects.
- Include feature/rules version, permitted-state identity, observer, phase/pending commitments, choices and relevant style/intent in appropriate cache keys. Bound cache sizes and avoid serializing the same growing history repeatedly.
- Preserve reference/optimized encoder parity, dynamic inference batching, shared-memory transport, overlapping actors and the single-GPU fast path.
- Charge feature work to the existing admitted decision budget where applicable; do not reset a whole-turn deadline to hide added overhead.

### Model and artifacts

- Use a new feature contract, provisionally `permitted-features/2.0.0`; finalize its exact layout before creating production data/model artifacts.
- All six groups are required before enabling the production preset. Intermediate development must not silently train with zero-filled missing feature groups.
- New state summaries reach value/utility as well as policy. Candidate consequences enter the action encoder. Preserve applicable personality conditioning/head semantics.
- Change the versioned model/input contract as needed; initialize fresh IL weights. Reject incompatible old weights, optimizer states and mixed-version batches explicitly.
- Do not change reward coefficients/normalization, PPO losses or teacher objectives as part of this revision.
- Re-encode old raw demonstrations only when their permitted observations contain sufficient facts. Preserve raw formats, choice IDs and supervision alignment where possible.
- Missing required source information must produce a documented compatibility failure or an explicit versioned raw-data extension/regeneration workflow. Unknown masks describe genuinely unresolved game information, not missing dataset fields that should have been recorded.
- Invalidate old encoded caches. Preserve raw teacher provenance; do not silently reuse or relabel teacher targets under a new model/feature identity. Any reuse requires an explicit checked conversion.
- Do not overwrite or delete old runs/packages. Old models remain usable through their pinned old runtime; this new path does not promise old-weight migration. The newly trained schema must support normal same-version IL handoff and self-play resume.

## 5. Step-by-step implementation

Stages are sequential. For a scoped request, complete only the named stage and its exit checks. Mark completion here with file/test evidence; do not mark learning strength as passed from fixture tests.

| Stage | Work | Exit evidence |
| --- | --- | --- |
| **FE0 — Inventory and contract** | Trace actual active entry points/settings, permitted data, both encoders, caches, model heads and downstream consumers. Specify every field/mask/scale and data reconstruction rule. Measure a small pre-change baseline. | Written field/source matrix; settings snapshot; bounded target/dedup definitions; initial dimension/parameter budget; identical reusable benchmark inputs. |
| **FE1 — Finance, order and opportunities** | Implement A, D and E as reusable permitted-view derivations. Preserve phase/seat semantics. | Focused financial/settlement, tie/order, phase and dedup checks; no extra full legal enumeration. |
| **FE2 — Realization and higher tiers** | Implement C and F state derivations; share stack/connectivity facts. Add the pure development before/after calculations needed by FE3. | Era-end exposure, action counts, beer contention, repeated levels, blockers, free development and current/next-era tests. |
| **FE3 — Candidate effects and caches** | Implement B and F action features with exact/bounded masks and explicit accounting basis. Separate static/dynamic caches and assemble the complete derived context. | Deterministic legal engine-transition comparisons, no double payment, partial-action tests, stale-cache/observer isolation tests. |
| **FE4 — Schema, model and data** | Freeze the complete feature layout; update reference/optimized encoders, dimensions, model/input contract, encoded replay and raw-data compatibility workflow. Initialize new IL model. | Encoder parity, finite tensors, dynamic candidate counts, parameter report, old-artifact rejection and demonstration-label preservation. |
| **FE5 — Pipeline integration** | Carry the same schema through IL, PPO, NN teachers/search, evaluation, batch/shared-memory transport and relevant export/personality compatibility gates. Add fresh-run and new-version resume checks. | Tiny IL -> self-play/teacher smoke; same-version save/load; output/seat mapping; batching/cache parity; unchanged reward/settings evidence. |
| **FE6 — Final checks and lab guide** | Run only targeted suites and a bounded paired before/after benchmark. Document dataset preparation and fresh training commands. Review the six-group checklist. | Verification/performance report; completed lab instructions; all mandatory fields wired; unresolved limitations explicitly listed. |

### Progress ledger

- [x] FE0 inventory, contract and baseline
- [x] FE1 financial safety, provisional order and legal opportunities
- [x] FE2 realization, era-end exposure and higher-tier development
- [x] FE3 action consequences and cache correctness
- [x] FE4 feature/model schema and dataset integration
- [x] FE5 end-to-end consumers and fresh training handoff
- [x] FE6 final verification, performance and lab instructions

For each completed stage append: date, implemented files, actual test command/result, artifacts, pending issues and next stage. No long campaign is needed to finish the implementation.

### 14 October 2026 — FE0 / FE1 evidence

FE0: See [field/source contract](FEATURE_ENGINEERING_FE0_CONTRACT.md) and its
ordered JSON matrix. The active settings snapshot, 14 reusable native permitted
positions (1,014 candidates), CPU reference/prepared cold/warm baseline and
provisional parameter report are in `artifacts/feature-engineering/fe0-fe3/`.
Commands: `python scripts/benchmark-feature-engineering.py --stage baseline`
and `--stage reference`, both completed. No production encoder/model/settings
changed. The user subsequently authorized exceeding 2M and retaining width 304.
The final FE3 provisional layout projects 2,590,449 parameters with utility and
2,656,282 with goals. Width 240 is an unselected alternative. FE4 must freeze
the actual schema/model and explicitly version its increased parameter limit;
no existing parameter check was bypassed.

FE1: `src/brass_ai/feature_engineering/finance.py`, shared `rules.py` and
`contracts.py` implement A/D/E. `python -m unittest test_fe1 -v` with
`src;tests/R2;tests/FeatureEngineering` on PYTHONPATH: **7 tests passed**.
Signed money, debt masks, final-era settlement, relative seats, current-order
ties, phase applicability and opportunity dedup are covered. Native transition
comparisons are added at FE3. Next: FE2 state realization/development, then FE3
candidate accounting and context/cache assembly. No learning-strength claim.

### 14 October 2026 — FE2 / FE3 completion evidence

FE2: `src/brass_ai/feature_engineering/industry.py` implements C/F state and pure
one-removal development effects. `python -m unittest test_fe2 -v`: **7 passed**.
Checks cover shared beer as individual feasibility, unflipped level-one exposure,
opening/partial/debt action counts, repeated tiers, productive incomplete removal,
undevelopable blockers, current/next-era eligibility and free development.

FE3: `effects.py` and `context.py` assemble all six groups using `contracts.py`,
`finance.py`, `industry.py` and `rules.py`. Explicit incremental and selected-part
bases keep payments, immediate VP and future era VP separate. Native transitions
verify build/market/resource/loan/merchant/development/second-Rail effects,
overbuilding, multi-resource costs, forced future merchant proceeds and settlement.
Finite immutable contexts retain every original choice ID/order. Full permitted
identity keys isolate seats, revision, cash, markets, pending state, beer and mats;
bounded static and dynamic caches remain separate. Histories are validated without
being reserialized into this additional derived key; caller deadlines remain shared.

Final command (project dependencies installed, `src;tests/R2;tests/FeatureEngineering`
on Windows PYTHONPATH): `python -m unittest test_fe1 test_fe2 test_fe3 -v`:
**32 passed in 15.870 seconds**, saved to
`artifacts/feature-engineering/fe0-fe3/focused-tests.log`. The bounded component
smoke `python scripts/benchmark-feature-engineering.py --stage derived` completed
on the exact 14 baseline positions / 1,014 candidates. Median/p95: cold
26.207/113.086 ms; warm 23.651/105.736 ms. Warm cache: 70 timed hits after 14
warmup misses; 191 dynamic calculations. Python allocation peaks: 1.22/3.84 MB.
This is additional derivation work; the active optimized encoder's baseline is
0.808/1.443 ms warm. No integrated performance/learning claim is made. Full
admission/content-identity work remains on hits; FE4/FE5 must share validation and
existing facts and measure the integrated path without weakening admission.

The ordered matrix now specifies 326 state and 52 action facts, seven components
per fact. Provisional appended dimensions: 5,489 state / 548 action. User-approved
width 304 projects 2,413,760 state-encoder parameters, 2,580,429 base, 2,590,449
with utility and 2,656,282 with goals. See the contract and JSON parameter budget.
`preservation-audit.json`: all **183** captured active source/config files retain
their byte hashes; active implementation and compatibility identities are unchanged.
No production feature/model/recipe/checkpoint or installed-game changes.

Next is FE4 schema/model/data compatibility and encoder parity, then FE5 consumer
integration and FE6 lab guide/performance checks. These stages remain incomplete;
the new derivations are not yet model inputs. No fresh training command is enabled
by FE0–FE3. No unresolved FE0–FE3 correctness failure is known from these bounded
checks; production integration and playing-strength effects remain unverified.

## 6. Focused verification and performance acceptance

### 14 October 2026 — FE4 / FE5 implementation evidence

FE4 freezes `configs/learning/features-2.0.0.json` at **5,489 state / 548 action**
inputs, retaining all 326 state / 52 action facts and seven components per fact.
Both encoders append the same mandatory A–F context. The style/intent tail remains
the final 12 state values. Width **304** is retained with the explicit versioned
`brass-four-seat-mlp-resnet/2.0.0` **3M** cap; counts are 2,413,760 state encoder,
2,580,429 base, 2,590,449 utility and 2,656,282 with goals. The personality model
uses its own revised 3.1M cap with the existing 100k extension allowance.

Learning/model/source fingerprints, encoded-version checks and `.features-v2.*`
caches reject old weights, encoded arrays, mixed batches and stale teacher/source
identities. `scripts/feature-engineering.py convert` provides explicit heuristic-only
raw conversion with full native replay/label/split validation and new outputs.
Two archived complete games (345 training / 346 validation decisions) were converted
without changing raw game objects, action IDs or source bytes. Negative cases reject
neural/search labels and missing required facts. No old-schema PPO import is added.

FE5 carries the layout through existing IL/PPO/inference/search/evaluation and
shared-memory paths. Oversized frames retain the queue fallback. Export tests cover
dynamic official, utility and personality/goal outputs; old packages remain pinned
to their old runtimes. The separate config helper binds a completed new IL model
and retains the current recipe. Only architecture/parameter-limit metadata changed
in the checked-in latest-model preset; its pinned 12/12/16/24 clocks, 0.08 exploration,
PPO/reward/teacher/curriculum/resource settings and null cumulative limit are retained.

Final focused command: `python -m unittest test_fe1 test_fe2 test_fe3 test_fe4
test_recovery test_exploration08 test_latest_model_recipe test_teacher_phase_revision
test_time_budgets test_inference_deadline_race test_teacher_readiness_review -q`
with `src`, `tests/R2`, `tests/Recovery`, `tests/FeatureEngineering` on PYTHONPATH:
**129 passed in 83.594 seconds** (`artifacts/feature-engineering/fe4-fe6/final-tests.log`).
Historical migration tests now use explicit test-only same-contract receipts;
production source-review registries were not expanded to allow v1 → v2 continuation.

The initial native smoke trained real random width-304 IL for one epoch, verified
IL completion/resume, handed its weights to a fresh PPO optimizer, played six ordinary
games over two iterations with exact Adam/RNG/curriculum/lineage resume, then ran four
seat-rotated diagnostic evaluation games and four reserved teacher turns. Further
verification after copying/fact-allocation optimization uses a new isolated output;
see the final FE6 report for its exact timing and usage. No production run is edited.

Mandatory regressions, grouped to avoid redundant suites:

1. **Money and effects:** negative income, signed balances, shortfall, pending debt, already settled rounds, last/era boundaries, gross cost versus proceeds, exact versus unresolved costs, resource double counting, immediate versus deferred VP.
2. **Realization and development:** shared beer does not imply joint sale; resource-industry flip conditions; disappearing level-one tiles; opening/partial-turn action counts; repeated tier copies; productive but incomplete development; undevelopable blockers; next-era targets; merchant bonus development.
3. **Order and legality:** official ties, final-round applicability, correct phase summaries, dedup without changing legal actions/labels, dynamic candidate counts and no-candidate handling at terminal boundaries.
4. **Parity and privacy:** same reference/optimized tensors, cached/uncached equality, changes to cash/market/beer/stack/pending state invalidate consequences, independent observer/seat mappings, hidden-state changes cannot alter an otherwise identical permitted input.
5. **Data and integration:** old encoded/model versions rejected; valid raw re-encoding preserves decisions/splits; unsupported rows fail explicitly; fresh IL handoff works; new-version checkpoint resume retains its own weights, optimizer and counters; export numerical/output mapping where applicable.
6. **Search/resources:** unchanged admitted shared deadlines, cancellation, legality, best-completed fallback and lease cleanup; new features do not create a second unbounded search or block worker retirement.

Use test-only engine transitions for representative completely determined legal actions as the factual oracle. For unresolved action bounds, verify realizable legal continuations fall within the bounds without requiring production-time exhaustive enumeration.

Record pre/post encoder median/p95 latency, candidates/second, cache hit behavior and incremental memory on identical permitted positions, phase mix and candidate lists. Then run a small paired integration smoke with the same seeds, hardware, concurrency, search budgets and model layout/weights where possible. An old-versus-new model comparison must be labeled as additionally changing dimensions/model work, not a pure encoder benchmark.

Report games/second or iteration time when measurable, completed simulations/second, qualified teacher samples/second, CPU/GPU utilization and peak memory. A short sample with zero qualified teachers is a measured zero, not a successful teacher-throughput result. Separate warm/cold caches and startup time. Do not infer speed from VRAM consumption.

Investigate material repeatable regressions before marking FE6 complete; remove redundant work first. Do not reduce approved search budgets, weaken quality gates or omit required feature groups to manufacture a speedup. Any remaining overhead and CPU-only/lab-hardware limitations must be reported.

Longer learning comparisons are a later lab activity, not an implementation exit gate. Track development progress, built-tier distribution, higher-tier flip rate, unflipped end-era exposure, insolvency, final VP, placement and wins. More development or a higher mean tier alone does not establish improvement. Preserve held-out evaluation separation.

## 7. Lab workflow and documentation deliverable

Use conda environment `ai-training` and root `/data/not_backed_up/yxu209/ai-training/`.

At FE6, create `docs/FEATURE_ENGINEERING_LAB.md` and add a concise pointer in the existing training guide. Document verified commands in this order:

1. Activate conda and inspect the new feature/model/schema versions.
2. Audit existing raw demonstration compatibility; re-encode supported records into a separate versioned cache or generate a new dataset when required.
3. Run a tiny fresh-IL smoke and check labels, masks and dimensions.
4. Run full fresh IL on the lab only when the user starts it; keep training/validation/test setup separation.
5. Evaluate the selected new IL checkpoint, then start a new self-play directory with the matching feature/model contract.
6. Resume that new run with its own checkpoint/state; do not import old-schema PPO progress or optimizer state.
7. Run diagnostic evaluation and later controlled learning comparisons with explicit model/schema identities.

Inspect actual CLI options before writing runnable commands; do not invent flags. Prefix project paths with the lab root. Existing dataset/run directories must remain intact. New output/cache/model paths should identify this feature revision. Explain required data regeneration, invalidated caches and why old models are rejected.

### 14 October 2026 — FE6 completion evidence

See [FE4–FE6 validation/performance report](FEATURE_ENGINEERING_FE4_FE6_REPORT.md)
and [lab guide](../docs/FEATURE_ENGINEERING_LAB.md). The guide includes inspection,
explicit raw conversion/regeneration, tiny smoke, fresh full IL, diagnostic evaluation,
new config/self-play start and same-version continuation commands. The config CLI
was exercised against the final completed smoke IL model. Existing run/dataset paths
and production source-review approvals were preserved.

After profiling redundant copying/fact allocation, the final source passed **129
focused tests**, fresh width-304 IL/resume, two ordinary games over two PPO iterations
with exact handoff/Adam/RNG/lineage, four seat-rotated evaluation games (zero failures)
and four native reserved teacher turns. Dynamic official, utility and goal ONNX
outputs pass CPU parity, largest absolute error below 0.000035. Native preflight
parity/device/optimized-worker checks also pass. An earlier six-game mixed-table
smoke is retained separately; no prototype weights/replays were imported into the
final run.

Native final phase probes: 4 qualified labels, **0 used / 4 unused** because probes
are incomplete teacher games; 72 completed of 103 attempted rollouts in 64.595 s.
Each phase completed one search and qualified one label; insufficient-coverage and
timeout counts were zero in this final sample. Native teacher policy drift is zero
(no update). A separate authored 80-row fixture used 80 distinct labels / 128
presentations over eight steps, reuse at most two, policy KL 0.000392383. Native
label yield and authored update mechanics are separate measurements.

Paired identical corpus/controlled-output benchmark: persistent-encoder miss median
1.162 → 3.225 ms; warm-hit median 0.895 → 0.875 ms. Cold construction median
2.702 → 8.739 ms; new facts/wider tensors retain measurable overhead. Python warm
allocation peak 1.896 → 4.861 MB. Two timed native games per arm varied substantially
(17.937/15.860 → 19.890/11.657 s); no end-to-end speedup or regression is established
from that small sample. All three controlled native choice sequences match. GPU
utilization/peak memory and production concurrency/teacher throughput remain lab
measurements; RSS was unavailable. No quality gate, feature group, search budget,
PPO setting or reward was weakened to improve these numbers.

Machine evidence: `artifacts/feature-engineering/fe4-fe6/review.json`, including
the final schema/source identities, parameter counts, preservation audit, source-bound
smoke, per-phase coverage/time/work/usage, exports and paired benchmark references.
The current latest-model preset retains its explicit 12/12/16/24 clocks and null
cumulative cap. No long campaign or installed-game change. Playing-strength and
learning-stability effects remain unproven; they are not implementation exit claims.

## 8. Small execution prompts

Initial request:

```text
Implement FE0 in ai-research/FEATURE_ENGINEERING_REVISION_PLAN.md.
Inspect the current code/settings and write the field contracts and baseline evidence.
Do not start FE1 or change the installed game AI. Report exit checks and next steps.
```

Subsequent request (replace the stage number):

```text
Implement FE1 in ai-research/FEATURE_ENGINEERING_REVISION_PLAN.md.
Review completed-stage evidence first, complete this stage and its focused checks,
update the progress ledger, and report any blocker. Preserve current training
semantics/settings and the installed game. Do not launch long training.
```

To authorize the complete implementation instead:

```text
Implement FE0 through FE6 in ai-research/FEATURE_ENGINEERING_REVISION_PLAN.md,
in order. Complete all six feature groups, fresh-IL/model/data integration,
focused checks and the small benchmark. Update stage evidence as you proceed.
Do not install a game patch or launch a long training/evaluation campaign.
```
