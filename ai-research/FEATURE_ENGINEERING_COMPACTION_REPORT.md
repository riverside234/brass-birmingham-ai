> Historical v8 compaction receipt. Active widths and teacher settings are now v9; see [the follow-up receipt](FEATURE_ENGINEERING_WIDTH_REVISION_REPORT.md). Features-4 data remains reusable.

# Feature compaction implementation receipt — v8

Implemented FC0–FC3 on 15 October 2026. The [active contract](GROUPED_PROJECTION_CONTRACT.md) supersedes earlier targets; [lab instructions](../docs/FEATURE_ENGINEERING_LAB.md) contain exact new-data/new-model and continuation commands.

**Final target:** 3,128 state / 327 action inputs; group input sizes 31 / 218 / 109 / 1,314 / 1,247 / 197 plus 12 context; projection widths 32 / 64 / 48 / 160 / 192 / 48; fusion 556 → 304; six width-304 residual blocks; candidate branch 327 → 128. Heads, activation/normalization and residual zero initialization are unchanged.

**Exact parameters:** base 1,904,445; utility 1,914,465; personality 1,980,298. State encoder 1,766,064. Base dense work is 1,797,760 MACs per position plus 97,280 per padded candidate. These are computed structural counts, not measured latency.

**Identities:** features/learning 4.0.0, derived context 3.0.0, grouped layout 4.0.0, architecture 8.0.0. Layout SHA-256: fe763afeb6a5276daff0f592d075be219b261324698e50af41829ebcb0440fd7. Validated implementation: 97547e1c2759fbf05b449c69e11434233005f586ab0dfa54b9c73ce94efffe2f.

## FC0: keep/remove/move/compact ledger

All conditional target omissions were admitted; there are **no dimensional exceptions**. These decisions are pinned to the supported four-player profile, not inferred from sample frequency.

| Decision | Source reason and preserved information |
| --- | --- |
| Compact 194 state and 36 action facts to triplets | Finance/order/development/effect sources emit exact, unknown or inapplicable values for these fields. Interval/one-sided-bound input raises an error. Existing scales and signs remain. |
| Keep one state and five action bounds | realization.card_supported_actions uses own cards plus possible draws. All five portion-finance fields keep seven components, including cost/proceeds/balance/projection/shortfall uncertainty. |
| Trim five neural history counters | Omit coal, iron, beer, finish and bonus_skip only. Keep build, develop, sell, loan, network, pass, scout, bonus_develop and liquidate. Full raw history, knowledge calculations and audit remain. This is an information tradeoff, not proven irrelevant history. |
| Remove hidden_discards columns | Observation.cs:45 emits Canal=1 / Rail=0 for every player. Admission checks this; raw fields remain. |
| Share settlement; omit duplicate debt and own spending difference | financial() uses the same has_settlement() for all four seats. Raw public debt is retained; own spending minus itself is zero. Negative payment, projected cash, shortfall and all four provisional ranks remain. Debt unknowns and final-Rail inapplicability remain distinct. |
| Remove eight current-head development facts per industry | All four raw mats retain level counts and complete top-two tile properties/presence. Current already_unlocked remains. Scales are not reinterpreted: derived cost/100 comes from raw cost/30 × 0.3; VP/100 from VP/20 × 0.2; beer/8 from beer/2 × 0.25. Absence uses the retained raw mat-head present bit. |
| Remove five fields for each target/horizon | target() selects a strictly higher eligible level at index > 0: already_unlocked is false, survives_canal follows level > 1, blocked follows presence/reachability, eligibility deltas use target flags minus raw head flags. Keep 150 development facts, blockers/unknown removal distance, both horizons and absent targets. |
| Remove fixed slot type sets and merchant bonus identities | Board.cs CreateRegions defines all 49 slot sets and nine merchant bonuses by region/index. Randomness affects IDs/merchant acceptance, not those constants. ImportSetup changes merchant accepted types, not bonuses. Source-derived frozen constants are checked on admission. |
| Remove slot.present and five placed-tile properties | Four relative-owner bits distinguish occupied/empty. PlayerMat.PrepareMat and read-only BaseIndustry properties fix coal/iron/developable/canal_only/rail_only for all 29 type/level keys, including repeated copies. Constructor audit found no conflicts. Pending payments use retained counters; overbuild reads full raw tiles; Canal removes level-one tiles. Retain eight placed properties including cost (liquidation uses cost // 2), current stock, scoring, sale beer and ownership. Mat encoding remains separate and full. |
| Remove main_action_available and 14 available facts | Retained phase and phase-aware target counts recover them without treating subchoice absence as no future main action. Keep every legal candidate and deduplication rule. |
| Remove eleven action facts | Resource-used indicators follow kind. Incremental effects award immediate VP only to actor; other seats retain flips/income/secured VP. Development removes one selected tile, free follows bonus_develop, survival follows result level, and before=after+1 for the same selected higher target with unchanged unknown/absent status. Keep internal facts and both accounting bases. |
| Move related facts before projection | Markets + eight realization facts join board_resources; supplies join cards_history; remaining actions/card support/in-progress/shared settlement join turn_global. Mats/development become industry_development. Current/provisional order and spending stay together. No moved value is duplicated. |

Rule sources inspected: src/BrassHost/Observation.cs, Model.cs and Position.cs; decompiled Board.cs CreateRegions/merchant preparation, PlayerMat.cs, BaseIndustry.cs, BuildingSlot.cs, BorderRegion.cs, BuildAction.cs, SellIndustryFlow.cs and CanalEra.cs; feature_engineering finance.py, industry.py, effects.py and rules.py. Source hashes for the static constructor audit are in manifest-generation.json. No proprietary rules or live-game files were edited.


## FC1: implementation and downstream bindings

The semantic contract still contains **326 state facts and 52 action facts**. Neural emission retains **195 state facts** (194 triplets + one seven-component field = 589) and **41 action facts** (36 triplets + five seven-component fields = 143). The explicit omission ledgers contain 131 state and 11 action facts. Strict semantic fact-set equality remains; arbitrary missing/unexpected facts are rejected.

Both reference and optimized encoders use separate placed-tile/mat lists, selected history kinds and declared component lengths. The static profile is admitted before reuse. New dimensions flow through batching, shared-memory validation, PPO, teacher leaf inference, utility/personality models and named exports. Cache filenames/metadata identify Features-4; stale attestations are rejected. Source/compatibility hashes cover the current manifests and derived code. Packaging/handoff includes the new feature and layout files.

Existing-source/config files changed in this revision:

- src/brass_ai/feature_engineering/context.py
- src/brass_ai/feature_engineering/contracts.py
- src/brass_ai/feature_engineering/data.py
- src/brass_ai/learning/contracts.py
- src/brass_ai/learning/features.py
- src/brass_ai/learning/model.py
- src/brass_ai/learning/state_layout.py
- src/brass_ai/learning/utility_model.py
- src/brass_ai/personality/handoff.py
- src/brass_ai/recovery/encoding.py
- src/brass_ai/recovery/league.py
- src/brass_ai/recovery/replay.py
- src/brass_ai/step5/league.py
- src/brass_ai/personality/laptop/package.py
- configs/step5/joint7-latest.json

Also added static_profile.py, features-4.0.0.json, state-groups-4.0.0.json, test_compaction.py and smoke_compaction.py. Updated scripts/feature-engineering.py inspection output; synchronized existing model/feature test expectations and variable-length semantic offsets; updated the active contract/lab documents and this receipt. Previous uncommitted performance, source-review and training updates were preserved. Earlier frozen manifests and source-review receipts are byte-unchanged.

The checked-in recipe changed **only architecture**. The generated fresh-run binding uses the new completed IL checkpoint **and its re-encoded rehearsal dataset**. No existing production checkpoint, population, dataset or run was edited. Current reward, curriculum, exploration, teacher settings, search budgets and all guards are preserved, including search_wall_fraction=null. Exact values are recorded in the active contract and resolved smoke configuration.

## FC2: focused debugging and one isolated lifecycle smoke

**Two focused checks passed in 3.677 seconds.** The combined semantic/encoder pass exercised ten authored positions and 760 candidate rows (13 action kinds), including repeated-tier development, optional merchant money proceeds, payments, pending/partial debt and final Rail. It checked compact exact/zero/unknown/absent states, real portion bounds and the card-support interval, omission reconstruction, static-profile rejection, reference/optimized parity, choice order, seat mapping, full/disjoint grouping and context preservation. The one-sided resource quote also exercised retained bounded cost encoding. The tiny batch checked finite forward/backward values, legal padding, every projection's gradients, unchanged heads, utility/personality entry points, computed counts and synthetic incompatible metadata rejection.

The smoke used one scratch directory: artifacts/feature-engineering/compaction-v8-20261015/smoke. It re-audited two complete existing heuristic raw games, then used **eight training and eight validation examples** as a fixture for **one IL optimizer update** and the real completion/handoff path. This test-local selection does not change production data loading or admit old neural targets.

It then collected **one ordinary game**, all four current-model seats at exploration 0.08, with **315 decisions**: 143 actor-eligible and 172 forced. Strict PPO behavior-probability verification ran before **one PPO optimizer step**. Actual actor usage was 143 distinct labels / 143 presentations / 0 unused eligible; critic usage was 64 distinct samples / 64 presentations. The existing calibration guard performed **three separate utility-head-only steps** from this game's fresh returns and marked calibration ready.

Matching checkpoint load restored exact weights, **68 Adam states** and RNG. Generation/game counters stayed at 1/1; population retained generations 0/1, curriculum and IL lineage. Checkpoint bytes and committed state remained unchanged. Resume was checked through the actual loader without another training iteration.

A spawned worker used the teacher's RemoteEvaluator/shared transport with the target schema, matching model ID/reward identity and a finite utility response. No tree simulations, timed teacher collection, labels or distillation ran. Personality import copied the exact utility parent; the newly initialized goal extension remains diagnostic.

Production actor ONNX export/load parity passed at **(batch,candidates)=(1,1),(2,7)** with padding and all ten named outputs. Maximum absolute error was **3.0517578e-05**, from the VP head; policy error was 4.1723251e-07. Runtime was ORT CPU 1.22.1. Existing tracer warnings about Python shape guards occurred; dynamic-shape parity passed. No deployment qualification is implied.

Elapsed smoke wall time was **401.02 seconds**, including harness repair pauses; the final continuation took **26.02 seconds**. Full native conversion took 84.25 seconds. The ordinary iteration took about 46.2 seconds (43.1 collection, 2.2 PPO). This is execution accounting, not an inference benchmark.

Debug fixes were confined to the harness: a final-Rail fixture had incorrectly broken deck conservation; the initial smoke omitted the new rehearsal binding; and its Windows leaf worker initially tried to pickle an endpoint lock. Fixed the fixture, bound rehearsal to completed IL, and acquired the spawn-safe channel inside the child. Reused the same audited data, completed IL and committed generation; **no second game or PPO update** was launched. Failed-attempt logs are retained alongside the final pass receipt.

Smoke-only overrides: CPU two threads, one actor/host, effective batch 512, microbatch 16, critic batch 64, KL probe 32, one PPO epoch, disabled rehearsal/scheduled evaluation/retention, and forced current-model table composition. These do not modify the production recipe.

## Scope and evidence

No full suite, seed sweep, timed teacher probes, full teacher/distillation collection, strength games, previous-model comparison or performance benchmark was run. Historical/legacy ordinary tables, GPU/OpenVINO/Unity deployment, full training convergence, scheduling and release-package qualification were not exercised by this smoke. Existing broader regressions were aligned with the target but not rerun. Validation stopped after the focused checks and smoke passed.

Evidence is in artifacts/feature-engineering/compaction-v8-20261015/: before.json and changed.json (preservation inventory), manifest-generation.json, inspect.json, focused.log/focused-2.log, smoke.log/smoke-resume.log/smoke-finish.log, smoke/report.json, audited data, committed IL/PPO checkpoints and actor export/parity files. The manifest-generation and documentation scripts retain the exact local procedure.

Use fresh audited/re-encoded data and a new compatible IL/self-play run. Same-target continuation commands are in the lab instructions; do not point them at an earlier architecture. Representation compaction and passing smoke checks demonstrate neither preserved/improved playing strength nor optimal widths or a measured deployment speedup.
