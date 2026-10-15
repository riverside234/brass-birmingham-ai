# Main model: widened compact grouped v9

This target implements [the compaction plan](FEATURE_ENGINEERING_COMPACTION_PLAN.md). See [the width/teacher revision receipt](FEATURE_ENGINEERING_WIDTH_REVISION_REPORT.md) and [the original compaction receipt](FEATURE_ENGINEERING_COMPACTION_REPORT.md) and [lab commands](../docs/FEATURE_ENGINEERING_LAB.md).

## Versioned contract

| Contract | Current identity |
| --- | --- |
| Architecture | brass-four-seat-mlp-resnet/9.0.0 |
| Neural features / learning compatibility | permitted-features/4.0.0 / four-seat-learning/4.0.0 |
| Derived encoding | permitted-derived-context/3.0.0 |
| Layout | grouped-state-layout/5.0.0 |
| Feature manifest | configs/learning/features-4.0.0.json |
| Index manifest | configs/learning/state-groups-5.0.0.json |
| Layout SHA-256 | f7bd96900365293a314e167256142104ede1f390a9a7820dce95cff91307476b |

Earlier identities are immutable. Features-3 was the archived hand-coverage variant; this model has no hand coverage. The feature manifest declares retained fields, source facts, per-field components/scales, group membership, raw field lists and static assumptions. Python feature declarations must match it. Its projection_widths field is immutable historical v8 proposal metadata; active learned widths are owned by the separately versioned layout and ModelConfig. The feature manifest, reference encoder and all feature/derived identities stay unchanged for dataset reuse. The prepared encoder's cache lifetime/sharing was subsequently optimized without changing values; see [the reuse report](ENCODER_REUSE_REPORT.md). The generated index manifest binds exact membership/order, widths and the full feature-manifest hash. Equal shapes alone do not admit an artifact.

## Representation and network

| Group | Inputs | Outputs |
| --- | ---: | ---: |
| turn_global | 31 | 32 |
| cards_history | 218 | 64 |
| player_economy | 109 | 48 |
| industry_development | 1,314 | 192 |
| board_resources | 1,247 | 224 |
| current_action | 197 | 48 |
| policy_context | 12 | 12 passthrough |
| Total | **3,128** | **620** |

An exact-only fact emits [value / scale, known, applicable]. Exact zero, unknown and inapplicable remain distinct. Compact fields reject real intervals and one-sided bounds. State realization.card_supported_actions and all five portion-finance fields retain seven components. Full internal Measure facts remain available for rules and audit.

Candidates retain all 184 categorical/card components, plus 36 compact facts × 3 and five bounded facts × 7: **327 inputs**. Legal rows, IDs/order, physical card multiplicity, target/merchant distinctions and observer-relative seat mappings are unchanged.

~~~text
state [B,3128]
  -> static gather and six disjoint splits
  -> six Linear(input,width) -> GELU projections
  -> concatenate 608 projected values + 12 context values
  -> Linear(620,304) -> GELU
  -> six residual blocks at width 304 -> LayerNorm(304)

candidate [B,A,327]
  -> Linear(327,128) -> GELU -> LayerNorm(128)
  -> concatenate with shared state context
  -> Linear(432,128) -> GELU -> Linear(128,1) -> legal mask

residual(x) = x + Linear(304,304)(GELU(Linear(304,304)(LayerNorm(x))))
~~~

Each residual branch's final Linear weight/bias starts at zero. State encoding occurs once per observation/model. Official four-seat heads, utility integration and the personality context/goal extension retain their existing behavior.

## Computed budgets and admission

| Component/model | Parameters |
| --- | ---: |
| Six projections | 561,856 |
| Fusion | 188,784 |
| Six residual blocks | 1,116,288 |
| Final state LayerNorm | 608 |
| State encoder | **1,867,536** |
| Base | **2,005,917** |
| Utility | **2,015,937** |
| Personality | **2,081,770** |

Medium base dense work is B × (1,899,168 + A × 97,280) MACs. Utility/personality fixed work is 1,909,152 / 1,974,800 MACs, with the same candidate work. One MAC is two FLOPs; encoding, gathers, nonlinearities, normalization, transport and search are excluded.

The supported small preset keeps width 128 for backbone/fusion output and the same six group widths and six blocks. Its base/utility/personality counts are 934,605 / 938,993 / 974,378. Main training remains width 304. Resource ceilings remain 3M base/utility and 3.1M personality; there is no new 2M limit.

Static admission rejects unsupported slot type sets, location-specific merchant bonuses, era-derived hidden-discard counts and omitted placed-tile properties. The catalogue has 29 type/level combinations with no conflicting variants for those properties. Dynamic merchant acceptance/beer, placed-tile resources/cost and full mat properties remain encoded. Full raw observations/history are preserved.

The already converted Features-4 heuristic IL dataset is reusable as-is, including its existing manifest and raw shards. No conversion command or native re-audit is needed for this width-only change. Existing audited-heuristic admission permits only an architecture-source difference; all feature/encoder/derived/rules/target hashes must match and neural behavior/search targets remain forbidden. Architecture-bound derived cache receipts are invalidated and safely re-encoded in memory during normal loading; old files are not relabeled or overwritten. Use fresh v9 IL weights and a separate self-play directory. Prior-width weights and neural replay remain incompatible. Same-target continuation preserves weights, Adam/RNG, progress, population, curriculum and lineage. Source-review flags do not bypass feature/layout compatibility.

The checked-in latest recipe uses teacher clocks **9/12/12/12 seconds** (early Canal / late Canal / early Rail / late Rail) and **execution.teacher_search_workers_per_turn=8**. Five simultaneous teacher games provide capacity for 40 private search workers, plus coordinators/inference. Each turn's clock is shared across its workers, both actions and continuations. The generic search_workers_per_turn stays 4, preserving ordinary opponent and evaluation search. Explicit historical configs keep their settings; the latest-model generator applies these new defaults. Exploration 0.08, temperature 1, entropy/value weights 0.005/0.5, curriculum, four-seat collection, disabled gate, insolvency 0.005 capped at 0.075, five teacher games, cadence 160 and eight distillation steps are preserved. search_wall_fraction=None/null disables only the cumulative restriction; deadlines, cancellation and quality/freshness/calibration guards remain active.

This is an experimental representation/capacity adjustment. Playing-strength preservation, improvement and deployment speedup are unproven.
