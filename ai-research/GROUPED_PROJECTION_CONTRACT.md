# Main model: compact grouped v8

This target implements [the compaction plan](FEATURE_ENGINEERING_COMPACTION_PLAN.md). See [the implementation receipt](FEATURE_ENGINEERING_COMPACTION_REPORT.md) and [lab commands](../docs/FEATURE_ENGINEERING_LAB.md).

## Versioned contract

| Contract | Current identity |
| --- | --- |
| Architecture | brass-four-seat-mlp-resnet/8.0.0 |
| Neural features / learning compatibility | permitted-features/4.0.0 / four-seat-learning/4.0.0 |
| Derived encoding | permitted-derived-context/3.0.0 |
| Layout | grouped-state-layout/4.0.0 |
| Feature manifest | configs/learning/features-4.0.0.json |
| Index manifest | configs/learning/state-groups-4.0.0.json |
| Layout SHA-256 | fe763afeb6a5276daff0f592d075be219b261324698e50af41829ebcb0440fd7 |

Earlier identities are immutable. Features-3 was the archived hand-coverage variant; this model has no hand coverage. The feature manifest declares retained fields, source facts, per-field components/scales, group membership, raw field lists and static assumptions. Python declarations must match it. The generated index manifest binds exact membership/order, widths and the full feature-manifest hash. Equal shapes alone do not admit an artifact.

## Representation and network

| Group | Inputs | Outputs |
| --- | ---: | ---: |
| turn_global | 31 | 32 |
| cards_history | 218 | 64 |
| player_economy | 109 | 48 |
| industry_development | 1,314 | 160 |
| board_resources | 1,247 | 192 |
| current_action | 197 | 48 |
| policy_context | 12 | 12 passthrough |
| Total | **3,128** | **556** |

An exact-only fact emits [value / scale, known, applicable]. Exact zero, unknown and inapplicable remain distinct. Compact fields reject real intervals and one-sided bounds. State realization.card_supported_actions and all five portion-finance fields retain seven components. Full internal Measure facts remain available for rules and audit.

Candidates retain all 184 categorical/card components, plus 36 compact facts × 3 and five bounded facts × 7: **327 inputs**. Legal rows, IDs/order, physical card multiplicity, target/merchant distinctions and observer-relative seat mappings are unchanged.

~~~text
state [B,3128]
  -> static gather and six disjoint splits
  -> six Linear(input,width) -> GELU projections
  -> concatenate 544 projected values + 12 context values
  -> Linear(556,304) -> GELU
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
| Six projections | 479,840 |
| Fusion | 169,328 |
| Six residual blocks | 1,116,288 |
| Final state LayerNorm | 608 |
| State encoder | **1,766,064** |
| Base | **1,904,445** |
| Utility | **1,914,465** |
| Personality | **1,980,298** |

Medium base dense work is B × (1,797,760 + A × 97,280) MACs. Utility/personality fixed work is 1,807,744 / 1,873,392 MACs, with the same candidate work. One MAC is two FLOPs; encoding, gathers, nonlinearities, normalization, transport and search are excluded.

The supported small preset keeps width 128 for backbone/fusion output and the same six group widths and six blocks. Its base/utility/personality counts are 844,397 / 848,785 / 884,170. Main training remains width 304. Resource ceilings remain 3M base/utility and 3.1M personality; there is no new 2M limit.

Static admission rejects unsupported slot type sets, location-specific merchant bonuses, era-derived hidden-discard counts and omitted placed-tile properties. The catalogue has 29 type/level combinations with no conflicting variants for those properties. Dynamic merchant acceptance/beer, placed-tile resources/cost and full mat properties remain encoded. Full raw observations/history are preserved.

Use explicit heuristic raw-data re-audit into new files, fresh compatible IL and a separate self-play directory. Incompatible weights/caches/teacher targets cannot migrate or be relabeled. Same-target continuation preserves weights, Adam/RNG, progress, population, curriculum and lineage. Source-review flags do not bypass feature/layout compatibility.

The checked-in recipe changes only its architecture identity. Exploration 0.08, temperature 1, entropy/value weights 0.005/0.5, current curriculum, four-seat collection, disabled gate, insolvency 0.005 capped at 0.075, teacher clocks 12/12/16/24 seconds, five teacher games, cadence 160 and eight distillation steps are preserved. search_wall_fraction=None/null disables only the cumulative restriction; deadlines, cancellation and quality/freshness/calibration guards remain active.

This is an experimental representation/capacity adjustment. Playing-strength preservation, improvement and deployment speedup are unproven.
