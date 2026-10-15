# FE0–FE3 implementation contract

This is the historical FE0–FE3 record. FE4/FE5 now enable the frozen layout in
`configs/learning/features-2.0.0.json`; see the revision-plan ledger and
`docs/FEATURE_ENGINEERING_LAB.md` for the active contract and fresh-run workflow.

This is a standalone `permitted-derived-context/0.1.0` implementation. It is
**not enabled in the active feature encoder or training preset**. FE4 must freeze
`permitted-features/2.0.0`, input dimensions, model/data compatibility and cache
conversion before any production use. All six groups are mandatory for assembly.

The ordered field/source/unit/scale/basis matrix is
`FEATURE_ENGINEERING_FE0_CONTRACT.json`, generated from
`brass_ai.feature_engineering.contracts.specification()`. Every quantity uses
`value, known, applicable, lower, lower_known, upper, upper_known`, in that order.
Unknown values have zero storage with false masks; inapplicable differs from
unknown, and an exact zero has true known/applicable masks. Lower/upper endpoints
can be independently known. Signed values are not clipped. Fixed scales are
documented per field; no normalization uses training statistics.

| Group | Owner and production inputs | Meaning and exclusions |
| --- | --- | --- |
| A | Four observer-relative seats; cash, actual income, debt, phase, era, round | Next settlement at fixed current position. Final Rail has none. A debtor's unresolved projection is unknown; public debt is retained. A new round's cash already includes the past settlement. |
| B | Acting observer plus four relative effect recipients; legal candidate, public board/markets/pending state | Separate incremental chosen-portion effects from mandatory completion of that selected part. Never include an optional second development/link/sale. Gross expenditure and proceeds remain separate. No simulator call per candidate. |
| C | Observer only; public industry tiles, graph, merchants/beer; own hand, deck count and turn context | Potential sale VP, secured flipped industry VP, and cumulative raw VP remain separate. Individual sale feasibility can reuse contested beer and does not certify joint sales. Action slots include an in-progress main action; future draws are bounded rather than guaranteed. |
| D | Four relative seats; public spending and current order | Provisional next order sorts spending ascending, preserving current-order ties. Spending difference is observer minus that seat. Final Rail has no next round; final Canal still determines Rail order. |
| E | Observer's already admitted legal candidates and current UI phase | Aggregate without changing choices, IDs or labels. An unavailable main-action summary during payment/bonus/debt is inapplicable, not evidence of no building/selling possibilities. |
| F | Observer's full ordered industry mats and era flags | Current stack head is never silently skipped. Each target is the first strictly higher eligible level by stack order. Every intervening physical copy counts. An undevelopable prefix blocks development-only reachability. No profitability ranking or affordability claim. |

Target status distinguishes presence, reachability, blockers and unlocked current
heads. Current-era and Canal-to-Rail targets are separate. A target is selected
even if blocked, so its attributes remain visible, but its removal distance is
unknown when development alone cannot reach it. Next-era targets are inapplicable
in Rail. Development effects remove exactly the selected top tile and retain the
original target identity for before/after distance; a free merchant choice uses
no iron and does not consume a second main-action slot.

Opportunity deduplication is category-specific: Build `(slot, industry)`, Sell
`(industry slot, merchant)`, Network/resource/liquidation `target`, development
`industry`, Scout sorted discarded-card multiset, Pass/Loan/Finish/Skip one target.
Card alternatives for a fixed action target remain distinct actual policy
candidates but do not multiply target counts. Scout multisets remain different
because the exchange itself is the effective opportunity.

Action fields use **the selected action portion before automatic turn/round
advancement**. This avoids mixing income settlements and era scoring into a
building price or treating resource subchoices as another action. Immediate VP
means merchant/action bonuses; secured industry VP is a potential future era
award. Any settlement projection is explicitly a fixed-position exposure, not
a promised balance. Resource-source choices can immediately flip other players'
industries; effects therefore use four relative recipients. Known/bounded masks
must remain honest when a future source, merchant bonus, or liquidation outcome
is unresolved. Tests compare action events separately from automatic boundaries.

Only the acting, nonterminal permitted observer with a nonempty admitted legal
list can assemble a context. Terminal positions do not create policy tensors.
Public rule calculations are shared in `feature_engineering/rules.py`. The
pinned native host is the test oracle. The boundary accepts only validated
permitted observations/legal choices, never a host, checkpoint or sampled full
world. Graph components/resource inventories are built once per observation.
Historical events are not used by these derivations; their growing list is not
serialized again to form the derived cache key. Observer, revision, phase,
pending counters, full relevant public state, own hand, choices and policy
style/intent are included. Static choice semantics and state-dependent effects
have separate bounded caches. Outputs retain every original choice ID/order.

## Active pipeline inventory

- `scripts/step5.py` → `step5/cli.py`: active `imitate` calls
  `recovery/demonstrations.train_bc`; `train` calls `recovery/league.train`.
- IL raw admission/encoding is `demonstrations.load_dataset` → recovery replay
  → `PreparedEncoder` → `updates.Dataset`/supervised epochs. The standalone
  `learning/features.Encoder` is the reference; `recovery/encoding.py` owns the
  optimized independent encode path and static action cache.
- Ordinary PPO, teacher collection/search and evaluation share the recovery
  inference and replay paths. `learning/batching`, `recovery/inference` and
  `shared_transport` carry dimensions/masks. New state summaries must feed all
  state heads and new action consequences the candidate encoder in FE4/FE5.
- `learning/contracts.compatibility` pins feature/encoder/model digests.
  `recovery/replay.cache_version` pins encoded caches; both must change at FE4.
  Raw replay can be re-encoded only after explicit checked old-version admission;
  the existing heuristic-data check is not a feature-version conversion.
- `learning/export`/personality contracts, model goals, rosters and deployment
  receipts depend on those identities/dimensions. No artifacts are requalified
  by this standalone stage. A fresh IL/self-play schema is required at FE4/FE5.
- Wire observation `v0.2.0` includes full ordered mats, actual income, debt,
  board regions/links, resource stock, merchant types/bonuses, pending counts,
  observer hand and admitted legal candidates. No wire contract is edited.

## FE0 baseline and parameter budget

Resolved current settings and hashes of active source/config files are saved in
`artifacts/feature-engineering/fe0-fe3/settings-snapshot.json`. This captures the
latest-model 20k/100k curriculum, no gate, exploration 0.08, current PPO/rewards,
teacher 12/12/16/24 seconds, five games/eight steps, null wall fraction, search,
resource controls, retention and evaluation schedules without copying old values.

The identical reusable corpus has 14 permitted native-host positions and 1,014
legal candidates, covering Canal/Rail, action/payment/continuation/bonus/debt.
Input SHA-256 is `0a6923e6fcd8b0497f0674c3bb6f4b728cfe213b314265989544f3b02e4ce28b`.
`reference.json` records uninstrumented CPU timing and a separate Python-allocation
pass. Median/p95 milliseconds: reference encoder 36.826/167.638, prepared cold
3.192/12.029, prepared warm 0.808/1.443. `baseline.json` retains an earlier
tracemalloc-instrumented exploratory timing; it is not the comparison baseline.
Neither report measures training/search throughput, GPU memory or playing strength.

Active dimensions remain 3,207 state / 184 action. The IL base including official
heads has 1,840,109 parameters; with utility it has 1,850,129 and with the existing
goal extension it has 1,915,962. The provisional seven-component layout has 326
state facts and 52 action facts: total dimensions 5,489 / 548 if appended unchanged.

**The user authorizes exceeding 2M parameters and retaining width 304.** The
analytic width-304 projection is 2,413,760 for the state encoder, 2,580,429 for
the base network including candidate/policy/official heads, 2,590,449 with
utility and 2,656,282 including the existing goal extension. The width-240
alternative in the JSON budget is not selected. FE4 must freeze the final layout
and introduce an explicit versioned parameter limit supporting it. No active
model, weight migration or parameter-limit bypass is implemented at FE0–FE3.
FE4 may compress redundant exact masks before freezing its final dimensions.

## FE1–FE3 exit evidence

`tests/FeatureEngineering/test_fe1.py`, `test_fe2.py` and `test_fe3.py` pass **32
tests**. The suite uses test-only pinned native-engine transitions for gross
construction cost/proceeds, market payments, last-resource flips, nonlinear loan
income, merchant bonuses, free development, second Rail links, multiple-resource
payments, overbuilding and next-round settlement/order. It also checks masks,
partial action/card counts, contested beer, repeated tiers, undevelopable blockers,
current/next-era targets, observer isolation, immutable outputs, bounded caches
and invalidation after revision/market/cash/pending/beer/stack changes. Cached and
uncached contexts match, with the original choice order and IDs preserved.

`derived.json` uses the same corpus as `reference.json`. The additional standalone
derived-context component measures **26.207/113.086 ms cold median/p95** and
**23.651/105.736 ms warm**, with 1,688/1,862 candidates per second respectively.
The warm run records 14 misses during warmup and 70 hits during timed calls, with
191 dynamic calculations for 1,014 original candidates. This is reuse of numeric
consequences, not removal or duplication of policy labels. Validation and content
identity are included even on a hit. Separate allocation passes peak at about
1.22 MB cold and 3.84 MB warm for Python allocations on this corpus; these are
not RSS or GPU-memory measurements. These timings measure an additional component,
not an encoder replacement or an end-to-end speedup. FE4/FE5 must reuse validated
admission and shared facts while preserving validation/privacy and then benchmark
the integrated path. A separate diagnostic `warm-profile.json` attributes 2.281
of 2.300 profiled seconds to schema validation on 14 cache hits; profiling overhead
is included and those seconds are not benchmark timings. No training, search
throughput or playing strength is measured.

`preservation-audit.json` confirms all 183 captured active source/config files
have their original byte hashes, and active implementation/compatibility identities
are unchanged. The provisional package is outside active encoding/training paths.
FE4 is next; production schema, old-artifact rejection, encoder parity and fresh-IL
integration remain its work. Existing runs/checkpoints are not converted.
