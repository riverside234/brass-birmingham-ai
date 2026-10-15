# Hand matching HM0 contract

**Archived:** Not used by the active encoder after the 15 October 2026 rollback.
The active feature layout is the original six-group Features 2 schema; see
[the current report](SIX_GROUP_FIVE_BLOCK_REPORT.md).

14 October 2026. This addition reuses all six Features-2 groups and their native
rule calculations. It does not add a strategic score, planning search or new
reward. The active entry points are Step 5 `imitate` and `train`, using
`recovery.demonstrations`, `PreparedEncoder`, `InferenceService` and the existing
Masked ISMCTS teacher pipeline. `Encoder` remains the parity reference.

## Fixed-board coverage

Only a main-action observation has the complete admitted build candidate set.
Deduplicate construction outcomes by **(slot, industry)**: the observer, board,
current stack head and era are fixed within the observation. The wire choice has
no other outcome-varying build field. Cards supporting an outcome are the card
types of those already-generated legal Build candidates, intersected with actual
positive hand counts. No alternative legal list is generated.

One bit per outcome and one bitset per present card type provide coverage.
Physical duplicate copies do not multiply opportunity counts. Counts remaining
after removal determine whether a card type still supplies its whole bitset.
An identified discard removes `card` plus every entry of `cards` jointly, with
full multiplicity. This includes all three Scout discards. A Build candidate's
selected outcome is removed from both loss counts: completing it is not a loss
of an unused opportunity. No new post-build network/slot state is predicted.

Wildcards are the two existing `wild:*` card types. All other existing card types
are ordinary. Another physical copy of the same type is an alternative card.
All support and delta results are exact only on this fixed admitted opportunity
set. They make no claim about subsequent action legality or future draws.
The loss fields isolate the consumed cards; they do not credit newly acquired
cards, including Scout's two new wildcards. They are not post-Scout hand forecasts.

## Fields and masks

The same seven Measure components used by A–F are retained: value, known,
applicable, lower, lower-known, upper, upper-known. Exact values are repeated in
their bounds; unknown, unavailable and actual zero remain distinct.

| Location | Field | Meaning | Scale |
| --- | --- | --- | ---: |
| State, existing card-vocabulary order (27) | `hand.support.<card>` | Distinct current Build outcomes supported by this card type; zero if absent from hand | 128 |
| State | `hand.wild_only` | Outcomes with wildcard support and no ordinary-card support | 128 |
| Action | `hand.lost_all` | Unused outcomes with no remaining supporting card after the identified joint removal | 128 |
| Action | `hand.lost_ordinary` | Unused outcomes losing their last ordinary support while retaining wildcard support | 128 |
| Build action | `hand.alternative_ordinary` | Remaining ordinary card/copy supports the selected build | 1 |
| Build action | `hand.alternative_wild` | Remaining wildcard card/copy supports the selected build | 1 |

State fields are unavailable outside main action. Action loss fields are
unavailable outside main action or without identified card removal. Alternative
fields are applicable only to main-action Build candidates. In an applicable
phase, a legal zero-build opportunity set has exact zero counts. Missing raw
hand/choices produce explicit admission failure, not zero-filled feature groups.

## Version, caching and model

Append the 28 state facts after A–F and before style[4]/intent[8]. Append four
action facts after A–F. Use new immutable `features-3.0.0.json`,
`permitted-features/3.0.0`, `four-seat-learning/3.0.0` and model architecture 3.0.0.
Old v1/v2 schema files remain untouched. Dimensions become **5,685 / 576**.
Width 304, four residual blocks and head semantics stay the same; expected
parameters: base **2,643,597**, utility **2,653,617**, personality **2,719,450**.
Existing explicit 3M / 3.1M caps remain sufficient.

Coverage is computed once per context. Residual bitsets are memoized by sorted
physical card removals. Base action effects continue to share card-independent
numeric/tensor blocks; hand deltas use a separate local cache. The complete
context/result cache still binds board, hand, phase, pending commitments, observer,
revision, choices, version and policy context. Absolute request clocks and
cancellation remain active through construction and tensor publication.

Raw heuristic v4 recordings can be explicitly re-audited and re-encoded into new
manifests/receipts and `.features-v3.*` sidecars without new teacher games. Preserve
records, action IDs/labels, setup splits and source hashes. Old encoded artifacts,
neural behavior, teacher targets, checkpoints and historical models are rejected
by normal admission. New-schema checkpoints retain the existing same-version
IL handoff/resume contract. No old neural weight/optimizer migration is supplied.

## HM0 evidence

Before-source/config hashes, an isolated pinned runtime and a 15-position corpus
(1,361 candidates, maximum 387) are saved under
`artifacts/feature-engineering/hm0-hm3/`. The corpus adds an actual legal
ordinary/wildcard hand to the existing 14 native positions.
`benchmark_hm.py` measures construction, persistent misses and cache hits plus
separate Python allocation peaks on this same corpus.

The three existing-feature semantic checks pass their focused native/encoded
regressions: affected-player attribution; action-specific connectivity/merchant
beer and individual-versus-joint feasibility; and distinct finance/time/sale
signals for the same stack target. Existing FE2 regressions additionally cover
incomplete development and era restrictions. No A–F correction/new field was
indicated by these bounded checks; final results are recorded in the HM report.
