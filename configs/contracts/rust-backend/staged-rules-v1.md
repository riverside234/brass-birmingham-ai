# Staged native rules — L05

These opt-in diagnostic operations execute four-player standard games under
`steam-brass-patch-1.1.0`. They extend `brass-native-control/0.1.0` with advertised
capabilities and use the complete [semantic state DTO](semantic-state-v1.md).
This trusted boundary includes hidden cards; production policy requests must
use the permitted observations and sampled worlds introduced in L06.

## Operations

After `hello` and `state_import`, the owner accepts:

| Request | Effect |
| --- | --- |
| `rules_snapshot {session}` | Export state, the complete ordered legal set, and explicit chance outcomes |
| `rules_apply {session, revision, choice}` | Apply one legal semantic choice at the exact current revision |
| `rules_chance {session, revision, card}` | Resolve one pending explicit card outcome at the exact current revision |

Successful responses have `status: "rules"`, `session`, `owner`, `state`,
`choices`, and `chance`. `chance` is null or contains `kind`, `remaining`, and
`outcomes` (`card`, `probability`). Both terminal and chance states have an empty
ordinary legal set. Terminal results remain inside `state.result`.

The Python diagnostic client exposes `rules_snapshot`, `apply_choice`, and
`resolve_chance`. It does not select this backend automatically. Both state
owners still have one computation thread and one private save slot each.

## Choice and transition semantics

Choices use the frozen C# host's canonical fields and semantic IDs;
ordering is ordinal by complete ID. Duplicate cards produce one semantic choice
per distinct type or Scout multiset. Rust generates the current staged choices
directly, without first expanding products of complete action plans.

The adapter covers ordinary build, network, develop, sell, loan, scout and pass;
coal/iron/beer payments; action continuations and finish; merchant development
bonuses; and debt liquidation. It preserves separate acting seat, engine flow
cursor and turn owner, resource consumption/flipping, income, spending, merchant
bonuses, ordered cards and history, round settlement, era scoring and cleanup,
official terminal tie-breaking and shared wins.

Build eligibility, nearest coal choices, own/connected beer, overbuilding,
preferred single-industry slots, network membership, wildcard restrictions,
optional second development/link and repeat sales match the selected C# rules.
The target income marker saturates at 99. Nottingham grants 3 VP; Shrewsbury
grants 4. Scout replaces traded cards at their original hand positions. Unbuilt
Canal-only mat tiles remain at the Rail boundary. Final Rail scoring does not
collect another round of income. Bonus-VP subtotals reset at both era boundaries.

`brass-upstream-rules` supplies physical connectivity and market pricing;
`fixedbitset` and the existing typed catalog remain shared infrastructure.
Staged action/flow translation is target-specific code. The vendored upstream
rules, lockfiles, local patch and license are preserved byte for byte.

## Explicit chance and rejection

Chance probabilities equal each card type's remaining multiplicity divided by
the pool size. Outcomes follow semantic card order. Each accepted outcome moves
one card from pool to ordered picks and increments revision; it adds no public
choice-history entry. Partial draws/shuffles preserve the paused physical card
zones. A completed draw resumes the turn; a completed 64-card Rail shuffle
settles and scores Canal, determines the next round order, and redeals in that
order. This does not depend on matching C# and Rust random-number generators.

Choices must belong to the full current legal set. Stale revisions, invalid
choices/outcomes, history/revision limits and checked numeric overflow are
errors. Mutation takes place on an owned transactional fork before commit;
rejected transitions preserve the current state and the independent save slot.
The worker also prepares the next response before committing, so a legal-set
error after a structurally valid but unreachable import cannot partly commit.
Successful transitions leave the save slot intact. Save/restore also preserves
pending payments, partial chance and residual settlement debts.

## Qualification boundary

The L05 suite compares complete private state, legal choices and chance on 546
accepted roots, 20 recorded games / 8,344 transitions, and 1,406 branch
alternatives. The root set includes the six separately corrected L04 authored
cases; the original inconsistent cases remain rejection fixtures. Reference
game captures must reproduce the original L02 projections at every step before
their additional private fields are admitted. Existing Python reward accounting
consumes the native public consequences and must reproduce all 20 frozen returns.

The worker continues to report `target_rules_qualified: false`. L06 privacy and
sampling and L07 expanded simulator qualification precede an admitted simulator.
Native reward-ledger/search implementation, features, neural inference, Unity
deployment and lab collection are later milestones. No L05 rule microbenchmark
is a matched NN/search speedup; that later test retains 4 seconds per whole turn.
