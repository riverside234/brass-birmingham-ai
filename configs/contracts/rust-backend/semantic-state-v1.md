# Trusted native state — `brass-native-state/1.0.0`

This is the L04 diagnostic/import boundary for four-player standard games under
`steam-brass-patch-1.1.0`, with explicit chance. It is separate from the frozen
L02 canonical projection and from production policy requests. The executable
schema is `native/brass-ai-core/src/state/dto.rs`: Serde denies unknown fields,
requires the typed fields and four-element seat arrays, and rejects unknown enums.
`NativeState::import` validates relationships before publishing a replacement.

## Authoritative fields

| Field | Meaning |
| --- | --- |
| `format`, `rules` | Exact contract versions |
| `context` | Protocol, rules, revision, actor, turn owner, phase, remaining actions, era, round, turn, terminal/chance flags |
| `current_player` | Engine flow cursor, which can differ from the debt actor and turn owner |
| `order` | Four distinct seats in current round order |
| `players` | Stable seat, signed cash/points/spending, income marker and verified derived income, bonus points, link stock, ordered industry mats, visible debt |
| `slots`, `links`, `merchants` | Stable topology, industry identities and quantities, link owners, merchant types/beer/bonus |
| `cards` | Ordered hands, deck, wild piles, discards and per-seat Rail discard offsets |
| `removed` | Exact C# board removed-industry identity list |
| `retired` | Remaining off-board identities: development, era mat pruning and liquidation can remove objects without adding them to the board removed list |
| `round_pending`, `debt_pending` | Pending round/settlement flow flags |
| `settlement_debts` | Four residual amounts used at settlement, including unpaid penalties during the pre-Rail shuffle pause |
| `pending` | Action kind, target, merchant, completed parts and outstanding coal/iron/beer; no turn-start approximation |
| `chance` | Null, or draw/rail-shuffle kind, total needed, remaining ordered pool and ordered picks already made |
| `history` | Ordered public staged choices with validated semantic IDs |
| `result` | Null until terminal; independently checked/reconstructed rank and win-share result |

Card identity is a semantic type, with multiplicity and ordered zone positions;
it is not a C# object GUID. The complete normal-card multiset has 64 cards, plus
four of each wildcard. In Rail, `discards[seat][:discard_starts[seat]]` contains
historical aliases of cards already redealt. Count only the suffix as physical
cards. Do not count a pending chance pool/picked list as another physical zone.

All 180 industry IDs occur exactly once across mats, slots, removed and retired.
Immutable tile/topology metadata comes from the separately verified C# catalog.
Tile resource bounds, market bounds, link stock, merchant distribution, income,
actor/phase/pending consistency and chance pool conservation are checked.
This is a structurally valid snapshot boundary, not a proof that an arbitrary
caller-supplied state is reachable through a full legal game. L05 qualifies moves.

The public player `debt` is exposed only in the debt phase. It is not a complete
substitute for `settlement_debts`: after all liquidation decisions, residual debt
can still reduce VP when the paused Rail shuffle finishes. Money already on a
player is not subtracted from this residual amount again. The diagnostic flow
exporter reads the actual frozen engine settlement dictionary.

The source C# RNG seed/state, engine event revision, object GUIDs, event journal,
UI liquidation lists and past presentation callbacks are excluded. Explicit
chance plus ordered cards/picks supply the future stochastic input. Semantic
host revision is retained. Derived hand counts/discard views/known wildcards
come from cards and history; permitted recall is a separate L06 search input.

## Native representation and operations

State has owned vectors/arrays and numeric tile/card/slot/link/merchant IDs.
Public history is typed and interned, with semantic choice strings reconstructed
on export. There is no retained JSON tree or C# replay journal inside Rust.
The immutable catalog can be shared. Every mutable collection, including the
pending chance and derived indexes, belongs to its state/session.

`brass-upstream-rules::utils::dsu::DisjointSetUnion` rebuilds physical connectivity,
including three-region links. `fixedbitset` indexes owned slots/links, network
regions and nonempty coal/iron/beer sources. These are physical indexes; distance,
payment eligibility and staged legal choices are L05 work. Income conversion
reuses the upstream pure income function and is verified against the fixtures.

The worker advertises `state_format` and these diagnostic capabilities:

- `state_import {session, state}`: validate a full DTO, then replace live state.
- `state_export {session}`: reconstruct a fresh DTO and diagnostic index view.
- `state_save {session}`: replace that owner's single private save slot.
- `state_restore {session}`: restore the saved state and rebuild indexes.
- `state_clone_check {session}`: mutate a private clone across its collections,
  verify the original/save point are unchanged, then restore and compare again.

Responses have `status: "state"`, `session`, `owner`, `state`, `indexes`, `saved`
and optional `clone_isolated`. A failed import preserves both live and saved
state. A successful import leaves the save slot intact, enabling A/save/B/restore
without transferring the snapshot to another session. EOF/shutdown clears both
owners; there is no persistent or cross-session snapshot handle.

The configuration remains two independent owners with one computation thread
each. `target_rules_qualified` remains false: this milestone cannot execute a
game, construct permitted beliefs, infer with a neural net or search.
