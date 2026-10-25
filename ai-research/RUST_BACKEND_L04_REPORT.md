# L04 — Native state import, export and restoration

Date: **2026-10-25**. Status: **complete**. Final qualification passed with zero
unexplained differences. The [qualification receipt](../artifacts/rust-backend/20261025-l04-qualified/qualification.json)
and raw logs, parity reports, measurements, source and executable archives are
under `artifacts/rust-backend/20261025-l04-qualified/`.

## Implementation

`native/brass-ai-core/src/state/` imports the strict, versioned
`brass-native-state/1.0.0` DTO into owned Rust data, reconstructs exports, and
provides private clone/save/restore. The state includes actions remaining,
ordinary/payment/continuation/bonus/debt phases, both eras, terminal results,
ordered cards, stable tile identities, public history, pending resources,
partially resolved chance draws, and pending round/settlement information.

The original upstream `GameRunner` remains a separate diagnostic capability.
Its atomic state cannot represent our C# staged phases directly. The new state
uses dense numeric identities and typed history; it retains no JSON tree or
checkpoint replay journal. Wire serialization reuses Serde; derived physical
connectivity reuses the upstream DSU; ownership/resource indexes reuse
`fixedbitset`; derived income reuses the upstream income function. No upstream
vendor module was edited.

The worker adds `state_import`, `state_export`, `state_save`, `state_restore` and
`state_clone_check`. Each of the **two independent sessions has one computation
thread** and one private save slot. A failed import leaves live and saved state
unchanged. Restoration rebuilds indexes. A private-fork diagnostic mutates all
mutable state categories, verifies isolation, then restores the saved state.

See the [state contract](../configs/contracts/rust-backend/semantic-state-v1.md)
and [native build instructions](../native/README.md).

## Reference capture and corrections

Separate C# diagnostic executables reference the exact frozen L01 host and
runtime. They do not rebuild or change the oracle or installed game. Raw capture
restored all **546** frozen roots and matched their original canonical
projections exactly. Inputs, raw captures and upgraded state DTOs are under
`artifacts/rust-backend/20261024-l04/`.

Three state details required explicit handling:

- C# does not put every developed, liquidated or era-pruned industry in its
  board removed list. A distinct `retired` identity partition preserves that
  list and accounts for all 180 tiles.
- Rail discard prefixes alias cards already redealt. Card conservation counts
  only active zones and discard suffixes. Ordered chance picks are retained
  separately from physical card zones.
- The debt actor can differ from the engine flow cursor. Public observations
  also hide residual debt during a rail-shuffle pause. The flow exporter read
  the actual settlement dictionary for **77** pending-settlement roots, so
  outstanding VP penalties survive restoration.

Six frozen authored fixtures are structurally inconsistent. The importer rejects
them without weakening validation or editing the frozen corpus:

| Original fixture | Defect | Separate C# correction |
| --- | --- | --- |
| `edge-shared-merchant-beer` | Link stock and merchant distribution | Balance link stock; restore the standard merchant pool while retaining both Gloucester Cotton destinations |
| `edge-sale-payment` | Same | Same corrected setup, replay original choices |
| `edge-free-development-bonus` | Same | Same corrected setup, replay original choices |
| `edge-sale-continuation` | Same | Same corrected setup, replay original choices |
| `edge-pending-debt` | Turn owner disagrees with round order | Align turn owner with the fourth turn; retain the debt actor |
| `edge-partial-debt` | Same | Same corrected setup, replay original choices |

`regressions-v2/regressions.json` contains the accepted corrected variants and
their source hashes. The earlier `regressions/` directory records the intermediate
repair attempt, before merchant conservation was corrected. The repair driver
asserts that C# changed only the explicitly corrected semantic fields.

The accepted main suite has **540 unchanged frozen roots + 6 corrected authored
roots**: 436 original development roots, 104 original holdout roots and six
corrected development cases. Three additional C# states cover a partial draw
and two different pick orders for a partial rail shuffle. Thus **549 accepted
snapshots** are checked, alongside rejection of the six original invalid roots.

| Accepted main-suite phase | Roots |
| --- | ---: |
| Ordinary action | 124 |
| Payment | 112 |
| Continuation | 123 |
| Merchant bonus | 6 |
| Debt | 61 |
| Draw chance | 80 |
| Rail-shuffle chance | 20 |
| Terminal | 20 |
| Total | 546 |

There are 277 Canal and 269 Rail roots. The independent Python graph traversal
checks connectivity, owned slots/links, network regions and resource indexes;
it does not use the Rust DSU or its numeric mapping.

## Qualification and measurements

Formatting and strict Clippy for project code passed. All **16 native tests**
passed in debug and release, including property tests with 256 generated cases
per property. All **17 real-process Python tests passed on each of debug,
release and profiling builds**, covering the 549 accepted snapshots, independent
indexes, private clone/restore, invalid imports and transport/owner behavior.
The six existing warnings in the byte-identical upstream modules are preserved.
Malformed inputs cover contracts, shape, immutable definitions, identity and
conservation errors, phase/actor/debt inconsistencies, forged history, chance
progress and terminal results. A final regression rejects a rail-shuffle pool
that includes a card still held in a hand, even when total card counts balance.

Repeated qualification also exposed an intermittent owner-retirement race in
the existing worker: a queued request can retain its reply sender after the
owner's receive loop exits, leaving the caller waiting. A separate owner-exit
channel now wakes that caller with `session_failed`. A deterministic test
recreates the queued-request/closed-owner case, and the existing panic test
checks that the other session remains usable. The interrupted attempt is
retained under `20261025-l04-final/`; `20261025-l04-verified/` is the earlier run
before the two final regressions. Only `20261025-l04-qualified/` represents the
final implementation.

Criterion measures typed-state import/export/clone/restore on initial and
complex Rail-payment positions. These timings exclude JSON transport and NN
work. A separate allocator probe measures 100 warm clones; allocated bytes are
not a count of bytes copied. These measurements establish no search speedup.

| Operation | Initial state, mean µs | Complex Rail payment, mean µs |
| --- | ---: | ---: |
| Import and validate typed DTO; rebuild indexes | 51.60 | 68.95 |
| Reconstruct exported DTO | 43.23 | 74.49 |
| Owned clone and drop | 3.72 | 5.17 |
| Restore saved state and rebuild indexes | 4.74 | 5.45 |

These are Criterion sample means with 20 samples per workload. Raw estimates
and confidence intervals are retained. The complex fixture is
`game-731009-root-026`, selected by occupied slots then history length among Rail
payment states; it has 266 public-history entries. Warm clone allocation traffic
was 65 allocations / 4,441 bytes per initial clone and 79 allocations / 26,361
bytes per complex clone. The shallow `NativeState` size is 1,760 bytes; owned
vector contents are additional. This is an initial allocation profile, not the
final search allocation budget or a comparison with a C# history replay.

All three executables report source fingerprint
`a707affe9c039dcdbe49c2ae6604c8a0709efd4ce867da9bcce6b5a0dd13e29a`
and Cargo lock hash
`36f78617f4ccc3bcf60caa46f42a97f90b057b07ef708b1cb0616b09d76c888b`.
The portable Windows x64 release executable is **1,699,111 bytes**, SHA-256
`fcdc4ea0567ed10bcd51880e9f6baf78d7f5cb23001d4165fc4bda5fa372fe04`.
It uses the pinned Rust 1.99.1 GNU toolchain and the L03 build profiles.

## Preservation and remaining work

The installed fallback, original saves, g890 checkpoint and frozen L01/L02/L03
evidence passed before/after preservation checks. All 40 imported upstream
modules remain byte-identical. The previously recorded unrelated
`scripts/export-test-nn.py` drift remains unchanged; its frozen source is still
preserved. The C#/Python backend stays
the default. No Unity integration, model change, NN evaluation, search or lab
training is part of L04.

`target_rules_qualified` remains false. **L05 is next:** staged legal choices,
transitions, resource payments, explicit chance application and turn/era/final
settlement parity. The first usable simulator remains L07. Permitted views and
belief/recall, NN inference and Rust search retain their later gates. The matched
NN/search benchmark remains **4 seconds per whole turn**, with two independent
one-thread sessions. Local settings remain **4s / 8s / 10s / 12s**, and local
qualification still precedes the lab migration.
