# L05 — Staged native rules, transitions and explicit chance

Date: **2026-10-25**. Status: **complete**. The
[qualification receipt](../artifacts/rust-backend/20261025-l05-qualified-v4/qualification.json)
records zero unexplained legal/state/reward differences. Final logs, per-profile
parity reports, raw measurements and source/executable archives are under
`artifacts/rust-backend/20261025-l05-qualified-v4/`.

## Implementation

The typed Rust state now generates the full ordered legal set for ordinary
actions, resource payments, continuations, merchant development bonuses and
forced liquidation. It applies staged choices and explicit card outcomes,
including turn settlement, era scoring/cleanup, the Rail redeal, terminal
tie-breaking and shared wins. Legal generation enumerates the current stage
directly, without constructing complete atomic plans first.

The main implementation is in `native/brass-ai-core/src/state/choices.rs`,
`phases.rs` and `chance.rs`. It reuses the existing upstream DSU, market pricing,
income conversion, typed catalog and `fixedbitset` indexes. All 40 imported
upstream modules and both dependency lockfiles are preserved. The custom code
translates our staged rules and engine flow; it does not replace the upstream
crate with another general game engine.

The opt-in worker adds `rules_snapshot`, `rules_apply` and `rules_chance`, with
matching methods on the Python diagnostic client. State remains private to
**two independent sessions with one computation thread each**. Every request
requires the current revision. A transition prepares one owned successor and
commits only after successful execution and response preparation. Illegal
choices/outcomes, stale revisions, numeric limits and next-state legal-set
errors preserve live state and the independent save slot.

The [staged rules contract](../configs/contracts/rust-backend/staged-rules-v1.md)
documents the boundary. [Native build instructions](../native/README.md) include
the capture and qualification commands.

## Independent reference and coverage

`RulesExporter.cs` is a separate diagnostic executable linked against the
hash-verified frozen C# runtime. It adds complete private state and settlement
debts to the reference capture without rebuilding the host, changing its rules,
or touching the installed game.

The main roots reuse the immutable L04 state exports and L02 legal/chance
projections: **540 original valid roots plus six separately corrected authored
roots**. The original six inconsistent fixtures remain rejection tests. For
each full-game command, the new C# capture must reproduce the original L02
projection, including every available masked view, before its additional
private fields become comparison inputs.

During qualification, concurrent g950 training work changed the shared recovery
sources. The preservation guard stopped that attempt. The final qualifier
materializes all 462 reference source files from the immutable L01 archive,
adds the four L02 harness files pinned by its own qualification receipt, and
adds the current native Python client as a separately hashed candidate overlay.
Every executed reference file must match the original manifest and archive;
Python import identities are recorded. Concurrent checkout drift is reported
before and after, while its files are left intact. This preserves the g890/L01
comparison identity without undoing the training updates.

| Suite | Cases | Comparison |
| --- | ---: | --- |
| Accepted roots | 546 | Complete state, exact ordered legal set and chance support/probabilities |
| Complete games | 20 / 8,344 transitions | Every private state field, choices, chance, terminal scores and existing reward returns |
| Alternative moves | 1,406 | Every alternative at authored/initial roots; representative choices at dense development roots; next-state import and restore |
| Existing L04 state suite | 549 accepted snapshots | Import/export, independent indexes, clone/restore and malformed-state rejection |
| Rejected rules requests | All seven boundary categories | Ordinary, payment, continuation, bonus, debt, chance and terminal state remain unchanged |

The roots span **277 Canal and 269 Rail** states; the largest ordered legal set
has **413 choices**. Full-game transitions are distributed as follows (the
64-card Rail shuffle is still part of the paused Canal boundary):

| Pre-transition boundary | Canal | Rail |
| --- | ---: | ---: |
| Ordinary action | 1,200 | 1,280 |
| Continuation | 1,299 | 1,315 |
| Payment | 389 | 248 |
| Merchant bonus | 4 | 1 |
| Debt liquidation | 52 | 76 |
| Explicit chance | 1,840 | 640 |

All 14 staged choice kinds occur in the full games and in the branch coverage.
The full games contain 5,864 choices and 2,480 chance outcomes; alternative
branches add 1,406 choice applications. These are comparison counts, not a
claim that every possible position or configuration has been qualified.

The full-game suite checks the existing Python reward implementation using
native public consequences: terminal utility, causal insolvency charges and
combined returns. This qualifies those consequences; the native search reward
ledger is still a later milestone.

The four holdout games and their 104 roots are correctness checks only. Timing
input selection uses development roots. Coverage details and named regression
locations are retained in each profile's `rules-parity.json` report.

## Retained regressions

The implementation and tests preserve these target-rule details:

- Scout replaces the two traded cards at their original hand positions,
  including repeated card types.
- Unbuilt Canal-only tiles remain on player mats when Rail starts.
- Nottingham merchant beer grants 3 VP; Shrewsbury grants 4 VP.
- Bonus-VP subtotals reset at both era boundaries. Final Rail scoring does not
  collect another round of income.
- Wild location cards exclude the rural brewery regions. Physical network
  emptiness permits the reference's initial network/build choices.
- Debt actor, engine flow cursor and turn owner remain distinct; residual
  settlement penalties survive the pre-Rail chance pause.
- A numeric overflow after a discard cannot partly apply an action. A worker
  response error caused by an unreachable trusted import cannot commit its
  otherwise prepared successor.

Concrete captured examples include `game-731001` step 3 (Scout) and step 236
(15 unbuilt Canal-only mat tiles survive the Rail transition), `game-731005`
step 308 (Nottingham), `game-731015` step 161 (Shrewsbury), and `game-731005`
step 406 (a nonzero bonus-VP subtotal resets at final scoring). Native tests
separately exercise arithmetic rollback, response-failure rollback, an entire
all-Pass game and generated legal prefixes with import/fork/save isolation.

## Qualification and measurements

Formatting and strict project Clippy passed. All **21 Rust tests passed in both
debug and release**, and all **21 real-process Python tests passed on each of
debug, release and profiling**. The native suite includes 64 generated staged
prefixes per run, retained state properties, full-game settlement and rollback
tests. The Python suite compares every game transition and all branch
alternatives, exercises imports/save/restore throughout play, and checks both
owner sessions. The isolated-reference guard also rejected a deliberately
altered reference file.

| Rule microbenchmark | Mean | 95% confidence interval |
| --- | ---: | ---: |
| Initial legal set, 388 choices | 357.10 µs | 329.34–386.92 µs |
| Initial validated build step | 217.23 µs | 206.90–228.26 µs |
| Complex Rail-payment legal set, 1 choice | 0.60 µs | 0.56–0.64 µs |
| Complex validated payment step | 15.26 µs | 14.18–16.39 µs |

These are Criterion sample means with 20 samples, 200 ms warm-up and a 1-second
measurement window per operation. Inputs are `game-731001-root-000` and
`game-731009-root-026`. The latter has eight occupied industries, three links
and 266 history entries, and is the most occupied captured development
Rail-payment root. Its single legal payment explains the small generation time;
it does not represent an ordinary Rail turn's branching cost.

The `rules` Criterion benchmark measures legal generation and validated staged
application. Application includes full legal validation, the transaction's
owned clone, mutation, history append and index reconstruction. The setup clone,
JSON/pipe transport, next-response serialization and neural inference are
outside this timing. Initial and dense Rail-payment inputs are recorded with
their root IDs and hashes. These measurements are not a matched search speedup.
No paired C# timing or neural search ran here, and no A8 performance target is
claimed from these microbenchmarks. L07 profiles broader simulator workloads.

## Provenance and preservation

| Identity | SHA-256 |
| --- | --- |
| Native source fingerprint | `2cfc098ac82a75833550c9450b2cdb22c377055d133256c41730c9af2717e067` |
| Unchanged workspace Cargo lock | `36f78617f4ccc3bcf60caa46f42a97f90b057b07ef708b1cb0616b09d76c888b` |
| Qualification receipt | `16ba5d84d1c1ec5e0e8e56b5fcae18ec293b9b80180ad3e49554ddc9ac3f69b2` |
| Portable release worker | `18020014f5f1f88aea90dd34a44c39e0adc63882af4d353cb59123e52d9181f3` |

The release executable is 1,856,121 bytes. All three build identities and
executables are preserved in the final run directory. The L01 reference archive,
L02 fixture/harness identities, L03/L04 evidence archives, upstream source and
license, installed three-DLL fallback, original 98 saves and g890 model/bundle
hashes remain valid. The
[final verification receipt](../artifacts/rust-backend/20261025-l05-qualified-v4/final-verification.json)
checks current source and binary hashes against both archives, verifies the C#
runtime used for capture, and records the final documentation hashes.

Earlier interrupted captures and qualification attempts remain as diagnostic
history. The completed L05 evidence is the `20261025-l05-qualified-v4` directory.

## Remaining boundary

L05 does not install the Rust backend in Unity or begin lab training. The worker
continues to advertise `target_rules_qualified: false`: **L06** adds permitted
observations, recall and sampled worlds; **L07** qualifies the usable simulator
adapter and expanded games. Native neural search follows later.

The g890 checkpoint remains the temporary migration test model and was not
executed or modified here. The matched search budget remains **4 seconds per
whole turn**; the local settings remain **4/8/10/12 seconds**. The migration still
requires local acceptance before lab deployment.
