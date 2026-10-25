# L05 — Staged native rules, transitions and explicit chance

Date: **2026-10-25**. Status: **qualification in progress**.

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

## Qualification and measurements

Final build, parity and measurement receipts will be recorded after the full
qualification run succeeds.

The `rules` Criterion benchmark measures legal generation and validated staged
application. Application includes full legal validation, the transaction's
owned clone, mutation, history append and index reconstruction. The setup clone,
JSON/pipe transport, next-response serialization and neural inference are
outside this timing. Initial and dense Rail-payment inputs are recorded with
their root IDs and hashes. These measurements are not a matched search speedup.

## Remaining boundary

L05 does not install the Rust backend in Unity or begin lab training. The worker
continues to advertise `target_rules_qualified: false`: **L06** adds permitted
observations, recall and sampled worlds; **L07** qualifies the usable simulator
adapter and expanded games. Native neural search follows later.

The g890 checkpoint remains the temporary migration test model and was not
executed or modified here. The matched search budget remains **4 seconds per
whole turn**; the local settings remain **4/8/10/12 seconds**. The migration still
requires local acceptance before lab deployment.
