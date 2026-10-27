# L13 — Local Alpha Test integration

Date: 2026-10-26. Status: **complete; opt-in Alpha Test installed**. L14/L15 remain pending.

L13 adds the packaged Rust backend and an **Alpha Test** entry in the existing
game AI selection panel. Alpha Test uses the pinned **temporary g890 NN**, not
the model still being trained. The native backend retains two independent
sessions with one CPU compute thread each and the qualified singleton batching
policy. L14 performance/strength and L15 complete-game acceptance remain pending.

## User-visible behavior

- Alpha Test is offered for standard four-player games. Selecting it changes
  every AI seat to the NN family; selecting V4 changes every AI seat to V4.
  Opposite-family seat choices are omitted. Human seats remain supported.
- Each AI seat independently selects **4s, 8s, 10s or 12s**. Different settings
  can coexist on one table. Changing family preserves the other seats' settings;
  new NN saves retain each seat's allowance and exact model/export identity.
- Action two and retries share the original ordinary-turn allowance. Mid-turn
  save/load preserves the remaining time and the actor's permitted recall.
- An invalid model, stale result, failed worker or disabled runtime stops the
  NN table. No V4 player is silently substituted. Existing V4 saves retain
  their versioned time mappings, including the old Max-to-12s mapping.

## Implementation

[`local.rs`](../native/brass-ai-worker/src/local.rs) exposes a small framed local
protocol around the qualified L12 Pair. The two persistent owners retain the
engine, features, search trees and CPU ORT sessions. Startup warms both paths;
Unity submits detached permitted inputs through its existing scheduler. The
control reader performs I/O only. Python is not part of the runtime.

The [native C# adapter](../src/BrassLocalAI/Native/NativeProcess.cs) verifies
package/source/model identities, drains cancellation and terminal responses,
and waits for actual retirement before releasing scheduler resources. Existing
Unity generation/revision guards and isolated complete-action validation remain
in front of each main-thread commit. Hiding the thinking panel does not cancel
work; history/menu/game replacement does. A weak per-game turn clock prevents
same-object retries from receiving a new allowance.

The [versioned NN save](../src/BrassLocalAI/Native/NativeGameState.cs) records
family, seat identity, allowance, model/export and a bounded clock/recall journal.
The three existing public-history/action adapters are reused unchanged. No seed,
private deck, full vendor events or opponents' private hands cross the boundary.
The NN liquidation route uses the public contract's color order for prompt
ownership and the game's existing persisted sale/save flow. V4 liquidation
order and valuation are preserved.

Local utility acceptance remains separate from measured readiness. The policy
binds the exact g890 checkpoint/model/export/reward identities to the user's
2026-10-25 local decision. `final_model` is false and `utility_calibration`
remains explicitly null. Another identity, absent receipt, lab scope or invented
calibration is rejected. Historical model manifests and qualification evidence
are unchanged.

See the [transport/save contract](../configs/contracts/rust-backend/local-native-v1.md).

## Evidence

All gates below passed. The final run is under
[`20261026-l13-qualified`](../artifacts/rust-backend/20261026-l13-qualified/).
The qualification script records each command and its output, snapshots the
compiled C# sources, packages the native Rust source and notices, and seals
artifact hashes only after all checks pass.

| Check | Coverage |
| --- | --- |
| Rust debug/release tests and Clippy | **82 tests in each profile**; owned code warning gate (historical vendor warnings retained) |
| Exact ONNX output test | **11 positions**, frozen L09 tensor/output oracles against the packaged model/runtime |
| Native process contracts | **8 passing tests**: startup/admission, identity and private-input rejection, all allowances, forced/debt/staged paths, cancellation, duplicate roots, EOF/shutdown |
| Production transport replay | 44 frozen complex roots; 39 original roots independently replayed in C#, five rotated roots retain native evidence |
| Native Unity player | **813 checks**: real panel and mixed-time save/load, old-save mappings, all four shadow/live allowances, both actions, reduced-clock persistence, hidden panel, history/menu/replacement cancellation, worker death, five liquidation choices across reload |
| Preservation | Exact game assembly, **14,941 unchanged existing methods**, all **48 embedded V4 profiles**, source-scope audit, original test NN and unrelated user document |

The [529-file artifact audit](../artifacts/rust-backend/20261026-l13-qualified/final-verification.json)
and [installed verification](../artifacts/rust-backend/20261026-l13-deployment/installed-verification.json)
pass. [Installation tests](../artifacts/rust-backend/20261026-l13-deployment/installation-tests.json)
verify exact rollback and recovery after an interrupted publication. The
[reviewed panel](../artifacts/rust-backend/20261026-l13-qualified/l13-alpha-panel.png)
and [mixed-time table](../artifacts/rust-backend/20261026-l13-qualified/l13-alpha-table.png)
show the final controls. Shadow delivery was 3.775 / 7.385 / 9.235 / 11.055 seconds
for 4 / 8 / 10 / 12-second settings, with 324 / 436 / 545 / 653 Unity frames.

Shadow timings report Unity delivery and responsiveness for these fixtures;
they are not a matched implementation-speed result or a playing-strength claim.
The 44-root replay uses a four-second per-root allowance. The whole-turn matched
study, representative latency distributions and complete matches remain L14/L15.

## Build and local deployment

Reproduce in a fresh evidence directory:

```powershell
python scripts/qualify-rust-local.py --out artifacts/rust-backend/<new-run>
python scripts/qualify-rust-local.py --out artifacts/rust-backend/<new-run> --audit
```

The native ZIP contains the worker, ONNX Runtime, unchanged model export,
policy/build manifests, source snapshot and dependency notices. It contains no
proprietary Steam assembly. The isolated game candidate has the unchanged
qualified 1.4.34 game assembly and scoped helper changes.

```powershell
./scripts/install-rust-alpha.ps1
./scripts/install-rust-alpha.ps1 -Action Rollback
# Only for a durable interrupted-installation intent:
./scripts/install-rust-alpha.ps1 -Action Recover
```

The installed package is `1.4.35-alpha-test`; its pre-installation backup is
`artifacts/patch-recovery/alpha-32ffabc13c9046d3ac3e54d90ccb9ffb/`.
The sealed qualification records the pre-installation state; the separate
installed verification records successful deployment and a fresh native startup.

The Alpha installer requires the sealed L13 receipt and exact baseline hashes,
backs up the previous helpers and receipt, verifies installed hashes and retains
recovery information. It identifies the result as `1.4.35-alpha-test`, with
`localReleaseQualified: false`. Rollback restores the prior installation and
keeps the retired native package and all NN saves. Those saves require the
matching Alpha Test package to resume. `BrassNative/disabled` is the runtime
stop switch; it removes Alpha Test from setup and prevents further NN commits.

No training, teacher or lab runtime was migrated by this step. The next step is
L14's four-second matched performance and decision-quality qualification.
