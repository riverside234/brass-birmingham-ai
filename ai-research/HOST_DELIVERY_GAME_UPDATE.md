# Host delivery game update — 1.4.44

Installed and verified on 2026-10-29 (America/Chicago). The game now uses the qualified host optimizations: revision-bound event reuse, a small post-action clock snapshot, direct observation tokens, reusable board indexes, selected-choice legality resolution and 60 FPS validation handoffs. Native search retains its 30 FPS rendering cap. Identity, actor, revision, cancellation, receipt and final legality checks remain active.

The [host comparison](HOST_DELIVERY_OPTIMIZATION_REPORT.md) measured 50–56% lower outside-native overhead and 71.1% fewer managed object bytes in its bounded allocation probe. That comparison used the previous 1.2-second K32 floor. This installation retains the exact measured C# binary and reduces only the native K32 delivery floor to 0.8 seconds. Shared Rust rules/search mathematics, model/export, ORT provider and all policy files are unchanged. K16 retains its previous timing.

| Search allowance | New K32 delivery reserve | Maximum search-work interval |
| --- | ---: | ---: |
| 4 seconds | 0.8 seconds | 3.2 seconds |
| 8 seconds | 0.8 seconds | 7.2 seconds |
| 10 seconds | 1.0 second | 9.0 seconds |
| 12 seconds | 1.2 seconds | 10.8 seconds |

The reserve is paid within the original turn clock. No retry replenishes that clock. Five native budget tests cover unchanged deadlines, narrowing, the new floor and retained larger defaults. Pre-install four-second smokes completed both actions in Canal at 2.828 seconds, late Rail at 3.029 seconds and a Rail root with effective K32 at 3.237 seconds. The K16 control completed in 3.504 seconds. Each passed 760 checks, including completed-turn save/load.

The actual installed player then passed **K16 in 3.567 seconds** and **K32 in 3.133 seconds**, with 760 / 760 checks, two committed actions and completed-turn save/load. Both use a four-second allowance, temperature 1.0 and two independent one-thread CPU sessions. NN remains the temporary g890 Alpha Test model. These are short functional checks, not a strength or latency-tail guarantee.

All 142 user save/history files match their pre-install hashes. The installer does not write menu preferences, and the smoke harness keeps its menu choices in memory. Direct registry preference measurement was unavailable under the tool's permissions; it is not reported as a measured pass. Interrupted-install recovery and exact rollback passed in a disposable copy with an unchanged saved-game sentinel. Only the AI assembly and three native package files were published; the game/helper assemblies and policy files are preserved.

With the game closed, rollback is:

```powershell
& .\scripts\install-host-delivery.ps1 -Action Rollback
```

For an interrupted publication, use `-Action Recover`. The backup is `C:\Program Files (x86)\Steam\steamapps\common\Brass Birmingham\artifacts\patch-recovery\host-delivery-8f939d91d60f4a0b8c66a34aff247a21`. Saves are preserved.

Evidence: [installed verification](../artifacts/host-delivery/20261029/installed-verification.json), [release qualification](../artifacts/host-delivery/20261029/release-1.4.44/qualification.json), [native build and tests](../artifacts/host-delivery/20261029/reserve08-native/native-build.json), [recovery test](../artifacts/host-delivery/20261029/upgrade-test.json), and [save preservation](../artifacts/host-delivery/20261029/user-data-verification.json). Installation changes no lab deployment or training bundle.
