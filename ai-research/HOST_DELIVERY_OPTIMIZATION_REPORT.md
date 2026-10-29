# Host delivery optimization — qualified local candidate

Date: 2026-10-29 (America/Chicago). Implemented and smoke-tested in the isolated Unity player. **This task does not install the candidate:** the game remains the verified 1.4.43 release. The frozen candidate is `artifacts/host-delivery/20261029/candidate/`.

The installed `search_v1` binary, V205-off setting, g890 temporary NN, ONNX Runtime, two independent one-thread sessions, all policies and both K-specific deadline reserves are unchanged. All matched live turns use a four-second search allowance and temperature 1.0. K16 retains its original reserve; K32 retains 1.2 seconds. These results do not establish playing strength or latency tails.

## Implementation

- After ordinary action one, `search_v1` captures only the clock-journal fields instead of replaying public history into a second complete NN input. The projected-successor path retains its full observation and comparison when that contract is present.
- Event history is reused only within one bridge, rules object and revision. Changing the revision/object reloads it; loading or moving history cancels the bridge. NN turns skip legacy intent-history arguments when there is no intent. Two ordinary actions now require two storage reads instead of eight; the profiler still sees six cheap-or-real `Events()` calls.
- Allowlisted observations build JSON tokens directly. General `Wire.Token` parsing semantics are unchanged. Board-object indexes bind the rules object and era; mutable tile/owner values remain live. The selected-choice resolver shares the canonical enumeration predicates, filters to the requested branch, and rejects missing or ambiguous selected IDs. Canonical complete enumeration stays available and supplies the oracle.
- Only detached validation delivery uses a 60 FPS cap. Native search remains at 30 FPS, concurrent search takes priority, and an existing lower software cap is respected. Optional-work suspension, cancellation ownership and exact frame/VSync restoration remain active.

Identity, revision, actor, cancellation, exact observation/composition/path equality, single-use receipt and current semantic legality checks still precede main-thread mutation. Action two is captured after the actual first action. The original turn clock is never replenished.

## Short matched Unity results

Each row is one fresh baseline/candidate pair with the same permitted root, native package, settings and selected actions. Execution order is counterbalanced. The displayed difference is total delivery minus native response-ready duration; it includes preparation and dispatch because those timers start at different boundaries.

| Root / setting | Baseline outside native timer | Candidate outside native timer | Reduction |
| --- | ---: | ---: | ---: |
| canal-k16 | 356.7 ms | 155.9 ms | 56.3% |
| canal-k32 | 330.3 ms | 156.1 ms | 52.7% |
| rail-k16 | 463.5 ms | 215.7 ms | 53.5% |
| rail-k32 | 476.1 ms | 236.3 ms | 50.4% |

Both actions and completed-turn save/load passed in all eight runs. Native response-ready timing varied between processes: the K16 late-Rail candidate took 3.701 seconds overall versus 2.432 seconds for baseline, while its outside-native portion still improved from 463.5 to 215.7 ms. The unchanged native binary took 3.485 versus 1.968 seconds in that pair. Therefore this smoke supports a host-side improvement, not a claim that every total turn becomes faster.

A separate K32 diagnostic pair measured C# time outside its containing native round trip at **338.8 → 149.1 ms**. Full NN captures fell from two to one; detached validation spans totaled **107.6 → 34.2 ms**, guarded commits **58.2 → 7.3 ms**, and event-history method spans **51.7 → 23.3 ms**. These spans are nested and must not be added together. Instrumentation changes timing; raw uninstrumented results remain separate.

## Fixed replies and correctness

The exact shipped Mono runtime checked 40 ordinary/staged states and 15 complete recorded actions. Both builds passed 3,719 checks and produced exactly equal observations, complete legal sets, committed states, semantic event payload/order/revisions and clock journals. Only the journal creation timestamp is excluded from the cross-process comparison. A separate 300-check oracle covers bonus skip/develop, explicit debt membership and a real seven-turn Canal-to-Rail transition with two legal liquidations; baseline/candidate results are identical.

Across eight matched actions, retained-path managed object allocations fell **383,082,424 → 110,887,688 bytes (71.1%)**. The small allocation-instrumented fixed-action timing sample had median **149.1 → 38.6 ms (74.1% lower)**. These are managed allocation traffic and instrumented isolated timings, not whole-Unity allocation totals, retained memory or an uninstrumented latency guarantee.

The actual Unity lifecycle suite passed **839 checks**, including queued/retiring work, stale actor/revision, same-revision load, fresh bridge after load, cancellation, receipt reuse boundaries, action-two validation, family/difficulty setup and worker failure. Separate post-action-one K16 cancellation and K32 loading smokes passed. Another **45 event/identity + 79 checkpoint/liquidation checks** passed. Saves remain once after completed human/AI End, with no per-action AI save.

## Preserved failed attempts

Two supplemental test attempts had incorrect fixture assumptions: no recorded ordinary root owned collateral, and the late-Canal replay entered real debt. The corrected test uses an explicit conserved debt perspective and executes the actual liquidation. Their failed evidence remains in `edge-candidate/` and `edge-v2-candidate/`.

The first broader lifecycle run found an older test-helper error: it returned the human checkpoint object after already saving a newer AI-turn checkpoint. The helper now reloads the newest checkpoint. The original failure remains in `validation-lifecycle/`; the corrected suite is `validation-lifecycle-v2/`. No production save behavior was changed to satisfy the test.

## Evidence and reproduction

See [assessment](../artifacts/host-delivery/20261029/assessment.json), [fixed-reply assessment](../artifacts/host-delivery/20261029/host-assessment.json), [protocol](../artifacts/host-delivery/20261029/protocol.json), and [candidate qualification](../artifacts/host-delivery/20261029/candidate/qualification.json).

The scoped builder is `scripts/build-host-delivery.py`. `scripts/test-host-delivery.py` runs the exact-Mono oracles/allocation probes; `scripts/check-host-delivery-edges.ps1` runs supplemental branches; `scripts/smoke-host-delivery.py` runs individual short matched pairs, diagnostic profiles, lifecycle smokes and sealing. Existing output directories are protected. Use a fresh study label when reproducing; retain all failed attempts. Native/lab packages, installed managed files, user preferences and gameplay saves are unchanged.
