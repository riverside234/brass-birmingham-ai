# Network input fix — patch 1.4.34

Date: 2026-10-24. Baseline: installed and archived 1.4.33. Local games only.

## Report and diagnosis

A human places one Canal link, possibly opens the industry mat or double-clicks,
then clicks End Turn. The next AI stops with “You can only place one link per
action in canal era.” Reloading the last saved turn clears the problem.

The preserved player log shows a human link-button callback first throwing a
`NullReferenceException` in `NetworkAction.PlaceLink`, followed by that exact
Canal exception in the AI live commit. The completed Network action has already
cleared its player. A stale button listener nevertheless calls it again. The
vendor code increments its internal link counter **before** dereferencing the
cleared player, leaving an invisible counter value of one. The game reuses that
Network action object for the next player. The AI's detached validation does not
contain the counter corruption and passes; committing to the live game fails.

Opening the industry mat is not established as a separate rules defect. The
regression opens/closes the actual mat prefab during a valid Network prompt and
checks that subsequent placement still works.

## Changes

- Guard the human link-selection entry point against completed, cancelled,
  replaced and inactive actions; wrong actors; history browsing; missing card
  preparation; pending resource payment; and noncurrent or unavailable links.
- Ignore obsolete UI callbacks without consuming cards, cash, actions or links.
  A small reentrancy guard also prevents a callback from entering itself.
- Validate domain action ownership, its prepared player and the current unowned
  board link **before** entering the vendor placement method. Invalid direct
  calls throw without incrementing the link counter or applying events.
- Allow AI search reconstruction, which restores a prepared player but does not
  retain the discarded card reference. The human UI still requires its selected
  card. Preserve Canal's one-link limit and Rail's two-link limit and payments.

Only two existing game methods change. The original human selection body is
preserved behind the new guard. All previous helper methods are identical to
1.4.33. The AI DLL and its 48 embedded profiles are byte-for-byte unchanged,
including two workers and Easy 4s / Mid 8s / Hard 10s / Max 12s.

## Reproduction and qualification

The isolated 1.4.33 regression invokes the real Unity link button twice, completes
the human replay, ends the turn, validates an AI Network choice and attempts its
live commit. It reproduces both the null-player failure and the subsequent Canal
exception for 2, 3 and 4 players: **36 assertions passed** confirming the defect.
The expected exceptions are caught inside the test.

The candidate suite exercises normal and interrupted Canal/Rail replays, duplicate
buttons, stale other-link callbacks, information panels, history browsing,
cancellation, direct unprepared engine calls, AI handoff and reconstructed
double-Rail search. **969 assertions passed**. Native presentation lifecycle tests
add **184 assertions**. Fresh live V4 turns and packaging checks are recorded in
the release qualification reports, along with the exact tested DLL hashes.

The V4 suite passed **3,117 assertions across nine live turns**. After installation,
the Network suite passed **969 assertions** and startup/menu passed **1,502**.
Packaging passed **86 checks**, including rollback, stale/wrong qualification
rejection and the standard installer dispatch to the combined installer.

Statistics, Game History, Card Tracker and their visual reports are retained from
the qualified baseline with their original fingerprints; they are not represented
as fresh captures. A method-by-method comparison verifies their code is preserved.

## Build and installation

1. Run `scripts/build-network-input-fix.ps1` against frozen 1.4.33.
2. Run `tests/BrassPatch/InformationUI/run-native.ps1` for `link`, `presentation`
   and `menu -Quick` using `artifacts/network-input-build`.
3. Run `tests/BrassPatch/Presentation/qualify-network-input.ps1`, then the
   InformationUI packaging tests against that build.
4. Install with `scripts/install-combined-patch.ps1 -BuildDirectory
   artifacts/network-input-build`; verify the installed link and menu suites.
5. Archive with `scripts/archive-network-input-release.ps1` and stage the release.

The installer verifies the baseline, backs up its three DLLs, publishes the new
files atomically and keeps online play disabled. Personal saves are not modified
by the regression fixtures. Restart the game to load the installed patch; an
already corrupted in-memory turn must be reloaded once.

For a later rebuild after staging a newer release, choose a fresh build directory
and pass `-BaselineHookInventory` pointing to
`artifacts/patch-releases/1.4.34/baseline/1.4.33/Managed/Assembly-CSharp.dll.manifest.json`.
That archive retains the matching 1.4.33 DLL and its hook inventory.
