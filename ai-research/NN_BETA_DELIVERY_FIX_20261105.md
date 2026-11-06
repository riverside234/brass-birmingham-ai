# Beta delivery patch — 1.4.52

The November 5 Beta failure occurred on the second action after a 12-second
search. The native reply was ready at 12,647.7 ms; Unity reached the loan commit
check at 12,951.4 ms. With a 13,000 ms absolute deadline and a required 100 ms
commit reserve, it correctly stopped. The avoidable delay was first-action
snapshot preparation and an additional Unity/scheduler handoff after search.

`NativeJob` now receives a detached **local validation observation** before
search starts. It validates the first returned action inside the existing
scheduler lease before publishing completion. This is the local observation,
not the differently shaped neural-network/public-history observation.
The receipt retains the original Unity thread ID. Unity still checks the full
live observation, composition, semantic action, lifetime, generation, request,
revision, actor, model and cancellation before committing, once. The second
action still captures and validates the actual state after the first commit.
The timeout message now says NN rather than incorrectly saying Alpha Test for
Beta. Rust search, weights, two one-thread sessions and all difficulty settings
are unchanged. The extra allowance remains 1,000 ms and commit reserve 100 ms.

Short validation used a copy of the reported save through revision 729; original
save files were never modified. On the same position with reply consumption
delayed to approximately 12,650 ms, the previous assembly took 153.1 ms after
resumption versus 107.1 ms for the patch (one matched smoke, not a statistical
benchmark). Both actions completed. The original failure was load-sensitive;
the matched baseline did not fail on this run. The patch removes work from
that critical path without relaxing deadline or legality checks.

Additional checks passed: 4-second Beta turn; cancellation after action one;
loading during the AI turn; expired reply rejected without committing; completed
turn save/load; exact event/state parity for the recorded actions; worker-thread
commit rejection; cancelled receipts; changed identity/composition; and one-use
receipts. Contract runs passed 23 validation checks and 79 checkpoint checks.
Installer, rollback and interrupted-install recovery passed in a disposable
installation. See `artifacts/nn-delivery/20261105` for the build, test receipts,
baseline comparison and installed verification.

Only `BrassLegacyAI.dll` is replaced. The installer checks the exact 1.4.51
baseline, source and native file hashes, saves a recovery copy, and publishes
atomically. Rollback: `pwsh -File scripts/install-nn-delivery.ps1 -Action Rollback`
while the game is closed. Reopen the game and reload the last completed-turn
checkpoint to recover from the already stopped AI turn.
