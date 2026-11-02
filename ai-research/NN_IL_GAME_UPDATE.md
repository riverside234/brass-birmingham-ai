# IL game option — 1.4.50

Installed on 2026-11-02. Choose **AI for whole table → NN → IL** in the player
selection panel. IL is a separate option alongside Alpha Test and Beta. Apply
persists the model choice across restarts; Cancel discards the draft. All AI seats
use the selected NN model, while their 4/8/10/12-second difficulties remain
independent. The largest-text layout keeps all four NN subtabs above the footer.

The source is the newly pulled
`test_nn/features4-v11-2-batch7-il-7f167b8c-gunknown-657fe1901ab3-21d50ddf1a35/`
bundle. Its original checkpoint SHA-256 is
`657fe1901ab35d144e5d86e16b42e795d281946829c87bcb8e0245927b70a263` and
model ID is `94bebe397ac8ab1f8fe3cfb46d7d293cdb8bc10116dcd5efaf2d2afdd813270b`.

IL has no trained utility head. An inference copy uses the existing UtilityNet
extension with an exactly zero residual, producing the installed search objective:

`utility = win_share + 0.1 × (4 − placement) / 3 + 0.001 × min(VP, 180)`

All original tensors and the policy, win, placement and VP outputs are preserved
exactly. No optimizer, fitting, training games or utility-calibration receipt is
created. The inference extension has its own model ID,
`c4cdf78a1b90ae9ade7156878009e34ee355ec71ee67226f3fd8a95d09335a94`; its
receipt retains the original IL identity. The original checkpoint is unchanged.
The exact ONNX export hash is
`a42304fa909cbc833cf258bcbe7d046647a272a6a1424def2e6d968bb079a141`.

The package uses **search_v1**, V205 disabled, and ONNX Runtime 1.22.1 CPU with
two independent one-thread sessions. Existing K8/K16 and temperature controls
remain available; old K32 saves remain readable. Alpha/Beta package bytes and
the installed 1.4.49 sell-replay-speed fix are preserved.

Short validation passed:

- ONNX parity at six batch/candidate shapes, including 772 candidates, plus exact
  original-head and zero-residual comparisons.
- 21 Rust worker tests. The engine, search and inference sources match the
  installed Beta source archive; only model admission and its Cargo feature differ.
- 5,651 Unity checks across menu/persistence, fresh-process restoration, existing
  setup behavior, IL/Alpha/Beta complete turns, IL cancellation and save/load.
- Visual inspection and a bounds assertion at 1080p with the largest text size.
- Disposable installation, interrupted-install recovery, exact rollback and save
  preservation.
- The actual installed game passed 765 checks on a 4-second/K16 Rail-era turn,
  including both actions and completed-turn save/load. Delivery was 4,184.98 ms
  within the existing 5,000 ms search-plus-delivery limit.

All 663 installed native file hashes match the candidate. The prior 441 native
files and all 161 user save/history files are unchanged. These are functional
smokes, with no playing-strength claim. The game and native workers are closed.

Evidence: [export receipt](../artifacts/nn-il/20261102/model/export-receipt.json),
[qualification](../artifacts/nn-il/20261102/release-1.4.50/qualification.json),
[installation recovery tests](../artifacts/nn-il/20261102/upgrade-test.json),
[installed smoke](../artifacts/nn-il/20261102/installed-il/verification.json) and
[final installed audit](../artifacts/nn-il/20261102/installed-verification.json).
The earlier layout attempt is retained separately and is excluded from final
qualification evidence.

With the game closed, `scripts/install-nn-il.ps1 -Action Recover` handles an
interrupted installation; `-Action Rollback` restores the exact previous 1.4.49
runtime and receipt using the recorded backup. Corresponding native/export
sources and dependency notices accompany `BrassNative/il-v11-2/`.
