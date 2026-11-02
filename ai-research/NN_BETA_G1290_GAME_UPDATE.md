# Beta g1290 — local release 1.4.48

Installed and verified on 2026-11-01.

The NN menu now includes **Beta**, selecting the exported g1290 checkpoint with
Neutral context for the whole table. Alpha Test continues to select g890. Apply
stores the model choice together with the existing setup preferences; Cancel
discards the draft. Different seat difficulties can share a Beta table. NN models
and V4 remain separate whole-table choices.

Pinned Beta identities:

| Item | Identity |
| --- | --- |
| Checkpoint generation | 1290 |
| Model | `73fd0aa12a7a6496414b03e106ae52367d865767922aed7b40ea88a7940891ff` |
| Restored checkpoint | `ed8edb039a1257d591cc0d2e2b87eb7a2d7aa411ce0f7710a2d7f8ad487dc3b0` |
| ONNX | `1ee3340c0279592eb28cf2ce0bdbd24691e63be60f81d0196984bfc27c921424` |
| Read-only utility receipt | `f56bd03a91e661c5be0f941765ec9420c4729ef6caca0f8d294d61b3d3550670` |

The supplied `best` bundle is a qualified g1290 policy anchor from a committed
g1360 run snapshot. The game uses the g1290 weights. Restoration verifies every
checkpoint part and its combined hash. ONNX export preserves the model ID and
checkpoint bytes; the model has 5,928,616 parameters. All policy and value outputs
passed PyTorch/ONNX Runtime comparisons across six batch/candidate shapes,
including 772 candidates. Maximum observed absolute errors were below 0.000050.

Beta runs the existing Rust engine and search_v1 implementation with a pinned
`local-beta` model admission feature. It uses ONNX Runtime CPU inference with two
independent sessions and one compute thread per session. V205 remains disabled.
The actual read-only utility receipt is preserved and checked against its source
model/checkpoint/request; no utility fitting or further training ran locally.
Switching model/K/temperature retires the previous worker before starting another.
Signed Beta saves bind the distinct backend, model, export and neutral profile.
Existing Alpha package files are unchanged.

Bounded qualification passed:

- 21 Rust worker tests and five inference contract tests.
- 779 Beta UI/save checks and 763 checks in a fresh Unity process.
- 1,048 regression checks for the existing settings and NN controls.
- Actual 4-second/K16 complete turns for Beta and Alpha: 765 checks each.
- 759 checks covering cancellation after Beta's first action, preventing action two.
- Beta turn checkpoint/save-load, changed model identity rejection and changed
  utility receipt rejection. The largest-text Beta page was visually inspected.
- Disposable interrupted-install recovery, exact rollback and save preservation.

The installed game passed 765 checks for a 4-second/K16 Beta turn, including both
actions and save/load. Delivery took 4,310.8 ms within the existing 5,000 ms total
deadline. All 441 native package file hashes match the candidate; the original
218 native files and all 161 user game/history files remain unchanged. The game
and native workers are closed. See the
[installed verification](../artifacts/nn-beta/20261101/installed-verification.json).

These are functional smokes, not a strength comparison. Evidence:
[qualification](../artifacts/nn-beta/20261101/release-1.4.48/qualification.json),
[export receipt](../artifacts/nn-beta/20261101/model/export-receipt.json),
[recovery/rollback checks](../artifacts/nn-beta/20261101/upgrade-test.json), and
[installed smoke](../artifacts/nn-beta/20261101/installed-beta/verification.json).

The installer adds `BrassNative/beta-g1290/` and replaces `BrassLegacyAI.dll` plus
the installation receipt. With the game closed, `scripts/install-nn-beta.ps1
-Action Recover` handles an interrupted install. `-Action Rollback` restores the
previous managed runtime and receipt and removes only the manifest-listed Beta
files, preserving saves. Source archives, hashes and dependency notices accompany
the installed native package.
