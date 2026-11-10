# beta-2 g1190 — local release 1.4.55

Adds **beta-2** to the NN page of the per-seat AI chooser, alongside Alpha Test,
Beta and IL. The choice uses the user-selected g1190 checkpoint from
`joint7-v11-2-g1290-restart-g1290`. Existing per-seat difficulty, K, temperature,
Apply/Cancel, saved setup and mixed-table behavior are retained.

## Exact identity

- Source bundle: `test_nn/joint7-v11-2-g1290-restart-g1290-ae347ae9-g1190-f228f4dd7f1a-1af78fc4bcc8/`.
- Checkpoint SHA256: `f228f4dd7f1a913ea11e908cb2ccb53b76cd14ca302da9e2e5479c505d1294f1`.
- Model ID: `d872ea925d6643ee3a3eeec99a52d16c6a78c2de1f757617f1aaaa47d175f864`.
- ONNX SHA256: `39a536967350d02ffcaa1b9aea6a3239777bae03783fabb58d4ae38e45bb73f9`.
- Reward ID: `49711b4abaa95940928db567fa76ab654a5d60c417e1b5eca8dbb44ffb5327c1`.

The original policy and learned utility head are exported without training or
changing checkpoint bytes. Companion heads remain in the source checkpoint;
search_v1 continues using its existing utility output. The null calibration
receipt stays null. Exact model/export/checkpoint admission is recorded separately
as local user acceptance; no statistical utility qualification is fabricated.

g1190's reward is win=1, placement=0.1, VP=0.001 with cap 180, insolvency=0.0025
with maximum cost 0.025. Its insolvency scale differs from the installed older
models. Both C# input construction and Rust admission now select the pinned reward
for the chosen model. This preserves the checkpoint's own training objective;
later training revisions are not applied to old weights.

## Packaging and verification

The new package is `BrassNative/beta2-g1190/`. It uses search_v1, V205 disabled,
c_scale=1, ONNX Runtime 1.22.1 CPU, and two independent one-thread sessions.
Difficulty remains 4/8/10/12 seconds plus the existing maximum one-second delivery
allowance. Existing K8/K16 and temperature choices, and saved K32 compatibility,
remain available.

The native build starts from the installed IL package's frozen source archive.
Only Cargo model selection, model constants and model-specific reward admission
change. The game engine, search and inference implementations are retained.
The managed build overlays only NativeModel, NativePackage and NativeInput on the
installed managed source snapshot. The model registry supplies the new UI card.
Existing native packages, Assembly-CSharp and BrassPatch are preserved exactly.

Reproduction, in order, using the existing local dependency environment:

```powershell
& ./artifacts/p6e/Scripts/python.exe -B scripts/build-nn-beta2.py export
& ./artifacts/p6e/Scripts/python.exe -B scripts/package-nn-beta2.py native
& ./artifacts/p6e/Scripts/python.exe -B scripts/package-nn-beta2.py package
foreach ($case in 'ui-write','ui-read','live-beta2','cancel-beta2','load-beta2','live-old-beta') {
    & ./artifacts/p6e/Scripts/python.exe -B scripts/package-nn-beta2.py $case
    if ($LASTEXITCODE -ne 0) { throw $case }
}
& ./artifacts/p6e/Scripts/python.exe -B scripts/package-nn-beta2.py qualify
& ./artifacts/p6e/Scripts/python.exe -B scripts/package-nn-beta2.py upgrade
pwsh -NoProfile -File scripts/install-nn-beta2.ps1
& ./artifacts/p6e/Scripts/python.exe -B scripts/package-nn-beta2.py installed-beta2
& ./artifacts/p6e/Scripts/python.exe -B scripts/package-nn-beta2.py installed-ui-read
& ./artifacts/p6e/Scripts/python.exe -B scripts/package-nn-beta2.py audit
```

Completed evidence directories are immutable to these commands. Use a new study
directory for a deliberate rebuild. The installer supports `-Action Rollback`
and `-Action Recover`, restoring the exact prior receipt/files from its recorded
backup. The installer refuses an active game or an unrelated changed file.

Evidence is under `artifacts/nn-beta2/20261110/`: model export/parity,
native-build, short Unity checks, qualification, installer recovery checks and
final installed verification. **Installed and verified on 2026-11-10**:

- ONNX parity passed all six batch/action shapes, up to 772 candidates; maximum
  utility error was 3.88e-7. All 21 native worker tests passed.
- Candidate UI selection, restart persistence, cancellation, stale-load rejection,
  complete-turn save/load and existing Beta compatibility passed. Largest-text UI
  capture is retained in `release-1.4.55/evidence/beta2-1080-largest.png`.
- Installed UI and rail-era turn checks passed (1,535 assertions). The installed
  4-second/K16 turn committed both actions; turn clock was 4,238.66 ms and host
  delivery was 4,338.74 ms. Candidate canal-era delivery was 4,351.03 ms. These are
  short functional smokes, not a strength or performance benchmark.
- Installation, exact rollback and interrupted-install recovery passed on the
  disposable fixture. All existing native packages and protected user saves/game
  history remained byte-for-byte unchanged.

The installed receipt is `artifacts/patch-installation.json`; final evidence is
`artifacts/nn-beta2/20261110/installed-verification.json`. The previous installed
runtime and receipt are retained at
`artifacts/patch-recovery/nn-beta2-f3c8db4b00b04e46b91e063c02f09f19/`.
