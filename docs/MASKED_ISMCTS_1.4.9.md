# Masked ISMCTS game package — 1.4.9

**Installed:** 30 September 2026 locally (`2026-10-01T02:13:01Z`).

All six existing v3 styles now use the shared C# Masked ISMCTS implementation. The game still offers **Easy 2 seconds, Mid 5 seconds and Hard 12 seconds**. Earlier saved games retain their **10-second** allowance. Both actions, validation and replanning share one turn deadline. Saved profile identities are checked against their original embedded bytes before a detached copy is upgraded for play.

The search samples plausible hidden worlds from the acting player's permitted observation. It stores that player's information sets and uses independently masked, fixed opponent policies. Completed paths back up four stable-seat returns. Pending or interrupted work does not become a visit. Dynamic legal choices, progressive top-K expansion, industry/setup/sale protection and positive tail exploration share the Python training contract. The legacy scorer provides priors, evaluations and the six styles' preferences; a complete legal incumbent remains available if the clock expires.

The previous 240-node target is retired. The one-million-operation completion guard, 100,000-simulation ceiling and bounded tree protect resources; they are not promises that every turn reaches those counts. Cash protection, timely industry conversion and merchant beer evaluation remain part of the v3 scorer. A short turn can return its incumbent with few completed simulations. Search is qualified for correctness and lifecycle behavior here, **not for improved playing strength**.

## Resource management and existing patches

The [1.4.7 execution lifecycle](AI_EXECUTION_RESOURCES_1.4.7.md) remains in use:

- Computation and waiting for a retiring backend use the 30 FPS cap with temporary VSync zero. Presentation, menu/history/game replacement, errors and destruction restore the previous settings.
- One turn job executes at a time. Cancellation requests do not count as completion. Cleanup and admission of the next job wait for actual backend/device retirement.
- Optional registered work restores its previous state. Detached turn buffers can be released; shared model weights remain owned by the backend.
- Typed request, revision, generation and stale-result guards remain active. The resource policy is independent of strategy and thinking time.

The Python CPU/GPU inference services follow the same completion rule: private pending tickets and simulator lanes are retired before reuse. Those services train the future NN opponent; random training weights are not installed into the live game by this package.

`Assembly-CSharp.dll` and `BrassPatch.dll` are byte-identical to the immediately preceding **1.4.8** package. Its city highlighting, link replay, saves, merchant/rules repairs and completed-match statistics UI are preserved. Only the AI helper DLL changes. Fresh native evidence covers the new helper; earlier UI/rules evidence is retained only for the two unchanged binaries.

## Package fingerprints

| Installed file | SHA-256 |
| --- | --- |
| Assembly-CSharp.dll | `C39FE4084D4AB4DECDC025A883A15A25E731366A954B80D753EAF685B9A5833C` |
| BrassPatch.dll | `40905DA52DAC68F2078C947267ECB738DBDDE4B4B8866E145F644F765F473C57` |
| BrassLegacyAI.dll | `7DDBFAE1B4C600496C5586BBA66631D2634F4EF406B02571E0CA4B8E47CB9C3D` |

The private `artifacts/patch-releases/1.4.9/` archive contains the binaries, compilation source pins, qualification, installation record, source, profiles, notices and an inventory. `1.4.8` is retained intact. The coherent pre-upgrade binaries also remain at `artifacts/patch-recovery/5ae028b0473a4933b360363b09118e4f/`. Game binaries and release archives stay outside Git.

## Install or return to the preceding package

The current game already has 1.4.9 installed. For later installation, close the game, verify/stage the archive, then use the existing transactional installer:

```powershell
& ./scripts/stage-archived-patch.ps1 -Version 1.4.9 -VerifyOnly
& ./scripts/stage-archived-patch.ps1 -Version 1.4.9
& ./scripts/install-patch.ps1
& ./scripts/test-installed-local-ai.ps1 -Smoke
```

To return to the preserved 1.4.8 combined patch, use the same first three commands with `-Version 1.4.8`. Staging changes only the build directory; installation records and live DLLs change only through the installer. Its durable intent and coherent previous-package backup support interrupted replacement. Do not mix individual DLLs from different packages. Original stock restoration remains available through `scripts/uninstall-patch.ps1`, subject to its existing save-compatibility checks.

## Research and training

The separately frozen lab bank is **`bank-927c8ddec148897fa542`**. It runs v3 Masked ISMCTS at **exactly 1,000 ms per whole turn**, without clock calibration or a qualification tournament. Five identities are training opponents; Pottery Flex's identity is held out for evaluation. Pottery actions remain legal and supported for every learner.

Fresh Step 5 uses a **1,840,109-parameter MLP-ResNet**, audited heuristic imitation and predominantly self-play, with completed search visits supplying a separate distillation loss. The lab NN teacher uses a separate three-second allowance; this does not change the v3 one-second contract. Follow the [conda guide](TRAINING_RECOVERY_LAB.md) and [compact exit evidence](MASKED_ISMCTS_EXIT_CHECKS.md). No old neural checkpoint or neural replay is imported.
