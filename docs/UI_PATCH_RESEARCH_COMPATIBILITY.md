# Installed patch 1.3.0 and research compatibility

27 September 2026. **The existing R0–R3 training/research code remains compatible with its preserved runtimes. No rules, AI-policy, model, contract, profile or frozen-bank changes were needed.** One small Unity test-staging repair prevents a new installation from contaminating an older research test player.

## Verified boundaries

- The installed three-DLL package matches the recorded and qualified **1.3.0** release. Compared with the research inventory, `Assembly-CSharp.dll` and `BrassPatch.dll` changed and `BrassLegacyAI.dll` was added.
- All frozen research source/configuration bytes are unchanged. The R1 host, development R3 worker and frozen R3 bank each retain their complete, matching 112-file runtime. The existing five training profiles and held-out Pottery Flex admission remain valid.
- The six S1 v2 profile resources embedded in the installed AI DLL match the research JSON files byte for byte. The deployed implementation is a separate fork under `src/BrassLocalAI`, with its own `local-game/0.3.0` contract; research remains four-player `0.2.0`.
- Comparing the installed game assembly with the preserved research assembly found **zero changed method bodies among 1,253 existing `BoardGameRules` methods**, and no removed existing method bodies. Menu, presentation, history/cancellation, metadata cloning and AI dispatch hooks changed.
- The release includes more than visual changes: saved strategy/intent metadata, save-lifecycle safeguards and `JsonIgnore` on a brewery runtime callback. These changes are documented in [the release notes](LOCAL_AI_1.3_RELEASE.md). The independent research dependency copies are unaffected; this check does not relabel the deployment fork as the old frozen benchmark.

## Executed checks

| Check | Result |
| --- | --- |
| R0 references/contracts | 37 tests pass |
| R1 host, restoration and observations | 22 tests pass |
| R2 model/training/checkpoint/replay | 24 tests pass |
| Frozen R3 bank | 31 tests pass |
| Existing saved checkpoint/replay | Original model identity preserved; replay compatibility header accepted |
| Unity 2019.1.7f1 / Mono, fresh isolated player copied from 1.3.0 | 25 bridge assertions; 64 turns, 124 actions, 32 Rail turns, 15 save/reloads, 7 debt episodes; final projection agrees for all four seats |
| Staging regression | All 111 R1 DLLs copied exactly; foreign deployment DLL removed from the R1 fixture; overlay dependencies replaced correctly; targeting the installed Managed directory rejected |
| Preservation | Installed DLLs, research runtime fingerprints, bank/pool/report identities and original qualification evidence retained |

The [machine-readable record](UI_PATCH_RESEARCH_COMPATIBILITY.json) identifies the tested files and evidence hashes. Raw results are under ignored `artifacts/research-compatibility/patch-1.3.0/`. The Unity match uses scripted human metadata and decisions, not a new physical mouse playtest. Existing 1.3.0 UI release evidence was fingerprint-checked; its full 252-game reliability matrix was not rerun. The unchanged research bank did not need another strength tournament.

## Minimal repair

Previously, a fresh research Unity player copied the installed game's Managed directory, then replaced only its engine/test assemblies. With 1.3.0 installed, this could leave the newer `BrassPatch.dll` or deployment `BrassLegacyAI.dll` beside a 1.1.0 research engine.

`scripts/sync-research-managed.ps1` now copies the selected research runtime's complete DLL set, verifies the copied bytes and removes foreign DLLs **only inside an isolated player below project `artifacts/`**. It rejects the installed directory and directory/file links. The R1 runtime, R3 bridge and historical overlay UI check scripts call it before their test assembly/startup weaving steps. No installed DLL or user save is modified.

## Rebuild and data compatibility

Keep the private 1.1.0 research dependencies for reproducible training. The historical `build-host.ps1`, `build-legacy.ps1`, `build-legacy-overlay.ps1`, R0 `--game-root` check and R1/R3 installed-binary checks intentionally reject a 1.3.0 installation. Their fingerprint checks remain strict; do not replace their pins with the installed hashes or rebuild a frozen identity using new DLLs. This is a build/installed-version precondition, not a failure of the preserved training runtimes.

Use the preserved runtimes for normal collection/training and `check-frozen-bank.py` for current bank validation:

```powershell
python scripts/check-frozen-bank.py --root . --bank bank-804c85204da65824cb9b --output artifacts/research-compatibility/new-bank-check.json
```

For the lab, keep `/data/not_backed_up/yxu209/ai-training/` as the root and follow [the frozen-bank preparation guide](R3_TACTICAL_BANK.md) with the matching private research DLLs. The installed deployment's saves/observations are not automatically R2 replay data; importing them or deploying a trained model through `local-game/0.3.0` requires an explicit adapter and separate validation. Existing checkpoints and research replay need no conversion.
