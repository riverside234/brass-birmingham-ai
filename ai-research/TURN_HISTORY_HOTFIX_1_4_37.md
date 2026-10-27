# City cancellation and turn checkpoint hotfix — 1.4.37

Date: **2026-10-26** (America/Chicago). **Installed and verified.** The full local Rust release remains qualified. This update fixes city cancellation and saves once after every completed human or AI turn, as requested. [Installed verification](../artifacts/city-cancel-fix/installed-verification.json).

## Cause and correction

The previous history code treated the unsaved event buffer as an action boundary. Under 1.4.36's human-End-only saving, that buffer also contained completed AI turns. Opening a city prepared a Build action without adding events; cancelling it could rewind to the previous AI's last action. The input guard then correctly blocked human input because the reconstructed actor was an AI.

The installed 1.4.36 reproduction switched **Yellow to Blue**, moving revision **88 to 86**. The fix recognizes `TurnStarted` as a rollback boundary, alongside completed actions. Closing a city without making a choice clears the selection without reconstructing the game. Event-bearing Cancel and Undo retain earlier AI turns and cannot cross into the previous player's turn.

The user's subsequent save-policy update is also implemented: **one checkpoint after a human clicks End Turn, and one after each AI completes its whole turn**. The checkpoint includes turn-end and next-player/round events before the next player is dispatched. Search, validation, individual actions and liquidation continuation do not write additional saves. A failed checkpoint blocks continuation; retry does not repeat gameplay events. Loading resumes from the latest completed-turn checkpoint. Existing saves are compatible.

## Bounded verification

- **296 assertions** cover 2/3/4-player V4 and 4-player NN tables in Canal and Rail, unsaved previous AI turns, repeated city Cancel, city Back, human input after Cancel, partially completed actions, first/second-action Undo, replay guards, empty pending histories, checkpoint timing and save/load. The same cases also pass on the shipped Unity Mono runtime.
- **79 liquidation/lifecycle assertions** cover pending debt, delayed and duplicate callbacks, save failure and retry for human and AI turns, game replacement and navigation.
- **755 Unity harness assertions**, plus the embedded 296-case regression, pass. Two actual live search turns use the unchanged **4-second** allowance: NN **4.28 seconds**, V4 **4.33 seconds** observed through completion/checkpoint checks. Each writes one AI-End save; neither saves during actions/search. These are short functional smokes, not a performance study.
- **15,137 existing method bodies** are preserved. The only changed existing bodies are two history methods and the turn-end checkpoint dispatcher. The AI executor DLL, 48 V4 profiles, and all **213 native package files** are unchanged.
- Disposable installation, exact rollback and interrupted-publication recovery pass. All **107 existing save/history files** remain unchanged by installation.

Receipts: [baseline reproduction](../artifacts/city-cancel-fix/baseline/turn-history-tests.json), [qualification](../artifacts/city-cancel-fix/candidate/qualification.json), [Unity cases](../artifacts/city-cancel-fix/unity-smoke/unity.turn-history.json), [preserved runtime](../artifacts/city-cancel-fix/preserved-runtime.json), [upgrade/recovery](../artifacts/city-cancel-fix/upgrade-test.json).

The native search, ONNX Runtime CPU provider, temporary g890 model, two independent one-thread sessions, four difficulties and table-family restriction are unchanged. Lab deployment is unchanged.

## Reproduce and recover

`scripts/build-turn-history-fix.py` builds from the sealed 1.4.36 sources and overlays only this update. `tests/BrassPatch/TurnHistory/run.ps1` runs focused domain/use-case checks. `tests/RustBackendLocal/run-unity.ps1 -BuildDirectory artifacts/city-cancel-fix/candidate -TurnHistory -MaxSeconds 60 -EvidenceDirectory <fresh-directory>` runs the short Unity coverage. The default build is sealed; preserve it before choosing a fresh output for further development.

With the game closed, `scripts/install-turn-history-fix.ps1 -Action Rollback` restores exact 1.4.36 managed binaries and its receipt. `-Action Recover` repairs an interrupted hotfix installation. The prior release's own rollback chain remains available after restoring it. Neither operation modifies saved games.

The 1.4.36 reports and sealed receipts retain their original save-policy results as historical evidence. This report and the current migration plan describe 1.4.37 behavior.
