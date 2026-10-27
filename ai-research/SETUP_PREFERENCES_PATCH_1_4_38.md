# Local menu patch 1.4.38

Installed and verified on 2026-10-27. This overlays 1.4.37 and its native search configuration update.

Clicking **Apply** now saves the setup immediately, without requiring a game to start. The next new-game menu restores the table's AI family, Human/AI controllers, each seat's V4 style and difficulty, player count and introductory setting. Random remains Random until game creation. Switching between V4 and NN retains the remembered V4 styles. Cancel, browsing unavailable models and rejected/stale callbacks do not save draft changes.

The whole-table selector now offers **V4 / NN**. **Alpha Test** and **Gamma** are nested under NN. Gamma remains unavailable; moving its tab does not enable an unfinished model. Unsupported tables omit Alpha Test. The chooser distinguishes the seat's Human/AI controller from the table's AI family.

Menu defaults use the versioned `BrassLocalAI.Setup.v2` PlayerPrefs entry. Existing V4 style/difficulty preferences remain a fallback. Invalid defaults fail safely to the legacy defaults. Saved games retain their own assignments; loading them does not replace menu preferences.

Validation used two short, separate Unity processes: **808 checks** for applying/reopening, cancellation, invalid preferences, legacy fallback and UI layout, and **774 checks** for fresh-process restoration and family changes. The isolated harness transfers the exact preference bytes between processes rather than writing the player's real preferences. Layouts were inspected at 1920×1080 and 1920×1200 with the largest text size. Installation, exact rollback and interrupted-install recovery also passed in a disposable fixture.

Only menu code in `BrassLegacyAI.dll` changed. Both other managed DLLs, all 48 embedded V4 profiles, all 213 native package files and all 121 existing save-directory files were preserved. ONNX Runtime CPU, two sessions × one thread, 4s/K=8 and 8s/10s/12s/K=16 remain installed.

- [Qualification](../artifacts/setup-preferences/candidate/qualification.json)
- [Installed verification](../artifacts/setup-preferences/installed-verification.json)
- [Runtime preservation](../artifacts/setup-preferences/preserved-runtime.json)
- [Installation/recovery checks](../artifacts/setup-preferences/upgrade-test.json)

With the game closed, `scripts/install-setup-preferences-patch.ps1 -Action Rollback` restores the preceding managed runtime and installation receipt. Use `-Action Recover` after an interrupted installation. Neither operation edits saved games or menu preferences.
