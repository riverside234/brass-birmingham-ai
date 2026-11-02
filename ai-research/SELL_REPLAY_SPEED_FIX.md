# Sell replay speed fix — 1.4.49

The installed 1.4.48 sell state machine disabled the screen-wide fast-forward
button when entering an industry, beer, merchant-bonus or next-sale selection.
The next animation did not restore it. Nested sale replay could also be followed
by the previous callback's disable-button epilogue.

`BrassPatch.Presentation.ReplaySpeed.BeforePlay` restores that button immediately
before a tagged replay sequence actually starts. This runs after the epilogue.
Only `DOTweenAnimationSequence.Play()` receives a new prefix; ordinary view and
selection animations are ignored, and hidden/disabled replay views are excluded.
Existing native speed, pause, resume and default-speed controls remain in use.

Short Unity-player qualification in `artifacts/sell-replay-speed/20261101/`:

- Baseline: 200 checks, reproducing the unavailable input in all 28 cases.
- Fixed candidate and installed game: 324 checks each, with native EventSystem
  raycasts/clicks, pause/resume, default-speed reset, callback count, domain-state
  preservation, selection handoff and disabled replay coverage.
- Two/four players and both eras; cotton, manufactured goods, pottery, brewery
  beer supply, industry income and merchant money/income animations.
- Twelve paired flip measurements: normal/fast simulated-duration ratio 1.99359
  (0.01-second measurement steps). The live tween speed is exactly 2.
- Existing presentation lifecycle regressions: 184 checks passed.
- Interrupted installation, recovery, installation and exact rollback passed in
  a disposable game tree. The live installation also passed its smoke test.
- All 441 native files, the AI assembly, and 157 save/history files stayed unchanged.

The final receipt is `artifacts/sell-replay-speed/20261101/installed-verification.json`.
The frozen candidate is `artifacts/sell-replay-speed/20261101/release-1.4.49/`.
Tests use the existing self-test startup, without initializing user persistence.

To restore the exact prior 1.4.48 managed runtime and installation receipt:

```powershell
./scripts/install-sell-replay-speed.ps1 -Action Rollback
```
