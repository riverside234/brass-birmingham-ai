# Native Scout regression scope

`UnityScoutRegression.cs` is compiled as a separate `UnityScoutRegression.dll`.
Entry point: `BrassPatch.Tests.Scout.UnityScoutRegression.Run()`, returning an assertion count.
Call it synchronously on Unity's main thread, after the initial scene is available, with normal framework/storage/network startup skipped. It is independent of the existing liquidation `UnityPresentationRegression.dll`; neither its source nor its 19-check popup suite is replaced.

The suite uses the game's real `ScoutActionView`, `IndustryCardView`, `WildIndustryCardView`, and `WildLocationCardView` components. Their standard references are supplied by a minimal in-memory RectTransform/Image hierarchy. No fake subclass overrides their input/animation behavior. It calls native `BaseCardView.OnMouseClicked` and advances DOTween's actual normal-update iterator in 1/30-second steps. It does not force `Complete(true)`, globally kill tweens, instantiate a persistence backend, open a user save, or load the normal gameplay scene.

Checks cover:

- Synchronous selected-card locking and a repeated click before any tween update.
- Normal completion, including reentrant cleanup/hide from the completing animation callback, exactly-once callback delivery, and transferred wild cards surviving view destruction.
- Cancellation early in animation, in the final pause, and after staggered selection has shifted the second card's sibling index.
- Idempotent cancellation cleanup, original hand order/positions/input flags, and restored layout.
- Native view destruction with early/final-pause animations outstanding, plus destruction during the introductory animation.
- Reopening the same cached view during its pending hide, after early and final-pause cancellation; original slot scales, fresh selection before show completion, and a successful new exchange.
- Absence of Unity/DOTween warnings and errors, and preservation of a separate unrelated tween.

The native callbacks in this suite are controlled counters. Domain action/card/event/save invariants and operation ownership are tested separately by the managed Scout suite. Cached-view tests execute the real view show/hide animations and the production presenter's preparation helper, then explicitly cancel the old hide as Router.ShowView does; they do not instantiate a full gameplay Router. The MenuScene does not supply the serialized gameplay Scout prefab. This test therefore does not certify serialized gameplay-prefab wiring, physical mouse hit testing, screen layout, controller navigation, or all possible timing interleavings. Real production components and tween updates are exercised; a full interactive gameplay session is not claimed.

**Installed Unity execution passed on 27 September 2026.** Unity `2019.1.7f1` / Mono runtime `4.0.30319.42000` completed all **128 ScoutPresentation assertions across nine scenarios**. All nine combined self-test suites passed; the separate original popup suite retained its **19 passing checks**. This confirms the native behaviors described above within the stated minimal-hierarchy and controlled-callback limits.

Execution evidence is retained in `artifacts/patch-build/unity-selftest-result.json`; `artifacts/patch-build/validation.json` records the corresponding validated candidate and in-player result. Exact installed binary fingerprints:

```text
Assembly-CSharp.dll  1C8CEF281B81474200268EED23448FAA3F5247F8EA49409A69EFC3982C5A2D0A
BrassPatch.dll       8315180FBA0A7D94D757BAC270C8944C56CFBA315F15DB2537ACEACA13F6EEFB
```
