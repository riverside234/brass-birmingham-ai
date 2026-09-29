# Lifecycle, cards, debt, and persistence cross-check

Review date: 2026-09-27. Review only; no production changes or save migration. The official rulebook is authoritative. The comparison implementation is pinned to `npow/brass-birmingham@fdeb63c6aa92cc4ba44a721482316e0b8db10e06`.

Installed assemblies used by both probes:

| Assembly | SHA-256 |
|---|---|
| Assembly-CSharp.dll | `34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A` |
| BrassPatch.dll | `4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D` |

Both runners reject any other game/helper pair before loading it. New fixture saves live only under `artifacts/reference-review/phases-scout/`; no existing player saves were used.

## R01 — Scout animation completion outlives cancellation

The actual `ScoutActionUseCase.SelectCard` passes `OnCompleteAction` to the second trade-card animation. Cancellation resets the counter and rolls game history back but does not invalidate that callback. `OnCompleteAction` later uses retained Scout/player references and calls `BaseActionUseCase.ConfirmAction` against the **current** history. `TurnFlow.ConfirmCurrentAction` does not require a current action before decrementing the action count. A rejected decrement at zero happens after `AggregateRoot.ApplyEvent` increments its revision, leaving a missing event revision.

The deterministic production-use-case probe uses normal local-game setup and commands, actual history rollback, and the patched filesystem repository. Presentation is replaced by a callback queue. Results in `phases-results.json`:

| Case | Measured result |
|---|---|
| Normal Scout, distinct trade cards | Revision 7, zero actions, seven hand cards including two wild cards; saves/reloads at revision 9 after EndTurn. |
| Cancel, then deliver old Scout completion | Rollback restores revision 4, one action, eight ordinary cards. Old callback then reaches revision 5 and zero actions without awarding wild cards; EndTurn saves/reloads at revision 7. |
| Cancel, legally Pass and confirm, then old completion | Pass reaches revision 7 / zero actions. Old completion throws `InvalidOperationException` and increments revision to 8 without committing event 8. EndTurn reaches revision 10; actual patched save rejects it with `Incoming event revisions are not contiguous.` The last good revision-4 save remains readable. |
| Cancel, start another Scout, then old completion | Old callback hides the new exchange presentation and throws on its newly reset card-selection list. This is a secondary reentrancy signal, not a separately counted P1. |

The official rules require one action per first Canal turn, two subsequently, and three discarded hand cards for Scout (action overview/turn rules, p. 6; detailed Scout rule, p. 11). Cancellation itself is a digital convenience; the rules consequence is consuming a turn without its required action/card exchange. The save refusal is application integrity damage, not deletion of a previously good save.

### Native timing and cancellation availability

The unchanged `ScoutActionView.AnimateCardToSlot` creates a local DOTween sequence and keeps no cancellable handle. Its final callback occurs after the card movements and an explicit one-second pause. There is no `OnDestroy` cancellation in this class or `BaseView`.

`ScoutActionPresenter.SetupCardInSlot` disables only `IHandOfCardsView` interactions after accepting the second trade card. It leaves the `ActionInProgressUseCase` cancel delegate installed. The exchange presentation places ActionInProgress above Scout, then puts HandOfCards on top. `ActionInProgressView.OnCancelClicked` still calls its controller; its Escape handler also has no local guard. The hand's topmost keyboard registration may consume Escape before the progress view, so source inspection does **not** establish that Escape reaches Cancel. Cancel-button activation is preserved in source, but serialized layout/raycast hit testing was not exercised.

Native follow-up executed through the installed isolated self-test hook on Unity 2019.1.7f1 / Mono 4.0.30319.42000. Gameplay prefabs are absent from MenuScene and Resources; the test therefore constructs only Transform/card targets and invokes the unchanged production Scout view method. DOTween is advanced through its real update path in 1/30-second steps. No game scene, normal initialization, user storage, or networking is initialized.

| Native case | Actual result |
|---|---|
| Normal second-card animation | One final callback at 2.933332 seconds. |
| Destroy Scout and card targets at 0.50 seconds, representing cancellation at 0.20 plus its 0.30-second hide | One final callback at 2.933332 seconds. DOTween catches missing-target warnings and preserves the sequence. |
| Destroy targets at 2.50 seconds, representing cancellation during the final pause at 2.20 | One final callback at 2.933332 seconds; no native warnings. |

These results establish that target destruction does not prevent late completion in the shipped Unity/DOTween runtime. They are **not** a full native click-through reproduction: the hierarchy is synthetic, cancellation is represented by destruction at the traced boundary, and the full Scout presenter/hand animations are not connected to the rule-command probe. Completing a replacement Pass before 2.933 seconds remains a timing-dependent escalation. The action-loss case has a larger late-pause window. The domain/use-case defect and save escalation are deterministic; ordinary serialized-UI reachability remains source-supported rather than end-to-end verified.

Fix plan: give each Scout attempt an operation identity and owning-game/action identity; invalidate it on cancel, restart, completion, and destruction; ignore stale callbacks before hiding views or mutating rules. Retain/kill the owned animation on cancel. Confirming an action must prevalidate ownership/current-action/remaining-count before emitting events. Add cancellation tests throughout the animation timeline, including a replacement Pass and save/reload. Keep the existing save continuity check: weakening it would conceal the revision gap.

Reference comparison: the web implementation performs its Scout once all three selections exist in [uiManager.js, selection flow](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/uiManager.js#L1000-L1025), avoiding this specific delayed Unity callback design. This is architectural comparison, not proof that the reference implements all cancellation rules correctly.

## R02 — Duplicate trade-card selection is accepted

`ScoutActionUseCase.SelectCard` does not reject a previously selected card. `ScoutAction.Scout` checks only the number of trade-card entries and wild-card status; it does not require distinct cards owned by the current player. The event handler records the same card twice in the discard pile and removes it from the hand only once.

Production probe: selecting the same trade card twice yields eight cards including two wild cards, with four discard entries but only three distinct IDs (one entry is the seeded initial discard). Normal distinct selection yields seven hand cards and four distinct discard IDs. Both cases save/reload successfully, so persistence does not reject the incorrect exchange.

The native hierarchy probe invokes inherited `BaseCardView.OnMouseClicked` twice before advancing the tween. Both calls are accepted, and Scout's final callback executes. `firstLockedImmediately` is `false`: the first card is marked ignored only in the sequence's first update, whereas the second-selection presenter lock occurs immediately after accepting it. This proves same-update reentrant input exposure, not a physical double-click frequency in the normal UI. Keep priority conditional/lower than a broadly reproducible P1 until actual UI delivery is established.

Fix plan: reject duplicate identities and cards absent from the current hand in the Scout domain command **before** emitting `WildCardsTaken` (the action-card discard already precedes this exchange). Validate canonical cards owned by the captured game, since IDs can repeat between games; preserve the no-wild restriction at action start. Validate the use-case selection too, and lock the selected card synchronously before scheduling animation. Test repeated selection through both domain and native card entry points, failed-command state immutability, distinct-card success, wild-card conservation, and reload.

The reference [UI explicitly rejects an already queued card index](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/uiManager.js#L1001-L1008), which motivated this check. Its [executeScout domain entry point](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L934-L964) does not itself enforce distinctness, ownership, eligibility, or wild-pile availability; do not copy its validation strategy as a domain fix.

## Other compared areas and exclusions

- Final Rail income suppression, first-Canal-round action count, later-round action count, expenditure turn ordering, and Canal-to-Rail reset align with the already patched lifecycle rules and the reference's corresponding `endRound` / `endCanalEra` code. No additional P1 established here.
- The native card-dealing helper's all-or-nothing partial refill differs from the reference's [draw-until-empty loop](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameState.js#L183-L188). Existing legal Pass/Scout coverage and deck parity do not expose an ordinary reachable shortfall; do not elevate an artificial helper input to a new P1.
- Reference [`resolveShortfall`](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameState.js#L693-L730) automatically chooses liquidation tiles. That is the reference's rules limitation, not a reason to undo the installed B09 human choice. No new debt reload issue was established in the patched event-based choice flow.
- Introductory scoring ambiguity remains excluded pending publisher clarification. Existing B01–B16 and the already repaired interrupted filesystem write issue are not recounted as new findings.

## Evidence and restoration audit

Source/probes: `phases-probe.cs`, `phases-probe.ps1`, `phases-results.json`; `phases-native-probe.cs`, `phases-native-run.ps1`. The failed prefab attempt is preserved separately as `phases-native-prefab-probe.cs`; its resource inventory is `artifacts/reference-review/phases-native-resource-catalog.json`.

Native outputs: `artifacts/reference-review/phases-native-results.json`, `phases-native-unity.log`, `phases-native-selftest-result.json`, `phases-native-run.json`. The self-test infrastructure calls this replaceable test slot `ActualPopup` / `popupChecks`; in this review run its four checks are the four Scout timing cases above, **not** four real popup-prefab checks. The rest of the unchanged self-test suites also completed. The original validation evidence was restored, not replaced with this review run.

Verified after the native run:

| Artifact | Result / original SHA-256 |
|---|---|
| Installed game/helper DLLs | Unchanged; hashes at top of this note. |
| `artifacts/patch-build/validation.json` | Read only, unchanged: `8B06A6E479BDA41510004B38FE826E32A8B68AF8694235EA02C7F414D6060BEE` |
| Original `UnityPresentationRegression.dll` | Restored: `1595DFF50DBA44DDE5BA7575ADA3888420275B96D278685FF319E1B52E2E34DD` |
| Original `unity-selftest-result.json` | Restored: `74297FC6F13BB32AEF2F3937AFB401401D79D7A7A62EDF3C2ED19DC18334DC5F` |
| Original `unity-selftest.log` | Restored: `CB3382B61BC923642A468D8024F691B0B38DCC6E0475D65597ED40BB0E8372E5` |

No installed script, production source, installed assembly, normal game startup, or user save was changed by this review.
