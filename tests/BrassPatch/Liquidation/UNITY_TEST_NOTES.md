# Unity popup validation

The first in-player run instantiated the shipped question popup and reached the transition to the second prompt. It then failed in DOTween's `RemoveActiveTween` because the harness used `Sequence.Complete(true)` outside DOTween's update loop.

The existing Router records animations under the requested view interface but removes completion entries using the concrete view class. When a hide callback immediately opens another popup, the Router can therefore kill the completing predecessor. The installed DOTween defers kills during its ordinary update loop. The harness's forced completion bypassed that deferral, causing an immediate despawn followed by a second despawn with `activeId == -1`.

The revised harness sets its native popup sequences to manual update and advances them using `DOTween.ManualUpdate` in 1/30-second steps. This executes the same tween update and deferred-kill logic used by normal game frames, while keeping the test synchronous in the self-test startup hook. It does not alter production code or suppress exceptions. The prefab, Router, text labels, native button events, callback ordering, and tween implementation remain real.

No ordinary gameplay caller of forced `Complete(true)` was found in the reviewed application. The synthetic failure alone therefore does not establish a new high-priority player-facing defect.

The normal-update retest passed in the installed Unity 2019.1.7f1 player, Mono runtime `4.0.30319.42000`, on 27 September 2026. All 19 popup assertions passed, including native labels, next-industry selection, keyboard confirmation, stale-button rejection, double-click handling, Router cleanup, and callback ordering. The complete six-suite self-test also passed. Evidence: `artifacts/patch-build/unity-selftest-result.json` and `unity-selftest.log`.
