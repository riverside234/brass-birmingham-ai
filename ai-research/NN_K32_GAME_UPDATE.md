# NN search K32 — local release 1.4.43

AI selection → NN → Alpha Test now offers **K8 / K16 / K32** in the existing cream/gold control style. The choice applies to the whole NN table at every **4/8/10/12-second** difficulty and both **1.0/1.5** temperatures. Seats can retain different time budgets. K16 remains the default; existing saved choices are preserved.

Apply saves the setting; Cancel discards the draft. K32 is included in the existing assignment integrity hash and restored from both menu preferences and completed-turn saves. Older saves without a K field still use K16 without being rewritten. K32 saves require this release to resume.

The installed backend remains **search_v1**, with V205 disabled, two independent CPU sessions and one compute thread each. The existing g890 test model, ONNX export, ONNX Runtime provider, and shared Rust rules/search core are preserved. The local worker adds an admission-time **1.2-second delivery reserve for K32** within the original turn deadline; K8/K16 keep their previous reserve. The maximum 12-second budget already reserved 1.2 seconds. No retry can extend the original deadline.

This reserve addresses the retained pre-fix smoke: one run requested action two after its clock expired, and another returned its native plan at 3.63 seconds but finished both Unity commits at 4.40 seconds. The measured extra interval included snapshot work, validation, frame scheduling and commit; it was not all validation CPU. Wider roots can leave less support for a full two-action plan, but that was not established as the cause of the first retry. Earlier work cutoff gives the existing legal completion and guarded commit path time to finish.

Short qualification passed 979 UI/preference checks and 812 fresh-process restart checks, all K/temperature/difficulty combinations, invalid/tampered settings, legacy saves, Apply/Cancel, and large-text layout at 1920×1080 and 1920×1200. Four Rust reserve tests cover unchanged deadlines, monotonic narrowing, preserved long-budget reserve and invalid reserves. Interrupted installation recovery and exact rollback passed with an unchanged save sentinel.

The actual installed game completed both actions and completed-turn save/load at K32 in **3631.47 ms at T1.0** and **3671.58 ms at T1.5**, with 754 checks per smoke. These are bounded functional checks, not a playing-strength or latency-tail guarantee. Installation was verified against all 218 native files and three managed assemblies.

Evidence: [installed verification](../artifacts/nn-k32/installed-verification.json), [qualification](../artifacts/nn-k32/candidate-v2/qualification.json), [reserve tests/build](../artifacts/nn-k32/native-build.json), [recovery test](../artifacts/nn-k32/upgrade-test.json), and [panel preview](../artifacts/nn-k32/candidate-v2/evidence/images/setup-nn-1080-largest.png). Initial failed/late smokes remain under `artifacts/nn-k32/live-t10*`.

With the game closed, exact rollback is `& ./scripts/install-nn-k32.ps1 -Action Rollback`; use `-Action Recover` after an interrupted installation. User saves are preserved. The separately frozen lab/training bundles are not changed.
