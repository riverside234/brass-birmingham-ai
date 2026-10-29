# NN move temperature — local release 1.4.41

The AI selection panel adds **Temperature: 1.0 / 1.5** beneath the NN search K controls. The choice applies to the whole NN table at every 4/8/10/12-second difficulty, independently of K8/K16 and the individual seats' time budgets. It uses the existing light theme, gold selected state, check marks and keyboard navigation.

Temperature means `policy logits / T` before search consumes each NN evaluation. The local Rust worker wraps each of its two owner-thread evaluators, so root proposals, descendant policies, staged choices and focused second-action evaluations use the selected temperature. Legal candidate order, model identity and all four value heads are unchanged. T1.0 is a fast pass-through that preserves the complete original evaluation. T1.5 recomputes stable log probabilities and probabilities after scaling; it does not add a probability floor or alter value utility. This is a policy temperature, not a separate final visit-count sampling rule.

Apply saves the menu default and commits the complete draft; Cancel discards the draft. New games bind `nn_temperature` in the assignment integrity hash. Missing temperature in an older menu preference or save defaults to **1.0**, preserving the existing K and difficulty. Loading a game uses its saved temperature independently of current menu settings. Changing temperature retires the previous worker before another is admitted, preventing reuse of policy statistics across temperatures.

The package includes four verified K/temperature policies. The original T1.0 policy files and g890 model/export/ONNX Runtime files are preserved. The local Rust source overlay changes only `local.rs`, its module declaration and `local_temperature.rs`; the shared game/search core is byte-identical to the previously installed source archive. The corresponding updated Rust source is shipped in `BrassNative/alpha-g890/native-source.zip`. Cache-only `search_v2_focus_only`, V205 disabled, and two independent CPU sessions with one compute thread each remain selected. The B01 lab bundle keeps its original source pins.

Short qualification passed:

- Four Rust tests: exact T1.0 identity; known T1.5 probabilities with unchanged identity/value heads; singleton/tie/extreme stability; invalid temperature and nonfinite-logit rejection.
- 939 UI/preference checks and 806 fresh-process restart checks, covering both temperatures at every K/difficulty combination, Apply/Cancel, invalid settings, legacy preferences/saves and assignment integrity. Layout was inspected at 1920×1080 and 1920×1200 using the largest text size.
- 753 checks per 4-second live smoke at T1.5/K8 and T1.5/K16, including switching from T1.0, worker retirement, both committed actions, completed-turn save/load and native readiness/response temperature binding. Candidate deliveries were 2,900.57 ms and 2,729.69 ms respectively. These are functional smokes, not a comparative performance or playing-strength study.
- Interrupted installation recovery and exact rollback on an isolated copy with an unchanged saved-game sentinel.

Evidence: [qualification](../artifacts/nn-temperature/candidate/qualification.json), [native build/tests](../artifacts/nn-temperature/native-build.json), [upgrade test](../artifacts/nn-temperature/upgrade-test.json), [installed K8 smoke](../artifacts/nn-temperature/installed-k8/verification.json), [installed K16 smoke](../artifacts/nn-temperature/installed-k16/verification.json), and [panel preview](../artifacts/nn-temperature/candidate/evidence/images/setup-nn-1080-largest.png).

The actual installed player passed **753 checks per K setting at T1.5**. Both actions completed in **2,987.67 ms at K8** and **3,100.64 ms at K16**. All 216 native package files and three managed assemblies were verified after the smokes; model/export/ORT files match the prior installation. See [final installed verification](../artifacts/nn-temperature/installed-verification.json) and [deployment receipt](../artifacts/nn-temperature/deployment.json). The game and native workers are closed.

With the game closed, exact rollback is available through:

```powershell
& ./scripts/install-nn-temperature.ps1 -Action Rollback
```

Use `-Action Recover` after an interrupted installation. Rollback restores the prior native binary, source archive, managed runtime, policy manifest and receipt. Games created with the temperature field require the temperature-capable release to resume; their saves are preserved by rollback.
