# R1 exit checks

27 September 2026. Scope: standard four-player rules host, legal decisions, restoration, permitted observations and initial H0 capture. Runtime contract `0.2.0`; existing installed rules patch `1.1.0`. See [implementation and usage](R1_HOST.md).

**Outcome: passed for the declared Windows runtime.** The implementation and checks below have run locally. The [versioned machine-readable summary](R1_EXIT_CHECKS.json) records the final results and hashes. Its generated original is `artifacts/r1-host/exit-checks.json`, produced by `scripts/verify-r1.py`; that verifier rejects stale host hashes, changed source, mismatched traces, failed reports and changed installed patch files. Generated game-state records remain ignored by Git.

| Exit gate | Evidence |
| --- | --- |
| Declared rules and dependencies | Original R0 profile preserved; all 111 managed DLLs pinned and checked before build/launch; installed patch pair unchanged |
| Legal choices and action lifecycle | 20 targeted host tests plus two schema/boundary tests; seven actions and all 14 decision kinds exercised |
| Complete small decision tree | Exhaustively enumerated five complete outcomes in a constrained Pass/develop fixture, including both resource orders and stopping after either first payment |
| Four-player progression | Five complete games, 1,551 decisions, both eras; loans, passes, Scout, build, develop, networks, sales, bonuses and debt |
| Restoration | 46 checkpoints covering ordinary actions, pending payments, continuations, bonuses, debt, Rail transition and game end |
| Runtime agreement | Identical canonical checkpoint hashes in standalone .NET Framework and Unity **2019.1.7f1** Mono; 5,393 replayed commands including overlapping checkpoint prefixes |
| Unexpected failure recovery | Exception injected after the loan handler has mutated state; full canonical pre-command state restored, then the same legal action succeeds |
| Information boundary | 328 counterfactual cases across all seats and encountered era/phase combinations; permitted observations match, and acting-seat choices match |
| Hypothetical worlds | All 328 sampled reconstructions execute forward to the next ordinary boundary; source state remains unchanged |
| Strict schema and stale work | Unknown/private fields rejected; invalid/incomplete/multiple actions leave state unchanged; duplicate/stale revisions and pre-reset/pre-restore transport identities rejected |
| Existing patch regressions | .NET: Core 1,081, State 315, Resources 99, Liquidation 56, Scout 125 assertions. Unity: Core 1,086 and the same other counts. State suite includes 15 replay cutoffs and five games |
| R0 compatibility | All 37 R0 tests still pass; both immutable references, 16 cached source hashes, ten mappings and 12 examples verify |
| H0 started | Five original-stock and five guarded-stock Hard games, seeds 4100–4104; all completed 64 turns/124 actions. One plain control game per profile verifies selector logging does not change event/action traces |
| Repository boundaries | Game DLLs, generated executable, private checkpoints, H0 traces and isolated Unity copy remain ignored; no installation or publishing performed |

The five primary trace games use seeds 1200, 1201, 1202, 1300 and 1301 with an independent deterministic coverage-policy seed. This deliberately weak policy explores legal choices; these are correctness runs, not playing-strength results. The exhaustive fixture is small and deliberately constrained; the full Brass state space has not been exhaustively enumerated.

Targeted tests cover owned and rival overbuilding, single-slot preference, multi-coal affordability/payment, equal-distance coal choices, board-iron priority, coal/iron ordering, one/two developments, forbidden development, first/second rail and stopping, remote owned beer, single-merchant beer, multiple sales, merchant bonus/empty bonus, Scout multisets/wilds, delayed replenishment, loan threshold, actor-correct liquidation and final outcomes. Existing regression suites add the audited game fixes and their persistence/state checks. A failed worker or illegal stock action would be recorded as a reliability failure, never a strategic win.

Unity validation uses a disposable game copy under `artifacts/r1-unity`. Its startup hook enters the test runner before the normal application starts; it does not open user saves. The weaver checks every method body and requires that only `InitUnity.Awake` and `InitUnity.OnDestroy` change. The rules methods remain those of the pinned installed patch. The installed game directory's patch files are verified again after testing.

Reproduce the exit checks from this workspace (or supply `-GameRoot` / `--game-root` where appropriate):

```powershell
python -m pip install -r requirements-r1.txt
pwsh -NoProfile -File scripts/build-host.ps1
python scripts/check-r1-traces.py
python scripts/test-r1.py
pwsh -NoProfile -File scripts/check-r1-runtime.ps1
pwsh -NoProfile -File scripts/check-r1-runtime.ps1 -Unity
# The Unity command launches the isolated check. Wait for its process to exit
# and artifacts/r1-unity/unity-result.json to report passed before aggregating.
pwsh -NoProfile -File scripts/capture-h0.ps1 -Games 5
python scripts/verify-r1.py
python -m unittest discover -s tests/R0 -p 'test_*.py' -v
python scripts/verify-r0.py --check-cache --game-root .
```

The local run used the bundled Python interpreter and already-installed pinned validator packages via `PYTHONPATH=artifacts/r0-validation-deps;src`; the portable commands above use an ordinary Python environment with the same requirements. No Torch or GPU was involved.

Limits and handoff:

- Windows .NET Framework and actual Unity Mono are validated. Linux and the lab server still need platform qualification; the clone does not include the proprietary runtime or a training loop.
- Checkpoint/replay favors correctness and has not been optimized for large-scale training or search throughput.
- H0 is **begun**, not complete. Selected-rule/action/setup/timing/outcome capture and a noninterference control exist. Independent candidate-alternative diagnosis, mixed tables, seat rotation and fairness conversion of stock policies remain.
- No improved legacy bank, neural model, S1/S2/S3 planner, asynchronous live-game bridge, difficulty calibration or personality evaluation is claimed. Those remain R2/R3 and later milestones.
- Future models, search callbacks, inference caches and training targets must pass their own counterfactual checks. R1 validates the environment inputs and sampled-world construction they will consume.
