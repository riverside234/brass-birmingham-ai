# P5/P6 implementation audit — 12 October 2026

Base: `05def96` (`Add personality difficulty and laptop runtime`). The checkout was clean at the start. This follow-up reviewed the implemented P5 calibration/evidence pipeline and P6 export, persistent service, native game bridge, saves and deployment checks. It does not repeat the earlier whole-training audit or qualify a model for release.

Eight confirmed defects were repaired. Separate reviewers owned P5, P6 runtime and native code; the integrating reviewer checked their execution paths and evidence. All experiments used authored fixtures or existing pinned native dependencies. No training campaign, real qualification study or live installation ran.

## Confirmed findings and repairs

| Severity | Defect and execution path | Repair and regression evidence |
| --- | --- | --- |
| High | **A new neural game could fail at its first save.** `PersonalityState.Attach` sealed an envelope without outer `game_id`. The existing `SaveExtensions.Merge` then added it. `LocalMenu.StartGame` immediately saved and called `LocalGameState.Read`, which rejected the changed seal. | Include the stable metadata game ID before sealing in `src/BrassPersonalityGame/State.cs`. The actual Attach → Merge → Read sequence reproduced `Neural save integrity differs` before repair. Native regressions cover roundtrip, unchanged legacy/envelope data, pending-plan clone/load, rewind and foreign-game rejection. |
| High | **Failed/crashed P5 jobs could be retried under another output directory.** `collection.execute` reserved only its chunk; the verifier saw only supplied chunks. Omitting the original attempt allowed replacement evidence for the same planned job. | A sealed panel-owned `reservations.json`, updated under the existing OS-backed run lock, assigns every job to its immutable chunk before collection. Loading checks exact ownership and reserved job coverage. Tests cover crash retry, overlap, concurrent claims, disjoint continuation and copied-chunk rejection. |
| High | **P5 reports accepted contradictory derived summaries.** `evaluation.load_report` recomputed metrics/gates but accepted resealed changes to failures, stock failures, timing and environment. Release planning trusted the editable failure list. | Recompute all those summaries and source receipts from pinned records. Release planning/loading uses the verified `no_failed_games` gate. Tests first accepted the damaged reports, then rejected them while retaining a valid roundtrip. |
| High | **A P6 package could claim the qualified rollout pool while using different ordered reference model IDs.** Runtime uses `models[1:]`; package validation previously checked only the actor against the roster. | Bind the digest of the ordered reference IDs to both manifest and roster pool IDs. Matching exports remain valid; reordered, substituted and mismatched pool fixtures fail. Existing graph metadata/hash checks remain intact. |
| High | **Certification could credit controls that never occurred, and overlook Play Now clock violations.** A benchmark timer cancelled after normal completion still left `interrupt='play-now'`. The certificate checked its label/status but not observed metrics; clock gates covered ordinary rows only. | Require observed cancellation/Play Now booleans and the corresponding status. Apply the no-reset/50 ms overrun gate to successful Play Now turns too. Positive synthetic evidence passes; unobserved controls, reset deadlines and excessive overruns fail without producing a certificate. |
| Medium | **P5 inference preparation failures bypassed cleanup.** Registration/channel allocation preceded the service context; an exception could leave owned queues, pipes and shared buffers open. | Protect preparation and close the service on failure in `difficulty/collection.py`. Injected channel-allocation failure reproduced missing cleanup before repair. |
| Medium | **Failed P6 forest creation or partial channel allocation leaked persistent inference resources.** Closing acquired worker endpoints alone left service-owned lanes alive, and the server retained the failed runtime for retry. | Retire the entire runtime on either failure, close native/inference resources with idempotent cleanup, and create a fresh runtime for the next request. Tests use real portable channel allocation with injected native/allocation failures; native workers/model execution are mocked in these specific failure tests. |
| Medium | **Non-object JSON killed the persistent service.** Handling `[]`, `null`, a number or a string called `.get` both in dispatch and its error handler. | Validate the decoded envelope type before assigning it to the error-response variable. The regression sends all four malformed values followed by a successful `hello`; diagnostics remain redacted. |

P5 admission/verification is explicitly versioned as `personality-difficulty/1.2.0`; reservations use `personality-difficulty-reservations/1.0.0`. P6 package/save schemas remain unchanged; exact source identities change and force fresh packages/staged helpers. No reward, objective, architecture, training schedule, worker count or search budget changed. P5/P6 still use two workers and 2/5/12-second whole-turn allowances.

## Verification

Evidence is under [`artifacts/p56-audit-20261012`](../artifacts/p56-audit-20261012/), except the short Windows-path pipeline sandbox `artifacts/p6a`.

| Check | Result / evidence |
| --- | --- |
| Initial P5/P6 Python baseline | 41 tests passed; `baseline.log`. |
| Focused P5 repair checks | 27 passed; `p5-after.log`. Before-fix evidence: `p5-before.log`, `p5-preparation-before.log`. |
| Focused runtime and save checks | 8 passed; `p6-runtime-final.log`. Before-fix evidence: `p6-runtime-before.log`, `p6-channel-before.log`. |
| Package/certificate and existing installation regressions | 6 passed; `package-after.log`. Before-fix evidence: `package-before.log`, `certification-clock-before.log`. |
| Complete personality Python suite | **121 passed in 97.494 seconds**, including P0 checkpoint/optimizer/counter restoration and P1–P4 contracts; `personality-suite.log`. |
| Fresh isolated native helper | Compiled and passed current source identity/file checks; `native-build-final.log`, `native-verification-final.json`. |
| Actual native save hooks | Six checks passed on the final helper; `native-save-final/report.json`. Before-fix product failure: `native-save-repro.log`. Initialization RPC is stubbed in this test. |
| Existing native bridge checks | 37 assertions passed without optional Python save input; seven observation/choice cases in `native-final/report.json`. Repeating with the actual Python pipeline's saved state passed **44 assertions**; `native-save-python/report.json`. |
| Complete native public-history parity games | Three games, 993 legal decisions, 369 checked completed-action boundaries, including both eras and sale/bonus/debt paths; `native-history-repro/report.json`. |
| Original Hard P5 native bridge | Three complete games, 377 stock actions including five liquidations. Strict observations/pre-settlement views and official results passed; `stock-history-report.json`. No search/neural strength claim. |
| Fresh P6 export and three-tier CPU smoke | ONNX export/parity and legal complete plans passed; `laptop-smoke/benchmark.json`. Fixture remained unqualified. |
| Fresh P5 collection | Two full authored-fixture games completed, including a table against three Original Hard seats, plus a real two-worker searched turn. No game failures or remaining CPU leases; `difficulty-smoke/smoke-report.json`. Pinned report reload passed. |
| Standalone bundled-runtime pipeline | Actual RPC, observed Play Now/cancellation, exact save restoration, late-control isolation, policy-only mandatory liquidation and seven native reconstructions passed. Interpreter/backend preflight, dry run, installation and byte-exact rollback passed **only in the marked isolated sandbox**; `../p6a/report.json`. |
| Preservation | All **1,743** snapshotted upstream source/config/bank/installed-DLL files match. Step 5, learning and P0–P4 identities match; `preservation-before.json`, `preservation-after.json`. |
| Final evidence validation | P5 report recomputation, P6 package validation, staged helper hashes/current identities and actual control flags passed; `final-validation.json`. |

The one-position CPU smoke measured admitted planning times of 1.843 / 4.547 / 10.922 seconds for Easy / Medium / Hard, with 0 / 0 / 4 completed simulations and 0 / 0 / 1 quality samples. There were no deadline resets, measured overruns or remaining CPU leases. Peak process-tree RSS was about 674 MiB. Other audit checks ran concurrently, so these are execution observations, not a speed comparison. No performance improvement is claimed.

P5 collection took 35.719 seconds; its two-second searched turn returned a legal fallback with no completed simulations and retired both leases. In the standalone P6 smoke, cold Play Now took 8.000 seconds including initialization (0.437 seconds inside planning), cancellation took 0.422 seconds, and warm mandatory liquidation took 0.172 seconds. Both controls were actually observed; none reset the admitted deadline or reported an overrun. These fixtures remained unqualified.

The final package is `6c53d7108c18f2aeab8816df2ed4d29e75cbfed3a2b57733af84384316b50166`; the final staged build is `10ab2fb003b13478dad5f10b691c35bdc07fdace4ef10cff7166fef852965c7e`. Final P5/P6 source identities are pinned in `final-validation.json`. Test-driver setup errors retained in `native-save-before.log` / `native-history-before.log` were compiler mistakes, not additional product findings; `native-save-repro.log` contains the actual pre-fix native defect.

## Compatibility and continuation

Existing Step 5/IL checkpoints, optimizer/RNG state, counters, datasets, frozen banks and P0–P4 artifacts retain their contracts. Continue those runs with their existing commands; do not reset, retrain or migrate them for this audit.

For P5, retain old evidence under its original identity. The new verifier intentionally refuses old panels/reports: they lack the new job-ownership guarantee. Once a real P4 roster is qualified, use a **new output root** (for example `P5="$ROOT/artifacts/personality/p5-audit-1.2"`) and the plan/collect/check commands in [the P5 guide](PERSONALITY_P5_LAB.md). Obtain new calibration and disjoint release evidence before claiming qualification. Do not delete or rewrite `reservations.json` to retry a failed game. A reserved crashed job requires an explicitly amended study; a different chunk name is no longer a retry mechanism.

For P6, re-export a new immutable candidate from compatible qualified P4/P5 evidence, restage its helper, and rerun the backend/Unity acceptance procedure in [the P6 guide](PERSONALITY_P6_DEPLOYMENT.md). Never replace an old package's runtime in place or edit an old neural save's package ID. Existing valid neural saves remain tied to their original immutable package/runtime. Envelopes already invalidated by the first-save bug remain rejected; no speculative save repair was added. Legacy-only saves retain their format.

Reproduce the Python regression suite from a dependency-provisioned environment:

```powershell
$env:PYTHONPATH="$PWD/src;$PWD/tests/R2;$PWD/tests/Recovery;$PWD/tests/Personality"
python -m unittest discover -s tests/Personality -p 'test_*.py' -v
```

Use fresh output directories for `smoke_difficulty.py`, `smoke_laptop.py`, `check_p56_native_audit.py`, `check_laptop_native.py` and `check_laptop_pipeline.py`. The latter requires a short Windows sandbox path and an interpreter whose own environment contains the package dependencies. Development `PYTHONPATH` additions are deliberately not inherited by installation preflight.

## Remaining limits

- There is no qualified real parent/roster or P5/P6 release evidence locally. Synthetic positive certificate tests exercise validator logic only; they are not release certificates for a real package.
- Windows CPU and isolated native hooks were exercised. Linux, lab CUDA, Intel graphics, multi-GPU behavior and representative real-model laptop performance were not measured.
- Actual Unity interaction, menu rendering, user-driven rewind/cancellation and post-game resource restoration still require the documented acceptance session. A stubbed native initialization plus a separate real Python pipeline is not a full interactive game test.
- Cold Play Now still waits for lazy model/native initialization. Native cooperative operations can outlast their soft cancellation point; the existing measured overrun gate remains in force.
- Abrupt OS/process termination during installation/rollback remains a manual journal/backup recovery case. Ordinary replacement failure recovery is covered by existing tests. No live deployment was attempted.
- This was a bounded implementation audit, not proof that the code is bug-free. No long qualification campaign or throughput benchmark was run.
