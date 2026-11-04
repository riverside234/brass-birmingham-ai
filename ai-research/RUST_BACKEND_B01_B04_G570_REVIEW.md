# g570 B01–B04 qualification review and r5 repair

Reviewed 2026-11-03. The reported failed chain **passed B01 and failed B02**.
Original report: `results/rust-lab-qualification/20261103T232335735601Z-20261103T232040081783Z-g570-all-fc1a5f74-025ddb53`,
received at commit `bea9eb6185ec501a19c27c0b30484ff3bb2d838c`.
Verified copies are retained under `artifacts/rust-backend/g570-review/lab-evidence/`;
every received file matches its original export byte count and SHA-256.
Original result folders and old bundles are unchanged.

## Diagnosis and fixes

| Finding | Change |
| --- | --- |
| B02 required Roslyn in a local Windows development cache, absent on the lab | Ship the unchanged public 4.12.0 NuGet package; verify its SHA-256 and restore compiler dependencies into the new evidence directory. Check startup, compile, initial decision, one action and checkpoint restore before the Rust build. |
| B02 verified the B01 inventory receipt but did not compare recorded executable inputs against the current checkout | Compare all `src/`, `configs/`, `training/runtime/` and authored-host builder inputs. Reject added, removed or changed executable inputs. Harness-only fixes may reuse a passing B01. B03/B04 inherit this entry check. |
| The bounded command helper called POSIX `killpg` unconditionally on timeout | Reap the owned process tree on Windows as well; tolerate a POSIX child exiting during cancellation and retain the original failure log. |
| Restarting the full chain unnecessarily repeated a passing B01 | Add `--start-at b02` with an explicit original B01 receipt. Check missing dependencies before any stage; preserve checkpoint/state checks, failure/deferral stops and Git-visible exports. |

The production implementation remains
`70a926e5dfc7aa30829c296b49cb7b53159534ccc5e7c8876c46a9609a44164d`,
teacher producer remains
`bc2c07c397fdc979c9d4376189e723af86620e6361e3cd7052f3500cea3709cb`,
and the shared core/source approvals are unchanged. Comparison against the
received B01 source/runtime inventory matched 659 files (cross-platform review
allows Git line-ending conversion; the live reuse gate checks exact raw pins).
All runtime inputs carried in r4 remain byte-identical.

## Evidence and review coverage

The actual lab B01 receipt is `passed: true`, completed at
`2026-11-03T23:23:11.762035+00:00`. Its 13 roots and nine staged transitions
include real `cuda:0` parity, two-model inference, cancellation and expired
requests. B01 qualifies that exact g570/source/device scope; it does not claim
full-search speed or teacher readiness. B02's native build/tests and 20 Python
adapter tests passed before the missing compiler stopped collection.

Short local checks collected 168 tests: **165 passed, three explicitly skipped**.
The skips are Windows unprivileged symlink creation and two optional native
teacher probes requiring the export-overlay binary. No native/CUDA result is
inferred from those skipped tests. Separately, the restored compiler built the
reference bootstrap and its one-action/checkpoint smoke passed.

Coverage includes B01 environment/imports, native include preparation,
inference deadlines and owner lifecycle; B02 behavior probabilities, forced
rows, branch-family publication/retry/cancellation; B02.5 teacher loss isolation,
shadow equivalence, adoption and resume; B03 tamper/truncation/duplicate rejection,
whole-family holdout, PPG/optimizer/RNG restoration; B04 task-only profiles,
readiness, stale epochs, cancellation, full legal logits, once-only verification
and the 4s search plus cumulative 1s post-work cutoff. The review found no
additional production-path defect in these bounded checks; real B02–B04 lab
execution remains Pending.

See `artifacts/rust-backend/g570-review/review.json` and the current
`distribution/rust-lab-g570/local-development.json` for test/log pins.
`distribution/rust-lab-g570/package-check.json` records the separate fresh-bundle
transport/integrity and bootstrap check. No full training iteration, production
checkpoint modification, game installation or long performance session was run.

## Lab continuation

Use the [r5 commands](../distribution/rust-lab-g570/README.md). Resume at B02 only
while the original g570 state and its B01 dependencies remain unchanged. If the
run advanced or executable inputs changed, qualify the new latest state from B01.
Keep `search_v1`, K16, DOMiNO/AZ_DB off and production teaching disabled.
