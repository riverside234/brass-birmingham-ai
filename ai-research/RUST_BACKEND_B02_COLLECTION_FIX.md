# B02 collection smoke repair (g570-r6)

Reviewed 2026-11-03, America/Chicago. Received report:
`results/rust-lab-qualification/20261104T003413309751Z-20261104T003002821892Z-g570-all-adb9f30b-d6f429a2`.

The r5 compiler/bootstrap fix passed. Collection stopped constructing a V4
reference policy: `Rebuild legacy runtime after source change: src/BrassLegacyAI/Program.cs`.
The shared runtime had already passed `runtime_directory`'s canonical-source
identity check. `LegacyWorker` then compared current raw bytes with build-time
raw bytes, rejecting text-equivalent newline/BOM changes after Git/ZIP transport.
An LF-build/CRLF-checkout regression fixture reproduces the failure.

The repair retains raw checks for ordinary legacy banks and frozen manifest
checks for historical rating references. For a current shared runtime whose raw
source differs, it verifies the recorded shared receipt, exact manifest hash,
current canonical source/profile/adapter identity and complete source inventory.
All executable and private dependency hashes must still match. No runtime
manifest, executable, policy ID or production state is rewritten. Semantic source
edits and tampered binaries/manifests still fail.

B02 now preflights distinct non-neural policies before starting diagnostic
families, and closes earlier policies if a later constructor fails. The default
worker ceiling is **60**, configurable with `--b02-workers` on the unified
launcher (`--workers` on B02). Effective capacity also respects the saved
`legacy_budget.global_workers` and CPU budget minus reserve. The smoke retains
three frozen parent families and at most one continuation per eligible native
family. It allocates only useful lanes (normally four or five), instead of the
previous two; it does not invent 60 games or allocate idle inference buffers.
`worker-settings.json` and the final smoke receipt record requested/training/CPU
limits, actual lanes and continuation capacity. The 240-second watchdog remains.
B04's two one-thread search sessions and 4s search plus at most 1s post-work are
unchanged.

Changed production fingerprint inputs are only `src/brass_ai/legacy.py` and
`src/brass_ai/rust_backend/lab_b02_smoke.py`. Exact reviewed source-continuation
routes preserve the recorded g570 recipe, complete model/Adam/RNG/PPG state and
utility readiness; this is not a recipe adoption. Implementation:
`20efa9ea06dad63f8301f0352d56f254c245b0a138caea2bd529dfc17873629e`.
The native core and teacher producer are unchanged. Because executable source
changed, the old B01 receipt remains evidence for r5 only: **rerun from B01**.
No stale-receipt bypass is added. All r6 lab gates remain Pending.

Short local regression/continuation/inference suites and fresh-package checks
are recorded in `distribution/rust-lab-g570/local-development.json` and
`package-check.json`. Lab evidence copies retain original export hashes under
`artifacts/rust-backend/g570-b02-collection/lab-evidence/`. No local full-game
campaign or production training was run. See the [r6 commands](../distribution/rust-lab-g570/README.md).
