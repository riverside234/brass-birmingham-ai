# B06: matched workload measurements

2026-11-07. Implemented; Linux/CUDA qualification **Pending**. B05 r2 passed on
the pinned g1250 checkpoint in
`results/rust-lab-qualification/20261107T070806231803Z-20261107T070539193547Z-g1250-b05-88497ac1-cf4704fd/evidence/qualification.json`.
It completed two PPO and two PPG steps with diagnostic checkpoint reload, 32
search-head steps on 128 frozen value rows, and four authored teacher/PPO steps.
These remain component diagnostics; production was unchanged and no games were collected.

Use the [B06 bundle and lab commands](../distribution/rust-lab-b06-g1250/README.md).
The harness lives under `scripts/`. A compatibility fix in the native Python
adapter and its sealed-family validator changes the training implementation
identity to `63564be49e282c4b2c0fff02b7640238c63d04d35f7c072bf8c017d12ce6ace5`;
search equations, native producer/binaries, checkpoint and optimizer behavior
are unchanged. The exact before/after file hashes are in the bundle's
`source-reuse.json`. Previous restricted resume routes are preserved for this
reviewed change; B05's update/teacher/restore evidence retains its original source
and scope. No production source adoption occurs.
It verifies every bundled file, the exact current checkpoint/config, and original
qualified collector/teacher binary dependencies. It holds the normal run lock,
checks source dependencies after work, and writes first-error details even when
a child fails before loading the model. Each phase runs in a bounded child process.

## Ordinary comparison

The compatibility review found that the native branch adapter still required
parent and continuation opening windows to match. Current v9 intentionally uses
T1.3/three on parents and T1.5/two after the excluded fork action. Native branch
construction now switches only its private policies' opening window, retaining
their frozen policy IDs; family validation derives the same branch window when
replaying probabilities/RNG. Parent settings, original packet and earlier branch
contracts remain unchanged. Regression fixtures exercise actual private-lane
construction, real v9 sampling/likelihood admission, wrong windows and historical
family admission. The never-published r1 archive is retained locally; use r2.

Both arms use the actual frozen planner and exact g1250 recipe, including current
neutral inputs, cache ownership, sparse probes, two-age PPG, and branch T1.5
openings. The first job in each actual `(table kind, backend route)` stratum is
selected without changing seeds, rotations, opponents or branch probability.
At most three parent families per arm run. V4 seats may occur in historical
tables; no obsolete standalone legacy-table requirement is imposed. Unsupported
mixed tables retain whole-table reference routing and their cost remains visible.

Reference uses the current persistent `ActorPool`, asynchronous family writer,
feature caches and branch scheduler. Candidate uses existing `collect_families`,
`collect_lane`, `FamilyStore` and diagnostic native replay admission. Natural
branch triggering is checked explicitly even though outputs cannot train.
No lockstep oracle executes in the timed loop. Afterwards both complete families
must agree on observations, legal order, selected actions, behavior identities,
results, branch semantics, prepared tensors, causal rewards and auxiliary targets.
Small probability serialization differences use the existing numerical tolerance.
Mismatched accepted work prevents any speedup claim.
Report collection-only ratios and collection-plus-first-admission/preparation
ratios separately; faster simulation cannot hide a slower accepted-data path.

Both arms keep the saved resource configuration (60 CPU work slots; active lanes
limited by the subset), fresh workers, the same frozen models preloaded onto CUDA,
and the existing inference scheduler/shared-memory/padding settings. Model loading
and planning are timed separately. Collection includes worker startup and sealed
family publication; reference host phases and native cost receipts expose nested
work. Admission, first/warm replay reads and first/warm preparation are distinct.
Phase times overlap and are never added as if independent. Process-tree memory
and CPU are sampled every 50ms; CPU is a lower bound because brief children can
escape sampling. CUDA allocation peaks and scheduler batch/queue telemetry are
reported separately. This is a bounded batch measurement, not a 320-game iteration
or saturated 60-worker scaling claim. If no natural fork occurs, its performance
coverage stays unmeasured; B02's earlier branch correctness evidence is preserved.

## Search and teacher overhead

Use the same typical and widest permitted roots, checkpoint, search_v1 equations,
K16 and 4s search plus at most 1s post-work. Freeze one calibration, verify
fixed-count zero-blend output parity, then time the old policy-only export and
all-seat w=0 export in ABBA pairs. Each pair runs two independent one-thread
sessions; eight root turns total. Diagnostic labels cannot enter teacher buffers.
Retain raw probes, partial failures, trajectory/round counts, effective K,
inference counts/waits/queue telemetry, selection, label, commit, verification/IPC,
post-work and total timing. Summaries expose sample counts, median, p90 and max.
Missing native probe encoding counters are null, never invented zero costs.

Full-table all-seat wall time, lane-seconds, inference requests/wait and full/fast
turn metrics come from the exact successful B04.5 g1250 tables shipped in this
bundle. That evidence is historical and distinct from the new root benchmark.
It is not compared as a speed ratio against the older g570 policy-only tables.
No additional full teacher games are needed.

The identical frozen policy labels go through the original table-batched validator
and actual streamed verifier in opposite order for the two tables. Compare accepted
bytes/targets/rejections, total work and one-use barrier cost separately. Original
identity, revision, reservation, model-logit, cancellation and allowance checks
remain; no metadata-only bypass. Buffer sealing/reads and a one-step search head
pass are isolated. The actual g1250 teacher KL forward/backward kernel is measured
twice on up to 64 real frozen labels, without an ordinary optimizer step. These
are kernel timings, not production beta calibration or full PPO-step overhead.

The ordinary update phase restores a complete isolated g1250 fork through B05's
qualified logical-CUDA-stream route. B05's authored bounded-input update/reload
helper is timed around the actual current PPO and PPG kernels. Current optimizers,
cached ages, companions and recipe remain intact. These phase estimates cannot
establish whole-iteration improvement. No new optimization is enabled without
lab evidence. Useful measured collection gain, validation cost and scope gaps
are reported for B07 review; one threshold never silently rolls back the backend.

## Validation and remaining gate

Short CPU tests exercise real family sealing/admission/cache/preparation, exact
trajectory comparison and rejection, frozen-plan selection, measured-work
accounting, verifier byte/target parity and tampering, CUDA flag/error propagation,
and Git-visible export behavior. CUDA matched performance remains Pending until
the lab command finishes. Earlier B01–B05 receipts are reused without rerunning
their qualification commands. The installed game and production training remain
unchanged; B07 owns explicit latest-state adoption and normal G-to-G+1 continuation.
