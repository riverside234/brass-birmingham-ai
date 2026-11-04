# B02 collection timeout repair (g570-r7)

Reviewed 2026-11-03, America/Chicago. Report:
`results/rust-lab-qualification/20261104T005813812452Z-20261104T005116206749Z-g570-all-c370fde6-41fd94d4`.

B01 passed on g570 / cuda:0. The r6 reference-runtime preflight also passed.
B02 then exceeded the unchanged 240-second collection watchdog, with one family
published and no collection receipt. The log contains no per-lane progress or
stacks, so it cannot prove the precise state of the remaining lanes at timeout.

Code review found that text-mode `Pipe` used `FileIO.readline` on unbuffered
stdout: one OS read per byte of a large oracle JSON reply. A short test using
the real C# bootstrap, the frozen self-play seed, 378 legal choices and ~113 KB
decisions reproduced severe contention across four lanes. Six replies per lane
took 10.2–11.0 seconds unbuffered versus 0.041–0.052 seconds buffered. This is
about 226x faster for this transport microbenchmark, not a claimed whole-game or
Linux/CUDA speedup. Replies were exactly equal and checkpoint restoration passed.

r7 buffers only text stdout in 64 KB blocks. Stdin remains immediate, binary
framing remains unchanged, and reply size limits, errors, deadlines and process
cleanup remain enforced. The oracle now uses the complete decision already
returned by `step` or `restore`, eliminating the redundant following `decision`
RPC. A fresh oracle is created for every lane and fork restoration replaces its
cached decision. Each lane reuses one independent Python feature encoder;
per-row observation, legal-choice and feature comparisons still run, as do
terminal/scoring/target, behavior, branch and atomic family/retry checks.

The smoke writes `collection-progress.json` every ten seconds with stage, lane,
step, era/phase and candidate count. The captured log gets thread stacks every
60 seconds if the smoke is still running. Both survive the outer process-group
watchdog and are included in the existing Git-visible result export. No private
game state is added to these diagnostics. The 240-second limit and 60-worker
ceiling remain; only useful lanes from the same three frozen parents and their
eligible continuations are allocated. No extra games, shorter trajectories,
resampling, recipe changes or training admission are introduced.

Validation includes real subprocess tests for large/small/fragmented text,
binary framing, partial-line timeout cleanup, oversized/truncated replies,
decision reuse and diagnostic failures. A four-lane real-oracle smoke checks
48 bounded decisions plus shuffled checkpoint restoration against fresh
decision replies. Continuation and B01–B04 harness checks are recorded in
`distribution/rust-lab-g570/local-development.json`; fresh extraction evidence
is in `package-check.json`. No full games or local training were run.

Only implementation inputs `lab_transport.py`, `lab_b02_smoke.py` and the new
`lab_smoke_progress.py` change. Implementation:
`a7d39190382b768058f699ba94a8e870b7ca18eb54c636392fc2eeec6b8dc31b`.
The Rust core/teacher producer and model/optimizer/RNG/recipe remain unchanged.
Exact source-continuation reviews preserve the full g570 state and utility
readiness; earlier approved teacher-adoption routes are retained without a new
recipe adoption. Because executable inputs changed, **rerun B01** for r7.
The received r6 B01 pass remains historical evidence; r7 lab gates are Pending.

Use the [current commands](../distribution/rust-lab-g570/README.md). Previous
archives and original result directories are not overwritten.
