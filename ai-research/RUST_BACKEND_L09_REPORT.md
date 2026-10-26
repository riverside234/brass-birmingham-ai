# L09 — Same-process CPU inference and session lifecycle

Date: **2026-10-25**. Status: **complete; local mechanical qualification passed**.
Evidence: [qualification receipt](../artifacts/rust-backend/20261025-l09-qualified-v2/qualification.json),
[supplemental checks](../artifacts/rust-backend/20261025-l09-qualified-v2/supplement.json)
and [final verification](../artifacts/rust-backend/20261025-l09-qualified-v2/final-verification.json).
Native source SHA-256:
`235ed74478f7c54b58474158dcac1fe5af9e705dda6eb6bc591cf7a702a69989`.

## Implemented boundary

The new [`brass-ai-inference` crate](../native/brass-ai-inference/src/lib.rs)
loads and evaluates the temporary g890 model inside the Rust AI worker. It reuses
the pinned `ort = 2.0.0-rc.10` binding and compatible ONNX Runtime **1.22.1**,
with existing `serde`, `sha2`, channels and allocation instrumentation. The
crate has no custom unsafe code. Cargo.lock, runtime bytes, model bytes, export
metadata and their licenses are preserved with the candidate artifacts.

There are **two independent sessions, one compute thread each**. Each owns its
persistent model, encoder buffers and batch scratch. Both ORT thread counts are
one; execution is sequential with spinning disabled. Model construction,
warm-up, synchronous forwards and destruction happen on the owning thread.
`CpuEvaluator` is neither Send nor Sync; a compile-fail test checks that boundary.
Loading and two warm-up forwards precede timed evaluation. Changing model identity
requires a new worker.

`Evaluator` is the shared typed interface for local CPU evaluation and
`ReferenceEvaluator<T: ReferenceTransport>`. The reference adapter checks complete
row identity, ordering and all outputs before publishing. The concrete Python
reference adapter uses the frozen exporter evaluator for parity. GPU transport
and collection remain Part 2 work; the local CPU path has no Python encoder or
inference RPC dependency.

Singleton inference borrows the encoder's contiguous arrays directly. Mixed
batches pack into capacity-reusing arrays with explicit false padding masks.
All legal candidates are retained. The five outputs are complete legal logits
and four stable-seat heads: win share, placement, VP and remaining utility. The
export already maps values to stable seats, so Rust does not rotate them again.
Native f64 log-softmax and visited-logit normalization preserve finite legal
logits for Gumbel; extreme logits, underflow, ties and forced choices need no
invented probability floor. Gumbel sampling and tree backup are L10 work.

The [evaluation contract](../configs/contracts/rust-backend/inference-v1.md)
defines exact tensor names/types/shapes and model/export/reward/profile,
observer/revision, request/batch/generation identities. The bundle loader checks
the caller's manifest pin, bounded file sizes, bundle-relative paths, hashes,
metadata and the graph's actual input/output schema. L09 accepts mechanical-only
manifests with the model's actual null calibration; it cannot promote a fabricated
calibration or claim live readiness.

The worker adds asynchronous diagnostic start/poll/cancel controls while keeping
all computation synchronous on its owner. Requests can carry permitted views or
owner-local native state handles; the latter generates the acting view and full
legal set inside Rust. Only one active ticket exists per owner. Queues allow up to
four pending positions, 8,192 candidates per position and 32,768 padded cells;
the configurable cell bound may flush earlier. Control transport remains bounded
at one MiB and reports oversized results explicitly without truncating choices.
Native search will call the typed evaluator without serializing each leaf.

Cancellation/expiry discards outputs and retains the active slot until the
synchronous runtime call finishes and its result is drained. Returned results
own their buffers. A runtime/output failure retires the evaluator; malformed
transport retires the client process. Polling rechecks expiry even after a reply
has finished, and the Python caller also counts transport in its deadline.
The diagnostic drain watchdog retires a nonreturning runtime after its bounded
grace period; it does not extend accepted search work or establish Unity latency.

## Model and runtime identity

| Item | Pinned value |
| --- | --- |
| Temporary model | g890, v11.2 W512 × 8, 5,928,616 parameters; not the final trained model |
| Model ID | `1145fecba3289511df1c518bb71fbfb5b2f6df351777151a5eb999b89cba0761` |
| Checkpoint SHA-256 | `ecad3b932a8dd9d3b5d0cc49b5e9bb94a9e98b4536ffdc86a44179d003995cb8` |
| ONNX SHA-256 | `64e4a8340f4beb7bbc4ee70219b5e461b1185bcde9e25c4bfc78e30936c91ae9` |
| Model manifest SHA-256 | `3fee37f18cd7963f6dfde7b56fc211b9d424263bf6f27bcc4e7cb6e7a7e132b7` |
| Runtime DLL SHA-256 | `c7151fd9844ad7c7d18525f1177e9ef62d91e4a6ac3583d0be700554a2b2b1d6` |
| Runtime configuration | CPUExecutionProvider, float32, opset 17, all graph optimizations, exact GELU, intra/inter 1, sequential |
| Utility admission | `utility_calibration = null`; mechanical tests only, live/strength/teacher admission pending |

The ONNX export is 24,134,155 bytes and the runtime DLL is 14,854,688 bytes.
The original checkpoint is read-only throughout export and qualification.
The frozen Python reference uses PyTorch 2.8.0+cpu and ONNX Runtime 1.22.1.

## Parity and failure qualification

The harness executes the immutable L01 reference source with a separately
identified native diagnostic overlay. It reuses the captured L05/L06 corpus and
preserves the previous L08 receipts. No candidate output is used to regenerate
the expected observations. Both full runs compare PyTorch, Python ORT and Rust
ORT, including complete probabilities/log-probabilities and all four value heads.

| Check | Evidence |
| --- | --- |
| Native tests | 44 regular tests plus one compile-fail ownership doctest, in debug and release |
| Actual-runtime golden test | Separately enabled in both builds: 11 positions in six categorical/mixed exporter batches, widths through 772 |
| Full debug and release corpus | 2,244 unique positions / 127,702 candidate rows in each build |
| Profiling-build parity | 100-position smoke check with 31 singleton/mixed pairs |
| Corpus phases | 791 action, 780 continuation, 441 payment, 214 debt, 18 bonus |
| Acting seats | 616 / 555 / 549 / 524 positions at seats 0 / 1 / 2 / 3 |
| Critic context | 1,122 unknown and 1,122 explicit descriptor cases per full run |
| Singleton versus mixed | 57 extra paired checks per full run; actual batch sizes 1–4 |
| Process suites | 9 inference lifecycle tests, 7 feature tests and 11 worker transport regressions pass in each of debug, release and profiling |
| Lifecycle cases | Cancellation, busy-until-drained, expiry before execution and after completion, stale/foreign tickets, bad identities, model reload, bounds, corrupt replies, shutdown, watchdog retirement |

Cross-runtime comparisons retain the frozen exporter's **atol = rtol = 2e-4**.
This is an output tolerance across different runtimes; L08's exact float32 input
contract is unchanged. Maximum absolute errors across the full debug/release
comparisons are:

| Head | PyTorch versus Rust ORT | Python ORT versus Rust ORT |
| --- | ---: | ---: |
| Legal policy logits | 0.0000457764 | 0 |
| Win share | 0.00000190735 | 0.00000137091 |
| Placement | 0.00000190735 | 0.00000119209 |
| VP | 0.0000839233 | 0.0000381470 |
| Remaining utility | 0.00000196695 | 0.00000137091 |

Maximum full-policy probability difference against Python ORT was
`2.4424906541753444e-15`. Python evaluates oracle batches of four; Rust's
cooperative scheduler sometimes flushes smaller batches. These comparisons
include their small value differences; legal logits matched exactly after f32
conversion. The ordinary Rust runtime test is explicitly ignored without its
pinned bundle/fixtures and explicitly executed by the qualification command.

The supplemental release check proves **identical actor logits/probabilities in
35 unknown-versus-explicit critic pairs**. Forty authored cyclic-seat cases cover
every era × decision phase × actor, including bonus seats absent from the captured
matrix. They preserve exact actor-policy invariance and correctly rotate all four
stable-seat value heads, while also matching Python ORT within tolerance
(maximum difference 0.00000378662). The rotation transformation is inherited from
the preserved L08 harness; these are authored tests, not additional C# trajectories.
The main correctness corpus includes held-out cases; timing selection uses only
development data.

## Performance and remaining gates

The [main batch sweep](../artifacts/rust-backend/20261025-l09-qualified-v2/batch-sweep.json)
completed **48 configurations / 5,760 positions**: four workloads, caps 1/2/4,
flushes 0/500/1000/2000 microseconds, 15 iterations and two owners. Each diagnostic
job has four positions available to its cooperative producer. With cap 1 it
immediately evaluates each position, so only one is pending at a time. Timed
inputs are native simulator handles; Python does no feature encoding or inference.

Immediate-singleton stage means, per position, in milliseconds:

| Captured development workload | Candidates | Native encoding | ORT forward | Binding | Output copies | Postprocessing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial Canal action | 388 | 0.401 | 3.845 | 0.0065 | 0.0324 | 0.0485 |
| Wide Canal action (`dense` fixture) | 413 | 0.503 | 5.143 | 0.0076 | 0.0474 | 0.0833 |
| Rail payment (`complex` fixture) | 1 | 0.095 | 3.190 | 0.0063 | 0.0039 | 0.0016 |
| Mixed: initial/payment/wide/payment | 388/1/413/1 | 0.312 | 4.996 | 0.0094 | 0.0302 | 0.0517 |

These are normal-allocator measurements under concurrent owners. They exclude
model load/warm-up. They are not complete-leaf latency: view/choice materialization,
validation and diagnostic transport add work outside these stage counters.
The raw records separately retain dispatch, queue age, packing and complete owner
work intervals. Singleton queue age averaged 0.00038–0.00052 ms and packing
0.000045–0.000053 ms. All singleton calls copied **zero input tensor bytes**;
output copies were 3,232 / 3,432 / 136 / 1,734 bytes per position for the four rows.
Owned candidate IDs and response metadata add allocations outside those byte counts.

At cap 4 / 2 ms, mean ORT time per position was 2.777 / 4.187 / 1.255 / 2.543 ms,
respectively. The mixed workload's padding ratio reached 2.057 padded cells per
valid cell. A zero flush produces singleton calls even when the configured cap
is larger. Actual intermediate batch sizes can be three when a partial queue ages
out. The ordered short sweep has visible timing variation and no confidence
intervals; it establishes instrumentation and candidate policies, not a final
speedup or optimal setting.

All **720 paired jobs overlapped** in shared-process monotonic work intervals.
The process had six OS threads before model loading, after loading and after the
sweep. Two compute threads used 13.83 and 13.38 CPU seconds over 18.51 wall seconds;
the control thread used 0.94 seconds and the other three registered zero at the
counter's resolution. Loading inference added no threads. Two persistent model
handles used 129.44 MiB process RSS after loading and 144.43 MiB after the sweep;
these are snapshots, not peak-memory guarantees or isolated model-size estimates.

The separate allocation probe warms every shape, then measures 100 forwards per
owner at each cap. It includes Rust input validation and owned outputs but excludes
the ONNX DLL's C allocator and prebuilt observations/features. Every row had zero
reallocations and equal allocated/deallocated byte totals. It does **not** claim a
zero-allocation evaluator:

| Workload | Rust allocations/position, cap 1 | Cap 2 | Cap 4 | Rust bytes/position, cap 1 |
| --- | ---: | ---: | ---: | ---: |
| Initial | 435 | 420.5 | 413.25 | 54,435 |
| Wide Canal | 460 | 445.5 | 438.25 | 56,502 |
| Rail payment | 41 | 26.5 | 19.25 | 4,080 |

Candidate-ID cloning, validation and owned result construction remain measured
optimization targets for the native search integration. L08's separately measured
zero-allocation warm encoder is preserved. Instrumented allocation-run timings
are not used as performance results.

The inherited complex fixture is a Rail payment with one legal choice. Two
additional sweeps therefore select Rail **action** roots before any timing,
using development fixtures only: maximum occupied tiles (then candidates/history)
and, separately, maximum legal candidate count. Each runs all 48 configurations
for five iterations, adding **3,840 positions** including the unchanged control
workloads. Both preserve six OS threads and overlap in all 240 paired jobs.

| Rail action root | Occupied industry tiles | Legal candidates | Singleton encoding | Singleton ORT |
| --- | ---: | ---: | ---: | ---: |
| `game-731009-root-024` | 7 | 6 | 0.091 ms | 2.818 ms |
| `game-731014-root-018` | 2 | 208 | 0.276 ms | 3.658 ms |

These are captured states with modest occupancy, not a fully packed late-Rail
worst case. L14 must expand complex-state trajectory benchmarks before final
tuning. Raw results and selected fixture hashes are in the supplemental receipt;
no held-out performance data or fastest timing was used to select either root.

The initial policy is **cap 1, pending limit 1, flush 0 microseconds**. The bounded
sweep covers caps 1/2/4 and flushes 0/500/1000/2000 microseconds with two owners.
The producer advances available local work instead of sleeping to fill a batch.
Final batching selection depends on native trajectory/round/plan performance in
L14. Queue age overlaps other work and is not additive CPU time.

These are encoding/inference microbenchmarks. The matched **four-second whole-turn
search** benchmark, live 4/8/10/12-second lifecycle qualification and complete
Gumbel trajectories remain later milestones. The temporary model's missing
calibration still blocks live/strength/teacher use. Installed gameplay, UI/table
rules, saves, checkpoint weights and lab training are preserved.

## Reproduction and evidence

Use the project's existing Python dependencies and pinned native toolchain,
choosing a new evidence directory:

```powershell
python scripts/qualify-rust-inference.py --out artifacts/rust-backend/<new-l09-run>
```

The script exports into a fresh immutable bundle, prepares golden tensors, checks
generated layouts/formatting/Clippy, runs native tests and three worker builds,
executes parity and profiling, and archives source/binaries/model/runtime/notices.
It verifies that sources, frozen references, installed binaries, saves and original
model files retain their recorded identities. Ordinary worker/rules operation
does not load ORT until inference is requested.
The [supplemental harness](../artifacts/rust-backend/20261025-l09-qualified-v2/supplement.py)
and [independent receipt verifier](../artifacts/rust-backend/20261025-l09-qualified-v2/verify-final.py)
are preserved alongside their outputs. The qualifier preserves debug, release and
profiling executables plus the model/runtime bundle in `executables.zip`, and the
implemented source/test/contracts/scripts in `implementation.zip`.

The initial full qualification correctly rejected colliding diagnostic fixture
request IDs. Those labels are now unique across sampled worlds, and the full run
was restarted; the failed evidence remains preserved. An earlier development
export also stopped on a missing `onnxsim` search path, fixed by using the existing
project dependency folder. Neither failure required changing model weights or
loosening numerical tolerances.
The first supplemental attempt referenced a mask attribute absent from the Python
encoder; its preserved script was corrected to construct the full legal mask,
matching the main parity harness. The corrected supplement passed independently
without changing the qualified Rust implementation or model bundle.

**Next: L10**, deterministic Gumbel IS-MCTS against the executable reference.
