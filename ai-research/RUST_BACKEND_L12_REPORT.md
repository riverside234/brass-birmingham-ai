# L12 — Two-session Gumbel coordinator and cooperative batching

Updated **2026-10-25**. **L12 is implemented and qualified. L13 is next.** Debug, release and
profiling checks pass; the two-session engineering sweep and final receipt audit
are complete. Installed gameplay and lab training remain unchanged.

Identity: `brass-gumbel-is-mcts/coordinator-v1`.
Native source SHA-256: `93415e701757878b931085e0fff7a65fc42a7dfc482a6abc73e6f57c446b8a71`.
Evidence: [`20261025-l12-qualified-v2`](../artifacts/rust-backend/20261025-l12-qualified-v2/).

## Implemented behavior

- [`Pair`](../native/brass-ai-worker/src/pair.rs) maintains **two persistent
  sessions, one compute thread each**. Each owns its tree, mutable sampled worlds,
  cache, reusable feature buffers and single-threaded ONNX Runtime evaluator.
  The coordinator performs no simulation or neural forward. Existing pinned
  Crossbeam channels provide bounded control/reply queues.
- Both sessions receive the same immutable permitted root logits, four-seat
  value, canonical actions and root Gumbels. Sampled-world/chance streams remain
  private. Only eligible root snapshots are pooled: sum completed R/C, recompute
  Q/G, and rank the union of survivors. One/no eligible session and conflicting
  survivor sets have independent recorded reference cases.
- Complete-plan support remains separate. A plan must independently meet the
  existing two-contribution minimum inside one owner's eligible watermark.
  Pooling two unsupported one-visit fragments cannot fabricate support. Both
  owners validate/adopt the selected plan, preserving continuation and action-two
  recall without another allowance.
- The scalar baseline remains **cap 1, pending 1, immediate flush**. Bounded
  cooperative waves support caps 2/4 and 0/0.5/1/2 ms flush ages. Root reservations
  occupy launch slots only; FIFO publication cannot cross the exact round barrier.
  Non-root selection sees completed counts. Identical new leaves in a wave share
  one inference reply, while each completed path contributes its own backup.
- Publication journals only touched edge counts/returns. Cancellation before an
  edge or after the final edge rolls back the whole backup and its unpublished
  expansion. No full-tree copy or shared tree lock is used. Reservations retire
  exactly once, and a late result cannot create visits or complete-plan support.
- Deep native work polls cancellation through an explicit thread-owned scope:
  generation, sampling, observations, application, traversal, encoding, expansion,
  backup, publication and completion. A compile-fail ownership test prevents the
  scope from moving threads. Non-cooperative synchronous inference keeps the pair
  busy until both owners actually return; failed dispatch and owner panic also
  drain the surviving owner before exposing idle state.
- The original **4/8/10/12-second** clock and **0.4/0.8/1.0/1.2-second** reserve
  remain. Caller remaining time becomes an absolute deadline before dispatch,
  root preparation and validation. Retries/continuations cannot replenish it.
  Forced roots and debt launch no NN. Cancellation/hard expiry exposes no action.
- Immutable information keys and legal-ID vectors use Serde's existing `Arc`
  support, retaining semantic hashes/equality and identical serialized structure.
  The private trajectory application path reuses its already admitted legal
  choice at the same revision instead of enumerating the full legal set again.
  State mutation still uses the qualified owned-copy transaction.

No new crate or version was added to the pinned dependency inventory. The implementation reuses
the pinned SlotMap, Crossbeam, Serde, ORT and `stats_alloc` crates. Serde's `rc`
feature and the existing allocation crate's worker dependency are explicit in
Cargo metadata/lock receipts. The normal executable uses the normal allocator;
allocation attribution has its own executable.

## Reference and verification

The independent small
[`reproduce-gumbel-coordinator.py`](../scripts/reproduce-gumbel-coordinator.py)
and [`coordinator-v1.json`](../tests/GumbelISMCTS/fixtures/coordinator-v1.json)
define five merger cases and two reservation/retirement traces. The detailed
[versioned contract](../configs/contracts/rust-backend/gumbel-coordinator-v1.md)
documents the scheduling semantics, absolute deadline and publication boundary.

| Check | Result |
| --- | --- |
| Native tests, including ownership doctests | **82 passed in debug; 82 in release** |
| Actual ONNX golden test | Passed in both builds, 11 positions per run |
| Independent Python oracle tests | **22 passed** |
| Process checks | **59 passed per debug/release/profiling build**: 11 deadline/output, 9 session, 12 search, 11 transport, 9 inference, 7 feature |
| Retained L10 count parity | 17 toy and 44 Brass cases per debug/release build |
| L11 singleton compatibility | 44 roots per recorded/native evaluation mode, both builds |
| Two-owner native NN corpus | 44 cases / 40 era-phase-actor cells, **1,536 trajectories and 55 complete actions** per build |
| C# plan replay | Debug: 99 choices; release: 99 choices, across 39 original checkpoints each |

Each paired corpus contains **22 search-supported decisions, 12 forced
completions and 10 debt decisions**. Formatting and strict own-code Clippy pass;
six preserved upstream warnings remain unchanged. Every binary handshake agrees
with the independent source fingerprint and lock receipt.

Receipts: [qualification](../artifacts/rust-backend/20261025-l12-qualified-v2/qualification.json),
[final audit](../artifacts/rust-backend/20261025-l12-qualified-v2/final-verification.json),
[debug replay](../artifacts/rust-backend/20261025-l12-qualified-v2/debug-coordinator/verification.json)
and [release replay](../artifacts/rust-backend/20261025-l12-qualified-v2/release-coordinator/verification.json).

The 44-case correctness corpus includes 40 era × phase × actor cells. All 44
singleton roots, with both recorded and actual native NN evaluation, are checked
against sealed L11 counts, returns, Gumbels and complete plans. The paired path
uses independent sampled worlds and cap-four scheduling; its pooled equations
are checked independently in Python. Original-root plans replay through the
frozen C# rules exporter. Five explicitly rotated-seat fixtures retain their
native/comparator evidence. Count correctness uses a **60-second watchdog**,
separate from four-second performance measurements.

The ten new core regressions cover pooled equations, malformed snapshots,
reservation quotas/retirement, singleton parity, repeatable wave interleavings,
duplicate expansion, late batches, transactional rollback, foreign handles and
deep generation failures. Three owner tests cover pooled continuation, narrowed
deadlines and panic/partial dispatch. Existing L11 tests retain partial-round,
last-backup, support-watermark, changed-legality and stale-context coverage.
Process tests inject all 11 failure stages, test 250 ms non-cooperative inference,
late runtime output, zero-estimate fallback, cap 1/2/4 quotas, actual NN batches,
the four reserve settings and cleanup after a later streamed case fails. Cancellation returns no action and keeps busy
ownership until runtime retirement.

## Four-second engineering measurements

The development screen selected cap 2 / pending 2 / flush 1 ms, but its held-out geometric completed-work ratio was **0.979** (-2.1%). The gain was not confirmed. Retain **cap 1 / pending 1 / immediate flush** for L13; larger batches remain experimental. See the [performance assessment](../artifacts/rust-backend/20261025-l12-qualified-v2/performance-assessment.json).

| Cap / pending | Flush (ms) | Completed trajectories | Paired geometric ratio | Supported searched turns | Maximum ready time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 / 1 | 0 | 7,194 | 1.000 | 8/8 | 3.644 |
| 2 / 2 | 0 | 7,450 | 1.030 | 8/8 | 3.645 |
| 2 / 2 | 0.5 | 7,198 | 0.998 | 8/8 | 3.647 |
| 2 / 2 | 1 | 7,774 | 1.078 | 8/8 | 3.648 |
| 2 / 2 | 2 | 7,351 | 1.021 | 8/8 | 3.640 |
| 4 / 4 | 0 | 7,600 | 1.053 | 8/8 | 3.645 |
| 4 / 4 | 0.5 | 7,446 | 1.039 | 8/8 | 3.650 |
| 4 / 4 | 1 | 7,610 | 1.058 | 8/8 | 3.660 |
| 4 / 4 | 2 | 7,295 | 1.016 | 8/8 | 3.648 |

The 90 development turns include 72 searched turns and 18 forced completions.
All finished inside their four-second allowance; the maximum internal ready
time was **3.660 s**. The retained baseline produced **7,194 completed
trajectories**, of which **5,269** were in eligible snapshots; **1,925**
partial-round contributions were excluded from recommendation. **1/16**
searched owner tournaments completed their entire frozen quota. Its latest
successful backup was **3.599865 s**,
before the 3.6-second work deadline. Work-stop/retirement can follow that point;
late work does not publish a backup. These counters are distinct from plan support.
Mean unused work allowance was **1.4 ms**
per searched owner. Frozen-quota completion does not start another tournament.

Held-out validation passed **8 four-second turns**; the maximum ready
time was **3.658 s**.
All **6 additional 8/10/12-second setting checks** passed. Every
measured turn retired both owners, and all recorded completed-backup timestamps
preceded the applicable work deadline. Observed maximum CPU polling gap in the
development sweep was **50.73 ms**; this is an observed wall interval, not a
real-time upper bound.

The final development process used **515.48 CPU seconds** over
**266.70 wall seconds** and peaked at
**308.4 MiB** working set, including runtime,
input/output and diagnostics. Streaming removed the earlier 5.26 GB accumulation.
The final raw data and protocol are in the
[benchmark receipt](../artifacts/rust-backend/20261025-l12-qualified-v2/benchmark/benchmark.json).
No result here is an L14 matched speedup or playing-strength acceptance.

### Allocation and copying decision

| Probe (100 calls) | Deep-copy allocations | Deep-copy bytes | Shared-key clone allocations |
| --- | ---: | ---: | ---: |
| Dense Canal, 413 choices | 234,500 | 15,306,800 | 0 |
| Wide Rail, 246 choices | 204,800 | 13,389,300 | 0 |
| Dense Rail payment, forced | 145,100 | 9,055,500 | 0 |

Semantic hash lookup and the warm feature encoder also make zero allocations in
these isolated probes. Constructing a fresh permitted information set and legal
choices still allocates; the deep-copy comparison includes two `Arc` containers
and is not a byte-identical historical binary benchmark.

Whole native-NN decision probes allocate approximately **25,111–27,375
objects and 1.55–2.11 MiB per completed trajectory**,
including admission, inference, plan support and final validation amortized over
128 trajectories per paired case. Forced payment is reported per decision instead.
These are allocation traffic, not retained bytes or a leak count.

Measured owned state copies account for **0.42–0.71%**
of owner elapsed time with native NN, and **0.64–1.08%**
with scripted inference across the five roots, including dense payment. Retain
the L07 owned-copy transaction; an undo journal is not justified by this share.
Fresh information/choice construction remains a larger allocation target. Copy
times cover the instrumented native state copies, not every tensor/string copy.


The protocol freezes five development roots and two held-out roots before
selection. Development includes a 413-choice Canal action, a 246-choice Rail
action, bonus and continuation roots, and dense Rail payment. The payment root
has one forced choice: it measures completion/copy cost, not search throughput.
Two repetitions compare nine cap/flush policies under **one four-second turn**,
with exactly **two sessions × one thread**. Caps above one can change non-root
selection order, so these are separately identified scheduling variants.

Timed quotas use a separately recorded cap-one, 64-trajectory-per-owner startup
calibration of this implementation/model. The conservative estimate includes
root preparation and freezes before each tournament; it is explicitly identified
as L12 evidence, not relabeled historical L09 timing. No restart spends unused
time. The sweep selects an engineering candidate only for at least 5% geometric
completed-work improvement with no fewer supported plans, then checks held-out
roots. This does not establish equal-time playing strength.

Normal timing and allocator runs use separate executables. Scripted and actual
NN attribution are also separate. Allocator counts represent traffic during the
turn; Windows peak working set and CPU time cover the whole diagnostic process,
including model startup and output. CPU polling gaps exclude the synchronous
evaluator interval; OS descheduling may enlarge wall-time gaps. Final adoption
is covered by delivery timestamps, not the search-component timing sum.
Ready timestamps describe the validated typed decision inside the research
process. Diagnostic JSON serialization and future Unity application are outside
that timestamp; production delivery remains an integration/latency gate.

## Scope and reproduction

This milestone supplies the research library and bounded diagnostic executables
`gumbel-pair` / `gumbel-pair-alloc`. Live `search` protocol capability, packaging,
g890-specific readiness admission and Unity shadow integration remain **L13**.
**L14** retains the matched implementation/algorithm comparison, equal-time
strength and production latency acceptance. **L15** retains full local games
and deployment checks. No mixed NN/V4 table or V4 fallback is introduced.

The pinned g890 checkpoint/export remains the temporary migration test model.
The user's utility-head acceptance is retained separately from its unchanged
null calibration receipt. Installed gameplay, saves, model artifacts and lab
training are preserved by before/after receipts. This milestone does not promote
the backend to training or treat the temporary NN as final.

```powershell
python scripts/qualify-rust-coordinator.py --out artifacts/rust-backend/my-new-l12-run
```

Use a fresh output directory with the pinned local toolchain/model/corpus. The
qualifier builds debug/release/profiling, checks scalar and paired paths, runs
the four-second sweep and seals source/binary/command/raw-output archives.
Historical completed milestone receipts remain immutable.

The first L12 run passed its behavior checks but was not selected: retaining all
90 expanded diagnostic results raised peak working set to 5.26 GB and could
bias timing. The harness now streams each result through a bounded writer,
releases it before the next admission and publishes the output file only after
success. The Python measurement process also drops expanded proofs before
launching the next measurement. A failure regression verifies partial-file
cleanup. The final audit also caught a Python/Rust source-fingerprint ordering
difference for a module file beside its test directory; Python now matches
Rust component ordering and every build verifies its handshake immediately.
The initial archives and failed final-audit receipt remain unchanged.
