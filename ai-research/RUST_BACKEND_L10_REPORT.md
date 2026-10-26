# L10 — Native Gumbel IS-MCTS

Date: **2026-10-25**. Qualification: **passed**. Search identity:
`brass-gumbel-is-mcts/count-v1`.

L10 adds the scalar Rust search, a diagnostic full Brass Python comparator, native
NN integration, trace/failure tests and component probes. It retains the fixed-count
paper-reference semantics. Whole-turn deadlines, complete-action plans, two-session
aggregation and Unity deployment remain L11–L15.

## Implementation

- `native/brass-ai-core/src/search/{math,halving,tree,trajectory}.rs` implements
  completed Q, visited-prior normalization, improved policy, Equation (3), fixed
  root Gumbels and post-backup sequential halving. Statistics use f64; ties retain
  the first canonical semantic action. All legal logits remain in the denominator.
- The arena reuses pinned **slotmap 1.1.1**, with generational handles and an extra
  owner identity. Typed hashing and semantic equality cover complete permitted
  information, actor recall, legal choices, model/profile/reward context and public
  utility budget. Hash collisions never establish node equality.
- `brass.rs` samples fresh detached worlds using L06's permitted-view sampler and
  invokes the qualified native rules directly. `reward.rs` preserves per-seat
  insolvency accounting and official terminal tiebreakers/reward caps. Each backup
  subtracts only costs incurred after that edge's observation.
- An owned pending leaf carries the world and bounded path across evaluation.
  Completion consumes it; backup preflight precedes expansion/publication. Invalid
  replies, stale handles, failed expansion, depth exhaustion and memory limits
  cannot invent a completed visit or truncate the legal action set.
- `native/brass-ai-inference/src/search.rs` connects the same-process Features-4
  encoder and session-owned CPU ORT evaluator. All actors use the root model on
  their own permitted views, with unknown critic descriptors and a fixed root
  personality context. There are no frozen-policy opponent rollouts.

The [count-mode contract](../configs/contracts/rust-backend/gumbel-search-v1.md)
specifies identities, limits, ordering, failure semantics and measurement scope.
The implementation adds read-only state accessors and typed `Hash` derives; it
does not change the qualified game rules or feature definitions.

## Comparison method and corpus

[`gumbel_brass_reference.py`](../tests/RustBackend/gumbel_brass_reference.py) runs
Python Gumbel selection and cost-aware backup over the **sealed L09 simulator**,
which has no Gumbel search implementation, with the frozen Python encoder/ORT
evaluator. L04–L07 supply the prior independent C# rules qualification. This is a
full Brass search comparator, not another independent implementation of the rules.

Every case records permitted shuffle indices, chance uniforms, acting-seat NN
inputs/replies and causal root ledgers. Rust first replays those exact NN replies;
then it reruns with native encoding and ORT. Comparisons cover complete node
information/legal sets, selected edges, first-new leaves, all four backup values,
counts, improved policies, round elimination and final action. Recorded-reply
numeric tolerance is 1e-10; native inference uses the existing 2e-4 tolerance.
Action IDs, counts and tree/tournament structure must match exactly in both modes.

The corpus has **44 Brass searches / 1,664 trajectories**, comprising 35 captured
actor/phase combinations, five labeled cyclic-seat rotations and four additional
late Rail searches. This covers all **40 era × phase × actor combinations**.
Initial budgets are 32 trajectories/K=8; the four late cases use 96/K=1. These are
correctness settings, not tuned deployment settings.

The Python reference records **1,414 nodes / 84,216 legal candidate rows**, **1,370
neural leaves**, **294 terminal leaves**, **5,381 selected decision edges**, **6,413
chance steps**, **1,241 opponent edges** and **168 cost-adjusted edge backups**.
The longest path has **68 decision/chance steps**. Debug and release match every
case in both recorded-reply and native-NN modes. The profiling build also passes
four Brass searches (128 trajectories per mode) and all 17 toy cases.

The toy suite includes the checked-in golden trace and 16 additional two-/four-seat
cases covering N=0/1/2/7/23/24/32/200, K=1, inadequate budgets and odd requested K.
Independent Rust tests cover literal Equation (3), prior-weighted completed Q,
underflow, near/exact ties, post-backup elimination, arena collision/ownership,
failed/dropped leaves, cost deltas and official terminal ties/VP capping.

## Qualification and profiles

Run [`20261025-l10-qualified`](../artifacts/rust-backend/20261025-l10-qualified/qualification.json)
passed with native source SHA-256
`629c33727290c094898930b960f265f2521d20060d12a9c63f4c49b4dfca756f`.
Evidence includes [debug parity](../artifacts/rust-backend/20261025-l10-qualified/debug-parity.json),
[release parity](../artifacts/rust-backend/20261025-l10-qualified/release-parity.json),
[profiling smoke parity](../artifacts/rust-backend/20261025-l10-qualified/profiling-parity.json)
and the [independent final audit](../artifacts/rust-backend/20261025-l10-qualified/final-verification.json).

- Formatting and strict Clippy pass for the workspace and all targets. The six
  pre-existing upstream warnings remain isolated from our lint gate.
- **53 native tests plus one ownership compile-fail doctest** pass in each of debug
  and release. A separately enabled actual-runtime golden test passes in each
  build, evaluating the 11 previously frozen tensor positions.
- **12 new search-boundary tests**, nine inference tests, seven feature tests and
  11 transport regressions pass in **each of debug/release/profiling**.
- The unchanged Python Gumbel oracle's **14 tests** pass. All **17 toy traces**
  match in all three builds. Debug/release each verify **44 Brass traces twice**,
  first with taped replies and then native inference.
- Native-NN versus taped responses differ by at most **7.11e-15 in logits** and
  **2.23e-16 in leaf utility** in these singleton searches; the maximum accumulated
  return difference is **4.45e-16**. Counts, paths, chance outcomes and final actions
  agree exactly. These measured errors do not replace the declared general ORT
  tolerance or extend this result to other models/platforms.

The qualified source/executable archives, reference oracle archive, per-command
logs, unchanged model receipts and complete raw traces are retained. The frozen
reference, upstream source, installed backend, original saves and g890 bytes pass
preservation checks. The capture is
[`20261025-l10-capture`](../artifacts/rust-backend/20261025-l10-capture/index.json).
Two initial development-harness errors (a missing direct Serde dependency and a
capture projection expecting a `views` field) were corrected before this complete
qualification; development artifacts remain separate.

The reproducible command is:

```powershell
python scripts/verify-rust-search.py --prepare --fixtures artifacts/rust-backend/<fresh-capture>
python scripts/qualify-rust-search.py --fixtures artifacts/rust-backend/<capture> --out artifacts/rust-backend/<fresh-run>
```

Use the project tool environment and frozen L09 Python reference source, as recorded
in the qualification command receipts. The prepare step requires a fresh folder.
The qualifier builds debug, release and profiling, runs native/property/process
tests, checks the existing ONNX golden tensors and seals sources and executables.

Normal-allocator profiles run **two independent sessions, one thread each**, with
persistent single-threaded ORT sessions and five measured fixed-count searches per
owner after loading and a warm search. Verbose trace-on/off runs are separate.
Component timers remain enabled in both; they are diagnostics, not an uninstrumented
four-second deployment comparison. Separate allocation probes exclude ORT's C
allocator and distinguish typed information construction, key copying, hash/equality
lookup, warm feature encoding and full searches. No speedup or strength claim is
inferred from these component measurements.

| Probe | Root choices | Mean core search, 32 trajectories | Wall time for 5 overlapping pairs / 320 trajectories | Rust allocations per trajectory | Allocated traffic per trajectory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense Canal action | 413 | 201.30 ms | 1,032.68 ms | 25,077 | 1.62 MB |
| Wide Rail action | 246 | 171.31 ms | 885.16 ms | 25,886 | 1.71 MB |
| Staged Canal bonus / era transition | 6 | 152.69 ms | 801.90 ms | 22,567 | 1.89 MB |

Mean core-search time includes root evaluation and completed traversals, excluding
model startup, permitted-root preparation outside `search::run`, final trace
materialization and tree destruction. Wall time includes that outer per-search
work. Allocation traffic includes temporary objects and is **not retained/peak
memory**. There are ten searches per row, not ten different game positions; no
confidence interval or tuned-optimum claim is made. The wide Rail fixture belongs
to the existing held-out corpus; these fixed-parameter diagnostic profiles perform
no tuning and do not constitute L14 held-out quality/strength validation.

The two owners overlap in every profile, keep separate CPU sessions and each
perform their own search/NN work on one thread. Trace-on wall times were
1,077.33 / 898.49 / 849.57 ms; the small ordered sample is only an instrumentation
check. For trace-off work, information/choice construction averages
1.207 / 1.063 / 0.786 ms per trajectory; lookup costs
0.372 / 0.415 / 0.261 ms. Native leaf evaluation, including feature encoding,
averages 3.323 / 2.632 / 2.621 ms per trajectory, with root preparation recorded
separately. Raw component counters are in the profile reports.

**Measured work for L12:** the scalar baseline still copies nested permitted
observations, canonical choice records and node keys. One typed key clone allocates
2,343 / 2,046 / 1,072 objects on these fixtures. Actor information/choice construction
allocates 7,374 / 6,580 / 2,984 objects per isolated call. In contrast, hash/equality
lookup and the warm feature encoder allocate **zero**. Prioritize shared immutable
public facts, key/choice ownership and buffer reuse; compare any optimization to
this sealed scalar baseline and rerun identical traces. These results do not claim
a zero-allocation production search or completion of L12's optimization gate.

## Model and rollout status

The unchanged g890 W512×8 test model remains pinned to checkpoint SHA-256
`ecad3b932a8dd9d3b5d0cc49b5e9bb94a9e98b4536ffdc86a44179d003995cb8`
and ONNX SHA-256
`64e4a8340f4beb7bbc4ee70219b5e461b1185bcde9e25c4bfc78e30936c91ae9`.
The utility-calibration field stays null. The user's explicit local utility-head
acceptance applies; historical L09 mechanical-only/runtime flags are preserved.
L13 still needs the explicit model-specific admission record for live integration.

This work does not change installed gameplay, NN/V4 table selection, four difficulty
allowances, model weights or lab training. The matched benchmark remains **4 seconds**.
**L11 is next:** whole-turn sessions and complete-plan selection.
