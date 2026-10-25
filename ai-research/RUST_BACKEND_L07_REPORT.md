# L07 — Native simulator adapter and local qualification

Date: **2026-10-25**. Status: **complete; local simulator qualification passed**.
Evidence: [qualification receipt](../artifacts/rust-backend/20261025-l07-qualified-v4/qualification.json)
and [final verification](../artifacts/rust-backend/20261025-l07-qualified-v4/final-verification.json).

## Implemented boundary

`select_backend(backend="rust-simulator")` now returns `NativeSpielGame`.
Its states provide the existing OpenSpiel-facing methods and attributes consumed
by Python `MaskedISMCTS`: explicit setup/chance, current player, semantic legal
actions, apply, permitted views/recall, terminal result/returns, clone/restore and
portable serialization. Variant B is selected explicitly. The default remains
the reference backend.

Rust owns initial order/merchant/card choices, board construction and dealing,
staged transitions, chance, observation memory and every simulated world.
The existing Python search code is unchanged. `from_observation` passes only a
permitted view and recall; Python emits the reference RNG's recorded shuffle
indices, and the qualified native sampler reconstructs the world. Consecutive
identical roots reuse their prepared native template.

Each of the **two independent one-thread sessions** owns up to 256 simulator
handles. Cloning shares an immutable native world; applying a move builds an
independent successor. Restore replaces the destination's world reference.
Monotonic handles cannot alias released states or the other session. Clone and
restore send acknowledgements without re-exporting boards; their Python wrapper
copies cached diagnostic views. Switching between states performs no history
journal replay, state re-import or board serialization.

Portable checkpoints preserve setup prefixes, private state, each seat's recall
and the previous permitted reveal during partial draws/Rail shuffles. They are
explicit trusted diagnostic artifacts. Production policy construction continues
to use the separate permitted-root boundary. Invalid actions, revisions,
checkpoints and capacity failures leave existing native worlds intact.

State close/garbage collection releases native handles, and game close releases
remaining owned objects. Cancelled Python trajectories close their generators
and release pending reservations/states. Worker retirement discards all states.
The native concurrent search scheduler and its cancellation remain L12 work.

## Qualification coverage

Formatting and strict workspace Clippy passed. All **31 native tests** passed
in debug and release; all **five adapter tests** passed in release, debug and
profiling builds. The **27 existing release process regressions** passed on the
identical candidate and were verified/reused as described below. The separate
setup-probability comparison also passed. The selected corpus has **zero
unexplained differences**. `target_rules_qualified` is now true for the supported
four-player standard simulator; this flag does not admit NN/search deployment
or training.

The qualifier materializes the immutable L01 source archive, pinned L02 harness
and separately identified native client overlay. It executes frozen Python
observation/recall/search code and uses sealed L04–L06 C# fixtures. The original
checkpoint, installed assemblies, saves, upstream modules and prior receipts
are checked before and after qualification.

The live search comparison also runs the exact L01/L05 explicit-chance C# host
and dependencies from isolated copies, with the frozen card composition. Its
manifest gets an explicitly recorded lookup/extension adaptation; executable
and dependency bytes remain unchanged. The first qualification was stopped to
close this isolation gap. A second preflight rejected the working directory's
different `r4-host` hash. Runs v3/v4 use the preserved reference host.

Run v3 passed all 27 earlier process regressions and four of the five adapter
tests. Its full-game adapter test had a harness `KeyError: views`: L05's state
stream was incorrectly used as the observation stream. The repaired test joins
each L05 state/command to the hash-verified L02 observation stream and asserts
matching game/command order. No simulator implementation changed for this repair.
Run v4 reuses only the 27 separately sealed successful regressions, requiring
identical native source, release binary, Python client and regression test hashes.
It reruns all five adapter tests in every build. The original failed run and
its error remain intact; it is not reported as an entirely passing suite.

After v4's tests and Criterion completed successfully, its allocation-log
collector selected a brace in a compiler warning instead of the JSON document.
The collector now recognizes the standalone JSON document. Recovery verified
all thirteen completed command/log receipts, all three parity reports and the
unchanged native/client/test identities, then completed the remaining allocation,
adapter-profile and packaging steps. The original failure, script and command
record are preserved in the [completion record](../artifacts/rust-backend/20261025-l07-qualified-v4/completion-resume.json).
No simulator changes or repeated game tests were needed for that collector fix.

| Area | Coverage |
| --- | --- |
| Adapter initialization and complete games | Twenty reference setups, 77 explicit setup choices each, then 8,344 staged/chance transitions through Canal, Rail and settlement; a supplementary frozen-adapter comparison verifies exact setup outcome sets/probabilities and four-seat information at all 1,540 setup boundaries |
| Per-transition comparison | Actor/phase, complete ordered legal set, chance probabilities, four views/recalls/information digests, terminal results and returns |
| Staged import and restore | All 546 accepted roots; clone and portable restore at every root; 86 additional complete-game checkpoints, one every 97 transitions |
| Permitted sampling | All 1,704 frozen root/observer/seed combinations, exact private sampled state and 440 continuation steps |
| Unchanged Python search | Canal action, Rail action and Rail payment roots; four completed trajectories per root/backend; identical deterministic evaluator inputs, root statistics and completed plans |
| Failure/lifetime | Stale/foreign handles, bad revisions/checkpoints, bounded storage, two-owner isolation, close/collection and cancellation with no completed visits or pending reservations |
| Earlier regressions | L03–L06 rules, 1,406 alternate moves, reward/cost ledgers, eight optional-merchant cases with 456 transitions, hidden counterfactuals, transport failure and state/privacy checks |

The scripted search comparison is a deterministic compatibility check, with an
exceptional timeout. It does not run NN weights or measure the later four-second
matched search benchmark. No Python search algorithm or training recipe was
changed to accommodate the adapter.

The supported matrix is four-player standard games and the action, payment,
continuation, bonus, debt, chance and terminal phases represented in the corpus.
Other player counts/configurations and malformed inputs remain explicit
rejections. The six inconsistent original authored roots remain rejected with
their separately captured corrected fixtures; no original evidence was rewritten.

The 546-root coverage is 124 action, 123 continuation, 112 payment, 61 debt,
six bonus, 80 draw-chance, 20 Rail-shuffle and 20 terminal roots: 277 Canal and
269 Rail. Its partitions remain 436 development, 104 held-out and six corrected
authored roots. The twenty original games remain sixteen development and four
held-out games. Full-game action-family counts include Build 275, Network 265,
Develop 418, Sell 12, Loan 286, Scout 250 and Pass 1,108; payments, bonuses,
128 liquidation choices and 2,480 chance outcomes are checked separately.
Rare sale/bonus/network branches also retain the authored and alternative-move
fixtures from L05. The sealed [rules coverage](../artifacts/rust-backend/20261025-l07-qualified-v3/release-rules-parity.json)
and [root coverage](../artifacts/rust-backend/20261025-l07-qualified-v3/release-state-parity.json)
record the exact groups and regression identities for this same candidate.

A7's native NN/encoding cancellation, search arena/cache tests, concurrent native
aggregation and deployment UI/settings checks remain attached to L08–L15.
L07 qualifies the simulator-facing subset; it does not substitute simulator
counts for those later gates.

## Rules and allocation profile

Criterion and `stats_alloc` use an initial root and a dense Rail payment root
selected only from development fixtures. Measurements separate legal generation,
one-time permitted-root preparation, sampling a prepared root, rules application,
application with observation memory, owned state copying, handle clone/restore,
and diagnostic snapshot construction. A separate wall-time profile includes
Python copies, framed IPC and JSON. The latter is intermediate diagnostic cost,
not the final native search path. Allocated bytes are not claimed as bytes copied.

Measurements used an Intel Core Ultra 5 135U on Windows x64, Rust 1.99.1,
portable `target-cpu=x86-64`, opt-level 3, ThinLTO and one codegen unit.
Criterion used twenty samples per operation, 0.2-second warm-up and one-second
measurement; it used the normal allocator. The independent `stats_alloc` probe
used one warm-up followed by 100 operations. These microbenchmarks execute one
operation at a time; they are not a two-session search-throughput result.
The worker retains two independent one-thread owners.

Fixtures were `game-731001-root-000` (initial action, no history) and
`game-731009-root-026` (dense Rail payment, 266 history entries). Both came from
the development partition. A dense board can have few choices at a payment
stage; its cheap legal-generation result does not represent a dense action root.

| Native operation | Initial mean, µs | Dense Rail payment mean, µs |
| --- | ---: | ---: |
| Complete staged legal set | 237.41 | 0.416 |
| Prepare permitted root once | 129.01 | 245.46 |
| Sample prepared root | 5.59 | 6.50 |
| Validated rules application | 142.39 | 7.19 |
| Application with observation/recall update | 173.12 | 61.16 |
| Owned state clone/drop | 2.44 | 3.39 |
| Diagnostic snapshot, all four seats | 309.44 | 179.73 |
| Handle clone + restore + release | 0.464 | 0.494 |

These are Criterion mean point estimates. Raw samples and 95% confidence
intervals are retained under `criterion/`; the handle-cycle mean intervals are
0.455–0.473 µs and 0.483–0.508 µs respectively. Small differences are not a
compiler or algorithm speedup claim.

| Operation | Initial allocations + reallocations | Initial allocated bytes | Dense allocations + reallocations | Dense allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| Legal set | 2,922 + 10 | 254,784 | 9 + 0 | 618 |
| Prepare root | 2,225 + 54 | 100,272 | 4,109 + 78 | 288,230 |
| Sample root | 143 + 0 | 8,095 | 130 + 1 | 27,977 |
| Apply rules | 1,679 + 11 | 172,658 | 112 + 1 | 49,080 |
| Apply with recall | 2,420 + 12 | 206,243 | 1,370 + 2 | 127,865 |
| Owned clone | 65 + 0 | 4,441 | 79 + 0 | 26,361 |
| Diagnostic snapshot | 5,408 + 10 | 358,398 | 4,497 + 0 | 288,038 |
| Handle clone/restore/release | 0 + 0 | 0 | 0 + 0 | 0 |

Counts and bytes are per operation, including temporary allocations. The native
state's shallow size is **1,760 bytes**; deep bytes copied are not instrumented
and remain explicitly unknown. Allocation traffic must not be substituted for
that quantity. The warm handle-cycle result does not imply that growing the
bounded handle registry never allocates.

| Diagnostic adapter operation | Initial median, ms | Dense median, ms |
| --- | ---: | ---: |
| Sample cached root and release | 2.67 | 3.00 |
| Clone, restore and release | 13.38 | 18.77 |
| Apply and restore | 9.72 | 12.09 |
| Fetch snapshot | 2.55 | 3.26 |

These thirty-sample wall times include Python snapshot copies and synchronous
IPC/JSON. They do not isolate each component's share. The contrast with the
native handle measurements identifies the diagnostic boundary as an important
remaining cost; no end-to-end NN search improvement is established here.

**Restoration decision:** retain owned transitions plus shared immutable native
handles. Owned state cloning is about 1.4% and 5.5% of application with recall
in these two probes, so a mutation journal is not justified for L07. It is
approximately 47% of the bare dense-payment rules microbenchmark; revisit that
cost once native feature/search consumers remove the larger surrounding work.
No apply/unapply journal was introduced or claimed faster. The chosen path
passed clone, checkpoint, cancellation and restoration checks in all builds.

Remaining native costs include semantic legal-choice construction, one-time
root validation, observation materialization and diagnostic copying. Repeated
root sampling and live-handle restoration avoid state parsing and history
replay. The zero-allocation production apply/encoding targets are **not yet
achieved** by this diagnostic interface. L08/L10 must measure typed consumers,
reusable buffers and on-demand observations before treating this as the final
native hot path; retain this adapter as the differential reference.

## Reproduction and remaining work

```powershell
python scripts/qualify-rust-simulation.py --out artifacts/rust-backend/my-new-l07-run
```

The output directory must be new. Qualification uses the offline pinned native
toolchain, saves raw per-build logs and parity reports, and archives implementation
and executables. Native tests run in debug/release; adapter tests run in all three
builds; the full earlier process suite also runs in release.

L08 is next: exact features and the admitted checkpoint interface. Native NN
inference/search, Unity installation, live play and lab training remain later
milestones. The g890 checkpoint remains temporary; the later matched benchmark
uses **4 seconds per whole turn**, two independent one-thread sessions, and the
deployment settings remain **4/8/10/12s**. The new plan requirement for V4-only or
NN-only deployment tables remains assigned to L13/L15.

See the [simulator contract](../configs/contracts/rust-backend/simulation-v1.md)
and [migration plan](RUST_AI_BACKEND_MIGRATION_PLAN.md).
