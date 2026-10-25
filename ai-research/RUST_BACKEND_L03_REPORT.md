# L03: isolated Rust core and worker

Date: 2026-10-24 (local). **Status: L03 complete.**

Evidence: [qualification receipt](../artifacts/rust-backend/20261024-l03-verified/qualification.json),
[measurement summary](../artifacts/rust-backend/20261024-l03-verified/measurement-summary.json),
[command logs](../artifacts/rust-backend/20261024-l03-verified/commands.json).

The migration now reuses established crates for its infrastructure and the
existing upstream game rules. The local diagnostic worker builds and runs
natively on Windows x64. It owns two independent sessions with one computation
thread each. This milestone does not enable Rust in Unity or lab training.

## Implemented

- `native/vendor/brass-upstream-rules/`: 40 original Rust modules, byte-identical
  to the frozen L01 archive at upstream commit
  `f8f0246718d6d3fb3f301742d4154811c90d5813`. The extraction retains hashes,
  the committed AGPL-3.0 license and the prior local patch. Web/database/UI,
  training and upstream PUCT are excluded.
- `native/brass-ai-core/`: typed seats, phases, context, choices and errors;
  strict Serde request types; bounded framing; diagnostic state ownership;
  source/dependency/compiler identity; property tests and measurement workloads.
- `native/brass-ai-worker/`: two dedicated state-owner threads using bounded
  Crossbeam channels, a versioned handshake, independent reset/clone operations,
  panic retirement and shutdown/EOF cleanup. The coordinator handles transport.
- `src/brass_ai/native/`: explicit backend selection that defaults to the
  existing `BrassSpielGame`, plus an opt-in framed worker adapter with bounded
  read/write waits and retirement after a broken transport.
- `scripts/qualify-rust-scaffold.py`: repeatable offline qualification with raw
  logs, executable/source hashes, PE architecture checks, dependency inventories
  and before/after preservation checks.

The plan now identifies crate reuse at each later milestone, including tree
handles, RNG primitives, exact tensor/inference bindings and optional Python
bindings. Brass-specific semantics and our Masked ISMCTS remain custom adapters;
a generic MCTS implementation cannot silently change the matched experiment.
See the [workspace README](../native/README.md) for pinned crates and commands.

## Validation

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Pass |
| Workspace Clippy, all targets, warnings denied | Pass |
| New Rust core/worker tests | 10 pass in debug; 10 pass in release |
| Preserved upstream tests | 31 pass in debug; 31 pass in release |
| Optional upstream `benchmark-deadlines` feature | 31 pass in debug; disabled in the worker |
| Real-process Python checks | 11 pass for each of debug, release and profiling |
| Current source/lock identity versus each executable's handshake | Pass |
| Windows PE architecture | AMD64 for each build |
| Frozen reference, original saves, installed game and test NN | Hash checks pass before and after qualification |

The five Rust property tests cover framed payload round trips and truncation,
seat and semantic-choice admission, deterministic initial states, and private
deal/RNG/bitset ownership. Process checks cover incompatible contracts, unknown
fields/capabilities, invalid player/session values, isolated resets, state
preservation after rejected operations, malformed/truncated/oversized input,
unexpected process death, response identity mismatch, full-pipe write timeout,
read timeout and cleanup. The reference host still creates a valid initial
chance state through the default selector.

The six existing unused-import/dead-code warnings in the frozen upstream source
remain visible in logs. The vendor crate is outside the new workspace's style
checks so those historical bytes are preserved. Its tests run separately; new
core and worker code pass the strict lint gate.

## Build identity and toolchain

Rust **1.99.1**, target **x86_64-pc-windows-gnu**, portable CPU baseline
**x86-64**. Release uses optimization 3, ThinLTO, one codegen unit, checked
overflow and unwind-based owner containment. The profiling build keeps symbols.
All direct crate versions and transitive checksums are locked.

The project-local Rust package needed an additional Windows `dlltool` helper.
The checksum-verified LLVM-MinGW **20261020** package supplies that tool and the
benchmark dependency's C compiler; Rust's bundled GCC linker retains the matching
runtime. Tools live under `artifacts/native-toolchain/`. The qualifier configures
only its child-process environment. No system PATH, installed game DLL or live
backend selection changed.

The initial qualification attempt also exposed a Windows path-ordering mismatch
in the Python source-hash verifier. It was corrected to use the same
case-sensitive ordering as Rust. The executable and independently recomputed
source hashes now agree. The failed attempt is preserved under
`artifacts/rust-backend/20261024-l03/`; completed evidence belongs in
`artifacts/rust-backend/20261024-l03-verified/`.

Native source fingerprint:
`5f562533ff5766c750e7b771dc37b81015bcc359a95939b62689d553b32069ae`.
Workspace Cargo lock SHA-256:
`f38532b3b0ec1503f05a033f679db12db97c7ad93d6acddc7b1f247c51c6db14`.
The release worker is **963,568 bytes**, SHA-256
`16866d9e159d1d04d73bd34f064fd698d06a0ca3277a936500c37f29f192e024`.
The receipt distinguishes the profiling build by its command and debug setting;
Cargo reports its inherited `PROFILE` value as `release`.

## Scaffold measurements

Criterion 0.8.2 used 20 samples per workload, 200 ms warmup and approximately
one second of measurement on this Windows 11 machine. The table reports its
slope estimate and 95% interval. Timing uses the optimized bench profile with
symbols and the ordinary allocator; the allocation probe runs separately.

| Workload | Estimate | 95% interval | Measured operations, excluding warmup |
| --- | --- | --- | ---: |
| Seeded upstream initial state, including drop | 15.05 µs | 14.03–15.83 µs | 66,990 initializations |
| Initial-state deep clone and drop | 4.56 µs | 4.39–4.78 µs | 204,750 clones |
| Initial action-type enumeration | 10.61 µs | 10.34–10.90 µs | 86,940 calls |
| Typed context-copy loop | 1.26 ns | 1.19–1.35 ns | 854,743,470 copies |

The copy loop includes `black_box` overhead and has outliers; it is a harness
sanity measurement. These are initial-state workloads, not complex-state rules
restoration or a comparison of NN/search speed with C#.

The warm allocation probe performed exactly **100 clones/drops**:
**8,700 allocations**, **8,700 deallocations**, **999,300 bytes allocated and
released**, and **zero reallocations**. That is 87 allocations and 9,993 bytes
per clone. The inline state wrapper occupies **2,128 bytes**; heap allocation
traffic is not a measurement of total bytes copied. This identifies a concrete
later optimization target while preserving upstream behavior for now. It does
not establish full-game or search lifecycle memory acceptance.

Raw samples/estimates are preserved under `criterion/`; the separate
`allocation.json` includes its build identity and exact counters. The runtime
dependency tree confirms that the worker does not link the upstream web,
database, PUCT or training stacks, or the benchmark allocation instrumentation.

Final preservation checks cover all **462 frozen reference files**, **40 imported
modules**, the **g890 checkpoint and pinned metadata**, **three installed DLLs**,
the recovery archives and **98 original save files**. The previously documented
change to `scripts/export-test-nn.py` remains separate; its freeze-time bytes are
preserved in the verified L01 archive and were not executed by L03.

## Boundary for subsequent work

The handshake reports `target_rules_qualified: false` and exposes only diagnostic
capabilities. C# state import and neural search return `unavailable_capability`.
The worker's rules identity is `daiyu-upstream-diagnostic/0.1.0`. Its state wrapper
currently supports seeded four-player initial positions; complete state import,
restoration and staged rules parity begin at **L04**.

Parallel neural-search scheduling, cancellation under search load, exact
Features-4, ONNX inference, full-game C# parity and live deployment remain later
gates. No NN was executed for L03. The supplied g890 checkpoint remains the
temporary test model, with its existing calibration gate intact. The matched
complex-state search test remains **4 seconds per complete turn**, with **two
independent sessions, one thread each**. The four local settings remain
**4s / 8s / 10s / 12s**. Local qualification still precedes lab migration.
