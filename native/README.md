# Native Brass backend: L03–L07

This workspace isolates the existing upstream rules and provides an opt-in
diagnostic worker. It is the foundation for the local-first migration in
[`RUST_AI_BACKEND_MIGRATION_PLAN.md`](../ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md).
The existing C#/Python backend remains the default. L04 adds complete trusted
state import/export and private save/restore. L05 adds staged legal choices,
transitions and explicit chance. L06 adds permitted views, own-hand recall and
detached world sampling. L07 adds an explicit OpenSpiel-compatible simulator
adapter with native setup, bounded handles and clone/restore. Features,
inference and our native Masked ISMCTS follow in L08 onward.
Nothing here installs a Unity patch or starts lab training.

## Reused code and crates

`vendor/brass-upstream-rules` contains 40 byte-identical modules extracted from
the frozen L01 archive. `UPSTREAM.json`, `upstream-local.patch` and `LICENSE`
retain origin, local changes, checksums and the committed AGPL-3.0 text.
The original web server, database, embedded UI, training code and PUCT are
excluded. The isolation manifest and module wrappers are new local files.

| Responsibility | Crate |
| --- | --- |
| Rules, board, cards, resources, validation | Local `brass-upstream-rules` |
| Existing rule data and RNG | `fixedbitset` 0.5.7, `once_cell` 1.21.3, `rand` 0.8.5 |
| Typed wire messages | `serde` 1.0.228, `serde_json` 1.0.149 |
| Typed seats and errors | `num_enum` 0.7.3, `thiserror` 2.0.21 |
| Bounded queues | `crossbeam-channel` 0.5.17 |
| Build provenance | `sha2` 0.10.9, `walkdir` 2.5.0 (build only) |
| Permitted identity and reproducible sampling | `sha2` 0.10.9, `rand` 0.8.5, `rand_chacha` 0.3.1 |
| Tests and measurements | `proptest` 1.6.0, `criterion` 0.8.2, `stats_alloc` 0.1.10 (development only) |

Both Cargo lockfiles are tracked. The small fixed-length framing and owner
threads use Rust's standard library. Additional crates are introduced when
their milestone has a real consumer; the plan identifies `slotmap`, `ort`,
`smallvec` and optional `pyo3` for subsequent work. L06 promotes the already
locked `rand`/`rand_chacha` packages to direct consumers without upgrading them.

## Build and qualification

Native target: `x86_64-pc-windows-gnu`, Rust **1.99.1**, portable `x86-64` CPU
baseline. Release uses optimization 3, ThinLTO, one codegen unit, overflow checks
and unwinding for the owner failure boundary. `profiling` adds symbols. Cargo's
two build jobs concern compilation, not the game's session/thread setting.

The project-local tools are under `artifacts/native-toolchain/`. Rust's bundled
GCC linker supplies its matching runtime; LLVM-MinGW supplies `dlltool` and the
benchmark dependency's C compiler. The minimal Rust package alone lacked a
working assembler for `dlltool`. No global PATH or system installation changes
are required. Qualification sets PATH and compiler variables only for its child
processes.

Tool provenance:

- [Official rustup installer](https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe),
  SHA-256 `00f139da614dced87d3c3d4121c43eebdac3c1bf8a5ee86b9d80dfd71c8d70fd`.
  Installed with project-local `RUSTUP_HOME`/`CARGO_HOME`, `--no-modify-path`,
  minimal profile, GNU host, toolchain `1.99.1`, and `rustfmt,clippy`.
- [LLVM-MinGW 20261020 release](https://github.com/mstorsjo/llvm-mingw/releases/tag/20261020),
  `llvm-mingw-20261020-msvcrt-x86_64.zip`, SHA-256
  `52a3e3f584a74c3cde8ceb0c6960549683178e03b36818cb02b7ed109a68ff67`.
  Extracted under the same tool directory, with its notices preserved.
- Direct versions are pinned; registry archives are checked by Cargo against
  `Cargo.lock`. Dependencies were fetched once. Qualification uses
  `--locked --offline`.

From the repository root, use the project's Python environment:

```powershell
python scripts/qualify-rust-scaffold.py --out artifacts/rust-backend/my-new-l03-run
```

For the current L04 implementation, first materialize diagnostic fixtures and
then use its dedicated qualifier (choose fresh evidence directories):

```powershell
python scripts/capture-rust-states.py --out artifacts/rust-backend/my-l04/raw
python scripts/complete-rust-state-capture.py --raw artifacts/rust-backend/my-l04/raw --out artifacts/rust-backend/my-l04/states
python scripts/capture-rust-state-regressions.py --capture artifacts/rust-backend/my-l04/states --out artifacts/rust-backend/my-l04/regressions
python scripts/qualify-rust-state.py --capture artifacts/rust-backend/my-l04/states --regressions artifacts/rust-backend/my-l04/regressions/regressions.json --out artifacts/rust-backend/my-l04/qualification
```

The frozen C# runtime is referenced by separate diagnostic executables; it is
never rebuilt or modified. Raw snapshot projections must match all frozen L02
roots before import qualification. `prepare-rust-state-catalog.py` regenerates
the immutable definitions from the captured initial state when deliberately
updating the catalog; it is not part of ordinary runtime state loading.
Six inconsistent authored L02 roots are retained as rejection tests. Their
corrected variants are generated by the frozen C# runtime from separately
corrected setups. The original corpus, checkpoints and projections stay intact.

For L05, capture full transitions from the same frozen runtime, then qualify
all native profiles. Each output directory must be new:

```powershell
python scripts/capture-rust-rules.py --out artifacts/rust-backend/my-l05/capture
python scripts/capture-rust-rule-branches.py --capture artifacts/rust-backend/my-l05/capture --out artifacts/rust-backend/my-l05/branches
python scripts/qualify-rust-rules.py --capture artifacts/rust-backend/my-l05/capture --branches artifacts/rust-backend/my-l05/branches --out artifacts/rust-backend/my-l05/qualification
```

The L05 capture reuses the hash-verified L04 roots and records complete private
state after every original L02 full-game command. It first requires each C#
projection to match the frozen L02 projection, including all four masked views.
Additional branch captures cover every alternative at authored roots and the
initial root, plus one choice per kind at selected dense development roots.
The qualifier checks complete state/choice/chance parity and existing reward
returns, while preserving previous evidence and the installed game.
It materializes the 462 reference source files from the hash-verified L01
archive, adds the separately pinned L02 comparison harness, and adds the current
native Python client as an explicit candidate overlay.
Python checks execute that isolated source tree, with import identities recorded.
Concurrent training edits in the shared checkout are recorded as drift and left
intact; they cannot silently change the reference used by these tests. New C#
captures still require the capture script's original source-preservation gate.

L06 reuses the sealed L05 exporter and original L02 views, executes the frozen
Python sampler with recorded shuffle indices, and compares all four sampled
observations plus continuations. Its capture and qualifier both use the isolated
reference archive so unrelated training changes cannot affect the oracle:

```powershell
python scripts/capture-rust-privacy.py --out artifacts/rust-backend/my-l06/capture
python scripts/capture-rust-privacy-edges.py --out artifacts/rust-backend/my-l06/edges
python scripts/qualify-rust-privacy.py --capture artifacts/rust-backend/my-l06/capture --edges artifacts/rust-backend/my-l06/edges --out artifacts/rust-backend/my-l06/qualification
```

The existing sealed L06 captures can also be reused by omitting `--capture` and `--edges` from
the qualification command. That qualifier runs the L03–L06 native/process
regressions in debug/release/profiling and records their exact source identities;
L06 adds no search or NN benchmark. Its contract is
[`permitted-observation-v1.md`](../configs/contracts/rust-backend/permitted-observation-v1.md).
`prepare_belief(session, view, recall)` accepts a single permitted view,
`sample_belief(session, random)` creates a trusted diagnostic sampled world, and
`observation(session, seat)` returns only that seat's information. Save/restore
includes recall and the prior reveal during chance. Sampling's prepared public
projection is immutable and private mutable worlds remain owner-local.
The supplementary edge capture preserves empty merchants with omitted `types`
fields, including a complete two-era game. Empty arrays remain distinct valid
wire identities, and explicit null is rejected.

L07 qualifies the usable diagnostic simulator, including full games through
the actual adapter and fixed-work comparisons through unchanged Python search:

```powershell
python scripts/qualify-rust-simulation.py --out artifacts/rust-backend/my-l07/qualification
```

It uses the isolated frozen Python reference and existing sealed captures.
Native tests run in debug/release; adapter tests run in debug/release/profiling;
the full L03–L06 process regression suite also runs in release. Criterion and
`stats_alloc` measure simple/dense native operations, while a separate adapter
profile includes transport and Python overhead. See the
[simulator contract](../configs/contracts/rust-backend/simulation-v1.md).

The script configures the local toolchain and Python reference dependencies,
checks preservation, runs formatting/Clippy/native/upstream tests, builds debug,
release and profiling executables, runs real-process Python checks, and records
Criterion and allocator measurements. Completed evidence cannot be overwritten.
It requires the preinstalled tools and populated Cargo cache; it never downloads
packages automatically. See `tool_environment()` in the script for direct Cargo
commands using the same environment. Run Cargo from `native/` so the workspace
configuration is applied.

Executables are in
`artifacts/rust-native-target/x86_64-pc-windows-gnu/{debug,release,profiling}/`.
They are console-protocol executables, not interactive applications.

## Protocol and ownership

Each frame is a four-byte little-endian byte length followed by UTF-8 JSON.
The maximum is 1 MiB; zero, oversized and truncated frames fail before parsing.
All requests carry `protocol`, `request_id` and a typed `request` object.
Unknown fields and operations are rejected. stdout contains only frames;
diagnostics go to stderr. An invalid JSON frame can be rejected while retaining
the next frame boundary; corrupt lengths terminate the process.

The required first message is:

```json
{
  "protocol": "brass-native-control/0.1.0",
  "request_id": "1",
  "request": {
    "op": "hello",
    "sessions": 2,
    "threads_per_session": 1,
    "required_capabilities": ["diagnostic_new", "diagnostic_snapshot", "diagnostic_clone_check"]
  }
}
```

The response identifies the source and Cargo lock hashes, compiler, target,
profile options, linker, upstream commit and exact capabilities. It explicitly
reports `target_rules_qualified: true` for the qualified four-player standard
simulator contract. This does not admit native NN search, unsupported game
variants, Unity deployment or training collection.

Two dedicated Rust owner threads each hold their private diagnostic/semantic
state, one bounded semantic save slot, and a request queue of capacity one.
The two diagnostic capabilities have separate state slots. The stdin coordinator does no simulations.
Diagnostic requests are synchronous; parallel search scheduling and cancellation
are subsequent milestones. Resetting one session leaves the other intact. State
clones own their mutable vectors, bitsets, deck and RNG. Immutable upstream
definitions may be shared. A panic retires that owner; shutdown/EOF joins owners.

Available operations:

- `diagnostic_new`: `session` 0 or 1, `players` exactly 4, `seed` a `u64`.
- `diagnostic_snapshot`: public initial-state summary for one session.
- `diagnostic_clone_check`: mutate a private fork and report isolation.
- `state_import`, `state_export`, `state_save`, `state_restore`,
  `state_clone_check`: trusted complete semantic states; see the
  [state contract](../configs/contracts/rust-backend/semantic-state-v1.md).
- `rules_snapshot`, `rules_apply`, `rules_chance`: ordered legal choices,
  transactional staged transitions and explicit card outcomes; see the
  [staged rules contract](../configs/contracts/rust-backend/staged-rules-v1.md).
- `simulation`: explicit setup, detached sampling, bounded state handles,
  transactional steps, clone/restore and trusted portable checkpoints.
- `shutdown`: join owners, acknowledge, exit.

`import_reference` (the old replay-journal placeholder) and `search` return
`unavailable_capability`. Use `state_import` for the versioned semantic DTO.
Invalid sessions,
player counts or messages do not replace a valid state. The diagnostic rules
identity is `daiyu-upstream-diagnostic/0.1.0`, not the C# rules identity.

The Python adapter uses one transport-only I/O thread to bound pipe writes and
reads. A timeout, EOF, invalid length or mismatched response identity kills and
retires the connection. It cannot switch an active game to Rust automatically.

```python
from brass_ai.native import select_backend

game = select_backend(root=repository_root)  # existing BrassSpielGame
game.close()

with select_backend(backend="rust-diagnostic", root=repository_root) as worker:
    first = worker.new_game(0, 890)
    second = worker.new_game(1, 891)
    assert first["owner"] != second["owner"]
    assert worker.clone_check(0)["clone_isolated"]

with select_backend(backend="rust-simulator", root=repository_root) as game:
    with game.new_initial_state() as state:
        assert state.is_chance_node()
        # Feed the normal explicit-chance setup choices, then use the same
        # state API consumed by the existing Python MaskedISMCTS.
        with state.clone() as fork:
            fork.apply_action(fork.legal_actions()[0])
            fork.restore(state)
```

## Measurements and remaining gates

Criterion measures seeded upstream initialization, initial-state deep clone/drop,
initial action-type enumeration and a typed context-copy loop. It writes raw
samples and estimates. A separate `stats_alloc` executable measures allocator
traffic over 100 warm clones; it does not change the normal worker allocator.
Allocated bytes and shallow struct size are not claimed to be deep bytes copied.
The L04 `state` benchmark additionally measures semantic import, export, clone
and restore on initial and complex Rail-payment fixtures. Its allocator probe
measures owned state clones separately from transport and JSON serialization.
The L05 `rules` benchmark measures legal-set generation and validated staged
application on an initial state and a dense Rail payment state selected only
from development fixtures. Validated application includes legal validation,
the transaction's owned clone, mutation, history and index updates. Input setup
clones, JSON transport and neural evaluation are outside those timings.

These are scaffold measurements, not matched neural-search speedups. No NN runs
here. g890 remains the temporary migration test model. The later matched test
uses **4 seconds per complete turn**, with **two independent sessions, one thread
each**, and complex-state fixtures. Local 4/8/10/12-second settings and the
local-before-lab promotion order remain in the migration plan.
