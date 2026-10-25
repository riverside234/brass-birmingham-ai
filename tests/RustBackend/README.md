# Local Rust migration reference harness

This implements L01 and L02 of the [migration plan](../../ai-research/RUST_AI_BACKEND_MIGRATION_PLAN.md).
It freezes and exercises the existing C#/Python oracle. It does not install a
Rust backend, modify the game, train a model or contact the lab.

The first qualified run is documented in the
[L01/L02 report](../../ai-research/RUST_BACKEND_L01_L02_REPORT.md): 546 roots,
20 replayed games, 24 focused tests and forty 4-second reference trials.
Its measured latency and memory limitations remain visible in that report.

The supplied generation-890 checkpoint in `test_nn/` is a **temporary test NN**.
Its hashes, v11.2 architecture and strict source compatibility pass. Its
`utility_calibration` is null. Mechanical tests and timings are allowed; this
does not qualify it for live neural search, search teaching or strength claims.
The run-state snapshot is generation 920; the selected checkpoint is still g890.

The matched comparison is **4 seconds for the whole ordinary turn**, shared by
**two independent sessions, each with one computational thread**. Each reference
session calls the original evaluator's encoding/forward/response routine inline;
the normal evaluator's extra inference executor is excluded. Torch intra-op and
inter-op are both one. C# host processes and their transport overhead remain
visible in variant A. There is no shared inference batching or extra NN worker.
The local game continues to offer 4/8/10/12 seconds, default Hard/10 seconds.

## Run

Use Python 3.12 with the existing repository dependencies. This Windows checkout
already has them under `artifacts/`; avoid changing the installed game runtime.

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\artifacts\r2-pythondeps;$PWD\artifacts\r4-pythondeps;$PWD\artifacts\r0-pythonlibs;$PWD\artifacts\r0-validation-deps"
$python = 'C:\Users\BobXu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$run = 'artifacts/rust-backend/20261024-g890-l01-l02'

# Freeze into a NEW run directory. Reads upstream and copies original saves;
# it never writes to the upstream checkout or original save directory.
& $python scripts/qualify-rust-backend.py freeze --run $run `
  --checkpoint test_nn/joint7-v11-2-grouped-lr-critic-1b96e228-g890-ecad3b932a8d-6bd7b0105b58/checkpoint.pt `
  --upstream C:\Users\BobXu\Desktop\brass `
  --save-root 'C:\Users\BobXu\AppData\LocalLow\Cublo\Brass Birmingham'

& $python scripts/qualify-rust-backend.py capture --run $run
& $python scripts/qualify-rust-backend.py qualify --run $run
& $python scripts/qualify-rust-backend.py traces --run $run
& $python scripts/qualify-rust-backend.py tests --run $run
& $python scripts/qualify-rust-backend.py benchmark --run $run
& $python scripts/qualify-rust-backend.py report --run $run
```

Freeze, sealed capture and benchmark outputs refuse overwriting. Capture can
recover completed games from an interrupted, unsealed capture. `--games N` and
`qualify --roots-only` are diagnostics and cannot pass L02. A new model or source
requires a new evidence run and a reviewed reference lock, not edited hashes.

## Evidence and boundaries

- `baseline.json`: exact source/model/runtime/config identities, selected reward,
  actual readiness, fallback/save recovery inventory, and predeclared targets.
  Reference source and upstream source/patch snapshots are kept locally.
- `local/fixtures/corpus.json`: reproducible recipe, all root/game file hashes,
  actual coverage and development/held-out assignment. Seeds 731017–731020 are
  held out from baseline timing and tuning; correctness replay still covers them.
- `local/fixtures/roots/`: trusted hidden checkpoints, ordered cards, staged
  state, all four masked views, recall and causal insolvency ledger.
- `local/fixtures/permitted/`: detached actor-only requests validated against the
  existing strict R1 observation/choice schema and the new request envelope.
  Unknown nested fields are rejected; only the observing seat's hand/recall is
  admitted. Trusted fixtures have a different schema and cannot be requests.
- `local/fixtures/games/`: twenty full four-player games, every staged choice and
  recorded random draw, exact reference transitions, terminal scores, official
  ties and the selected reward's terminal and causal-cost-adjusted returns.
- `scripted-nn.json.gz`: exact request-bound frozen NN replies for future trace
  tests. `trusted-sampled-worlds.json.gz` separately stores explicit sampled-world
  outcomes; equal seeds across languages are never presented as RNG equivalence.
- `qualification.json`: every root restored and every game replayed. A mismatch
  writes `first-difference.json` with source identities and the first changed
  field/index. Semantic legal-choice ordering is preserved.
- `negative-tests.json`/`.log`: deliberate state/legal/view/return mutations,
  hidden-field injection, changing unavailable hidden information, clock/thread
  validation, random-tape order/outcome changes and scripted-response binding.
- `matched-protocol.json` is written before measurement. `baseline-a-trials.jsonl`
  contains eight ordinary-turn roots (six era/round strata plus multi-sale and
  Rail resource cases) × five repetitions with the real test model.
  It retains zero-completion roots, fallback use, preparation, session CPU time,
  retirement and response-ready latency. Timing cannot establish playing strength.

The semantic projection excludes only the host transport UUID, and separates the
opaque C# restoration hash from cross-language equality. Explicit chance keeps
the already-picked card order as well as probabilities/remaining count: two Rail
deals with the same remaining pool can still deliver different hands. C# internal
RNG/revision fields inside its opaque hash are reference restore checks, not Rust
representation requirements. Full recorded transitions test future consequences.

The initial corpus is four-player full-game research rules. Two/three-player,
tutorial and online modes are unsupported migration inputs. The installed game
has its own rules/release identity and is qualified against Rust later. Arena,
cache, cancellation, optimized builds, native features/ONNX and full search
selection/backup traces remain explicit L03–L15 gates.

Generated evidence and original saves stay in ignored `artifacts/`. The tracked
fixture recipe, schemas and tests contain no proprietary assemblies or user saves.
