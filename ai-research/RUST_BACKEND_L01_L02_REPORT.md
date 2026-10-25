# Rust backend migration: L01 and L02

Run: `20261024-g890-l01-l02`. Date: 24 October 2026 (local time).
Status: **L01 complete for mechanical testing; L02 complete**. The test model's
live/strength admission remains pending utility calibration.

This implements the reference freeze and comparison harness from the
[migration plan](RUST_AI_BACKEND_MIGRATION_PLAN.md). The installed game remains
on patch 1.4.34. Rust integration starts at L03; local deployment and lab work
remain later gates.

## Frozen inputs

| Input | Identity / disposition |
| --- | --- |
| Reference source HEAD | `df41b07625ed190e3332326f3c3118c404e026f7` |
| Reference inventory | 462 source/config/runtime-input files, archived with per-file SHA-256 |
| Reference inventory digest | `8d8fcdeef503913df92cc9bb95a65c07bbfbe9ef1cba46d520f5e6921ceaf0f6` |
| Rules oracle | Four-player `steam-brass-patch-1.1.0`, protocol `0.2.0`, fused C# research host |
| Upstream Rust | `riverside234/brass-birmingham-daiyu-ai`, commit `f8f0246718d6d3fb3f301742d4154811c90d5813` |
| Upstream working changes | 71 source files archived, local binary diff recorded; committed LICENSE retained despite its local deletion |
| Installed fallback | 1.4.34 binaries, installation receipt and stock Assembly-CSharp backup copied and hash checked |
| Original saves | 98 save/history files copied to a recovery archive; original files untouched |
| Local settings | 4 / 8 / 10 / 12 seconds; Hard / 10 seconds remains the default |
| Matched benchmark | **4 seconds for the whole ordinary turn; two independent sessions, one computational thread each** |

The research host and installed Unity game have separate rules identities.
Their live parity is a later gate. Two/three-player, tutorial and online modes
are outside this initial reference corpus.

The temporary model is the user-supplied
[g890 checkpoint](../test_nn/joint7-v11-2-grouped-lr-critic-1b96e228-g890-ecad3b932a8d-6bd7b0105b58/checkpoint.pt).
It has **5,928,616 parameters**, v11.2, hidden width 512 and eight layers,
Features-4 widths 3,128/327, and `seat-descriptors/1.0.0` inputs `[B,4,34]`
with the established unknown-descriptor default. The transferred run-state
snapshot is generation 920; the checkpoint itself is generation 890.

- Checkpoint SHA-256: `ecad3b932a8dd9d3b5d0cc49b5e9bb94a9e98b4536ffdc86a44179d003995cb8`.
- Model ID: `1145fecba3289511df1c518bb71fbfb5b2f6df351777151a5eb999b89cba0761`.
- Reward: `brass-combined-return/2.0.0`; win 1.0, placement 0.1, VP 0.001
  capped at 180, insolvency event cost 0.02 capped at 0.075.
- Bundle hashes, strict checkpoint loading, architecture, finite weights,
  model identity and source compatibility passed. Weights were not changed.
- `utility_calibration` is null and the existing readiness check is false.
  The user authorized this checkpoint for mechanical tests and timing. These
  results do not admit it for live neural play, strength or search teaching.
  The final NN is still training and will need its own pinned evidence.
- The neutral frozen opponent policy uses this same checkpoint. Transferred
  IL lineage receipts are preserved; original lab IL artifacts were not copied
  or independently requalified here. ONNX export/parity belongs to L08/L09.

All identities, source snapshots and recovery paths are in
[baseline.json](../artifacts/rust-backend/20261024-g890-l01-l02/baseline.json).
The upstream pin is also tracked in
[rust-engine.lock.json](../configs/references/rust-engine.lock.json).

## Implemented harness

The [driver](../scripts/qualify-rust-backend.py) supports `freeze`, `capture`,
`qualify`, `traces`, `tests`, `benchmark` and read-only `report` operations.
[Run instructions](../tests/RustBackend/README.md) give the exact local Python
and dependency setup. Generated private states, game saves and binaries stay
in ignored `artifacts/`; the schemas, recipe, implementation and tests are
tracked source deliverables.

The versioned permitted-request schema accepts one actor's observation, legal
choices, own recall and public reward ledger. It rejects unknown nested fields,
trusted checkpoints, other hands, illegal clock settings and duplicate choices.
Trusted debug imports have a separate schema. Changing unavailable hidden
information cannot change the serialized policy request.

The canonical comparison retains all four masked views, semantic legal-choice
order, private ordered cards, staged context, chance probabilities and terminal
results. It excludes the transport game UUID. The C# opaque restoration hash
remains a separate check; Rust need not imitate C# internal RNG/revision fields.
Partially completed Rail deals also retain the ordered cards already picked,
since the remaining pool alone cannot determine the eventual hands.

Mismatch reports identify the first changed field/index and both source
identities. Random tapes record the actual draws, supports, order, probabilities
and selected outcomes. Scripted NN replies bind to exact permitted requests.
Sampled worlds are stored separately as trusted diagnostic data; matching
numeric seeds is not treated as cross-language RNG equivalence.

## Qualification evidence

The sealed corpus contains **546 roots**: 532 from 20 complete games and 14
authored complex roots. Four full games, seeds 731017–731020, provide **104
held-out roots** for later candidate evaluation. They participate in correctness
replay but are excluded from baseline timing and tuning.

Capture coverage includes both eras, staged payment/continuation/bonus/debt,
multi-sale beer decisions, development, Rail resource depletion, empty decks,
dense networks, liquidation, terminal scoring and explicit chance outcomes.
Fourteen independent assertions cover loans, resource depletion/flip, merchant
beer, liquidation and capped reward arithmetic with tied scores. These counts
are an initial corpus, not an exhaustive rules proof.

| Check | Result |
| --- | --- |
| Exact root restoration and canonical comparison | 546 / 546 passed |
| Complete four-player games | 20 / 20 passed; 8,344 replayed transitions |
| Authored staged edge transitions | 5 / 5 passed |
| Independently specified expectations | 14 / 14 passed |
| Boundary, negative, tape, comparison and saved-request tests | 24 passed; no failures, errors or skips |
| Serialized permitted-request identities | 426 / 426 matched the sealed corpus |
| Request-bound real NN replies | 28 verified |
| Explicit sampled worlds, root view/legal parity | 56 / 56 passed |

Receipts: [reference replay](../artifacts/rust-backend/20261024-g890-l01-l02/local/fixtures/qualification.json),
[NN/world traces](../artifacts/rust-backend/20261024-g890-l01-l02/local/fixtures/trace-qualification.json),
[focused tests](../artifacts/rust-backend/20261024-g890-l01-l02/local/negative-tests.json),
and [sealed corpus/coverage](../artifacts/rust-backend/20261024-g890-l01-l02/local/fixtures/corpus.json).

## Benchmark contract

Variant A uses the existing Python Masked ISMCTS and C# reference host with
the actual g890 test weights. Each session owns a private tree, RNG, evaluator
and simulation hosts. NN calls run synchronously on that session's owner;
the evaluator's usual extra inference executor is excluded. Torch intra-op,
inter-op and numerical-library thread limits are one. Owner IDs and thread
settings are checked and recorded. C# processes and transport overhead remain
part of the reference cost.

Eight development roots are selected by era/round stratum and largest legal
set, plus authored multi-sale and Rail roots. Selection is independent of timing.
Each starts at the beginning of a turn. Five deterministically rotated
repetitions give 40 trials. Both sessions receive one shared absolute deadline;
staged completion and the second action receive no fresh allowance. Complete
plans require the reference's two-visit support and retain its prior tie-break.

The evidence separates checkpoint loading, session startup and root preparation
from timed response readiness. It records completed/attempted/interrupted
trajectories, NN work, fallback use, actual worker retirement, staged choices,
CPU and process memory. Zero-completion roots remain in the report. No finalization
reserve is added to the selected neutral reference path; observed overhead is
reported, including overruns.

The protocol and targets are frozen before measurements: fixed-work speedup
2×, real-NN completed-work speedup 1.25×, critical-group median regression at
most 5%, and response-ready engineering targets of 4.05 s observed p95 / 4.15 s
maximum. They remain future candidate targets. Five repetitions per root are
an initial screen and cannot establish tail latency or playing strength.

## Measured baseline A

Hardware/runtime: Intel Core Ultra 5 135U, Windows 11, 14 logical CPUs,
AC power, Balanced power scheme, PyTorch 2.8.0 CPU. The same two session owner
threads served all 40 turns, with one intra-op/inter-op thread each and no
inference executor. No qualification/test job overlapped timing. System-wide
CPU load averaged approximately 17.1% over the measurement interval, including
startup/preparation; a complete background-process inventory and qualified
temperature sensor were unavailable.

| Root | Legal choices | Completed trajectories in five 4 s trials | Response median / maximum |
| --- | ---: | --- | ---: |
| Canal early, `game-731013-root-000` | 399 | 5, 6, 5, 6, 5 | 4.094 / 4.125 s |
| Canal middle, `game-731003-root-004` | 413 | 2, 2, 1, 1, 1 | 4.188 / 4.219 s |
| Canal late, `game-731014-root-008` | 142 | 1, 2, 1, 2, 2 | 4.360 / 4.390 s |
| Rail early, `game-731009-root-014` | 169 | 2, 2, 2, 2, 2 | 4.438 / 4.515 s |
| Rail middle, `game-731014-root-018` | 208 | 1, 0, 0, 1, 2 | 4.563 / 4.875 s |
| Rail late, `game-731010-root-024` | 124 | 2, 2, 2, 2, 2 | 4.610 / 5.141 s |
| Multi-sale, `edge-shared-merchant-beer` | 150 | 3, 3, 3, 3, 3 | 4.171 / 4.219 s |
| Rail resources, `edge-rail-action` | 119 | 1, 1, 0, 0, 2 | 4.234 / 4.297 s |

Across 40 turns, **85 of 163 attempted trajectories completed** and 78 were
interrupted. Four turns completed none. All 40 resolved a legal complete turn;
30 used the reference incumbent/fallback and ten completed without fallback.
The timed sessions processed 1,542 NN positions in 1,542 batches. These are
complete Masked ISMCTS frontier trajectories, not individual rules steps or
upstream PUCT simulations.

| Boundary | Median | Observed p95 | Maximum |
| --- | ---: | ---: | ---: |
| Last search owner retired | 4.016 s | 4.063 s | 4.125 s |
| Final staged turn resolved, including C# validation/application | 4.321 s | 4.735 s | 5.141 s |
| Excluded root restoration/warm-up | 0.962 s | 4.034 s | 4.166 s |

Checkpoint loading took 0.404 s and session/host startup took 9.461 s, before
the warm trials. The additional work after search retirement had a median
of 0.281 s and maximum of 1.141 s. That path includes the diagnostic host's
full snapshots/checkpoints and staged validation/application; it is **not an
installed Unity dispatch or animation measurement**. Future comparisons must
preserve this harness boundary or report a separately versioned boundary.

**The reference fails the frozen 4.05 s p95 / 4.15 s maximum response targets.**
No threshold was lowered. Deadline retirement and complete-turn delivery remain
work for L11–L15; successful L01 measurement does not mean these targets passed.
This run supplies A only. There is no B/C result or measured Rust speedup yet.

Peak aggregate RSS sampled at turn boundaries was 1,105.7 MiB; the sum of
individual process high-water marks was 1,183.0 MiB. Aggregate post-turn RSS
rose from 854.5 MiB after the first turn to 1,081.7 MiB after the last, a
227.2 MiB increase across this mixed-root sequence. This is not a synchronized
peak or a controlled post-retirement leak test. The reference's retained
caches/host memory need attribution; the future 64 MiB growth criterion is
not marked passed by these measurements.

Evidence: [frozen protocol](../artifacts/rust-backend/20261024-g890-l01-l02/local/matched-protocol.json),
[raw trials](../artifacts/rust-backend/20261024-g890-l01-l02/local/baseline-a-trials.jsonl),
[benchmark result](../artifacts/rust-backend/20261024-g890-l01-l02/local/baseline-a.json),
and [derived statistics](../artifacts/rust-backend/20261024-g890-l01-l02/local/baseline-analysis.json).

## Scope and provenance notes

A separate workspace update changed the checkpoint-export helper after the
freeze. The original helper is retained in the hash-checked source archive;
capture, replay and timing do not execute it. The reference game/search/model
sources and original test bundle remain hash guarded. This narrow provenance
difference is recorded in
[source-drift.json](../artifacts/rust-backend/20261024-g890-l01-l02/local/source-drift.json).
The other workspace changes were preserved.

Initial unsealed capture evidence was retained when the chance projection was
extended to include the ordered already-picked cards. The sealed corpus uses
the corrected projection and is replayed in full. A descriptor-label metadata
correction also preserves the original baseline and its hash. Neither changed
weights, rewards or reference source identity.

Native engine/search parity, fixed-work Rust comparisons, cancellation/lifecycle,
native feature encoding, ONNX and full search selection/backup traces remain
L03–L15. The next implementation step is L03. Local acceptance must precede
Part 2 lab migration. No backend was installed and no training job was started.

The machine-readable step ledger is
[local/progress.json](../artifacts/rust-backend/20261024-g890-l01-l02/local/progress.json).
The [final integrity check](../artifacts/rust-backend/20261024-g890-l01-l02/local/final-verification.json)
confirms unchanged installed binaries, test weights/bundle, reference runtime,
frozen archives and recovery backups, plus current passing harness receipts.
