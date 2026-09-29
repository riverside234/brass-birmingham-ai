# R4 / R5 exit checks

Local verification: 29 September 2026, Windows, Python 3.12, PyTorch 2.8.0 CPU,
OpenSpiel 2.0.2 and the preserved rule-patch 1.1.0 runtime. This is the additional
experiment delivery from item 6 of the [implementation plan](../ai-research/IMPLEMENTATION_PLAN.md).
Commands and promotion conditions are in [the lab guide](R4_R5_SEARCH_LAB.md).

## Implementation boundary

- R4 is a real OpenSpiel adapter and masked ISMCTS experiment, with an isolated
  explicit-chance worker. It is not a two-player toy fixture or a deterministic
  wrapper that skips hidden card draws.
- S3 is an independent complete-turn root action/observation tree with one/two
  future-root-turn horizons, four-seat values and fresh finalist confirmation.
- Comparisons, optional standalone S3 teacher updates and promotion records use
  separate outputs. They never resume or rewrite current Step 5 training.
- R0's immutable reference pins, R1's host, R2 interfaces, current R3 v3 bank,
  Step 5 source/configurations, distributed neural weights and installed game
  DLLs remain unchanged by this delivery.

Step 5's implementation identity remains
`688ce053b57aea695d1b876886629584a16eed499f6ea1f8705e93b60e9905aa`.
The frozen bank remains `bank-af17b1aadbe7f9b21fa7`.
The new authored search worker's SHA-256 is
`3b165f8e407ac34ec0df3e80df4b2e1bc547b66095871bcc984eec02aa7e1734`.

## Executed checks

The [machine-readable correctness record](R4_R5_EXIT_CHECKS.json) passes with
implementation identity
`1cbbb96029f2597394b2a4a8d042e5ff78a2312867d369fb6c033ff1ab29ef66`.

| Check | Observed result |
| --- | --- |
| Focused unit tests | 10 passed: stable action IDs, four-seat values, private history keys, future-turn settings, complete paired rotations and promotion rejection conditions |
| Matched full R1/R4 game | 303 decisions; all legal choices matched and all 1,212 player-view comparisons passed |
| Explicit chance | 4 order selections, 9 merchant selections, 64 initial card selections, 60 replenishment draws and 64 Rail shuffle selections; normalized probabilities and matched terminal returns |
| Restoration and private delivery | 7 independent decision clones, pending-draw and pending-shuffle clone/serialization checks, and 91 checks that intermediate private chance events reveal no card |
| Counterfactual privacy | Both planners passed for all four seats: changing only inaccessible live state preserves allowed inputs, selected outputs and search statistics at fixed work |
| Real opponent replies | R4 privacy tests exercised 302 masked opponent choices, with zero optimized opponent nodes; S3 exercised 1,372 reply choices and eight completed fresh confirmations |
| Additional S3 depth | A two-future-turn search completed fresh confirmation and six leaves at depth 2; an already-expired request was rejected |

The full correctness run took approximately 460 seconds on this local machine.
Its fixed-work privacy cases use generous deadlines to test deterministic
information access. They are not laptop playing-speed measurements.

Regression checks: 23 Step 5 tests, 24 R2 tests and the
54-path Step 5 distribution check pass. All six frozen v3 policies retain their
qualification fingerprints and admission. The isolated search runtime installer
successfully validates 112 files (the authored worker plus 111 private pinned
dependencies), without overwriting the R1 worker.

## Quick experiment scope

At the user's request, the longer 12-game workflow run was stopped after six
complete games: three S3, two R4 and one S2. All six completed and saved valid
complete-game replays. The unfinished comparison is not a balanced tournament
and is not used to rank variants or claim an equal-budget advantage. Its
deliberately short 0.2-second search allowance also produced recorded timing
overruns. Partial outputs remain under `artifacts/searchlab/paired-smoke/` for
inspection; no promotion record was created.

`scripts/smoke-searchlab.py` provides the smaller default workflow: one decision
per variant, with no games to finish and no changes to training. The longer
commands remain available for later research, not a prerequisite for trying
this additional experiment.

The quick command passed locally in approximately seven seconds including
startup. All three selected moves were accepted by the R1 host. With a requested
one-second allowance, the measured decision paths took S2 1.204 s, R4 0.937 s and
S3 1.110 s. No fresh-confirmation teacher was produced; these were legal fallback/
partial-search checks, not evidence that one second is enough for useful depth.

## Remaining acceptance limits

These checks qualify the local implementation and information boundary, not
better playing strength, calibrated personalities, human enjoyment or Unity
deployment. Small smoke checkpoints and one paired setup cannot demonstrate a
search advantage. A failed equal-budget/promotion test must remain a failed gate.

Run the correctness suite on Linux/Mono and separately for CUDA before collecting
lab comparisons. Intel GPU inference and laptop behavior while Unity renders
remain later deployment checks. The reconstruction/IPC cost can consume a short
turn allowance; all timing overruns and incomplete searches are reported. No
new search is selected for main training or installed in the game by these tools.
