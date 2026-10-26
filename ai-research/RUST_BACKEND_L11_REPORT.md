# L11 — Whole-turn Gumbel sessions and complete plans

Updated **2026-10-25**. **L11 is implemented and qualified.** Debug and release
pass the full native/Brass plan checks; profiling passes the process boundary
suite. **L12 is next.**

Candidate identity: `brass-gumbel-is-mcts/session-v1`.
Native source SHA-256: `b81171e2c126a7376a2ba3077e86050fcdd491091065fa325b2769d5ee7b2394`.
Evidence directory: [`20261025-l11-qualified-v2`](../artifacts/rust-backend/20261025-l11-qualified-v2/).
The installed game, temporary g890 checkpoint/export, historical calibration
receipts and lab training are unchanged.

## Implemented behavior

- [`search/session`](../native/brass-ai-core/src/search/session/mod.rs) owns one
  admitted turn, game/generation/export identity, monotonic request/revision
  guards, fixed personality/model/reward context, own-hand recall and remaining
  budget. Changed roots start fresh tree/tournament statistics. Only identical
  completed NN evaluations may enter the bounded, export-scoped cache.
- The original turn clock survives payments, continuations, bonuses, action two
  and replanning. The pinned reference reserve is `min(1.2s, 10% of allowance)`:
  **0.4/0.8/1.0/1.2 seconds** for the four **4/8/10/12-second** settings.
- Timed roots freeze a count quota from remaining work time and a matching
  completed-cost or recorded startup estimate. Missing/zero estimates select
  bounded fallback. N never increases, and unused time does not start a new
  tournament. The launch safeguard uses the last eight completed local costs,
  requiring four samples before its lower-quartile estimate is available.
- Balanced rounds publish root snapshots and plan-support watermarks. Interrupted
  rounds/remainders cannot change the recommendation. K=1 and normal quota
  completion follow the separately tested snapshot conventions.
- Completed action prefixes have their own contribution ledger. A partial NN
  leaf cannot grant plan support. Selection requires at least two eligible
  completed contributions; a second action needs its own supported full prefix.
  Historical support queries use bounded cumulative returns and binary search.
- Plans preserve ordered semantic choices and exact permitted preconditions.
  They replay on a detached rules state before delivery. First and second
  actions remain separate ID lists. Staged roots explicitly identify their
  starting phase and produce a complete suffix.
- A whole-action commit can acknowledge intermediate own-hand recall through an
  exact observed-view/context match to a validated continuation or deterministic
  action postcondition. The known recall prefix must match. This preserves a
  valid second action and permits identical NN-cache reuse without tree reuse.
  A changed precondition discards the continuation and replans on the same clock.
- Bounded completion makes no further NN calls, performs at most 96 choices,
  and reports zero search support. Forced roots launch no NN/search; debt uses
  its own policy/completion path. Cancellation or expired completion returns no
  action. No fallback invokes V4.

No dependency upgrade or new crate was needed: the session layer uses the
existing SlotMap-backed search, native rules/inference adapters, Serde and Rust
standard-library collections/clock primitives. The count-v1 oracle is retained.

## References and validation

The small independent
[`reproduce-gumbel-sessions.py`](../scripts/reproduce-gumbel-sessions.py) and
[`session-v1.json`](../tests/GumbelISMCTS/fixtures/session-v1.json) were written
before session integration. They cover ten quota/round interruption scenarios
and six startup/completed-cost quota examples. The detailed contract is
[`gumbel-session-v1.md`](../configs/contracts/rust-backend/gumbel-session-v1.md).

| Check | Debug result | Release/profiling result |
| --- | --- | --- |
| Native tests, including ownership compile-fail doctest | 68 passed, including 14 L11 regressions | Release: 68 passed |
| Actual ORT golden test | Passed, 11 positions | Release: passed, 11 positions |
| Python equation/trace/session oracle tests | 18 passed | Shared deterministic oracle |
| Session/search/transport process tests | 9 + 12 + 11 passed | Release and profiling: all 32 passed each |
| Retained L10 toy/full-Brass count parity | 17 toy cases; 44 Brass cases / 1,664 trajectories per evaluation mode | Release: identical discrete traces, within declared numeric tolerances |
| L11 plans with recorded and actual native NN evaluation | 44 cases, 40 era × phase × actor cells | Release: same actions and support; all C# replays pass |

The 44-case L11 corpus contains **22 search-supported decisions, 12 forced
completions and 10 debt decisions**, producing **55 complete actions**. Each
recorded/native evaluation mode completes **768 search trajectories**. Counts,
root returns, selected actions and complete-plan support are checked against
independent extraction from the sealed L10 traces. No partial contribution is
promoted to plan support.

The frozen C# rules exporter replays **101 semantic choices across 39 original
checkpoint cases**, checking the permitted view and legality before each choice.
The remaining five cases are the explicitly labeled cyclic-seat fixtures from
L10; their evidence is the sealed comparator/native rules, not an invented C#
checkpoint. Search, fallback, forced and debt cases are reported separately.
The nine boundary tests additionally exercise an actual-NN one-trajectory case:
its partial leaf gives zero complete-plan support and requires bounded completion.

These are **fixed-count correctness runs with a 60-second exceptional watchdog**,
not per-root performance measurements or the matched benchmark. The future
matched comparison remains **one 4-second whole-turn allowance**, with **two
independent sessions, one thread each**. Fake-clock regressions test all four
setting/reserve pairs and actual interrupted scalar search boundaries.

The 14 native session regressions include stale identity/revision/context
rejection, no repeated-root tournament, invalid requests leaving the clock intact,
late NN replies, before/after-round interruption, bounded completion expiry,
partial leaves, support watermarks, action-two support, changed legality,
whole-action recall acknowledgment and fresh counts with exact NN-cache reuse.

## Scope remaining for L12 and later

This is the scalar session foundation. **L12** still owns two-session scheduling
and pooled recommendation, cooperative pending paths/batching, polling inside
expensive operations, transactional deadline publication and in-flight runtime
retirement. The diagnostic harness rejects late replies at scalar boundaries;
that does not qualify production cutoff/delivery latency. Proof/key copying and
cache memory need the planned L12 allocation work.

**L13** still owns the native worker protocol/Unity adapter, live action validation
and separate commits/presentation, the g890 model-specific readiness acceptance,
packaging and the no-mixed-NN/V4 table UI. **L14/L15** retain the four-second matched
benchmark, all-setting latency checks, decision-quality and full-game gates.
No installed backend or lab training migration is claimed here.

The first unsealed qualification attempt was interrupted to add the whole-action
recall regression. Earlier harness size/temp-directory/assertion failures remain
in development evidence. This candidate uses a fresh qualification directory;
previous completed milestone evidence was not rewritten.

Reproduction entry point:

```powershell
python scripts/qualify-rust-sessions.py --out artifacts/rust-backend/my-new-l11-run
```

Use a new output directory and the pinned local toolchain/model/corpus. The
qualifier preserves prior source/model/installed/save hashes and records source,
binary, command, raw output and archive receipts.

Final evidence: [qualification receipt](../artifacts/rust-backend/20261025-l11-qualified-v2/qualification.json), [independent audit](../artifacts/rust-backend/20261025-l11-qualified-v2/final-verification.json), [debug plan replay](../artifacts/rust-backend/20261025-l11-qualified-v2/debug-plans/verification.json) and [release plan replay](../artifacts/rust-backend/20261025-l11-qualified-v2/release-plans/verification.json). Formatting and strict own-code Clippy pass; the six preserved upstream warnings remain unchanged.
