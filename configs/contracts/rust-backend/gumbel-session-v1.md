# Whole-turn Gumbel sessions, version 1

Identity: `brass-gumbel-is-mcts/session-v1`. The scalar count oracle remains
`brass-gumbel-is-mcts/count-v1`. L11 introduces the single-session adaptation;
L12 owns two-session merging, batching, expensive-operation polling and runtime
retirement. This is not a deployed worker capability or a latency qualification.

## Lifetime and freshness

`search/session` binds one admitted ordinary turn to game instance, generation,
export, `(era, round, turn, turn_owner, actor, is_debt)`, feature/model/reward/profile
context and a caller-owned monotonic clock. Request numbers increase. Revisions
cannot move backwards; a changed root requires a newer revision. A repeated root
cannot start another tournament. Invalid input cannot shorten the active clock
or alter its recall. Replacing/navigating a game requires retirement and a new
coordinator-admitted instance, not a retry with more time.

The caller creates `Budget` once at admission. Reserve is the personality
`SearchSession` reference's `min(1.2 seconds, 0.10 * allowance)`: 0.4/0.8/1.0/1.2
seconds at the four 4/8/10/12-second settings. Subsequent requests may only narrow
the existing deadlines. Action two, staged choices, invalidated continuations and
replanning retain that clock. The four-second matched benchmark remains L14.

The session owns the observer's recall. It appends the newly observed own hand
when it changes; caller-supplied nonempty recall must match. An empty field
delegates recall to the session. When a whole action commits between requests,
an exact new permitted-view/context match to a validated continuation acknowledges
its intermediate own-hand revision. The saved recall must extend the known
prefix; no sampled future hand is adopted. Style/intent/model/profile/reward/export are
fixed for the turn. Remaining utility can only decrease and must match the causal
ledger validated by the Brass adapter. Trees/counts/tournaments are always fresh
at a changed root. Only completed NN evaluations with exactly equal typed keys,
actor and full legal choices may be cached; cache entries contain no tree visits.
The bounded FIFO cache is scoped to this session's immutable export identity.

## Frozen quota and recommendations

The executable independent reference is
[`reproduce-gumbel-sessions.py`](../../../scripts/reproduce-gumbel-sessions.py),
with recorded [`session-v1.json`](../../../tests/GumbelISMCTS/fixtures/session-v1.json).
It tests interruption before the first balanced round, on/between round
boundaries, after the final backup, during remainder work, K=1 and zero work.

Count mode uses exact N/K with the caller's deadline as an exceptional watchdog.
Timed mode freezes `floor(remaining_work_ns / estimated_trajectory_ns)` after
root evaluation and before the tournament. The estimate is the lower quartile
of the last eight completed, identity/context/phase-matched local costs, requiring
four samples, or a separately measured startup estimate bound to model, export,
reward and phase. The receipt identifies the source. Missing/zero estimates give
N=0 and bounded fallback; a mismatched identity is rejected. Positive estimates
cannot produce more than the one-million-trajectory memory/work bound. N never
increases; no restart consumes leftover time. Launch admission also uses the
last-eight lower-quartile safeguard. Unknown descriptors remain mandatory.

Only completed backups count. A balanced round publishes a small root-statistics
snapshot, its survivors and completed-trajectory support watermark. K=1 snapshots
after each completed trajectory. Normal quota completion includes the survivor's
floor remainder. An interrupted round/remainder cannot update the eligible
snapshot or eliminate candidates. Partial completed work remains diagnostic.
The last balanced snapshot supplies all candidate scores; no eligible snapshot
means the common Gumbel-plus-prior proposal is available only to fallback.
Canonical ties and L10 math remain unchanged. The coordinator merger is L12.

## Complete actions and support

Each proof step contains the exact permitted actor view/recall, full legal
semantics, context, causal budget and selected semantic ID. A completed action
ends at `finish`; a debt response is one independently validated `liquidate`.
The plan never crosses sampled chance or predicts an unknown future hand.
First/second actions are separate lists, compatible with the current C# plan
format; the Unity bridge will validate and commit each separately in L13.
Payment/bonus/continuation roots return the complete remaining suffix and retain
their `start_phase`; they must not be misrepresented as fresh action boundaries.

Only a timely completed trajectory containing a complete action contributes to
the plan ledger. A neural leaf inside that action adds tree evidence, not plan
support. The ledger records complete prefixes and their original backup serial
numbers and cost-adjusted four-seat returns. At a snapshot watermark, require at
least two completed contributions to a matching prefix. Select first by eligible
Gumbel candidate rank; within a candidate select plan support, then root-actor
mean return, then canonical semantic IDs. An optional second action requires its
own supported full prefix. Future contributions cannot qualify an older snapshot.

Replay selected plans on a detached permitted reconstruction and require every
precondition and legal choice to match. A saved continuation may supply only a
matching current suffix; otherwise discard it and replan with the remaining
budget. Completed-path proof storage is bounded to 65,536 steps and two million
prefix contributions; exhaustion fails explicitly instead of truncating choices.

If no supported plan qualifies, finish one current action under the same hard
deadline, with at most 96 semantic choices and no additional NN calls. Root uses
the eligible recommendation or prior proposal; later choices prefer Finish,
then merchant beer, then canonical order. Fallback support is zero and its
provenance is separate. Forced choices launch no NN/search; debt uses its own
bounded policy/completion path and launches no tournament. No path invokes V4.
Cancellation retires the result; expiry or failed completion returns no action.

## Qualification boundary

[`verify-rust-sessions.py`](../../../scripts/verify-rust-sessions.py) independently
extracts plan contributions from sealed L10 trajectories, compares recorded and
actual native NN runs, and replays original-root plans through the frozen C#
rules exporter. Five explicitly rotated seat fixtures retain their L10/native
rules evidence instead of pretending to have an original C# checkpoint.
`gumbel-session` is a bounded diagnostic file harness. Count diagnostics use an
explicit 60-second watchdog, not a fresh timing benchmark allowance per root.
No new NN calibration, installed-game change or lab admission follows from L11.

Checks bracket traversal, NN evaluation, proof construction, backup and each
completion/validation step. Late responses cannot enter the cache or publish
counts. Deep internal polling and in-flight runtime termination remain L12;
bounded step counts do not establish a production wall-clock overrun bound.
