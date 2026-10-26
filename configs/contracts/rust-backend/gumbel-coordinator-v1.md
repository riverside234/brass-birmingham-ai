# Two-session Gumbel coordinator, version 1

Identity: `brass-gumbel-is-mcts/coordinator-v1`. This extends the qualified
[`session-v1`](gumbel-session-v1.md) rules. The independent executable is
[`reproduce-gumbel-coordinator.py`](../../../scripts/reproduce-gumbel-coordinator.py),
with recorded merger/reservation cases in
[`coordinator-v1.json`](../../../tests/GumbelISMCTS/fixtures/coordinator-v1.json).
It specifies a Brass adaptation, not an additional guarantee from the paper.

## Owners and turn lifetime

`brass-ai-worker::pair::Pair` owns two persistent, named compute threads.
Each constructs, uses and drops its own single-threaded CPU evaluator, reusable
feature buffers, private session, tree, sampling/chance stream and NN cache.
The caller only passes envelopes, pools root evidence and tracks retirement.
There is no shared mutable tree, cross-session batch or extra inference pool.
Bounded Crossbeam channels provide one command and one reply slot per owner.

Owner zero prepares one immutable permitted root: canonical legal IDs, logits,
four-seat cached value and one uniform/Gumbel per legal action. Both owners use
that exact root. Their world/chance streams are independently seeded; owner one's
diagnostic shuffle tape is cleared. Shared `Arc` keys/action lists retain full
semantic equality and hashes. No hidden deal enters a key or policy tensor.

Admission creates one monotonic turn clock. The four settings remain 4/8/10/12
seconds, with the existing 10% reserve capped at 1.2 seconds. Dispatch converts
caller remaining time to an absolute deadline before queueing/root preparation.
Each owner narrows its existing budget; neither retry nor continuation creates
time. Both owners adopt the single selected proof so later staged/second-action
continuations have the same preconditions. Forced and debt paths use no NN.

## Cooperative waves

The retained baseline is cap=1, pending=1, flush=0. Experimental caps are 2/4,
pending is between cap and 4, candidate-cell bounds are 8,192–32,768, and flush
ages are 0/500/1,000/2,000 microseconds. Zero flush sends immediately. No owner
sleeps to fill its own batch. A wave stops at its pending/age limit or the current
round's exact quota barrier; padded rows are bounded by the cell limit.

Root launch slots are reserved FIFO in canonical round-robin order. Reservations
do not change counts, returns, non-root selection, elimination or plan support.
Within a wave, non-root Equation (3) sees only completed work. Consequently caps
above one can select different paths than the scalar oracle. They are separately
identified scheduling variants; a throughput improvement alone cannot promote
them as equal-quality search. Cap one must retain L11 scalar traces/plans.

Pending trajectories own handles/proof data, not references into a mutable tree.
Equal new-leaf information sets in a wave share one evaluator reply. The first
FIFO arrival inserts the node; followers verify identical logits/value and back
up that first-new-leaf value once per completed path. All batch replies must pass
identity, shape, numerical and time checks before publication. Evaluation IDs
include generation, coordinator request, owner and local sequence.

## Transaction and retirement boundary

Backup preflight computes all returns/counts before mutation. Publication journals
only touched edge values; it polls before each changed edge and after the final
edge. An interrupted transaction restores all edges and removes its unpublished
expansion. Successful final checking is the backup's linearization point. Its
timestamp must precede the work deadline; then that completed contribution updates
the plan ledger and exact round quota. The last required backup may eliminate
candidates; an interrupted last backup cannot. No full-tree copy is involved.

Every exit drains or retires the outstanding launch reservations exactly once.
Late NN replies create no visits or support. Generation, sampling, observation,
application, traversal, encoding, expansion, backup, publication and completion
poll an explicit thread-local work scope. Its Rust type cannot move threads.
Synchronous ORT calls cannot be interrupted safely: cancellation invalidates the
result, but the pair stays busy until both owners return and retire. A failed
dispatch, owner panic or finalization error also drains the surviving owner.
No action is exposed after cancellation/hard expiry and no fallback gets a new
allowance. Scope restoration and response identity checks survive unwinding.

The CPU polling-gap metric excludes synchronous runtime time. Runtime and root
wait are separately recorded. OS descheduling can enlarge measured wall gaps;
these measurements are not a hard real-time delivery guarantee.

## Pooled recommendation

Use only each owner's latest eligible balanced-round/quota snapshot. Validate
common logits/value, legal widths, finite returns, count sum/watermark and survivor
bounds. Sum completed four-seat R and C with checked arithmetic. Recompute Q,
v_mix and Gumbel scores using the common root priors/value and common Gumbels.
Rank the union of eligible survivors, resolving exact ties by canonical ID order.
Private nodes and unbalanced newer work are never pooled. With one eligible
owner, use it alone; with neither, use common Gumbel-plus-logit fallback.

Search-supported plans must independently meet the existing minimum support of
two in one eligible owner's watermark. Two unsupported one-visit prefixes do not
become a supported plan by pooling. Rank by eligible candidate, then prefix
support, actor mean return, semantic action IDs and owner index. Revalidate the
selected proof on both owners. Otherwise complete the pooled proposal under the
same bounded completion rules with support zero and separate provenance.

## Qualification and measurement

[`verify-rust-coordinator.py`](../../../scripts/verify-rust-coordinator.py) compares
all 44 recorded and actual-NN singleton roots against sealed L11 results, checks
two-owner pooled equations independently in Python, and replays selected plans in
the frozen C# exporter for 39 original checkpoints. Five rotated-seat cases keep
their labeled native evidence. Fixed-count correctness uses a 60-second watchdog.

[`benchmark-rust-coordinator.py`](../../../scripts/benchmark-rust-coordinator.py)
freezes development/held-out roots and selection criteria before measurement.
It measures the nine cap/flush settings using four-second turns, two repetitions,
then checks the selected engineering candidate on held-out roots. Startup cost
receipts come from a separately labeled cap-one, 64-trajectory-per-owner
calibration bound to the same model/export/reward/phase; estimates freeze before
each tournament. This is an explicit L12 measured estimate, not historical L09
timing. Missing estimates remain N=0, with no invented search support.

Normal timings use `gumbel-pair`; allocation attribution uses a separate
`gumbel-pair-alloc` executable and reports traffic, not peak live bytes. Scripted
and actual NN component probes are separate. Record sampled process CPU/peak
working set, queue/root wait, copies, first-leaf latency, batch occupancy/padding,
completed versus eligible work, plan coverage and response/retirement times.
Model initialization is outside admission and reported at process scope.

This is a research library/harness qualification. L13 owns production framing,
packaging, Unity shadow admission and g890 live-readiness identity checks. L14
owns matched implementation-speed, equal-time strength and production latency
acceptance. Installed gameplay, training and the null calibration receipt remain
unchanged. No V4 mixing or fallback is introduced.
