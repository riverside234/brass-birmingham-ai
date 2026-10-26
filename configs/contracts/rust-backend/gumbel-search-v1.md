# Native fixed-count Gumbel IS-MCTS contract

Identity: `brass-gumbel-is-mcts/count-v1`. This is L10's scalar correctness path,
implemented in `native/brass-ai-core/src/search/`. It is callable entirely in
Rust with the L09 session-owned evaluator. It does not yet produce complete
Unity actions, use whole-turn deadlines, merge two trees or create teacher labels.

The equations and small-budget conventions are frozen by
[`reproduce-gumbel-is-mcts.py`](../../../scripts/reproduce-gumbel-is-mcts.py) and
[`GUMBEL_IS_MCTS_REFERENCE.md`](../../../ai-research/GUMBEL_IS_MCTS_REFERENCE.md).
The full Brass comparator is
[`gumbel_brass_reference.py`](../../../tests/RustBackend/gumbel_brass_reference.py).
It executes Python selection/backup over the sealed L09 simulator and Python
ORT. That simulator was independently qualified against the C# rules in L04–L07;
this comparator does not claim another independent implementation of the rules.

## Input and identity

`brass::Input` takes the acting seat's detached permitted observation, its allowed
recall, frozen feature/model/reward/profile context, exact reward specification,
and causal public cost ledger. It has no actual opponent hands, live deck or live
RNG field. The recorded shuffle tape is optional; otherwise every trajectory uses
a fresh stream of the L06 ChaCha12/Fisher–Yates sampler. This remains a
current-view belief approximation, not the paper's history/policy posterior.

Each node key includes the acting seat's complete permitted information and
recall, complete canonical choices, feature/model/reward/profile context,
remaining cost budgets and settled income events. Typed `Hash` plus full `Eq`
handles hash collisions. The full sampled world is never the node key or the
evaluator argument. Legal semantics are checked on repeated node lookup and at
every freshly sampled root. No JSON conversion occurs in the Rust trajectory.

The initial policy contract uses the same root model and fixed personality
context for all actors, unknown critic descriptors, and stable absolute-seat
utility output. Heterogeneous opponent models and per-actor personality changes
are outside this identity. The reward specification must hash to the declared
reward ID; the model adapter also checks its manifest's model/export/reward IDs.

The supplied g890 model's utility head is accepted for local readiness by the
user's 2026-10-25 decision. L10 reuses the unchanged qualified L09 export and
historical mechanical manifest; it does not fabricate a calibration receipt or
implement live deployment admission. L13 carries the model-specific acceptance
into the local loader before live integration.

## Search semantics

- Legal choices sort by complete semantic string ID, with strict uniqueness.
  Chance outcomes also sort by semantic string ID. Recorded uniforms and shuffle
  indices, not equal cross-language seeds, define exact replay.
- Full legal logits and four-seat cached values are evaluated before counted
  trajectories. One fixed Gumbel per legal action samples initial root top K.
- f64 completed-Q, visited-prior normalization, improved policy and Equation (3)
  match the executable oracle. Exact ties choose the first canonical action;
  unequal finite scores remain unequal, with no epsilon tie band. No Q rescaling,
  prior floor, frozen-policy rollout, negamax or progressive widening is added.
- Every trajectory samples a fresh world, handles explicit chance and stops at
  the first new actor information set or exact terminal settlement. New leaves
  cache the NN vector without an outgoing visit. The root model sees each actor's
  own simulated private observation, including payment/continuation/bonus/debt.
- Sequential halving uses the effective power-of-two K and equal additional
  round quotas. Only a completed backup can trigger elimination. N=0 returns
  the labeled Gumbel/prior fallback; K=1 has no halving rounds. Remainder visits
  go to the survivor and cannot change it. Constants default to N=200, K=16,
  c1=50, c2=.1 solely as reproduction settings.
- `CostLedger` ports public debt-event accounting. Charge each seat once per
  income event, cap cumulative costs, retain the preceding Finish over chance,
  and exclude the final Rail income event. Observe public costs before actor
  evaluation. Each backed-up edge receives `leaf_utility - (leaf_spent -
  edge_spent)` in absolute seat order. Terminal utility uses both official
  tiebreakers, split win shares, pairwise placement credit and the declared VP cap.

## Ownership and failure

Each private tree uses pinned `slotmap 1.1.1` generational handles with an additional
arena identity to reject foreign handles. Clearing a root invalidates old handles.
Node/edge limits fail explicitly; they never truncate legal choices. The default
limits are 16,384 nodes and 2,000,000 legal edges. These are logical bounds, not a
claim that retained observations already meet L14's final memory target.

A suspended `Pending` owns its sampled world, path, leaf information and cost
ledger values. It borrows no tree or tensor while inference executes. Completing
it consumes the object. Backup preflight checks all handles, edges, cost deltas,
counts and sums, including repeated edges, before expansion/publication. Dropping
or failing a pending leaf publishes no visits. This scalar path has no concurrent
expansion followers or pending scheduling slots; those are L12 work.

Malformed inputs, wrong identities, nonfinite values, replay exhaustion/leftovers,
depth exhaustion, inconsistent legality and memory limits return explicit errors.
The default 256-step diagnostic guard never invents a bootstrap return. Failure
aborts the scalar request; it does not produce a partial tournament recommendation.

## Verification and measurement

`gumbel-trace.exe INPUT.json OUTPUT.json` is an offline file diagnostic, not a new
live worker command. It supports the toy and full Brass, with either exact taped
actor-view evaluator replies or same-process native Features-4 + ORT. Verbose
trace output is opt-in; normal runs do not retain per-node/trajectory JSON.

`verify-rust-search.py --prepare` records permitted sampled-world indices, chance
uniforms, causal ledgers and Python NN responses. Replay compares every node,
selected edge, first-new leaf, four-seat backup, round and survivor. Taped numeric
parity uses 1e-10 absolute/relative tolerance; actual NN replay uses L09's 2e-4
output tolerance, while action IDs, counts, expansions and survivor sets must
still match exactly. Native unit/property tests independently check the equations,
ties, underflow, tiny budgets, terminal rewards, failures and arena ownership.

`qualify-rust-search.py` builds debug/release/profiling, runs relevant regressions,
and seals sources, dependencies, model identities, fixture/report hashes and
binaries. Component timers distinguish sampling, information/choice construction,
lookup, selection, rule application, feature encoding, inference and backup.
Allocation probes use `stats_alloc`; ORT's C allocator is excluded. Performance
probes run two independent owners, each with one thread and its own persistent
ORT session, after loading and warm-up. They are fixed-count component profiles,
not the later matched four-second whole-turn comparison or playing-strength evidence.
