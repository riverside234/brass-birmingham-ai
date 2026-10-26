# Executable Gumbel IS-MCTS reference

Updated **2026-10-25**. [`scripts/reproduce-gumbel-is-mcts.py`](../scripts/reproduce-gumbel-is-mcts.py) is the small, standard-library Python oracle for the Rust implementation in [migration steps L10–L12](RUST_AI_BACKEND_MIGRATION_PLAN.md#l10--implement-deterministic-gumbel-is-mcts-in-rust). It implements fixed-count algorithm mechanics on a finite hidden-type game. It does not run Brass, load a neural model, reproduce the paper's training/results, or replace the installed search. L08/L09 are qualified. L10 qualifies the native scalar implementation and full Brass comparator; see the [L10 results](RUST_BACKEND_L10_REPORT.md). L11 qualifies scalar whole-turn sessions and complete-action plans; see the [L11 results](RUST_BACKEND_L11_REPORT.md) and [session reference](../scripts/reproduce-gumbel-sessions.py). Two-session scheduling and integration remain L12–L15.

## Sources and precedence

The algorithm source is Li and Wellman, [*A Meta-Game Evaluation Framework for Deep Multiagent Reinforcement Learning*, arXiv:2405.00243v1](https://arxiv.org/abs/2405.00243v1): §6, Algorithm 1, Appendix B.1/B.2, Equation (3), Algorithm 3 and Table 7. The attached 19-page PDF's SHA-256 is `29bd21b7197c68c1569e3eea4e8df354c3531f8ef9210efbd5cea69db23d720e`. This source supplies algorithm definitions, not permission to run experiments or change training.

The code reference is [`olivkoch/gumbel-mcts` at `f7cc656af7a037f1283b08b454982edba4d86e21`](https://github.com/olivkoch/gumbel-mcts/tree/f7cc656af7a037f1283b08b454982edba4d86e21). Its [sparse kernels](https://github.com/olivkoch/gumbel-mcts/blob/f7cc656af7a037f1283b08b454982edba4d86e21/src/kernels/gumbel_sparse_kernels.py) and [sparse scheduler](https://github.com/olivkoch/gumbel-mcts/blob/f7cc656af7a037f1283b08b454982edba4d86e21/src/gumbel_mcts/gumbel_sparse.py) illustrate flat node/edge storage, cached priors/values, root candidate forcing and batched scheduling. The reproduction is an original implementation of the paper's equations; it does not import or execute that code. Preserve its [MIT notice](https://github.com/olivkoch/gumbel-mcts/blob/f7cc656af7a037f1283b08b454982edba4d86e21/LICENSE) if code is reused later.

The paper takes precedence where these references differ. Do **not** copy the repository's two-player negamax, concrete-board tree identity, min/max Q normalization, dense entropy-dependent K, or different halving quotas into the paper reference. None supplies Brass's information sets, permitted beliefs, four-seat values, staged plans or deadline rules. Source revision, individual blob identities, optional local snapshot hashes and executable/golden hashes are recorded in [`gumbel-is-mcts.lock.json`](../configs/references/gumbel-is-mcts.lock.json). Historical Masked ISMCTS locks are unchanged.

## Run and replay

From the repository root, with Python 3.10 or newer (no packages required):

```powershell
python scripts/reproduce-gumbel-is-mcts.py --out artifacts/gumbel-is-mcts/20261025-reference/paper-parameters.json
python scripts/reproduce-gumbel-is-mcts.py --replay artifacts/gumbel-is-mcts/20261025-reference/paper-parameters.json
python -m unittest discover -s tests/GumbelISMCTS -v
python scripts/reproduce-gumbel-is-mcts.py --replay tests/GumbelISMCTS/fixtures/reference-v1.json
```

Defaults use Table 7's **N=200, K=16, c1=50, c2=0.1**. `--simulations`, `--k`, `--seed`, and `--players 2|4` select a toy case. Replay takes all settings and uniform draws from the trace and checks the entire result, including selections, information sets, values and backups. The recorded draws, not equal Python/Rust seeds, define cross-language randomness. The checked-in [32-simulation golden trace](../tests/GumbelISMCTS/fixtures/reference-v1.json) is a compact porting fixture; regenerating it alone is not evidence of correctness.

Validated locally: **14 tests pass**; full 200-simulation and checked-in 32-simulation traces replay exactly. The default run selects toy action 6, completes 200 trajectories, allocates **48 / 48 / 48 / 50** trajectories to the four halving rounds, and spends the remaining **6** on the survivor. These are mechanics checks, not speed or strength results.

## Mathematical contract

For each actor information set `s`, store legal logits `l[a]`, normalized prior `p[a]`, cached actor-view value vector `v`, completed edge counts `n[a]`, and accumulated absolute-seat returns `R[a][i]`. Let `i` be the actor at this information set and `N = sum(n)`.

```text
q[a] = R[a][i] / n[a]                                      if n[a] > 0
v_mix = v[i]                                               if N == 0
v_mix = (v[i] + N * sum_visited(p[b] * q[b])
                       / sum_visited(p[b])) / (1 + N)      otherwise
q[a] = v_mix                                              if n[a] == 0
G(q[a]) = c2 * (c1 + max_b n[b]) * q[a]
improved_policy[a] = softmax(l[a] + G(q[a]))
root_score[a] = g[a] + l[a] + G(q[a])
```

The visited-edge prior average is **not** visit weighted. Compute the visited prior normalization directly from visited logits with stable softmax; this is algebraically equivalent and avoids an underflow-induced `0/0`. Do not floor priors or add Q normalization. Appendix B writes `v(s,a)` although Algorithm 1 defines a state value `v(s_i(h))`; this reference explicitly interprets it as the cached state value's actor component, consistent with the algorithm's input. This notation choice is part of the reference contract.

At non-root information sets, Equation (3)'s squared-discrepancy minimizer is exactly:

```text
argmax_a (improved_policy[a] - n[a] / (1 + N))
```

The independent test also evaluates the literal squared expression. Ties use the first canonical semantic action. There are no non-root Gumbels, PUCT/FPU terms, protected action families or progressive widening in this reference.

At the root, draw each `g[a] = -log(-log(u[a]))` **once**, with `0 < u < 1`. Initialize candidates with top K by `g + l`. Each halving round visits every surviving action equally, with per-action quota `floor(N / (survivor_count * log2(K)))`. Scores use completed cumulative edge statistics; the quota is additional work in that round. Halve only **after** the completing backup. Gumbels stay fixed throughout the search.

For every trajectory, sample a fresh world from the root belief, traverse with each acting player's own information, explicitly sample chance, and stop at the **first new actor information set** or a terminal state. A new node caches its policy/value but gets no outgoing visit until a later traversal. Back up the leaf's absolute-seat vector along every completed edge; each node selects using its own actor's component. Never alternate the sign or substitute the root's private view for another actor's view.

The toy's public signal has likelihood 0.8 under opponent 1's high type and 0.2 under its low type. A uniform type prior therefore gives posterior 0.8 after that signal. Other opponent types remain independent uniform bits. The toy uses this exact posterior, hidden-type acceptance decisions, explicit terminal chance and general-sum returns. Its evaluator receives only an `Info` object and supplies deterministic logits and zero bootstrap values. This is a stand-in evaluator, not a claim that a learned value model is exact.

## Explicit executable conventions

Algorithm 3 does not completely specify small/odd budgets, one legal action or floor remainders. The reference chooses and tests these conventions:

| Case | Reference behavior |
| --- | --- |
| Root initialization | Expand/evaluate the permitted root once before counted trajectories; retain evaluation cost separately in later timed implementations |
| K or legal count not a power of two | Effective K is the largest power of two no larger than requested K or the legal count; full legal priors remain intact |
| Insufficient budget | Reduce K until `N >= K*log2(K)`, or K=1; no zero-quota elimination loop |
| N=0 | Gumbel-plus-prior fallback, zero completed visits; explicitly labeled `prior_fallback` |
| K=1 | All trajectories select that candidate; no division by `log2(1)` |
| Floor remainder | Spend on the final survivor; do not rerun elimination or change the returned winner |
| Ties | Ascending canonical semantic action IDs; candidates visited in that order within each quota pass |
| Numeric behavior | Finite legal logits/values, stable softmax, double-precision accumulation, no hidden prior floors or Q rescaling |
| Invalid inputs / depth guard | Explicit error; the 256-step diagnostic guard never creates a bootstrap value or partial visit |

The script is synchronous and owns one tree. It intentionally supplies **no wall-clock scheduling, two-session merger, complete Brass plans, asynchronous NN batching, trained-network admission or production performance claims**. Those adaptations are specified and gated by A4–A5 and L08–L14 of the migration plan. A one-lane mathematical fixture does not change the requested performance configuration of **two independent sessions, one compute thread each**.

## Rust implementation and acceptance

1. L08/L09 qualify the unchanged model on the newly required actor/phase leaves, including relative-to-absolute seat mapping, unknown critic descriptors, full legal logits and utility/reward contexts. The user accepted the pinned g890 utility head for local readiness on 2026-10-25 despite its null calibration receipt; the migration plan records that model-specific decision. Historical L08/L09 evidence remains unchanged.
2. L10 implements the paper math in `native/brass-ai-core/src/search/` and replays this golden trace. The diagnostic [`gumbel_brass_reference.py`](../tests/RustBackend/gumbel_brass_reference.py) uses the same oracle rules over the qualified simulator with recorded Brass worlds/chance/NN responses, comparing every selection, expansion and backup against the native core. See the [native count-mode contract](../configs/contracts/rust-backend/gumbel-search-v1.md) and [`qualify-rust-search.py`](../scripts/qualify-rust-search.py). The toy is not the full-game comparator.
3. L11 qualifies the separately versioned scalar session adaptation: frozen timed quotas, completed-round snapshots, whole-turn context/recall, supported complete actions and bounded completion. Its [independent reference](../scripts/reproduce-gumbel-sessions.py), [recorded cases](../tests/GumbelISMCTS/fixtures/session-v1.json) and [native contract](../configs/contracts/rust-backend/gumbel-session-v1.md) preserve the L10 foundation. L12 qualifies the separately versioned two-session coordinator with an [independent merger/pending reference](../scripts/reproduce-gumbel-coordinator.py), [recorded fixtures](../tests/GumbelISMCTS/fixtures/coordinator-v1.json), [native contract](../configs/contracts/rust-backend/gumbel-coordinator-v1.md) and [results report](RUST_BACKEND_L12_REPORT.md). The cap-one baseline preserves scalar results; larger cooperative waves retain an explicit scheduling-variant identity and require L14 quality evidence. Scalar session correctness does not imply production latency or playing-strength qualification.
4. Retain crate reuse, typed arenas, generational handles, reusable contiguous buffers, stable semantic IDs, safe ownership and exactly-once backup. Optimize actual bottlenecks under debug/release tests, property tests and recorded-draw differential traces; retain a simple scalar implementation as the oracle. No JSON, Python callbacks or shared mutable tree belongs in a production trajectory.
5. L14 separates algorithm changes from implementation speed: legacy A/B, Python Gumbel `G_py`, Rust reference-evaluator `G_ref`, and native ONNX `G`. All timing comparisons use **4-second whole turns and 2 × 1 compute threads**. Equal-time strength and valid-plan coverage must qualify the new algorithm independently of higher trajectory counts.

Brass's L06 sampler conditions on permitted current information, not the paper's exact history-and-policy posterior. Keep that explicit approximation versioned; do not claim exact Bayesian Brass beliefs. The paper's idealized improvement claims do not establish improvement for approximate beliefs/values, four-player Brass, deadline truncation or replica aggregation. Part 2 reuses only a locally accepted implementation and does not enable disabled training teachers or alter policy-only collection automatically.
