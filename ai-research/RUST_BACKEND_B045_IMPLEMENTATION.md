# B04.5 implementation and g1250 qualification

2026-11-06 (America/Chicago). Implementation follows
[the search-value plan](RUST_BACKEND_B045_SEARCH_VALUE_PLAN.md). Linux/CUDA
qualification is **Pending**. Production teaching remains behind B05/B07.

The r2 transfer fixes the g1250 attempt ending `2caeea3b-b0d5a86e`: the cleanup
digest fixture caught a same-size rewrite with unchanged POSIX metadata. The
cache returned the previous digest, so qualification stopped before building
Rust. Cleanup now reads content for plan, quarantine and final deletion on all
platforms. Deterministic identical-stat regressions cover the digest and the
actual quarantine boundary; altered files stay in place. This adds a cleanup
read on Linux, while collection, admission, inference and PPG optimizations
remain unchanged. Exact source-review routes preserve current continuation.

Exports remain under `results/rust-lab-qualification/` as requested. They check
Git visibility in the results repository, include local rules to keep logs
visible and byte-exact, and print exact `git -C results` commands. The launcher
runs the export regressions and preserves a traceback on failure. Existing
evidence can be re-exported without repeating qualification. No native/search
algorithm or production teacher change is included in r2.

## Integration

The lab-only Rust teacher protocol is now `brass-lab-teacher/2.0.0`. Each
one-thread table session searches all 64 turns. The 16 reserved turns retain
4s/K16 and the existing full-action continuation and improved-policy labels.
The other 48 use 0.8s/K4 with 0.2s cumulative post-work, without policy-label
construction or policy verification. Both cutoffs survive continuation and
replanning. The installed Unity/ORT runtime and its sealed search_v1 source
are unchanged.

Value-only turn validation bypasses the policy model lock, so a fast turn does
not spend its 0.2s allowance waiting for another lane's policy verification.
The trunk ramp scales both the state encoder and shared descriptor encoder;
regression checks verify these gradients and the main-head exclusion.

Every root exports its permitted state tensor, stable/relative seat mapping,
remaining insolvency budgets, completed edge counts/returns, network value and
four-seat root mean. A Python oracle verifies `(v + sum(returns))/(1 + N)`.
Final capped insolvency costs produce the full-game and lambda targets. Value
rows have their own eight-age, reward-bound shards and never enter ordinary
replay, GAE, PPO or PPG targets. They reuse the label buffer's per-age policy
staleness rule, including retained label probes for ages four through seven.

The registered optional `search_value_head` has a zero final layer and its own
Adam. Legacy checkpoints install no new parameters. Extended checkpoints record
the head contract; the companion Adam and stage/buffer state restore separately.
Stages A/B cache detached trunk features once per pass, then perform up to 32
head steps. The stage-C API uses the existing PPG `Selection` snapshot/restore
for model, value Adam and companion together; callers supply the ordinary
phase's frozen policy probe, held-out loss, current beta and LR schedule.

G0/G1 use seeded, 200-refit game-cluster intervals over complete eight-iteration
held-out windows. The state machine records blend, lambda, scale, gate history,
head-to-head result and ordinary guard baseline. Manual forward jumps fail.
Stage C restores rejected intervals and returns to head-only on G2 or ordinary
metric guard failure. Checkpoints retain this state across interruption.

The specified lambda selection compares candidate return estimates against the
final utility-to-go. This criterion naturally favors lambda=1 (its target equals
the outcome); all candidate errors are reported and the selection is frozen.
A different bias/variance selection criterion requires a separate plan revision.

## Compatibility with current training performance work

The reference is
`results/joint7-v11-2-g1290-trained-restart-g1250-20261107T041141451177Z`.
Its checkpoint SHA is
`de427bb22e38cc5dc9d4e79151cccdb321650c02978f79bcdad746c3e6be7985`.

- Keep its recorded execution configuration: 24 processes, two hosts per
  process, demand routing, shared slots, candidate-aware batching, CPU budget
  and current caches. The bounded teacher smoke uses only two admitted lanes.
- Search requests use the existing GPU queue and shared-memory transport. The
  sixth numerical output carries request-bound value source and blend. Ordinary
  requests keep their existing forward path and output arrays. Zero blend
  returns the original utility bytes and performs no extra head/encoder work.
- The correction pass does not replace admitted-input reuse, collection
  scheduling, feature caches, PPG packing, policy caches, local-scoring heads,
  opening exploration or g1247 updates.
- g1250's ordinary PPG cap is 0.003. B04.5's combined cap remains
  `min(0.002, ordinary_cap)` against the **pre-PPG** probe. If ordinary PPG has
  already exhausted that budget, skip the correction trunk pass. Do not change
  ordinary PPG to make room.
- Teacher jobs remain first in the collection barrier. No legacy search worker
  multiplier applies: one Rust search session and one thread per table.

The source review registry carries forward the exact incoming routes approved
for the g1250 implementation, plus B03's already reviewed legacy continuation
fixture route. Existing config restrictions remain; no checkpoint signature,
optimizer, counter or throughput calibration is rewritten by qualification.

## Lab qualification

Use [the g1250 bundle commands](../distribution/rust-lab-g1250/README.md).
The bundle includes its sealed baseline inputs and needs no historical
qualification ZIP. It does not rerun B01/B02/B02.5/B03/B04. It builds the new
native teacher, excludes one GPU warmup per root and takes three warm probes
per root type, compares full-profile zero blend to the unchanged search
equations at fixed count, probes both time budgets, collects two shadow tables
(one train and one held-out), and performs one isolated stage-A update.

The user authorizes a null utility receipt for this exact g1250 **shadow** only.
The report preserves null and records the assumption; false or mismatched
receipts fail. Source checkpoint, state, optimizer, RNG, population and runtime
pins are checked again after the test. No source transition or teacher adoption
is performed on the training run. Each attempt exports a new Git-visible results
folder, including failed attempts.

`evaluate-rust-search-value.py` implements the separate 256-game, 2v2,
four-rotation head-to-head using 4s/K16 on every turn. Its interval resamples
setup clusters to retain correlation between rotations. It is deliberately
excluded from smoke qualification; run it once after stage B has trained a head.

## Remaining gates

B05 must wire the tested lifecycle into the committed production iteration and
qualify interruption, cache/optimizer retention and stage-C callbacks against
the current ordinary PPG implementation. B06 measures all-seat throughput and
GPU contention with ordinary collection. B07 canaries at most two all-seat
tables at w=0. Neither the source update nor a passing B04.5 smoke enables
production teaching or deploys the correction head to Unity/ONNX.
