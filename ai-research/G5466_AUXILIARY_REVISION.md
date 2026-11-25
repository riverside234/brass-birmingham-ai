# g5466 auxiliary revision and iteration speed

Run `joint7-v11-2-g4970-rail025-soft8-beta0.02`, committed at g5466 in the newest export
(`joint7-v11-2-g4970-rail025-soft8-beta0.02-g5466-20261125T040736512851Z`; the request named g5450). The training
changes are adopted once with `--auxiliary-revision` at a committed boundary, then kept
(`src/brass_ai/recovery/training_revision_g5466.py`). The speed changes are code-only.

## Request (2026-11-24)

Performance:

1. Optimize the preallocated MCTS node pool (node statistics in several Vecs, backup allocating temporaries), if the
   gain is worth it; skip it if the gain is small or negative.
2. Speed up the 32.5 s post-commit cleanup and the 15.8 s full GC pass.
3. Cut rating-evaluation time: remove hashing and validation that are not needed, and disable the bootstrap.
4. Decide, with short smoke tests, whether Rust should write ordinary-game exports straight into preallocated shared
   memory (about 13.35 GiB per 320 games at g5177).

Training:

0. The Rail-end heads update in PPG only. Treat them exactly like the other auxiliary heads in training and updates.
1. PPG `grad_clip` 2.2, with the Adam state rescaled.
2. The era heads' own clip 6 in PPO and PPG, rescaled. **Skipped** at the user's rule: their norms reached 1 in only
   one of the last 30 iterations (max 1.58 at g5446; otherwise at most 0.67).
3. The soft head's own PPO clip 6, rescaled. **Skipped**: 0 of 1,380 steps clipped at 1.0 in 30 iterations (P90 at
   most 0.12).
4. Every ownership/auxiliary head at the PPO rate (linked), fixed: no warm-up or decay.

## Training changes (`--auxiliary-revision`)

| | before (g5402 recipe) | after |
| --- | --- | --- |
| Rail-end outputs in PPO | not frozen, not preserved, no PPO-phase gradient | frozen and preserved like Canal/Rail (owner KL + VP Huber at weight .25) |
| PPG clip | 1.0 (81% of steps clipped in g5436-g5465) | 2.2 (about the logged P95) |
| PPG Adam | built on gradients clipped at 1.0 | moments rescaled once at adoption (below) |
| Era heads, PPG rate | PPG peak (.65 x PPO) x PPG warm-up/cosine factor: 3.9e-6 at peak, 3.9e-9 at the end | the PPO rate (6e-6 at g5465), fixed for the whole phase, head-only steps included |
| Era heads, PPO rate | the PPO step rate | unchanged |
| Soft head rate | the PPO step rate | unchanged |
| Head clips (era heads, soft) | 1.0 | unchanged (items 2-3 skipped) |

What already matched before this change: the Rail-end outputs trained in the same PPG batches and steps as Canal/Rail,
their own warm-up has ended (multiplier 1.0, `stage: full`), they sit in the ownership stop's criterion at their loss
weight (.25 >= the .25 entry weight), PPO-game Canal rows use the actual game-end map, and search rows use lambda
returns. So item 0 only needed PPO: `CompanionV3.preservation_weights()` dropped the Rail-end outputs; with
`rail_end_ppo` set from the revision it keeps them, so `FrozenValues` snapshots them and PPO's preservation loss
reaches their layers. Their loss weight stays the run's .25 (`--rail-end-weight 0.25`, the user's setting).

"The PPO rate" in PPG is the rate PPO's last step of the same iteration applied, the one PPG's own peak is .65 of
(`ppg_g210` `rate['ppo_lr']`). It is reported as `value_phase.era_head_rate`.

### PPG Adam rescale

Clipping scales the whole gradient to `min(n, c)` along its direction, so Adam's moments hold `E[min(n, 1)]` and
`E[min(n, 1)^2]`. At adoption, once, on the adoption generation's checkpoint (`ppg_g210.initialize` ->
`rescale_ppg`), every PPG Adam `exp_avg` is multiplied by `E[min(n, 2.2)] / E[min(n, 1)]` and `exp_avg_sq` by
`E[min(n, 2.2)^2] / E[min(n, 1)^2]`. The receipt is kept in the Adam group, so a later checkpoint is not rescaled
again; a crashed attempt reloads the same checkpoint and rescales it the same way.

The norms come from the logged `value_phase.gradient_clipping` of the last 30 committed iterations (state.recent_training,
the run's uncompacted reports and `retention-history.json`): about 1,000 attempted steps, Adam's beta2 horizon. Each
iteration is a normal with its logged mean and `sd = (P95 - mean) / 1.645`, pooled by steps. Three fits were compared
on g5436-g5465 (916 steps):

| fit | exp_avg x | exp_avg_sq x | predicted clip rate at 1.0 (logged 81.0%) |
| --- | --- | --- | --- |
| normal, mean + P95 (used) | 1.497 | 2.425 | 80.7% |
| lognormal, median + P95 | 1.406 | 2.147 | 78.0% |
| lognormal, median + mean | 1.446 | 2.309 | 74.7% |

All three put the settled step at 0.95-0.96x today's. Without the rescale the first PPG phases would step about
1.4-1.5x that size until the second moment caught up (about 1,000 steps). The revision refuses a fit whose predicted
clip rate is more than 10 points from the logged one.

## Gradient-norm logging (follow-up request, code-only)

- The era heads report their own pre-clip total gradient norm (what their clip sees) per phase, over every attempted
  head step, restored ones included: `local_scoring_auxiliary.gradient_norms.ppo` and `.ppg`, each with steps, steps
  above the heads' 1.0 clip, mean, median, P98 and max. Until now only `gradient_norm_max`, over both phases, was
  recorded.
- PPO's `actor.gradient_norms` reports P98 in place of P95. The other distributions built by
  `review_metrics.distribution` (advantages, returns, rewards) keep P95.
- PPG's `value_phase.gradient_clipping.pre_clip_norm` reports P98 in place of P95, with the maximum beside it.
- The PPG Adam rescale reads either percentile: the committed history logged P95, and iterations logged from this
  change on carry P98. Each uses its own z (1.645 or 2.054) in the normal fit, so the factors do not depend on which
  was logged.
- Cost: one device scalar per head step (about 100 per iteration) and one host copy per report.

## Speed changes (code-only)

| item | before | change | expected |
| --- | --- | --- | --- |
| 5-iteration cleanup | plan 32.4 s, of which the pin scan 28.4 s (g5464) | one `PassResolver` per pass | scan ~5-8 s; about -4 to -5 s per iteration on average |
| commit cleanup | `owner_cleanup` 6.2 s every iteration | teacher check only for ages retiring now | most of the 6.2 s; `owner_cleanup_parts` shows the rest |
| full GC pass | 15-16 s every iteration (4.6 s after a restart, rising) | cycles broken; full pass every 5 iterations | about -12 s per iteration on average |
| rating slot | 102.5 s at g5450 (bootstrap 42.7 s, intervals 2.1 s, replay checks 139 worker-s) | no bootstrap; outcome-only replay read | about -45 to -48 s per slot, every 20 iterations |

**Pin scan.** Every reference went through `project_path`, which re-resolved the root, re-read and re-validated the
relocation metadata, lstat-ed every parent directory and ran a final `realpath`. The live lab state alone holds 1,498
path strings (992 teacher-label paths, each resolved twice). `learning/paths.PassResolver` checks the root, the
relocation file and each directory once per pass and returns the same paths with the same refusals (`_parts` already
refuses `..`, and every existing component is lstat-ed and refused if it is a link, so the final `realpath` cannot
change the answer). On a scratch run built from the lab's g5466 state on the laptop, the scan found the same 9,524 pins
in 12.5 s instead of 74.5 s. A new resolver is built for every pass, as before.

**Commit cleanup.** `TeacherBuffer.committed` re-checked every retired label shard since g4970 (975 of them: hash the
shard, re-read its receipt and `owner.json`) at every commit, and deleted nothing: the state's own `retired` list
pins them. It now checks only the ages it still holds rows for, the ones retiring this commit (about 2). The commit
report gains `owner_cleanup_parts` (seconds per owner).

**Full GC pass.** The pass walked the whole heap (about 14M long-lived objects) to free the ~3.7M cyclic objects each
iteration made; the object counts were flat (13.8-14.2M frozen, 3.72-3.79M unreachable over g5400-g5460). On the PPG
phase fixture the recurring cycles were two closures over `self`:

- `SearchRows.prepare` built `Corrections(lambda ix: self.corrections[ix], ...)`, so every search pool (rows, era maps
  and three device tensors, 36k roots on the lab) waited for the full pass. It now passes the tensor's own
  `__getitem__`.
- `InputAudit.check` registered `weakref.ref(row, lambda dead: self._forget(key, dead))`, so every audit and its
  per-row proofs became one cycle. The callback now holds the audit weakly.

The fixture's recurring cyclic garbage per PPG phase fell from 2,087 objects to 131 (standard-library JSON-encoder
closures); a PPO update leaves 5. The full pass now runs every 5 iterations (`BRASS_GC_FULL_EVERY`, 1 restores every
iteration). In between the heap stays frozen with young passes only, as in the tail after collection. If a pass finds
more than 500k cyclic objects per iteration since the one before, it runs every iteration again until it finds less.

**Rating slot.** `plackett_luce.estimate(bootstrap=False)` does the full fit only: every interval is `None`, the
pairwise differences are not listed (that list was most of the 24.9 MB report), and the report says
`bootstrap_refits: 0`. The regression alert reads the point estimate, `highest_lower_bound` is `None`, and the plot
shows "No interval (bootstrap off)" and the point change. Rating replays are read for their outcomes only (game id,
seat policies, setup, result) without the full replay checks, the replay hash or the feature cache; the seat and setup
identity checks stay. Native-format shards take the full read. `BRASS_RATING_BOOTSTRAP=1` and
`BRASS_RATING_REPLAY_CHECKS=1` restore each.

## Decisions on items 1 and 4

**MCTS node pool: skipped.** One profiled game on the engine harness (`tmp/engine-perf`, 32,128 simulations, same
search digest as the uninstrumented build) with an allocation counter on each span:

| | allocations per simulation |
| --- | --- |
| whole search | 2,341 |
| backup | 2.1 |
| node insert | 0.1 |
| selection scratch | 16 |
| node statistics Vecs (in evaluate) | about 4 |
| decision materialization / view building | 1,366 / 1,005 |

What a node pool and reused scratch could remove is about 22 allocations, about 1 us of the 200 us engine cost per
simulation, and the CPU forward is about two thirds of teacher search time: about 0.2% of search time.

**Rust writing exports into shared memory: not adopted.** A Linux smoke test (WSL2, `tmp/g5466-shm/smoke.py`, 42.7 MB
per game, median of 12 games, twice) of the three ways to move one export:

| design | lane | worker | publisher | total per game |
| --- | --- | --- | --- | --- |
| today: pipe, then the lane copies and hashes into a lease | 122-129 ms (83 ms of it the pipe) | pipe writes | 34 ms | about 200 ms |
| Rust writes into a fresh lease | 0.5 ms | 43 ms | 44 ms | about 88 ms |
| Rust writes into a reused preallocated segment | 0.4 ms | 33 ms | 43 ms | about 77 ms |

That saves about 110-125 ms of CPU per game, 35-40 CPU-s per iteration: about 6% of the ordinary path's CPU (lanes 315 s
plus publishers 263 s for 320 games at g5465). The 83 s collection is set by the 54 teacher tables (74 s), which hold
the CPUs, and by lanes starting late (last lane ready at 43 s), so the expected gain is at most 1-3 s per iteration,
for a cross-language change to the export protocol that must stay byte-exact. Creating segments costs 0.5 ms on Linux;
the lab's 166 ms `shared_mapping_setup_seconds` is timed in the busy trainer process when it issues the lease, not on
the lane.

## Cost

| change | time | memory |
| --- | --- | --- |
| Rail-end in PPO | about +0.5-1 s per iteration (freeze copy 8.0 s and PPO forward/backward 17.6 s today; the forward already computes the outputs) | frozen outputs 2,140 -> 3,202 floats per row, 1.07 -> 1.60 GB host |
| PPG clip 2.2, rescale | 0 (one multiply at load) | 0 |
| fixed head rate | 0 | 0 |
| GC cadence | -12 s per iteration on average | cyclic garbage of up to 4 iterations waits for the next pass (small after the fixes; the fallback bounds it) |
| pin scan, commit check | -9 to -10 s per iteration on average | 0 |
| rating slot | about -2.3 s per iteration on average | no rating intervals |

## Checks after adoption

- The trainer prints `g5466 auxiliary revision: PPG clip 1 -> 2.2; PPG Adam exp_avg x..., exp_avg_sq x...`; the first
  PPG report carries `adam_rescale` (factors near 1.50 and 2.42; they are recomputed from the 30 iterations before the
  adoption generation).
- `value_phase.gradient_clipping`: `max_norm` 2.2, `clipped_percent` near 5% instead of 81%. PPG takes bigger steps,
  so watch its KL checks and beta.
- `value_phase.era_head_rate.rate` equals the PPO rate; `local_scoring_auxiliary.companion_lr.ppg` equals it too (it
  was 3.9e-9, the end of the cosine).
- `actor.value_preservation.floats_per_row` 3,202; `rail_end_heads.in_ppo_preservation` true.
- `iteration_gc.full_collection.ran` false on four iterations in five; on the fifth, `unreachable_per_iteration`
  below 500k (otherwise the pass returns every iteration by itself).
- `commit_timings.owner_cleanup_seconds` (was 6.2 s) and `owner_cleanup_parts`; at the next 5-iteration cleanup
  `cleanup.timings.references_seconds` (was 28.4 s).
- The next rating slot's `timing-report.json`: `bootstrap_seconds` about 0, `replay_validation.worker_seconds` far
  below 139.

## Tests

- `tests/Recovery/test_auxiliary_revision_g5466.py` (10): recipe and transitions, the fit against a Monte Carlo sample
  and the logged clip rate (P95 and P98 records), the norms read from the run's files, the PPG Adam rescaled once
  (receipt survives a checkpoint), the rescale keeping the step size across the clip change, Rail-end frozen and
  preserved, the era heads' per-phase norms with P98, PPO's P98, and a full `ppg_g210.fit` with every era-head PPG
  step at the fixed PPO rate.
- `tests/Recovery/test_iteration_speed_g5466.py` (10): the pass resolver against `resolve_project_path` (paths and
  refusals), the pin scan, the GC cadence and fallback, both cycles gone, the commit check, and the rating slot without
  bootstrap or full replay checks.
- `test_iteration_speed_g3658`: the two GC-window tests run at `BRASS_GC_FULL_EVERY=1`, the g3658 behavior they
  describe. `test_rating_retention`'s cumulative-fit test runs at `BRASS_RATING_BOOTSTRAP=1`: it checks that
  compaction keeps the bootstrap samples. `test_optimizer_revision_g5402`'s clip-report test reads P98.

With the related suites (25 modules, 269 tests), 11 fail, all on unmodified HEAD too: the 9 already in
`docs/KNOWN_TEST_FAILURES/KNOWN_TEST_FAILURES.md`, and 2 `test_policy_anchor` configuration tests, confirmed with
`git stash -u` and added there.

## Execution identity

`3aea42362141ce2bf63a2b039780b8931acda542f82af179862d9ae4d9bc7c6e` (predecessor `a7aad72d...`), entries
`g5466_auxiliary_revision` and `g5466_iteration_speed` in `configs/learning/b07-replay-location-review.json`.

## Command

```
python scripts/continue-rust-g4970.py --expected-generation 5466 --iterations 10 --share-target 0.065 --auxiliary-revision
```

Use the run's committed generation in `--expected-generation`. `--optimizer-revision` is already adopted and is not
needed again.
