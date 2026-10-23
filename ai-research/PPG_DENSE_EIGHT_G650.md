# Dense eight-iteration PPG continuation from g650

The continuation starts from **g650**, as subsequently confirmed by the user.
The g650 export retains full resume checkpoints for g649 and g650; the files
found for g640 were PPG data/label caches, not a full model/Adam/RNG checkpoint.
No production run or checkpoint was modified in this implementation workspace.

## Recipe and checkpoint selection

`ppg-value-phase/13.0.0` uses eight committed iterations, uniform age weights,
and a phase every eight iterations. Loss acceptance uses ordinary learner
held-out games from **all eight ages**, normally 144 games under the 8/8/2 quotas.
Whole parent/branch families remain excluded from auxiliary fitting and policy
cloning/probes. Eligible learner, snapshot and branch critic rows still train
against their recorded MC targets and original seat descriptors. Every epoch
reshuffles in the main process.

The user confirmed `ppg-last-accepted-loss/1.0.0`:

- Compare each loss check with the **previous accepted checkpoint**, using the
  same fixed held-out games and existing 1.5 paired-SE allowance.
- Save each accepted checkpoint, including a plateau or a tolerance-safe loss
  increase. Minimum loss is reported for diagnosis only.
- Reject a mean loss increase above 1.5 SE. With patience 1, stop and restore
  the last accepted core weights, value Adam, auxiliary heads and auxiliary Adam
  together. A rejected final interval is never retained.

The loss checks remain every 16 steps (also at epoch tails), scheduled KL
measurements every 32 steps, and the epoch ceiling remains five. PPG still has
**no hard KL cap**, with the saved fixed beta (1 in the g650 run) and no beta
adaptation. Complete-distribution KL, including age 0, is still measured.
The constant PPG LR remains **0.35 × that iteration's actual last PPO LR**.
Value Adam still resets at phase start as in the existing recipe.

Branch probability is **0.5% per eligible decision** (`rate_per_decision=0.005`),
for both this continuation and newly generated defaults. The g650 export actually
records 1.0%; the separate narrow migration also supports the prior 1.5% setting.
It preserves opening temperatures, post-fork counters, entropy baseline/floors,
rollout origin, maximum branches per game and all other branch options.

PPO LR feedback cuts now divide the multiplier by **1.25** (multiply by 0.8),
instead of 1.5. This applies to all existing protective-cut triggers: a stop
before half the planned steps, two early stops in the three-iteration window,
or post-PPO KL above 1.5 times target. The triggers and PPO hard guard are unchanged;
a single late KL stop still follows the existing window rule. Cooldown, clamps,
raise threshold and all saved controller evidence stay intact. The decision
report records `cut_divisor`. Current/default config generation uses the new
value; an explicit archived 1.5 configuration retains its historical behavior.
PPG keeps its existing 0.35-times-actual-PPO-LR linkage and fixed beta.

PPO stopping now has an explicit `update.kl_control` contract:
**2,048 fixed nonforced probe rows, exact categorical KL only**. The existing
per-step raw-T1 and actual-behavior exact guards remain, with `target_kl=0.01`.
Each sums over the complete legal-action distribution, then averages across
probe positions. If a tiny collection has fewer eligible rows, use all available
rows and report the actual support; never pad by duplicating decisions.

The g650 export already used 2,048 rows and fed the exact raw-T1 value
`0.007636939611984417` to the LR controller. The removed path is the secondary
**sampled full-dataset epoch stop**, which previously could also set the
early-stop flag and affect LR feedback. Sampled chosen-action KL remains a
diagnostic only for the new contract; nonfinite-policy failures and strict
pre-update behavior-probability verification are retained. Archived explicit
recipes retain their original secondary guard. The fixed probe and old policies
are unchanged throughout each PPO update. Exactness refers to the action sum
at each probe state, not an exhaustive average over every training state.

Reports retain exact T1/behavior KL and sampled diagnostics separately, with
requested/actual probe rows and `sampled_kl_stop_enabled=false`. The previous
LR window, multiplier and cooldown are preserved; newly collected evidence
uses the exact-only stopping contract. No extra probe forward pass was added.

The reward remains `event_cost=0.04`, `max_cost=0.10`. Other PPO/DCPG settings, policy collection,
opponents, IL identity, model layout, GAE/snapshot evidence, economic baselines,
warmup, LR-controller state and all unrelated settings remain unchanged.

Successive tolerance-safe loss increases can accumulate with this requested
selection rule. A final loss higher than the phase's minimum is therefore
intentional, and both losses and the comparison checkpoint are reported.

## Cache and preparation performance

New caches use the original dense FP32 `critic.npy` / `actions.npy` mmap layout.
There is no feature bit unpacking, constant-column reconstruction or descriptor
dictionary reconstruction in minibatches. The scoring JSON sidecar is decoded
once when an age is opened; its immutable target maps are reused.

PPG reuses the persistent PPO preparation pool (four single-thread CPU workers
in the saved g650 config). Workers read the admitted, sealed dense arrays
directly, without copying the full replay to PPO's temporary input staging file.
Prefetch is bounded by the worker count and results are consumed in original
order. RNG, row sampling and all model/GPU work stay in the main process.

The initial file-output transport was slower in the local CPU benchmark. Dense
PPG now uses bounded shared-memory result slots in the same preparation service,
with private tensor copies before a slot can be reused. Targets and descriptors
travel with the corresponding input batch. Slots and pending work are released
on normal completion, early stopping and exceptions. Existing PPO transport is
unchanged.

Ownership labels use vectorized metadata gathers and unique sealed
game/era/seat-mapping targets. The bounded CPU target cache holds up to 65,536
entries, avoiding the former 4,096-entry churn over an eight-age phase. Missing
labels, ownership rotations, validity masks, VP scales, losses and gradients
retain their original semantics. No model output or autograd graph is cached.

## Adoption and continuation

The config helper writes only a new configuration. It checks committed generation
and checkpoint hash, changes only `value_phase`, branch probability, the PPO
LR cut divisor and `update.kl_control`, and refuses
unrelated changes through the reviewed source-transition contract.

At trainer startup, after successful checkpoint restore and durable reviewed
adoption, only the owned old PPG caches are retired. Eight **fresh dense** ages
are required; old targets are not reconstructed or relabeled. A restart before
the first new checkpoint repeats this migration idempotently. A restart after
commit restores the persisted schedule instead of moving its origin again.
Prior phase counts and the last-phase identity remain recorded.

From g650, the first new phase is **g658** (data g651–658), then **g666**
(g659–666). The following command runs g651–670:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 650 --apply && \
python scripts/configure-ppg-eight-fast.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 650 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppg8-dense-g650.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg8-dense-g650.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 650 --iterations 20
```

The first command archives an interrupted, uncommitted collection; it does not
rewind committed training. Fresh config generation also uses the new dense-eight
recipe; archived explicit sixteen-age helpers/configurations remain supported.

## Validation scope

Focused regressions cover exact configuration bounds and source review, restart
cadence, startup cache retirement, ordered process batches with duplicate row
indices, legal selectors, unchanged targets/descriptors/RNG, ownership target
and gradient parity, and restoration of the last accepted state even when its
loss exceeds the historical minimum. Old minimum-loss and hard-KL recipes keep
their historical selection behavior.

The isolated smoke uses eight small authored iterations, one real auxiliary
phase and an exact replay of that phase after a g7 checkpoint restart. Its
48 held-out games reflect reduced fixture quotas, not the production 144-game
window. It checks core/companion weights, both kinds of Adam, RNG and gate state.
Benchmark results are local CPU measurements, not a claim about lab CUDA speed,
training stability or playing strength.

## Measured validation and performance

- The broader regression run exercised 78 tests across dense replay, archived
  minimum/one-SE selection, sixteen-age replay, DCPG, beta feedback, ownership
  heads and PPO preparation. Its two outstanding cases were a fixture's epoch-tail
  expectation (55, not 64 steps) and a missing carried-forward source approval
  for the archived g470 helper. Both were corrected and rerun with the focused
  tests; the epoch-tail loss check itself was correct.
- The PPG/branch focused run passed **8/8 tests in 58.37 s**. It also checks the g650
  branch-rate-only adoption, new-run default, unchanged entropy state and
  rejection of unrelated edits. Raw output:
  [`final-branch-tests.log`](../artifacts/ppg-dense-eight-g650-final-branch-tests.log).
- After the additional PPO LR request, **19/19 LR and resume tests passed in
  25.90 s**, including the combined g650 helper, all three cut triggers,
  unchanged late-stop boundary, clamps/cooldown, preserved Adam/RNG/streaks,
  shadow/non-PPO isolation and exact checkpoint restart. Explicit archived
  divisors still produce their original cuts. Raw output:
  [`lr-cut-tests.log`](../artifacts/ppg-dense-eight-g650-lr-cut-tests.log).
- The final combined regression run passed **25/25 tests in 64.27 s**: actual
  2,048-row full-action KL, independent T1/behavior guards, diagnostic-only
  approximate KL, nonfinite failures, legacy guard compatibility, all dense PPG
  regressions and the combined g650 config/Adam continuation. Raw output:
  [`exact-kl-tests.log`](../artifacts/ppg-dense-eight-g650-exact-kl-tests.log).
- The isolated CPU smoke passed real PPO/PPG and ownership-head training, then
  reproduced the complete g8 result after restoring g7. Core/auxiliary weights,
  both optimizer states, RNG and schedule/gate state matched exactly. The
  incoming checkpoint was unchanged. There were 35 retained PPG steps over
  five small epochs, with 869 distinct training rows and 48 held-out games.
  This fixture is deliberately much smaller than a production phase.
- Final smoke wall time was **83.51 s**, including the restart and benchmark; its
  auxiliary phase took **19.60 s**. Main-process peak RSS was **513,658,880 bytes**;
  this excludes worker RSS. No GPU was available for this local measurement.

Same authored tensors and index order, 32 batches of 512 rows, warmed workers,
median of three timed trials:

| Input preparation | Initial trial, seconds | Final trial, seconds |
| --- | ---: | ---: |
| Packed cache, synchronous | 0.59113 | 0.51422 |
| Dense mmap, synchronous | 0.55609 | 0.44527 |
| Dense mmap, four workers and shared-memory output | 0.48487 | 0.36402 |

The preparation path took **18% less time** than packed synchronous preparation
in the initial trial and **29% less** in the final trial. Ownership metadata/label
assembly fell from **0.02641 s to 0.00904 s** (about **2.9x**) initially and from
**0.01803 s to 0.00502 s** (about **3.6x**) in the final trial. Complete tensor, target, descriptor,
ownership-label and gradient parity were checked. The failed file-IPC experiment
was replaced before these reported measurements.

Raw smoke and timing data:
[final report](../artifacts/ppg-dense-eight-g650-exact-smoke/report.json) and
[initial trial](../artifacts/ppg-dense-eight-g650-shared-smoke/report.json).
The final smoke includes the exact-only PPO guard and unchanged checkpoint
restart behavior. Timings vary with local CPU conditions; use within-trial paired
comparisons, not a comparison of the two trials' total phase times. The smaller eight-age phase processes
roughly half the former sixteen-age data but runs twice as often. These component
timings therefore do **not** establish a full server-phase speedup or reduced
amortized training time; check the next production log for those measurements.
