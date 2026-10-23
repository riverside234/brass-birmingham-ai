# Delayed critic preservation and eight-iteration PPG, g590

The later [g600 beta hotfix](PPG_BETA_G600.md) resets the saved cloning beta to 1
and adds clean low-KL decreases while retaining adaptive increases. Use its
continuation command for the current run; the g590 validation below is historical.

The exported run is committed at **g590**, with its last PPG phase at **g588**.
This explicit revision keeps all eight replay ages and moves the next phases to
**g596, g604, g612, ...**. It preserves the original schedule origin, completed
phase count, weights, all Adam states, RNG, curriculum, population and lineage.
No production checkpoint or run state was modified in this checkout.

## Effective settings

| Setting | Revision |
| --- | --- |
| Value recipe | `ppg-value-phase/10.0.0` |
| PPO value objective | `ppo-critic-gradient/2.0.0`, frozen-output preservation |
| Preservation coefficient | beta-V = 1; core value weight remains 0.5 |
| PPG interval / train / validation | 8 committed iterations / all 8 ages / all 8 ages |
| Sampling | Uniform age weights; full random permutation each epoch |
| PPG LR | Constant **0.35 times the actual final PPO step's LR** in that iteration |
| PPG LR curve | No warmup or decay; retries use the same rate |
| Loss / scheduled policy checks | Every 16 / 32 planned optimizer steps, plus short epoch tails |
| Unchanged limits | Five-epoch ceiling, patience 1, 1.5-SE paired allowance, overall and age-0 KL <= 0.002, beta-clone maximum 32 |
| Optimizers | PPO Adam retained; value Adam still resets at phase start; joint keep-best/rollback includes value and companion Adam |

For g590's actual PPO LR of `6.25e-6`, the corresponding PPG rate would be
`2.1875e-6` throughout its phase. This is an example, not a constant across
iterations: PPO warmup and the applied LR-controller multiplier are included.
The rate is captured before the controller chooses the next iteration's rate.
An iteration with no PPO steps cannot supply a linked auxiliary update.
The companion's separate configured head LR remains `1e-4`, now constant within
PPG as the phase-relative multiplier is one. Its configuration was not changed.

## PPO critic behavior

Immediately before advantage calculation, an eval/no-grad pass records every
critic row's four-seat win distribution, bounded rank predictions, VP and utility,
using its true recorded descriptors without dropout. Learner, forced, snapshot
and eligible branch rows are represented. Held-out rows can supply frozen
advantages/diagnostics but never enter the critic training batches.

GAE uses the **same cached utility values**, not another later model prediction.
Actual behavior probabilities and the separate raw-T1/opening PPO reference
remain verified and frozen exactly as before.

Core preservation is:

```text
0.5 * (KL(old_win || new_win)
       + 0.25 * Huber((new_rank - 1)/3, (old_rank - 1)/3)
       + 0.10 * Huber(new_vp/100, old_vp/100)
       + MSE(new_utility, old_utility))
```

The core currently has no categorical rank logits: rank is a continuous bounded
prediction per seat. The implementation therefore retains the existing rank/VP
regression forms and scales rather than inventing a new head or applying MSE to
unscaled VP. Win probabilities are sufficient for the requested categorical KL;
soft-target CE has the same gradient. Architecture and checkpoint tensors are
unchanged.

Ownership/VP companion outputs are also frozen: 49 industry slots and 69 links,
with five owner probabilities and four normalized VP predictions each. Their
existing owner and VP weights are retained, using KL and normalized Huber to
their frozen predictions. No future ownership map or fresh scoring label enters
this PPO loss. Existing apply/shadow/off behavior and activation ramp are retained.

Preservation gradients reach the shared state/entity encoders, descriptor encoder
and value heads. The actor, entropy and rehearsal losses are unchanged. The
ordinary utility-calibration fit is disabled in this mode: the g590 log showed
201 such fresh-target head steps before PPO, which would otherwise violate the
delayed-critic boundary. All fresh core/local value targets are fitted during PPG.
The pre-PPO snapshot is recreated from the critic left by the preceding iteration
and any completed value phase; no snapshot survives an interrupted collection.

## Separate checks and retention

Loss checks at steps 16, 48, ... can create provisional selections. Scheduled KL
checks occur at multiples of 32 and epoch tails. The final chosen weights receive
their **own** complete-policy KL check, including selections from earlier steps.
A safe later endpoint does not certify an earlier model.

A second bounded selection holds only explicitly KL-verified states. If a
provisional selection or retry rollback target fails either cap, its weights,
value Adam and companion state are discarded in favor of that verified selection.
The final retained state is checked again before commit. Reports distinguish
scheduled checks, retention checks and rejected provisional states. This can add
checks when retaining an earlier state; it does not make the regular KL interval
16 steps or waive either cap. Bounded beta retries, fixed post-PPO raw policy
references, corrected strata sampling and immutable MC targets remain in place.

## Evidence and reporting

`actor.value_preservation` contains the end-of-PPO preservation loss and components,
weighted objective, four-seat utility RMS drift on the ordinary PPO policy probe,
frozen-row/role counts and bytes, actor KL, and one gradient record per PPO step.
Gradient records include exact actor-term and preservation-term shared-trunk norms
and cosine, before clipping. Actor measurements exclude entropy/rehearsal.
Preservation gradients are accumulated separately and recombined before clipping
to avoid cancellation when the preservation gradient is very small.

`value_phase.value_sag` records elapsed committed iterations since the previous
scheduled phase, accepted phase steps, and current ordinary learner held-out EV
both after PPO and after PPG. Branches and snapshot seats are excluded from that
evidence. These fields are retained in compact history. Economic baselines,
current lambdas, R2 state, gate thresholds and existing rolling evidence are kept.
Going forward, only **accepted PPG fitting** counts as fresh critic work for GAE;
PPO preservation is not relabeled as value fitting. Existing zero-step/window
health rules and PPO/PPG KL safeguards remain active.

The available detach-baseline logs give these comparison points:

| Committed generation | Exact raw-T1 actor KL | PPO steps | Actual PPO LR |
| --- | ---: | ---: | ---: |
| 589 | 0.00498565 | 50 | 6.25e-6 |
| 590 | 0.00660766 | 48 | 6.25e-6 |

New ordinary-run results should be compared with those and a broader supported
window. Synthetic smoke results cannot establish a better actor/critic tradeoff
or stronger play.

## Official DCPG cross-check

The [official DCPG update](https://github.com/snu-mllab/DCPG/blob/main/dcpg/algos/dcpg.py)
freezes the old critic before policy optimization, penalizes value drift through
the shared network, and fits returns in a separate auxiliary phase while cloning
the frozen policy. Its [configuration](https://github.com/snu-mllab/DCPG/blob/main/configs/dcpg.yaml)
sets `value_dist_coef=1.0`. Our cached-output reference is equivalent to a frozen
network on the same deterministic inputs. The requested adaptation extends the
scalar penalty to our existing outcome/utility/local heads and keeps our eight-age
MC replay, true seat descriptors, guarded selection, LR recipe and optimizer-reset
policy. It is not a claim to reproduce the paper's exact algorithm or results.

## Runtime and compatibility

The snapshot has **16 core floats plus 1,062 companion floats per row**, not about
30 floats. At g590's 143,189 critic rows this is **617,430,968 bytes (~589 MiB)**,
stored on CPU and transferred by minibatch. The cache is iteration-local and
excluded from checkpoints and eight-age disk replay. CPU/GPU memory telemetry
and actual bytes are reported. No extra model architecture parameters are added.

PPO now backpropagates preservation through the encoder and does an extra actor
gradient evaluation for the requested per-step diagnostics, plus initial/final
critic scans. Removing the 201-step calibration and halving phase frequency saves
work, while 16-step held-out checks add PPG validation work. The net GPU iteration
time needs new lab measurements; no speedup is asserted here.

The core compatibility identity remains
`ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
Source/config adoption is explicit and narrow. Existing critic replay stays valid;
stale PPO behavior is neither relabeled nor admitted. IL remains
`artifacts/step5/features4-v11-2-batch7-il`.

## Validation

All **10 new DCPG regressions pass** (`artifacts/dcpg-new-regressions.log`,
59.9 seconds). They cover true-descriptor frozen outputs, all head gradients,
absence of fresh PPO/calibration fitting, the unchanged behavior verifier and
frozen GAE utilities, strict configuration, actual-PPO LR capture, separate
loss/KL checks, unsafe provisional selection rollback, and g590 migration.
The gradient comparison uses relative float32 tolerance for differently ordered
norm reductions; it also directly checks the combined parameter gradients.

The broader check covered **72 tests**. After correcting that norm-comparison
tolerance and rerunning its regression, **70 passed**. Two pre-existing archived
resume tests in `test_health_ev_window.Continuation` still request direct source
approval from g55-era implementations (`e2b6bf...` and `c13907...`) to today's
source. Those routes were absent for the previous g590 implementation too; the
source-review guard continues to reject them. This revision preserves all 74,158
previous review entries and adds only the narrow current routes. The requested
g590 config transition, checkpoint-state migration and repeated restart tests pass.
See `artifacts/dcpg-focused-tests.log` and the corrected rerun
`artifacts/dcpg-gradient-test.log`.

The main implementation is in `recovery/value_preservation.py` and
`recovery/dcpg_config.py`, integrated through `updates.py`, `actor_value.py`,
`ppg_auxiliary.py` and their configuration/resume validators. Companion accounting,
console/compact history, the lab instructions and explicit source-review registry
were updated. `scripts/configure-dcpg.py` generates the continuation config;
`scripts/smoke-dcpg.py` and `tests/Recovery/test_dcpg.py` provide isolated validation.

The [isolated smoke report](../artifacts/dcpg-g590-smoke-verified/report.json)
completed 16 authored iterations on CPU in 389.6 seconds, with two CPU threads.
It ran 128 PPO steps and PPG only at iterations 8 and 16. Checkpoint reloads at
4, 8, 12 and 16 preserved weights, PPO/value/companion Adam, RNG, phase clock and
epoch state exactly. All earlier checkpoint checksums stayed unchanged. Expired
replay was deleted after commit; both phases trained all eight ages, including
eligible snapshot and branch rows, with 48 ordinary learner holdout games.

| Synthetic phase | Retained PPG steps | Constant PPG LR | Overall KL | Age-0 KL |
| --- | ---: | ---: | ---: | ---: |
| 8 | 35 | 1.4127273e-6 | 0.00001297 | 0.00001697 |
| 16 | 35 | 2.1000000e-6 | 0.00004958 | 0.00003154 |

These tiny epochs have only seven steps, exercising short-tail checks. Separate
focused regressions exercise the 16/32/48 boundaries and ensure an unsafe
16-step selection cannot borrow a safe 32-step endpoint's KL result.

Across the smoke, end-of-PPO core preservation loss ranged from 2.24e-7 to
4.15e-5; utility RMS drift ranged from 0.000169 to 0.002332. Actor KL ranged from
2.11e-6 to 0.0001412. All 128 steps recorded trunk gradient diagnostics; their
cosines ranged from -0.165 to 0.116. The report includes EV before/after each
phase and the elapsed-iteration sag series. These fixture measurements cannot
be compared directly with the real g589/g590 actor KL or establish playing
strength or a reliable EV trend.

Peak sampled process high-water during the value phases was 535,166,976 bytes;
peak replay disk usage was 27,660,569 bytes. The frozen-output cache itself was
702,856 bytes for 163 fixture rows. These are tiny-test resource measurements,
not lab-scale estimates. The exported lab directory contains reports rather
than its complete checkpoint or GPU runtime, so full g590 checkpoint execution
and GPU timings remain to be measured on the lab machine.

## Continuation

After syncing the revised code, run from the lab repository to continue **g591–610**:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 590 --apply && \
python scripts/configure-dcpg.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 590 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-dcpg-g590.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-dcpg-g590.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 590 --iterations 20
```

The recovery command archives an incomplete next iteration; it does not rewind a
committed generation. The config helper verifies the committed checkpoint checksum
and writes only the new config. Do not repeat `--fresh-ppg-buffer` or the old local
head adoption command. For another interruption, use the actual committed
generation in the recovery/train guards and continue with this same config.
