# Four-iteration PPG and detached PPO critic, g580

The g580 export is committed at generation 580. Its last PPG phase was g576,
its next phase was g584, and its last actual PPO learning rate was 6.25e-6.
The supplied export contains reports, not the lab's full checkpoint or replay
files. No existing training checkpoint, cache or run state was edited here.

## Effective recipe

| Setting | Applied revision |
| --- | --- |
| Contract | `ppg-value-phase/9.0.0` |
| Cadence | Every four committed iterations; from g580: g584, g588, g592, g596, g600 |
| Training / validation | All eight eligible cached ages, uniform weights; whole-game holdouts and branch families remain excluded from fitting |
| Shuffle | Fresh `torch.randperm` over all eligible rows each epoch, including repeated phases; no sampling replacement for critic rows |
| LR peak | Actual last PPO step's LR in the same iteration, including PPO warmup and the applied multiplier |
| PPG curve | Existing linear 0.1x-to-1x warmup across the first 32 planned steps, then linear decay to 0.3x at the cap; retries reuse the same rates |
| PPO value path | Detach shared encoder outputs before outcome, utility and ownership/VP heads; continue training those heads and critic-only descriptors |
| PPG value path | Shared encoder gradients remain active; policy cloning and both exact KL guards remain active |
| Preserved throughput recipe | Fixed five-epoch ceiling, 32-step KL/heldout checks, patience 1, 1.5-SE paired allowance, overall and age-0 KL caps 0.002 |

The first phase of a fresh run waits until eight collections exist, then runs
every four. Existing g580 replay is retained, so g584 uses g577–584 and g588
uses g581–588. Both fitting and holdout evidence span all eight ages. Expired
payloads are still deleted only after a new checkpoint commit.

The linked peak is captured immediately after PPO, before the LR controller's
decision for the next iteration. A raise/cut therefore cannot retrospectively
change this phase's reference rate. With a 6.25e-6 peak, a sufficiently long
phase starts at 6.25e-7, reaches 6.25e-6 on step 32, and ends at 1.875e-6.
The actual rate follows the future iteration's PPO; g580's number is an example,
not a new fixed setting. No PPO steps means no linked auxiliary update.
The separate companion Adam retains its configured head rate and the existing
phase-relative curve. Linking the core value Adam does not change that head LR.

## Official implementation cross-check

Compared with OpenAI's [ppg.py](https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/ppg.py)
and [ppo.py](https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/ppo.py):

| Detail | Review result |
| --- | --- |
| Detachment | Official `arch="detach"` passes detached encoder features into the true value head. Our PPO follows that boundary for all critic heads; actor gradients remain enabled. |
| Auxiliary representation learning | Official code has a separate auxiliary value head with an encoder gradient path. We retain existing heads and enable their encoder gradients during PPG to preserve v11.2 checkpoints and outputs. This is an intentional adaptation. |
| Policy cloning | Official auxiliary loss uses KL(old policy \|\| updated policy). Ours freezes raw temperature-1 complete legal distributions after PPO, under eval/no-grad, and keeps this direction for the whole phase. |
| Epoch order | Official auxiliary minibatches reshuffle segments each epoch. Ours reshuffles eligible rows across all eight ages and preserves saved RNG on restart. |
| Targets | Official PPO forms value targets from GAE. Our existing recorded Monte Carlo targets and seat descriptors stay immutable. |
| Optimizers and buffer | Official code has a separate auxiliary Adam and clears its segment buffer after a phase. Our requested adaptation retains rolling disk replay, phase-start Adam reset, linked warmup/decay, and guarded model/Adam selection. |

Our additional controls remain explicit: ordinary-game paired acceptance,
disjoint cloning/probe samples, population correction for sampling strata,
branch/fork eligibility, overall and age-0 KL caps, bounded beta retries and
joint rollback of core/value Adam/companion weights/companion Adam. Current
ordinary holdouts alone feed current GAE evidence; overlapping eight-age PPG
validation windows are not counted as independent GAE iterations. This review
does not claim identical dynamics to the official implementation.

## Checkpoint and configuration behavior

The new signed config adds `update.value_gradient` and sets
`value_phase.learning_rate_source="ppo-last-applied"` with `learning_rate=null`.
The core model layout, parameters, IL identity and inference outputs are unchanged.
Core learning compatibility stays
`ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
The versioned cadence keeps the original origin, actual last completed phase and
phase count, plus a durable cadence-transition receipt. Restarting g580 before
the next checkpoint repeats the same migration, rather than advancing or resetting it.

A peak-rate receipt lives in the value Adam parameter group. Resume validates
the saved step rate against that saved peak, not a guessed PPO base. The migration
preserves every Adam tensor and RNG state; only the normal next PPG phase resets
value moments as before. Adopting the preceding throughput recipe archives old
adaptive epoch votes and fixes the cap at five. All economic, LR, entropy, GAE
and snapshot evidence remains unchanged.

Reports retain the peak's source generation, actual step rates, cadence, replay
ages and PPO gradient mode. Console adds `value_peak=...[PPO]` and
`critic=PPO-heads/PPG-shared`. The configuration helper reads the committed run
under its lock, verifies the checkpoint hash and writes only a new config.
It preserves the completed IL directory and every unrelated setting.

## Continuation

After syncing sources and the matching review registry, run from the lab repo:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 580 --apply && \
python scripts/configure-ppg-phasic.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 580 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-ppg-phasic-g580.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-phasic-g580.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 580 --iterations 20
```

This continues g581–600. The recovery helper archives unfinished collection work
without rewinding the committed checkpoint; it is a no-op when there is none.
Do not repeat auxiliary-head adoption or `--fresh-ppg-buffer`. Source-freshness
checks continue to reject unfinished old-source PPO trajectories.

## Validation and runtime scope

Focused regressions cover exact prediction parity at the detach boundary, zero
shared-encoder critic gradients, trainable heads, restored PPG/actor gradients,
all companion modes, strict defaults, uniform epoch coverage, dynamic rate
curves, g580/fresh schedule boundaries, preserved optimizer/cache/gate state,
and narrow idempotent configuration generation.

Validation passed **44 focused regressions** (43 together plus one separate LR
handoff regression): the new phase/gradient
checks, preserved throughput-cache parity, joint companion rollback, raw-policy
KL, branch/opening behavior and cache rejection, eight-age sampling and the
canonical source registry. The LR handoff test exercises actual multiplied PPO
steps followed by a next-rate change, verifying that the PPG peak remains bound
to the rate that was used.

Implementation identity:
`8dd7b4737bcbcd85d1f935c63e66687a125734094dbd065fd8421b18bc083b3c`.
The review registry preserves all 74,142 existing explicit entries, including
the preceding throughput work, and adds 16 narrow entries. The source-transition
helpers continue to require checkpoint, population and source-freshness checks.

The principal changes are in `recovery/ppg_phasic.py`, `ppg_config.py`,
`actor_value.py`, `ppg_auxiliary.py`, `ppg_epochs.py`, `updates.py` and
`local_scoring_auxiliary.py`. Configuration/resume/history/console integration
is updated alongside the source registry. New entry points are
`scripts/configure-ppg-phasic.py` and `scripts/smoke-ppg-phasic.py`; focused
tests are in `tests/Recovery/test_ppg_phasic.py`. The prior throughput files
and their independent changes remain intact.

The isolated smoke uses twelve tiny authored CPU collections with two PPG phases
and checkpoint restarts at 4, 8, 10 and 12. It exercises two overlapping eight-age
windows, real PPO/value fitting, companion heads, cache expiry and exact resume.
Its measurements do not estimate production GPU throughput or playing strength.

The completed smoke passed in **262.7 seconds** on CPU: **96 PPO steps** and
**70/70 retained PPG steps** across phases 8 and 12. Both phases used eight ages,
48 ordinary held-out games, and all 869 / 871 eligible critic rows respectively.
Their peaks tracked the actual warming PPO rate: **4.03636e-6** then
**5.78182e-6**. Overall/age-0 KL was **0.00001221 / 0.00002038** in the first
phase and **0.00004460 / 0.00005376** in the second, below 0.002. Exact restart
checks passed at 4, 8, 10 and 12, including weights, PPO/value Adam, companion
state, RNG, peak receipts and cadence. The phase cache remained about 4.82 MB
on this small fixture; process peak RSS was about 547 MB. These are fixture
measurements, not estimates for the production replay window. The report is
`artifacts/ppg-phasic-g580-smoke/report.json`.

At the same per-phase budget, four-iteration scheduling offers twice as much
PPG work per ordinary iteration as the preceding eight-iteration throughput recipe.
The eight-age disk and phase working sets are unchanged. PPO no longer needs
the critic encoder's backward activations, but wall-time savings are not measured
here. Larger PPG peaks can encounter the existing KL guards earlier; caps and
rollback remain unchanged. No long training or evaluation campaign is run.
