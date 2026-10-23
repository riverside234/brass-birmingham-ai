# PPG beta hotfix and feedback, g600

The g600 report records effective PPG cloning **beta = 8**, even though the
configuration's starting beta is 1. This revision explicitly resets the saved
effective beta to **1 once**, through reviewed continuation. Changing the starting
config value alone would not change the restored adaptive state.

`ppg-value-phase/11.0.0` keeps the existing KL-triggered doubling/retry logic and
the 1–32 bounds. After a scheduled phase with accepted training, halve beta for
the **next** phase if both final retained overall and age-0 KL are strictly below
`0.5 * 0.002 = 0.001`. Do not decrease after a KL trip/retry, with no accepted
steps, or on a waiting iteration. Equality at 0.001 does not qualify. Minimum
beta is 1. The 0.002 safeguards and verified keep-best rollback remain active.

The config adopts this feedback contract; the checkpoint's actor/value options
record its adoption. Restarts from the old g600 checkpoint consistently adopt 1;
after the first new checkpoint, later adapted beta values resume unchanged.
The migration is reported with `beta_before=8`, `beta_after=1`. Each subsequent
iteration records `beta_feedback`, `beta_clone_final` (used during the phase),
`beta_clone_next`, decision and reason. The console shows `PPGbeta=used->next`.
Fresh configurations also use this feedback rule, starting from 1.

All other settings remain: PPG every eight committed iterations; eight uniform
replay ages and eight-age holdouts; constant 0.35 × actual PPO LR; held-out checks
every 16 steps, scheduled KL checks every 32; five-epoch cap and patience 1.
PPO preservation beta-V remains 1. No model/Adam/RNG, replay, phase-clock, progress,
population, economic baseline or gate evidence is reset. The last phase was g596;
the next phases remain **g604, g612 and g620**.

## Validation

**20 focused tests passed** (19-test suite in 99.8 seconds, plus the new-contract
KL-retry test in 28.6 seconds). Logs: `artifacts/ppg-beta-g600-tests.log` and
`artifacts/ppg-beta-retry-g600-test.log`. The isolated optimizer test confirmed
beta increases from 1 to 2 after an injected age-0 KL trip, retains a safe state,
and does not immediately decrease again. All execution used temporary authored
CPU fixtures, not the lab's g600 checkpoint.

Focused tests cover the strict half-cap boundary, both KL signals, minimum and
maximum bounds, suppression after a trip/retry, zero accepted steps and waiting
iterations; explicit g600 source/config admission; idempotent config generation;
one-time effective-beta reset with unchanged weights/Adams/RNG/replay/schedule;
retention of later adapted beta on restart; and real isolated auxiliary updates
with injected KL failures. The pre-existing DCPG and companion rollback tests
are also run. Full training and evaluations are not launched.

## Continue from g600

After syncing the code, run in the lab repository to continue **g601–620**:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 600 --apply && \
python scripts/configure-dcpg.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 600 --adaptive-beta \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-dcpg-beta1-g600.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-dcpg-beta1-g600.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 600 --iterations 20
```

The config helper verifies the committed checkpoint checksum and writes a new
config only. The effective-beta hotfix takes effect on reviewed training resume;
the lab checkpoint is not rewritten by this helper. The local export contains
reports, so production adoption runs on the lab machine using the command above.
Do not reset the replay buffer or re-adopt the local scoring heads. For later
continuation, keep this config and use the actual committed generation guards.
