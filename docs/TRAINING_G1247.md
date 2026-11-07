# g1247 continuation: PPG every iteration and beta reset

Continue `joint7-v11-2-g1290-trained-restart` at committed g1247. The new
`ppg-value-phase/24.0.0` restores PPG **every iteration**, starting at g1248,
and a peak LR of **0.65 times that iteration's last actually applied PPO LR**.
The linear 16-step warmup and cosine decay to 0.001× peak remain unchanged.

On adoption only, reset saved beta from **0.25 to 0.125**, recording the old
value and transition generation. The floor remains 0.125; subsequent clean
low-KL decreases and hard-cap increases remain active. A restart after a new
checkpoint preserves adapted beta rather than repeating the hotfix. Retrying
preparation before g1248 is safe. Preserve the five-phase trip window, 0.003
KL cap, early-trip multiplier, loss16/KL32 checks, two-epoch cap, and compatible
two-iteration replay. No cache refresh is needed.

Only the PPG contract, cadence and LR factor change in the signed config.
Model/companion weights, Adam/RNG, progress, population, reward/entropy,
PPO/DCPG, GAE/critic evidence and calibration remain intact. The old cadence
origin is retained in the adoption receipt; every-iteration phases need no
modulo scheduling state. Older signed contracts keep their prior behavior.

The g1247 export reports zero retained PPG steps and `heldout_loss_increased`.
These changes are requested experiments, not evidence that value fitting or
playing strength will improve. Scheduling every iteration doubles fitting
opportunities versus every two; actual runtime depends on rejection timing.

## Continue g1248–g1267

Sync reviewed code to the lab, then run from its repository root:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 1247 --apply && \
python scripts/configure-training-g1247.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 1247 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json && \
CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/step5.py \
  --cuda-visible-devices 7 --device cuda \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1247 --iterations 20
```

The helper verifies the full committed checkpoint hash, rejects pending work
and conflicting output, and writes only a config. The beta hotfix is applied
during reviewed training initialization, then saved with the next checkpoint.
GPU 7 is logical cuda:0. Once the run advances, reuse this config with the
actual committed generation; do not rerun the generation-pinned helper.

## Validation

`tests/Recovery/test_training_g1247.py` checks exact configuration scope, actual
PPO-linked LR, beta safeguards, retry-safe reset, preserved replay/optimizers/
RNG/GAE state, and no repeated reset after an adapted beta is saved. Its small
CPU smoke runs two consecutive real PPG phases with synthetic rows. The
g1241 tests cover unchanged older-contract cadence and continuation behavior.
No production checkpoint edits, long training or evaluation games are run.

Validated 2026-11-06: **8 tests passed**, including full checkpoint/helper
retry continuity and older cadence regressions. The two consecutive CPU smoke
phases each retained **4/4** steps, with exact policy KL **2.94e-7 / 2.61e-7**
against cap 0.003. Both used peak **4.875e-6** from PPO LR **7.5e-6**.
The reset was repeatable before commit and did not repeat after saving adapted
beta 0.1875. `git diff --check` passed. Log: `artifacts/g1247-validation.log`.
