# g1241 continuation: PPG every two iterations

Continue the existing `joint7-v11-2-g1290-trained-restart` run from committed
generation **1241**. This is not a fresh run. The original IL folder stays
`features4-v11-2-batch7-il`.

## Changes

The exact reviewed transition is `ppg-value-phase/22.0.0` → `23.0.0`:

- Beta floor **0.25 → 0.125**. Preserve the saved beta and last five phase
  outcomes; this is not a one-time beta reset. The g1241 report has beta 0.25.
  Clean accepted phases with KL < 0.0015 can now reduce it to 0.125. Preserve
  the early hard-cap response (≤16 retained steps: ×1.5), 3-of-5 ×2 precedence,
  maximum 32 and hard KL cap 0.003.
- Peak PPG LR **0.65 → 0.5 times this iteration's last actually applied PPO
  LR**, before the controller selects the next iteration's rate. Preserve
  linear warmup from 0.1× peak to peak over steps 1–16, then cosine decay to
  0.001× peak by the existing two-epoch budget. The new factor drives both
  training and LR provenance; it is not just a config/report change.
- PPG runs every **two newly committed iterations**. Persist the cadence
  origin at g1241: g1242 buffers only; g1243 runs PPG on g1242+g1243;
  subsequent phases run at g1245, g1247, etc. Restarting never resets this
  origin. Rejected or empty scheduled phases still consume their scheduled
  slot, without fabricating accepted training work.

The compatible two-iteration critic cache is retained and staged every
iteration, including skipped phases. Expired files are still deleted only
after checkpoint commit. Off-cycle iterations measure current whole-game
held-out health evidence, without opening old replay, creating cloning policy
references, drawing shuffle RNG, changing either value optimizer, or inserting
fake clean results into beta's five-phase history. GAE evidence receives zero
fresh fitting steps on skipped phases. Current/previous whole-game holdouts,
recorded targets, loss checks every 16 steps and KL checks every 32 plus
retention checks remain unchanged.

All other signed configuration is preserved, including the 224/96 self-play /
history tables, PFSP, reward/entropy progress, epsilon, companion heads, PPO
and DCPG objectives, sparse PPO probes and throughput settings. Model weights,
Adam/RNG, counters, lineage, GAE/critic evidence and compatible replay survive
the reviewed transition. A new actor/value payload version persists cadence;
old signed PPG recipes retain their existing behavior.

## Launch

After syncing this reviewed code to the lab, run from the repository root:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 1241 --apply && \
python scripts/configure-training-g1241.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 1241 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-g1241.json && \
CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/step5.py \
  --cuda-visible-devices 7 --device cuda \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-g1241.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1241 --iterations 20
```

This requests **g1242–g1261**, with ten scheduled PPG phases. GPU 7 is logical
`cuda:0` inside the process. The helper verifies the committed full checkpoint
and hash, rejects interrupted pending work and conflicting outputs, and writes
only a config. Retry safely before advancing. Once training advances, reuse
the generated config and the actual committed generation rather than rerunning
the generation-pinned helper.

## Log observations and validation

The exported g1239–g1241 PPG phases each attempted 16 steps and retained zero,
stopping on `heldout_loss_increased`, without KL hits. Their reported PPG time
was 21.7–22.4 seconds each. Lowering the beta floor alone does not change these
rejections: beta only drops after a qualifying accepted phase. The reduced LR
is an experiment, not a guarantee of better value fitting or playing strength.

The new cadence halves scheduled fitting opportunities; current cache writes
and holdout measurements still run every iteration, so total PPG-related wall
time will not necessarily halve. No production GPU speedup is claimed.

Focused regressions are in `tests/Recovery/test_training_g1241.py`. They cover
strict config scope, LR curve and actual-PPO linkage, beta limits and skipped
feedback, committed cadence boundaries, exact checkpoint/Adam/RNG/calibration
preservation, retry-safe preparation, compatible cache reuse and restart across
both skipped and active phases. The isolated CPU smoke uses synthetic recorded
rows and the real PPG fitting/companion paths; it does not play evaluation games.

Validated 2026-11-06: **18 focused tests passed**, including legacy g1041
planning/resume, beta safeguard regressions and the isolated CPU smoke. The new
smoke retained **4/4** steps, with exact raw policy KL **1.18180e-7** and peak LR
**3.75e-6** from actual PPO LR **7.5e-6**. Skipped iterations preserved model,
value/companion Adam and RNG exactly, kept beta history unchanged, and resumed
the same cadence. The full-checkpoint preparation test preserved weights,
PPO Adam/RNG and utility calibration and rejected incompatible output or
pending work. Log: `artifacts/g1241-validation.log`.

`git diff --check` passed. Production checkpoints were not modified; no long
training or evaluation was launched.
