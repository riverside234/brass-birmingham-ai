# g1041 continuation: 70/30 tables and early PPG KL response

This is a reviewed continuation of `joint7-v11-2-g1290-trained-restart` at
committed generation **1041**, beginning with g1042. It includes both
[PPO sparse probes](PPO_SPARSE_PROBES.md) and
[collection/rating/cleanup throughput changes](SELFPLAY_THROUGHPUT.md).
The original IL folder is unchanged.

## Changes

- PPG `ppg-value-phase/22.0.0` adds a beta response after any hard-cap hit when
  the phase retains **0–16 accepted optimizer steps**: beta increases by **1.5**
  immediately in the state committed for the next phase. Count retained work,
  not attempted steps; a trip at step 32 can retain only step 16 or zero.
  There is no within-phase retry. Existing 3-of-5 trips still double beta and
  take precedence; the multipliers never stack. Maximum remains 32, minimum
  0.25. Clean accepted phases with retained KL strictly below 0.0015 still
  halve beta. A held-out-loss stop without a KL hit does not trigger a raise.
  The existing saved beta and trip window are preserved on adoption.
- From ordinary game **333120**, collect **224 self-play / 96 historical / 0
  standalone v4** tables per iteration. Keep four learners on self-play tables
  and two learners plus two frozen opponents on history tables. The frozen v4
  ensemble remains eligible through PFSP. Historical scheduling before this
  boundary is retained so the six learner-seat-pair rotation continues without
  restarting or backdating counts.
- PFSP probabilities remain normalized over eligible historical opponents.
  For conditional probability `p`, expected exposure is now `192*p` opponent
  seats per collection, or `0.15*p` of all table seats. Actual observed counts
  drive the estimator: 192 opponent seats and 384 pairwise comparisons per
  collection. No artificial score/count rescaling or duplicate loss weighting.
  PPO still averages eligible rows; 896 self-play plus 192 historical learner
  seats means approximately **82.35% / 17.65%** learner-seat support. Decision
  fractions depend on game lengths and are reported from actual samples.
  This increases learner-seat support by about 11.9% from 972 to 1088, so PPO
  preparation/update work may grow even with the throughput changes. There
  are also more self-play tables eligible to branch, at the unchanged branch
  rate. No production wall-time improvement has been measured for this mix.
- The existing LR controller detects the changed collection composition and
  clears its short KL window while preserving its multiplier, cooldown, Adam
  state and independent health evidence.
- Adopt `ppo-sparse-probe-control/1.0.0`: exact PPO guards after step 1, every
  fourth step, and epoch tails; full logging-only post-epoch scans disabled.
  Full behavior reconstruction and the final exact probe remain active.
- Adopt the throughput settings, including **16** bootstrap processes within
  the configured CPU budget, one numerical-library thread each, unchanged 200
  deterministic paired resamples and full-fit warm starts. Immutable collection
  input/request reuse, overlapping rating validation and safe retention hash
  reuse are source changes included in this reviewed implementation.

PPO/PPG targets, loss weights, entropy/reward schedule progress, model and
companion weights, optimizers/RNG, warmup, training counters, PFSP estimates,
history, and lineage are preserved. The existing compatible two-iteration PPG
cache is reused; no cache refresh is needed. PPG remains every iteration, up to
two epochs, loss checks every 16 steps and KL checks every 32 steps (plus
retention checks), with the unchanged 0.003 safeguard and 0.65× PPO peak LR
schedule. These changes do not demonstrate improved playing strength.

## Launch from g1041

Sync the reviewed code to the lab, then run from the repository root. The first
command archives interrupted/uncommitted work using the existing recovery
mechanism. The preparation helper verifies the full g1041 checkpoint hash and
writes only a config; it rejects pending work, the wrong generation and a
conflicting output. It is safe to retry before the first new generation commits.

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-iteration 1041 --apply && \
python scripts/configure-training-g1041.py \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --expected-generation 1041 \
  --output configs/step5/joint7-v11-2-g1290-trained-restart-g1041.json && \
CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/step5.py \
  --cuda-visible-devices 7 --device cuda \
  --config configs/step5/joint7-v11-2-g1290-trained-restart-g1041.json train \
  --run artifacts/step5/joint7-v11-2-g1290-trained-restart \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1041 --iterations 20
```

This requests **g1042–g1061**. GPU 7 becomes logical `cuda:0`; startup and
iteration metadata report the effective visibility mask and device UUID.
Reuse the generated config on subsequent continuation, using the actual new
committed generation; do not rerun the generation-pinned preparation helper.

## Log observations and validation

The exported g1039/g1040/g1041 phases retained **0 / 16 / 0** PPG steps and
stopped for `heldout_loss_increased`, without KL trips. The new beta condition
would not change those outcomes. The export still has the old every-step PPO
probe contract and four bootstrap processes, so generating the combined config
above is necessary to adopt the requested performance settings.

Focused checks and a small isolated CPU PPG smoke are in
`tests/Recovery/test_training_g1041.py`. They cover strict config adoption,
0/16/17 retained-step boundaries, trip precedence/clamps, actual 320-table
planning and six-pair balance, PFSP probability/count normalization, LR evidence
reset, full checkpoint/Adam/RNG/calibration retry continuity, and sealed replay
reuse across the PPG contract change. No production checkpoints or live games
are needed for these tests.

Validated 2026-11-06: **39 tests passed** (six new checks plus 33 sparse-PPO,
throughput, continuation and legacy-beta regressions). The isolated CPU PPG
smoke retained **4/4** attempted steps, measured raw policy KL
**2.01455e-7** against the unchanged **0.003** cap, and restored its two-age
buffer and value Adam state exactly. Its clean low-KL phase moved beta from
0.5 to 0.25 under the existing rule. The early-hit, 16/17 boundary and 3-of-5
precedence checks use authored feedback, not a playing-strength experiment.
Logs: `artifacts/g1041-validation.log` and `artifacts/g1041-regressions.log`.
`git diff --check` passed. No long training, live evaluation campaign or physical
GPU benchmark was launched.
