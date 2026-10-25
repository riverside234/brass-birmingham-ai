# g1050 PPG continuation

Date: 2026-10-25. Export checked:
`results/joint7-v11-2-grouped-lr-critic-g1050-20261025T163535686370Z`.

## Changes

`ppg-value-phase/17.0.0` changes only the beta floor, loss-check interval and
version identity from the g950 recipe:

- **Beta floor 0.5**, upper bound 32. Continuing from v16 sets beta to **0.5 once**,
  before the first new PPO/PPG iteration. Adaptive increases/decreases and the
  saved five-phase KL-trip history remain active. Later restarts restore the
  checkpoint's beta instead of resetting it again.
- **Held-out loss checks every 8 attempted steps**, plus short epoch tails.
  They still use every held-out game and its branch family from both cached ages.
- Periodic **KL checks remain every 16 attempted steps**, plus epoch tails and
  verification of otherwise unchecked retained states. The cap stays **0.002**.
  A step-8 loss acceptance is provisional: it cannot be kept unless its exact
  KL is certified. A safe step-16 policy does not certify a different step-8 policy.

The phase still runs every iteration, trains two uniformly shuffled critic ages
for up to two epochs, and uses peak LR = 0.65 × actual last PPO LR. Linear warmup
is 0.1 × peak at step 1 to peak at step 16, followed by cosine decay to 0.001 ×
peak at the full budget. GAE/snapshot evidence remains current-only. The strict
non-increasing loss acceptance rule, PPO objectives, rewards, exploration,
opponent sampling, rating settings and ownership training are unchanged.

Model weights, PPO/value Adam moments, auxiliary state, RNG, training progress,
population, lineage, independent health evidence and compatible replay caches
are preserved. The conversion records a versioned receipt and supports a retry
before the first new checkpoint. **No fresh-cache reset is needed.** Fresh config
generation also selects v17 and its 0.5 floor; fresh-run initial beta remains 4.

More frequent loss checks increase validation overhead for a phase that runs
the same number of steps. Earlier stopping may offset that cost. No throughput
or strength improvement is established by this change.

## Latest saved report

The export is committed at **g1050**, with full report `iteration-1049/report.json`
and checkpoint path `generation-1050.pt`. Its saved recipe is v16 and beta is **1**.
The phase attempted 16 steps and kept zero because held-out loss increased;
maximum attempted probe KL was **0.00020465**, below 0.002. Loss acceptance used
**15,125 rows from 45 game identities** across g1049 and g1050 (including branch
families). This confirms a loss rejection, not a KL stop. The new 8-step interval
may retain an earlier safe improvement, but does not guarantee fewer rejected phases.

## Validation

**30 focused tests passed** in 234.7 seconds. The two rollback tests also passed
after strengthening them to compare against the original pre-rollback snapshots
(20.9 seconds). Focused validation and an isolated restart smoke are recorded in
`artifacts/g1050-validation.log` and `artifacts/g1050-smoke/report.json`.
The isolated smoke passed in 61.0 seconds on this local CPU run: three authored
small PPO/PPG iterations retained 2, 4 and 4 PPG steps. Its fresh-fixture beta
adapted from 4 to 2, 1 and 0.5. Repeating the third iteration from the saved second
checkpoint reproduced model/auxiliary weights, Adam/RNG, beta/trips and gate state
exactly. The separate adoption regression verifies the current-run 1 → 0.5 hotfix.
No production game or checkpoint was used for the smoke; its runtime is not a
server performance benchmark.
The tests cover exact config changes, unchanged LR schedule, beta floor and
feedback, one-time migration/retry, preservation of cache bytes/Adam/RNG/gates,
read-only generation-bound config generation, and loss8/KL16 rollback.

In particular, forced KL scenarios check both outcomes after a step-16 failure:
retain step 8 only if its own KL passes, or restore the starting state if it fails.
They also check that loss rejection after a safe later probe still verifies the
earlier retained checkpoint, including matching Adam/auxiliary state and retained
row/step accounting. Legacy explicit v14/v15/v16 behavior remains supported.

Changed implementation: `src/brass_ai/recovery/ppg_g210.py`, `config.py`,
`resume.py`, `league.py`, the pinned source-review registry, and
`scripts/configure-training-g1050.py`. Added focused tests/smoke and updated
fresh-default expectations and the two lab guides. Production checkpoints were
not edited and no long training or evaluation was launched.

## Continue g1051–1070

Sync the changes, then run from the lab repository root:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 1050 --apply && \
python scripts/configure-training-g1050.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 1050 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-training-g1050.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-training-g1050.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1050 --iterations 20
```

The first command handles an interrupted uncommitted iteration without rewinding
newer committed progress. The helper verifies the complete g1050 checkpoint and
hash, checks the narrow reviewed transition and writes only the new config.
The trainer applies the one-time beta hotfix under its existing run lock.
Keep the completed IL directory unchanged; do not manually edit state or optimizer files.
