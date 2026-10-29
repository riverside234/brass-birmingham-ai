# g1320 insolvency reduction and PPG cache refresh

Date: 2026-10-28 (America/Chicago).

The supplied incremental export is
`results/joint7-v11-2-grouped-lr-critic-g1320-20261027T075134827445Z`.
Its manifest points to the preceding `g1320-20261027T074209515491Z` bundle for
`state.json`. That state confirms generation **1320**, **422,400 ordinary games**,
and `generation-1320.pt` with SHA-256
`9cd37f82ae2a1a8ed4f3bdeb791719cb974c50d8fd4fccf81f1e8de854a03be4`.
The continuation helper verifies the actual checkpoint on the training machine;
these reports do not replace it.

## Narrow change

- Insolvency `event_cost`: **0.02 → 0.01**. `max_cost`: **0.075**, unchanged.
  Costs remain causal, once per insolvency settlement event, capped per seat/game.
- Learner and search reward specifications stay identical. The explicit reviewed
  migration changes reward metadata without changing model tensors or Adam/RNG.
  The old utility-calibration receipt is invalidated; subsequent fitting follows
  the existing DCPG/PPG recipe. Frozen opponent checkpoints are unchanged.
- The existing PPG reward-change path drops old cache receipts, then deletes
  only owned PPG cache files after the new config is durably adopted. It checks
  directory ownership and all filenames before deleting anything. Repeating
  startup after interruption is safe; after the first new-reward checkpoint,
  ordinary resumes preserve its fresh cache.
- **g1321** trains/validates on fresh current data only. **g1322** restores the
  current+previous window using g1321/g1322, both under the new reward. Old targets
  are never relabeled or admitted. No manual cache deletion is needed.
- All other configuration is preserved: every-iteration PPG, two-age buffer,
  two-epoch cap, loss16/KL32, .002 policy cap, saved adaptive beta, PPO/PPG/auxiliary
  optimizer state, opening/branching, GAE/snapshot/anchor evidence, population,
  training counters, completed IL and checkpoint lineage.
- Fresh self-play config generation also selects 0.01/0.075. Explicit historical
  configs retain their original values until a reviewed transition is requested.

Implementation: `training_revision_g1320.py`, `configure-training-g1320.py`, exact
reward migration admission, resume handler, fresh config generation and source
review registry. Focused tests and the isolated smoke cover continuation and
fresh-cache restart behavior. Existing unrelated Rust/search edits are untouched.

## Validation

**9 focused regressions passed** in 21.3 seconds. They cover the exact config
change, unchanged cap and causal returns, reward fingerprints, explicit migration
requirement, checkpoint/generation checks, read-only helper retries, preservation
of both Adams/RNG/beta/gates, cleanup only after adoption, unknown-file protection,
cache refill and preserved historical source-review restrictions.
The initial text-fixture hash assertion was adjusted for Windows Git CRLF checkout;
the LF-normalized export matches its original manifest hash. Production binary
checkpoint validation remains byte-exact. Final log:
`artifacts/g1320-validation-final.log`.

The isolated CPU smoke passed in **34.8 seconds**, using two authored old-reward
iterations followed by migration and two fresh-reward iterations. Retained PPG
steps were **2, 4, 2, 4**; validation ages were **[1], [1,2], [3], [3,4]**.
All retained policy KL measurements were below .002. Repeating the final iteration
from the preceding checkpoint exactly reproduced model weights, PPO/value/auxiliary
Adam, RNG, beta/trips and gate state. Only disposable fixture caches were deleted.
Result: `artifacts/g1320-reward-smoke/report.json`. CUDA was unavailable; this is
an integration/restart check, not a production throughput benchmark.

Implementation identity:
`83ebfadaaccf862ebd06fb78cecb4b34362ad977781adf67f118851df244f015`.
The registry retains all **74,611** existing approval records and appends **52**
exact source/config routes. No production checkpoints, long training or evaluation
campaigns were run.

## Continue g1321–1340

After syncing the revised source and review registry, run from the lab repository:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 1320 --apply && \
python scripts/configure-training-g1320.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 1320 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-training-g1320.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-training-g1320.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1320 --iterations 20
```

The first command archives uncommitted work left by interruption. The config
helper writes only a new config, checks the full checkpoint hash, and rejects a
stale generation or a conflicting output file. It does not rewind training.
On first training startup, expect `PPG fresh reset ... next phase g1321`.

The initial phase has only one age of data, so its training and held-out support
will temporarily be smaller. Cache cleanup adds one-time startup I/O. No playing
strength improvement or production speedup is inferred from the small smoke.
