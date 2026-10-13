# Two-stage self-play champion selection

The active lab and smoke presets use `two-stage-champion/1.0.0`:

1. Freeze the latest committed candidate, incumbent and evaluation settings. Precommit 72 distinct validation setup seeds.
2. Screen the first 8 setups: all six candidate seat pairs per setup, 48 games. Continue only if candidate win share is **at least 56%**. The screen cannot promote a model.
3. Complete **64 additional setups**, 384 games, using the same two checkpoints. Do not stop early after a favorable intermediate result or add further setups after failure.
4. Pool **all 72 setups / 432 games**. Promote only when candidate win share is **at least 55%** and the **95% setup-cluster bootstrap lower bound is strictly greater than 49%**. Exactly 49% fails.

Every game is an ordinary four-player game with two independent seats per model. Win share is the sum of the official winner shares of the candidate's two seats, including official ties and tiebreakers. Its parity baseline is 50%. Policies use neutral focus, zero exploration and **no search**. Evaluation replays do not enter PPO or teacher distillation.

The interval resamples setup clusters, keeping all six seat pairs together (4,000 deterministic bootstrap draws, 2.5th/97.5th percentiles). The pooled mean weights the two stages 8:64, not 1:1. This is the requested repeated training-selection rule; the interval is not adjusted for screening or repeated challenges. A lower bound above 49% can still include 50% and therefore does not establish a statistically significant advantage above parity.

## Use for joint-7 generation 150

Use [joint7-g150-gate72.json](../configs/step5/joint7-g150-gate72.json). It is identical to the generation-150 control preset except for `champion`. Existing control/distill4 experiment presets retain their recorded gate configuration. Old single-stage report decisions remain reproducible under their original thresholds.

From the lab project root, verify the actual stopped run first:

```bash
python scripts/joint7_preflight.py \
  --root /data/not_backed_up/yxu209/ai-training \
  --run artifacts/step5/v4-selfplay-joint \
  --evidence results/v4-selfplay-joint-g0150-20261013T173011108713Z \
  --config configs/step5/joint7-g150-gate72.json
```

Then use the new preset with the existing training command and `--resume-reviewed-source`, for example:

```bash
python scripts/step5.py --root /data/not_backed_up/yxu209/ai-training \
  --config configs/step5/joint7-g150-gate72.json --device cuda \
  train --run artifacts/step5/v4-selfplay-joint \
  --imitation-run artifacts/step5/v4-batch7-joint \
  --resume-reviewed-source --iterations 1
```

The checked source/config transition preserves checkpoint weights, Adam/RNG, generation, population, accepted champion, and schedule counters. Source/config review remains mandatory; no historical result is relabelled. The gate still runs at its existing 6,400-game cadence. A pending evaluation from an older reviewed source gets a separate deterministic evaluation directory.

## Reports and restart

`spec.json` freezes all 72 setup seeds and both checkpoints before screening. `screen-report.json` records the completed screen; `confirmation-report.json` records the additional 64 setups after completion. Interrupted confirmation reuses the screen and resumes the same remaining game jobs. No champion or schedule change is published from a partial panel.

The final `report.json` contains the spec, outcomes, stage summaries, pooled interval and exact thresholds. `stage` is `screen_rejected` or `combined`; `confirmation_required` is an intermediate decision only. A completed root report is recalculated and reused on restart without new games. Accepted evidence remains protected by retention; unfinished stages are retained, and obsolete completed stage caches are retired with their panel. Compact history retains both stages and the pooled interval. The existing report exporter includes `*-report.json` automatically.
