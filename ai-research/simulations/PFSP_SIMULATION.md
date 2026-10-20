# Stored PFSP simulation and trainer cross-check

Date: 2026-10-19.

The supplied simulation is saved unchanged as [pfsp_sim_original.py](pfsp_sim_original.py). Its SHA-256 is `d4a3b527e6da783d8ee7da1fb1d577ccc47744021210297fa6ae6fffb2ab22d1`; `.gitattributes` preserves its original bytes across Windows/Linux checkouts. It came from the user's attachment `de4abf70-c95e-4301-bbcf-942b4ecc0a58/Pasted text.txt`.

Use the maintained [scripts/pfsp_sim.py](../../scripts/pfsp_sim.py) for future experiments. **Rerun this simulation when history-table share, games per iteration, scored-pool capacity, IL seat share or estimator half-life changes.** It requires Python and NumPy, and does not load or modify models, training state or configs.

## Rerun from the repository root

Current 320 ordinary games × 45% history = 144 history tables per iteration:

The corrected v4 gate v2 changes 160/144/16 tables to **160/154/6** after
qualification. History then supplies **308 opponent seats and 616 pairwise
comparisons**, a 6.94% increase in actual comparisons. Run the pre-gate case
below and use `--tables 154` for the post-gate case (48.125% of all tables).
All added history seats use the frozen PFSP distribution. S/C estimates take
the extra actual outcomes with the existing five-iteration half-life; stored
scores, reciprocal IL share, capacity and admission safeguards are not reset
or artificially rescaled. History learner-pair rotation counts actual history
tables, including the transferred slots.

```bash
python scripts/pfsp_sim.py --games 320 --history-share 0.45 \
  --output results/pfsp-history45.json

python scripts/pfsp_sim.py --games 320 --tables 154 \
  --output results/pfsp-history154.json
```

For example, to explore a future 30% history share:

```bash
python scripts/pfsp_sim.py --games 320 --history-share 0.30 \
  --output results/pfsp-history30.json
```

These commands do not change the trainer's mixture. The percentage is a fraction of **tables**, not seats. Alternatively pass `--tables 96` for an exact history-table count. Nonintegral `games * history-share` is rejected: use the actual planned count rather than silently rounding. The two history seats and two learner seats yield four comparisons per table. An effective history fraction is recorded when `--tables` overrides the share.

Other options: `--caps 16 23` (scored members, excluding IL), `--seeds 16` (dynamic seeds 0–15), `--half-life 5`, `--min-count 15`, and `--il-fixed` (old 20% IL share). Default IL share is `1/(N+2)`, where N is scored members. IL alone receives 100%. `--seeds 2` is useful for a quick CLI check, not a reliable comparison across scenarios.

The maintained version defaults to C≥15 eviction support, matching the trainer's safeguard. `--min-count 0` restores the supplied simulator's eviction support and seeded trajectories. To reproduce its original output formatting and timing definition exactly, run:

```bash
python ai-research/simulations/pfsp_sim_original.py
python ai-research/simulations/pfsp_sim_original.py --il-fixed
```

Choose a new JSON filename for each experiment; existing results are never overwritten. Each report records options, fixed experiment settings, NumPy version, both script hashes, per-seed dynamic results, static precision arrays and elapsed time. The default experiment has 400 iterations per case, a 60-iteration static burn-in, and forgetting at iteration 250. Static precision uses seed 0; the dynamic experiment uses the requested seed count. Changing `--seeds` does not add static replications.

## Cross-reference findings

The trainer's unweighted S/C decay, prior, squared hardness weights, uniform floor, reciprocal IL share and four learner/snapshot comparisons match the supplied NumPy implementation. Direct parity tests use identical generated outcomes, also injecting official-residual-tie scores and a no-comparison iteration. The old quality-score estimator matches the supplied quality baseline. No trainer defect was found in these checks; the trainer source and its reviewed continuation identity are unchanged by this follow-up.

The maintained simulator fixes two edge cases in the supplied script:

- `argmax` on an all-false crossing condition returned zero, incorrectly reporting a one-iteration response. Missing/evicted observations and nonpositive/undefined plateau changes now remain unavailable (`null`), instead of manufacturing a crossing.
- Empty scored pools caused `rng.choice(0)` and empty probability normalization failures. The generator now supports IL-only tables with all four comparisons. The dynamic experiment still skips pre-admission startup, preserving the original random sequence.

Response timing now uses the **end** of a trailing three-iteration smoothing window. The event is elapsed iteration 0; indices 0–2 cannot produce an observed crossing before elapsed iteration 2. This shifts ordinary reported crossings one iteration later than the original centered-window label. Precision predictions with no observations are unavailable rather than infinite.

| Aspect | Supplied / maintained simulation | Trainer |
| --- | --- | --- |
| PFSP eviction support | Original C≥0; maintained default C≥15 | C≥15 |
| Eviction choice | Highest eligible estimated outrank; protects newest two and age<20 | Within 0.05 of easiest eligible, then nearest-other-model probe KL |
| Admission | Every five iterations from 15; no health/KL model | Health window and distinctness checks; affects the next collection |
| Capacity | Separate constant-cap experiments at 16 and 23 | 16 before 100k committed games, then 23 |
| IL strength | Kept at a 95% learner pairwise win probability in dynamic experiment | Actual frozen completed-IL checkpoint |
| Outcomes | Synthetic Gumbel seat utilities; continuous, without ties | Actual game results with official tiebreaks and residual ties |
| Age reports | Recorded after update, so age≤20 corresponds to pre-update age<20 | Planning summary uses committed age<20 |

The simulation's looser admission and simplified eviction are assumptions, not instructions to remove trainer safeguards. It does not emulate cached probe KL, checkpoints, rating, PPO or actual game economics. The earlier [independent benchmark](../../tests/Recovery/benchmark_pfsp.py) remains useful for testing the trainer's estimator and eviction on a fixed synthetic pool; it is a different experiment.

## Validation

- 33 focused tests passed: new reference/parity/edge-case tests plus existing PFSP and quality-pool regressions. [Test log](../../artifacts/pfsp-simulation-20261019/regressions.log).
- The archived original reproduced the supplied figures: at cap 16, forgotten-member mean share was 0.024 → 0.070 → 0.107 → 0.119 at event/+5/+10/+20; PFSP evictions 0/16, quality-score evictions 5/16. [Original log](../../artifacts/pfsp-simulation-20261019/original.log).
- The maintained default completed in 3.41 seconds, with 16 dynamic seeds at each cap. Corrected half/90% response medians were 7/14 iterations at cap 16 and 6/13.5 at cap 23; no forgotten PFSP evictions. The latter also includes the C≥15 support safeguard. [Full JSON](../../artifacts/pfsp-simulation-20261019/history45.json).
- A bounded 30%-history rerun verified 96 tables and 384 comparisons per iteration. It used only two dynamic seeds and does not establish a recommendation to reduce history share. [JSON](../../artifacts/pfsp-simulation-20261019/history30.json).

All timings are local simulation timings, not training throughput. These results do not demonstrate improved playing strength or guarantee stability in real games.

The v2 correction was checked with fresh **144-table and 154-table runs**,
16 dynamic seeds per cap, under `artifacts/v4-history-g213/pfsp-144.json` and
`pfsp-154.json`. See [the correction report](../V4_HISTORY_G213_REVISION.md)
for measured results and exact continuation commands. The archived original
simulation is unchanged.
