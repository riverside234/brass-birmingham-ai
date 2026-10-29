# g1322 reward and exploration revision

## Effective settings

| Setting | Previous | New |
| --- | ---: | ---: |
| Insolvency event cost | 0.01 | **0.005** |
| Insolvency maximum cost | 0.075 | **0.05** |
| `canal_action`, `rail_action` entropy coefficients | 0.005 | **0.015** |
| `canal_other`, `rail_other` entropy coefficients | 0.005 | **0.005** |
| Opening temperature | 1.3 | **2.0** |
| Opening action decisions per neural seat | 3 | **3** |
| Ordinary epsilon mixing | 0.05 | **0.08** |

These are fixed entropy-loss coefficients on the raw T1 policy. Adaptive entropy
remains disabled. Saved entropy baselines, controller history and promotion origin
are not reset. Console and full/compact reports show the effective coefficients,
rather than dormant adaptive alpha values.

The reward cap is reached after ten insolvency events. Learner and search reward
specifications match. The original family/industry/context exploration distribution
is preserved. Existing ordinary neural seats, including frozen historical seats
under the current `snapshot_seats=match_learner` recipe, adopt .08. Bot, teacher and
evaluation policy settings are unchanged.

The opening, post-fork branch and decoupled PPO contracts are versioned together.
Each neural seat's first three action decisions use T2.0; other decisions use T1.
Post-fork counters reset per seat, excluding the fork move. Branch points still
skip the first three action decisions, and branch probability stays 1%.
Eligible opening rows retain the raw-T1 clipping ratio and detached capped
`old_T1 / actual_behavior` weight. Actual behavior includes T2 and epsilon .08;
its probabilities and provenance remain strictly checked. Old behavior is never
relabeled. Existing masking, forced-row filtering and KL guards are retained.

## Continuation and cache refresh

The user reports two completed iterations after the g1320 command. No g1322 export
was supplied locally. The helper reads the real lab state under its run lock and
requires committed g1322 plus its full checkpoint's exact hash. It writes only the
new config, checks the exact reviewed source/config transition, and rejects stale
generation flags or conflicting config files.

On startup, the existing reward migration preserves model weights, PPO/value/
auxiliary Adam, RNG, saved beta/trips, population, checkpoint lineage and progress.
The original economic and entropy baselines and attained GAE/snapshot state remain.
Existing collection-regime rules handle changed exploration evidence; LR warmup
and its multiplier do not restart. Utility-calibration eligibility is invalidated
as required by the existing reward migration.

The reward change explicitly invalidates old PPG target receipts. Only after
durable config adoption, startup audits and deletes owned PPG cache artifacts.
Unknown files and unsafe paths fail before deletion. Interrupted adoption is safe
to retry. Once a new-reward checkpoint commits, the reset does not repeat.

- **g1323:** current new-recipe data only.
- **g1324 onward:** current plus previous new-recipe iteration, including their
  separate held-out games.

PPG remains every iteration, two cached ages, two epochs, loss checks every 16
steps and periodic KL checks every 32 plus tails/retained-state verification.
The .002 KL safeguard, beta adaptation and LR schedule are unchanged.

## Continue g1323–1342

After syncing revised source and the source-review registry, run from the lab repo:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 1322 --apply && \
python scripts/configure-training-g1322.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 1322 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-training-g1322.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-training-g1322.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1322 --iterations 20
```

The recovery command archives incomplete work after Ctrl+C and does not rewind
committed progress. Expect `PPG fresh reset ... next phase g1323` on adoption.
The completed IL folder is unchanged. Later clean resumes use the generated
config with the actual committed generation; do not rerun old hotfix helpers.

## Implementation and validation

Changed source: `training_revision_g1322.py`, `exploration.py`,
`opening_objective.py`, `branching.py`, `config.py`, `utility.py`, `resume.py`,
`updates.py`, and effective entropy console reporting in `league.py`.
Added `scripts/configure-training-g1322.py`, focused regressions and an isolated
PPO/PPG migration/restart smoke. Fresh-config assertions in historical tests now
expect the latest recipe while their archived transition assertions remain intact.

**23 focused tests passed in 31.3 seconds.** Coverage includes exact config and
reward changes, fresh config regeneration, the ten-event cap, actual .08/T2
probabilities, legal masks, decoupled objective/gradients, stale behavior rejection,
three post-fork decisions, effective group entropy/reporting, baseline preservation,
read-only generation/checkpoint checks, safe cache cleanup/retries, optimizer/RNG
preservation and immutable source-review history. Log:
`artifacts/g1322-validation.log`. An initial fresh-config regeneration failure was
fixed by explicitly handling the new cap in fresh recipe generation; historical
resume migrations retain their exact original restrictions.

The **isolated CPU smoke passed in 34.1 seconds**, with two small authored old-recipe
iterations, migration, two new-recipe iterations and an exact restart comparison.
Retained PPG steps were **2, 4, 2, 4**; held-out ages were **[1], [1,2], [3], [3,4]**.
All retained KL checks stayed within .002. Final weights, PPO/value/auxiliary Adam,
RNG, beta/trips and gate state matched exactly after restart. No production games
or checkpoint were used. Result: `artifacts/g1322-recipe-smoke/report.json`.

Implementation identity:
`8746640733360fd984d0a35a38aa8de6748bf058a9cc53952e86fc54dc1c58b1`.
Learning compatibility remains unchanged. The review registry preserves **74,663**
existing approvals and adds **54** exact routes. Existing unrelated Rust/search
updates remain intact, including the separate local-scoring receipt refactor.

No production training checkpoint is modified by this implementation work.
Cache retirement adds one-time startup I/O; g1323 temporarily has only one age
of training/validation data. The changes add no model passes or search work.
Changing sampled actions can change game lengths and candidate counts, so no
production runtime or playing-strength improvement is inferred.
