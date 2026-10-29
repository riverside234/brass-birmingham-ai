# g1324 fork-action mixture revision

Fork actions now use **50% temperature-2 policy + 50% existing exploration q**.
The old mixture was 70% T1 + 25% T2 + 5% q. This changes the alternative fork move;
branch continuations keep their existing policy settings.

The sampler excludes the parent action, normalizes each component on its remaining
support, selects a component, then samples an action. The recorded probability is
the full mixture marginal; the component and conditional probability are also
retained. If q has no supported alternative, the available T2 component receives
all mass. If neither component has support, no branch is created.

## Versioning and preservation

`selfplay-branching/6.0.0` binds the new mixture, seeds and collection provenance.
Capture and replay verification use the same contract-indexed mixture. Versions
1–5 retain their exact original sampler and deterministic draws, so historical
branch replay remains verifiable. Stale pending collections are never relabeled.
Python and the Rust lab collection path both use this shared capture function.

Only `branching.contract` and `branching.move_distribution` change in the run config.
The following remain: 1% branch probability, at most one branch per eligible self-play game,
three opening decisions excluded as fork points, fresh three-decision T2 windows
after the fork, epsilon .08, fixed entropy .015 action/.005 other, insolvency
.005/.05, decoupled opening PPO, PPG recipe and its safeguards. The forced fork
move remains excluded from actor and critic training.

Weights, PPO/value/auxiliary Adam, RNG, beta, baselines, population, progress and
lineage are retained. Valid PPG replay is retained because the reward/targets and
critic input contract are unchanged; its old branch identities remain original.
Normal cache expiry still occurs after checkpoint commit. Existing collection-regime
evidence rules remain in force. The g1322 reward-cache reset is not repeated.

## Continue g1325–1344

After syncing source and the source-review registry, run from the lab repository:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 1324 --apply && \
python scripts/configure-training-g1324.py \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-generation 1324 \
  --output configs/step5/joint7-v11-2-grouped-lr-critic-training-g1324.json && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-training-g1324.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --expected-generation 1324 --iterations 20
```

The recovery command archives incomplete work after interruption. The helper reads
locked committed state, checks g1324 and the full checkpoint's exact hash, verifies
the reviewed transition and writes only the new config. Repeating it at the same
generation is idempotent. It refuses stale generations, changed checkpoints and
conflicting output files. No g1324 export was supplied locally; the lab command
checks the actual state rather than assuming an exported report is a checkpoint.

## Implementation and validation

Changed: `branching.py`, `exploration.py`, fresh config generation, resume migration
registration and source-review receipts. Added `training_revision_g1324.py`, the
generation-bound config helper, focused tests and a small isolated restart smoke.
Historical tests keep their original transition assertions; only fresh-default
expectations advance to the new recipe.

**20 focused tests passed in 50.7 seconds.** They cover exact 50/50 component
weights, conditional and marginal probabilities, parent-action exclusion, empty
support, deterministic draws, unchanged legacy sampling, actual capture/replay,
tamper rejection, post-fork opening resets, unchanged optimizer/RNG/cache state,
fresh defaults, generation/checkpoint checks and historical source approvals.
Log: `artifacts/g1324-validation.log`.

The **isolated CPU PPO/PPG smoke passed in 52.4 seconds**. Three small authored
iterations retained 2, 4 and 4 PPG steps; replay retained the normal two-age window.
Repeating the third iteration after restart exactly matched model weights,
PPO/value/auxiliary Adam, RNG, beta/trips and gate state. This authored-fixture
smoke verifies training integration/restart, not native game throughput or
playing strength. Result: `artifacts/g1324-branch-smoke/report.json`.

Implementation identity:
`603d1d236ca3da4511467956a9b656b41a4f03017d814b1abd8f060467a4c05c`.
Learning compatibility is unchanged. All 74,717 existing source approvals are
preserved; 56 exact source/config routes were added. Scoped whitespace checks pass.

No production training, checkpoint modification or evaluation campaign is run.
The mixture adds no neural inference calls. Different fork actions can change
continuation lengths, so this is not a demonstrated runtime or strength improvement.
