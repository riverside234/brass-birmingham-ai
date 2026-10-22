# g503 follow-up: decoupled opening PPO and PPG re-enable

Implemented 22 October 2026. The final requested PPG acceptance interval is **16 optimizer steps**, with **patience 1**. This supersedes the provisional eight-step setting. Existing model/feature layouts and the completed IL checkpoint are unchanged.

## Effective recipe

| Setting | Current follow-up |
| --- | --- |
| PPG | Enabled, `ppg-value-phase/6.0.0` |
| Acceptance | Every 16 steps, also at a shorter epoch tail; patience 1 |
| Replay / schedule | Eight committed iterations, uniform weights, phase every eight fresh iterations |
| Held-out acceptance | All eight iterations' ordinary learner held-out games; existing whole-family exclusions |
| PPG policy guard | Overall and age-0 exact raw-policy KL each capped at 0.002, checked every optimizer step |
| Value optimizer | Existing separate Adam; reset at phase start; 0.1x to 1x warmup across the first 16 steps, then decay to 0.3x at the cap |
| Epoch / beta controls | Existing adaptive 2-epoch initial cap, bounds 1–6; beta bounded at 32; saved beta retained |
| Ordinary exploration | Epsilon 0.00; probabilistic action sampling retained |
| Openings | Temperature 2 for each neural seat's first two action decisions, including the existing post-fork reset |
| Snapshot critic | Fixed 1.00, with prior recovery/decay state archived |
| Insolvency | Event cost 0.010, maximum cost 0.075 |
| Branching | 1.5% per eligible decision, still at most one fork per ordinary self-play game |
| PPO batching | Microbatch 512, candidate-cell cap 131072, four preparation workers; effective batch 2048 |

The branch fork-move distribution remains unchanged; the fork move and copied prefix remain excluded from PPO. Teacher distillation remains off. The current curriculum, PFSP sampling, ratings, reward targets, GAE, entropy coefficients and original entropy/economic baselines remain intact.

## Opening-only objective and provenance

`update.opening_objective` records `opening-ppo-objective/1.0.0`. For eligible nonforced temperature-2 opening actions:

```text
r = pi_new_T1(chosen) / frozen_pi_old_T1(chosen)
w = stop_gradient(min(2, frozen_pi_old_T1(chosen) / actual_behavior(chosen)))
loss = -mean(w * min(r * advantage, clip(r, 1-clip, 1+clip) * advantage))
```

The denominator is the recorded collection probability, including temperature and epsilon mixing. There is no minibatch weight renormalization. Raw old-policy probabilities are captured once during the existing pre-PPO behavior-verification scan and frozen before any actor update. The existing utility-only calibration does not alter the policy.

Non-opening rows keep their behavior-policy clipping ratio. Value losses, advantages, entropy losses, legal masks and forced-action filtering are unchanged. The correction does **not** make the recorded returns or GAE fully on-policy for temperature 1.

Collection receipts carry the objective contract through shard loading, branch expansion and compact reader IPC. Epsilon remains part of policy identities and collection signatures. Stale objective/epsilon caches are rejected before actors start; previous behavior is never relabeled.

Reports separate `temperature1_categorical_kl`, `behavior_categorical_kl` and opening-only raw-policy KL. Console `KLT1` is raw temperature-1 movement and the LR feedback signal; `KLbehavior` is the separate actual-behavior policy movement. Both exact PPO guards retain the existing 0.01 limit, while the full-data sampled guard remains a behavior-policy check. Behavior reconstruction is still checked separately before updating. Compact history retains these fields and opening weight/support statistics.

## Continuation and changed files

`scripts/joint7-g503-ppg.py` reads locked committed state, composes this revision with the previously requested g503 changes, validates the exact reviewed transition, and writes only the new config. It preserves weights, PPO/value Adam state, RNG, counters, population and lineage. The existing per-phase value-Adam reset still occurs when a scheduled auxiliary phase begins.

Re-enabling PPG at committed g503 starts an empty cache window without backfilling old reward/behavior data. Phases are due at **g511 and g519**. GAE evidence survives re-enable; the LR controller's measurement window resets for the new collection/objective identity while its multiplier and warmup state are retained. The original entropy baseline is not reset. Unchanged old signed recipes remain valid under their old contracts.

Principal files:

- `src/brass_ai/recovery/opening_objective.py`, `updates.py`, `review_metrics.py`: objective, frozen reference, strict verification and separate KL measurements/guards.
- `ordinary.py`, `replay.py`, `branching.py`, `ingress.py`: durable objective provenance.
- `ppg_config.py`, `actor_value.py`, `value_phase.py`: versioned 16-step/patience-1 PPG enablement and continuation.
- `config.py`, `exploration.py`, `resume.py`, `lr_controller.py`, `history.py`, `league.py`: explicit defaults, reviewed transitions, data identity, checkpoint recipe and reporting.
- `scripts/joint7-g503-ppg.py`, `configs/step5/joint7-v11-2-grouped-lr-critic-ppg-openings-g503.json`, the learning source-review registries, and `docs/MLP_V11_LAB.md`.
- `tests/Recovery/test_opening_ppo.py`, `smoke_opening_ppo.py`, and updated fresh-default assertions in `test_ppg_disabled.py` / `test_exploration_revision.py`.

Unrelated existing chart edits and the completed g503 fixes were preserved. Training checkpoints and exported results were not modified.

## Validation

The final 16-step revision passed **19 focused objective/continuation checks**, plus **two updated fresh-default regressions**. Coverage includes positive/negative clipping, detached/capped weights, the exploration-inclusive denominator, legal masks, strict behavior verification, independent exact KL guards, branch openings, shard/IPC provenance, stale-cache rejection, preservation of non-opening PPO weights/Adam/RNG, and interrupted adoption. The PPG regression checks at steps 16 and 32, then restores the best step-16 weights and matching Adam when patience is exhausted.

A broader 105-case regression run exercised categorical KL guards, eight-age PPG, disabled PPG, exploration, controller floors and the earlier g503 fixes. Its two failures were obsolete fresh-default assertions expecting PPG off and epsilon 0.02; both were updated and passed on rerun. The final targeted invocation also included two mistyped test selectors; those selectors were corrected and both intended tests passed. There are no unresolved test-case failures.

The isolated CPU smoke used the current medium v11.2 model (**5,928,616 parameters**) and authored mixed-table/branch fixtures. It ran g504–511, then replayed g511 from the g510 disk checkpoint. Total wall time was **212.1 seconds**, on two CPU threads. This was a small fixture run with reduced row/clone/probe batches, not production games or a full-size throughput test.

| Smoke measurement | Observed |
| --- | ---: |
| Retained / attempted PPG steps | 56 / 56 |
| Acceptance checks | Steps 16, 28 (epoch end), 44, 56 (epoch end) |
| Distinct critic rows used | 870: 774 learner and 96 snapshot; 6 branch rows included |
| Held-out acceptance support | 48 ordinary games across all eight ages |
| Cloning / probe support | 320 / 125 positions |
| Final PPG overall exact KL | 0.00003188 |
| Final PPG age-0 exact KL | 0.00003698 |
| Peak replay cache | 21,326,157 bytes, about 20.34 MiB |
| Process lifetime peak RSS | 840,605,696 bytes, about 801.66 MiB |
| Continuous vs restarted phase | Bit-identical weights, both Adams, RNG, schedule and epoch-controller state |
| Incoming checkpoint bytes | Unchanged |

The final fixture PPO update had 73 nonforced opening rows, including one branch-opening row. Actual behavior remained unchanged and verified. Its mean importance weight was 0.9892, range 0.8456–1.1314. Raw-T1/behavior probe KL were 0.0003762 / 0.0002073 on 121 positions. These are implementation diagnostics, not evidence of improved playing strength.

Local smoke output: `artifacts/g503-opening/smoke16/report.json`. Reproduce the focused checks and smoke with:

```bash
PYTHONPATH=src:tests/Recovery python -m unittest -v \
  test_opening_ppo \
  test_ppg_disabled.DisabledEight.test_defaults_and_exact_g460_config_review \
  test_exploration_revision.ControllerTests.test_strict_configuration_and_fresh_defaults
PYTHONPATH=src:tests/Recovery python tests/Recovery/smoke_opening_ppo.py \
  --output artifacts/g503-opening-smoke-new
```

The smoke requires a new output directory. No long training or evaluation campaign was launched.

## Runtime implications

Compared with eight-step acceptance, 16 steps approximately halves paired held-out check frequency away from epoch tails. Compared with the archived 32-step recipe, it approximately doubles that check frequency. Patience 1 can end unproductive phases sooner. The 0.002 KL checks still run every optimizer step.

The opening correction reuses existing forward passes for old-policy capture and KL probes, adding probability tensors/softmax calculations rather than another model. PPG re-enablement restores eight-age disk-cache and auxiliary compute costs. Full-size GPU runtime and memory have not been measured here.

## Exact g504–523 command

After syncing this revision on the lab machine, run this from the repository. The earlier g503 helper does not need to run first:

```bash
cd /data/not_backed_up/yxu209/ai-training

python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 503 --apply && \
python scripts/joint7-g503-ppg.py --expected-generation 503 && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-ppg-openings-g503.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

Recovery archives an incomplete next collection after Ctrl+C and refuses a different committed generation or an ambiguous pending checkpoint. The completed IL filename stays unchanged. Later clean continuations use only the training command.

Reviewed source: `38206ae9ecec5db1b291066978738d247d40ebe112fc27b4f3a4332796caffca`. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`.
