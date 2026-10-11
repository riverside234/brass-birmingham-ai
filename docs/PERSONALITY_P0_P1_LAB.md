# Post-Step-5 personalities: P0 handoff and P1 goal head

Implemented 10 October 2026. These commands use the **completed, stopped Step 5 run on the lab**. They do not need a model on the laptop. Wait for your current training run to finish; the importer refuses an active run. Replace `selfplay-best-v1` below with the actual completed run folder.

P0/P1 is a separate, bounded experiment. The installed game and Step 5 training, IL handoff, search settings and checkpoint readers are unchanged. Goal-biased Masked ISMCTS, persistent intent and archive discovery are now implemented in the separate [P2/P3 workflow](PERSONALITY_P2_P3_LAB.md). The full personality league and release qualification remain later work. The four pilot profile codes are input controls, not yet proven playing personalities.

## 1. Inspect the accepted model

Use your existing conda environment and dependencies. No new package installation is required.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/personality.py inspect \
  --from-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1
```

The check resolves `best.json` to the immutable `state.json` champion entry and checks its hash against `best.pt`. It validates the accepted gate's specification, report and result, the model's run/generation lineage, the rules/features contract and the calibrated utility head. An IL-only initial incumbent, mismatching reward, incomplete gate or changed dependency is rejected. It does **not** fall back to the latest learner. If no PPO candidate has passed Step 5's gate, the real P0 handoff must wait.

This implementation accepts the current reward fingerprint:

```text
W + 0.10 * pairwise_placement + 0.001 * scored_VP - min(0.005 * insolvency_events, 0.025)
```

Returns stay undiscounted and subtract only future insolvency costs at each decision. The utility output remains unbounded; it is not a probability. Changing the reward requires an explicit future migration, rather than editing a receipt. A passed Step 5 selection gate is preliminary training evidence, not a personality release certificate.

## 2. Freeze the P0 handoff

Use a new output directory:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/personality.py handoff \
  --from-run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1 \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --trace-shards 4 --references 3
```

The handoff takes the existing run/cleanup lock, pins dependencies **before copying**, and verifies every copied hash. It preserves the parent, an age-spread reference sample, reward/calibration metadata, accepted selection receipts, resolved configuration, runtime/bank contracts and available ordinary training shards with their authoritative receipts. It records every shard's game/setup/split identity. Old style/intent fields retain their original meaning; absent new personality context is explicitly absent.

`manifest.json` and `pins.json` describe the protected bundle. The source gains explicit pins; its model, state, population and selection are unchanged. With zero-hour Step 5 cleanup, keep these pins while the handoff is in use. Optional cleanup review:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/selfplay-best-v1
```

This command is the existing cleanup dry run; do not add `--apply` merely to inspect it. P0 never recovers missing training data from evaluation games. If ordinary traces were already retired, the manifest says so; use fresh warm-up collection in the next section. You can explicitly set `--trace-shards 0` when only fresh discovery data is wanted.

## 3. Warm up the new head with the core frozen

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/personality.py --device cuda warmup \
  --handoff /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/personality/goal-pilot-v1 \
  --epochs 3
```

This uses the copied ordinary training shards only for auxiliary supervision. It starts a new optimizer, copies all trained policy/official-value/utility tensors, and keeps those tensors and the new context projection frozen. It checks that the copied tensors remain bit-for-bit unchanged. The goal learning rate is `1e-4`.

If no suitable traces remain, add `--games 20 --workers 4` to collect fresh policy games first. Each table has one learner seat and three frozen parent seats, with private rules hosts and batched inference. Twenty games cover all five input codes in all four seats: neutral, industrialist, networker, supplier and investor. These are provisional controls; no heuristic teacher or recovery-opponent schedule is started. `--device cpu` is also supported.

The medium architecture has **1,915,962 total parameters**: the copied 1,850,129-parameter network plus 65,833 goal/context parameters. The new contract checks a 100,000-parameter extension limit and a 2,100,000 total limit. Its named outputs are policy logits; four-seat win share, placement, VP and remaining utility; four goal-factor logits; and six-by-three event-confidence logits. Import initially preserves all five old outputs, including the trained utility correction and remaining cost budgets.

Goal labels use **public realized outcomes**, never future features or opponents' cards. They cover sales, a newly built industry's later flip, resource consumption, network extension, liquidity recovery and flexibility. Industry factors include Pottery and all other industries. Region uses public city areas. Horizons count the current remaining own turn and up to two/four own turns in the same era. Missing final effects, short horizons and ambiguous simultaneous completions are masked. An unsold build is not a completed industry goal. These are behavioral proxies, not proof of optimal strategy or the model's mental intent.

## 4. Measure held-out goal diagnostics, then run a small joint pilot

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/personality.py --device cuda validate-goals \
  --handoff /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/personality/goal-pilot-v1 \
  --games 20 --workers 4

python /data/not_backed_up/yxu209/ai-training/scripts/personality.py --device cuda joint-pilot \
  --handoff /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/personality/goal-pilot-v1 \
  --games 20 --workers 4 --epochs 2

python /data/not_backed_up/yxu209/ai-training/scripts/personality.py --device cuda validate-goals \
  --handoff /data/not_backed_up/yxu209/ai-training/artifacts/personality/handoff-v1 \
  --run /data/not_backed_up/yxu209/ai-training/artifacts/personality/goal-pilot-v1 \
  --games 20 --workers 4
```

The joint pilot collects **fresh current-model learner trajectories**. Frozen parent actions never become learner PPO targets. It records and checks the exact action probability, actual predecision profile and model identity. Saved intent is explicitly absent in this P1 collector; the separate P2 controller produces it causally. Old traces cannot enter PPO through the auxiliary importer.

The pilot takes LR, clipping, KL limit, entropy/value weights, batch limits and gradient clipping from the handoff's resolved Step 5 settings. Current values are LR `1e-5`, clip `0.2`, KL `0.01`, entropy `0.005`, value weight `0.5`, gradient clip `5`, actor/critic batch `2048`, microbatch `512`, and `131072` candidate cells. Goal loss has weight `0.05`. Actor batches exclude forced choices; separate critic/goal batches can use them. The current CLI defaults to two joint epochs. There are no style rewards, search-value pseudo-targets or Pass penalties.

`validate-goals` uses fresh grouped validation seeds, applies no gradients and leaves model bytes unchanged. Reports include masked goal loss, confidence Brier/ECE, separate official-head losses, remaining-utility MSE/MAE, official win share/place/VP by profile, and raw/weighted reward components. These small diagnostics are neither a matched strength tournament nor calibrated difficulty levels. Confidence is reported as **unqualified**. A joint update clears inherited search readiness; P2 must check/recalibrate the changed utility evaluator on appropriate fresh data before teaching.

## 5. Checkpoints, traces and scope

The separate `personality-checkpoint/1.0.0` stores the named-output model, context/goal/reward contracts, source fingerprint, optimizer and RNG. Step 5 readers intentionally reject this new model format. Repeating a completed warm-up or joint command on the same run continues its model and same-phase optimizer. Transitioning from frozen warm-up to joint training creates a fresh optimizer for the enlarged trainable parameter set. Every joint invocation recollects current-model PPO data. After joint training, start a separate run if you want to repeat frozen-core preparation.

`initial.pt` preserves the imported control; `model.pt` is the P1 learner, **not an automatically promoted incumbent**. The frozen Step 5 parent is retained throughout this pilot. The base generated-replay format is unchanged: new profile/role/model contexts live in a hash-bound sidecar. Training, joint and diagnostic seed ranges are separate, with grouped train/validation partition checks. Old generated IL datasets and their readers are unchanged.

Operations are bounded to 1–32 fresh games, at most eight host workers and 1–10 passes per call; up to 32,768 prepared examples are accepted. Use fewer handoff shards if that limit is reached. Four tensor threads are the CLI default. P0/P1 uses small batched policy collection and does not yet implement the planned 70/25/5 full personality league or P2's resource-managed search queues. Do not treat this pilot as the large Step 5 training driver.

Completed operation reports are retained, with the last eight referenced in `state.json`. These bounded pilots do not run automatic campaign cleanup. A partial operation is rejected rather than silently overwritten; preserve it for diagnosis and use a new run for a retry. Source changes invalidate P0/P1 compatibility explicitly, so finish an operation before updating code. Do not use local smoke fixtures as a production handoff.

## Verification

Local verification on 10 October passed **13 P0/P1 tests and 23 existing combined-reward/champion tests**. The final native smoke completed three four-player games in **68.7 seconds** on CPU, with 70 non-forced learner decisions, 32 labeled training turn roots and 16 held-out turn roots. Reload, frozen-parent preservation and unchanged validation checkpoint bytes passed. These numbers describe fixture operation, not trained-model performance. The unchanged Step 5 implementation fingerprint is `826b6beb04241a73a5d1ec0d712176ff9df4370a26cf01bb05ee92f7a05b4f16`.

The focused suite checks accepted-parent rejection, hashes/pins, old-reader rejection, neutral output equivalence including nonzero utility corrections, parameter count, public-only labels, era/horizon masks, all-industry access, frozen-core warm-up, learner-only PPO probabilities and checkpoint/optimizer restoration:

```sh
cd /data/not_backed_up/yxu209/ai-training/
PYTHONPATH=src:tests/R2:tests/Recovery:tests/Personality \
  python -m unittest discover -s tests/Personality -p 'test_*.py' -v
```

A small optional native smoke creates **authored fixture weights/evidence**, runs two fresh joint games and one held-out diagnostic game, and checks reload and source preservation:

```sh
python /data/not_backed_up/yxu209/ai-training/tests/Personality/smoke.py \
  --output /data/not_backed_up/yxu209/ai-training/artifacts/personality/native-smoke-local
```

Local CPU verification does not establish CUDA throughput, playing strength, real-parent goal quality, or P2 search readiness. Those require the completed lab parent and later comparison gates in [the personality plan](../ai-research/PERSONALITIES_AND_DIFFICULTY_PLAN.md).
