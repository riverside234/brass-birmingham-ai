# g412 PPO input-cost regression

Reviewed `results/joint7-v11-2-grouped-lr-critic-g0412-20261021T173550049006Z` on 21 October 2026. The export confirms **committed generation 412 / 131,840 ordinary games**, with the entropy-groups hotfix already adopted. Only exported evidence is available locally; no live lab checkpoint, config or training state was modified.

## Findings

| Metric | g410, before input-cache update | g412, after update |
| --- | ---: | ---: |
| Collection wall time | 190.91 s | 146.06 s |
| Total reported PPO time | 88.78 s | 98.99 s |
| PPO steps completed / planned | 52 / 52 | 44 / 44 |
| Forward/backward total | 18.67 s | 14.47 s |
| Per-step KL-probe total | 13.62 s | 7.33 s |
| Probability scans total | 17.54 s | 12.72 s |
| Dataset preparation | 8.55 s | 11.94 s |
| Time outside existing named PPO scopes | 30.08 s | 52.27 s |
| Head-only utility calibration steps | 213 | 186 |

g411 also shows the regression: **100.21 s PPO**, 44 steps and **54.14 s** outside the old named scopes. The measured probe and training kernels are faster; reverting those improvements is not supported by these logs. Collection also contains fewer branches after the separately requested rate reduction, so its change cannot be attributed solely to inference caching. The g412 timing tail lacks cleanup/whole-iteration totals; those are unavailable, not zero.

**Confirmed code regression:** the new `InputAudit.check` hashed the complete candidate matrix before each state-only value batch narrowed it to one candidate. It then hashed the temporary slice as well. Head-only utility calibration performs three training passes plus before/after measurements, amplifying that cost. Dataset construction also eagerly hashed and validated all actions, including unused snapshot-critic candidates. Calibration was included in total PPO wall time but omitted from the named timing scopes. The exports cannot determine exactly how much of the 22.20 s increase outside those scopes came from each operation; the bounded benchmark below reproduces the unnecessary work directly.

g411/g412 each completed all **44 planned PPO steps**, without early stopping; exact categorical KL was **0.005102 / 0.006360**. Shared economic health passed, LR stayed **6.25e-6**, PPG remained disabled and teacher steps stayed zero. The existing mixed-v4 EV warning remains a review signal; this performance repair changes no gate or threshold.

## Repair

- Dataset construction retains full original shape/layout checks, with selector admission performed when candidates enter collation.
- Stock state-only value batches check original shapes before slicing and fully validate the consumed candidate's selectors. They do not hash unused action tails or build temporary admission proofs. CPU and pinned-prefetch paths agree. A custom action-dependent value model still receives every candidate.
- Policy/BC batches retain every legal candidate and full selector admission. Cached probe/scan content seals, stale-behavior rejection, strict probability reconstruction, masks and exact KL after every PPO step remain active.
- New non-overlapping timing scopes expose `ppo.behavior_identity`, `ppo.critic_prepare`, `ppo.utility_calibration`, `ppo.gae`, `ppo.probe_prepare` and `ppo.probe_final` in full iteration reports. Calibration behavior, its three passes and all optimizer math remain unchanged.

No config regeneration or new hotfix is required. The existing collection inference changes, model architecture, checkpoints, Adam/RNG, training progress, population, lineage, LR/GAE/snapshot evidence, entropy floors, branching, reward and opponent settings are preserved.

Changed files: `src/brass_ai/recovery/updates.py`, exact source/handoff review registries, `tests/Recovery/test_value_batch_performance.py`, `tests/Recovery/test_selfplay_input_cache_resume.py`, the new `tests/Recovery/benchmark_ppo_value_inputs.py`, this review and `docs/MLP_V11_LAB.md`.

## Validation and runtime scope

**67 focused tests passed; two CUDA-only tests skipped** (69 total, 54.077 s). They cover exact value outputs/gradients, calibration receipts/Adam/RNG, full-PPO parity, consumed-selector rejection, original-layout checks, custom value models, cached-input mutation rejection, behavior probabilities, streaming inference, exact g412 config/source admission, prior migration routes and interrupted-collection recovery. Evidence: `artifacts/g412-ppo-review/focused-tests.log`.

The isolated smoke passed with a fresh **5,928,616-parameter** model, **64 authored rows**, **two PPO steps**, and a repeat from the same saved checkpoint. Weights, Adam and RNG matched exactly; the incoming checkpoint bytes stayed unchanged. All 56 eligible actor rows were used and both per-step KL checks ran. This computation smoke uses the existing authored epsilon-0.08/temperature-1 fixture; focused regressions separately cover the current epsilon-0.02 and per-row temperature-2 behavior. It does not simulate complete games or evaluate strength. Evidence: `artifacts/g412-ppo-review/ppo-smoke/report.json`.

The CPU benchmark alternates before/after order over three trials, using **512 authored positions**, **1–772 candidates**, five value-only packing passes and two PyTorch threads. Baseline: commit `b33c37be5031e8a522eb56ab3b066b9d8cee4296`. All resulting tensors, masks and seat mappings match exactly.

| CPU path | Before median | After median |
| --- | ---: | ---: |
| Cold critic Dataset construction | 0.08751 s | 0.01239 s |
| Five value-input packing passes | 0.33546 s | 0.06776 s |
| Bytes hashed during those passes | 474,450,840 | 0 |

The packing path is **4.95× faster**, about **80% less packing time**, in this fixture. This is not a 4.95× whole-PPO speedup. Local PyTorch is CPU-only; production GPU end-to-end savings remain to be measured. The new calibration timing makes that comparison possible without changing the training recipe. Evidence: `artifacts/g412-ppo-review/value-input-benchmark/report.json` and `comparison.json`.

All **26,486 existing expanded source approvals** and all prior IL handoff entries were preserved. There are **7,910 appended exact routes**, retaining existing approved migration chains and adding the unchanged-config g412 transition to implementation `d0860bb58611f1645d27d6a3df2e5c308d567854808315d8d434b104895ad151`. Learning compatibility remains `ef6a1b5126de42da8f3277dc9357b6ffb94a435f3faa7fee8dd312bb5a242a54`. No long training or evaluations ran.

## Continue after Ctrl+C

After syncing the revised source and review registries to the lab machine, run from its repository root with training stopped:

```bash
python scripts/recover-step5-iteration.py --root "$PWD" \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --expected-iteration 412 --apply && \
python scripts/step5.py \
  --config configs/step5/joint7-v11-2-grouped-lr-critic-entropy-groups.json train \
  --run artifacts/step5/joint7-v11-2-grouped-lr-critic \
  --imitation-run artifacts/step5/features4-v11-2-batch7-il \
  --resume-reviewed-source --iterations 20
```

Recovery archives any incomplete next collection without changing committed state; it is a no-op if none exists. It refuses an uncommitted checkpoint/report requiring separate commit recovery. Training starts with **g413** and, if all 20 iterations complete, reaches **g432**. Do not rerun `joint7-entropy-groups.py`; that migration is already recorded. The original IL path is unchanged.
