# Fresh Masked ISMCTS training in conda

Recipe `step5-training/2.2.0` implements M0–M5 of the [revision](../ai-research/MASKED_ISMCTS_REVISION_PLAN.md). The main model is a randomly initialized **MLP-ResNet with 1,840,109 parameters**. Only Masked ISMCTS is active. Existing neural weights, optimizers, populations and neural replay are not imported. An audited, entirely heuristic dataset may be reused. Resume only inside the same new run, with the same config, code and dataset receipt.

## Transfer and preflight

Keep conda `ai-training`. Use the project's pinned requirements; no new environment is needed if those dependencies already work.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python -m pip install -r /data/not_backed_up/yxu209/ai-training/requirements-searchlab.txt
```

Sync this revision's source/configs and tracked `training/` directory first. Privately copy the laptop's `artifacts/step5-transfer/private-rules-runtime.zip` to `/data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime.zip`. This includes proprietary rules dependencies, so it remains outside Git. Project-authored worker executables and the frozen bank source/profiles are tracked. Existing bootstrap weights remain archival and are never loaded by this recipe.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py import-runtime --bundle /data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime.zip
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda preflight
```

The bundle importer verifies the full member inventory and hashes before writing. It refuses to replace different runtime files. If an old clone has conflicting privately generated runtime files, use a clean clone/checkout and retain the old run separately. Preflight checks full-game rules parity, the selected bank and a device backward pass. It is not a v3 clock calibration or strength tournament.

## Generate and audit heuristic demonstrations

The new six-style bank is `bank-927c8ddec148897fa542`. Its five training identities use **1,000 ms per whole turn**, including both actions and replanning. Pottery Flex's opponent identity is evaluation-only; learner Pottery choices remain legal and receive the same exploration support as every industry. The bank's strength is unmeasured; imitation supplies legal/economic foundations rather than a strength ceiling.

For a first dataset, 500–1,000 complete games gives independent train/validation/test setups and broad style coverage. Start smaller to verify throughput. New generation rotates the five admitted teachers. Existing audited heuristic manifests can be loaded with their immutable archived provenance; do not pass old neural replay to `demo-audit`.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-generate --output /data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/masked-v3 --games 1000
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-audit --replays '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/masked-v3/*.jsonl.gz' --output /data/not_backed_up/yxu209/ai-training/training/datasets/masked-v3.json
```

Use a new manifest name: auditing never overwrites an existing receipt. Audit replays every transition against the rules engine, verifies complete legal lists, deduplicates games, partitions by setup and rejects held-out/unadmitted teachers. BC balances teacher/era/phase/action labels without forcing a Pottery build rate. Validation is held out by setup. Early stopping selects the best validation weights after at least three epochs, at most ten, with patience two.

## Fresh imitation, then self-play

One command initializes random residual weights, does heuristic BC inside that new run, resets optimizer moments, then starts population PPO. **Omit `--initial` and `--from-run`.** Both are rejected. Do not copy checkpoints or a previous population into this directory.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-fresh --demonstrations /data/not_backed_up/yxu209/ai-training/training/datasets/masked-v3.json --iterations 20
```

Repeat the identical command to resume that run. The source/config signature, current checkpoint hash and every population path are checked. After a crash before the first league commit, an identity-matching imitation progress receipt permits BC to restart from the original random seed; it does not import partial weights. A config/code change requires a new directory. Run without `--demonstrations` only if deliberately testing random self-play; then there is no BC or rehearsal.

Twenty iterations is a pipeline/early-learning run (6,400 ordinary games), not a promise of strong play. Continue based on held-out economics and opponent-panel progress, rather than loss alone. Monitor unflipped tiles, cash distress, sales/merchant beer, legal opportunities versus chosen industries, VP and seat-specific outcomes. Later self-play should establish strength beyond the heuristic teachers.

| Setting | Main lab preset |
| --- | --- |
| Ordinary collection | 320 games/iteration; 28 single-host actors |
| Heavy collection | 8 tables × 4 private root workers = 32 search workers |
| CPU allowance | 28 ordinary + 32 heavy + 4 tensor threads = 64 planning allowance; not an OS affinity cap |
| v3 teacher clock | Exactly 1 second; no calibration, no 240-node target; completion may retire late and is measured |
| NN search | 3-second teacher clock, 4 private root workers, 4 pending paths / batch per replica; independent warm simulator lanes |
| Central CPU/GPU inference | Batch up to 64 positions, 131,072 candidate cells, 1 ms partial flush; CPU feature preparation |
| PPO | Effective batch 2,048; microbatch 512; independent critic batch 2,048; 2 epochs; LR 3e-5; KL stop 0.01 |
| BC | LR 1e-4; main effective/microbatch settings; complete legal policy denominator |
| Curriculum | v3 tables 40% → 20% → 10% → 5% at 2,000 / 5,000 / 10,000 games |
| Search teachers | Eligible from 640 games; schedule every 1,600; 4 games; at most 10% collection/update wall-time share |
| Routine evaluation | 8-game monitor every 1,600 games; 16-game broader screen every 6,400 |
| Retention | 8 recent + 4 anchor weight-only opponents; 2 full resume checkpoints; 4 replay rounds; bounded development shards/traces |

The first search collection occurs at the end of a chunk after the ready threshold. Subsequent due points advance by 1,600 games. Search actions are external behavior: they never become PPO neural likelihoods. Distillation needs completed visits on at least two alternatives, mixes 90% normalized visits with 10% full-support prior, and uses a separate supervised loss. A search that cannot complete meaningful evidence skips distillation and reports the reason. NN teachers use a separate 3-second lab clock because full NN rollouts need more completed evidence; v3 stays exactly one second. Reference/smoke NN presets stay at one second. Do not fabricate labels or silently lengthen the v3 clock.

For fewer resources, select `--config /data/not_backed_up/yxu209/ai-training/configs/step5/lab-small.json` before creating a new run. `lab-reference.json` uses one search worker; `lab-16x2.json` uses 16 heavy tables × 2 workers. All v3 presets still use one second. `recovery-smoke.json` is a compact local scaffold check. None of these is permission to migrate an old source-bound run in place.

## Evaluation and selected-backend probe

Scheduled evaluations use frozen snapshots from the same new run. Evaluation games never enter PPO or BC. For a manual 8-game development screen, choose two distinct generations from that run:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-fresh/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-fresh/population/weights-REFERENCE_MODEL_ID.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-screen --setups 2
```

Replace `REFERENCE_MODEL_ID` with a path recorded in the run's `state.json`. Broader and formal panels require two distinct same-run references. The full shortlist preset includes held-out Pottery Flex. Add `--search` to evaluate the selected search backend; ordinary monitor panels stay fast. Search/evaluation reports do not promote a model without their separate evidence gates.

A quick decision probe compares only the selected backend against its own NN prior and, optionally, the same search with top-K fully opened. It is a coverage/latency check, not a tournament:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/masked-search.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/search/lab-gpu.json --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-fresh/generation-0020.pt --compare-unpruned --output /data/not_backed_up/yxu209/ai-training/artifacts/masked-search/probe.json
```

The CPU path is mandatory and uses `configs/search/lab-cpu.json`; laptop CPU uses `configs/search/laptop-cpu.json` (one search worker, two tensor threads, eight pending paths). Standalone probes can override the NN clock with `--seconds`; this never changes the v3 bank's fixed one-second clock. No S1/S2/S3/R4 comparison command remains active.

## Storage and diagnostics

Cleanup runs after committed iterations, preserves current/resume/population/evaluation-anchor weights and referenced dataset shards, and quarantines retired files for 24 hours. First inspect a manual plan:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-fresh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-fresh --apply
```

Use `pin --run ... --paths ...` before a long independent evaluation when retaining an older generation. `restore --manifest` accepts the exact quarantine receipt; `--apply --purge` deletes only matured, unreferenced quarantine entries inside the owned run. Active runs and outside-workspace paths are rejected.

Report inference batch positions/cells, padding, expired requests, queue/encoding/inference timing, host RPC/IPC timing, completed simulations, distinct alternatives and turn retirement. Warm private lanes avoid per-batch host swapping; obsolete search generations release their lanes instead of filling the lightweight policy cache. Parallel roots use private trees/channels/RNG and await actual completion before reuse. Device utilization is diagnostic: a 1.84M model does not need to fill 90 GB, and slower rule simulation may dominate. GPU/lab throughput and playing strength require fresh measurements on the server.

The live 1.4.9 game patch is C# heuristic Masked ISMCTS, with the same six v3 styles and 2/5/12-second options. It preserves the [1.4.7 resource lifecycle](AI_EXECUTION_RESOURCES_1.4.7.md). The future trained NN deploys through that lifecycle after qualification; this revision does not install untrained random NN weights as opponents.
