# Fresh training in conda: selected v4 teachers and Masked ISMCTS NN

Recipe `step5-training/2.2.0` uses a fresh **1,840,109-parameter MLP-ResNet** and **Masked ISMCTS + NN** for search/distillation. The source-based six-style revision and comparison requirements are documented in [the opening-book note](../ai-research/TOURNAMENT_STYLE_V3.md). The user selected the previous advanced bank, `bank-3dc65253a610191d66a5`, as **v4**, with productive-development stall reset and opening-biased candidates. Further strength tests were stopped; this is an explicit selection rather than a tournament promotion. The rejected network/Scout additions are excluded. `training/opponents/current.json` and the distribution both identify that exact bank. The main lab preset is **24 queued heavy tables, four private search workers per decision**, with one shared 60-slot CPU admission gate under your 64-core allocation. Waiting for admission does not consume the search clock.

Imitation and heuristic evaluation use searched Masked **v4** at **one second/four workers**. Early RL uses its distinct search-free incumbent; at 10,000 games, when heuristic opponents occupy 5% of tables, it switches to searched v4. Internal `v3-*` profile IDs and `training/runtime/v3*` paths remain compatibility aliases for the selected bank. Neural weights, optimizers, populations and neural replay are not imported. Audited entirely heuristic datasets may be reused with their original receipts. Resume only within the same new run and unchanged source/config/dataset.

## Transfer and preflight

Keep conda `ai-training`. Use the project's pinned requirements; no new environment is needed if those dependencies already work.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python -m pip install -r /data/not_backed_up/yxu209/ai-training/requirements-searchlab.txt
```

Sync this revision's source/configs and tracked `training/` directory first. Read the selected bank from `training/opponents/current.json`; `training/runtime/distribution.json` must identify the same bank. The game uses **two workers**, lab heuristic and NN search **four**. Privately copy the freshly exported `artifacts/step5-transfer/private-rules-runtime-v4.zip` to `/data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime-v4.zip`. The distribution manifest records every member and SHA-256; import verifies the exact inventory before writing. The bundle includes proprietary game dependencies and remains outside Git. Authored workers and frozen source/profiles are tracked; archival bootstrap weights are never loaded.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py import-runtime --bundle /data/not_backed_up/yxu209/ai-training/artifacts/step5-transfer/private-rules-runtime-v4.zip
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda preflight
```

The bundle importer verifies the full member inventory and hashes before writing. It refuses to replace different runtime files. If an old clone has conflicting privately generated runtime files, use a clean clone/checkout and retain the old run separately. Preflight checks full-game rules parity, the selected bank and a device backward pass. It is not a v3 clock calibration or strength tournament.

Use the bundle exported with this source revision: **10 authored workers and 904 private runtime files**, including the selected bank and immutable S1, early Masked, prior Masked, industry, economic and advanced controls. Bundle SHA-256: `49103db526cf01d43b6374d45f124da2a0c6ac309543bc6a640e295c28764be8`. Current authored executables and frozen bank source/profiles are tracked under `training/`; vendor DLLs stay private. An old bundle is incompatible even if its filename matches. On a conflicting lab checkout, retain the old runs and import into a clean checkout.

The NN config declares `evaluator_kind: neural`, `q_transform: none` and `leaf_policy: new_root_information_frontier`. `future_turns` is removed. Search finishes the current turn, then evaluates a new ordinary acting-root information set; later supported visits can expand it further. Choice/tree/clock limits still interrupt safely. Heuristic reply limits and Q normalization are rejected by the NN schema. CPU batching, shared CUDA inference, self-play curriculum and retention remain enabled. Old search teachers are incompatible; generate new labels after this revision.

## Generate and audit heuristic demonstrations

Read the active bank identity from `training/opponents/current.json`. Its five training styles use **1,000 ms per complete turn and four private root workers**, with K 4/2, guarded sibling Q and zero voluntary reply turns. The new worker implements the forest internally, sharing a CPU heuristic batching/cache queue instead of starting four separate policy processes. Advanced Pottery Variants's opponent identity remains evaluation-only; learner Pottery choices retain full legality and exploration support. Imitation supplies foundations; the mostly self-play curriculum establishes strength beyond the teachers.

For a first dataset, 500–1,000 complete games gives independent train/validation/test setups and broad style coverage. Start smaller to verify throughput. New generation rotates the five admitted teachers. Existing audited heuristic manifests can be loaded with their immutable archived provenance; do not pass old neural replay to `demo-audit`.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-generate --output /data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/masked-24x4-tournament-v4 --games 1000
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py demo-audit --replays '/data/not_backed_up/yxu209/ai-training/artifacts/demonstrations/masked-24x4-tournament-v4/*.jsonl.gz' --output /data/not_backed_up/yxu209/ai-training/training/datasets/masked-24x4-tournament-v4.json
```

Use a new manifest name: auditing never overwrites an existing receipt. Audit replays every transition against the rules engine, verifies complete legal lists, deduplicates games, partitions by setup and rejects held-out/unadmitted teachers. BC balances teacher/era/phase/action labels without forcing a Pottery build rate. Validation is held out by setup. Early stopping selects the best validation weights after at least three epochs, at most ten, with patience two.

## Fresh imitation, then self-play

One command initializes random residual weights, does heuristic BC inside that new run, resets optimizer moments, then starts population PPO. **Omit `--initial` and `--from-run`.** Both are rejected. Do not copy checkpoints or a previous population into this directory.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda train --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh --demonstrations /data/not_backed_up/yxu209/ai-training/training/datasets/masked-24x4-tournament-v4.json --iterations 20
```

Repeat the identical command to resume that run. The source/config signature, current checkpoint hash and every population path are checked. After a crash before the first league commit, an identity-matching imitation progress receipt permits BC to restart from the original random seed; it does not import partial weights. A config/code change requires a new directory. Run without `--demonstrations` only if deliberately testing random self-play; then there is no BC or rehearsal.

Twenty iterations is a pipeline/early-learning run (6,400 ordinary games), not a promise of strong play. Continue based on held-out economics and opponent-panel progress, rather than loss alone. Monitor unflipped tiles, cash distress, sales/merchant beer, legal opportunities versus chosen industries, VP and seat-specific outcomes. Later self-play should establish strength beyond the heuristic teachers.

| Setting | Main lab preset |
| --- | --- |
| Ordinary collection | 320 games/iteration; 28 single-host actors |
| Heavy collection | 24 queued tables × 4 private workers; searched decisions share 60 active CPU slots |
| CPU allowance | Shared FIFO gate: 60 active work slots + 4 reserved cores; ordinary turns cost one slot, searched turns four; not OS affinity |
| v4 teacher clock | v4 + Masked ISMCTS; 1-second shared turn deadline after admission; 4 private roots; K 4/2 and guarded heuristic Q. Completion/selection may retire late and is measured |
| NN search | 3-second teacher clock, K 16/8, raw Q, new-root-information frontier; 4 private root workers, selection-only virtual loss 0.02, 4 pending paths / batch per replica; independent warm simulator lanes |
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
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/step5/development-evaluation.json evaluate --candidate /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh/generation-0020.pt --references /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh/population/weights-REFERENCE_MODEL_ID.pt --output /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-screen --setups 2
```

Replace `REFERENCE_MODEL_ID` with a path recorded in the run's `state.json`. Broader and formal panels require two distinct same-run references. The full shortlist preset includes held-out Advanced Pottery Variants. Add `--search` to evaluate the selected search backend; ordinary monitor panels stay fast. Search/evaluation reports do not promote a model without their separate evidence gates.

A quick decision probe compares only the selected backend against its own NN prior and, optionally, the same search with top-K fully opened. It is a coverage/latency check, not a tournament:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/masked-search.py --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/search/lab-gpu.json --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh/generation-0020.pt --compare-unpruned --output /data/not_backed_up/yxu209/ai-training/artifacts/masked-search/probe.json
```

The CPU path is mandatory and uses `configs/search/lab-cpu.json`; laptop CPU uses `configs/search/laptop-cpu.json` (two private search workers, two tensor threads, eight pending paths per root). Standalone probes can override the NN clock with `--seconds`; this never changes the v3 bank's fixed one-second clock. The frozen S1 comparison remains a historical diagnostic control. S2/S3 and alternative neural-search comparisons remain retired.

## Storage and diagnostics

Cleanup runs after committed iterations, preserves current/resume/population/evaluation-anchor weights and referenced dataset shards, and quarantines retired files for 24 hours. First inspect a manual plan:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh
python /data/not_backed_up/yxu209/ai-training/scripts/step5.py cleanup --run /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh --apply
```

Use `pin --run ... --paths ...` before a long independent evaluation when retaining an older generation. `restore --manifest` accepts the exact quarantine receipt; `--apply --purge` deletes only matured, unreferenced quarantine entries inside the owned run. Active runs and outside-workspace paths are rejected.

Report inference batch positions/cells, padding, expired requests, queue/encoding/inference timing, host RPC/IPC timing, completed simulations, distinct alternatives and turn retirement. Warm private lanes avoid per-batch host swapping; obsolete search generations release their lanes instead of filling the lightweight policy cache. Parallel roots use private trees/channels/RNG and await actual completion before reuse. Device utilization is diagnostic: a 1.84M model does not need to fill 90 GB, and slower rule simulation may dominate. GPU/lab throughput and playing strength require fresh measurements on the server.

The selected **v4 / patch 1.4.20** uses C# heuristic Masked ISMCTS, six advanced-opening styles, **two game workers** and the existing **2/5/12-second** options. Lab search remains at four workers. Its previous `smoke-7` screen tied 6–6 on wins and averaged 88.79 versus 84.71 VP against the economic control; the user then explicitly selected this bank and stopped further strength tests. This selection does not certify tournament superiority. Productive development resets intent age, and opening goals influence build/setup candidates. The [1.4.7 resource lifecycle](AI_EXECUTION_RESOURCES_1.4.7.md) remains in force. Trained NN deployment is a later qualification step.

## Historical rule-based comparison commands

These commands document earlier S1/Masked diagnostics. They are not required for the v4 transfer or fresh training, and no further strength comparison was run for this selection.

Run eight matched four-player games (two setups × four seat rotations). Each table has one Balanced serial 1s/240-work opponent, one Balanced 1s/two-worker/720-total-work opponent, and two rotating serial S1 styles.

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-s1-training.py --tables 16 --setups 2 --output /data/not_backed_up/yxu209/ai-training/artifacts/s1-training-smoke/lab
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-s1-training.py --compare-masked --tables 16 --setups 2 --output /data/not_backed_up/yxu209/ai-training/artifacts/s1-vs-masked-smoke/lab
```

The second command compares S1 with Masked ISMCTS, both at one second and two private workers, alongside the same serial S1 anchors. Selection uses official win rate, then lower mean rank, then higher mean VP; a complete tie keeps S1. The previous threefold work multiplier applies only to the S1 control's 240-to-720 node allowance; it does not multiply Masked simulations. Active presets keep this compatibility field at one.

That historical preset supported 16 concurrent heavy tables; eight smoke games can occupy at most eight. A local laptop smoke used `--tables 2`. Reports provide VP, official split win rate, first-place frequency, mean rank, work counts and actual retirement times. Those smoke games never enter training. The **current** main preset queues 24 heavy tables with four workers per decision. [The recorded local results](V3_S1_TRAINING.md) retain the earlier selection evidence.


## Parallel execution and measurements

The shared admission gate applies to imitation generation, ordinary RL, later searched v3 opponents, NN distillation and evaluation. Search clocks start after admission and all workers retire before a turn lease releases. NN trees own their nodes privately; pending paths use virtual loss only during selection. The central inference queue locks private reply channels, groups frozen-model requests and flushes by the earliest deadline. Heuristic v3 uses a CPU queue, not the GPU.

Ordinary encoding caches semantic action vectors with exact R2 parity; replay gzip uses compression level one. Host persistence, audited seals, derived feature receipts and asynchronous writes remain. Track `cpu_admission`, `cpu.admission.wait`, queue/encoding/RPC times, valid/padded candidate cells and completed search support. These changes were tested locally on CPU; measure Linux/Mono/CUDA throughput on the lab before claiming a speedup.

Historical era-comparison command (eight games, both one second/four workers; not required for this release):

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-era-v3.py --tables 8 --seed 842400 --output /data/not_backed_up/yxu209/ai-training/artifacts/rail-era/lab-confirmation
```

This small standalone smoke runs at most eight tables (32 search slots); the main dataset/training preset queues 24 tables under the shared 60-slot admission gate. It is diagnostic and never enters BC/PPO. Preserve each output directory.



The six **v4** styles are Adaptive Advanced Book, Coventry Goods Opening, External-Beer Three Cotton, Worcester Iron-Cotton, Zero-Loan Industry and Advanced Pottery Variants. Opening/expansion evaluation accounts for funded access, current public beer and real resource-market receipts. The selected bank is `bank-3dc65253a610191d66a5`; the rejected network/Scout revision is not active. Resolve the admitted bank from `training/opponents/current.json`; see [the source and comparison note](../ai-research/TOURNAMENT_STYLE_V3.md).
