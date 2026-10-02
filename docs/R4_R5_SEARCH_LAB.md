# Selected search experiment: Masked ISMCTS

The neural R4/R5 and S1/S2/S3 comparison roadmap is retired by the implemented [Masked ISMCTS revision](../ai-research/MASKED_ISMCTS_REVISION_PLAN.md). On 1 October, the requested quick comparisons retained rule-based v3 + Masked ISMCTS for imitation generation and training/evaluation opponents, with one second and the 16×2 lab preset. Original S1 is a frozen diagnostic control. See [both screens and the selection](V3_S1_TRAINING.md). Neural search and distillation continue to use Masked ISMCTS. The explicit-chance OpenSpiel adapter remains an implementation dependency. Historical configs remain in `configs/searchlab/archive/`.

Use the [current conda lab guide](TRAINING_RECOVERY_LAB.md) for fresh MLP-ResNet training, one-second Masked **v4** teachers/opponents, Masked ISMCTS NN teachers, evaluation and cleanup. `scripts/r4r5.py`, `verify-searchlab.py` and `smoke-searchlab.py` still exit with a migration message.

The additional quick experiment is `scripts/masked-search.py`. It loads only a checkpoint from a new fresh residual run and compares its policy-only preferred move with Masked ISMCTS. `--compare-unpruned` adds the same backend with all legal candidates opened immediately. Both variants receive the same whole-decision clock. Legal choices, protected industries and PPO probabilities are unchanged by expansion pruning.

```sh
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/masked-search.py --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-24x4-tournament-v4-fresh/generation-0020.pt --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/search/lab-gpu.json --compare-unpruned --output /data/not_backed_up/yxu209/ai-training/artifacts/masked-search/probe.json
```

Use `lab-cpu.json` or `laptop-cpu.json` for the required CPU path. `--seconds` and `--simulations` bound this decision probe; they do not calibrate or change v3's 1,000ms training allowance. The report records source/model/config identity, complete legal incumbents, completed root statistics, coverage and actual latency. It makes no playing-strength claim and performs no promotion.

Run compact engineering checks with `scripts/verify-masked-ismcts.py --integration`. These include CPU batch parity, legal-mask/top-K separation, hidden-world invariance, completed four-seat visit labels, fresh training/resume, separate distillation, ONNX parity and retention. No old comparison tournament is required.

## Strength-repair contract

The quick probe now uses `masked-ismcts/1.1.0`: visited/unseen PUCT competition, supported continuation evidence and distinct private replica warmups. All three active search presets explicitly use neural raw Q and the new-root-information frontier. They have no `future_turns`, heuristic normalization or heuristic opponent-turn cap. Top-K is temporary expansion pruning; full policy legality, protected industries and CPU/GPU batching are preserved.

Use a checkpoint produced by a **new** `masked-24x4-tournament-v4-fresh` run with this source. Existing source-bound runs and older search-teacher labels are not resumed/imported. The probe remains an additional decision experiment, with no automatic model promotion. See the [search contract and repair evidence](MASKED_ISMCTS_STRENGTH_1.4.11.md).


## Era/parallel revision

The lab additional experiment uses four private NN roots and the same bounded central CPU/GPU inference service as Step 5. It no longer forwards GPU work through four independent local evaluators. Pending-path virtual loss is selection-only (0.02); raw Q, four-seat evidence and the NN information-frontier horizon remain. Private channels are locked, and each forest drains before reuse. The lab probe acquires four CPU admission slots before starting its clock. The laptop preset and installed game use two workers. Top-K and full-expansion comparisons still receive the same budget and perform no promotion. Main collection/evaluation uses 24 queued heavy tables under the shared 60-slot gate. See [era repair and diagnostics](MASKED_V3_ERA_REPAIR.md).

For a small throughput check, `--repeat 2` reports cold and repeated decisions
separately using persistent workers. `--pending-paths 2` can reduce concurrent
unfinished paths on a slow CPU without changing legality, Q units or the
information-frontier horizon. These optional flags change only that probe;
compare completed evidence as well as inference batch size. An expired search
with no completed support must fall back and contribute no distillation label.



The advanced-opening v3 candidate replaces the traditional six labels with Coventry, external-beer cotton, Worcester iron/cotton, zero-loan industry, advanced pottery and an adaptive book. Opening/expansion evaluation accounts for funded access, current public beer and real resource-market receipts. The user stopped that comparison and selected the previous advanced version as v4, including productive-development progress and opening-biased candidate selection. Its bank is bank-3dc65253a610191d66a5; the rejected network/Scout revision is not active. Resolve the admitted bank from `training/opponents/current.json`; see [the source and comparison note](../ai-research/TOURNAMENT_STYLE_V3.md).
