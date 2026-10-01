# Selected search experiment: Masked ISMCTS

The previous R4/R5 and S1/S2/S3 comparison roadmap is retired by the implemented [Masked ISMCTS revision](../ai-research/MASKED_ISMCTS_REVISION_PLAN.md). The explicit-chance OpenSpiel adapter remains an implementation dependency. It is not a separate algorithm choice. Historical configs are preserved in `configs/searchlab/archive/`; frozen banks, old results and source signatures remain archival evidence.

Use the [current conda lab guide](TRAINING_RECOVERY_LAB.md) for fresh MLP-ResNet training, the direct-one-second v3 bank, NN search teachers, evaluation and cleanup. `scripts/r4r5.py`, `verify-searchlab.py` and `smoke-searchlab.py` now exit with a migration message.

The additional quick experiment is `scripts/masked-search.py`. It loads only a checkpoint from a new fresh residual run and compares its policy-only preferred move with Masked ISMCTS. `--compare-unpruned` adds the same backend with all legal candidates opened immediately. Both variants receive the same whole-decision clock. Legal choices, protected industries and PPO probabilities are unchanged by expansion pruning.

```sh
conda activate ai-training
python /data/not_backed_up/yxu209/ai-training/scripts/masked-search.py --checkpoint /data/not_backed_up/yxu209/ai-training/artifacts/step5/masked-fresh/generation-0020.pt --device cuda --config /data/not_backed_up/yxu209/ai-training/configs/search/lab-gpu.json --compare-unpruned --output /data/not_backed_up/yxu209/ai-training/artifacts/masked-search/probe.json
```

Use `lab-cpu.json` or `laptop-cpu.json` for the required CPU path. `--seconds` and `--simulations` bound this decision probe; they do not calibrate or change v3's 1,000ms training allowance. The report records source/model/config identity, complete legal incumbents, completed root statistics, coverage and actual latency. It makes no playing-strength claim and performs no promotion.

Run compact engineering checks with `scripts/verify-masked-ismcts.py --integration`. These include CPU batch parity, legal-mask/top-K separation, hidden-world invariance, completed four-seat visit labels, fresh training/resume, separate distillation, ONNX parity and retention. No old comparison tournament is required.
