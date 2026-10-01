# Masked ISMCTS M0–M5 exit checks

**Result:** M0–M5 implemented; compact engineering checks passed on the laptop's Windows CPU runtime. Game package **1.4.9** is installed. Lab CUDA throughput, sustained CPU/GPU utilization and playing strength remain to be measured. This is not a new v3 qualification tournament or a promotion of random neural weights.

## Component exits

| Component | Implemented exit |
| --- | --- |
| M0 | OpenSpiel v2.0.2 commit/file pins, adaptation map, separate legal/padding/expansion masks, stable four-seat utility, model/search/teacher contracts and notices |
| M1 | Fresh residual model, CPU/GPU batch interfaces, independently masked fixed opponents, private simulator lanes/root forests, progressive top-K and completed-only visit teachers |
| M2 | One shared C# Masked ISMCTS source compiled into research and live v3; complete legal incumbents, cash/conversion/merchant support and masked reply simulation |
| M3 | Installed, separately archived 1.4.9; all six styles, 2/5/12-second clocks, old 10-second saves and resource lifecycle preserved |
| M4 | New immutable six-style bank, direct 1,000 ms lab budget, five training identities plus held-out Pottery Flex, authored workers and private runtime bundle |
| M5 | Fresh recipe 2.2.0, audited heuristic BC, self-play curriculum, same-run population/resume, visit distillation, evaluation/retention and current lab guide; earlier algorithm/config entry points retired |

The new model has **1,840,109 total parameters**. Constructor, checkpoint and export checks enforce the current two-million cap; the separately declared later goal/personality extension cap is 2.10 million. Padding logits are negative infinity and have zero softmax probability. Oversized legal candidate sets are chunked with a global denominator, rather than truncated. Root/interior top-K starts at 16/8, widens with visits and protects setup, industry and sale/beer families.

## Executed checks

| Check | Passed evidence |
| --- | --- |
| Focused Python contracts/numerics | 5 tests: pins/configs, fresh model/compatibility, CPU batch and candidate-chunk parity, cancellation, protected expansion/unpruning and completed-visit teacher/forest arithmetic |
| Shared C# runtime | 88 assertions; all six four-player styles, two/three-player compatibility, isolation/masking, actual completed iterations, four-seat statistics and clocks/saves |
| Managed asynchronous execution | 10 checks, including delayed completion after cancellation |
| Native Unity compact style/difficulty matrix | 8 cases, 1,212 assertions, all six styles and three clocks; no native errors |
| Native resource lifecycle | 20 checks; 18 jobs submitted/completed, 15 cancellation requests, maximum 1 executing and 0 remaining |
| Industry conversion | 435 assertions |
| Credit/cash/merchant-beer regressions | 72 credit + 60 cash/beer assertions |
| Saved-profile compatibility | 54 assertions each for producers 1.3.2, 1.4.0 and 1.4.3 |
| Installed native startup/menu/game creation | 259 assertions, including all 20 resource checks; installed DLLs match the candidate |
| Runtime transfer/preflight | 339 privately bundled members; two complete rules-parity games; current bank admission and CPU backward pass |
| Fresh end-to-end learning | Audited heuristic BC; two complete four-player training games; same-run resume to generation 2; four seat-rotated evaluation games; retention preserves required weights |
| NN search and distillation | 3 completed paths on 3 root alternatives; one worker batches up to 3 positions; one separate supervised distillation update |
| ONNX | Two dynamic-shape CPU parity batches; maximum finite-output absolute error `3.624e-5` |
| Selected-backend probe | Same 15-second allowance for top-K and full expansion; 392 complete legal choices retained; legal complete-turn incumbents in both settings |

The tiny heuristic fixture's held-out BC loss was **4.6023 → 4.5283 → 4.3784** across three epochs. It checks data reuse and optimization, not generalization or strength. Audited heuristic archives may differ only in their old architecture-source hash; rules, features, encoder, targets and provenance still match. Neural checkpoint/replay compatibility remains exact, and old neural initialization/import attempts are rejected.

The functional NN search used 26 evaluated positions in 14 batches, maximum batch 3, completing three paths in about **9.61 seconds** on CPU. The independent probe completed 1 top-K path and 3 full-expansion paths within their clocks. These are small timing/coverage observations with smoke weights, not evidence that one expansion setting is stronger. Incomplete paths contribute no visits or teacher evidence; inadequate distinct completed alternatives skip distillation.

## Fingerprinted evidence

The final learning/search implementation identity is:

`e4814f99e3a09b084bf98f4e5180d0c61a76fa2cb54cefcc566cad2d9687bb2f`

The aggregate [receipt](../artifacts/masked-exit/exit-summary.json) pins the current configs, source identity, bank, distribution, private bundle and individual reports. Its source-bound integration report is [here](../artifacts/masked-exit/integration-1790822680084331900/report.json); the separate [decision probe](../artifacts/masked-exit/probe.json) uses a generation from that same fresh run. The current `state.json` signature matches the implementation identity. Initial failures during development remain diagnostics, rather than current exit evidence.

Production DLL/source fingerprints and fresh native reports are in `artifacts/patch-build/build-manifest.json`, `local-ai-qualification.json`, `validation.json` and `local-ai-installed.json`. Only unchanged `Assembly-CSharp.dll` and `BrassPatch.dll` retain the prior package's statistics/rules/UI evidence. Frozen earlier banks and the complete 1.4.8 archive inventory remain unchanged. No earlier search strength claim transfers to this bank.

New bank manifest SHA-256:

`2075067f68824f0aa452bd1bc676e74c7f7e1f8414dab8ba2c76f63170f8a7dd`

Private transfer ZIP SHA-256:

`35345bc54785728c5faf8b563abc67fe216ccf3790b33acb7f17b82454ae8bf0`

The worker/bank files, new contracts and source are eligible for Git; proprietary DLLs, generated checkpoints, exit reports and release packages remain ignored. Existing curated bootstrap weights are archival only.

## Reproduce compact checks

Run Python commands inside the project's configured conda environment on the lab, or with the corresponding local dependencies:

```sh
python /data/not_backed_up/yxu209/ai-training/scripts/verify-masked-ismcts.py
python /data/not_backed_up/yxu209/ai-training/scripts/verify-masked-ismcts.py --integration
```

The integration check optionally exercises the archived tiny heuristic manifest when it is privately available. Without that fixture it checks fresh random self-play; the production recipe uses the audited manifest supplied by `--demonstrations`.

Windows live-game checks:

```powershell
& ./tests/MaskedISMCTS/run-runtime.ps1
& ./tests/BrassLocalAI/run-execution.ps1
& ./tests/BrassLocalAI/run-unity.ps1 -MaskedQuick
& ./scripts/test-installed-local-ai.ps1 -Smoke
```

Follow [release/rollback instructions](MASKED_ISMCTS_1.4.9.md) and the [fresh conda training guide](TRAINING_RECOVERY_LAB.md). Future personality/goal fine-tuning remains after substantial Step 5 training.
