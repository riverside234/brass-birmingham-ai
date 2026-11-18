# g2400 long-search C_scale ablation

- **Updated:** 2026-11-14 (America/Chicago).
- **Checkpoint:** frozen g2400 from `artifacts/step5/joint7-v11-2-g1190-restart-3`.
- **Related records:** [report index and historical g1760 study](C_SCALE_SELECTION_REPORT.md), [g2400 short search](G2400_SHORT_SEARCH_REPORT.md).
- **Measurement status:** the g2400 long-search result table has not been supplied in this conversation or found in the local result exports. This record documents the updated protocol and the existing configured value; it does not claim a newly measured long-search winner.

## Updated test protocol

| Item | Setting |
|---|---|
| Candidate searches | Actual native K8 / 640-simulation searches at each scale |
| Reference | K8 / 3,000 simulations, default C_scale = 0.10 |
| Candidate scales | 0.10, 0.15, 0.20, 0.25, 0.30 |
| Frozen model | g2400 |
| Original requested source | g2399 archived positions |
| Original requested quota | 200 normal + 50 cotton positions |
| Search-head weight | w = 0 |
| Opening sampling | T = 2 for the first three actions per seat |
| CPU workers | 60 |
| Intended comparison mix | Normal:cotton = 85:15 |

The current reader also supports pooling several source generations and explicit group quotas. That capability alone is not evidence that a new long-search run was completed.

## Recorded configuration and historical decision

The [g2400 training export](../../results/joint7-v11-2-g1190-restart-3-g2400-20261115T022541658333Z/state.json) records long-search C_scale **0.10** and short-search C_scale **0.70** in `teacher_phase.revision_g1491.count_search`. These are the training settings recorded by that export, not the outcome of a new g2400 long-search ablation.

The [historical g1760 report](C_SCALE_SELECTION_REPORT.md#historical-g1760-record-2026-11-12) recommended long-search **0.15** at K8 after considering its older tables and a search-noise argument. Its K8 Test C measured the lowest overall reference-move NLL at **0.10 (1.3737)** and the lowest held-out NLL at **0.20 (1.3393)**. Those historical values remain attributed to g1760.

A new g2400 numerical ranking is pending the user's long-search table. Do not use the g2400 short-search mean Q gaps as long-search results.

## Metrics to record

The long-search script retains the legacy column name `short_scale`; in this report it means the candidate **long-search** C_scale.

- `match_long_move`: candidate played action equals the reference played action.
- `match_long_label`: candidate improved-policy argmax equals the reference improved-policy argmax.
- `move_NLL`: negative log probability of the reference played action under the candidate's improved policy, computed from the candidate's own search statistics.

For the label-distillation comparison, inspect **reference-move NLL** alongside the two agreement metrics. NLL evaluates the candidate distribution against sampled reference moves; it does not itself prove true-outcome calibration or stronger training.

When normal and cotton result rows are available, compute each aggregate as `0.85 * normal + 0.15 * cotton`. Keep their raw sample-weighted total separately. A 200/50 sample has an 80:20 raw mix.

## Reproduction command

This is the configured test command, not a claim that its results have been recorded:

```bash
python scripts/experiments/policy-c-scale/long-search.py \
  --generation 2400 --collected-generation 2399 \
  --normal-positions 200 --cotton-positions 50 \
  --lanes 60 --limit 250 --ref-scale 0.1 \
  --scales 0.10 0.15 0.20 0.25 0.30
```

Sources: [long-search.py](../../scripts/experiments/policy-c-scale/long-search.py), [shared study reader/runner](../../scripts/experiments/policy-c-scale/_study.py), and the g2400 export linked above. No new lab search or production configuration change was performed for this documentation update.

## g3270 follow-up: K8/640 and K8/96 (2026-11-18)

These are the console results supplied in the g3270 follow-up. The exact checkpoint, source generations, and command-line settings for these runs are unverified. Both candidate budgets report 700/700 valid normal positions and 200/200 valid cotton positions, with 700 and 200 attempts respectively. The combined rows below preserve the reported 700:200 sample mix.

Both budgets were tested with [long-search.py](../../scripts/experiments/policy-c-scale/long-search.py) against a K8/3000 reference. The output column `short_scale` is the candidate C_scale for each budget. `target_KL` measures KL from the reference improved policy to the candidate distribution. `differs_from_policy` measures how often the candidate's played action differs from the raw network policy's top action. The `policy` row is the raw policy baseline.

### K8/640

#### Combined, 900 valid positions

| short_scale | target_KL | match_long_move | match_long_label | move_NLL | differs_from_policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| policy | 0.528684 | 60.4% | 75.4% | 1.5109 | 0.0% |
| 0.10 | 0.318986 | 85.4% | 80.7% | 1.4080 | 35.4% |
| 0.15 | 0.273688 | 88.3% | 83.4% | 1.4210 | 35.8% |
| 0.20 | 0.258244 | 89.8% | 85.0% | 1.4510 | 36.2% |
| 0.25 | 0.282275 | 91.4% | 85.1% | 1.5217 | 37.6% |
| 0.30 | 0.295533 | 91.9% | 86.1% | 1.5906 | 38.1% |

#### Normal, 700 valid positions

| short_scale | target_KL | match_long_move | match_long_label | move_NLL | differs_from_policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| policy | 0.554590 | 59.3% | 75.1% | 1.5776 | 0.0% |
| 0.10 | 0.330736 | 85.3% | 80.9% | 1.4656 | 36.3% |
| 0.15 | 0.274731 | 88.1% | 83.7% | 1.4679 | 36.6% |
| 0.20 | 0.253374 | 90.1% | 85.1% | 1.4907 | 37.1% |
| 0.25 | 0.277449 | 92.0% | 85.0% | 1.5603 | 38.3% |
| 0.30 | 0.289395 | 92.0% | 86.1% | 1.6371 | 38.6% |

#### Cotton, 200 valid positions

| short_scale | target_KL | match_long_move | match_long_label | move_NLL | differs_from_policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| policy | 0.438012 | 64.5% | 76.5% | 1.2772 | 0.0% |
| 0.10 | 0.277859 | 86.0% | 80.0% | 1.2062 | 32.5% |
| 0.15 | 0.270040 | 89.0% | 82.5% | 1.2568 | 33.0% |
| 0.20 | 0.275289 | 88.5% | 84.5% | 1.3118 | 33.0% |
| 0.25 | 0.299168 | 89.5% | 85.5% | 1.3865 | 35.0% |
| 0.30 | 0.317017 | 91.5% | 86.0% | 1.4278 | 36.5% |

### K8/96

#### Combined, 900 valid positions

| short_scale | target_KL | match_long_move | match_long_label | move_NLL | differs_from_policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| policy | 0.528684 | 60.4% | 75.4% | 1.5109 | 0.0% |
| 0.10 | 0.514793 | 81.1% | 75.9% | 1.5113 | 34.3% |
| 0.15 | 0.521966 | 81.4% | 76.1% | 1.5233 | 34.1% |
| 0.20 | 0.532323 | 82.2% | 76.4% | 1.5400 | 34.8% |
| 0.25 | 0.549914 | 82.3% | 77.2% | 1.5619 | 35.0% |
| 0.30 | 0.574406 | 82.3% | 76.9% | 1.5879 | 35.8% |

#### Normal, 700 valid positions

| short_scale | target_KL | match_long_move | match_long_label | move_NLL | differs_from_policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| policy | 0.554590 | 59.3% | 75.1% | 1.5776 | 0.0% |
| 0.10 | 0.540035 | 80.9% | 75.7% | 1.5807 | 35.0% |
| 0.15 | 0.543666 | 81.3% | 76.0% | 1.5906 | 34.9% |
| 0.20 | 0.550549 | 82.0% | 76.4% | 1.6050 | 35.7% |
| 0.25 | 0.567219 | 82.1% | 77.0% | 1.6308 | 35.7% |
| 0.30 | 0.590999 | 82.4% | 76.7% | 1.6593 | 36.4% |

#### Cotton, 200 valid positions

| short_scale | target_KL | match_long_move | match_long_label | move_NLL | differs_from_policy |
| --- | ---: | ---: | ---: | ---: | ---: |
| policy | 0.438012 | 64.5% | 76.5% | 1.2772 | 0.0% |
| 0.10 | 0.426444 | 82.0% | 76.5% | 1.2684 | 32.0% |
| 0.15 | 0.446018 | 82.0% | 76.5% | 1.2879 | 31.5% |
| 0.20 | 0.468531 | 83.0% | 76.5% | 1.3124 | 31.5% |
| 0.25 | 0.489348 | 83.0% | 78.0% | 1.3209 | 32.5% |
| 0.30 | 0.516331 | 82.0% | 77.5% | 1.3377 | 33.5% |

