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

