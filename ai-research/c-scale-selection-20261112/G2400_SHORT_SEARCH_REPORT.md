# g2400 short-search C_scale ablation

- **Recorded:** 2026-11-14 (America/Chicago).
- **Checkpoint:** frozen g2400 from `artifacts/step5/joint7-v11-2-g1190-restart-3`.
- **Evidence:** three user-supplied lab console tables, reproduced below at their original precision. The underlying per-position results and lab `summary.json` files were not supplied.
- **Related records:** [report index and historical g1760 study](C_SCALE_SELECTION_REPORT.md), [g2400 long search](G2400_LONG_SEARCH_REPORT.md).
- **Status:** offline parameter comparison; the recommendations here do not constitute a training configuration change.

## Recommendation

With **85% normal / 15% cotton** weighting and mean Q gap as the primary diagnostic, **0.90 is the provisional first choice; 0.70 is the other leading candidate**. **0.50** is the alternative when matching the reference's played action is the primary goal.

For the newest, pooled S3 sample:

| Candidate | Weighted action agreement | Weighted mean Q gap | Interpretation |
|---|---:|---:|---|
| 0.90 | 83.27% | 0.0057275 | Lowest mean Q gap among tested scales |
| 0.70 | 84.44% | 0.0060170 | 1.17 percentage points more agreement than 0.90; close on Q gap |
| 0.50 | 84.95% | 0.0069725 | Highest agreement |

- 0.90 has a **4.8% lower** weighted mean Q gap than 0.70.
- 0.70 has a **13.7% lower** weighted mean Q gap than 0.50, with only **0.51 percentage points less** agreement.
- On these sample means, 0.40, 0.60, 0.80 and 1.00 each have another candidate that is better on both agreement and mean Q gap.
- 0.90 also has the lowest weighted mean Q gap in S1 and S2. Agreement instead peaks at 0.60 in those g2399 samples.
- These are rankings of observed means, not statistically established strength tiers. No confidence intervals or independent match results were supplied.

## Search protocol

| Item | Setting |
|---|---|
| Search implementation | Actual native searches rerun separately at each candidate scale |
| Short candidate budget | K8 / 96 simulations |
| Reference budget | K8 / 3,000 simulations |
| Reference C_scale | 0.10, the default in the supplied commands |
| Search-head weight | w = 0 |
| Opening sampling | T = 2 for the first three actions per seat |
| CPU workers | 60 |
| Candidate scales | 0.40, 0.50, 0.60, 0.70, 0.80, 0.90; also 1.00 in S2/S3 |
| Source groups | Tables collected with normal exploration settings versus cotton-targeted tables |
| Cotton forcing during probes | Disabled for both source groups; group names describe the source games |
| Position eligibility | Archived long-search roots with multiple legal choices and zero recorded insolvency spend, with matching source/model/reward receipts |

The checkpoint stays at g2400 when the source generation changes. Both the reference and candidate searches use its weights. Archived g2396–g2399 labels identify positions; their old Q values are not the new reference result.

This protocol differs from the older g1760 rescore-only tests: changing a candidate's scale here also changes its actual search, including candidate selection. Do not compare the old and new raw match rates as a controlled model-strength improvement.

For S3, eligible positions from all three generations are pooled and shuffled deterministically within the normal/cotton groups. There is no fixed per-generation quota. All scales in a study use the same valid positions. A failed pair is replaced from the remaining pool when possible; all three supplied runs completed their requested quotas without extra attempts.

## Weighting and metric interpretation

Use the requested deployment mix, independently of the number of sampled positions:

```text
weighted metric = 0.85 * normal metric + 0.15 * cotton metric
```

The printed `total` rows use the actual sample mix: **80:20** in S1/S2 and **83.33:16.67** in S3. They are preserved as raw results below but are not the requested 85:15 comparison.

- `match_long_move`: exact agreement with the reference's played action.
- `differs_from_policy`: departure from the frozen policy's top-logit action; a larger value alone does not establish better decisions.
- `mean_Q_gap`: mean of reference Q(reference played action) minus reference Q(candidate played action), with matches contributing zero. This is a signed difference; negative and positive differences can cancel. It is not regret against a known optimal action.
- `miss_Q_gap`: the same difference, conditional on disagreement and direct reference visits to both actions.
- `Q_supported_misses`: supported disagreements / all disagreements. In S3, every search candidate's disagreements are supported; the policy row has two unsupported disagreements.
- Unvisited-action Q values in `mean_Q_gap` use the reference's completed mixed-value estimate. The conditional `miss_Q_gap` excludes unsupported disagreements.

Mean Q gap is the primary proxy here because a different action can have almost the same estimated value, while another disagreement can be costly. Its usefulness depends on the reference Q estimates. Agreement is useful for reproducing reference behavior, and opening T=2 introduces sampling variability into that behavior.

Do not directly weight the two conditional `miss_Q_gap` means by 85:15: first account for each group's supported-miss rate. The table below weights only the unconditional metrics.

Weighted agreement below is reconstructed from the integer miss totals and group sizes, avoiding an extra rounding error from printed percentages. Weighted Q gaps are calculated from the five-decimal console means; additional displayed digits describe the arithmetic, not additional measurement precision.

## Weighted comparison across the three records

| C_scale | S1 agreement | S1 mean Q gap | S2 agreement | S2 mean Q gap | S3 agreement | S3 mean Q gap |
|---|---:|---:|---:|---:|---:|---:|
| policy | 64.55% | 0.0104840 | 62.64% | 0.0113170 | 61.18% | 0.0121245 |
| 0.40 | 85.70% | 0.0070685 | 81.36% | 0.0080250 | 84.31% | 0.0079375 |
| 0.50 | 85.70% | 0.0071705 | 81.64% | 0.0077800 | 84.95% | 0.0069725 |
| 0.60 | 86.00% | 0.0064870 | 81.85% | 0.0073530 | 84.46% | 0.0075335 |
| 0.70 | 84.13% | 0.0067290 | 80.13% | 0.0076260 | 84.44% | 0.0060170 |
| 0.80 | 84.85% | 0.0066060 | 80.64% | 0.0074660 | 83.59% | 0.0065015 |
| 0.90 | 84.73% | 0.0060285 | 80.58% | 0.0069750 | 83.27% | 0.0057275 |
| 1.00 | not tested | not tested | 79.51% | 0.0069975 | 81.76% | 0.0064260 |

S1 and S2 use the same g2399 source pool and overlap; do not add their sample sizes as independent evidence. S3 uses different collection generations. Positions within a table are correlated. The source training positions also participated in model training, so this is a tuning study rather than an independent generalization test.

## Raw console records

### S1: g2399; 200 normal + 50 cotton positions; scales 0.40–0.90.

**total: 250/250 valid positions (250 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 64.4% | 0.0% | +0.01024 | +0.02877 | 89/89 |
| 0.40 | 85.6% | 32.8% | +0.00679 | +0.04713 | 36/36 |
| 0.50 | 85.6% | 33.2% | +0.00688 | +0.04781 | 36/36 |
| 0.60 | 86.0% | 34.4% | +0.00621 | +0.04438 | 35/35 |
| 0.70 | 84.0% | 35.6% | +0.00652 | +0.04074 | 40/40 |
| 0.80 | 84.8% | 34.8% | +0.00640 | +0.04209 | 38/38 |
| 0.90 | 84.8% | 37.6% | +0.00580 | +0.03815 | 38/38 |

**normal: 200/200 valid positions (200 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 65.0% | 0.0% | +0.01121 | +0.03202 | 70/70 |
| 0.40 | 86.0% | 32.5% | +0.00791 | +0.05649 | 28/28 |
| 0.50 | 86.0% | 33.0% | +0.00803 | +0.05736 | 28/28 |
| 0.60 | 86.0% | 34.5% | +0.00730 | +0.05211 | 28/28 |
| 0.70 | 84.5% | 36.0% | +0.00735 | +0.04739 | 31/31 |
| 0.80 | 85.0% | 35.5% | +0.00723 | +0.04819 | 30/30 |
| 0.90 | 84.5% | 37.5% | +0.00672 | +0.04335 | 31/31 |

**cotton: 50/50 valid positions (50 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 62.0% | 0.0% | +0.00637 | +0.01677 | 19/19 |
| 0.40 | 84.0% | 34.0% | +0.00230 | +0.01438 | 8/8 |
| 0.50 | 84.0% | 34.0% | +0.00230 | +0.01438 | 8/8 |
| 0.60 | 86.0% | 34.0% | +0.00188 | +0.01344 | 7/7 |
| 0.70 | 82.0% | 34.0% | +0.00321 | +0.01784 | 9/9 |
| 0.80 | 84.0% | 32.0% | +0.00307 | +0.01922 | 8/8 |
| 0.90 | 86.0% | 38.0% | +0.00211 | +0.01509 | 7/7 |

### S2: g2399; 400 normal + 100 cotton positions; scales 0.40–1.00.

**total: 500/500 valid positions (500 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 62.6% | 0.0% | +0.01112 | +0.02972 | 187/187 |
| 0.40 | 81.4% | 34.8% | +0.00786 | +0.04224 | 93/93 |
| 0.50 | 81.6% | 35.8% | +0.00766 | +0.04161 | 92/92 |
| 0.60 | 81.8% | 37.0% | +0.00723 | +0.03973 | 91/91 |
| 0.70 | 80.0% | 37.8% | +0.00754 | +0.03770 | 100/100 |
| 0.80 | 80.6% | 37.8% | +0.00733 | +0.03779 | 97/97 |
| 0.90 | 80.6% | 39.4% | +0.00682 | +0.03515 | 97/97 |
| 1.00 | 79.6% | 39.6% | +0.00686 | +0.03361 | 102/102 |

**normal: 400/400 valid positions (400 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 62.7% | 0.0% | +0.01192 | +0.03199 | 149/149 |
| 0.40 | 81.2% | 35.2% | +0.00852 | +0.04542 | 75/75 |
| 0.50 | 81.8% | 36.2% | +0.00814 | +0.04458 | 73/73 |
| 0.60 | 82.0% | 38.0% | +0.00771 | +0.04281 | 72/72 |
| 0.70 | 80.5% | 39.0% | +0.00789 | +0.04048 | 78/78 |
| 0.80 | 80.8% | 39.0% | +0.00788 | +0.04095 | 77/77 |
| 0.90 | 80.5% | 39.8% | +0.00744 | +0.03814 | 78/78 |
| 1.00 | 79.2% | 40.5% | +0.00741 | +0.03569 | 83/83 |

**cotton: 100/100 valid positions (100 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 62.0% | 0.0% | +0.00790 | +0.02080 | 38/38 |
| 0.40 | 82.0% | 33.0% | +0.00522 | +0.02898 | 18/18 |
| 0.50 | 81.0% | 34.0% | +0.00574 | +0.03019 | 19/19 |
| 0.60 | 81.0% | 33.0% | +0.00533 | +0.02806 | 19/19 |
| 0.70 | 78.0% | 33.0% | +0.00613 | +0.02786 | 22/22 |
| 0.80 | 80.0% | 33.0% | +0.00512 | +0.02562 | 20/20 |
| 0.90 | 81.0% | 38.0% | +0.00434 | +0.02286 | 19/19 |
| 1.00 | 81.0% | 36.0% | +0.00466 | +0.02454 | 19/19 |

### S3: Pooled g2396, g2397 and g2398; 500 normal + 100 cotton positions; scales 0.40–1.00.

**total: 600/600 valid positions (600 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 61.0% | 0.0% | +0.01215 | +0.03140 | 232/234 |
| 0.40 | 84.2% | 33.5% | +0.00804 | +0.05079 | 95/95 |
| 0.50 | 84.8% | 34.8% | +0.00709 | +0.04674 | 91/91 |
| 0.60 | 84.3% | 34.8% | +0.00761 | +0.04857 | 94/94 |
| 0.70 | 84.3% | 35.7% | +0.00610 | +0.03894 | 94/94 |
| 0.80 | 83.5% | 36.5% | +0.00657 | +0.03981 | 99/99 |
| 0.90 | 83.2% | 35.8% | +0.00582 | +0.03455 | 101/101 |
| 1.00 | 81.7% | 37.3% | +0.00648 | +0.03535 | 110/110 |

**normal: 500/500 valid positions (500 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 62.8% | 0.0% | +0.01185 | +0.03216 | 184/186 |
| 0.40 | 85.6% | 33.0% | +0.00700 | +0.04861 | 72/72 |
| 0.50 | 86.0% | 34.6% | +0.00593 | +0.04236 | 70/70 |
| 0.60 | 85.6% | 34.2% | +0.00683 | +0.04742 | 72/72 |
| 0.70 | 85.4% | 35.0% | +0.00530 | +0.03633 | 73/73 |
| 0.80 | 84.4% | 36.0% | +0.00587 | +0.03761 | 78/78 |
| 0.90 | 84.2% | 35.4% | +0.00491 | +0.03106 | 79/79 |
| 1.00 | 82.6% | 36.8% | +0.00591 | +0.03395 | 87/87 |

**cotton: 100/100 valid positions (100 attempted)**

| C_scale | match_long_move | differs_from_policy | mean_Q_gap | miss_Q_gap | Q_supported_misses |
|---|---:|---:|---:|---:|---:|
| policy | 52.0% | 0.0% | +0.01368 | +0.02849 | 48/48 |
| 0.40 | 77.0% | 36.0% | +0.01325 | +0.05759 | 23/23 |
| 0.50 | 79.0% | 36.0% | +0.01288 | +0.06133 | 21/21 |
| 0.60 | 78.0% | 38.0% | +0.01152 | +0.05235 | 22/22 |
| 0.70 | 79.0% | 39.0% | +0.01008 | +0.04800 | 21/21 |
| 0.80 | 79.0% | 39.0% | +0.01008 | +0.04800 | 21/21 |
| 0.90 | 78.0% | 38.0% | +0.01036 | +0.04710 | 22/22 |
| 1.00 | 77.0% | 40.0% | +0.00935 | +0.04065 | 23/23 |

## Reproduction commands

Run from the lab repository root after updating the scripts.

S1 (the original smaller scan):

```bash
python scripts/policy-c-scale/short-search.py \
  --generation 2400 --collected-generation 2399 \
  --normal-positions 200 --cotton-positions 50 \
  --lanes 60 --limit 250 --ref-scale 0.1 \
  --scales 0.4 0.5 0.6 0.7 0.8 0.9
```

S2 (expanded g2399 scan):

```bash
python scripts/policy-c-scale/short-search.py \
  --generation 2400 --collected-generation 2399 \
  --normal-positions 400 --cotton-positions 100 \
  --lanes 60 --limit 500 --ref-scale 0.1 \
  --scales 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

S3 (latest pooled scan):

```bash
python scripts/policy-c-scale/short-search.py \
  --generation 2400 --collected-generations 2396 2397 2398 \
  --normal-positions 500 --cotton-positions 100 \
  --lanes 60 --limit 600 --ref-scale 0.1 \
  --scales 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

## Sources and limits

- Numerical results: the user's S1, S2 and S3 console outputs in the C_scale study conversation.
- Implementation: [short-search.py](../../scripts/policy-c-scale/short-search.py) and [shared study reader/runner](../../scripts/policy-c-scale/_study.py).
- Source-pool evidence: [g2400 export state](../../results/joint7-v11-2-g1190-restart-3-g2400-20261115T022541658333Z/state.json). The export timestamp is UTC; the update date above uses America/Chicago.
- No new lab search was executed while writing this report. No production search setting was changed.
- Actual VP/win-rate superiority and statistical separation of 0.90 versus 0.70 remain unmeasured here.

