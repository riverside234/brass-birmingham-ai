# g1730 value-head ablations

The lab comparisons used the frozen g1730 checkpoint from the g1190 restart-3 run, before the search-head refresh. Lower mean squared error (MSE) is better throughout. The fresh fits trained only the head under test with the encoder fixed.

## Datasets and targets

| Dataset | Role in the comparison |
|---|---|
| 256 complete active teacher-buffer games | Search-value fitting data. The initial experiments split these by game into 205 fit games and 51 buffer-validation games. Later experiments fit on all 256. |
| 4 latest official teacher held-out games | External check for the initial buffer-validation runs. These games were outside the 256-game fitting buffer. |
| 100 retained official teacher held-out games | Larger external check containing the four latest games and older official holdouts. The utility-head comparison also evaluated on these games. |
| Ordinary PPO games from iteration 1729 | Utility-head fitting and recent PPO holdout. The accompanying saved-head check reported 304 critic-fit games and 16 whole-game critic holdouts. |

The search-value head learns a correction to the utility head. Its fitting target uses the search-value λ-return. At λ = 1.0 it follows the realized utility-to-go. Values below 1.0 blend in later search-root value estimates. The λ sweeps report MSE against final utility-to-go for all arms. The utility-head width test uses ordinary PPO utility targets. [Search-value target code](../../src/brass_ai/recovery/search_value_phase.py) · [g1730 export](../../results/joint7-v11-2-g1190-restart-3-g1730-20261111T202244924262Z/export-manifest.json)

## Search-value head width

These are means over seeds 4504, 4505, and 4506. Each seed chose the checkpoint with the lowest MSE on the 51 buffer-validation games from steps 16, 64, 128, 256, and 512.

| λ | Hidden width | Selected steps by seed | Fit MSE | Buffer-validation MSE | Four-game official MSE |
|---:|---:|---|---:|---:|---:|
| 1.0 | 32 | 256, 256, 256 | 0.157562 | 0.160011 | 0.155886 |
| 1.0 | 64 | 256, 256, 256 | 0.156775 | 0.160145 | 0.154777 |
| 1.0 | 256 | 128, 128, 64 | 0.157549 | 0.160219 | 0.156127 |
| 0.8 | 64 | 512, 512, 512 | 0.157103 | 0.159731 | 0.155303 |
| 0.8 | 128 | 512, 512, 512 | 0.156611 | 0.159584 | 0.155229 |
| 0.8 | 256 | 512, 512, 512 | 0.156230 | 0.159517 | 0.155236 |

On the four official games, the utility-only baseline (w = 0) was **0.158822** and the saved search head was **0.154610**. At λ = 1.0, fresh width 64 led on these four games. At λ = 0.8, width 256 led on buffer validation and widths 128 and 256 differed by 0.000007 on the four official games.

The saved search head had MSE **0.156799** against the utility-only **0.160845** on the 256 active buffer games. On all 100 official holdouts, its MSE was **0.169906** against **0.167630** for utility alone. That reversal motivated the larger holdout tests.

### Width 32 vs. 64 at λ = 0.8 on the larger holdout

| Hidden width | Fit step | 100-game official MSE | Older-only MSE |
|---:|---:|---:|---:|
| 32 | 256 | 0.167630 | 0.168118 |
| 64 | 256 | 0.167787 | 0.168297 |
| 32 | 512 | 0.167951 | 0.168470 |
| 64 | 512 | 0.168232 | 0.168774 |

Width 32 had lower MSE at both step budgets. At 256 steps its 100-game result matched the utility-only baseline to the reported precision. Both widths had higher held-out MSE at 512 steps.

## Search-value λ ablation

Fresh width-64 fits used all 256 buffered games. The table gives the reported means across seeds on the 100 retained official games. The utility-only baseline was **0.167630**.

| Fit step | λ = 1.0 MSE | λ = 0.8 MSE |
|---:|---:|---:|
| 256 | 0.168449 | **0.167787** |
| 512 | 0.169644 | **0.168232** |

λ = 0.8 had lower MSE at both budgets. Both settings exceeded the utility-only MSE. In the earlier four-game check, the validation-selected width-64 fit at λ = 1.0 averaged 0.154777 versus 0.155303 at λ = 0.8, with selection at 256 and 512 steps respectively.

The follow-up **λ = 0.6 vs. 0.8** was reported qualitatively as favoring **0.8**. The discussion contains no numeric MSE or fit details for the 0.6 arm.

## Utility head width

Fresh utility heads used the same frozen g1730 encoder and ordinary PPO fitting data. The same recent PPO and retained teacher holdouts were used for both widths.

| Hidden width | Fit step | PPO fit MSE | Recent PPO holdout MSE | 100-game teacher holdout MSE |
|---:|---:|---:|---:|---:|
| 32 | 256 | 0.159605 | **0.145219** | **0.168604** |
| 256 | 256 | 0.157575 | 0.146508 | 0.170060 |
| 32 | 512 | 0.158050 | 0.146598 | 0.169940 |
| 256 | 512 | 0.154184 | 0.148028 | 0.171717 |

Width 256 achieved lower fitting MSE at both budgets. Width 32 had lower MSE on both held-out datasets at both budgets. Extending the fit from 256 to 512 steps lowered fitting MSE and raised both holdout errors for both widths.

The numerical tables above come from the lab output pasted in the ablation discussion. The export supplies the checkpoint and run context.
