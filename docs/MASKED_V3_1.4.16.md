# Masked v3 patch 1.4.16

Installed on 2026-10-01. This small update sets **two search workers in the game**, following the user's
final preference. Lab training and NN distillation retain **four workers** and
the **24-table** preset. The era economics and merchant-development fixes from
[1.4.15](MASKED_V3_1.4.15.md) remain the same.

Game thinking options remain **2/5/12 seconds**. Saved strategy identities
remain `brass-local-ai/1.4.15`, including saved clocks; no assignments or user
saves are rewritten. Four-owner lab and two-owner game forests are checked
separately, with private states and completed-return merging. Resource modes,
cancellation, retirement and input guards are retained.

Only the authored AI helper changes. Qualified UI, Card Tracker, city locator
and rules binaries remain byte-identical. The atomic installer retains the
previous release. The frozen lab bank stays **`bank-3f7221cadcd472aafdb9`**.

The strategy's last eight-game comparison used one second/four workers for both
versions: **78.75 versus 76.0 mean VP, four wins each**. This update does not
claim a new strength result for two workers. See [diagnosis and checks](MASKED_V3_ERA_REPAIR.md)
and [the updated conda lab guide](TRAINING_RECOVERY_LAB.md).
