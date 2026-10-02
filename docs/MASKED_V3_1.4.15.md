# Masked v3 patch 1.4.15

Superseded by [patch 1.4.16](MASKED_V3_1.4.16.md), which sets two game search
workers while keeping four for lab training. Strategy and clocks are unchanged.

Installed on 2026-10-01 at the user's request after correctness checks and the
eight-game comparison. All six v3 styles use **four private Masked ISMCTS root
workers**. Easy/Mid/Hard thinking options remain **2/5/12 seconds**; saved
ten-second assignments remain compatible. The backend's CPU/GPU resource
management, cancellation, input guards and retirement are retained.

Canal priorities retain income, completion and carry-over investments. Rail
priorities price real rail/coal/beer costs and preserve feasible industry
construction/sales. Development and loans now target useful, funded upgrades
instead of repeated tile discards or a fixed £15 borrowing threshold. Free
merchant development considers the follow-up investment and can skip unusable
upgrades. Industry and link forecasts share the same uncertain flip event.

The final one-second/four-worker comparison against the previous frozen Masked
v3 completed eight games: **78.75 versus 76.0 mean VP, four wins each**. The
candidate's average rank was 2.375 versus 2.625. This does not demonstrate a
win-rate improvement. See [diagnosis and exit evidence](MASKED_V3_ERA_REPAIR.md).

Only `BrassLegacyAI.dll` changes from installed 1.4.13. Qualified UI, Card
Tracker, city highlighting and rules binaries are retained byte-for-byte.
Prior binaries are backed up by the atomic installer; user saves are not edited.

Training uses `bank-3f7221cadcd472aafdb9`. Imitation uses one-second/four-worker
searched v3; early RL uses its search-free variant and later switches to the
searched variant as v3 tables fall to 5%. NN distillation stays Masked ISMCTS
plus the fresh 1.84M MLP-ResNet with four workers and central inference batching.
The lab preset queues **24 tables** under a shared 60-slot CPU admission budget
within the 64-core allocation. Follow the [updated conda guide](TRAINING_RECOVERY_LAB.md)
and transfer the newly exported private rules bundle before starting a new run.
