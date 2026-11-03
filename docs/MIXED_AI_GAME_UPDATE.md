# Per-seat AI and 2x animations — 1.4.51

The local setup panel selects a controller for one seat at a time. V4, Alpha Test,
Beta and IL can share a table, including all-AI tables. The “AI for whole table”
controls are removed. Human, V4 and NN have direct pages; the NN roster is generated
from `NativeModel.All`, so future properly packaged models use the same selection,
save and dispatch path without combination-specific restrictions. Gamma remains
an unavailable placeholder.

Each seat retains its own model, difficulty, K and temperature. Menu preferences
survive reopening/restarting; version-2 whole-table preferences migrate to per-seat
defaults. The current NN models still require standard four-player rules. Model
weights, native search engines, search_v1, V205 being disabled, and the two
independent one-thread ONNX Runtime sessions are unchanged.

New tables containing NN seats use `brass-local-mixed-save/1.0.0`. Assignment hashes
bind each physical seat's backend, exact model/export/profile and execution settings.
V4 intent and NN recall/clock records coexist. Dispatch and liquidation select the
actor's backend, and model changes retire the previous native process under the
existing exclusive scheduler. Old V4 and homogeneous NN saves remain readable
without rewriting their identities. Rollback to 1.4.50 cannot read new mixed saves;
retain 1.4.51 to resume those games.

Animation startup and action-reset speed default to 2x. Fast-forward is still 2x.
Only the two float initializers in `UIProperties` change; gameplay time, search
budgets and other game methods are unchanged.

Validation uses short Unity smokes: independent choices and cancellation, separate
process preference restore, model-bound save/load and tamper rejection, mixed live
4s/K16 turns with both actions and completed-turn checkpoints, stale cancellation,
and animation startup/reset. UI review prioritizes 1920×1080 and 1920×1200 at the
largest text setting; 1280×720 is also checked. This is functional validation,
not a strength or long-match claim.

Build/evidence: `artifacts/mixed-ai/20261102/`. Reproduction:

```powershell
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py build
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py ui-write
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py ui-read
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py live
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py qualify
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py upgrade
./scripts/install-mixed-ai.ps1
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py installed-read
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py installed-live
./artifacts/p6e/Scripts/python.exe scripts/build-mixed-ai.py audit
```

The builder preserves existing evidence. Reproduction needs fresh artifact labels
or a clean matching baseline. Installation verifies the exact 1.4.50 receipt and
files, records a recovery backup, and atomically updates only Assembly-CSharp.dll,
BrassLegacyAI.dll and the installation receipt. The unchanged BrassPatch.dll,
native packages and user save/history files are fingerprinted. Rollback/recovery:
`./scripts/install-mixed-ai.ps1 -Action Rollback` / `-Action Recover`.
