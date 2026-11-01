# NN industry focus tags — local release 1.4.46

Installed on 2026-10-31. **NN → Alpha Test** now offers **Neutral, Cotton,
Manufacturer, Pottery, Brewery, Iron and Coal**. Each NN seat can choose its own
tag and difficulty. Apply saves the selection; Cancel discards the draft. Tags
restore when creating a game in a fresh process, and remain remembered when
switching the table between NN and V4. The two AI families still cannot share a
table. The roster supports filtering, keyboard navigation and scrolling at the
largest text size.

The selected tag reaches the existing g890 test NN through the exact
`step5.policy.context_for` style/intent encoding. The Rust worker accepts only the
seven matching profiles, checks the ticket against the input profile, and
rejects a tag with the wrong encoding. Native saves bind each seat's tag and
profile in the assignment hash. Older preferences and signed neutral saves
without `nn_focus` retain neutral behavior without rewriting saved payloads.

This patch retains search_v1, V205 disabled, the installed test model, ONNX Runtime
CPU inference, two independent sessions with one compute thread per session,
K8/K16 menu choices, temperature choices and all four search allowances. Search
continues to have its existing additional one-second delivery grace. The build
uses the latest installed K32-removal/public-history-bonus source snapshot, with
nine scoped managed source changes and only the local Rust transport file
overlaid. Training and lab qualification bundles are outside this update.

Short validation passed:

- 21 Rust worker tests, including all seven encodings and unknown-profile rejection.
- 831 UI/preference checks and 758 checks in a separate fresh Unity process,
  covering independent tags, mixed time budgets, Apply/Cancel, filtering, family
  switching, actual game creation, save/load, tampering and old neutral saves.
- Seven actual 4-second/K16 native searches, plus mismatched encoding/ticket rejection.
- Complete 4-second/K16 Unity turns for Neutral, Cotton and Coal: both actions
  committed and the completed-turn checkpoint reloaded successfully.
- Interrupted installation recovery and exact rollback on a disposable copy,
  including an unchanged save sentinel.
- The installed game passed 764 checks on a 4-second/K16 Cotton turn, including
  both actions and save/load. Total measured delivery was 4,694.6 ms, within the
  existing 5,000 ms total deadline. All three managed assembly hashes and 218
  native package file hashes match the candidate. All 146 user game/history files
  remain unchanged. The game and native worker are closed.

These are functional smoke tests, not a playing-strength comparison. Preference
restart tests transfer the exact serialized preference bytes between fresh
Unity processes through the existing isolated self-test store. The first UI
attempt caught and fixed a family-switch branch before installation; its failed
evidence is retained alongside the passing run.

Evidence: [qualification](../artifacts/nn-industry-focus/20261031/release-1.4.46/qualification.json),
[seven-tag native smoke](../artifacts/nn-industry-focus/20261031/protocol-smoke.json),
[installation recovery/rollback](../artifacts/nn-industry-focus/20261031/upgrade-test.json),
[installed verification](../artifacts/nn-industry-focus/20261031/installed-verification.json).

With the game closed, `scripts/install-nn-industry-focus.ps1 -Action Rollback`
restores the exact previous runtime and installation receipt. `-Action Recover`
handles an interrupted installation. The installed receipt pins the retained
backup. Industry-tagged saves require the corresponding tag-capable runtime.
