# Single neutral NN menu — local release 1.4.47

Installed on 2026-11-01. **NN → Alpha Test** again offers one NN option with
**Neutral** context. The six industry tag tiles are removed, the original card
layout is restored, and setup rows display Alpha Test. Former industry-tag menu
preferences normalize to neutral when creating a game, retaining the other setup
settings. Apply/Cancel, saved preferences and independent seat difficulties remain
available.

Only `BrassLegacyAI.dll` and the installation receipt are replaced. The existing
Rust engine, model, search_v1, V205-disabled configuration, K/temperature controls
and native package are retained. Signed assignments in existing games remain
readable with their original identities; the backend retains the prior profile
support for save compatibility. New NN games offer only the neutral profile.

Short validation passed 1,048 UI/preference checks and 820 checks after a fresh
Unity restart. This includes all six former industry preferences reverting to the
single neutral choice, unavailable industry choices being rejected, Apply/Cancel,
mixed difficulties and save compatibility. A complete 4-second/K16 Neutral turn
passed 764 checks, including both actions and a completed-turn save/load. The
largest-text layout was visually inspected at 1080p. Disposable installation tests
passed interrupted-install recovery, exact rollback and save preservation.

Evidence: [qualification](../artifacts/nn-neutral-ui/20261031/release-1.4.47/qualification.json),
[installation checks](../artifacts/nn-neutral-ui/20261031/upgrade-test.json), and
[installed smoke](../artifacts/nn-neutral-ui/20261031/installed-neutral/verification.json).

`scripts/install-nn-neutral-ui.ps1 -Action Recover` handles an interrupted install.
`-Action Rollback` restores the exact previous 1.4.46 runtime and receipt using the
backup recorded by the installer. Run these commands with the game closed.
