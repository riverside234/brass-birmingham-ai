# V3 live search settings — package 1.4.3

**Installed and verified:** package **1.4.3**, 29 September 2026, in the Steam game
directory. Installed-file verification completed at 08:14 UTC (03:14 America/Chicago).

The local v3 S1 executor uses a **10,000 ms whole-turn search allowance** and a
**240-node whole-turn work limit**. Both actions and any replanning share the time
allowance. Search can finish sooner when its work is exhausted; validation and
main-thread application can add overhead after search.

All six profiles in the current research bank,
`bank-af17b1aadbe7f9b21fa7`, specify `work_limit: 240` and `reply_work: 2048`.
Their source files were checked against the frozen manifest and embedded profiles.
The live node limit now matches that bank; reply work remains **2,048**. The bank's
research time setting is 3,000 ms; the live 10,000 ms allowance is the user's
explicit setting, not a change to the research bank.

Only the execution copy receives these settings. Existing saved profiles and
intent identities retain their original data. Strategies, S1 planning, fallback,
cash/conversion policy, research files and save format are unchanged. Play now and
online play remain disabled.

`Assembly-CSharp.dll` and `BrassPatch.dll` are byte-identical to installed 1.4.2,
preserving its player-panel, white-tile, inspection-panel and save/Continue fixes.
The only game-runtime source change is the live budget configuration in
`src/BrassLocalAI/Unity/LocalTurnBridge.cs`. Package qualification verifies this
restricted source diff and retains the earlier rules/UI evidence only for those
unchanged binaries.

Installer hardening: Windows file sharing/locking and replacement error 1175 now
receive up to six attempts at the same atomic replacement, with at most three
seconds of backoff. The live target is never deleted as a workaround; persistent
errors still enter the existing rollback/recovery path. This addresses replacement
failures encountered in the disposable packaging tests.

Fresh checks: the runtime/S1/save suite and 36 native AI turns across six strategies,
2/3/4 players and introductory/full games. Native checks compare the running
executor's time, node and reply limits to the requested settings and bank profiles,
and retain click guards, single completion and save/reload checks.

Fresh validation passed: **1,030 runtime/S1/save assertions** and **36 native turns /
17,403 assertions**, with no unexpected native errors. Every native turn checked
10,000 ms, 240 nodes, the unchanged bank reply budget and S1 mode.

Packaging and interrupted-install recovery passed **79 assertions** after the
bounded retry was added. Installed startup/menu/game-creation smoke passed **57
assertions**. All **125 existing patch hooks** and the original backup were verified.
Personal saves were not opened or modified.

Installed AI DLL SHA-256:
`0862C2AAFB4674EDAC63385BE10196D2C1C3CD40FF62FC2C6B8CCA8D028D0A1A`.
The source, binaries and qualifying reports are preserved under
`artifacts/patch-releases/1.4.3`; prior releases and recovery copies remain available.
