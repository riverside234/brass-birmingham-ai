# Brass: Birmingham patch 1.4.23

Date: 3 October 2026 (America/Chicago).

Status: **installed and verified** on 3 October 2026.

## Game History

- The main-menu **Online Game** entry becomes **Game History** from its first visible frame.
- Completed local full and introductory matches are recorded automatically, for two to four players. Opening Statistics is not required.
- The scrollable light history page retains the latest **10 distinct games**. Each row shows mode, recorded date, final ranks, player colors and VP. Focus/hover reveals full names and opponent descriptions.
- Selecting a game opens the **same ten-page Statistics panel** used after a match, with the same calculations, plots, filters, evidence and text controls. **Back to history** restores the selected row and scroll position.
- The default text preset is **Larger (115%)**. Gamma remains unavailable and network play remains disabled.

## Statistics fixes and plot improvements

The same changes apply to live end-game Statistics and archived reports:

- Player names are literal text, preventing formatting-like nicknames from changing font sizes, colors or visible labels.
- Long chart row labels receive enough vertical space. Income/cash curve endpoint labels retain their intended right gutter.
- Geography labels avoid each other, with light backing to keep map edges clear of text. Heatmap tooltips and exact-data values use **VP** or **units**, instead of confusing categorical axis names.
- Horizontal charts and timelines repeat their number scale at the top, so the scale is visible before scrolling through a long plot.
- Subtle alternating bands make rows easier to follow. Stacked bars show counts or percentages inside segments where space permits; exact values remain available for every segment.
- Industry VP uses a blue intensity scale. Continuous heatmaps have a visible color ramp, explicit units and a labeled total column. Turn-position heatmaps have separate colored keys for each position.

The [Statistics review](../rules-audit/STATISTICS_PANEL_REVIEW_1.4.23.md) records the reproduced defects and data checks. Plot inspection targets **1920 × 1200**, as requested; existing 1080p layout regressions are retained.

## Storage and retention

The archive lives under Unity's per-user data directory:

`C:/Users/BobXu/AppData/LocalLow/Cublo/Brass Birmingham/GameHistoryStatistics/history.json`

One flushed, atomic bank replacement commits a new report and evicts the oldest together. A transient `history.pending.json` supports recovery. The canonical bank contains at most ten report snapshots; retired games retain only a hashed identity and sequence to prevent duplicate re-admission.

Recording preserves the first admission order and **Recorded** date when the same finished game is reviewed again. A full report cannot be replaced by a weaker summary. Summary-only and unavailable reports are explicitly labeled. A corrupt/unsupported individual report does not hide other readable rows.

The patch does not backfill an invented previous ten games by scanning saves. Recording begins with this patch; an older completed save encountered normally can be recorded on first observation. Saved games are not pruning targets. Automated checks use disposable fixture storage.

## Validation and installation record

The candidate derives from accepted **1.4.22**. Binary comparison confirms that **14,268 unrelated method bodies** and all embedded resources are preserved. The AI DLL and its 48 profiles are unchanged; this release does not claim a fresh AI-strength benchmark.

| Candidate native suite | Passed checks |
| --- | ---: |
| Statistics | 5,613 |
| Game History, storage and viewer reuse | 3,741 |
| Main menu / AI selection | 492 |
| Card Tracker | 458 |
| Presentation lifecycle | 184 |
| Link replay | 765 |

All six suites passed without reported native errors. History coverage includes nine completed match fixtures, all ten Statistics pages, full-field/derived-model persistence checks, latest-ten retention, duplicate suppression, summary upgrades, corrupt reports, interrupted publication and file locking. Statistics review checks raw-entry totals, player/era filters, resource tabs, exact units, long names and rendered geometry through both entry paths.

The native runner captured **30 distinct plots at 1920 × 1200 in each entry path**. Manual review covered the live plots and archived examples across all ten pages, including the corrected map, numeric scales, stacked labels, resource heatmaps and efficiency plots. Existing 1080p automated layout regressions were retained; no additional 1080p manual plot review was performed.

The installer passed **74 checks using artificial files**, then installed the qualified three-DLL package at **15:32 Chicago time on 3 October 2026**. The recoverable 1.4.22 binaries and receipt are in [the preceding-release backup](../artifacts/patch-recovery/4e67e6c8354e454ea2df72efe499610f). Offline-only status is retained.

| Installed file | SHA-256 |
| --- | --- |
| Assembly-CSharp.dll | `50BF8A2F8A167C1FE5679D37C72A5D5D1F1B01BDFA144EBE5811F5DAF7552699` |
| BrassPatch.dll | `7191226A323C6ED4E1C949BDF1464902EC48DF3DC6BD986ECF7131D7B42F9069` |
| BrassLegacyAI.dll | `72BFD3599C0FF591C9369E8674CF297A2CE9F35E98FB7C1EAC4C0FF3E70F2EFF` |

The installed game passed **3,741 History**, **492 Menu** and **5,613 Statistics** native checks, with fingerprints matching the qualified candidate. These used disposable fixture storage, not personal saved games or history.

Release binaries, source snapshots, test reports and screenshots are retained in [the 1.4.23 archive](../artifacts/patch-releases/1.4.23). The archive inventory and staging verifier check every retained file by SHA-256.

Implementation plan: [Game History](../rules-audit/GAME_HISTORY_STATISTICS_PLAN.md).
