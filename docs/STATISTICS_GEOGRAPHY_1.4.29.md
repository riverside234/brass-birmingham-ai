# Patch 1.4.29 — Statistics geography

Date: 7 October 2026  
Base: accepted **1.4.27**. Version 1.4.28 was previously withdrawn; this release uses 1.4.29.  
Status: **installed and verified** over 1.4.27; online play remains disabled.

## Using the map

Open **Statistics → Networks & Geography → Geography** from either the completed game or **Game History**.

- Choose **End of Canal** or **End of Rail** beside the player buttons. The board shows the moment of scoring, before links and Canal level-I tiles are removed. Introductory games have Canal only.
- The diagram has no permanent city names. Hover a city dot, merchant diamond or farm triangle to inspect it; keyboard focus opens the same tag. The tag appears below the selected location with space around the dot.
- The light city tag shows physical vacancies, tile owner, industry, level, historical flip status, and owners/counts of directly incident links. A vacancy does not establish permission to build. Connections mean incident owned links, not everyone reachable through the network.
- Owned links use player colors and distinct owner markers. **Printed routes** controls the faint era-specific background network. Zero-VP links are retained; the southwest farm route remains one branched link with three endpoints.
- **Measure** switches between occupancy, industry VP and link contributions. Industry VP is the selected era's industry award. Link contributions can reward multiple incident links; they are not extra industry points. The introductory bonus is labeled separately.
- Player and region filters emphasize the complete board. Other occupants still count toward vacancies. Action, industry and evidence filters are explicitly suspended on Geography; search filters its exact-data ledger.
- Scroll below the map for **Era-end city snapshot** exact data. Use **Network results** for the existing route rankings, link-source plots and supporting boundary records.

## Implementation and corrections

Both entry points use one shared query, renderer and tag. Historical snapshots are derived from existing schema-1 report records and cached per report; hovering does not replay a match or load a live board. Board-relative positions cover all 27 locations, including the central farm west of Cannock and Gloucester east of Worcester.

Snapshot checks reconcile all-player slot counts and ownership. Build, flip and removal revisions select the correct era's occupants, including carried tiles, overbuilds and liquidation. Missing or inconsistent evidence displays an unavailable message. The maximum tag reserve is identical in both eras, preventing the map from changing scale just because the number of occupied slots changes.

The patch preserves the accepted game and AI binaries exactly. An assembly comparison confirms **14,641 existing methods** remain unchanged, with all changes confined to the permitted Statistics types and all embedded resources preserved. Card Tracker placement/layering, player selection, V4 AI, disabled Gamma, gameplay fixes, live-view presentation suspension, History schema 1 and the latest-10 History limit retain their existing behavior.

## Qualification

- Native end-game Statistics: **5,696 checks passed**, no reported test/runtime errors.
- Native Game History: **3,906 checks passed**, including additional pointer, keyboard and repeated-hover checks.
- Manually inspected **16 native captures at 1920 × 1200**, using the default second and largest text presets, in both viewer contexts and both eras. Native geometry checks also cover southern/edge merchants, the farm and below-dot placement.
- Independent synthetic data checks cover Canal cleanup, carryover flip state, overbuild/liquidation, all-player capacity, changing era link ownership, zero-VP branched links, direct-link ownership, and unavailable history. Schema-1 encode/decode produces identical real-match snapshots.
- Installer rejection/recovery: **134 checks passed** using artificial files, including failed-evidence rejection, fingerprint mismatches, locked-file recovery and repeat installation.
- Installed-game Statistics: **5,712 checks passed**, including the additional pointer, keyboard and repeated-hover checks, with no reported test/runtime errors. Installed DLL fingerprints match the qualified candidate.

Tests use generated matches and artificial History/installer fixtures, not personal saved games. Original menu, tracker and AI execution reports retain their original tested-binary fingerprints; they are not presented as fresh executions against this helper DLL.

## Files and recovery

Candidate and evidence: `artifacts/statistics-geography-build/`  
Release archive after installation: `artifacts/patch-releases/1.4.29/`  
Installation receipt: `artifacts/patch-installation.json`  
The installer records the recoverable 1.4.27 backup in `previousReleaseBackup` and preserves offline mode and configured save paths. No report schema migration is needed.

Implementation plan: [Statistics geography map revision](../rules-audit/STATISTICS_GEOGRAPHY_MAP_REVISION_PLAN.md).
