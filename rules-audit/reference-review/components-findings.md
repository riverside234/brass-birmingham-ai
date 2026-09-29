# Component/setup differential review against npow/brass-birmingham

Reviewed 27 September 2026. **No new P1 defect in the installed Steam patch was established in this component/setup/map scope.** The comparison found 1,825 matching checks and two merchant-mix differences. The official full-board setup illustration supports the Steam four-player merchant mix and contradicts the reference implementation. Do not import the reference merchant mix as a game fix.

## Inputs and reproducibility

- Installed `Assembly-CSharp.dll`: `34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A`.
- Installed `BrassPatch.dll`: `4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D`.
- Reference revision: `fdeb63c6aa92cc4ba44a721482316e0b8db10e06`.
- Reviewed `artifacts/reference-npow-brass/js/gameData.js` SHA-256: `26CBF5E31CC40B126F2F66DE555B64F97A670F5C2B64F1B76C84D2C8CF684B81`.
- Authority: Roxley's publisher-hosted rulebook, local `rules-audit/official-rulebook.pdf`, SHA-256 `BB627A11C769957FBB5F26A210E3957AE27A2BAD054BA9CB3B4FADC6C2BA73E5`; printed pp. 4-5. Publisher URL: https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281.
- Reviewed `docs/LOCAL_PATCH.md` and the historical `rules-audit/BUG_FIX_PLAN.md`; this does not reopen B01-B16 or the save-integrity repair.
- No applicable `AGENTS.md` was found in the workspace file search.

Run in the repository root:

```powershell
pwsh -NoProfile -File rules-audit/reference-review/components-probe.ps1
& 'C:/Users/BobXu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe' rules-audit/reference-review/components-compare.cjs
```

The PowerShell probe rejects unknown assembly hashes, loads the installed patched DLL and helper in a separate process, creates fresh games with two, three, and four players and seed `20260927`, and records their real production setup objects. It never opens persistence or a user's save. The only mutations are fresh in-memory game construction and calling `Build()` on isolated mat tiles to inspect their Canal-era resource production. No installed binary or existing source was modified.

The Node comparator evaluates only the fully reviewed data-only `gameData.js`, in a fresh VM without `require`, `process`, network, or DOM bindings and with a one-second timeout. It never executes the reference's game, AI, UI, or other application code. Retained outputs: `components-dll-data.json`, `components-comparison-results.json`.

## Actual comparison coverage

The 1,825 matches are individual comparisons, including repeated deterministic component values in three player-count setups; they are not 1,825 independent rule interpretations or played games.

| Scope | Comparison |
| --- | --- |
| Industry tiles | All 45 tile positions: type/level multiplicities, cash/coal/iron cost, VP, income increase, link VP, development permission, era eligibility, beer to sell, Canal-era resource production |
| Board | All 20 city slot multisets, both farm brewery slots, complete Canal and Rail link multisets, including the shared Kidderminster-Worcester-southern-farm link |
| Cards | Complete location/industry card multisets for 2/3/4 players, including Cotton Mill/Manufacturer dual cards; totals 40/54/64 |
| Merchant locations | Each location's slot count, minimum player count, bonus type; total active slots; beer attached only to nonblank tiles |
| Initial state | Every player's money, income marker, hand/discard count and link supply; market stocks and wild-card counts |

Two representation differences are intentionally normalized. C# `IsForCanalEra`/`IsForRailEra` describe era restrictions; both false means the tile is legal in either era. C# `Central`/`SouthWest` correspond to reference `northern`/`southern`, and `viaBrewery` becomes a third endpoint on the one shared link. These are not rules disagreements. Brewery resource production was compared in Canal only here; the existing patched build-resource matrix already covers quantities in both eras.

Relevant implementation locations:

- `ai-research/decompiled/BoardGameRules.Entities.Players/PlayerMat.cs:17` (tile data; brewery era quantity function at line 133).
- `ai-research/decompiled/BoardGameRules.Entities.Board/Board.cs:436` (slots/merchants), line 579 (links), lines 622-623 (three-endpoint farm link).
- `ai-research/decompiled/BoardGameRules.Entities.Cards/CardsFactory.cs:53` (card multiplicities).
- Reference `gameData.js:53` (industries), line 127 (cities), line 420 (links), line 467 (cards).

Agreement with another implementation is differential evidence only. This review does not claim that the reference independently proves every printed tile value or card multiplicity. The official p. 4 map/setup illustration was also visually checked; its merchant symbols, farm layout, starting stocks, and location slots are consistent with the Steam data.

## Reference discrepancy: three-/four-player merchant mixes

Production Steam setup uses `MerchantTilesFactory.cs:23-36`: the two-player set contains two blanks, one universal buyer, one cotton buyer, and one manufacturer buyer. Three players add a blank and pottery (lines 33-34). Four players add manufacturer and cotton (lines 35-36).

The reference instead adds pottery and manufacturer at `gameData.js:403`, then universal buyer and cotton at line 406. The unchanged production `StartGame()` data reproduces the disagreement immediately, without an arranged gameplay fixture:

| Players | Installed patch mix | Reference mix | Patch/reference initial merchant beer |
| --- | --- | --- | --- |
| 2 | 2 blank, 1 any, 1 cotton, 1 manufacturer | Same | 3 / 3 |
| 3 | 3 blank, 1 any, 1 cotton, 1 manufacturer, 1 pottery | 2 blank, 1 any, 1 cotton, 2 manufacturer, 1 pottery | 4 / 5 |
| 4 | 3 blank, 1 any, 2 cotton, 2 manufacturer, 1 pottery | 2 blank, 2 any, 2 cotton, 2 manufacturer, 1 pottery | 6 / 7 |

**Adjudication:** The official printed p. 4 setup illustration shows all nine tiles: Warrington has universal/cotton, Nottingham cotton/manufacturer, Shrewsbury manufacturer, Gloucester blank/pottery, and Oxford blank/blank. That is three blank tiles and one universal buyer, exactly the Steam four-player mix. The full-board high-resolution inspection is retained at `artifacts/reference-review/components-official-board.png`. The external reference's four-player extra universal buyer is therefore a confirmed reference-data error, not a current game defect. Its extra buyer also adds one merchant beer under its own `gameState.js:101-115` initialization.

The precise player-count pips on individual merchant tiles are too blurred in the rulebook illustration to independently read all minimum-player assignments. Consequently, the three-player difference is recorded without claiming that this illustration alone proves each tile's three-player eligibility. It does not justify changing Steam: the reference has already failed the complete-set check, and this review has no authoritative evidence against Steam's three-player subset.

**Reachability and severity limits:** Both differences are deterministic setup data, so they occur on every reference setup of the respective player count, regardless of shuffle seed. They could change external-reference selling opportunities and free beer. The Steam four-player data is supported by the official illustration; no illegal Steam action, blocked turn, wrong score, or altered user save was reproduced. No new high-priority Steam finding is reported from this scope. Normal production action flows, Unity visual interaction, and arbitrary seeds were outside this component probe.
