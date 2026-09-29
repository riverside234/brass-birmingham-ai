# B09: choosing industries during income debt

The patched local/hotseat application saves a pending settlement before prompting an indebted human. The existing question popup lists the owner's name/color, region, industry type and level, flipped status, liquidation proceeds, and remaining shortfall. **Next industry** cycles through all owned alternatives. **Liquidate** removes exactly the chosen tile, saves the resulting events, then continues settlement. AI decisions are confined to AI-owned tiles. The round ends only after every player has covered the shortfall or has no industries left. Surplus proceeds and the original VP-loss cap are retained.

No new Unity assets or save-event types are introduced. `IndustrySellingStarted`, `IndustrySold`, and `IndustrySellingEnded` now reconstruct the debt dictionaries, removed-industry summary, VP-loss summary, pending phase, and current decision owner. A loaded unfinished settlement resumes its prompt. Failed saves offer a retry without repeating a turn or sale. Repeated clicks, delayed save completion, and callbacks from a different opened game are guarded.

The patch is local-only. The PlayGame, EndTurn, save-completion, and liquidation-entry paths decline network games before executing commands, saving, or starting a spectator/network flow. The existing popup returns to the menu; the separate main-menu patch blocks the ordinary online entry.

## Follow-up review: debt reconstruction and cloning

The old debt handlers were insufficient for a pending human decision: initial debt and sales were maintained only by commands. Moving those mutations into existing handlers fixes that prerequisite for B09. This does not imply that the old synchronous completed settlement necessarily corrupted normal saves; the earlier completed-settlement comparisons passed.

The empty `BoardGameRulesGame` copy constructor, shallow `BrassApplicationGame` copy constructor, and incomplete `Player.Clone` remain unsuitable for a full gameplay snapshot. The inspected ordinary persistence path constructs a game from its configuration and replays events. A repository snapshot branch exists, but the reviewed application contains no producer of gameplay snapshot payloads. Metadata cloning is used when saving metadata; event reconstruction supplies the actual players/game state afterward. No additional high-priority ordinary-play clone defect was established, so these unrelated constructors are not speculatively patched.

## Validation

`tests/BrassPatch/Liquidation/Run-LiquidationTests.ps1` runs under Windows PowerShell/.NET Framework against the woven candidate DLL. The tests exercise the original public production use cases and domain methods with delayed presenter/animation/save stubs, plus the original in-memory repository and game factory. Fixture setup uses deterministic production event types to establish specific indebted states; it is not represented as a naturally played match.

Coverage includes ownership/phase rejection without revision changes, pending/partial/completed save reconstruction, sale surplus, multiple humans, mixed AI/humans, no-assets VP loss, next-choice navigation, duplicate click/EndTurn, save retry, stale callbacks after opening another game, and the local-only online guard. Actual Unity popup rendering and live online services are intentionally outside this harness; no network backend is contacted.

The implementation uses the shipped popup's hide-animation completion to create the next prompt. This avoids retaining button listeners or allowing the old keyboard handler to clear callbacks for the following prompt. The same callback ordering is represented by the delayed presentation stubs.
