# State and result repairs

`patches.json` redirects the narrow production entry points to `StateFixes.cs`. It targets the audited assembly fingerprint; the central weaver also removes every game-global `DevelopMerchantBonus.Instance` read/write. No helper stores a current game in static mutable state.

| Finding | Implementation |
| --- | --- |
| B01 | Final Rail settlement gate before debt events; ordinary rounds and final Canal retain income settlement. |
| B05/B06 | Lexicographic VP/income-level/cash comparison; equal keys share competition ranks. Stable display order does not consume random numbers. Existing result tuple schema is retained. The summary labels each co-winner, and the obsolete single-winner analytics event is skipped for draws. |
| B07 | Loan menu uses the same three-level lower bound as execution. |
| B12 | Event handlers, turn flow, human/tutorial sale flow, and AI sale helper resolve their owning bonus. Rejected End Turn is checked before event revision advances. |
| B13 | The existing three introductory component events reconstruct breakdowns while applying each original VP change exactly once. Missing legacy breakdowns display an explicit unavailable label instead of crashing. |
| B14 | Exhausted manufacturer stack makes the strategic rule inapplicable; the normal selector continues. |
| B16 | Both era-end handlers restore all recorded dictionaries and clear only the current-era bonus counter. Event dictionaries are copied to avoid mutating persisted payloads. |

The introductory result comparator also orders rank 0 before lower ranks; the original comparator reversed this display order. Joint-winner and unavailable-breakdown annotations are currently English text, matching this patch's documentation; localization integration remains a presentation enhancement.

The introductory unflipped level-2 bonus interpretation is deliberately unchanged here. The source-only copy-constructor concern is also separate from these event-based repairs. These helpers require a real `BoardGameRulesGame` aggregate, as production uses; they fail explicitly when called with detached domain objects that have no owning game.

Run `tests/BrassPatch/State/run.ps1` after building the staging patch. The harness compiles against, and executes, the patched production assembly on Windows .NET Framework. It covers production repository reconstruction, deterministic replay callbacks, public AI decisions, and results presenters. It does not run Unity animation or send analytics, and does not read or write user save files.
