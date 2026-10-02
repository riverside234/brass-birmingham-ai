# Four-player Card Tracker: design and implementation plan

**Date:** 1 October 2026.

**Status:** Implemented, qualified and installed as patch **1.4.12**, then refined and installed as **1.4.13** on 1 October 2026. The latest installed-location verification passes 539 tracker checks and 259 startup/custom-game/resource checks. See the [initial release notes](../docs/CARD_TRACKER_1.4.12.md) and [appearance revision](../docs/CARD_TRACKER_APPEARANCE_1.4.13.md).

**Baseline:** Qualified installed patch 1.4.11. Preserve its repaired AI DLL exactly, along with the existing rules, persistence and statistics fixes. Build and qualify the tracker in dedicated staging so other ongoing AI research work is not included accidentally.

**User decisions:** Enable the tracker only in four-player games; disable it in two- and three-player games. Use public counting, rather than inspecting hidden cards. Place its toggle immediately to the right of the iron market, city indicators near their cities, industry counters to the left of the existing industry sidebar, and Wild Card holders at the top of the screen.

**Implementation revisions requested by the user:** Opening the tracker pauses the game. Enable it only during an idle human turn, after replay and animations finish. Disable it during AI thinking, AI turns, action resolution, scoring and other replay. These decisions replace the original always-live interaction proposal below.

**Appearance follow-up, implemented in 1.4.13:** Make tracker surfaces slightly translucent while keeping text and symbols opaque. Use jade dots with dark rims, shadows and small highlights for unseen cards, and slashed slate-gray rings for played cards. Raise the iron-market button and legend at least 48 physical pixels above their previous minimum, then clear any higher visible hand artwork dynamically. Four-corner bounds account for raised or tilted cards. Reserve this footer space when fitting the map and placing details. Native checks cover hand clearance and opacity at three resolutions in day/night modes. Only tracker appearance/layout sources changed; the accepted 1.4.12 game and AI DLLs remain byte-identical, and earlier rules, pause, eligibility, statistics and persistence implementations are preserved.

## 1. Recommended feature

Add a single **Card Tracker** toggle that displays three coordinated overlays:

1. **City badges:** one dot for each ordinary Location card of that city. Jade filled dots mean not yet seen discarded; slashed slate-gray rings mean publicly discarded this era.
2. **Industry counters:** large, compact counters immediately left of the six existing industry cards. Each shows **Unseen** and **Played**, with Cotton Mill and Manufacturer explicitly sharing one pool.
3. **Wild Card strip:** four player labels across the top, each with a location-pin symbol and a factory symbol showing which Wild Cards that player currently holds.

Counts are computed automatically from public card events when the tracker opens, then stay frozen while the game is paused. Clicking a badge opens its explanation; it does not manually decrement a count. Reopening reconstructs the current counts, including events that occurred while the tracker was closed.

The button is enabled only on an idle human turn in a four-player local game. AI turns, thinking, replay, scoring, mid-action states and finished games cannot open it, including through stale click handlers. Opening records and pauses both Unity and DOTween clocks; closing restores their exact prior speeds. A tracker-owned input shield blocks underlying gameplay while its map pan, zoom, details and card list remain usable. No AI search is interrupted or shortened.

## 2. Rules that determine the counts

The rules cross-check uses the [retained official English rulebook](official-rulebook.pdf), its [searchable text](official-rulebook.txt), and the publisher's [official setup text](https://files.roxley.com/Brass-Birmingham-Rulebook-2018.11.20-highlights.pdf). Page references below are printed page numbers.

| Rule | Consequence for the tracker |
| --- | --- |
| Four players use all Location cards, p. 4. | No player-count card removals. This does not waive player-area setup. |
| Every player draws eight hand cards and one face-down discard, p. 5, steps 8–9. | Four-player Canal setup has **four hidden discards**, one per player. |
| The ordinary card set contains 64 cards; Wild Cards are a separate set of eight, p. 1. | Use a 64-card ordinary manifest and separate Wild Card ownership. |
| Action discards are public; passing also costs a card, pp. 6–7. | Count all public ordinary discards, regardless of the action they paid for. |
| Scout spends three ordinary cards and grants one Wild Location plus one Wild Industry, p. 11. | Count the action card and both trade cards; track two acquisitions. |
| Spent Wild Cards return to their supply, p. 11. | Spending a wild changes its holder, not any ordinary city/industry count. |
| Canal discards are reshuffled for Rail, including the hidden bottom cards; no new bottom discard is dealt, p. 6. | Reset ordinary counters for Rail. The hidden-setup uncertainty applies to Canal only. |
| Build permits alternative card routes, pp. 8–9. | Exhausted ordinary city cards do not make a city safe from construction. |

**Answer to the four-player setup question:** There is no four-player exemption from the face-down setup discard. The fresh installed-game fixture agrees with the rulebook: **32 cards in hands + 28 in the draw deck + 4 face-down discards = 64 ordinary cards**. See [deck evidence](card-tracker-deck-evidence.json).

### Public counting, as selected by the user

For each ordinary card category:

- **Total (T):** its fixed multiplicity in the four-player deck.
- **Played (D):** copies publicly discarded at the currently displayed revision in this era. “Played” includes filler discards and Scout trade cards; it does not mean Build actions only.
- **Unseen (U):** T minus D. These copies have not been publicly accounted for as discards.

Use **Unseen**, rather than an unqualified **Left** or **In deck**, on screen. Unseen cards can be in any player's hand, in the draw deck, or among the four hidden Canal setup discards. The tracker does not identify opponents' ordinary hand cards.

At Canal setup, all 64 ordinary cards are unseen even though only 60 can be played that era. The tracker must not examine the four hidden cards to turn their dots gray. At the completed Canal era, 60 public ordinary discards and four unseen setup cards are expected when history is complete.

Show a small era legend:

- **Canal:** “Unseen includes 4 hidden setup discards.”
- **Rail:** “Unseen cards are in hands or the draw deck.”

In a tooltip, Canal playable copies for a category can be bounded conservatively between **max(0, U − 4)** and **U**. These bounds describe possibilities, not probabilities. Do not subtract four from every category or pretend the setup identities are known.

### Meaning of a depleted city badge

When Birmingham reaches zero unseen ordinary Location cards, all three of its ordinary cards have been publicly accounted for this era. Show a gray badge with a small **crossed-card** symbol and the text **0 unseen**.

Do not put a padlock over the city or label it **Safe**, **Locked** or **Cannot build**. A Wild Location card can still target it. A suitable Industry or Wild Industry card can also allow building through the player's network, subject to the normal rules; the network-free entry exception can matter too. A later Scout can create another wild-card opportunity. Empty ordinary city cards therefore rule out only that particular card route.

Suggested tooltip:

> Birmingham — Location cards  
> Played: 3 / 3 · Unseen: 0  
> No ordinary Birmingham cards remain unplayed this era. Wild Location cards and legal industry-card builds can still reach this city.

## 3. Verified four-player card manifest

The [fresh fixture evidence](card-tracker-deck-evidence.json) records the installed assembly/helper fingerprints, 64 distinct ordinary card IDs, per-category totals and setup sizes. It uses a newly constructed in-memory game and does not inspect a user's save or current match.

The previous [component audit](reference-review/components-findings.md) independently compared the complete card multiset with the reference implementation. Its probe includes the initial discard piles: the reported 64 is the complete ordinary deck, not a 64-card playable remainder.

### Ordinary Location cards — 41 total

| City | Cards | City | Cards |
| --- | ---: | --- | ---: |
| Belper | 2 | Birmingham | 3 |
| Burton-on-Trent | 2 | Cannock | 2 |
| Coalbrookdale | 3 | Coventry | 3 |
| Derby | 3 | Dudley | 2 |
| Kidderminster | 2 | Leek | 2 |
| Nuneaton | 1 | Redditch | 1 |
| Stafford | 2 | Stoke-on-Trent | 3 |
| Stone | 2 | Tamworth | 1 |
| Uttoxeter | 2 | Walsall | 1 |
| Wolverhampton | 2 | Worcester | 2 |

The current city categories contain **one to three cards**, so dots are especially suitable. Do not add a fourth dot just because the game has four players.

The two unnamed farm breweries and the merchant towns have no ordinary Location cards in this manifest. Omit city-card badges there. Their absence is not a zero-card warning.

### Ordinary Industry cards — 23 total

| Physical card family | Cards | Sidebar mapping |
| --- | ---: | --- |
| Brewery | 5 | Brewery |
| Coal Mine | 3 | Coal Mine |
| Iron Works | 4 | Iron Works |
| Cotton Mill / Manufacturer | 8 | Cotton Mill **and** Manufacturer, one shared pool |
| Pottery | 3 | Pottery |

There are five physical industry-card families and six sidebar industry types. The eight dual-purpose cards must not become eight Cotton cards plus eight Manufacturer cards in totals.

### Wild Cards — tracked separately

| Type | Supply size | Symbol |
| --- | ---: | --- |
| Wild Location | 4 | Location pin |
| Wild Industry | 4 | Factory silhouette |

The manifest is a versioned, static public rule datum. Obtain it from verified setup/component data, not the shuffled deck of an ongoing match. During implementation, test both full and introductory four-player setup before enabling both modes.

## 4. Layout and appearance

### Toggle beside the iron market

- Label: **Card Tracker**, with a small card-and-dots icon and a clear On/Off state.
- Place it to the right of the actual iron-market control in the bottom-left area, using its measured bounds and an 8–12 pixel gap.
- Start around 160 × 44 physical display pixels with at least a 40-pixel click target. Reflow beside/above the market at small widths rather than covering market cubes, prices, resource prompts or the player's hand.
- Anchor to the safe-area canvas. Do not hard-code one screen coordinate or reparent the iron market.
- Default Off for a new match. Keep the larger-text preference for that view session; reopening captures fresh public counts. The paused/open state is not saved.
- Hide the button and all tracker overlays for two- and three-player games. Gate on the actual configured player count, including human and AI seats.
- Hide during main-menu/loading/tutorial screens and after gameplay transitions to finished-game review. The existing Statistics button then retains its current location and behavior.

### City dots

Place a small rounded capsule near each city banner, outside building slots and the usual green/white build highlights. Birmingham's sequence is:

| Publicly discarded | Display | Detail |
| ---: | --- | --- |
| 0 | Three filled teal-green dots | Played 0 · Unseen 3 |
| 1 | Two filled dots + one gray dot | Played 1 · Unseen 2 |
| 2 | One filled dot + two gray dots | Played 2 · Unseen 1 |
| 3 | Three gray dots + crossed-card symbol | Played 3 · Unseen 0 |

Use approximately 12–14 pixel dots with a visible border and 4–5 pixel spacing at normal display size. Keep them readable across board zoom levels. The capsule should have a dark, mostly opaque background rather than the faint transparency that previously made city highlights hard to see.

Starting palette for visual qualification:

| Element | Suggested color |
| --- | --- |
| Unseen dot | Teal-green #69C8B4 |
| Publicly discarded dot | Slate gray #858A90, with a distinct hollow/marked treatment |
| Text | Warm ivory #F2E9D8 |
| Capsule | Dark brown #292622 at roughly 92% opacity |
| Border/active control | Muted gold #D4B478 |

Color does not imply build legality. Shape, count and the legend distinguish the tracker from existing build highlights. A zero-card badge stays readable rather than fading out.

Dots are the first-release default. Segmented pies could be a later appearance preference; their wedges must represent individual copies and retain the same text/tooltip definitions.

Keep badges attached to their city's board position but compensate for zoom so they do not shrink into unreadable pixels. Use measured banner bounds plus city-specific offsets for dense areas such as Birmingham, Walsall and the Black Country. At low zoom, use short leader lines or compact count capsules when dots would collide. Hover/click always identifies the city; never merge neighboring cities into an ambiguous count.

### Industry sidebar

Place a compact two-line badge to the **left of each existing industry card**:

    Unseen 3
    Played 1

Use 20–21 pixel count text by default and 18-pixel secondary labels, accounting for canvas scale as the statistics readability patch does. Counts are global card counts, not remaining industry tiles on the displayed player's mat.

Show matching Cotton and Manufacturer counts with a small connecting bracket/shared-card symbol. Both tooltips say **Shared Cotton / Manufacturer card pool**. Discarding one dual card updates both displays once, while the total ordinary-card summary decreases by only one.

Attach to each row's actual industry type, not its child index alone. Keep the badges aligned if the sidebar updates, temporarily hides, changes player or opens the development board. Do not cover cost, resource, VP or income tooltips.

### Wild Card strip at the top

Show four compact player chips in a stable seat order, for example:

    Alice [pin filled] [factory outline]   Bot 2 [pin outline] [factory filled]

- Use player name plus a small player-color accent. Identify players by stable seat/color internally, so repeated names and changing turn order do not mix their cards.
- A filled gold icon means **Held**. A clearly dim outlined icon means **Not held** when history is complete. An icon with a question mark means **Unknown** if ownership cannot be reconstructed.
- Location pin = Wild Location; factory = Wild Industry. Use existing game icon assets or simple vector/code-drawn symbols with consistent outlines, not platform-dependent emoji.
- Hover/click identifies the player and type: “Alice currently holds a Wild Location card, acquired through Scout.”
- Public Scout history identifies the acquired types. Do not enumerate an opponent's current private hand to fill the strip.
- Reserve top space for the tracker header and four stable player chips, and leave the right sidebar clear. Keep names at a readable physical font size; long names may ellipsize, with the full name in the symbol's detail. Fit the map below the header and above the footer when opening, then restore the exact previous map pose on closing.
- The strip shows current ownership. A player who used a wild must not remain marked simply because they previously took Scout.

### Overall interaction

Turning On shows all three sections together. A small legend beside the toggle opens definitions and an optional scrollable **card list** containing all 25 ordinary categories. The list provides readable numeric access when the map is crowded; it uses the same projection as the overlays.

Show era and global summary in that legend:

    Canal · Played 12 · Unseen 52
    Unseen includes 4 hidden setup discards.

Hover tooltips are passive. Clicking a badge pins its detail popover; clicking its close button or pressing Escape dismisses that popover first. Escape then closes the card list, if open, and finally closes the tracker and resumes play. Consume all gameplay keys while paused, including the closing frame, so the same Escape/Return press cannot also advance gameplay or open the main menu.

The paused overlay owns pointer input. Its background accepts map drag and wheel zoom; badges and detail/list controls accept their own input. Underlying building slots, market resources, action buttons and replay controls cannot receive tracker clicks. Existing panels retain their normal behavior after the tracker closes. Opening is unavailable throughout animations, so closing it never requires cancelling a replay.

## 5. Counting and event handling

Implement a small read-only **CardTrackerProjection** with:

- Game identity, configured player count, era, displayed revision and history-completeness status.
- The immutable four-player manifest.
- Per-category public discard counts.
- Per-player current Wild Location/Wild Industry ownership and acquisition revision.
- Seen event revisions and per-era physical-card accounting for duplicate/conflict detection.

Normalize Location cards by RegionName and Industry cards by their complete sorted BuildingTypes set. Wild subclasses have separate keys. Display names are localized labels, not dictionary identities.

| Public event/state | Tracker effect |
| --- | --- |
| GameStarted | Initialize manifest and players from public configuration. No setup card identities or seed are read. |
| CardDiscarded: ordinary card | Increment the matching ordinary category once when this discard has been publicly presented. |
| CardDiscarded: Wild Location/Industry | Clear that player's matching Held icon; no ordinary count changes. |
| WildCardsTaken | Count the two ordinary CardsToTrade and mark both wild types Held for that player. The action-payment CardDiscarded remains a separate, single count. |
| RailEraStarted | Begin a new ordinary-card epoch with zero public discards; discard prior Canal accounting. |
| Presented revision moves backward | Rebuild from the retained public prefix at that revision; remove effects of undone/cancelled events. |
| Presented revision jumps forward | Fold the intervening public prefix once, including skipped presentation batches. |
| GameEnded / exit to menu | Remove gameplay overlays and release their session state. |

Verify the actual era-start event ordering before binding reset to a method. The visual reset occurs when Rail setup becomes the presented state, not when the AI or engine first computes a future Rail state.

All ordinary card uses count: Build, Network, Develop, Sell, Loan, Scout and Pass. A multi-industry Sell or double-rail Network still consumes only its actual action card. Merchant bonus actions do not fabricate another discard. Card preview/selection alone never changes the count.

Deduplicate event delivery by game identity and event revision. A repeated animation, UI refresh or save/reload notification is not another discard. Physical identities provide an additional per-era consistency check; unexpected duplicates/conflicting categories should produce an unavailable/unknown state, not silent clamping.

Wild ownership follows **acquire → hold → spend → return**, including repeated Scout episodes across the match. Do not reset it from a lifetime Scout counter or assume both wild types are always spent together.

### Public information boundary

Allow only public discard payloads, public Scout trades/acquisitions, player identity, player count, era and the displayed event cursor. The ordinary deck's contents, opponent hands, hidden setup discards, shuffled seed and pending AI choices are excluded.

Do not invoke the complete-game StatisticsCollector for live tracking. It reconstructs a detached full game, applies events and can inspect card objects that are inappropriate for this public overlay. Likewise, the AI host's private ExportCards is not a tracker API.

The existing AI host's Observation demonstrates the relevant boundary: it skips one initial Canal discard and uses a Rail discard-start offset. Reuse that reasoning, not its private full-state serialization.

### Era and resume edge cases

The engine can retain old Canal discard references during the Rail deal. Scanning every entry of every DiscardPile would count old cards again. Prefer event epochs; any fallback scan must use verified current-era start offsets and exclude the initial Canal hidden card.

Rebuild on continue/resume from persisted and uncommitted public events, capped at the **currently presented** revision. Merge duplicate revisions consistently and reject conflicting records. Toggle visibility must not affect accounting, so turning the tracker on halfway through a game produces the same result as having it visible from the start.

If the required history is missing, show **? / History incomplete** for affected categories and wild holders. Never substitute zero, silently use private hands or present a partial count as complete. Reopening complete history should restore accurate counts.

No save-format change is necessary: counts are a projection of existing events. An optional persisted On/Off appearance preference is separate from authoritative card state.

## 6. Integration with the current patch

Implemented files:

| File/group | Responsibility |
| --- | --- |
| src/BrassPatch/CardTracker/CardTrackerProjection.cs | Verified manifest, public history adapter, era reset, rollback and wild ownership |
| src/BrassPatch/CardTracker/CardTrackerEntryPoints.cs | Match/router ownership, human-turn and four-player gates, pause token, toggle and teardown |
| src/BrassPatch/CardTracker/CardTrackerOverlay.cs | City anchors, sidebar badges, top chips, legend and details |
| src/BrassPatch/CardTracker/CardTrackerWidgets.cs | Dots, crossed-card and wild icons; readable palette and hit targets |
| src/BrassPatch/CardTracker/manifest.json | Narrow verified presentation/lifecycle hooks |
| tests/BrassPatch/CardTracker/ | Projection fixtures and native Unity integration checks |
| docs/CARD_TRACKER_1.4.12.md | Definitions, usage, qualification and rollback notes |

Useful verified code anchors:

- MarketView finds the iron-market button at **SafeArea/Popup/Markets/IronMarket**. Measure that RectTransform for toggle placement; the tracker must not alter market selection listeners.
- IndustriesSummaryView binds the permanent six-row sidebar at **SafeArea/Popup/RightPanelBackground/IndustriesContainer**. This is the primary industry-badge anchor. IndustriesDetailsView is the separate development/details panel and must not be mistaken for the permanent sidebar.
- BoardView.GetRegionViewForGivenRegionName exposes the region views already used by the city-locator patch. Place badges in a dedicated overlay layer, rather than inside building-slot objects that can be cloned into region details.
- ReplayUseCase.EventReplayCompleted advances history to the presented event revision. ReplayPresenter.UpdateGamePresentation refreshes the board, markets, HUD, industries and hand.
- Existing CityLocator, PresentationLifecycle, LinkReplay and StatisticsEntryPoints own their respective highlights, replay recovery, information-panel input and finished-game button.

During implementation:

1. Audit current method signatures and hook manifests against the verified assembly before weaving.
2. Bind a read-only session to gameplay start/resume/exit and the current router. Keep one owner per match/view lifecycle.
3. Observe the presented revision after public event presentation and ordinary presentation refreshes. Never read a latest engine revision that is ahead of AI animation.
4. Use narrow postfix/observer hooks where available. The weaver rejects duplicate target methods: compose with an existing helper when a target is already patched, or select another appropriate refresh point. Do not overwrite existing manifests.
5. Marshal visual changes to the Unity main thread. Cache the small projection and refresh only changed badges; avoid full-history replay or object allocation each frame.
6. Destroy only tracker-owned UI on exit, load, board replacement and match change. Do not leave static listeners, orphan badges or overlays from another match.
7. Catch tracker failures locally, mark the tracker unavailable and keep gameplay running. No tracker method may own or call an event-replay completion callback.

The new toggle must not call Play now, Skip to my turn, CancelCurrentReplay, KillAllAnimations, ConfirmNextAction or any gameplay command. AI search settings and the disabled Play now control remain unaffected. The human-only eligibility check means pausing never races with an AI worker or commit. Restore the pause on close, view disable/destroy, match replacement or unexpected eligibility/revision change.

## 7. Verification and acceptance

### Rules and projection checks

| Scenario | Required result |
| --- | --- |
| Fresh full four-player game | 64 ordinary categories accounted for; 41 location + 23 industry; all 64 unseen; eight-card hands, one hidden discard each, 28-card draw deck |
| Four-player introductory game | Same public setup accounting if confirmed by the actual mode; no artificial Rail reset after its Canal-only ending |
| Two-/three-player game | No tracker button, overlays, public-count session or listeners |
| Ordinary action discard | Exactly one correct decrement in Unseen / increment in Played |
| Discarded city card used as filler | City count changes even though the action was not Build |
| Scout | Three ordinary cards accounted for, two separate wild types acquired |
| Shared Cotton/Manufacturer discard | Both sidebar views change; physical ordinary total changes once |
| Wild use as Build or filler | Matching holder icon clears; ordinary counts do not change |
| Repeated Scout / different wild spending order | Current ownership stays correct through every acquire/spend episode |
| Multi-Sell, double-rail Network, merchant bonus | Count only actual card payloads |
| Duplicate event/replay notification | No duplicate decrement |
| Preview, partial action, cancellation and undo | Match the displayed public revision; no count from selection alone or abandoned future events |
| Toggle Off for several turns, then On | Same state as continuous tracking |
| Canal ending | 60 public ordinary discards + 4 unseen setup cards, with complete history |
| Rail start | 64 unseen ordinary cards, zero Played, no hidden-setup warning and no stale Canal counts |
| Continue/resume in either era | Same projection as uninterrupted play at that presented revision |
| Gapped/conflicting history | Explicit unknown/unavailable indicators; no fake zeros |
| Depleted city | Crossed-card warning only; tooltip explicitly permits alternative legal build routes |

Add a public-information independence test: fixtures with the same public history but different opponent hands, draw order and hidden setup card identities must produce identical tracker output. Use the complete internal fixture only as an independent test oracle; it is not a runtime source.

Projection invariants with complete public history:

- For every ordinary category, 0 ≤ Played ≤ Total and Unseen = Total − Played.
- Sum of category totals is 64; the dual-purpose pool contributes eight once.
- At any completed Canal point, sum of Unseen equals ordinary cards in hands + draw deck + four hidden setup cards.
- At completed Rail points, sum of Unseen equals ordinary cards in hands + draw deck, using current-era objects/offsets.
- Every wild acquisition and spend preserves public holder/supply conservation; history gaps explicitly weaken certainty.

### Native UI and regression checks

- Capture and inspect the actual Unity UI at **1280 × 720, 1920 × 1080 and 2560 × 1440**, both board appearances and several zoom/pan positions.
- Verify city badges do not cover names, slots, resource cubes, links or the existing rounded green/white highlights. Check dense neighboring cities and both farm breweries.
- Verify industry counters remain outside existing tooltip and card hit targets, including exhausted mat tiles and development-panel transitions.
- Check long/duplicate player names, stable seat ordering, both wild icons, incomplete-history icons and narrow-screen wrapping.
- Check readable color/shape distinctions and physical text size, including a larger-text setting. Zero counts remain visible.
- During AI thinking, turns and animations, verify the button is disabled and stale/direct opening requests are rejected. Verify replay/scoring and mid-action rejection independently. The AI DLL must remain identical to the accepted baseline.
- Opening must freeze both Unity and DOTween clocks. Inspection must not mutate events or advance the game revision. Closing, disabling or replacing the view must restore the exact prior speeds and map pose; Escape, Return and pointer input must not leak into underlying actions.
- During link placement, resource selection, round/era scoring and game ending, information panels remain closable and the tracker causes no replay deadlock.
- Test resize, exit-to-menu, rejoin, restart and consecutive matches with different player counts. Confirm one overlay owner and no UI stacking or stale cards.
- Run affected native presentation, link-replay, statistics and startup/menu suites on the final candidate, plus meaningful public-card/Scout and lifecycle checks. Recheck installer/package allowlists, preserve accepted baseline AI qualification with exact binary/source evidence, and verify the installed tracker.

## 8. Implementation sequence

1. **Public projection:** Pin the manifest, define counts, implement Scout/shared-pool/wild handling, and qualify hidden-information independence plus era boundaries.
2. **Board overlay:** Add four-player gating, the iron-market toggle, city dots, industry counters and the Wild Card strip. Qualify readability and input behavior in the real Unity prefabs.
3. **Lifecycle completion:** Connect presented revisions, resume, rollback, skipped presentation, incomplete history and teardown. Verify against current replay and AI input guards.
4. **Release qualification:** Run meaningful projection/native regressions, review screenshots, document controls and counting limitations, then produce a versioned patch candidate with a recoverable backup.
5. **Install and verify:** Use the verified atomic installer with the dedicated candidate, preserve the latest accepted fixes and online-disabled setup, then verify the installed files and tracker entry points. Keep a recoverable backup and a complete versioned release archive.

The feature is complete when an idle human turn in a four-player local match can open an accurate paused public snapshot, all three overlays are readable, closing restores play without state changes, and AI/action/replay states plus two-/three-player matches cannot open the tracker.

## 9. Sources and retained evidence

- [Roxley Brass: Birmingham product and rulebook listing](https://roxley.com/products/brass-birmingham), checked 1 October 2026.
- [Publisher-hosted English rulebook used for the setup cross-check](https://files.roxley.com/Brass-Birmingham-Rulebook-2018.11.20-highlights.pdf): printed pp. 1, 4–9 and 11.
- [Local official rulebook](official-rulebook.pdf) and [extracted text](official-rulebook.txt).
- [Fresh installed-game deck evidence](card-tracker-deck-evidence.json): ordinary multiplicities, physical identity count, hidden-discard setup and assembly fingerprints.
- [Earlier component/setup comparison](reference-review/components-findings.md) and [probe source](reference-review/components-probe.ps1): complete ordinary deck, including initial discard piles.
- [Existing public observation boundary](../src/BrassLocalAI/Host/Observation.cs): hidden Canal discard exclusion, known wild types and Rail offsets.
- [Existing statistics projection](../src/BrassPatch/Statistics/StatisticsProjection.cs): CardDiscarded versus Scout trade-card event structure; not a live tracker data source.
- [Existing city locator](../src/BrassPatch/CityLocator/CityLocatorFixes.cs), [link replay recovery](../src/BrassPatch/Presentation/LinkReplay.cs) and [statistics entry points](../src/BrassPatch/Statistics/StatisticsEntryPoints.cs): UI anchors and lifecycles to preserve.
- [Installed statistics readability release](../docs/STATISTICS_READABILITY_1.4.10.md): current typography, presentation and regression baseline.

The layout, palette and interaction recommendations are design choices for this digital version. The rulebook establishes card accounting and permitted actions; it does not specify a Card Tracker interface.
