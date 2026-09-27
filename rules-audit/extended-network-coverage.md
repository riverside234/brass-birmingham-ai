# Extended Network, Sell, and Beer audit

Date: 2026-09-27. This is an independent second pass after the first audit's B01–B11 findings. **No additional distinct P1 defect was established in this assigned scope.** This is positive tested coverage, not a claim that the game is defect-free.

## New evidence

- [C# matrix and human-flow harness](extended-network-matrix.cs)
- [PowerShell runner](extended-network-matrix.ps1)
- [Original-DLL results](extended-network-matrix-results.json)

The suite passes **1,282 assertions**, with an empty failure list, against SHA-256 `600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB`. All execution is against the installed original assembly, using isolated in-memory games. No save, installed binary, or previous audit artifact was changed.

Run with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File rules-audit/extended-network-matrix.ps1
```

## What the second pass adds

| Coverage | Count | Independent checks |
|---|---:|---|
| Network candidate fixtures | 670 | 600 first-link states and 70 second-link states; original eligible set versus separately implemented graph, ownership, cost, beer and inventory oracle |
| Executed rail placements | 105 | First and second rail payment, mandatory coal, single barrel for double rail, ownership and tile decrement |
| Beer connectivity combinations | 324 | Own versus opponent brewery; all 27 destinations; six brewery locations including both farms; original query versus independent graph traversal |
| Original human Sell flow | 30 | 2/3/4 players, both eras, zero/one/two beer industries, brewery-only or brewery-plus-merchant supply, resource auto-selection, action completion |
| Original human Network cancellation/undo | 21 | Seven action stages across 2/3/4 players; production history/repository restores state; a subsequent single-rail action verifies counter reset |
| Multiple industries sold in one human action | 12 | Both eras and all player counts; zero/one/two beer industries; two merchants used across separate industries; one card/action total; optional early stop |
| Original human Sell cancellation/undo | 18 | Six stages across all player counts, including partial beer, a flipped brewery and industry, zero-beer continuation, and completed-action undo |

Counts overlap assertions: one fixture can contain several assertions. These are not 1,282 naturally played games.

### Network candidate and payment matrix

The independent reference traversal uses `Link.HasOwner`, raw endpoints, and its own queue; it does **not** call `Board.GetConnectedRegions`, `DoesRouteExists`, or `GetShortestRouteTo`. It combines endpoints only for evaluating a prospective link, computes the player's network directly from owned industries and links, and computes one-cube market prices separately from the production market implementation.

The first-link matrix includes 2/3/4 players, both eras, money values around £3/£5/£10 and market costs, zero/one/many remaining link tiles, coal-market stock 0/1/2/3/12/13/14, blank networks, opponent-only links, disconnected own industries, connected coal mines and the southwest farm link. The second-link matrix starts every market-touching rail on an empty board, pays its coal, and reevaluates all possible second rails with one versus two initial link tiles and disconnected own beer. Stock 0/1/2/13/14 covers exhausted-market £8 purchases and a price transition between the two rails.

All sets and charges matched the reference. In particular, the UI query does not offer an unaffordable second rail in these cases; the engine consumes coal separately after both placements; a single remaining link tile prevents a second placement; own disconnected beer remains legal.

Relevant original call chain:

- `BoardGameRules.Entities.Actions/NetworkAction.cs:175` — eligible connection query; `:198` — base costs; `:211` — connection affordability.
- `BrassApplication.UseCases.Game.Actions.NetworkAction/NetworkActionUseCase.cs:209` — second-link selection refreshes candidates after first-link payment.
- `BrassApplication.AI.AIHelpers/BoardHelper.cs:507` — original AI's second-link helper also refreshes through the rule query.
- `BrassApplication.AI.AIHelpers/NetworkActionAIHelper.cs:15` — places and pays each rail before recursively taking a second eligible rail.
- `BoardGameRules.Entities.Markets/BaseMarket.cs:39` — actual market buying price, independently cross-checked at boundaries.

The existing farther-mine issue B03 is unaffected: this new candidate matrix validates the existence and affordability of coal supply, not the correctness of every source offered to a multi-endpoint link. B03 remains an open defect from the first pass.

### Human Sell and multiple-sale sequencing

The harness calls the production `SellActionUseCase` and `BeerDeliveryUseCase`; it stubs presentation and queues replay-completion callbacks, reproducing the order in which the UI receives normal completed animations. Single-source beer automatically goes through the original `BeerDeliveryUseCase.StartBeerDelivery` logic. Choice cases invoke the same `SelectBeerSource` method that the controller/delegate invokes after a click.

The single-industry cases verify one discard, one action deduction, industry flip and completion for beer costs 0/1/2. The multiple-sale cases sell level III, II and V manufactured goods using two separate merchant barrels and one brewery barrel, in the same action. The brewery flips when exhausted. The optional early-stop cases finish after the zero-beer sale and preserve unsold industries and beer. These cases passed.

Relevant original methods:

- `BrassApplication.UseCases.Game.Actions.SellAction/SellActionUseCase.cs:61` — industry choice; `:70` — brewery/merchant candidates; `:103` — replay-driven state machine; `:166` — subsequent-sale choice; `:181` — finish.
- `BrassApplication.UseCases.Game.Actions.BeerDelivery/BeerDeliveryUseCase.cs:30` — automatic singleton source consumes up to the required amount.
- `BoardGameRules.Entities.Actions/SellAction.cs:31` — zero-beer flip and required-beer setup; `:147` — industry flip/income.

Existing B04 remains open: unlike these valid multi-industry cases, a **single** two-beer industry cannot use two merchants. The old original-AI reproduction already demonstrates that violation; this suite does not relabel it as a new issue.

### Cancellation and undo

These cases additionally exercise production `BrassGameHistory`, `BaseRepository`, `BrassBirminghamFactory`, `InMemoryGameStorage`, and `InMemorySnapshotStorage`. Baseline fixtures are assembled with replayable original events, saved to the in-memory repository, and then acted upon via the original human usecase. Cancellation invokes the original usecase's `CancelAction`; completed actions invoke the original history's `UndoAction`.

The comparison covers era/round/action count, every player's money/link tiles/income/hand/discard count/mat tile IDs, both market stocks, every link's owner, and all board industry IDs/flip/resource states.

Network stages: after discard; first rail placed but unpaid; first coal paid; second-link choice entered; second rail placed but unpaid; second coal paid but beer not yet selected; completed double action undone. Every state restored, and the reused usecase subsequently completed a correctly priced single rail without a stale beer requirement.

Sell stages: after discard; level V industry selected; first of two brewery beers consumed; both beers consumed and brewery/industry flipped; a second zero-beer sale completed; three-industry action completed and undone. Every baseline fingerprint restored, with no current action left behind.

Relevant methods: `BaseActionUseCase.CancelAction` (`BrassApplication.UseCases.Game.Actions.BaseAction/BaseActionUseCase.cs:33`), `NetworkActionUseCase.CancelAction` (`:191`), and `BrassGameHistory.UndoAndCancelImpl` (`BrassApplication.GameHistory/BrassGameHistory.cs:153`). No claim is made about a real rendered button, animation cancellation, or replay skipping: that needs production `ReplayUseCase`/Unity testing and is a separate audit area.

## Rules checked

The local official publisher rulebook, [Roxley's PDF](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281), provides the oracle: printed p8 for resources, p9 for farm connections, p10 for Sell, and p11 for Network. Checks cover action payment, network membership, post-placement coal, two-rail beer, and separate industry sales. The rule requiring an opponent's beer to connect to the **second** rail remains intentional; accepting only first-rail connectivity would introduce a defect.

## Limits and remaining risk

The generated fixtures test boundaries and normal usecase call paths, not the reachability frequency of each state in natural games. Some fixtures arrange events directly instead of playing the setup turns. Source and resource oracle independence helps detect agreement failures but cannot prove all rules implemented correctly. Rendered interactions, stale raycast targets, asynchronous Unity animation timing, online concurrency, and production `ReplayUseCase` are not executed here. Existing B03, B04 and B10 are neither fixed nor disproved by these passing cases.
