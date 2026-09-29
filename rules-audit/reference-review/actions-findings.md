# Build, Develop and Network reference comparison

Reviewed 27 September 2026 against npow/brass-birmingham revision `fdeb63c6aa92cc4ba44a721482316e0b8db10e06` and the **currently installed patched Steam binaries**:

```text
Assembly-CSharp.dll 34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A
BrassPatch.dll      4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D
```

**No new current-game P1 was established in this action/resource scope.** B02/B03/B08 are already repaired and are not counted again. The reference is useful as an independent implementation, but a reproduced reference-only rail error demonstrates why a difference must be checked against the publisher's rulebook before being reported against Steam.

## Current-DLL evidence

Run `pwsh -NoProfile -File rules-audit/reference-review/actions-probe.ps1`. The runner checks both installed hashes before loading either assembly, creates fresh in-memory games, and writes only `actions-results.json` beside itself. **57 assertions passed, zero failed**, comprising the same 19 assertions for each of two, three and four players:

| Focus | Concrete outcome | Current source | Pinned reference | Rulebook printed pages |
| --- | --- | --- | --- | --- |
| Build card/network | Cotton Mill I at Worcester accepts an industry card when the player has no board presence, rejects that card after an isolated Stafford brewery is added, and accepts the Worcester location card | `Board.cs:157–175`; `BuildAction.cs:202–258,400–447,555–598` | [gameLogic.js:130–163](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L130-L163) | 8–9, Your Network and Build |
| Two-coal build | Pottery III at Coventry, £22, one coal at Coventry and one at linked Nuneaton, no merchant connection: offered and completed through the original `BuildActionAIHelper`; £0, no resource debt, both mines flipped | `ResourcesFixes.cs:57–65`; `BuildAction.cs:400–447`; `BuildActionAIHelper.cs:15–34` | [gameLogic.js:166–221](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L166-L221), [gameState.js:380–449](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameState.js#L380-L449) | 8, Consuming Coal |
| Rail nearest coal | Birmingham–Dudley rail with Dudley coal and more distant Redditch coal connected to Birmingham: offered; nearest set is Dudley only; far source rejected; nearest source consumes one and clears the debt | `NetworkAction.cs:175–198`; `ResourcesFixes.cs:79–123` | [gameLogic.js:401–442](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L401-L442), [gameLogic.js:631–649](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L631-L649) | 8 and 11, Coal and Network |
| Develop | Two Cotton Mill removals with one disconnected opponent iron cube and one market cube: exact-cash second removal accepted, one-pound-short rejected; ordinary AI helper removes two, flips the iron works and settles exact cash/iron; Pottery I remains prohibited | `DevelopAction.cs:27–62,78–102`; `DevelopActionAIHelper.cs:13–34` | [gameLogic.js:656–728](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L656-L728), [gameState.js:453–478](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameState.js#L453-L478) | 8 and 11, Iron and Develop |

Source names above are rooted in `ai-research/decompiled/BoardGameRules.Entities.Actions`, `ai-research/decompiled/BoardGameRules.Entities.Board`, `ai-research/decompiled/BrassApplication.AI.AIHelpers`, or `src/BrassPatch/Resources` as appropriate. The original decompiled source is interpreted together with the replacement manifest, not mistaken for current installed behavior.

The following additional differences and risky-looking paths were source-traced but did **not** establish new caller-reachable P1 behavior:

- **Overbuilding:** the patched occupied-space exception preserves the vacant single-icon rule for empty targets (`ResourcesFixes.cs:67–77`). Opponent resource overbuilds require a higher tile and empty board plus market (`BuildAction.cs:670–701`, `Board.cs:201–211`). The reference independently checks those conditions at [gameLogic.js:77–114](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L77-L114). Authority: printed pp. 9–10. Existing B08 is not new.
- **Industry-card type on a dual-icon overbuild:** broad raw slot enumeration is subsequently filtered by `IsCardValidForBuildingInSlot` at `BuildAction.cs:555–565`; the normal UI supplies that validated list in `BuildActionUseCase.cs:99–118`. A direct unchecked public Build call is not a normal UI exploit.
- **Iron board priority:** raw `ConsumingIron.SupplyIron` does not itself reject a market argument while board iron exists, but human `IronDeliveryUseCase.StartIronDelivery` at lines 34–58 offers the market only when no board source exists; the original Build and Develop AI helpers choose board iron first and re-query after depletion. No normal-caller bypass was identified. Authority: printed p. 8.
- **Second rail:** `NetworkAction.cs:175–198,298–303` checks the second rail's coal, cash, link supply and connected opponent brewery/any owned brewery; `NetworkActionUseCase` places and pays each rail separately. The independent reference models the same relevant beer restriction at [gameLogic.js:503–542](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L503-L542). Authority: printed p. 11. The previously retained broad rail oracle/matrix already exercises these boundaries; this review did not rerun that entire old matrix or count its assertions as new evidence.
- **Development type helper:** `CanDevelopGivenTypeOfIndustry` checking IronWorks regardless of argument remains suspicious in isolation, but the identified production caller requests IronWorks. Human selection queries the player's actual mat. It remains an API hardening observation, not a new ordinary-turn failure.

## Confirmed reference limitation: single-rail coal source

Run `node rules-audit/reference-review/actions-reference-probe.cjs`. It loads `gameData.js`, `gameState.js`, and `gameLogic.js` from the pinned checkout into a local VM; it does not edit the reference. Evidence is retained in `actions-reference-results.json`.

**Fixture:** Rail Era, own coal mines at Dudley and Redditch with two cubes each, and a pre-existing Birmingham–Redditch rail. Build Birmingham–Dudley. The closest source after placement is Dudley itself. The current Steam code correctly consumes Dudley coal. The reference's ordinary `executeNetwork` reports success but reduces **Redditch from 2 to 1**, leaving **Dudley at 2**.

The reference calls `findCheapestCoalForLink` before placing the rail ([lines 413–430](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L413-L430)). Its selector retains the first endpoint's mine ordering and sorts by free/cost without globally comparing distance ([lines 631–649](https://github.com/npow/brass-birmingham/blob/fdeb63c6aa92cc4ba44a721482316e0b8db10e06/js/gameLogic.js#L631-L649)). The probe's control asks after temporary placement and gets Dudley for this particular graph. This is a **reference-only rule defect**, not another Steam finding, and must not be copied into the local patch. Authority: printed p. 8 requires nearest coal; p. 11 evaluates coal after the new rail is placed.

## Limits

The 57 assertions are targeted current-DLL method tests, not 57 independent whole games. Fixtures use actual installed mat tiles and board/map objects but arrange the positions and era rather than playing setup turns. The tests establish legal predicate/execution behavior for those states; they do not measure frequency in natural games. No real Unity clicks, animation timing, online play or user save files were exercised. The official local rulebook remains the authority: `rules-audit/official-rulebook.pdf`, SHA-256 `BB627A11C769957FBB5F26A210E3957AE27A2BAD054BA9CB3B4FADC6C2BA73E5`, [publisher PDF](https://cdn.shopify.com/s/files/1/0246/2190/8043/files/Brass-Birmingham-Rulebook.pdf?v=1758826281).
