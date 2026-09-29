# R1 four-player rules host

27 September 2026. R1 implements the shared environment from main-plan stages 0–3. It runs standard four-player games against the preserved rules runtime **1.1.0**, using runtime contract **0.2.0**. The [exit record](R1_EXIT_CHECKS.md) distinguishes implementation evidence from future work. The game now has a separate 1.3.0 deployment; see [compatibility and rebuild boundaries](UI_PATCH_RESEARCH_COMPATIBILITY.md).

The source is an original adapter around the user's game assemblies. It contains no copied game implementation, assets, Daiyu code or OpenSpiel code. R0's reference pins and contract `0.1.0` remain unchanged. Its model/search prototypes are separate from this runtime schema; R2/R3 must explicitly target `0.2.0`.

## Rules and runtime

The original [rules profile](../configs/rules/steam-brass-patch-1.1.0.json) remains immutable because R0 pins its bytes. [Dependency pins](../configs/rules/runtime-dependencies-1.1.0.json) extend it to all 111 installed managed DLLs. Building checks both sets before copying assemblies into ignored `artifacts/r1-host/guarded`. The Python client checks those pins and the built executable's recorded hash before launch.

The existing patch supplies the audited rules repairs: multi-coal affordability, nearest coal across link endpoints, one merchant's beer per sale, overbuilding, loan threshold, final-Rail settlement/ties, stable tile identity, game-owned bonuses, Scout and liquidation guards. R1 leaves that installed patch unchanged. New adapter preconditions cover gaps in low-level APIs: owned cards, action phase, source availability, board iron before market iron, developable tiles of the requested type, completed payments, and exactly one action confirmation. The vendor's develop-type helper tests the wrong industry; this adapter inspects the actual next tile instead.

Low-level engine methods are not a legality oracle. New policies use this provider and never execute arbitrary vendor action methods. Host rules and the installed game's guarded rules were tested in both .NET Framework and the game's actual Unity Mono runtime. This is finite regression evidence, not a proof of every possible Brass position.

## Build and run

Validated platform: Windows, Python 3, PowerShell 7 and .NET Framework; no .NET SDK, GPU, Torch or OpenSpiel installation is needed. Use your own patch-1.1.0 game installation. The first build can acquire the SHA-256-pinned Roslyn compiler from NuGet; subsequent builds work offline.

```powershell
python -m pip install -r requirements-r1.txt
pwsh -NoProfile -File scripts/build-host.ps1 -GameRoot 'D:\SteamLibrary\steamapps\common\Brass Birmingham'
python scripts/check-r1-traces.py
python -m unittest discover -s tests/R1 -p 'test_*.py' -v
pwsh -NoProfile -File scripts/check-r1-runtime.ps1
```

When the repository itself is the game directory, omit `-GameRoot`. The dependencies and generated records stay under `artifacts/`, which Git excludes. A clean clone alone cannot run the engine: provide the locally owned, matching game DLLs. The client accepts a `mono=` executable on non-Windows systems as an experimental launch path; Linux and the lab computer have **not** been validated. [R2's training scaffold](R2_TRAINING.md) now consumes this interface and can train transferred compatible replay shards without the engine.

Trusted orchestration example, with `src` on `PYTHONPATH`:

```python
from brass_ai.host import RulesHost
from brass_ai.runtime_contracts import policy_inputs

with RulesHost() as environment:
    context = environment.reset(seed=42)
    view, choices = policy_inputs(environment)
    # A policy receives only view and choices. This example deliberately passes.
    choice = next(c for c in choices if c['kind'] == 'pass')
    context = environment.apply(context, choice)
    finish = next(c for c in environment.choices() if c['kind'] == 'finish')
    context = environment.apply(context, finish)
```

## Interface and action coverage

The worker reads one JSON request per line and writes one correlated response. Every request declares `protocol: "0.2.0"`; mutations require the current `game_id` and `revision`. Reset and explicit restore renew the transport identity, rejecting outstanding pre-load replies. Choice IDs contain semantic locations, industry types, permitted card types and resource sources; they contain no vendor GUIDs or engine seed.

| Operation | Behavior |
| --- | --- |
| `reset(seed, setup?)` | Standard four-player game, or a trusted explicit setup/public-position reconstruction |
| `context` | Actor, ordinary turn owner, phase, remaining actions, era/round/turn, revision and transport identity |
| `observe(seat)` | Detached, allowlisted observation for one seat |
| `choices` | All next semantic choices for the actual actor, in stable sorted order |
| `apply(revision, game_id, choice)` | Validate membership, execute one decision, roll back an unexpected partial failure |
| `apply_action(revision, game_id, choices)` | Reconstruct and validate one complete ordinary action; adopt only after full success |
| `advance` | Idempotent automatic progression to a decision or game end; also called after finishing actions/debts |
| `checkpoint`, `restore(checkpoint)` | Trusted reconstruction record and replay; never policy input |
| `canonical_hash` | Trusted diagnostic digest including hidden cards, RNG progress and engine state |
| `result` | Four stable seats, VP/income/cash, shared competition ranks and win-share utilities |

Phases are `action`, `payment`, `continuation`, `bonus`, `debt`, `terminal`. Action selection includes a card and initial target. Payments consume one cube/barrel per choice so all source allocations and meaningful resource orders can be represented. A completed sub-action exposes `finish`, plus any legal additional development, sale or second rail. Bonus and debt have their own actor/choice handling. Deck draws, round ordering, income and era transitions remain engine operations.

Coverage includes all seven actions, locations/industries and overbuilding, exact coal/iron/beer sources, equal-distance coal, both coal/iron payment orders, merchant selection, ordered multiple sales, optional second development/rail, free development/skip, Scout's three distinct physical cards, wild cards and owned liquidation assets. There is no strategic shortlist. Physically duplicate cards with the same legal meaning are collapsed into one semantic choice; Scout enumerates every distinct card multiset. Developing one tile and paying its iron before offering tile two preserves the same complete legal outcomes as choosing both tiles first. A single-cube payment representation similarly preserves source-allocation outcomes.

An action plan ends with one `finish`; it cannot omit unpaid resources, confirm twice, or contain another ordinary action. A bonus is resolved before that finish. Current-turn planning, pruning and neural ranking belong to R3/R2, not the legal provider.

## Restoration and isolation

Checkpoints record an explicit initial card/merchant/order setup, versioned choice commands and card outcomes after each command. Replay reselects each legal semantic choice, reconstructing private in-progress action fields, merchant bonuses, debt and automatic transitions through the same execution path. Explicit card outcomes cover chance boundaries instead of assuming equal numeric seeds imply equal shuffles on every runtime. A mismatch outside a chance boundary is rejected.

The canonical digest includes public state/history, all hands/discards/deck, RNG initialization/progress, engine revision, pending round/debt flags and removed industries. Exported JSON is detached; vendor event objects are not retained as shared snapshots. Neither empty copy constructors nor action cancellation is used for rollback. Unexpected exceptions after mutation restore the pre-command record; an injected post-loan-mutation failure tests that path.

Use one active game per worker process. Keep several persistent `RulesHost` instances for isolated branches; do not share a worker concurrently between threads. Only trusted reconstruction/transaction code creates temporary host instances within a worker. The host never enters stock `BoardHelper`; the installed patch owns merchant bonus state per game. The stock-AI runner is separate and sequential. These boundaries avoid mixing stock static context with search branches or a live Unity game.

Checkpoint copying and replay currently favor correctness over throughput. Replaying an entire history for every action validation is unsuitable as the final high-volume search implementation. Profile and optimize it before scaling R2/R3; the API and correctness checks are the foundation for that work.

## Information boundary and sampled worlds

`Observe` explicitly constructs primitives; it does not serialize an engine object, event or card instance. It includes the observing hand, public board/mats/resources/spending/order, hand counts, current-era face-up discards and executed public decisions. Setup discard identities remain hidden even from their owners. Wild-card knowledge is derived from public Scout/discard history. Seed, deck order, other private hands, internal identities and transport identity are absent from policy features.

The vendor retains Canal discard references after the Rail shuffle. R1 tracks that archive separately so it is neither exposed as current Rail discards nor subtracted twice by the belief sampler. `sample_world_setup(view, composition, search_seed)` subtracts permitted known cards from the public 64-card composition, samples other hands, all four unseen Canal setup discards and the deck, and enforces wild-card conservation. It never accepts a real hidden pool/checkpoint. Sampling is broad and uniform subject to hard constraints; behavioral inference is future work.

The host reconstructs these worlds from public positions and sampled cards, including payment, continuation, bonus and debt phases. Reflection is confined to the pinned reconstruction adapter where the engine lacks setters; it is not used to expose private state. Each simulated opponent must subsequently receive `Observe(its_seat)`, never the root view or the sampled full state. Strict schemas and `policy_inputs` implement that boundary. They are a fairness interface for project policies, not an OS sandbox against malicious Python code.

S1/S2/S3, neural inputs/targets, opponent action probabilities and policy-cache isolation do not yet exist. R1 verifies the observation/choice/world foundations; those later consumers must retain the counterfactual checks when implemented.

## H0 capture and remaining milestones

Run `pwsh scripts/capture-h0.ps1 -Games 5` with the verified original assembly at ignored `artifacts/patch-baseline/Assembly-CSharp.dll`. Original and guarded Hard run in separate directories/processes. The runner records one selected rule, action/event trace, timings, explicit setup, scores and failures, and runs an uninstrumented control. A selector wrapper calls the original selector exactly once and records its returned rule; it never reruns applicability, priority or print methods for explanation.

These are diagnostic all-Hard tables, not a revised legacy bank or a fairness-qualified training pool. H0 has begun; independent candidate-alternative diagnosis, mixed tables, seat rotations and promotion remain. H1–H3, live replacement integration, model training, difficulty/personality calibration and GPU inference are unchanged future work.
