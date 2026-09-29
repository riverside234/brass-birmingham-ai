# R3 four-player legacy bank

For the current six-style v3 generation and its deployment/research checks, see [the v3 follow-up](LOCAL_AI_V3_CONVERSION.md). The original H0–H2 measurements below remain historical.

This is a CPU heuristic implementation, with no neural weights or LLM dependency. It uses the declared R1 rules and semantic legal choices. The public library is in `src/BrassLegacyAI`; the process client is `src/brass_ai/legacy.py`. Installed game files and the R1/R2 host remain unchanged.

The H0–H2 foundation below is now extended by [H3–H5 tactical profiles, mandatory reply search and frozen-bank integration](R3_TACTICAL_BANK.md). [Current exit checks](R3_EXIT_CHECKS.md) distinguish verified functionality, pool admission and playing-strength gates. Historical H0–H2 results remain in [their original exit record](R3_H0_H2_EXIT_CHECKS.md). A frozen profile is not automatically a qualified opponent.

## H0: baseline and diagnosis

`scripts/capture-h0.ps1` runs the original and guarded stock policies in separate directories. `scripts/diagnose-legacy.py` adds a four-player control/recorded/diagnosed comparison. Stock `GetNextAIMove` and its selector are invoked exactly once per action. Logging records the returned rule, publicly played cards, context, latency, outcomes and independently generated legal alternatives with evaluator ledgers. It never calls stock applicability/printing again for an explanation.

Diagnosis reconstructs a separate world from the acting player's allowed observation. The live canonical state, including its RNG, is checked before and after every diagnostic decision. Entire public action/state/outcome traces must match the control. Wildcard plays are captured even though spent wildcards return to their decks rather than entering the discard pile. Guarded stock retains its original liquidation strategy; in mixed games its debtor's choices are obtained from an isolated trusted copy. Failures remain reliability failures and are never credited as wins.

## H1: scored complete actions

`Evaluator` is pure over an observation. Its ledger separates banked points, expected remaining industry points, link points, a single discounted cash-flow forecast, feasible development potential and a modest strongest-rival comparison. It reads the actual income-marker bands. Banked Canal points are not awarded a second time; surviving tiles retain their Rail opportunity. Merchant VP bonuses are already included in banked VP; their separate era-summary counter is not added again. Border regions contribute two link points regardless of their merchant tiles. Terminal evaluation uses four-seat official VP/income/cash ordering and split first-place utility, without converting leftover cash into VP.

Nonterminal numbers are heuristic VP equivalents, not probabilities of winning. Completion probabilities, future spending capacity, demand and development potential are approximations. They still need tuning and human playtesting; the implementation does not claim expert-level economic forecasting.

Proposals are ranked **legal semantic choices**, not old stock helper commands. Compound actions specify every discarded card, building/link, resource source, merchant, optional continuation, bonus and final confirmation. Candidate narrowing retains action/industry diversity and low-value setup families. Payments are bounded greedy completions; stop and extended versions are considered for development, sale and networking. This is bounded search, not exhaustive enumeration of every complete-turn permutation. The underlying R1 legality API remains complete.

## H2: complete turns

S0 scores complete ordinary actions. S1 compares sequences of up to two complete ordinary actions, including loans, development, Scout and network setup. The first Canal turn contains one action. Search stops before ordinary replenishment or an era transition; unseen new cards therefore cannot improve a current-turn plan. Scout's publicly specified wildcards can enable the next action.

After both actions of the last seat's final Rail turn, no opponent decisions remain. The evaluator projects the final public board score exactly, without drawing cards or advancing the engine. It ranks these plans by official win share and placement before applying personality. The nonterminal style window is never applied to terminal win-share units.

Search receives only a strict observation and legal choices. Beliefs conserve the public deck composition, own hand, public discards, known wildcards and hand sizes. Neither real opponent cards, real future deck order, vendor IDs nor the game seed enter the policy. The same deterministic sampling seed and semantic tie break make fixed-work decisions repeatable. S1 does not simulate opponents; the implemented H4/S2 extension is described in [R3_TACTICAL_BANK.md](R3_TACTICAL_BANK.md).

A complete validated incumbent is built before widening. Each request reports nodes, rejected candidates, elapsed time, budget termination and overruns. Reaching a deadline cannot turn a partially paid action into a move. Finishing/validating an incumbent can exceed very small caps; recorded decision/execution timings include that cost. Historical H2 results sum decision/execution timers. H3–H5 additionally records whole-turn timing including outer observation capture; IPC and rendered presentation remain outside it. The provisional L2 setting is a **3,000 ms whole-turn search allowance**, with 240 reconstruction nodes as an additional limit. These are experiment settings, not hard real-time guarantees or difficulty labels. S0's 250 ms default is a reliability/throughput experiment, not a claim of equivalent strength.

Rail Tempo may prefer links only within 1.5 evaluator points of the best plan. Balanced has no style allowance. The profiles do not share private intent or coordinate against a human. Original H2 v1 profiles remain stateless. H3 v2 profiles persist per-seat focus and use conditional tactical preferences.

## Build and run

On this Windows checkout:

```powershell
./scripts/build-legacy.ps1 -ManagedPath artifacts/r1-host/guarded
python scripts/run-legacy.py --root . --profile balanced-v1 --seeds 9501 9502 --seats 0 1 2 3 --milliseconds 3000 --output my-pilot
python scripts/diagnose-legacy.py --root .
python -m unittest discover -s tests/legacy-ai -v
./scripts/check-legacy-bridge.ps1
```

Each `--seats 0 1 2 3` run seats one candidate against three guarded stock bots in four seat assignments per setup. `--seats 4` is an all-candidate reliability run; `--seats -1` is all guarded stock. `--fixed-work --work 240` is deterministic regression mode, not a wall-clock measurement. Output directories must be new so a previous report cannot be silently overwritten.

The default lab root is `/data/not_backed_up/yxu209/ai-training/`. For example:

```bash
cd /data/not_backed_up/yxu209/ai-training/
python scripts/run-legacy.py --profile balanced-v1 --seeds 9601 --seats 0 1 2 3 --output lab-pilot
```

The lab requires a **private copy of the matching runtime/dependencies**, kept under `artifacts/legacy-ai/runtime/`, and Mono to execute the .NET worker. Git contains no game binaries or assets. Linux/Mono equivalence is not qualified here; repeat the R1/R3 exit checks there before generating production training data. Training R2 from compatible replay shards does not require this game runtime. The RTX GPU is unnecessary for the legacy bank.

## Optional game bridge

`scripts/build-legacy-overlay.ps1` produces a separate optional overlay in `artifacts/legacy-ai/overlay/`. It patches AI routing only and verifies the method diff. It does not install the overlay. The factory hook preserves the vendor behavior types for save compatibility; profile resolution and four-player/Hard-only routing happen at the asynchronous turn hook, including loaded games. Two-/three-player and other AI settings retain their existing path.

`TurnCoordinator` copies observations/choices, searches on a worker, publishes complete incumbents, and rejects stale game/revision/actor/request/profile/schema responses. `LegacyBehavior` executes a prepared action once. The Unity adapter applies one action on the game thread, revalidates the second action on a later frame, retains one turn deadline, and offers Play Now when an incumbent exists. End-turn/round handling remains with the vendor presenter. Debt selection in the optional UI overlay retains the guarded game's existing debt flow; the headless bank has its explicit liquidation interface.

The live observer merges saved event history with uncommitted events at the current revision, so wildcard knowledge and Rail discard accounting survive save/load and undo. It does not inspect opponents' hidden cards to reconstruct that knowledge. Profile hashes are persisted in a game-keyed sidecar; no custom AI object enters vendor saves. Sidecar mismatch stops visibly. Unexpected mid-application failure stops the turn visibly rather than pretending cancellation rolled it back.

`live.example.json` is disabled by default. The optional overlay must pass its remaining interactive acceptance checks before installation: rendered one-human/three-bot games, menu/restart during thinking, Play Now UI, event animations/bonus/debt presentation, and save/load of supported interactive boundaries. Headless Unity tests alone do not establish that acceptance. Disable by removing/turning off the local `live.json`; `scripts/restore-legacy-overlay.ps1` restores a known overlay to the exact patched 1.1.0 assembly and retains saves/sidecars. It rejects unknown installed files. Do not overwrite the installed rules patch with unqualified files.

## Evidence and next gate

See [R3_EXIT_CHECKS.md](R3_EXIT_CHECKS.md) for current checks/admission and [R3_H0_H2_EXIT_CHECKS.md](R3_H0_H2_EXIT_CHECKS.md) for earlier pilots. H5 implements prespecified tournaments and pool gates. Stronger-Hard, calibrated difficulty and human enjoyment require their separate release evidence.
