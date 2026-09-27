# Brass: Birmingham AI feasibility review

Reviewed 27 September 2026, America/Chicago.

Follow-up: `DESIGN.md` records the user's priorities for four-player-only training, distinct personalities, difficulty levels above the original Hard AI, and optional Intel GPU inference. Its model-size and deployment recommendations supersede the initial conservative proposal below.

**A lightweight, offline replacement AI is technically feasible.** The installed game has a clear AI interface, and its original rules engine runs outside Unity. The strongest evidence is an executable probe: 15 complete Hard-AI matches finished successfully on this computer, across all three player counts. A new AI has not yet been trained or installed.

The intended deployment is a compact game-playing model on the laptop CPU. The lab RTX GPU would be used during training; normal play would work offline without the lab server. This is a conventional game AI using structured state and legal moves.

**What was inspected and verified**

| Item | Finding |
| --- | --- |
| Game | Cublo, Brass Birmingham, Steam app 1287920 |
| Runtime | Unity 2019.1.7 family, Mono scripting backend |
| Main code | `BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll`, 1,413,632 bytes |
| Code accessibility | Managed .NET code decompiled successfully with meaningful type and method names |
| Existing Hard AI | Hand-written, state-dependent priority rules; the selector chooses the highest-priority applicable rule |
| Rules engine | `BoardGameRules.*`; the tested paths execute without Unity, graphics, Steam, or network services |
| Laptop CPU | Registry reports Intel Core Ultra 5 135U |
| Original assembly | SHA-256 unchanged after inspection and probes |

Assembly SHA-256:

`600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB`

The review added this analysis folder only. It did not install a mod loader, replace a game DLL, access game saves, or connect to the lab server. `decompiled` contains local inspection output from the installed assembly; `tools` contains the official ILSpy command-line package, version 9.1.0.7988. These inspection outputs are not needed to play the game.

**The replacement points are concrete**

The AI contract is:

```csharp
Action GetNextAIMove(BrassApplicationGame brassApplicationGame);
```

`IAiBehavior` also defines a diagnostic `PrintNextAIMove` method. `AiBehaviorFactory.CreateAiBehavior(PlayerMetadata)` constructs the Easy, Medium, and Hard implementations. `PlayerMetadata.AIBehaviour` stores the behavior used during play.

`PlayAiTurnUseCase.PlayAITurn` repeatedly requests an action, executes its returned delegate, and calls `TurnFlow.ConfirmCurrentAction()`. A replacement can implement this same contract and return a delegate that applies a chosen move through the game's existing action APIs. The adapter must preserve this lifecycle: it must not confirm the action a second time or silently consume a turn after an invalid move.

The proposed first integration is a configurable replacement for the existing Hard difficulty, using a BepInEx/Harmony patch to the factory. That avoids needing a new menu option initially. The factory is also used when constructing loaded-game player behaviors. A configurable fallback can retain the original behavior, but it must occur before any partial action changes the game state. Runtime loading and patch compatibility remain untested in this review. [BepInEx documentation](https://docs.bepinex.dev/) describes the Mono plugin and runtime-patching mechanism.

The original `HardAiBehavior` initializes `BoardHelper`, invokes `PrioritySelector`, and executes the selected rule. `RuleSetFactory.CreateBasicRuleSet` lists 33 candidate rule classes before difficulty filtering. Examples cover loans, breweries, industrial development, selling, and links. The inspected Hard decision path has no tree search or learned policy. This explains the design's strategic limitations; it does not by itself prove which particular decisions cause weak play.

| Local evidence in `decompiled` | What it establishes |
| --- | --- |
| `BrassApplication.AI.AiBehaviors/IAiBehavior.cs:9` | Decision interface |
| `BrassApplication.AI.AiBehaviors/AiBehaviorFactory.cs:7` | Behavior construction point |
| `BrassApplication.AI.AiBehaviors/HardAiBehavior.cs` | Hard AI decision path |
| `BrassApplication.AI.RuleBasedSystem.RuleSelectors/PrioritySelector.cs:17` | Priority-based selection |
| `BrassApplication.UseCases.Game.PlayAiTurn/PlayAiTurnUseCase.cs:25` | Action execution and confirmation |
| `BoardGameRules.Entities.Game/BoardGameRulesGame.cs` | Rules initialization and game state |

**Executed feasibility test**

`headless-smoke.ps1` loads the installed assembly and constructs an offline game in memory. It runs the original Hard AI, confirms actions, advances rounds and both eras, and records final scores. It includes the original forced-industry-sale rule for debt, although that path was not exercised in these 15 matches. Each process runs its games sequentially; the three player-count batches ran in separate processes.

| Players | Completed games | Total actions | Whole-match elapsed time |
| --- | ---: | ---: | ---: |
| 2 | 5/5 | 390 | 0.36–1.03 s |
| 3 | 5/5 | 525 | 0.50–1.11 s |
| 4 | 5/5 | 620 | 0.68–1.28 s |

All 15 reached the end of the rail era: 1,535 actions total. The five integer seeds were 20260927 through 20260931. One additional two-player run repeated seed 20260927; scores, ranking, action counts, turns, and revision matched the earlier run.

These are small feasibility tests using the old AI, PowerShell 7.6.5, and .NET 10.0.11 on the laptop. They are not a benchmark of a future model, search throughput, GPU training, or a proof that every rule edge case works. The repeat comparison checks summaries rather than every event. Equivalence with the game's Unity Mono runtime and execution on the lab server still need validation.

Results are saved in `smoke-2p.json`, `smoke-3p.json`, `smoke-4p.json`, `smoke-repeat.json`, and `repeat-check.json`. From the game directory, a probe can be reproduced with:

```powershell
pwsh -NoProfile -File .\ai-research\headless-smoke.ps1 -Players 2 -Games 5
```

The script checks the assembly fingerprint before running. It requires PowerShell 7 and the installed game assembly; it is an inspection harness, not yet a training environment.

**A suitable design for the laptop**

Use a small neural network to score legal candidate moves from structured game features. Inputs would describe the public board, industries, resources, money, income, turn order, remaining actions, era, the acting player's cards, and each candidate move. The environment supplies legal choices, so the model does not need to learn the rules through illegal-move penalties.

A reasonable initial design budget is approximately 0.2–1 million parameters, or 0.8–4 MB of float32 weights before packaging. This is a proposed size budget, not an implemented architecture or measured performance claim. Start with CPU inference. The Intel integrated GPU is optional; no dedicated inference GPU is required by this design. A small dense model could run directly in C#, or through ONNX Runtime in a local worker process. ONNX Runtime provides C# inference and CPU thread controls, but compatibility inside this older Unity Mono process has not been established. [C# inference documentation](https://onnxruntime.ai/docs/get-started/with-csharp.html), [CPU thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

Begin with policy-only move selection. If testing shows benefit, add a bounded look-ahead, for example a configurable one-second budget per action. Inference latency, move generation, search, and battery usage should be measured separately on the laptop. The time budget is a design target, not a promised response time.

On the lab server, use CPU workers to produce games and the RTX GPU to train batched model updates. Train against the original AI and a pool of earlier versions of the new AI, then continue self-play. Imitating the original AI can help initialize a model, but imitation alone is unlikely to fix its weaknesses. The exact GPU variant, server OS, CPU capacity, and installed training software were not inspected; those affect the eventual training setup and throughput.

**The engineering work still required**

1. **Build the training interface.** Add reset, player observation, legal choices, step, terminal scores, and reproducible seeded episodes around the verified rules engine. A process-based worker is the safest starting point for running multiple games at once.
2. **Represent complete moves.** Existing build, link, sell, resource, and card-validation helpers are reusable, but there is no verified single method that enumerates every complete legal move. Coal/iron/beer sourcing, discarded cards, double rail links, multiple sales, merchant development bonuses, scouting, and forced liquidation need explicit handling. A simpler first prototype could learn which existing rule to select; its strength would remain constrained by the original rules' candidate choices.
3. **Restrict the observation.** The engine exposes every player's `Hand` and `Board.CardDeck`. Only the acting player's hand and public information should enter the policy. Exclude opponent cards, exact future draws, the RNG seed, and RNG state. Search must sample plausible hidden states from permitted information. Information-set search is an established approach for this problem. [Cowling, Powley and Whitehouse, Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf).
4. **Make simulation isolation reliable before adding search.** `BoardGameRulesGame` has an empty copy constructor; `BrassApplicationGame`'s copy constructor shares the same underlying game. They are not usable deep clones. Existing event replay is promising but needs correctness and speed tests. `DevelopMerchantBonus.Instance` and `BoardHelper` contain mutable static game state, so arbitrary simultaneous games or search branches in one process can interfere. Use process isolation initially and resolve these references before in-process branching.
5. **Train and evaluate.** Use four-player games from the first training prototype, following the user's clarified priority. Evaluate on held-out seeds with rotated seats, report win rate and score margins with uncertainty, and compare both against stock Hard and previous model versions. A high win rate against one weak opponent is insufficient evidence of broadly strong play. See `DESIGN.md` for personality and difficulty calibration.
6. **Package and test the in-game replacement.** Validate factory interception, action completion, fallback behavior, saving/loading, era changes, game completion, hidden-information restrictions, and CPU timing. Keep the model and adapter optional and version the feature/action schema with the model.

The next implementation milestone is a reusable legal-move environment plus a minimal in-game adapter. The current review establishes that the installed engine is accessible and runnable, substantially reducing the uncertainty around that work. It does not yet establish a stronger opponent.
