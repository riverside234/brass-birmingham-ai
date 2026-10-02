# Selected Masked v4 — installed patch 1.4.20

Installed and archived on 2 October 2026. The user selected the previous advanced revision, **`bank-3dc65253a610191d66a5`**, as **v4** and stopped further strength tests. This is a user-directed selection, not a claim that the earlier tied screen passed its strength gate. The later network/Scout experiment remains excluded.

## Requested fixes

- Productive development changes the intent progress key and resets its stall age. Developing toward a higher tier no longer abandons the opening merely because several decisions elapsed. An unchanged goal still accumulates stall age; reaching the desired tier stops further development bias.
- Opening intent contributes a bounded bias when ranking build families and loan/develop/network setup candidates in `FastIncumbents`, as well as ordinary proposals. Complete-turn economic scoring still chooses among the resulting legal plans.

Both fixes were already present in the selected immutable bank. The game uses those search/evaluation source bytes; its public labels now end in **v4**. The six choices are Adaptive Advanced Book, Coventry Goods Opening, External-Beer Three Cotton, Worcester Iron-Cotton, Zero-Loan Industry and Advanced Pottery Variants. Internal `*-v3-masked-economy` identifiers remain compatibility aliases.

## Game and training

| Context | Selected configuration |
| --- | --- |
| Game | Two private search workers; existing 2/5/12-second choices |
| Lab imitation and heuristic evaluation | Masked v4, one second per complete turn, four private search workers |
| Lab early RL opponents | Search-free incumbent from this same bank |
| Lab later RL opponents | One-second/four-worker Masked v4 at 10,000 games / 5% heuristic tables |
| Lab scheduling | 24 queued heavy tables; shared 60 CPU slots plus four reserved cores |
| Neural search/distillation | Masked ISMCTS + NN; existing four-worker batching configuration |
| Neural initialization | Fresh 1,840,109-parameter MLP-ResNet; no imported checkpoints |

`training/opponents/current.json`, `lab/current.json`, the admitted pool and `training/runtime/distribution.json` identify the selected bank. Five teacher styles are admitted for training; Advanced Pottery Variants remains an evaluation-only opponent identity. Learner Pottery actions are unrestricted.

The authored workers and frozen source/profiles are eligible for Git. Proprietary runtime dependencies remain in the private bundle **`artifacts/step5-transfer/private-rules-runtime-v4.zip`**, SHA-256 `49103db526cf01d43b6374d45f124da2a0c6ac309543bc6a640e295c28764be8`. The distribution includes ten authored workers and 904 private runtime files. Follow [the conda lab guide](TRAINING_RECOVERY_LAB.md) to transfer/import it and start a new source-bound run on `/data/not_backed_up/yxu209/ai-training/`.

## Verification

All checks passed against the installed package:

| Check | Passed assertions |
| --- | ---: |
| Masked search runtime | 99 |
| Era/workers/batching | 22 |
| Industry conversion | 17 |
| Economy, intent progress and v4 labels | 37 |
| Execution lifecycle | 10 |
| Sales and merchant bonuses | 412 |
| Borrowing | 72 |
| Cash and beer | 60 |
| Native six-style play, clocks, cancellation and resources | 1,218 |
| Saved 1.3.2 / 1.4.0 / 1.4.3 profiles | 54 each |
| Saved 1.4.9 / 1.4.11 / 1.4.15 / 1.4.17 / 1.4.18 profiles | 252 each |

The merchant-beer fixture begins within a sale to test the payment/bonus choice independently of a legitimate early-Rail decision to build first. Qualification verifies all 54 named sale scenarios across the six styles, rather than relying only on a variable count of replayed continuation choices.

CPU training preflight passed rules parity and a device backward pass. No fresh strength tournament was run. CUDA throughput is not measured by this local CPU check.

Only `BrassLegacyAI.dll` changes relative to qualified 1.4.19. Installed helper SHA-256: `c785597b44463c5bd0c64ee49404014532d0b8b200ff0c908099106a543984a0`. The installer verified the game files and preserved a recovery backup. The immutable archive is `artifacts/patch-releases/1.4.20/` (176 verified files); `installed-verification.json` records the exact installed fingerprints.
