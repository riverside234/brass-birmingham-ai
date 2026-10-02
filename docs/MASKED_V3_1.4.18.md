# v3: useful industry upgrades and grouped sales

This revision keeps Masked ISMCTS, the six v3 styles and the existing 2/5/12-second game options. The game uses **two search workers**. Lab imitation, searched legacy opponents and NN distillation use **four**, with **24 queued tables** and the existing shared CPU admission gate.

## Industry investment

Iron is the development resource (the game calls it iron rather than steel). Development proposals quote the actual market cost after available board iron, preserve enough funding to build the replacement, and allow time for its sale or resource consumption. Higher tiers can be useful in both eras; level two and above can survive Canal into Rail.

The planner can consider a staged upgrade that needs two Develop actions, including clearing three low Cotton tiles. It does not reward every tile discarded. A high tier with no useful card, buyer, funding or time is not an upgrade opportunity. Full candidate plans still execute through the strict rules host.

The investment ledger values the strongest available mat opportunity and discounts a second alternative to 35%. It no longer adds six independently funded future investments together. This prevents development across unrelated industries from crowding out actual builds. Merchant free development is ranked by its improvement to that same portfolio; it can still unlock a useful Pottery investment.

An unflipped level-two sellable industry retains discounted Rail value when Canal actions run out. A credible Canal flip can increase that carry-over forecast. Only real flips and official scoring create banked points.

## Sale timing

The planner explicitly compares building a companion industry and selling the feasible group as action two. During a sale continuation, it keeps adding legal sales, paying beer and resolving each merchant bonus through the rules host.

A safe lone industry can wait for a funded companion. An already-ready group keeps a strong immediate sale prior. Early single sales remain useful near an era deadline, when delaying risks the beer supply, to capture merchant beer, or to protect income and avoid forced liquidation/credit trouble. A bounded three-point action-opportunity cost discourages unnecessary single sales, capped at four per complete turn; it is a heuristic and is never added to official VP. Terminal win shares remain exact.

## Promotion and verification

Compare the candidate against the frozen current Masked v3 (`bank-3f7221cadcd472aafdb9`) and the old S1 v3 (`bank-d82438ccec287dbc8a64`). Each comparison uses three setup/style pairs, all four seat rotations, one second and four workers per decision: **12 games per control, 24 in total**.

Promotion requires complete games without errors and **at least one** improvement against **each** control: more official win share, higher mean VP, or lower mean numeric rank. This is a small directional screen, not a statistically established strength claim. Earlier candidates and their rejected results stay separate; their games do not enter training.

```sh
conda activate ai-training
cd /data/not_backed_up/yxu209/ai-training/
python /data/not_backed_up/yxu209/ai-training/scripts/smoke-industry-v3.py --root /data/not_backed_up/yxu209/ai-training/ --bank BANK_FROM_CURRENT_JSON --output /data/not_backed_up/yxu209/ai-training/artifacts/industry-v3/lab-screen --tables 24
```

Replace `BANK_FROM_CURRENT_JSON` with the identity in `training/opponents/current.json`. The private distribution includes both controls; their code and runtime bytes remain frozen. The old S1 control's work ceiling is increased so dividing it among four workers does not prematurely truncate its one-second clock.

Deployment also requires fresh strict-choice, sale/beer/credit, worker retirement, resource restoration, native game and saved-profile checks. Existing saves retain their original profile bytes and assignments while using the updated deployment behavior. The package replaces only `BrassLegacyAI.dll`, preserving the installed 1.4.17 UI and rules binaries exactly.

## Lab training

See [the conda guide](TRAINING_RECOVERY_LAB.md). The admitted bank supplies searched imitation teachers and legacy evaluation opponents, the early search-free v3 opponents, and the later one-second searched v3 opponents. Pottery Flex remains an opponent-identity evaluation holdout; the learner can freely build and develop Pottery.

The neural model still starts freshly, with 1,840,109 parameters. Neural search/distillation retains Masked ISMCTS + NN, raw neural Q and the information-frontier horizon. This heuristic revision does not replace neural values or add the heuristic sale cost to NN training rewards. Regenerate teacher data if you want the new behavior throughout imitation; previously audited heuristic datasets keep their archived provenance.
