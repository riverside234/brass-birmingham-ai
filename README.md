# Brass: Birmingham AI training project

Rules-host implementation and research for an offline replacement opponent in the
Steam version of Brass: Birmingham. The target is four-player games with varied
personalities and calibrated difficulty, trained on a lab GPU and played locally
on a laptop.

**Current status:** R0 reference pins, R1's four-player host, R2's model/training
scaffold and R3's six-style CPU legacy bank are implemented locally. R3 includes
persistent intent, complete-turn planning, required reply search, immutable
versions and evidence-gated training/evaluation pools. See the
[R3 guide](docs/R3_TACTICAL_BANK.md) and [exit checks](docs/R3_EXIT_CHECKS.md) for
results and remaining gates. The [v3 update](docs/LOCAL_AI_V3_CONVERSION.md)
adds cash protection, deadline-aware industry completion, merchant routes and beer
bonuses to the six S1 Custom Game strategies and a matching frozen research bank
(introduced in package 1.4.1; the bank uses the user-accepted 28-game limited validation).
The [1.4.2 presentation and save update](docs/PRESENTATION_SAVE_1.4.2.md) repairs
player-panel layout, white industry tiles, panel closing during animations and
save/Continue defects. The [1.4.4 menu update](docs/AI_STYLE_AND_DIFFICULTY_1.4.4.md)
adds a direct style chooser and independent **Easy 2s / Mid 5s / Hard 12s** search
allowances for each bot. New setups default to Hard; existing local-AI saves retain
the preceding package's **10-second** allowance. The research bank's work setting
remains **240**, with unchanged S1 planning, adaptive intent and fallback behavior.
The [1.4.5 initialization fix](docs/AI_MENU_INITIALIZATION_1.4.5.md) makes the new
controls visible on first opening the menu, before any click or refresh.
The [1.4.6 link replay repair](docs/LINK_PLACEMENT_REPLAY_1.4.6.md) fixes link
placement replay. The [1.4.7 resource update](docs/AI_EXECUTION_RESOURCES_1.4.7.md)
caps rendering at 30 FPS during computation, restores normal presentation,
serializes retiring AI jobs and provides optional simulation suspension and
cleanup contracts for future inference backends. It applies independently of
search allowances and strategies. **Package 1.4.7 is installed and verified.**
The [compatibility check](docs/UI_PATCH_RESEARCH_COMPATIBILITY.md)
preserves the research bank and training interfaces. [Step 5](docs/STEP5_LAB_TRAINING.md)
adds self-play-first population PPO, neural search distillation, diverse-opponent
evaluation and optional Reptile meta-RL. Strong neural playing skill and calibrated
difficulty remain to be demonstrated through substantial training and evaluation.

[R4/R5 search experiments](docs/R4_R5_SEARCH_LAB.md) add the OpenSpiel adapter and
POMCP-inspired future-observation search alongside Step 5. Separate correctness
gates, frozen-checkpoint comparisons and promotion records preserve the current
training run. Implementation does not establish a search advantage.

## Included work

- [Feasibility review](ai-research/REVIEW.md)
- [Model, hardware, personality and difficulty design](ai-research/DESIGN.md)
- [Post–Step 5 personality fine-tuning, learned style discovery and difficulty calibration plan](ai-research/PERSONALITIES_AND_DIFFICULTY_PLAN.md)
- [Turn planning, hidden information and thinking-time experiments](ai-research/SEARCH_DESIGN.md)
- [Daiyu AI and OpenSpiel reference integration plan](ai-research/REFERENCE_INTEGRATION.md)
- [Legacy Hard improvements and six tactical benchmark profiles](ai-research/LEGACY_HARD_AI_PLAN.md)
- [Implementation plan](ai-research/IMPLEMENTATION_PLAN.md)
- [Implemented R0 reference pins and contracts](docs/R0_CONTRACTS.md)
- [R0 exit-check results](docs/R0_EXIT_CHECKS.md)
- [R1 host setup, API and information boundary](docs/R1_HOST.md)
- [R1 exit-check results and H0 capture](docs/R1_EXIT_CHECKS.md)
- [R2 model, training and lab commands](docs/R2_TRAINING.md)
- [R2 exit-check results](docs/R2_EXIT_CHECKS.md)
- [R3 four-player legacy bank, planner and optional game bridge](docs/R3_LEGACY_BANK.md)
- [R3 H0–H5 exit checks and qualification](docs/R3_EXIT_CHECKS.md)
- [Step 5 self-play, search, meta-RL and lab transfer instructions](docs/STEP5_LAB_TRAINING.md)
- [Step 5 larger lab preset, parallel simulators and continuing an existing trained run](docs/STEP5_SCALING.md)
- [Step 5 larger evaluation preset, parallel seat rotations and lab commands](docs/STEP5_EVALUATION_SCALING.md)
- [Step 5 executed local checks and unqualified smoke weights](docs/STEP5_EXIT_CHECKS.md)
- [R4/R5 additional search experiments and lab instructions](docs/R4_R5_SEARCH_LAB.md)
- [R4/R5 executed exit checks](docs/R4_R5_EXIT_CHECKS.md)
- [Rules audit and proposed fixes](rules-audit/BUG_FIX_PLAN.md)
- Original PowerShell/C# probe scripts under `ai-research/` and `rules-audit/`

Game executables, Unity assets, managed game assemblies, decompiled source,
downloaded tools/reference documents, generated probe results and experiment
checkpoints are excluded from Git. Curated weights under `training/models/` and
the project-authored host/v3 workers under `training/runtime/` have explicit
exceptions; their game dependencies still require a private transfer. References in the research reports to local
decompiled files or generated results are inspection evidence, not files included
in a clone. Some audit links refer to the original laptop's absolute paths.

## Clone on the lab computer

```sh
git clone https://github.com/riverside234/ai-training.git /data/not_backed_up/yxu209/ai-training/
cd /data/not_backed_up/yxu209/ai-training/
```

Keep your own game installation outside the clone. The host/bank builders may
copy the required private assemblies under ignored `artifacts/`; those runtime
dependencies are not downloaded or provided by this repository. R2 can train compatible replay shards without the
engine. New match collection needs the verified R1 runtime until a separate
implementation exists. The default lab root is
`/data/not_backed_up/yxu209/ai-training/`; R2's [setup guide](docs/R2_TRAINING.md)
includes CPU/CUDA installation, fresh initialization, collection, training and export.
For the current population trainer, use the [Step 5 lab guide](docs/STEP5_LAB_TRAINING.md).
It includes a fingerprint-checked private rules/runtime bundle, loadable R2
bootstrap weights, CUDA/Mono preflight and resumable self-play/meta-RL commands.

The R1 host was verified on Windows .NET Framework and Unity's actual Mono runtime.
Point its builder at your patch-1.1.0 installation; the full managed dependency
inventory is fingerprinted before use:

```powershell
python -m pip install -r requirements-r1.txt
pwsh -NoProfile -File scripts/build-host.ps1 -GameRoot 'D:\SteamLibrary\steamapps\common\Brass Birmingham'
python scripts/check-r1-traces.py
python scripts/test-r1.py
```

Replace the example path with your installation directory. If `-GameRoot` is
omitted, the builder uses the repository root. Generated assemblies and traces stay
under ignored `artifacts/`. See the [R1 guide](docs/R1_HOST.md) for APIs, stock baseline
capture and runtime parity checks. Linux and the lab machine have not been validated.

The older `ai-research/headless-smoke.ps1` is a historical probe requiring the
unmodified stock assembly. It is deliberately not retargeted to the patched host.

The older `rules-audit` scripts currently expect the game directory at the
repository root, and some require Windows PowerShell 5.1 / .NET Framework.
They are retained as audit fixtures, not a portable training entry point.

## Repository boundaries

The root `.gitignore` retains the original Visual Studio rules and adds a default
exclude policy for this Steam folder. It explicitly permits the research/audit
source files and future project directories: `src`, `tests`, `training`, `configs`,
`scripts`, `docs`, and `.github`. Add an intentional exception when introducing
another project directory. Keep secrets and generated training artifacts local.

Before committing, inspect what will be included:

```sh
git status --short
git diff --cached --stat
git ls-files
```

Do not force-add excluded game files. `.gitignore` applies to untracked files;
it does not remove files already committed to history.

R3 snapshots contain original code/configuration and hashes; game runtime files
stay private under ignored `artifacts/`. The [bank guide](docs/R3_TACTICAL_BANK.md)
explains lab preparation, pool selection, mixed neural/legacy collection and R2
value training. Linux/Mono and rendered human-play acceptance remain separate
from the Windows automated checks.
R0 validation can still run without game files; see its [commands and scope](docs/R0_CONTRACTS.md).

## Local rules patch

A reversible local-game patch for the rules audit is implemented under `src/BrassPatch`,
with a version-checked assembly weaver, regression tests, and installation/restoration
scripts. See [patch documentation](docs/LOCAL_PATCH.md). R1's host consumes this
patched engine; the R2 scaffold does not yet replace its AI. Online play is
disabled while this patch is installed, and games created with its versioned IDs
require the patch to continue.
