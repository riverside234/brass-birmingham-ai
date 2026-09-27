# Brass: Birmingham AI training project

Research and implementation planning for an offline replacement opponent in the
Steam version of Brass: Birmingham. The target is four-player games with varied
personalities and calibrated difficulty, trained on a lab GPU and played locally
on a laptop.

**Current status:** feasibility probes, a rules audit, and design/implementation
plans. A replacement AI, reusable training environment, and neural training loop
have not been implemented yet. Cloning this repository does not start training.

## Included work

- [Feasibility review](ai-research/REVIEW.md)
- [Model, hardware, personality and difficulty design](ai-research/DESIGN.md)
- [Implementation plan](ai-research/IMPLEMENTATION_PLAN.md)
- [Rules audit and proposed fixes](rules-audit/BUG_FIX_PLAN.md)
- Original PowerShell/C# probe scripts under `ai-research/` and `rules-audit/`

Game executables, Unity assets, managed game assemblies, decompiled source,
downloaded tools/reference documents, generated probe results and model
checkpoints are excluded from Git. References in the research reports to local
decompiled files or generated results are inspection evidence, not files included
in a clone. Some audit links refer to the original laptop's absolute paths.

## Clone on the lab computer

```sh
git clone https://github.com/riverside234/ai-training.git
cd ai-training
```

Keep your own game installation and required game assemblies outside the clone.
They are runtime dependencies for the current probes and are not downloaded or
provided by this repository. Future training will also need access to the verified
rules engine until a separate implementation exists.

The headless feasibility probe was verified with PowerShell 7.6.5 / .NET 10 on
Windows. Point it at the game installation on that computer:

```powershell
pwsh -NoProfile -File ./ai-research/headless-smoke.ps1 -Players 4 -Games 1 -GameRoot 'D:\SteamLibrary\steamapps\common\Brass Birmingham' -OutputPath ./ai-research/smoke-lab.json
```

Replace the example path with your installation directory. Alternatively, set
`BRASS_GAME_ROOT` before running the probe. If neither is set, the probe uses the
repository root, preserving operation in the original Steam installation folder.
The assembly fingerprint is checked before executing game code. A successful run
tests the original rules and original AI; it is not neural training. Linux and
the lab machine's runtime have not been validated yet.

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

The first implementation milestone is the complete four-player legal-decision
environment, state restoration and player-information boundary described in the
[implementation plan](ai-research/IMPLEMENTATION_PLAN.md).
