# Migration plans

All migration-plan documents are grouped here. Run script commands from the
repository root; moving a plan does not change a CLI's location.

| Plan | Scope | Status |
| --- | --- | --- |
| [Further script migration](FURTHER_SCRIPT_MIGRATION_PLAN.md) | Reviews 120 top-level scripts against the g3170 workflow | Completed: 76 moves; 44 paths retained |
| [Training script migration](TRAINING_SCRIPT_MIGRATION_PLAN.md) | Historical training recipes/restarts and retired Rust lab bundles | Completed: 94 moves, with results recorded in the plan |
| [Local-game script migration](LOCAL_GAME_SCRIPT_MIGRATION_PLAN.md) | Builds, installs, and checks grouped by patch feature | First pass completed: 145 moves; release-source dependencies retained |
| [Rust backend migration](RUST_AI_BACKEND_MIGRATION_PLAN.md) | Architecture, implementation milestones, and qualification history | Technical migration history, separate from script layout |

The earlier plans retain their implementation history. The further plan
records the additional moves and compatibility checks.
See the [script workflow index](../../scripts/README.md) for runnable paths.
