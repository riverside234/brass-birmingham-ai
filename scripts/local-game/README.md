# Local game patches

Each row keeps a patch feature's build, install, and verification commands
together. The package row includes the shared staging and install scripts.
A dash means the feature has no dedicated script in this directory for that step.
See the [local patch guide](../../docs/LOCAL_PATCH.md) for the managed package.

## Package and runtime

| Feature | Build or package | Install or restore | Verify or archive |
| --- | --- | --- | --- |
| Rules and local AI bundle | [build-patch.ps1](../build-patch.ps1)<br>[build-local-ai.ps1](../build-local-ai.ps1) | [stage-archived-patch.ps1](../stage-archived-patch.ps1)<br>[install-patch.ps1](../install-patch.ps1)<br>[install-combined-patch.ps1](../install-combined-patch.ps1)<br>[uninstall-patch.ps1](../uninstall-patch.ps1) | [test-unity-patch.ps1](../test-unity-patch.ps1)<br>[test-installed-local-ai.ps1](../test-installed-local-ai.ps1) |
| Shared install helper | — | [patch-install-common.ps1](../patch-install-common.ps1) | — |
| Legacy bridge and overlay | [build-legacy.ps1](../build-legacy.ps1)<br>[build-legacy-overlay.ps1](../build-legacy-overlay.ps1)<br>[LegacyWeaver.cs](../LegacyWeaver.cs) | [restore-legacy-overlay.ps1](../restore-legacy-overlay.ps1)<br>[sync-research-managed.ps1](../sync-research-managed.ps1) | [verify-legacy-integration.py](../verify-legacy-integration.py)<br>[check-legacy-bridge.ps1](../check-legacy-bridge.ps1)<br>[check-legacy-overlay-ui.ps1](../check-legacy-overlay-ui.ps1) |
| Mixed AI | [build-mixed-ai.py](../build-mixed-ai.py) | [install-mixed-ai.ps1](../install-mixed-ai.ps1) | [verify-mixed-legacy.py](../verify-mixed-legacy.py) |
| Rust local runtime | [build-rust-local-package.py](../build-rust-local-package.py) | [install-rust-local-release.ps1](../install-rust-local-release.ps1)<br>[install-rust-alpha.ps1](../install-rust-alpha.ps1) | [qualify-rust-local.py](../qualify-rust-local.py)<br>[qualify-rust-release.py](../qualify-rust-release.py)<br>[verify-rust-local.py](../verify-rust-local.py)<br>[finalize-rust-local.py](../finalize-rust-local.py)<br>[test-rust-local-upgrade.ps1](../test-rust-local-upgrade.ps1)<br>[test-rust-release-deployment.ps1](../test-rust-release-deployment.ps1) |

## UI and rules patches

| Feature | Build or package | Install or restore | Verify or archive |
| --- | --- | --- | --- |
| AI selection UI | [build-ai-selection-ui.ps1](../build-ai-selection-ui.ps1) | — | [archive-ai-selection-ui-release.ps1](../archive-ai-selection-ui-release.ps1) |
| Player selection | [build-player-selection.ps1](../build-player-selection.ps1) | — | [archive-player-selection-release.ps1](../archive-player-selection-release.ps1) |
| Game history | [build-game-history.ps1](../build-game-history.ps1) | — | [archive-game-history-release.ps1](../archive-game-history-release.ps1) |
| Turn history fix | [build-turn-history-fix.py](../build-turn-history-fix.py) | [install-turn-history-fix.ps1](../install-turn-history-fix.ps1) | [qualify-turn-history-fix.py](../qualify-turn-history-fix.py) |
| Game finishing margin | [build-game-finishing-margin.ps1](../build-game-finishing-margin.ps1) | — | — |
| Network input fix | [build-network-input-fix.ps1](../build-network-input-fix.ps1) | — | [archive-network-input-release.ps1](../archive-network-input-release.ps1) |
| Public history bonus | — | — | [test-public-history-bonus.py](../test-public-history-bonus.py) |
| Information UI | [build-information-ui.ps1](../build-information-ui.ps1) | — | [archive-information-ui-release.ps1](../archive-information-ui-release.ps1) |
| Card tracker | [build-card-tracker.ps1](../build-card-tracker.ps1) | — | [archive-card-tracker-release.ps1](../archive-card-tracker-release.ps1) |
| Card tracker light | [build-card-tracker-light.ps1](../build-card-tracker-light.ps1) | — | [archive-card-tracker-light-release.ps1](../archive-card-tracker-light-release.ps1) |
| Card tracker log button | [build-card-tracker-log-button.ps1](../build-card-tracker-log-button.ps1) | — | [archive-card-tracker-log-button-release.ps1](../archive-card-tracker-log-button-release.ps1) |
| Card tracker for three players | [build-card-tracker-three.py](../build-card-tracker-three.py) | [install-card-tracker-three.ps1](../install-card-tracker-three.ps1) | — |
| Statistics base and readability | — | — | [archive-statistics-release.ps1](../archive-statistics-release.ps1)<br>[archive-statistics-readability-release.ps1](../archive-statistics-readability-release.ps1) |
| Statistics light | [build-statistics-light.ps1](../build-statistics-light.ps1) | — | [archive-statistics-light-release.ps1](../archive-statistics-light-release.ps1) |
| Statistics review | [build-statistics-review.ps1](../build-statistics-review.ps1) | — | [archive-statistics-review-release.ps1](../archive-statistics-review-release.ps1) |
| Statistics geography | [build-statistics-geography.ps1](../build-statistics-geography.ps1) | — | [archive-statistics-geography-release.ps1](../archive-statistics-geography-release.ps1) |
| Statistics geography refinement | [build-statistics-geography-refinement.ps1](../build-statistics-geography-refinement.ps1) | — | [archive-statistics-geography-refinement-release.ps1](../archive-statistics-geography-refinement-release.ps1) |
| Statistics geography polish | [build-statistics-geography-polish.ps1](../build-statistics-geography-polish.ps1) | — | [archive-statistics-geography-polish-release.ps1](../archive-statistics-geography-polish-release.ps1) |
| Statistics efficiency | [build-statistics-efficiency.ps1](../build-statistics-efficiency.ps1) | — | [archive-statistics-efficiency-release.ps1](../archive-statistics-efficiency-release.ps1) |
| Setup preferences | [build-setup-preferences-patch.py](../build-setup-preferences-patch.py) | [install-setup-preferences-patch.ps1](../install-setup-preferences-patch.ps1) | [qualify-setup-preferences-patch.py](../qualify-setup-preferences-patch.py) |
| Sell replay speed | [fix-sell-replay-speed.py](../fix-sell-replay-speed.py) | [install-sell-replay-speed.ps1](../install-sell-replay-speed.ps1) | — |
| Execution history | — | — | [archive-execution-release.ps1](../archive-execution-release.ps1) |
| Link replay | — | — | [archive-link-replay-release.ps1](../archive-link-replay-release.ps1) |

## Local AI and search patches

| Feature | Build or package | Install or restore | Verify or archive |
| --- | --- | --- | --- |
| Era AI | [build-era-ai.ps1](../build-era-ai.ps1) | — | [archive-era-release.ps1](../archive-era-release.ps1) |
| Masked ISMCTS and strength | — | — | [verify-masked-ismcts.py](../verify-masked-ismcts.py)<br>[archive-masked-release.ps1](../archive-masked-release.ps1)<br>[archive-masked-strength-release.ps1](../archive-masked-strength-release.ps1) |
| V4 search time | [build-v4-search-time.ps1](../build-v4-search-time.ps1) | — | [archive-v4-search-time-release.ps1](../archive-v4-search-time-release.ps1) |
| V4 four-times search | [build-v4-four-times.ps1](../build-v4-four-times.ps1) | — | [archive-v4-four-times-release.ps1](../archive-v4-four-times-release.ps1) |
| NN beta | [nn-beta.py](../nn-beta.py) | [install-nn-beta.ps1](../install-nn-beta.ps1) | [smoke-installed-nn-beta.ps1](../smoke-installed-nn-beta.ps1) |
| NN beta 2 | [build-nn-beta2.py](../build-nn-beta2.py)<br>[package-nn-beta2.py](../package-nn-beta2.py) | [install-nn-beta2.ps1](../install-nn-beta2.ps1) | — |
| NN C_scale | [build-nn-c-scale.py](../build-nn-c-scale.py) | [install-nn-c-scale.ps1](../install-nn-c-scale.ps1) | — |
| NN delivery | [fix-nn-delivery.py](../fix-nn-delivery.py) | [install-nn-delivery.ps1](../install-nn-delivery.ps1) | — |
| NN IL | [build-nn-il.py](../build-nn-il.py)<br>[nn-il.py](../nn-il.py) | [install-nn-il.ps1](../install-nn-il.ps1) | [smoke-installed-nn-il.ps1](../smoke-installed-nn-il.ps1) |
| NN industry focus | [nn-industry-focus.py](../nn-industry-focus.py) | [install-nn-industry-focus.ps1](../install-nn-industry-focus.ps1) | [smoke-installed-nn-industry-focus.ps1](../smoke-installed-nn-industry-focus.ps1) |
| NN K32 | [build-nn-k32.py](../build-nn-k32.py)<br>[build-nn-k32-native.py](../build-nn-k32-native.py) | [install-nn-k32.ps1](../install-nn-k32.ps1) | [qualify-nn-k32.py](../qualify-nn-k32.py)<br>[finalize-nn-k32.py](../finalize-nn-k32.py)<br>[smoke-installed-nn-k32.ps1](../smoke-installed-nn-k32.ps1) |
| NN neutral UI | [nn-neutral-ui.py](../nn-neutral-ui.py) | [install-nn-neutral-ui.ps1](../install-nn-neutral-ui.ps1) | [smoke-installed-nn-neutral-ui.ps1](../smoke-installed-nn-neutral-ui.ps1) |
| NN opening temperature | [build-nn-opening-temperature.py](../build-nn-opening-temperature.py) | [install-nn-opening-temperature.ps1](../install-nn-opening-temperature.ps1) | [qualify-nn-opening-temperature.py](../qualify-nn-opening-temperature.py) |
| NN search choice | [build-nn-search-choice.py](../build-nn-search-choice.py) | [install-nn-search-choice.ps1](../install-nn-search-choice.ps1) | [qualify-nn-search-choice.py](../qualify-nn-search-choice.py)<br>[smoke-installed-nn-search-choice.ps1](../smoke-installed-nn-search-choice.ps1) |
| NN temperature | [build-nn-temperature.py](../build-nn-temperature.py)<br>[build-nn-temperature-native.py](../build-nn-temperature-native.py) | [install-nn-temperature.ps1](../install-nn-temperature.ps1) | [qualify-nn-temperature.py](../qualify-nn-temperature.py)<br>[smoke-installed-nn-temperature.ps1](../smoke-installed-nn-temperature.ps1) |
| Fixed game reserve | [fixed-game-reserve.py](../fixed-game-reserve.py) | [install-fixed-reserve.ps1](../install-fixed-reserve.ps1) | [check-fixed-reserve-user-data.ps1](../check-fixed-reserve-user-data.ps1)<br>[smoke-installed-fixed-reserve.ps1](../smoke-installed-fixed-reserve.ps1) |
| Host delivery | [build-host-delivery.py](../build-host-delivery.py)<br>[build-host-delivery-release.py](../build-host-delivery-release.py) | [install-host-delivery.ps1](../install-host-delivery.ps1) | [test-host-delivery.py](../test-host-delivery.py)<br>[smoke-host-delivery.py](../smoke-host-delivery.py)<br>[smoke-host-delivery-release.py](../smoke-host-delivery-release.py)<br>[finalize-host-delivery-installation.py](../finalize-host-delivery-installation.py)<br>[check-host-delivery-edges.ps1](../check-host-delivery-edges.ps1)<br>[check-host-delivery-user-data.ps1](../check-host-delivery-user-data.ps1)<br>[smoke-installed-host-delivery.ps1](../smoke-installed-host-delivery.ps1) |
| Search delivery grace | [search-delivery-grace.py](../search-delivery-grace.py) | [install-search-delivery-grace.ps1](../install-search-delivery-grace.ps1) | [check-search-delivery-budget.ps1](../check-search-delivery-budget.ps1)<br>[check-search-delivery-grace-user-data.ps1](../check-search-delivery-grace-user-data.ps1)<br>[finalize-search-delivery-grace.ps1](../finalize-search-delivery-grace.ps1)<br>[smoke-installed-search-delivery-grace.ps1](../smoke-installed-search-delivery-grace.ps1) |
| Search v1 default | [build-search-v1-default.py](../build-search-v1-default.py) | [install-search-v1-default.ps1](../install-search-v1-default.ps1) | [qualify-search-v1-default.py](../qualify-search-v1-default.py)<br>[smoke-installed-search-v1.ps1](../smoke-installed-search-v1.ps1) |
| Search v2 | [build-search-v2-managed.py](../build-search-v2-managed.py)<br>[search-v2-tools.py](../search-v2-tools.py) | [install-search-v2.ps1](../install-search-v2.ps1) | [smoke-search-v2-unity.py](../smoke-search-v2-unity.py)<br>[verify-search-v2-delivery.py](../verify-search-v2-delivery.py)<br>[verify-search-v2-native.py](../verify-search-v2-native.py)<br>[smoke-installed-search-v2.ps1](../smoke-installed-search-v2.ps1) |
| Search v3 | [build-search-v3.py](../build-search-v3.py)<br>[search_v3_patch.py](../search_v3_patch.py) | — | [run-search-v3-smoke.py](../run-search-v3-smoke.py) |
