# Training migrations

These scripts prepare or adopt settings for specific committed runs or
generations. For routine continuation, see [training](../README.md).

## Saved generation recipe changes

- [configure-training-g370.py](configure-training-g370.py)
- [configure-training-g570.py](configure-training-g570.py)
- [configure-training-g611.py](configure-training-g611.py)
- [configure-training-g760.py](configure-training-g760.py)
- [configure-training-g850.py](configure-training-g850.py)
- [configure-training-g950.py](configure-training-g950.py)
- [configure-training-g1041.py](configure-training-g1041.py)
- [configure-training-g1050.py](configure-training-g1050.py)
- [configure-training-g1120.py](configure-training-g1120.py)
- [configure-training-g1241.py](configure-training-g1241.py)
- [configure-training-g1247.py](configure-training-g1247.py)
- [configure-training-g1320.py](configure-training-g1320.py)
- [configure-training-g1322.py](configure-training-g1322.py)
- [configure-training-g1324.py](configure-training-g1324.py)
- [configure-training-g1340.py](configure-training-g1340.py)

## Joint-7 continuation recipes

- [joint7-entropy-groups.py](joint7-entropy-groups.py)
- [joint7-entropy95.py](joint7-entropy95.py)
- [joint7-exploration.py](joint7-exploration.py)
- [joint7-g110-hotfix.py](joint7-g110-hotfix.py)
- [joint7-g503.py](joint7-g503.py)
- [joint7-g503-ppg.py](joint7-g503-ppg.py)
- [joint7-g540-ppg.py](joint7-g540-ppg.py)
- [joint7-panel-gates.py](joint7-panel-gates.py)
- [joint7-ppg-eight.py](joint7-ppg-eight.py)
- [joint7-ppg-window.py](joint7-ppg-window.py)
- [joint7-ppo-only.py](joint7-ppo-only.py)
- [joint7-qualified-anchor.py](joint7-qualified-anchor.py)
- [joint7-snapshot-r2.py](joint7-snapshot-r2.py)
- [joint7-throughput.py](joint7-throughput.py)
- [joint7-v4-share.py](joint7-v4-share.py)
- [joint7-vp-cap.py](joint7-vp-cap.py)

## PPO, PPG, and self-play configuration

- [configure-dcpg.py](configure-dcpg.py)
- [configure-entropy-schedule.py](configure-entropy-schedule.py)
- [configure-gae-floor.py](configure-gae-floor.py)
- [configure-pfsp.py](configure-pfsp.py)
- [configure-ppg-eight-fast.py](configure-ppg-eight-fast.py)
- [configure-ppg-g210.py](configure-ppg-g210.py)
- [configure-ppg-phasic.py](configure-ppg-phasic.py)
- [configure-ppg-sixteen.py](configure-ppg-sixteen.py)
- [configure-ppg-throughput.py](configure-ppg-throughput.py)
- [configure-ppo-lr-threshold.py](configure-ppo-lr-threshold.py)
- [configure-ppo-probes.py](configure-ppo-probes.py)
- [configure-selfplay-throughput.py](configure-selfplay-throughput.py)

## Teacher, scoring, and evaluation configuration

- [configure-domino.py](configure-domino.py)
- [configure-domino-context-evaluation.py](configure-domino-context-evaluation.py)
- [configure-local-scoring-aux.py](configure-local-scoring-aux.py)
- [configure-teacher-phase.py](configure-teacher-phase.py)
- [prepare-rust-post-transfer-reward.py](../../prepare-rust-post-transfer-reward.py)

## Historical restarts

- [continue-rust-g1190-restart.py](continue-rust-g1190-restart.py)
- [continue-rust-g1490-restart.py](continue-rust-g1490-restart.py)
- [continue-rust-revised.py](continue-rust-revised.py)
- [joint7_restart160.py](joint7_restart160.py)
- [prepare-neutral-selfplay.py](prepare-neutral-selfplay.py)
- [prepare-rust-branch-restart.py](prepare-rust-branch-restart.py)
- [prepare-rust-g1190-restart.py](prepare-rust-g1190-restart.py)
- [prepare-rust-g1190-restart-2.py](prepare-rust-g1190-restart-2.py)
- [prepare-rust-g1490-restart.py](prepare-rust-g1490-restart.py)

## Retained recovery entry points

`prepare-rust-post-transfer-reward.py` keeps its top-level path because current
transfer output and the reward runtime name it. The earlier in-place rollback
and g160 reconstruction commands now live here with the historical restarts.

## Retired lab bundles

The generation-specific bundle builders are indexed with the other
[legacy Rust lab tools](../../legacy/rust-lab/README.md).
