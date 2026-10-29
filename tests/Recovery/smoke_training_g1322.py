"""Small PPO/PPG recipe transition, fresh replay and exact restart check."""
from brass_ai.recovery.training_revision_g1322 import configure
from test_training_g1322 import smoke_config,collect_revised
import smoke_training_g1320 as migration_smoke

# Exercise the existing complete migration/cleanup/restart harness with the
# current reward, actual T2/epsilon-.08 behavior and per-group entropy losses.
migration_smoke.smoke_config=smoke_config
migration_smoke.configure=configure
migration_smoke.common.collect=collect_revised


if __name__=='__main__':migration_smoke.main()
