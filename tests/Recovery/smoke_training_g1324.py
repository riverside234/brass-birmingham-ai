"""Three small authored PPO/PPG iterations and exact restart with branch v6."""
from brass_ai.recovery.training_revision_g1324 import configure
from test_training_g1324 import smoke_config
from test_training_g1322 import collect_revised
import smoke_training_g850 as harness


def config():
    model,opt,c=smoke_config()
    return model,opt,configure(c)


harness.config=config
harness.collect=collect_revised

if __name__=='__main__':harness.main()
