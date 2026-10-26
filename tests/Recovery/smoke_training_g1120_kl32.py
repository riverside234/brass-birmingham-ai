"""Bounded three-action T1.3 PPO and loss16/KL32 PPG restart smoke."""
import smoke_training_g850 as smoke
from test_training_g1120 import collect_revised
from test_training_g1120_kl32 import smoke_config

smoke.config=smoke_config
smoke.collect=collect_revised

if __name__=='__main__':smoke.main()
