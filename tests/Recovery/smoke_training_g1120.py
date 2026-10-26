"""Bounded T1.3 three-decision PPO/loss16 PPG smoke and exact restart."""
import smoke_training_g850 as smoke
from test_training_g1120 import smoke_config,collect_revised

smoke.config=smoke_config
smoke.collect=collect_revised

if __name__=='__main__':smoke.main()
