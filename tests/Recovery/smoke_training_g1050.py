"""Bounded two-age PPO/PPG restart smoke using loss8/KL16 and the .5 beta floor."""
import smoke_training_g850 as smoke
from test_training_g1050 import smoke_config

smoke.config=smoke_config

if __name__=='__main__':smoke.main()
