"""Reuse the bounded ownership/branch PPO+PPG restart smoke with the g950 recipe."""
import smoke_training_g850 as smoke
from test_training_g950 import smoke_config

smoke.config=smoke_config

if __name__=='__main__':smoke.main()
