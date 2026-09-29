"""Lab-root entry point for Step 5 self-play, search distillation and meta-RL."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.step5.cli import main

if __name__ == '__main__':
    main()
