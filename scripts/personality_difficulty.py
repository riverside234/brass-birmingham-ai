"""Separate P5 entry point. Uses one allocated GPU; does not resume/train Step 5."""
from pathlib import Path
import os
import sys

if __name__ == '__main__':
    # Restrict visibility before importing torch/checkpoint RNG support.
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',',1)[0]
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from brass_ai.personality.difficulty.cli import main
    main()
