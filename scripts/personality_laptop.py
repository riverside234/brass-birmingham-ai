"""Separate P6 entry point; never changes Step 5 checkpoints/settings."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.personality.laptop.cli import main
if __name__=='__main__':main()
