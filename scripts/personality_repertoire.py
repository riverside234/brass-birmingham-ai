"""Separate lab P3 entry point; archive tooling is never loaded by the game."""
import os
from pathlib import Path
import sys

if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    os.chdir(root);sys.path.insert(0,str(root/'src'))
    from brass_ai.personality.repertoire.cli import main
    main()
