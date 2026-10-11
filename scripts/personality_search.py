"""Separate lab P2 entry point; active Step 5 and the installed game are unchanged."""
import os
from pathlib import Path
import sys

if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    os.chdir(root);sys.path.insert(0,str(root/'src'))
    from brass_ai.personality.search.cli import main
    main()
