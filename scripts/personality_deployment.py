"""Separate lab P4 entry point; one shared model, no game patch or Step 5 changes."""
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    root=Path(__file__).resolve().parent.parent
    os.chdir(root);sys.path.insert(0,str(root/'src'))
    from brass_ai.personality.deployment.cli import main
    main()
