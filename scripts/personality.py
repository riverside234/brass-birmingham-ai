"""Lab entry point for the separate post-Step-5 personality P0/P1 pilot."""
import os
from pathlib import Path
import sys


def main():
    root=Path(__file__).parent.parent
    os.chdir(root);sys.path.insert(0,str(root.resolve()/'src'))
    from brass_ai.personality.cli import main as run
    run()


if __name__=='__main__':main()
