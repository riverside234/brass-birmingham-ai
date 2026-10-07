"""Lab-root entry point for Step 5 self-play, search distillation and meta-RL."""
import os
from pathlib import Path
import sys


def main():
    # Pin cwd before importing multiprocessing: Linux can unlink the launch
    # directory, making later spawn fail in os.getcwd(). __file__ is absolute
    # on supported Python versions; chdir before resolve() consults the old cwd.
    # CLI artifact/config paths still resolve against the explicit --root.
    root = Path(__file__).parent.parent
    os.chdir(root)
    sys.path.insert(0, str(root.resolve() / 'src'))
    from brass_ai.step5.cli import main as run
    run()


if __name__ == '__main__':
    main()
