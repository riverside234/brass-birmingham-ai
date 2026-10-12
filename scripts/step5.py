"""Lab-root entry point for Step 5 self-play, search distillation and meta-RL."""
import os
from pathlib import Path
import sys


def _limit_default_gpu(argv):
    # Do this before importing Torch: checkpoint RNG capture also enumerates
    # visible GPUs. The normal one-GPU path must not initialize idle devices.
    # An explicit inference override retains the caller's logical GPU indices.
    flags=[arg.split('=',1)[0] for arg in argv if arg.startswith('--')]
    if not any('--inference-devices'.startswith(flag) for flag in flags):
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
        os.environ['CUDA_VISIBLE_DEVICES']=visible.split(',',1)[0]


def main():
    # Pin cwd before importing multiprocessing: Linux can unlink the launch
    # directory, making later spawn fail in os.getcwd(). __file__ is absolute
    # on supported Python versions; chdir before resolve() consults the old cwd.
    # CLI artifact/config paths still resolve against the explicit --root.
    root = Path(__file__).parent.parent
    os.chdir(root)
    sys.path.insert(0, str(root.resolve() / 'src'))
    _limit_default_gpu(sys.argv[1:])
    from brass_ai.step5.cli import main as run
    run()


if __name__ == '__main__':
    main()
