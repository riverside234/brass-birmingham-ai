"""Profile an export using its saved parity inputs, without rebuilding a model."""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brass_ai.learning.onnx_tools import cpu_options
from brass_ai.learning.onnx_profile import profile_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, help='Saved .parity.npz; defaults to the model sidecar')
    parser.add_argument('--threads', type=int, choices=(1, 2), default=1)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=21)
    args = parser.parse_args()
    session = ort.InferenceSession(str(args.model), cpu_options(args.threads), providers=['CPUExecutionProvider'])
    names = [value.name for value in session.get_inputs()]
    del session
    with np.load(args.inputs or args.model.with_suffix('.parity.npz'), allow_pickle=False) as data:
        indices = sorted({int(key.split(':')[0]) for key in data.files if ':input:' in key})
        feeds = [(f"batch-{data[f'{i}:input:states'].shape[0]}-candidates-{data[f'{i}:input:actions'].shape[1]}",
                  {name: data[f'{i}:input:{name}'].copy() for name in names}) for i in indices]
    report = profile_model(args.model, feeds, args.output, threads=args.threads,
                           warmups=args.warmups, repeats=args.repeats)
    print(f"Profiled {len(report['cases'])} shapes: {args.output / 'PROFILE.md'}")


if __name__ == '__main__':
    main()
