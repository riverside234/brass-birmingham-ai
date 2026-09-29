"""One real held-out setup in all four seats; no strength qualification."""
import argparse
from pathlib import Path
import sys
import uuid

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / 'src'))

import torch
from brass_ai.bank import read, sha
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.contracts import LAB_ROOT, project_path, require
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.step5.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--mono')
    args = parser.parse_args()
    require(args.root.is_absolute(), 'Use an absolute project root')
    require(args.device != 'cuda' or torch.cuda.is_available(), 'CUDA unavailable')
    root = args.root.resolve()
    torch.set_num_threads(2)
    directory = project_path(root, 'artifacts/step5/evaluation-checks/' + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    paths = []
    for seed in (841, 842):
        torch.manual_seed(seed)
        model = FourSeatNet(ModelConfig.preset('small'))
        path = directory / (str(seed) + '.pt')
        save_checkpoint(path, model, training=dict(mode='evaluation-smoke', strength='unqualified'))
        paths.append(path)
    before = [sha(p) for p in paths]
    config = read(root / 'configs/step5/evaluation.json')
    config.update(setups=1, workers=4, threads=2, include_legacy=False)
    report = evaluate(root, directory / 'result', paths[0], paths[1:], config, device=args.device, mono=args.mono)
    result = report['results']['recent-neural']
    require(result['games'] == 4 and result['seat_counts'] == {str(s): 1 for s in range(4)}, 'Incorrect evaluation rotations')
    require(not report['promotion_eligible'] and report['failed_games'] == 0, 'Smoke cannot qualify playing strength')
    require([sha(p) for p in paths] == before, 'Evaluation changed checkpoint files')
    print('Passed four-seat evaluation: ' + str(directory / 'result/report.json'))


if __name__ == '__main__':
    main()
