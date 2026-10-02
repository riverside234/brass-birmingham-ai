"""Exercise the privately transferred original Hard/v3 runtime on the lab."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.bank import load_bank,LAB_ROOT, FrozenOpponent, below, read, require

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=LAB_ROOT)
parser.add_argument('--mono')
parser.add_argument('--seed', type=int, default=856001)
parser.add_argument('--v3-seat', type=int, choices=[-1, 0, 1, 2, 3], default=-1,
                    help='Default -1 uses four original guarded Hard seats; 0..3 inserts one v3 seat')
parser.add_argument('--output', default='artifacts/step5/transferred-legacy-check.json')
args = parser.parse_args()
root = args.root.resolve()
bank = read(root / 'training/opponents/current.json')['bank']
opponent = FrozenOpponent(root, bank + '/' + next(n for n in load_bank(root,bank)['payload']['profiles'] if n.startswith('balanced-v3')), purpose='diagnostic', mono=args.mono)
try:
    match = opponent.worker.call('run', seed=args.seed, player_count=4,
        candidate_seat=args.v3_seat, profile=opponent.profile, fixed_work=opponent.fixed_work,
        record_selectors=True)
finally:
    opponent.close()
require(match['completed'] and not match.get('failure') and match['turns'] == 64 and match['actions'] == 124,
        'Transferred stock/v3 runtime did not complete a four-player game')
path = below(root, args.output); path.parent.mkdir(parents=True, exist_ok=True)
require(not path.exists(), 'Preserve the prior runtime check; select a new output')
path.write_text(json.dumps(match, separators=(',', ':')) + '\n', encoding='utf-8')
print(json.dumps(dict(passed=True, bank=bank, stock='stock-hard-guarded-1.1.0',
                     v3_seat=args.v3_seat, actions=match['actions'], turns=match['turns'], result=match['result']), indent=2))
