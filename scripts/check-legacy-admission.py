"""Exercise actual admitted identities and reject cross-pool/unqualified use."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.bank import LAB_ROOT, FrozenOpponent, admission, below, read, require, sha
from brass_ai.qualification import write_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mono')
    args = parser.parse_args()
    root = args.root.resolve()
    pool_path = root / 'training/opponents/pool.json'
    pool = read(pool_path)
    require(pool['training'] or pool['evaluation'], 'No profiles passed admission')
    require(not (set(pool['training']) & set(pool['evaluation'])), 'Pools overlap')
    executed = []
    rejected = []
    for purpose in ('training', 'evaluation'):
        other = 'evaluation' if purpose == 'training' else 'training'
        for identity in sorted(pool[purpose]):
            denied = False
            try:
                admission(root, identity, other)
            except ValueError:
                denied = True
            require(denied, 'Admitted identity crossed into the other pool')
            rejected.append(dict(identity=identity, purpose=other))
            opponent = FrozenOpponent(root, identity, purpose=purpose, mono=args.mono)
            try:
                worker = opponent.worker
                context = worker.call('reset', seed=53001)
                before = worker.call('hash')
                observation = worker.call('observe', seat=context['actor'])
                decision = worker.call('plan', observation=observation, choices=worker.call('choices'),
                                       profile=opponent.profile, fixed_work=opponent.fixed_work)
                require(before == worker.call('hash'), 'Planning changed the source game')
                selected = decision['selected']['actions']
                require(len(selected) == context['actions_remaining'], 'Incomplete opening turn')
                for action in selected:
                    context = worker.call('commit', game_id=context['game_id'],
                                          revision=context['revision'], choices=action)
                require(context['actor'] != observation['seat'], 'Committed opening did not advance')
                executed.append(dict(identity=identity, purpose=purpose, committed_actions=selected))
            finally:
                opponent.close()
    for bank in {identity.split('/')[0] for identity in pool['training']}:
        for name in read(root / 'training/opponents/bank' / bank / 'manifest.json')['payload']['profiles']:
            identity = bank + '/' + name
            if identity in pool['training']:
                continue
            denied = False
            try:
                admission(root, identity, 'training')
            except ValueError:
                denied = True
            require(denied, 'Unadmitted configuration entered training')
            rejected.append(dict(identity=identity, purpose='training'))
    report = dict(passed=True, pool_sha256=sha(pool_path), executed=executed,
                  rejected=rejected, verifier_sha256=sha(Path(__file__)))
    write_new(below(root, args.output), report)
    print(dict(passed=True, executed_profiles=len(executed), rejected_uses=len(rejected)))


if __name__ == '__main__':
    main()
