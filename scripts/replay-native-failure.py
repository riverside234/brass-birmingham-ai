"""Inspect or replay an L15 permitted-input failure bundle in an isolated worker.

Never opens a save or commits to Unity. Timed search is not deterministic: replay
reports the newly observed reply and the original semantic paths separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    assert path.stat().st_size <= 2*1024*1024, 'Oversized failure bundle'
    bundle = json.loads(path.read_text(encoding='utf-8-sig'))
    assert bundle['format'] == 'brass-native-failure/1.0.0'
    assert bundle['trusted_full_state_included'] is False
    request = bundle['request']
    assert request['op'] == 'search'
    assert hashlib.sha256(request['input_json'].encode()).hexdigest() == request['ticket']['root_sha256']
    permitted = json.loads(request['input_json'])
    assert set(permitted) <= {'view','recall','context','reward','ledger','sample_seed'}
    assert 'shuffle_tape' not in permitted
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    bundle = load(args.bundle)
    print(json.dumps(dict(stage=bundle['stage'], game_id=bundle['request']['ticket']['game_id'],
        revision=bundle['request']['ticket']['revision'], has_reply=bundle.get('response') is not None,
        trusted_full_state_included=False)))
    if not args.package:
        assert not args.out, '--out requires --package'
        return
    assert args.out and not args.out.exists(), 'Choose a new output file'
    package = args.package.resolve()
    api = runpy.run_path(str(ROOT/'tests/RustBackendLocal/test_local.py'))
    worker = api['Worker'](package)
    try:
        hello = worker.hello['data']
        assert hello['build']['source_sha256'] == bundle['request']['ticket']['source_sha256'], 'Use the exact failed build'
        assert hello['provider'] == 'CPUExecutionProvider'
        if bundle.get('policy') is not None:
            assert api['read'](package/'policy.json') == bundle['policy'], 'Policy differs'
        if bundle.get('model_manifest') is not None:
            assert api['read'](package/'model/manifest.json') == bundle['model_manifest'], 'Model differs'
        worker.send(bundle['request'])
        response = worker.recv(timeout=25)
        args.out.write_text(json.dumps(dict(format='brass-native-failure-replay/1.0.0',
            bundle_sha256=api['sha'](args.bundle), readiness=hello, original_reply=bundle.get('response'),
            replay_reply=response, committed=False,
            interpretation='Same permitted request and pinned runtime; a timing-dependent action difference is not itself a rules failure.'),indent=2)+'\n')
    finally:
        worker.close()


if __name__ == '__main__':
    main()
