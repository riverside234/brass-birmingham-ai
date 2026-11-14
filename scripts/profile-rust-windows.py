"""Freeze and measure a Windows NN worker using the installed game protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / 'artifacts/rust-backend/20261025-l10-capture'
POLICIES = ('policy-k8.json', 'policy.json', 'policy-k32.json')
ALLOWANCES = (4000, 8000, 10000, 12000)
ROOTS = (0, 3, 26, 8)
EXTRA_ROOTS = (4, 9, 25, 42)


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    with path.open('rb') as stream:
        return dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def freeze(output):
    fixtures = []
    for index in ROOTS + EXTRA_ROOTS:
        path = CAPTURES / f'brass-{index:03d}-request.json'
        case = read(path)['cases'][0]
        fixture = case['input']
        fixture.pop('shuffle_tape', None)
        fixture['context']['profile_id'] = 'alpha-test-g890-neutral/1.0.0'
        width = len(case['replies'][0]['info']['actions'])
        fixtures.append(dict(index=index, source=pin(path), input=fixture, legal_choices=width))
    write(output, fixtures)
    return fixtures


def measure(game, fixtures, report, *, quick=False):
    worker_type = runpy.run_path(str(ROOT / 'tests/RustBackendLocal/test_local.py'))['Worker']
    package = game / 'BrassNative/alpha-g890'
    paths = [package / name for name in (*POLICIES, 'warmup.json', 'build.json', 'brass-ai-local.exe',
                                          'model/manifest.json', 'model/model.onnx', 'model/onnxruntime.dll')]
    inputs = {f['index']: f for f in fixtures}
    result = dict(host=platform.platform(), logical_cpus=os.cpu_count(), package=[pin(p) for p in paths],
                  fixtures=pin(report.parent / 'fixtures.json'), cold=[], turns=[])
    cases = zip(ROOTS, ALLOWANCES, strict=True)
    selected = list(cases)
    if quick:
        selected = selected[:1]
    for policy in POLICIES:
        began = time.perf_counter()
        worker = worker_type(package, policy=policy)
        try:
            ready = worker.hello['data']
            assert ready['measured_readiness']['warmup_passed'], ready
            result['cold'].append(dict(policy=policy, seconds=time.perf_counter() - began, readiness=ready))
            searches = selected + ([(i, 4000) for i in EXTRA_ROOTS] if not quick and policy == POLICIES[0] else [])
            for index, allowance in searches:
                fixture = inputs[index]
                inp = fixture['input']
                view = inp['view']
                ticket = dict(game_instance=f'frozen-{index}', game_id='frozen', generation=1,
                              revision=view['context']['revision'], actor=view['seat'],
                              profile=inp['context']['profile_id'], source_sha256=ready['build']['source_sha256'])
                started = time.perf_counter()
                request = worker.send(dict(op='search', input=inp, ticket=ticket, allowance_ms=allowance,
                                           remaining_ms=allowance + ready['delivery_grace_ms']))
                reply = worker.recv(timeout=20)
                seconds = time.perf_counter() - started
                data = reply['data']
                assert reply['id'] == request and reply['status'] == 'ok' and data['retired'] and data['actions'], reply
                assert data['response_ready_ns'] < data['deadline_ns'], data
                owners = data['replay']['owners']
                result['turns'].append(dict(policy=policy, fixture=index, allowance_ms=allowance,
                    legal_choices=fixture['legal_choices'], seconds=seconds, completed=data['completed'],
                    coverage=[sum(n > 0 for n in o['snapshot']['stats']['count']) if o['snapshot'] else 0 for o in owners],
                    headroom_ms=(data['deadline_ns']-data['response_ready_ns'])/1e6,
                    actions=data['actions'], provenance=data['provenance']))
                print(policy, index, data['completed'], round(seconds, 3), flush=True)
        finally:
            worker.close()
        assert all(pin(Path(p['path'])) == p for p in result['package'])
        write(report, result)


def unity(game, output):
    """Use the existing matched-turn driver in a disposable Unity player."""
    historical = ROOT / 'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures'
    cache = output / 'unity-cache'
    cache.mkdir()
    cases = [dict(name=f'live-92781-{era}-{number}', seed=92781, era=era, round=number, partition='frozen')
             for era, number in (('CanalEra', 2), ('RailEra', 7))]
    for case in cases:
        name = case['name'] + '.json'
        (cache / name).write_bytes((historical / name).read_bytes())
    inputs = output / 'unity-fixtures.json'
    write(inputs, cases)
    result = output / 'unity-matched.json'
    env = {key: value for key, value in os.environ.items() if not key.startswith('BRASS_')}
    env.update(BRASS_L14_LIVE_FIXTURES=str(inputs), BRASS_L14_LIVE_RESULT=str(result),
               BRASS_L14_LIVE_CACHE=str(cache))
    subprocess.run([str(game / 'BrassBirminghamUnity.exe'), '-batchmode', '-screen-fullscreen', '0',
                    '-screen-width', '1920', '-screen-height', '1080', '-logFile', str(output / 'unity.log'),
                    '--brass-local-ai-test', '--brass-native-matched-test'], cwd=game, env=env,
                   timeout=90, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    evidence = read(result)
    assert evidence['passed'] and evidence['complete'] and len(evidence['rows']) == len(cases), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--game', type=Path, default=ROOT)
    parser.add_argument('--fixtures', type=Path)
    parser.add_argument('--unity-player', type=Path, help='Disposable Unity player with the matched-turn test driver')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / 'fixtures.json'
    if args.fixtures is None:
        fixtures = freeze(frozen)
    else:
        frozen.write_bytes(args.fixtures.read_bytes())
        fixtures = read(frozen)
    measure(args.game.resolve(), fixtures, output / 'windows-native.json', quick=args.quick)
    if args.unity_player is not None:
        unity(args.unity_player.resolve(), output)


if __name__ == '__main__':
    main()
