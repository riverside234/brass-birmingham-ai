"""Capture L06 permitted roots and recorded-shuffle worlds using pinned Python/C#."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import random
import runpy
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
L05 = runpy.run_path(str(ROOT / 'scripts/qualify-rust-rules.py'))
BASE = L05['BASE']
read, write, receipt, sha = (BASE[k] for k in ('read', 'write', 'receipt', 'sha'))
RULES = ROOT / 'artifacts/rust-backend/20261025-l05-capture-v2'
FIXTURES = BASE['FROZEN'] / 'local/fixtures'


class ShuffleRecorder(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.draws = []

    def _randbelow(self, n):
        result = super()._randbelow(n)
        self.draws.append(dict(bound=n, index=result))
        return result


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        yield from map(json.loads, stream)


def capture(out):
    assert out.is_relative_to(ROOT / 'artifacts/rust-backend') and not out.exists()
    out.mkdir(parents=True)
    reference = L05['prepare_reference'](out)
    before = BASE['verify_frozen'](reference_source_root=reference)
    write(out / 'preservation-before.json', before)
    l05 = L05['verify_capture'](RULES)
    executable = RULES / 'exporter/RulesExporter.exe'
    assert sha(executable) == l05['exporter_sha256']
    runtime = read(BASE['FROZEN'] / 'reference-runtime.json')
    runtime_checks = []
    for item in runtime['files']:
        # The exporter runs beside exact copies of the frozen assemblies.
        original = ROOT / item['path']
        assert sha(original) == item['sha256']
        candidate = executable.parent / original.name
        assert sha(candidate) == item['sha256']
        runtime_checks.append(receipt(candidate))
    spec = importlib.util.spec_from_file_location('frozen_beliefs', reference / 'src/brass_ai/beliefs.py')
    beliefs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(beliefs)
    composition = read(reference / 'configs/contracts/v0.2.0/card-composition.json')
    old = runpy.run_path(str(ROOT / 'scripts/capture-rust-rules.py'))
    oracle = old['Oracle'](executable)
    corpus = read(FIXTURES / 'corpus.json')
    corrected = {x['original_id']: x for x in read(old['REGRESSIONS'])['cases']}
    root_states = {r['id']: r for r in rows(RULES / 'roots.jsonl.gz')}
    groups = set()
    coverage = Counter()
    samples = roots = suffix_steps = 0
    files = [FIXTURES / 'corpus.json', old['REGRESSIONS']]
    try:
        with gzip.open(out / 'roots.jsonl.gz', 'wt', encoding='utf-8') as roots_out, gzip.open(out / 'samples.jsonl.gz', 'wt', encoding='utf-8') as samples_out:
            for item in corpus['roots']:
                path = FIXTURES / item['path']
                assert sha(path) == item['sha256']
                files.append(path)
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    fixture = json.load(stream)
                projection = corrected[item['id']]['snapshot'] if item['id'] in corrected else fixture['projection']
                state = root_states[item['id']]['expected']['state']
                views = projection.get('views')
                roots_out.write(json.dumps(dict(id=item['id'], partition=item['partition'], state=state,
                    views=views, recall=fixture['recall']), separators=(',', ':')) + '\n')
                roots += 1
                if views is None or state['context']['terminal']:
                    continue
                for seat, view in enumerate(views):
                    rng = ShuffleRecorder(890000 + samples)
                    with patch.object(beliefs.random, 'Random', return_value=rng):
                        setup = beliefs.sample_world_setup(view, composition, 890000 + samples)
                    assert [d['bound'] for d in rng.draws] == list(range(len(rng.draws)+1, 1, -1))
                    expected = oracle.call(seed=1, setup=setup, projection=True)
                    snapshot = expected.pop('snapshot')
                    expected['views'] = snapshot['views']
                    assert expected['views'][seat] == view, (item['id'], seat, 'reference root changed')
                    key = (view['context']['era'], view['context']['phase'], seat)
                    suffix = []
                    # One trajectory per era/phase/seat group, chosen by frozen
                    # corpus order. This exercises reconstruction beyond its root.
                    if key not in groups:
                        groups.add(key)
                        current = expected
                        for _ in range(12):
                            if current['state']['context']['terminal']:
                                break
                            if current['chance']:
                                command = dict(card=current['chance']['outcomes'][0]['card'])
                            else:
                                choices = current['choices']
                                choice = next((c for c in choices if c['kind'] in ('finish', 'pass', 'bonus_skip')), choices[0])
                                command = dict(choice=choice['id'])
                            after = oracle.call(**command, projection=True)
                            snapshot = after.pop('snapshot')
                            after['views'] = snapshot.get('views')
                            suffix.append(dict(command=command, expected=after))
                            current = after
                            suffix_steps += 1
                    samples_out.write(json.dumps(dict(root=item['id'], partition=item['partition'], seat=seat,
                        view=view, recall=fixture['recall'][seat], draws=rng.draws, expected=expected,
                        suffix=suffix), separators=(',', ':')) + '\n')
                    samples += 1
                    coverage[':'.join(map(str, key))] += 1
                    if samples % 100 == 0:
                        print('sampled worlds', samples, flush=True)
        after = BASE['verify_frozen'](reference_source_root=reference)
        assert L05['preserved_identity'](before) == L05['preserved_identity'](after)
        write(out / 'preservation-after.json', after)
        write(out / 'capture.json', dict(format='brass-native-privacy-capture/1.0.0',
            completed_utc=datetime.now(timezone.utc).isoformat(), roots=roots, sampled_worlds=samples,
            trajectories=len(groups), transitions=suffix_steps, coverage=dict(coverage),
            reference_sources=[receipt(reference / p) for p in ('src/brass_ai/beliefs.py', 'src/brass_ai/searchlab/spiel.py')],
            python_reference=receipt(out / 'python-reference.json'), exporter=receipt(executable),
            runtime=runtime_checks, inputs=[receipt(p) for p in files], rules_capture=receipt(RULES / 'capture.json'),
            script=receipt(Path(__file__)), files={n:sha(out/n) for n in ('roots.jsonl.gz','samples.jsonl.gz')},
            sampling='Exact frozen sample_world_setup; recorded Python _randbelow draws; same supplied setup imported by frozen C#',
            preservation=after))
        print('L06 capture passed', roots, samples, suffix_steps, flush=True)
    finally:
        oracle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    capture(parser.parse_args().out.resolve())
