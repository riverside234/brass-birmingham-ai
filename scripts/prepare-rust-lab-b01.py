#!/usr/bin/env python3
"""Package pinned B01 inputs for transfer. Reads receipts; runs no qualification."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from brass_ai.learning.contracts import read_json, require
from brass_ai.rust_backend.lab_contract import ARCHIVE, METHOD, POLICIES, admit, sha, verify_inputs


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def prepare(output):
    base = ROOT / 'artifacts/search-v1-default'
    package = base / 'candidate/BrassNative/alpha-g890'
    references = {
        'source.zip': package / 'native-source.zip',
        'candidate.json': base / 'candidate/build-manifest.json',
        'deployment.json': base / 'deployment.json',
        'installed-smoke.json': base / 'installed-k16/verification.json',
        'local-qualification.json': base / 'candidate/qualification.json',
        'local-native-build.json': package / 'build.json',
        'local-acceptance.json': ROOT / 'artifacts/rust-backend/20261026-local-release/release-acceptance.json',
    }
    references.update({name: package / name for name in POLICIES})
    require(sha(references['source.zip']) == ARCHIVE, 'Selected installed search_v1 source archive differs')
    profile = read_json(ROOT / 'configs/search/lab-search-v1.json')
    profile_id = admit(profile)
    captures = ROOT / 'artifacts/rust-backend'
    rules_file = captures / '20261025-l05-capture-v2/roots.jsonl.gz'
    views_file = captures / '20261025-l06-capture/roots.jsonl.gz'
    branches_file = captures / '20261025-l05-branches/branches.jsonl.gz'
    rules = {r['id']: r for r in rows(rules_file)}
    views = {r['id']: r for r in rows(views_file)}
    candidates = [r for r in rules.values() if r['id'] in views and r['expected']['choices']
                  and r['expected']['state']['context']['actor'] >= 0]
    selected = {}
    # Bounded coverage: each stage/era, all acting seats, and the widest root.
    for row in candidates:
        c = row['expected']['state']['context']
        selected.setdefault(('phase', c['era'], c['phase']), row)
        selected.setdefault(('actor', c['actor']), row)
    selected[('widest',)] = max(candidates, key=lambda r: len(r['expected']['choices']))
    unique = {r['id']: r for r in selected.values()}
    require(0 < len(unique) <= 24, 'B01 root coverage exceeds its bounded smoke cap')
    roots = []
    for key, row in sorted(unique.items()):
        state = row['expected']['state']; actor = state['context']['actor']
        require(state == views[key]['state'], 'Frozen rules/observation fixture mismatch')
        roots.append(dict(id=key, expected=row['expected'], view=views[key]['views'][actor]))
    transitions = {}
    for row in rows(branches_file):
        c = row['before']['state']['context']
        transitions.setdefault((c['era'], c['phase']), row)
    require(0 < len(transitions) <= 24, 'B01 transition coverage exceeds its bounded smoke cap')
    output = Path(output).resolve()
    require(output.is_relative_to(ROOT / 'distribution'), 'Qualification bundles belong under distribution/')
    require(not output.exists(), 'B01 input bundle already exists; choose a new file')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='b01-package-') as temp:
        temp = Path(temp)
        for name, path in references.items():
            (temp / name).write_bytes(path.read_bytes())
        write(temp / 'fixtures.json', dict(contract='brass-lab-smoke-fixtures/1.0.0', roots=roots,
            transitions=list(transitions.values()), selection='First per era/stage/actor plus widest root; no games',
            captures=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
                      for p in (rules_file, views_file, branches_file)]))
        # Preserve the native vendor's attribution omitted by the study's source-only ZIP.
        vendor = ROOT / 'native/vendor/brass-upstream-rules'
        for name in ('LICENSE', 'UPSTREAM.json', 'upstream-local.patch'):
            (temp / ('vendor-' + name)).write_bytes((vendor / name).read_bytes())
        write(temp / 'inputs.json', dict(contract='brass-lab-b01-inputs/2.0.0', method=METHOD, profile_id=profile_id,
              files={p.name: sha(p) for p in sorted(temp.iterdir()) if p.is_file()}))
        verify_inputs(temp, profile)
        with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(temp.iterdir()):
                archive.write(p, p.name)
    print(json.dumps(dict(bundle=str(output), sha256=sha(output), roots=len(roots),
                          transitions=len(transitions), qualification_run=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    prepare(parser.parse_args().out)
