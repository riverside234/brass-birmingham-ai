"""Execute workflow metadata and mocked Release uploads; needs PyYAML + Bash."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    workflow = yaml.load((ROOT / '.github/workflows/step5-transfer-release.yml').read_text(), Loader=yaml.BaseLoader)
    steps = workflow['jobs']['package-and-release']['steps']
    bash = os.environ.get('TRANSFER_TEST_BASH') or shutil.which('bash')
    assert bash, 'Install Bash or set TRANSFER_TEST_BASH'
    for step in steps:
        subprocess.run([bash, '-n'], input=step['run'], text=True, check=True)
        for source in re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", step['run'], re.S):
            compile(source, step['name'], 'exec')
    export = next(s for s in steps if s['name'] == 'Export and validate the stopped run')['run']
    upload = next(s for s in steps if s['name'] == 'Upload all parts, then publish the Release')['run']
    assert '--dependencies repo' in export
    metadata = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", export, re.S)[0]

    with tempfile.TemporaryDirectory(prefix='workflow-smoke-', dir=ROOT / 'artifacts') as tmp:
        area = Path(tmp)
        bundle = area / 'selfplay-migration.tar.zst'
        parts = []
        for number in (1, 2):
            path = area / f'{bundle.name}.part{number:04d}'
            path.write_bytes(b'fixture-upload-data')
            parts.append(dict(name=path.name, bytes=path.stat().st_size))
        index = area / f'{bundle.name}.parts.json'
        index.write_text(json.dumps(dict(parts=parts)))
        result = dict(bundle=str(index), git_commit='fixture-commit', run='artifacts/step5/selfplay',
                      source_root='/fixture/project', dependencies='repo', runtime_files=904,
                      bundled_runtime_files=904, bundled_runtime_bytes=254220007,
                      iteration=2, games_done=640, zstd_level=3, compression_threads=2,
                      compressed_bytes=sum(p['bytes'] for p in parts), parts=len(parts))
        env = dict(BUNDLE_DIR=area.as_posix(), BUNDLE=bundle.as_posix(),
                   RELEASE_TAG='fixture-transfer', GITHUB_SHA='fixture-commit', GH_REPO='fixture/test',
                   SOURCE_ROOT=ROOT.as_posix())
        cases = [{}, {'git_commit': 'wrong-commit'}, {'dependencies': 'external'},
                 {'runtime_files': 0}, {'bundled_runtime_files': 903}]
        for changed in cases:
            (area / 'export-result.json').write_text(json.dumps(dict(result, **changed)))
            try:
                with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(metadata, 'workflow metadata', 'exec'), {})
            except SystemExit:
                assert changed, 'Valid metadata was rejected'
            else:
                assert not changed, f'Invalid metadata was accepted: {changed}'
        notes = (area / 'release-notes.md').read_text()
        assert '904' in notes and 'no source-machine access' in notes

        stub = r'''
        gh() {
          printf '%s\n' "$*" >> "$MOCK_GH_LOG"
          if [[ "$1 $2" == 'release upload' && "$4" == *"$MOCK_FAIL_SUFFIX" && "$MOCK_FAIL_SUFFIX" != 'none' ]]; then
            return 17
          fi
          if [[ "$1 $2" == 'release view' ]]; then
            printf 'https://example.invalid/fixture-release\n'
          fi
        }
        '''
        for visibility in ('public', 'private'):
            for failure in ('none', '.part0002', '.parts.json'):
                case = visibility + failure
                log = area / f'calls-{case}.txt'
                summary = area / f'summary-{case}.md'
                case_env = dict(os.environ, **env, MOCK_GH_LOG=log.as_posix(), MOCK_FAIL_SUFFIX=failure,
                                GITHUB_REPOSITORY_VISIBILITY=visibility, GITHUB_STEP_SUMMARY=summary.as_posix())
                run = subprocess.run([bash, '-s'], input=stub + upload, text=True,
                                     env=case_env, capture_output=True)
                calls = log.read_text().splitlines()
                assert calls[0].startswith('release create fixture-transfer --draft'), calls
                if failure == 'none':
                    assert run.returncode == 0, run.stderr
                    index_call = next(i for i, c in enumerate(calls) if c.startswith('release upload') and c.endswith('.parts.json'))
                    publish_call = next(i for i, c in enumerate(calls) if c.startswith('release edit'))
                    assert index_call == 3 and publish_call > index_call, calls
                    assert 'https://example.invalid/fixture-release' in summary.read_text()
                else:
                    assert run.returncode == 17, (run.returncode, run.stderr)
                    assert not any(c.startswith('release edit') for c in calls), calls
                    assert not summary.exists(), calls
                    if failure == '.part0002':
                        assert not any(c.endswith('.parts.json') for c in calls), calls
                print(f'Mock Release upload: {visibility}, {failure}: passed')

    print('Workflow syntax, metadata and uploads passed; no GitHub request was made.')


if __name__ == '__main__':
    main()
