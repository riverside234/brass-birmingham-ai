"""Capture trusted semantic states through a separate, frozen-host diagnostic.

L01/L02 inputs and runtimes remain immutable. The exporter references public
methods on that exact host; it never exposes private chance views to a policy.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'artifacts/rust-backend/20261024-g890-l01-l02'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build(out, *, flow=False, repair=False):
    reference = read(FROZEN / 'reference-runtime.json')
    runtime = Path(reference['path'])
    target = out / 'exporter'
    target.mkdir(parents=True, exist_ok=True)
    for item in reference['manifest']['files']:
        source = runtime / item['name']
        assert sha(source) == item['sha256'], 'Reference runtime drift: ' + str(source)
        dest = target / source.name
        if not dest.exists():
            shutil.copyfile(source, dest)
        assert sha(dest) == item['sha256']
    compiler = ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    source = ROOT / 'tests/RustBackendNative/StateExporter.cs'
    name='FixtureRepairExporter' if repair else ('FlowExporter' if flow else 'StateExporter')
    args = ['/nologo', '/target:exe', '/langversion:7.3', '/nostdlib+', '/deterministic+', '/optimize+',
            f'/out:"{target / (name + ".exe")}"', f'/main:{name}', f'/reference:"{target / "BrassHost.exe"}"']
    args += [f'/reference:"{p}"' for p in sorted(target.glob('*.dll'))] + [f'"{source}"']
    if flow or repair:args.append(f'"{ROOT / ("tests/RustBackendNative/"+name+".cs")}"')
    rsp = target / 'build.rsp'
    rsp.write_text('\n'.join(args), encoding='utf-8-sig')
    subprocess.run([str(compiler), '@' + str(rsp)], check=True, timeout=120)
    return target / (name + '.exe')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / 'artifacts/rust-backend')
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'states.jsonl.gz').exists(), 'Capture is immutable; choose a fresh directory'
    setup = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
    setup['verify_frozen']()
    from brass_ai.rust_backend.contracts import canonical, first_difference
    executable = build(out)
    p = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, encoding='utf-8',
                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    def export(checkpoint=None):
        p.stdin.write(json.dumps({} if checkpoint is None else {'checkpoint': checkpoint})+'\n');p.stdin.flush()
        reply=json.loads(p.stdout.readline())
        assert reply['ok'], reply.get('message')
        return reply
    try:
        initial = export()['state']
        (out / 'initial-state.json').write_text(json.dumps(initial,indent=2)+'\n')
        fixture_root = FROZEN / 'local/fixtures'
        corpus = read(fixture_root / 'corpus.json')
        counts = Counter(); entries=[]
        with gzip.open(out / 'states.jsonl.gz', 'wt', encoding='utf-8') as stream:
            for index, entry in enumerate(corpus['roots']):
                path = fixture_root / entry['path']; assert sha(path)==entry['sha256']
                with gzip.open(path,'rt',encoding='utf-8') as f: fixture=json.load(f)
                reply = export(fixture['checkpoint']); snapshot=reply['snapshot']
                for key in ('views','chance','result'): snapshot.setdefault(key,None)
                difference=first_difference(fixture['projection'],canonical(snapshot))
                assert difference is None, (entry['id'],difference)
                state=reply['state']; kind=('chance:'+state['chance']['kind']) if state['chance'] else state['context']['phase']
                counts[kind]+=1;counts['era:'+state['context']['era']]+=1
                row=dict(id=entry['id'],partition=entry['partition'],source=entry['sha256'],state=state)
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
                entries.append(dict(id=entry['id'],state_sha256=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode()).hexdigest()))
                if index % 50 == 0: print(f'captured {index+1}/{len(corpus["roots"])}',flush=True)
        receipt=dict(format='brass-native-state-capture/1.0.0',count=len(entries),coverage=dict(counts),entries=entries,
                     original_corpus_sha256=sha(fixture_root/'corpus.json'),states_sha256=sha(out/'states.jsonl.gz'),
                     exporter_source_sha256=sha(ROOT/'tests/RustBackendNative/StateExporter.cs'),exporter_sha256=sha(executable),
                     reference_runtime=reference if (reference:=read(FROZEN/'reference-runtime.json')) else None)
        (out/'capture.json').write_text(json.dumps(receipt,indent=2)+'\n')
    finally:
        p.stdin.close(); p.wait(timeout=15); p.stdout.close(); p.stderr.close()
    print('Captured complete states:',len(entries),dict(counts),flush=True)


if __name__=='__main__': main()
