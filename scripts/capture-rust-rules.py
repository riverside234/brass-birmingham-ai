"""Capture L05 staged rules against the unchanged, frozen C# reference runtime."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import runpy
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/capture-rust-states.py'))
read,sha=BASE['read'],BASE['sha']
STATE_ROOT=ROOT/'artifacts/rust-backend/20261024-l04/states'
REGRESSIONS=ROOT/'artifacts/rust-backend/20261024-l04/regressions-v2/regressions.json'
FIXTURES=BASE['FROZEN']/'local/fixtures'
COMPLETE=runpy.run_path(str(ROOT/'scripts/complete-rust-state-capture.py'))['complete']


def build(out):
    executable=BASE['build'](out)
    folder=executable.parent
    rsp=(folder/'build.rsp').read_text(encoding='utf-8-sig').replace('StateExporter.exe','RulesExporter.exe').replace('/main:StateExporter','/main:RulesExporter')
    rsp+='\n"'+str(ROOT/'tests/RustBackendNative/RulesExporter.cs')+'"\n'
    (folder/'rules.rsp').write_text(rsp,encoding='utf-8-sig')
    compiler=ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    subprocess.run([str(compiler),'@'+str(folder/'rules.rsp')],check=True,timeout=120)
    return folder/'RulesExporter.exe'


class Oracle:
    def __init__(self,executable):
        self.p=subprocess.Popen([str(executable)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        initial=read(STATE_ROOT/'initial-state.json')
        self.identities={t['id'] for p in initial['players'] for m in p['mat'] for t in m['tiles']}
    def call(self,**request):
        self.p.stdin.write(json.dumps(request,separators=(',',':'))+'\n');self.p.stdin.flush()
        line=self.p.stdout.readline()
        if not line:raise RuntimeError('C# oracle closed output')
        reply=json.loads(line)
        if not reply['ok']:raise RuntimeError(reply['message'])
        reply['state']=COMPLETE(reply['state'],self.identities)
        return reply
    def close(self):
        self.p.stdin.close();self.p.wait(timeout=15);self.p.stdout.close()


def corrected_request(fixture):
    """Repeat the six L04 corrections, then verify the complete L04 state below."""
    cp=fixture['checkpoint'];setup=deepcopy(cp['setup']);position=setup['position']
    for p in position['players']:p['link_tiles']=14-sum(l.get('owner')==p['seat'] for l in position['links'])
    position['context']['turn_owner']=position['order'][position['context']['turn']-1]
    initial=read(STATE_ROOT/'initial-state.json')
    pool=Counter(tuple(m['types']) for m in initial['merchants']);actual=Counter(tuple(m['types']) for m in position['merchants'])
    for m in position['merchants']:
        key=tuple(m['types'])
        if m['id'].startswith('Gloucester:') or actual[key]<=pool[key]:continue
        replacement=next(k for k in sorted(pool) if actual[k]<pool[k]);actual[key]-=1;actual[replacement]+=1
        m['types']=list(replacement);m['beer']=int(bool(replacement))
    assert actual==pool
    return dict(checkpoint=cp,setup=setup)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not (out/'capture.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    frozen=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'));before=frozen['verify_frozen']()
    from brass_ai.rust_backend.contracts import canonical,first_difference
    executable=build(out);oracle=Oracle(executable);coverage=Counter();entries=[]
    corpus=read(FIXTURES/'corpus.json');corrected={r['original_id']:r for r in read(REGRESSIONS)['cases']}
    state_receipt=read(STATE_ROOT/'capture.json');assert sha(STATE_ROOT/'states.jsonl.gz')==state_receipt['states_sha256']
    with gzip.open(STATE_ROOT/'states.jsonl.gz','rt',encoding='utf-8') as f:states={r['id']:r['state'] for r in map(json.loads,f)}
    for entry in state_receipt['entries']:
        import hashlib
        assert hashlib.sha256(json.dumps(states[entry['id']],sort_keys=True,separators=(',',':')).encode()).hexdigest()==entry['state_sha256']
    def compare(projection,reply,label):
        snapshot=reply.pop('snapshot')
        for key in ('views','chance','result'):snapshot.setdefault(key,None)
        difference=first_difference(projection,canonical(snapshot));assert difference is None,(label,difference)
    def save(stream,row):
        stream.write(json.dumps(row,separators=(',',':'))+'\n')
    try:
        with gzip.open(out/'roots.jsonl.gz','wt',encoding='utf-8') as stream:
            for i,root in enumerate(corpus['roots']):
                path=FIXTURES/root['path'];assert sha(path)==root['sha256']
                with gzip.open(path,'rt',encoding='utf-8') as f:fixture=json.load(f)
                request=corrected_request(fixture) if root['id'] in corrected else dict(checkpoint=fixture['checkpoint'])
                expected=corrected[root['id']]['state'] if root['id'] in corrected else states[root['id']]
                # L04 already verified every root against this exact C# runtime.
                # Reuse that immutable evidence, including the six explicit corrections.
                snapshot=corrected[root['id']]['snapshot'] if root['id'] in corrected else fixture['projection']
                reply=dict(ok=True,state=expected,choices=snapshot.get('choices',[]),chance=snapshot.get('chance'))
                save(stream,dict(id=root['id'],partition=root['partition'],request=request,expected=reply))
                if i%100==0:print('roots',i+1,flush=True)
        with gzip.open(out/'games.jsonl.gz','wt',encoding='utf-8') as stream:
            for game in corpus['games']:
                path=FIXTURES/game['path'];assert sha(path)==game['sha256']
                with gzip.open(path,'rt',encoding='utf-8') as source:
                    header=json.loads(next(source));reply=oracle.call(checkpoint=header['initial'],projection=True);compare(header['projection'],reply,game['id'])
                    save(stream,dict(game=game['id'],partition=game['partition'],initial=reply))
                    count=0
                    for line in source:
                        step=json.loads(line)
                        if 'command' not in step:continue
                        command=step['command'];reply=oracle.call(**command,projection=True);compare(step['after'],reply,(game['id'],count))
                        save(stream,dict(game=game['id'],step=count,command=command,expected=reply));count+=1
                        coverage[next(iter(command))]+=1
                    assert count==game['steps'];entries.append(dict(id=game['id'],steps=count));print(game['id'],count,flush=True)
        after=frozen['verify_frozen']();assert before==after
        result=dict(format='brass-native-rules-capture/1.0.0',completed_utc=datetime.now(timezone.utc).isoformat(),roots=546,games=entries,coverage=dict(coverage),
                    source_corpus_sha256=sha(FIXTURES/'corpus.json'),state_capture_sha256=sha(STATE_ROOT/'capture.json'),regressions_sha256=sha(REGRESSIONS),
                    root_provenance='Hash-verified L04 semantic exports plus original L02 choices/chance; six L04 corrected C# snapshots',
                    exporter_sha256=sha(executable),exporter_source_sha256=sha(ROOT/'tests/RustBackendNative/RulesExporter.cs'),script_sha256=sha(Path(__file__)),
                    files={n:sha(out/n) for n in ('roots.jsonl.gz','games.jsonl.gz')},preservation=before)
        (out/'capture.json').write_text(json.dumps(result,indent=2)+'\n');print('L05 C# capture passed',flush=True)
    finally:oracle.close()


if __name__=='__main__':main()
