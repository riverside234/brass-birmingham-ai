"""Preserve rejected L02 authored roots and capture independently corrected C# variants."""
import argparse
from copy import deepcopy
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT/'artifacts/rust-backend')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'regressions.json').exists()
    capture=args.capture.resolve();receipt=json.loads((capture/'capture.json').read_text())
    assert hashlib.sha256((capture/'states.jsonl.gz').read_bytes()).hexdigest()==receipt['states_sha256']
    with gzip.open(capture/'states.jsonl.gz','rt',encoding='utf-8') as f:rows=[json.loads(l) for l in f]
    exporter=runpy.run_path(str(ROOT/'scripts/capture-rust-states.py'));complete=runpy.run_path(str(ROOT/'scripts/complete-rust-state-capture.py'))['complete']
    initial=json.loads((capture/'initial-state.json').read_text());identities={t['id'] for p in initial['players'] for m in p['mat'] for t in m['tiles']}
    merchant_pool=Counter(tuple(m['types']) for m in initial['merchants'])
    def repair_merchants(merchants):
        actual=Counter(tuple(m['types']) for m in merchants)
        if actual==merchant_pool:return False
        # Preserve the two Cotton destinations which define this authored edge.
        for m in merchants:
            key=tuple(m['types'])
            if m['id'].startswith('Gloucester:') or actual[key]<=merchant_pool[key]:continue
            replacement=next(k for k in sorted(merchant_pool) if actual[k]<merchant_pool[k])
            actual[key]-=1;actual[replacement]+=1;m['types']=list(replacement);m['beer']=1 if replacement else 0
        assert actual==merchant_pool
        return True
    corpus_root=exporter['FROZEN']/'local/fixtures';corpus=json.loads((corpus_root/'corpus.json').read_text());sources={r['id']:r for r in corpus['roots']}
    executable=exporter['build'](out,repair=True)
    p=subprocess.Popen([str(executable)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    regressions=[]
    try:
        for row in rows:
            original=row['state'];expected=deepcopy(original);issues=[]
            placed=[sum(l.get('owner')==seat for l in original['links']) for seat in range(4)]
            for seat in range(4):
                if original['players'][seat]['link_tiles']+placed[seat]!=14:
                    issues.append('link tile conservation');expected['players'][seat]['link_tiles']=14-placed[seat]
            context=expected['context'];turn_owner=expected['order'][context['turn']-1]
            if context['turn_owner']!=turn_owner:issues.append('turn owner disagrees with order');context['turn_owner']=turn_owner
            if repair_merchants(expected['merchants']):issues.append('merchant tile conservation')
            if not issues:continue
            assert row['id'].startswith('edge-')
            source=sources[row['id']];path=corpus_root/source['path'];assert exporter['sha'](path)==source['sha256']
            with gzip.open(path,'rt',encoding='utf-8') as stream:fixture=json.load(stream)
            checkpoint=fixture['checkpoint'];setup=deepcopy(checkpoint['setup']);position=setup['position']
            # Correct the authored setup, then let C# replay the original choices.
            for player in position['players']:
                owner=player['seat'];player['link_tiles']=14-sum(l.get('owner')==owner for l in position['links'])
            position['context']['turn_owner']=position['order'][position['context']['turn']-1]
            repair_merchants(position['merchants'])
            p.stdin.write(json.dumps(dict(checkpoint=checkpoint,setup=setup))+'\n');p.stdin.flush()
            reply=json.loads(p.stdout.readline());assert reply['ok'],reply.get('message')
            state=complete(reply['state'],identities)
            assert state==expected, 'Correction changed unexpected semantic fields: '+row['id']
            regressions.append(dict(id='corrected-'+row['id'],original_id=row['id'],original_source_sha256=source['sha256'],rejection_reasons=sorted(set(issues)),state=state,snapshot=reply['snapshot']))
    finally:
        p.stdin.close();p.wait(timeout=15);p.stdout.close()
    assert len(regressions)==6
    result=dict(format='brass-native-state-regressions/1.0.0',source_capture_sha256=exporter['sha'](capture/'capture.json'),
                exporter_sha256=exporter['sha'](executable),exporter_source_sha256=exporter['sha'](ROOT/'tests/RustBackendNative/FixtureRepairExporter.cs'),
                script_sha256=exporter['sha'](Path(__file__)),cases=regressions)
    (out/'regressions.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Captured six corrected authored roots; originals retained for rejection tests.')


if __name__=='__main__':main()
