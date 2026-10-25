"""Add an explicit off-board partition to the verified raw C# snapshots.

C# Board.RemovedIndustries excludes developed tiles and tiles pruned from mats
at era changes. These objects cannot re-enter play. Keep its removed list intact
and materialize the remaining stable IDs separately; never modify frozen roots.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete(state, identities):
    live = [t['id'] for p in state['players'] for m in p['mat'] for t in m['tiles']]
    live += [s['tile']['id'] for s in state['slots'] if s.get('tile')]
    used = live + state['removed']
    assert len(used) == len(set(used)) and set(used) <= identities
    state['retired'] = sorted(identities - set(used))
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not (out/'capture.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((args.raw/'capture.json').read_text())
    assert sha(args.raw/'states.jsonl.gz')==receipt['states_sha256']
    initial=json.loads((args.raw/'initial-state.json').read_text())
    identities={t['id'] for p in initial['players'] for m in p['mat'] for t in m['tiles']}
    assert len(identities)==180
    initial['settlement_debts']=[0]*4
    (out/'initial-state.json').write_text(json.dumps(complete(initial,identities),indent=2)+'\n')
    exporter=runpy.run_path(str(ROOT/'scripts/capture-rust-states.py'))
    executable=exporter['build'](out,flow=True)
    corpus_root=exporter['FROZEN']/'local/fixtures'
    corpus=json.loads((corpus_root/'corpus.json').read_text())
    frozen={r['id']:r for r in corpus['roots']}
    worker=subprocess.Popen([str(executable)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                            text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    def export(identity,picks=None):
        source=frozen[identity];path=corpus_root/source['path'];assert sha(path)==source['sha256']
        with gzip.open(path,'rt',encoding='utf-8') as stream:fixture=json.load(stream)
        worker.stdin.write(json.dumps(dict(checkpoint=fixture['checkpoint'],picks=picks or []))+'\n');worker.stdin.flush()
        reply=json.loads(worker.stdout.readline());assert reply['ok'],reply.get('message')
        return reply
    entries=[];retired=Counter();flow_count=0;supplemental=[];partial_sources={}
    with gzip.open(args.raw/'states.jsonl.gz','rt',encoding='utf-8') as source,gzip.open(out/'states.jsonl.gz','wt',encoding='utf-8') as dest:
        for line,entry in zip(source,receipt['entries'],strict=True):
            row=json.loads(line)
            digest=hashlib.sha256(json.dumps(row['state'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            assert row['id']==entry['id'] and digest==entry['state_sha256']
            state=row['state']
            if state['debt_pending']:
                actual=export(row['id'])['state'];settlement=actual.pop('settlement_debts')
                assert actual==state, 'Flow exporter changed a frozen semantic field: '+row['id']
                state['settlement_debts']=settlement;flow_count+=1
                if flow_count%10==0:print('captured settlement state',flow_count,flush=True)
            else:state['settlement_debts']=[0]*4
            if state['chance'] and state['chance']['needed']>=2 and not state['chance']['picked']:
                partial_sources.setdefault(state['chance']['kind'],(row['id'],state['chance']['pool']))
            complete(row['state'],identities);retired[len(row['state']['retired'])]+=1
            entries.append(dict(id=row['id'],state_sha256=hashlib.sha256(json.dumps(row['state'],sort_keys=True,separators=(',',':')).encode()).hexdigest()))
            dest.write(json.dumps(row,separators=(',',':'))+'\n')
    for kind,(identity,pool) in partial_sources.items():
        picks=[pool[0]]
        if kind=='rail-shuffle':picks.append(next(x for x in pool if x!=picks[0]))
        for order in ([picks,list(reversed(picks))] if len(picks)>1 else [picks]):
            reply=export(identity,order)
            supplemental.append(dict(id=f'partial-{kind}-{len(supplemental)}',source=identity,picks=order,
                                     state=complete(reply['state'],identities),snapshot=reply['snapshot']))
    worker.stdin.close();worker.wait(timeout=15);worker.stdout.close()
    (out/'supplemental.json').write_text(json.dumps(supplemental,indent=2)+'\n')
    receipt.update(entries=entries,raw_capture=str(args.raw.resolve()),raw_capture_receipt_sha256=sha(args.raw/'capture.json'),
                   materializer_sha256=sha(Path(__file__)),states_sha256=sha(out/'states.jsonl.gz'),retired_tile_count_histogram=dict(retired),
                   settlement_oracle_roots=flow_count,flow_exporter_sha256=sha(executable),flow_exporter_source_sha256=sha(ROOT/'tests/RustBackendNative/FlowExporter.cs'),
                   supplemental_sha256=sha(out/'supplemental.json'),supplemental_count=len(supplemental))
    (out/'capture.json').write_text(json.dumps(receipt,indent=2)+'\n')
    shutil.copyfile(__file__,out/Path(__file__).name)
    print('Completed',len(entries),'verified state DTOs; retired tile histogram',dict(retired))


if __name__=='__main__':main()
