"""Capture every staged alternative in authored edge roots and selected complex roots."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[1]
CAPTURE=runpy.run_path(str(ROOT/'scripts/capture-rust-rules.py'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();source=args.capture.resolve()
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not (out/'capture.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    with gzip.open(source/'roots.jsonl.gz','rt',encoding='utf-8') as stream:roots=list(map(json.loads,stream))
    selected=[r for r in roots if r['id'].startswith('edge-')]
    selected += [r for r in roots if r['id']=='game-731001-root-000']
    # Development-only selection. Never use held-out game roots to select workloads.
    for phase in ('action','payment','continuation','debt'):
        candidates=[r for r in roots if r['partition']=='development' and r['id'].startswith('game-') and r['expected']['state']['context']['phase']==phase and r['expected']['state']['context']['era']=='RailEra' and not r['expected']['state']['chance']]
        if candidates:selected.append(max(candidates,key=lambda r:(sum('tile' in s for s in r['expected']['state']['slots']),len(r['expected']['state']['history']))))
    executable=source/'exporter/RulesExporter.exe';oracle=CAPTURE['Oracle'](executable);coverage=Counter();entries=[];total=0
    try:
        with gzip.open(out/'branches.jsonl.gz','wt',encoding='utf-8') as stream:
            for root in selected:
                expected=root['expected'];commands=[dict(choice=c['id']) for c in expected['choices']]
                if expected['chance']:commands=[dict(card=c['card']) for c in expected['chance']['outcomes']]
                if root['id'].startswith('game-') and not root['id'].endswith('-000'):
                    # One alternative per staged kind at each dense imported boundary.
                    unique={}
                    for command in commands:unique.setdefault(command.get('choice','chance').split('|')[0],command)
                    commands=list(unique.values())
                for command in commands:
                    before=oracle.call(**root['request']);assert before==expected,root['id']
                    after=oracle.call(**command)
                    row=dict(root=root['id'],before=before,command=command,expected=after)
                    stream.write(json.dumps(row,separators=(',',':'))+'\n');total+=1
                    coverage[command.get('choice','chance').split('|')[0]]+=1
                entries.append(dict(root=root['id'],alternatives=len(commands)));print(root['id'],len(commands),flush=True)
        result=dict(format='brass-native-rules-branches/1.0.0',roots=entries,transitions=total,coverage=dict(coverage),
                    roots_sha256=CAPTURE['sha'](source/'roots.jsonl.gz'),exporter_sha256=CAPTURE['sha'](executable),
                    script_sha256=CAPTURE['sha'](Path(__file__)),files={'branches.jsonl.gz':CAPTURE['sha'](out/'branches.jsonl.gz')})
        (out/'capture.json').write_text(json.dumps(result,indent=2)+'\n');print('Captured',total,'alternatives',flush=True)
    finally:oracle.close()


if __name__=='__main__':main()
