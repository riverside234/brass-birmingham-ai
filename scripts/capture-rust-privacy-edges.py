"""Frozen C# regressions for optional empty-merchant fields in the R1 allowlist."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import importlib.util
import json
from pathlib import Path
import runpy
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
CAP=runpy.run_path(str(ROOT/'scripts/capture-rust-privacy.py'))
L05,BASE=CAP['L05'],CAP['BASE']
read,write,receipt,sha=(BASE[k] for k in ('read','write','receipt','sha'))


def capture(out):
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists()
    out.mkdir(parents=True)
    reference=L05['prepare_reference'](out)
    before=BASE['verify_frozen'](reference_source_root=reference)
    source=ROOT/'artifacts/rust-backend/20261025-l06-capture'
    frozen=L05['verify_capture'](source)
    for item in [frozen['exporter'],*frozen['runtime']]:assert receipt(ROOT/item['path'])==item
    spec=importlib.util.spec_from_file_location('frozen_beliefs',reference/'src/brass_ai/beliefs.py')
    beliefs=importlib.util.module_from_spec(spec);spec.loader.exec_module(beliefs)
    composition=read(reference/'configs/contracts/v0.2.0/card-composition.json')
    views=next(CAP['rows'](source/'roots.jsonl.gz'))['views']
    empty=[i for i,m in enumerate(views[0]['merchants']) if not m['types']]
    assert len(empty)==3
    oracle=runpy.run_path(str(ROOT/'scripts/capture-rust-rules.py'))['Oracle'](ROOT/frozen['exporter']['path'])
    cases=steps=0
    try:
        with gzip.open(out/'edges.jsonl.gz','wt',encoding='utf-8') as output:
            for variant,indices in [('first-empty',empty[:1]),('all-empty',empty)]:
                for seat,original in enumerate(views):
                    view=deepcopy(original)
                    for i in indices:del view['merchants'][i]['types']
                    rng=CAP['ShuffleRecorder'](890+cases)
                    with patch.object(beliefs.random,'Random',return_value=rng):setup=beliefs.sample_world_setup(view,composition,890+cases)
                    initial=oracle.call(seed=1,setup=setup,projection=True)
                    snapshot=initial.pop('snapshot');initial['views']=snapshot['views']
                    assert initial['views'][seat]==view
                    complete=variant=='all-empty' and seat==0
                    trace=[];current=initial
                    for _ in range(650 if complete else 12):
                        if current['state']['context']['terminal']:break
                        if current['chance']:command=dict(card=current['chance']['outcomes'][0]['card'])
                        else:
                            choices=current['choices'];choice=next((c for c in choices if c['kind'] in ('pass','finish')),choices[0])
                            command=dict(choice=choice['id'])
                        after=oracle.call(**command,projection=True)
                        snapshot=after.pop('snapshot');after['views']=snapshot.get('views')
                        for i in indices:assert 'types' not in after['state']['merchants'][i]
                        trace.append(dict(command=command,expected=after));current=after
                    if complete:assert current['state']['context']['terminal']
                    output.write(json.dumps(dict(id=f'{variant}-seat-{seat}',view=view,recall=[],draws=rng.draws,
                                                 expected=initial,suffix=trace,complete_game=complete),separators=(',',':'))+'\n')
                    cases+=1;steps+=len(trace)
                    print(variant,seat,'steps',len(trace),flush=True)
        after=BASE['verify_frozen'](reference_source_root=reference)
        assert L05['preserved_identity'](before)==L05['preserved_identity'](after)
        write(out/'capture.json',dict(format='brass-native-privacy-edges/1.0.0',completed_utc=datetime.now(timezone.utc).isoformat(),
              cases=cases,transitions=steps,complete_games=1,source=receipt(source/'capture.json'),
              reference_sources=[receipt(reference/p) for p in ('src/brass_ai/beliefs.py','src/brass_ai/searchlab/spiel.py')],
              runtime=frozen['runtime'],exporter=frozen['exporter'],script=receipt(Path(__file__)),
              files={'edges.jsonl.gz':sha(out/'edges.jsonl.gz')},preservation=after))
    finally:oracle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    capture(parser.parse_args().out.resolve())
