"""Two real P5 fixture games (neural and original Hard), plus one searched turn.

No trained model or qualification is claimed. This creates new owned artifacts.
"""
import argparse
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'src',ROOT/'tests/R2',ROOT/'tests/Personality'):sys.path.insert(0,str(path))

import torch
from smoke_search import fixture
from test_difficulty import named
from brass_ai.bank import sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest, require, compatibility
from brass_ai.learning.reward import CostLedger
from brass_ai.host import RulesHost
from brass_ai.personality.contracts import source_id
from brass_ai.personality.deployment.contracts import sealed
from brass_ai.personality.search.contracts import controller, Profile
from brass_ai.personality.search.readiness import load
from brass_ai.personality.search.inference import SharedInference
from brass_ai.personality.search.engine import SearchSession
from brass_ai.personality.difficulty import panels, collection, evaluation, native, ui
from brass_ai.personality.difficulty.contracts import resolve_search, default_profiles
from brass_ai.search.common import SearchConfig, turn_key
from brass_ai.step5.league import implementation_id, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mono')
    a=p.parse_args();out=a.output.resolve();require(out.is_relative_to(ROOT/'artifacts') and not out.exists(),'Use a new smoke directory')
    torch.set_num_threads(2);torch.manual_seed(78361);began=time.monotonic()
    before=(implementation_id(),source_id(),compatibility());ready=fixture(out)
    model,parents,r,handoff,pool=load(ROOT,ready)
    v={k:x for k,x in named().items() if k!='id'}
    v['shared_model']=dict(ready=ready.relative_to(ROOT).as_posix(),ready_id=r['id'],model_id=r['model']['model_id'])
    v['pool_id']=pool;cfg=SearchConfig(**v['search']).validate()
    for m in v['members']:
        m['model_id']=r['model']['model_id']
        m['candidate_id']=controller(m['model_id'],model_id(parents[0]),Profile.parse(m['profile']),cfg,pool)['candidate_id']
    v['controls']=[controller(r['model']['model_id'],model_id(parents[0]),Profile.parse(c['profile']),cfg,pool) for c in v['controls']]
    v['dependencies']=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in (ready/'ready.json',ready/'model.pt')]
    directory=out/'fixture-roster';directory.mkdir();write_json(directory/'manifest.json',sealed(v))
    native.runtime(ROOT)  # explicit build-host command first
    plan=panels.plan(ROOT,directory,out/'panel',setups=2,repetitions=1,fixture=True,tables=1)
    chunk=collection.execute(ROOT,out/'panel',out/'chunk',games=2,tables=1,device='cpu',mono=a.mono,fixture_direct=True)
    report=evaluation.check(ROOT,out/'panel',[out/'chunk'],out/'checked')
    require(not report['p5_qualified'] and not report['release_qualified'] and not report['failures'], 'Fixture smoke failed or received qualification')
    require(chunk['inference']['cpu_admission']['active']==0 and chunk['inference']['error'] is None, 'Unretired resource or inference failure')
    require(any(j['job']['lineup']=='original-hard' and j['stock_actions']>0 for j in
        [__import__('json').loads((out/'chunk'/item['path']).read_text(encoding='utf-8')) for item in chunk['records']]),'Stock bridge not exercised')
    # One real 2-second/two-worker root/turn exercises unchanged P2 clocks; not a full searched tournament.
    search=resolve_search(v,default_profiles(),'easy');execution=handoff['signature']['config']['execution']
    with SharedInference(execution,'cpu') as service:
        ident=service.register(model);pids=[service.register(m) for m in parents]
        channels=collection.TrackedChannels(service.channels(1+search.workers*(1+len(pids))))
        session=SearchSession(ROOT,service,channels,ident,pids,search,[Profile()]*4,r['goal_calibration'],mono=a.mono)
        try:
            session.begin(73,782);ledger=CostLedger(r['reward']);rng=random.Random(782)
            with RulesHost(native.runtime(ROOT),mono=a.mono) as host:
                c=host.reset(73);key=turn_key(c);seat=c['actor'];start=time.monotonic()
                while not c['terminal'] and c['actor']==seat and turn_key(c)==key:
                    view=host.observe(seat);choices=host.choices();ledger.observe(view)
                    chosen,_,_=session.select(view,choices,rng,ledger)
                    ledger.chosen(view,choices[chosen]);c=host.apply(c,choices[chosen])
                elapsed=time.monotonic()-start
            session.finish_turn();timing=dict(seconds=elapsed,search=session.telemetry,cpu=service.cpu.report())
        finally:session.close();channels.close()
    require(before==(implementation_id(),source_id(),compatibility()),'Upstream training compatibility changed')
    menu=ui.menu(named(16,8),fixture=True);write_json(out/'public-menu.json',menu)
    result=dict(scope='Untrained authored fixtures only; no difficulty/human/release evidence',seconds=time.monotonic()-began,
                games=2,original_hard_baseline=plan['baseline']['identity'],qualification=report['status'],
                collection_seconds=chunk['seconds'],inference=chunk['inference'],searched_turn=timing,
                public_style_count=len(menu['choices']),step5_implementation=before[0])
    write_json(out/'smoke-report.json',result);print(result,flush=True)


if __name__=='__main__':main()
