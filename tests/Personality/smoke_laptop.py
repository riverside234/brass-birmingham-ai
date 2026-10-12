"""P6 authored 16x8 fixture export + real native whole-turn smoke; never release."""
import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'src',ROOT/'tests/R2',ROOT/'tests/Personality'):sys.path.insert(0,str(p))

import torch
from smoke_search import fixture
from test_difficulty import named
from brass_ai.bank import sha
from brass_ai.host import RulesHost
from brass_ai.learning.batching import model_id
from brass_ai.personality.search.readiness import load
from brass_ai.personality.search.contracts import controller,Profile
from brass_ai.search.common import SearchConfig
from brass_ai.personality.laptop import package,benchmark
from brass_ai.personality.laptop.contracts import write,seal


def prepare(out,ready=None):
    ready=Path(ready).resolve() if ready else fixture(out)
    model,parents,r,handoff,pool=load(ROOT,ready)
    v={k:x for k,x in named(16,8).items() if k!='id'}
    v['shared_model']=dict(ready=ready.relative_to(ROOT).as_posix(),ready_id=r['id'],model_id=r['model']['model_id']);v['pool_id']=pool
    cfg=SearchConfig(**v['search']).validate()
    for m in v['members']:
        m['model_id']=r['model']['model_id'];m['candidate_id']=controller(m['model_id'],model_id(parents[0]),Profile.parse(m['profile']),cfg,pool)['candidate_id']
    v['controls']=[controller(r['model']['model_id'],model_id(parents[0]),Profile.parse(c['profile']),cfg,pool) for c in v['controls']]
    v['dependencies']=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in (ready/'ready.json',ready/'model.pt')]
    write(out/'roster/manifest.json',seal(v))
    package.build(ROOT,out/'roster',out/'package',fixture=True)
    return out/'package'


def cases():
    with RulesHost(ROOT/'artifacts/r1-host/guarded') as host:
        c=host.reset(73)
        # Advance four cheap completed turns to reach an ordinary two-action turn.
        while c['round']<2:
            choices=host.choices();chosen=next((x for x in choices if x['kind']=='pass'),choices[0])
            c=host.apply(c,chosen)
        return [dict(view=host.observe(c['actor']),choices=host.choices(),seed=998)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--ready',type=Path)
    p.add_argument('--tiers',nargs='+',default=['easy']);a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('Use a fresh fixture directory')
    torch.set_num_threads(2);torch.manual_seed(78361);out.mkdir(parents=True)
    bundle=prepare(out,a.ready);positions=cases();write(out/'positions.json',positions)
    r=benchmark.measure(bundle,positions,out/'benchmark.json',tiers=a.tiers,fixture=True)
    print({k:r[k] for k in ('fixture','qualified')});print(r['results']);print(benchmark.recommendation(r))
    if not all(r['available'] for r in r['results']):raise RuntimeError('Fixture integration smoke failed')

if __name__=='__main__':main()
