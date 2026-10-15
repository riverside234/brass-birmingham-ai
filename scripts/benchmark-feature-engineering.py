"""Bounded FE0/FE3 CPU benchmark on identical saved permitted inputs; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/FeatureEngineering')]
from brass_ai.learning.contracts import digest,compatibility,require
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import champion_mix_config,time_budget_config
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import implementation_id,write_json


def measure(factory,positions,repeats=5):
    samples=[];candidates=0
    obj=factory()
    for row in positions:obj.encode(row['view'],row['choices'])
    for _ in range(repeats):
        for row in positions:
            started=time.perf_counter();obj.encode(row['view'],row['choices'])
            samples.append(time.perf_counter()-started);candidates+=len(row['choices'])
    tracemalloc.start()
    memory_obj=factory()
    for row in positions:memory_obj.encode(row['view'],row['choices'])
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop();ordered=sorted(samples)
    result=dict(calls=len(samples),median_ms=statistics.median(samples)*1000,
        p95_ms=ordered[int(.95*(len(ordered)-1))]*1000,candidates_per_second=candidates/sum(samples),peak_python_bytes=peak)
    if hasattr(obj,'report'):result['cache']=obj.report()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('baseline','reference','derived'),required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/feature-engineering/fe0-fe3')
    a=parser.parse_args();out=a.output.resolve();require(out.is_relative_to(ROOT/'artifacts'),'Output must be under artifacts')
    out.mkdir(parents=True,exist_ok=True);inputs=out/'benchmark-inputs.json'
    if a.stage=='baseline':
        require(not inputs.exists(),'Preserve the original baseline inputs')
        from fe_fixtures import benchmark_positions
        positions=benchmark_positions();write_json(inputs,positions)
        cfg=json.loads((ROOT/'configs/step5/joint7-latest.json').read_text(encoding='utf-8'))
        resolved=champion_mix_config(time_budget_config(cfg));model=FourSeatNet(ModelConfig.preset())
        utility=UtilityNet(utility_config(model.config,reward_spec(resolved)))
        paths=[p for folder in ('src/brass_ai','configs/learning','configs/step5','configs/contracts','src/BrassTrainingHost','src/BrassHost') for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.cs','.json')]
        write_json(out/'settings-snapshot.json',dict(implementation=implementation_id(),compatibility=compatibility(),
            active_config=resolved,config_sha256=digest(cfg),state_dim=model.config.state_dim,action_dim=model.config.action_dim,
            base_parameters=model.parameter_count,utility_parameters=utility.parameter_count,
            files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
        result=dict(reference=measure(Encoder,positions),prepared_cold=measure(lambda:ColdEncoder(),positions),prepared_warm=measure(PreparedEncoder,positions))
    elif a.stage=='reference':
        positions=json.loads(inputs.read_text(encoding='utf-8'))
        result=dict(reference=measure(Encoder,positions),prepared_cold=measure(ColdEncoder,positions),prepared_warm=measure(PreparedEncoder,positions))
    else:
        positions=json.loads(inputs.read_text(encoding='utf-8'))
        from brass_ai.feature_engineering.context import DerivedContextBuilder
        class Derived:
            def __init__(self,cold=False):self.builder=DerivedContextBuilder();self.cold=cold
            def encode(self,view,choices):
                if self.cold:self.builder=DerivedContextBuilder()
                return self.builder.build(view,choices)
            def report(self):
                return dict(scope='last call only; builder reset every call' if self.cold else 'all warmup and timed calls',
                            **self.builder.report())
        result=dict(derived_cold=measure(lambda:Derived(True),positions),derived_warm=measure(Derived,positions))
    result.update(inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),positions=len(positions),
        candidates=sum(len(p['choices']) for p in positions),platform=platform.platform(),python=sys.version,
        scope='CPU wall time including validation/content identity; separate tracemalloc pass measures Python allocations, not total RSS/GPU memory. Same saved raw inputs; standalone derivations, no model/search/training integration.')
    write_json(out/f'{a.stage}.json',result);print(json.dumps(result,indent=2))


class ColdEncoder:
    def encode(self,view,choices):return PreparedEncoder().encode(view,choices)


if __name__=='__main__':main()
