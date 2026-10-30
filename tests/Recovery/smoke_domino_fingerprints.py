"""Small fixed-position fingerprint benchmark using real smoke-game inputs."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,require,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.batching import model_id
from brass_ai.learning.features import Encoder
from brass_ai.recovery.critic_data import fingerprint
from brass_ai.recovery.domino_fingerprints import Fingerprints
from brass_ai.recovery.il_kl_probe import Probe
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.recovery.telemetry import memory
from brass_ai.step5.policy import context_for
from brass_ai.step5.league import write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--collection',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use new isolated output');out.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(17343)
    c=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z/recorded-config.json')
    m=FourSeatNet(ModelConfig.preset(c['model_size']));o=torch.optim.AdamW(m.parameters());migrate(m,o,reward_spec(c));m.eval()
    other=copy.deepcopy(m)
    with torch.no_grad():next(other.parameters()).add_(.001)
    p=out/'frozen.pt';save_checkpoint(p,other);mid=model_id(m);oid=model_id(other)
    family=json.load(gzip.open(sorted((a.collection/'control').glob('*.family.json.gz'))[0]))
    records=[r for r in family['parent']['game']['records'] if len(r['choices'])>1]
    selected=np.linspace(0,len(records)-1,min(64,len(records)),dtype=int).tolist();records=[records[i] for i in selected]
    enc=Encoder();encoded=[enc.encode(r['view'],r['choices'],*context_for('neutral')) for r in records]
    rows=[dict(phase=r['view']['context']['phase'],era=r['view']['context']['era'],chosen=r['chosen'],
        kinds=[v['kind'] for v in r['choices']],industries=[v.get('industry') for v in r['choices']]) for r in records]
    offsets=np.r_[0,np.cumsum([len(e.actions) for e in encoded])]
    probe=SimpleNamespace(metadata=dict(probe_hash=digest(dict(rows=rows,positions=selected))),rows=rows,encoded=encoded,
        arrays=dict(offsets=offsets,reference=np.zeros(offsets[-1],dtype=np.float32)),
        parts=lambda update:[list(range(i,min(i+8,len(rows)))) for i in range(0,len(rows),8)])
    started=time.perf_counter();neutral=fingerprint(probe,Probe.probabilities(probe,m,c['update']));neutral_seconds=time.perf_counter()-started
    provider=Fingerprints(SOURCE_ROOT,out,probe,m,c['update'],{oid:dict(path=str(p),sha256=sha(p))},neutral=neutral)
    before=torch.get_rng_state().clone()
    for model in (mid,oid):
        for z in range(7):provider.get(model,z)
    cold=copy.deepcopy(provider.report)
    require(cold['misses']==13 and len(provider.models)==2,'Fingerprint pass/model duplication')
    again=Fingerprints(SOURCE_ROOT,out,probe,m,c['update'],{oid:dict(path=str(p),sha256=sha(p))})
    started=time.perf_counter()
    for model in (mid,oid):
        for z in range(7):require(again.get(model,z)==provider.get(model,z),'Persisted fingerprint changed')
    warm_seconds=time.perf_counter()-started
    require(again.report['misses']==0 and torch.equal(before,torch.get_rng_state()),'Fingerprint cache misses or RNG mutation')
    write_json(out/'report.json',dict(passed=True,parameters=m.parameter_count,positions=len(rows),legal_actions=int(offsets[-1]),
        neutral_seconds=neutral_seconds,cold=cold,warm_seconds=warm_seconds,warm=again.report,memory=memory('cpu'),
        scope='CPU, 64 actual native-smoke positions, complete legal distributions; current 512-wide model plus one frozen model, all seven contexts. Production uses its unchanged 2048-position probe; not a GPU throughput estimate.'))
    print(dict(passed=True,parameters=m.parameter_count,neutral_seconds=neutral_seconds,cold_seconds=cold['seconds'],warm_seconds=warm_seconds),flush=True)


if __name__=='__main__':main()
