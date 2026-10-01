"""Quick selected-backend probe: NN alone, Masked ISMCTS, optional full expansion.

This is a decision/coverage screen, not a promotion tournament. It never loads
an old checkpoint, starts another algorithm, calibrates v3, or changes training.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    import torch
    from brass_ai.learning.contracts import LAB_ROOT,project_path,require
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.host import RulesHost
    from brass_ai.bank import read,sha
    from brass_ai.step5.league import implementation_id,write_json
    from brass_ai.search.common import SearchConfig
    from brass_ai.search.ismcts import MaskedSearchPolicy
    from brass_ai.recovery.parallel import parallelize
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=LAB_ROOT)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--config',default='configs/search/laptop-cpu.json')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--mono');p.add_argument('--seed',type=int,default=18510001)
    p.add_argument('--seconds',type=float);p.add_argument('--simulations',type=int)
    p.add_argument('--compare-unpruned',action='store_true')
    p.add_argument('--output',default='artifacts/masked-search/probe.json')
    a=p.parse_args();require(a.root.is_absolute(),'Use an absolute project root');root=a.root.resolve()
    cfg=read(project_path(root,a.config));require(cfg['format']=='masked-search-probe/1.0.0','Wrong probe config')
    torch.set_num_threads(cfg['threads'])
    model,raw=load_checkpoint(project_path(root,a.checkpoint),device=a.device)
    require(model.config.architecture=='brass-four-seat-mlp-resnet/1.0.0' and raw['training'].get('fresh_run'),'Only a fresh residual run checkpoint is accepted')
    config=SearchConfig(**cfg['search']).validate()
    config=replace(config,**({} if a.seconds is None else {'seconds':a.seconds}),**({} if a.simulations is None else {'simulations':a.simulations})).validate()
    with RulesHost(root/'artifacts/r1-host/guarded',mono=a.mono) as host:
        c=host.reset(a.seed)
        while c['round']==1:
            legal=host.choices();choice=next((x for x in legal if x['kind']=='pass'),legal[0])
            c=host.apply(c,choice)
        view,choices=host.observe(c['actor']),host.choices()
    results={};variants=['top-k','full-expansion'] if a.compare_unpruned else ['top-k']
    for name in variants:
        setting=replace(config,root_top_k=2**30,interior_top_k=2**30) if name=='full-expansion' else config
        private=replace(setting,workers=1)
        policy=MaskedSearchPolicy(model,root,a.device,config=private,mono=a.mono)
        try:
            if setting.workers>1:
                policy=parallelize(policy,setting.workers,setting.cpu_role,
                    lambda _:MaskedSearchPolicy(model,root,a.device,config=private,mono=a.mono))
            priors,_=policy.base.predict([(view,choices,random.Random(a.seed))])[0]
            results.setdefault('nn-policy',dict(choice=choices[max(range(len(priors)),key=lambda i:priors[i])]['id'],legal_choices=len(choices)))
            began=time.monotonic();selected,_,teacher=policy.decide(view,choices,random.Random(a.seed),began+setting.seconds)
            stats=policy.last_statistics or dict(completed=0,root={},expansion=[False]*len(choices))
            results[name]=dict(choice=choices[selected]['id'],whole_decision_seconds=time.monotonic()-began,
                complete_legal_incumbent=bool(policy.last_continuation),search=stats,teacher=teacher,config=setting.to_dict())
        finally:policy.close()
    report=dict(format='masked-search-probe/1.0.0',implementation=implementation_id(),checkpoint_sha256=sha(project_path(root,a.checkpoint)),
        model_id=raw['model_id'],parameters=model.parameter_count,device=a.device,seed=a.seed,results=results,
        scope='same-budget decision coverage; no strength/promotion claim')
    write_json(project_path(root,a.output),report)
    print({name:dict(choice=r['choice'],completed=r.get('search',{}).get('completed')) for name,r in results.items()})

if __name__=='__main__':main()
