"""Small paired public-root timing screen, not playing-strength qualification."""
from pathlib import Path
import random
import time

from brass_ai.host import RulesHost
from brass_ai.learning.contracts import project_path,require
from brass_ai.learning.reward import CostLedger
from .contracts import Profile,search_config
from .readiness import load
from .inference import SharedInference
from .engine import SearchSession,validate_personality_teacher


def probe(root,bundle,*,seconds=12.,workers=4,repeats=2,device='cpu',mono=None,unpruned=False):
    require(1<=repeats<=8,'Use a small timing screen')
    model,parents,ready,handoff,_=load(root,bundle)
    execution=dict(handoff['signature']['config']['execution'],processes=1,hosts_per_process=1)
    config=search_config(ready['reward'],seconds=seconds,workers=workers,unpruned=unpruned)
    service=SharedInference(execution,device=device);ident=service.register(model);ids=[service.register(p) for p in parents]
    channels=service.channels(1+workers*(1+len(ids)));rows=[]
    with service:
        session=SearchSession(root,service,channels,ident,ids,config,[Profile()]*4,ready['goal_calibration'],mono=mono)
        try:
            with RulesHost(project_path(root,handoff['signature']['config']['host']),mono=mono) as host:
                for repeat in range(repeats):
                    c=host.reset(87210000+repeat);view=host.observe(c['actor']);choices=host.choices()
                    for alpha in ((0.,.15) if repeat%2==0 else (.15,0.)):
                        session.profiles=[Profile(alpha=alpha)]*4;session.begin(87210000+repeat,repeat)
                        ledger=CostLedger(ready['reward']);ledger.observe(view);before=dict(session.telemetry)
                        tick=time.monotonic();selected,label,info=session.select(view,choices,random.Random(129+repeat),ledger)
                        if label is not None:validate_personality_teacher(label,dict(view=view,seat=view['seat'],choices=choices,chosen=selected,
                            teacher=None,behavior=dict(kind='external')),ident,ready['reward'])
                        elapsed=time.monotonic()-tick;session.finish_turn()
                        require(service.cpu.report()['active']==0,'Probe released before workers retired')
                        rows.append(dict(repeat=repeat,alpha=alpha,elapsed=elapsed,legal=0<=selected<len(choices),teacher=label is not None,
                            goal=info['goal'],completed=session.telemetry['completed']-before['completed'],
                            prior_seconds=session.telemetry['prior_seconds']-before['prior_seconds'],
                            overrun_seconds=session.telemetry['overruns']-before['overruns'],reason=info.get('reason')))
        finally:session.close()
    summary={}
    for alpha in (0.,.15):
        subset=[r for r in rows if r['alpha']==alpha]
        seconds_used=sum(r['elapsed'] for r in subset);simulations=sum(r['completed'] for r in subset)
        summary[str(alpha)]=dict(completed=simulations,seconds=seconds_used,completed_per_second=simulations/max(seconds_used,1e-9))
    base=summary['0.0']['completed_per_second'];biased=summary['0.15']['completed_per_second']
    return dict(rows=rows,summary=summary,relative_throughput_change=biased/base-1 if base else None,
        inference=service.report(),goal_bias_enabled=ready['goal_calibration']['bias_ready'],unpruned=unpruned,
        scope='Paired initial public-root smoke; timing noise and no playing-strength inference')
