"""Small P1 policy-only pilot: private hosts, batched NN, one learner per table.

This is not the P2 search/batching integration or a replacement Step 5 collector.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.host import RulesHost
from brass_ai.learning.batching import collate,model_id as base_id
from brass_ai.learning.contracts import CONFIG_DIR,digest,read_json,require
from brass_ai.learning.replay import split_for,terminal_targets
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.replay import write_shard
from brass_ai.step5.league import seeds_for,write_json
from brass_ai.step5.policy import exploration_distribution
from .checkpoint import model_id
from .contracts import TRACE,PROFILES,context,context_vector,contract


def distribution(logits,choices,*,exploration=.05):
    p=torch.softmax(logits.float(),-1)
    q=p.new_zeros(p.shape)
    for i,c in enumerate(choices):q[i,:len(c)]=torch.as_tensor(exploration_distribution(c),device=p.device,dtype=p.dtype)
    return (1-exploration)*p+exploration*q


def collect(host_directory,learner,parent,output,*,games=4,workers=4,seed=2710000,
            partition='train',device='cpu',mono=None,max_decisions=2000):
    require(type(games) is int and 1<=games<=32 and type(workers) is int and 1<=workers<=8,'P1 pilot bounds exceeded')
    require(partition in ('train','validation'),'Pilot partition must be training or goal calibration')
    out=Path(output);require(not out.exists(),'Use a new pilot collection directory')
    pins=read_json(CONFIG_DIR/'host-1.0.0.json')
    require(sha(Path(host_directory)/'BrassHost.exe')==pins['host_sha256'],'P1 requires the qualified rules host')
    lid,pid=model_id(learner),base_id(parent)
    spec=dict(format=TRACE,contract=contract(),model_id=lid,parent_id=pid,partition=partition,
              profiles=PROFILES,exploration=.05,temperature=1.,seed=seed,games=games,workers=workers)
    seeds,_=seeds_for(seed,games,partition);encoder=PreparedEncoder();completed=[];contexts={};began=time.monotonic()
    learner.eval();parent.eval()
    def start_game(host,index,s):
        c=host.reset(s);seat=index%4;profile=PROFILES[index%len(PROFILES)]
        identity='personality-pilot/'+digest(dict(model=lid,profile=profile,exploration=.05))
        opponent='personality-parent/'+pid
        g=dict(game_id=uuid.uuid4().hex,setup_group=digest(dict(rules=pins['rules'],setup_seed=s)),
               players=4,split=split_for(digest(dict(rules=pins['rules'],setup_seed=s))),
               opponents=[identity if x==seat else opponent for x in range(4)],records=[],result=None)
        return dict(host=host,ctx=c,game=g,seat=seat,personality=context(profile),
            rng=[random.Random(digest(dict(seed=seed,index=index,seat=x))) for x in range(4)])
    def observe(item):
        return item['host'].observe(item['ctx']['actor']),item['host'].choices()
    with ExitStack() as stack:
        hosts=[stack.enter_context(RulesHost(host_directory,mono=mono)) for _ in range(min(games,workers))]
        pool=stack.enter_context(ThreadPoolExecutor(max_workers=len(hosts)))
        for start in range(0,games,len(hosts)):
            live=[start_game(h,start+i,s) for i,(h,s) in enumerate(zip(hosts,seeds[start:start+len(hosts)]))]
            for item in live:contexts[item['game']['game_id']]=[]
            while live:
                views=list(pool.map(observe,live));groups={True:[],False:[]}
                for item,(view,choices) in zip(live,views):
                    require(len(item['game']['records'])<max_decisions,'Incomplete P1 game reached decision limit')
                    e=encoder.encode(view,choices,[0.]*4,[0.]*8)
                    groups[view['seat']==item['seat']].append((item,view,choices,e))
                actions=[]
                for is_learner,items in groups.items():
                    if not items:continue
                    inputs=collate([x[3] for x in items],device)
                    with torch.inference_mode():
                        if is_learner:
                            z=torch.as_tensor(np.stack([context_vector(x[0]['personality']) for x in items]),device=device)
                            budget=torch.full((len(items),4),learner.core.config.utility['insolvency']['max_cost'],device=device)
                            # Policy/goals do not depend on budget; P1 does not use
                            # this unused utility output for action selection.
                            logits=learner(*inputs,personality=z,remaining=budget)['policy_logits']
                        else:logits=parent(*inputs)[0]
                        probs=distribution(logits,[x[2] for x in items]).cpu().numpy()
                    for (item,view,choices,_),p in zip(items,probs):
                        p=p[:len(choices)].astype(np.float64);p/=p.sum()
                        chosen=item['rng'][view['seat']].choices(range(len(p)),weights=p,k=1)[0]
                        row=dict(step=len(item['game']['records']),seat=view['seat'],view=view,choices=choices,chosen=chosen,
                            style=[0.]*4,intent=[0.]*8,teacher=None,seconds=0.,
                            behavior=dict(kind='model',policy_id=item['game']['opponents'][view['seat']],
                                          probability=float(p[chosen]),temperature=1.))
                        item['game']['records'].append(row)
                        contexts[item['game']['game_id']].append(dict(step=row['step'],role='learner' if is_learner else 'incumbent',
                            context=item['personality'] if is_learner else None,model_id=lid if is_learner else pid))
                        actions.append((item,choices[chosen]))
                next_context=list(pool.map(lambda x:x[0]['host'].apply(x[0]['ctx'],x[1]),actions))
                for (item,_),c in zip(actions,next_context):item['ctx']=c
                for item in live[:]:
                    if item['ctx']['terminal']:
                        item['game']['result']=item['host'].call('result');terminal_targets(item['game']['result'])
                        completed.append(item['game']);live.remove(item)
                        print(f'P1 {partition} collection: {len(completed)}/{games}',flush=True)
    out.mkdir(parents=True)
    replay=out/'games.jsonl.gz'
    write_shard(replay,completed,dict(producer=TRACE,purpose='training' if partition=='train' else 'evaluation',spec=digest(spec)))
    metadata=dict(format=TRACE,spec=spec,spec_id=digest(spec),replay=dict(path=replay.name,sha256=sha(replay)),
        contexts=contexts,contexts_id=digest(contexts),seconds=time.monotonic()-began,scope='Fresh P1 policy pilot; no search or strength qualification')
    write_json(out/'manifest.json',metadata)
    return metadata
