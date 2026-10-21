"""Two native ordinary games, one forced test fork, one PPO step and exact restart."""
import argparse
import copy
from pathlib import Path
import time

import torch

from brass_ai.bank import sha
from brass_ai.host import RulesHost
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,require,digest
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import exploration as E
from brass_ai.recovery.execution import run_game,neural_spec
from brass_ai.recovery.host_runtime import training_host
from brass_ai.recovery.ingress import load_collection
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.replay import write_shard,cache_receipt,read_shards
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.telemetry import Timings
from brass_ai.recovery.updates import Dataset,policy_measure,ppo_update
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.recovery.value_phase import split,metrics
from brass_ai.step5.league import seeds_for,write_json
from brass_ai.step5.policy import NeuralPolicy
from test_exploration_revision import config


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new artifact directory')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(2802);started=time.monotonic()
    cfg=config();model=FourSeatNet(ModelConfig.preset('medium'))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    reward=reward_spec(cfg);migrate(model,optimizer,reward);model.eval()
    state=E.initial(cfg,0);settings=E.effective(cfg,state,20);settings['branching']=dict(E.BRANCH,rate_per_decision=1.)
    behavior=Behavior(model_id(model),cfg['exploration'],settings)
    policy=NeuralPolicy(model,exploration=cfg['exploration'],focus='mixed',exploration_version=3,opening=E.OPENING,
        batch_size=8,max_candidate_cells=131072)
    records=[];kinds={};collection=[]
    with RulesHost(training_host(SOURCE_ROOT)) as host:
        for i,seed in enumerate(seeds_for(53241,2)[0]):
            group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed));kinds[group]='selfplay'
            proof=dict(producer='recovery/2.0.0',purpose='training',kind='selfplay',exploration=settings)
            job=dict(index=i,seed=seed,sampling_seed=seed*7,stable_game_id='exploration-smoke-'+str(i),
                policies=[neural_spec(behavior.evaluator.model_id,cfg['exploration'],'mixed',version=3,opening=E.OPENING)]*4,
                focuses={str(s):'neutral' for s in range(4)},provenance=proof)
            if i==0:job['branching']=settings['branching']
            began=time.monotonic();game=run_game(host,job,[policy]*4,Timings(),2000)
            elapsed=time.monotonic()-began;path=out/f'game-{i}.jsonl.gz'
            write_shard(path,[game],dict(producer='smoke',purpose='training',
                jobs=[dict(index=i,seed=seed,provenance=proof)]))
            records.append(dict(job=i,game_id=game['game_id'],path=str(path),sha256=sha(path),features=cache_receipt(path)))
            collection.append(dict(seconds=elapsed,ordinary_rows=len(game['records']),
                branch_rows=sum(len(g['records'])-1 for g in game.get('branches',[]))))
            print(f'Native collection {i+1}/2: {elapsed:.1f}s, {collection[-1]}',flush=True)
    policy.close()
    games,data,diagnostics=load_collection(records,behavior.policy_id,kinds,insolvency=cfg['insolvency'],reward=reward)
    require(len(games)==3 and diagnostics['health_counts']['tables']==2,'Branches entered ordinary health')
    require(sum(bool(d[0].get('_branch')) for d in data)==collection[0]['branch_rows'],'Branch move/prefix row admission differs')
    from brass_ai.recovery.branching import validate_opening_rows
    # load_collection returns compact game summaries; inspect the authoritative
    # shard for the forced and subdecision rows omitted from those summaries.
    branch=read_shards([records[0]['path']],receipts={records[0]['path']:records[0]['features']})[0]['branches'][0]
    validate_opening_rows(branch['records'][1:])
    openings={s:sum(r['seat']==s and r['behavior']['temperature']==2. for r in branch['records'][1:]) for s in range(4)}
    require(all(n==2 for n in openings.values()),'Branch did not restart two openings per seat')
    ds=Dataset(data,'cpu');cfg['update'].update(epochs=1,effective_batch_size=65536,critic_batch_size=64,microbatch_size=16)
    kl=policy_measure(model,ds,range(len(data)),behavior,cfg['update'],Timings(),verify=True)
    require(abs(kl)<1e-7,'Collected probabilities do not reconstruct')
    train,held,selection=split(data,.05,cfg['value_phase']['holdout'])
    parent=games[0]['game_id']
    require(all((i in train)==(next(j for j,d in enumerate(data) if d[2][0]==parent) in train)
                for i,d in enumerate(data) if d[0].get('_branch')),'Branch leaked across holdout')
    before=out/'incoming.pt';save_checkpoint(before,model,optimizer,step=185,training=dict(exploration=state))
    original=sha(before);results=[];finals=[]
    for restart in (False,True):
        net,raw=load_checkpoint(before);adam=torch.optim.AdamW(net.parameters())
        resume_optimizer(raw,adam)
        require(_same_payload(dict(net.state_dict()),raw['state_dict']) and _same_payload(adam.state_dict(),raw['optimizer']),
            'Checkpoint did not preserve weights/Adam')
        report=ppo_update(net,adam,games,behavior,cfg['update'],learner_data=data,insolvency=cfg['insolvency'],reward=reward)
        require(report['optimizer_steps']==1,'Bounded PPO smoke exceeded one step')
        final=dict(weights=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(adam.state_dict()),rng=torch.get_rng_state().clone())
        finals.append(final)
        results.append(dict(restarted=restart,ppo_steps=report['optimizer_steps'],
            categorical_probe=report['categorical_probe'],branch_sample_usage=report['branch_sample_usage'],
            value_evidence=metrics(net,ds,held,cfg['update'],evidence=True)))
    require(_same_payload(*finals),'Repeated checkpoint restore changed PPO result or RNG')
    require(sha(before)==original,'Incoming checkpoint changed')
    write_json(out/'report.json',dict(passed=True,seconds=time.monotonic()-started,parameters=model.parameter_count,
        ordinary_games=2,branch_games=1,branch_openings_by_seat=openings,collection=collection,diagnostics=diagnostics['exploration_metrics'],
        exact_probability_reconstruction=True,exact_restored_update=True,results=results,
        production_branch_rate=cfg['branching']['rate_per_decision'],
        scope='Isolated randomly initialized current v11.2 model; test forces one early fork (rate=1) to exercise admission. Two native ordinary games; one PPO step repeated from the same checkpoint. No strength or production-overhead estimate.'))


if __name__=='__main__':main()
