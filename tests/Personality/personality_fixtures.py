"""Authored unit-test evidence only, never a qualified or trained production AI."""
import copy
from itertools import combinations
from pathlib import Path
import shutil
import torch

from fixtures import position,result
from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest,read_json
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.learning.reward import specification
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.replay import split_for
from brass_ai.recovery.champion import CONTRACT,decision
from brass_ai.recovery.replay import write_shard
from brass_ai.step5.league import write_json,seeds_for,implementation_id


def parent(size='small'):
    model=UtilityNet(utility_config(ModelConfig.preset(size),specification()))
    model.utility_calibration=dict(ready=True,reward_id=digest(specification()),scope='authored test fixture only')
    return model


def episode(rounds=6):
    records=[];cash=[17+s for s in range(4)]
    for rnd in range(1,rounds+1):
        for seat in range(4):
            for phase in ('action','continuation'):
                view,choices=position(seat)
                if phase=='continuation':cash[seat]+=30;choices=[dict(id=f'finish-{rnd}-{seat}',kind='finish')]
                for s,p in enumerate(view['players']):p['cash']=cash[s]
                view['context'].update(round=rnd,revision=len(records),phase=phase,actions_remaining=1)
                records.append(dict(step=len(records),seat=seat,view=view,choices=choices,chosen=0,
                    behavior=dict(kind='random',policy_id='authored-fixture',probability=1/len(choices),temperature=1.),
                    style=[0.]*4,intent=[0.]*8,teacher=None,seconds=0.))
    seed=seeds_for(3810000,1)[0][0];group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
    return dict(game_id='authored-personality-fixture',setup_group=group,players=4,split=split_for(group),
                opponents=['authored-fixture']*4,result=result(),records=records)


def qualified_run(directory,*,traces=True,size='small'):
    root=SOURCE_ROOT;run=Path(directory);run.mkdir(parents=True)
    cfg=read_json(root/'configs/step5/lab.json')
    signature=dict(config=cfg,compatibility=compatibility(),implementation=implementation_id(),
        legacy_pool=read_json(root/'training/opponents/pool.json'),teachers={})
    model=parent(size);entries=[]
    for gen in (0,1):
        with torch.no_grad():model.outcomes[-1].bias.add_(.001)
        path=run/f'generation-{gen:04d}.pt'
        identity=save_checkpoint(path,model,step=gen,training=dict(mode='step5-population-ppo',
            fresh_run=run.relative_to(root).as_posix(),signature=digest(signature),fixture=True))
        entries.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path),model_id=identity,generation=gen))
    gate_dir=run/'evaluations/champion-00006400';gate_dir.mkdir(parents=True)
    seeds=seeds_for(3920000,8,'validation')[0]
    spec=dict(format=CONTRACT,candidate=entries[1],incumbent=entries[0],seeds=seeds,config=cfg['champion'])
    rows=[dict(seed=s,candidate_seats=list(pair),win=1.,candidate_vp=100.,incumbent_vp=70.,
        candidate_rank=1.5,incumbent_rank=3.5) for s in seeds for pair in combinations(range(4),2)]
    outcome=decision(rows,cfg['champion']);report=dict(format=CONTRACT,spec_id=digest(spec),result=outcome,
        outcomes=rows,failed_games=0,scope='Authored unit-test fixture, not actual qualification')
    write_json(gate_dir/'spec.json',spec);write_json(gate_dir/'report.json',report)
    gate=dict(games_done=6400,candidate_generation=1,incumbent_generation=0,**outcome,
        report=(gate_dir/'report.json').relative_to(root).as_posix(),report_sha256=sha(gate_dir/'report.json'))
    state=dict(format='step5-recovery-league/2.2.0',signature=signature,history=entries,population=entries,
        evaluation_anchor=entries[0],champion=entries[1],origin=dict(fixture=True),
        champion_gate=dict(accepted_games=6400,accepted_gate=gate),games_done=6400)
    write_json(run/'state.json',state)
    write_json(run/'best.json',dict(format=CONTRACT,checkpoint=entries[1],selected_games=6400,gate=gate))
    shutil.copyfile(root/entries[1]['path'],run/'best.pt')
    if traces:
        path=run/'iteration-0000/games.jsonl.gz';g=episode()
        write_shard(path,[g],dict(purpose='training',kind='selfplay',spec=digest(signature)))
        write_json(path.parent/'report.json',dict(execution=[dict(path=path.relative_to(root).as_posix(),sha256=sha(path),game_id=g['game_id'])]))
    return run
