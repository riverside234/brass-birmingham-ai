"""Fresh direct-play calibration with actual causal saved-intent inputs."""
from brass_ai.learning.batching import model_id as base_id
from brass_ai.learning.contracts import digest,require
from brass_ai.learning.replay import terminal_targets
from brass_ai.learning.reward import replay_returns
from brass_ai.personality.checkpoint import model_id
from brass_ai.personality.contracts import PROFILES,context_vector
from brass_ai.personality.labels import goal_labels,blank
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.step5.league import seeds_for
from .contracts import Profile,controller,search_config


def collect_examples(root,bundle,model,parent,handoff,output,*,games,workers,seed,partition,device,mono):
    from .collection import collect,read
    require(1<=games<=32 and 1<=workers<=8 and partition in ('train','validation'),'Bounded calibration collection required')
    ident=model_id(model);pid=base_id(parent);pool=digest([pid]);config=search_config(handoff['reward'])
    provisional=dict(scope='direct-readiness-probe',utility_ready=False,model=dict(model_id=ident),reward=handoff['reward'],
        # This explicitly unqualified collection must exercise predicted goals
        # before calibrating them. It has no search tilt, teacher or promotion.
        goal_calibration=dict(temperature=1.,bias_ready=True,scope='Unqualified direct probe for calibration only'))
    provisional['id']=digest(provisional);jobs=[]
    for i,s in enumerate(seeds_for(seed,games,partition)[0]):
        profiles=[Profile(style=PROFILES[(i+j)%len(PROFILES)],alpha=0.) for j in range(4)]
        jobs.append(dict(index=i,seed=s,sampling_seed=seed+i*37,partition=partition,neural_seats=[i%4],parent_indices=[0]*4,
            profiles=[p.to_dict() for p in profiles],controllers=[controller(ident,pid,p,config,pool,decision_mode='direct')['candidate_id'] for p in profiles]))
    collect(root,bundle,output,jobs,config,tables=workers,device=device,mono=mono,decision_mode='direct',
        _calibration=(model,[parent],provisional,handoff,pool))
    manifest,pairs=read(output,teacher_game=False)
    return examples(pairs,manifest,ident,partition,handoff['reward'])


def examples(pairs,manifest,ident,partition,reward):
    require(manifest['spec']['calibration_probe'] and manifest['spec']['model_id']==ident,'Wrong calibration source')
    encoder=PreparedEncoder();data=[]
    for game,side in pairs:
        require(game['split']==partition,'Calibration split changed')
        goals=goal_labels(game);returns,_=replay_returns(game['records'],game['result'],reward);targets=terminal_targets(game['result'])
        for row,info in zip(game['records'],side['records']):
            if info['role']=='parent':continue
            require(info['model_id']==ident and info['teacher'] is None,'Calibration cannot reuse teachers/opponents')
            i=row['step'];ctx=info['context']
            data.append(dict(row=row,encoded=encoder.encode(row['view'],row['choices'],row['style'],row['intent']),personality=context_vector(ctx),
                goals=goals.get(i,blank()),utility=returns[i]['utility'],remaining=returns[i]['remaining'],targets=targets,
                game_id=game['game_id'],setup_group=game['setup_group'],split=partition,on_policy=False,
                saved_intent=ctx['saved_intent'] is not None,context_origin='actual-p2-direct-readiness'))
            require(len(data)<=32768,'Calibration example cap exceeded')
    require(data,'No learner calibration positions')
    return data
