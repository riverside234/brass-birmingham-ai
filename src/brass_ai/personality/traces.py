"""Read unchanged base replays and bind optional actual personality contexts."""
from pathlib import Path
import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,read_json,require
from brass_ai.learning.replay import terminal_targets
from brass_ai.learning.reward import replay_returns
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.replay import read_shards
from .contracts import TRACE,contract,context,context_vector,local_file
from .labels import goal_labels,blank


def examples(games,reward,*,metadata=None,learner_only=False,roots_only=False,max_examples=32768):
    encoder=PreparedEncoder();output=[]
    require(not learner_only or metadata is not None,'PPO requires actual collected personality context')
    for game in games:
        goals=goal_labels(game);returns,_=replay_returns(game['records'],game['result'],reward)
        targets=terminal_targets(game['result'])
        extra=metadata['contexts'][game['game_id']] if metadata else None
        if extra:
            require(len(extra)==len(game['records']) and [x['step'] for x in extra]==list(range(len(extra))),
                    'Missing/changed personality decision attribution')
            require(len({r['seat'] for r,x in zip(game['records'],extra) if x['role']=='learner'})==1,
                    'P1 requires exactly one learner seat')
        for row in game['records']:
            i=row['step'];info=extra[i] if extra else None
            if info:
                require(info['role'] in ('learner','incumbent'),'Unknown policy role')
                # A matching ID and scalar likelihood are not enough to make
                # externally selected/search actions on-policy. Bind the same
                # sampling recipe used by collection before assigning PPO rows.
                require(row['behavior']['kind']=='model' and row['behavior']['temperature']==1.
                    and row['teacher'] is None,'P1 requires unsearched model behavior at temperature 1')
                if info['role']=='learner':
                    require(info['model_id']==metadata['spec']['model_id'] and info['context'] is not None
                        and info['context']['saved_intent'] is None,'Wrong learner/context; P1 has no committed goal controller')
                    expected='personality-pilot/'+digest(dict(model=info['model_id'],profile=info['context']['profile'],exploration=.05))
                    require(row['behavior']['policy_id']==expected and row['teacher'] is None,'Relabeled learner behavior')
                else:
                    require(info['context'] is None and info['model_id']==metadata['spec']['parent_id']
                        and row['behavior']['policy_id']=='personality-parent/'+info['model_id'],'Relabeled incumbent behavior')
            if learner_only and info['role']!='learner':continue
            if roots_only and i not in goals:continue
            # Missing legacy context is an explicit neutral extension only for
            # auxiliary supervision. It is never claimed as on-policy input.
            z=info['context'] if info and info['context'] is not None else context()
            e=game.encoded(row,encoder) if hasattr(game,'encoded') else encoder.encode(row['view'],row['choices'],row['style'],row['intent'])
            output.append(dict(row=row,encoded=e,personality=context_vector(z),goals=goals.get(i,blank()),
                utility=returns[i]['utility'],remaining=returns[i]['remaining'],targets=targets,
                game_id=game['game_id'],setup_group=game['setup_group'],split=game['split'],
                on_policy=bool(info and info['role']=='learner' and game['split']=='train'),context_origin='recorded' if info else 'legacy-absent-auxiliary-only'))
            require(len(output)<=max_examples,'P1 trace cap exceeded; select fewer shards/games')
    require(output,'No eligible personality examples')
    return output


def read_collection(directory,model_identity=None,*,partition='train',reward):
    directory=Path(directory);meta=read_json(directory/'manifest.json')
    require(meta['format']==TRACE and meta['spec']['contract']==contract() and meta['spec_id']==digest(meta['spec']),
            'Personality trace contract changed')
    require(meta['contexts_id']==digest(meta['contexts']),'Collected personality contexts changed')
    require(meta['spec']['partition']==partition and meta['spec']['exploration']==.05 and meta['spec']['temperature']==1.,
            'Wrong trace partition/behavior recipe')
    if model_identity is not None:require(meta['spec']['model_id']==model_identity,'PPO requires fresh current-model rollouts')
    path=local_file(directory,meta['replay']['path']);require(sha(path)==meta['replay']['sha256'],'Personality replay changed')
    games=read_shards([path],partition=partition)
    require(len(games)==meta['spec']['games'] and {g['game_id'] for g in games}==set(meta['contexts'])
        and all(g['split']==partition for g in games),'Incomplete/foreign personality games')
    return examples(games,reward,metadata=meta,learner_only=True),meta,games
