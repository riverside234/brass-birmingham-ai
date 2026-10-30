"""Immutable context receipts, independent of model/action/setup RNG streams."""
import copy
import random
from brass_ai.learning.contracts import digest,require
from brass_ai.step5.policy import context_for
from .domino_config import CONTEXT,ENCODING,NAMES


def strategy(seed,index,seat):
    require(all(type(n) is int for n in (seed,index,seat)) and index>=0 and 0<=seat<4,'Invalid strategy stream key')
    return random.Random(digest(dict(contract=CONTEXT,seed=seed,index=index,seat=seat))).randrange(7)


def assignments(policies,rotation,*,seed,index,learner_id,current,checkpoints,learner_checkpoint=None):
    result=[]
    for seat in range(4):
        p=policies[(seat-rotation)%4]
        if p['kind']!='neural': result.append(None);continue
        z=strategy(seed,index,seat);role='learner' if p['policy_id']==learner_id else 'frozen'
        mid=p['model_id'];require(role!='learner' or mid==current,'DOMiNO learner weights differ')
        require(mid in checkpoints,'Missing DOMiNO checkpoint identity')
        family=digest(dict(policy_id=p['policy_id'],context_contract=CONTEXT,context_encoding=ENCODING))
        record=dict(contract=CONTEXT,model_id=mid,checkpoint_sha256=learner_checkpoint if role=='learner' and learner_checkpoint is not None else checkpoints[mid],training_role=role,
            actor_origin='incoming-learner' if role=='learner' else 'frozen-opponent',
            legacy_policy_id=p['policy_id'],behavior_family_id=family,strategy_id=z,context=NAMES[z],
            context_encoding=ENCODING,ordinary_game_index=index,physical_seat=seat,probability=1/7)
        record['seat_policy_id']=digest(dict(behavior_family_id=family,strategy_id=z))
        record['receipt']=digest(record);result.append(record)
    return dict(contract=CONTEXT,seats=result)


def validate_assignment(record):
    require(type(record) is dict and record.get('contract')==CONTEXT and record.get('context_encoding')==ENCODING,
        'Missing/unsupported ordinary strategy receipt')
    require(type(record['strategy_id']) is int and 0<=record['strategy_id']<7 and
        record['context']==NAMES[record['strategy_id']] and record['probability']==1/7,
        'Invalid strategy identity/probability')
    require(record['receipt']==digest({k:v for k,v in record.items() if k!='receipt'}), 'Strategy receipt changed')
    require(record['behavior_family_id']==digest(dict(policy_id=record['legacy_policy_id'],context_contract=CONTEXT,context_encoding=ENCODING)) and
        record['seat_policy_id']==digest(dict(behavior_family_id=record['behavior_family_id'],strategy_id=record['strategy_id'])),
        'Strategy behavior identity differs')
    require(record['training_role'] in ('learner','frozen') and record['actor_origin']==
        ('incoming-learner' if record['training_role']=='learner' else 'frozen-opponent'),'Invalid strategy training role')
    require(type(record['physical_seat']) is int and 0<=record['physical_seat']<4 and
        type(record['ordinary_game_index']) is int and record['ordinary_game_index']>=0,
        'Invalid strategy seat/progress')
    return record


def bind(game,proof):
    assignment=copy.deepcopy(proof['diversity_assignment'])
    require(assignment['contract']==CONTEXT and len(assignment['seats'])==4,'Invalid strategy table')
    learner=proof['learner_behavior']
    for seat,record in enumerate(assignment['seats']):
        rows=[r for r in game['records'] if r['seat']==seat]
        table=proof.get('critic_table')
        if table is not None:
            planned=table['seats'][seat]
            require((record is not None)==(planned['role'] in ('learner','snapshot','il')) and
                planned.get('strategy')==record,'Critic strategy differs from collection assignment')
        if record is None:
            require(all(r['behavior']['kind']!='model' for r in rows),'Neural seat lacks strategy assignment')
            continue
        validate_assignment(record)
        require(record['physical_seat']==seat and record['ordinary_game_index']==proof['ordinary_game_index'] and
            game['opponents'][seat]==record['legacy_policy_id'], 'Strategy seat/progress differs')
        if record['training_role']=='learner':
            require(record['model_id']==learner['model_id'] and record['legacy_policy_id']==learner['policy_id'],
                    'Strategy actor attribution differs from incoming learner')
        else:require(record['legacy_policy_id']!=learner['policy_id'],'Frozen strategy must retain its separate behavior family')
        style,intent=context_for(record['context'])
        for row in rows:
            require(row['style']==style and row['intent']==intent and row['behavior']['kind']=='model' and
                row['behavior']['policy_id']==record['legacy_policy_id'], 'Collected strategy/context does not match immutable assignment')
    game.diversity_assignment=assignment
    return assignment


def learner_row(row,policy_id):
    a=row.get('_domino')
    if a is None:return row['behavior']['policy_id']==policy_id
    validate_assignment(a)
    return a['training_role']=='learner' and a['legacy_policy_id']==policy_id and row['behavior']['policy_id']==policy_id


def install(job,assigned,rngs):
    a=job['provenance'].get('diversity_assignment')
    if a is None:return
    require(a['contract']==CONTEXT and len(a['seats'])==len(assigned)==len(rngs)==4,'Invalid strategy table')
    # Validate every seat before changing any live policy context. A missing
    # receipt must not silently select the old mixed cycle or neutral history.
    for seat,record in enumerate(a['seats']):
        require((record is not None)==(assigned[seat].kind=='model'),'Neural seat lacks strategy assignment')
        if record is None:continue
        validate_assignment(record);p=assigned[seat]
        require(record['physical_seat']==seat and record['ordinary_game_index']==job['provenance']['ordinary_game_index'] and
            p.policy_id==record['legacy_policy_id'] and p.evaluator.model_id==record['model_id'],
            'Strategy routed to wrong weights/seat')
    for seat,record in enumerate(a['seats']):
        if record is None:continue
        p=assigned[seat]
        p.contexts[rngs[seat]]=context_for(record['context'])
