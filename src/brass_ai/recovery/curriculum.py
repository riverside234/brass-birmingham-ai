"""Persistent, bounded searched-v4 recovery window for ordinary training games."""
from brass_ai.learning.contracts import digest,require
from .config import table_kind,latest_model_recipe,v4_search_enabled


def activate_boost(state,config):
    setting=config.get('legacy_boost')
    if setting is None:return
    if 'legacy_boost' not in state:
        require(not latest_model_recipe(config) or state['games_done']==0,
                'Latest-model curriculum must start in a fresh run at zero')
        start=state['games_done']
        state['legacy_boost']=dict(start_games=start,until_games=start+sum(s['games'] for s in setting['stages']),config=digest(setting))
    boost_window(state,config)


def boost_window(state,config):
    setting=config.get('legacy_boost')
    if setting is None:return None
    window=state.get('legacy_boost')
    require(isinstance(window,dict) and set(window)=={'start_games','until_games','config'},
            'Missing/malformed persistent searched-v4 window')
    start=window['start_games'];end=window['until_games']
    require(not latest_model_recipe(config) or start==0,'Latest-model curriculum cannot be rebased')
    require(type(start) is int and type(end) is int and 0<=start<=state['games_done'] and
            end==start+sum(s['games'] for s in setting['stages']) and window['config']==digest(setting),
            'Searched-v4 window changed')
    return window


def training_table(index,config,window):
    boosted=window is not None and window['start_games']<=index<window['until_games']
    stages=config['curriculum']
    if boosted and not latest_model_recipe(config):
        end=window['start_games']
        for stage in config['legacy_boost']['stages']:
            end+=stage['games']
            if index<end:
                stages=[dict(stage,until=None)];break
    return table_kind(index,stages),boosted


def tables_before(index,kind,config,window):
    """Count a table kind in [0,index), without aliasing its quota schedule.

    Each fixed-quota segment has period 20. Counting complete cycles also
    handles resumed/partial collections and archived curriculum/boost changes
    without a new mutable counter or a scan of the whole training history.
    """
    require(type(index) is int and index>=0 and kind in ('selfplay','historical','legacy'),
            'Invalid ordinary table count')
    boundaries={0,index}
    boundaries.update(s['until'] for s in config['curriculum'] if s['until'] is not None and 0<s['until']<index)
    if window is not None and not latest_model_recipe(config):
        end=window['start_games']
        if 0<end<index:boundaries.add(end)
        for stage in config['legacy_boost']['stages']:
            end+=stage['games']
            if 0<end<index:boundaries.add(end)
    total=0;boundaries=sorted(boundaries)
    for start,stop in zip(boundaries,boundaries[1:]):
        cycle=[training_table(i,config,window)[0] for i in range(start,min(start+20,stop))]
        full,tail=divmod(stop-start,20)
        total+=full*cycle.count(kind)+cycle[:tail].count(kind)
    return total


def curriculum_report(start,done,config):
    """Absolute ordinary-game intervals, including a boundary-crossing batch."""
    stages=[];begin=0
    for stage in config['curriculum']:
        end=stage['until']
        stages.append(dict(start_games=begin,until_games=end,
            quotas={k:stage[k] for k in ('selfplay','historical','legacy')},
            ordinary_games_this_iteration=max(0,min(done,end if end is not None else done)-max(start,begin))))
        if end is not None:begin=end
    return dict(start_games=start,completed_ordinary_games=done,stages=stages,
        next_stage=next(i for i,s in enumerate(config['curriculum']) if s['until'] is None or done<s['until']),
        scope='Ordinary game tables only; teacher/evaluation games and optimizer steps do not advance this clock')


def searched_legacy(index,config,boosted):
    return v4_search_enabled(config) and (boosted or index>=config['legacy_opponents']['search_free_until_games'])


def boost_report(start,done,config,window):
    if window is None:return dict(enabled=False)
    stages=[];begin=window['start_games']
    for stage in config['legacy_boost']['stages']:
        end=begin+stage['games']
        stages.append(dict(start_games=begin,until_games=end,
            quotas={k:stage[k] for k in ('selfplay','historical','legacy')},
            ordinary_games_this_iteration=max(0,min(done,end)-max(start,begin))))
        begin=end
    return dict(enabled=True,**window,stages=stages,
        ordinary_games_this_iteration=max(0,min(done,window['until_games'])-max(start,window['start_games'])),
        remaining_games=max(0,window['until_games']-done),
        scope='Ordinary game tables; legacy tables contain one learner and three searched-v4 opponents')
