"""Persistent, bounded searched-v4 recovery window for ordinary training games."""
from brass_ai.learning.contracts import digest,require
from .config import table_kind


def activate_boost(state,config):
    setting=config.get('legacy_boost')
    if setting is None:return
    if 'legacy_boost' not in state:
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
    require(type(start) is int and type(end) is int and 0<=start<=state['games_done'] and
            end==start+sum(s['games'] for s in setting['stages']) and window['config']==digest(setting),
            'Searched-v4 window changed')
    return window


def training_table(index,config,window):
    boosted=window is not None and window['start_games']<=index<window['until_games']
    stages=config['curriculum']
    if boosted:
        end=window['start_games']
        for stage in config['legacy_boost']['stages']:
            end+=stage['games']
            if index<end:
                stages=[dict(stage,until=None)];break
    return table_kind(index,stages),boosted


def searched_legacy(index,config,boosted):
    return boosted or index>=config['legacy_opponents']['search_free_until_games']


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
