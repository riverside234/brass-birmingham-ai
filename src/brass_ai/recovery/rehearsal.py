"""Rehearsal weight from committed ordinary-game and iteration progress."""
import math
from brass_ai.learning.contracts import require

DEFAULT_FLOOR=dict(contract='rehearsal-floor/1.0.0',initial=.03,hold_iterations=60,
                   decrement=.01,interval_iterations=60)


def validate(config):
    require(type(config) is dict and set(config)==set(DEFAULT_FLOOR) and
            config['contract']==DEFAULT_FLOOR['contract'],'Invalid rehearsal floor contract')
    for name in ('initial','decrement'):
        require(type(config[name]) in (int,float) and math.isfinite(config[name]) and
                0<config[name]<=.5,'Invalid rehearsal floor '+name)
    for name in ('hold_iterations','interval_iterations'):
        require(type(config[name]) is int and config[name]>0,'Invalid rehearsal floor '+name)
    return config


def schedule(config,games,iteration):
    require(type(games) is int and games>=0 and type(iteration) is int and iteration>=0,
            'Invalid committed rehearsal progress')
    floor=0.;spec=config.get('rehearsal_floor')
    if spec:
        validate(spec)
        drops=0 if iteration<spec['hold_iterations'] else 1+(iteration-spec['hold_iterations'])//spec['interval_iterations']
        floor=max(0.,round(spec['initial']-drops*spec['decrement'],12))
    decayed=config['rehearsal_weight']*max(0.,1-games/config['rehearsal_until'])
    # An explicit zero rehearsal weight still disables rehearsal for diagnostics.
    enabled=bool(config['rehearsal_manifest'] and config['rehearsal_weight'])
    return dict(weight=max(floor,decayed) if enabled else 0.,floor=floor,decayed_weight=decayed,
                committed_games=games,committed_iterations=iteration,enabled=enabled,
                scope='Weight for the next collection/update, using configured hold and interval at committed progress')
