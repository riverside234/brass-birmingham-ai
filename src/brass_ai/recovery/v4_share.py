"""One-way v4 table-share gate, driven by the existing held-out v4 panel."""
import copy
import math

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import digest,project_path,require

CONTRACT='v4-table-share-gate/1.0.0'
MIGRATION='step5-v4-share-and-critic-holdout/1.0.0'
DEFAULT=dict(contract=CONTRACT,enabled=True,source='rating-v4-panel',win_share_above=.95,
    reference_games=320,reduced_v4_games=6)


def validate(options):
    require(type(options) is dict and set(options)==set(DEFAULT)
        and options['contract']==CONTRACT and type(options['enabled']) is bool
        and options['source']=='rating-v4-panel','Invalid v4 share gate')
    require(type(options['win_share_above']) in (int,float)
        and math.isfinite(options['win_share_above']) and options['win_share_above']==.95
        and type(options['reference_games']) is int and options['reference_games']==320
        and type(options['reduced_v4_games']) is int and options['reduced_v4_games']==6,
        'V4 share gate requires >95% and six v4 tables per 320')
    return options


def configure(config):
    """Explicit reviewed upgrade: retain PPO-only and every unrelated option."""
    from .critic_holdout import quota_options
    from .value_phase import disabled_options
    require(type(config.get('value_phase')) is dict and 'holdout' in config['value_phase'],
        'V4 share continuation requires the current stratified critic recipe')
    result=quota_options(disabled_options(config))
    result['v4_share_gate']=copy.deepcopy(DEFAULT)
    return result


def initial(config,generation,games):
    return dict(contract=CONTRACT,config=digest(validate(config['v4_share_gate'])),
        origin_generation=generation,origin_games=games,activated=None,last_panel=None)


def validate_state(saved,config,generation,games):
    require(type(saved) is dict and set(saved)==set(initial(config,0,0))
        and saved['contract']==CONTRACT and saved['config']==digest(config['v4_share_gate']),
        'V4 share gate state/config differs')
    for key,limit in (('origin_generation',generation),('origin_games',games)):
        require(type(saved[key]) is int and 0<=saved[key]<=limit,'Invalid v4 gate origin')
    evidence=saved['last_panel']
    if evidence is not None:
        require(type(evidence) is dict and set(evidence)=={'generation','candidate','games','setups','win_share','panels','spec'}
            and type(evidence['generation']) is int and 0<evidence['generation']<=generation
            and type(evidence['games']) is int and evidence['games']==config['rating']['panel_setups']*4
            and type(evidence['setups']) is int and evidence['setups']==config['rating']['panel_setups']
            and type(evidence['win_share']) in (int,float) and math.isfinite(evidence['win_share'])
            and 0<=evidence['win_share']<=1
            and evidence['candidate']==f'generation-{evidence["generation"]:06d}',
            'Invalid v4 gate panel evidence')
        for key in ('panels','spec'):
            ref=evidence[key]
            require(type(ref) is dict and set(ref)=={'path','sha256'} and type(ref['path']) is str
                and type(ref['sha256']) is str and len(ref['sha256'])==64,'Invalid v4 gate panel receipt')
    activation=saved['activated']
    if activation is not None:
        require(type(activation) is dict and set(activation)=={'generation','games'}
            and type(activation['generation']) is int and type(activation['games']) is int
            and saved['origin_generation']<=activation['generation']<=generation
            and saved['origin_games']<=activation['games']<=games
            and config['v4_share_gate']['enabled']
            and evidence is not None and evidence['generation']<=activation['generation']
            and evidence['win_share']>config['v4_share_gate']['win_share_above'],
            'Invalid v4 gate activation')
    return saved


def prepare(root,state,config):
    """Read only committed panel receipts; persist the result before planning.

    On adoption use the latest completed panel, not a historical peak. Once
    triggered, retain the trigger receipt even if subsequent panels regress.
    This is a training-share decision, not deployment qualification.
    """
    options=config.get('v4_share_gate')
    if options is None:return None
    validate(options)
    saved=copy.deepcopy(state.get('v4_share_gate',initial(config,state['iteration'],state['games_done'])))
    validate_state(saved,config,state['iteration'],state['games_done'])
    if not options['enabled'] or saved['activated'] is not None:return saved
    rating=state['ratings'];slots=rating['slots']
    if not slots:return saved
    slot=max(slots,key=lambda s:s['generation'])
    require(saved['last_panel'] is None or saved['last_panel']['generation']<=slot['generation'],
        'V4 share gate rating progress went backwards')
    if saved['last_panel'] is not None and saved['last_panel']['generation']==slot['generation']:return saved
    contents={}
    for key in ('panels','spec'):
        path=project_path(root,slot[key]['path'])
        require(sha(path)==slot[key]['sha256'],'V4 share gate rating receipt changed')
        contents[key]=read(path)
    spec,panels=contents['spec'],contents['panels']
    from .plackett_luce import V4
    require(spec['generation']==slot['generation'] and spec['config']==config['rating']
        and rating['config']==config['rating'] and panels['candidate']==spec['candidate']
        and spec['agents'][V4]==rating['agents'][V4] and spec['agents'][V4]['policy'].get('search_free') is True,
        'V4 share gate reference/panel differs')
    panel=panels['panels']['v4-panel']
    saved['last_panel']=dict(generation=slot['generation'],candidate=panels['candidate'],
        games=panel['games'],setups=panel['setups'],win_share=panel['learner_win_share'],
        panels=copy.deepcopy(slot['panels']),spec=copy.deepcopy(slot['spec']))
    validate_state(saved,config,state['iteration'],state['games_done'])
    if panel['learner_win_share']>options['win_share_above']:
        saved['activated']=dict(generation=state['iteration'],games=state['games_done'])
    return validate_state(saved,config,state['iteration'],state['games_done'])


def legacy_before(index,config,window,saved):
    """Actual v4-table ordinal; remains continuous when ten tables are moved."""
    from .curriculum import tables_before
    base=tables_before(index,'legacy',config,window)
    if not saved or saved['activated'] is None:return base
    origin=saved['activated']['games']
    if index<=origin:return base
    before=tables_before(origin,'legacy',config,window)
    return before+(base-before)*6//16


def table(index,config,window,saved):
    from .curriculum import training_table
    kind,boosted=training_table(index,config,window)
    if kind=='legacy' and saved and saved['activated'] is not None and index>=saved['activated']['games']:
        if legacy_before(index+1,config,window,saved)==legacy_before(index,config,window,saved):
            kind='selfplay'
    return kind,boosted


def report(saved,composition):
    if saved is None:return None
    counts={kind:composition.get(kind,0) for kind in ('selfplay','historical','legacy')}
    total=sum(counts.values())
    return dict(**copy.deepcopy(saved),reduced=saved['activated'] is not None,
        tables=counts,table_fractions={k:n/total if total else None for k,n in counts.items()},
        history_opponent_seats=2*counts['historical'],history_pairwise_comparisons=4*counts['historical'],
        scope='One-way gate on held-out one-learner/three-v4 win share, strictly >95%; changes ordinary tables only. PFSP histories and rating lineups unchanged.')
