"""One-way v4 table-share gate, driven by the existing held-out v4 panel."""
import copy
import math

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import digest,project_path,require

LEGACY_CONTRACT='v4-table-share-gate/1.0.0'
LEGACY_MIGRATION='step5-v4-share-and-critic-holdout/1.0.0'
LEGACY_DEFAULT=dict(contract=LEGACY_CONTRACT,enabled=True,source='rating-v4-panel',win_share_above=.95,
    reference_games=320,reduced_v4_games=6)
CONTRACT='v4-table-share-gate/2.0.0'
MIGRATION='step5-v4-share-to-history/1.0.0'
DEFAULT=dict(LEGACY_DEFAULT,contract=CONTRACT,released_tables='historical')
CURRENT_CONTRACT='v4-table-share-gate/3.0.0'
CURRENT_DEFAULT=dict(contract=CURRENT_CONTRACT,enabled=True,source='rating-v4-panel',
    win_share_at_least=.9,reference_games=320,reduced_v4_games=6,released_tables='historical')
HISTORY_CONTRACTS=(CONTRACT,CURRENT_CONTRACT)


def reached(options, win_share):
    return win_share>=options['win_share_at_least'] if options['contract']==CURRENT_CONTRACT else win_share>options['win_share_above']


def validate(options):
    require(type(options) is dict,'Invalid v4 share gate')
    template=(LEGACY_DEFAULT if options.get('contract')==LEGACY_CONTRACT else
        CURRENT_DEFAULT if options.get('contract')==CURRENT_CONTRACT else DEFAULT)
    require(set(options)==set(template)
        and options['contract']==template['contract'] and type(options['enabled']) is bool
        and options['source']=='rating-v4-panel','Invalid v4 share gate')
    if options['contract'] in HISTORY_CONTRACTS:
        require(options['released_tables']=='historical','Released v4 tables must go to history')
    threshold='win_share_at_least' if options['contract']==CURRENT_CONTRACT else 'win_share_above'
    require(type(options[threshold]) in (int,float)
        and math.isfinite(options[threshold]) and options[threshold]==template[threshold]
        and type(options['reference_games']) is int and options['reference_games']==320
        and type(options['reduced_v4_games']) is int and options['reduced_v4_games']==6,
        'V4 share gate threshold/size differs from its versioned contract')
    return options


def configure_legacy(config):
    """Reproduce the original signed migration without changing its meaning."""
    from .critic_holdout import quota_options
    from .value_phase import disabled_options
    require(type(config.get('value_phase')) is dict and 'holdout' in config['value_phase'],
        'V4 share continuation requires the current stratified critic recipe')
    result=quota_options(disabled_options(config))
    result['v4_share_gate']=copy.deepcopy(LEGACY_DEFAULT)
    return result


def configure(config):
    """Explicit reviewed upgrade: move released v4 tables into the PFSP pool."""
    result=configure_legacy(config)
    result['v4_share_gate']=copy.deepcopy(DEFAULT)
    if 'v4_share_gate' in config:
        validate(config['v4_share_gate'])
        result['v4_share_gate']['enabled']=config['v4_share_gate']['enabled']
    return result


def configure_90(config):
    result=copy.deepcopy(config)
    previous=config.get('v4_share_gate')
    if previous is not None:validate(previous)
    result['v4_share_gate']=dict(CURRENT_DEFAULT,enabled=previous['enabled'] if previous else True)
    return result


def initial(config,generation,games):
    options=validate(config['v4_share_gate'])
    result=dict(contract=options['contract'],config=digest(options),
        origin_generation=generation,origin_games=games,activated=None,last_panel=None)
    if options['contract'] in HISTORY_CONTRACTS:
        result['allocation_origin']=dict(generation=generation,games=games)
    return result


def validate_state(saved,config,generation,games):
    require(type(saved) is dict and set(saved)==set(initial(config,0,0))
        and saved['contract']==config['v4_share_gate']['contract'] and saved['config']==digest(config['v4_share_gate']),
        'V4 share gate state/config differs')
    for key,limit in (('origin_generation',generation),('origin_games',games)):
        require(type(saved[key]) is int and 0<=saved[key]<=limit,'Invalid v4 gate origin')
    if saved['contract'] in HISTORY_CONTRACTS:
        origin=saved['allocation_origin']
        require(type(origin) is dict and set(origin)=={'generation','games'}
            and type(origin['generation']) is int and type(origin['games']) is int
            and saved['origin_generation']<=origin['generation']<=generation
            and saved['origin_games']<=origin['games']<=games,'Invalid v4 allocation origin')
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
            and reached(config['v4_share_gate'],evidence['win_share']),
            'Invalid v4 gate activation')
    return saved


def migrate_state(state,config):
    """Preserve the latch and evidence; never count past v1 transfers as history."""
    saved=state.get('v4_share_gate')
    if saved is None:return None,None
    before=state['signature']['config'];generation=state['iteration'];games=state['games_done']
    validate_state(saved,before,generation,games)
    if before['v4_share_gate']==config.get('v4_share_gate'):
        return copy.deepcopy(saved),None
    current=config.get('v4_share_gate',{}).get('contract')==CURRENT_CONTRACT
    require((before['v4_share_gate']['contract'] in (LEGACY_CONTRACT,CONTRACT) if current else
        before['v4_share_gate']['contract']==LEGACY_CONTRACT)
        and config.get('v4_share_gate')==(configure_90(before) if current else configure(before))['v4_share_gate'],
        'Unreviewed v4 share state migration')
    result=copy.deepcopy(saved)
    result.update(contract=config['v4_share_gate']['contract'],config=digest(config['v4_share_gate']),
        allocation_origin=copy.deepcopy(saved.get('allocation_origin',dict(generation=generation,games=games))))
    # Re-read the latest committed receipt at 90%; an older inspected panel
    # must not activate the gate if a newer panel has regressed.
    if current and result['activated'] is None:result['last_panel']=None
    validate_state(result,config,generation,games)
    return result,dict(contract='v4-share-threshold-90/1.0.0' if current else MIGRATION,from_digest=digest(saved),to_digest=digest(result),
        generation=generation,games=games,
        preserved='Gate origin, activation and triggering panel; PFSP S/C, membership and all training state',
        released_tables='historical')


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
    retired=rating.get('panel_gates',{}).get('retired',{}).get('v4-panel') if 'panel_gates' in config else None
    if retired is not None:
        # A resumed rating backlog can retire the panel, then commit a later
        # main-only slot before ordinary collection reaches this gate. Use the
        # verified retirement trigger, not the later slot's absent panel.
        from . import panel_gates
        require(panel_gates.prepare(root,state,config)['retired']['v4-panel']==retired,
            'V4 share gate retirement evidence differs')
        saved['last_panel']={key:copy.deepcopy(retired[key]) for key in
            ('generation','candidate','games','setups','win_share','panels','spec')}
        saved['activated']=dict(generation=state['iteration'],games=state['games_done'])
        return validate_state(saved,config,state['iteration'],state['games_done'])
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
    if reached(options,panel['learner_win_share']):
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
            kind=('historical' if saved['contract'] in HISTORY_CONTRACTS
                and index>=saved['allocation_origin']['games'] else 'selfplay')
    return kind,boosted


def history_before(index,config,window,saved):
    """History ordinal includes newly transferred tables, never earlier v1 work."""
    from .curriculum import tables_before
    base=tables_before(index,'historical',config,window)
    if not saved or saved['contract'] not in HISTORY_CONTRACTS or saved['activated'] is None:return base
    origin=max(saved['allocation_origin']['games'],saved['activated']['games'])
    if index<=origin:return base
    released=(tables_before(index,'legacy',config,window)-tables_before(origin,'legacy',config,window)
        -legacy_before(index,config,window,saved)+legacy_before(origin,config,window,saved))
    return base+released


def report(saved,composition):
    if saved is None:return None
    counts={kind:composition.get(kind,0) for kind in ('selfplay','historical','legacy')}
    total=sum(counts.values())
    return dict(**copy.deepcopy(saved),reduced=saved['activated'] is not None,
        tables=counts,table_fractions={k:n/total if total else None for k,n in counts.items()},
        history_opponent_seats=2*counts['historical'],history_pairwise_comparisons=4*counts['historical'],
        released_tables='historical' if saved['contract'] in HISTORY_CONTRACTS else 'selfplay',
        threshold=.9 if saved['contract']==CURRENT_CONTRACT else .95,
        comparison='>=' if saved['contract']==CURRENT_CONTRACT else '>',
        scope='One-way held-out one-learner/three-v4 gate. Every actual history seat uses the frozen PFSP plan; main rating lineups unchanged.')
