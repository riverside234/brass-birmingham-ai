"""Deterministic whole-game critic holdouts stratified by ordinary table type."""
import copy
import hashlib
import math

from brass_ai.learning.contracts import digest,require

LEGACY_CONTRACT='ppg-critic-holdout/2.0.0'
LEGACY_DEFAULT=dict(contract=LEGACY_CONTRACT,minimum_games_per_table=2)
CONTRACT='ppg-critic-holdout/3.0.0'
DEFAULT=dict(contract=CONTRACT,minimum_games_per_table=2,reference_games=320,
    games_per_table=dict(selfplay=8,historical=8,legacy=2))
TABLES=('selfplay','historical','legacy')


def validate(options):
    modern=type(options) is dict and options.get('contract')==CONTRACT
    require(type(options) is dict and set(options)==set(DEFAULT if modern else LEGACY_DEFAULT)
        and options['contract'] in (CONTRACT,LEGACY_CONTRACT)
        and type(options['minimum_games_per_table']) is int and 1<=options['minimum_games_per_table']<=8,
        'Invalid stratified critic holdout')
    if modern:
        require(type(options['reference_games']) is int and options['reference_games']==320
            and type(options['games_per_table']) is dict and set(options['games_per_table'])==set(TABLES)
            and all(type(n) is int for n in options['games_per_table'].values())
            and options['games_per_table']==DEFAULT['games_per_table'],
            'Critic holdout requires 8 self-play, 8 history and 2 v4 games per 320')
    return options


def quota_options(options):
    result=copy.deepcopy(options);value=result['value_phase']
    require('holdout' in value,'Quota holdout requires stratified critic diagnostics')
    validate(value['holdout'])
    value['holdout']=copy.deepcopy(DEFAULT)
    return result


def select(games,fraction,options):
    validate(options)
    require(type(fraction) in (int,float) and math.isfinite(fraction) and 0<fraction<.5,
        'Invalid critic holdout fraction')
    require(type(games) is dict and all(type(g) is str and g and k in TABLES for g,k in games.items()),
        'Invalid critic holdout game/table identities')
    held=[]
    from .branching import family
    total=len({family(g) for g in games})
    for kind in TABLES:
        members=sorted({family(g) for g in games if games[g]==kind})
        require(all(g in games and games[g]==kind for g in members),'Branch parent missing from critic selection')
        members.sort(key=lambda g:(hashlib.sha256((options['contract']+':'+kind+':'+g).encode()).digest(),g))
        # Round half up, guarantee support when possible, and always leave a
        # training game in each represented stratum. A singleton trains only.
        target=(total*options['games_per_table'][kind]/options['reference_games']
            if options['contract']==CONTRACT else len(members)*fraction)
        count=min(max(0,len(members)-1),max(options['minimum_games_per_table'],math.floor(target+.5)))
        selected=set(members[:count])
        held.extend(g for g in games if family(g) in selected)
    return dict(**options,fraction=fraction,games=dict(sorted(games.items())),heldout_games=sorted(held))


def validate_selection(selection,options=None,fraction=None):
    fields=DEFAULT if type(selection) is dict and selection.get('contract')==CONTRACT else LEGACY_DEFAULT
    require(type(selection) is dict and set(selection)==set(fields)|{'fraction','games','heldout_games'},
        'Invalid critic holdout selection receipt')
    actual={k:selection[k] for k in fields}
    if options is not None:require(actual==options and selection['fraction']==fraction,'Critic holdout recipe differs')
    require(selection==select(selection['games'],selection['fraction'],actual),'Critic holdout selection differs')
    return selection


def partition(data,fraction,options):
    games={}
    for row,_,identity,_ in data:
        game=identity[0];kind=row.get('_table_kind','selfplay')
        require(game not in games or games[game]==kind,'One game has conflicting critic table types')
        games[game]=kind
    selection=select(games,fraction,options);held=set(selection['heldout_games'])
    train=[i for i,d in enumerate(data) if d[2][0] not in held]
    validation=[i for i,d in enumerate(data) if d[2][0] in held]
    by_table={}
    from .branching import family
    for kind in TABLES:
        total=len({family(g) for g,k in games.items() if k==kind})
        count=len({family(g) for g in held if games[g]==kind})
        by_table[kind]=dict(games=total,train_games=total-count,heldout_games=count,
            effective_fraction=count/total if total else None,
            support='absent' if not total else 'singleton_training_only' if total==1 else
                    'limited_population' if count<options['minimum_games_per_table'] else 'ready')
    return train,validation,dict(**copy.deepcopy(options),fraction=fraction,
        train_rows=len(train),heldout_rows=len(validation),
        train_games=len(games)-len(held),heldout_games=len(held),heldout_games_digest=digest(sorted(held)),
        by_table=by_table,selection=selection,
        parent_games=len({family(g) for g in games}),
        heldout_parent_games=len({family(g) for g in held}),
        scope='Outcome-independent hash ranking within table type; complete games excluded from all critic fits and caches; actor still uses these games. Rare strata oversampled in held-out diagnostics.')
