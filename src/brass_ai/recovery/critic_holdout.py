"""Deterministic whole-game critic holdouts stratified by ordinary table type."""
import hashlib
import math

from brass_ai.learning.contracts import digest,require

CONTRACT='ppg-critic-holdout/2.0.0'
DEFAULT=dict(contract=CONTRACT,minimum_games_per_table=2)
TABLES=('selfplay','historical','legacy')


def validate(options):
    require(type(options) is dict and set(options)==set(DEFAULT) and options['contract']==CONTRACT
        and type(options['minimum_games_per_table']) is int and 1<=options['minimum_games_per_table']<=8,
        'Invalid stratified critic holdout')
    return options


def select(games,fraction,options):
    validate(options)
    require(type(fraction) in (int,float) and math.isfinite(fraction) and 0<fraction<.5,
        'Invalid critic holdout fraction')
    require(type(games) is dict and all(type(g) is str and g and k in TABLES for g,k in games.items()),
        'Invalid critic holdout game/table identities')
    held=[]
    from .branching import family
    for kind in TABLES:
        members=sorted({family(g) for g in games if games[g]==kind})
        require(all(g in games and games[g]==kind for g in members),'Branch parent missing from critic selection')
        members.sort(key=lambda g:(hashlib.sha256((CONTRACT+':'+kind+':'+g).encode()).digest(),g))
        # Round half up, guarantee support when possible, and always leave a
        # training game in each represented stratum. A singleton trains only.
        count=min(max(0,len(members)-1),max(options['minimum_games_per_table'],math.floor(len(members)*fraction+.5)))
        selected=set(members[:count])
        held.extend(g for g in games if family(g) in selected)
    return dict(**options,fraction=fraction,games=dict(sorted(games.items())),heldout_games=sorted(held))


def validate_selection(selection,options=None,fraction=None):
    require(type(selection) is dict and set(selection)==set(DEFAULT)|{'fraction','games','heldout_games'},
        'Invalid critic holdout selection receipt')
    actual={k:selection[k] for k in DEFAULT}
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
    return train,validation,dict(contract=CONTRACT,fraction=fraction,
        minimum_games_per_table=options['minimum_games_per_table'],train_rows=len(train),heldout_rows=len(validation),
        train_games=len(games)-len(held),heldout_games=len(held),heldout_games_digest=digest(sorted(held)),
        by_table=by_table,selection=selection,
        parent_games=len({family(g) for g in games}),
        heldout_parent_games=len({family(g) for g in held}),
        scope='Outcome-independent hash ranking within table type; complete games excluded from all critic fits and caches; actor still uses these games. Rare strata oversampled in held-out diagnostics.')
