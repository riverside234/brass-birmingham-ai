"""Capture and replay the independent C# reference before importing Rust rules."""
from __future__ import annotations

from collections import Counter
import copy
import gzip
import json
from pathlib import Path
import random

from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json, require
from brass_ai.learning.reward import CostLedger, terminal_utility
from .contracts import (VERSION, RecordedDraws, canonical, comparison,
                        permitted_request, trusted_debug, validate_debug)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_gzip(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, separators=(',', ':'), allow_nan=False)


def read_gzip(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream: return json.load(stream)


def snapshot(host):
    value = host.call('search_snapshot')
    # Newtonsoft omits nulls; the versioned projection uses explicit nulls.
    for key in ('views', 'chance', 'result'): value.setdefault(key, None)
    return value


def advance(host, state, command):
    require(set(command) in ({'choice'}, {'card'}), 'Invalid staged transition')
    value = host.call('search_step', game_id=state['context']['game_id'],
                      revision=state['context']['revision'], **command)
    for key in ('views', 'chance', 'result'): value.setdefault(key, None)
    return value


def update_recall(recall, state):
    if state.get('views') is not None:
        for seat, view in enumerate(state['views']):
            if not recall[seat] or recall[seat][-1]['hand'] != view['own_hand']:
                recall[seat].append(dict(revision=view['context']['revision'], hand=list(view['own_hand'])))


def restore(host, checkpoint):
    host.call('restore', checkpoint=checkpoint)
    result = snapshot(host)
    require(result['checkpoint']['hash'] == checkpoint['hash'], 'Reference restore hash changed')
    return result


def coverage(state):
    return coverage_projection(canonical(state))


def coverage_projection(p):
    c = p['context']
    tags = ['era:'+c['era'], 'phase:'+c['phase']]
    if p['chance']:
        tags.append('chance:'+p['chance']['kind'])
    else:
        tags.extend('legal:'+k for k in sorted({choice['kind'] for choice in p['choices']}))
        if c['terminal']: tags.append('terminal')
        elif c['actions_remaining'] == 1 and c['round'] == 1 and c['era'] == 'CanalEra': tags.append('opening')
        if p['views']:
            view = p['views'][0]
            if view['deck_count'] == 0: tags.append('empty-deck')
            if sum('owner' in x for x in view['links']) >= 12: tags.append('dense-network')
            if len(p['choices']) >= 100: tags.append('legal-count>=100')
            if any(x['cash'] <= 5 for x in view['players']): tags.append('low-cash')
    return tags


def capture(root, out, baseline, *, games=None):
    from brass_ai.searchlab.spiel import BrassSpielGame
    root, out = Path(root), Path(out)
    require(not (out/'corpus.json').exists(), 'Corpus already sealed; choose a new output directory')
    recipe = read_json(SOURCE_ROOT/'tests/RustBackend/fixtures/recipe.json')
    count = recipe['games'] if games is None else games
    require(1 <= count <= recipe['games'], 'Invalid game count')
    source = baseline['reference_source']['digest']; reward = baseline['model']['reward']
    model = baseline['model']['model_id']; roots = []; games_out = []; counts = Counter()
    engine = BrassSpielGame(root=root, fast=True)
    try:
        host = engine.host
        for number in range(count):
            seed = recipe['first_seed']+number
            partition = 'holdout' if number in recipe['holdout_seed_offsets'] else 'development'
            host.reset(seed); state = snapshot(host); initial = copy.deepcopy(state['checkpoint'])
            recall = [[] for _ in range(4)]; ledger = CostLedger(reward)
            tape = RecordedDraws(random.Random(seed+1000000)); seen = set(); selected = 0
            game_id = f'game-{seed}'; trace_path = out/'games'/(game_id+'.jsonl.gz')
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            # Reuse only COMPLETE unsealed games with matching source/seed and
            # a terminal footer. Qualification still replays every transition.
            # An interrupted current game is recaptured from its initial seed.
            saved = None
            if trace_path.exists():
                try:
                    with gzip.open(trace_path,'rt',encoding='utf-8') as existing:
                        header=json.loads(next(existing)); last=None
                        require(header['source']==source and header['seed']==seed,'Unsealed capture source changed')
                        for line in existing: last=json.loads(line)
                    if last and 'terminal' in last: saved=last['terminal']
                except (EOFError,OSError,json.JSONDecodeError,StopIteration): pass
            if saved is not None:
                for path in sorted((out/'roots').glob(game_id+'-root-*.json.gz')):
                    f=validate_debug(read_gzip(path));require(f['source']==source,'Captured root source changed')
                    p=f['projection']
                    roots.append(dict(id=f['id'],path=path.relative_to(out).as_posix(),sha256=file_sha(path),
                        projection_digest=digest(p),partition=partition,seed=seed,step=p['context']['revision'],
                        tags=coverage_projection(p),legal_count=len(p['choices'])))
                require(sum(e['seed']==seed for e in roots)>=recipe['minimum_roots_per_game'],'Missing roots for completed game')
                with gzip.open(trace_path,'rt',encoding='utf-8') as existing:
                    initial_row=json.loads(next(existing)); previous=initial_row['projection']
                    for line in existing:
                        row=json.loads(line); counts.update(coverage_projection(previous))
                        if 'terminal' in row:break
                        if 'choice' in row['command']:
                            choice=next(c for c in previous['choices'] if c['id']==row['command']['choice'])
                            counts['chosen:'+choice['kind']]+=1
                        previous=row['after']
                games_out.append(dict(id=game_id,path=trace_path.relative_to(out).as_posix(),sha256=file_sha(trace_path),
                                      partition=partition,source=source,**saved))
                print(json.dumps(dict(reused_complete_game=number+1,roots=len(roots))),flush=True)
                continue
            def save_root(label):
                nonlocal selected
                identity = f'{game_id}-{label}-{selected:03d}'
                path = Path('roots')/(identity+'.json.gz')
                debug = trusted_debug(state, identity=identity, source=source, recall=recall, ledger=ledger.snapshot())
                write_gzip(out/path, debug)
                entry = dict(id=identity, path=path.as_posix(), sha256=file_sha(out/path),
                             projection_digest=digest(debug['projection']), partition=partition, seed=seed,
                             step=step, tags=coverage(state), legal_count=len(state['choices']))
                roots.append(entry); selected += 1
            with gzip.open(trace_path, 'wt', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(format=VERSION, source=source, seed=seed, initial=initial,
                                              projection=canonical(state)), separators=(',', ':'))+'\n')
                for step in range(recipe['maximum_steps_per_game']):
                    update_recall(recall, state)
                    if state['views'] is not None: ledger.observe(state['views'][0])
                    tags = coverage(state); counts.update(tags)
                    c = state['context']; chance = state['chance']
                    key = (c['era'], c['phase'], c['round']//3, bool(chance))
                    # Sampling is structural and predeclared, not chosen by speed.
                    if key not in seen:
                        save_root('root'); seen.add(key)
                    if c['terminal']:
                        terminal = terminal_utility(state['result'], reward).tolist()
                        result = dict(format='rust-game-result/1.0.0', seed=seed, steps=step,
                                      result=state['result'], terminal_utility=terminal,
                                      insolvency_cost=ledger.spent.tolist(),
                                      game_utility=[v-cost for v,cost in zip(terminal,ledger.spent)],
                                      ledger=ledger.snapshot())
                        stream.write(json.dumps(dict(terminal=result), separators=(',', ':'))+'\n')
                        break
                    draw_start = len(tape.tape)
                    if chance:
                        outcomes = chance['outcomes']
                        index = tape.choose('chance:'+chance['kind'], [x['card'] for x in outcomes],
                                            [x['probability'] for x in outcomes])
                        command = dict(card=outcomes[index]['card'])
                    else:
                        choices = state['choices']; families = sorted({x['kind'] for x in choices})
                        family = families[tape.choose('action-family', families)]
                        options = [x for x in choices if x['kind'] == family]
                        choice = options[tape.choose('semantic-choice', [x['id'] for x in options])]
                        ledger.chosen(state['views'][c['actor']], choice)
                        command = dict(choice=choice['id']); counts['chosen:'+family] += 1
                    after = advance(host, state, command)
                    stream.write(json.dumps(dict(step=step, command=command, draws=tape.tape[draw_start:],
                                                  after=canonical(after)), separators=(',', ':'))+'\n')
                    state = after
                else: raise ValueError('Reference game exceeded explicit step guard')
            require(selected >= recipe['minimum_roots_per_game'], 'Insufficient varied roots in game')
            games_out.append(dict(id=game_id, path=trace_path.relative_to(out).as_posix(), sha256=file_sha(trace_path),
                                  partition=partition, source=source, **result))
            print(json.dumps(dict(captured_game=number+1, steps=step, roots=len(roots))), flush=True)
        edges = capture_edges(root, out, engine, source, reward)
        roots.extend(edges)
    finally: engine.close()
    for entry in roots:
        # No hidden record is ever included in a policy request or scripted NN key.
        fixture = read_gzip(out/entry['path']); p = fixture['projection']
        if p['chance'] is not None or p['context']['terminal']: continue
        seat = p['context']['actor']
        request = permitted_request(p['views'][seat], p['choices'], fixture['recall'][seat],
                                    request_id=entry['id'], model_id=model, reward_context=fixture['ledger'])
        entry['request_digest'] = digest(request)
        write_gzip(out/'permitted'/(entry['id']+'.json.gz'), request)
    manifest = dict(format=VERSION, source=source, recipe=recipe, recipe_digest=digest(recipe),
                    root_count=len(roots), game_count=len(games_out), roots=roots, games=games_out,
                    coverage=dict(sorted(counts.items())), complete=count == recipe['games'],
                    edge_coverage=dict(sorted(Counter(tag for entry in edges for tag in entry['tags']).items())),
                    independent_expectations_sha256=file_sha(out/'independent-expectations.json'),
                    edge_transitions_sha256=file_sha(out/'edge-transitions.json.gz'),
                    trusted_artifacts='roots/ and games/; do not pass to policy or commit user saves',
                    pending_gates=['Native parity, clone/cache/cancellation, feature/NN/search traces: L03-L15'])
    require(len(roots) >= recipe['minimum_roots'] or count < recipe['games'], 'Insufficient root corpus')
    write_json(out/'corpus.json', manifest)
    return manifest


def file_sha(path):
    import hashlib
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def harness_receipt(*extra):
    names=['src/brass_ai/rust_backend/contracts.py','src/brass_ai/rust_backend/fixtures.py',
           'configs/contracts/rust-backend/1.0.0/wire.schema.json','tests/RustBackend/fixtures/recipe.json',*extra]
    return {name:file_sha(SOURCE_ROOT/name) for name in sorted(set(names))}


def receipt_current(receipt):
    return bool(receipt) and all(file_sha(SOURCE_ROOT/name)==value for name,value in receipt.items())


def capture_edges(root, out, engine, source, reward):
    """Independent arithmetic expectations plus established authored edge setups.

    These cases are composed by hand, not learned from the captured outcome.
    Existing feature fixtures supply semantic positions, not their answers.
    """
    import importlib.util
    path = SOURCE_ROOT/'tests/FeatureEngineering/fe_fixtures.py'
    spec = importlib.util.spec_from_file_location('rust_edge_positions', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    positions = module.benchmark_positions(); entries = []; transitions=[]; edge_checks=[]
    for number, case in enumerate(positions):
        state = engine.from_observation(case['view'], 830000+number)
        current = snapshot(engine.host); recall = state.recall; ledger = CostLedger(reward)
        ledger.observe(current['views'][0]); identity = 'edge-'+case['name']
        fixture = trusted_debug(current, identity=identity, source=source, recall=recall, ledger=ledger.snapshot())
        relative = Path('roots')/(identity+'.json.gz'); write_gzip(out/relative, fixture)
        entries.append(dict(id=identity, path=relative.as_posix(), sha256=file_sha(out/relative),
                            projection_digest=digest(fixture['projection']), partition='development', seed=830000+number,
                            step=0, tags=coverage(current)+['authored-edge'], legal_count=len(current['choices'])))
        specs={
            'develop-payment':('iron',None),
            'sale-payment':('beer',None),
            'rail-first-coal':('coal','Coventry:1'),
            'pending-debt':('liquidate','Dudley:0'),
            'partial-debt':('liquidate','Dudley:1')}
        if case['name'] in specs:
            kind,target=specs[case['name']]
            choice=next(c for c in current['choices'] if c['kind']==kind and (target is None or c.get('target')==target))
            after=advance(engine.host,current,dict(choice=choice['id']));view=after['views'][0]
            checks=[]
            def expect(field,expected,actual):
                require(expected==actual,'Independent edge expectation failed: '+case['name']+' '+field)
                checks.append(dict(case=case['name'],field=field,expected=expected,actual=actual))
            if case['name']=='develop-payment':
                expect('phase','continuation',after['context']['phase']);expect('pending.iron',0,view['pending']['iron'])
            elif case['name']=='sale-payment':
                expect('phase','bonus',after['context']['phase'])
                expect('Gloucester:0.beer',0,next(m['beer'] for m in view['merchants'] if m['id']=='Gloucester:0'))
            elif case['name']=='rail-first-coal':
                tile=next(s['tile'] for s in view['slots'] if s['id']=='Coventry:1')
                expect('Coventry:1.resource',0,tile['resource']);expect('Coventry:1.flipped',True,tile['flipped'])
            elif case['name']=='pending-debt':
                expect('actor',current['context']['actor'],after['context']['actor'])
                expect('Dudley:0.tile',None,next(s.get('tile') for s in view['slots'] if s['id']=='Dudley:0'))
            elif case['name']=='partial-debt':
                expect('phase','action',after['context']['phase'])
                expect('Dudley:1.tile',None,next(s.get('tile') for s in view['slots'] if s['id']=='Dudley:1'))
            transitions.append(dict(id=identity,checkpoint=fixture['checkpoint'],before=fixture['projection'],
                                    command=dict(choice=choice['id']),after=canonical(after),checks=checks))
            edge_checks.extend(checks)
    # Pin literal known rule assertions in addition to host-to-host replay.
    engine.host.reset(830101); before = snapshot(engine.host)
    seat = before['context']['actor']; choice = next(c for c in before['choices'] if c['kind'] == 'loan')
    after = advance(engine.host, before, dict(choice=choice['id']))
    a, b = before['views'][seat]['players'][seat], after['views'][seat]['players'][seat]
    checks = [dict(case='loan', field='cash_delta', expected=30, actual=b['cash']-a['cash']),
              dict(case='loan', field='income_delta', expected=-3, actual=b['income']-a['income']),
              dict(case='loan', field='discard_count_delta', expected=1, actual=len(b['discards'])-len(a['discards']))]
    # Official ties: equal VP compares income, then cash. Utility VP is capped;
    # the official winner uses the uncapped VP score.
    scores = [(200,0,1),(181,20,100),(181,20,100),(100,30,100)]
    result = dict(objective='win-share/1.0.0', seats=[dict(seat=i,vp=v,income=n,cash=c,
        rank=[1,2,2,4][i],utility=float(i==0)) for i,(v,n,c) in enumerate(scores)])
    actual = terminal_utility(result,reward).tolist()
    expected = [1+reward['placement']+reward['vp']*180,
                reward['placement']*.5+reward['vp']*180,
                reward['placement']*.5+reward['vp']*180,reward['vp']*100]
    require(all(abs(a-b)<1e-12 for a,b in zip(actual,expected)), 'Independent reward/tie expectation failed')
    checks.append(dict(case='uncapped-winner-capped-reward-tied-second',field='utility',expected=expected,actual=actual))
    require(all(x['expected']==x['actual'] for x in checks[:3]), 'Independent loan expectation failed')
    write_json(out/'independent-expectations.json',dict(source=source, checks=checks+edge_checks, passed=True))
    write_gzip(out/'edge-transitions.json.gz',dict(source=source,transitions=transitions))
    return entries


def qualify(root, out, baseline, *, replay_games=True):
    from brass_ai.searchlab.spiel import BrassSpielGame
    out = Path(out); corpus = read_json(out/'corpus.json'); source = baseline['reference_source']['digest']
    require(corpus['source']==source, 'Corpus/reference source differs')
    engine = BrassSpielGame(root=root, fast=True); roots_passed = games_passed = transitions = 0
    edge_count=0
    try:
        for entry in corpus['roots']:
            path = out/entry['path']; require(file_sha(path)==entry['sha256'], 'Fixture bytes changed: '+entry['id'])
            fixture = validate_debug(read_gzip(path)); require(fixture['source']==source, 'Fixture source differs')
            restored = restore(engine.host, fixture['checkpoint'])
            report = comparison(fixture['projection'], canonical(restored), expected_source=source, actual_source=source)
            if not report['passed']:
                write_json(out/'first-difference.json',dict(fixture=entry['id'],**report))
                raise ValueError('Reference root mismatch: '+entry['id']+' '+report['first_difference']['path'])
            roots_passed += 1
            if roots_passed%50==0:
                print(json.dumps(dict(restored_roots=roots_passed,total_roots=len(corpus['roots']))),flush=True)
        require(file_sha(out/'independent-expectations.json')==corpus['independent_expectations_sha256'],'Independent expectations changed')
        require(file_sha(out/'edge-transitions.json.gz')==corpus['edge_transitions_sha256'],'Edge trace changed')
        for edge in read_gzip(out/'edge-transitions.json.gz')['transitions']:
            state=restore(engine.host,edge['checkpoint']);require(canonical(state)==edge['before'],'Edge root differs')
            after=advance(engine.host,state,edge['command'])
            report=comparison(edge['after'],canonical(after),expected_source=source,actual_source=source)
            if not report['passed']:
                write_json(out/'first-difference.json',dict(fixture=edge['id'],**report))
                raise ValueError('Authored edge transition differs: '+report['first_difference']['path'])
            edge_count+=1
        if replay_games:
            for entry in corpus['games']:
                require(file_sha(out/entry['path'])==entry['sha256'], 'Game trace bytes changed')
                ledger = CostLedger(baseline['model']['reward'])
                with gzip.open(out/entry['path'], 'rt', encoding='utf-8') as stream:
                    header=json.loads(next(stream)); state=restore(engine.host,header['initial'])
                    require(canonical(state)==header['projection'], 'Initial reference state differs')
                    for line in stream:
                        event=json.loads(line)
                        if state['views'] is not None: ledger.observe(state['views'][0])
                        if 'terminal' in event:
                            expected=event['terminal']
                            require(state['context']['terminal'], 'Game trace ended before terminal')
                            require(state['result']==expected['result'], 'Terminal scores differ')
                            utility=terminal_utility(state['result'],baseline['model']['reward']).tolist()
                            require(utility==expected['terminal_utility'] and ledger.spent.tolist()==expected['insolvency_cost'],
                                    'Terminal utility/causal costs differ')
                            require([v-c for v,c in zip(utility,ledger.spent)]==expected['game_utility'],'Game return differs')
                            continue
                        draws=RecordedDraws(tape=event['draws'])
                        if state['chance']:
                            options=state['chance']['outcomes']
                            index=draws.choose('chance:'+state['chance']['kind'],[x['card'] for x in options],[x['probability'] for x in options])
                            require(event['command']==dict(card=options[index]['card']), 'Recorded chance differs')
                        else:
                            choices=state['choices']; families=sorted({c['kind'] for c in choices})
                            family=families[draws.choose('action-family',families)]
                            options=[c for c in choices if c['kind']==family]
                            choice=options[draws.choose('semantic-choice',[c['id'] for c in options])]
                            require(event['command']==dict(choice=choice['id']), 'Recorded action differs')
                            ledger.chosen(state['views'][state['context']['actor']],choice)
                        draws.assert_consumed(); state=advance(engine.host,state,event['command'])
                        report=comparison(event['after'],canonical(state),expected_source=source,actual_source=source)
                        if not report['passed']:
                            write_json(out/'first-difference.json',dict(game=entry['id'],step=event['step'],**report))
                            raise ValueError('Reference transition mismatch: '+report['first_difference']['path'])
                        transitions+=1
                games_passed+=1
                print(json.dumps(dict(replayed_games=games_passed, transitions=transitions)),flush=True)
    finally: engine.close()
    report=dict(format='rust-reference-qualification/1.0.0',source=source,passed=True,
                roots=roots_passed,games=games_passed,transitions=transitions,authored_edge_transitions=edge_count,
                corpus_sha256=file_sha(out/'corpus.json'),harness=harness_receipt(),scope='C# oracle replay; Rust not implemented')
    write_json(out/'qualification.json',report)
    return report
