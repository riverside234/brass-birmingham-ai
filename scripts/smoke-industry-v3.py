"""Matched full four-player games: revised v3 vs current Masked and old S1.

Three seeds/style pairs and four seat rotations per control. All actors have
one second and four private search workers. These games never enter training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, closing
from functools import lru_cache
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT, load_bank, read, sha
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import project_path, require
from brass_ai.learning.replay import terminal_targets
from brass_ai.recovery.industry_bank import CURRENT_CONTROL, S1_CONTROL
from brass_ai.recovery.legacy_search import LabLegacyPolicy
from brass_ai.step5.league import write_json

STYLE_PAIRS = (('balanced', 'rail-tempo'), ('cotton', 'goods-beer'), ('supplier', 'pottery-flex'))


@lru_cache(maxsize=8)
def frozen_factory(root, bank):
    load_bank(root, bank)
    path = root/'training/opponents/bank'/bank/'source/src/brass_ai/recovery/legacy_search.py'
    spec = importlib.util.spec_from_file_location('brass_ai.recovery._control_'+bank.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.LabLegacyPolicy


def profile_name(root, bank, style):
    names = [n for n in load_bank(root, bank)['payload']['profiles'] if n.startswith(style+'-v3-')]
    require(len(names) == 1, 'Ambiguous style in comparison bank')
    return names[0]


def play(root, out, new_bank, control, index, seed, rotation, mono):
    styles = STYLE_PAIRS[index//4]
    entries = [('new', styles[0]), ('old', styles[0]), ('new', styles[1]), ('old', styles[1])]
    assigned = [entries[(s-rotation) % 4] for s in range(4)]
    began = time.monotonic(); turns = []
    with ExitStack() as stack:
        policies = []
        for role, style in assigned:
            bank = new_bank if role == 'new' else control
            factory = LabLegacyPolicy if role == 'new' else frozen_factory(root, bank)
            # S1's historical node cap must not truncate its one-second clock
            # after dividing work among four owners. Its C# code stays frozen.
            budget = dict(role='reference', seconds=1., search_workers=4,
                          global_workers=60, work_multiplier=16 if bank == S1_CONTROL else 1)
            policy = factory(root, bank+'/'+profile_name(root, bank, style), budget, mono=mono, purpose='diagnostic')
            policies.append(stack.enter_context(closing(policy)))
        host = stack.enter_context(RulesHost(root/'artifacts/r1-host/guarded', mono=mono))
        context = host.reset(seed); states = [dict(intent=None, work=0, turn=None) for _ in range(4)]
        while not context['terminal']:
            require(len(turns) < 160, 'Game did not progress')
            seat = context['actor']; view = host.observe(seat); choices = host.choices(); state = states[seat]
            key = tuple(context[k] for k in ('era', 'round', 'turn', 'turn_owner'))
            if state['turn'] != key:
                state.update(turn=key, work=0, seed=seed+len(turns)*101, deadline=time.monotonic()+1.)
            started = time.monotonic(); before = policies[seat].telemetry['nodes']
            plan = policies[seat].decision(view, choices, state); elapsed = time.monotonic()-started
            require(plan['actions'] and all(plan['actions']), 'Missing complete continuation')
            state['intent'] = plan.get('intent'); events = []
            for action in plan['actions']:
                for identity in action:
                    legal = {c['id']: c for c in host.choices()}
                    require(identity in legal, 'Illegal complete continuation')
                    choice = legal[identity]
                    if choice['kind'] in ('build', 'develop', 'bonus_develop'):
                        current = host.observe(seat)
                        mat = next(m for m in current['players'][seat]['mat'] if m['type'] == choice['industry'])
                        events.append(dict(kind=choice['kind'], industry=choice['industry'],
                                           level=mat['tiles'][0]['level'], target=choice.get('target')))
                    context = host.apply(context, choice)
            detail = policies[seat].last_decision
            turns.append(dict(seat=seat, era=view['context']['era'], round=view['context']['round'],
                seconds=elapsed, nodes=policies[seat].telemetry['nodes']-before,
                actions=[a[0].split('|')[0] for a in plan['actions']], complete_actions=plan['actions'],
                events=events, cash=view['players'][seat]['cash'], income=view['players'][seat]['income'],
                intent={k:plan['intent'].get(k) for k in ('focus','target','book_family','milestone','goal_level','reason')} if plan.get('intent') else None,
                ledger=plan.get('ledger'), completed=detail.get('completed', detail.get('search', {}).get('completed', 0)),
                cpu_batching=detail.get('search', {}).get('cpu_batching', {}),
                search_timing_ms=detail.get('search', {}).get('timing_ms', {}),
                network={k:detail.get('search', {}).get(k,0) for k in ('network_candidates','network_completed','network_rejected')}))
        result = host.call('result'); targets = terminal_targets(result)
    metrics = []
    for seat, (role, style) in enumerate(assigned):
        mine = [t for t in turns if t['seat'] == seat]
        builds = [dict(e, era=t['era']) for t in mine for e in t['events'] if e['kind'] == 'build']
        sales = [sum(id.startswith('sell|') for id in a) for t in mine for a in t['complete_actions'] if a[0].startswith('sell|')]
        metrics.append(dict(role=role, style=style, seat=seat, vp=targets['vp'][seat],
            win_share=targets['win_share'][seat], first_place=int(targets['placement'][seat] == 1), rank=targets['placement'][seat],
            mean_turn_seconds=statistics.mean(t['seconds'] for t in mine), max_turn_seconds=max(t['seconds'] for t in mine),
            nodes=sum(t['nodes'] for t in mine), builds=len(builds), sum_build_level=sum(e['level'] for e in builds),
            canal_builds=sum(e['era'] == 'CanalEra' for e in builds),
            canal_level2plus=sum(e['era'] == 'CanalEra' and e['level'] >= 2 for e in builds),
            rail_builds=sum(e['era'] == 'RailEra' for e in builds),
            rail_level3plus=sum(e['era'] == 'RailEra' and e['level'] >= 3 for e in builds),
            development_actions=sum(a == 'develop' for t in mine for a in t['actions']),
            sale_actions=len(sales), sold_tiles=sum(sales), group_sale_actions=sum(n >= 2 for n in sales)))
    report = dict(index=index, control=control, seed=seed, rotation=rotation, completed=True,
                  lineup=assigned, result=result, metrics=metrics, turns=turns, seconds=time.monotonic()-began)
    write_json(out/control/f'game-{index:02d}.json', report)
    return report


def summarize(rows, role):
    seats = [m for r in rows for m in r['metrics'] if m['role'] == role]
    if not seats: return {}
    totals = {k: sum(m[k] for m in seats) for k in ('builds', 'sum_build_level', 'sale_actions', 'sold_tiles', 'group_sale_actions')}
    return dict(seat_games=len(seats), mean_vp=statistics.mean(m['vp'] for m in seats),
        win_share=sum(m['win_share'] for m in seats), win_rate=statistics.mean(m['win_share'] for m in seats),
        first_places=sum(m['first_place'] for m in seats), mean_rank=statistics.mean(m['rank'] for m in seats),
        mean_turn_seconds=statistics.mean(m['mean_turn_seconds'] for m in seats), max_turn_seconds=max(m['max_turn_seconds'] for m in seats),
        mean_build_level=totals['sum_build_level']/max(1, totals['builds']),
        mean_canal_level2plus=statistics.mean(m['canal_level2plus'] for m in seats),
        mean_rail_level3plus=statistics.mean(m['rail_level3plus'] for m in seats),
        mean_development_actions=statistics.mean(m['development_actions'] for m in seats),
        tiles_per_sale=totals['sold_tiles']/max(1, totals['sale_actions']), **totals)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--bank'); parser.add_argument('--output', default='artifacts/industry-v3/smoke-1')
    parser.add_argument('--tables', type=int, default=4); parser.add_argument('--seed', type=int, default=843100)
    parser.add_argument('--mono'); args = parser.parse_args(); root = args.root.resolve()
    bank = args.bank or read(root/'artifacts/industry-v3/candidate-bank.json')['bank']
    require(1 <= args.tables <= 24, 'Invalid table count')
    candidate = load_bank(root, bank); controls = (CURRENT_CONTROL, S1_CONTROL)
    out = project_path(root, args.output); require(not out.exists(), 'Preserve earlier smoke evidence'); out.mkdir(parents=True)
    for control in controls: load_bank(root, control); (out/control).mkdir()
    rows = []; failures = []
    with ThreadPoolExecutor(max_workers=args.tables) as pool:
        jobs = {pool.submit(play, root, out, bank, control, block*4+r, args.seed+block, r, args.mono): (control, block*4+r)
                for control in controls for block in range(3) for r in range(4)}
        for future in as_completed(jobs):
            try: rows.append(future.result())
            except Exception as e: failures.append(dict(control=jobs[future][0], index=jobs[future][1], error=str(e)))
            print(dict(completed=len(rows), failures=failures, total=24), flush=True)
    comparisons = {}
    for control in controls:
        group = [r for r in rows if r['control'] == control]
        policies = {role: summarize(group, role) for role in ('new', 'old')}; n, o = policies['new'], policies['old']
        improvements = [k for k, better in [('win_share', bool(n and o and n['win_share'] > o['win_share'])),
                        ('mean_vp', bool(n and o and n['mean_vp'] > o['mean_vp'])),
                        ('mean_rank', bool(n and o and n['mean_rank'] < o['mean_rank']))] if better]
        comparisons[control] = dict(games=len(group), runtime_files=load_bank(root, control)['payload']['runtime_files'],
            policies=policies, improved_metrics=improvements, eligible=len(group) == 12 and bool(improvements))
    passed = len(rows) == 24 and not failures; promote = passed and all(c['eligible'] for c in comparisons.values())
    report = dict(format='masked-industry-smoke/1.0.0', passed=passed, promote=promote, games=len(rows), failures=failures,
        new_bank=bank, new_runtime_files=candidate['payload']['runtime_files'], comparisons=comparisons,
        seconds=1., search_workers=4, s1_work_multiplier=16, local_tables=args.tables, lab_tables=24,
        logical_cpus=os.cpu_count(), seed=args.seed, winner='new' if promote else 'not-promoted',
        selection_rule='Against EACH control, more official win share OR higher mean VP OR lower mean numeric rank; all 24 games must finish without errors.',
        evidence={r['control']+f'/game-{r["index"]:02d}.json': sha(out/r['control']/f'game-{r["index"]:02d}.json') for r in rows},
        runner_sha256=sha(Path(__file__)), limitation='12 games per control, three setups with four seat rotations. Directional smoke; not statistically established playing strength.')
    write_json(out/'report.json', report)
    print(dict(passed=passed, promote=promote, games=len(rows), new_bank=bank, comparisons={b:{k:v for k,v in c.items() if k!='runtime_files'} for b,c in comparisons.items()}, failures=failures), flush=True)
    require(passed, 'Smoke failed')


if __name__ == '__main__': main()
