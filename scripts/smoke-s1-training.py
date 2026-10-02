"""Quick four-player S1 budget and S1-versus-Masked v3 comparisons.

The lab table count changes throughput; two private workers change a decision.
Every table has one candidate of each budget and two rotating serial styles.
Results stay diagnostic and never enter imitation/PPO/distillation datasets.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, closing
import os
from pathlib import Path
import platform
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.bank import LAB_ROOT, read, sha
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import project_path, require
from brass_ai.learning.replay import terminal_targets
from brass_ai.step5.league import write_json

MASKED_BANK = 'bank-981744ddae4aaf07f1cf'
S1_BANK = 'bank-d82438ccec287dbc8a64'


def frozen_policy(root, bank):
    """Use the byte-verified frozen coordinator, independently of the trainer."""
    import importlib.util
    from brass_ai.bank import load_bank
    load_bank(root, bank)
    path = root / 'training/opponents/bank' / bank / 'source/src/brass_ai/recovery/legacy_search.py'
    spec = importlib.util.spec_from_file_location('brass_ai.recovery._frozen_legacy_control', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LabLegacyPolicy


def play(root, directory, index, seed, rotation, mono, compare_masked=False):
    bank = S1_BANK
    serial = dict(role='reference', seconds=1.0, search_workers=1, global_workers=32, work_multiplier=1)
    parallel = dict(role='lab', seconds=1.0, search_workers=2, global_workers=32, work_multiplier=3)
    roles = ('s1-16x2', 'masked-16x2') if compare_masked else ('serial', '16x2')
    labels = [*roles, ('cotton', 'goods-beer', 'supplier', 'rail-tempo')[index % 4],
              ('goods-beer', 'supplier', 'rail-tempo', 'cotton')[index % 4]]
    assigned = [labels[(seat - rotation) % 4] for seat in range(4)]
    began = time.monotonic()
    with ExitStack() as stack:
        policies = []
        for label in assigned:
            if label == 'masked-16x2':
                policy = frozen_policy(root, MASKED_BANK)(root, MASKED_BANK + '/balanced-v3-masked-strength',
                    parallel, mono=mono, purpose='diagnostic')
            else:
                policy = frozen_policy(root, S1_BANK)(root,
                    bank + '/' + ('balanced' if label in roles else label) + '-v3-lab-1s',
                    parallel if label in ('16x2', 's1-16x2') else serial, mono=mono, purpose='diagnostic')
            policies.append(stack.enter_context(closing(policy)))
        host = stack.enter_context(RulesHost(root / 'artifacts/r1-host/guarded', mono=mono))
        context = host.reset(seed)
        states = [dict(turn=None, intent=None, work=0) for _ in range(4)]
        turns = []
        while not context['terminal']:
            require(len(turns) < 160, 'Game failed to progress')
            seat = context['actor']; view = host.observe(seat); choices = host.choices()
            state = states[seat]; key = (context['era'], context['round'], context['turn'], seat)
            if state['turn'] != key:
                state.update(turn=key, work=0, seed=seed, deadline=time.monotonic() + 1.0)
            started = time.monotonic()
            plan = policies[seat].decision(view, choices, state)
            seconds = time.monotonic() - started
            state['intent'] = plan.get('intent')
            require(plan['actions'] and all(plan['actions']), 'Empty S1 continuation')
            for action in plan['actions']:
                for identity in action:
                    legal = host.choices(); options = {c['id']: c for c in legal}
                    require(identity in options, 'Illegal S1 complete-turn continuation')
                    context = host.apply(context, options[identity])
            detail = policies[seat].last_decision
            turns.append(dict(seat=seat, seconds=seconds,
                nodes=detail.get('nodes', policies[seat].telemetry['nodes'] -
                    sum(t['nodes'] for t in turns if t['seat']==seat)),
                unique_plans=detail.get('unique_plans', len(detail.get('plans', {}))),
                partitions=detail.get('partitions', len(policies[seat].workers)),
                work_allowance=detail.get('work_allowance')))
        result = host.call('result'); targets = terminal_targets(result)
    metrics = {}
    for label in roles:
        seat = assigned.index(label); decisions = [t for t in turns if t['seat'] == seat]
        metrics[label] = dict(vp=targets['vp'][seat], win_share=targets['win_share'][seat],
            first_place=int(targets['placement'][seat] == 1), rank=targets['placement'][seat],
            nodes=sum(t['nodes'] for t in decisions), turns=len(decisions),
            mean_turn_seconds=statistics.mean(t['seconds'] for t in decisions),
            max_turn_seconds=max(t['seconds'] for t in decisions),
            unique_plans=sum(t['unique_plans'] for t in decisions))
    report = dict(format='v3-s1-budget-match/1.0.0', completed=True, index=index,
        seed=seed, rotation=rotation, lineup=assigned, bank=bank,
        budgets=dict(serial=serial, lab_16x2=parallel), result=result, metrics=metrics,
        turns=turns, seconds=time.monotonic()-began)
    write_json(directory / f'game-{index:02d}.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=LAB_ROOT)
    parser.add_argument('--mono'); parser.add_argument('--setups', type=int, default=2)
    parser.add_argument('--compare-masked', action='store_true')
    parser.add_argument('--tables', type=int, default=16)
    parser.add_argument('--seed', type=int, default=830100)
    parser.add_argument('--output', default='artifacts/s1-training-smoke')
    args = parser.parse_args(); root = args.root.resolve()
    require(args.setups > 0 and 1 <= args.tables <= 16, 'Invalid smoke size/table allowance')
    out = project_path(root, args.output)
    require(not out.exists(), 'Use a new smoke output; preserve prior evidence')
    out.mkdir(parents=True)
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=args.tables) as pool:
        jobs = {pool.submit(play, root, out, block*4+rotation, args.seed+block, rotation, args.mono, args.compare_masked):
                block*4+rotation for block in range(args.setups) for rotation in range(4)}
        for future in as_completed(jobs):
            try: rows.append(future.result())
            except Exception as error: failures.append(dict(index=jobs[future], error=str(error)))
            print(dict(completed=len(rows), failures=failures, total=args.setups*4), flush=True)
    summary = {}
    for label in (('s1-16x2', 'masked-16x2') if args.compare_masked else ('serial', '16x2')):
        values = [row['metrics'][label] for row in rows]
        if values:
            summary[label] = dict(games=len(values),
                mean_vp=statistics.mean(v['vp'] for v in values),
                win_rate=statistics.mean(v['win_share'] for v in values),
                first_place_rate=statistics.mean(v['first_place'] for v in values),
                mean_rank=statistics.mean(v['rank'] for v in values),
                mean_nodes_per_game=statistics.mean(v['nodes'] for v in values),
                mean_turn_seconds=statistics.mean(v['mean_turn_seconds'] for v in values),
                max_turn_seconds=max(v['max_turn_seconds'] for v in values))
    current = dict(bank=S1_BANK,manifest_sha256=sha(root/'training/opponents/bank'/S1_BANK/'manifest.json'))
    report = dict(format='v3-s1-budget-smoke/1.0.0', passed=len(rows)==args.setups*4 and not failures,
        games=len(rows), setups=args.setups, rotations=4, seed=args.seed,
        bank=current['bank'], manifest_sha256=current['manifest_sha256'], seconds=1.0,
        lab_preset='16 concurrent tables x 2 private S1 workers; total work multiplier 3',
        local_tables=args.tables, platform=platform.system(), logical_cpus=os.cpu_count(),
        policies=summary, failures=failures, evidence={f'game-{r["index"]:02d}.json':
            sha(out / f'game-{r["index"]:02d}.json') for r in rows},
        limitation='Small matched smoke, not strength qualification. Win rate uses official split win shares; first place counts tied rank-one finishes. Table concurrency is throughput, not extra thinking time. No neural search settings changed.')
    if args.compare_masked:
        report.update(format='v3-s1-masked-smoke/1.0.0', masked_bank=MASKED_BANK,
            selection_rule='Official win rate, then lower mean rank, then higher mean VP; exact tie keeps S1',
            selected_backend=('masked_ismcts' if
                (summary['masked-16x2']['win_rate'], -summary['masked-16x2']['mean_rank'], summary['masked-16x2']['mean_vp']) >
                (summary['s1-16x2']['win_rate'], -summary['s1-16x2']['mean_rank'], summary['s1-16x2']['mean_vp']) else 'S1') if not failures else None)
    write_json(out / 'report.json', report)
    print(report, flush=True)
    require(report['passed'], 'S1 budget smoke failed')


if __name__ == '__main__':
    main()
