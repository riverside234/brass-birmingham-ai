"""Replay a completed comparison to explain score/action differences.

No new AI decisions, training examples or strength games are generated. Only
recorded legal choices are applied to the pinned rules host; final official
results must exactly match. Intermediate diagnostics are descriptive.
"""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from brass_ai.bank import read, sha
from brass_ai.host import RulesHost


def industries(view, seat):
    return [s for s in view['slots'] if s.get('tile') and s['tile']['seat'] == seat]


def snapshot(view, seat):
    mine = industries(view, seat)
    selling = [s['tile'] for s in mine if s['tile']['type'] in ('CottonMill', 'Manufacturer', 'Pottery')
               and not s['tile']['flipped']]
    return dict(cash=view['players'][seat]['cash'], income=view['players'][seat]['income'],
                own_beer=sum(s['tile']['resource'] for s in mine if s['tile']['type'] == 'Brewery'),
                unsold=len(selling), unsold_vp=sum(t['vp'] for t in selling))


def awards(view, seat):
    mine = industries(view, seat)
    merchants = {m['id'].split(':')[0] for m in view['merchants']}
    flipped = [s for s in view['slots'] if s.get('tile') and s['tile']['flipped']]
    link_points = 0
    for link in view['links']:
        if link.get('owner') != seat:
            continue
        ends = set(link['regions'])
        link_points += sum(s['tile']['link_vp'] for s in flipped if s['id'].split(':')[0] in ends)
        link_points += 2 * len(merchants & ends)
    return dict(industry_vp=sum(s['tile']['vp'] for s in mine if s['tile']['flipped']),
                link_vp=link_points, bonus_vp=view['players'][seat]['bonus_vp'],
                unflipped_level1=sum(not s['tile']['flipped'] and s['tile']['level'] == 1 for s in mine),
                unflipped=sum(not s['tile']['flipped'] for s in mine), **snapshot(view, seat))


def replay(root, path, mono=None):
    game = read(path)
    stats = [Counter() for _ in range(4)]
    phase_stats = [{e: Counter() for e in ('CanalEra', 'RailEra')} for _ in range(4)]
    endings = [{} for _ in range(4)]
    scout_turns = []
    with RulesHost(root / 'artifacts/r1-host/guarded', mono=mono) as host:
        context = host.reset(game['seed'])
        for index, turn in enumerate(game['turns']):
            seat = context['actor']; era = context['era']
            assert (seat, era, context['round']) == (turn['seat'], turn['era'], turn['round'])
            counts = stats[seat]; counts['turns'] += 1; counts['nodes'] += turn['nodes']
            counts['completed_search'] += turn['completed']; counts['turn_seconds'] += turn['seconds']
            for action in turn['complete_actions']:
                view = host.observe(seat)
                start = snapshot(view, seat)
                legal = {c['id']: c for c in host.choices()}
                first = legal[action[0]]; kind = first['kind']
                counts[kind] += 1; phase_stats[seat][era][kind] += 1
                if kind == 'scout':
                    scout_turns.append((seat, index, era))
                parts = []
                for identity in action:
                    before = host.observe(seat)
                    choice = next(c for c in host.choices() if c['id'] == identity)
                    parts.append(choice)
                    if choice['kind'] == 'build':
                        tile = next(m for m in before['players'][seat]['mat'] if m['type'] == choice['industry'])['tiles'][0]
                        counts['build_' + choice['industry']] += 1
                        phase_stats[seat][era]['build_' + choice['industry']] += 1
                        phase_stats[seat][era]['build_level_sum'] += tile['level']
                        phase_stats[seat][era]['level3plus'] += tile['level'] >= 3
                        if choice.get('card', '').startswith('wild:'):
                            counts['wild_on_build'] += 1
                        if era == 'RailEra' and tile['level'] <= 2:
                            counts['rail_low_build'] += 1
                    elif choice.get('card', '').startswith('wild:'):
                        counts['wild_on_' + choice['kind']] += 1
                    if choice['kind'] == 'beer':
                        if choice['target'].startswith('merchant:'):
                            source = 'merchant'
                        else:
                            brewery = next(s for s in before['slots'] if s['id'] == choice['target'])['tile']
                            source = 'own' if brewery['seat'] == seat else 'opponent'
                        counts['beer_' + source + '_for_' + kind] += 1
                    context = host.apply(context, choice)
                    if context['era'] != era or context['terminal']:
                        after = host.observe(seat)
                        for actor in range(4):
                            ending = awards(before, actor)
                            ending['official_award'] = after['players'][actor]['vp'] - before['players'][actor]['vp']
                            # Merchant bonus VP was already banked when used.
                            assert ending['industry_vp'] + ending['link_vp'] == ending['official_award'], (path.name, actor, era, ending)
                            endings[actor][era] = ending
                if kind == 'network':
                    n = sum(c['kind'] == 'network' for c in parts)
                    counts['double_rail' if era == 'RailEra' and n == 2 else 'single_' + era] += 1
                if kind == 'sell':
                    n = sum(c['kind'] == 'sell' for c in parts)
                    counts['sold_tiles'] += n; counts['group_sales'] += n >= 2
                if kind == 'build' and start['unsold'] and start['own_beer'] == 0:
                    counts['build_with_unsold_no_own_beer'] += 1
        actual = host.call('result')
        assert actual == game['result'], 'Replay final result differs'
    rows = []
    for seat, (role, style) in enumerate(game['lineup']):
        counts = dict(stats[seat])
        for actor, at, era in scout_turns:
            if actor != seat:
                continue
            following = [t for t in game['turns'][at + 1:] if t['seat'] == seat and t['era'] == era][:2]
            # Follow-up counts show exploitation, not a causal VP uplift.
            counts['scout_build_within_two_turns'] = counts.get('scout_build_within_two_turns', 0) + any(
                'build' in t['actions'] for t in following)
        metric = next(m for m in game['metrics'] if m['seat'] == seat)
        rows.append(dict(control=game['control'], index=game['index'], role=role, style=style,
                         seat=seat, vp=metric['vp'], rank=metric['rank'], wins=metric['win_share'],
                         counts=counts, phases=phase_stats[seat], endings=endings[seat]))
    return rows


def summarize(rows):
    counts = Counter()
    for row in rows:
        counts.update(row['counts'])
    eras = {}
    for era in ('CanalEra', 'RailEra'):
        keys = {k for row in rows for k in row['endings'][era]}
        eras[era] = {k: statistics.mean(row['endings'][era][k] for row in rows) for k in sorted(keys)}
        kinds = {k for row in rows for k in row['phases'][era]}
        eras[era]['mean_actions'] = {k: statistics.mean(row['phases'][era].get(k, 0) for row in rows) for k in sorted(kinds)}
    return dict(seat_games=len(rows), first_places=sum(r['wins'] for r in rows),
                mean_vp=statistics.mean(r['vp'] for r in rows), mean_rank=statistics.mean(r['rank'] for r in rows),
                mean_nodes_per_turn=counts['nodes']/counts['turns'],
                mean_completed_search_per_turn=counts['completed_search']/counts['turns'],
                totals=dict(counts), eras=eras)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--screen', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]); parser.add_argument('--mono')
    args = parser.parse_args(); root = args.root.resolve(); directory = args.screen.resolve()
    report = read(directory/'report.json'); assert report['passed'] and report['games'] == 24
    rows = []
    for relative, fingerprint in report['evidence'].items():
        path = directory/relative; assert sha(path) == fingerprint
        rows.extend(replay(root, path, args.mono))
    groups = {}
    for control in report['comparisons']:
        group = [r for r in rows if r['control'] == control]
        groups[control] = dict(overall={role: summarize([r for r in group if r['role'] == role]) for role in ('new', 'old')},
            styles={style: {role: summarize([r for r in group if r['style'] == style and r['role'] == role]) for role in ('new', 'old')}
                    for style in sorted({r['style'] for r in group})})
    result = dict(format='masked-economy-diagnosis/1.0.0', passed=True, replayed_games=len(report['evidence']),
        original_report_sha256=sha(directory/'report.json'), script_sha256=sha(Path(__file__)),
        comparisons=groups, seats=rows,
        limitation='Descriptive matched traces; style cells have four seat-games and mechanism attribution requires an ablation, not a new seed search.')
    target = directory/'diagnosis.json'; assert not target.exists(), 'Preserve earlier diagnosis'
    import json
    target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(dict(passed=True, replayed_games=len(report['evidence']), report=str(target)), flush=True)


if __name__ == '__main__': main()
