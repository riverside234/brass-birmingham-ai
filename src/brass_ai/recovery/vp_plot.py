"""Ordinary self-play learner VP, pooled into fixed ten-generation groups.

This is reporting evidence only: no rating refit, policy inference or training
state changes. A small cumulative history survives bounded report retention.
"""
import gzip
import json
import math
from pathlib import Path
import re

from .rating_plot import DEFAULT_MAX_POINTS, RECENT_GENERATIONS, display_points, _endpoint_label, _ticks


FORMAT = 'selfplay-vp-history/1.0.0'
GROUP_SIZE = 10
HISTORY_PATH = Path('ratings/vp-history.json')


def _read(path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8-sig') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _initial(state):
    entry = next((e for e in state.get('history', []) if e.get('generation') == 0), {})
    return {k: entry[k] for k in ('path', 'sha256', 'model_id') if k in entry}


def _totals(report):
    # Full reports and compact/recent history have different containers. Never
    # use top-level health_counts: those include history/v4 learner seats.
    composition = report.get('by_composition', {}).get('selfplay', {})
    counts = composition.get('health_counts') or report.get('health_by_composition', {}).get('selfplay')
    if not counts or 'vp_sum' not in counts or 'seat_games' not in counts:
        seats = (composition.get('learner_by_seat') or
                 report.get('learner_by_composition_seat', {}).get('selfplay', {}))
        if not seats or not all('vp_sum' in row and 'seat_games' in row for row in seats.values()):
            return None
        counts = dict(vp_sum=sum(row['vp_sum'] for row in seats.values()),
                      seat_games=sum(row['seat_games'] for row in seats.values()))
    n, total = counts['seat_games'], counts['vp_sum']
    if type(n) is not int or n < 0 or type(total) not in (int, float) or not math.isfinite(total):
        raise ValueError('Invalid self-play learner VP totals')
    if n == 0:
        return None
    row = dict(vp_sum=float(total), seat_games=n)
    if 'tables' in counts:
        if type(counts['tables']) is not int or counts['tables'] < 0:
            raise ValueError('Invalid self-play table count')
        row['tables'] = counts['tables']
    return row


def _merge(rows, row):
    generation = row['generation']
    old = rows.get(generation)
    if old is not None:
        if (old['seat_games'] != row['seat_games'] or
                not math.isclose(old['vp_sum'], row['vp_sum'], rel_tol=1e-12, abs_tol=1e-7) or
                ('tables' in old and 'tables' in row and old['tables'] != row['tables'])):
            raise ValueError(f'Conflicting self-play VP evidence for generation {generation}')
        row = dict(old, **row)
    rows[generation] = row


def collect(run, *, state=None, history_sources=()):
    """Read committed reports; optional older bundles must share the lineage.

    Iteration N's report belongs to committed generation N+1. The state counter
    is authoritative, so an interrupted, not-yet-committed collection is ignored.
    """
    run = Path(run)
    state = _read(run/'state.json') if state is None else state
    committed = state['iteration']
    if type(committed) is not int or committed < 0:
        raise ValueError('Invalid committed generation for VP history')
    initial = _initial(state)
    rows = {}
    for source in [*(Path(p) for p in history_sources), run]:
        saved = state if source == run else _read(source/'state.json')
        if source != run and (not initial or _initial(saved) != initial):
            raise ValueError('VP history source belongs to a different run lineage')
        limit = min(committed, saved['iteration'])
        cache = source/HISTORY_PATH
        if cache.is_file():
            history = _read(cache)
            if history.get('format') != FORMAT or history.get('initial_checkpoint') != initial:
                raise ValueError('Incompatible self-play VP history')
            for item in history['iterations']:
                generation = item['generation']
                if type(generation) is not int or generation < 1:
                    raise ValueError('Invalid VP history generation')
                if generation <= limit:
                    totals = _totals({'health_by_composition': {'selfplay': item}})
                    if totals is not None:
                        _merge(rows, dict(generation=generation, **totals))

        def add(report, iteration=None):
            index = report.get('iteration')
            if type(index) is not int or (iteration is not None and index != iteration):
                raise ValueError('Iteration identity differs in self-play VP history')
            if not 0 <= index < limit:
                return
            totals = _totals(report)
            if totals is not None:
                _merge(rows, dict(generation=index+1, **totals))

        retained = source/'retention-history.json'
        if not retained.is_file():
            retained = retained.with_suffix('.json.gz')
        if retained.is_file():
            for name, entry in _read(retained).items():
                match = re.fullmatch(r'iteration-(\d+)/report\.json', name)
                if match and int(match[1]) < limit and 'summary' in entry:
                    add(entry['summary'], int(match[1]))
        for path in sorted(source.glob('iteration-*/report.json')):
            match = re.fullmatch(r'iteration-(\d+)', path.parent.name)
            if match and int(match[1]) < limit:
                add(_read(path), int(match[1]))
        for report in saved.get('recent_training', []):
            add(report)
    return dict(format=FORMAT, initial_checkpoint=initial, committed_generation=committed,
                iterations=[rows[g] for g in sorted(rows)])


def save_history(history, path):
    """Replace only this small derived report, never a checkpoint or state.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(history, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
    temporary = path.with_suffix('.json.tmp')
    try:
        temporary.write_text(raw, encoding='utf-8', newline='\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def groups(history, generation):
    """Pool VP sums and seat counts, not means of means; never fill missing data."""
    end = min(generation, history['committed_generation'])
    rows = {r['generation']: r for r in history['iterations'] if r['generation'] <= end}
    result = []
    for last in range(GROUP_SIZE, end+1, GROUP_SIZE):
        first = last-GROUP_SIZE+1
        part = [rows[g] for g in range(first, last+1) if g in rows]
        complete = len(part) == GROUP_SIZE
        seats = sum(r['seat_games'] for r in part)
        total = math.fsum(r['vp_sum'] for r in part)
        result.append(dict(first=first, last=last, iterations=len(part), seat_games=seats,
                           vp_sum=total, mean_vp=total/seats if complete and seats else None,
                           tables=sum(r['tables'] for r in part) if all('tables' in r for r in part) else None))
    return result


def display_groups(data, max_points=DEFAULT_MAX_POINTS):
    """Bound the overview using PL's adjacent buckets, pooling VP by seats.

    First/latest available groups stay exact. Missing groups are omitted before
    pooling; the recent panel still shows the original groups and their gaps.
    """
    data = [row for row in data if row['mean_vp'] is not None]
    if not data:
        return []
    buckets = display_points([r['last'] for r in data], [0.] * len(data), max_points)
    result = []; offset = 0
    for bucket in buckets:
        part = data[offset:offset+bucket['count']]; offset += bucket['count']
        if len(part) == 1:
            result.append(part[0])
            continue
        seats = sum(r['seat_games'] for r in part)
        total = math.fsum(r['vp_sum'] for r in part)
        result.append(dict(first=part[0]['first'], last=part[-1]['last'],
                           generation=bucket['generation'], seat_games=seats, vp_sum=total,
                           mean_vp=total/seats))
    return result


def render(history, output, *, generation=None, preview=None, max_points=DEFAULT_MAX_POINTS):
    """Match progress.svg's visual style without implying rating uncertainty."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    generation = history['committed_generation'] if generation is None else generation
    data = groups(history, generation)
    valid = [r for r in data if r['mean_vp'] is not None]
    shown = display_groups(data, max_points)
    averaged = any('generation' in r for r in shown)
    shown_count = sum(r['mean_vp'] is not None for r in shown)
    latest = valid[-1] if valid else None
    previous = next((r for r in valid if r['last'] == latest['first']-1), None) if latest else None
    recent_start = max(0, generation-RECENT_GENERATIONS)
    ink = '#203B43'; muted = '#5D737C'; blue = '#167D8D'; dark = '#0A6473'
    grid = '#E5ECEE'; paper = '#F6F8F7'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
            'text.color': ink, 'axes.labelcolor': muted, 'xtick.color': muted, 'ytick.color': muted,
            'svg.fonttype': 'none', 'svg.hashsalt': 'brass-vp-progress-v1',
            'axes.unicode_minus': False, 'lines.solid_capstyle': 'round'}):
        fig = plt.figure(figsize=(18., 8.4), facecolor=paper)
        try:
            text = fig.text
            text(.055, .975, 'BRASS BIRMINGHAM  /  SELF-PLAY', fontsize=7.4, weight='bold', color=muted)
            text(.055, .931, 'Self-play learner VP', fontsize=23, weight='bold')
            text(.945, .967, 'GENERATION ' + str(generation), ha='right', fontsize=8.5, weight='bold',
                 color=dark, bbox=dict(boxstyle='round,pad=.5,rounding_size=.25',
                                      facecolor='#E5F0EE', edgecolor='none'))
            text(.945, .934, 'Ordinary self-play tables · learner seats only', ha='right', fontsize=8.5, color=muted)
            for left, width in ((.043, .283), (.341, .308), (.664, .293)):
                fig.add_artist(FancyBboxPatch((left, .817), width, .092,
                    boxstyle='round,pad=0,rounding_size=.01', transform=fig.transFigure,
                    facecolor='white', edgecolor='#E0E8E7', linewidth=.7, zorder=0))
            text(.060, .888, 'LATEST 10-ITERATION MEAN VP', fontsize=7.2, weight='bold', color=muted)
            text(.060, .850, f'{latest["mean_vp"]:,.2f}' if latest else 'Not available',
                 fontsize=20 if latest else 17, weight='bold', color=dark if latest else muted)
            text(.060, .831, f'g{latest["first"]}–g{latest["last"]}' if latest else 'No complete group with VP evidence',
                 fontsize=8, color=muted)
            text(.358, .888, 'CHANGE VS PREVIOUS 10 ITERATIONS', fontsize=7.2, weight='bold', color=muted)
            text(.358, .851, f'{latest["mean_vp"]-previous["mean_vp"]:+,.2f}' if previous else 'Not available',
                 fontsize=19 if previous else 17, weight='bold', color=dark if previous else muted)
            text(.358, .831, f'Compared with g{previous["first"]}–g{previous["last"]}' if previous
                 else 'Previous adjacent group unavailable', fontsize=8, color=muted)
            text(.681, .888, 'TRAINING COVERAGE', fontsize=7.2, weight='bold', color=muted)
            missing = len(data)-len(valid)
            coverage = [f'{len(valid):,} complete groups / {len(valid)*GROUP_SIZE:,} iterations',
                        f'{sum(r["seat_games"] for r in valid):,} learner-seat games with VP evidence',
                        f'{missing:,} groups missing evidence · {generation % GROUP_SIZE} iterations in unfinished group']
            for yy, label in zip((.865, .847, .829), coverage):
                text(.681, yy, label, fontsize=8.1, color=ink if yy == .865 else muted)

            def panel(rect, title, subtitle, start, ticks):
                ax = fig.add_axes(rect, facecolor='white')
                ax.set_axisbelow(True); ax.grid(axis='y', color=grid, linewidth=.8)
                ax.grid(axis='x', color=grid, linewidth=.6, alpha=.45)
                for spine in ('left', 'right', 'top'):
                    ax.spines[spine].set_visible(False)
                ax.spines['bottom'].set_color('#D2DEDF')
                ax.tick_params(axis='both', length=0, pad=8, labelsize=8.6)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
                ax.set_xlabel('Committed training generation', fontsize=9, labelpad=12)
                ax.set_ylabel('Mean learner VP', fontsize=9, labelpad=9)
                ax.set_xlim(start-max(1, (generation-start)*.015), generation+max(4, (generation-start)*.025))
                ax.set_xticks(_ticks(start, generation, ticks))
                text(rect[0], .784, title, fontsize=11.5, weight='bold')
                text(rect[0], .760, subtitle, fontsize=8.4, color=muted)
                return ax

            ax = panel([.075, .190, .460, .527], 'Full training history',
                       f'{shown_count} points / {len(valid)} complete groups · ' +
                       ('adjacent pooled averages' if averaged else f'maximum {max_points} points'), 0, 7)
            # Only available groups occupy space; labels retain their real generations.
            positions = list(range(len(shown)))
            tick_positions = sorted({round(i*(len(shown)-1)/min(7, len(shown)-1))
                                     for i in range(min(8, len(shown)))}) if len(shown)>1 else positions
            ax.set_xlim(-.5, max(.5, len(shown)-.5))
            ax.set_xticks(tick_positions, [str(shown[i]['last']) for i in tick_positions])
            ax.set_xlabel('Generation at group end · missing VP groups omitted', fontsize=9, labelpad=12)
            detail = panel([.635, .190, .313, .527], 'Recent progress',
                           f'g{recent_start}–g{generation} · same groups · separate y-axis', recent_start, 4)
            for axes, part, name in ((ax, shown, 'history'),
                                      (detail, [r for r in data if r['last'] >= recent_start], 'recent')):
                present = [r for r in part if r['mean_vp'] is not None]
                if not present:
                    axes.set_ylim(0, 1); axes.set_yticks([])
                    axes.text(.5, .5, 'No complete 10-iteration group available', transform=axes.transAxes,
                              ha='center', va='center', fontsize=9, color=muted)
                    continue
                x = positions if name == 'history' else [r['last'] for r in part]
                y = [r['mean_vp'] if r['mean_vp'] is not None else math.nan for r in part]
                line, = axes.plot(x, y, color=blue, linewidth=2., zorder=4)
                line.set_gid('vp-' + name + '-line')
                ordinary = [(xx,r) for xx,r in zip(x,part) if r['mean_vp'] is not None and r != latest]
                dots = axes.scatter([xx for xx,r in ordinary], [r['mean_vp'] for xx,r in ordinary],
                                    s=23, facecolor='white', edgecolor=blue, linewidth=1.05, zorder=5)
                dots.set_gid('vp-' + name + '-markers')
                low = min(r['mean_vp'] for r in present); high = max(r['mean_vp'] for r in present)
                span = max(2., high-low)
                axes.set_ylim(low-span*.12, high+span*.16)
                if latest in present:
                    last = axes.scatter([x[part.index(latest)]], [latest['mean_vp']], s=53, color=dark,
                                        edgecolor='white', linewidth=1.4, zorder=6)
                    last.set_gid('vp-' + name + '-latest-point')
            if latest and latest['last'] >= recent_start:
                _endpoint_label(detail, latest['last'], latest['mean_vp'], dark, prefix='vp')
            recent_positions = [i for i,r in enumerate(shown) if r['last'] >= recent_start]
            if recent_start > 0 and recent_positions:
                start = recent_positions[0]-.5
                ax.axvspan(start, len(shown)-.5, color=blue, alpha=.035, linewidth=0, zorder=0)
                ax.axvline(start, color=blue, alpha=.25, linewidth=.8, linestyle=(0, (3, 4)), zorder=1)
            fig.legend(handles=[Line2D([], [], color=blue, marker='o', markerfacecolor='white',
                       linewidth=1.8, markersize=4, label='Seat-weighted pooled mean VP')],
                       loc='center left', bbox_to_anchor=(.066, .095), frameon=False, fontsize=8.4, labelcolor=muted)
            text(.055, .055, 'Full history: adjacent 10-iteration groups pooled; first/latest exact. Recent panel: original groups. Zero-VP seats included.'
                 if averaged else 'Each point = total VP / learner-seat games across 10 committed iterations. Zero-VP seats are included.',
                 fontsize=8.1, color=muted)
            text(.055, .033, 'History/v4, branch and evaluation games excluded. Left: missing groups take no space. Right: original generation spacing.',
                 fontsize=8.1, color=muted)
            text(.945, .011, 'Training score trend · not a controlled playing-strength evaluation', fontsize=8, color=muted, ha='right')
            fig.savefig(output, format='svg', facecolor=paper, metadata=dict(Date=None,
                Title='Brass Birmingham — self-play learner VP',
                Description=f'Fixed ten-iteration groups through generation {generation}. '
                    f'Full history: {shown_count} displayed points (cap {max_points}); adjacent seat-weighted averages, first/latest exact. '
                    'Full-history spacing omits missing groups. Recent detail and summary cards use original groups. ' + '; '.join(coverage)))
            output = Path(output)
            output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf-8').splitlines())+'\n',
                              encoding='utf-8', newline='\n')
            if preview is not None:
                fig.savefig(preview, format='png', dpi=150, facecolor=paper)
        finally:
            plt.close(fig)


def companion(output, generation, *, run=None, state=None, persist=False, history_sources=(), preview=None,
              max_points=DEFAULT_MAX_POINTS):
    """Generate the sibling chart; standalone redraws only read source evidence."""
    output = Path(output).with_name('progress_vp.svg')
    history = (collect(run, state=state, history_sources=history_sources) if run is not None else
               dict(format=FORMAT, initial_checkpoint={}, committed_generation=generation, iterations=[]))
    render(history, output, generation=generation, preview=preview, max_points=max_points)
    if persist:
        if run is None:
            raise ValueError('A training run is required to persist VP history')
        save_history(history, Path(run)/HISTORY_PATH)
    return output
