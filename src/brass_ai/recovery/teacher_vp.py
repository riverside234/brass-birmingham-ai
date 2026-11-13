"""Uncapped final VP for completed teacher tables and the three progress curves."""
from collections import deque
import math
from pathlib import Path

from brass_ai.learning.contracts import require

FORMAT = 'teacher-vp-history/1.0.0'
HISTORY_PATH = Path('ratings/teacher-vp-history.json')
CHART = 'progress_teacher_vp.svg'
COTTON = 'cotton_exploration'
STANDARD = 'standard_exploration'
GROUPS = (COTTON, STANDARD, 'all')
SPLITS = ('train', 'heldout')
MOVING_AVERAGE_ITERATIONS = 10
HISTORY_MAX_POINTS = 40
RECENT_MAX_POINTS = 30


def summarize(tables):
    """Reduce the four terminal seat scores once, alongside table admission."""
    counts = {split: {group: [0, 0] for group in GROUPS} for split in SPLITS}
    for split, cotton, scores in tables:
        require(len(scores) == 4 and all(type(score) is int for score in scores), 'Invalid teacher table VP')
        total = sum(scores)
        for group in ('all', COTTON if cotton else STANDARD):
            counts[split][group][0] += 1
            counts[split][group][1] += total

    def stat(tables, vp):
        return dict(tables=tables, vp_sum=vp, mean_vp=vp/(4*tables) if tables else None,
                    mean_table_vp=vp/tables if tables else None)

    result = {g: stat(counts['train'][g][0]+counts['heldout'][g][0],
                      counts['train'][g][1]+counts['heldout'][g][1]) for g in GROUPS}
    result['by_split'] = {split: {g: stat(*values) for g, values in groups.items()} for split, groups in counts.items()}
    return result


def _earlier_tables(report):
    """Extract the table identities and group ordering from an older collection report."""
    collection = (report.get('teacher_phase') or {}).get('collection') or {}
    jobs = collection.get('post_admission_per_table_seconds') or {}
    actions = (collection.get('teacher_actions') or {}).get('per_table') or []
    splits = [split for split in SPLITS for _ in range((collection.get('tables') or {}).get(split, 0))]
    if len(jobs) != len(splits) or not jobs:
        return None
    if actions and len(actions) == len(jobs) and [row['split'] for row in actions] == splits:
        cotton = [row['cotton_exploration'] for row in actions]
    else:
        metrics = collection.get('collection_metrics') or []
        if len(metrics) != len(jobs):
            return None
        cotton = ['cotton' in (m.get('cpu_inference') or {}).get('revision', {}) for m in metrics]
    actor = report.get('actor') or {}
    reward = actor.get('reward_spec') or (actor.get('reward') or {}).get('spec') or {}
    if not (reward.get('vp', 0) > 0 and reward.get('vp_cap') is not None):
        return None
    return list(zip(jobs, splits, cotton, strict=True)), reward['vp'], reward['vp_cap']


class History:
    """Build the small history in the same committed-report scan as ordinary VP."""
    def __init__(self, state, sources):
        from .vp_plot import _initial, _read

        self.initial = _initial(state)
        self.generation = state['iteration']
        self.rows, self.earlier, self.payments = {}, {}, {}
        for source in sources:
            path = Path(source)/HISTORY_PATH
            if not path.is_file():
                continue
            saved = _read(path)
            require(saved.get('format') == FORMAT and saved.get('initial_checkpoint') == self.initial,
                    'Teacher VP history belongs to another run')
            limit = min(self.generation, saved['committed_generation'])
            for row in saved['iterations']:
                generation = row['generation']
                require(type(generation) is int and generation > 0, 'Invalid teacher VP generation')
                if generation <= limit:
                    self.add_scores(generation, row['scores'])

    def add_scores(self, generation, scores):
        require(generation not in self.rows or self.rows[generation] == scores,
                f'Conflicting teacher VP evidence for generation {generation}')
        self.rows[generation] = scores

    def add_report(self, report):
        generation = report['iteration'] + 1
        scores = ((report.get('teacher_phase') or {}).get('collection') or {}).get('vp')
        if scores is not None:
            self.add_scores(generation, scores)
        elif generation not in self.rows:
            earlier = _earlier_tables(report)
            if earlier is not None:
                require(generation not in self.earlier or self.earlier[generation] == earlier,
                        'Conflicting historical teacher table identities')
                self.earlier[generation] = earlier
        payments = (report.get('search_value') or {}).get('vp_payments_per_table') or {}
        for table, payment in payments.items():
            if payment.get('contract') == 'canal-and-final-delta/1.0.0':
                total = payment['total']
                require(table not in self.payments or self.payments[table] == total, 'Conflicting teacher VP payouts')
                self.payments[table] = total

    def build(self):
        for generation, (jobs, rate, cap) in self.earlier.items():
            if generation in self.rows:
                continue
            tables = []
            for job, split, cotton in jobs:
                payouts = self.payments.get('teacher/' + job)
                if payouts is None or len(payouts) != 4 or not all(math.isfinite(v) for v in payouts):
                    break
                scores = [round(v/rate) for v in payouts]
                # A capped reward cannot tell us the actual terminal score.
                if any(score >= cap or not math.isclose(v, score*rate, abs_tol=1e-10)
                       for v, score in zip(payouts, scores, strict=True)):
                    break
                tables.append((split, cotton, scores))
            else:
                self.add_scores(generation, summarize(tables))
        return dict(format=FORMAT, initial_checkpoint=self.initial, committed_generation=self.generation,
                    iterations=[dict(generation=g, scores=self.rows[g]) for g in sorted(self.rows)])


def moving_averages(rows):
    """Pool final VP over the trailing generations, weighted by table count."""
    window = deque()
    result = {}
    for generation, scores in sorted(rows.items()):
        window.append((generation, scores))
        while window[0][0] <= generation-MOVING_AVERAGE_ITERATIONS:
            window.popleft()
        means = {}
        for group in GROUPS:
            if not scores[group]['tables']:
                means[group] = math.nan
                continue
            tables = sum(row[group]['tables'] for _, row in window)
            vp = sum(row[group]['vp_sum'] for _, row in window)
            means[group] = vp/(4*tables)
        result[generation] = means
    return result


def marker_indices(observations, limit):
    """Spread markers across finite observations, including the endpoints."""
    indices = [i for i, value in enumerate(observations) if math.isfinite(value)]
    if len(indices) <= limit:
        return indices
    return [indices[round(i*(len(indices)-1)/(limit-1))] for i in range(limit)]


def render(history, output, *, generation=None):
    """Plot the three moving averages in the full history and latest 200 iterations."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    import matplotlib
    from .rating_plot import RECENT_GENERATIONS, _ticks

    generation = history['committed_generation'] if generation is None else generation
    rows = {r['generation']: r['scores'] for r in history['iterations'] if r['generation'] <= generation}
    if not rows:
        return None
    means = moving_averages(rows)
    generations = list(means)
    recent_start = max(1, generation-RECENT_GENERATIONS+1)
    recent = range(recent_start, generation+1)
    recent_count = sum(g in means for g in recent)
    paper, ink, muted = '#F7FAFA', '#18333B', '#49656C'
    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                               'svg.hashsalt': 'teacher-vp', 'path.simplify': False}):
        figure = Figure(figsize=(15.5, 5.4), facecolor=paper)
        FigureCanvasAgg(figure)
        figure.text(.07, .918, 'Teacher table VP', fontsize=23, color=ink, weight='bold')
        figure.text(.07, .864, f'Final VP per seat · training and held-out tables · {MOVING_AVERAGE_ITERATIONS}-iteration moving average',
                    fontsize=11, color=muted)
        positions = range(len(generations))
        overview_ticks = sorted({round(i*(len(generations)-1)/7) for i in range(8)})

        def panel(rect, title, subtitle, x, values, marker_limit, *, recent_panel=False):
            axes = figure.add_axes(rect, facecolor='white')
            figure.text(rect[0], .751, title, fontsize=12, weight='bold', color=ink)
            figure.text(rect[0], .714, subtitle, fontsize=9, color=muted)
            for group, label, color, style in zip(GROUPS, ('Cotton exploration', 'Standard exploration', 'All tables'),
                                                  ('#007D79', '#A75521', '#314E87'), ('-', '--', '-.'), strict=True):
                y = [means[g][group] if g in means else math.nan for g in values]
                line, = axes.plot(x, y, color=color, linestyle=style, marker='o', markersize=2.7,
                                  markevery=marker_indices(y, marker_limit), linewidth=1.5, label=label)
                prefix = 'teacher-vp-recent-' if recent_panel else 'teacher-vp-'
                line.set_gid(prefix + group.replace('_', '-'))
            axes.set_xlabel('Committed generation', color=muted)
            axes.set_ylabel('Mean VP', color=muted)
            axes.yaxis.set_major_locator(MaxNLocator(nbins=6))
            axes.grid(axis='y', color='#E0EBED', linewidth=.8)
            axes.set_axisbelow(True)
            axes.spines[['right', 'top']].set_visible(False)
            axes.spines[['left', 'bottom']].set_color('#ABC2C6')
            axes.tick_params(colors=muted, labelsize=9)
            return axes

        overview = panel((.07, .165, .49, .51), 'Full training history',
                         f'{len(rows):,} recorded iterations · up to {HISTORY_MAX_POINTS} markers per curve',
                         positions, generations, HISTORY_MAX_POINTS)
        overview.set_xlim(-.6, len(generations)-.4)
        overview.set_xticks(overview_ticks, [str(generations[i]) for i in overview_ticks])
        first_recent = sum(g < recent_start for g in generations)
        if 0 < first_recent < len(generations):
            overview.axvspan(first_recent-.5, len(generations)-.4,
                             color='#314E87', alpha=.07, linewidth=0, zorder=0)
        detail = panel((.648, .165, .32, .51), 'Recent 200 iterations',
                       f'g{recent_start}–g{generation} · {recent_count:,} recorded · up to {RECENT_MAX_POINTS} markers per curve',
                       recent, recent, RECENT_MAX_POINTS, recent_panel=True)
        detail.set_xlim(recent_start-2, generation+2)
        detail.set_xticks(sorted({recent_start, *_ticks(recent_start, generation, 5)}))
        if not recent_count:
            detail.set_yticks([])
            detail.text(.5, .5, 'No teacher VP available in this window', ha='center', va='center',
                        transform=detail.transAxes, color=muted, fontsize=9)
        handles, labels = overview.get_legend_handles_labels()
        figure.legend(handles, labels, loc='center left', bbox_to_anchor=(.066, .807), ncol=3,
                      frameon=False, labelcolor=ink, fontsize=9)
        output = Path(output)
        figure.savefig(output, format='svg', metadata=dict(Date=None, Title='Teacher table VP',
            Description=f'Trailing {MOVING_AVERAGE_ITERATIONS}-iteration final VP per seat, pooled by table count '
                        'across training and held-out teacher tables. '
                        'Series distinguish cotton-targeted tables, standard exploration, and all tables. '
                        f'Full history plots all {len(rows)} recorded iterations with up to {HISTORY_MAX_POINTS} markers per curve. '
                        f'Recent detail spans g{recent_start}–g{generation} with {recent_count} recorded iterations, '
                        f'actual generation spacing and up to {RECENT_MAX_POINTS} markers per curve.'))
    output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf-8').splitlines())+'\n',
                      encoding='utf-8', newline='\n')
    return output
