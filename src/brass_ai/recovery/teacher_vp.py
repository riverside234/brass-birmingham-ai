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
FULL_HISTORY_START = 2269
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
    """Plot comparable teacher VP from g2269 and a recent 400-generation view."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import MaxNLocator
    import matplotlib
    from .rating_plot import RECENT_GENERATIONS, _ticks

    generation = history['committed_generation'] if generation is None else generation
    rows = {r['generation']: r['scores'] for r in history['iterations'] if r['generation'] <= generation}
    if not rows:
        return None
    # Start the moving average inside the comparable period as well as cutting the x-axis.
    means = moving_averages({g: scores for g, scores in rows.items() if g >= FULL_HISTORY_START})
    generations = list(means)
    # Archived reports before the change can still show their original recent window.
    recent_means = means if generation >= FULL_HISTORY_START else moving_averages(rows)
    recent_start = max(1, generation-RECENT_GENERATIONS+1)
    recent = range(recent_start, generation+1)
    recent_count = sum(g in recent_means for g in recent)
    latest = next(reversed(recent_means), None)
    paper, ink, muted = '#F5F7FA', '#182C43', '#4E6175'
    grid, border, window = '#E2E8F0', '#D4DEE9', '#EAF1F9'
    series = (
        (COTTON, 'Cotton exploration', '#AE4718', '-', 'D'),
        (STANDARD, 'No cotton exploration', '#006CA7', (0, (3.2, 2.2)), 's'),
        ('all', 'All tables', '#253550', '-', 'o'),
    )
    styles = {
        group: dict(color=color, linestyle=dashes, marker=marker, markersize=3.8,
                    linewidth=2.15 if group == 'all' else 1.75,
                    markerfacecolor=color if group == 'all' else 'white',
                    markeredgecolor='white' if group == 'all' else color,
                    markeredgewidth=.8)
        for group, _, color, dashes, marker in series
    }
    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                               'svg.hashsalt': 'teacher-vp', 'path.simplify': False,
                               'lines.solid_capstyle': 'round', 'lines.dash_capstyle': 'round',
                               'lines.solid_joinstyle': 'round', 'lines.dash_joinstyle': 'round'}):
        figure = Figure(figsize=(17.5, 6.8), facecolor=paper)
        FigureCanvasAgg(figure)
        figure.text(.055, .924, 'Teacher table VP', fontsize=25, color=ink, weight='bold')
        figure.text(.055, .874, f'Final VP per seat · training and held-out tables · '
                    f'trailing {MOVING_AVERAGE_ITERATIONS}-iteration mean', fontsize=11, color=muted)
        figure.text(.945, .874, f'Latest moving averages · g{latest}' if latest is not None
                    else 'No comparable VP recorded', ha='right', fontsize=10, color=muted)

        # Each series key also shows its exact latest moving average.
        latest_values = recent_means.get(latest, {})
        for left, (group, label, color, _, _) in zip((.055, .355, .655), series, strict=True):
            figure.add_artist(FancyBboxPatch((left, .765), .29, .077,
                boxstyle='round,pad=0,rounding_size=.009', transform=figure.transFigure,
                facecolor='white', edgecolor=border, linewidth=.75, zorder=1))
            figure.add_artist(Line2D([left+.015, left+.030, left+.045], [.803]*3,
                transform=figure.transFigure, markevery=[1], **styles[group]))
            figure.text(left+.055, .803, label, fontsize=10.5, color=ink, va='center')
            value = latest_values.get(group, math.nan)
            figure.text(left+.275, .803, f'{value:,.2f}' if math.isfinite(value) else '—',
                        fontsize=19, color=color, va='center', ha='right', weight='bold')

        def panel(rect, title, subtitle, x, values, data, marker_limit, prefix, empty_label):
            axes = figure.add_axes(rect, facecolor='white')
            figure.text(rect[0], .705, title, fontsize=13, weight='bold', color=ink)
            figure.text(rect[0], .667, subtitle, fontsize=9.5, color=muted)
            finite = []
            for group, label, _, _, _ in series:
                y = [data[g][group] if g in data else math.nan for g in values]
                finite.extend(value for value in y if math.isfinite(value))
                line, = axes.plot(x, y, label=label, markevery=marker_indices(y, marker_limit), **styles[group])
                line.set_gid(prefix + group.replace('_', '-'))
            axes.set_ylabel('Mean VP per seat', color=muted, fontsize=10, labelpad=9)
            axes.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axes.grid(axis='y', color=grid, linewidth=.8)
            axes.grid(axis='x', color=grid, linewidth=.65, alpha=.55)
            axes.set_axisbelow(True)
            axes.spines[['left', 'right', 'top']].set_visible(False)
            axes.spines['bottom'].set_color(border)
            axes.tick_params(colors=muted, labelsize=9, length=0, pad=7)
            if finite:
                low, high = min(finite), max(finite)
                pad = max(.5, (high-low)*.08)
                axes.set_ylim(low-pad, high+pad)
            else:
                axes.set_yticks([])
                axes.text(.5, .5, empty_label, ha='center', va='center',
                          transform=axes.transAxes, color=muted, fontsize=10)
            return axes

        positions = range(len(generations))
        overview_ticks = sorted({round(i*(len(generations)-1)/7) for i in range(8)}) if generations else []
        overview = panel((.073, .160, .428, .474), 'Full training history',
                         f'From g{FULL_HISTORY_START} · {len(generations):,} recorded iterations',
                         positions, generations, means, HISTORY_MAX_POINTS, 'teacher-vp-',
                         f'No teacher VP recorded from g{FULL_HISTORY_START}')
        pad = max(.6, (len(generations)-1)*.012)
        overview.set_xlim(-pad, max(pad, len(generations)-1+pad))
        overview.set_xticks(overview_ticks, [str(generations[i]) for i in overview_ticks])
        overview.set_xlabel('Generation · recorded iterations only', color=muted, fontsize=10, labelpad=9)
        first_recent = sum(g < recent_start for g in generations)
        if 0 < first_recent < len(generations):
            overview.axvspan(first_recent-.5, overview.get_xlim()[1], facecolor=window, linewidth=0,
                             zorder=0).set_gid('teacher-vp-recent-window')
            figure.text(overview.get_position().x1, .667, 'Recent window', ha='right',
                        fontsize=9, color=muted, bbox=dict(facecolor=window, edgecolor='none', pad=3))

        detail = panel((.575, .160, .370, .474), f'Recent {RECENT_GENERATIONS} iterations',
                       f'g{recent_start}–g{generation} · {recent_count:,} recorded · separate VP scale',
                       recent, recent, recent_means, RECENT_MAX_POINTS, 'teacher-vp-recent-',
                       'No teacher VP available in this window')
        pad = max(2, (generation-recent_start)*.012)
        detail.set_xlim(recent_start-pad, generation+pad)
        # Keep the true range endpoints and avoid an adjacent grid tick crowding the first label.
        min_gap = (generation-recent_start)*.10
        ticks = [g for g in _ticks(recent_start, generation, 5) if g-recent_start >= min_gap or g == generation]
        detail.set_xticks(sorted({recent_start, *ticks}))
        detail.set_xlabel('Generation', color=muted, fontsize=10, labelpad=9)

        output = Path(output)
        figure.savefig(output, format='svg', metadata=dict(Date=None, Title='Teacher table VP',
            Description=f'Trailing {MOVING_AVERAGE_ITERATIONS}-iteration final VP per seat, pooled by table count '
                        'across training and held-out teacher tables. '
                        'Series distinguish tables with cotton exploration, without cotton exploration, and all tables. '
                        f'Full history starts at g{FULL_HISTORY_START} and plots {len(generations)} recorded iterations '
                        f'with up to {HISTORY_MAX_POINTS} markers per curve. '
                        f'Recent detail spans g{recent_start}–g{generation} with {recent_count} recorded iterations, '
                        f'actual generation spacing, a separate VP scale and up to {RECENT_MAX_POINTS} markers per curve.'))
    output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf-8').splitlines())+'\n',
                      encoding='utf-8', newline='\n')
    return output
