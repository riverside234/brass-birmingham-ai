"""Bounded common-fit PL history, exact recent detail and paired evidence.

Shared by automatic rating slots and standalone redraws. Rendering never refits
ratings, collects games, or changes the saved statistical evidence.
"""
import math
from pathlib import Path
from statistics import fmean


DEFAULT_MAX_POINTS = 30
RECENT_GENERATIONS = 100


def display_points(generations, scores, max_points=DEFAULT_MAX_POINTS):
    """Average adjacent non-overlapping groups; first/latest remain exact.

    This restores the earlier display reduction, rather than fitting a trend.
    Every checkpoint contributes once; intervals and the recent panel use the
    original evidence. The total dot count includes the exact latest point.
    """
    if type(max_points) is not int or not 3 <= max_points <= DEFAULT_MAX_POINTS:
        raise ValueError('Maximum displayed PL points must be an integer from 3 to 30')
    if not generations or len(generations) != len(scores):
        raise ValueError('Expected matching nonempty PL generations and scores')
    def bucket(start, stop):
        return dict(generation=fmean(generations[start:stop]), score=fmean(scores[start:stop]),
                    count=stop-start, first=generations[start], last=generations[stop-1])
    count = len(generations)
    if count <= max_points:
        return [bucket(i, i+1) for i in range(count)]
    middle = count-2; groups = max_points-2
    return [bucket(0, 1)] + [bucket(1+i*middle//groups, 1+(i+1)*middle//groups)
        for i in range(groups)] + [bucket(count-1, count)]


def _interval(value, interval, label):
    low, high = interval
    if not all(math.isfinite(v) for v in (value, low, high)) or low > high:
        raise ValueError('Invalid PL point estimate or interval: ' + label)
    return value, low, high


def paired_change(report, previous, latest):
    """Read latest-minus-previous from this fit's paired bootstrap evidence.

    Separate marginal intervals cannot reconstruct this interval. Archived
    reports without the pair must say unavailable rather than approximate it.
    """
    pairs = [row for row in report.get('differences', [])
             if {row['a'], row['b']} == {previous, latest}]
    if not pairs:
        return None
    if len(pairs) != 1:
        raise ValueError('Duplicate paired PL comparison')
    row = pairs[0]
    delta, low, high = _interval(row['points_delta'], row['points_delta_ci95'], 'paired change')
    if row['b'] != latest:
        delta, low, high = -delta, -high, -low
    expected = report['ratings'][latest]['points'] - report['ratings'][previous]['points']
    if not math.isclose(delta, expected, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError('Paired PL comparison does not match the current fit')
    return dict(delta=delta, low=low, high=high,
                status='improved' if low > 0 else 'declined' if high < 0 else 'uncertain')


def coverage_labels(report, checkpoints):
    """Keep per-slot support distinct from the accumulated common-fit games."""
    protocol = report.get('protocol', {})
    main = protocol.get('main_games')
    # Current slots always use four seat rotations per setup. Offline refits
    # may omit protocol; total unique setups is not evidence for the last slot.
    latest = (f'Latest slot: {main:,} games / {main // 4:,} setups'
              if type(main) is int and main > 0 and main % 4 == 0
              else 'Latest slot: support not recorded')
    total = report.get('games')
    accumulated = (f'Common fit: {total:,} games' if type(total) is int and total >= 0
                   else 'Common fit: game count unavailable')
    return latest, accumulated, f'{checkpoints:,} rated checkpoints'


def _ticks(start, stop, count):
    """Uncrowded generation ticks, always retaining the real latest value."""
    step = max(10, math.ceil((stop - start) / (count * 10)) * 10)
    ticks = list(range(math.ceil(start / step) * step, stop + 1, step))
    if not ticks:
        ticks = [stop]
    elif ticks[-1] != stop:
        if len(ticks) > 1 and stop - ticks[-1] < step * .55:
            ticks[-1] = stop
        else:
            ticks.append(stop)
    return ticks


def _endpoint_label(axes, generation, score, color):
    """Reserve a physical gutter so endpoint text never covers the evidence.

    Measure the formatted value rather than assuming three-digit ratings. The
    label remains inside the canvas for negative and large scores, and its
    leader still points to the exact, unshifted checkpoint.
    """
    from matplotlib.font_manager import FontProperties

    figure = axes.figure
    renderer = figure.canvas.get_renderer()
    label = f'{score:,.1f}'
    font = FontProperties(size=10, weight='bold')
    width = renderer.get_text_width_height_descent(label, font, ismath=False)[0]
    gutter = max(renderer.points_to_pixels(64), width + renderer.points_to_pixels(24))
    position = axes.get_position()
    right = min(position.x1, .978 - gutter / figure.bbox.width)
    axes.set_position([position.x0, position.y0, right-position.x0, position.height])
    low, high = axes.get_ylim()
    y = min(.94, max(.06, (score-low)/(high-low)))
    x = 1 + renderer.points_to_pixels(12) / axes.get_window_extent(renderer).width
    annotation = axes.annotate(label, xy=(generation, score), xycoords='data',
        xytext=(x, y), textcoords='axes fraction', ha='left', va='center',
        fontproperties=font, color=color, annotation_clip=False, clip_on=False,
        bbox=dict(boxstyle='round,pad=.35,rounding_size=.22', facecolor='white',
                  edgecolor='#C9DEDF', linewidth=.8),
        arrowprops=dict(arrowstyle='-', color=color, linewidth=.9,
                        shrinkA=4, shrinkB=6, relpos=(0., .5)), zorder=8)
    annotation.set_gid('pl-recent-latest-label')
    annotation.arrow_patch.set_gid('pl-recent-latest-leader')


def render(report, output, *, preview=None, agents=None, max_points=DEFAULT_MAX_POINTS):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch, Patch
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    ratings = report['ratings']
    if agents is None:
        points = sorted((int(name.removeprefix('generation-')), name)
            for name in ratings if name.startswith('generation-') and
            name.removeprefix('generation-').isdigit() and int(name.removeprefix('generation-')) > 0)
    else:
        points = sorted((entry['checkpoint']['generation'], name)
            for name, entry in agents.items() if name in ratings and entry['kind'] == 'model'
            and entry['checkpoint']['generation'] > 0)
    if not points:
        raise ValueError('The report has no rated training checkpoints')
    if len({g for g, _ in points}) != len(points):
        raise ValueError('Duplicate rated checkpoint generation')

    def values(name):
        row = ratings[name]
        return _interval(row['points'], row['points_ci95'], name)

    data = [values(name) for _, name in points]
    x = [g for g, _ in points]; y = [v for v, _, _ in data]
    shown = display_points(x, y, max_points); averaged = len(shown) < len(points)
    history_x = [row['generation'] for row in shown]; history_y = [row['score'] for row in shown]
    lower = [lo for _, lo, _ in data]; upper = [hi for _, _, hi in data]
    latest, latest_name = points[-1]; score, low, high = data[-1]
    v4, v4low, v4high = values('v4'); anchor = values('generation-0')[0]
    if anchor != 0:
        raise ValueError('Expected the fixed IL reference at zero PL points')
    previous = points[-2] if len(points) > 1 else None
    change = paired_change(report, previous[1], latest_name) if previous else None
    recent_start = max(0, latest - RECENT_GENERATIONS)
    recent = [i for i, generation in enumerate(x) if generation >= recent_start]
    rx = [x[i] for i in recent]; ry = [y[i] for i in recent]
    rlo = [lower[i] for i in recent]; rhi = [upper[i] for i in recent]
    coverage = coverage_labels(report, len(points))

    ink = '#203B43'; muted = '#5D737C'; blue = '#167D8D'; dark = '#0A6473'
    amber = '#9C753F'; grid = '#E5ECEE'; paper = '#F6F8F7'
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
            'text.color': ink, 'axes.labelcolor': muted, 'xtick.color': muted, 'ytick.color': muted,
            'svg.fonttype': 'none', 'svg.hashsalt': 'brass-pl-progress-v7',
            'axes.unicode_minus': False, 'lines.solid_capstyle': 'round'}):
        fig = plt.figure(figsize=(14.4, 8.4), facecolor=paper)
        try:
            text = fig.text
            text(.055, .975, 'BRASS BIRMINGHAM  /  SELF-PLAY', fontsize=7.4, weight='bold', color=muted)
            text(.055, .931, 'Self-play strength', fontsize=23, weight='bold')
            text(.945, .967, 'GENERATION ' + str(latest), ha='right', fontsize=8.5, weight='bold',
                 color=dark, bbox=dict(boxstyle='round,pad=.5,rounding_size=.25',
                                      facecolor='#E5F0EE', edgecolor='none'))
            text(.945, .934, 'Plackett–Luce · one common fit · IL = 0', ha='right', fontsize=8.5, color=muted)

            # Three compact cards leave most of the canvas for evidence.
            for left, width in ((.043, .283), (.341, .308), (.664, .293)):
                fig.add_artist(FancyBboxPatch((left, .817), width, .092,
                    boxstyle='round,pad=0,rounding_size=.01', transform=fig.transFigure,
                    facecolor='white', edgecolor='#E0E8E7', linewidth=.7, zorder=0))
            text(.060, .888, 'LATEST PL SCORE', fontsize=7.2, weight='bold', color=muted)
            text(.060, .850, f'{score:,.1f}', fontsize=20, weight='bold', color=dark)
            text(.060, .831, f'95% interval  {low:,.1f} to {high:,.1f}', fontsize=8, color=muted)
            text(.358, .888, 'CHANGE VS ' + (f'g{previous[0]}' if previous else 'PREVIOUS'),
                 fontsize=7.2, weight='bold', color=muted)
            if change is None:
                text(.358, .853, 'Not available', fontsize=17, color=muted)
                text(.358, .831, 'First rated checkpoint' if previous is None else 'Paired interval not recorded',
                     fontsize=8, color=muted)
            else:
                status = {'improved': 'Improvement supported', 'declined': 'Decline supported',
                          'uncertain': 'Change uncertain'}[change['status']]
                status_color = dark if change['status'] == 'improved' else amber if change['status'] == 'declined' else muted
                text(.358, .851, f'{change["delta"]:+,.1f}', fontsize=19, weight='bold', color=status_color)
                text(.470, .856, status, fontsize=8.1, color=status_color)
                text(.358, .831, f'Paired 95% interval  {change["low"]:+,.1f} to {change["high"]:+,.1f}',
                     fontsize=8, color=muted)
            text(.681, .888, 'EVALUATION COVERAGE', fontsize=7.2, weight='bold', color=muted)
            for yy, label in zip((.865, .847, .829), coverage):
                text(.681, yy, label, fontsize=8.1, color=ink if yy == .865 else muted)

            def panel(rect, title, subtitle):
                left, bottom, width, height = rect
                ax = fig.add_axes(rect, facecolor='white')
                ax.set_axisbelow(True); ax.grid(axis='y', color=grid, linewidth=.8)
                ax.grid(axis='x', color=grid, linewidth=.6, alpha=.45)
                for spine in ('left', 'right', 'top'):
                    ax.spines[spine].set_visible(False)
                ax.spines['bottom'].set_color('#D2DEDF')
                ax.tick_params(axis='both', length=0, pad=8, labelsize=8.6)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
                ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
                ax.set_xlabel('Committed training generation', fontsize=9, labelpad=12)
                text(left, .784, title, fontsize=11.5, weight='bold')
                text(left, .760, subtitle, fontsize=8.4, color=muted)
                ax.set_ylabel('PL points', fontsize=9, labelpad=9)
                return ax

            history_subtitle = (f'{len(shown)} points / {len(points)} checkpoints · adjacent averages'
                                if averaged else f'{len(points)} exact checkpoints · maximum {max_points} points')
            ax = panel([.075, .190, .523, .527], 'Full training history', history_subtitle)
            detail = panel([.709, .190, .239, .527], 'Recent progress',
                           f'g{recent_start}–g{latest} · exact points · separate y-axis')
            ax.set_xlim(-max(1, latest * .015), latest + max(5, latest * .025))
            ax.set_xticks(_ticks(0, latest, 7))
            bottom = min(anchor, v4low, *lower, *y); top = max(v4high, *upper, *y)
            span = max(100., top - bottom)
            ax.set_ylim(bottom - span * .06, top + span * .09)
            if recent_start > 0:
                window = ax.axvspan(recent_start, latest, color=blue, alpha=.035, linewidth=0, zorder=0)
                window.set_gid('pl-recent-window')
                ax.axvline(recent_start, color=blue, alpha=.25, linewidth=.8, linestyle=(0, (3, 4)), zorder=1)
                # Axes-relative label stays legible when the highlighted time
                # window becomes very narrow in a long training history.
                ax.text(.985, .97, 'Recent window →', transform=ax.transAxes, ha='right', va='top',
                        fontsize=7.5, color=muted)
            ax.axhspan(v4low, v4high, color=amber, alpha=.045, linewidth=0, zorder=1)
            ax.axhline(v4, color=amber, linewidth=1., linestyle=(0, (5, 4)), alpha=.75, zorder=2)
            ax.axhline(anchor, color='#A0AFB3', linewidth=.9, linestyle=(0, (2, 4)), zorder=2)
            for value, label, color in ((v4, f'v4  {v4:,.0f}', amber), (anchor, 'IL  0', '#82979D')):
                ax.annotate(label, xy=(1., value), xycoords=('axes fraction', 'data'),
                    xytext=(7, 0), textcoords='offset points', fontsize=8.2, color=color, va='center')

            # Both panels use untouched marginal bootstrap bounds. Include MAP
            # values in limits even when percentile intervals exclude them.
            def curve(axes, xx, yy, lows, highs, prefix, size, *, band_x=None):
                band = axes.fill_between(xx if band_x is None else band_x, lows, highs,
                                         color=blue, alpha=.075, linewidth=0, zorder=2)
                band.set_gid('pl-raw-pointwise-intervals' if prefix == 'history' else 'pl-recent-intervals')
                line, = axes.plot(xx, yy, color=blue, linewidth=2., zorder=4)
                line.set_gid('pl-' + prefix + '-line')
                dots = axes.scatter(xx[:-1], yy[:-1], s=size, facecolor='white', edgecolor=blue,
                                    linewidth=1.05, zorder=5)
                dots.set_gid('pl-' + prefix + '-markers')
                if prefix == 'recent':
                    axes.vlines(xx, lows, highs, color=blue, alpha=.20, linewidth=.9, zorder=3)
                else:
                    axes.vlines([latest], [low], [high], color=dark, alpha=.5, linewidth=1.2, zorder=5)
                last = axes.scatter([latest], [score], s=53, color=dark, edgecolor='white', linewidth=1.4, zorder=6)
                last.set_gid('pl-latest-point' if prefix == 'history' else 'pl-recent-latest-point')

            curve(ax, history_x, history_y, lower, upper, 'history', 20, band_x=x)
            detail.set_xlim(recent_start - 2, latest + 4)
            detail.set_xticks(_ticks(recent_start, latest, 4))
            rbottom = min(*rlo, *ry); rtop = max(*rhi, *ry); rspan = max(20., rtop - rbottom)
            detail.set_ylim(rbottom - rspan * .12, rtop + rspan * .16)
            curve(detail, rx, ry, rlo, rhi, 'recent', 23)
            _endpoint_label(detail, latest, score, dark)
            if len(recent) == 1:
                detail.text(.04, .94, 'One checkpoint in this window', transform=detail.transAxes,
                            va='top', fontsize=8.2, color=muted)
            handles = [Line2D([], [], color=blue, marker='o', markerfacecolor='white', linewidth=1.8,
                              markersize=4, label='Adjacent averages (full history)' if averaged else 'Exact checkpoint rating'),
                       Patch(facecolor=blue, alpha=.12, edgecolor='none', label='Original pointwise 95% intervals'),
                       Line2D([], [], color=amber, linewidth=1., linestyle=(0, (5, 4)), label='v4 reference · no search')]
            fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.066, .095), ncol=3,
                       frameon=False, fontsize=8.4, handlelength=2.2, columnspacing=2.4, labelcolor=muted)
            text(.055, .055, ('Full history: adjacent checkpoints averaged; first/latest exact. Recent panel: exact checkpoints on a separate y-axis.'
                 if averaged else 'Both panels show exact checkpoints. Recent panel uses a separate y-axis range.'), fontsize=8.1, color=muted)
            text(.055, .033, 'Band: original pointwise 95% intervals, not intervals for averages. Change card: paired comparison from the same fit.',
                 fontsize=8.1, color=muted)
            text(.945, .011, f'PL points = 173.7 × θ  ·  {report.get("bootstrap_refits", 0):,} bootstrap refits',
                 fontsize=8, color=muted, ha='right')
            metadata = dict(Title='Brass Birmingham — self-play PL strength', Date=None,
                Description=f'Common-fit PL progression through generation {latest}: {len(shown)} displayed points from {len(points)} checkpoints (cap {max_points}). '
                    + ('Adjacent checkpoints averaged for display; first/latest remain exact. ' if averaged else 'All displayed checkpoints are exact. ')
                    + f'Recent detail covers generations {recent_start}–{latest}: all {len(recent)} points exact, with a separate y-axis. '
                    f'Unmodified pointwise 95% setup-bootstrap intervals. Latest {score:.3f}, interval [{low:.3f}, {high:.3f}]. '
                    + (f'Paired change versus generation {previous[0]}: {change["delta"]:.3f}, '
                       f'95% interval [{change["low"]:.3f}, {change["high"]:.3f}]. ' if change else 'Paired change unavailable. ')
                    + '; '.join(coverage) + '. IL anchor 0; v4 reference ' + f'{v4:.3f}.')
            fig.savefig(output, format='svg', facecolor=paper, metadata=metadata)
            output = Path(output)
            output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf-8').splitlines()) + '\n',
                              encoding='utf-8', newline='\n')
            if preview is not None:
                fig.savefig(preview, format='png', dpi=150, facecolor=paper)
        finally:
            plt.close(fig)

