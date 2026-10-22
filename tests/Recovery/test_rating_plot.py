"""Bounded display averages, exact evidence and shared automatic rendering."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from brass_ai.recovery import rating_plot, ratings


ROOT = Path(__file__).resolve().parents[2]


def fixture():
    report = dict(games=256, setups=32, bootstrap_refits=200,
        protocol=dict(main_games=128, panel_games=64), ratings={
        'generation-0': dict(points=0., points_ci95=[0., 0.]),
        'v4': dict(points=40., points_ci95=[20., 60.]),
        # A percentile interval can exclude its MAP point estimate.
        'generation-000010': dict(points=100., points_ci95=[120., 150.]),
        'generation-000020': dict(points=180., points_ci95=[140., 220.])},
        differences=[dict(a='generation-000010', b='generation-000020',
                          points_delta=80., points_delta_ci95=[15., 125.])])
    agents = {name: dict(kind='v4') if name == 'v4' else
        dict(kind='model', checkpoint=dict(generation=int(name.split('-')[1])))
        for name in report['ratings']}
    return report, agents


def long_fixture(count=47):
    report, agents = fixture()
    for i in range(3, count+1):
        name = f'generation-{i*10:06d}'; score = 100.+i*12+(i % 5)*3
        report['ratings'][name] = dict(points=score, points_ci95=[score-40, score+50])
        agents[name] = dict(kind='model', checkpoint=dict(generation=i*10))
    previous, latest = (f'generation-{g*10:06d}' for g in (count-1, count))
    delta = report['ratings'][latest]['points']-report['ratings'][previous]['points']
    report['differences'].append(dict(a=previous, b=latest, points_delta=delta,
                                     points_delta_ci95=[delta-40., delta+50.]))
    return report, agents


def command():
    spec = importlib.util.spec_from_file_location('render_pl_progress_cli', ROOT/'scripts/render-pl-progress.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def labels(path):
    return ' '.join(ET.parse(path).getroot().itertext())


class DisplayPoints(unittest.TestCase):
    def test_short_histories_remain_exact(self):
        for size in (1, 2, 29, 30):
            x = [10*i for i in range(size)]; y = [i*i+.25 for i in x]
            shown = rating_plot.display_points(x, y)
            self.assertEqual([p['generation'] for p in shown], x)
            self.assertEqual([p['score'] for p in shown], y)
            self.assertTrue(all(p['count'] == 1 for p in shown))

    def test_long_histories_average_every_row_once_and_keep_first_latest_exact(self):
        for size in (1, 2, 30, 31, 47, 100, 1000):
            x = [i*i+10 for i in range(size)]; y = [i*2.5+(i % 7)*11 for i in range(size)]
            for cap in (3, 12, 30):
                with self.subTest(size=size, cap=cap):
                    shown = rating_plot.display_points(x, y, cap)
                    self.assertEqual(len(shown), min(cap, size))
                    self.assertEqual(sum(p['count'] for p in shown), size)
                    self.assertEqual((shown[0]['generation'], shown[0]['score']), (x[0], y[0]))
                    self.assertEqual((shown[-1]['generation'], shown[-1]['score']), (x[-1], y[-1]))
                    cursor = 0
                    for row in shown:
                        end = cursor+row['count']
                        self.assertEqual((row['first'], row['last']), (x[cursor], x[end-1]))
                        self.assertAlmostEqual(row['generation'], sum(x[cursor:end])/row['count'])
                        self.assertAlmostEqual(row['score'], sum(y[cursor:end])/row['count'])
                        cursor = end
                    self.assertEqual(cursor, size)

    def test_invalid_cap_or_mismatched_data_is_rejected(self):
        for cap in (None, True, 2, 0, -1, 3.5, '30', 31, 100):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, 'integer from 3 to 30'):
                rating_plot.display_points([10], [1.], cap)
        for x, y in (([], []), ([10], []), ([10, 20], [1.])):
            with self.assertRaisesRegex(ValueError, 'matching nonempty'):
                rating_plot.display_points(x, y)


class PairedEvidence(unittest.TestCase):
    def pair(self, report):
        return rating_plot.paired_change(report, 'generation-000010', 'generation-000020')

    def test_saved_pair_is_used_not_independent_marginal_intervals(self):
        report, _ = fixture()
        result = self.pair(report)
        self.assertEqual(result, dict(delta=80., low=15., high=125., status='improved'))
        report['ratings']['generation-000010']['points_ci95'] = [-1000., 1000.]
        self.assertEqual(self.pair(report), result)

    def test_reversed_pair_flips_delta_and_interval_endpoints(self):
        report, _ = fixture()
        report['differences'] = [dict(a='generation-000020', b='generation-000010',
            points_delta=-80., points_delta_ci95=[-125., -15.])]
        self.assertEqual(self.pair(report), dict(delta=80., low=15., high=125., status='improved'))

    def test_crossing_and_touching_zero_are_uncertain(self):
        for interval in ([-10., 125.], [0., 125.], [-10., 0.]):
            report, _ = fixture(); report['differences'][0]['points_delta_ci95'] = interval
            self.assertEqual(self.pair(report)['status'], 'uncertain')

    def test_decline_requires_negative_upper_bound(self):
        report, _ = fixture()
        report['ratings']['generation-000020']['points'] = 80.
        report['differences'][0].update(points_delta=-20., points_delta_ci95=[-40., -1.])
        self.assertEqual(self.pair(report)['status'], 'declined')

    def test_missing_pair_remains_unknown(self):
        report, _ = fixture(); report.pop('differences')
        self.assertIsNone(self.pair(report))

    def test_inconsistent_duplicate_nonfinite_or_invalid_pair_rejected(self):
        for mutation in ('wrong-fit', 'duplicate', 'nan', 'reversed-bounds'):
            report, _ = fixture(); pair = report['differences'][0]
            if mutation == 'wrong-fit': pair['points_delta'] = 79.
            if mutation == 'duplicate': report['differences'].append(copy.deepcopy(pair))
            if mutation == 'nan': pair['points_delta_ci95'][0] = float('nan')
            if mutation == 'reversed-bounds': pair['points_delta_ci95'] = [120., 100.]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.pair(report)

    def test_slot_and_accumulated_counts_are_distinct_panels_excluded(self):
        report, _ = fixture()
        self.assertEqual(rating_plot.coverage_labels(report, 2),
            ('Latest slot: 128 games / 32 setups', 'Common fit: 256 games', '2 rated checkpoints'))
        report.pop('protocol')
        self.assertEqual(rating_plot.coverage_labels(report, 2)[0], 'Latest slot: support not recorded')


class RatingPlot(unittest.TestCase):
    def test_automatic_and_standalone_render_identical_style_without_mutations(self):
        report, agents = fixture(); before = copy.deepcopy((report, agents))
        original_fonttype = matplotlib.rcParams['svg.fonttype']
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            auto = Path(tmp)/'auto.svg'; manual = Path(tmp)/'manual.svg'
            ratings.plot_report(auto, report, agents); command().render(report, manual)
            self.assertEqual(auto.read_bytes(), manual.read_bytes())
            root = ET.parse(auto).getroot(); text = labels(auto)
            self.assertEqual(root.attrib['viewBox'], '0 0 1296 604.8')
            for label in ('Full training history', 'Recent progress', 'GENERATION 20',
                          '180.0', '140.0 to 220.0', '+80.0', '+15.0 to +125.0',
                          'Improvement supported', 'Latest slot: 128 games / 32 setups',
                          'Common fit: 256 games', '200 bootstrap refits'):
                self.assertIn(label, text)
            self.assertIsNone(root.find('.//{http://purl.org/dc/elements/1.1/}date'))
        self.assertEqual((report, agents), before)
        self.assertEqual(matplotlib.rcParams['svg.fonttype'], original_fonttype)
        self.assertEqual(plt.get_fignums(), [])

    def test_long_plot_limits_dots_while_recent_points_and_all_bounds_remain_raw(self):
        report, agents = long_fixture(); before = copy.deepcopy((report, agents))
        lines = []; scatters = []; bands = []; intervals = []
        original_plot = Axes.plot; original_scatter = Axes.scatter
        original_band = Axes.fill_between; original_vlines = Axes.vlines
        def plot(ax, *args, **kwargs):
            result = original_plot(ax, *args, **kwargs); lines.extend(result); return result
        def scatter(ax, *args, **kwargs):
            result = original_scatter(ax, *args, **kwargs); scatters.append(result); return result
        def band(ax, x, y1, y2, *args, **kwargs):
            bands.append((list(x), list(y1), list(y2)))
            return original_band(ax, x, y1, y2, *args, **kwargs)
        def vlines(ax, x, ymin, ymax, *args, **kwargs):
            intervals.append((list(x), list(ymin), list(ymax)))
            return original_vlines(ax, x, ymin, ymax, *args, **kwargs)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            auto = Path(tmp)/'auto.svg'
            with patch.object(Axes, 'plot', plot), patch.object(Axes, 'scatter', scatter), \
                 patch.object(Axes, 'fill_between', band), patch.object(Axes, 'vlines', vlines):
                ratings.plot_report(auto, report, agents)
            text = labels(auto)
            for label in ('47 rated checkpoints', 'Change uncertain', 'g270–g470',
                          '30 points / 47 checkpoints', 'first/latest exact',
                          'not intervals for averages', 'Recent window'):
                self.assertIn(label, text)
        curve = next(line for line in lines if line.get_gid() == 'pl-history-line')
        raw_x = list(range(10, 471, 10))
        raw_y = [report['ratings'][f'generation-{g:06d}']['points'] for g in raw_x]
        expected = rating_plot.display_points(raw_x, raw_y)
        self.assertEqual(list(curve.get_xdata()), [row['generation'] for row in expected])
        self.assertEqual(list(curve.get_ydata()), [row['score'] for row in expected])
        self.assertEqual(len(curve.get_xdata()), 30)
        recent = next(line for line in lines if line.get_gid() == 'pl-recent-line')
        self.assertEqual(list(recent.get_xdata()), list(range(270, 471, 10)))
        self.assertEqual(list(recent.get_ydata()), raw_y[-21:])
        markers = next(s for s in scatters if s.get_gid() == 'pl-history-markers')
        self.assertEqual(len(markers.get_offsets()), 29)
        latest = next(s for s in scatters if s.get_gid() == 'pl-latest-point')
        self.assertEqual(len(markers.get_offsets())+len(latest.get_offsets()), 30)
        for band, generations in zip(bands, (list(range(10, 471, 10)), list(range(270, 471, 10)))):
            self.assertEqual(band[0], generations)
            for offset in (0, 1):
                self.assertEqual(band[1+offset],
                    [report['ratings'][f'generation-{g:06d}']['points_ci95'][offset] for g in generations])
        self.assertEqual(intervals[1], bands[1])
        self.assertEqual((report, agents), before)

    def test_cli_smaller_cap_averages_history_only_without_rewriting_report(self):
        report, _ = long_fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            source = Path(tmp)/'report.json'; output = Path(tmp)/'progress.svg'
            source.write_text(json.dumps(report), encoding='utf-8'); before = source.read_bytes()
            with patch.object(sys, 'argv', ['render-pl-progress.py', '--report', str(source), '--max-points', '12']), \
                 patch('builtins.print'):
                command().main()
            self.assertIn('12 displayed points from 47 checkpoints (cap 12)', labels(output))
            self.assertIn('all 21 points exact', labels(output))
            self.assertEqual(source.read_bytes(), before)

    def test_summary_cards_are_shorter_and_plot_area_is_taller(self):
        report, _ = long_fixture(); snapshots = []
        from matplotlib.figure import Figure
        from matplotlib.patches import FancyBboxPatch
        save = Figure.savefig
        def snapshot(figure, *args, **kwargs):
            snapshots.append(([a.get_height() for a in figure.artists if isinstance(a, FancyBboxPatch)],
                              [ax.get_position().height for ax in figure.axes]))
            return save(figure, *args, **kwargs)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp, patch.object(Figure, 'savefig', snapshot):
            rating_plot.render(report, Path(tmp)/'progress.svg')
        cards, plots = snapshots[0]
        self.assertEqual(len(cards), 3)
        self.assertTrue(all(height < .10 for height in cards))
        self.assertEqual(len(plots), 2)
        self.assertTrue(all(height > .52 for height in plots))

    def test_latest_label_has_its_own_gutter_and_stays_inside_canvas(self):
        from matplotlib.figure import Figure

        save = Figure.savefig
        for case in ('normal', 'flat', 'negative', 'large', 'single'):
            report, _ = long_fixture()
            report.pop('differences')
            score = report['ratings']['generation-000470']['points']
            if case in ('negative', 'large'):
                score = -500. if case == 'negative' else 12345678.9
                report['ratings']['generation-000470'] = dict(points=score, points_ci95=[score-40, score+50])
            elif case == 'flat':
                for generation in range(270, 471, 10):
                    report['ratings'][f'generation-{generation:06d}'] = dict(
                        points=score, points_ci95=[score-40, score+50])
            elif case == 'single':
                report['ratings'] = {k: v for k, v in report['ratings'].items()
                    if k in ('generation-0', 'v4', 'generation-000470')}
            before = copy.deepcopy(report)
            snapshots = []

            def snapshot(figure, *args, **kwargs):
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                axes = figure.axes[1]
                label = next(t for t in axes.texts if t.get_gid() == 'pl-recent-latest-label')
                snapshots.append((label.get_bbox_patch().get_window_extent(renderer).frozen(),
                    axes.get_window_extent(renderer).frozen(), figure.bbox.frozen(), label.xy))
                return save(figure, *args, **kwargs)

            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp, \
                    patch.object(Figure, 'savefig', snapshot):
                path = Path(tmp)/'progress.svg'
                rating_plot.render(report, path)
                badge, plot, canvas, target = snapshots[0]
                self.assertGreater(badge.x0, plot.x1 + 3)
                self.assertGreaterEqual(badge.y0, plot.y0)
                self.assertLessEqual(badge.y1, plot.y1)
                self.assertLess(badge.x1, canvas.x1 - 3)
                self.assertEqual(target, (470, score))
                self.assertEqual(report, before)
                root = ET.parse(path).getroot()
                self.assertIsNotNone(root.find(".//*[@id='pl-recent-latest-label']"))
                self.assertIsNotNone(root.find(".//*[@id='pl-recent-latest-leader']"))

    def test_percentile_endpoints_are_not_clamped_or_recentered(self):
        report, agents = fixture(); calls = []; original = Axes.vlines
        def vlines(ax, x, ymin, ymax, *args, **kwargs):
            calls.append((list(x), list(ymin), list(ymax)))
            return original(ax, x, ymin, ymax, *args, **kwargs)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp, patch.object(Axes, 'vlines', vlines):
            ratings.plot_report(Path(tmp)/'progress.svg', report, agents)
        self.assertIn(([10, 20], [120., 140.], [150., 220.]), calls)

    def test_temperature_aliases_and_unrated_agents_are_not_checkpoints(self):
        report, agents = fixture()
        report['ratings']['generation-000020-temperature-0.7'] = dict(points=900., points_ci95=[800., 1000.])
        agents['not-yet-rated'] = dict(kind='model', checkpoint=dict(generation=30))
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            auto = Path(tmp)/'auto.svg'; manual = Path(tmp)/'manual.svg'
            ratings.plot_report(auto, report, agents); rating_plot.render(report, manual)
            self.assertEqual(auto.read_bytes(), manual.read_bytes())
            self.assertNotIn('900.0', labels(manual))

    def test_no_active_kl_reference_marker_or_read_dependency(self):
        report, _ = fixture()
        report['coverage'] = dict(active_kl_reference='generation-000010')
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            with_anchor = Path(tmp)/'a.svg'; without = Path(tmp)/'b.svg'
            rating_plot.render(report, with_anchor)
            report['coverage']['active_kl_reference'] = 'missing-checkpoint'
            rating_plot.render(report, without)
            self.assertEqual(with_anchor.read_bytes(), without.read_bytes())
            self.assertNotIn('KL reference', labels(with_anchor))

    def test_one_checkpoint_and_missing_paired_evidence_are_explicit(self):
        for first in (True, False):
            report, _ = fixture(); report.pop('differences')
            if first: report['ratings'].pop('generation-000010')
            with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
                path = Path(tmp)/'progress.svg'; rating_plot.render(report, path)
                text = labels(path)
                self.assertIn('First rated checkpoint' if first else 'Paired interval not recorded', text)
                self.assertNotIn('Improvement supported', text)

    def test_nonfinite_rating_fails_before_writing_output(self):
        report, _ = fixture(); report['ratings']['generation-000020']['points'] = float('nan')
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            path = Path(tmp)/'progress.svg'
            with self.assertRaisesRegex(ValueError, 'Invalid PL point estimate'):
                rating_plot.render(report, path)
            self.assertFalse(path.exists())

    def test_writer_failure_closes_figure(self):
        report, _ = fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            with self.assertRaises(OSError):
                rating_plot.render(report, Path(tmp)/'missing'/'progress.svg')
        self.assertEqual(plt.get_fignums(), [])

    def test_training_slot_calls_real_renderer_and_restart_keeps_chart(self):
        # Eight authored outcomes and four refits; no host, model or training.
        from test_strength_rating import Slots, fixture_rows
        from brass_ai.recovery.quality_config import RATING
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            root = Path(tmp); c = dict(RATING, setups=2, panel_setups=1, bootstrap=4)
            state = Slots().state(root, c); cfg = dict(rating=c, host='host', execution={})
            with patch('brass_ai.recovery.retention.slim_checkpoint', side_effect=lambda root,run,e,old=None:e), \
                 patch.object(ratings, 'collect_slot', side_effect=lambda root,directory,spec,*args: fixture_rows(spec)) as collector:
                fitted = ratings.schedule(root, root, state, cfg)
                path = root/'ratings/slots/generation-000010/progress.svg'; before = path.read_bytes()
                self.assertIn('Full training history', labels(path))
                self.assertIn('Recent progress', labels(path))
                self.assertIn('Latest slot: 8 games / 2 setups', labels(path))
                self.assertEqual(fitted['games'], 8)
                self.assertIsNone(ratings.schedule(root, root, state, cfg))
                self.assertEqual(path.read_bytes(), before)
                self.assertEqual(collector.call_count, 1)


if __name__ == '__main__':
    unittest.main()
