"""Committed self-play VP attribution, retention, grouping and chart integration."""
import copy
import gzip
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from brass_ai.recovery import rating_plot, vp_plot
from test_rating_plot import command, fixture


ROOT = Path(__file__).resolve().parents[2]


def summary(iteration, total=400., seats=4):
    return dict(iteration=iteration, health_by_composition={
        'selfplay': dict(vp_sum=total, seat_games=seats, tables=seats//4),
        'historical': dict(vp_sum=90000., seat_games=200),
        'legacy': dict(vp_sum=80000., seat_games=100)},
        health_counts=dict(vp_sum=999999., seat_games=1000),
        branching=dict(vp_sum=70000., seat_games=100))


class VPProgress(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'artifacts')
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.state = dict(iteration=23, history=[dict(generation=0,
            path='artifacts/run/initial.pt', sha256='a'*64, model_id='b'*64)])
        self.write('state.json', self.state)

    def write(self, name, value):
        path = self.run/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def retained(self, rows):
        return {f'iteration-{r["iteration"]:04d}/report.json': dict(summary=r) for r in rows}

    def test_fixed_groups_pool_totals_not_iteration_means_or_mixed_tables(self):
        rows = [summary(i) for i in range(23)]
        rows[0] = summary(0, 0.)  # Zero VP is still four learner-seat games.
        rows[9] = summary(9, 8000., 40)
        self.write('retention-history.json', self.retained(rows))
        history = vp_plot.collect(self.run)
        data = vp_plot.groups(history, 23)
        self.assertEqual([(r['first'], r['last']) for r in data], [(1, 10), (11, 20)])
        self.assertEqual(data[0]['seat_games'], 76)
        self.assertAlmostEqual(data[0]['mean_vp'], (8*400+8000)/76)
        self.assertEqual(data[1]['mean_vp'], 100.)
        self.assertEqual(len(history['iterations']), 23)
        self.assertEqual(len(vp_plot.groups(history, 19)), 1)

    def test_full_compact_recent_and_gzip_deduplicate_and_exclude_pending(self):
        self.state['iteration'] = 10
        self.state['recent_training'] = [summary(9)]
        self.write('state.json', self.state)
        with gzip.open(self.run/'retention-history.json.gz', 'wt', encoding='utf-8') as f:
            json.dump(self.retained([summary(i) for i in range(10)]), f)
        full = dict(iteration=9, by_composition=dict(selfplay=dict(
            health_counts=summary(9)['health_by_composition']['selfplay'])))
        self.write('iteration-0009/report.json', full)
        # Even malformed pending output is not loaded.
        self.write('iteration-0010/report.json', {'unfinished': True})
        before = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        data = vp_plot.groups(vp_plot.collect(self.run), 10)
        self.assertEqual((data[0]['mean_vp'], data[0]['seat_games']), (100., 40))
        self.assertTrue(all(p.read_bytes() == raw for p, raw in before.items()))

    def test_seat_sum_fallback_is_still_selfplay_only(self):
        self.write('iteration-0000/report.json', dict(iteration=0, by_composition={
            'selfplay': dict(learner_by_seat={'0': dict(vp_sum=0., seat_games=2),
                                            '1': dict(vp_sum=200., seat_games=2)}),
            'historical': dict(learner_by_seat={'0': dict(vp_sum=90000., seat_games=99)})}))
        row = vp_plot.collect(self.run)['iterations'][0]
        self.assertEqual(row, dict(generation=1, vp_sum=200., seat_games=4))

    def test_history_survives_report_retention_and_restart(self):
        self.state['iteration'] = 10; self.write('state.json', self.state)
        self.write('retention-history.json', self.retained([summary(i) for i in range(10)]))
        history = vp_plot.collect(self.run)
        vp_plot.save_history(history, self.run/vp_plot.HISTORY_PATH)
        self.state['iteration'] = 20; self.write('state.json', self.state)
        self.write('retention-history.json', self.retained([summary(i, 480.) for i in range(10, 20)]))
        resumed = vp_plot.collect(self.run)
        self.assertEqual([g['mean_vp'] for g in vp_plot.groups(resumed, 20)], [100., 120.])
        vp_plot.save_history(resumed, self.run/vp_plot.HISTORY_PATH)
        before = (self.run/vp_plot.HISTORY_PATH).read_bytes()
        vp_plot.save_history(vp_plot.collect(self.run), self.run/vp_plot.HISTORY_PATH)
        self.assertEqual((self.run/vp_plot.HISTORY_PATH).read_bytes(), before)
        # A restored earlier commit cannot show future generations from the cache.
        self.state['iteration'] = 10; self.write('state.json', self.state)
        self.assertEqual(len(vp_plot.collect(self.run)['iterations']), 10)

    def test_older_bundles_require_same_lineage_and_consistent_overlap(self):
        older = self.run/'older'; older.mkdir()
        (older/'state.json').write_text(json.dumps(dict(self.state, iteration=10)), encoding='utf-8')
        (older/'retention-history.json').write_text(json.dumps(self.retained([summary(i) for i in range(10)])), encoding='utf-8')
        self.write('retention-history.json', self.retained([summary(i) for i in range(9, 20)]))
        merged = vp_plot.collect(self.run, history_sources=[older])
        self.assertEqual(len(merged['iterations']), 20)
        self.write('iteration-0009/report.json', summary(9, 999.))
        with self.assertRaisesRegex(ValueError, 'Conflicting.*generation 10'):
            vp_plot.collect(self.run, history_sources=[older])
        self.state['history'][0]['path'] = 'artifacts/another-run/initial.pt'
        self.write('state.json', self.state)
        with self.assertRaisesRegex(ValueError, 'different run lineage'):
            vp_plot.collect(self.run, history_sources=[older])

    def test_invalid_totals_and_wrong_iteration_identity_are_rejected(self):
        self.write('iteration-0000/report.json', summary(1))
        with self.assertRaisesRegex(ValueError, 'Iteration identity'):
            vp_plot.collect(self.run)
        self.write('iteration-0000/report.json', summary(0, math.nan))
        with self.assertRaisesRegex(ValueError, 'Invalid self-play learner VP'):
            vp_plot.collect(self.run)

    def test_missing_groups_take_no_overview_space_and_still_break_recent_lines(self):
        self.state['iteration'] = 33; self.write('state.json', self.state)
        self.write('retention-history.json', self.retained([summary(i, 400.+i) for i in
                   [*range(10), *range(20, 33)]]))
        history = vp_plot.collect(self.run); before = copy.deepcopy(history)
        data = vp_plot.groups(history, 33)
        self.assertIsNone(data[1]['mean_vp'])
        calls = []; original = Axes.plot
        def capture(axes, x, y, *args, **kwargs):
            calls.append((x, y))
            return original(axes, x, y, *args, **kwargs)
        with patch.object(Axes, 'plot', capture):
            vp_plot.render(history, self.run/'progress_vp.svg')
        self.assertEqual(calls[0][0], [0, 1])
        self.assertTrue(all(math.isfinite(y) for y in calls[0][1]))
        self.assertEqual(calls[1][0], [10, 20, 30])
        self.assertTrue(math.isnan(calls[1][1][1]))
        text = ' '.join(ET.parse(self.run/'progress_vp.svg').getroot().itertext())
        self.assertIn('1 groups missing evidence', text)
        self.assertIn('3 iterations in unfinished group', text)
        self.assertIn('Previous adjacent group unavailable', text)
        self.assertEqual(history, before)
        self.assertEqual(plt.get_fignums(), [])

    def test_late_and_sparse_history_fills_overview_with_real_generation_labels(self):
        from matplotlib.figure import Figure
        history=dict(committed_generation=1507,iterations=[dict(generation=g,seat_games=4,
            vp_sum=0. if g<=1300 else 400.) for g in (*range(1291,1311),*range(1491,1501))])
        captured=[];save=Figure.savefig
        def capture(fig,*args,**kwargs):
            ax=fig.axes[0]
            line=next(line for line in ax.lines if line.get_gid()=='vp-history-line')
            captured.append((list(line.get_xdata()),list(line.get_ydata()),ax.get_xlim(),
                             [t.get_text() for t in ax.get_xticklabels()]))
            return save(fig,*args,**kwargs)
        with patch.object(Figure,'savefig',capture):vp_plot.render(history,self.run/'sparse.svg')
        x,y,limits,labels=captured[0]
        self.assertEqual(x,[0,1,2]);self.assertEqual(y,[0.,100.,100.])
        self.assertEqual(limits,(-.5,2.5));self.assertEqual(labels,['1300','1310','1500'])
        self.assertIn('missing VP groups omitted',(self.run/'sparse.svg').read_text(encoding='utf-8'))

    def test_sparse_overview_pools_only_available_groups_with_exact_endpoints(self):
        history=dict(committed_generation=2000,iterations=[dict(generation=g,seat_games=4+g%3,vp_sum=float(g))
            for g in range(1,2001) if (g-1)//10%2==0])
        data=vp_plot.groups(history,2000);valid=[r for r in data if r['mean_vp'] is not None]
        shown=vp_plot.display_groups(data,30)
        self.assertEqual(len(shown),30)
        self.assertEqual((shown[0],shown[-1]),(valid[0],valid[-1]))
        self.assertEqual(sum(r['seat_games'] for r in shown),sum(r['seat_games'] for r in valid))
        self.assertEqual(math.fsum(r['vp_sum'] for r in shown),math.fsum(r['vp_sum'] for r in valid))
        self.assertTrue(all(r['mean_vp']==r['vp_sum']/r['seat_games'] for r in shown))

    def test_cli_generates_both_charts_without_changing_source_evidence(self):
        report, _ = fixture()
        path = self.write('ratings/slots/generation-000020/report.json', report)
        self.write('retention-history.json', self.retained([summary(i, 400.+i) for i in range(23)]))
        before = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        import sys
        with patch.object(sys, 'argv', ['render-pl-progress.py', '--report', str(path)]):
            command().main()
        self.assertTrue(path.with_name('progress.svg').is_file())
        svg = path.with_name('progress_vp.svg')
        self.assertTrue(svg.is_file())
        text = ' '.join(ET.parse(svg).getroot().itertext())
        self.assertIn('Self-play learner VP', text)
        self.assertIn('g11–g20', text)
        self.assertIn('103.62', text)
        self.assertTrue(all(p.read_bytes() == raw for p, raw in before.items()))
        self.assertFalse((self.run/vp_plot.HISTORY_PATH).exists())

    def test_companion_is_empty_when_history_is_unavailable(self):
        report, _ = fixture()
        rating_plot.render_progress(report, self.run/'progress.svg')
        text = ' '.join(ET.parse(self.run/'progress_vp.svg').getroot().itertext())
        self.assertIn('No complete group with VP evidence', text)
        self.assertNotIn('0.00', text)

    def test_vp_writer_failure_closes_figure(self):
        with self.assertRaises(OSError):
            vp_plot.render(dict(committed_generation=10, iterations=[]), self.run/'missing/progress_vp.svg')
        self.assertEqual(plt.get_fignums(), [])

    def test_custom_pl_output_cannot_overwrite_the_companion_chart(self):
        path = self.run/'progress_vp.svg'; path.write_text('existing VP chart')
        report, _ = fixture()
        with self.assertRaisesRegex(ValueError, 'reserved for the companion VP chart'):
            rating_plot.render_progress(report, path)
        self.assertEqual(path.read_text(), 'existing VP chart')


if __name__ == '__main__':
    unittest.main()
