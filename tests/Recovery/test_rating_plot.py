"""The automatic rating curve and redraw CLI share the saved progress style."""
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

from brass_ai.recovery import rating_plot,ratings


ROOT=Path(__file__).resolve().parents[2]


def fixture():
    report=dict(games=256,setups=32,bootstrap_refits=200,ratings={
        'generation-0':dict(points=0.,points_ci95=[0.,0.]),
        'v4':dict(points=40.,points_ci95=[20.,60.]),
        # Percentile intervals are allowed to exclude the MAP point estimate.
        'generation-000010':dict(points=100.,points_ci95=[120.,150.]),
        'generation-000020':dict(points=180.,points_ci95=[140.,220.])})
    agents={name:dict(kind='v4') if name=='v4' else
        dict(kind='model',checkpoint=dict(generation=int(name.split('-')[1])))
        for name in report['ratings']}
    return report,agents


def long_fixture(count=47):
    report,agents=fixture()
    for i in range(3,count+1):
        name=f'generation-{i*10:06d}';score=100.+i*12+(i%5)*3
        report['ratings'][name]=dict(points=score,points_ci95=[score-40,score+50])
        agents[name]=dict(kind='model',checkpoint=dict(generation=i*10))
    return report,agents


class DisplayPoints(unittest.TestCase):
    def test_below_cap_preserves_every_checkpoint(self):
        for size in (1,2,29,30):
            with self.subTest(size=size):
                x=[10*i for i in range(1,size+1)];y=[i*i+.25 for i in x]
                shown=rating_plot.display_points(x,y)
                self.assertEqual([p['generation'] for p in shown],x)
                self.assertEqual([p['score'] for p in shown],y)
                self.assertTrue(all(p['count']==1 for p in shown))

    def test_long_irregular_histories_average_all_rows_once_and_keep_endpoints(self):
        for size in (31,32,47,100,1000):
            with self.subTest(size=size):
                x=[i*i+10 for i in range(size)];y=[i*2.5+(i%7)*11 for i in range(size)]
                shown=rating_plot.display_points(x,y)
                self.assertEqual(len(shown),30)
                self.assertEqual(sum(p['count'] for p in shown),size)
                self.assertEqual((shown[0]['generation'],shown[0]['score']),(x[0],y[0]))
                self.assertEqual((shown[-1]['generation'],shown[-1]['score']),(x[-1],y[-1]))
                cursor=0
                for row in shown:
                    end=cursor+row['count']
                    self.assertEqual((row['first'],row['last']),(x[cursor],x[end-1]))
                    self.assertAlmostEqual(row['generation'],sum(x[cursor:end])/row['count'])
                    self.assertAlmostEqual(row['score'],sum(y[cursor:end])/row['count'])
                    cursor=end
                self.assertEqual(cursor,size)
                middle=[p['count'] for p in shown[1:-1]]
                self.assertLessEqual(max(middle)-min(middle),1)

    def test_cap_is_strictly_validated(self):
        for cap in (None,True,2,0,-1,3.5,'30'):
            with self.subTest(cap=cap),self.assertRaisesRegex(ValueError,'integer of at least 3'):
                rating_plot.display_points([10],[1.],cap)
        self.assertEqual(len(rating_plot.display_points(list(range(20)),list(range(20)),3)),3)


class RatingPlot(unittest.TestCase):
    def test_automatic_and_standalone_render_identical_saved_style(self):
        report,agents=fixture();before=copy.deepcopy((report,agents))
        spec=importlib.util.spec_from_file_location('render_pl_progress',ROOT/'scripts/render-pl-progress.py')
        command=importlib.util.module_from_spec(spec);spec.loader.exec_module(command)
        original_fonttype=matplotlib.rcParams['svg.fonttype']
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            auto=Path(tmp)/'auto.svg';manual=Path(tmp)/'manual.svg'
            ratings.plot_report(auto,report,agents)
            command.render(report,manual)
            self.assertEqual(auto.read_bytes(),manual.read_bytes())
            root=ET.parse(auto).getroot();labels=' '.join(root.itertext())
            self.assertEqual(root.attrib['viewBox'],'0 0 950.4 590.4')
            for label in ('Self-play strength','GENERATION 20','180.0','140 – 220',
                          '256 games · 32 setups','200 bootstrap refits'):
                self.assertIn(label,labels)
            self.assertIsNone(root.find('.//{http://purl.org/dc/elements/1.1/}date'))
        self.assertEqual((report,agents),before)
        self.assertEqual(matplotlib.rcParams['svg.fonttype'],original_fonttype)
        self.assertEqual(plt.get_fignums(),[])

    def test_long_plot_caps_means_but_retains_raw_intervals_and_original_marker_sizes(self):
        report,agents=long_fixture();before=copy.deepcopy((report,agents))
        lines=[];scatters=[];bands=[];intervals=[]
        original_plot=Axes.plot;original_scatter=Axes.scatter
        original_band=Axes.fill_between;original_vlines=Axes.vlines
        def plot(ax,*args,**kwargs):
            result=original_plot(ax,*args,**kwargs);lines.extend(result);return result
        def scatter(ax,*args,**kwargs):
            result=original_scatter(ax,*args,**kwargs);scatters.append(result);return result
        def band(ax,x,y1,y2,*args,**kwargs):
            bands.append((list(x),list(y1),list(y2)))
            return original_band(ax,x,y1,y2,*args,**kwargs)
        def vlines(ax,x,ymin,ymax,*args,**kwargs):
            intervals.append((list(x),list(ymin),list(ymax)))
            return original_vlines(ax,x,ymin,ymax,*args,**kwargs)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            auto=Path(tmp)/'auto.svg';manual=Path(tmp)/'manual.svg'
            with patch.object(Axes,'plot',plot),patch.object(Axes,'scatter',scatter),\
                    patch.object(Axes,'fill_between',band),patch.object(Axes,'vlines',vlines):
                ratings.plot_report(auto,report,agents)
            rating_plot.render(report,manual)
            self.assertEqual(auto.read_bytes(),manual.read_bytes())
            labels=' '.join(ET.parse(auto).getroot().itertext())
            self.assertIn('30/47 points shown (cap 30)',labels)
            self.assertIn('first/latest exact',labels)
            self.assertIn('not a confidence interval for grouped means',labels)
        curve=next(line for line in lines if line.get_gid()=='pl-history-line')
        self.assertEqual(len(curve.get_xdata()),30)
        self.assertEqual(curve.get_xdata()[-1],470)
        self.assertEqual(curve.get_ydata()[-1],report['ratings']['generation-000470']['points'])
        markers=next(s for s in scatters if s.get_gid()=='pl-history-markers')
        self.assertEqual(len(markers.get_offsets()),29)
        self.assertEqual(list(markers.get_sizes()),[25])
        latest=next(s for s in scatters if s.get_gid()=='pl-latest-point')
        self.assertEqual(len(latest.get_offsets()),1)
        self.assertEqual(list(latest.get_sizes()),[61])
        self.assertEqual(bands[0][0],list(range(10,471,10)))
        self.assertEqual(bands[0][1],[report['ratings'][f'generation-{g:06d}']['points_ci95'][0]
            for g in range(10,471,10)])
        self.assertEqual(bands[0][2],[report['ratings'][f'generation-{g:06d}']['points_ci95'][1]
            for g in range(10,471,10)])
        self.assertEqual(intervals,[([470],[report['ratings']['generation-000470']['points_ci95'][0]],
            [report['ratings']['generation-000470']['points_ci95'][1]])])
        self.assertEqual((report,agents),before)

    def test_cli_supports_custom_cap_without_rewriting_report(self):
        report,_=long_fixture()
        spec=importlib.util.spec_from_file_location('render_pl_progress_cli',ROOT/'scripts/render-pl-progress.py')
        command=importlib.util.module_from_spec(spec);spec.loader.exec_module(command)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            source=Path(tmp)/'report.json';output=Path(tmp)/'progress.svg'
            source.write_text(json.dumps(report),encoding='utf-8');before=source.read_bytes()
            with patch.object(sys,'argv',['render-pl-progress.py','--report',str(source),'--max-points','12']),\
                    patch('builtins.print'):
                command.main()
            self.assertIn('12/47 points shown (cap 12)',' '.join(ET.parse(output).getroot().itertext()))
            self.assertEqual(source.read_bytes(),before)

    def test_percentile_endpoints_are_not_clamped_or_recentered(self):
        report,agents=fixture();calls=[];original=Axes.vlines
        def vlines(ax,x,ymin,ymax,*args,**kwargs):
            calls.append((list(x),list(ymin),list(ymax)))
            return original(ax,x,ymin,ymax,*args,**kwargs)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp,patch.object(Axes,'vlines',vlines):
            ratings.plot_report(Path(tmp)/'progress.svg',report,agents)
        self.assertEqual(calls[0],([10,20],[120.,140.],[150.,220.]))

    def test_report_only_redraw_ignores_temperature_comparison_aliases(self):
        report,_=fixture()
        report['ratings']['generation-000020-temperature-0.7']=dict(points=900.,points_ci95=[800.,1000.])
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            path=Path(tmp)/'progress.svg';rating_plot.render(report,path)
            labels=' '.join(ET.parse(path).getroot().itertext())
        self.assertIn('180.0',labels)
        self.assertNotIn('900.0',labels)

    def test_nonfinite_rating_fails_before_writing_output(self):
        report,_=fixture();report['ratings']['generation-000020']['points']=float('nan')
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
            path=Path(tmp)/'progress.svg'
            with self.assertRaisesRegex(ValueError,'Invalid PL point estimate'):
                rating_plot.render(report,path)
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
