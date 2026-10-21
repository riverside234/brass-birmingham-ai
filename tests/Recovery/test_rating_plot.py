"""The automatic rating curve and redraw CLI share the saved progress style."""
import copy
import importlib.util
from pathlib import Path
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
