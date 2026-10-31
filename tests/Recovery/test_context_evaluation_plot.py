"""SVG semantic checks and synthetic visual fixtures, never live results."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
import torch
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery.context_evaluation_plot import render,VERSION,LABELS
from test_context_evaluation import fixture_spec,fixture_rows
from brass_ai.recovery.context_evaluation_analysis import analyze


def example(identical=False):
    s=fixture_spec();r=analyze(s,fixture_rows(s,identical));r['synthetic']=True
    r['cost']=dict(additional_seconds=132.,incremental_wall_fraction=.032)
    r['history']=[dict(generation=r['generation'],protocol_id=r['protocol_id'],mixture_vp=r['contexts'][-1]['paired']['vp'],
        incremental_wall_fraction=.032,additional_seconds=132.)]
    return r


def fixtures(out):
    r=example();cases={'normal':r,'identical':example(True),'incomplete':dict(r,status='incomplete')}
    wide=copy.deepcopy(r);wide['generation']=1000000
    for i,row in enumerate(wide['contexts'][1:]):row['paired']['vp']=dict(mean=50+i,ci95=[-240+i,-130+i])
    wide['history']=[dict(generation=wide['generation']-50*(11-i),protocol_id=r['protocol_id'],
        mixture_vp=dict(mean=i-4,ci95=[i-8,i+3]),incremental_wall_fraction=None if i in (3,4) else .025+.002*i,
        additional_seconds=100+i) for i in range(12)]
    cases['wide-history']=wide
    timings={}
    for name,value in cases.items():
        directory=out/name;directory.mkdir(parents=True,exist_ok=True)
        (directory/'report.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
        timings[name]=render(value,directory/'progress.svg',preview=directory/'preview.png')
    (out/'rendering.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
    return timings


class Plot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.report=example()

    def test_committed_values_labels_receipts_and_deterministic_svg(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            directory=Path(temp);r=copy.deepcopy(self.report);p=random.getstate();n=np.random.get_state();t=torch.get_rng_state().clone()
            with patch('brass_ai.recovery.context_evaluation_collect.collect',side_effect=AssertionError('no games')),\
                 patch('brass_ai.recovery.context_evaluation_analysis.analyze',side_effect=AssertionError('no statistics')):
                stats=render(r,directory/'a.svg');render(r,directory/'b.svg')
            raw=(directory/'a.svg').read_bytes();self.assertEqual(raw,(directory/'b.svg').read_bytes());ET.fromstring(raw)
            text=raw.decode()
            for name in ('DOMiNO context evaluation g1330','Uniform mixture','16 paired setups','384 new + 64 reused','SYNTHETIC FIXTURE',*LABELS):self.assertIn(name,text)
            self.assertLess(stats['bytes'],2_000_000);self.assertEqual(r,self.report)
            self.assertEqual(random.getstate(),p);np.testing.assert_equal(n,np.random.get_state());self.assertTrue(torch.equal(t,torch.get_rng_state()))

    def test_incomplete_and_future_history_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'a.svg';render(dict(self.report,status='incomplete'),path)
            self.assertIn('INCOMPLETE PANEL',path.read_text(encoding='utf-8'))
            bad=copy.deepcopy(self.report);bad['history'][0]['generation']+=1
            with self.assertRaises(ValueError):render(bad,path)


if __name__=='__main__':unittest.main()
