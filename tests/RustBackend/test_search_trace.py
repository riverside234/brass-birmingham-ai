"""L10 process boundary failures and privacy; full trace suite runs separately."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

@unittest.skipUnless(os.environ.get('BRASS_SEARCH_BINARY') and os.environ.get('BRASS_SEARCH_FIXTURES'),'set the qualified L10 binary and fixtures')
class SearchBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=Path(os.environ['BRASS_SEARCH_BINARY'])
        cls.fixtures=Path(os.environ['BRASS_SEARCH_FIXTURES'])
        cls.request=json.loads((cls.fixtures/'brass-000-request.json').read_text())

    def invoke(self,request,success=False):
        with tempfile.TemporaryDirectory(prefix='l10-',dir=ROOT/'artifacts') as folder:
            source=Path(folder)/'input.json';target=Path(folder)/'output.json'
            source.write_text(json.dumps(request),encoding='utf-8')
            p=subprocess.run([self.binary,source,target],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
            if success:
                self.assertEqual(p.returncode,0,p.stderr.decode(errors='replace'))
                return json.loads(target.read_text())['results'][0]
            self.assertNotEqual(p.returncode,0)
            self.assertFalse(target.exists())
            return p.stderr.decode(errors='replace')

    def changed(self):return copy.deepcopy(self.request)

    def test_root_only_permitted_input(self):
        r=self.changed();r['cases'][0]['input']['actual_opponent_hands']=[['location:Birmingham']]*4
        self.assertIn('unknown field',self.invoke(r))

    def test_recorded_response_identity_mismatch(self):
        r=self.changed();r['cases'][0]['replies'][1]['info']['key']['context']['profile_id']='foreign'
        self.assertIn('input mismatch',self.invoke(r))

    def test_missing_response(self):
        r=self.changed();r['cases'][0]['replies']=[]
        self.assertIn('tape exhausted',self.invoke(r))

    def test_nonfinite_response(self):
        r=self.changed();r['cases'][0]['replies'][0]['value'][0]=float('nan')
        self.invoke(r)

    def test_incomplete_legal_logits(self):
        r=self.changed();r['cases'][0]['replies'][0]['logits'].pop()
        self.assertIn('width mismatch',self.invoke(r))

    def test_wrong_reward_identity(self):
        r=self.changed();r['cases'][0]['input']['reward']['win']=2.
        self.assertIn('reward identity mismatch',self.invoke(r))

    def test_bad_root_budget(self):
        r=self.changed();r['cases'][0]['input']['context']['remaining'][0]=0.001
        self.assertIn('causal ledger',self.invoke(r))

    def test_no_privileged_descriptors(self):
        r=self.changed();r['cases'][0]['input']['context']['stable_descriptors']=[[0.,0.,0.,0.,1.]+[0.]*29]*4
        self.assertIn('unknown critic descriptors',self.invoke(r))

    def test_invalid_shuffle(self):
        r=self.changed();r['cases'][0]['input']['shuffle_tape'][0][0]=999
        self.assertIn('outside support',self.invoke(r))

    def test_depth_guard_does_not_bootstrap(self):
        r=self.changed();r['cases'][0]['config']['max_depth']=1
        self.assertIn('depth guard',self.invoke(r))

    def test_tree_cap_does_not_truncate_choices(self):
        r=self.changed();r['cases'][0]['config']['max_edges']=1
        self.assertIn('tree memory cap',self.invoke(r))

    def test_recorded_randomness_overrides_seed(self):
        r=self.changed();r['cases'][0]['config']['seed']=9983;r['cases'][0]['input']['sample_seed']=773
        a=self.invoke(r,success=True)
        expected=json.loads((self.fixtures/'brass-000-expected.json').read_text())
        self.assertEqual(a['selected_action'],expected['selected_action'])
        self.assertEqual([n['count'] for n in a['nodes']],[n['count'] for n in expected['nodes']])
        for n in a['nodes']:
            k=n['info']['key'];view=k['information']['observation']['view']
            self.assertEqual(n['info']['actor'],view['seat']);self.assertEqual(view['seat'],view['context']['actor'])
            self.assertNotIn('hands',view);self.assertNotIn('deck',view)

if __name__=='__main__':unittest.main()
