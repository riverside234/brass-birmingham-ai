"""L11 diagnostic boundary: caller bypasses, missing estimates and stale roots."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
FIXTURES=ROOT/'artifacts/rust-backend/20261025-l10-capture'

class SessionBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=Path(os.environ.get('BRASS_SESSION_BINARY',ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/debug/gumbel-session.exe'))
        if not cls.binary.exists():raise unittest.SkipTest('Build the L11 diagnostic binary')

    def case(self,index=0):
        c=json.loads((FIXTURES/f'brass-{index:03d}-request.json').read_text(encoding='utf-8'))['cases'][0]
        c.pop('kind');c.update(mode=dict(kind='count'),allowance_ms=4000)
        c['config']['simulations']=0
        return c

    def invoke(self,c,ok=True,native=False):
        with tempfile.TemporaryDirectory(prefix='l11-',dir=ROOT/'artifacts') as d:
            p=Path(d)/'input.json';out=Path(d)/'output.json'
            body=dict(cases=[c])
            if native:
                manifest=ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/model/manifest.json'
                body.update(manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
                c['replies']=None
            p.write_text(json.dumps(body,allow_nan=False),encoding='utf-8')
            result=subprocess.run([str(self.binary),str(p),str(out)],capture_output=True,text=True,timeout=30,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if not ok:
                self.assertNotEqual(result.returncode,0);self.assertFalse(out.exists());return result.stderr
            self.assertEqual(result.returncode,0,result.stderr)
            return json.loads(out.read_text(encoding='utf-8'))['results'][0]

    def test_missing_estimate_uses_bounded_fallback_without_support(self):
        c=self.case();c['mode']=dict(kind='timed',startup=None)
        r=self.invoke(c)['decisions'][0]
        self.assertEqual(r['completed'],0);self.assertIsNone(r['snapshot'])
        self.assertEqual(r['provenance'],'gumbel_prior_fallback')
        self.assertEqual(r['actions'][0]['support'],0)
        self.assertEqual(r['budget']['reserve_ns'],400_000_000)
        self.assertEqual(r['action_ids'][0][-1],'finish|||||')

    def test_forced_root_never_evaluates_or_searches(self):
        c=self.case(2);c['replies']=[];c['config']['simulations']=200
        out=self.invoke(c);r=out['decisions'][0]
        self.assertEqual(out['neural']['positions'],0)
        self.assertEqual(r['completed'],0);self.assertEqual(r['provenance'],'forced_completion')

    def test_debt_is_separate_non_search_path(self):
        c=self.case(6);c['config']['simulations']=200
        r=self.invoke(c)['decisions'][0]
        self.assertEqual(r['completed'],0);self.assertEqual(r['provenance'],'debt_policy')
        self.assertEqual(len(r['action_ids']),1);self.assertEqual(len(r['action_ids'][0]),1)
        self.assertTrue(r['action_ids'][0][0].startswith('liquidate|'))

    def test_same_root_cannot_receive_another_tournament(self):
        c=self.case();c['followups']=[copy.deepcopy(c['input'])]
        self.assertIn('root already attempted',self.invoke(c,False))

    def test_context_change_cannot_reset_turn(self):
        c=self.case();other=copy.deepcopy(c['input']);other['context']['profile_id']='different'
        c['followups']=[other];self.assertIn('turn context/utility changed',self.invoke(c,False))

    def test_stale_revision_and_changed_turn_are_rejected(self):
        c=self.case(3);other=copy.deepcopy(c['input']);other['view']['context']['revision']-=1
        c['followups']=[other];self.assertIn('stale revision',self.invoke(c,False))
        c=self.case();other=copy.deepcopy(c['input']);other['view']['context']['turn']=1 if other['view']['context']['turn']!=1 else 2
        c['followups']=[other];self.assertIn('different turn',self.invoke(c,False))

    def test_privileged_input_is_not_a_session_field(self):
        c=self.case();c['input']['opponent_hands']=[['location:Birmingham']]
        self.assertIn('unknown field',self.invoke(c,False))

    def test_startup_estimate_is_bound_to_export(self):
        c=self.case();ctx=c['input']['context']
        c['mode']=dict(kind='timed',startup=dict(model_id=ctx['model_id'],reward_id=ctx['reward_id'],export_sha256='b'*64,
            phase=c['input']['view']['context']['phase'],estimate=dict(nanoseconds=10_000_000,source='L10 measured fixture')))
        self.assertIn('startup estimate identity mismatch',self.invoke(c,False))

    def test_actual_nn_partial_leaf_requires_zero_support_completion(self):
        c=self.case();c['config']['simulations']=1
        out=self.invoke(c,native=True);r=out['decisions'][0]
        self.assertEqual(r['completed'],1);self.assertEqual(r['plan_events'],0)
        self.assertEqual(r['provenance'],'bounded_policy_completion')
        self.assertEqual(r['actions'][0]['support'],0);self.assertTrue(r['action_ids'][0][-1].startswith('finish|'))
        self.assertEqual(out['neural']['positions'],2)

if __name__=='__main__':unittest.main()
