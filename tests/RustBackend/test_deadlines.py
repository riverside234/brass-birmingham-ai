"""L12 owner retirement, batching, shared draws and failure boundaries."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT/'artifacts/rust-backend/20261025-l10-capture'
MANIFEST = ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/model/manifest.json'


class DeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ.get('BRASS_PAIR_BINARY', ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/debug/gumbel-pair.exe'))
        if not cls.binary.exists():
            raise unittest.SkipTest('Build gumbel-pair')

    def case(self, index=0, n=32, cap=1):
        source = json.loads((FIXTURES/f'brass-{index:03d}-request.json').read_text(encoding='utf-8'))['cases'][0]
        source['input']['shuffle_tape'] = None
        source['config']['simulations'] = n
        return dict(name=f'case-{index}', input=source['input'], config=source['config'],
                    mode=dict(kind='count'), allowance_ms=4000,
                    policy=dict(cap=cap, pending_limit=cap, max_candidate_cells=8192*cap, flush_us=0 if cap==1 else 2000))

    def invoke(self, cases, native=False, delay=0, success=True):
        if isinstance(cases, dict):
            cases=[cases]
        with tempfile.TemporaryDirectory(prefix='l12-', dir=ROOT/'artifacts') as folder:
            source=Path(folder)/'in.json'; target=Path(folder)/'out.json'
            body=dict(manifest=str(MANIFEST), manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                      scripted=not native, delay_ms=delay, cases=cases)
            source.write_text(json.dumps(body, allow_nan=False), encoding='utf-8')
            process=subprocess.run([self.binary,source,target], capture_output=True, timeout=90,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if not success:
                self.assertNotEqual(process.returncode,0)
                self.assertFalse(target.exists())
                self.assertFalse(target.with_suffix('.partial').exists())
                return process.stderr.decode(errors='replace')
            self.assertEqual(process.returncode,0,process.stderr.decode(errors='replace'))
            return json.loads(target.read_text(encoding='utf-8'))['results']

    def test_two_private_owners_common_draws_and_pooled_completed_counts(self):
        result=self.invoke(self.case())[0]['result']
        self.assertEqual(result['status'],'complete',result)
        decision=result['decision']; lanes=decision['lanes']
        self.assertEqual((decision['sessions'],decision['threads_per_session']),(2,1))
        self.assertEqual({l['owner'] for l in lanes},{'brass-search-0','brass-search-1'})
        self.assertEqual(lanes[0]['outcome']['gumbels'],lanes[1]['outcome']['gumbels'])
        self.assertTrue(all(l['outcome']['completed']==32 for l in lanes))
        pooled=decision['selection']['pool']
        self.assertEqual(sum(pooled['stats']['count']),64)
        self.assertEqual(pooled['stats']['count'],[a+b for a,b in zip(*[l['outcome']['snapshot']['stats']['count'] for l in lanes])])

    def test_missing_startup_has_zero_search_support(self):
        case=self.case();case['mode']=dict(kind='timed',startup=None)
        decision=self.invoke(case)[0]['result']['decision']
        self.assertEqual(decision['selection']['pool']['eligible'],[])
        self.assertTrue(all(a['support']==0 for a in decision['result']['actions']))
        self.assertEqual(decision['result']['budget']['reserve_ns'],400_000_000)

    def test_forced_and_debt_bypass_search(self):
        results=self.invoke([self.case(2),self.case(6)])
        for row in results:
            self.assertEqual(row['result']['status'],'complete',row)
            self.assertTrue(all(l['outcome']['completed']==0 for l in row['result']['decision']['lanes']))
            self.assertEqual(sum(l['neural']['positions'] for l in row['result']['decision']['lanes']),0)
        self.assertEqual(sum(l['neural']['positions'] for l in results[0]['result']['decision']['lanes']),0)

    def test_pending_cap_never_overshoots_round_or_count_quota(self):
        rows=self.invoke([self.case(n=23,cap=cap) for cap in (1,2,4)])
        for row in rows:
            decision=row['result']['decision']
            for lane in decision['lanes']:
                out=lane['outcome']
                self.assertEqual(out['completed'],23)
                self.assertEqual(out['snapshot']['support_watermark'],23)
                self.assertEqual(out['retired_pending'],0)
                self.assertLessEqual(out['schedule']['max_pending'],row['policy']['pending_limit'])

    def test_cancel_keeps_pair_busy_until_noncooperative_runtime_retires(self):
        case=self.case();case['cancel_after_ms']=100
        row=self.invoke(case,delay=250)[0]
        self.assertEqual(row['result']['status'],'retired')
        self.assertTrue(row['busy_after_cancel']);self.assertTrue(row['blocked_admission'])
        self.assertFalse(row['busy_after_retirement'])
        self.assertGreaterEqual(row['elapsed_ns'],200_000_000)
        self.assertNotIn('decision',row['result'])

    def test_late_runtime_reply_never_returns_an_action(self):
        case=self.case();case['allowance_ms']=50
        row=self.invoke(case,delay=150)[0]
        self.assertEqual(row['result']['status'],'retired')
        self.assertGreaterEqual(row['elapsed_ns'],120_000_000)
        self.assertFalse(row['busy_after_retirement'])

    def test_injected_stage_errors_retire_both_owners(self):
        cases=[]
        for stage in ('generation','sampling','observation','application','traversal','encoding','expansion','backup','publication','completion','inference'):
            case=self.case(n=64,cap=4)
            case.update(name=stage,faults=[dict(stage=stage,after=1),None])
            cases.append(case)
        for row in self.invoke(cases):
            self.assertEqual(row['result']['status'],'retired',row['name'])
            self.assertFalse(row['busy_after_retirement'])

    def test_actual_nn_batched_search_returns_valid_completed_plans(self):
        row=self.invoke(self.case(n=32,cap=4),native=True)[0]
        self.assertEqual(row['result']['status'],'complete',row)
        decision=row['result']['decision']
        self.assertTrue(all(l['neural']['positions']>0 for l in decision['lanes']))
        self.assertTrue(all(l['outcome']['completed']==32 for l in decision['lanes']))
        self.assertTrue(all(a[-1].startswith('finish|') for a in decision['result']['action_ids']))

    def test_invalid_batch_and_privileged_inputs_are_rejected(self):
        case=self.case();case['policy']['cap']=8
        self.assertIn('policy bounds',self.invoke(case,success=False))
        case=self.case();case['input']['opponent_hands']=[['location:Birmingham']]
        self.assertIn('unknown field',self.invoke(case,success=False))

    def test_all_four_allowances_keep_reference_reserve(self):
        cases=[]
        for seconds in (4,8,10,12):
            c=self.case(n=0);c['allowance_ms']=seconds*1000;cases.append(c)
        for seconds,row in zip((4,8,10,12),self.invoke(cases)):
            self.assertEqual(row['result']['decision']['result']['budget']['reserve_ns'],seconds*100_000_000)

    def test_later_case_failure_removes_partial_output(self):
        bad=self.case();bad['policy']['cap']=8
        self.assertIn('policy bounds',self.invoke([self.case(n=0),bad],success=False))


if __name__=='__main__':
    unittest.main()
