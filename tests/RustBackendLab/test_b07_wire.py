"""Focused array framing, cancellation, forced RNG and native recovery checks."""
import copy
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time
import unittest
from unittest.mock import Mock
from brass_ai.rust_backend.lab_transport import Pipe,Collector,DecisionEncoder,MAX_ARRAY_FRAME
import test_b02_timeout as transport_tests


class Wire(transport_tests.Transport):
    def test_sampled_timers_keep_all_bytes_and_calls_without_claiming_full_times(self):
        source='''import json,struct,sys
r=sys.stdin.buffer;w=sys.stdout.buffer
for i in range(65):
 n=struct.unpack('<I',r.read(4))[0];q=json.loads(r.read(n))
 v=json.dumps(dict(ok=True,id=q['id'],protocol=q['protocol'],data={})).encode()
 w.write(struct.pack('<I',len(v))+v);w.flush()
'''
        pipe=self.pipe(source,timeout=3);pipe.sample_every=64
        for _ in range(65):pipe.call('echo')
        self.assertEqual(pipe.metrics['echo_calls'],65);self.assertEqual(pipe.metrics['timing_sampled_calls'],1)
        self.assertEqual(pipe.metrics['received_bytes'],pipe.metrics['echo_received_bytes'])
        self.assertGreater(pipe.metrics['reply_decode_sampled_seconds'],0.)
        self.assertNotIn('round_trip_seconds',pipe.metrics)
    # Inherit the existing real subprocess fragmentation/deadline coverage.
    def test_array_reply_is_bounded_and_exact(self):
        source='''import json,struct,sys
r=sys.stdin.buffer;n=struct.unpack('<I',r.read(4))[0];req=json.loads(r.read(n))
raw=bytes(range(256))*900
header=json.dumps(dict(ok=True,id=req['id'],protocol=req['protocol'],data={'bytes':len(raw)})).encode()
data=struct.pack('<I',len(header))+header+raw
sys.stdout.buffer.write(struct.pack('<I',len(data)|(1<<31))+data);sys.stdout.buffer.flush()
'''
        pipe=self.pipe(source,timeout=3);reply=pipe.call('array')
        self.assertEqual(bytes(reply['_arrays']),bytes(range(256))*900)
        self.assertEqual(pipe.metrics['calls'],1)

    def test_oversized_array_or_truncated_header_closes_lane(self):
        for data in (struct.pack('<I',(MAX_ARRAY_FRAME+1)|(1<<31)),struct.pack('<II',4|(1<<31),99)):
            source=f"import sys,struct;r=sys.stdin.buffer;n=struct.unpack('<I',r.read(4))[0];r.read(n);sys.stdout.buffer.write({data!r});sys.stdout.buffer.flush()"
            pipe=self.pipe(source,timeout=3)
            with self.assertRaises(ValueError):pipe.call('bad')
            self.assertTrue(pipe.closed)

    def test_inflight_cancellation_retires_process(self):
        pipe=self.pipe("import sys,time;sys.stdin.buffer.read(4);time.sleep(30)",timeout=10)
        event=threading.Event();pipe.cancel_event=event
        timer=threading.Timer(.1,event.set);timer.start();self.addCleanup(timer.join)
        began=time.monotonic()
        with self.assertRaisesRegex(ValueError,'canceled'):pipe.call('blocked')
        self.assertLess(time.monotonic()-began,2);self.assertTrue(pipe.closed)
        self.assertFalse(pipe.thread.is_alive())

    def test_record_batch_rejects_stale_cursor_lengths_and_extra_bytes(self):
        c=object.__new__(Collector);c.token='token'
        good=dict(wire='ordinary-records/1',token='token',start=0,end=1,rows=1,lengths=[[2,2]],_arrays=memoryview(b'{}{}'))
        c.call=Mock(return_value=good);self.assertEqual([tuple(bytes(x) for x in pair) for pair in c.records({'rows':1})],[(b'{}',b'{}')])
        for change in ({'token':'stale'},{'start':1},{'end':0},{'lengths':[[3,2]]},{'_arrays':memoryview(b'{}{}x')}):
            c.call=Mock(return_value=dict(good,**change))
            with self.assertRaises(ValueError):list(c.records({'rows':1}))


@unittest.skipUnless(os.environ.get('BRASS_B07_COLLECTOR'),'Set BRASS_B07_COLLECTOR for short real native wire checks')
class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=os.environ['BRASS_B07_COLLECTOR']
        cls.roots=json.loads(Path(os.environ['BRASS_B07_FIXTURES']).read_text())['roots']
    def test_forced_rng_branch_features_records_and_recovery_match_unbatched(self):
        from rust_lab_b07_transport import walk
        for root in (self.roots[0],self.roots[-1]):
            origin=dict(state=root['expected']['state'],rail_keys=list(range(64)))
            old=walk(self.binary,origin,legacy=True,limit=24 if root==self.roots[0] else 96)
            new=walk(self.binary,origin,limit=24 if root==self.roots[0] else 96)
            for key in ('trace','features','branches','rng','complete','forced'):self.assertEqual(old[key],new[key],key)
            self.assertGreater(new['forced'],0)
            if root==self.roots[-1]:self.assertIsNotNone(new['complete'])

    def test_stale_reply_retires_lane_and_fresh_origin_recovers(self):
        origin=dict(state=self.roots[0]['expected']['state'],rail_keys=list(range(64)))
        with Collector(self.binary) as lane:
            lane.begin('recover',origin,100)
            before=copy.deepcopy(lane.current)
            with self.assertRaisesRegex(ValueError,'stale'):
                from brass_ai.rust_backend.lab_transport import feature_context
                lane.call('feature_arrays',token='stale',revision=0,context=feature_context('test',[0.]*4,[0.]*8))
            self.assertTrue(lane.closed)
        with Collector(self.binary) as lane:
            lane.begin('recover',origin,100)
            self.assertEqual(lane.current['decision'],before['decision'])

    def test_cancellation_discards_unacknowledged_forced_records(self):
        from rust_lab_b07_transport import walk
        origin=dict(state=self.roots[0]['expected']['state'],rail_keys=list(range(64)))
        # The first reference row includes the unchanged real sampler metadata.
        row,evidence=walk(self.binary,origin,legacy=True,limit=1)['trace'][0]
        with Collector(self.binary) as lane:
            lane.begin('cancel',origin,100)
            DecisionEncoder(lane,evidence['model_id']).encode(row['view'],row['choices'],row['style'],row['intent'])
            lane.append(row,evidence);self.assertTrue(lane.forced)
            event=threading.Event();event.set();lane.cancel_event=event
            with self.assertRaisesRegex(ValueError,'canceled'):lane.call('finish')
            self.assertTrue(lane.closed)
        with Collector(self.binary) as lane:
            lane.begin('cancel',origin,100)
            self.assertEqual(lane.current['step'],0)
            self.assertEqual(lane.current['decision']['view'],row['view'])
            cp=lane.checkpoint(lane.current['decision']['view']['context']['revision'])
            lane.release();lane.begin('cancel',{},100,checkpoint=cp,indices=list(range(len(cp['payload']['state']['cards']['deck'])-1,0,-1)))
            self.assertEqual(lane.current['decision']['view'],row['view'])
