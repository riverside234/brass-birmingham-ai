"""L13 real-process contract tests. No Python inference service is started."""
import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(os.environ.get('BRASS_LOCAL_PACKAGE',ROOT/'artifacts/rust-backend/l13-dev/BrassNative/alpha-g890'))
PROTOCOL='brass-local-native/1.0.0'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class Worker:
    def __init__(self,package=PACKAGE):
        env=dict(os.environ,PYTHONPATH='',PATH=str(package))
        self.p=subprocess.Popen([str(package/'brass-ai-local.exe'),str(package/'policy.json'),sha(package/'policy.json')],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,cwd=package)
        self.output=queue.Queue();self.id=0
        def reader():
            try:
                while True:
                    h=self.p.stdout.read(4)
                    if len(h)!=4:break
                    n=struct.unpack('<I',h)[0];assert 0<n<=1048576
                    self.output.put(json.loads(self.p.stdout.read(n)))
            finally:self.output.put(None)
        self.reader=threading.Thread(target=reader,daemon=True);self.reader.start()
        self.hello=self.recv();assert self.hello['status']=='ready',self.hello
    def send(self,operation,id=None):
        operation=copy.deepcopy(operation)
        if 'input' in operation:
            value=json.dumps(operation.pop('input'),separators=(',',':'))
            operation['input_json']=value;operation['ticket']['root_sha256']=hashlib.sha256(value.encode()).hexdigest()
        self.id=self.id+1 if id is None else id
        b=json.dumps(dict(protocol=PROTOCOL,id=self.id,request=operation)).encode()
        self.p.stdin.write(struct.pack('<I',len(b))+b);self.p.stdin.flush();return self.id
    def recv(self,timeout=65):
        value=self.output.get(timeout=timeout)
        if value is None:raise AssertionError('worker exited: '+self.p.stderr.read().decode(errors='replace'))
        assert value['protocol']==PROTOCOL
        return value
    def search(self,index=0,allowance=4000,remaining=None,instance=None,generation=2):
        case=read(ROOT/f'artifacts/rust-backend/20261025-l10-capture/brass-{index:03d}-request.json')['cases'][0]
        inp=case['input'];inp.pop('shuffle_tape',None);inp['recall']=[]
        inp['context']['profile_id']='alpha-test-g890-neutral/1.0.0'
        return dict(op='search',ticket=dict(game_instance=instance or f'local-{index}-{self.id}',game_id='test',generation=generation,
            revision=inp['view']['context']['revision'],actor=inp['view']['seat'],profile=inp['context']['profile_id'],root_sha256='a'*64,
            source_sha256=self.hello['data']['build']['source_sha256']),allowance_ms=allowance,remaining_ms=remaining or allowance,input=inp)
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.close()
            try:self.p.wait(20)
            except subprocess.TimeoutExpired:self.p.kill();self.p.wait();raise
        self.reader.join(2)
        self.p.stdin.close()
        self.p.stdout.close();self.p.stderr.close()

class LocalTests(unittest.TestCase):
    def setUp(self):self.w=Worker()
    def tearDown(self):self.w.close()
    def test_readiness_is_distinct_from_acceptance_and_health(self):
        d=self.w.hello['data'];self.assertEqual((d['sessions'],d['threads_per_session']),(2,1))
        self.assertIsNone(d['utility_acceptance']['utility_calibration'])
        self.assertFalse(d['utility_acceptance']['final_model'])
        self.assertTrue(d['measured_readiness']['warmup_passed']);self.assertGreater(d['measured_readiness']['native_nn_positions'],0)
        self.w.send(dict(op='health'));self.assertFalse(self.w.recv()['data']['busy'])
    def test_four_allowances_and_forced_debt_staged(self):
        for index,ms in [(0,4000),(3,8000),(26,10000),(8,12000),(41,4000),(42,4000),(30,4000)]:
            start=time.monotonic();req=self.w.search(index,ms);rid=self.w.send(req);reply=self.w.recv();elapsed=time.monotonic()-start
            self.assertEqual((reply['id'],reply['status']),(rid,'ok'),reply)
            data=reply['data'];self.assertEqual(data['ticket']['game_instance'],req['ticket']['game_instance']);self.assertTrue(data['actions']);self.assertTrue(data['retired'])
            self.assertLess(data['response_ready_ns'],data['deadline_ns']);self.assertLess(elapsed,ms/1000+1)
            self.assertEqual(data['replay']['version'],'brass-local-replay/1.0.0')
            self.assertEqual([owner['owner'] for owner in data['replay']['owners']],[0,1])
            self.assertEqual(sum(owner['completed'] for owner in data['replay']['owners']),data['completed'])
    def test_delivery_reserve_completes_without_search_at_all_difficulties(self):
        for allowance in (4000,8000,10000,12000):
            self.w.send(self.w.search(0,allowance,remaining=300))
            response=self.w.recv();self.assertEqual(response['status'],'ok',response)
            data=response['data'];self.assertTrue(data['actions']);self.assertTrue(data['retired'])
            self.assertEqual(data['requested_k'],8 if allowance==4000 else 16)
            self.assertEqual(data['completed'],0)
            self.assertLess(data['response_ready_ns'],data['deadline_ns'])
    def test_cancel_drains_before_reuse_and_stale_cancel_does_not_kill_current(self):
        rid=self.w.send(self.w.search());self.w.send(dict(op='cancel',target=rid+100))
        self.assertEqual(self.w.recv()['status'],'error')
        self.w.send(dict(op='health'));self.assertTrue(self.w.recv()['data']['busy'])
        self.w.send(dict(op='cancel',target=rid));replies=[self.w.recv(),self.w.recv()]
        self.assertEqual({r['status'] for r in replies},{'cancelling','error'})
        self.assertTrue(next(r for r in replies if r['id']==rid)['data']['retired'])
        self.w.send(self.w.search(41,generation=3));self.assertEqual(self.w.recv()['status'],'ok')
    def test_full_identity_and_research_inputs_rejected(self):
        for field,value in [('profile','unaccepted'),('source_sha256','f'*64),('revision',999999),('actor',99),('generation',0)]:
            r=self.w.search();r['ticket'][field]=value;self.w.send(r);self.assertEqual(self.w.recv()['status'],'error')
        for field,value in [('model_id','f'*64),('profile_id','different'),('style',[1.,0.,0.,0.])]:
            r=self.w.search();r['input']['context'][field]=value;self.w.send(r);self.assertEqual(self.w.recv()['status'],'error')
        r=self.w.search();r['input']['shuffle_tape']=[];self.w.send(r);self.assertEqual(self.w.recv()['status'],'error')
        r=self.w.search();r['input']['private_deck']=[];self.w.send(r);self.assertEqual(self.w.recv()['status'],'error')
        r=self.w.search();value=json.dumps(r.pop('input'),separators=(',',':'));r['input_json']=value
        self.w.send(r);self.assertEqual(self.w.recv()['status'],'error')  # mismatched root digest
    def test_no_replenishment_or_duplicate_root(self):
        r=self.w.search(0,remaining=500,instance='same');self.w.send(r);self.w.recv()
        r['remaining_ms']=4000;self.w.send(r);self.assertEqual(self.w.recv()['status'],'error')
    def test_k_tracks_seat_allowance_with_short_remainders_and_mixed_seats(self):
        # One persistent process switches difficulties, including returning to 4s.
        # This is a bounded smoke, not four full-duration searches.
        for ms in (8000,4000,12000,10000,4000):
            self.w.send(self.w.search(0,ms,remaining=1600))
            response=self.w.recv();self.assertEqual(response['status'],'ok',response)
            d=response['data'];expected=8 if ms==4000 else 16
            self.assertEqual(d['requested_k'],expected)
            self.assertTrue(d['actions']);self.assertTrue(d['retired'])
            self.assertEqual((d['sessions'],d['threads_per_session']),(2,1))
            for owner in d['replay']['owners']:
                self.assertGreater(owner['effective_k'],0)
                self.assertLessEqual(owner['effective_k'],expected)
                # Later roots use measured trajectory costs. A short remainder
                # may legitimately reduce K to fit a complete halving schedule.
                if owner['quota']['simulations']>=expected*(expected.bit_length()-1):
                    self.assertEqual(owner['effective_k'],expected)
    def test_allowance_cannot_change_within_the_same_turn(self):
        r=self.w.search(0,8000,remaining=500,instance='fixed-allowance')
        self.w.send(r);self.assertEqual(self.w.recv()['status'],'ok')
        r['allowance_ms']=4000;r['remaining_ms']=300
        self.w.send(r);reply=self.w.recv();self.assertEqual(reply['status'],'error')
        self.assertIn('turn allowance changed',reply['data']['error'])
    def test_shutdown_active_retires_before_stopped(self):
        search=self.w.send(self.w.search());stop=self.w.send(dict(op='shutdown'))
        replies=[self.w.recv(),self.w.recv()]
        self.assertEqual(replies[0]['id'],search);self.assertEqual(replies[-1]['id'],stop)
        self.assertEqual(replies[-1]['status'],'stopped');self.assertTrue(replies[-1]['data']['retired'])
        self.w.p.wait(10)
    def test_eof_and_invalid_frame_retire_active(self):
        self.w.send(self.w.search());self.w.p.stdin.write(struct.pack('<I',1048577));self.w.p.stdin.flush()
        self.w.p.wait(20);self.assertEqual(self.w.p.returncode,0)

class AdmissionTests(unittest.TestCase):
    def test_unaccepted_model_and_fabricated_calibration_fail_startup(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts/rust-backend') as directory:
            target=Path(directory);shutil.copytree(PACKAGE,target,dirs_exist_ok=True)
            original=read(target/'policy.json')
            for field,value in [('model_id','0'*64),('export_sha256','0'*64),('checkpoint_sha256','0'*64),('scope','lab'),('final_model',True),('utility_calibration',{'passed':True})]:
                policy=copy.deepcopy(original);policy['acceptance'][field]=value
                (target/'policy.json').write_text(json.dumps(policy))
                p=subprocess.run([str(target/'brass-ai-local.exe'),str(target/'policy.json'),sha(target/'policy.json')],capture_output=True,timeout=20)
                self.assertNotEqual(p.returncode,0);self.assertFalse(p.stdout);self.assertIn(b'acceptance',p.stderr)
            for missing in ('utility_calibration','model_id'):
                policy=copy.deepcopy(original);del policy['acceptance'][missing]
                (target/'policy.json').write_text(json.dumps(policy))
                p=subprocess.run([str(target/'brass-ai-local.exe'),str(target/'policy.json'),sha(target/'policy.json')],capture_output=True,timeout=20)
                self.assertNotEqual(p.returncode,0);self.assertFalse(p.stdout);self.assertIn(b'missing field',p.stderr)

if __name__=='__main__':unittest.main()
