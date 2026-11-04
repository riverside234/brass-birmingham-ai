"""Real subprocess framing/deadline tests and oracle/progress regressions."""
import json
from pathlib import Path
import queue
import sys
import tempfile
import unittest
from unittest.mock import Mock

from brass_ai.rust_backend.lab_transport import Pipe,MAX_FRAME
from brass_ai.rust_backend.lab_b02_smoke import Oracle
from brass_ai.rust_backend.lab_smoke_progress import Progress
from brass_ai.learning.contracts import SOURCE_ROOT


class Transport(unittest.TestCase):
    def pipe(self, source, **kwargs):
        pipe=Pipe([sys.executable,'-u','-c',source],**kwargs)
        self.addCleanup(pipe.close)
        return pipe

    def test_large_repeated_text_replies_without_waiting_for_buffer_fill(self):
        source='''import json,sys
for line in sys.stdin:
 r=json.loads(line)
 data={'op':r['op'],'value':r['value']}
 sys.stdout.write(json.dumps({'ok':True,'data':data})+'\\n');sys.stdout.flush()
'''
        pipe=self.pipe(source,binary=False,timeout=3)
        for size in (1,150000,2,70000):
            value='x'*size
            self.assertEqual(pipe.call('echo',value=value),dict(op='echo',value=value))

    def test_fragmented_text_reply(self):
        pipe=self.pipe("import sys,time;sys.stdin.readline();sys.stdout.write('{\"ok\":true,');sys.stdout.flush();time.sleep(.05);sys.stdout.write('\"data\":17}\\n');sys.stdout.flush()",binary=False,timeout=3)
        self.assertEqual(pipe.call('split'),17)

    def test_binary_framing_remains_exact(self):
        source='''import json,struct,sys
r=sys.stdin.buffer
while True:
 h=r.read(4)
 if not h:break
 req=json.loads(r.read(struct.unpack('<I',h)[0]))
 data=json.dumps(dict(ok=True,id=req['id'],protocol=req['protocol'],data=req['value'])).encode()
 sys.stdout.buffer.write(struct.pack('<I',len(data))+data);sys.stdout.buffer.flush()
'''
        pipe=self.pipe(source,timeout=3)
        for value in ('small','x'*150000):self.assertEqual(pipe.call('echo',value=value),value)

    def test_partial_line_timeout_kills_process_and_reader(self):
        pipe=self.pipe("import sys,time;sys.stdin.readline();sys.stdout.write('{');sys.stdout.flush();time.sleep(30)",binary=False,timeout=.5)
        with self.assertRaises(queue.Empty):pipe.call('hang')
        self.assertIsNotNone(pipe.process.poll());self.assertFalse(pipe.thread.is_alive())
        self.assertTrue(pipe.closed)

    def test_truncated_and_oversized_text_still_fail(self):
        for output in ('{}', 'x'*(MAX_FRAME+1)):
            # Pass only the size in argv; avoid the OS argument length limit.
            source=f"import sys;sys.stdin.readline();sys.stdout.write('x'*{len(output)});sys.stdout.flush()"
            pipe=self.pipe(source,binary=False,timeout=3)
            with self.assertRaisesRegex(ValueError,'Bootstrap frame limit'):pipe.call('bad')
            self.assertIsNotNone(pipe.process.poll())


class OracleReplies(unittest.TestCase):
    def test_step_and_restore_supply_next_decision_without_extra_rpc(self):
        pipe=Mock();pipe.call.side_effect=[dict(state=1),dict(state=2),dict(state=3)]
        oracle=Oracle(pipe)
        self.assertEqual(oracle.decision(),dict(state=1));oracle.decision()
        oracle.step('move');self.assertEqual(oracle.decision(),dict(state=2))
        oracle.restore({'checkpoint':'fork'},123);self.assertEqual(oracle.decision(),dict(state=3))
        self.assertEqual([c.args[0] for c in pipe.call.call_args_list],['decision','step','restore'])

    def test_terminal_reply_is_not_requeried(self):
        pipe=Mock();pipe.call.return_value=dict(context={'terminal':True},result={'vp':[1,2,3,4]})
        oracle=Oracle(pipe);oracle.step('last')
        self.assertTrue(oracle.decision()['context']['terminal']);pipe.call.assert_called_once_with('step',choice='last')


class DiagnosticProgress(unittest.TestCase):
    def test_failure_is_written_and_monitor_stops(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'progress.json';progress=Progress(path,interval=60)
            with self.assertRaisesRegex(ValueError,'mismatch'),progress:
                with progress.stage('0/parent','oracle-compare',step=9):raise ValueError('mismatch')
            value=json.loads(path.read_text())
            self.assertEqual(value['lanes']['0/parent']['operation'],'oracle-compare')
            self.assertEqual(value['lanes']['0/parent']['stage'],'failed')
            self.assertFalse(progress.thread.is_alive());self.assertFalse(path.with_suffix('.tmp').exists())

    def test_success_reports_latest_stage(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'progress.json'
            with Progress(path,interval=60) as progress:
                progress.update('0/parent','native-collection',step=41)
                progress.update('collection','complete')
            self.assertEqual(json.loads(path.read_text())['lanes']['0/parent']['step'],41)
