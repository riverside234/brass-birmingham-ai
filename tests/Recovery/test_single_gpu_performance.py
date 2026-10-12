"""Single-device defaults, strict replay reuse and pass-local retention caches."""
import contextlib
import copy
import io
import importlib.util
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch,MagicMock

import torch

from fixtures import game
from brass_ai.learning.contracts import SOURCE_ROOT,decode_json
from brass_ai.recovery.config import EXECUTION
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.recovery.replay import CheckedGame,check_game
from brass_ai.recovery.execution import _ActorSerializedGame as SerializedGame
from brass_ai.recovery import inference,retention
from brass_ai.step5 import cli
from brass_ai.step5.league import write_json


class SingleGPUPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_strict_json_fast_path_preserves_values_and_rejects_nested_duplicates(self):
        self.assertEqual(decode_json('{"a":[1,2.5,true,null,{"name":"beer"}]}'),
                         {'a':[1,2.5,True,None,{'name':'beer'}]})
        for text,key in [('{"a":1,"a":2}','a'),('{"v":[{"x":1,"x":2}]}','x'),
                         ('{"x":1,"y":2,"y":3,"x":4}','y'),('{"a":1,"\\u0061":2}','a')]:
            with self.assertRaisesRegex(ValueError,'Duplicate JSON key: '+key):decode_json(text)
        for value in ('NaN','Infinity','-Infinity'):
            with self.assertRaisesRegex(ValueError,'Non-finite'):decode_json('{"x":'+value+'}')

    def test_writer_reuses_checked_inputs_after_verifying_seals(self):
        original=game();checked=check_game(original)
        checked.actor_validated=True  # same proof attached by play_game
        with patch('brass_ai.recovery.replay.BOUNDARY.check',side_effect=AssertionError('redundant schema walk')):
            serialized=SerializedGame(checked)
        self.assertEqual(decode_json(serialized.line),original)
        self.assertFalse(hasattr(serialized.game,'actor_validated'))

    def test_changed_input_label_context_and_metadata_cannot_use_proof(self):
        for change in (lambda g:g['records'][0]['view'].update(opponent_hands=[]),
                       lambda g:g['records'][0].update(chosen=100),
                       lambda g:g['records'][0]['style'].__setitem__(0,.5),
                       lambda g:g.update(game_id='changed')):
            checked=check_game(game());checked.actor_validated=True;change(checked)
            with self.assertRaisesRegex(ValueError,'changed'):SerializedGame(checked)
        checked=check_game(game());checked.actor_validated=True;checked['records'][0]['extra']='not in wire format'
        with self.assertRaisesRegex(ValueError,'fields'):SerializedGame(checked)

    def test_bound_but_unvalidated_game_still_runs_full_schema_checks(self):
        source=game();encoder=PreparedEncoder()
        arrays=[encoder.encode(r['view'],r['choices'],r['style'],r['intent']) for r in source['records']]
        source['records'][0]['view']['opponent_hands']=[]
        with self.assertRaises(Exception):SerializedGame(CheckedGame(source).bind(arrays))
        with self.assertRaises(Exception):SerializedGame(source)

    def test_single_selected_or_auto_gpu_has_no_router_and_only_one_initialization(self):
        for selected in (['cuda:1'],['auto']):
            with patch.object(torch.cuda,'is_available',return_value=True),\
                 patch.object(torch.cuda,'device_count',return_value=2 if selected==['cuda:1'] else 1),\
                 patch.object(inference,'MultiInferenceService',side_effect=AssertionError('router')):
                service=inference.InferenceService(EXECUTION,'cuda',inference_devices=selected)
            try:
                self.assertIsInstance(service,inference.InferenceService)
                self.assertEqual(service.device,'cuda:1' if selected==['cuda:1'] else 'cuda:0')
                self.assertFalse(hasattr(service,'router'))
                with patch.object(inference,'memory',return_value={}):
                    self.assertEqual(service.report()['replica_count'],1)
            finally:service.__exit__(None,None,None)

    def test_default_device_uses_one_cuda_gpu_or_falls_back_to_cpu(self):
        self.assertEqual(cli.parser().parse_args(['preflight']).device,'auto')
        for available in (True,False):
            with patch.object(torch.cuda,'is_available',return_value=available),\
                 patch.object(cli,'preflight',return_value={}) as preflight,contextlib.redirect_stdout(io.StringIO()):
                cli.main(['--root',str(SOURCE_ROOT),'preflight'])
            self.assertEqual(preflight.call_args.kwargs['device'],'cuda' if available else 'cpu')

    def test_launcher_limits_visibility_before_torch_and_preserves_explicit_overrides(self):
        spec=importlib.util.spec_from_file_location('step5_launcher',SOURCE_ROOT/'scripts/step5.py')
        launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
        for visible,expected in [(None,'0'),('3,5','3'),('GPU-allocated, GPU-other','GPU-allocated'),('',''),('-1','-1')]:
            with patch.dict(os.environ):
                if visible is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
                else:os.environ['CUDA_VISIBLE_DEVICES']=visible
                launcher._limit_default_gpu(['--device','cuda','train'])
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],expected)
        for args in (['--inference-devices','auto'],['--inference-devices=cuda:0,cuda:1']):
            with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='3,5'):
                launcher._limit_default_gpu(args)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'3,5')

    def test_retention_resolves_repeated_dependencies_once_but_refreshes_next_pass(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);run=root/'artifacts/run';run.mkdir(parents=True)
            control=root/'artifacts/ref/pins.json';target=run/'keep.pt'
            target.write_bytes(b'weights')
            write_json(control,{'repeated':[str(target)]*50})
            original=Path.resolve;counts={}
            def resolve(path,*args,**kwargs):
                counts[path]=counts.get(path,0)+1
                return original(path,*args,**kwargs)
            with patch.object(Path,'resolve',resolve):pins=retention.references(root,run)
            self.assertIn(target,pins);self.assertEqual(counts[target],1)
            write_json(control,{'repeated':[]})
            self.assertNotIn(target,retention.references(root,run))

    def test_reader_startup_overlaps_collection_without_a_readiness_barrier(self):
        from brass_ai.recovery import ingress
        pool=MagicMock();pool.__enter__.return_value=pool
        future=MagicMock();future.result.return_value=True;pool.submit.return_value=future
        admission=object()
        with patch.object(ingress,'ProcessPoolExecutor',return_value=pool) as factory:
            with ingress.reader_pool(3,admission) as readers:
                self.assertIs(readers,pool)
                self.assertEqual(pool.submit.call_count,3)
                future.result.assert_not_called()
            self.assertEqual(factory.call_args.kwargs['max_workers'],3)
            self.assertEqual(factory.call_args.kwargs['initargs'],(admission,))
        self.assertEqual(future.result.call_count,3)
        pool.__exit__.assert_called_once()

    def test_previous_source_resumes_identical_weights_optimizer_and_counters(self):
        from brass_ai.learning.checkpoint import resume_optimizer
        from brass_ai.recovery.resume import load_resume,commit_transition,check_signature
        from test_time_resume import signatures
        from test_teacher_work_resume import TeacherWorkResumeTests
        fixture=TeacherWorkResumeTests();self.addCleanup(fixture.doCleanups)
        def source_only():
            _,new=signatures();old=copy.deepcopy(new)
            old['implementation']='f19424e3d4da14f8759fbc5877bfd459771ca54194d6945d2dfcb90ccc23aa5d'
            return old,new
        fixture.signatures=source_only
        run,state,new,original,optimizer=fixture.fixture()
        checkpoint=run/'generation-0070.pt';before=checkpoint.read_bytes()
        model,raw,changed=load_resume(SOURCE_ROOT,run,state,new,reviewed=True)
        self.assertTrue(changed);self.assertFalse(model.utility_calibration['ready'])
        for key,value in original.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
        restored=torch.optim.AdamW(model.parameters(),lr=new['config']['learning_rate'])
        resume_optimizer(raw,restored)
        self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
        self.assertEqual(restored.state_dict()['param_groups'],optimizer.state_dict()['param_groups'])
        for key,values in optimizer.state_dict()['state'].items():
            for name,value in values.items():
                torch.testing.assert_close(restored.state_dict()['state'][key][name],value,rtol=0,atol=0)
        updated=commit_transition(run,state,new)
        for key,value in state.items():
            if key!='signature':self.assertEqual(updated[key],value,key)
        self.assertEqual(updated['source_transitions'][-1]['config_changes'],{})
        self.assertEqual(checkpoint.read_bytes(),before)
        self.assertFalse(load_resume(SOURCE_ROOT,run,updated,new)[2])
        unsupported=copy.deepcopy(new);unsupported['config']['learning_rate']*=2
        with self.assertRaisesRegex(ValueError,'unchanged config'):check_signature(state,unsupported,True)


if __name__=='__main__':unittest.main()
