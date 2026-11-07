"""Bounded B07 phase reuse and exactly-one production continuation checks."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

from brass_ai.learning.contracts import SOURCE_ROOT as ROOT, digest
from brass_ai.rust_backend.lab_contract import pin
from brass_ai.step5.league import write_json
from brass_ai.rust_backend.lab_replay_compat import qualified_implementation as implementation_id
import rust_lab_b07_completion as C

spec = importlib.util.spec_from_file_location('b07_continue_tests', ROOT/'scripts/continue-rust-lab-b07.py')
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)


class Completion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'artifacts')
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.snapshot = dict(generation=1250,checkpoint={'sha256':'c'*64},config_sha256='d'*64,
                             state_file={'sha256':'e'*64},signature_sha256='f'*64)
        self.identity = dict(implementation=implementation_id(),generation=1250,
                             checkpoint_sha256='c'*64,config_sha256='d'*64)

    def evidence(self):
        (self.out/'evidence').mkdir()
        build = self.out/'build.json';write_json(build, {'exact':'build'})
        checked = dict(passed=True,matched={'equal':True},performance={'useful_accepted_work_gain':True})
        q = dict(self.identity,milestone='B07',passed=True,ordinary=checked,ordinary_build=pin(build))
        entries = {}
        for name,value in [('qualification',q),('ordinary',checked),('snapshot',self.snapshot)]:
            data = json.dumps(value).encode()
            filename = name+'.json'
            if name == 'snapshot':data=gzip.compress(data,mtime=0);filename+='.gz'
            (self.out/'evidence'/filename).write_bytes(data)
            entries[name] = dict(file=filename,sha256=hashlib.sha256(data).hexdigest())
        return {'completed_ordinary':{'evidence':entries}},q

    def test_reuse_exact_pass_without_collecting_or_rebuilding(self):
        ref,q=self.evidence()
        with patch.object(C,'DIRECTORY',self.out),patch.object(C,'verify_unchanged') as verify:
            actual,checked=C.ordinary(ref,self.snapshot)
        self.assertEqual(actual,q);self.assertEqual(checked,q['ordinary'])
        verify.assert_called_once_with(self.snapshot)

    def test_reuse_rejects_changed_evidence_state_boundary_or_build(self):
        ref,q=self.evidence()
        with patch.object(C,'DIRECTORY',self.out),patch.object(C,'verify_unchanged'):
            for key,value in [('generation',1251),('config_sha256','wrong'),('state_file',{'sha256':'wrong'})]:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    C.ordinary(ref,dict(self.snapshot,**{key:value}))
            path=Path(q['ordinary_build']['path']);path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'build changed'):C.ordinary(ref,self.snapshot)
            (self.out/'evidence/ordinary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'evidence changed'):C.ordinary(ref,self.snapshot)

    def resume_receipt(self):
        detail=dict(passed=True,pass_index=1,tables=2,blend_weight=0,checkpoint_reloaded=True,buffer_reloaded=True)
        path=self.out/'canary-1-checks.json';write_json(path,detail)
        q=dict(self.identity,milestone='B07',passed=False,failed_phase='canary-2',
               completion_workflow=C.workflow_id(),completion_state=self.snapshot['state_file'],
               completed_phases={'canary-1':pin(path)})
        receipt=self.out/'qualification.json';write_json(receipt,q)
        return receipt,q,path

    def test_failed_later_phase_reuses_only_earlier_pass(self):
        receipt,q,path=self.resume_receipt()
        _,result=C.resume(receipt,self.snapshot)
        self.assertEqual(list(result),['canary-1'])
        q['completion_workflow']='old';write_json(receipt,q)
        with self.assertRaisesRegex(ValueError,'exact B07'):C.resume(receipt,self.snapshot)

    def test_resume_rejects_tampered_or_failed_phase(self):
        receipt,q,path=self.resume_receipt();value=json.loads(path.read_text());value['passed']=False;write_json(path,value)
        with self.assertRaisesRegex(ValueError,'evidence changed'):C.resume(receipt,self.snapshot)
        q['completed_phases']['canary-1']=pin(path);write_json(receipt,q)
        with self.assertRaisesRegex(ValueError,'failed phase'):C.resume(receipt,self.snapshot)

    def test_completed_source_and_checkpoint_are_exact(self):
        for key,value in [('implementation','wrong'),('checkpoint_sha256','wrong')]:
            with self.subTest(key=key),self.assertRaises(ValueError):C.boundary(dict(self.identity,**{key:value}),self.snapshot)

    def test_shipped_source_review_and_resume_command_keep_exact_saved_recipe(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.cli import parser
        review=C.reviewed_source()
        snapshot=json.loads(gzip.decompress((C.DIRECTORY/'evidence/b07-r4-snapshot.json.gz').read_bytes()))
        state=snapshot['state'];target=dict(state['signature'],implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True))
        config='configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json'
        self.assertEqual(json.loads((ROOT/config).read_text()),state['signature']['config'])
        args=parser().parse_args(['--root',str(ROOT),'--config',config,'--device','cuda','--mono','mono',
            'train','--run','artifacts/step5/joint7-v11-2-g1290-trained-restart','--expected-generation','1251','--iterations','1'])
        self.assertEqual(args.expected_generation,1251);self.assertEqual(args.iterations,1)
        self.assertFalse(args.fresh);self.assertIsNone(args.initial)
        changed=copy.deepcopy(target);changed['config']['selfplay_recipe']='changed'
        with self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)

    def test_old_failed_teacher_is_rebuilt_and_successful_phases_cannot_change_teacher(self):
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        path=self.out/'teacher.json';write_json(path,dict(implementation='old',producer='old'))
        self.assertIsNone(C.resumable_teacher(pin(path),{}))
        with self.assertRaisesRegex(ValueError,'different teacher'):C.resumable_teacher(pin(path),{'canary-1':{}})
        write_json(path,dict(implementation=implementation_id(),producer=source_fingerprint()))
        self.assertEqual(C.resumable_teacher(pin(path),{'canary-1':{}}),str(path.resolve()))
        receipt=pin(path);path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'build changed'):C.resumable_teacher(receipt,{})

    def test_ordinary_rebind_keeps_exact_old_receipt_and_rejects_teacher_or_tampering(self):
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        value=dict(implementation='old',producer='old-native',collector={'exact':'bytes'})
        path=self.out/'original.json';write_json(path,value)
        review=dict(from_implementation='old')
        reference=dict(build_reuse={'ordinary':dict(original_receipt=pin(path),from_producer='old-native',
                                                   ordinary_compile_inputs_unchanged=True)})
        write_json(self.out/'reference.json',reference)
        with patch.object(C,'DIRECTORY',self.out),patch.object(C,'reviewed_source',return_value=review):
            updated=C.reuse_build(value,'ordinary')
            self.assertEqual(updated['collector'],value['collector'])
            self.assertEqual(updated['producer'],source_fingerprint())
            self.assertEqual(C.reuse_build(updated,'ordinary'),updated)
            with self.assertRaisesRegex(ValueError,'new teacher build'):C.reuse_build(value,'teacher')
            with self.assertRaisesRegex(ValueError,'differs from original'):
                C.reuse_build(dict(updated,collector={'tampered':True}),'ordinary')
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'receipt changed'):C.reuse_build(value,'ordinary')

    def test_checkpoint_only_teacher_reuse_preserves_binary_and_checks_original_receipt(self):
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        name='src/brass_ai/recovery/search_value_phase.py'
        value=dict(implementation='r6',producer=source_fingerprint(),binary={'sha256':'unchanged'})
        path=self.out/'r6-teacher.json';write_json(path,value)
        review=dict(from_implementation='r6',to_implementation=implementation_id(),changes={n:dict(
            after=hashlib.sha256((ROOT/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest())
            for n in (name,'src/brass_ai/recovery/search_value_iteration.py')})
        write_json(self.out/'reference.json',dict(checkpoint_source_review=review,
            teacher_checkpoint_reuse={'original_receipt':pin(path)}))
        with patch.object(C,'DIRECTORY',self.out),patch.object(C,'reviewed_source',return_value={}):
            rebound=C.reuse_build(value,'teacher')
            self.assertEqual(rebound['binary'],value['binary'])
            self.assertEqual(C.reuse_build(rebound,'teacher'),rebound)
            with self.assertRaisesRegex(ValueError,'differs from original'):
                C.reuse_build(dict(rebound,binary={'sha256':'changed'}),'teacher')
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'receipt changed'):C.reuse_build(value,'teacher')


class Workflow(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.run=self.base/'run';self.run.mkdir()
        self.out=self.base/'output';self.out.mkdir()
        from brass_ai.recovery.config import LATEST_MODEL_RECIPE
        self.source=dict(bc_run=(self.base/'completed-il').relative_to(ROOT).as_posix(),
            path=(self.base/'completed-il/candidate.pt').relative_to(ROOT).as_posix(),sha256='il-checkpoint',
            manifest=(self.base/'demonstrations.json').relative_to(ROOT).as_posix(),manifest_sha256='dataset',
            model_id='il-model',bc_signature='il-signature')
        self.config={'teacher_phase':{'mode':'off'},'saved_recipe':'unchanged','execution':{'workers':60},
            'selfplay_recipe':LATEST_MODEL_RECIPE,'imitation_checkpoint':copy.deepcopy(self.source),
            'imitation':{'manifest':None}}
        self.state=dict(iteration=1250,history=[{'sha256':'checkpoint'}],
            signature={'config':self.config,'imitation_source':copy.deepcopy(self.source)})
        write_json(self.run/'state.json',self.state)
        self.args=SimpleNamespace(run=str(self.run),expected_generation=1250,qualification=None,
            resume_qualification=None,teacher_build=None,offline=False,mono='mono')
        self.calls=[];self.report={}

    def invoke(self,name,argv):
        self.calls.append((name,argv))
        if name=='remaining':
            write_json(self.out/'remaining/qualification.json',{'adoption_ready':True})
        elif name=='training':
            actual=json.loads(Path(argv[argv.index('--config')+1]).read_text())
            self.assertEqual(actual,self.config)
            self.assertEqual(argv[argv.index('--iterations')+1],'1')
            self.assertEqual(argv[argv.index('--expected-generation')+1],'1250')
            # Parse through the real trainer CLI; the latest-model recipe needs
            # this even on resume, where learner weights/Adam come from self-play.
            from brass_ai.step5.cli import parser
            parsed=parser().parse_args(argv[2:])
            self.assertEqual(Path(parsed.imitation_run),ROOT/self.source['bc_run'])
            self.assertEqual(parsed.run,str(self.run));self.assertEqual(parsed.iterations,1)
            self.assertNotIn('--fresh',argv);self.assertNotIn('--initial',argv)
            self.assertNotIn('--from-run',argv);self.assertNotIn('--resume-reviewed-source',argv)
            self.state['iteration']=1251;write_json(self.run/'state.json',self.state)
        else:
            if name=='adopt':
                self.state['native_backend']={'mode':'rust','adopted_generation':1250}
                write_json(self.run/'state.json',self.state)
            write_json(self.out/name/'qualification.json',{'passed':True})

    def execute(self,invoke=None):
        with patch('brass_ai.rust_backend.lab_backend.validate_state'):
            W.execute(self.args,self.out,self.report,invoke or self.invoke)

    def test_full_sequence_keeps_committed_config_and_trains_exactly_once(self):
        self.execute()
        self.assertEqual([n for n,_ in self.calls],['remaining','adopt','verify-rollback','training','verify-next'])
        self.assertEqual(self.report['training_iterations_launched'],1)
        self.assertTrue(self.report['passed'])

    def test_repeat_after_commit_only_verifies_never_starts_g_plus_two(self):
        self.state.update(iteration=1251,native_backend={'mode':'rust','adopted_generation':1250})
        write_json(self.run/'state.json',self.state);self.execute()
        self.assertEqual([n for n,_ in self.calls],['verify-rollback','verify-next'])
        self.assertEqual(self.report['training_iterations_launched'],0)

    def test_interruption_after_adoption_reuses_selected_backend(self):
        self.state['native_backend']={'mode':'rust','adopted_generation':1250}
        write_json(self.run/'state.json',self.state);self.execute()
        self.assertEqual([n for n,_ in self.calls],['verify-rollback','training','verify-next'])
        self.assertEqual(self.report['training_command'],self.calls[1][1])

    def test_missing_or_changed_il_binding_fails_before_any_child(self):
        original=copy.deepcopy(self.state)
        for adopted in (False,True):
            for invalid in (None,{},dict(self.source,sha256='wrong'),dict(self.source,bc_run='../outside')):
                with self.subTest(adopted=adopted,source=invalid):
                    self.state=copy.deepcopy(original)
                    self.state['signature']['imitation_source']=invalid
                    if adopted:self.state['native_backend']={'mode':'rust','adopted_generation':1250}
                    write_json(self.run/'state.json',self.state)
                    with self.assertRaises(ValueError):self.execute()
                    self.assertEqual(self.calls,[])
                    self.assertEqual(json.loads((self.run/'state.json').read_text()),self.state)

    def test_adoption_cannot_change_legacy_imitation_binding(self):
        self.config.pop('selfplay_recipe')
        write_json(self.run/'state.json',self.state)
        def changed(name,argv):
            self.invoke(name,argv)
            if name=='adopt':
                self.state['signature']['imitation_source']['bc_run']=str(self.base/'other-il')
                write_json(self.run/'state.json',self.state)
        with self.assertRaisesRegex(ValueError,'changed the committed imitation dependency'):self.execute(changed)
        self.assertNotIn('training',[n for n,_ in self.calls])

    def test_no_adoption_after_incomplete_qualification(self):
        def fail(name,argv):
            self.calls.append((name,argv));write_json(self.out/'remaining/qualification.json',{'adoption_ready':False})
        with self.assertRaisesRegex(ValueError,'gates'):self.execute(fail)
        self.assertEqual([n for n,_ in self.calls],['remaining'])
        self.assertEqual(json.loads((self.run/'state.json').read_text()),self.state)

    def test_training_failure_does_not_claim_commit_or_launch_again(self):
        def fail(name,argv):
            if name=='training':raise RuntimeError('interrupted iteration')
            self.invoke(name,argv)
        with self.assertRaisesRegex(RuntimeError,'interrupted'):self.execute(fail)
        self.assertNotIn('verify-next',[n for n,_ in self.calls]);self.assertNotIn('passed',self.report)

    def test_stale_generation_reference_switch_or_new_recipe_are_rejected(self):
        for state in [dict(self.state,iteration=1252),dict(self.state,iteration=1251),
                      dict(self.state,native_backend={'mode':'reference','adopted_generation':1250})]:
            with self.subTest(state=state),self.assertRaises(ValueError):W.stage(state,1250)
        def changed(name,argv):
            self.invoke(name,argv)
            if name=='adopt':
                self.state['signature']['config']=dict(self.config,saved_recipe='wrong')
                write_json(self.run/'state.json',self.state)
        with self.assertRaisesRegex(ValueError,'recipe'):self.execute(changed)


class TrainingArguments(unittest.TestCase):
    setUp=Completion.setUp

    def test_actual_g1250_recipe_passes_pinned_imitation_run_to_real_cli(self):
        from brass_ai.step5.cli import parser
        config=json.loads((ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json').read_text())
        state={'signature':{'config':config,'imitation_source':copy.deepcopy(config['imitation_checkpoint'])}}
        before=copy.deepcopy(state)
        argv=W.training_command(ROOT,ROOT/'artifacts/step5/joint7-v11-2-g1290-trained-restart',
            self.out/'config.json',1250,'mono',state=state)
        parsed=parser().parse_args(argv[2:])
        self.assertEqual(Path(parsed.imitation_run),ROOT/config['imitation_checkpoint']['bc_run'])
        self.assertEqual(parsed.expected_generation,1250)
        self.assertEqual(parsed.iterations,1)
        self.assertEqual(state,before)

    def test_legacy_dependency_is_forwarded_but_no_dependency_is_invented(self):
        state={'signature':{'config':{}}}
        self.assertEqual(W.imitation_arguments(ROOT,self.out,state),[])
        state['signature']['imitation_source']={'bc_run':str(self.out/'il')}
        self.assertEqual(W.imitation_arguments(ROOT,self.out,state),['--imitation-run',str(self.out/'il')])

    def test_project_relocation_uses_current_root_and_preserves_saved_binding(self):
        from brass_ai.learning.paths import make_relocation,RELOCATION_FILE
        from brass_ai.recovery.config import LATEST_MODEL_RECIPE
        prior='/prior/lab/project'
        write_json(self.out/RELOCATION_FILE,make_relocation(self.out,[prior],'a'*64))
        saved={'bc_run':prior+'/artifacts/il','path':prior+'/artifacts/il/candidate.pt',
               'manifest':prior+'/artifacts/data.json','sha256':'unchanged'}
        cfg=dict(saved,bc_run='artifacts/il',path='artifacts/il/candidate.pt',manifest='artifacts/data.json')
        state={'signature':{'config':{'selfplay_recipe':LATEST_MODEL_RECIPE,'imitation_checkpoint':cfg},
                            'imitation_source':saved}}
        before=copy.deepcopy(state)
        self.assertEqual(W.imitation_arguments(self.out,self.out/'run',state),
                         ['--imitation-run',str(self.out/'artifacts/il')])
        self.assertEqual(state,before)

    def test_outside_project_self_dependency_and_conflicting_imitation_fail(self):
        for source,config in [({'bc_run':'../outside'},{}),({'bc_run':str(self.out)},{}),
                              ({'bc_run':str(self.out/'il')},{'imitation':{'manifest':'data.json'}})]:
            with self.subTest(source=source,config=config),self.assertRaises(ValueError):
                W.imitation_arguments(ROOT,self.out,{'signature':{'config':config,'imitation_source':source}})


class Launcher(unittest.TestCase):
    setUp=Completion.setUp
    def test_completion_runs_only_missing_phases_and_reuses_original_binary(self):
        import rust_lab_b07_retry as retry
        import rust_lab_b06_metrics as metrics
        Q=W.module('qualify-rust-lab-b07.py')
        run=self.out/'run';run.mkdir();output=self.out/'completion'
        bundle=self.out/'bundle';bundle.mkdir()
        write_json(bundle/'reference.json',dict(self.identity,prerequisites={},b06={},completed_ordinary={}))
        original=self.out/'qualified-build.json';write_json(original,{'binary':'original'})
        from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
        teacher=self.out/'teacher-build.json';write_json(teacher,{'implementation':implementation_id(),'producer':source_fingerprint()})
        prior=dict(ordinary_build=pin(original))
        checked=dict(passed=True,matched={'equal':True},performance={'useful_accepted_work_gain':True})
        done=dict(passed=True,pass_index=1,tables=2,blend_weight=0,checkpoint_reloaded=True,buffer_reloaded=True)
        calls=[]
        def command(argv,**kwargs):
            if '--phase' in argv:
                phase=argv[argv.index('--phase')+1];calls.append(phase)
                value=dict(done,pass_index=2) if phase=='canary-2' else {'passed':True}
                write_json(output/(phase+'-checks.json'),value)
            else:calls.append('fixtures')
        base=SimpleNamespace(python_environment=lambda:{'PYTHONPATH':''},command=command)
        with patch.object(Q,'DIRECTORY',bundle),patch.object(Q,'module',return_value=base),\
             patch.object(Q,'verify',return_value={'implementation':implementation_id()}),\
             patch.object(Q.platform,'system',return_value='Linux'),patch.object(metrics,'measurement_preflight',return_value={}),\
             patch.object(retry,'original_runtime',return_value=({},{})),patch.object(retry,'verify_ordinary',side_effect=lambda v,_:v) as verify,\
             patch.object(retry,'verify_teacher',side_effect=lambda v,_:v),patch.object(retry,'build_teacher') as build,\
             patch.object(retry,'prior_ordinary',return_value={'performance':{}}),\
             patch.object(C,'ordinary',return_value=(prior,checked)),\
             patch.object(C,'resume',return_value=({'teacher_build':pin(teacher)},{'canary-1':done})),\
             patch('brass_ai.rust_backend.lab_snapshot.capture',return_value=(self.snapshot,None,None)),\
             patch('brass_ai.rust_backend.lab_snapshot.verify_unchanged'),patch('rust_lab_results.export',return_value='test-export'),\
             patch.object(sys,'argv',['qualifier','--complete-remaining','--resume-from',str(teacher),'--run',str(run),
                                     '--out',str(output)]),patch.object(Q,'ROOT',self.out):
            # Output safety still applies in the fake checkout.
            output=self.out/'artifacts/rust-backend/completion'
            sys.argv[-1]=str(output)
            Q.main()
        self.assertEqual(calls,['fixtures','canary-2','continuation'])
        build.assert_not_called();verify.assert_called_once_with({'binary':'original'},{})
        report=json.loads((output/'qualification.json').read_text())
        self.assertTrue(report['adoption_ready']);self.assertTrue(report['passed'])
        self.assertEqual(set(report['completed_phases']),{'canary-1','canary-2','continuation'})


if __name__=='__main__':unittest.main()
