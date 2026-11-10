"""The g1290 restart into a new run: recipe, Stage A, value-buffer pause and the built run."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.recovery import branch_restart as B,entropy_schedule as S,ppg_g210 as P,quality_pool
from brass_ai.recovery.teacher_gradient_qualification import restore
from brass_ai.recovery import search_value_schedule as V,training_revision_g1294 as T94,training_revision_g1307 as T07
from brass_ai.recovery.collector_runtime import recipe as without_runtime
from brass_ai.recovery.search_value_training import StageMachine
from brass_ai.rust_backend import lab_shadow_config as C,lab_teacher_activation as A
from brass_ai.step5.league import write_json

EXPORT=ROOT/'results/joint7-v11-2-g1290-trained-restart-g1290-20261109T041246972332Z'
LATER=ROOT/'results/joint7-v11-2-g1290-trained-restart-g1294-20261110T000540895274Z'


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state=read(EXPORT/'state.json')
        cls.shadow=B.recipe(cls.state['signature']['config'],share_target=.1)
        cls.on=A.configure(cls.shadow,.1,revise_stages=True)

    def test_the_first_four_iterations_measure_beta_in_shadow_mode(self):
        term=self.shadow['teacher_phase']['term']
        self.assertEqual(self.shadow['teacher_phase']['mode'],'shadow')
        self.assertEqual((term['share_target'],term['shadow_iterations']),(.1,4))
        self.assertTrue(C.is_shadow(self.shadow))
        self.assertFalse(A.is_on(self.shadow))

    def test_teaching_then_uses_the_g1307_recipe_without_the_raise_and_with_stage_a30(self):
        self.assertTrue(A.is_on(self.on))
        # The recipe the run used at g1293 (share .1), plus the g1294 PPG/head parts and the g1307 floor.
        from brass_ai.recovery.search_value_config import max_steps_options
        g1293=without_runtime(read(LATER/'recorded-config.json'))
        self.assertNotIn(S.RAISE,g1293['update'][S.KEY])
        expected=T07.configure(max_steps_options(P.kl_stop_options(g1293)))
        expected['teacher_phase']['search_value']['schedule']['stage_a_iterations']=30
        self.assertEqual(self.on,expected)
        self.assertEqual(self.on['update'][S.KEY],self.state['signature']['config']['update'][S.KEY])
        self.assertEqual(self.on['value_phase']['contract'],P.KL_STOP_CONTRACT)
        self.assertEqual(self.on['teacher_phase']['search_value']['steps_per_iteration'],64)
        self.assertTrue(T07.applied(self.on))

    def test_later_continuations_leave_it_alone(self):
        later=dict(self.state,iteration=1307)
        self.assertEqual(T94.configure(self.on,later),self.on)
        self.assertEqual(T07.configure(self.on),self.on)
        self.assertIsNone(T94.summary(self.on)['entropy'])

    def test_the_shadow_to_on_switch_is_a_reviewed_transition(self):
        signature=self.state['signature']
        shadow=dict(signature,config=self.shadow);on=dict(signature,config=self.on)
        self.assertTrue(C.signature_transition(shadow,on))
        self.assertEqual(C.collection_signature(shadow),C.collection_signature(on))
        ten=copy.deepcopy(self.on);ten['teacher_phase']['search_value']['schedule']['stage_a_iterations']=10
        self.assertFalse(C.signature_transition(dict(signature,config=ten),on))
        self.assertFalse(C.signature_transition(on,dict(signature,config=ten)))

    def test_any_valid_share_target_works_in_all_three_commands(self):
        signature=self.state['signature']
        for target in (.4,.25,1.):
            shadow=B.recipe(signature['config'],share_target=target)
            self.assertEqual(shadow['teacher_phase']['term']['share_target'],target)
            on=A.configure(shadow,target,revise_stages=True)
            self.assertTrue(C.is_shadow(shadow) and A.is_on(on))
            self.assertEqual(on['teacher_phase']['term']['share_target'],target)
            self.assertTrue(C.signature_transition(dict(signature,config=shadow),dict(signature,config=on)))
            # continue-rust-shadow.py may be given the share target after the restart was prepared with another one.
            prepared=dict(signature,config=self.shadow)
            self.assertTrue(C.signature_transition(prepared,dict(signature,config=shadow)))
        for bad in (0,-.1,1.5):
            with self.assertRaisesRegex(ValueError,'share target'):B.recipe(signature['config'],share_target=bad)

    def test_only_reviewed_stage_a_lengths(self):
        bad=copy.deepcopy(self.shadow);bad['teacher_phase']['search_value']['schedule']['stage_a_iterations']=20
        with self.assertRaises(ValueError):V.validate(bad['teacher_phase']['search_value'])
        with self.assertRaises(ValueError):B.recipe(self.state['signature']['config'],share_target=.1,stage_a_iterations=20)

    def test_an_old_run_cannot_take_the_restart_recipe(self):
        with self.assertRaisesRegex(ValueError,'teacher-on'):B.recipe(self.shadow,share_target=.1)


class ShareTargetChange(unittest.TestCase):
    """After the four shadow iterations qualify the gradient ratio, the share target only rescales beta."""
    @classmethod
    def setUpClass(cls):
        cls.state=read(EXPORT/'state.json')   # a real teacher-on state with its qualification
        cls.q=cls.state['teacher_phase']['weight']['gradient_qualification']

    def signature(self,target):
        signature=copy.deepcopy(self.state['signature'])
        signature['config']['teacher_phase']['term']['share_target']=target
        return signature

    def test_any_target_is_a_reviewed_on_to_on_transition_with_the_same_qualification(self):
        from brass_ai.recovery.teacher_gradient_qualification import fixed_beta
        for target in (.01,.1,.4,1.):
            signature=self.signature(target)
            self.assertTrue(C.signature_transition(self.state['signature'],signature))
            self.assertTrue(C.approved(self.state,signature))
            q=A.qualification(self.state,signature['config'])
            self.assertEqual((q['observations'],q['raw_median'],q['qualified_generation']),
                (self.q['observations'],self.q['raw_median'],self.q['qualified_generation']))
            self.assertEqual(q['beta'],fixed_beta(target,self.q['raw_median']))
            updated=dict(copy.deepcopy(self.state),signature=signature)
            with mock.patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value=dict(receipt={})):
                C.rebind_backend(self.state,updated)
            self.assertEqual(updated['native_backend']['teacher']['beta'],q['beta'])
            A.validate_backend(updated,updated['native_backend'])
            # The next iteration stores the same qualification under the new target.
            updated['teacher_phase']['weight']['gradient_qualification']=restore(self.q,target,4)
            A.validate_backend(updated,updated['native_backend'])

    def test_the_teacher_term_uses_the_rescaled_beta_without_new_shadow_iterations(self):
        from brass_ai.recovery.teacher_term import TeacherTerm
        from brass_ai.recovery.teacher_gradient_qualification import fixed_beta
        weight=self.state['teacher_phase']['weight']
        config=copy.deepcopy(self.state['signature']['config']['teacher_phase']);config['term']['share_target']=.4
        term=TeacherTerm.__new__(TeacherTerm)
        # Only the beta restore is under test; no rows or model are needed.
        with mock.patch('brass_ai.recovery.teacher_term.policy_logits',return_value=[]):
            term.__init__(None,config,[],seed=0,generation=1290,state=weight)
        self.assertEqual(term.beta,fixed_beta(.4,self.q['raw_median']))
        self.assertTrue(term.calibrated and term.qualification['passed'])

    def test_a_clamped_beta_is_flagged_for_the_new_target(self):
        low=restore(self.q,.001,4);high=restore(self.q,1.,4)
        self.assertEqual((low['beta'],low['clamped']),(.01,True))
        self.assertFalse(high['clamped'])


class StageA(unittest.TestCase):
    def options(self,a):
        state=read(EXPORT/'state.json')
        config=B.recipe(state['signature']['config'],share_target=.1,stage_a_iterations=a)
        return config['teacher_phase']['search_value']

    def test_stage_a_ends_after_its_configured_length(self):
        for a in (10,30):
            saved=dict(read(EXPORT/'state.json')['search_value']['stage']);saved.pop('buffer_rebuild')
            machine=StageMachine(saved=saved,options=self.options(a))
            generation=saved['generation']
            for count in range(saved['stage_a_iterations']+1,a+1):
                generation+=1;state=machine.advance(generation,{})
                self.assertEqual(state['stage'],'A' if count<a else 'B',(a,count))

    def test_a_committed_counter_beyond_the_length_is_refused(self):
        saved=dict(read(EXPORT/'state.json')['search_value']['stage'],stage_a_iterations=10)
        with self.assertRaises(ValueError):StageMachine(saved=saved,options=self.options(10))
        StageMachine(saved=saved,options=self.options(30))


class Pause(unittest.TestCase):
    def phase(self,generations):
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        phase=SearchValuePhase.__new__(SearchValuePhase)
        state=read(EXPORT/'state.json')
        options=B.recipe(state['signature']['config'],share_target=.1)['teacher_phase']['search_value']
        saved=dict(state['search_value']['stage'],buffer_rebuild=dict(origin_generation=1291,required_ages=8,
            ready_generation=None,instruction_id='a'*64))
        phase.options=options;phase.report={};phase.machine=StageMachine(saved=saved,options=options)
        rows=[dict(split='train',game_id=f'g{g}',index=0,generation=g,terminal_utility=[0.]*4,u=[0.]*4,r=[0.]*4)
              for g in generations]
        phase.buffer=type('Buffer',(),dict(rows=lambda self,generation,split:rows if split=='train' else []))()
        steps=[]
        phase.companion=type('Companion',(),dict(seed=1,head_only=lambda self,rows,generation,steps_:steps.append(1) or dict(steps=1)))()
        phase.companion.head_only=lambda rows,generation,steps:steps_append(steps)
        def steps_append(n):
            steps.append(n);return dict(steps=n)
        phase.trunk_scale=lambda:0.
        return phase,steps

    def test_counts_pause_through_the_first_eight_iterations(self):
        for generation in range(1294,1298):
            phase,steps=self.phase(range(1290,generation+1))
            with mock.patch('brass_ai.recovery.search_value_schedule.diagnostics',return_value={}):
                report=phase._scheduled_after_ppg(generation,guarded=None)
            self.assertTrue(report['buffer_rebuild']['paused'],generation);self.assertEqual(steps,[])
            self.assertEqual(report['stage']['stage_a_iterations'],4)
        phase,steps=self.phase(range(1291,1299))
        with mock.patch('brass_ai.recovery.search_value_schedule.diagnostics',return_value={}):
            report=phase._scheduled_after_ppg(1298,guarded=None)
        self.assertFalse(report['buffer_rebuild']['paused']);self.assertEqual(steps,[64])
        self.assertEqual(report['stage']['stage_a_iterations'],5)


class Build(unittest.TestCase):
    """A small real run tree built from the exported g1290 state, then loaded back."""
    def setUp(self):
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.recovery.search_value_iteration import SearchValueIteration,stamp
        temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(temp.cleanup)
        base=Path(temp.name);self.source=base/'source';self.run=base/'restart'
        old='artifacts/step5/joint7-v11-2-g1290-trained-restart/';new=self.source.relative_to(ROOT).as_posix()+'/'
        state=read(EXPORT/'state.json')
        state={k:B.rewrite(v,old,new) if k!='signature' else v for k,v in state.items()}
        config=state['signature']['config']
        torch.manual_seed(5)
        model=UtilityNet(utility_config(ModelConfig.preset(config['model_size']),reward_spec(config)))
        identity=model_id(model)
        # Stand-ins for every file the state refers to, with matching receipts.
        def stand_in(receipt):
            path=ROOT/receipt['path'];path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(('stand-in '+receipt['path']).encode());receipt['sha256']=sha(path)
        for where,receipt in B.receipts(state,new):
            if where.startswith(('/il_kl_limit',)+B.RECORD_PREFIXES):continue
            stand_in(receipt)
        last=state['history'][-1];last['model_id']=identity
        agent=state['ratings']['agents']['generation-001290']['checkpoint'];agent['model_id']=identity
        self.missing=next(m for m in state['quality_pool']['members'] if m['generation']==1045)
        self.unprobed=next(m for m in state['quality_pool']['members'] if m['generation']==1090)
        pfsp=__import__('brass_ai.recovery.pfsp_pool',fromlist=['seal']);pfsp.seal(state['quality_pool'])
        sv=SearchValueIteration(self.source,config,dict(iteration=1289,history=[dict(generation=1289,model_id=identity,sha256='a'*64)]),model)
        value=sv.payload(1290);value['stage']=dict(value['stage'],stage_a_iterations=4)
        state['search_value']=stamp(value)
        path=ROOT/agent['path']
        save_checkpoint(path,model,None,step=1290,training=dict(fresh_run=new[:-1],signature=digest(state['signature']),
            quality_pool_digest=quality_pool.state_digest(state['quality_pool']),teacher_phase=state['teacher_phase'],
            search_value=value,exploration=state['exploration'],startup_initialization=state['startup_initialization'],
            il_kl_limit=state['il_kl_limit'],native_backend=state['native_backend'],machine_transfer=state.get('machine_transfer')))
        agent['sha256']=sha(path)
        (ROOT/self.missing['path']).unlink()
        (ROOT/self.unprobed['probe']['path']).unlink()
        write_json(self.source/'state.json',state)
        snapshot=base/'export';snapshot.mkdir()
        write_json(snapshot/'state.json',state)
        report=EXPORT/'iteration-1289/report.json';(snapshot/'iteration-1289').mkdir()
        (snapshot/'iteration-1289/report.json').write_bytes(report.read_bytes())
        write_json(snapshot/'export-manifest.json',dict(run=new[:-1],committed_generation=1290,files=[
            dict(path='state.json',source=new+'state.json',sha256=sha(snapshot/'state.json')),
            dict(path='iteration-1289/report.json',source=new+'iteration-1289/report.json',sha256=sha(snapshot/'iteration-1289/report.json'))]))
        self.snapshot=snapshot;self.state=state;self.model=model
        probe=type('Probe',(),dict(metadata=dict(probe_hash=state['il_kl_limit']['probe']['probe_hash']),
            probabilities=lambda self,model,update:np.full(8,.125,dtype=np.float32)))()
        self.receipt=dict(path=(self.run/'il-kl/manifest.json').relative_to(ROOT).as_posix(),sha256='b'*64,
            probe_hash=state['il_kl_limit']['probe']['probe_hash'])
        real=load_checkpoint
        def loader(path,**kw):
            return (self.model,{}) if Path(path).name.startswith('weights-'+self.unprobed['model_id']) else real(path,**kw)
        for patcher in (mock.patch('brass_ai.recovery.il_kl_probe.import_probe',return_value=(probe,self.receipt)),
                        mock.patch('brass_ai.learning.checkpoint.load_checkpoint',side_effect=loader)):
            patcher.start();self.addCleanup(patcher.stop)

    def build(self):
        return B.prepare(ROOT,self.source,self.snapshot,self.run,share_target=.1)

    def test_the_new_run_loads_with_fresh_optimizers_and_kept_state(self):
        summary=self.build()
        self.assertEqual(summary['dropped_members'],[dict(generation=1045,model_id=self.missing['model_id'])])
        self.assertEqual(summary['regenerated_probes'],[1090])
        state=read(self.run/'state.json');config=state['signature']['config']
        self.assertEqual(state['iteration'],1290);self.assertEqual(state['source_transitions'],[])
        self.assertNotIn('machine_transfer',state);self.assertNotIn('teacher',state['native_backend'])
        self.assertEqual(state['teacher_phase']['weight'],{})
        self.assertEqual(state['il_kl_limit']['probe'],self.receipt)
        self.assertEqual(state['exploration'],self.state['exploration'])
        new=self.run.relative_to(ROOT).as_posix()+'/'
        self.assertTrue(all(e['path'].startswith(new) for e in state['history']))
        for m in state['quality_pool']['members']:
            if 'path' in m:self.assertEqual(sha(ROOT/m['path']),m['sha256'])
        quality_pool.validate_state(state['quality_pool'],config['opponent_pool'],1290)
        from brass_ai.recovery.resume import load_resume
        model,raw,changed=load_resume(ROOT,self.run,state,state['signature'])
        self.assertFalse(changed)
        self.assertEqual(model_id(model),model_id(self.model))
        # PPO: a new AdamW with its warmup and controller.
        from brass_ai.learning.search_value import ordinary_parameters
        from brass_ai.learning.checkpoint import resume_optimizer
        from brass_ai.recovery.resume import apply_optimizer_config
        optimizer=torch.optim.AdamW(ordinary_parameters(model),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        resume_optimizer(raw,optimizer,allow_cuda_topology_change=True);apply_optimizer_config(optimizer,config)
        self.assertEqual(optimizer.state_dict()['state'],{})
        from brass_ai.recovery import exploration
        restored=exploration.restore(config,state,raw['training']['exploration'])
        self.assertEqual(S.effective(config,restored),dict.fromkeys(S.GROUPS,.001))
        from brass_ai.recovery.actor_value import TrainingPhases
        phases=TrainingPhases(model,config,raw=raw,generation=1290,root=ROOT,run=self.run)
        self.assertEqual(phases.beta,.25);self.assertEqual(phases.buffer.receipts,[])
        self.assertEqual(phases.state['active_lambda'],.95)
        self.assertTrue(phases.state['table_enabled']['historical'])
        from brass_ai.recovery.teacher_phase import TeacherPhase
        teacher=TeacherPhase(ROOT,self.run,config,state,model,raw)
        self.assertEqual(teacher.weight,{});self.assertTrue(teacher.ready)
        machine=teacher.search_value.phase.machine
        self.assertEqual((machine.state['stage'],machine.state['stage_a_iterations'],machine.stage_a),('A',4,30))
        self.assertEqual(machine.state['buffer_rebuild']['origin_generation'],1291)
        self.assertEqual(list(teacher.search_value.phase.buffer.receipts),[])
        with mock.patch('brass_ai.rust_backend.lab_backend.checked_pin',return_value=dict(passed=True,
                implementation=state['native_backend']['implementation'])),\
                mock.patch('brass_ai.rust_backend.lab_backend.pin',side_effect=lambda path:state['native_backend']['rollback']):
            __import__('brass_ai.rust_backend.lab_backend',fromlist=['validate_state']).validate_state(state,raw,check_runtime=False)

    def test_the_source_run_is_untouched_and_a_rerun_refuses(self):
        before={p:sha(p) for p in self.source.rglob('*') if p.is_file()}
        self.build()
        self.assertEqual(before,{p:sha(p) for p in self.source.rglob('*') if p.is_file()})
        with self.assertRaisesRegex(ValueError,'already has training progress'):self.build()

    def test_an_unfinished_attempt_can_start_over_with_another_share_target(self):
        before={p:sha(p) for p in self.source.rglob('*') if p.is_file()}
        with mock.patch('brass_ai.recovery.il_kl_probe.import_probe',side_effect=ValueError('stopped midway')):
            with self.assertRaisesRegex(ValueError,'stopped midway'):self.build()
        self.assertTrue((self.run/B.RECEIPT).exists() and any(p.name.startswith('weights-') for p in self.run.rglob('*.pt')))
        summary=B.prepare(ROOT,self.source,self.snapshot,self.run,share_target=.4)
        self.assertTrue(summary['complete'])
        config=read(self.run/'state.json')['signature']['config']
        self.assertEqual(config['teacher_phase']['term']['share_target'],.4)
        self.assertEqual(before,{p:sha(p) for p in self.source.rglob('*') if p.is_file()})
        # A finished preparation is never replaced.
        with self.assertRaisesRegex(ValueError,'already has training progress'):self.build()

    def test_a_marker_of_another_recipe_without_the_flag_is_not_cleared(self):
        write_json(self.run/B.RECEIPT,dict(binding=dict(other=1),complete=True))
        with self.assertRaisesRegex(ValueError,'another source or recipe'):self.build()

    def retire_archive(self,keep):
        """The source run compacted its ratings again: the archive this state names is gone."""
        archive=self.state['ratings']['archive'];source=ROOT/archive['path']
        relative=archive['path'][len(self.source.relative_to(ROOT).as_posix())+1:]
        if keep:
            (self.snapshot/relative).parent.mkdir(parents=True,exist_ok=True)
            (self.snapshot/relative).write_bytes(source.read_bytes())
        source.unlink();return relative

    def test_a_superseded_rating_archive_comes_from_the_export(self):
        relative=self.retire_archive(True)
        self.build()
        placed=self.run/relative
        self.assertEqual(sha(placed),self.state['ratings']['archive']['sha256'])
        self.assertEqual(read(self.run/'state.json')['ratings']['archive']['path'],placed.relative_to(ROOT).as_posix())

    def test_a_file_in_neither_place_is_refused(self):
        self.retire_archive(False)
        with self.assertRaisesRegex(ValueError,'no longer holds'):self.build()

    def test_an_exported_copy_with_other_bytes_is_not_used(self):
        relative=self.retire_archive(True);(self.snapshot/relative).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'no longer holds'):self.build()

    def test_weights_from_another_generation_are_refused(self):
        with self.assertRaisesRegex(ValueError,'retained snapshot'):
            B.prepare(ROOT,self.source,self.snapshot,self.run,weights=self.state['evaluation_anchor']['path'])


if __name__=='__main__':unittest.main()
