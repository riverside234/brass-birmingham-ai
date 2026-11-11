"""The g1490 rollback restart: its recipe variant, one-iteration beta, cotton receipts and the built run."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT,digest
from brass_ai.recovery import branch_restart as B,teacher_gradient_qualification as Q,training_revision_g1491 as R
from brass_ai.recovery import lr_controller as L,quality_pool
from brass_ai.recovery.collector_runtime import recipe
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.teacher_phase import plan_jobs
from brass_ai.recovery.teacher_phase_config import buffer_identity
from brass_ai.rust_backend import lab_shadow_config as C,lab_teacher_activation as A
from brass_ai.step5.league import write_json

EXPORT=ROOT/'results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z'
LATER=ROOT/'results/joint7-v11-2-g1290-restart-g1290-g1500-20261110T232721030490Z'
SOURCE='artifacts/step5/joint7-v11-2-g1290-restart-g1290/'
PHASES=('canal_early','canal_late','rail_early','rail_late')


def committed():
    state=read(EXPORT/'state.json')
    return state,recipe(state['signature']['config'])


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state,cls.before=committed()
        cls.after=B.rollback_recipe(cls.state['signature']['config'],cls.state,share_target=.75)

    def test_exactly_the_requested_fields_change(self):
        validate_config(self.after)
        self.assertEqual({k for k in self.after if self.after[k]!=self.before[k]},{'teacher_phase'})
        old,new=self.before['teacher_phase'],self.after['teacher_phase']
        self.assertEqual({k for k in set(old)|set(new) if old.get(k)!=new.get(k)},{'mode','tables','term',R.KEY})
        self.assertEqual((new['mode'],new['tables']),('shadow',dict(train=32,held_out=4)))
        self.assertEqual(new['buffer'],dict(iterations=4,age_weights=[4,3,2,1],max_staleness_ratio=1.))
        self.assertEqual(new['term'],dict(old['term'],rows_per_step=72,shadow_iterations=1,share_target=.75))
        self.assertEqual(self.after['update']['epochs'],2)
        from brass_ai.recovery.teacher_cotton import settings
        self.assertEqual(new[R.KEY],dict(contract=R.CONTRACT,variant=R.RESTART,labels=dict(searches='long-only'),
            exploration=dict(contract='root-visit-count-sampling/1.0.0',temperature=1.,opening_actions_per_era=7,
                long_search='always',cotton=settings()),calibration=dict(shadow_iterations=1,ppo_lr='hold')))
        self.assertIsNone(R.long_weight(self.after))
        self.assertEqual(R.shadow_iterations(self.after),1)
        self.assertTrue(R.holds_ppo_lr(self.after))

    def test_it_is_a_reviewed_shadow_recipe_projecting_to_the_committed_one(self):
        self.assertTrue(C.is_shadow(self.after))
        self.assertFalse(A.is_on(self.after))
        expected=copy.deepcopy(self.before);expected['teacher_phase']['mode']='shadow'
        expected['teacher_phase']['term'].update(shadow_iterations=2,share_target=.75)
        self.assertEqual(R.previous(self.after),expected)

    def test_the_in_run_g1491_marker_is_unchanged(self):
        g1491=R.configure(self.before,self.state)
        self.assertNotIn('variant',g1491['teacher_phase'][R.KEY])
        self.assertEqual(digest(R.block()),digest(g1491['teacher_phase'][R.KEY]))
        self.assertEqual(R.shadow_iterations(g1491),2)
        with self.assertRaisesRegex(ValueError,'Another g1491 revision variant'):
            R.configure(g1491,self.state,variant=R.RESTART)

    def test_other_values_are_not_reviewed(self):
        for path,value in [('teacher_phase.tables.train',28),('teacher_phase.tables.train',48),
                           ('teacher_phase.buffer.age_weights',[4,3,2,2]),('teacher_phase.term.rows_per_step',64),
                           ('teacher_phase.term.rows_per_step',2048),('teacher_phase.term.shadow_iterations',2),
                           ('update.epochs',1),(f'teacher_phase.{R.KEY}.labels.searches','long-and-short'),
                           (f'teacher_phase.{R.KEY}.calibration.shadow_iterations',2)]:
            bad=copy.deepcopy(self.after);target=bad;keys=path.split('.')
            for k in keys[:-1]:target=target[k]
            target[keys[-1]]=value
            with self.subTest(path=path,value=value),self.assertRaises(ValueError):validate_config(bad)
        # The long-only variant is only reviewed with its cotton exploration.
        bad=copy.deepcopy(self.after);bad['teacher_phase'][R.KEY]['exploration'].pop('cotton')
        with self.assertRaises(ValueError):validate_config(bad)

    def test_jobs_label_long_searches_only_and_explore_as_before(self):
        jobs,_=plan_jobs(self.after,self.state,model_id='m'*64,export_hash='a'*64,epoch=1491,ready=True)
        self.assertEqual([j['split'] for j in jobs].count('train'),32)
        self.assertEqual([j['split'] for j in jobs].count('heldout'),4)
        for job in jobs:
            revision=job['teacher_revision']
            self.assertIs(revision['short_labels'],False)
            self.assertEqual('cotton' in revision['exploration'],job['split']=='train')
            self.assertEqual({k:v for k,v in revision['exploration'].items() if k!='cotton'},
                dict(contract='root-visit-count-sampling/1.0.0',temperature=1.,opening_actions_per_era=7,long_search=True))
            self.assertEqual(R.validate_job(job),revision)
        changed=copy.deepcopy(jobs[0]);changed['teacher_revision']['short_labels']=True
        # Short labels with this exploration are the g1491 job, which is still reviewed as such.
        self.assertEqual(R.validate_job(changed),changed['teacher_revision'])
        changed['teacher_revision']['exploration']['opening_actions_per_era']=8
        with self.assertRaisesRegex(ValueError,'Unreviewed g1491'):R.validate_job(changed)

    def test_short_rows_are_refused_by_admission(self):
        from brass_ai.recovery.teacher_labels import check_turn
        job=dict(reservations={'0':dict(canal_early=3)},teacher_revision=R.job_block(self.after))
        check_turn(dict(seat=0,phase='canal_early'),3,job)
        with self.assertRaisesRegex(ValueError,'Short-search labels need'):
            check_turn(dict(seat=0,phase='canal_early',profile='fast'),2,job)

    def test_new_labels_retire_the_old_labels_and_beta(self):
        from brass_ai.recovery.teacher_phase import committed_weight
        self.assertNotEqual(buffer_identity(self.after),buffer_identity(self.before))
        self.assertEqual(committed_weight(self.after,self.state),{})


class Transitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state,cls.before=committed()
        cls.shadow=B.rollback_recipe(cls.state['signature']['config'],cls.state,share_target=.75)
        cls.on=A.configure(cls.shadow,.75,revise_stages=True)

    def signature(self,config):
        return dict(self.state['signature'],config=config)

    def test_it_is_never_adopted_inside_a_run(self):
        self.assertFalse(C.signature_transition(self.state['signature'],self.signature(self.shadow)))

    def test_one_calibration_iteration_then_teaching_on(self):
        self.assertTrue(A.is_on(self.on))
        self.assertEqual({k for k in self.on['teacher_phase'] if self.on['teacher_phase'][k]!=self.shadow['teacher_phase'][k]},{'mode'})
        self.assertEqual({k for k in self.on if self.on[k]!=self.shadow[k]},{'teacher_phase'})
        self.assertTrue(C.signature_transition(self.signature(self.shadow),self.signature(self.on)))
        self.assertTrue(C.signature_transition(self.signature(self.on),self.signature(self.on)))
        self.assertFalse(C.signature_transition(self.signature(self.on),self.signature(self.shadow)))
        other=copy.deepcopy(self.on);other['teacher_phase']['term']['share_target']=.5
        self.assertTrue(C.signature_transition(self.signature(self.on),self.signature(other)))
        self.assertFalse(R.holds_ppo_lr(self.on))
        self.assertEqual(C.collection_signature(self.signature(self.shadow)),C.collection_signature(self.signature(self.on)))
        # The g1491 variant cannot be swapped in.
        g1491=A.configure(R.configure(self.before,self.state),.75,revise_stages=True)
        self.assertFalse(C.signature_transition(self.signature(self.on),self.signature(g1491)))


class Qualification(unittest.TestCase):
    """One full 32+4 shadow iteration fixes beta; teaching resumes at Stage C."""

    @classmethod
    def setUpClass(cls):
        cls.state,cls.before=committed()
        cls.shadow=B.rollback_recipe(cls.state['signature']['config'],cls.state,share_target=.75)
        cls.on=A.configure(cls.shadow,.75,revise_stages=True)

    def measure(self,generation,previous,raw,tables=None):
        context=dict(tables=tables or {'train':32,'heldout':4},labels_by_phase={p:30 for p in PHASES},
            model_id='m%d'%generation,export_hash='h%d'%generation)
        term=SimpleNamespace(config=self.shadow['teacher_phase'],mode='shadow',generation=generation,qualification=previous,
            qualification_context=context,beta=1.,calibrated=False,
            steps=[dict(raw_share=v,cosine=.01,actor_norm=1.,teacher_norm=v,share=v) for v in raw])
        state={};Q.finish(term,state)
        return state['gradient_qualification']

    def test_one_full_iteration_qualifies_and_authorizes_teaching(self):
        q=self.measure(1491,None,[4.,5.,6.])
        self.assertTrue(q['passed'])
        self.assertEqual((q['required_iterations'],q['raw_median'],q['qualified_generation']),(1,5.,1492))
        self.assertEqual(q['beta'],Q.fixed_beta(.75,5.))
        state=copy.deepcopy(self.state)
        state.update(iteration=1492,signature=dict(state['signature'],config=self.shadow))
        state['exploration']['entropy_schedule']['last_generation']=state['exploration']['generation']=1492
        state['history']=state['history'][:-1]+[dict(generation=g,model_id='m%d'%g,sha256='h%d'%g,path='x') for g in (1491,1492)]
        state['teacher_phase']['weight']=dict(gradient_qualification=q)
        self.assertEqual(A.authorize(state,self.on)['beta'],q['beta'])
        self.assertEqual(Q.restore(q,.75,1),q)

    def test_a_48_table_iteration_does_not_qualify(self):
        q=self.measure(1491,None,[4.,5.],tables={'train':48,'heldout':4})
        self.assertFalse(q['passed']);self.assertEqual(q['observations'],[])
        self.assertIn('32+4',q['reason'])

    def test_the_continue_command_counts_one_calibration_iteration(self):
        spec=importlib.util.spec_from_file_location('continue_g1491',ROOT/'scripts/continue-rust-g1491.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        # The restart starts with no committed beta or qualification.
        state=dict(self.state,signature=dict(self.state['signature'],config=self.shadow),
                   teacher_phase=dict(self.state['teacher_phase'],weight={}))
        self.assertEqual(m.calibrated(state),(False,1))
        q=self.measure(1491,None,[4.,5.,6.])
        state=dict(state,iteration=1492,teacher_phase=dict(state['teacher_phase'],weight=dict(gradient_qualification=q)))
        self.assertEqual(m.calibrated(state),(True,0))
        self.assertEqual(m.calibrated(dict(state,signature=dict(state['signature'],config=self.on))),(True,0))


class Cotton(unittest.TestCase):
    """Short searches still explore, unlabelled: the label sequence may skip allowance, never go back."""

    def counts(self,targeted):
        return dict(action_decisions=10,legal_decisions=5,eligible_decisions=4,covered_decisions=3,no_evidence=1,
            no_cotton_visits=0,triggers=targeted,targeted_builds=targeted,actual_builds=targeted)

    def receipt(self,used,long,short):
        from brass_ai.recovery.teacher_cotton import CONTRACT
        return dict(contract=CONTRACT,targeted_by_era_seat=used,builds_by_era_seat=copy.deepcopy(used),
            long=self.counts(long),short=self.counts(short))

    def row(self,used_before,targeted=True):
        from brass_ai.recovery.teacher_cotton import CONTRACT
        return dict(view=dict(context=dict(era='CanalEra')),teacher=dict(seat=0,cotton_exploration=dict(
            contract=CONTRACT,targeted=targeted,used_before=used_before)))

    def job(self,short):
        return dict(teacher_revision=dict(short_labels=short,exploration=dict(cotton={})))

    def test_long_only_labels_allow_short_search_spending_between_them(self):
        from brass_ai.recovery.teacher_cotton import validate_receipt
        used=[[2,0,0,0],[0,0,0,0]]
        # A short search spent the first allowance; the long label saw it spent.
        validate_receipt(self.receipt(used,1,1),self.job(False),[self.row(1)])
        with self.assertRaisesRegex(ValueError,'allowance sequence'):
            validate_receipt(self.receipt(used,1,1),self.job(True),[self.row(1)])
        with self.assertRaisesRegex(ValueError,'allowance sequence'):
            validate_receipt(self.receipt([[3,0,0,0],[0]*4],2,1),self.job(False),[self.row(1),self.row(1)])
        with self.assertRaisesRegex(ValueError,'quota differs'):
            validate_receipt(self.receipt(used,2,0),self.job(False),[self.row(1)])
        with self.assertRaisesRegex(ValueError,'quota differs'):
            validate_receipt(self.receipt([[1,0,0,0],[0]*4],1,0),self.job(False),[self.row(1)])
        # With every search labelled the exact sequence still holds.
        validate_receipt(self.receipt([[1,0,0,0],[0]*4],1,0),self.job(True),[self.row(0)])


class Verifier(unittest.TestCase):
    def test_long_rows_of_a_revised_table_check_logits_once_per_table(self):
        from brass_ai.recovery.teacher_phase_config import TIMING
        from brass_ai.rust_backend import lab_teacher_finalization as F
        state,_=committed()
        config=B.rollback_recipe(state['signature']['config'],state,share_target=.75)
        jobs,_=plan_jobs(config,state,model_id='m'*64,export_hash='a'*64,epoch=1491,ready=True)
        job=jobs[0];identity=job['expected']['job_id']
        seat,phase='0','canal_early';round_=job['reservations'][seat][phase]
        v=F.TurnVerifier.__new__(F.TurnVerifier)
        v.jobs={identity:job};v.cache={};v.lock=threading.Lock();v.cancel=threading.Event();v.diagnostic=False
        v.sources=set();v.encoders=threading.local();v.search_limits=None;v.history=[];v._check_model=lambda:None
        v.model=object()
        label=dict(seat=0,phase=phase)
        accepted=[SimpleNamespace(row=dict(view=dict(context=dict(round=round_))),label=label,encoded='e')]
        meta=dict(job=job['expected'],timing=TIMING,profile='full',rows=[dict(teacher=label)],remaining_ns=0,
            native_post_ns=0,search_ns=0,index=1)
        with mock.patch.object(F,'prepare_rows',return_value=(accepted,{})),mock.patch.object(F,'check_logits') as check:
            v.verify(meta,0.)
            check.assert_not_called()
            self.assertEqual(len(v.cache[identity]['deferred']),1)
            self.assertEqual(v.settle(identity)['rows'],1)
            check.assert_called_once()


class Term(unittest.TestCase):
    def test_the_long_only_variant_keeps_the_single_pool_loss(self):
        from test_b025 import config,example,model_fixture
        from brass_ai.recovery.teacher_term import TeacherTerm
        model=model_fixture()
        options=copy.deepcopy(config()['teacher_phase'])
        options.update(mode='on');options['term']['rows_per_step']=3
        options[R.KEY]=R.block(variant=R.RESTART)
        rows=[(i%2,example(model,seat=i%4,round=1+i%8)[0]) for i in range(4)]
        term=TeacherTerm(model,options,rows,seed=5,generation=12,state=dict(beta=1.,calibrated=True))
        self.assertIsNone(term.long_weight)
        self.assertNotIn('long_weight',term.report())
        short=copy.deepcopy(rows);short[0][1].row['teacher'].update(profile='fast',requested_k=8)
        with self.assertRaisesRegex(ValueError,'need the g1491 recipe'):
            TeacherTerm(model,options,short,seed=5,generation=12)


class Build(unittest.TestCase):
    """A small real run tree built from the exported g1491 state, then loaded back."""

    def setUp(self):
        from brass_ai.learning.model import ModelConfig
        from brass_ai.learning.utility_model import UtilityNet,utility_config
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.recovery.search_value_iteration import SearchValueIteration,stamp
        from brass_ai.recovery.local_scoring_auxiliary import Companion,KEY as COMPANION,bundle_id
        temp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');self.addCleanup(temp.cleanup)
        base=Path(temp.name);self.source=base/'source';self.run=base/'restart'
        new=self.source.relative_to(ROOT).as_posix()+'/';self.new=new
        state=read(EXPORT/'state.json')
        state={k:B.rewrite(v,SOURCE,new) if k!='signature' else v for k,v in state.items()}
        config=state['signature']['config']
        torch.manual_seed(5)
        model=UtilityNet(utility_config(ModelConfig.preset(config['model_size']),reward_spec(config)))
        identity=model_id(model)
        slot=next(s for s in state['ratings']['slots'] if s['generation']==1400)
        keep={slot[k]['path'] for k in ('records','report','spec','panels','bootstrap')}
        archive=state['ratings']['archive']
        # Stand-ins for every file the state refers to; slot 1400 is gone (the source compacted it).
        for where,receipt in B.receipts(state,new):
            if where.startswith(('/il_kl_limit',)+B.RECORD_PREFIXES) or receipt['path'] in keep:continue
            path=ROOT/receipt['path'];path.parent.mkdir(parents=True,exist_ok=True)
            if receipt is archive:path.write_bytes((EXPORT/receipt['path'][len(new):]).read_bytes())
            else:path.write_bytes(('stand-in '+receipt['path']).encode());receipt['sha256']=sha(path)
        self.assertEqual(sha(ROOT/archive['path']),archive['sha256'])
        previous=state['history'][-2];self.assertEqual(previous['generation'],1490)
        previous['model_id']=identity
        agent=state['ratings']['agents']['generation-001490']['checkpoint'];agent['model_id']=identity
        # The learner's g1490 payloads, one generation behind the exported state.
        exploration=copy.deepcopy(state['exploration'])
        exploration['generation']=exploration['entropy_schedule']['last_generation']=1490
        sv=SearchValueIteration(self.source,config,dict(iteration=1489,history=[dict(generation=1489,model_id=identity,sha256='a'*64)]),model)
        value=sv.payload(1490);value['stage']=dict(copy.deepcopy(state['search_value']['stage']),generation=1489)
        companion=Companion(model,config[COMPANION],raw=dict(model_id=identity)).payload(0)
        companion.pop('identity')
        companion.update(generation=1490,age=1490,retained_steps=dict(ppo=40000,ppg=40000),
            warmup=dict(companion['warmup'],generation=1490,qualifying_iterations=1000,fitting_steps=40000,released_at=2,encoder_stage=5))
        companion['identity']=bundle_id(companion)
        path=ROOT/agent['path']
        save_checkpoint(path,model,None,step=1490,training=dict(fresh_run=new[:-1],signature=digest(state['signature']),
            quality_pool_digest='0'*64,teacher_phase=state['teacher_phase'],search_value=value,exploration=exploration,
            startup_initialization=state['startup_initialization'],auxiliary_initialization=state.get('auxiliary_initialization'),
            il_kl_limit=state['il_kl_limit'],native_backend=state['native_backend'],v4_share_gate=state['v4_share_gate'],
            **{COMPANION:companion}))
        # The pool member and population entry name the same retained file.
        for _,receipt in B.receipts(state,new):
            if receipt['path']==agent['path']:receipt['sha256']=sha(path)
        write_json(self.source/'state.json',state)
        snapshot=base/'export';snapshot.mkdir()
        write_json(snapshot/'state.json',state)
        (snapshot/'iteration-1490').mkdir()
        (snapshot/'iteration-1490/report.json').write_bytes((EXPORT/'iteration-1490/report.json').read_bytes())
        write_json(snapshot/'export-manifest.json',dict(run=new[:-1],committed_generation=1491,files=[
            dict(path='state.json',source=new+'state.json',sha256=sha(snapshot/'state.json')),
            dict(path='iteration-1490/report.json',source=new+'iteration-1490/report.json',sha256=sha(snapshot/'iteration-1490/report.json'))]))
        # The later export: only its committed rating archive matters here.
        later=base/'later';(later/'ratings').mkdir(parents=True)
        name='ratings/history-5cb5bb67ff13b24695f1e072b180f22bac015c6197b2e11843a981bb93161ac0.json'
        (later/name).write_bytes((LATER/name).read_bytes())
        write_json(later/'state.json',dict(ratings=dict(archive=dict(path=new+name,sha256=sha(later/name)))))
        write_json(later/'export-manifest.json',dict(run=new[:-1],committed_generation=1500,files=[
            dict(path='state.json',source=new+'state.json',sha256=sha(later/'state.json')),
            dict(path=name,source=new+name,sha256=sha(later/name))]))
        self.snapshot,self.later,self.state,self.model,self.weights=snapshot,later,state,model,agent['path']
        probe=type('Probe',(),dict(metadata=dict(probe_hash=state['il_kl_limit']['probe']['probe_hash']),
            probabilities=lambda self,model,update:np.full(8,.125,dtype=np.float32)))()
        self.receipt=dict(path=(self.run/'il-kl/manifest.json').relative_to(ROOT).as_posix(),sha256='b'*64,
            probe_hash=state['il_kl_limit']['probe']['probe_hash'])
        patcher=mock.patch('brass_ai.recovery.il_kl_probe.import_probe',return_value=(probe,self.receipt))
        patcher.start();self.addCleanup(patcher.stop)

    def build(self,**kw):
        options=dict(share_target=.75,weights=self.weights,rollback=dict(later=self.later));options.update(kw)
        return B.prepare(ROOT,self.source,self.snapshot,self.run,**options)

    def test_the_new_run_starts_from_g1491_with_the_g1490_learner(self):
        before={p:sha(p) for p in self.source.rglob('*') if p.is_file()}
        summary=self.build()
        self.assertEqual(before,{p:sha(p) for p in self.source.rglob('*') if p.is_file()})
        identity=model_id(self.model)
        self.assertEqual((summary['learner']['generation'],summary['learner']['model_id']),(1490,identity))
        self.assertEqual((summary['compacted_rating_slots'],summary['ppo_lr_multiplier']),([1400],1.25))
        self.assertEqual(summary['schedule'],dict(shadow_iterations=1,teacher_on_from_iteration=2,
            search_head_stage='C',search_head_paused_iterations=8))
        state=read(self.run/'state.json');config=state['signature']['config']
        self.assertEqual(state['iteration'],1491)
        self.assertEqual((state['history'][-1]['generation'],state['history'][-1]['model_id']),(1491,identity))
        self.assertEqual(state['resume'],[state['history'][-1]])
        self.assertEqual(R.variant_of(config),R.RESTART)
        self.assertEqual(config['teacher_phase']['term']['share_target'],.75)
        # Ratings: the compaction the source made at g1500, verified and applied.
        self.assertEqual([s['generation'] for s in state['ratings']['slots']],list(range(1410,1500,10)))
        self.assertTrue(state['ratings']['archive']['path'].startswith(self.run.relative_to(ROOT).as_posix()))
        self.assertEqual(sha(ROOT/state['ratings']['archive']['path']),state['ratings']['archive']['sha256'])
        from brass_ai.recovery.ratings import validate_state as ratings_state
        ratings_state(ROOT,state['ratings'],config['rating'])
        # Teacher labels, beta and the backend receipt.
        self.assertEqual(state['teacher_phase']['weight'],{})
        self.assertEqual(state['teacher_phase']['buffer']['receipts'],[])
        self.assertEqual(state['teacher_phase']['buffer']['identity'],buffer_identity(config))
        self.assertNotIn('teacher',state['native_backend'])
        self.assertEqual(state['native_backend']['shadow']['tables'],dict(train=32,held_out=4))
        self.assertEqual(state['native_backend']['shadow']['teacher_build'],self.state['native_backend']['shadow']['teacher_build'])
        quality_pool.validate_state(state['quality_pool'],config['opponent_pool'],1491)
        # The run loads: checkpoint, optimizers, schedules, teacher phase, companion and backend.
        from brass_ai.recovery.resume import load_resume,apply_optimizer_config
        model,raw,changed=load_resume(ROOT,self.run,state,state['signature'])
        self.assertFalse(changed);self.assertEqual(model_id(model),identity);self.assertEqual(raw['step'],1491)
        from brass_ai.learning.search_value import ordinary_parameters
        from brass_ai.learning.checkpoint import resume_optimizer
        optimizer=torch.optim.AdamW(ordinary_parameters(model),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        resume_optimizer(raw,optimizer,allow_cuda_topology_change=True);apply_optimizer_config(optimizer,config)
        self.assertEqual(optimizer.state_dict()['state'],{})
        group=optimizer.param_groups[0]
        self.assertEqual((group[L.KEY]['multiplier'],group[L.KEY]['window'],L.multiplier(group)),(1.25,[],1.))
        from brass_ai.recovery import exploration
        exploration.restore(config,state,raw['training']['exploration'])
        from brass_ai.recovery.actor_value import TrainingPhases
        phases=TrainingPhases(model,config,raw=raw,generation=1491,root=ROOT,run=self.run)
        self.assertEqual(phases.buffer.receipts,[])
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        companion=Companion(model,config['local_scoring_auxiliary'],raw=raw,generation=1491)
        self.assertEqual((companion.generation,companion.warmup['generation'],companion.retained_steps['ppg']),(1491,1491,40000))
        from brass_ai.recovery.teacher_phase import TeacherPhase
        teacher=TeacherPhase(ROOT,self.run,config,state,model,raw)
        self.assertEqual(teacher.weight,{});self.assertTrue(teacher.ready)
        self.assertEqual(teacher.buffer.iterations,4)
        machine=teacher.search_value.phase.machine
        self.assertEqual((machine.state['stage'],machine.state['w'],machine.state['scale']),('C',1.,1.))
        self.assertEqual(machine.state['buffer_rebuild']['origin_generation'],1492)
        with mock.patch('brass_ai.rust_backend.lab_backend.checked_pin',return_value=dict(passed=True,
                implementation=state['native_backend']['implementation'])),\
                mock.patch('brass_ai.rust_backend.lab_backend.pin',side_effect=lambda path:state['native_backend']['rollback']):
            __import__('brass_ai.rust_backend.lab_backend',fromlist=['validate_state']).validate_state(state,raw,check_runtime=False)
        with self.assertRaisesRegex(ValueError,'already has training progress'):self.build()

    def test_the_restart_refuses_without_the_later_rating_archive(self):
        with self.assertRaisesRegex(ValueError,r'Rating slots \[1400\]'):self.build(rollback={})

    def test_a_later_archive_that_does_not_extend_this_one_is_refused(self):
        name=next(p for p in (self.later/'ratings').iterdir())
        data=read(name);data['games']=data['games'][1:]
        from brass_ai.recovery.ratings import ARCHIVE_FORMAT
        self.assertEqual(data['format'],ARCHIVE_FORMAT)
        write_json(name,data)
        manifest=read(self.later/'export-manifest.json')
        for f in manifest['files']:
            if f['path'].startswith('ratings/'):f['sha256']=sha(name)
        later=read(self.later/'state.json');later['ratings']['archive']['sha256']=sha(name)
        write_json(self.later/'state.json',later)
        next(f for f in manifest['files'] if f['path']=='state.json')['sha256']=sha(self.later/'state.json')
        write_json(self.later/'export-manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'not this state\'s archive'):self.build()

    def test_only_the_g1490_snapshot_seeds_the_learner(self):
        with self.assertRaisesRegex(ValueError,'retained snapshot of g1490'):
            self.build(weights=self.state['ratings']['agents']['generation-001480']['checkpoint']['path'])


if __name__=='__main__':unittest.main()
