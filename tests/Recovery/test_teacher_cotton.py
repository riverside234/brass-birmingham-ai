"""Short cotton admission, recovery, recipe and continuation regressions. No games/benchmarks."""
import copy
import importlib.util
import unittest
from unittest.mock import Mock, patch

from brass_ai.learning.contracts import SOURCE_ROOT as ROOT, digest
from brass_ai.recovery import teacher_cotton as C, training_revision_g1491 as R, teacher_labels as L


class Recipe(unittest.TestCase):
    def test_training_jobs_only_and_old_recipe_remains_loadable(self):
        from test_training_revision_g1491 import committed
        from brass_ai.recovery.teacher_phase import plan_jobs
        from brass_ai.recovery.config import validate_config
        state,before=committed();after=R.configure(before,state)
        jobs,_=plan_jobs(after,state,model_id='m'*64,export_hash='e'*64,epoch=1491,ready=True)
        for job in jobs:
            self.assertEqual(C.enabled(job),job['split']=='train')
            R.validate_job(job)
        self.assertEqual(after['reward'],before['reward'])
        self.assertEqual(after['teacher_phase']['search'],before['teacher_phase']['search'])
        self.assertEqual(after['teacher_phase']['fast_search'],before['teacher_phase']['fast_search'])
        old=copy.deepcopy(after);old['teacher_phase'][R.KEY]=R.block(cotton=False)
        validate_config(old)
        self.assertEqual(R.configure(old,state),old)
        self.assertNotEqual(digest(R.job_block(old)),digest(R.job_block(after)))
        bad=copy.deepcopy(jobs[0]);bad['split']='heldout'
        with self.assertRaises(ValueError):R.validate_job(bad)


class Labels(unittest.TestCase):
    def example(self, *, targeted=True, choice='c'):
        from test_b025 import example, model_fixture
        row,_=example(model_fixture())
        label=copy.deepcopy(row.label)
        choices=[dict(id='a',kind='loan'),dict(id='c',kind='build',industry='CottonMill',target='X')]
        label.update(legal_ids=['a','c'],logits=[0.,-5.],counts=[90,10],mean_returns=[.2,.1],
            node_value=.15,c1=50,c2=.1,max_n=90,played_action=choice,explored=True,
            completed_trajectories=100,node_trajectories=100,frozen_n=128,effective_k=2)
        target,mixed=L.improved_policy(label['logits'],label['counts'],label['mean_returns'],label['node_value'])
        label.update(target=target.tolist(),v_mix=mixed)
        label['cotton_exploration']=dict(contract=C.CONTRACT,targeted=targeted,probability=.28 if choice=='c' else .72,
            used_before=0,cotton_visits=10,total_visits=100,covered_locations=1)
        return label,choices

    def test_behavior_probability_and_improved_target_are_independent(self):
        label,choices=self.example()
        original=copy.deepcopy(label);original.pop('cotton_exploration')
        self.assertEqual(L.validate_improved_policy(label,choices).tolist(),L.validate_improved_policy(original,choices).tolist())
        self.assertLess(label['target'][1],.01) # targeted move is NOT the teacher's preferred action
        for choice in ('a','c'):
            current,choices=self.example(targeted=False,choice=choice)
            L.validate_improved_policy(current,choices)

    def test_bad_probability_counts_cap_and_played_move_are_rejected(self):
        for key,value in [('probability',float('nan')),('probability',.5),('used_before',3),('targeted',1),
                          ('cotton_visits',9),('covered_locations',2),('total_visits',101)]:
            label,choices=self.example();label['cotton_exploration'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):L.validate_improved_policy(label,choices)
        label,choices=self.example(targeted=True,choice='a')
        with self.assertRaises(ValueError):L.validate_improved_policy(label,choices)

    def test_table_cap_sequence_and_committed_counts(self):
        label,_=self.example()
        def row(used):
            l=copy.deepcopy(label);l['seat']=0;l['cotton_exploration']['used_before']=used
            return dict(teacher=l,view=dict(context=dict(era='CanalEra')))
        rows=[row(i) for i in range(3)]
        counts=dict(triggers=3,targeted_builds=3,actual_builds=4,covered_decisions=3,eligible_decisions=4,
            legal_decisions=5,action_decisions=6,no_evidence=0,no_cotton_visits=1)
        receipt=dict(contract=C.CONTRACT,long=counts,short={k:0 for k in counts},
            targeted_by_era_seat=[[3,0,0,0],[0]*4],builds_by_era_seat=[[4,0,0,0],[0]*4])
        job=dict(teacher_revision=R._job_block(R.block()['exploration']),split='train')
        C.validate_receipt(receipt,job,rows)
        bad=copy.deepcopy(rows);bad[1]['teacher']['cotton_exploration']['used_before']=0
        with self.assertRaises(ValueError):C.validate_receipt(receipt,job,bad)
        with self.assertRaises(ValueError):C.validate_receipt(receipt,{},rows)
        stats=C.aggregate([dict(cpu_inference=dict(revision=dict(cotton=receipt))),{}])
        self.assertEqual(stats['natural_builds'],1)
        self.assertEqual(stats['coverage_rate'],.75)


class Continuation(unittest.TestCase):
    def module(self):
        spec=importlib.util.spec_from_file_location('cotton_continue',ROOT/'scripts/continue-rust-g1491.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def test_requested_command_and_fresh_vs_stale_worker(self):
        m=self.module()
        args=m.shared().parser(default_mode='on',default_stage_revision=True,teacher_fast=128,inference='cpu',run=m.RUN).parse_args(
            ['--expected-generation','1491','--iterations','100','--share-target','0.4'])
        self.assertEqual((args.expected_generation,args.iterations,args.share_target),(1491,100,.4))
        state=dict(iteration=1491)
        with patch('brass_ai.rust_backend.lab_shadow.teacher_build',return_value=dict(core_source='core')),\
             patch('brass_ai.rust_backend.lab_cpu_runtime.resolve') as resolve,patch.object(m.subprocess,'run') as run:
            m.ensure_cpu(ROOT/'artifacts/example',state);run.assert_not_called()
            resolve.side_effect=[ValueError('stale worker'),{}]
            m.ensure_cpu(ROOT/'artifacts/example',state)
            command=run.call_args.args[0]
            self.assertIn('--focused-regressions',command)
            self.assertNotIn('--skip-regressions',command)

    def test_hundred_iteration_command_adopts_cotton_in_shadow_then_resumes_teaching(self):
        from test_training_revision_g1491 import committed
        m=self.module();shared=m.shared();state,before=committed();calls=[]
        def continue_(args,**kw):
            calls.append(kw)
            if kw['mode']=='shadow':
                config=R.configure(before,state)
                self.assertEqual(R.job_block(config)['exploration']['cotton'],C.settings())
                return dict(state,iteration=1493,signature=dict(state['signature'],config=config),
                    teacher_phase=dict(state['teacher_phase'],weight=dict(gradient_qualification=dict(passed=True))))
            return state
        shared.continuation=continue_
        with patch.object(m,'shared',return_value=shared),patch.object(m,'state_of',return_value=state),\
             patch.object(m,'interrupted',return_value=False),patch.object(m,'ensure_cpu') as prepare,\
             patch('sys.argv',['continue-rust-g1491.py','--expected-generation','1491','--iterations','100','--share-target','0.4']):
            m.main()
        prepare.assert_called_once()
        self.assertEqual([(c['mode'],c['expected'],c['iterations']) for c in calls],[('shadow',1491,2),('on',1493,98)])


class Runtime(unittest.TestCase):
    def test_teacher_plan_binds_the_derived_core_without_relabelling_the_gpu_core(self):
        from brass_ai.rust_backend.lab_cpu_runtime import teacher_plan
        value=dict(teacher=dict(path='worker'),teacher_execution_core='derived',receipt='pin',runtime='runtime')
        settings=dict(bundles={'model':'bundle'},runtime='runtime',exported_bundles=[])
        with patch('brass_ai.rust_backend.lab_cpu_runtime.resolve',return_value=value) as resolve,\
             patch('brass_ai.rust_backend.lab_cpu_runtime.worker_settings',return_value=settings):
            plan=teacher_plan(ROOT,dict(core_source='sealed'),None,[dict(expected=dict(model_id='model'))],
                              dict(teacher_budget=dict(mode='count'),pinning='none'))
        self.assertEqual(plan['core_source'],'derived')
        resolve.assert_called_once_with(ROOT,teacher_core='sealed')


if __name__=='__main__':unittest.main()
