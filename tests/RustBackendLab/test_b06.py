"""B06 acceptance, real replay/cache/preparation and verifier regression fixtures."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from brass_ai.learning.contracts import SOURCE_ROOT,digest
sys.path.insert(0,str(SOURCE_ROOT/'scripts'))
import rust_lab_b06_metrics as M
import rust_lab_b06_ordinary as O
import rust_lab_b06_teacher as T


def launcher():
    spec=importlib.util.spec_from_file_location('b06_tests',SOURCE_ROOT/'scripts/qualify-rust-lab-b06.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class B06(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def temporary(self):
        t=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(t.cleanup);return Path(t.name)

    def test_actual_plan_strata_not_retired_legacy_kind_and_no_redraw(self):
        from test_b02 import job
        jobs=[job(i) for i in range(5)]
        jobs[1]['provenance']['kind']='historical'
        jobs[2]['provenance']['kind']='historical'
        jobs[2]['policies'][2]=dict(kind='lab-legacy',policy_id='v4')
        before=copy.deepcopy(jobs)
        selected=O.freeze_jobs(dict(jobs=jobs))
        self.assertEqual([j['index'] for j in selected],[0,1,2]);self.assertEqual(jobs,before)
        for j in selected:self.assertEqual(j,before[j['index']])
        self.assertNotIn('branching',selected[0])
        self.assertEqual(M.choose_jobs([jobs[0]])[0],jobs[0])
        jobs[3]['provenance']['kind']='unsupported'
        with self.assertRaises(ValueError):M.choose_jobs(jobs)

    def test_distribution_exposes_small_samples_outliers_empty_and_nonfinite(self):
        self.assertEqual(M.distribution([1,2,100]),dict(samples=3,median=2.,p90=100.,maximum=100.))
        self.assertIsNone(M.distribution([])['median'])
        for values in ([float('nan')],[-1],[float('inf')]):
            with self.assertRaises(ValueError):M.distribution(values)

    def test_speedup_requires_equal_work_and_does_not_claim_iteration_gain(self):
        reference=dict(collection_seconds=2.,accepted_work_seconds=3.,work={'families':3})
        candidate=dict(collection_seconds=1.,accepted_work_seconds=4.,work={'families':3})
        report=M.performance(reference,candidate,True)
        self.assertEqual(report['collection_ratio_reference_over_rust'],2.)
        self.assertIsNone(report['whole_iteration_speedup'])
        self.assertFalse(report['useful_accepted_work_gain'])
        self.assertFalse(M.performance(candidate,reference,True)['useful_collection_gain'])
        with self.assertRaises(ValueError):M.performance(reference,dict(candidate,work={'families':2}),True)
        with self.assertRaises(ValueError):M.performance(reference,candidate,False)

    def test_complete_family_comparison_rejects_discrete_or_behavior_drift(self):
        from test_b03 import family_fixture
        from brass_ai.rust_backend.lab_admission import convert
        f,a,_=family_fixture(branch=True);g=convert(f,a,allow_diagnostic=True)
        same=copy.deepcopy(g);same['records'][0]['seconds']=10
        same['branches'][0]['branch']['elapsed_seconds']=99
        self.assertEqual(M.compare_games([g],[same])['work']['branches'],1)
        for key in ('chosen','behavior','choices'):
            bad=copy.deepcopy(g);row=bad['records'][0]
            if key=='chosen':row[key]=(row[key]+1)%len(row['choices'])
            elif key=='behavior':row[key]['probability']*=.5
            else:row[key]=list(reversed(row[key]))
            with self.subTest(key=key),self.assertRaises((ValueError,AssertionError)):M.compare_games([g],[bad])
        with self.assertRaises(ValueError):M.compare_games([g],[dict(g,branches=[])])

    def test_native_benchmark_real_seal_admission_warm_cache_and_target_pipeline(self):
        from test_b03 import family_fixture
        f,a,c=family_fixture(kind='historical');job=f['frozen_job'];output=self.temporary()
        service=Mock();service.channels.return_value.acquire.side_effect=lambda:Mock()
        execution=dict(cpu_budget=64,cpu_reserve=4,processes=24,hosts_per_process=2,model_cohorts=False)
        snapshot=dict(generation=10,model_id='a'*64,state=dict(signature=dict(config=c)))
        with M.Meter() as meter,patch.object(O,'Pipe'),patch.object(O,'collect_lane',return_value=f['parent']):
            games,report=O.native(SOURCE_ROOT,[job],service,execution,
                dict(collector={'path':'mock'},bootstrap={'binary':{'path':'mock'}}),output,'mono',meter,f['producer'],snapshot)
            self.assertEqual(report['scheduler']['counts']['rust'],1)
            self.assertEqual(M.work(games)['families'],1)
            rows=O.prepared_rows(games,c,job['provenance']['learner_behavior']['policy_id'])
            M.compare_rows(rows,copy.deepcopy(rows))
            bad=copy.deepcopy(rows);bad[0][3].state.setflags(write=True);bad[0][3].state[0]+=1
            with self.assertRaises(AssertionError):M.compare_rows(rows,bad)
            self.assertIn('admitted_replay_warm_read',meter.report()['phases'])

    def test_reference_benchmark_reads_real_actor_shard_and_prepares_identical_targets(self):
        from test_b03 import family_fixture
        from brass_ai.recovery.replay import write_shard,cache_receipt
        f,a,c=family_fixture(kind='legacy');job=f['frozen_job'];output=self.temporary();path=output/'actor.jsonl.gz'
        write_shard(path,[f['parent']['game']],job['provenance'])
        record=dict(path=str(path),features=cache_receipt(path))
        pool=Mock();pool.__enter__=Mock(return_value=pool);pool.__exit__=Mock(return_value=False)
        pool.collect.return_value=[record];pool.processes=[object()];pool.lanes=2;pool.last_collection_report={}
        with patch.object(O,'ActorPool',return_value=pool),patch('brass_ai.recovery.host_runtime.local_scoring_host',return_value=output),M.Meter() as meter:
            games,report=O.reference(SOURCE_ROOT,[job],Mock(),{},output,'mono',meter,{})
        self.assertEqual(report['lanes'],2)
        original=copy.deepcopy(job);actual=pool.collect.call_args.args[0][0]
        original.pop('path',None);actual.pop('path',None);self.assertEqual(original,actual)
        from brass_ai.rust_backend.lab_admission import convert
        candidate=convert(f,a)
        M.compare_games(games,[candidate])
        learner=job['provenance']['learner_behavior']['policy_id']
        M.compare_rows(O.prepared_rows(games,c,learner),O.prepared_rows([candidate],c,learner))

    def test_native_initiating_failure_survives_sibling_cancellation(self):
        from test_b03 import family_fixture
        f,_,_=family_fixture();job=f['frozen_job'];output=self.temporary()
        service=Mock();service.channels.return_value.acquire.side_effect=lambda:Mock()
        execution=dict(cpu_budget=64,cpu_reserve=4,processes=24,hosts_per_process=2)
        def cancelled(jobs,store,native,*args,**kwargs):
            try:native(job,None,Mock())
            except ValueError:pass
            raise ValueError('secondary cancellation')
        with M.Meter() as meter,patch.object(O,'Pipe'),patch.object(O,'collect_lane',side_effect=ValueError('initiating failure')),patch.object(O,'collect_families',side_effect=cancelled):
            with self.assertRaisesRegex(ValueError,'initiating failure'):
                O.native(SOURCE_ROOT,[job],service,execution,dict(collector={'path':'mock'},bootstrap={'binary':{'path':'mock'}}),
                         output,'mono',meter,f['producer'],{})

    def test_current_branch_window_preserves_policy_identity_and_validates_real_probabilities(self):
        from test_b03 import family_fixture
        from brass_ai.recovery.exploration import FORK_T15_BRANCH,THREE_OPENING,FORK_OPENING
        from brass_ai.rust_backend.lab_collection import branch_opening
        from brass_ai.rust_backend.lab_families import validate_family,validate_part
        from brass_ai.rust_backend.lab_admission import convert
        f,a,c=family_fixture(branch=True,branching=FORK_T15_BRANCH)
        job=f['frozen_job'];before=copy.deepcopy(job)
        self.assertEqual(branch_opening(job,THREE_OPENING),FORK_OPENING)
        validate_family(f,job,f['producer']);game=convert(f,a,allow_diagnostic=True)
        self.assertEqual(job,before)
        rows=game['branches'][0]['records'][1:]
        for row in rows:
            if row['view']['context']['phase']=='action' and row['behavior']['action_decision']<2:
                self.assertEqual(row['behavior']['temperature'],1.5)
                self.assertEqual(row['behavior']['policy_id'],job['policies'][row['seat']]['policy_id'])
        bad=copy.deepcopy(f['child']);bad['game']['records'][1]['behavior']['temperature']=1.3
        with self.assertRaises(ValueError):validate_part(bad,job,child=True)
        badjob=copy.deepcopy(job);badjob['branching']['continuation_opening']['temperature']=1.7
        with self.assertRaises(ValueError):branch_opening(badjob,THREE_OPENING)
        with self.assertRaises(ValueError):branch_opening(job,FORK_OPENING)

    def test_native_branch_lane_installs_continuation_window_before_sampling(self):
        from test_b03 import family_fixture
        from brass_ai.recovery.exploration import FORK_T15_BRANCH,FORK_OPENING
        from brass_ai.rust_backend import lab_collection as C
        from brass_ai.step5.policy import NeuralPolicy
        from brass_ai.recovery.execution import neural_spec
        f,_,_=family_fixture(branch=True,branching=FORK_T15_BRANCH);job=f['frozen_job']
        row=copy.deepcopy(f['parent']['game']['records'][16]);meta=f['child']['game']['branch']
        packet=dict(metadata=meta,job_id=C.frozen_id(job),row=row,contexts=[([0.]*4,[0.]*8)]*4,
                    counts=[5]*4,checkpoint={'payload':{'state':{'cards':{'deck':[]}}}})
        packet['sha256']=digest(packet)
        lane=Mock();lane.__enter__=Mock(return_value=lane);lane.__exit__=Mock(return_value=False)
        lane.current=dict(decision=dict(view=row['view'],choices=row['choices']),step=1,terminal=False)
        policies=[]
        def make(spec,*_):
            p=NeuralPolicy(None,evaluator=Mock(model_id=spec['model_id']),focus=spec['focus'],exploration=spec['exploration'],
                           exploration_version=3,opening=copy.deepcopy(spec['opening']))
            def sample(_):
                self.assertEqual(p.opening,FORK_OPENING);self.assertEqual(p.policy_id,spec['policy_id'])
                raise RuntimeError('sampling reached with current opening')
            p.select_batch=sample;policies.append(p);return p
        with patch.object(C,'Collector',return_value=lane),patch.object(C,'DecisionEncoder'),patch.object(C,'make_policy',side_effect=make):
            with self.assertRaisesRegex(RuntimeError,'sampling reached'):
                C.collect_lane(SOURCE_ROOT,'mock',Mock(),job,Mock(),{'max_decisions':100},continuation=packet)

    def test_exact_g1250_source_route_preserves_config_restrictions(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        state=json.loads((SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g1250-20261107T041141451177Z/state.json').read_text())
        target=copy.deepcopy(state['signature']);target['implementation']=implementation_id()
        self.assertTrue(check_signature(state,target,reviewed=True))
        target['config']['learning_rate']*=1.123
        with self.assertRaises(ValueError):check_signature(state,target,reviewed=True)

    def stream_fixture(self):
        from test_b025 import model_fixture,example
        from test_b045 import value_fixture
        model=model_fixture().eval();_,job,_=value_fixture(model)
        row,_=example(model,job=job,round=job['reservations']['0']['canal_early'])
        table=dict(job_id=job['expected']['job_id'],rows=[row.row],metrics=dict(searched_turns=64,reserved_turns=16,
            turns=[dict(index=i,profile='full',native_post_ns=10_000_000,search_ns=4_000_000_000) for i in range(1,65)],
            verification=[dict(index=i,rows=int(i==1),allowance_seconds=.95) for i in range(1,65)]))
        return model,job,table

    def test_stream_and_batch_use_identical_frozen_rows_and_keep_identity_checks(self):
        from brass_ai.rust_backend.lab_admission import admit_teacher_rows
        from brass_ai.learning.batching import model_id
        model,job,table=self.stream_fixture();history=[dict(generation=10,model_id=model_id(model))]
        batch,rejected=admit_teacher_rows(table['rows'],job=job,model=model,history=history)
        stream,dropped,seconds=T.replay_stream(model,history,job,table)
        self.assertEqual([r.row for r in batch],[r.row for r in stream]);self.assertEqual(rejected,dropped)
        self.assertGreater(seconds,0)
        for change in ('missing','duplicate','epoch','target','late'):
            bad=copy.deepcopy(table)
            if change=='missing':bad['metrics']['verification'].pop()
            elif change=='duplicate':bad['metrics']['turns'][1]['index']=1
            elif change=='epoch':bad['rows'][0]['teacher']['epoch']+=1
            elif change=='target':bad['rows'][0]['teacher']['target'][0]+=1
            else:bad['metrics']['verification'][0]['allowance_seconds']=-1
            with self.subTest(change=change),self.assertRaises(ValueError):T.replay_stream(model,history,job,bad)

    def test_probe_schedule_balances_both_arms_without_shared_sessions(self):
        self.assertEqual(T.probe_schedule(),[('policy_only',0),('all_seat_w0',0),('all_seat_w0',1),('policy_only',1)])

    def test_probe_schema_does_not_invent_missing_table_counters(self):
        turn={k:1 for k in ('search_ns','native_post_ns','post_search_ns','selection_ns','labels_ns','commit_ns','verification_ipc_ns','elapsed_ns')}
        probe=dict(metrics=dict(labels=1,turns=[turn],searches=[dict(completed=8,rounds=1,effective_k=16)]),
                   inference_requests=10,gpu_wait_ns=2e9,input_bytes=64)
        report=T.summarize_turns([probe]);self.assertEqual(report['inference_requests'],10)
        self.assertEqual(report['gpu_wait_seconds'],2);self.assertIsNone(report['encoding_seconds'])

    def test_cli_child_flags_and_structured_early_error(self):
        m=launcher();args=m.arguments([]);out=self.temporary();args.run=out;args.out=out;args.phase='ordinary'
        child=m.arguments(m.child_command(args,'teacher',out)[2:])
        self.assertEqual((child.device,child.phase,child.mono),('cuda:0','teacher','mono'))
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):m.arguments(['--device','cuda:1'])
        with patch.object(m.platform,'system',return_value='Windows'):
            with self.assertRaisesRegex(ValueError,'Linux/CUDA'):m.run_child(args,{})
        report=json.loads((out/'ordinary-checks.json').read_text())
        self.assertFalse(report['passed']);self.assertIn('Linux/CUDA',report['error'])

    def test_meter_samples_owned_process_and_keeps_nested_phase_scope(self):
        with M.Meter() as meter:
            with meter.phase('outer'):
                with meter.phase('inner'):sum(range(100))
            report=meter.report()
        self.assertGreater(report['peak_process_tree_rss_bytes'],0)
        self.assertGreaterEqual(report['phases']['outer']['total_seconds'],report['phases']['inner']['total_seconds'])

    def test_measurement_dependency_is_bundled_pinned_and_checks_native_import(self):
        requirements=(SOURCE_ROOT/M.REQUIREMENTS).read_text().splitlines()
        self.assertIn('psutil=='+M.PSUTIL_VERSION,requirements)
        for error in (ModuleNotFoundError('psutil'),ImportError('broken extension'),OSError('missing shared library')):
            with self.subTest(error=error),patch.object(M.importlib,'import_module',side_effect=error):
                with self.assertRaisesRegex(RuntimeError,'python -m pip install -r'):
                    M.measurement_preflight()
        with patch.object(M.importlib,'import_module',return_value=SimpleNamespace(__version__='0.0')):
            with self.assertRaisesRegex(ValueError,'found 0.0'):M.measurement_preflight()
        self.assertEqual(M.measurement_preflight()['psutil'],M.PSUTIL_VERSION)

    def test_missing_dependency_exports_actionable_error_before_fixtures_or_run_lock(self):
        m=launcher();written={};out=[]
        def write(path,value):written[Path(path).name]=copy.deepcopy(value)
        def export(path):out.append(path);return path
        with patch.object(m,'verify_bundle',return_value={'implementation':'test'}), \
             patch.object(m,'arguments',return_value=m.arguments(['--fixtures-only'])), \
             patch('brass_ai.learning.contracts.read_json',return_value={'prerequisites':{}}), \
             patch('brass_ai.step5.league.write_json',side_effect=write), \
             patch('rust_lab_results.export',side_effect=export), \
             patch('brass_ai.recovery.retention.run_lock') as lock, \
             patch('subprocess.Popen') as child,patch.dict(sys.modules,{'psutil':None}), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'same Python environment'):m.main()
        self.assertEqual(len(out),1);self.addCleanup(out[0].rmdir)
        child.assert_not_called();lock.assert_not_called()
        report=written['qualification.json']
        self.assertEqual(report['failed_phase'],'environment')
        self.assertFalse(report['passed']);self.assertFalse(report['production_modified'])
        self.assertIn(M.REQUIREMENTS,report['error'])
        self.assertNotIn('fixtures.log',written)

    @staticmethod
    def process(pid,cpu):
        process=Mock(pid=pid)
        process.children.return_value=[];process.create_time.return_value=float(pid)
        process.cpu_times.return_value=SimpleNamespace(user=cpu,system=0.)
        process.memory_info.return_value=SimpleNamespace(rss=1024)
        return process

    def test_meter_baselines_already_running_children(self):
        import psutil
        parent=self.process(1,10.);child=self.process(2,25.);parent.children.return_value=[child]
        with patch.object(psutil,'Process',return_value=parent),M.Meter() as meter:
            report=meter.report()
        self.assertEqual(report['sampled_process_tree_cpu_seconds'],0.)
        self.assertEqual(report['peak_process_tree_rss_bytes'],2048)

    def test_meter_tolerates_exited_child_but_rejects_inaccessible_counters(self):
        import psutil
        parent=self.process(1,10.);child=self.process(2,1.);parent.children.return_value=[child]
        child.cpu_times.side_effect=psutil.NoSuchProcess(child.pid)
        with patch.object(psutil,'Process',return_value=parent),M.Meter() as meter:
            report=meter.report()
        self.assertGreater(report['exited_process_reads'],0)
        self.assertEqual(report['peak_process_tree_rss_bytes'],1024)
        child.cpu_times.side_effect=psutil.AccessDenied(child.pid)
        with patch.object(psutil,'Process',return_value=parent),patch.object(M.threading,'Thread') as thread:
            with self.assertRaises(psutil.AccessDenied):M.Meter().__enter__()
            thread.assert_not_called()

    def test_meter_background_failure_invalidates_report_and_keeps_primary_error(self):
        import psutil
        for primary in (False,True):
            parent=self.process(1,10.);parent.children.side_effect=[[],RuntimeError('sampler regression')]
            meter=M.Meter()
            with self.subTest(primary=primary),patch.object(psutil,'Process',return_value=parent):
                with self.assertRaisesRegex(RuntimeError,'original collection error' if primary else 'measurements are incomplete') as caught:
                    with meter:
                        self.assertTrue(meter.stop.wait(2),'Sampler failure was not signaled')
                        with self.assertRaisesRegex(RuntimeError,'sampler regression'):meter.report()
                        if primary:raise RuntimeError('original collection error')
                self.assertFalse(meter.thread.is_alive())
                if primary:self.assertIn('sampler regression',caught.exception.__notes__[0])

    def test_meter_cuda_initialization_failure_does_not_leak_sampler(self):
        with patch.object(torch.cuda,'reset_peak_memory_stats',side_effect=RuntimeError('CUDA unavailable')), \
             patch.object(M.threading,'Thread') as thread:
            with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'):M.Meter('cuda:0').__enter__()
            thread.assert_not_called()


if __name__=='__main__':unittest.main()
