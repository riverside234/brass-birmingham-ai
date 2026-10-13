"""Historical diversity, bounded two-collection scheduling and continuation."""
import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery.config import SEARCH_EVERY_GAMES,SEARCH_WALL_FRACTION,validate_config
from brass_ai.recovery.population import retain,sample_opponents,rollout_opponents
from brass_ai.recovery.teacher_collection import collect_due,due_collections
from brass_ai.recovery.teacher_schedule import CONTRACT,PHASES,counters,search_allowance
from brass_ai.recovery.resume import check_signature,commit_transition
from brass_ai.step5.league import implementation_id,write_json


def entry(generation):
    return dict(generation=generation,model_id=digest(generation),path=f'artifacts/mock/generation-{generation}.pt',sha256='mock')


class HistoricalPopulationTests(unittest.TestCase):
    def test_streaming_retention_matches_full_history_and_retains_middle_generations(self):
        cfg=dict(recent=16,anchors=48);full=[];stream=[]
        for i in range(1001):
            full.append(entry(i));stream=retain(stream+[full[-1]],cfg)
            self.assertEqual(stream,retain(full,cfg))
            self.assertLessEqual(len(stream),64)
        generations=[e['generation'] for e in stream]
        self.assertEqual(generations[0],0)
        self.assertEqual(generations[-16:],list(range(985,1001)))
        self.assertGreater(sum(100<=g<=700 for g in generations),15)
        self.assertEqual(stream,retain(list(reversed(full))+[full[-1]],cfg))

    def test_seeded_sampling_includes_old_il_and_recent_without_current(self):
        entries=[entry(i) for i in range(81)];cfg=dict(recent=16,anchors=48)
        pool=retain(entries,cfg);current=entries[-1]['model_id']
        a=sample_opponents(pool,current,entries[0],random.Random(97),16,6000)
        b=sample_opponents(pool,current,entries[0],random.Random(97),16,6000)
        self.assertEqual(a,b);self.assertNotIn(current,{e['model_id'] for e in a})
        self.assertTrue(1300<sum(e['generation']==0 for e in a)<1700)
        self.assertTrue(2600<sum(0<e['generation']<64 for e in a)<3400)
        self.assertEqual(sample_opponents([entries[0]],entries[0]['model_id'],entries[0],random.Random(0),16),[entries[0]]*3)
        rollouts=rollout_opponents(pool,current,entries[0],random.Random(97),16)
        self.assertEqual(rollouts[0]['generation'],0);self.assertLessEqual(len(rollouts),3)
        self.assertTrue(any(1<e['generation']<64 for e in rollouts))


class TeacherCollectionTests(unittest.TestCase):
    def fixture(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        state=dict(next_search_games=800,search_seconds=10.,population=[],history=[],evaluation_anchor=entry(0))
        service=SimpleNamespace(register=lambda model:'learner',models={'learner':None})
        jobs_seen=[]
        def collect(jobs):
            jobs_seen.extend(jobs)
            return [dict(path=j['path'],game_id=str(j['seed'])) for j in jobs]
        heavy=SimpleNamespace(collect=collect)
        teaching=dict(contract=CONTRACT,phases={p:dict(counters(),scheduled_turns=20,qualified_samples=20)
                                               for p,_,_,_ in PHASES})
        update=dict(updated=True,optimizer_steps=8,teacher_rows=80,teacher_setups=5,sample_presentations=128,
            requested_steps=8,max_observed_reuse=2,samples_by_phase={p:20 for p,_,_,_ in PHASES})
        return cfg,state,service,heavy,teaching,update,jobs_seen

    def run_fixture(self,ordinary=10000.,failed=False,fraction=None):
        cfg,state,service,heavy,teaching,update,jobs=self.fixture()
        cfg['search_wall_fraction']=fraction
        if failed:update.update(updated=False,optimizer_steps=0,teacher_rows=0)
        # Avoid starting the engine: these tests exercise the actual collection
        # coordinator's fresh jobs, budget accounting and per-batch update cap.
        with patch('brass_ai.recovery.teacher_collection.rollout_opponents',return_value=[]), \
             patch('brass_ai.recovery.teacher_collection.read_shards',return_value=[]), \
             patch('brass_ai.recovery.teacher_collection.collection_report',return_value=teaching), \
             patch('brass_ai.recovery.teacher_collection.distill_update',side_effect=lambda *a,**kw:copy.deepcopy(update)) as train, \
             patch('brass_ai.recovery.teacher_collection.time.monotonic',side_effect=[0.,200.,200.,400.]):
            result=collect_due(SOURCE_ROOT,SOURCE_ROOT/'artifacts/mock',SOURCE_ROOT/'artifacts/mock/iteration-0070',
                cfg,state,{},SimpleNamespace(config=SimpleNamespace(utility=cfg['search']['reward']),utility_calibration=dict(ready=True,reward_id=digest(cfg['search']['reward']))),None,service,heavy,done=960,cursor=18000000,
                ordinary_seconds=ordinary,ordinary_game_seconds=1.)
            calls=list(train.call_args_list)
        return result,jobs,calls

    def test_two_due_collections_have_distinct_seeds_paths_and_eight_steps_each(self):
        (report,next_due,cursor,spent),jobs,calls=self.run_fixture()
        self.assertEqual((report['collection_count'],next_due,spent),(2,1120,410.))
        self.assertEqual((len(jobs),report['teacher_rows'],report['optimizer_steps']),(10,160,16))
        for key in ('path','seed','sampling_seed','index'):self.assertEqual(len({j[key] for j in jobs}),10)
        self.assertEqual([c.kwargs['steps'] for c in calls],[8,8])
        self.assertEqual(report['teaching']['phases']['rail_late']['scheduled_turns'],40)
        self.assertEqual(report['max_observed_reuse'],2)

    def test_budget_rechecked_before_second_collection_and_failure_never_retries(self):
        (report,next_due,_,spent),jobs,calls=self.run_fixture(ordinary=1350.,fraction=.5)
        self.assertEqual((len(jobs),next_due,spent),(5,960,210.))
        self.assertTrue(report['budget']['deferred'])
        (report,next_due,_,_),jobs,calls=self.run_fixture(failed=True)
        self.assertEqual((len(jobs),len(calls),next_due),(10,2,1120))
        self.assertFalse(report['updated']);self.assertEqual(report['optimizer_steps'],0)

    def test_warmup_no_backlog_burst_disabled_and_budget(self):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        self.assertEqual((due_collections(320,640,cfg),due_collections(640,640,cfg),due_collections(960,800,cfg)),(0,1,2))
        self.assertEqual(due_collections(50000,640,cfg),2)
        self.assertEqual(due_collections(50000,640,dict(cfg,search_every_games=0)),0)
        self.assertEqual((SEARCH_EVERY_GAMES,SEARCH_WALL_FRACTION),(160,None))
        self.assertAlmostEqual(search_allowance(100,0,.5),100.)
        self.assertAlmostEqual(search_allowance(100,300,.5),0.)
        self.assertIsNone(search_allowance(100,1000))
        for p in (SOURCE_ROOT/'configs/step5').glob('lab*.json'):
            c=validate_config(read(p))
            self.assertEqual(c['population'],dict(recent=16,anchors=48))
            self.assertEqual((c['audit_workers'],c['execution']['max_models']),(16,72))
            self.assertEqual((c['search_every_games'],c['search_collections_per_iteration']),(160,2))
            self.assertEqual((c['search']['seconds'],c['distill_steps'],c['search_games']),(12.,8,5))


class Review70ResumeTests(unittest.TestCase):
    def test_reviewed_config_upgrade_keeps_checkpoint_adam_and_repeat_resume(self):
        import torch
        from test_reviewed_resume import ReviewedResumeTests
        from brass_ai.recovery.resume import load_resume
        from brass_ai.learning.checkpoint import resume_optimizer
        ReviewedResumeTests.setUpClass();fixture=ReviewedResumeTests()
        try:
            run,state,_,original,optimizer=fixture.fixture()
            previous=state['signature'];cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
            cfg.update(model_size='small',rehearsal_manifest=previous['imitation_source']['manifest'])
            signature=dict(previous,config=cfg,implementation=implementation_id())
            before=(run/'generation-0002.pt').read_bytes()
            model,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=True)
            self.assertTrue(changed)
            restored=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);resume_optimizer(raw,restored)
            self.assertEqual(restored.param_groups[0]['lr'],1e-5)
            self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
            for key,value in original.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
            for key,values in optimizer.state_dict()['state'].items():
                for name,value in values.items():torch.testing.assert_close(value,restored.state_dict()['state'][key][name],rtol=0,atol=0)
            updated=commit_transition(run,state,signature)
            self.assertEqual(updated['next_search_games'],800)
            self.assertEqual((run/'generation-0002.pt').read_bytes(),before)
            self.assertFalse(load_resume(SOURCE_ROOT,run,updated,signature)[2])
        finally:fixture.doCleanups()

    def test_exact_reviewed_signature_resumes_generation70_without_reset(self):
        state=read(SOURCE_ROOT/'results/selfplay-v2-70/state.json')
        old=state['signature'];cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
        cfg['rehearsal_manifest']=old['imitation_source']['manifest']
        signature=dict(old,config=cfg,implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,True))
        self.assertEqual((state['iteration'],state['history'][-1]['generation']),(70,70))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);write_json(run/'state.json',state)
            updated=commit_transition(run,state,signature)
            self.assertEqual(updated['next_search_games'],22560)
            for field in ('iteration','games_done','next_seed','history','population','context_counter','evaluations'):
                self.assertEqual(updated[field],state[field])
        altered=copy.deepcopy(signature);altered['config']['learning_rate']=2e-5
        with self.assertRaisesRegex(ValueError,'unchanged'):check_signature(state,altered,True)


if __name__=='__main__':unittest.main()
