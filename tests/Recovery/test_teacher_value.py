"""Current feature-layout fixtures for teacher outcome value learning."""
import copy
from collections import Counter
import json
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.model import ModelConfig
from brass_ai.learning.replay import terminal_targets
from brass_ai.learning.reward import replay_returns,specification,terminal_utility
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.recovery.config import TEACHER_VALUE,UPDATE_V21,teacher_value_config,validate_config,latest_model_config
from brass_ai.recovery.execution import _collection_recipe
from brass_ai.recovery.replay import check_game,write_shard
from brass_ai.recovery.teacher_schedule import CONTRACT,PHASES
from brass_ai.recovery.teacher_value import TeacherValue,admit_collection,provenance
from brass_ai.recovery.updates import Dataset,prepared,distill_update
from brass_ai.recovery.utility import calibrate,calibration_ready
from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import visit_teacher
from brass_ai.step5.league import seeds_for
from fe_fixtures import Factory
from fixtures import game,rotate


def equal_tree(test,left,right):
    if isinstance(left,torch.Tensor):test.assertTrue(torch.equal(left,right))
    elif isinstance(left,dict):
        test.assertEqual(set(left),set(right))
        for k in left:equal_tree(test,left[k],right[k])
    elif isinstance(left,(list,tuple)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):equal_tree(test,a,b)
    else:test.assertEqual(left,right)


class TeacherValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);factory=Factory()
        try:cls.view,cls.choices=factory.position()
        finally:factory.close()

    def fixture(self,*,games=2,diagnostics=True,model_size='small',dense_labels=False):
        torch.manual_seed(1923)
        reward=specification();model=UtilityNet(utility_config(ModelConfig.preset(model_size),reward))
        model.utility_calibration=dict(ready=True,reward_id=digest(reward),scope='authored fixture')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.0001)
        opts=dict(TEACHER_VALUE,enabled=True,diagnostics=diagnostics,samples_per_game=12)
        cfg=dict(UPDATE_V21,effective_batch_size=16,microbatch_size=8,critic_batch_size=16)
        quality=dict(min_samples=8,min_games=2,min_samples_per_phase=2,batch_size=16,max_reuse=2)
        identity=model_id(model);expected=provenance(identity,reward,opts);episodes=[];bindings={}
        for n in range(games+1):
            episode=game();seed=seeds_for(85000+n*100,1)[0][0]
            episode.update(game_id=f'teacher-value-fixture-{n}',
                setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed)),split='train')
            records=[]
            for phase,era,first,_ in PHASES:
                for seat in range(4):
                    row=copy.deepcopy(episode['records'][seat]);i=len(records)
                    row.update(step=i,view=rotate(self.view,(seat-self.view['seat'])%4),choices=copy.deepcopy(self.choices[:3]))
                    row['view']['context'].update(revision=i,era=era,round=first)
                    for player in row['view']['players']:player['hidden_discards']=int(era=='CanalEra')
                    if seat==2 and not dense_labels:row['choices']=row['choices'][:1]
                    if n==games:row['behavior']['kind']='model'
                    if (seat==0 or dense_labels) and n<games:
                        stats=dict(root={c['id']:dict(visits=v) for c,v in zip(row['choices'],[9,2,1])},expansion=[True]*3)
                        row['teacher']=visit_teacher(row['choices'],stats,[.7,.2,.1],model=identity,
                            observer=seat,config=SearchConfig(contract='masked-ismcts/1.2.0',reward=reward))
                    records.append(row)
            episode['records']=records
            episodes.append(check_game(episode))
            bindings[episode['game_id']]=dict(expected,game_digest=digest(episode))
        ordinary=prepared(episodes[-1:],insolvency=reward['insolvency'],reward=reward)
        return model,optimizer,episodes[:-1],ordinary,opts,cfg,quality,bindings

    def auxiliary(self,fixture):
        model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        return TeacherValue(model,games,ordinary,opts,cfg,bindings,ordinary_policy_id='fixture')

    def test_enabled_fresh_default_strict_bounds_and_unrelated_recipe_unchanged(self):
        cfg=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');self.assertEqual(cfg['teacher_value'],TEACHER_VALUE)
        validate_config(cfg);self.assertTrue(TEACHER_VALUE['enabled']);self.assertFalse(TEACHER_VALUE['diagnostics'])
        self.assertEqual((cfg['search_games'],cfg['distill_steps'],cfg['search_every_games']),(5,8,160))
        self.assertEqual(list(cfg['teacher_work_budget']['seconds_per_phase'].values()),[9.,12.,12.,12.])
        self.assertIsNone(cfg['search_wall_fraction'])
        for k,v in (('probe_positions',129),('samples_per_game',0),('max_samples',129),('max_reuse',3),
                    ('teacher_fraction',1.),('loss_weight',float('nan')),('enabled',1)):
            with self.subTest(k=k),self.assertRaises(ValueError):teacher_value_config(dict(TEACHER_VALUE,**{k:v}))

    def test_fresh_generation_enables_value_but_preserves_explicit_opt_out_and_legacy_omission(self):
        base=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');base.pop('teacher_value')
        before=copy.deepcopy(base)
        generated=latest_model_config(base,base['imitation_checkpoint'])
        self.assertEqual(generated['teacher_value'],TEACHER_VALUE)
        self.assertTrue(generated['teacher_value']['enabled'])
        self.assertEqual(base,before)
        # A stored legacy signature is not silently reinterpreted on resume.
        self.assertFalse(teacher_value_config(base.get('teacher_value'))['enabled'])
        generated['teacher_value']['enabled']=False
        self.assertFalse(latest_model_config(generated,base['imitation_checkpoint'])['teacher_value']['enabled'])

    def test_value_eligibility_is_separate_and_seats_targets_and_rng_are_preserved(self):
        fixture=self.fixture();model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        torch_before=torch.get_rng_state().clone();python_before=random.getstate();numpy_before=np.random.get_state()
        with patch('brass_ai.recovery.teacher_value.replay_returns',wraps=replay_returns) as returns:
            aux=self.auxiliary(fixture)
        self.assertEqual(returns.call_count,2)
        self.assertTrue(torch.equal(torch_before,torch.get_rng_state()));self.assertEqual(python_before,random.getstate())
        np.testing.assert_array_equal(numpy_before[1],np.random.get_state()[1])
        self.assertEqual(aux.report()['valid_value_positions'],32);self.assertEqual(len(aux.teacher.data),24)
        self.assertTrue(any(d[0]['teacher'] is None for d in aux.teacher.data))
        self.assertTrue(any(len(d[0]['choices'])==1 for d in aux.teacher.data))
        for row,target,identity,encoded in aux.teacher.data:
            self.assertEqual(target,terminal_targets(games[0]['result']))
            np.testing.assert_array_equal(encoded.stable_to_relative,[(s-row['seat'])%4 for s in range(4)])
            np.testing.assert_allclose(row['_reward_return'],terminal_utility(games[0]['result'],model.config.utility))
        self.assertEqual(len(aux.probe.indices),24);self.assertEqual(len(set(aux.probe.indices)),24)

    def test_target_cost_to_go_and_remaining_budgets_match_live_accounting(self):
        # Reuse the causal event fixtures, but preserve real static board fields
        # in every encoded row. Events before a state reduce only its budget;
        # only future events reduce that state's utility target.
        fixture=list(self.fixture());model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        raw=copy.deepcopy(dict(games[0]));raw['records']=copy.deepcopy(games[0]['records'])
        for i,row in enumerate(raw['records']):
            row['view']['context'].update(era='CanalEra',round=1+i//4)
            for player in row['view']['players']:player['hidden_discards']=1
            if i in (1,5,9,13):
                row['view']['context']['phase']='debt'
                for p in row['view']['players']:p['debt']=5 if p['seat']==0 else 0
        expected,summary=replay_returns(raw['records'],raw['result'],model.config.utility)
        self.assertAlmostEqual(summary['cost'][0],.020)
        fixture[2]=[raw];fixture[7]={raw['game_id']:dict(provenance(model_id(model),model.config.utility,opts),game_digest=digest(raw))}
        aux=self.auxiliary(fixture)
        for row,_,_,_ in aux.teacher.data:
            np.testing.assert_allclose(row['_reward_return'],expected[row['step']]['utility'])
            np.testing.assert_allclose(row['_reward_remaining'],expected[row['step']]['remaining'])
            key=(raw['game_id'],row['seat'],row['view']['context']['revision'])
            self.assertEqual(key in aux.probe.fit_keys,row['_reward_remaining']==[.075]*4)
        self.assertAlmostEqual(expected[0]['remaining'][0],.075)
        self.assertAlmostEqual(expected[1]['remaining'][0],.070)
        self.assertAlmostEqual(expected[0]['utility'][0]-expected[-1]['utility'][0],-.020)

    def test_stale_unlabelled_reward_duplicate_and_ordinary_sources_rejected(self):
        for change in ('source','model','raw','duplicate','ordinary','reward'):
            fixture=list(self.fixture());model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
            if change=='source':bindings[games[0]['game_id']]['implementation']='0'*64
            elif change=='model':bindings[games[0]['game_id']]['model_id']='0'*64
            elif change=='raw':games[0]['records'][1]['chosen']=1
            elif change=='duplicate':fixture[2]=games+[games[0]]
            elif change=='ordinary':ordinary[0][0]['behavior']['policy_id']='stale'
            else:bindings[games[0]['game_id']]['reward_id']='0'*64
            with self.subTest(change=change),self.assertRaises(ValueError):self.auxiliary(fixture)

    def test_selection_is_bounded_before_auxiliary_encoding(self):
        fixture=list(self.fixture(games=5));model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        fixture[4]=dict(opts,samples_per_game=2,max_samples=7,probe_positions=3)
        fixture[7]={g['game_id']:dict(provenance(model_id(model),model.config.utility,fixture[4]),game_digest=digest(g)) for g in games}
        from brass_ai.recovery.replay import CheckedGame
        original=CheckedGame.encoded;calls=[]
        def encode(game,row,encoder,**kw):
            calls.append((game['game_id'],row['step']))
            return original(game,row,encoder,**kw)
        with patch.object(CheckedGame,'encoded',encode):aux=self.auxiliary(fixture)
        self.assertEqual(len(calls),7);self.assertEqual(len(set(calls)),7)
        self.assertLessEqual(max(Counter(g for g,_ in calls).values()),2)
        self.assertEqual(len(aux.probe.indices),3);self.assertEqual(aux.report()['valid_value_positions'],80)

    def test_independent_validity_does_not_bypass_policy_gate_or_add_steps(self):
        fixture=list(self.fixture());model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        quality=dict(quality,min_samples=100)
        aux=self.auxiliary(fixture);before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
        report=distill_update(model,opt,games,cfg,steps=8,quality=quality,auxiliary=aux)
        self.assertFalse(report['updated']);self.assertEqual(report['optimizer_steps'],0)
        self.assertEqual(aux.report()['valid_value_positions'],32);self.assertEqual(aux.report()['distinct_positions_trained'],0)
        equal_tree(self,before,model.state_dict());self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(calibration_ready(model));self.assertIsNotNone(aux.report()['skip_reason'])

    def test_diagnostics_do_not_change_weights_adam_rng_samples_or_calibration(self):
        fixture=self.fixture();initial,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        results=[];rng=torch.get_rng_state().clone()
        for enabled in (False,True):
            model=copy.deepcopy(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.0001)
            local=(model,optimizer,games,ordinary,dict(opts,diagnostics=enabled),cfg,quality,bindings)
            aux=self.auxiliary(local);torch.set_rng_state(rng.clone())
            report=distill_update(model,optimizer,games,cfg,steps=8,quality=quality,auxiliary=aux)
            self.assertFalse(calibration_ready(model));self.assertEqual(report['optimizer_steps'],2)
            calibration=calibrate(model,optimizer,Dataset(ordinary,'cpu',policy=False),cfg,auxiliary=aux)
            results.append((model.state_dict(),optimizer.state_dict(),torch.get_rng_state(),aux.report(),calibration))
            self.assertEqual(aux.report()['max_auxiliary_reuse'],2)
            self.assertEqual(calibration['receipt']['steps'],3)
            c=calibration['receipt'];self.assertAlmostEqual(c['after']['mse'],.75*c['ordinary_after']['mse']+.25*c['teacher_after']['mse'])
        for i in (0,1,2):equal_tree(self,results[0][i],results[1][i])
        for k in ('distinct_positions_trained','sample_presentations','auxiliary_presentations','phases'):
            self.assertEqual(results[0][3][k],results[1][3][k])
        probe=results[1][3]['diagnostics']
        self.assertEqual(probe['extra_forward_positions']+probe['reused_fit_positions'],2*probe['positions'])
        self.assertGreater(probe['reused_fit_positions'],0)
        self.assertGreater(probe['elapsed_seconds'],0.)
        self.assertEqual(probe['reused_calibration_positions'],probe['positions'])
        self.assertTrue(all(p['available'] for p in probe['stages'].values()))
        self.assertGreater(probe['stages']['post_update']['prediction_change_from_before']['utility']['mean_absolute'],0.)
        self.assertEqual(probe['stages']['post_update']['errors']['vp_mae'],probe['stages']['post_calibration']['errors']['vp_mae'])

    def test_policy_only_keeps_value_heads_unchanged_auxiliary_updates_existing_heads(self):
        fixture=self.fixture();initial,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        old=copy.deepcopy(initial.state_dict());rng=torch.get_rng_state().clone();steps=[]
        for enabled in (False,True):
            model=copy.deepcopy(initial);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
            aux=self.auxiliary((model,optimizer,games,ordinary,opts,cfg,quality,bindings)) if enabled else None
            torch.set_rng_state(rng.clone());report=distill_update(model,optimizer,games,cfg,steps=8,quality=quality,auxiliary=aux)
            steps.append(report['optimizer_steps'])
            self.assertTrue(all(r['value_loss_training']==enabled for r in report['updates']))
            for prefix in ('outcomes.','utility_head.'):
                changed=any(not torch.equal(v,model.state_dict()[k]) for k,v in old.items() if k.startswith(prefix))
                self.assertEqual(changed,enabled)
            self.assertEqual(set(old),set(model.state_dict()))
        self.assertEqual(steps,[2,2])

    def test_mixed_calibration_cannot_waive_the_ordinary_readiness_gate(self):
        fixture=self.fixture();model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        aux=self.auxiliary(fixture);ds=Dataset(ordinary,'cpu',policy=False)
        model.eval()
        with torch.no_grad():
            ix,inputs,_=ds.value_batch(list(range(len(ds.data))),model)
            ds.utility.copy_(model(*inputs,remaining=ds.remaining[ix])[-1])
            aux.teacher.utility.fill_(5.)  # Deliberately conflicting test targets.
        aux.seen[0]=1;model.utility_calibration['ready']=False
        before=copy.deepcopy(model.state_dict())
        result=calibrate(model,opt,ds,cfg,auxiliary=aux)['receipt']
        self.assertLess(result['after']['mse'],result['before']['mse'])
        self.assertGreater(result['ordinary_after']['mse'],result['ordinary_before']['mse'])
        self.assertFalse(result['ready'])
        for k,v in before.items():
            if not k.startswith('utility_head.'):self.assertTrue(torch.equal(v,model.state_dict()[k]))

    def test_receipts_bind_direct_only_games_and_compact_history_is_bounded(self):
        fixture=self.fixture();model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        expected=provenance(model_id(model),model.config.utility,opts)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'teacher.jsonl.gz';jobs=[]
            for i,g in enumerate(games):
                seed=seeds_for(85000+i*100,1)[0][0]
                jobs.append(dict(index=i,seed=seed,provenance=dict(purpose='training',kind='masked-ismcts-teacher',teacher_value=expected),
                    policies=[dict(model_id=model_id(model),search=dict(reward=model.config.utility),teacher_schedule=CONTRACT)]*4))
            recipe=_collection_recipe(jobs,{})
            write_shard(path,games,dict(collection=digest(recipe),recipe=recipe,jobs=jobs))
            records=[dict(job=i,game_id=g['game_id'],path=str(path),sha256=sha(path)) for i,g in enumerate(games)]
            self.assertEqual(admit_collection(games,records,jobs,expected),{g['game_id']:bindings[g['game_id']] for g in games})
            stale=copy.deepcopy(jobs);stale[0]['provenance']['teacher_value']['implementation']='0'*64
            with self.assertRaisesRegex(ValueError,'Stale'):admit_collection(games,records,stale,expected)
        from brass_ai.recovery.history import compact
        aux=self.auxiliary(fixture);aux.begin(model)
        report=aux.report();report['vectors']=[0]*10000
        archived=compact(dict(search=dict(teacher_value=report,collections=[dict(teacher_value=report)])))
        self.assertEqual(archived['summary_version'],5);self.assertLess(len(json.dumps(archived)),14000)
        self.assertNotIn('vectors',archived['search']['teacher_value'])

    def test_collection_pipeline_admits_real_shards_calibrates_and_keeps_schedule(self):
        from brass_ai.recovery.teacher_collection import collect_due
        from brass_ai.recovery.history import compact
        fixture=self.fixture();model,opt,games,ordinary,opts,cfg,quality,bindings=fixture
        config=read(SOURCE_ROOT/'configs/step5/joint7-latest.json')
        config.update(teacher_value=opts,update=cfg,distillation=quality,search_games=2)
        state=dict(next_search_games=640,search_seconds=0.,history=[],population=[],evaluation_anchor={})
        service=SimpleNamespace(register=model_id,models={})
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp)
            def collect(jobs):
                complete=[]
                for job,episode in zip(jobs,games):
                    raw=copy.deepcopy(dict(episode))
                    raw['setup_group']=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
                    complete.append(check_game(raw))
                path=directory/'completed.jsonl.gz';recipe=_collection_recipe(jobs,config['execution'])
                write_shard(path,complete,dict(purpose='training',collection=digest(recipe),recipe=recipe,jobs=jobs))
                return [dict(job=j['index'],game_id=g['game_id'],path=str(path),sha256=sha(path)) for j,g in zip(jobs,complete)]
            teaching=dict(phases={p:dict(searches=4) for p,_,_,_ in PHASES})
            def refit(auxiliary):return calibrate(model,opt,Dataset(ordinary,'cpu',policy=False),cfg,auxiliary=auxiliary)
            with patch('brass_ai.recovery.teacher_collection.rollout_opponents',return_value=[]), \
                 patch('brass_ai.recovery.teacher_collection.collection_report',return_value=teaching):
                report,next_due,cursor,elapsed=collect_due(SOURCE_ROOT,directory,directory,config,state,{},model,opt,
                    service,SimpleNamespace(collect=collect),done=640,cursor=30000000,
                    ordinary_seconds=0.,ordinary_game_seconds=.1,recalibrate=refit,
                    ordinary_data=ordinary,ordinary_policy_id='fixture')
        self.assertEqual(next_due,800);self.assertGreater(cursor,30000000);self.assertGreater(elapsed,0.)
        self.assertEqual(report['collection_count'],1);self.assertTrue(report['value_loss_training'])
        self.assertEqual(report['teacher_value']['valid_value_positions'],32)
        batch=report['collections'][0];self.assertEqual(batch['teacher_rows'],8)
        self.assertEqual(batch['post_distill_calibration']['receipt']['teacher_fraction'],.25)
        archive=compact(dict(search=report))
        self.assertEqual(archive['search']['teacher_value']['distinct_completed_games'],2)
        self.assertEqual(archive['search']['collection_summaries'][0]['teacher_value']['valid_value_positions'],32)


if __name__=='__main__':unittest.main()
