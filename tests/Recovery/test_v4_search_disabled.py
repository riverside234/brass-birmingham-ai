"""Fresh recipe search switch, immutable IL binding and every evaluation bank."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json
from brass_ai.learning.model import ARCHITECTURE
from brass_ai.recovery.config import latest_model_config, validate_config, v4_search_enabled
from brass_ai.recovery.curriculum import searched_legacy
from brass_ai.recovery.execution import _collection_recipe
from brass_ai.recovery.legacy_search import legacy_spec
from brass_ai.recovery.ordinary import plan_collection
from brass_ai.recovery import evaluation, quality_pool, ratings
from brass_ai.step5.evaluation import evaluation_config, validate_evaluation_config


def config():
    return read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json')


class V4SearchDisabled(unittest.TestCase):
    def test_fresh_generation_upgrades_old_config_without_changing_IL_or_PPO(self):
        old=read_json(SOURCE_ROOT/'configs/step5/joint7-v11-1-grouped-lr.json')
        before=copy.deepcopy(old);new=latest_model_config(old,old['imitation_checkpoint'])
        self.assertEqual(old,before)
        allowed={'selfplay_recipe','curriculum','population','table_plan','opponent_pool','rating','v4_search_enabled','evaluation_schedule'}
        self.assertTrue({k for k in set(old)|set(new) if old.get(k)!=new.get(k)}<=allowed)
        self.assertFalse(new['v4_search_enabled']);self.assertFalse(new['teacher_enabled'])
        self.assertEqual(new['evaluation_schedule'],dict(old['evaluation_schedule'],enabled=False))
        self.assertTrue(new['rating']['enabled'])
        self.assertEqual(new['imitation_checkpoint'],old['imitation_checkpoint'])
        self.assertEqual(new['rehearsal_manifest'],old['rehearsal_manifest'])
        self.assertEqual(new['curriculum'],[dict(until=None,selfplay=50,historical=45,legacy=5)])
        self.assertEqual(new['opponent_pool']['max_members'],24)
        self.assertEqual(new,latest_model_config(new,new['imitation_checkpoint']))
        self.assertEqual(config(),latest_model_config(config(),config()['imitation_checkpoint']))

    def test_boolean_validation_and_old_configs_are_not_silently_changed(self):
        c=config();old=copy.deepcopy(c);old.pop('v4_search_enabled')
        self.assertTrue(v4_search_enabled(validate_config(old)))
        self.assertNotIn('v4_search_enabled',old)
        for value in (None,0,1,'false',{},[]):
            with self.subTest(value=value):
                bad=dict(c,v4_search_enabled=value)
                with self.assertRaisesRegex(ValueError,'v4_search_enabled'):validate_config(bad)
                e=evaluation_config(c);e['v4_search_enabled']=value
                with self.assertRaisesRegex(ValueError,'v4_search_enabled'):validate_evaluation_config(e)

    def test_disabled_at_every_age_even_with_legacy_boost(self):
        c=config()
        for age in (0,9999,10000,20000,100000,10**9):
            for boosted in (False,True):
                self.assertFalse(searched_legacy(age,c,boosted))
        old=copy.deepcopy(c);old.pop('v4_search_enabled');old['legacy_opponents']['search_free_until_games']=10000
        self.assertFalse(searched_legacy(9999,old,False))
        self.assertTrue(searched_legacy(9999,old,True))
        self.assertTrue(searched_legacy(10000,old,False))

    def test_ordinary_320_tables_keep_quota_pairs_and_frozen_search_free_opponents(self):
        c=config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);path=root/'weights.pt';path.write_bytes(b'planning fixture')
            entry=dict(model_id='IL',generation=0,path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path))
            pool=quality_pool.initial(entry,c['opponent_pool'],dict(path='probe',sha256='a'*64,probe_hash='probe'))
            state=dict(iteration=0,games_done=0,next_seed=c['seed'],context_counter=0,
                population=[entry],history=[entry],evaluation_anchor=entry,quality_pool=pool)
            service=Mock();service.models={'IL':None}
            slow={'v4':dict(kind='lab-legacy',policy_id='searched',search_free=False)}
            fast={'v4':dict(kind='lab-legacy',policy_id='fast',search_free=True)}
            planned=plan_collection(SOURCE_ROOT,root,state,c,dict(config=c),service,'learner',['v4'],slow,fast,None)
            self.assertEqual(planned['composition'],dict(selfplay=160,historical=144,legacy=16))
            self.assertEqual(planned['context_counter'],960)
            for j in planned['jobs']:
                learners=[p for p in j['policies'] if p.get('model_id')=='learner']
                self.assertTrue(all(p['exploration']==.08 for p in learners))
                self.assertEqual(len(learners),4 if j['provenance']['kind']=='selfplay' else 2)
                if j['provenance']['kind']=='legacy':
                    self.assertEqual(j['provenance']['legacy_mode'],'fast')
                    self.assertEqual(j['policies'].count(fast['v4']),2)
                if j['provenance']['kind']=='historical':
                    self.assertTrue(all(p['exploration']==.05 for p in j['policies'] if p not in learners))

    def test_evaluation_propagates_switch_independently_of_neural_search(self):
        c=config()
        for neural_search in (False,True):
            e=evaluation_config(c,use_search=neural_search,lineups=['mixed-v3'],include_legacy=True)
            self.assertFalse(e['v4_search_enabled']);self.assertEqual(e['use_search'],neural_search)
            self.assertEqual(e['legacy_opponent_work_budget'],c['legacy_opponent_work_budget'])
            self.assertFalse(evaluation_config(e)['v4_search_enabled'])
        old=copy.deepcopy(c);old.pop('v4_search_enabled')
        self.assertTrue(v4_search_enabled(evaluation_config(old)))

    def test_policy_and_collection_identities_distinguish_search_modes(self):
        c=config();profile=c['rating']['v4_profile']
        slow=legacy_spec(SOURCE_ROOT,profile,c['legacy_budget'],work_budget=c['legacy_opponent_work_budget'])
        fast=legacy_spec(SOURCE_ROOT,profile,c['legacy_budget'],search_free=True,work_budget=c['legacy_opponent_work_budget'])
        self.assertNotEqual(fast['policy_id'],slow['policy_id'])
        job=dict(index=0,seed=3,sampling_seed=7,policies=[slow]*4,provenance={},path='ignored')
        recipe=_collection_recipe([job],c['execution'])
        changed=_collection_recipe([dict(job,policies=[fast]*4)],c['execution'])
        self.assertNotEqual(digest(recipe),digest(changed))
        self.assertEqual(digest(changed),digest(_collection_recipe([dict(job,policies=[fast]*4,path='relocated')],c['execution'])))

    def test_ratings_remain_search_free(self):
        c=config();entry=dict(model_id='IL',generation=0,path='initial.pt',sha256='a'*64)
        with patch.object(ratings,'check_dependencies'):
            state=ratings.initialize(SOURCE_ROOT,SOURCE_ROOT/'artifacts/unused',dict(population=[entry]),c)
        self.assertTrue(state['agents']['v4']['policy']['search_free'])

    def test_all_scheduled_evaluation_banks_use_ordinary_workers_and_reject_changed_cache(self):
        c=config();c['retention']['enabled']=False;c['threads']=1
        collected=[];replays={}
        class Service:
            def __init__(self,*a,**k):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def register_checkpoint(self,*a):pass
            def report(self):return {}
        class Pool(Service):
            width=lanes=1
            def __init__(self,*a,heavy=False,**k):self.heavy=heavy
            def collect(self,jobs):
                collected.append((self.heavy,jobs));result=[]
                for j in jobs:
                    game_id=Path(j['path']).stem
                    replays[j['path']]=dict(game_id=game_id,split='validation',
                        setup_group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=j['seed'])),
                        opponents=[j['policies'][(s-j['rotation'])%4]['policy_id'] for s in range(4)])
                    result.append(dict(job=j['index'],game_id=game_id,path=j['path'],sha256='a'*64,cached=False))
                return result
        def checkpoint(path,**kwargs):
            return SimpleNamespace(config=SimpleNamespace(architecture=ARCHITECTURE)),dict(model_id=Path(path).stem,
                training=dict(fresh_run='artifacts/same-run',mode='step5-population-ppo'))
        mocks=dict(InferenceService=Service,ActorPool=Pool,load_checkpoint=checkpoint,sha=lambda p:'a'*64,
            read_shards=lambda paths,**k:[replays[p] for p in paths],
            outcome=lambda g,c,s:dict(seed=s,seat=g['opponents'].index(c)),
            summarize_outcomes=lambda rows:dict(games=len(rows),win_share_setup_bootstrap_95=[0,1]),
            pottery_metrics=lambda *a:dict(pottery_warning=False),economy=lambda *a:{})
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp,patch.multiple(evaluation,**mocks):
            out=Path(tmp)/'evaluation'
            e=evaluation_config(c,panel='full',include_legacy=True,use_search=False,setups=1,
                lineups=['recent-neural','mixed-neural','mixed-v3','heldout-pottery-v3'])
            result=evaluation.evaluate(SOURCE_ROOT,out,'candidate.pt',['ref1.pt','ref2.pt'],e)
            self.assertFalse(result['v4_search_enabled']);self.assertEqual(result['games'],16)
            self.assertEqual(len(collected),4)
            for heavy,jobs in collected:
                self.assertFalse(heavy)
                for j in jobs:
                    for p in j['policies']:
                        if p['kind']=='lab-legacy':self.assertTrue(p['search_free'])
            self.assertTrue(result['spec']['teachers']);self.assertTrue(result['spec']['held'])
            changed=dict(e,v4_search_enabled=True)
            with self.assertRaisesRegex(ValueError,'Evaluation specification changed'):
                evaluation.evaluate(SOURCE_ROOT,out,'candidate.pt',['ref1.pt','ref2.pt'],changed)


if __name__=='__main__':unittest.main()
