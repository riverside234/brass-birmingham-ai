"""Causal debt costs, unchanged replay/value labels, and bounded v4 exposure."""
import copy
from collections import Counter
from pathlib import Path
import random
import tempfile
import unittest

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.replay import terminal_targets
from brass_ai.recovery.config import INSOLVENCY,LEGACY_BOOST,UPDATE_V21,validate_config
from brass_ai.recovery.curriculum import activate_boost,boost_window,training_table,searched_legacy,boost_report
from brass_ai.recovery.insolvency import events,returns,advantage_costs
from brass_ai.recovery.ingress import load_collection,reader_pool
from brass_ai.recovery.replay import write_shard,read_shards,cache_receipt
from brass_ai.recovery.resume import check_signature,commit_transition
from brass_ai.recovery.source_reviews import expand_source_reviews
from brass_ai.recovery.updates import prepared,ppo_update
from brass_ai.step5.league import implementation_id,write_json
from test_revision import RevisionTests


def row(seat=0,round=1,era='CanalEra',phase='action',kind='pass',debts=None,cash=None,income=None):
    return dict(seat=seat,chosen=0,choices=[dict(kind=kind)],view=dict(
        context=dict(era=era,round=round,phase=phase),slots=[],
        players=[dict(seat=s,cash=(cash or {}).get(s,10),income=(income or {}).get(s,0),
                      debt=(debts or {}).get(s,0)) for s in range(4)]))


def onpolicy_debt():
    model,policy,game=RevisionTests().onpolicy(set())
    last=game['records'][-1];last['view']['context']['phase']='debt'
    last['view']['players'][0]['debt']=4;last['view']['players'][3]['debt']=3
    slot=last['view']['slots'][0]
    slot['tile']=copy.deepcopy(last['view']['players'][3]['mat'][0]['tiles'][0])
    last['choices']=[dict(id='liquidate-a',kind='liquidate',target=slot['id'])]
    index,p,_=policy.select_batch([(last['view'],last['choices'],random.Random(7))])[0]
    last.update(chosen=index,behavior=dict(kind='model',policy_id=policy.policy_id,probability=p,temperature=1.))
    return model,policy,game


class InsolvencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_grouped_liquidations_count_once_and_only_prior_decisions_pay(self):
        rows=[row(0),row(1),row(2,phase='debt',kind='liquidate',debts={0:5,2:2}),
              row(0,phase='debt',kind='liquidate',debts={0:5}),
              row(0,phase='debt',kind='liquidate',debts={0:1}),row(0,round=2)]
        costs,summary=returns(rows,INSOLVENCY)
        self.assertEqual(costs,[.005,0,0,0,0,0])
        self.assertEqual([s['events'] for s in summary],[1,0,1,0])
        self.assertEqual(summary[0]['shortfall'],5)
        self.assertEqual(summary[2]['shortfall'],2)
        self.assertEqual(len(events(rows)),2)

    def test_paid_negative_income_and_final_rail_are_exempt(self):
        solvent=[row(0,phase='continuation',kind='finish',cash={0:5},income={0:-5}),row(0,round=2)]
        self.assertEqual(events(solvent),[])
        final=[row(0,era='RailEra',round=8,phase='continuation',kind='finish',cash={0:0},income={0:-10})]
        final[0]['view']['slots']=[dict(tile=dict(seat=0))]
        self.assertEqual(events(final),[])

    def test_no_asset_automatic_settlement_and_final_canal(self):
        rows=[row(0),row(3,phase='continuation',kind='finish',cash={0:1},income={0:-4}),row(0,round=2)]
        costs,summary=returns(rows,INSOLVENCY)
        self.assertEqual(costs,[.005,0,0]);self.assertEqual(summary[0]['shortfall'],3)
        rows=[row(0,round=8),row(3,round=8,phase='continuation',kind='finish',cash={0:0},income={0:-2}),
              row(0,era='RailEra',round=1)]
        self.assertEqual(returns(rows,INSOLVENCY)[0],[.005,0,0])

    def test_repeated_events_cap_per_seat_not_per_tile_or_decision(self):
        self.assertEqual(INSOLVENCY,dict(event_cost=.005,max_cost=.075))
        # Preserve the historical five-event cap, and explicitly exercise the
        # current seven-event and full fifteen-event costs across both eras.
        for cap,count,total,after_first in ((.025,7,.025,.020),(.075,7,.035,.030),(.075,15,.075,.070)):
            with self.subTest(cap=cap,events=count):
                rows=[]
                for n in range(count):
                    phase=dict(round=n%8+1,era='CanalEra' if n<8 else 'RailEra')
                    rows.extend([row(0,**phase),row(0,**phase,phase='debt',kind='liquidate',debts={0:8})])
                costs,summary=returns(rows,dict(event_cost=.005,max_cost=cap))
                self.assertEqual(summary[0]['events'],count);self.assertAlmostEqual(summary[0]['cost'],total)
                self.assertAlmostEqual(costs[0],total);self.assertAlmostEqual(costs[1],after_first)
                if cap==.025:self.assertEqual(costs[-4:],[0,0,0,0])
                else:
                    self.assertAlmostEqual(costs[-2],.005);self.assertAlmostEqual(costs[-1],0.)
                self.assertTrue(all(0<=c<=cap+1e-8 for c in costs))

    def test_actor_return_changes_but_official_labels_and_replay_do_not(self):
        model,policy,game=onpolicy_debt();original=copy.deepcopy(game)
        plain=prepared([game]);shaped=prepared([game],insolvency=INSOLVENCY)
        self.assertEqual(game,original)
        for a,b in zip(plain,shaped):self.assertEqual(a[1],b[1])
        self.assertEqual(shaped[0][1],terminal_targets(game['result']))
        costs,metrics=advantage_costs(shaped,INSOLVENCY)
        self.assertEqual(costs,[.005,0,0,0]);self.assertEqual(metrics['events'],2)
        cfg=dict(UPDATE_V21,epochs=1,effective_batch_size=4,microbatch_size=2,critic_batch_size=4)
        results=[];models=[]
        for setting in (None,INSOLVENCY):
            net=copy.deepcopy(model);torch.manual_seed(42)
            results.append(ppo_update(net,torch.optim.AdamW(net.parameters(),lr=1e-5),[game],policy,cfg,insolvency=setting))
            models.append(net)
        self.assertAlmostEqual(results[0]['advantage']['mean']-results[1]['advantage']['mean'],.005/3,places=6)
        self.assertEqual(results[1]['forced_decisions'],1)
        self.assertTrue(any(not torch.equal(v,models[1].state_dict()[k]) for k,v in models[0].state_dict().items()))
        self.assertFalse(results[0]['insolvency']['enabled'])
        with self.assertRaisesRegex(ValueError,'Missing derived'):advantage_costs(plain,INSOLVENCY)

    def test_parallel_ingress_preserves_costs_and_ppo_update(self):
        model,policy,game=onpolicy_debt()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            records=[]
            for i in range(2):
                episode=copy.deepcopy(game);episode['game_id']=f'debt-{i}'
                path=Path(temp)/f'{i}.jsonl.gz';write_shard(path,[episode],dict(purpose='training'))
                records.append(dict(path=str(path),job=i,game_id=episode['game_id'],sha256=sha(path),features=cache_receipt(path)))
            games=read_shards([r['path'] for r in records],partition='train',receipts={r['path']:r['features'] for r in records})
            with reader_pool(2) as readers:
                headers,data,_=load_collection(records,policy.policy_id,{game['setup_group']:'selfplay'},
                    executor=readers,workers=2,insolvency=INSOLVENCY)
            expected=prepared(games,insolvency=INSOLVENCY)
            self.assertEqual(advantage_costs(expected,INSOLVENCY),advantage_costs(data,INSOLVENCY))
            self.assertNotIn('players',data[0][0]['view'])
            cfg=dict(UPDATE_V21,epochs=1,effective_batch_size=8,microbatch_size=2,critic_batch_size=8)
            output=[]
            for source,preprocessed in ((games,None),(headers,data)):
                net=copy.deepcopy(model);torch.manual_seed(91)
                result=ppo_update(net,torch.optim.AdamW(net.parameters(),lr=1e-5),source,policy,cfg,
                                  learner_data=preprocessed,insolvency=INSOLVENCY)
                output.append((net,result))
            self.assertEqual(output[0][1]['epochs'],output[1][1]['epochs'])
            self.assertTrue(all(torch.equal(v,output[1][0].state_dict()[k]) for k,v in output[0][0].state_dict().items()))


class BoostTests(unittest.TestCase):
    def fixture(self,start=22400):
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json');state=dict(games_done=start)
        activate_boost(state,cfg);return cfg,state

    def test_exact_twenty_thousand_quotas_seats_and_expiry(self):
        from brass_ai.recovery.config import table_rotation
        cfg,state=self.fixture();window=boost_window(state,cfg);totals=Counter();seats=Counter()
        for i in range(22400,42400):
            kind,boosted=training_table(i,cfg,window);self.assertTrue(boosted);totals[kind]+=1
            if kind=='legacy':
                seats[table_rotation(i)]+=1;self.assertTrue(searched_legacy(i,cfg,boosted))
        self.assertEqual(totals,dict(selfplay=3000,historical=5000,legacy=12000))
        self.assertEqual(list(seats.values()),[3000]*4)
        middle=Counter(training_table(i,cfg,window)[0] for i in range(42400,52400))
        self.assertEqual(middle,dict(selfplay=4500,historical=2500,legacy=3000))
        self.assertTrue(training_table(42400,cfg,window)[1])
        later=Counter(training_table(i,cfg,window)[0] for i in range(52400,52720))
        self.assertEqual(later,dict(selfplay=224,historical=80,legacy=16))
        self.assertFalse(training_table(52400,cfg,window)[1])
        report=boost_report(42240,42560,cfg,window)
        self.assertEqual([s['ordinary_games_this_iteration'] for s in report['stages']],[160,160])
        self.assertEqual(report['ordinary_games_this_iteration'],320)
        self.assertEqual(boost_report(52320,52640,cfg,window)['ordinary_games_this_iteration'],80)

    def test_restart_never_rearms_and_boost_overrides_search_free_warmup(self):
        cfg,state=self.fixture(0);old=copy.deepcopy(state['legacy_boost'])
        self.assertTrue(searched_legacy(5,cfg,True));self.assertFalse(searched_legacy(5,cfg,False))
        state['games_done']=640;activate_boost(state,cfg);self.assertEqual(state['legacy_boost'],old)
        state['games_done']=34000;activate_boost(state,cfg);self.assertEqual(state['legacy_boost'],old)
        self.assertEqual(boost_report(34000,34320,cfg,old)['remaining_games'],0)
        state['legacy_boost']['until_games']+=1
        with self.assertRaisesRegex(ValueError,'window changed'):boost_window(state,cfg)

    def test_presets_and_invalid_settings(self):
        for name in ('lab','lab-16x2','lab-24x4','lab-small','lab-reference','smoke','recovery-smoke'):
            cfg=validate_config(read(SOURCE_ROOT/f'configs/step5/{name}.json'))
            self.assertEqual(cfg['insolvency'],INSOLVENCY)
            if name.startswith('lab'):self.assertEqual(cfg['legacy_boost'],LEGACY_BOOST)
        for key,change in [('insolvency',dict(event_cost=.2,max_cost=.2)),
                           ('insolvency',dict(event_cost=-.01,max_cost=.1)),
                           ('legacy_boost',dict(stages=[dict(LEGACY_BOOST['stages'][0],games=20001)])),
                           ('legacy_boost',dict(stages=[dict(LEGACY_BOOST['stages'][0],selfplay=70)])),
                           ('legacy_boost',dict(stages=[dict(LEGACY_BOOST['stages'][0],games=30020)]))]:
            cfg,_=self.fixture();cfg[key]=change
            with self.assertRaises(ValueError):validate_config(cfg)

    def test_reviewed_generation70_upgrade_records_window_once(self):
        state=read(SOURCE_ROOT/'results/selfplay-v2-70/state.json');previous=state['signature']
        cfg=read(SOURCE_ROOT/'configs/step5/lab.json');cfg['rehearsal_manifest']=previous['imitation_source']['manifest']
        signature=dict(previous,config=cfg,implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,True))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);write_json(run/'state.json',state)
            updated=commit_transition(run,state,signature)
            self.assertEqual((updated['legacy_boost']['start_games'],updated['legacy_boost']['until_games']),(22400,52400))
            self.assertFalse(check_signature(updated,signature))
            activate_boost(updated,cfg)
            self.assertEqual(updated['legacy_boost']['until_games'],52400)
            self.assertEqual(updated['history'],state['history'])
            self.assertEqual(updated['next_seed'],state['next_seed'])
        bad=copy.deepcopy(signature);bad['config']['insolvency']['event_cost']=.03
        with self.assertRaisesRegex(ValueError,'unchanged'):check_signature(state,bad,True)

    def test_prior_review_and_two_transitions_before_ppo_keep_exact_checkpoint(self):
        from test_reviewed_resume import ReviewedResumeTests
        from brass_ai.recovery.resume import load_resume
        from brass_ai.learning.checkpoint import resume_optimizer
        ReviewedResumeTests.setUpClass();fixture=ReviewedResumeTests()
        try:
            run,state,_,original,adam=fixture.fixture()
            # Adopt the preceding reviewed version without producing a new
            # checkpoint, then adopt this revision. Restart must follow both
            # signed state transitions to the same original checkpoint.
            previous='2ea375ddfa37d3aa07786eba035f3d17e38fd94876734fee5317bed58af68360'
            cfg=read(SOURCE_ROOT/'configs/step5/lab.json')
            cfg.update(model_size='small',rehearsal_manifest=state['signature']['imitation_source']['manifest'])
            # Reconstruct the actual historical target from its pinned review.
            # Copying today's preset imports later reward/champion/search fields
            # and fabricates an unapproved historical configuration.
            reviews=expand_source_reviews(read(SOURCE_ROOT/'configs/learning/selfplay-resume-sources.json'))['transitions']
            review=next(r for r in reviews if r['from_implementation']==state['signature']['implementation']
                        and r['to_implementation']==previous)
            self.assertEqual(review['learning_compatibility'],digest(state['signature']['compatibility']))
            before=copy.deepcopy(state['signature']['config'])
            for key,change in review['config_changes'].items():
                self.assertEqual(before.get(key),change['from'])
                before[key]=copy.deepcopy(change['to'])
            signature=dict(state['signature'],implementation=previous,config=before)
            self.assertTrue(check_signature(state,signature,True))
            interim=commit_transition(run,state,signature)
            latest=dict(signature,implementation=implementation_id(),config=cfg)
            model,raw,changed=load_resume(SOURCE_ROOT,run,interim,latest,reviewed=True)
            self.assertTrue(changed)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']);resume_optimizer(raw,optimizer)
            self.assertTrue(torch.equal(torch.get_rng_state(),raw['rng']))
            updated=commit_transition(run,interim,latest)
            self.assertFalse(load_resume(SOURCE_ROOT,run,updated,latest)[2])
            self.assertEqual(updated['legacy_boost']['start_games'],state['games_done'])
            self.assertEqual(updated['legacy_boost']['until_games'],state['games_done']+30000)
            for key,value in original.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
            for key,values in adam.state_dict()['state'].items():
                for name,value in values.items():torch.testing.assert_close(value,optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
        finally:fixture.doCleanups()


if __name__=='__main__':unittest.main()
