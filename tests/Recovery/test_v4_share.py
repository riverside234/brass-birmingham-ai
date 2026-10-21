"""V4 panel gate, exact quotas, PFSP continuity and whole-game EV coverage."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock,patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import v4_share as gate,critic_holdout as holdout,quality_pool as q
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config,latest_model_config
from brass_ai.recovery.history import compact
from brass_ai.recovery.lr_controller import collection_signature
from brass_ai.recovery.ordinary import plan_collection
from brass_ai.recovery.plackett_luce import V4
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import check_signature,commit_transition
from brass_ai.recovery.value_phase import disabled_options
from brass_ai.step5.league import write_json,implementation_id
from test_gae_value_phase import fixture
from test_pfsp_pool import pool
from test_ppg_refinement import rows
from test_ppg_window import upgraded

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0210-20261020T145156876449Z'
EXPORT213=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0213-20261020T173708767483Z'


def config():
    return gate.configure(read(EXPORT/'state.json')['signature']['config'])


def panel(root,cfg,rate,*,generation=210,games=None):
    candidate=f'generation-{generation:06d}'
    reference=dict(kind='v4',policy=dict(search_free=True,policy_id='frozen-v4'))
    path=root/f'panels-{generation}.json';spec=root/f'spec-{generation}.json'
    write_json(path,dict(candidate=candidate,panels={'v4-panel':dict(
        games=cfg['rating']['panel_setups']*4 if games is None else games,
        setups=cfg['rating']['panel_setups'],learner_win_share=rate)}))
    write_json(spec,dict(generation=generation,candidate=candidate,config=cfg['rating'],agents={V4:reference}))
    receipt=lambda p:dict(path=p.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(p))
    return dict(config=cfg['rating'],agents={V4:reference},
        slots=[dict(generation=generation,panels=receipt(path),spec=receipt(spec))])


class ShareGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_strict_threshold_latest_panel_latch_restart_and_missing_evidence(self):
        c=config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);state=dict(iteration=210,games_done=67200,ratings=dict(slots=[]))
            waiting=gate.prepare(SOURCE_ROOT,state,c)
            self.assertIsNone(waiting['activated']);self.assertIsNone(waiting['last_panel'])
            state['ratings']=panel(root,c,.95)
            at=gate.prepare(SOURCE_ROOT,state,c);self.assertIsNone(at['activated'])
            state['ratings']=panel(root,c,31/32)
            passed=gate.prepare(SOURCE_ROOT,state,c)
            self.assertEqual(passed['activated'],dict(generation=210,games=67200))
            state['v4_share_gate']=json.loads(json.dumps(passed))
            # No automatic rollback or replay of activation on later weakness.
            state.update(iteration=220,games_done=70400,ratings=panel(root,c,.5,generation=220))
            self.assertEqual(gate.prepare(SOURCE_ROOT,state,c),passed)
            # Newly adopting the feature must not use an earlier peak.
            del state['v4_share_gate']
            state['ratings']['slots'].insert(0,panel(root,c,1.,generation=210)['slots'][0])
            self.assertIsNone(gate.prepare(SOURCE_ROOT,state,c)['activated'])

    def test_config_panel_and_receipt_validation(self):
        c=config();validate_config(c)
        self.assertFalse(c['value_phase']['enabled'])
        self.assertEqual(c,gate.configure(c))
        fresh=latest_model_config(c,c['imitation_checkpoint'])
        self.assertEqual(fresh['v4_share_gate'],gate.CURRENT_DEFAULT)
        self.assertEqual(fresh['value_phase']['holdout'],holdout.DEFAULT)
        for key,value in (('win_share_above',float('nan')),('reference_games',True),
                          ('reduced_v4_games',7),('enabled',1),('source','ordinary'),
                          ('released_tables','selfplay')):
            bad=copy.deepcopy(c);bad['v4_share_gate'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_config(bad)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp)
            for rate,count in ((float('nan'),32),(1.,31),(-.1,32),(1.1,32)):
                # write_json disallows NaN itself; exercise state validation separately.
                if rate!=rate:continue
                state=dict(iteration=210,games_done=67200,ratings=panel(root,c,rate,games=count))
                with self.assertRaises(ValueError):gate.prepare(SOURCE_ROOT,state,c)
            state=dict(iteration=210,games_done=67200,ratings=panel(root,c,1.))
            (root/'panels-210.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'receipt changed'):gate.prepare(SOURCE_ROOT,state,c)
        missing=copy.deepcopy(c);missing['rating']['enabled']=False
        with self.assertRaises(ValueError):validate_config(missing)

    def test_counts_all_offsets_continuous_ordinals_and_no_extra_rng(self):
        c=config();python_rng=random.getstate();rng=torch.get_rng_state().clone()
        for origin in (0,1,17,200,67200,67203):
            s=gate.initial(c,210,origin)
            s['activated']=dict(generation=210,games=origin)
            before=gate.legacy_before(origin,c,None,None)
            for start in (origin,origin+320,origin+641):
                counts=Counter(gate.table(i,c,None,s)[0] for i in range(start,start+320))
                self.assertEqual(counts,dict(selfplay=160,historical=154,legacy=6))
                self.assertEqual(gate.legacy_before(start+320,c,None,s)-gate.legacy_before(start,c,None,s),6)
                self.assertEqual(gate.history_before(start+320,c,None,s)-gate.history_before(start,c,None,s),154)
            self.assertEqual(gate.legacy_before(origin,c,None,s),before)
        self.assertEqual(random.getstate(),python_rng);self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_real_planner_extra_PFSP_games_balanced_pairs_and_restart(self):
        c=config()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            root=Path(temp);path=root/'weights.pt';path.write_bytes(b'planning fixture')
            p=pool(4,generation=210);fp=dict(values=[1/8]*16+[1/6]*12)
            for member in p['members']:
                member.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),fingerprint=fp)
            state=dict(iteration=210,games_done=67200,next_seed=c['seed'],context_counter=201600,
                quality_pool=p,population=[q.reference(m) for m in p['members']],history=[q.reference(p['members'][0])],
                evaluation_anchor=q.reference(p['members'][0]))
            service=Mock();service.models={m['model_id']:None for m in p['members']}
            profiles=[f'v4-{i}' for i in range(5)]
            bots={s:dict(kind='legacy',policy_id=s,search_free=True) for s in profiles}
            def plan(s):
                return plan_collection(SOURCE_ROOT,root,s,c,dict(config=c),service,'current',
                    profiles,bots,bots,None,fingerprints=dict(learner=fp,rule_bot=fp))
            state['v4_share_gate']=gate.initial(c,210,67200);base=plan(state)
            state['ratings']=panel(root,c,1.)
            state['v4_share_gate']=gate.prepare(SOURCE_ROOT,state,c)
            frozen=copy.deepcopy(state);new=plan(state)
            self.assertEqual(new,plan(json.loads(json.dumps(state))));self.assertEqual(state,frozen)
            self.assertEqual(base['composition'],dict(selfplay=160,historical=144,legacy=16))
            self.assertEqual(new['composition'],dict(selfplay=160,historical=154,legacy=6))
            self.assertEqual(len(new['quality_tables']),154)
            self.assertEqual(new['quality_planning'],base['quality_planning'])
            # Original history slots retain their independent opponent draws;
            # actual history ordinals now include the ten extra tables.
            for setup,old in base['quality_tables'].items():
                current=new['quality_tables'][setup]
                without_seat=lambda r:{k:v for k,v in r.items() if k!='seat'}
                self.assertEqual([without_seat(r) for r in current['opponents']],
                                 [without_seat(r) for r in old['opponents']])
            self.assertEqual(sum(new['historical_generations'].values()),308)
            self.assertEqual(new['cursor'],base['cursor'])
            self.assertEqual(new['context_counter']-state['context_counter'],960)
            history_pairs=Counter(tuple(t['learner_seats']) for t in new['quality_tables'].values())
            self.assertEqual(set(history_pairs),set(q.PAIRS))
            self.assertEqual(sorted(history_pairs.values()),[25,25,26,26,26,26])
            pairs=[];variants=Counter()
            for job in new['jobs']:
                policies=[job['policies'][(s-job['rotation'])%4] for s in range(4)]
                if job['provenance']['kind']=='legacy':
                    pairs.append(tuple(i for i,p in enumerate(policies) if p.get('model_id')=='current'))
                    variants.update(p['policy_id'] for p in policies if p.get('model_id')!='current')
                    self.assertEqual(len(job['provenance']['legacy_profiles']['seats']),2)
            self.assertEqual(set(pairs),set(q.PAIRS));self.assertEqual(len(pairs),6)
            self.assertEqual(set(variants),set(profiles))
            self.assertLessEqual(max(variants.values())-min(variants.values()),1)
            r=gate.report(state['v4_share_gate'],new['composition'])
            self.assertEqual(r['history_opponent_seats'],308)
            self.assertEqual(r['history_pairwise_comparisons'],616)
            self.assertEqual(compact(dict(v4_share_gate=r))['v4_share_gate'],r)
            # Existing LR controller resets its evidence when actual mix changes.
            def sample(plan):
                return dict(by_composition={k:dict(tables=n,learner_trajectories=n*(4 if k=='selfplay' else 2))
                    for k,n in plan['composition'].items()})
            self.assertNotEqual(collection_signature(c,sample(base)),collection_signature(c,sample(new)))
            # Feed the actual new plan through the estimator. All four pairs
            # on every added table count, once, at the frozen probabilities.
            games=[dict(setup_group=setup,opponents=t['policy_ids'],result={})
                for setup,t in new['quality_tables'].items()]
            with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(placement=[1,2,3,4])):
                updated,report=q.update_scores(p,games,new['quality_tables'],new['quality_planning'],
                    generation=211,games_done=67520)
                replayed,again=q.update_scores(json.loads(json.dumps(p)),games,new['quality_tables'],
                    new['quality_planning'],generation=211,games_done=67520)
            self.assertEqual((updated,report),(replayed,again));self.assertEqual(p,frozen['quality_pool'])
            self.assertEqual(report['exposure']['history_tables'],154)
            self.assertEqual(report['exposure']['opponent_seats'],308)
            self.assertEqual(report['exposure']['pairwise_comparisons'],616)
            self.assertEqual(report['exposure']['history_table_fraction'],154/320)
            self.assertEqual(report['exposure']['half_life_iterations'],5)
            decay=.5**.2
            for old,member,r in zip(p['members'],updated['members'],report['members']):
                self.assertAlmostEqual(member['C'],decay*old['C']+r['comparisons'])
                self.assertAlmostEqual(member['S'],decay*old['S']+(r['outrank_rate'] or 0)*r['comparisons'])
            self.assertEqual(sum(m['comparisons'] for m in report['members']),616)

    def test_holdout_exact_8_8_2_before_and_after_gate_branches_do_not_inflate(self):
        from brass_ai.recovery.branching import family
        for counts in ((160,144,16),(160,154,6)):
            data=rows(counts)
            train,held,r=holdout.partition(data,.05,holdout.DEFAULT)
            self.assertEqual([r['by_table'][k]['heldout_games'] for k in holdout.TABLES],[8,8,2])
            self.assertEqual((r['heldout_parent_games'],r['parent_games']),(18,320))
            self.assertEqual(len(held),72);self.assertEqual(len(train),1208)
            self.assertEqual(r,holdout.partition(list(reversed(data)),.05,holdout.DEFAULT)[2])
            holdout.validate_selection(r['selection'],holdout.DEFAULT,.05)
            # Add an actual branch-family identity, for every self-play parent.
            branches=[(copy.deepcopy(d[0]),None,(d[2][0]+'/branch/1',d[2][1]),None)
                for d in data if d[0]['_table_kind']=='selfplay']
            _,_,b=holdout.partition(data+branches,.05,holdout.DEFAULT)
            self.assertEqual(b['heldout_parent_games'],18)
            self.assertEqual({family(g) for g in b['selection']['heldout_games']},set(r['selection']['heldout_games']))
        # Archived selections continue to reproduce their original hash/quota.
        old=holdout.partition(rows(),.05,holdout.LEGACY_DEFAULT)[2]
        self.assertEqual([old['by_table'][k]['heldout_games'] for k in holdout.TABLES],[8,7,2])
        holdout.validate_selection(old['selection'])

    def test_legacy_gate_retains_its_allocation_and_migrates_without_backdating(self):
        c=gate.configure_legacy(config());validate_config(c)
        self.assertEqual(c['v4_share_gate'],gate.LEGACY_DEFAULT)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            state=dict(signature=dict(config=c),iteration=210,games_done=67200,
                ratings=panel(Path(temp),c,1.))
            state['v4_share_gate']=gate.prepare(SOURCE_ROOT,state,c)
            state.update(iteration=213,games_done=68160)
            saved=copy.deepcopy(state);old=state['v4_share_gate']
            self.assertEqual(Counter(gate.table(i,c,None,old)[0] for i in range(67840,68160)),
                dict(selfplay=170,historical=144,legacy=6))
            target=gate.configure(c);new,receipt=gate.migrate_state(state,target)
            self.assertEqual(state,saved)
            self.assertEqual(new['activated'],old['activated']);self.assertEqual(new['last_panel'],old['last_panel'])
            self.assertEqual(new['origin_games'],old['origin_games'])
            self.assertEqual(new['allocation_origin'],dict(generation=213,games=68160))
            self.assertEqual(receipt['from_digest'],digest(old))
            self.assertEqual(Counter(gate.table(i,target,None,new)[0] for i in range(67840,68160)),
                dict(selfplay=170,historical=144,legacy=6))
            self.assertEqual(gate.history_before(68160,target,None,new),gate.history_before(68160,c,None,old))
            self.assertEqual(gate.legacy_before(68160,target,None,new),gate.legacy_before(68160,c,None,old))
            self.assertEqual(Counter(gate.table(i,target,None,new)[0] for i in range(68160,68480)),
                dict(selfplay=160,historical=154,legacy=6))
            for start,stop in ((68160,68177),(68177,68210),(68210,68480)):
                counts=Counter(gate.table(i,target,None,new)[0] for i in range(start,stop))
                self.assertEqual(gate.history_before(stop,target,None,new)-gate.history_before(start,target,None,new),
                    counts['historical'])
            again=dict(state,signature=dict(config=target),v4_share_gate=json.loads(json.dumps(new)))
            self.assertEqual(gate.migrate_state(again,target),(new,None))
            self.assertEqual(gate.prepare(SOURCE_ROOT,again,target),new)

    def test_generation213_transition_preserves_training_and_latch_state(self):
        state=read(EXPORT213/'state.json');original=copy.deepcopy(state)
        target=gate.configure(state['signature']['config']);validate_config(target)
        self.assertEqual({k for k in target if target[k]!=state['signature']['config'].get(k)},{'v4_share_gate'})
        signature=dict(state['signature'],config=target,implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,reviewed=True))
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,signature)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            run=Path(temp);write_json(run/'state.json',state)
            updated=commit_transition(run,state,signature)
            self.assertEqual(state,original);self.assertEqual(read(run/'state.json'),updated)
            self.assertEqual({k for k in state if state[k]!=updated[k]},
                {'signature','source_transitions','v4_share_gate'})
            self.assertEqual(updated['quality_pool'],original['quality_pool'])
            self.assertEqual(updated['v4_share_gate']['origin_games'],67200)
            self.assertEqual(updated['v4_share_gate']['allocation_origin'],dict(generation=213,games=68160))
            self.assertIsNone(updated['v4_share_gate']['activated'])
            self.assertEqual(updated['v4_share_gate']['last_panel']['win_share'],.8125)
            self.assertEqual(gate.migrate_state(updated,target),(updated['v4_share_gate'],None))
            self.assertEqual(read(run/updated['source_transitions'][-1]['backup']),original)
        for key in ('learning_rate','exploration'):
            bad=copy.deepcopy(signature);bad['config'][key]*=1.1
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
        bad=copy.deepcopy(state);bad['v4_share_gate']['config']='0'*64
        with self.assertRaises(ValueError):gate.migrate_state(bad,target)

    def test_gate_only_change_keeps_PPO_GAE_evidence_and_both_optimizers(self):
        from test_gae_value_phase import ppo,finish
        model,opt,cfg,(behavior,data)=fixture()
        cfg=gate.configure(upgraded(cfg));cfg['v4_share_gate']=copy.deepcopy(gate.LEGACY_DEFAULT)
        phase=TrainingPhases(model,cfg)
        update=ppo(model,opt,cfg,behavior,data,phase);phase.fit(model);finish(phase,update,1)
        saved=phase.payload();weights=copy.deepcopy(model.state_dict())
        adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
        resumed=TrainingPhases(model,gate.configure(cfg),raw=dict(training={KEY:saved}),generation=1)
        self.assertTrue(_same_payload(resumed.payload(),saved))
        self.assertTrue(_same_payload(model.state_dict(),weights))
        self.assertTrue(_same_payload(opt.state_dict(),adam))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(resumed.disable_migration);self.assertIsNone(resumed.holdout_migration)

    def test_holdout_transition_preserves_optimizers_lambda_weights_and_exact_restart(self):
        model,adam,cfg,_=fixture();cfg=disabled_options(upgraded(cfg))
        cfg['value_phase']['holdout']=copy.deepcopy(holdout.LEGACY_DEFAULT)
        phase=TrainingPhases(model,cfg)
        from brass_ai.recovery import gae
        phase.state=gae.initial(cfg['advantage'],210)
        phase.state.update(active_lambda=.95,stage_origin=206,value_iterations=30,ev_streak=4,healthy_streak=4,
            table_enabled=dict(historical=True,legacy=True),table_origins=dict(historical=100,legacy=110))
        phase.critic_state.update(generation=210,snapshots_active=True,snapshot_weight=.75,
            activation_generation=15,learner_ev_baseline=.2)
        from brass_ai.recovery.critic_data import recovery_initial
        phase.critic_state['weight_recovery']=recovery_initial(0)  # Pre-R2 checkpoint.
        phase.critic_state['weight_recovery']['healthy_streak']=3
        saved=phase.payload();target=holdout.quota_options(cfg)
        weights=copy.deepcopy(model.state_dict());opt=copy.deepcopy(adam.state_dict());rng=torch.get_rng_state().clone()
        resumed=TrainingPhases(model,target,raw=dict(training={KEY:saved}),generation=210)
        self.assertTrue(_same_payload(resumed.optimizer.state_dict(),saved['optimizer']))
        self.assertTrue(_same_payload(model.state_dict(),weights));self.assertTrue(_same_payload(adam.state_dict(),opt))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(resumed.state['active_lambda'],.95);self.assertEqual(resumed.state['stage_origin'],206)
        self.assertEqual(resumed.state['value_iterations'],30)
        self.assertEqual(resumed.state['table_origins'],phase.state['table_origins'])
        self.assertEqual(resumed.state['table_evidence']['origin'],210)
        self.assertEqual(resumed.state['stage_evidence']['entries'],[])
        self.assertEqual(resumed.critic_state['snapshot_weight'],.75)
        self.assertEqual(resumed.critic_state['weight_recovery']['healthy_streak'],0)
        restarted=TrainingPhases(model,target,raw=dict(training={KEY:resumed.payload()}),generation=210)
        self.assertTrue(_same_payload(resumed.payload(),restarted.payload()))
        self.assertIsNone(restarted.holdout_migration)
        # Also compose directly from the pre-PPO-only v7 g210 checkpoint.
        ppg=copy.deepcopy(saved);ppg['contract']='actor-value-phases/7.0.0'
        ppg.pop('critic_evidence');ppg['options']['value_phase']['enabled']=True
        combined=TrainingPhases(model,target,raw=dict(training={KEY:ppg}),generation=210)
        self.assertIsNotNone(combined.disable_migration);self.assertIsNotNone(combined.holdout_migration)
        self.assertEqual(combined.state['active_lambda'],.95)
        self.assertEqual(combined.critic_evidence['origin'],210)
        self.assertEqual(combined.state['value_iterations'],0)
        self.assertTrue(_same_payload(combined.optimizer.state_dict(),saved['optimizer']))

    def test_reviewed_export_and_PPO_only_continuation_reject_unrelated_changes(self):
        state=read(EXPORT/'state.json');before=copy.deepcopy(state)
        signature=dict(state['signature'],config=config(),implementation=implementation_id())
        self.assertTrue(check_signature(state,signature,reviewed=True))
        previous=copy.deepcopy(state);previous['signature'].update(
            config=disabled_options(state['signature']['config']),
            implementation='fd908ff8b5e073b2744b8a034dd7304a2803a5c6ad1d75dd9e0be6c67640fd08')
        self.assertTrue(check_signature(previous,signature,reviewed=True))
        self.assertEqual(state,before)
        for key in ('learning_rate','exploration'):
            bad=copy.deepcopy(signature);bad['config'][key]*=1.1
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)


if __name__=='__main__':unittest.main()
