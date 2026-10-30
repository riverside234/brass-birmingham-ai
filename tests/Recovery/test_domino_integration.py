"""Collector, fingerprint, adoption and unchanged-loss integration contracts."""
import copy
import importlib.util
from collections import Counter
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint
from brass_ai.learning.features import Encoded
from brass_ai.recovery import domino_config as C,domino_sampling as A,domino_state as S,domino_adoption as adoption
from brass_ai.recovery.domino_auxiliary import Diversity
from brass_ai.recovery.domino_fingerprints import Fingerprints,inventory
from brass_ai.recovery import quality_pool as Q,pfsp_pool as P,updates
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from brass_ai.step5.policy import context_for
from test_domino import fixture,collect
from test_pfsp_pool import pool
from test_gae_value_phase import examples


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_native_branch_decimal_roundtrip_is_bounded_not_relabelled(self):
        from brass_ai.rust_backend.lab_families import same_fork_evidence
        old=dict(chosen=1,seed='fixed',probability=.2,raw_probabilities=[.023413803559246306,.9765861964407537])
        wire=copy.deepcopy(old);wire['raw_probabilities'][0]=.023413803559246302
        self.assertTrue(same_fork_evidence(wire,old));self.assertEqual(old['raw_probabilities'][0],.023413803559246306)
        wire['probability']+=1e-15;self.assertFalse(same_fork_evidence(wire,old));wire['probability']=old['probability']
        wire['raw_probabilities'][0]+=.000001;self.assertFalse(same_fork_evidence(wire,old))

    def test_all_neural_seats_and_same_weights_remain_separate_roles(self):
        from brass_ai.recovery.execution import neural_spec
        from brass_ai.recovery.critic_data import planned_table,check_table
        from brass_ai.recovery.replay import CheckedGame
        learner=neural_spec('same',.08,'mixed');frozen=neural_spec('same',.08,'neutral')
        fp=dict(values=[1/8]*16+[1/6]*12);provider=Mock();provider.get.return_value=fp
        for rotation in range(4):
            ps=[learner,learner,frozen,dict(kind='legacy',policy_id='v4')]
            receipt=A.assignments(ps,rotation,seed=12,index=50,learner_id=learner['policy_id'],current='same',
                checkpoints={'same':'1'*64},learner_checkpoint='2'*64)
            table=planned_table(ps,rotation,'same',100,[dict(model_id='same',generation=0,fingerprint=fp)],fp,fp,
                learner_policy_id=learner['policy_id'],strategies=receipt,context_fingerprints=provider)
            records=[];policies=[];rngs=[random.Random(i) for i in range(4)];assigned=[]
            for seat,a in enumerate(receipt['seats']):
                spec=ps[(seat-rotation)%4];policies.append(spec['policy_id'])
                assigned.append(SimpleNamespace(kind='model' if spec['kind']=='neural' else 'external',
                    policy_id=spec['policy_id'],evaluator=SimpleNamespace(model_id='same'),contexts={}))
                if a is None:continue
                style,intent=context_for(a['context'])
                records.append(dict(seat=seat,style=style,intent=intent,teacher=None,behavior=dict(kind='model',policy_id=spec['policy_id'])))
                self.assertEqual(a['checkpoint_sha256'],'2'*64 if a['training_role']=='learner' else '1'*64)
                self.assertEqual(a['physical_seat'],seat)
            proof=dict(diversity_assignment=receipt,ordinary_game_index=50,learner_behavior=dict(model_id='same',policy_id=learner['policy_id']))
            g=CheckedGame(opponents=policies,records=records);A.bind(g,proof);check_table(table,g,learner['policy_id'])
            self.assertEqual(sum(A.learner_row(dict(r,_domino=receipt['seats'][r['seat']]),learner['policy_id']) for r in records),2)
            A.install(dict(provenance=proof),assigned,rngs)
            for seat,a in enumerate(receipt['seats']):
                if a is not None:self.assertEqual(assigned[seat].contexts[rngs[seat]],context_for(a['context']))
            bad=copy.deepcopy(g);bad['records'][0]['intent']=[1.]*8
            with self.assertRaisesRegex(ValueError,'context'):A.bind(bad,proof)

    def test_planner_off_parity_pfsp_draws_rotation_and_restart(self):
        from brass_ai.recovery.ordinary import plan_collection
        from brass_ai.recovery.panel_gates import initial
        c=read(SOURCE_ROOT/'configs/step5/joint7-latest.json');value=pool(4,generation=85)
        fp=dict(values=[1/8]*16+[1/6]*12)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);path=root/'weights.pt';path.write_bytes(b'planner fixture')
            for m in value['members']:m.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),fingerprint=fp)
            P.seal(value)
            state=dict(iteration=85,games_done=27200,next_seed=c['seed'],context_counter=81600,quality_pool=value,
                population=[Q.reference(m) for m in value['members']],history=[Q.reference(value['members'][0])],
                evaluation_anchor=Q.reference(value['members'][0]),ratings=dict(panel_gates=initial(85)))
            service=Mock();service.models={m['model_id']:None for m in value['members']}
            bots={str(i):dict(kind='legacy',policy_id=str(i),search_free=True) for i in range(5)}
            provider=Mock();provider.get.return_value=fp
            def plan(cfg):return plan_collection(SOURCE_ROOT,root,state,cfg,dict(config=cfg),service,'current',list(bots),bots,bots,None,
                fingerprints=dict(learner=fp,rule_bot=fp,contexts=provider))
            legacy=plan(c);off=plan(dict(c,diversity=C.defaults('off')))
            # The signature binds an explicit off block but execution stays exact.
            for a,b in zip(legacy['jobs'],off['jobs']):
                a=copy.deepcopy(a);b=copy.deepcopy(b);a['provenance'].pop('spec');b['provenance'].pop('spec')
                self.assertEqual(a,b)
            enabled=dict(c,diversity=C.defaults());before=copy.deepcopy(state);new=plan(enabled);again=plan(enabled)
            self.assertEqual(new,again);self.assertEqual(state,before);self.assertEqual(new['context_counter'],81600)
            self.assertEqual(new['composition'],legacy['composition']);self.assertEqual(new['quality_planning'],legacy['quality_planning'])
            for old,j in zip(legacy['jobs'],new['jobs']):
                self.assertEqual(j['policies'],old['policies']);self.assertEqual(j['seed'],old['seed']);self.assertEqual(j['rotation'],old['rotation'])
                self.assertEqual(sum(a is not None for a in j['provenance']['diversity_assignment']['seats']),2 if j['provenance']['kind']=='legacy' else 4)
            self.assertTrue(any(o['strategy_id']!=0 for t in new['quality_tables'].values() for o in t['opponents']))

    def test_context_fingerprints_reuse_neutral_and_reject_corruption(self):
        from brass_ai.recovery.critic_data import fingerprint
        from brass_ai.recovery.il_kl_probe import Probe
        m,_,c=fixture();_,data=examples(m,games=2,decisions=2)
        encoded=[];rows=[];offsets=[0]
        for r,_,_,e in data:
            state=e.state.copy();state[-12:]=context_for('neutral')[0]+context_for('neutral')[1]
            encoded.append(Encoded(state,e.actions,e.stable_to_relative,e.choice_ids))
            rows.append(dict(phase='action',era='CanalEra',kinds=[v['kind'] for v in r['choices']],industries=[v.get('industry') for v in r['choices']],chosen=r['chosen']))
            offsets.append(offsets[-1]+len(e.actions))
        probe=SimpleNamespace(metadata=dict(probe_hash='authored-fixed'),rows=rows,encoded=encoded,
            arrays=dict(offsets=np.array(offsets),reference=np.zeros(offsets[-1],dtype=np.float32)),parts=lambda update:[list(range(len(rows)))])
        neutral=fingerprint(probe,Probe.probabilities(probe,m,c['update']))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            before=torch.get_rng_state().clone();f=Fingerprints(SOURCE_ROOT,Path(tmp),probe,m,c['update'],{},neutral=neutral)
            mutable=copy.deepcopy(m);changed=Fingerprints(SOURCE_ROOT,Path(tmp)/'changed',probe,mutable,c['update'],{})
            with torch.no_grad():next(mutable.parameters()).add_(.001)
            with self.assertRaisesRegex(ValueError,'weights changed'):changed.get(changed.current,1)
            neutral_result=f.get(model_id(m),0);self.assertEqual(neutral_result['values'],neutral['values']);self.assertEqual(f.report['misses'],0)
            for z in range(1,7):f.get(model_id(m),z)
            self.assertEqual(f.report['misses'],6);self.assertEqual(len(f.models),1)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            again=Fingerprints(SOURCE_ROOT,Path(tmp),probe,m,c['update'],{})
            self.assertEqual(again.get(model_id(m),3),f.get(model_id(m),3));self.assertEqual(again.report['misses'],0)
            path=again.directory/(digest(again.identity(model_id(m),3))+'.json');value=read(path);value['identity']['strategy_id']=4;write_json(path,value)
            with self.assertRaisesRegex(ValueError,'cache changed'):Fingerprints(SOURCE_ROOT,Path(tmp),probe,m,c['update'],{}).get(model_id(m),3)
            value['identity']['strategy_id']=3;value['fingerprint']['context_identity']['strategy_id']=4
            value['seal']=digest(value['fingerprint']);write_json(path,value)
            with self.assertRaisesRegex(ValueError,'context/probe'):Fingerprints(SOURCE_ROOT,Path(tmp),probe,m,c['update'],{}).get(model_id(m),3)

    def test_statistics_regime_bootstraps_without_rebaselining_other_controls(self):
        c=C.defaults();s=S.initial(c,1326,'old');s['quality']=[1.]*7;s['logits']=[.1]*6
        n=S.change_regime(s,c,'new');self.assertEqual(n['quality'],[None]*7);self.assertEqual(n['logits'],s['logits'])
        self.assertEqual(n['adam'],s['adam']);self.assertEqual(n['generation'],1326);self.assertEqual(n['archive'][0]['quality'],[1.]*7)
        self.assertIs(S.change_regime(n,c,'new'),n);self.assertEqual(S.coefficients(n,c).tolist(),[1.]*7)

    def test_adoption_archives_pfsp_and_only_resets_replay_manifest(self):
        m,o,c=fixture();old=pool(generation=1326);old['members'][1].update(S=99.,C=100.);P.seal(old)
        new,receipt=adoption.pool_transition(old)
        for a,b in zip(old['members'],new['members']):
            self.assertEqual({k:v for k,v in a.items() if k not in ('S','C','alarm_streak')},
                {k:v for k,v in b.items() if k not in ('S','C','alarm_streak')})
            self.assertEqual({k:b[k] for k in ('S','C','alarm_streak')},P.prior(old['config']))
        self.assertEqual(receipt['archived_scores'][old['members'][1]['model_id']]['S'],99.)
        e=dict(model_id=model_id(m),sha256='1'*64,generation=1326);state=dict(iteration=1326,history=[e])
        cfg=C.configure({k:v for k,v in c.items() if k!='diversity'},state)
        saved=dict(cache_origin=1322,rolling_buffer=[dict(generation=1325)],optimizer=dict(moments='unchanged'),beta_clone=4.,gae={'lambda':.95})
        raw=dict(training={KEY:saved});before=copy.deepcopy(raw)
        rewritten,r=adoption.prepare(SOURCE_ROOT,SOURCE_ROOT/'artifacts',state,raw,cfg,raw)
        self.assertEqual(raw,before);self.assertEqual(rewritten['training'][KEY],dict(saved,cache_origin=1326,rolling_buffer=[]))
        state['domino_adoption']=r
        self.assertEqual(adoption.prepare(SOURCE_ROOT,SOURCE_ROOT/'artifacts',state,raw,cfg,raw),(rewritten,r))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);oldcache=run/'old-owned-cache';oldcache.write_bytes(b'old')
            adoption.archive(run,r);adoption.archive(run,r);self.assertTrue(oldcache.exists())
        with self.assertRaisesRegex(ValueError,'bound full checkpoint'):
            adoption.prepare(SOURCE_ROOT,SOURCE_ROOT/'artifacts',dict(state,iteration=1327),raw,cfg,raw)

    def test_inventory_verifies_complete_inference_contract_and_rng(self):
        m,o,c=fixture()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'weights.pt';save_checkpoint(path,m,o);mid=model_id(m)
            state=dict(quality_pool=dict(members=[dict(model_id=mid,path=str(path),sha256=sha(path))]))
            before=torch.get_rng_state().clone();self.assertEqual(list(inventory(SOURCE_ROOT,state)),[mid]);self.assertTrue(torch.equal(before,torch.get_rng_state()))
            state['quality_pool']['members'][0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'unavailable checkpoint'):inventory(SOURCE_ROOT,state)

    def test_adoption_cache_expiry_waits_for_new_state_commit(self):
        from test_ppg_eight import stage_only
        from brass_ai.recovery import gae,snapshot_r2
        m,o,c=fixture();old={k:v for k,v in c.items() if k!='diversity'}
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,old,root=SOURCE_ROOT,run=run)
            stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            p.state=gae.initial(old['advantage'],1);p.critic_state['generation']=1
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
            p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            raw=dict(model_id=model_id(m),step=1,model_config=m.config.to_dict(),training={KEY:copy.deepcopy(p.payload())})
            state=dict(iteration=1,history=[dict(generation=1,model_id=model_id(m),sha256='1'*64)])
            new=C.configure(old,state);rewritten,_=adoption.prepare(SOURCE_ROOT,run,state,raw,new,raw)
            hashes={f.name:sha(f) for f in p.buffer.directory.iterdir()}
            q=TrainingPhases(m,new,raw=rewritten,generation=1,root=SOURCE_ROOT,run=run)
            q.diversity=Diversity(m,new['diversity'],raw=raw,generation=1,regime='r')
            self.assertIsNone(q.startup(1));self.assertEqual(q.buffer.receipts,[])
            self.assertTrue(_same_payload(p.optimizer.state_dict(),q.optimizer.state_dict()))
            self.assertEqual(hashes,{f.name:sha(f) for f in p.buffer.directory.iterdir()})
            _,rows=collect(m,q,2,games=42);ds=updates.Dataset(rows,'cpu');q.prepare(ds)
            actor=[i for i in range(len(rows)) if len(rows[i][3].choice_ids)>1]
            q.buffer.stage(q.critic_dataset,q.train_ids,q.heldout_ids,actor,q.collection_source,m,q.split['selection'])
            with self.assertRaisesRegex(ValueError,'checkpoint/state commit'):q.buffer.committed(2)
            self.assertTrue(q.buffer.path(1,'critic.npy').exists())
            write_json(run/'state.json',dict(iteration=2));q.buffer.committed(2)
            self.assertFalse(q.buffer.path(1,'critic.npy').exists());self.assertTrue(q.buffer.path(2,'domino.json').exists())

    def test_shadow_ppo_ppg_match_existing_losses_and_optimizers(self):
        from brass_ai.recovery.utility import reward_spec
        m,o,c=fixture();c['diversity']['mode']='shadow';other=copy.deepcopy(m)
        oo=torch.optim.AdamW(other.parameters());oo.load_state_dict(copy.deepcopy(o.state_dict()))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp)/'shadow');p.diversity=Diversity(m,c['diversity'],regime='r')
            base={k:v for k,v in c.items() if k!='diversity'};q=TrainingPhases(other,base,root=SOURCE_ROOT,run=Path(tmp)/'base')
            b,rows=collect(m,p,1,games=42);q.snapshot_data=copy.deepcopy(p.snapshot_data);q.collection_source=p.collection_source
            rng=torch.get_rng_state().clone();reward=reward_spec(c)
            def run(net,opt,phases):
                updates.ppo_update(net,opt,[dict(split='train')],b,c['update'],learner_data=copy.deepcopy(rows),reward=reward,
                    insolvency=reward['insolvency'],phases=phases)
                return phases.fit(net)
            a=run(m,o,p);after=torch.get_rng_state().clone();torch.set_rng_state(rng);z=run(other,oo,q)
            self.assertTrue(_same_payload((m.state_dict(),o.state_dict(),p.optimizer.state_dict(),after),
                (other.state_dict(),oo.state_dict(),q.optimizer.state_dict(),torch.get_rng_state())))
            self.assertEqual(a['optimizer_steps'],z['optimizer_steps']);self.assertFalse(p.diversity.steps)

    def test_g1326_helper_is_generation_bound_readonly_and_strict(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        spec=importlib.util.spec_from_file_location('domino_helper',SOURCE_ROOT/'scripts/configure-domino.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z/state.json')
        m,o,_=fixture()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/run';run.mkdir(parents=True)
            checkpoint=run/'generation-1326.pt';save_checkpoint(checkpoint,m,o,step=1326)
            e=dict(model_id=model_id(m),generation=1326,path='artifacts/run/generation-1326.pt',sha256=sha(checkpoint))
            state['history'][-1]=e;state['quality_pool']['members']=[e];write_json(run/'state.json',state)
            before=(sha(run/'state.json'),sha(checkpoint));output='configs/step5/domino-g1326.json'
            receipt=helper.generate(root,run,output,1326)
            self.assertEqual(receipt,helper.generate(root,run,output,1326));self.assertFalse(receipt['run_modified'])
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
            new=read(root/output);self.assertEqual({k:v for k,v in new.items() if k!='diversity'},state['signature']['config'])
            self.assertEqual(new['diversity']['adoption']['checkpoint_sha256'],e['sha256'])
            with self.assertRaisesRegex(ValueError,'Stale expected generation'):helper.generate(root,run,output,1324)
            bad=copy.deepcopy(new);bad['exploration']=.01
            with self.assertRaises(ValueError):check_signature(state,dict(state['signature'],implementation=implementation_id(),config=bad),reviewed=True)
            from brass_ai.recovery.exploration import fixed_010_config,FIXED_ENTROPY_010
            high_output='configs/step5/domino-g1326-entropy010.json'
            high=helper.generate(root,run,high_output,1326,entropy_010=True)
            self.assertEqual(high,helper.generate(root,run,high_output,1326,entropy_010=True))
            self.assertEqual(read(root/high_output),fixed_010_config(new))
            self.assertEqual(high['fixed_entropy'],FIXED_ENTROPY_010)
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
            with self.assertRaisesRegex(ValueError,'Existing output differs'):
                helper.generate(root,run,output,1326,entropy_010=True)
            changed=read(root/high_output);changed['branching']['rate_per_decision']=.015
            with self.assertRaises(ValueError):
                check_signature(state,dict(state['signature'],implementation=implementation_id(),config=changed),reviewed=True)
            checkpoint.write_bytes(b'changed checkpoint')
            with self.assertRaisesRegex(ValueError,'full checkpoint'):helper.generate(root,run,output,1326)

    def test_existing_domino_source_update_needs_no_readoption_or_config_reset(self):
        from brass_ai.recovery.resume import check_signature
        from brass_ai.step5.league import implementation_id
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z/state.json')
        state['signature']['config']=C.configure(state['signature']['config'],state)
        state['signature']['implementation']='cd3943e2ea84d436fb009fa3bce936c84115aa5bcb5b360b0fb3be7b97e31cc1'
        state['iteration']=1327
        before=copy.deepcopy(state);target=dict(state['signature'],implementation=implementation_id())
        self.assertTrue(check_signature(state,target,reviewed=True));self.assertEqual(state,before)
        with self.assertRaisesRegex(ValueError,'resume-reviewed-source'):check_signature(state,target)
        changed=copy.deepcopy(target);changed['config']['exploration']=.079
        with self.assertRaises(ValueError):check_signature(state,changed,reviewed=True)


if __name__=='__main__':unittest.main()
