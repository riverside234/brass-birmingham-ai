"""Narrow recipe adoption, actual behavior, cosine PPG and reward-cache restart."""
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch,Mock

import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_g210 as G,exploration as E,opening_objective as O,updates,value_phase as V,gae,snapshot_r2
from brass_ai.recovery.training_revision_g850 import configure
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import reward_spec,migrate
from brass_ai.recovery.telemetry import Timings
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_g210 import config,prepare
from test_ppg_eight import stage_only
from test_opening_ppo import opening_rows
from test_gae_value_phase import fixture,finish
from test_exploration_revision import measured

EXPORT=SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g0850-20261024T181807288965Z'


def smoke_config():
    model,opt,c=config();c=G.cosine_options(c)
    c['update']['opening_objective']=copy.deepcopy(O.T15)
    c['insolvency']=dict(event_cost=.02,max_cost=.075)
    c['search']['reward']['insolvency']=copy.deepcopy(c['insolvency'])
    model.config=replace(model.config,utility=reward_spec(c))
    return model,opt,c


def collect_revised(model,phase,generation,*,branches=False):
    from test_ppg_eight import collect
    from test_value_buffer import source
    _,data=collect(model,phase,generation,branches=branches)
    behavior,data=opening_rows(model,data,.05,opening=E.REVISED_OPENING)
    for row,_,_,_ in data:
        if row['behavior']['temperature']==2.:row['behavior']['temperature']=1.5
        row[O.ROW_KEY]=copy.deepcopy(O.T15)
    ds=updates.Dataset(data,'cpu');model.eval()
    with torch.no_grad():
        for part in ds.microbatches(range(len(data)),32,131072):
            ix,inputs,q=ds.batch(part,.05);prob=updates.row_probabilities(ds,ix,model(*inputs)[0],q,behavior)
            for j,i in enumerate(part):data[i][0]['behavior']['probability']=float(prob[j,data[i][0]['chosen']])
    phase.collection_source=dict(source(behavior),snapshot_rows=True)
    return behavior,data


class Recipe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_recipe_strict_and_unchanged_fields_and_effective_entropy(self):
        state=read(EXPORT/'state.json');old=state['signature']['config'];new=validate_config(configure(old))
        self.assertEqual(configure(new),new)
        changed={k for k in new if old.get(k)!=new[k]}
        self.assertEqual(changed,{'exploration','update','search','insolvency','value_phase','entropy_control','opening','branching','rating_compute'})
        self.assertEqual(new['update'],dict(old['update'],entropy_weight=.005,opening_objective=O.T15))
        restored=E.restore(new,state,state['exploration'])
        self.assertEqual(restored,state['exploration'])
        self.assertEqual(E.effective(new,restored,850)['alphas'],dict.fromkeys(E.GROUPS,.005))
        advanced,report=E.advance(restored,new,measured(.01),generation=851,warmup_done=True,healthy=True)
        self.assertEqual(advanced['baseline'],restored['baseline'])
        for g in E.GROUPS:
            self.assertEqual(advanced['groups'][g]['alpha'],restored['groups'][g]['alpha'])
            self.assertEqual(report['groups'][g]['effective_alpha'],.005)
        for key,value in [('max_epochs',3),('beta_clone_min',1.),('kl_check_steps',32)]:
            with self.assertRaises(ValueError):G.validate(dict(new['value_phase'],**{key:value}))
        bad=copy.deepcopy(new);bad['entropy_control']['mode']='apply'
        with self.assertRaises(ValueError):validate_config(bad)
        self.assertEqual(validate_config(configure(old,rating_workers=8))['rating_compute']['workers'],8)
        from brass_ai.recovery.config import latest_model_config
        fresh=latest_model_config(old,old['imitation_checkpoint'])
        from brass_ai.recovery.training_revision_g1120 import configure_kl32 as newest
        expected=newest(G.short_options(G.linear_options(new)))
        for k in ('value_phase','opening','branching','insolvency','exploration','rating_compute'):
            self.assertEqual(fresh[k],expected[k],k)
        from brass_ai.learning.reward import CostLedger
        ledger=CostLedger(reward_spec(new))
        for expected in (.02,.04,.06,.075,.075):
            ledger.charge(0);self.assertAlmostEqual(float(ledger.spent[0]),expected)

    def test_cosine_and_half_beta_boundaries(self):
        c=G.COSINE_DEFAULT
        rates=[G.rate_factor(i,100,c) for i in range(1,101)]
        self.assertAlmostEqual(rates[0],.1);self.assertEqual(rates[15],1.);self.assertAlmostEqual(rates[-1],.3)
        self.assertEqual(rates[:16],sorted(rates[:16]));self.assertEqual(rates[15:],sorted(rates[15:],reverse=True))
        self.assertLess(G.rate_factor(4,4,c),1.)  # Tiny phases do not compress the 16-step warmup.
        r=dict(kl_guard_tripped=False,optimizer_steps=1,probe_kl=.000999)
        self.assertEqual(G.beta_after_phase(1.,[],r,c)[0],.5)
        self.assertEqual(G.beta_after_phase(.5,[],r,c)[0],.5)
        self.assertEqual(G.beta_after_phase(.5,[True,True],dict(r,kl_guard_tripped=True),c)[0],1.)
        self.assertEqual(G.beta_after_phase(1.,[],dict(r,optimizer_steps=0),c)[0],1.)
        self.assertEqual(G.beta_after_phase(1.,[],dict(r,probe_kl=.001),c)[0],1.)

    def test_t15_mixed_behavior_reconstruction_provenance_and_fork_counts(self):
        m,_,c,(_,data)=fixture(games=3,decisions=3)
        b,data=opening_rows(m,data,.05,opening=E.REVISED_OPENING)
        for row,_,_,_ in data:
            if row['behavior']['temperature']==2.:row['behavior']['temperature']=1.5
            row[O.ROW_KEY]=copy.deepcopy(O.T15)
        ds=updates.Dataset(data,'cpu');m.eval()
        with torch.no_grad():
            for part in ds.microbatches(range(len(data)),32,131072):
                ix,inputs,q=ds.batch(part,.05);probs=updates.row_probabilities(ds,ix,m(*inputs)[0],q,b)
                for j,i in enumerate(part):data[i][0]['behavior']['probability']=float(probs[j,data[i][0]['chosen']])
        ds=updates.Dataset(data,'cpu');ds.opening_reference=O.Reference(ds,b,O.T15)
        updates.policy_measure(m,ds,range(len(data)),b,c['update'],Timings(),verify=True)
        self.assertTrue(ds.opening_reference.frozen);self.assertTrue(ds.opening_reference.mask.any())
        self.assertFalse(ds.opening_reference.old.requires_grad)
        with self.assertRaises(ValueError):O.Reference(ds,b,O.DEFAULT)
        data[0][0]['behavior']['exploration']=0.
        with self.assertRaises(ValueError):updates.verify_behavior_metadata(ds,b)
        from brass_ai.recovery.branching import validate_opening_rows
        rows=[]
        for n in range(3):
            for seat in range(4):
                rows.extend([dict(seat=seat,view=dict(context=dict(phase='bonus')),behavior=dict(temperature=1.,action_decision=None)),
                    dict(seat=seat,view=dict(context=dict(phase='action')),behavior=dict(temperature=1.5 if n<2 else 1.,action_decision=n))])
        validate_opening_rows(rows,E.REVISED_OPENING)
        with self.assertRaises(ValueError):validate_opening_rows(rows)

    def test_reward_cache_transition_keeps_Adam_beta_rng_and_evidence(self):
        m,opt,c=config();c['insolvency']=dict(event_cost=.04,max_cost=.10)
        c['search']['reward']['insolvency']=copy.deepcopy(c['insolvency'])
        m.config=replace(m.config,utility=reward_spec(c))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
            stage_only(m,p,1);write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1)
            p.state=gae.initial(c['advantage'],1);p.critic_state['generation']=1
            p.critic_state['weight_recovery']=snapshot_r2.initial(1,p.critic_state['snapshot_weight'])
            for optimizer in (opt,p.optimizer):
                optimizer.zero_grad(set_to_none=True)
                parameter=next(m.parameters());parameter.grad=torch.zeros_like(parameter);optimizer.step()
            p.beta=2.;p.trips=[True,False];p.buffer.pending=copy.deepcopy(p.buffer.receipts)
            raw=dict(model_config=m.config.to_dict(),training={KEY:p.payload()})
            new=G.cosine_options(c);new['insolvency']=dict(event_cost=.02,max_cost=.075)
            new['search']['reward']['insolvency']=copy.deepcopy(new['insolvency'])
            before=copy.deepcopy((m.state_dict(),opt.state_dict(),p.optimizer.state_dict(),p.state,p.critic_state,torch.get_rng_state()))
            m._reviewed_cap_upgrade=True;migrate(m,opt,reward_spec(new))
            q=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(q.beta,2.);self.assertEqual(q.trips,[True,False]);self.assertEqual(q.buffer.receipts,[])
            self.assertTrue(_same_payload(before,(m.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            write_json(run/'state.json',dict(iteration=1,signature=dict(config=c)))
            with self.assertRaises(ValueError):q.startup(1)
            write_json(run/'state.json',dict(iteration=1,signature=dict(config=new)))
            self.assertTrue(q.startup(1)['deleted_files'])
            again=TrainingPhases(m,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual(again.startup(1)['deleted_files'],[])

    def test_branch_execution_resets_all_seats_with_t15_policies(self):
        from brass_ai.recovery.branching import continue_branch
        context=dict(phase='action',era='CanalEra')
        row=dict(step=25,seat=2,view=dict(context=context),choices=[{'id':0},{'id':1}],chosen=0,
            style=[0.]*4,intent=[0.]*8,
            behavior=dict(probability=.5,component='policy',temperature=1.,action_decision=3))
        meta=dict(contract=E.REVISED_BRANCH['contract'],seed='12345678',decision=25,chosen=1,marginal_probability=.3)
        packet=dict(row=row,metadata=meta,counts=[3,4,4,2],contexts=[{'s':i} for i in range(4)])
        assigned=[SimpleNamespace(contexts={},action_counts={},opening=E.REVISED_OPENING) for _ in range(4)]
        host=Mock();host.call.return_value=context;host.apply.return_value={'terminal':True}
        parent=dict(game_id='parent',setup_group='setup',split='train',opponents=['model']*4)
        def turns(host,job,policies,timing,cap,context,game,rngs,encoded,encoder,lease):
            self.assertNotIn('branching',job)
            for seat,p in enumerate(policies):
                self.assertEqual(p.action_counts[rngs[seat]],0)
                self.assertEqual(p.opening['temperature'],1.5)
            self.assertEqual(game['records'][0]['behavior']['component'],'branch_move')
            return game
        with patch('brass_ai.recovery.execution._run_turns',side_effect=turns),patch('brass_ai.recovery.replay.CheckedGame'):
            continue_branch(host,dict(branching=E.REVISED_BRANCH,provenance={}),parent,packet,assigned,None,2000,Mock(),None)
        self.assertEqual(packet['counts'],[3,4,4,2])

    def test_real_phase_cosine_steps_and_sixteen_step_safe_rollback(self):
        m,opt,c=config();c=G.cosine_options(c);c['value_phase']['batch_size']=2
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            p=TrainingPhases(m,c,root=SOURCE_ROOT,run=Path(tmp));prepare(m,p,1)
            metric=V.metrics;capture={}
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence') and not kw.get('unknown'):
                    step=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    result['loss']=1.-.001*step
                    if step==16:capture.update(model=copy.deepcopy(m.state_dict()),adam=copy.deepcopy(p.optimizer.state_dict()))
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',side_effect=[.001,.003,0.]):r=p.fit(m)
            self.assertEqual(r['attempted_steps'],32);self.assertEqual(r['optimizer_steps'],16)
            self.assertEqual([x['step'] for x in r['kl_checks']],[16,32])
            self.assertTrue(_same_payload(capture['model'],m.state_dict()));self.assertTrue(_same_payload(capture['adam'],p.optimizer.state_dict()))
            peak=r['peak_learning_rate'];rates=r['attempted_learning_rates']
            self.assertAlmostEqual(rates[0],peak*.1);self.assertEqual(rates[15],peak)
            self.assertLess(rates[16],peak);self.assertEqual(r['kl_check_steps'],16)

    def test_two_age_loss_acceptance_and_current_only_gate_evidence(self):
        from test_value_buffer import commit
        m,opt,c=smoke_config()
        # This pure value-phase fixture retains the original authored PPO behavior.
        c['update'].pop('opening_objective',None)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
            prepare(m,p,1);first=p.fit(m)
            update=dict(categorical_probe=dict(categorical_kl=0.),controller_warmup_done=True,optimizer_steps=1,kl_early_stop=False,gae={})
            finish(p,update,1);commit(run,p,1)
            self.assertEqual(first['acceptance_generations'],[1])
            prepare(m,p,2)
            metric=V.metrics;start=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
            seen=[]
            def metrics(net,ds,ids,update,**kw):
                result=metric(net,ds,ids,update,**kw)
                if kw.get('evidence'):
                    step=max((int(s['step']) for s in p.optimizer.state.values()),default=0)
                    pooled=hasattr(ds.data,'parts')
                    if pooled:
                        seen.append((len(ds.data.parts),len(ids)))
                        result['loss']=1. if step==start else 1.1
                    else:result['loss']=1. if step==start else .9
                return result
            with patch.object(V,'metrics',metrics),patch.object(V.RawPolicyReference,'measure',return_value=0.):r=p.fit(m)
            self.assertEqual(r['acceptance_generations'],[1,2]);self.assertEqual(r['validation_iterations'],2)
            self.assertEqual(r['optimizer_steps'],0);self.assertEqual(r['early_stop_reason'],'heldout_loss_increased')
            self.assertTrue(seen);self.assertTrue(all(parts==2 and rows==r['acceptance_heldout_rows'] for parts,rows in seen))
            self.assertEqual(sum(s['rows'] for s in r['acceptance_support']),r['acceptance_heldout_rows'])
            self.assertGreater(r['acceptance_heldout_rows'],r['heldout_rows'])
            self.assertEqual(r['heldout_rows'],len(p.heldout_ids))
            self.assertEqual(r['heldout_after']['games'],len({p.critic_dataset.data[i][2][0] for i in p.heldout_ids}))

    def test_g850_helper_and_review_are_narrow_and_read_only(self):
        spec=importlib.util.spec_from_file_location('g850_helper',SOURCE_ROOT/'scripts/configure-training-g850.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        state=read(EXPORT/'state.json')
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/joint7-v11-2-grouped-lr-critic';run.mkdir(parents=True)
            checkpoint=root/state['history'][-1]['path'];checkpoint.write_bytes(b'isolated-checkpoint-stand-in')
            state['history'][-1]['sha256']=sha(checkpoint);write_json(run/'state.json',state)
            before=(sha(run/'state.json'),sha(checkpoint))
            report=helper.generate(root,run,'configs/step5/g850.json',850)
            self.assertEqual(report,helper.generate(root,run,'configs/step5/g850.json',850))
            self.assertEqual(before,(sha(run/'state.json'),sha(checkpoint)))
            from brass_ai.recovery.resume import check_signature
            signature=dict(state['signature'],implementation=implementation_id(),config=configure(state['signature']['config']))
            self.assertTrue(check_signature(state,signature,reviewed=True))
            signature['config']['learning_rate']*=2
            with self.assertRaises(ValueError):check_signature(state,signature,reviewed=True)
            with self.assertRaises(ValueError):helper.generate(root,run,'configs/step5/stale.json',849)


if __name__=='__main__':unittest.main()
