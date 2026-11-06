"""g1041 recipe, exact planning/continuation and bounded two-age PPG smoke."""
import copy
import importlib.util
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import ppg_g210 as P,quality_pool as Q,pfsp_pool as F,v4_history as H,v4_share as V
from brass_ai.recovery import exploration as E,lr_controller as LR
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.training_revision_g1041 import configure,exposure
from brass_ai.recovery.resume import check_signature,load_resume,commit_transition,apply_optimizer_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id,write_json

BUNDLE=SOURCE_ROOT/'results/joint7-v11-2-g1290-trained-restart-g1041-20261106T081744641630Z'


class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original=read(BUNDLE/'state.json');cls.old=cls.original['signature']['config']
        cls.new=validate_config(configure(cls.old,cls.original))

    def test_exact_scope_versions_and_rejection(self):
        self.assertEqual(configure(self.new,self.original),self.new)
        self.assertEqual({k for k in self.old if self.old[k]!=self.new[k]},
            {'value_phase','curriculum','table_plan','update','rating_compute'})
        self.assertEqual(self.new['update'],dict(self.old['update'],kl_control=self.new['update']['kl_control']))
        self.assertEqual(self.new['update']['kl_control']['probe_check_steps'],4)
        self.assertFalse(self.new['update']['kl_control']['full_dataset_kl'])
        self.assertEqual(self.new['rating_compute']['workers'],16)
        self.assertEqual(self.new['value_phase']['kl_check_steps'],32)
        self.assertEqual(self.new['table_plan']['origin_games'],333120)
        for section,key,value in [('value_phase','kl_cap',.004),('table_plan','origin_games',0),
                                  ('table_plan','pair_rotation','random')]:
            c=copy.deepcopy(self.new);c[section][key]=value
            with self.assertRaises(ValueError):validate_config(c)
        bad=copy.deepcopy(self.new);bad['value_phase']['early_kl_beta']['retained_steps_at_most']=16.
        with self.assertRaises(ValueError):validate_config(bad)
        state=copy.deepcopy(self.original);state['quality_pool']['v4_history']['activated']=None
        with self.assertRaisesRegex(ValueError,'activation'):configure(self.old,state)

    def test_beta_retained_boundary_priority_clamp_and_legacy(self):
        c=self.new['value_phase']
        def run(steps,trip=True,beta=1.,trips=(),kl=0.):
            return P.beta_after_phase(beta,list(trips),dict(optimizer_steps=steps,attempted_steps=32,
                kl_guard_tripped=trip,probe_kl=kl),c)
        for steps in (0,1,16):
            b,_,r=run(steps);self.assertEqual(b,1.5);self.assertTrue(r['early_kl_trigger'])
        self.assertEqual(run(17)[0],1.)
        self.assertEqual(run(0,False)[0],1.)
        self.assertEqual(run(16,False,kl=.0015)[0],1.)
        self.assertEqual(run(16,False,kl=.001499)[0],.5)
        self.assertEqual(run(16,False,beta=.25,kl=0.)[0],.25)
        self.assertEqual(run(16,beta=30.)[0],32.)
        self.assertEqual(run(16,trips=[True,True])[0],2.)
        self.assertEqual(run(17,trips=[True,True])[0],2.)
        self.assertEqual(P.beta_after_phase(1.,[],dict(kl_guard_tripped=True,optimizer_steps=0),self.old['value_phase'])[0],1.)
        b,trips,_=run(16)
        restored=read_json_roundtrip(dict(beta=b,trips=trips))
        self.assertEqual(run(0,beta=b,trips=trips),run(0,beta=restored['beta'],trips=restored['trips']))

    def test_ordinals_and_actual_planner_sampling_counts(self):
        from brass_ai.recovery.ordinary import plan_collection
        from brass_ai.recovery.history import compact
        old=self.old;new=copy.deepcopy(self.new);new['local_scoring_auxiliary']['mode']='off'
        state=copy.deepcopy(self.original);origin=state['games_done'];share=state['v4_share_gate']
        for index in (0,320,origin-1,origin):
            self.assertEqual(V.history_before(index,new,None,share),V.history_before(index,old,None,share))
        self.assertEqual(V.history_before(origin+320,new,None,share)-V.history_before(origin,new,None,share),96)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            path=Path(tmp)/'weights.pt';path.write_bytes(b'planner-only')
            for m in state['quality_pool']['members']:
                if not H.bot(m):m.update(path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path))
            F.seal(state['quality_pool'])
            state['population']=[Q.reference(m) for m in state['quality_pool']['members'] if not H.bot(m)]
            state['signature']['config']=new
            service=Mock();service.models={m['model_id']:None for m in state['population']}
            values=state['quality_pool']['v4_history']['policies']
            settings=E.effective(new,E.restore(new,state,state['exploration']),1041)
            def plan():return plan_collection(SOURCE_ROOT,Path(tmp),state,new,state['signature'],service,'current',
                sorted(values),values,values,None,exploration=settings)
            before=copy.deepcopy(state);result=plan();self.assertEqual(result,plan());self.assertEqual(state,before)
            self.assertEqual(result['composition'],dict(selfplay=224,historical=96))
            pairs=Counter(tuple(t['learner_seats']) for t in result['quality_tables'].values())
            self.assertEqual(sorted(pairs.values()),[16]*6)
            expected=F.distribution(state['quality_pool']);self.assertEqual(result['quality_planning'],expected)
            self.assertAlmostEqual(sum(r['p'] for r in expected['members']),1.)
            mix=result['table_mix'];self.assertEqual(mix['learner_seats'],dict(selfplay=896,historical=192,legacy=0))
            self.assertAlmostEqual(sum(r['expected_opponent_seats'] for r in mix['members']),192.)
            self.assertAlmostEqual(sum(r['all_seat_probability'] for r in mix['members']),.15)
            games=[]
            for job in result['jobs']:
                group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
                games.append(dict(setup_group=group,opponents=[job['policies'][(s-job['rotation'])%4]['policy_id'] for s in range(4)],result={}))
            with patch('brass_ai.learning.replay.terminal_targets',return_value=dict(placement=[1,2,3,4])):
                pool,report=Q.update_scores(state['quality_pool'],games,result['quality_tables'],expected,
                    generation=1042,games_done=origin+320)
            self.assertEqual(report['exposure']['opponent_seats'],192)
            self.assertEqual(report['exposure']['pairwise_comparisons'],384)
            self.assertEqual(report['exposure']['history_table_fraction'],.3)
            self.assertAlmostEqual(sum(r['p'] for r in pool['next_planning']['members']),1.)
            report['table_mix']=mix
            self.assertEqual(compact(dict(quality_pool=report))['quality_pool']['table_mix'],mix)

    def test_LR_resets_evidence_for_actual_volume_not_multiplier(self):
        def sampling(a,b):return dict(by_composition=dict(selfplay=dict(tables=a,learner_trajectories=4*a),
            historical=dict(tables=b,learner_trajectories=2*b),legacy=dict(tables=0,learner_trajectories=0)))
        before=LR.collection_signature(self.old,sampling(166,154));after=LR.collection_signature(self.new,sampling(224,96))
        self.assertNotEqual(before,after)
        state=LR.initial_state(self.new['ppo_lr_controller']);state.update(multiplier=1.5,data_signature=before,
            window=[dict(kl=.001,early_stop=False)]*2)
        m=dict(kl=.007,steps_done=48,steps_planned=48,early_stop=False,warmup_done=True,data_signature=after,
            economic_health=dict(contract='selfplay-economic-health/3.0.0',ok=True))
        next_state,r=LR.decide(state,m)
        self.assertEqual(next_state['multiplier'],1.5);self.assertEqual(len(next_state['window']),1)
        self.assertEqual(next_state['data_signature'],after)

    def test_real_checkpoint_and_preparation_are_retry_safe(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.learning.checkpoint import save_checkpoint,resume_optimizer
        from brass_ai.recovery.utility import migrate,reward_spec
        spec=importlib.util.spec_from_file_location('g1041_helper',SOURCE_ROOT/'scripts/configure-training-g1041.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);state=copy.deepcopy(self.original);old=state['signature']['config'];old['model_size']='small'
            model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
            migrate(model,opt,reward_spec(old));apply_optimizer_config(opt,old)
            p=next(model.parameters());p.grad=torch.ones_like(p);opt.step();opt.zero_grad(set_to_none=True)
            model.utility_calibration=dict(ready=True,reward_id=digest(reward_spec(old)))
            path=run/'generation-1041.pt';identity=save_checkpoint(path,model,opt,step=1041,
                training=dict(fresh_run=run.relative_to(SOURCE_ROOT).as_posix(),signature=digest(state['signature']),
                              exploration=state['exploration']))
            entry=dict(generation=1041,path=path.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(path),model_id=identity)
            state['history']=[entry];state['population']=[entry];state['evaluation_anchor']=entry
            write_json(run/'state.json',state);before=(run/'state.json').read_bytes();checksum=sha(path)
            output=SOURCE_ROOT/'configs/step5'/('test-g1041-'+run.name+'.json');self.addCleanup(lambda:output.unlink(missing_ok=True))
            args=(SOURCE_ROOT,str(run),str(output),1041)
            a=helper.generate(*args);self.assertEqual(a,helper.generate(*args));self.assertEqual((run/'state.json').read_bytes(),before)
            new=read(output);signature=dict(state['signature'],config=new,implementation=implementation_id())
            with self.assertRaises(ValueError):check_signature(state,signature)
            bad=copy.deepcopy(signature);bad['config']['learning_rate']*=2
            with self.assertRaises(ValueError):check_signature(state,bad,reviewed=True)
            saved=torch.load(path,map_location='cpu',weights_only=True)
            for retry in range(2):
                model,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=retry==0)
                opt=torch.optim.AdamW(model.parameters(),lr=new['learning_rate']);resume_optimizer(raw,opt);apply_optimizer_config(opt,new)
                self.assertTrue(_same_payload(saved['state_dict'],dict(model.state_dict())))
                self.assertTrue(_same_payload(saved['optimizer'],opt.state_dict()))
                self.assertTrue(torch.equal(saved['rng'],torch.get_rng_state()))
                self.assertEqual(model.utility_calibration,saved['training']['utility_calibration'])
                self.assertEqual(sha(path),checksum)
                if retry==0:state=commit_transition(run,state,signature)
            self.assertEqual(a,helper.generate(*args))
            (run/'iteration-1041').mkdir();(run/'iteration-1041/partial').write_text('pending')
            with self.assertRaisesRegex(ValueError,'interrupted'):helper.generate(*args)

    def test_cache_adoption_and_isolated_PPG_smoke(self):
        from brass_ai.learning.model import FourSeatNet,ModelConfig
        from brass_ai.recovery.utility import migrate,reward_spec
        from brass_ai.recovery.actor_value import TrainingPhases,KEY
        from brass_ai.recovery.local_scoring_auxiliary import Companion
        from brass_ai.learning.batching import model_id
        from brass_ai.recovery import gae
        from test_ppg_g210 import prepare
        from test_gae_value_phase import finish
        from test_value_buffer import commit as commit_cache
        old=copy.deepcopy(self.old);old['model_size']='small';old['update']['preparation_workers']=0
        old['value_phase'].update(batch_size=128,clone_rows=24,clone_batch_size=8)
        model=FourSeatNet(ModelConfig.preset('small'));opt=torch.optim.AdamW(model.parameters(),lr=old['learning_rate'])
        migrate(model,opt,reward_spec(old))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);phase=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
            phase.local_scoring=Companion(model,old['local_scoring_auxiliary'],raw=dict(model_id=model_id(model)))
            prepare(model,phase,1,companion=True)
            phase.buffer.stage(phase.critic_dataset,phase.train_ids,phase.heldout_ids,phase.actor_ids,
                phase.collection_source,model,phase.split['selection'])
            phase.state=gae.initial(old['advantage'],1);phase.critic_state['generation']=1;phase.beta=.5;phase.trips=[True,False]
            raw=dict(model_config=model.config.to_dict(),training={KEY:phase.payload(),
                'local_scoring_auxiliary':phase.local_scoring.payload(1)})
            write_json(run/'state.json',dict(iteration=1));phase.buffer.committed(1)
            before=copy.deepcopy((model.state_dict(),opt.state_dict(),phase.optimizer.state_dict(),phase.state,phase.critic_state,torch.get_rng_state()))
            hashes={p.name:sha(p) for p in phase.buffer.directory.iterdir()}
            new=configure(old,self.original);q=TrainingPhases(model,new,raw=raw,generation=1,root=SOURCE_ROOT,run=run)
            self.assertEqual((q.beta,q.trips),(.5,[True,False]));self.assertIsNone(q.startup(1))
            self.assertEqual(q.buffer.receipts,phase.buffer.receipts)
            self.assertEqual(hashes,{p.name:sha(p) for p in phase.buffer.directory.iterdir()})
            self.assertTrue(_same_payload(before,(model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),q.state,q.critic_state,torch.get_rng_state())))
            q.local_scoring=Companion(model,new['local_scoring_auxiliary'],raw=raw,generation=1)
            prepare(model,q,2,companion=True);r=q.fit(model)
            self.assertGreater(r['optimizer_steps'],0);self.assertLessEqual(r['probe_kl'],.003)
            update=dict(categorical_probe=dict(categorical_kl=0.),controller_warmup_done=True,optimizer_steps=1,kl_early_stop=False,gae={})
            finish(q,update,2);payload=q.payload();commit_cache(run,q,2)
            resumed=TrainingPhases(model,new,raw=dict(training={KEY:payload}),generation=2,root=SOURCE_ROOT,run=run)
            self.assertEqual((resumed.beta,resumed.trips),(q.beta,q.trips))
            self.assertEqual([r['generation'] for r in resumed.buffer.receipts],[1,2])
            self.assertTrue(_same_payload(q.optimizer.state_dict(),resumed.optimizer.state_dict()))
            print('g1041 isolated PPG smoke:',{k:r[k] for k in ('optimizer_steps','attempted_steps','probe_kl','kl_cap','beta_clone_next','early_stop_reason')})


def read_json_roundtrip(value):
    import json
    return json.loads(json.dumps(value))


if __name__=='__main__':unittest.main()
