"""Production PPG guards move companion weights, Adam and retained rows together."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import ppg_auxiliary as A,ppg_config as P,updates
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from test_ppg_eight import setup,collect
from test_local_scoring import annotate


class Rollback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def run_fit(self,trip=None,retry=False):
        model,opt,cfg=setup();cfg=P.coverage_options(cfg);cfg[KEY]=copy.deepcopy(DEFAULT)
        cfg['value_phase'].update(batch_size=128,acceptance_steps=2)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);p=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
            a=Companion(model,cfg[KEY]);a.age=7;p.local_scoring=a
            for g in range(1,9):
                _,data=collect(model,p,g,branches=True);annotate(data);annotate(p.snapshot_data)
                ds=updates.Dataset(data,'cpu');p.prepare(ds)
                actor=[i for i in range(len(data)) if len(data[i][3].choice_ids)>1]
                p.after_ppo(model,actor,actor[:4],{},critic_steps=1,critic_rows=len(p.train_ids))
                p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,actor,p.collection_source,model,p.split['selection'])
                if g<8:write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
            # A nonempty PPO-head Adam must survive every PPG reset and retry.
            ix,inputs,_=ds.value_batch([0,1],model);a.begin('ppo')
            _,loss=a.forward(model,inputs,ds,ix);loss.mean().backward();a.step('ppo');model.zero_grad()
            ppo=copy.deepcopy(a.optimizers['ppo'].state_dict());calls=[];saved={};restores=[]
            step=a.step;probes=A.Reference.probes;snapshot=a.snapshot;restore=a.restore;paired_states=[]
            def steps():return max((int(s['step']) for s in p.optimizer.state.values()),default=0)
            def recording(phase):
                step(phase);calls.append(1)
                if len(calls)==4:saved.update(aux=snapshot(),core=copy.deepcopy(model.state_dict()),adam=copy.deepcopy(p.optimizer.state_dict()))
            def paired_snapshot():
                s=snapshot();paired_states.append((copy.deepcopy(s),copy.deepcopy(model.state_dict()),copy.deepcopy(p.optimizer.state_dict())))
                return s
            def paired_restore(s):
                matches=[r for r in paired_states if _same_payload(r[0],s)]
                self.assertTrue(matches)
                self.assertTrue(any(_same_payload(r[1],model.state_dict()) and _same_payload(r[2],p.optimizer.state_dict()) for r in matches))
                restore(s);self.assertTrue(_same_payload(snapshot(),s));restores.append(steps())
            def fake_losses(net,data,ids,update):
                names=sorted({data.data[i][2][0] for i in ids});signs={g:(.02 if j%2 else -.02) for j,g in enumerate(names)}
                n=steps();base=1. if n<2 else .98 if n<4 else .981 if n==4 else .97 if retry else .99
                return {i:base+(signs[data.data[i][2][0]] if n==4 else 0.) for i in ids}
            def guarded(ref,net):
                r=probes(ref,net)
                if trip and steps()>=5 and (not retry or len(calls)==5):r[trip]=.003
                return r
            with patch.object(a,'step',recording),patch.object(a,'snapshot',paired_snapshot),patch.object(a,'restore',paired_restore),patch.object(A,'losses',fake_losses),patch.object(A.Reference,'probes',guarded):
                report=A.fit(model,p.optimizer,p.buffer,p.critic_dataset,p.train_ids,p.heldout_ids,
                    cfg['value_phase'],cfg['update'],beta=8.,snapshot_weight=1.,dropout=None,
                    run_phase=True,ppo_work=p.ppo_work,epoch_cap=2,companion=a)
            self.assertTrue(_same_payload(ppo,a.optimizers['ppo'].state_dict()))
            self.assertTrue(restores)
            self.assertLessEqual(report['probe_kl'],.002);self.assertLessEqual(report['age0_probe_kl'],.002)
            held=set(p.split['selection']['heldout_games'])
            self.assertTrue(held.isdisjoint({x[0] for x in a.retained if x[0] not in {ds.data[i][2][0] for i in (0,1)}}))
            if not retry:
                self.assertEqual(report['optimizer_steps'],4)
                self.assertTrue(_same_payload(snapshot(),saved['aux']))
                self.assertTrue(_same_payload(model.state_dict(),saved['core']))
                self.assertTrue(_same_payload(p.optimizer.state_dict(),saved['adam']))
                self.assertEqual(report['distinct_value_rows_used'],512)
            else:
                self.assertEqual(report['beta_retries'],1);self.assertEqual(report['beta_clone_final'],16.)
            return report

    def test_one_se_rejection_restores_coverage_selected_heads_and_Adam(self):
        self.assertEqual(self.run_fit()['early_stop_reason'],'validation_patience')

    def test_overall_and_age0_guard_restore_all_states(self):
        for metric in ('overall','age0'):
            with self.subTest(metric=metric):self.assertTrue(self.run_fit(metric)['kl_guard_tripped'])

    def test_beta_retry_restores_joint_best_and_keeps_PPO_moments(self):
        self.run_fit('age0',retry=True)


if __name__=='__main__':unittest.main()
