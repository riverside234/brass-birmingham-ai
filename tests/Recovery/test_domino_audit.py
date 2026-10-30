"""Cross-receipt admission, vectorized advantages and current-only PPG gradients."""
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import domino_sampling as A,domino_config as C,domino_state as S
from brass_ai.recovery import updates,value_phase
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.domino_auxiliary import Diversity,PPGCompanions,row_key
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json
from brass_ai.step5.policy import NeuralPolicy,context_for
from test_domino import fixture,collect


def ready(d):
    from brass_ai.recovery.domino_features import width
    d.state.update(centroids=[(np.arange(4*width())*(z+1)*1e-5).tolist() for z in range(7)],
        quality=[1.]*7,seat_games=[32]*7,distinct_games=[16]*7,logits=[-.3,-.1,0.,.1,.2,.3])


def old_combine(d,task,ds,ids,frozen):
    out=task.clone();c=S.coefficients(d.state,d.config)
    for i in ids:
        r=ds.rows[i]
        if r.get('_branch') or '_domino_target' not in r:continue
        z=r['_domino']['strategy_id'];adv=r['_domino_target']-float(frozen[i])
        out[i]=c[z]*task[i]+(1-c[z])*adv
    return out


class Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def prepared(self,run):
        m,o,c=fixture();p=TrainingPhases(m,c,root=SOURCE_ROOT,run=run)
        d=p.diversity=Diversity(m,c['diversity'],regime='audit');ready(d)
        b,rows=collect(m,p,1,games=42);ds=updates.Dataset(rows,'cpu');p.prepare(ds)
        return m,o,c,p,d,ds,b

    def test_missing_swapped_and_conflicting_contexts_fail_before_training(self):
        from types import SimpleNamespace
        from brass_ai.recovery.execution import neural_spec
        from brass_ai.recovery.replay import CheckedGame
        from brass_ai.recovery.critic_data import planned_table,check_table
        learner=neural_spec('same',.08,'mixed');frozen=neural_spec('same',.08,'neutral')
        specs=[learner,learner,frozen,frozen]
        assignment=A.assignments(specs,0,seed=92,index=27,learner_id=learner['policy_id'],current='same',checkpoints={'same':'a'*64})
        policies=[NeuralPolicy(None,exploration=s['exploration'],focus=s['focus'],
            evaluator=SimpleNamespace(model_id='same')) for s in specs]
        rngs=[random.Random(i) for i in range(4)];rng_before=[r.getstate() for r in rngs]
        proof=dict(diversity_assignment=assignment,ordinary_game_index=27,
            learner_behavior=dict(policy_id=learner['policy_id'],model_id='same'))
        missing=copy.deepcopy(proof);missing['diversity_assignment']['seats'][2]=None
        with self.assertRaisesRegex(ValueError,'lacks strategy'):A.install(dict(provenance=missing),policies,rngs)
        self.assertTrue(all(not p.contexts for p in policies))
        swapped=copy.deepcopy(proof);swapped['diversity_assignment']['seats'][:2]=reversed(swapped['diversity_assignment']['seats'][:2])
        with self.assertRaisesRegex(ValueError,'wrong weights/seat'):A.install(dict(provenance=swapped),policies,rngs)
        A.install(dict(provenance=proof),policies,rngs)
        rows=[]
        for s,p in enumerate(policies):
            style,intent=p.context_features({},rngs[s])
            self.assertEqual((style,intent),context_for(assignment['seats'][s]['context']))
            rows.append(dict(seat=s,style=style,intent=intent,teacher=None,behavior=dict(kind='model',policy_id=p.policy_id)))
        self.assertEqual(rng_before,[r.getstate() for r in rngs])
        g=CheckedGame(opponents=[p.policy_id for p in policies],records=rows)
        fp=dict(values=[1/8]*16+[1/6]*12)
        provider=SimpleNamespace(get=lambda *args:fp)
        table=planned_table(specs,0,'same',27,[dict(model_id='same',generation=0,fingerprint=fp)],fp,fp,
            learner_policy_id=learner['policy_id'],strategies=assignment,context_fingerprints=provider)
        proof['critic_table']=table;A.bind(g,proof);check_table(table,g,learner['policy_id'])
        bad=copy.deepcopy(table);a=bad['seats'][0]['strategy'];a['strategy_id']=(a['strategy_id']+1)%7
        a['context']=C.NAMES[a['strategy_id']];a['seat_policy_id']=digest(dict(behavior_family_id=a['behavior_family_id'],strategy_id=a['strategy_id']))
        a['receipt']=digest({k:v for k,v in a.items() if k!='receipt'});A.validate_assignment(a)
        with self.assertRaisesRegex(ValueError,'differs from collection'):A.bind(g,dict(proof,critic_table=bad))
        with self.assertRaisesRegex(ValueError,'differs from collection'):check_table(bad,g,learner['policy_id'])
        with self.assertRaisesRegex(ValueError,'lacks strategy'):A.bind(g,missing)

    def test_vectorized_advantages_match_scalar_and_leave_neutral_branch_and_rng(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            _,_,_,p,d,ds,_=self.prepared(Path(tmp));ids=list(range(len(ds.rows)))[::-1]
            for dtype in (torch.float32,torch.float64):
                task=torch.linspace(-2,3,len(ds.rows),dtype=dtype);frozen=torch.linspace(-.2,.1,len(p.critic_dataset.rows))
                expected=old_combine(d,task,ds,ids,frozen);rng=torch.get_rng_state().clone()
                result=d.combine(task,ds,ids,frozen)
                torch.testing.assert_close(result,expected,rtol=0,atol=0)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for i,row in enumerate(ds.rows):
                    if row.get('_branch') or row['_domino']['strategy_id']==0:self.assertEqual(result[i],task[i])
            self.assertTrue(torch.equal(d.combine(task,ds,[],frozen),task))

    def test_disk_metadata_is_current_only_and_does_not_decode_features(self):
        from brass_ai.recovery.ppg_replay import Rows
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            run=Path(tmp);m,_,_,p,d,ds,_=self.prepared(run)
            actor=[i for i in range(len(ds.rows)) if not ds.forced_cpu[i]]
            p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,actor,p.collection_source,m,p.split['selection'])
            write_json(run/'state.json',dict(iteration=1));p.buffer.committed(1);d.committed(1)
            b,rows=collect(m,p,2,games=42);current=updates.Dataset(rows,'cpu');p.prepare(current)
            actor=[i for i in range(len(rows)) if not current.forced_cpu[i]]
            p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,actor,p.collection_source,m,p.split['selection'])
            self.assertFalse(any(d.eligible(ds,range(len(ds.rows)))))
            with p.buffer.datasets('cpu',policy=False) as (replay,_,receipts):
                ids=list(range(len(replay.data)));expected=[row_key(replay,i) in d.train_keys for i in ids]
                query=ids[::-1]+ids[:8];expected=[expected[i] for i in query]
                with patch.object(Rows,'field',side_effect=AssertionError('Decoded a complete critic row')):
                    self.assertEqual(d.eligible(replay,query),expected)
                    self.assertEqual(d.eligible(replay,query),expected)
                self.assertEqual(sum(expected[:-8]),len(d.train_keys))
                current_part=replay.data.parts[0];g=next(g for g in current_part.domino if any(k[0]==g for k in d.train_keys))
                seat=next(str(k[1]) for k in d.train_keys if k[0]==g)
                current_part.domino[g][seat]=dict(current_part.domino[g][seat],receipt='0'*64)
                d._indices.clear()
                with self.assertRaisesRegex(ValueError,'context/role'):d.eligible(replay,ids)

    def test_ppg_masks_and_single_encoder_forward_with_or_without_local_heads(self):
        from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            m,_,c,p,d,_,_=self.prepared(Path(tmp));ds=p.critic_dataset
            train=next(i for i in p.train_ids if row_key(ds,i) in d.train_keys)
            hold=next(i for i in p.heldout_ids if row_key(ds,i) in d.targets)
            snap=next(i for i,r in enumerate(ds.roles) if r=='snapshot')
            branch=next(i for i,r in enumerate(ds.rows) if r.get('_branch'))
            ids=[train,hold,snap,branch];ix,inputs,_=ds.value_batch(ids,m)
            with torch.no_grad():d.head[-1].weight.fill_(.1)
            for local in (None,Companion(m,dict(DEFAULT,mode='off')),Companion(m,DEFAULT)):
                m.zero_grad(set_to_none=True);d.begin('ppg');wrapper=PPGCompanions(local,d,p.optimizer,.5);wrapper.batch(ds,ids)
                predictions=[]
                def watch(module,inputs,out):out.retain_grad();predictions.append(out)
                with patch.object(m,'encode_state',wraps=m.encode_state) as encode:
                    hook=d.head.register_forward_hook(watch)
                    try:
                        out,loss=wrapper.forward(m,inputs,ds,ix,dropout=(0,0,1.,'value'))
                        (out[4].square().mean()+loss.mean()).backward()
                    finally:hook.remove()
                self.assertEqual(encode.call_count,1)
                self.assertGreater(abs(float(predictions[0].grad[0])),0.)
                self.assertFalse(bool(predictions[0].grad[1:].any()))
                self.assertGreater(sum(float(v.grad.abs().sum()) for n,v in m.named_parameters() if n.startswith('state_encoder') and v.grad is not None),0.)
                with torch.no_grad():
                    expected=updates.value_output(m,inputs,ds,ix,dropout=(0,0,1.,'value'))
                for a,b in zip(out[1:],expected[1:]):torch.testing.assert_close(a,b,rtol=0,atol=0)
                actual_grads=[v.grad.clone() if v.grad is not None else None for v in (*m.parameters(),*d.head.parameters())]
                m.zero_grad(set_to_none=True);d.begin('ppg')
                expected=updates.value_output(m,inputs,ds,ix,dropout=(0,0,1.,'value'))
                pred=d.prediction(m,inputs,ds,ix)
                auxiliary=(pred[0]-d.targets[row_key(ds,train)]).square()*.5
                (expected[4].square().mean()+auxiliary).backward()
                for before,param in zip(actual_grads,(*m.parameters(),*d.head.parameters())):
                    if before is None:self.assertIsNone(param.grad)
                    else:torch.testing.assert_close(before,param.grad,rtol=2e-5,atol=2e-6)

    def test_ppg_cache_cannot_overwrite_a_seat_strategy(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            m,_,_,p,_,ds,_=self.prepared(Path(tmp));critic=p.critic_dataset
            first=next(i for i,r in enumerate(critic.rows) if not r.get('_branch'))
            second=next(i for i in range(first+1,len(critic.rows)) if critic.data[i][2]==critic.data[first][2])
            a=copy.deepcopy(critic.rows[second]['_domino']);a['strategy_id']=(a['strategy_id']+1)%7;a['context']=C.NAMES[a['strategy_id']]
            a['seat_policy_id']=digest(dict(behavior_family_id=a['behavior_family_id'],strategy_id=a['strategy_id']))
            a['receipt']=digest({k:v for k,v in a.items() if k!='receipt'});critic.rows[second]['_domino']=a
            actor=[i for i in range(len(ds.data)) if not ds.forced_cpu[i]]
            with self.assertRaisesRegex(ValueError,'changed within cached'):
                p.buffer.stage(critic,p.train_ids,p.heldout_ids,actor,p.collection_source,m,p.split['selection'])

    def test_rejected_ppg_restores_base_head_and_all_adam_state(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            m,_,c,p,d,ds,_=self.prepared(Path(tmp));critic=p.critic_dataset
            for opt,parameters in ((p.optimizer,m.parameters()),(d.optimizers['ppg'],d.head.parameters())):
                for param in parameters:param.grad=torch.full_like(param,.01)
                opt.step();opt.zero_grad(set_to_none=True)
            before=copy.deepcopy((m.state_dict(),p.optimizer.state_dict(),d.snapshot()))
            companion=PPGCompanions(None,d,p.optimizer,.5)
            actor=[i for i in p.learner_train_ids if not ds.forced_cpu[i]];calls=[];original=value_phase.metrics
            def reject(*args,**kwargs):
                result=original(*args,**kwargs);calls.append(result)
                if len(calls)==3:result=dict(result,loss=result['loss']+100)
                return result
            cfg=dict(c['value_phase'],learning_rate=1e-5,batch_size=4096,max_epochs=1,acceptance_steps=1,kl_check_steps=1)
            with patch.object(value_phase,'metrics',side_effect=reject):
                report=value_phase.fit(m,p.optimizer,ds,p.learner_train_ids,p.heldout_ids,actor,actor[:8],cfg,c['update'],beta=1.,
                    value_dataset=critic,value_train_ids=p.train_ids,validation_dataset=critic,companion=companion)
            self.assertGreater(report['attempted_steps'],0);self.assertEqual(report['optimizer_steps'],0)
            self.assertTrue(_same_payload(before,(m.state_dict(),p.optimizer.state_dict(),d.snapshot())))


if __name__=='__main__':unittest.main()
