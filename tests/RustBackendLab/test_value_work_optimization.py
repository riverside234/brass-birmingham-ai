"""Matched numeric/byte/work regressions; no local timing benchmarks."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.contracts import digest
from brass_ai.learning.seat_descriptors import critic_inputs,validate,unknown_numpy
from brass_ai.recovery import updates as U
from brass_ai.recovery.ppg_cache_writer import write_critic
from brass_ai.recovery.value_buffer import row_dtype
from brass_ai.recovery.value_phase import RawPolicyReference
from brass_ai.recovery.retention import _same_payload
import test_value_batch_performance as batch_fixture


class Forwards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        f=batch_fixture.ValueBatchPerformanceTests();return f.model(True),U.Dataset(f.data(True),'cpu')

    def test_value_predictions_gradients_adam_and_rng_match_full_forward(self):
        initial,ds=self.fixture();results=[]
        ix,inputs,_=ds.batch([3,1,0,2])
        for fast in (False,True):
            m=copy.deepcopy(initial);adam=torch.optim.AdamW(m.parameters(),lr=1e-5);torch.manual_seed(991)
            out=U.value_output(m,inputs,ds,ix,heads_only=True) if fast else m(*inputs,remaining=ds.remaining[ix],descriptors=critic_inputs(ds,ix.tolist()))
            loss=U.value_losses(out,tuple(t[ix] for t in ds.targets))[0].sum()+(out[4]-ds.utility[ix]).square().sum()
            loss.backward();adam.step()
            results.append((tuple(x.detach().clone() for x in out[1:]),m.state_dict(),adam.state_dict(),torch.get_rng_state()))
        self.assertTrue(_same_payload(results[0],results[1]))

    def test_policy_predictions_gradients_adam_and_rng_match_full_forward(self):
        initial,ds=self.fixture();results=[];_,inputs,_=ds.batch([3,1,0,2])
        for fast in (False,True):
            m=copy.deepcopy(initial);adam=torch.optim.AdamW(m.parameters(),lr=1e-5);torch.manual_seed(992)
            logits=U.policy_output(m,inputs) if fast else m(*inputs)[0]
            loss=-torch.log_softmax(logits.float(),-1)[:,0].sum();loss.backward();adam.step()
            results.append((logits.detach(),m.state_dict(),adam.state_dict(),torch.get_rng_state()))
        self.assertTrue(_same_payload(results[0],results[1]))

    def test_value_pass_never_constructs_candidates_and_reference_never_constructs_value_heads(self):
        m,ds=self.fixture();ix,inputs,_=ds.batch([0,1,2,3])
        with patch.object(m,'score_prepared',side_effect=AssertionError('Unneeded candidate scoring')):
            U.value_output(m,inputs,ds,ix,heads_only=True)
        with patch.object(m,'heads_from_encoded',side_effect=AssertionError('Unneeded value heads')):
            reference=RawPolicyReference(m,ds,[0,1,2,3],dict(microbatch_size=2,max_candidate_cells=6,bucket=True))
            self.assertAlmostEqual(reference.measure(m),0.)
            reference.loss([0,1],reference.inputs([0,1]),m).sum().backward()

    def test_action_dependent_custom_model_keeps_original_forward(self):
        f=batch_fixture.ValueBatchPerformanceTests();m=f.model(False);ds=U.Dataset(f.data(False),'cpu')
        from brass_ai.learning.model import FourSeatNet
        class Custom(FourSeatNet):
            def forward(self,*inputs,**kwargs):
                out=super().forward(*inputs,**kwargs)
                return (*out[:3],out[3]+inputs[1].sum((1,2))[:,None])
        custom=Custom(m.config)
        ix,inputs,_=ds.batch([0,1]);expected=custom(*inputs)
        with patch.object(custom,'forward',wraps=custom.forward) as forward:
            actual=U.value_output(custom,inputs,ds,ix,heads_only=True)
            self.assertTrue(_same_payload(actual,expected));self.assertEqual(forward.call_count,1)
            self.assertTrue(torch.equal(U.policy_output(custom,inputs),expected[0]));self.assertEqual(forward.call_count,2)


class Cache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        f=batch_fixture.ValueBatchPerformanceTests();data=f.data(True)
        for row,_,_,_ in data:
            row.update(_critic_role='learner',_table_kind='selfplay',behavior=dict(policy_id='same-policy'))
        data[1][0].update(_critic_role='snapshot',behavior=dict(policy_id='opponent-policy'))
        ds=SimpleNamespace(data=data)
        return ds,dict(policy_id='same-policy',snapshot_rows=True),data[0][0]['_reward_id'],dict(games={d[2][0]:'selfplay' for d in data})

    def scalar_reference(self,ds,ids,generation):
        a=np.empty(len(ids),dtype=row_dtype());games={};contexts={};values=[]
        for j,i in enumerate(ids):
            row,target,identity,encoded=ds.data[i];context=row['view']['context'];key=digest(context)
            if key not in contexts:contexts[key]=len(contexts);values.append(copy.deepcopy(context))
            e=a[j];e['state']=encoded.state;e['action']=encoded.actions[0];e['mapping']=encoded.stable_to_relative
            e['targets']=[target[k] for k in ('win_share','placement','vp')];e['utility']=row['_reward_return'];e['remaining']=row['_reward_remaining']
            e['seat']=row['seat'];e['game']=games.setdefault(identity[0],len(games));e['context']=contexts[key];e['forced']=len(encoded.choice_ids)==1
            e['descriptors']=validate(row.get('_seat_descriptors',unknown_numpy()));e['role']=('learner','snapshot').index(row['_critic_role'])
            e['table']=('selfplay','historical','legacy').index(row['_table_kind']);e['descriptor_generation']=row.get('_descriptor_generation',generation-1)
        return a,games,values

    def test_byte_identical_block_writer_keeps_order_duplicates_and_all_metadata(self):
        ds,source,reward,selection=self.fixture();ids=[3,1,0,2,1]
        expected,games,contexts=self.scalar_reference(ds,ids,12)
        for size in (1,2,256):
            actual=np.empty(len(ids),dtype=row_dtype())
            g,c,stats=write_critic(actual,ds,ids,source,reward,selection,12,block_rows=size)
            self.assertEqual(actual.tobytes(),expected.tobytes());self.assertEqual((g,c),(games,contexts))
            self.assertEqual(stats['rows'],5);self.assertEqual(stats['finite_array_checks'],((5+size-1)//size)*5)

    def test_source_seat_prefix_role_and_descriptor_checks_stay_mandatory(self):
        for key,value in [('_reward_id','wrong'),('_critic_role','foreign'),('_table_kind','legacy'),
                          ('_copied_prefix',True),('behavior',{'policy_id':'foreign'}),('_seat_descriptors',np.full((4,34),np.nan))]:
            ds,source,reward,selection=self.fixture();ds.data[0][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                write_critic(np.empty(4,dtype=row_dtype()),ds,list(range(4)),source,reward,selection,12)
        ds,source,reward,selection=self.fixture();row,target,(game,seat),encoded=ds.data[0]
        ds.data[0]=(row,target,(game,(seat+1)%4),encoded)
        with self.assertRaisesRegex(ValueError,'mismatched'):
            write_critic(np.empty(4,dtype=row_dtype()),ds,list(range(4)),source,reward,selection,12)

    def test_nonfinite_and_destination_float_overflow_are_rejected_before_publication(self):
        for value in (float('nan'),float('inf'),1e300):
            ds,source,reward,selection=self.fixture();ds.data[0][0]['_reward_return']=[value]*4
            with np.errstate(over='ignore'),self.assertRaisesRegex(ValueError,'Nonfinite'):
                write_critic(np.empty(4,dtype=row_dtype()),ds,list(range(4)),source,reward,selection,12)

    def test_metric_scans_reuse_sealed_groups_and_batch_result_transfers(self):
        from brass_ai.recovery import value_phase as V
        from brass_ai.recovery.config import UPDATE_V21
        from brass_ai.recovery.ppg_replay import CriticDataset,Rows

        source_ds,source,reward,selection=self.fixture()
        model=batch_fixture.ValueBatchPerformanceTests().model(True)
        parts=[]
        with tempfile.TemporaryDirectory() as directory:
            try:
                for age,indices in enumerate(([3,1],[2,0])):
                    array=np.empty(len(indices),dtype=row_dtype())
                    games,contexts,_=write_critic(array,source_ds,indices,source,reward,selection,12)
                    path=Path(directory)/f'age-{age}.npy'
                    np.save(path,array)
                    parts.append(Rows(path,dict(rows=len(indices),game_ids=list(games),contexts=contexts,files={})))
                ds=CriticDataset(parts,'cpu');ids=[3,0,1,2]*9;config=dict(UPDATE_V21,microbatch_size=1)
                with patch.object(ds,'metric_metadata',None):
                    expected=V.metrics(model,ds,ids,config,evidence=True)
                transferred=[];cpu=torch.Tensor.cpu
                def transfer(tensor,*args,**kwargs):
                    if tensor.ndim==2 and tensor.shape[1]==12:transferred.append(len(tensor))
                    return cpu(tensor,*args,**kwargs)
                cache={}
                with patch.object(V,'metric_groups',wraps=V.metric_groups) as group,patch.object(torch.Tensor,'cpu',transfer):
                    for _ in range(2):
                        self.assertEqual(V.metrics(model,ds,ids,config,evidence=True,group_cache=cache),expected)
                    self.assertEqual(group.call_count,1)
                    self.assertEqual(transferred,[32,4,32,4])
                    V.metrics(model,ds,list(reversed(ids)),config,evidence=True,group_cache=cache)
                    self.assertEqual(group.call_count,2)
            finally:
                for part in parts:part.close()


class Targets(unittest.TestCase):
    def test_flat_target_copy_matches_old_recurrence_and_isolates_numeric_and_nested_mutation(self):
        from brass_ai.recovery.search_value_phase import SearchValuePhase
        rows=[dict(game_id='g',index=i,state=[float(j) for j in range(3000)],mapping=[0,1,2,3],remaining=[.1]*4,
            u=[.2+.01*i]*4,r=[.3]*4,target=[0.]*4,terminal_utility=[.25]*4,job={'binding':{'model':'frozen'}}) for i in (2,1)]
        for lam in (.5,.8,.95,1.):
            phase=SimpleNamespace(machine=SimpleNamespace(state={'lam':lam}))
            expected=copy.deepcopy(sorted(rows,key=lambda r:r['index']));following=np.asarray(expected[-1]['terminal_utility'])
            for i in reversed(range(len(expected))):
                next_u=np.asarray(expected[i+1]['u'] if i+1<len(expected) else expected[i]['terminal_utility'])
                boot=np.asarray(expected[i+1]['r'] if i+1<len(expected) else expected[i]['terminal_utility'])
                following=-(next_u-expected[i]['u'])+(1-lam)*boot+lam*following
                expected[i].update(target=following.tolist(),target_lambda=lam)
            actual=SearchValuePhase._targets(phase,rows)
            self.assertEqual(actual,expected)
            actual[0]['state'][0]=999;actual[0]['job']['binding']['model']='changed'
            self.assertEqual(rows[1]['state'][0],0.);self.assertEqual(rows[1]['job']['binding']['model'],'frozen')


class Calibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def run_phase(self,*,deferred,accepted):
        from brass_ai.recovery import value_phase as V,ppg_g210 as G
        from brass_ai.recovery.config import UPDATE_V21
        f=batch_fixture.ValueBatchPerformanceTests();model=f.model(True);ds=U.Dataset(f.data(True),'cpu')
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-6)
        config=dict(G.TEACHER_FLOOR_DEFAULT,batch_size=2,clone_rows=2,clone_batch_size=2,learning_rate=1e-6)
        update=dict(UPDATE_V21,microbatch_size=2,max_candidate_cells=6)
        original=V.metrics;validation_calls=0;train_calls=0
        def measure(m,data,ids,*args,**kwargs):
            nonlocal validation_calls,train_calls
            result=original(m,data,ids,*args,**kwargs)
            if list(ids)==[0,1,2]:train_calls+=1
            if list(ids)==[3] and not kwargs.get('unknown'):
                validation_calls+=1
                # Fixed deterministic loss decisions exercise rejected and
                # accepted transactions; calibration uses real predictions.
                result['loss']=1./validation_calls if accepted else float(validation_calls)
            return result
        torch.manual_seed(665)
        with patch.object(V,'defer_train_metrics',return_value=deferred),patch.object(V,'metrics',side_effect=measure):
            report=V.fit(model,optimizer,ds,[0,1,2],[3],[0,2],[0,2],config,update,beta=.25)
        return model,optimizer,torch.get_rng_state(),report,train_calls

    def test_rejected_phase_skips_full_train_scan_without_changing_weights_adam_rng_or_gates(self):
        before=self.run_phase(deferred=False,accepted=False);after=self.run_phase(deferred=True,accepted=False)
        self.assertEqual((before[4],after[4]),(1,0))
        self.assertTrue(_same_payload((before[0].state_dict(),before[1].state_dict(),before[2]),
            (after[0].state_dict(),after[1].state_dict(),after[2])))
        for key in ('optimizer_steps','attempted_steps','early_stop_reason','probe_kl','interval_reports'):
            self.assertEqual(before[3][key],after[3][key])
        self.assertEqual(after[3]['optimizer_steps'],0);self.assertIsNone(after[0].utility_calibration)
        self.assertFalse(after[3]['train_metrics_measured']);self.assertIsNone(after[3]['train_before']['loss'])
        self.assertEqual(after[3]['full_train_metric_rows_avoided'],3)

    def test_accepted_phase_measures_exact_pre_and_post_weights_preserving_adam_rng_and_readiness(self):
        before=self.run_phase(deferred=False,accepted=True);after=self.run_phase(deferred=True,accepted=True)
        self.assertEqual((before[4],after[4]),(2,2));self.assertGreater(after[3]['optimizer_steps'],0)
        self.assertTrue(_same_payload((before[0].state_dict(),before[1].state_dict(),before[2],before[0].utility_calibration),
            (after[0].state_dict(),after[1].state_dict(),after[2],after[0].utility_calibration)))
        for key in ('optimizer_steps','attempted_steps','probe_kl','interval_reports','train_before','train_after','heldout_after'):
            self.assertEqual(before[3][key],after[3][key])
        self.assertEqual(after[3]['full_train_metric_rows_avoided'],0)
        self.assertGreaterEqual(after[3]['timings']['train_before_seconds'],0.)


if __name__=='__main__':unittest.main()
