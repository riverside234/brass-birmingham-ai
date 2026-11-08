"""Tiny numeric/binding regressions for canal/final VP reward timing."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.contracts import digest
from brass_ai.learning.post_transfer_reward import specification
from brass_ai.learning.reward import replay_returns,terminal_utility
from brass_ai.learning.reward_timing import replay_payments,teacher_payments,vp_indices,policy_returns
from brass_ai.recovery.updates import prepared,Dataset
from brass_ai.recovery.gae import compute


class EraVP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def setUp(self):self.reward=specification(True)

    def result(self):
        scores=[200,170,100,90]
        return dict(objective='win-share/1.0.0',seats=[dict(seat=s,vp=v,income=0,cash=0,rank=s+1,
            utility=1. if s==0 else 0.) for s,v in enumerate(scores)])

    def records(self):
        return [dict(seat=i%4,view=dict(context=dict(era=era,round=1,phase='action'),players=[
            dict(seat=s,vp=v,debt=0) for s,v in enumerate(scores)]),choices=[dict(kind='pass')],chosen=0)
            for i,(era,scores) in enumerate([('CanalEra',[0]*4),('CanalEra',[10]*4),
                ('RailEra',[60,90,180,200]),('RailEra',[80,95,180,200])])]

    def test_once_only_after_scoring_and_terminal_difference_uses_one_shared_cap(self):
        records=self.records();original=copy.deepcopy(records);result=self.result()
        labels,summary=replay_returns(records,result,self.reward)
        p=summary['vp_payments'];np.testing.assert_allclose(p['canal'],[.06,.09,.18,.18])
        np.testing.assert_allclose(p['final_delta'],[.12,.08,-.08,-.09],atol=1e-15)
        np.testing.assert_allclose(np.array(p['canal'])+p['final_delta'],[.18,.17,.1,.09])
        self.assertEqual(p['canal_before_record'],2);self.assertEqual(records,original)
        terminal=terminal_utility(result,self.reward)
        for i,row in enumerate(labels):
            credit=np.array(p['canal']) if i>=2 else np.zeros(4)
            np.testing.assert_allclose(row['utility'],terminal-credit)
            np.testing.assert_allclose(row['anchored_utility'],terminal)
        # No clamping the difference: a lower final score must correct an
        # earlier payout, preserving the exact original capped total.
        np.testing.assert_allclose(labels[2]['utility'],terminal-p['canal'])

    def test_bad_boundary_and_missing_parent_credit_rejected_but_rail_branch_inherits_exactly(self):
        rows=self.records();result=self.result()
        for invalid in (rows[:2],rows+[rows[0]], [dict(r,view=dict(r['view'],context=dict(r['view']['context'],round=3))) for r in rows]):
            with self.assertRaises(ValueError):replay_payments(invalid,result,self.reward)
        with self.assertRaisesRegex(ValueError,'parent canal payout'):replay_returns(rows[2:],result,self.reward)
        _,p=replay_payments(rows,result,self.reward)
        labels,_=replay_returns(rows[2:],result,self.reward,initial_vp_paid=p['canal'])
        expected=replay_returns(rows,result,self.reward)[0][2:]
        self.assertEqual(labels,expected)

    def test_old_reward_keeps_terminal_only_labels_byte_equivalent(self):
        old=specification(False);rows=self.records();labels,summary=replay_returns(rows,self.result(),old)
        self.assertNotIn('vp_payments',summary)
        for row in labels:
            self.assertEqual(set(row),{'remaining','utility'})
            np.testing.assert_allclose(row['utility'],terminal_utility(self.result(),old))

    def test_preparation_and_dataset_keep_paid_credit_and_anchor_values_without_new_inputs(self):
        from test_b025 import example,model_fixture
        label,_=example(model_fixture());encoded=label.encoded
        rows=[]
        for i,raw in enumerate(self.records()):
            view=copy.deepcopy(label.row['view']);view['context'].update(raw['view']['context'])
            for player,score in zip(view['players'],raw['view']['players'],strict=True):player['vp']=score['vp'];player['debt']=0
            row=dict(label.row,step=i,seat=view['seat'],view=view,style=[0.]*4,intent=[0.]*8,teacher=None,
                chosen=0,seconds=0.,behavior=dict(kind='model',policy_id='fixture',probability=1.,temperature=1.))
            rows.append(row)
        game=dict(game_id='timed',records=rows,result=self.result())
        # Encoded tensors are an authored frozen fixture, not a new game. The
        # normal target/metadata route still runs with the public score views.
        with patch('brass_ai.recovery.updates.PreparedEncoder.encode',return_value=encoded):
            data=prepared([game],reward=self.reward,insolvency=self.reward['insolvency'])
        ds=Dataset(data,'cpu',policy=False)
        expected=np.asarray([r['utility'] for r in replay_returns(rows,self.result(),self.reward)[0]])
        np.testing.assert_allclose(policy_returns(ds).numpy(),expected,atol=1e-7)
        np.testing.assert_allclose(ds.utility.numpy(),np.tile(terminal_utility(self.result(),self.reward),(4,1)),atol=1e-7)
        self.assertEqual(ds.encoded[0].state.shape,encoded.state.shape)
        self.assertIn('canal_vp_payment',data[0][0]['_reward_trajectory'])
        branch=copy.deepcopy(game);branch['game_id']='timed/branch';branch['records']=branch['records'][2:]
        branch['branch']=dict(parent_game_id=game['game_id'],prior_events=[0]*4)
        for i,row in enumerate(branch['records']):row['step']=i
        with patch('brass_ai.recovery.updates.PreparedEncoder.encode',return_value=encoded):
            child=prepared([branch],reward=self.reward,insolvency=self.reward['insolvency'],vp_parent=game)
        self.assertEqual(len(child),1)  # Fork action stays excluded.
        np.testing.assert_allclose(policy_returns(Dataset(child,'cpu',policy=False)).numpy()[0],expected[-1],atol=1e-7)

    def test_gae_includes_payment_at_boundary_and_no_double_credit_in_values(self):
        # Two observations in one seat's chain, one before and one after canal.
        data=[({},None,('game',0),None),({},None,('game',0),None)]
        for row in data:row[0]['view']=dict(context=dict(era='CanalEra',phase='action',round=1))
        raw=torch.tensor([[.5]*4,[.7]*4]);paid=torch.tensor([[0.]*4,[.06]*4])
        ds=SimpleNamespace(utility=raw,targets=None,vp_paid=paid)
        future=policy_returns(ds)[:,0];mc=torch.tensor([.2,.3])
        actual,_=compute(data,[0,1],mc,future,.95,[])
        # next reward is .5 - .64 = -.14, next value is .64-.3=.34;
        # current value .5-.2=.3, so delta=-.1 and GAE=-.1+.95*.3.
        np.testing.assert_allclose(actual.numpy(),[.185,.3],atol=1e-7)

    def teacher_rows(self):
        from test_b025 import model_fixture
        from test_b045 import value_fixture
        row,job,_=value_fixture(model_fixture(),generation=1284,seat=2)
        job['search']['reward']=self.reward;job['expected']['reward_id']=digest(self.reward)
        result=[]
        for i in range(64):
            r=copy.deepcopy(row);era='CanalEra' if i<32 else 'RailEra';round_no=(i%32)//4+1
            r.update(index=i+1,job=job['expected'],reward_id=job['expected']['reward_id'],
                remaining=[.025]*4,spent=[0.]*4,turn_key=[era,round_no,i%4,2],
                profile='full' if i%4==0 else 'fast',phase=('canal_' if i<32 else 'rail_')+('early' if round_no<=4 else 'late'))
            for stable,vp in enumerate([60,90,180,200]):r['state'][vp_indices()[r['mapping'][stable]]]=float(np.float32(vp/100))
            result.append(r)
        return result,job

    def test_teacher_existing_root_tensors_reconstruct_rotated_canal_payment_and_seal_future_rows(self):
        from brass_ai.recovery.search_value_rows import complete_game,copy_row
        rows,job=self.teacher_rows();p=teacher_payments(rows,self.result(),self.reward)
        np.testing.assert_allclose(p['canal'],[.06,.09,.18,.18]);self.assertEqual(p['canal_before_root'],33)
        completed=complete_game(rows,self.result(),[0.]*4,job)
        self.assertEqual(completed[0]['vp_payments'],p)
        self.assertEqual(completed[31]['vp_paid'],[0.]*4);self.assertEqual(completed[32]['vp_paid'],p['canal'])
        np.testing.assert_allclose(completed[32]['future_u'],np.asarray(completed[32]['u'])-p['canal'])
        detached=copy_row(completed[32]);detached['vp_paid'][0]=999
        self.assertEqual(completed[32]['vp_paid'],p['canal'])

    def test_teacher_incomplete_or_nonintegral_boundary_cannot_invent_payment(self):
        rows,_=self.teacher_rows()
        with self.assertRaisesRegex(ValueError,'complete eras'):teacher_payments(rows[1:32]+rows[33:],self.result(),self.reward)
        rows[32]['state'][vp_indices()[0]]=.12345
        with self.assertRaisesRegex(ValueError,'Nonintegral'):teacher_payments(rows,self.result(),self.reward)


if __name__=='__main__':unittest.main()
