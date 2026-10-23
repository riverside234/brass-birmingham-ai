"""Companion contracts/gradients, native passive capture and transactional replay."""
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.learning.local_scoring_heads import LocalScoringHeads
from brass_ai.learning.batching import model_id
from brass_ai.recovery import local_scoring_targets as targets,local_scoring_auxiliary as auxiliary
from brass_ai.recovery import updates
from brass_ai.recovery.retention import _same_payload
from test_gae_value_phase import fixture


def envelope(game='authored'):
    """Authored scoring fixture, not represented as native evaluation evidence."""
    from brass_ai.learning.board_topology import topology
    board=topology();eras=[];previous=[0]*4
    for era in ('CanalEra','RailEra'):
        slots=[dict(id=n,owner=(j%5),vp=200 if j==1 else j%3 if j%5 else 0,valid=True,instance='tile:'+str(j)) for j,n in enumerate(board['slot_order'])]
        links=[dict(id=n,owner=j%5 if board['link_eras'][n]==era.replace('Era','') else 0,
            vp=0,valid=board['link_eras'][n]==era.replace('Era','')) for j,n in enumerate(board['link_order'])]
        industry=[sum(r['vp'] for r in slots if r['owner']==s+1) for s in range(4)]
        after=[a+b for a,b in zip(previous,industry)]
        eras.append(dict(era=era,slots=slots,links=links,before=previous,
            audit=dict(industry=industry,network=[0]*4,bonus=[0]*4,bonus_since_recording=[0]*4,outside_adjustment=[0]*4,outside_scoring=[0]*4,after=after)))
        previous=after
    native=dict(contract=targets.CONTRACT,rules=targets.identity()['rules'],producer='local-scoring-host/1.0.0',
        start_revision=0,initial_vp=[0]*4,eras=eras,final_vp=previous)
    g=dict(game_id=game,setup_group='setup',records=[dict(view=dict(context=dict(era=e))) for e in ('CanalEra','RailEra')],
        result=dict(seats=[dict(seat=s,vp=v) for s,v in enumerate(previous)]))
    return targets.bind(native,g,dict(runtime_sha256='a'*64,host_sha256='b'*64)),g


def annotate(data):
    maps={}
    for j,(row,_,(game,_),_) in enumerate(data):
        if game not in maps:maps[game]=envelope(game)[0]
        row['step']=j;row['view']['context']['revision']=j
        row['_local_scoring']=dict(envelope=maps[game],era=row['view']['context']['era'])
    return data


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_labels_exact_bounds_masks_digest_and_missing(self):
        e,g=envelope();targets.validate(e,g)
        self.assertEqual(targets.unpack(targets.pack(e)),e)
        self.assertTrue(any(r['vp']>180 for r in e['receipt']['eras'][0]['slots']))
        for edit in ('sha','owner','mask','total','era','game','order'):
            bad=copy.deepcopy(e)
            if edit=='sha':bad['sha256']='0'*64
            elif edit=='owner':bad['receipt']['eras'][0]['slots'][0]['owner']=5
            elif edit=='mask':bad['receipt']['eras'][0]['links'][0]['valid']=False
            elif edit=='total':bad['receipt']['eras'][0]['audit']['industry'][0]+=1
            elif edit=='era':bad['receipt']['eras'].pop(0)
            elif edit=='game':bad['game_id']='foreign'
            elif edit=='order':bad['receipt']['eras'][0]['slots'].reverse()
            if edit!='sha':bad['sha256']=targets.seal(bad)
            with self.assertRaises(ValueError):targets.validate(bad,g)
        self.assertIsNone(targets.row_target(g,g['records'][0]))

    def test_all_observer_rotations_and_vp_not_clipped(self):
        m,_,_,(_,data)=fixture(games=2);annotate(data);a=auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT))
        from brass_ai.learning.features import Encoded
        for observer in range(4):
            row,target,ident,e=data[0];mapping=np.array([(s-observer)%4 for s in range(4)],dtype=np.int64)
            row=dict(row,seat=observer);encoded=Encoded(e.state,e.actions,mapping,e.choice_ids)
            ds=updates.Dataset([(row,target,(ident[0],observer),encoded)],'cpu');ix,x,_=ds.batch([0]);a.begin('ppo');a.forward(m,x,ds,ix)
            key=((row['_local_scoring']['envelope']['sha256'],row['_local_scoring']['era'],observer,tuple(mapping)),'slots')
            classes,values,mask=a.target_cache[key];labels=targets.map_for(row)['slots']
            self.assertTrue(mask.all())
            for j,label in enumerate(labels):
                owner=label['owner'];self.assertEqual(classes[j],mapping[owner-1]+1 if owner else 0)
                self.assertAlmostEqual(float(values[j].sum())*20,label['vp'])
                if owner:self.assertEqual(int(values[j].argmax()) if label['vp'] else mapping[owner-1],mapping[owner-1])

    def test_one_se_selection_restores_companion_Adam_with_core(self):
        from brass_ai.recovery.ppg_selection import Selection
        m,opt,_,(_,data)=fixture(games=2);annotate(data);ds=updates.Dataset(data,'cpu');ix,x,_=ds.batch([0,1])
        a=auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT));games=dict(a=1.,b=1.)
        selection=Selection(m,opt,games,companion=a)
        a.begin('ppg',reset=True);_,loss=a.forward(m,x,ds,ix);loss.mean().backward();a.step('ppg')
        selection.consider(m,opt,dict(a=.9,b=.9),[1,2],[],1,1);saved=a.snapshot()
        a.begin('ppg');_,loss=a.forward(m,x,ds,ix);loss.mean().backward();a.step('ppg')
        self.assertFalse(selection.consider(m,opt,dict(a=1.2,b=1.2),[1,2,3],[],2,2))
        selection.restore(m,opt);self.assertTrue(_same_payload(saved,a.snapshot()))

    def test_parameter_counts_initialization_rng_and_old_core_identity(self):
        h=LocalScoringHeads();self.assertEqual(sum(p.numel() for p in h.parameters()),91410)
        m,opt,c,(b,data)=fixture(games=2)
        before=model_id(m);r=torch.get_rng_state().clone();py=random.getstate();npstate=np.random.get_state()
        a=auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT))
        self.assertTrue(torch.equal(r,torch.get_rng_state()));self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(npstate[1],np.random.get_state()[1]));self.assertEqual(before,model_id(m))
        for layer in (a.heads.slots.owner,a.heads.links.owner,a.heads.slots.vp,a.heads.links.vp):
            self.assertEqual(float(layer.weight.detach().abs().sum()),0.)

    def test_one_encoding_parity_head_only_and_shared_gradients(self):
        from unittest.mock import patch
        m,opt,c,(b,data)=fixture(games=2);annotate(data);ds=updates.Dataset(data,'cpu')
        a=auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT));m.eval()
        for n in (1,2):
            ix,x,_=ds.batch(range(n));expected=updates.value_output(m,x,ds,ix)
            a.begin('ppo')
            with patch.object(m,'encode_state',wraps=m.encode_state) as encode:
                out,loss=a.forward(m,x,ds,ix);self.assertEqual(encode.call_count,1)
            for original,new in zip(expected[1:],out[1:]):self.assertTrue(torch.equal(original,new))
            before=model_id(m);adam=copy.deepcopy(opt.state_dict())
            m.zero_grad();loss.mean().backward();a.step('ppo')
            self.assertEqual(before,model_id(m));self.assertTrue(_same_payload(adam,opt.state_dict()))
            self.assertTrue(all(p.grad is None or not p.grad.any() for p in m.parameters()))
        a.age=2;self.assertEqual(a.scale,.2);a.begin('ppo');m.zero_grad()
        _,loss=a.forward(m,x,ds,ix);loss.mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.state_encoder.parameters() if p.grad is not None),0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.entity_encoder.parameters() if p.grad is not None),0.)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in m.descriptor_encoder.parameters() if p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for p in m.policy_scorer.parameters()))

    def test_missing_labels_no_step_and_partial_batch_normalization(self):
        m,_,_,(_,data)=fixture(games=2);annotate(data);ds=updates.Dataset(data,'cpu')
        a=auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT));ix,x,_=ds.batch([0,1]);a.begin('ppo')
        _,full=a.forward(m,x,ds,ix);data[1][0].pop('_local_scoring');a.begin('ppo')
        _,partial=a.forward(m,x,ds,ix)
        self.assertEqual(float(partial[1].detach()),0.);self.assertEqual(float(partial[0].detach()),float(full[0].detach()))
        data[0][0].pop('_local_scoring');a.begin('ppo');_,none=a.forward(m,x,ds,ix);a.step('ppo')
        self.assertIsNone(none);self.assertFalse(a.optimizers['ppo'].state)

    def test_bounded_heldout_diagnostics_and_disabled_mode_preserve_state(self):
        m,_,_,(_,data)=fixture(games=2);annotate(data);ds=updates.Dataset(data,'cpu')
        a=auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT));before=model_id(m)
        result=a.diagnostics(m,ds,list(range(len(data))),maximum=4)
        self.assertEqual(result['rows'],4);self.assertTrue(result['groups'])
        for group in result['groups'].values():
            self.assertEqual(group['current_empty']+group['current_occupied'],group['entities'])
            self.assertTrue(np.isfinite(group['vp_mae']));self.assertTrue(np.isfinite(group['owner_ce']))
        self.assertEqual(before,model_id(m));self.assertFalse(a.retained)
        saved=a.payload(0);config=dict(copy.deepcopy(auxiliary.DEFAULT),mode='off')
        off=auxiliary.Companion(m,config,raw=dict(training={auxiliary.KEY:saved}))
        state=copy.deepcopy(off.optimizers['ppg'].state_dict());off.begin('ppg',reset=True,rate_factor=.1)
        ix,x,_=ds.batch([0,1]);out,loss=off.forward(m,x,ds,ix);off.step('ppg')
        self.assertIsNone(loss);self.assertTrue(_same_payload(state,off.optimizers['ppg'].state_dict()))

    def test_exact_bundle_resume_modes_and_joint_rollback(self):
        m,_,_,(_,data)=fixture(games=2);annotate(data);ds=updates.Dataset(data,'cpu');ix,x,_=ds.batch([0,1])
        a=auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT));a.begin('ppo');_,loss=a.forward(m,x,ds,ix);loss.mean().backward();a.step('ppo')
        saved=a.payload(1);b=auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT),raw=dict(training={auxiliary.KEY:saved}),generation=1)
        self.assertTrue(_same_payload(saved,b.payload(1)))
        core_before=model_id(m);ppo_before=copy.deepcopy(a.optimizers['ppo'].state_dict());snap=a.snapshot()
        a.begin('ppg',reset=True,rate_factor=.1);_,loss=a.forward(m,x,ds,ix);loss.mean().backward();a.step('ppg');a.restore(snap)
        self.assertTrue(_same_payload(snap,a.snapshot()));self.assertTrue(_same_payload(ppo_before,a.optimizers['ppo'].state_dict()))
        self.assertEqual(core_before,model_id(m))
        with self.assertRaisesRegex(ValueError,'Missing adopted'):auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT),generation=1)
        bad=copy.deepcopy(saved);bad['heads']['slots.owner.bias'][0]+=1
        with self.assertRaisesRegex(ValueError,'identity'):auxiliary.Companion(m,copy.deepcopy(auxiliary.DEFAULT),raw=dict(training={auxiliary.KEY:bad}),generation=1)


if __name__=='__main__':unittest.main()
