"""Four-phase fixed-IL KL evidence and PPO-only increase admission."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from brass_ai.learning.contracts import digest
from brass_ai.recovery import lr_controller as controller
from brass_ai.recovery.history import compact
from brass_ai.recovery.learning_rate import configure as configure_warmup
from brass_ai.recovery.review_metrics import CategoricalProbe
from brass_ai.recovery.teacher_schedule import PHASES,phase_for
from test_ppo_lr_controller import config,measurements,ready


class GroupAdmission(unittest.TestCase):
    def decision(self,**kwargs):
        state=ready()
        for _ in range(3):state,report=controller.decide(state,measurements(**kwargs))
        return state,report

    def test_versioned_defaults_and_strict_group_settings(self):
        cfg=config()['ppo_lr_controller']
        self.assertEqual(cfg,controller.DEFAULT_CONTROLLER)
        self.assertEqual((cfg['anchor_kl_budget'],cfg['anchor_group_kl_budget'],cfg['anchor_group_min_positions']),(.15,.30,64))
        for key,value in (('anchor_group_kl_budget',0.),('anchor_group_kl_budget',float('nan')),
                          ('anchor_group_kl_budget',True),('anchor_group_min_positions',0),
                          ('anchor_group_min_positions',True),('anchor_group_min_positions',64.),
                          ('anchor_group_min_positions',513)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                controller.validate_config(dict(cfg,**{key:value}))
        for key in controller.GROUP_FIELDS:
            bad=copy.deepcopy(cfg);del bad[key]
            with self.assertRaises(ValueError):controller.validate_config(bad)

    def test_mean_and_group_boundaries_are_inclusive_and_independent(self):
        groups=measurements()['anchor_groups']
        for row in groups.values():row['anchor_distance']=.10
        groups['rail_late']['anchor_distance']=.30
        _,report=self.decision(anchor_distance=.15,anchor_groups=groups)
        self.assertEqual(report['decision'],'up')
        self.assertEqual(report['health']['anchor_group_max'],.30)
        _,report=self.decision(anchor_distance=.150001,anchor_groups=groups)
        self.assertIn('anchor_budget_exceeded',report['health']['reasons'])
        self.assertEqual(report['decision'],'hold')
        groups['rail_late']['anchor_distance']=.300001
        _,report=self.decision(anchor_distance=.14,anchor_groups=groups)
        self.assertEqual(report['decision'],'hold')
        self.assertEqual(report['health']['reasons'],['anchor_group_budget_exceeded:rail_late'])

    def test_insufficient_missing_nonfinite_or_unassigned_evidence_only_blocks_raises(self):
        base=measurements()['anchor_groups']
        cases=[]
        for entry,reason in ((dict(positions=63,anchor_distance=.9),'insufficient_samples'),
                             (dict(positions=0,anchor_distance=None),'insufficient_samples'),
                             (dict(positions=64,anchor_distance=float('nan')),'unavailable'),
                             (dict(positions=64,anchor_distance=float('inf')),'unavailable'),
                             (dict(positions=True,anchor_distance=.1),'unavailable'),
                             (dict(positions=-1,anchor_distance=.1),'unavailable')):
            groups=copy.deepcopy(base);groups['rail_late']=entry
            cases.append((dict(anchor_groups=groups),'anchor_group_'+reason+':rail_late'))
        missing=copy.deepcopy(base);del missing['rail_late']
        cases.extend(((dict(anchor_groups=missing),'anchor_group_unavailable:rail_late'),
                      (dict(anchor_groups=None),'anchor_group_unavailable:rail_late'),
                      (dict(anchor_unassigned_positions=1),'anchor_group_unassigned_positions'),
                      (dict(anchor_unassigned_positions=None),'anchor_group_coverage_unavailable')))
        for kwargs,reason in cases:
            with self.subTest(reason=reason,kwargs=kwargs):
                state,report=self.decision(**kwargs)
                self.assertEqual(report['decision'],'hold');self.assertIn(reason,report['health']['reasons'])
                digest(report)  # Nonfinite input is represented as missing evidence.
                _,cut=controller.decide(state,measurements(kl=.02,**kwargs))
                self.assertEqual(cut['decision'],'down')

    def test_recovery_has_no_hysteresis_or_additional_cooldown(self):
        groups=measurements()['anchor_groups'];groups['rail_late']['anchor_distance']=.31
        state,report=self.decision(anchor_distance=.14,anchor_groups=groups)
        self.assertEqual(report['decision'],'hold');self.assertEqual(state['cooldown'],0)
        groups['rail_late']['anchor_distance']=.29
        state,report=controller.decide(state,measurements(anchor_distance=.14,anchor_groups=groups))
        self.assertEqual(report['decision'],'up');self.assertEqual(state['multiplier'],1.25)
        self.assertEqual(set(state),set(ready()))

    def test_legacy_contract_is_not_silently_reinterpreted(self):
        old={k:v for k,v in controller.DEFAULT_CONTROLLER.items() if k not in controller.GROUP_FIELDS}
        old['contract']=controller.LEGACY_CONTRACT
        controller.validate_config(old)
        state=controller.initial_state(old);state['health_window']=[100.]*5
        for _ in range(3):state,report=controller.decide(state,measurements(anchor_groups=None))
        self.assertEqual(report['decision'],'up');self.assertEqual(report['contract'],controller.LEGACY_CONTRACT)
        self.assertNotIn('anchor_groups',report['health'])
        with self.assertRaises(ValueError):controller.validate_config(dict(old,anchor_group_kl_budget=.30))
        # The optimizer application happens only after the external signature review.
        p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=6e-6)
        configure_warmup(opt,6e-6,0,.1,'selfplay');opt.param_groups[0][controller.KEY]=copy.deepcopy(state)
        controller.configure(opt,controller.DEFAULT_CONTROLLER)
        actual=opt.param_groups[0][controller.KEY]
        self.assertEqual({k:v for k,v in actual.items() if k!='config'},
                         {k:v for k,v in state.items() if k!='config'})
        self.assertEqual(actual['config'],controller.DEFAULT_CONTROLLER)

    def test_finish_iteration_and_compact_history_keep_group_evidence(self):
        p=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([p],lr=6e-6)
        configure_warmup(opt,6e-6,0,.1,'selfplay');controller.configure(opt,controller.DEFAULT_CONTROLLER)
        opt.param_groups[0][controller.KEY]=ready()
        opt.param_groups[0][controller.KEY]['window']=[dict(kl=.002,early_stop=False)]*2
        groups=measurements()['anchor_groups'];groups['rail_late']['anchor_distance']=.31
        probe=dict(positions=256,categorical_kl=.002,anchor_distance=.1,anchor_groups=groups,anchor_unassigned_positions=0)
        update=dict(categorical_probe=probe,optimizer_steps=10,optimizer_steps_planned=10,
                    kl_early_stop=False,controller_warmup_done=True,learning_rates=[6e-6])
        sampling=dict(by_composition=dict(selfplay=dict(tables=1,learner_trajectories=4)))
        with patch.object(controller,'game_health',return_value=dict(selfplay_vp_per_seat=100.,zero_vp_fraction=.01)):
            report=controller.finish_iteration(opt,update,config(),sampling,[],'learner',{},dict(model_id='fixed-il'))
        self.assertEqual(report['decision'],'hold')
        self.assertEqual(report['health']['anchor_groups']['rail_late'],
                         dict(positions=64,anchor_distance=.31,status='budget_exceeded'))
        summary=compact(dict(actor=dict(lr_controller=report,categorical_probe=probe)))
        self.assertEqual(summary['actor']['lr_controller'],report);digest(summary)


class ProbeFixture:
    """Unequal groups, variable legal masks, and reordered candidate buckets."""
    def __init__(self):
        self.rows=[];logits=[];widths=[]
        for group,era,first,last in PHASES:
            count=128 if group=='canal_early' else 64
            for i in range(count):
                self.rows.append(dict(view=dict(context=dict(era=era,round=first if i%2 else last))))
                widths.append(2+i%2)
                logits.append([0.,-2.5,0.] if group=='rail_late' else [0.,0.,0.])
        self.logits=torch.tensor(logits);self.widths=widths

    def microbatches(self,indices,size,cells,bucket):
        ordered=sorted(indices,key=lambda i:self.widths[i]) if bucket else list(indices)
        for start in range(0,len(ordered),size):yield ordered[start:start+size]

    def batch(self,indices,exploration):
        ix=torch.tensor(indices,dtype=torch.long)
        legal=torch.arange(3)[None,:]<torch.tensor([self.widths[i] for i in indices])[:,None]
        q=torch.tensor([.6,.3,.1]).expand(len(indices),-1)*legal
        q=q/q.sum(-1,keepdim=True)
        return ix,(ix[:,None].float(),torch.zeros(len(indices),3,1),legal,torch.zeros(len(indices),4)),q


class FixedPolicy(torch.nn.Module):
    def __init__(self,logits):
        super().__init__();self.register_buffer('logits',logits);self.calls=0

    def forward(self,x,a,legal,mapping):
        self.calls+=1
        return (self.logits[x[:,0].long()].masked_fill(~legal,-torch.inf),)


class ProbeGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_existing_phase_boundaries(self):
        self.assertEqual(PHASES,(('canal_early','CanalEra',1,4),('canal_late','CanalEra',5,8),
                                 ('rail_early','RailEra',1,6),('rail_late','RailEra',7,8)))
        for name,era,first,last in PHASES:
            for round_ in range(first,last+1):self.assertEqual(phase_for(dict(era=era,round=round_)),name)

    def test_ordinary_reader_compaction_preserves_rounds_for_the_real_probe(self):
        from brass_ai.recovery.ingress import _prepare_inner
        ds=ProbeFixture();data=[]
        for i,row in enumerate(ds.rows):
            row['view']['context']['phase']='action'
            row.update(seat=i%4,chosen=0,behavior=dict(kind='model',policy_id='learner'),
                       choices=[dict(kind='pass'),dict(kind='loan')],style=[0.]*4,intent=[0.]*8,teacher=None)
            data.append((row,{},('game',i%4),object()))
        receipt=dict(path='fixture.jsonl.gz',sha256='checked',game_id='game')
        games=[dict(split='train',game_id='game',records=ds.rows)]
        with patch('brass_ai.recovery.ingress.sha',return_value='checked'), \
             patch('brass_ai.recovery.ingress.read_shards',return_value=games), \
             patch('brass_ai.recovery.ingress.summarize',return_value={}), \
             patch('brass_ai.recovery.ingress.prepared',return_value=data):
            _,packed,_,_=_prepare_inner(([receipt],'learner',{},None,None))
        for original,actual in zip(data,packed):
            self.assertIs(actual[3],original[3])
            self.assertEqual(actual[0]['view']['context'],original[0]['view']['context'])
        ds.rows=[item[0] for item in packed]
        model=FixedPolicy(ds.logits);anchor=FixedPolicy(torch.zeros_like(ds.logits))
        probe=CategoricalProbe(model,ds,list(range(len(ds.rows))),SimpleNamespace(exploration=.08,temperature=1.),
                              dict(microbatch_size=17,max_candidate_cells=8192,bucket=True))
        report=probe.measure(model,anchor=anchor)
        self.assertEqual(report['anchor_unassigned_positions'],0)
        self.assertEqual({p:r['positions'] for p,r in report['anchor_groups'].items()},
                         dict(canal_early=128,canal_late=64,rail_early=64,rail_late=64))

    def test_weighted_means_masks_mixture_no_extra_forwards_or_rng(self):
        ds=ProbeFixture();indices=list(reversed(range(len(ds.rows))))
        model=FixedPolicy(ds.logits);anchor=FixedPolicy(torch.zeros_like(ds.logits))
        cfg=dict(microbatch_size=17,max_candidate_cells=8192,bucket=True)
        rng=torch.get_rng_state().clone()
        probe=CategoricalProbe(model,ds,indices,SimpleNamespace(exploration=.08,temperature=1.),cfg)
        result=probe.measure(model,anchor=anchor)
        batches=math.ceil(len(indices)/17)
        self.assertEqual(model.calls,2*batches);self.assertEqual(anchor.calls,batches)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertAlmostEqual(result['categorical_kl'],0.,places=7)
        self.assertEqual(result['anchor_unassigned_positions'],0)
        _,(_,_,legal,_),q=ds.batch(list(range(len(ds.rows))),.08)
        p=.92*torch.softmax(ds.logits.masked_fill(~legal,-torch.inf),-1)+.08*q
        ref=.92*legal.float()/legal.sum(-1,keepdim=True)+.08*q
        expected=(ref*(ref.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())).sum(-1).tolist()
        self.assertAlmostEqual(result['anchor_distance'],sum(expected)/len(expected),places=7)
        for name,_,_,_ in PHASES:
            ids=[i for i,row in enumerate(ds.rows) if phase_for(row['view']['context'])==name]
            actual=result['anchor_groups'][name]
            self.assertEqual(actual['positions'],len(ids))
            self.assertAlmostEqual(actual['anchor_distance'],sum(expected[i] for i in ids)/len(ids),places=7)
        unweighted=sum(row['anchor_distance'] for row in result['anchor_groups'].values())/4
        self.assertGreater(abs(result['anchor_distance']-unweighted),.01)
        # The aggregate passes while the supported late-Rail group fails.
        self.assertLess(result['anchor_distance'],.15)
        self.assertGreater(result['anchor_groups']['rail_late']['anchor_distance'],.30)
        state=ready()
        for _ in range(3):
            state,report=controller.decide(state,measurements(anchor_distance=result['anchor_distance'],
                anchor_groups=result['anchor_groups'],anchor_unassigned_positions=result['anchor_unassigned_positions']))
        self.assertEqual(report['health']['reasons'],['anchor_group_budget_exceeded:rail_late'])
        self.assertEqual(report['decision'],'hold')

    def test_empty_and_unmapped_rows_are_visible_and_not_relabelled(self):
        ds=ProbeFixture();ds.rows[0]['view']['context'].pop('round')
        model=FixedPolicy(ds.logits);anchor=FixedPolicy(torch.zeros_like(ds.logits))
        cfg=dict(microbatch_size=4,max_candidate_cells=8192,bucket=True)
        for indices in ([],[0,1]):
            probe=CategoricalProbe(model,ds,indices,SimpleNamespace(exploration=.08,temperature=1.),cfg)
            result=probe.measure(model,anchor=anchor)
            self.assertEqual(sum(r['positions'] for r in result['anchor_groups'].values())+
                             result['anchor_unassigned_positions'],len(indices))
            self.assertEqual(result['anchor_unassigned_positions'],1 if indices else 0)
            self.assertEqual(result['anchor_groups']['rail_late'],dict(positions=0,anchor_distance=None))
            self.assertNotIn('anchor_groups',probe.measure(model))
            digest(result)


if __name__=='__main__':unittest.main()
