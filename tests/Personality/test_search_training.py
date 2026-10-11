import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from fixtures import position
from personality_fixtures import parent
from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,digest,read_json
from brass_ai.learning.reward import specification
from brass_ai.personality import checkpoint
from brass_ai.personality.contracts import context,contract
from brass_ai.personality.model import import_parent
from brass_ai.personality.search.contracts import identity,Profile,search_config
from brass_ai.personality.search.engine import teacher
from brass_ai.personality.search import teachers
from brass_ai.personality.search.continuation import continue_pilot
from brass_ai.personality.search.readiness import confidence_calibration,distilled_source
from brass_ai.personality.search.calibration import examples as calibration_examples
from brass_ai.personality.contracts import GOALS
from brass_ai.recovery.teacher_schedule import PHASES
from brass_ai.step5.league import write_json


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_uninformative_or_miscalibrated_goal_cannot_enable_bias(self):
        import numpy as np
        labels=[dict(confidence=[[0.]*3]*6,confidence_mask=[[True]*3]*6) for _ in range(8)]
        result=confidence_calibration(np.zeros((8,6,3)),labels,['one']*4+['two']*4)
        self.assertFalse(result['bias_ready']);self.assertEqual(result['ece'],.5)

    def test_distilled_readiness_requires_the_exact_pilot_generation(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);pilot=directory/'pilot';pilot.mkdir();distilled=directory/'distilled';distilled.mkdir()
            (distilled/'model.pt').write_bytes(b'authored-lineage-fixture')
            state=dict(model_sha256='current-pilot')
            ready=dict(id='ready',handoff_sha256='handoff',origin=dict(
                pilot=pilot.relative_to(SOURCE_ROOT).as_posix(),pilot_model_sha256='current-pilot'))
            report=dict(updated=True,contract=identity(),handoff_sha256='handoff',parent_ready_id='ready',
                parent_ready='unused-mocked-ready',model_sha256=sha(distilled/'model.pt'))
            write_json(distilled/'report.json',report)
            with patch('brass_ai.personality.search.readiness.load',return_value=(None,None,ready,None,None)):
                self.assertEqual(distilled_source(SOURCE_ROOT,pilot,state,distilled,'handoff'),distilled/'model.pt')
                ready['origin']['pilot_model_sha256']='earlier-pilot-generation'
                with self.assertRaisesRegex(ValueError,'PPO source changed'):
                    distilled_source(SOURCE_ROOT,pilot,state,distilled,'handoff')
                ready['origin']['pilot_model_sha256']='current-pilot'
                ready['origin']['pilot']='artifacts/another-pilot'
                with self.assertRaisesRegex(ValueError,'PPO source changed'):
                    distilled_source(SOURCE_ROOT,pilot,state,distilled,'handoff')
                ready['origin']['pilot']=pilot.relative_to(SOURCE_ROOT).as_posix();ready['id']='different-readiness'
                with self.assertRaisesRegex(ValueError,'PPO source changed'):
                    distilled_source(SOURCE_ROOT,pilot,state,distilled,'handoff')

    def test_calibration_uses_actual_saved_intent_and_is_never_ppo_or_discovery(self):
        from personality_fixtures import episode
        from brass_ai.personality.repertoire.traces import from_collection
        game=episode(2);game['split']='train';intent=dict(schema=GOALS,family='industry_sequence',industry='CottonMill',region='unspecified',
            horizon=4,confidence=.9,age=1,created_turn='recorded-prior-turn',revision_reason='productive-progress')
        records=[dict(role='discovery' if r['seat']==0 else 'parent',teacher=None,model_id='model',context=context('industrialist',intent)) for r in game['records']]
        manifest=dict(spec=dict(calibration_probe=True,model_id='model'))
        data=calibration_examples([(game,dict(records=records))],manifest,'model','train',specification())
        self.assertTrue(all(d['saved_intent'] and not d['on_policy'] for d in data))
        self.assertTrue(all(d['context_origin']=='actual-p2-direct-readiness' for d in data))
        with patch('brass_ai.personality.search.collection.read',return_value=(manifest,[])):
            with self.assertRaisesRegex(ValueError,'cannot fit discovery'):from_collection('unused')

    def test_policy_only_distill_reuse_gate_and_successor_ppo_state(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            directory=Path(tmp);pilot=directory/'pilot';pilot.mkdir();bundle=directory/'ready';bundle.mkdir()
            model,_=import_parent(parent());optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
            # Materialize Adam state for every tensor, as a continuing PPO pilot.
            sum(p.square().sum() for p in model.parameters()).backward();optimizer.step();optimizer.zero_grad()
            ident=checkpoint.save(pilot/'model.pt',model,optimizer,training=dict(phase='joint'))
            checkpoint.save(pilot/'initial.pt',model)
            state=dict(format='personality-pilot/1.0.0',contract=contract(),model_sha256=sha(pilot/'model.pt'),model_id=ident,
                handoff_sha256='handoff-fixture',warmup_runs=1,joint_runs=20,validation_runs=0)
            write_json(pilot/'state.json',state)
            ready=dict(id='ready',model=dict(model_id=ident),reward=specification(),handoff_sha256=state['handoff_sha256'],
                origin=dict(pilot=pilot.relative_to(SOURCE_ROOT).as_posix(),pilot_model_sha256=state['model_sha256']))
            write_json(bundle/'ready.json',ready);collection=directory/'teacher';collection.mkdir()
            cfg=search_config(specification());pairs=[]
            for game_index in range(4):
                rows=[];side=[];reservations={}
                for phase,era,lo,hi in PHASES:
                    view,choices=position();view['context'].update(era=era,round=lo)
                    trace=[('root',choices[0]['id'])];sig=digest(trace)
                    stats=dict(root={c['id']:dict(visits=n,returns=[1.]*4,available=3) for c,n in zip(choices,[2,1,0]) if n},expansion=[True]*3,
                        plans={sig:dict(signature=sig,first=choices[0]['id'],trace=trace,visits=2,returns=[1.]*4)})
                    scope=dict(context=context(),goal=None,profile=Profile(),inject=False,information='root',view_hash=digest(view),id='context',calibration_id='fixture')
                    label=teacher(choices,stats,[.4,.3,.3],[.4,.3,.3],model=ident,seat=0,config=cfg,scope=scope,remaining=[.025]*4,selected=0)
                    rows.append(dict(view=view,choices=choices,seat=0,chosen=0,teacher=None,style=[0.]*4,intent=[0.]*8,behavior=dict(kind='external')))
                    side.append(dict(teacher=label));reservations[phase]=lo
                pairs.append((dict(records=rows,split='train',setup_group=str(game_index)),dict(records=side,reservations=[reservations]*4)))
            manifest=dict(id='teacher-fixture',spec=dict(ready_id=ready['id'],model_id=ident));write_json(collection/'manifest.json',manifest)
            old={k:v.clone() for k,v in model.state_dict().items()}
            with patch.object(teachers,'load',return_value=(model,[],ready,{},'pool')),patch.object(teachers,'read',return_value=(manifest,pairs)):
                report=teachers.distill(SOURCE_ROOT,bundle,collection,directory/'distilled')
                self.assertTrue(report['updated']);self.assertLessEqual(report['max_presentations'],2)
                with self.assertRaisesRegex(ValueError,'already presented'):teachers.distill(SOURCE_ROOT,bundle,collection,directory/'reused')
            learned,raw=checkpoint.load(directory/'distilled/model.pt')
            self.assertFalse(learned.core.utility_calibration['ready']);self.assertFalse(report['value_loss_training'])
            for key in old:
                if key.startswith(('core.outcomes','core.utility_head','goal_','confidence_')):self.assertTrue(torch.equal(old[key],learned.state_dict()[key]))
            self.assertTrue(any(not torch.equal(v,learned.state_dict()[k]) for k,v in old.items()))
            original=sha(pilot/'model.pt')
            with patch('brass_ai.personality.search.continuation.load',return_value=(model,[],ready,{},'pool')):
                result=continue_pilot(SOURCE_ROOT,pilot,directory/'distilled',directory/'successor')
            self.assertTrue(result['ppo_optimizer_preserved']);self.assertEqual(result['joint_runs_preserved'],20)
            self.assertEqual(sha(pilot/'model.pt'),original)
            _,before=checkpoint.load(pilot/'model.pt');_,after=checkpoint.load(directory/'successor/model.pt')
            self.assertEqual(before['optimizer']['param_groups'],after['optimizer']['param_groups'])
            for key,state in before['optimizer']['state'].items():
                for name,tensor in state.items():self.assertTrue(torch.equal(tensor,after['optimizer']['state'][key][name]))
            self.assertEqual(after['training']['phase'],'joint')


if __name__=='__main__':unittest.main()
