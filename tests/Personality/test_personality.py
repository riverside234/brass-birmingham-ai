import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fixtures import position
from personality_fixtures import parent,episode,qualified_run
from brass_ai.bank import sha
from brass_ai.learning.batching import collate
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,digest,read_json
from brass_ai.learning.features import Encoder
from brass_ai.learning.reward import specification,replay_returns
from brass_ai.recovery.retention import run_lock,references,plan
from brass_ai.step5.league import write_json,implementation_id
from brass_ai.personality import checkpoint,handoff
from brass_ai.personality.contracts import (context,context_vector,CONTEXT_DIM,FAMILIES,INDUSTRIES,
    contract,local_file)
from brass_ai.personality.model import import_parent,base_outputs
from brass_ai.personality.labels import goal_labels,loss_rows,confidence_report,events
from brass_ai.personality.traces import examples
from brass_ai.personality.training import fit,forward,execute
from brass_ai.personality.collection import distribution


class PersonalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def temp(self):
        value=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(value.cleanup);return Path(value.name)

    def encoded(self):
        encoder=Encoder();rows=[encoder.encode(*position(s)) for s in range(4)]
        return collate(rows)

    def test_named_outputs_exact_neutral_import_and_parameter_budget(self):
        source=parent('medium')
        with torch.no_grad():
            source.utility_head[-1].weight.normal_(0,.01);source.utility_head[-1].bias.fill_(1.2)
        model,report=import_parent(source)
        args=self.encoded();budget=torch.tensor([[0.,.005,.01,.025]]*4)
        with torch.no_grad():
            expected=source(*args,remaining=budget)
            actual=model(*args,personality=torch.zeros(4,CONTEXT_DIM),remaining=budget)
        for a,b in zip(expected,base_outputs(actual)):self.assertTrue(torch.equal(a,b))
        self.assertTrue((actual['remaining_utility_4']>1.).all())  # Not clipped to probabilities.
        self.assertEqual(report['parameters'],dict(base=1850129,extension=65833,total=1915962))
        self.assertEqual(set(actual['goals']),{'family','industry','region','horizon'})
        self.assertEqual(actual['confidence_logits'].shape,(4,6,3))
        self.assertFalse(report['rejected']);self.assertEqual(len(report['copied']),len(source.state_dict()))

    def test_old_reader_rejects_new_checkpoint_and_restore_is_exact(self):
        directory=self.temp();model,_=import_parent(parent());model.freeze_core()
        data=examples([episode()],specification(),roots_only=True)
        optimizer,report=fit(model,data,'cpu',epochs=1)
        identity=checkpoint.save(directory/'model.pt',model,optimizer,training=dict(phase='warmup'))
        restored,raw=checkpoint.load(directory/'model.pt')
        self.assertEqual(identity,checkpoint.model_id(restored));self.assertTrue(raw['optimizer']['state'])
        for name,t in model.state_dict().items():self.assertTrue(torch.equal(t,restored.state_dict()[name]))
        with self.assertRaises(ValueError):load_checkpoint(directory/'model.pt')
        optimizer2,_=fit(restored,data,'cpu',epochs=1,resume=raw)
        self.assertGreater(float(next(iter(optimizer2.state.values()))['step']),float(next(iter(optimizer.state.values()))['step']))

    def test_handoff_verifies_parent_copies_dependencies_and_pins_cleanup(self):
        directory=self.temp();run=qualified_run(directory/'source')
        source=implementation_id();manifest=handoff.create(SOURCE_ROOT,run,directory/'handoff',reference_count=2)
        loaded,model=handoff.load(directory/'handoff')
        self.assertEqual(manifest,loaded);self.assertEqual(model.config.utility,specification())
        self.assertEqual(len(manifest['traces']),1);self.assertEqual(len(manifest['references']),1)
        pins=references(SOURCE_ROOT,run)
        self.assertIn((run/'generation-0001.pt').resolve(),pins)
        self.assertIn((run/'iteration-0000/games.jsonl.gz').resolve(),pins)
        plan(SOURCE_ROOT,run)  # Existing cleanup handles the new pins without code changes.
        self.assertEqual(source,implementation_id())
        with self.assertRaisesRegex(ValueError,'new immutable'):handoff.create(SOURCE_ROOT,run,directory/'handoff')
        (directory/'handoff/parent.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'dependency changed'):handoff.load(directory/'handoff')

    def test_handoff_rejects_active_il_only_and_tampered_best(self):
        directory=self.temp();run=qualified_run(directory/'source',traces=False)
        with run_lock(run):
            with self.assertRaisesRegex(ValueError,'active'):handoff.inspect(SOURCE_ROOT,run)
        best=read_json(run/'best.json');best['checkpoint']['generation']=0
        state=read_json(run/'state.json');state['champion']['generation']=0
        write_json(run/'best.json',best);write_json(run/'state.json',state)
        with self.assertRaisesRegex(ValueError,'post-IL'):handoff.inspect(SOURCE_ROOT,run)
        best['checkpoint']['generation']=state['champion']['generation']=1
        write_json(run/'best.json',best);write_json(run/'state.json',state)
        (run/'best.pt').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'bytes changed'):handoff.inspect(SOURCE_ROOT,run)

    def test_handoff_rejects_reward_and_gate_mismatch(self):
        directory=self.temp();run=qualified_run(directory/'source',traces=False)
        state=read_json(run/'state.json');state['signature']['config']['reward']['vp']=.0005
        write_json(run/'state.json',state)
        with self.assertRaisesRegex(ValueError,'another reward'):handoff.inspect(SOURCE_ROOT,run)
        state['signature']['config']['reward']['vp']=.001;write_json(run/'state.json',state)
        path=run/'evaluations/champion-00006400/report.json';report=read_json(path);report['failed_games']=1
        write_json(path,report)
        with self.assertRaisesRegex(ValueError,'report changed'):handoff.inspect(SOURCE_ROOT,run)

    def test_legacy_traces_are_auxiliary_only_and_no_evaluation_ingress(self):
        data=examples([episode()],specification(),roots_only=True)
        self.assertTrue(all(not d['on_policy'] and d['context_origin']=='legacy-absent-auxiliary-only' for d in data))
        model,_=import_parent(parent())
        with self.assertRaisesRegex(ValueError,'PPO rejects'):fit(model,data,'cpu',epochs=1,joint=True)
        bad=copy.deepcopy(data);bad[0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'cannot train'):fit(model,bad,'cpu',epochs=1)
        directory=self.temp();run=qualified_run(directory/'source')
        path=run/'iteration-0000/games.jsonl.gz'
        with gzip.open(path,'rt') as f:lines=f.readlines()
        header=json.loads(lines[0]);header['provenance']['purpose']='evaluation';lines[0]=json.dumps(header)+'\n'
        with gzip.open(path,'wt') as f:f.writelines(lines)
        report=read_json(path.parent/'report.json');report['execution'][0]['sha256']=sha(path);write_json(path.parent/'report.json',report)
        with self.assertRaisesRegex(ValueError,'ordinary training'):handoff.create(SOURCE_ROOT,run,directory/'handoff')

    def test_context_boundary_no_implicit_hindsight_or_unknown_fields(self):
        self.assertTrue(np.array_equal(context_vector(context()),np.zeros(CONTEXT_DIM)))
        c=context('supplier');c['opponent_hand']=['hidden']
        with self.assertRaises(ValueError):context_vector(c)
        with self.assertRaises(ValueError):local_file(self.temp(),'../escape.pt')
        with self.assertRaises(ValueError):context('unknown')

    def test_public_labels_own_turn_horizons_ambiguity_and_censoring(self):
        game=episode();labels=goal_labels(game);liquid=FAMILIES.index('liquidity_recovery')
        self.assertEqual(labels[0]['values']['family'],liquid)
        self.assertEqual(labels[0]['confidence'][liquid],[1.,1.,1.])
        self.assertTrue(labels[0]['confidence_mask'][FAMILIES.index('sale_chain')][2])
        last=max(labels);self.assertFalse(labels[last]['confidence_mask'][0][2])
        altered=copy.deepcopy(game)
        for row in altered['records'][1:]:row['view']['own_hand']=['private-other']
        self.assertEqual(goal_labels(altered),labels)  # Future hands cannot affect targets.
        self.assertEqual(len(labels),24)  # Both actions/subchoices never count as extra own turns.

    def test_unsold_build_is_not_completed_industry_goal_and_pottery_is_supported(self):
        game=episode();slot=game['records'][0]['view']['slots'][0]['id']
        tile=copy.deepcopy(game['records'][0]['view']['players'][0]['mat'][0]['tiles'][0])
        tile.update(type='Pottery',seat=0,flipped=False,id='test-pottery')
        for row in game['records'][1:]:next(s for s in row['view']['slots'] if s['id']==slot)['tile']=copy.deepcopy(tile)
        before=events(game)
        self.assertFalse(any(e['family']=='industry_sequence' for e in before))
        for row in game['records'][3:]:next(s for s in row['view']['slots'] if s['id']==slot)['tile']['flipped']=True
        self.assertTrue(any(e['family']=='industry_sequence' and e['industry']=='Pottery' for e in events(game)))
        self.assertIn('Pottery',INDUSTRIES)

    def test_goal_horizons_stop_at_era_and_ambiguous_completion_is_masked(self):
        game=episode()
        for row in game['records'][8:]:row['view']['context']['era']='RailEra'
        labels=goal_labels(game)
        self.assertFalse(labels[0]['confidence_mask'][0][2])  # Cannot use next-era turns to finish a Canal horizon.
        game=episode();link=game['records'][0]['view']['links'][0]['id']
        for row in game['records'][1:]:next(x for x in row['view']['links'] if x['id']==link)['owner']=0
        label=goal_labels(game)[0]  # Simultaneous public network + liquidity effects.
        self.assertFalse(label['mask']['family']);self.assertFalse(label['mask']['region'])
        self.assertTrue(label['confidence'][FAMILIES.index('network_extension')][0])

    def test_warmup_changes_only_auxiliary_parameters_and_reports_masks(self):
        model,_=import_parent(parent());before={k:v.clone() for k,v in model.state_dict().items()}
        data=examples([episode()],specification(),roots_only=True);_,report=fit(model,data,'cpu',epochs=2)
        changed=[k for k,v in model.state_dict().items() if not torch.equal(v,before[k])]
        self.assertTrue(changed);self.assertTrue(all(k.startswith(('goal_','confidence_')) for k in changed))
        self.assertTrue(np.isfinite(report['after']['goal_loss']));self.assertGreater(report['after']['confidence']['samples'],0)

    def test_joint_update_requires_matching_behavior_and_uses_remaining_utility(self):
        model,_=import_parent(parent());data=examples([episode()],specification())
        data=[d for d in data if d['row']['seat']==0]
        for d in data:d['on_policy']=True
        with torch.no_grad():
            out=forward(model,data,list(range(len(data))),'cpu')
            p=distribution(out['policy_logits'],[d['row']['choices'] for d in data])
            for i,d in enumerate(data):d['row']['behavior']['probability']=float(p[i,d['row']['chosen']])
        optimizer,report=fit(model,data,'cpu',epochs=1,joint=True,cfg=dict(batch_size=64,microbatch_size=8))
        self.assertGreater(report['actor_rows'],0);self.assertTrue(np.isfinite(report['after']['utility_mse']))
        self.assertEqual(report['reward'],specification())
        bad=copy.deepcopy(data);bad[0]['row']['behavior']['probability']=.999
        with self.assertRaisesRegex(ValueError,'likelihood'):fit(model,bad,'cpu',epochs=1,joint=True)

    def test_cli_warmup_resume_and_protected_source(self):
        directory=self.temp();run=qualified_run(directory/'source');before=sha(run/'state.json')
        handoff.create(SOURCE_ROOT,run,directory/'handoff',trace_shards=1)
        first=execute(SOURCE_ROOT,directory/'handoff',directory/'pilot',epochs=1)
        second=execute(SOURCE_ROOT,directory/'handoff',directory/'pilot',epochs=1)
        state=read_json(directory/'pilot/state.json');self.assertEqual(state['warmup_runs'],2)
        self.assertEqual(before,sha(run/'state.json'))
        self.assertTrue(first['parent_outputs_preserved'] and second['parent_outputs_preserved'])


if __name__=='__main__':unittest.main()
