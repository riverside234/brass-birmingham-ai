"""Bounded diagnostics parity and crash-safe amortized retention."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock

import numpy as np
import torch
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import il_kl_probe as probes,il_kl_limit,retention
from brass_ai.recovery.config import RETENTION,bounded_fields
from brass_ai.step5.league import write_json
import test_auto_il_kl as probe_fixtures
from test_auto_il_kl import ToyPolicy


def probe_fixture(rows=8):
    data=probe_fixtures.ProbeTests().data();selected=[data[i%len(data)] for i in range(rows)]
    arrays=dict(states=np.stack([d[3].state for d in selected]),
        actions=np.concatenate([d[3].actions[:2+i%2] for i,d in enumerate(selected)]),
        seats=np.stack([d[3].stable_to_relative for d in selected]),
        offsets=np.array([0]+list(np.cumsum([2+i%2 for i in range(rows)]))))
    arrays['reference']=np.concatenate([np.full(2+i%2,1/(2+i%2),dtype=np.float32) for i in range(rows)])
    metadata=dict(rows=[dict(**d[0]['view']['context'],group=probes.group_for(d[0]['view']['context']),
        kinds=['sell','pass','loan'][:2+i%2],choice_ids=list(d[3].choice_ids[:2+i%2])) for i,d in enumerate(selected)])
    return probes.Probe(metadata,arrays,copy.deepcopy(il_kl_limit.DEFAULT))


class ProbeOptimization(unittest.TestCase):
    def test_full_distribution_parity_floors_masks_and_no_il_comparison(self):
        probe=probe_fixture();model=ToyPolicy();model.train();rng=torch.get_rng_state().clone()
        update=dict(microbatch_size=3,max_candidate_cells=9)
        reference=probe.arrays['reference'].copy();reference[:2]=[1e-10,1.-1e-10]
        report,values=probe.evaluate(model,update,reference=reference)
        expected=[]
        for a,b in zip(probe.arrays['offsets'][:-1],probe.arrays['offsets'][1:]):
            p=torch.tensor(reference[a:b])[None,:];q=torch.tensor(values[a:b])[None,:]
            expected.append(float(probes.stable_kl(p,q,torch.ones_like(p,dtype=torch.bool),1e-6)))
        self.assertAlmostEqual(report['kl_mean'],np.mean(expected),places=12)
        self.assertNotIn('original_il',report);self.assertFalse(report['original_il_diagnostics'])
        self.assertTrue(model.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        json.dumps(report,allow_nan=False)

    def test_cache_reuses_only_admitted_inputs_never_probabilities(self):
        probe=probe_fixture();model=ToyPolicy();update=dict(microbatch_size=3,max_candidate_cells=9)
        with patch.object(probes,'collate',wraps=probes.collate) as collate:
            first=probe.probabilities(model,update);calls=collate.call_count
            second=probe.probabilities(model,update)
            self.assertEqual(collate.call_count,calls);np.testing.assert_array_equal(first,second)
            with torch.no_grad():model.bias.add_(torch.tensor([1.,-1.,2.]))
            self.assertFalse(np.array_equal(first,probe.probabilities(model,update)))
            self.assertEqual(collate.call_count,calls)
        with patch.object(model,'forward',side_effect=AssertionError('duplicate inference')):
            probe.measure(model,update,probabilities=second)
        tiny=probe_fixture();tiny._input_limit=1;tiny.probabilities(model,update)
        self.assertEqual(tiny._input_bytes,0);self.assertFalse(tiny._inputs)
        for bad in (second*2,np.full_like(second,np.nan)):
            with self.assertRaises(ValueError):probe.measure(model,update,probabilities=bad)


class AuxiliarySchedule(unittest.TestCase):
    def test_committed_cadence_restart_skip_and_release(self):
        from brass_ai.recovery.league import finish_training_diagnostics
        for generation in (1341,1342,1344,1345,1350):
            companion=Mock(active=True,generation=generation-1)
            companion.report.return_value={};companion.diagnostics.return_value=dict(rows=12,groups={},seconds=.1)
            phases=Mock();report=finish_training_diagnostics(None,phases,companion)['heldout']
            self.assertEqual(report['measured'],generation%5==0)
            self.assertEqual(companion.diagnostics.call_count,int(generation%5==0))
            self.assertEqual(report['maximum_rows'],512)
            self.assertEqual(report['next_generation'],(generation//5+1)*5)
            phases.release.assert_called_once_with()
            if generation%5:self.assertNotIn('rows',report)

    def test_default_512_rows_reduced_metrics_eligibility_and_modes(self):
        from test_gae_value_phase import fixture
        from test_local_scoring import annotate
        from brass_ai.recovery import updates,local_scoring_auxiliary as aux
        torch.set_num_threads(2)
        model,_,_,(_,data)=fixture(games=48,decisions=3);annotate(data)
        dataset=updates.Dataset(data,'cpu');dataset.rows[0]['_branch']=True
        dataset.roles[1]='snapshot'
        companion=aux.Companion(model,copy.deepcopy(aux.DEFAULT));model.train();companion.heads.train()
        rng=torch.get_rng_state().clone()
        result=companion.diagnostics(model,dataset,range(len(data)))
        self.assertEqual(result['rows'],512)
        self.assertTrue(model.training);self.assertTrue(companion.heads.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for group in result['groups'].values():
            self.assertEqual(set(group),{'rows','entities','positive','none','positive_errors','ce_sum','vp_abs_sum','owner_ce','vp_mae'})
            self.assertEqual(group['positive']+group['none'],group['entities'])
        small=companion.diagnostics(model,dataset,[0,1,2])
        self.assertEqual(small['rows'],1)
        with patch.object(model,'encode_state',side_effect=RuntimeError('fixture')):
            with self.assertRaises(RuntimeError):companion.diagnostics(model,dataset,[2])
        self.assertTrue(model.training);self.assertTrue(companion.heads.training)


def retention_fixture(root,n=130):
    run=root/'artifacts/run';run.mkdir(parents=True)
    keep=run/'keep.pt';keep.write_bytes(b'live')
    write_json(run/'state.json',dict(best='artifacts/run/keep.pt',last='artifacts/run/keep.pt',epoch=10,iteration=10))
    for i in range(n):(run/f'epoch-{i:04}.pt').write_bytes((str(i)*4096).encode())
    return run


class CleanupOptimization(unittest.TestCase):
    def fixture(self,n=130):
        t=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts');self.addCleanup(t.cleanup)
        root=Path(t.name);return root,retention_fixture(root,n),dict(RETENTION,quarantine_hours=0)

    def test_batched_journal_preserves_pins_and_bounds_writes(self):
        root,run,cfg=self.fixture()
        with patch.object(retention,'write_json',wraps=write_json) as writes:
            result=retention.prune(root,run,cfg,apply=True)
        self.assertEqual(result['permanently_deleted_files'],130)
        self.assertLess(writes.call_count,25)
        self.assertEqual((run/'keep.pt').read_bytes(),b'live')
        self.assertTrue(retention.settled(read(Path(result['manifest']))))
        self.assertEqual(retention.purge(root,run,cfg)['permanently_deleted_files'],0)

    def test_interrupted_move_reconciles_mixed_batch_and_rescues_new_pin(self):
        root,run,cfg=self.fixture(8);replace=Path.replace;count=0
        def fail(source,target):
            nonlocal count
            if source.parent==run and source.name.startswith('epoch-'):
                count+=1
                if count==3:raise OSError('move fault')
            return replace(source,target)
        with patch.object(Path,'replace',fail),self.assertRaisesRegex(OSError,'move fault'):
            retention.prune(root,run,cfg,apply=True)
        write_json(root/'artifacts/consumer/pins.json',dict(paths=['artifacts/run/epoch-0000.pt']))
        result=retention.purge(root,run,cfg)
        self.assertEqual(result['rescued_dependencies'],1)
        self.assertEqual(result['permanently_deleted_files'],1)
        self.assertTrue((run/'epoch-0000.pt').exists())
        self.assertTrue((run/'epoch-0002.pt').exists())
        self.assertTrue(all(retention.settled(read(p)) for p in (run/'.retention').glob('cleanup-*.json')))
        self.assertEqual(retention.prune(root,run,cfg,apply=True)['permanently_deleted_files'],6)

    def test_interrupted_delete_retries_and_rejects_tampered_pending_file(self):
        root,run,cfg=self.fixture(8);unlink=Path.unlink;count=0
        def fail(path,*args,**kwargs):
            nonlocal count
            if '.retention-trash' in path.parts and path.name.startswith('epoch-'):
                count+=1
                if count==3:raise OSError('delete fault')
            return unlink(path,*args,**kwargs)
        with patch.object(Path,'unlink',fail),self.assertRaisesRegex(OSError,'delete fault'):
            retention.prune(root,run,cfg,apply=True)
        pending=next((run/'.retention-trash').rglob('epoch-0002.pt'));original=pending.read_bytes()
        pending.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'Quarantine changed'):retention.purge(root,run,cfg)
        pending.write_bytes(original)
        self.assertEqual(retention.purge(root,run,cfg)['permanently_deleted_files'],6)
        self.assertTrue((run/'keep.pt').exists())

    def test_five_committed_iteration_cadence_survives_restart_and_failed_pass(self):
        root,run,cfg=self.fixture(1);state=read(run/'state.json')
        for generation in (1341,1344):
            write_json(run/'state.json',dict(state,iteration=generation))
            with patch.object(retention,'prune',side_effect=AssertionError('not due')):
                r=retention.scheduled_prune(root,run,cfg)
            self.assertFalse(r['performed']);self.assertEqual(r['next_cleanup_generation'],1345)
        write_json(run/'state.json',dict(state,iteration=1345))
        with patch.object(retention,'prune',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):retention.scheduled_prune(root,run,cfg)
        self.assertFalse((run/'.retention/automatic.json').exists())
        r=retention.scheduled_prune(root,run,cfg);self.assertTrue(r['performed'])
        self.assertFalse(retention.scheduled_prune(root,run,cfg)['performed'])
        write_json(run/'state.json',dict(state,iteration=1349))
        self.assertEqual(retention.scheduled_prune(root,run,cfg)['next_cleanup_generation'],1350)
        write_json(run/'state.json',dict(state,iteration=1350))
        self.assertTrue(retention.scheduled_prune(root,run,cfg)['performed'])
        self.assertFalse(retention.scheduled_prune(root,run,dict(cfg,enabled=False))['performed'])
        self.assertEqual(bounded_fields(dict(cfg,every_iterations=1),RETENTION,'retention')['every_iterations'],1)
        for value in (0,True,21,1.5):
            with self.assertRaises(ValueError):bounded_fields(dict(cfg,every_iterations=value),RETENTION,'retention')


class CurrentContinuation(unittest.TestCase):
    def test_actual_g1340_export_accepts_same_helper_without_rewriting_run(self):
        from brass_ai.learning.contracts import digest
        from brass_ai.recovery import resume
        from brass_ai.step5.league import implementation_id
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1340-20261031T210153391863Z/state.json')
        spec=importlib.util.spec_from_file_location('g1340_helper',SOURCE_ROOT/'scripts/configure-training-g1340.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as tmp:
            root=Path(tmp);run=root/'artifacts/step5/demo';run.mkdir(parents=True)
            checkpoint=run/'generation-1340.pt';checkpoint.write_bytes(b'isolated helper fixture')
            state['history'][-1].update(path=checkpoint.relative_to(root).as_posix(),sha256=sha(checkpoint))
            write_json(run/'state.json',state);before=sha(run/'state.json');checkpoint_before=sha(checkpoint)
            output='configs/step5/g1340.json'
            report=helper.generate(root,run,output,1340)
            self.assertEqual(report['generation'],1340);self.assertEqual(report['next_context_generation'],1390)
            self.assertEqual(before,sha(run/'state.json'));self.assertEqual(checkpoint_before,sha(checkpoint))
            config=read(root/output)
            self.assertEqual(config['exploration'],.05);self.assertEqual(config['insolvency'],dict(event_cost=.01,max_cost=.075))
            current=dict(state['signature'],config=config,implementation=implementation_id())
            self.assertTrue(resume.check_signature(state,current,reviewed=True))
            self.assertFalse(resume.evaluation_receipt_preserved(state,current,dict(training={'signature':digest(state['signature'])}),{}))
            with self.assertRaises(ValueError):resume.check_signature(state,current,reviewed=False)
            bad=copy.deepcopy(current);bad['config']['learning_rate']*=2
            with self.assertRaises(ValueError):resume.check_signature(state,bad,reviewed=True)

    def test_direct_performance_only_review_preserves_calibration_and_exact_config(self):
        from brass_ai.learning.contracts import digest
        from brass_ai.recovery import resume
        from brass_ai.step5.league import implementation_id
        state=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1340-20261031T210153391863Z/state.json')
        state['signature']['implementation']='b514fd9ee9944d93bfd596968cba77c93a4c742118119c74610c5541dce6c29f'
        current=dict(state['signature'],implementation=implementation_id())
        self.assertTrue(resume.check_signature(state,current,reviewed=True))
        self.assertTrue(resume.evaluation_receipt_preserved(state,current,dict(training={'signature':digest(state['signature'])}),{}))
        bad=copy.deepcopy(current);bad['config']['exploration']=.021
        with self.assertRaises(ValueError):resume.check_signature(state,bad,reviewed=True)


if __name__=='__main__':unittest.main()
