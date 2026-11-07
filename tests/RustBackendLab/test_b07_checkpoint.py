"""Focused publication/recovery debugging; no games or timing benchmarks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from brass_ai.learning.checkpoint import save_checkpoint, load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest
from brass_ai.recovery.admitted_inputs import _freeze
from brass_ai.recovery.search_value_iteration import SearchValueIteration, stamp
from brass_ai.recovery import search_value_phase as phase_module
from test_b025 import model_fixture
from test_b05 import recipe


class Publication(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.model=model_fixture();self.config=recipe('shadow')
        self.phase=SearchValueIteration(self.root,self.config,dict(iteration=10),self.model)
        self.phase.phase.label_ages=[dict(generation=10,rows=[_freeze(dict(
            value=[1,1.,True,None,-0.,'汉字'],nested={'history':[[2],[3]]}))])]

    def test_one_conversion_same_snapshot_save_verify_and_strict_restart(self):
        before=digest(self.phase.phase.label_ages)
        with patch.object(phase_module,'label_snapshot',wraps=phase_module.label_snapshot) as convert:
            payload=self.phase.payload(11)
            self.assertIs(payload,self.phase.payload(11))
            save_checkpoint(self.root/'next.pt',self.model,step=11,training={'search_value':payload})
            state=dict(iteration=11,search_value=stamp(payload))
            self.phase.committed(state)
            self.assertEqual(convert.call_count,1)
        self.assertIsNone(self.phase._publication)
        restored,raw=load_checkpoint(self.root/'next.pt')
        self.assertEqual(digest(raw['training']['search_value']['label_ages']),before)
        with self.assertRaisesRegex(ValueError,'checkpoint/state'):
            SearchValueIteration(self.root,self.config,dict(iteration=10),restored,raw)
        resumed=SearchValueIteration(self.root,self.config,state,restored,raw)
        self.assertIsNone(resumed._publication)
        with self.assertRaises(ValueError):self.phase.committed(state)
        resumed.phase.label_ages.append(dict(generation=11,rows=[]))
        with patch.object(phase_module,'label_snapshot',wraps=phase_module.label_snapshot) as convert:
            next_payload=resumed.payload(12)
            resumed.committed(dict(iteration=12,search_value=stamp(next_payload)))
            self.assertEqual(convert.call_count,1)
        self.assertEqual(len(payload['label_ages']),1)

    def test_failed_save_can_retry_exact_snapshot_without_conversion_or_partial_file(self):
        with patch.object(phase_module,'label_snapshot',wraps=phase_module.label_snapshot) as convert:
            payload=self.phase.payload(11)
            with patch('brass_ai.learning.checkpoint.torch.save',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    save_checkpoint(self.root/'next.pt',self.model,step=11,training={'search_value':payload})
            self.assertFalse((self.root/'next.pt').exists())
            self.assertFalse((self.root/'next.pt.tmp').exists())
            self.assertIs(self.phase.payload(11),payload)
            save_checkpoint(self.root/'next.pt',self.model,step=11,training={'search_value':payload})
            with self.assertRaises(ValueError):self.phase.committed(dict(iteration=11,search_value={}))
            self.phase.committed(dict(iteration=11,search_value=stamp(payload)))
            self.assertEqual(convert.call_count,1)

    def test_snapshot_isolation_and_mutation_detection_even_with_forged_state(self):
        payload=self.phase.payload(11);expected=copy.deepcopy(payload)
        # Mutations of plain live age metadata cannot change the detached snapshot.
        self.phase.phase.label_ages[0]['generation']=99
        self.assertEqual(payload,expected)
        for change in (lambda p:p['label_ages'][0].update(generation=99),
                       lambda p:p.update(reward_id='wrong'),
                       lambda p:p['buffer']['receipts'].append({'file':'foreign'}),
                       lambda p:p['stage'].update(w=.9)):
            change(payload)
            with self.assertRaisesRegex(ValueError,'snapshot changed'):self.phase.payload(11)
            with self.assertRaisesRegex(ValueError,'durable checkpoint/state'):
                self.phase.committed(dict(iteration=11,search_value=stamp(payload)))
            payload.clear();payload.update(copy.deepcopy(expected))
        for action in (lambda:self.phase.admit([],[]),lambda:self.phase.prepare_ppg(None,11),lambda:self.phase._finish()):
            with self.assertRaisesRegex(ValueError,'publication is pending'):action()
        with self.assertRaises(ValueError):self.phase.payload(12)
        self.phase.committed(dict(iteration=11,search_value=stamp(payload)))

    def test_nonfinite_rejection_leaves_no_stale_publication(self):
        for value in (float('nan'),float('inf'),float('-inf')):
            self.phase.phase.label_ages=[_freeze({'value':value})]
            with self.assertRaises(ValueError):self.phase.payload(11)
            self.assertIsNone(self.phase._publication)
        self.phase.phase.label_ages=[_freeze({'value':0.})]
        payload=self.phase.payload(11)
        self.assertEqual(payload['label_ages'],[{'value':0.}])
        payload['label_ages'][0]['value']=1.
        self.assertEqual(self.phase.phase.label_ages[0]['value'],0.)


if __name__=='__main__':unittest.main()
