"""g5685: a resumed run reads nothing from the IL replay dataset (artifacts/il-replay, deleted on the lab): not its
IL lineage, not the dataset manifest behind its committed IL-KL probe, and not rehearsal rows (rehearsal is off)."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.recovery import il_kl_limit as limits, il_kl_probe as probes


def committed_probe(case):
    """A fixed probe built from a toy IL dataset (test_auto_il_kl's fixture), then the dataset deleted."""
    from test_auto_il_kl import ProbeTests, ToyPolicy
    t = tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts'); case.addCleanup(t.cleanup)
    root = Path(t.name); dataset = root / 'artifacts/il-replay/toy'; dataset.mkdir(parents=True)
    source = dataset / 'validation.jsonl.gz'; source.write_bytes(b'validation rows')
    manifest = dataset / 'manifest.json'
    manifest.write_text(json.dumps(dict(entries=[dict(split='validation', path='artifacts/il-replay/toy/validation.jsonl.gz',
                                                      sha256=sha(source))])))
    origin = dict(bc_run='artifacts/step5/toy-il', path='artifacts/step5/toy-il/candidate.pt', sha256='b' * 64,
                  model_id='a' * 64, manifest='artifacts/il-replay/toy/manifest.json', manifest_sha256=sha(manifest),
                  bc_signature='c' * 64)
    model, config = ToyPolicy(), copy.deepcopy(limits.DEFAULT); update = dict(microbatch_size=3, max_candidate_cells=100)
    run = root / 'artifacts/step5/run'
    with patch('brass_ai.recovery.demonstrations.load_dataset', return_value=(ProbeTests.data(None), {})):
        probe, receipt = probes.load(root, run, origin, model, config, update, 1)
    for p in (source, manifest): p.unlink()
    dataset.rmdir(); dataset.parent.rmdir()
    return root, run, origin, model, config, update, probe, receipt


class Probe(unittest.TestCase):
    def test_a_committed_probe_loads_without_the_il_dataset(self):
        root, run, origin, model, config, update, probe, receipt = committed_probe(self)
        self.assertFalse((root / 'artifacts/il-replay').exists())
        with patch('brass_ai.recovery.demonstrations.load_dataset', side_effect=AssertionError('IL dataset read')):
            loaded, again = probes.load(root, run, origin, model, config, update, 1, receipt)
        self.assertEqual(again, receipt); self.assertEqual(loaded.metadata, probe.metadata)
        self.assertEqual(loaded.measure(model, update)['kl_mean'], 0.)
        # The committed descriptor still binds the run's signed IL identity.
        for key, value in (('manifest_sha256', '0' * 64), ('model_id', 'd' * 64), ('sha256', 'e' * 64),
                           ('manifest', 'artifacts/il-replay/other/manifest.json')):
            with self.assertRaisesRegex(ValueError, 'source/config differs'):
                probes.load(root, run, dict(origin, **{key: value}), model, config, update, 1, receipt)
        # An uncommitted probe (a new run's restart before its first commit) still verifies its build inputs.
        with self.assertRaises(FileNotFoundError):
            probes.load(root, run, origin, model, config, update, 1)

    def test_the_diagnostic_fork_restores_the_probe_without_the_il_dataset(self):
        from brass_ai.rust_backend.lab_continuation import restore_probe
        root, run, origin, model, config, update, probe, receipt = committed_probe(self)
        fork = types.SimpleNamespace(root=root, resolve=lambda path: path,
                                     state=dict(signature=dict(imitation_source=origin)))
        restored = restore_probe(fork, dict(il_kl_limit=config), receipt)
        self.assertEqual(restored.metadata, probe.metadata)
        fork.state = dict(signature=dict(imitation_source=dict(origin, manifest_sha256='0' * 64)))
        with self.assertRaisesRegex(ValueError, 'source/config differs'): restore_probe(fork, dict(il_kl_limit=config), receipt)


class Handoff(unittest.TestCase):
    def test_a_resumed_run_binds_its_committed_il_origin_without_the_lineage(self):
        from brass_ai.recovery import demonstrations
        from brass_ai.recovery.league import imitation_handoff
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT / 'artifacts') as t:
            root = Path(t); run = root / 'artifacts/step5/run'; run.mkdir(parents=True)
            origin = dict(bc_run='artifacts/step5/toy-il', path='artifacts/step5/toy-il/candidate.pt', sha256='b' * 64,
                          model_id='a' * 64, manifest='artifacts/il-replay/toy/manifest.json', manifest_sha256='f' * 64,
                          bc_signature='c' * 64)
            fresh = (torch.nn.Linear(1, 1), dict(origin))
            with patch.object(demonstrations, 'completed_imitation', return_value=fresh) as walked:
                self.assertIs(imitation_handoff(root, run, 'artifacts/step5/toy-il'), fresh)  # a new run walks it
                self.assertEqual(walked.call_count, 1)
                (run / 'state.json').write_text(json.dumps(dict(signature=dict(imitation_source=origin))))
                self.assertEqual(imitation_handoff(root, run, 'artifacts/step5/toy-il'), (None, origin))
                self.assertEqual(walked.call_count, 1)  # a resumed run does not
                with self.assertRaisesRegex(ValueError, 'committed imitation source'):
                    imitation_handoff(root, run, 'artifacts/step5/other-il')
                (run / 'state.json').write_text(json.dumps(dict(signature={})))
                with self.assertRaisesRegex(ValueError, 'committed imitation source'):
                    imitation_handoff(root, run, 'artifacts/step5/toy-il')

    def test_rehearsal_stays_off_so_its_il_manifest_is_never_read(self):
        from brass_ai.recovery.rehearsal import schedule
        from test_auto_il_kl import DEFAULT_FLOOR
        config = dict(rehearsal_manifest='artifacts/il-replay/toy/manifest.json', rehearsal_weight=0., rehearsal_until=10000,
                      rehearsal_floor=dict(DEFAULT_FLOOR, initial=.03))
        for games, iteration in ((0, 0), (1_819_200, 5685), (10 ** 9, 10 ** 6)):
            self.assertEqual(schedule(config, games, iteration)['weight'], 0.)


if __name__ == '__main__':
    unittest.main()
