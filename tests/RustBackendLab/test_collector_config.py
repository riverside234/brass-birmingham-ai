"""Training-config options cross spawn without changing frozen learning jobs."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import SOURCE_ROOT as ROOT, digest
from brass_ai.recovery import collector_runtime as runtime, ingress, replay_binding
from brass_ai.recovery.config import validate_config
from brass_ai.rust_backend import lab_perf, lab_pool, lab_shadow_config
from brass_ai.step5.league import write_json


def worker_settings():
    return runtime.current(), lab_perf.mode(), replay_binding.enabled()


class CollectorConfig(unittest.TestCase):
    def setUp(self):
        self.base = read(ROOT/'configs/step5/joint7-v11-2-g1290-trained-restart-g1247.json')
        self.shadow = lab_shadow_config.configure(self.base)
        self.strict = dict(collector_validation='strict', replay_bindings=False)

    def test_explicit_settings_validate_and_leave_exact_recipe_unchanged(self):
        for mode in runtime.MODES:
            for enabled in (False, True):
                settings = dict(collector_validation=mode, replay_bindings=enabled)
                config = runtime.configured(self.shadow, settings)
                validate_config(config)
                self.assertTrue(lab_shadow_config.is_shadow(config))
                self.assertEqual(runtime.from_config(config), settings)
                self.assertEqual(runtime.recipe(config), self.shadow)
                self.assertEqual(digest(runtime.recipe(config)), digest(self.shadow))
        self.assertNotIn('collector_validation', self.shadow['execution'])

    def test_invalid_options_and_unrelated_recipe_edits_are_rejected(self):
        for key, bads in (('collector_validation', ('fast', 1, None, [])),
                          ('replay_bindings', (0, 1, '1', None))):
            for bad in bads:
                with self.subTest(key=key, bad=bad):
                    config = copy.deepcopy(self.shadow)
                    config['execution'][key] = bad
                    with self.assertRaises(ValueError): validate_config(config)
        config = runtime.configured(self.shadow)
        config['learning_rate'] *= 2
        self.assertFalse(lab_shadow_config.is_shadow(config))
        config = runtime.configured(self.shadow)
        config['execution']['unknown_runtime_option'] = True
        with self.assertRaises(ValueError): validate_config(config)
        self.assertFalse(lab_shadow_config.is_shadow(config))

    def test_config_beats_environment_and_restores_scope_after_error(self):
        with patch.dict(os.environ, dict(BRASS_COLLECTOR_VALIDATION='strict', BRASS_REPLAY_BINDINGS='0')):
            self.assertEqual(runtime.from_config(self.shadow), runtime.DEFAULTS)
            self.assertEqual(worker_settings(), (self.strict, 'strict', False))
            with self.assertRaisesRegex(RuntimeError, 'cancel'):
                with runtime.scope(runtime.from_config(self.shadow)):
                    self.assertEqual(worker_settings(), (runtime.DEFAULTS, 'production_fast', True))
                    with ThreadPoolExecutor(max_workers=1) as threads:
                        self.assertEqual(threads.submit(worker_settings).result(), worker_settings())
                    raise RuntimeError('cancel')
            self.assertEqual(worker_settings(), (self.strict, 'strict', False))
        settings = runtime.from_config(self.shadow)
        with runtime.scope(settings):
            settings['replay_bindings'] = False
            snapshot = runtime.current(); snapshot['replay_bindings'] = False
            self.assertTrue(replay_binding.enabled())

    def test_spawned_and_inline_readers_receive_active_config(self):
        previous_admission = ingress._admission
        with runtime.scope(self.strict):
            with ingress.reader_pool(2) as readers:
                results = [readers.submit(worker_settings) for _ in range(3)]
                self.assertTrue(all(f.result(timeout=30) == (self.strict, 'strict', False) for f in results))
            with self.assertRaisesRegex(RuntimeError, 'cancel'):
                with ingress.reader_pool(1, admission=object()) as readers:
                    self.assertIsNone(readers)
                    self.assertEqual(worker_settings(), (self.strict, 'strict', False))
                    raise RuntimeError('cancel')
        self.assertIs(ingress._admission, previous_admission)

    def test_native_worker_initializers_install_settings_before_work(self):
        config = runtime.configured(self.shadow, self.strict)
        connection = Mock()
        connection.send.side_effect = lambda _: self.assertEqual(worker_settings(), (self.strict, 'strict', False))
        stop = Mock(); stop.is_set.return_value = True
        with runtime.scope(runtime.DEFAULTS), patch('torch.set_num_threads'):
            lab_pool._writer(connection, 'unused', {}, config, 'ordinary', None, stop)
            connection.send.assert_called_once_with(dict(ready=True))
        class StopBeforeThreads(Exception): pass
        def stop_before_threads(**kwargs):
            self.assertEqual(worker_settings(), (self.strict, 'strict', False))
            raise StopBeforeThreads()
        with runtime.scope(runtime.DEFAULTS), patch('torch.set_num_threads'), \
                patch.object(lab_pool, 'ThreadPoolExecutor', side_effect=stop_before_threads):
            with self.assertRaises(StopBeforeThreads):
                lab_pool._actors([connection], ROOT, {}, config['execution'], 'mono', None, None, stop, 'unused')

    def test_public_training_installs_config_without_changing_pending_signature(self):
        from brass_ai.recovery import league
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            root = Path(temp); run = root/'artifacts/run'; run.mkdir(parents=True)
            pending = run/'pending'; pending.write_bytes(b'existing job')
            config = runtime.configured(self.shadow, self.strict)
            def run_stages(root, run_path, signed, **kwargs):
                self.assertEqual(signed, self.shadow)
                self.assertEqual(worker_settings(), (self.strict, 'strict', False))
                self.assertEqual(pending.read_bytes(), b'existing job')
                raise RuntimeError('cancel')
            before = runtime.current()
            with patch('brass_ai.recovery.automatic_training.run', side_effect=run_stages):
                with self.assertRaisesRegex(RuntimeError, 'cancel'):
                    league.train(root, 'artifacts/run', config)
            self.assertEqual(runtime.current(), before)

    def test_launcher_upgrades_legacy_config_and_preserves_user_settings_and_pending(self):
        spec = importlib.util.spec_from_file_location('collector_config_continue', ROOT/'scripts/continue-rust-shadow.py')
        launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as temp:
            root = Path(temp); run = root/'artifacts/run'; run.mkdir(parents=True)
            checkpoint = run/'generation-1257.pt'; checkpoint.write_bytes(b'checkpoint bytes')
            state = dict(iteration=1257, history=[dict(generation=1257,
                path='artifacts/run/generation-1257.pt', sha256=sha(checkpoint))],
                signature=dict(config=self.shadow), native_backend={})
            write_json(run/'state.json', state)
            pending = run/'iteration-1257/native-ordinary/part'; pending.parent.mkdir(parents=True)
            pending.write_bytes(b'completed pending family')
            output = root/'configs/step5/run-rust-shadow32-no-branch-share025.json'
            write_json(output, self.shadow)  # File made by the previous launcher.
            with patch('brass_ai.rust_backend.lab_backend.validate_state'), \
                    patch('brass_ai.rust_backend.lab_shadow.teacher_build'), \
                    patch('brass_ai.rust_backend.lab_shadow_config.archive_pending') as archive:
                _, path, left = launcher.prepare(root, 'artifacts/run', 1257, 2)
                self.assertEqual(left, 2)
                self.assertEqual(runtime.from_config(read(path)), runtime.DEFAULTS)
                write_json(path, runtime.configured(read(path), self.strict))
                launcher.prepare(root, 'artifacts/run', 1257, 2)
                self.assertEqual(runtime.from_config(read(path)), self.strict)
                invalid = read(path); invalid['execution']['replay_bindings'] = '1'
                write_json(path, invalid)
                with self.assertRaisesRegex(ValueError, 'replay_bindings'):
                    launcher.prepare(root, 'artifacts/run', 1257, 2)
                self.assertEqual(read(path), invalid)
                archive.assert_not_called()
            self.assertEqual(read(run/'state.json'), state)
            self.assertEqual(pending.read_bytes(), b'completed pending family')


if __name__ == '__main__': unittest.main()
