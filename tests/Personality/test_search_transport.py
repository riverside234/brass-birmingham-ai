"""Deployment fused-host selection, portable packaging and native parity."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT
from brass_ai.learning.features import Encoder
from brass_ai.learning.reward import specification
from brass_ai.personality.laptop.package import _copy_search_runtime
from brass_ai.personality.laptop.runtime import Runtime
from brass_ai.personality.search.contracts import search_config
from brass_ai.personality.search.engine import Worker
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.searchlab.spiel import BrassSpielGame, action_id
from fe_fixtures import benchmark_positions


class SearchTransportTests(unittest.TestCase):
    def test_worker_requests_fused_host_without_changing_search_config(self):
        config=search_config(specification(),workers=2)
        base=SimpleNamespace(evaluator=Mock());opponents=[Mock()];cancelled=Event()
        with patch('brass_ai.personality.search.engine.MaskedSearchPolicy.__init__',return_value=None) as init:
            worker=Worker(SOURCE_ROOT,config,base,opponents,None,cancelled,1)
        init.assert_called_once_with(None,SOURCE_ROOT,config=config,mono=None,
            evaluator=base.evaluator,opponents=opponents,fast_host=True)
        self.assertIs(worker.cancelled,cancelled);self.assertIs(worker.base,base)

    def test_laptop_plan_revalidation_requests_fused_host(self):
        with patch('brass_ai.personality.laptop.runtime.validate',return_value={'models':[]}), \
                patch('brass_ai.personality.laptop.runtime.PortableInference') as inference, \
                patch('brass_ai.personality.laptop.runtime.BrassSpielGame') as game:
            runtime=Runtime(SOURCE_ROOT,fixture=True)
            try:game.assert_called_once_with(root=SOURCE_ROOT/'runtime',mono=None,fast=True)
            finally:runtime.close()
            inference.return_value.__exit__.assert_called_once()

    def test_portable_bundle_uses_verified_fast_binary_without_training_files(self):
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as folder:
            out=Path(folder);_copy_search_runtime(SOURCE_ROOT,out)
            root=out/'runtime'
            card_path=Path('configs/contracts/v0.2.0/card-composition.json')
            (root/card_path).parent.mkdir(parents=True)
            shutil.copyfile(SOURCE_ROOT/card_path,root/card_path)
            self.assertFalse((root/'training').exists())
            metadata=json.loads((root/'artifacts/r4-host/runtime.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['kind'],'search')
            self.assertEqual(metadata['extension'],'brass-search-chance/1.0.0')
            game=BrassSpielGame(root=root,fast=True)
            try:
                self.assertTrue(game.host.fast)
                snapshot=game.host.call('search_snapshot')
                self.assertEqual(snapshot['choices'],game.host.choices())
                self.assertEqual(snapshot['views'][snapshot['context']['actor']],game.host.observe(snapshot['context']['actor']))
            finally:game.close()
            # Copied binary bytes remain checked; a tampered package cannot
            # obtain the fast path by merely retaining its manifest.
            exe=root/'artifacts/r4-host/BrassHost.exe'
            with exe.open('ab') as stream:stream.write(b'changed')
            from brass_ai.host import HostError
            with self.assertRaisesRegex(HostError,'fingerprint mismatch'):
                BrassSpielGame(root=root,fast=True)

    def test_missing_optimized_runtime_preserves_legacy_capability_fallback(self):
        with patch('brass_ai.recovery.host_runtime.training_host',return_value=SOURCE_ROOT/'artifacts/r4-host'):
            game=BrassSpielGame(root=SOURCE_ROOT,fast=True)
        try:
            self.assertTrue(game.host.fast_checked);self.assertFalse(game.host.fast)
            row=benchmark_positions()[0]
            state=game.from_observation(row['view'],192)
            self.assertEqual(state.legal,row['choices'])
            state.apply_action(action_id(state.legal[0]['id']))
            self.assertIsNotNone(state.checkpoint)
        finally:game.close()

    def test_native_fixture_reconstruction_advancement_and_branch_parity(self):
        old=BrassSpielGame(root=SOURCE_ROOT);fast=BrassSpielGame(root=SOURCE_ROOT,fast=True)
        oracle=Encoder();encoder=PreparedEncoder()
        def equal(a,b):
            self.assertEqual(a.checkpoint['hash'],b.checkpoint['hash'])
            # Each independent native host deliberately mints its own opaque
            # game ID for stale-request protection; public state must match.
            self.assertEqual({k:v for k,v in a.context.items() if k!='game_id'},
                {k:v for k,v in b.context.items() if k!='game_id'})
            self.assertNotEqual(a.context['game_id'],b.context['game_id'])
            for name in ('pending','views','legal','recall','actions','final_returns','final_result'):
                self.assertEqual(getattr(a,name),getattr(b,name),name)
        def next_action(state):
            if state.pending:return state.chance_outcomes()[0][0]
            choice=next((c for c in state.legal if c['kind']=='finish'),state.legal[0])
            return action_id(choice['id'])
        try:
            self.assertTrue(fast.host.fast)
            for i,row in enumerate(benchmark_positions()):
                with self.subTest(case=row['name']):
                    a=old.from_observation(row['view'],192+i)
                    b=fast.from_observation(row['view'],192+i);equal(a,b)
                    self.assertEqual(a.legal,row['choices'])
                    expected=oracle.encode(a.views[a.current_player()],a.legal)
                    actual=encoder.encode(b.views[b.current_player()],b.legal)
                    for field in ('state','actions','stable_to_relative'):
                        np.testing.assert_array_equal(getattr(expected,field),getattr(actual,field))
                    forks=a.clone(),b.clone();saved=copy.deepcopy(b.checkpoint)
                    for _ in range(4):
                        if a.is_terminal():break
                        action=next_action(a);a.apply_action(action);b.apply_action(action);equal(a,b)
                    # All clones must remain detached. Replay restoration is
                    # checked from the live initial game: arbitrary imported
                    # mid-game roots also fail hash verification in the legacy
                    # oracle (recorded separately by the benchmark report).
                    self.assertEqual(forks[1].checkpoint,saved)
                    if row['name']=='canal-opening':
                        for _ in range(16):
                            action=next_action(forks[0])
                            for state in forks:state.apply_action(action)
                            equal(*forks)
        finally:old.close();fast.close()


if __name__=='__main__':unittest.main()
