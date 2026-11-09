"""Tiny native-run archives and one-time empty-buffer adoption, no game runs."""
import copy
from pathlib import Path
import subprocess
import sys
import unittest
import zipfile
from unittest.mock import patch

import torch

import test_step5_transfer as fixtures
from brass_ai.learning.source_identity import implementation_sources
from brass_ai.recovery import transfer_resume as R
from brass_ai.recovery.retention import _same_payload
from brass_ai.learning.paths import make_relocation, resolve_project_path

transfer=fixtures.transfer
ROOT=fixtures.ROOT


class NativeTransfer(unittest.TestCase):
    put=fixtures.TransferTests.put
    entry=fixtures.TransferTests.entry
    archive_bytes=fixtures.TransferTests.archive_bytes
    move_source=fixtures.TransferTests.move_source

    def pin(self,path):
        return dict(path=str(path),sha256=transfer.sha(path),bytes=path.stat().st_size)

    def setUp(self):
        fixtures.TransferTests.setUp(self)
        # Exercise the real shared file-inventory recipe on a tiny authored
        # checkout; contents are inert fixtures, never executable runtimes.
        while True:
            try:
                transfer.source_identity(self.root,recipe=4)
                break
            except FileNotFoundError as error:
                name=Path(error.filename).relative_to(self.root).as_posix()
                self.put(name,b'# source fixture\n')
        self.state['signature']['config']['rehearsal_weight']=0.
        p=self.put('artifacts/step5/selfplay/il-kl/probe.npz',b'fixed-array-fixture')
        m=self.put('artifacts/step5/selfplay/il-kl/manifest.json',dict(probe_hash='a'*64,
            rows=[{} for _ in range(2048)],cache=self.pin(p)))
        self.state['il_kl_limit']=dict(probe=dict(self.pin(m),probe_hash='a'*64))
        self.state['il_kl_limit']['probe']['path']=m.relative_to(self.root).as_posix()
        binary=self.put('artifacts/native/collector',b'collector binary')
        tests=self.put('artifacts/native/tests.log',b'passed')
        b01=self.put('artifacts/native/original/qualification.json',dict(passed=True))
        self.put('artifacts/native/original/inputs/fixtures.json',dict(roots=[]))
        build=self.put('artifacts/native/build.json',dict(collector=self.pin(binary),tests=self.pin(tests),
            bootstrap=dict(binary=self.pin(self.root/'training/runtime/throughput/ordinary/BrassHost.exe'),dependencies=[]),
            b01=self.pin(b01)))
        qualification=self.put('artifacts/native/qualification.json',dict(passed=True,teacher_build=self.pin(build)))
        rollback=self.put('artifacts/native/diagnostic-fork.json',dict(payload='not-exported'))
        self.state['native_backend']=dict(contract='brass-ordinary-backend/1.0.0',mode='rust',build=self.pin(build),
            qualification=self.pin(qualification),rollback=self.pin(rollback))
        self.state['signature']['implementation']=transfer.source_identity(self.root,recipe=4)
        self.put('artifacts/step5/selfplay/state.json',self.state)
        self.omitted=[]
        for name in ('.ppg-eight-cache/generation-000002.critic.npy','teacher-labels/rows.jsonl.gz',
                     'search-values/000001-train.json.gz','il-kl/.prepared/large.npy'):
            self.omitted.append(self.put('artifacts/step5/selfplay/'+name,b'not transferred'))
        self.omitted.append(self.put('artifacts/step5/il/.prepared/full-il.npy',b'not transferred'))

    def export(self):
        return transfer.export(self.root,self.run,self.bundle,archive_format='zip',dependencies='bundled',part_mb=0)

    def test_recipe_matches_trainer_and_clean_interpreter_needs_no_torch_or_numpy(self):
        from brass_ai.step5.league import implementation_id
        self.assertEqual(transfer.source_identity(ROOT,recipe=4),implementation_id())
        # -S excludes site-packages; source/transfer inspection must stay stdlib.
        code="import sys;sys.path.insert(0,'src');from brass_ai.learning.source_identity import implementation_id;print(implementation_id('.'));assert 'torch' not in sys.modules and 'numpy' not in sys.modules"
        result=subprocess.run([sys.executable,'-S','-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertIn(implementation_id(),result.stdout)

    def test_archive_excludes_every_replay_and_original_il_cache_and_retains_fixed_probe_and_runtime(self):
        self.export();m=transfer.inspect_bundle(self.bundle);names={f['path'] for f in m['files']}
        self.assertEqual(m['source_recipe'],4);self.assertEqual(m['buffer_rebuild']['auxiliary_buffer_iterations'],8)
        for p in self.omitted:self.assertNotIn(p.relative_to(self.root).as_posix(),names)
        self.assertNotIn('artifacts/demos/train.jsonl.gz',names)
        self.assertIn('artifacts/step5/selfplay/il-kl/probe.npz',names)
        self.assertIn('artifacts/step5/selfplay/il-kl/manifest.json',names)
        self.assertIn('artifacts/native/collector',names)
        self.assertIn('artifacts/native/original/inputs/fixtures.json',names)
        with self.assertRaisesRegex(ValueError,'compact mode'):
            transfer.export(self.root,self.run,self.area/'full.zip',mode='full',archive_format='zip')

    def test_matching_qualified_source_accepted_but_unknown_source_or_rehearsal_rejected(self):
        actual=self.state['signature']['implementation'];qualified='f'*64
        self.put('configs/learning/b07-replay-location-review.json',dict(contract='brass-b07-replay-location-review/1.0.0',
            execution_implementation=actual,qualified_implementation=qualified))
        self.state['signature']['implementation']=qualified;self.put('artifacts/step5/selfplay/state.json',self.state)
        self.export()
        self.state['signature']['implementation']='e'*64;self.put('artifacts/step5/selfplay/state.json',self.state)
        with self.assertRaisesRegex(ValueError,'Source differs'):transfer.inventory(self.root,self.run)
        self.state['signature']['implementation']=qualified;self.state['signature']['config']['rehearsal_weight']=.1
        self.put('artifacts/step5/selfplay/state.json',self.state)
        with self.assertRaisesRegex(ValueError,'rehearsal to be disabled'):transfer.inventory(self.root,self.run)

    def test_available_startup_seed_json_travels_without_adding_unreferenced_seed_tensors(self):
        p=self.put('artifacts/step5/selfplay/seed/manifest.json',dict(seed_generation=1290,history_through=1330))
        self.put('artifacts/step5/selfplay/seed/unreferenced-old-weights.pt',b'not a live model dependency')
        self.state['signature']['config']['fresh_selfplay_seed']=dict(manifest=p.relative_to(self.root).as_posix(),sha256=transfer.sha(p))
        self.put('artifacts/step5/selfplay/state.json',self.state);self.export()
        names={r['path'] for r in transfer.inspect_bundle(self.bundle)['files']}
        self.assertIn(p.relative_to(self.root).as_posix(),names)
        self.assertNotIn('artifacts/step5/selfplay/seed/unreferenced-old-weights.pt',names)

    def test_startup_only_seed_json_remains_optional_on_an_already_compact_restored_run(self):
        self.state['signature']['config']['fresh_selfplay_seed']=dict(manifest='artifacts/step5/selfplay/seed/manifest.json',sha256='a'*64)
        self.put('artifacts/step5/selfplay/state.json',self.state);self.export()
        self.assertNotIn('artifacts/step5/selfplay/seed/manifest.json',
            {r['path'] for r in transfer.inspect_bundle(self.bundle)['files']})

    def test_restore_preserves_bytes_and_authorizes_reset_only_at_checkpoint_boundary(self):
        self.export();m=transfer.inspect_bundle(self.bundle);state_bytes=(self.run/'state.json').read_bytes()
        self.move_source();result=transfer.restore(self.root,self.bundle)
        self.assertTrue(result['verified']);self.assertEqual((self.run/'state.json').read_bytes(),state_bytes)
        self.assertIn('--expected-generation 2',result['resume']);self.assertIn('--resume-reviewed-source',result['resume'])
        original=dict(weights=torch.tensor([3.]),rng=torch.get_rng_state(),training=dict(actor_value_phases=dict(
            rolling_buffer=[dict(generation=2)],cache_origin=0,optimizer=dict(adam=torch.tensor([7.])),beta_clone=.25)))
        before=copy.deepcopy(original);state=copy.deepcopy(self.state)
        receipt,reset=R.prepare(self.root,self.run,state,original)
        self.assertTrue(reset);derived=R.ppg_payload(original,receipt,reset)
        self.assertEqual(derived['training']['actor_value_phases']['rolling_buffer'],[])
        self.assertEqual(derived['training']['actor_value_phases']['cache_origin'],2)
        self.assertIs(derived['weights'],original['weights'])
        self.assertIs(derived['training']['actor_value_phases']['optimizer'],original['training']['actor_value_phases']['optimizer'])
        self.assertTrue(_same_payload(original,before))
        # Retry before the first durable commit repeats the same reset safely.
        self.assertEqual(R.prepare(self.root,self.run,state,original),(receipt,True))
        self.put('artifacts/step5/selfplay/state.json',state)
        from brass_ai.recovery.retention import references
        pins=references(self.root,self.run)
        self.assertIn(self.run/R.MARKER,pins)
        self.assertIn(self.root/receipt['original_state']['path'],pins)
        state['iteration']=3;state['history'].append(dict(generation=3))
        committed=dict(original,training=dict(original['training'],machine_transfer=receipt))
        self.assertEqual(R.prepare(self.root,self.run,state,committed),(receipt,False))
        self.assertIs(R.ppg_payload(committed,receipt,False),committed)
        with self.assertRaisesRegex(ValueError,'receipt missing'):R.prepare(self.root,self.run,state,original)

    def test_relocation_keeps_native_pin_spelling_but_accesses_destination_bytes(self):
        self.export();destination=self.area/'destination';result=transfer.restore(destination,self.bundle,relocate=True)
        self.assertTrue(result['verified']);self.assertTrue(transfer.restore(destination,self.bundle,verify_only=True)['verified'])
        old=self.state['native_backend']['build'];mapped=resolve_project_path(destination,old['path'])
        from brass_ai.rust_backend import lab_paths,lab_backend
        from brass_ai.rust_backend.lab_contract import pin
        with patch.object(lab_paths,'SOURCE_ROOT',destination):
            self.assertEqual(pin(old['path']),old)
            self.assertEqual(lab_backend.checked_pin(old),transfer.read(mapped))
            mapped.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'evidence changed'):lab_backend.checked_pin(old)

    def test_source_in_place_restore_never_installs_a_rebuild_marker(self):
        self.export();transfer.restore(self.root,self.bundle)
        self.assertFalse((self.run/R.MARKER).exists())

    def rewrite(self,change,name):
        files=self.archive_bytes();m=transfer.json.loads(files['transfer.json']);change(m)
        m['id']=transfer.digest({k:v for k,v in m.items() if k!='id'});files['transfer.json']=transfer.encode(m)
        bundle=self.area/name
        with zipfile.ZipFile(bundle,'w') as out:
            for key,value in files.items():out.writestr(key,value)
        return bundle

    def test_executable_permission_is_restored_for_direct_native_workers(self):
        self.export()
        def change(m):
            for item in m['files']:
                if item['path']=='artifacts/native/collector':item['executable']=True
        bundle=self.rewrite(change,'executable.zip');self.move_source()
        chmod=Path.chmod;changed=[]
        def record(path,mode,*args,**kwargs):
            changed.append((path,mode));return chmod(path,mode,*args,**kwargs)
        with patch.object(Path,'chmod',record):self.assertTrue(transfer.restore(self.root,bundle)['verified'])
        self.assertTrue(any(p==self.root/'artifacts/native/collector' and mode & 0o100 for p,mode in changed))

    def test_invalid_rebuild_policy_rejected_before_destination_writes(self):
        self.export()
        bundle=self.rewrite(lambda m:m['buffer_rebuild'].update(auxiliary_buffer_iterations=7),'bad-policy.zip')
        destination=self.area/'bad-destination'
        with self.assertRaisesRegex(ValueError,'buffer-rebuild transfer policy'):transfer.restore(destination,bundle,relocate=True)
        self.assertFalse(destination.exists())


if __name__=='__main__':unittest.main()
