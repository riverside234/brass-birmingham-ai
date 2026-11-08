"""PI04/PI05 CPU regressions: capabilities, exact data, failure boundaries."""
import copy
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from brass_ai.learning.contracts import SOURCE_ROOT,digest
from brass_ai.recovery import replay_binding as B
from brass_ai.recovery.replay import check_game
from brass_ai.rust_backend import lab_admission as A,lab_features as F
from test_b03 import family_fixture


class Bindings(unittest.TestCase):
    def game(self,**kwargs):
        family,admission,config=family_fixture(**kwargs)
        return A.convert(family,admission,allow_diagnostic=True),family,admission,config

    def test_detaches_mutable_aliases_and_reuses_full_history_and_labels(self):
        game,_,_,_=self.game();original=game['records'][0];before=digest(game)
        original_array=game.encoded(original,None);original_values=original_array.state.copy()
        B.seal_game(game)
        original['view']['history'].append(dict(unrelated=True))
        original_array.state.setflags(write=True);original_array.state[0]+=17
        self.assertEqual(digest(game),before)
        np.testing.assert_array_equal(game.encoded(game['records'][0],None).state,original_values)
        with patch('brass_ai.recovery.replay.digest',side_effect=AssertionError('Repeated replay hash')):
            for _ in range(3):
                game.assert_metadata()
                for row in game['records']:game.encoded(dict(row,_reward_id='unchanged-labels'),None)
        with self.assertRaises(TypeError):game['records'][0]['behavior']['probability']=.5

    def test_owned_reader_views_do_not_expose_writable_children_through_copy_or_iteration(self):
        from brass_ai.recovery.admitted_inputs import OwnedTree
        # OwnedTree is only for fresh private JSON with no borrowed aliases.
        value=OwnedTree().freeze(json.loads('{"rows":[{"items":[1,2]}]}'))
        arrays=[value['rows'],value.get('rows'),dict(value)['rows'],value.copy()['rows'],
                (value|{})['rows'],({}|value)['rows'],next(iter(value.values())),dict(value.items())['rows']]
        for rows in arrays:
            members=[rows[0],rows[:][0],rows.copy()[0],list(rows)[0],next(iter(reversed(rows))),
                     (rows+rows)[0],([]+rows)[0],(rows*2)[0]]
            for member in members:
                with self.assertRaises(TypeError):member['items'].append(3)
        self.assertEqual(json.dumps(value,sort_keys=True),'{"rows": [{"items": [1, 2]}]}')
        restored=pickle.loads(pickle.dumps(value));restored['rows'][0]['items'].append(3)
        self.assertEqual(value['rows'][0]['items'],[1,2])

    def test_metadata_labels_context_and_array_layout_changes_still_fail(self):
        game,_,_,_=self.game();B.seal_game(game);row=game['records'][0]
        for changed in (dict(row,chosen=(row['chosen']+1)%len(row['choices'])),
                        dict(row,behavior=dict(row['behavior'],probability=.9))):
            with self.assertRaisesRegex(ValueError,'record changed'):game.encoded(changed,None)
        with self.assertRaisesRegex(ValueError,'context changed'):game.encoded(dict(row,style=[1.]*4),None)
        game['game_id']='changed'
        with self.assertRaisesRegex(ValueError,'metadata changed'):game.assert_metadata()
        encoded=game.features[next(iter(game.features))]
        with self.assertRaises(TypeError):encoded.actions=np.zeros_like(encoded.actions)
        with self.assertRaises(ValueError):encoded.state.setflags(write=True)
        encoded.actions.shape=(1,encoded.actions.size)
        with self.assertRaisesRegex(ValueError,'layout changed'):B.checked_encoded(encoded)

    def test_mutable_legacy_game_retains_original_mutation_detection(self):
        game,_,_,_=self.game();row=game['records'][0]
        row['view']['history'].append(dict(changed=True))
        with self.assertRaisesRegex(ValueError,'record changed'):game.encoded(row,None)

    def test_context_derivation_preserves_base_and_exact_action_arrays(self):
        game,_,_,_=self.game(kind='historical');B.seal_game(game)
        row=next(r for r in game['records'] if any(r['intent']))
        from brass_ai.recovery.encoding import PreparedEncoder
        base=game.encoded(row,None);neutral=dict(row,style=[0.]*4,intent=[0.]*8)
        value=game.encoded(neutral,PreparedEncoder(),allow_context=True)
        self.assertIs(value.actions,base.actions);self.assertIs(game.encoded(neutral,None,allow_context=True),value)
        np.testing.assert_array_equal(value.state[:-12],base.state[:-12])
        np.testing.assert_array_equal(value.state[-12:],np.zeros(12))
        self.assertTrue(np.any(base.state[-8:]));self.assertTrue(B.checked_encoded(value))
        key=next(k for k,v in game.features.items() if v is value)
        game.features[key]=pickle.loads(pickle.dumps(value))
        with self.assertRaisesRegex(ValueError,'features changed'):game.encoded(neutral,None,allow_context=True)

    def test_pickle_strips_capability_and_receiving_process_readmits_once(self):
        game,_,_,_=self.game();B.seal_game(game)
        restored=pickle.loads(pickle.dumps(game))
        self.assertFalse(hasattr(restored,'_immutable_binding'))
        encoded=restored.encoded(restored['records'][0],None)
        self.assertFalse(B.checked_encoded(encoded))
        admitted=B.seal_encoded(encoded);self.assertTrue(B.checked_encoded(admitted))
        encoded.state=encoded.state.copy();encoded.state[0]+=1
        self.assertNotEqual(encoded.state[0],admitted.state[0])

    def test_known_arrays_skip_selector_walks_and_hash_once_per_dataset(self):
        game,_,_,_=self.game();B.seal_game(game);row=game['records'][0];e=game.encoded(row,None)
        from brass_ai.recovery.batch_cache import InputAudit
        audit=InputAudit();before=B.report()
        with patch('brass_ai.recovery.batch_cache.array_digest',side_effect=AssertionError('Repeated array hash')):
            first=audit.seal([e]);self.assertEqual(audit.seal([e]),first)
        counts=B.since(before);self.assertEqual(counts['array_hashes_executed'],3)
        self.assertEqual(counts['array_hashes_reused'],3)
        self.assertEqual(audit.checks,0)

    def test_pack_preparation_and_branch_expand_keep_exact_outputs_without_rehashing(self):
        from brass_ai.recovery.branching import expand
        from rust_lab_b06_ordinary import prepared_rows
        from rust_lab_b06_metrics import compare_rows
        game,family,admission,config=self.game(branch=True)
        policy=family['frozen_job']['provenance']['learner_behavior']['policy_id']
        expected=prepared_rows(expand([game]),config,policy)
        B.seal_game(game)
        with patch('brass_ai.recovery.branching.validate_child',side_effect=AssertionError('Repeated branch replay')):
            games=expand([game])
        actual=prepared_rows(games,config,policy);compare_rows(expected,actual)
        before=B.report();compare_rows(actual,prepared_rows(games,config,policy))
        counts=B.since(before)
        self.assertNotIn('input_hash_executed',counts)
        self.assertGreater(counts['reward_targets_reused'],0)
        with patch.object(F,'label_key',side_effect=AssertionError('Repeated label hash')):
            raw,meta=F.pack(games,'a'*64)
        self.assertTrue(raw);self.assertEqual(len(meta['rows']),sum(len(g['records']) for g in games))

    def test_cold_publication_read_and_fallback_preserve_arrays_without_python_encoding(self):
        from brass_ai.rust_backend import lab_pool as P
        for kind in ('selfplay','legacy'):
            family,admission,config=family_fixture(kind=kind)
            part=family['parent'];part['game']=check_game(part['game']);part['game'].actor_validated=True
            with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as directory:
                publisher=P.Publisher(directory,family['producer'],config,'ordinary')
                with patch('brass_ai.recovery.encoding.PreparedEncoder.encode',side_effect=AssertionError('Re-encoding')):
                    handle=P.part_write(Path(directory)/'parts',part)
                    record=publisher.publish(family['frozen_job'],handle,None)
                    game=A.read(record['path'],record['native_admission'],reuse_validation=False)
                self.assertIsInstance(game._immutable_binding,B.GameBinding)
                self.assertTrue(all(B.checked_encoded(game.encoded(r,None)) for r in game['records']))

    def test_inline_ingress_drops_game_owners_and_keeps_plain_shared_metadata(self):
        from brass_ai.recovery.ingress import load_collection
        from brass_ai.recovery.utility import reward_spec
        from brass_ai.recovery.admitted_inputs import OwnedDict,OwnedList
        family,a,config=family_fixture()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            record=A.publish(Path(temp)/'replay.jsonl.gz',family,a,validation_receipt=True)
            policy=a['job']['provenance']['learner_behavior']['policy_id'];group=family['parent']['game']['setup_group']
            headers,data,_=load_collection([record],policy,{group:'selfplay'},workers=1,
                insolvency=config.get('insolvency'),reward=reward_spec(config))
            def check(value):
                self.assertNotIsInstance(value,(OwnedDict,OwnedList))
                if isinstance(value,dict):
                    for child in value.values():check(child)
                elif isinstance(value,(tuple,list)):
                    for child in value:check(child)
            check(headers);check([(r,t,identity) for r,t,identity,_ in data])
            self.assertIs(data[0][0]['_local_scoring']['envelope'],headers[0]['local_scoring_targets'])
            self.assertTrue(all(B.checked_encoded(e) for *_,e in data))

    def test_failed_publication_is_retained_and_recollected_without_claiming_old_bytes(self):
        from brass_ai.rust_backend import lab_pool as P
        family,a,config=family_fixture();job=a['job']
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            publisher=P.Publisher(temp,family['producer'],config,'ordinary')
            handle=P.part_write(Path(temp)/'parts',family['parent'])
            with patch.object(F,'write',side_effect=RuntimeError('Interrupted feature publication')):
                with self.assertRaisesRegex(RuntimeError,'Interrupted'):publisher.publish(job,handle,None)
            old=publisher.path(job).read_bytes();self.assertIsNone(publisher.load(job))
            family['parent']['game']['records'][0]['seconds']=.125
            result=publisher.publish(job,P.part_write(Path(temp)/'parts',family['parent']),None)
            self.assertEqual(A.read(result['path'],result['native_admission'])['records'][0]['seconds'],.125)
            retained=list((Path(temp)/'unclaimed').glob('*/replay.jsonl.gz'))
            self.assertEqual(len(retained),1);self.assertEqual(retained[0].read_bytes(),old)
            again=publisher.load(job);self.assertEqual(again['sha256'],result['sha256'])

    def test_owned_publication_reuses_canonical_encoding_and_original_family_digest(self):
        from brass_ai.rust_backend import lab_pool as P
        import gzip
        family,a,config=family_fixture()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            publisher=P.Publisher(temp,family['producer'],config,'ordinary')
            result=publisher.publish(a['job'],P.part_write(Path(temp)/'parts',family['parent']),None)
            _,stored=map(json.loads,gzip.decompress(Path(result['path']).read_bytes()).splitlines())
            self.assertEqual(stored['sha256'],digest({k:v for k,v in stored.items() if k!='sha256'}))
            self.assertEqual(stored['sha256'],result['native_admission']['family_sha256'])
            self.assertEqual(result['binding_checks']['family_json_encoding_executed'],1)
            self.assertEqual(result['binding_checks']['replay_family_encoding_reused'],1)

    def test_foreign_arrays_and_nonfinite_values_do_not_acquire_capability(self):
        game,_,_,_=self.game();encoded=game.encoded(game['records'][0],None)
        bad=copy.deepcopy(encoded);bad.state[0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Non-finite'):B.seal_encoded(bad)
        self.assertFalse(B.checked_encoded(encoded))
        from brass_ai.recovery.batch_cache import InputAudit
        audit=InputAudit();audit.check(encoded);self.assertEqual(audit.checks,1)

    def test_reviewed_old_feature_archive_is_read_once_without_reencoding(self):
        from brass_ai.recovery import feature_receipts
        game,_,_,_=self.game()
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'replay';receipt=F.write(path,[game],'a'*64)
            _,manifest=F._cache_paths(path);meta=json.loads(manifest.read_bytes());meta['encoder']='e'*64
            raw=json.dumps(meta,separators=(',',':')).encode();manifest.write_bytes(raw)
            receipt.update(validator='e'*64,manifest_sha256=hashlib.sha256(raw).hexdigest())
            with patch.object(feature_receipts,'review',return_value=dict(previous_encoders=['e'*64])),\
                    patch('brass_ai.recovery.encoding.PreparedEncoder.encode',side_effect=AssertionError('Re-encoding')):
                arrays=F.read(path,[game],receipt,'a'*64)
                self.assertEqual(len(arrays),len(game['records']))
                for e,row in zip(arrays,game['records'],strict=True):np.testing.assert_array_equal(e.state,game.encoded(row,None).state)
                self.assertIsNone(F.read(path,[game],dict(receipt,validator='f'*64),'a'*64))

    def test_known_old_cold_coverage_is_preserved_but_unknown_or_strict_replays_semantics(self):
        from brass_ai.rust_backend import lab_perf as V,lab_replay_compat as R,lab_families
        family,a,_=family_fixture();coverage=V.coverage(V.selection(a['job'],ordinal=1,selected_mode='production_fast'))
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            with patch.object(A,'_validation_version',return_value='e'*64):
                record=A.publish(Path(temp)/'replay.jsonl.gz',family,a,validation_receipt=True,coverage=coverage)
            with patch.object(R,'reviewed_validator',return_value=True),\
                    patch.object(lab_families,'check_game',side_effect=AssertionError('Full semantic recheck')):
                A.read(record['path'],record['native_admission'],reuse_validation=False,fast_recovery=True)
                with patch.dict(os.environ,BRASS_COLLECTOR_VALIDATION='strict'),self.assertRaisesRegex(AssertionError,'semantic recheck'):
                    A.read(record['path'],record['native_admission'],reuse_validation=False,fast_recovery=True)
            with patch.object(R,'reviewed_validator',return_value=False),\
                    patch.object(lab_families,'check_game',side_effect=AssertionError('Unknown source strict')):
                with self.assertRaisesRegex(AssertionError,'Unknown source strict'):
                    A.read(record['path'],record['native_admission'],reuse_validation=False,fast_recovery=True)

    def test_previous_adapter_identity_needs_reviewed_source_exact_parent_and_evidence(self):
        from brass_ai.rust_backend import lab_perf_runtime as R,lab_backend
        from brass_ai.rust_backend.lab_contract import pin
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            path=Path(temp)/'evidence.log';path.write_bytes(b'passed')
            original=dict(_receipt_sha256='a'*64,bootstrap={'exact':'original'},b01={'sha256':'b'*64})
            previous=dict(contract=R.CONTRACT,ordinary_source='c'*64,parent_build_sha256='a'*64,
                bootstrap=original['bootstrap'],b01=original['b01'],wire_regressions_passed=True,
                _receipt_sha256='d'*64,**{k:pin(path) for k in ('tests','build_log','performance_tests')})
            with patch('brass_ai.bank.read',return_value=dict(ordinary_predecessors=['c'*64])),\
                    patch.object(lab_backend,'check_build',return_value=previous):
                self.assertEqual(R.predecessor({},original)['_receipt_sha256'],'d'*64)
                previous['ordinary_source']='f'*64
                with self.assertRaisesRegex(ValueError,'Unknown previous'):R.predecessor({},original)
                previous['ordinary_source']='c'*64;path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'evidence changed'):R.predecessor({},original)

    def test_direct_pipe_read_fills_one_buffer_and_rejects_truncation(self):
        from types import SimpleNamespace
        from brass_ai.rust_backend.lab_transport import Pipe
        class Chunks(io.BytesIO):
            def readinto(self,target):return super().readinto(target[:3])
            def read(self,*args):raise AssertionError('Intermediate read allocation')
        pipe=object.__new__(Pipe);pipe.metrics=Counter();pipe.process=SimpleNamespace(stdout=Chunks(b'abcdefgh'))
        self.assertEqual(pipe._read(8),b'abcdefgh');self.assertEqual(pipe.metrics['receive_buffer_bytes'],8)
        pipe.process.stdout=Chunks(b'ab')
        with self.assertRaisesRegex(ValueError,'closed output'):pipe._read(8)

    def test_only_combined_export_can_receive_two_bounded_array_pages(self):
        from types import SimpleNamespace
        import struct
        from brass_ai.rust_backend.lab_transport import Pipe,CONTRACT,MAX_ARRAY_FRAME
        payload=b'x'*(MAX_ARRAY_FRAME+1)
        header=json.dumps(dict(protocol=CONTRACT,id=1,ok=True,data={'bytes':len(payload)})).encode()
        raw=struct.pack('<I',(len(payload)+len(header)+4)|(1<<31))+struct.pack('<I',len(header))+header+payload
        for op in ('export_page','training_arrays'):
            process=SimpleNamespace(stdin=io.BytesIO(),stdout=io.BytesIO(raw),poll=lambda:0,wait=lambda **kwargs:0)
            with patch('brass_ai.rust_backend.lab_transport.subprocess.Popen',return_value=process):
                with Pipe(['fixture']) as pipe:
                    if op=='export_page':self.assertEqual(len(pipe.call(op)['_arrays']),len(payload))
                    else:
                        with self.assertRaisesRegex(ValueError,'frame limit'):pipe.call(op)

    def test_supplied_features_do_not_skip_unknown_observation_schema(self):
        from jsonschema import ValidationError
        family,a,_=family_fixture();original=check_game(family['parent']['game'])
        arrays=[original.encoded(row,None) for row in original['records']]
        family['parent']['game']['records'][0]['view']['foreign_private_field']=1
        family['sha256']=digest({k:v for k,v in family.items() if k!='sha256'})
        with self.assertRaises(ValidationError):A.convert(family,a,arrays=arrays)

    def test_component_timings_cannot_pass_if_sources_change_mid_probe(self):
        import importlib.util
        from types import SimpleNamespace
        from brass_ai.step5 import league
        path=SOURCE_ROOT/'tests/RustBackendLab/profile_collector_performance.py'
        spec=importlib.util.spec_from_file_location('pi_profile_guard',path);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as temp:
            args=SimpleNamespace(families=2,decisions=8,workers=1,reverse=False,bindings_compare=True)
            def arm(*args,**kwargs):return dict(semantic_outputs=[],prepared_rows=0,_prepared=[[],[]])
            with patch.object(probe,'arm',side_effect=arm),patch.object(league,'implementation_id',side_effect=['a'*64,'b'*64]):
                with self.assertRaisesRegex(ValueError,'Source changed during'):probe.run(args,Path(temp))
            self.assertEqual(len(list(Path(temp).glob('*/report.json'))),2)

    def test_canonical_request_reuse_keeps_original_digest_with_escaped_text_and_budget(self):
        from brass_ai.learning.batching import Request
        from brass_ai.learning.reward import MAX_INSOLVENCY_COST
        from brass_ai.recovery.admitted_inputs import request
        from brass_ai.rust_backend.lab_perf import admit
        game,_,_,_=self.game();row=game['records'][0]
        view=copy.deepcopy(row['view']);view['history'].append(dict(text='café " \\ city'))
        frozen,choices=admit(view,row['choices'])
        for budget in (None,(0.,MAX_INSOLVENCY_COST/4,MAX_INSOLVENCY_COST/2,MAX_INSOLVENCY_COST)):
            args=('test','model',frozen,choices,1e20)
            expected=Request(*args,remaining=budget)
            with patch('brass_ai.learning.batching.Request.digest',side_effect=AssertionError('Repeated request JSON')):
                actual=request(*args,remaining=budget)
                self.assertEqual(actual.input_digest,expected.input_digest)
                self.assertEqual(actual.digest(),expected.input_digest)

    def test_unselected_branch_cannot_skip_necessary_origin_binding(self):
        from brass_ai.rust_backend import lab_perf as V
        f,a,_=family_fixture(branch=True);job=f['frozen_job']
        choice=next(V.selection(job,ordinal=i,selected_mode='production_fast') for i in range(1,64)
                    if not V.selection(job,ordinal=i,selected_mode='production_fast')['branch'])
        # Supply exact arrays only to isolate structural/branch admission.
        from brass_ai.recovery.replay import CheckedGame
        for part in (f['parent'],f['child']):
            original=copy.deepcopy(part['game']);checked=check_game(original)
            arrays=[checked.encoded(r,None) for r in original['records']]
            if part is f['child']:part['game']['records'][0]['view']['players'][0]['cash']+=1
            part['game']=CheckedGame(part['game']).bind(arrays)
        f['sha256']=digest({k:v for k,v in f.items() if k!='sha256'})
        with self.assertRaises(ValueError):
            A.convert(f,a,allow_diagnostic=True,_coverage=V.coverage(choice,child=True))

    def test_long_windows_result_export_keeps_portable_names_and_bytes(self):
        import rust_lab_results as results
        import shutil
        root=Path(tempfile.mkdtemp(dir=SOURCE_ROOT/'artifacts')).resolve()
        def cleanup():
            self.assertTrue(root.is_relative_to((SOURCE_ROOT/'artifacts').resolve()))
            shutil.rmtree(results.io_path(root))
        self.addCleanup(cleanup)
        evidence=root/'artifacts/rust-backend'/('pi04-export-'+48*'x')
        evidence.mkdir(parents=True);name='a'*96+'.json';results.io_path(evidence/name).write_bytes(b'{"passed":true}\n')
        out=results.export(evidence,root=root);manifest=json.loads((out/'export-manifest.json').read_text())
        report=manifest['files'][0]
        self.assertEqual(results.io_path(out/report['path']).read_bytes(),b'{"passed":true}\n')
        self.assertNotIn('?',report['path']);self.assertEqual(report['sha256'],report['source_sha256'])


@unittest.skipUnless(os.environ.get('BRASS_B07_COLLECTOR'),'Set native short-test binary')
class NativeTail(unittest.TestCase):
    def setUp(self):
        self.binary=os.environ['BRASS_B07_COLLECTOR']
        roots=json.loads(Path(os.environ['BRASS_B07_FIXTURES']).read_text())['roots']
        self.origin=dict(state=roots[-1]['expected']['state'],rail_keys=list(range(64)))

    def test_combined_pages_keep_actions_rng_features_seals_and_reduce_calls(self):
        from rust_lab_b07_transport import walk
        before=walk(self.binary,self.origin,raw_export=True,combined_export=False)
        after=walk(self.binary,self.origin,raw_export=True,combined_export=True)
        for k in ('trace','features','branches','rng','complete','forced','training_features'):self.assertEqual(before[k],after[k],k)
        self.assertIsNotNone(after['complete']);self.assertGreater(after['metrics']['ipc_export_page_calls'],0)
        self.assertLess(after['metrics']['ipc_calls'],before['metrics']['ipc_calls'])

    def test_export_rejects_stale_token_cursor_and_retained_feature_corruption(self):
        import rust_lab_b07_transport as wire
        from brass_ai.rust_backend.lab_transport import Collector
        for fault in ('token','cursor','features'):
            class Fault(Collector):
                def completed_export(inner):
                    if fault=='features':
                        key=next(iter(inner.retained_features));raw,_=inner.retained_features[key]
                        inner.retained_features[key]=(raw,'0'*64)
                        return super().completed_export()
                    return inner.call('export_page',token='stale' if fault=='token' else inner.token,
                        record_start=1 if fault=='cursor' else 0,feature_start=0,reuse=[None]*64)
            with patch.object(wire,'Collector',Fault),self.assertRaisesRegex(ValueError,'foreign export token|stale export cursors|retained feature differs'):
                wire.walk(self.binary,self.origin,raw_export=True)

    def test_cancelled_export_retires_process_without_publishing_or_hanging(self):
        import threading
        import rust_lab_b07_transport as wire
        from brass_ai.rust_backend.lab_transport import Collector
        class Cancel(Collector):
            def completed_export(inner):
                inner.cancel_event=threading.Event();inner.cancel_event.set()
                return super().completed_export()
        with patch.object(wire,'Collector',Cancel),self.assertRaisesRegex(ValueError,'canceled during IPC'):
            wire.walk(self.binary,self.origin,raw_export=True)

    def test_last_export_retires_lane_and_allows_clean_game_reset(self):
        import rust_lab_b07_transport as wire
        from brass_ai.rust_backend.lab_transport import Collector
        origin=self.origin
        class Reset(Collector):
            def completed_export(inner):
                result=super().completed_export()
                if not inner.retired:raise AssertionError('Last page did not retire lane')
                calls=inner.metrics['calls'];inner.release()
                if inner.metrics['calls']!=calls:raise AssertionError('Redundant release RPC')
                inner.begin('new-game',origin,5000)
                if inner.retained_features or inner.forced:raise AssertionError('Previous game retained state')
                inner.release();return result
        with patch.object(wire,'Collector',Reset):self.assertIsNotNone(wire.walk(self.binary,self.origin,raw_export=True)['complete'])


if __name__=='__main__':unittest.main()
