import copy
import random
import unittest

from brass_ai.learning.contracts import digest,read_json
from brass_ai.rust_backend.contracts import (RecordedDraws,ScriptedReplies,permitted_request,
                                           schema,validate_debug)
from support import evidence,root_fixture


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=root_fixture()
        cls.model=read_json(evidence()/'baseline.json')['model']['model_id']

    def request(self,fixture=None,**changes):
        f=fixture or self.fixture;p=f['projection'];seat=p['context']['actor']
        args=dict(view=p['views'][seat],choices=p['choices'],recall=f['recall'][seat],
                  request_id='boundary-test',model_id=self.model,reward_context=f['ledger'])
        args.update(changes)
        return permitted_request(**args)

    def test_valid_detached_permitted_request(self):
        result=self.request();schema('request').validate(result)
        result['view']['own_hand'].clear()
        self.assertTrue(self.request()['view']['own_hand'])
        self.assertEqual(self.request()['budget'],dict(total_ms=4000,remaining_ms=4000,sessions=2,threads_per_session=1))

    def test_hidden_unknown_fields_are_rejected_recursively(self):
        seat=self.fixture['projection']['context']['actor']
        for field in ('checkpoint','hands','deck','setup','trusted_debug'):
            with self.subTest(field=field):
                view=copy.deepcopy(self.fixture['projection']['views'][seat]);view[field]=['secret']
                with self.assertRaises(Exception):self.request(view=view)
        view=copy.deepcopy(self.fixture['projection']['views'][seat]);view['players'][0]['hand']=['secret']
        with self.assertRaises(Exception):self.request(view=view)
        choices=copy.deepcopy(self.fixture['projection']['choices']);choices[0]['hidden_deck']=['secret']
        with self.assertRaises(Exception):self.request(choices=choices)

    def test_unavailable_hidden_world_cannot_change_policy_request(self):
        changed=copy.deepcopy(self.fixture);seat=changed['projection']['context']['actor']
        changed['projection']['cards']['deck'].reverse()
        changed['projection']['cards']['hands'][(seat+1)%4].reverse()
        changed['checkpoint']['setup']['cards']['deck'].reverse()
        self.assertEqual(self.request(),self.request(changed))

    def test_trusted_import_is_not_a_policy_request(self):
        with self.assertRaises(Exception):schema('request').validate(self.fixture)
        with self.assertRaises(Exception):self.request(view=self.fixture)
        request=self.request();request['checkpoint']=self.fixture['checkpoint']
        with self.assertRaises(Exception):schema('request').validate(request)

    def test_sealed_permitted_files_match_corpus(self):
        from brass_ai.rust_backend.fixtures import read_gzip
        directory=evidence()/'local/fixtures';corpus=read_json(directory/'corpus.json');checked=0
        for entry in corpus['roots']:
            if 'request_digest' not in entry:continue
            request=read_gzip(directory/'permitted'/(entry['id']+'.json.gz'))
            self.assertEqual(digest(request),entry['request_digest'],entry['id'])
            self.assertEqual(request['request_id'],entry['id'])
            self.assertEqual(request['model_id'],self.model)
            checked+=1
        self.assertGreaterEqual(checked,100)

    def test_remaining_clock_and_session_contract_reject_invalid_inputs(self):
        for remaining in (-1,4001,4.5):
            with self.subTest(remaining=remaining),self.assertRaises(Exception):self.request(remaining_ms=remaining)
        for field in ('sessions','threads_per_session'):
            request=self.request();request['budget'][field]=4
            with self.assertRaises(Exception):schema('request').validate(request)

    def test_all_four_game_settings_keep_two_single_thread_sessions(self):
        for milliseconds in (4000,8000,10000,12000):
            request=self.request(total_ms=milliseconds,remaining_ms=milliseconds-300)
            self.assertEqual(request['budget']['total_ms'],milliseconds)
            self.assertEqual(request['budget']['sessions'],2)
            self.assertEqual(request['budget']['threads_per_session'],1)
        with self.assertRaises(Exception):self.request(total_ms=6000)

    def test_ledger_rejects_nonfinite_or_excess_cost(self):
        for cost in (float('nan'),float('inf'),-.01,.17):
            with self.subTest(cost=cost),self.assertRaises(ValueError):
                self.request(reward_context=dict(spent=[cost,0.,0.,0.],settled=[]))

    def test_recall_cannot_carry_another_hand_or_future_revision(self):
        with self.assertRaises(ValueError):self.request(recall=[dict(revision=100000,hand=['secret'])])
        with self.assertRaises(Exception):self.request(recall=[dict(revision=0,hand=[],opponent_hand=['secret'])])

    def test_duplicate_choices_rejected(self):
        choices=copy.deepcopy(self.fixture['projection']['choices']);choices.append(choices[0])
        with self.assertRaises(ValueError):self.request(choices=choices)

    def test_trusted_hand_mismatch_is_detected(self):
        changed=copy.deepcopy(self.fixture);changed['projection']['cards']['hands'][0].clear()
        with self.assertRaises(ValueError):validate_debug(changed)

    def test_recorded_draws_replay_outcomes_and_detect_changed_order(self):
        draws=RecordedDraws(random.Random(91));expected=[draws.choose('chance',['a','b','c'],[2,1,3]) for _ in range(5)]
        replay=RecordedDraws(tape=draws.tape)
        self.assertEqual(expected,[replay.choose('chance',['a','b','c'],[2,1,3]) for _ in range(5)])
        replay.assert_consumed()
        with self.assertRaises(ValueError):RecordedDraws(tape=draws.tape).choose('chance',['b','a','c'],[2,1,3])
        changed=copy.deepcopy(draws.tape);changed[0]['selected']='wrong'
        with self.assertRaises(ValueError):RecordedDraws(tape=changed).choose('chance',['a','b','c'],[2,1,3])
        with self.assertRaises(ValueError):RecordedDraws(tape=draws.tape).assert_consumed()

    def test_scripted_nn_replies_bind_exact_permitted_request(self):
        request=self.request();entry=dict(request_digest=digest(request),reply=dict(utility=[.1,.2,.3,.4]))
        script=ScriptedReplies([entry]);self.assertEqual(script.predict(request),entry['reply']);script.assert_consumed()
        changed=copy.deepcopy(request);changed['choices'].reverse()
        with self.assertRaises(ValueError):ScriptedReplies([entry]).predict(changed)
        with self.assertRaises(ValueError):script.predict(request)


if __name__=='__main__':unittest.main()
