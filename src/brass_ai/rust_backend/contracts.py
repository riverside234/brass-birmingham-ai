"""Versioned local parity boundary, deliberately separate from policy inference.

Only ``permitted_request`` may feed a policy. Trusted fixtures contain private
cards and replay checkpoints and must stay in the diagnostic process.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import math

from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json, require
from brass_ai.runtime_contracts import validate

VERSION = 'rust-backend-fixtures/1.0.0'
REQUEST = 'rust-permitted-request/1.0.0'
DEBUG = 'rust-trusted-debug/1.0.0'
PROJECTION = 'rust-state-projection/1.0.0'


def wire_ledger(ledger):
    if ledger is None: return None
    require(set(ledger) == {'spent','settled'}, 'Unexpected reward ledger fields')
    require(len(ledger['spent'])==4 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=.16
                                           for x in ledger['spent']), 'Invalid causal cost ledger')
    return dict(spent=list(ledger['spent']), settled=[list(item) for item in ledger['settled']])


@lru_cache(maxsize=None)
def schema(kind):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    wire = read_json(SOURCE_ROOT/'configs/contracts/v0.2.0/wire.schema.json')
    contract = read_json(SOURCE_ROOT/'configs/contracts/rust-backend/1.0.0/wire.schema.json')
    registry = Registry().with_resource(wire['$id'], Resource.from_contents(wire))
    Draft202012Validator.check_schema(contract)
    return Draft202012Validator({**contract, '$ref': '#/$defs/'+kind}, registry=registry)


def permitted_request(view, choices, recall, *, request_id, model_id, remaining_ms=4000,
                      reward_context=None,total_ms=4000):
    """Accept permitted arguments only; reject unknown fields at every wire level.

    There is intentionally no overload accepting a checkpoint, environment,
    fixture or four-seat views. Validation precedes copying, so hidden extras
    cannot be silently serialized through a future caller.
    """
    validate('observation', view)
    validate('choices', choices)
    require(view['seat'] == view['context']['actor'] and not view['context']['terminal'],
            'A permitted request requires the current observing actor')
    require(choices and len({c['id'] for c in choices}) == len(choices), 'Empty/duplicate legal choices')
    value = dict(format=REQUEST, request_id=request_id, model_id=model_id,
                 view=copy.deepcopy(view), choices=copy.deepcopy(choices), recall=copy.deepcopy(recall),
                 budget=dict(total_ms=total_ms, remaining_ms=remaining_ms, sessions=2, threads_per_session=1),
                 reward_context=wire_ledger(reward_context))
    schema('request').validate(value)
    require(remaining_ms<=total_ms,'Remaining clock exceeds original turn allowance')
    require(all(item['revision'] <= view['context']['revision'] for item in recall), 'Future recall')
    require(not recall or recall[-1]['hand'] == view['own_hand'], 'Recall does not end at current own hand')
    return value


def canonical(snapshot):
    """Semantic projection, not an implementation-specific checkpoint hash.

    Preserve legal list order, all permitted views, private ordered card piles,
    staged context and explicit chance. Omit only the transport game UUID. The
    C# opaque hash remains separate restore evidence; it includes its own RNG
    and engine revision and is not a Rust equivalence requirement.
    """
    context = {k: v for k, v in snapshot['context'].items() if k != 'game_id'}
    checkpoint = snapshot['checkpoint']
    cards = (checkpoint['commands'][-1]['cards_after'] if checkpoint['commands']
             else checkpoint['setup']['cards'])
    progress = chance_progress(snapshot.get('chance'), checkpoint['commands'])
    return copy.deepcopy(dict(format=PROJECTION, context=context, views=snapshot.get('views'),
                              choices=snapshot.get('choices', []), chance=snapshot.get('chance'),
                              chance_progress=progress,cards=cards, result=snapshot.get('result')))


def chance_progress(chance,commands):
    """Ordered already-picked cards matter during a multi-card Rail deal.

    Pool probabilities/remaining count alone would conflate two pending deals
    that eventually deliver different hands. The trusted journal retains them.
    """
    if chance is None:return None
    picked=[]
    for command in reversed(commands):
        if 'chance' not in command:break
        picked.append(command['chance'])
    picked.reverse()
    return dict(picked=picked,needed=chance['remaining']+len(picked))


def trusted_debug(snapshot, *, identity, source, recall, ledger):
    result = dict(format=DEBUG, id=identity, source=source, checkpoint=copy.deepcopy(snapshot['checkpoint']),
                  projection=canonical(snapshot), recall=copy.deepcopy(recall), ledger=wire_ledger(ledger))
    validate_debug(result)
    return result


def validate_debug(value):
    schema('debug').validate(value)
    p = value['projection']; views = p['views']
    if p['chance'] is not None:
        require(views is None and not p['choices'] and p['context']['actor'] == -1,
                'Chance cannot expose temporary policy views or choices')
        outcomes = p['chance']['outcomes']
        require(outcomes and abs(sum(x['probability'] for x in outcomes)-1) < 1e-12,
                'Chance probabilities must sum to one')
        require(all(math.isfinite(x['probability']) and x['probability'] > 0 for x in outcomes),
                'Invalid chance probability')
        require(len({x['card'] for x in outcomes}) == len(outcomes), 'Duplicate chance outcome')
    else:
        require(views is not None and len(views) == 4, 'Need all four masked views')
        for seat, view in enumerate(views):
            validate('observation', view)
            require(view['seat'] == seat, 'Stable masked seat order changed')
            require(view['own_hand'] == sorted(p['cards']['hands'][seat]), 'Masked hand differs from trusted cards')
        validate('choices', p['choices'])
        require(len({c['id'] for c in p['choices']}) == len(p['choices']), 'Duplicate semantic choice')
    if p['context']['terminal']:
        validate('result', p['result'])
        require(not p['choices'], 'Terminal state has choices')
    return value


def first_difference(expected, actual, path='$', *, atol=0.0):
    """Stable first field/index/semantic choice mismatch; no sorting of choices."""
    if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if math.isfinite(expected) and math.isfinite(actual) and abs(expected-actual) <= atol:
            return None
    elif type(expected) is type(actual):
        if isinstance(expected, dict):
            for key in sorted(set(expected) | set(actual)):
                if key not in expected or key not in actual:
                    return dict(path=path+'.'+key, expected=expected.get(key, '<missing>'), actual=actual.get(key, '<missing>'))
                difference = first_difference(expected[key], actual[key], path+'.'+key, atol=atol)
                if difference: return difference
            return None
        if isinstance(expected, list):
            if len(expected) != len(actual):
                return dict(path=path+'.length', expected=len(expected), actual=len(actual))
            for index, (a, b) in enumerate(zip(expected, actual)):
                difference = first_difference(a, b, f'{path}[{index}]', atol=atol)
                if difference: return difference
            return None
        if expected == actual: return None
    return dict(path=path, expected=expected, actual=actual)


def comparison(expected, actual, *, expected_source, actual_source):
    difference = first_difference(expected, actual)
    return dict(format='rust-parity-comparison/1.0.0', passed=difference is None,
                expected_source=expected_source, actual_source=actual_source,
                expected_digest=digest(expected), actual_digest=digest(actual), first_difference=difference)


class RecordedDraws:
    """Portable outcomes include support/order and the actual uniform draw.

    Rust replays these values; matching seeds alone is never called RNG parity.
    """
    def __init__(self, rng=None, tape=None):
        self.rng, self.tape, self.index = rng, ([] if tape is None else tape), 0

    def choose(self, tag, identities, weights=None):
        require(identities and len(set(identities)) == len(identities), 'Invalid draw support')
        weights = list(weights) if weights is not None else [1.0]*len(identities)
        require(len(weights) == len(identities) and all(math.isfinite(w) and w >= 0 for w in weights)
                and sum(weights) > 0, 'Invalid draw weights')
        probabilities = [w/sum(weights) for w in weights]
        if self.rng is None:
            require(self.index < len(self.tape), 'Random tape exhausted')
            record = self.tape[self.index]
            require(record['tag'] == tag and record['support'] == identities and record['probabilities'] == probabilities,
                    'Random tape support/order/probabilities differ')
            number = record['uniform']
        else:
            number = self.rng.random()
        require(type(number) in (int,float) and 0 <= number < 1, 'Invalid uniform draw')
        cumulative = 0.0; selected = len(probabilities)-1
        for index, probability in enumerate(probabilities):
            cumulative += probability
            if number < cumulative:
                selected = index; break
        if self.rng is None:
            require(record['selected'] == identities[selected], 'Random tape selected outcome differs')
        else:
            self.tape.append(dict(tag=tag, support=list(identities), probabilities=probabilities,
                                  uniform=number, selected=identities[selected]))
        self.index += 1
        return selected

    def assert_consumed(self):
        require(self.index == len(self.tape), 'Unused random tape entries')


class ScriptedReplies:
    """Exact request-keyed NN replies for later native trace comparisons."""
    def __init__(self, entries):
        self.entries, self.index = entries, 0

    def predict(self, request):
        require(self.index < len(self.entries), 'Scripted NN tape exhausted')
        entry = self.entries[self.index]
        require(entry['request_digest'] == digest(request), 'Scripted NN request differs')
        self.index += 1
        return copy.deepcopy(entry['reply'])

    def assert_consumed(self):
        require(self.index == len(self.entries), 'Unused scripted NN replies')
