"""Exact execution-source review for the B07 replay-location repair.

Keep the qualified collection signature while finishing an interrupted native
iteration. The full execution fingerprint remains distinct and is recorded in
the next checkpoint/report. This is not a general source-upgrade exemption.
"""
import copy
from functools import lru_cache

from brass_ai.bank import read
from brass_ai.learning.contracts import CONFIG_DIR,digest,require
from brass_ai.step5.league import implementation_id

REVIEW=CONFIG_DIR/'b07-replay-location-review.json'
QUALIFIED='9efcf34f853d75ab844366b6819da4b0921be38d3e2dfb806cf282ace76e7a30'


def execution():
    actual=implementation_id()
    review=read(REVIEW)
    require(review['contract']=='brass-b07-replay-location-review/1.0.0' and
            review['qualified_implementation']==QUALIFIED and
            isinstance(review['execution_implementation'],str) and len(review['execution_implementation'])==64 and
            all(c in '0123456789abcdef' for c in review['execution_implementation']),
            'Unknown B07 replay-location review')
    # All implementation files, including this module, enter the actual digest.
    # Any unreviewed code edit gets its own identity and cannot reuse old state.
    matched=actual==review['execution_implementation']
    return dict(implementation=actual,
        qualified_implementation=review['qualified_implementation'] if matched else actual,
        replay_location_review=digest(review) if matched else None)


def qualified_implementation():
    return execution()['qualified_implementation']


def accepts(implementation):
    current=execution()
    return implementation in (current['implementation'],current['qualified_implementation'])


def accepts_execution(saved):
    """Only exact reviewed predecessor checkpoints may cross this repair.

    Keep their old execution/review receipt intact; subsequent commits record
    the actual new execution. Unknown source or review digests still fail.
    """
    current=execution()
    if saved==current:return True
    return (current['replay_location_review'] is not None and
            saved in read(REVIEW).get('predecessor_executions',[]) and
            saved.get('qualified_implementation')==current['qualified_implementation'])


@lru_cache(maxsize=8)
def reviewed_validator(identity):
    """Known reviewed code may retain its original coverage on cold recovery.

    This never grants a warm heap capability or relabels an old disk receipt.
    Exact replay/features are still checked by the current reader.
    """
    current=execution()
    return current['replay_location_review'] is not None and identity in {
        v['implementation'] for v in read(REVIEW).get('predecessor_executions',[])}


def training_signature(state,signature):
    """Keep frozen jobs/receipts byte-identical only for the exact native fix."""
    value=state.get('native_backend')
    if value is None or state['signature']==signature:return signature
    current=execution();old=state['signature']
    from .lab_shadow_config import is_shadow,approved
    target=dict(signature,implementation=old['implementation'])
    require(current['replay_location_review'] is not None and
            signature['implementation']==current['implementation'] and
            old['implementation']==value['implementation']==current['qualified_implementation'] and
            value['mode'] in ('rust','reference') and
            (old['config'].get('teacher_phase',{}).get('mode','off')=='off' or is_shadow(old['config'])) and
            (target==old or approved(state,target)),
            'Native continuation source/config differs from the exact replay-location review')
    return copy.deepcopy(target)
