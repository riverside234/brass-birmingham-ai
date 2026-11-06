"""Explicit execution-only adoption; model/training state is unchanged."""
import copy
from brass_ai.learning.contracts import require

MIGRATION='selfplay-throughput/1.0.0'


def configure(config):
    from .plackett_luce import COMPUTE,validate_compute
    result=copy.deepcopy(config)
    block=result.setdefault('rating_compute',copy.deepcopy(COMPUTE))
    validate_compute(block);block['workers']=16
    require(16<=result['audit_workers'] and 16<=result['execution']['cpu_budget']-result['execution']['cpu_reserve'],
            '16 bootstrap workers require at least 16 available reader/CPU slots')
    return result
