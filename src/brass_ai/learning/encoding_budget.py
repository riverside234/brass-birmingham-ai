"""Cooperative feature work uses the request's existing absolute turn clock."""
import math
import time
from .contracts import require


class EncodingStopped(ValueError):
    def __init__(self, reason):
        super().__init__('Feature encoding '+reason+' at the admitted decision deadline/cancellation gate')
        self.reason = reason


def check_budget(deadline=None, cancelled=None):
    require(deadline is None or math.isfinite(deadline),'Invalid feature decision deadline')
    if cancelled is not None and cancelled.is_set():
        raise EncodingStopped('cancelled')
    if deadline is not None and time.monotonic() >= deadline:
        raise EncodingStopped('expired')
