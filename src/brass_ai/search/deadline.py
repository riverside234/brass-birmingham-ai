"""Optimistic launch admission, without changing a simulation's leaf or return."""
from collections import deque
import math


def configure_clock(bot, policy):
    """Common game/IL/teacher gate; CPU placement never changes work semantics."""
    bot.time_budget_primary=getattr(policy,'time_budget_primary',False)
    bot.deadline_admission=bot.time_budget_primary or getattr(policy,'teacher_work_budget',None) is None


class DeadlineAdmission:
    def __init__(self):
        self.completed = deque(maxlen=8)

    def observe(self, seconds):
        if math.isfinite(seconds) and seconds > 0:
            self.completed.append(seconds)

    @property
    def estimate(self):
        # Only local completed paths; tolerate slow outliers and startup cost.
        if len(self.completed) < 4:
            return 0.
        return sorted(self.completed)[(len(self.completed) - 1) // 4]

    def admit(self, remaining):
        return remaining > 0 and remaining >= self.estimate
