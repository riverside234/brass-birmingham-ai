"""Allowlisted public DTOs and deterministic templates, not Unity/P6 deployment.

Default indicators NEVER consume a goal or observation. Opt-in hints accept the
same seat's validated IntentState snapshot, not an arbitrary raw network output.
"""
from dataclasses import dataclass

from brass_ai.learning.contracts import require
from brass_ai.personality.contracts import context
from brass_ai.personality.search.contracts import Profile
from brass_ai.personality.deployment.contracts import validate_roster
from .contracts import HINT_MODES, TIERS, UI, qualified_report


def menu(value, report=None, *, fixture=False):
    """Only public archetype IDs/names: no member, checkpoint, or controller IDs."""
    validate_roster(value)
    require(len(value['archetypes']) == 16 and len(value['members']) == 128,
            'The selection interface requires 16 styles with eight members each')
    qualified = not fixture and qualified_report(report, value)
    return dict(format=UI, status='P5-qualified; P6 pending' if qualified else
                ('Fixture preview — unqualified' if fixture else 'Calibration pending'),
                choices=[dict(archetype_id=g['id'], name=g['name'], description=g['description'],
                              tiers=list(TIERS) if qualified else [], enabled=qualified)
                         for g in value['archetypes']],
                hint_modes=list(HINT_MODES), default_hint_mode='public',
                qualified_hint_mode=report['profiles']['hint_mode'] if qualified else None,
                hint_notice='AI intent hints deliberately reveal a broad part of this AI’s own plan. '
                            'They are uncertain intentions, not explanations of every decision.',
                release_qualified=False)


def selection(public_menu, archetype_id, tier, hint_mode='public', *, preview=False):
    choices = [x for x in public_menu['choices'] if x['archetype_id'] == archetype_id]
    require(len(choices) == 1 and tier in TIERS and hint_mode in HINT_MODES, 'Invalid public selection')
    require(preview or (choices[0]['enabled'] and tier in choices[0]['tiers']), 'Difficulty is not qualified')
    require(preview or public_menu['qualified_hint_mode']==hint_mode, 'Different hint mode requires calibration')
    # Choosing another hint mode or clock cannot inherit the calibrated claim.
    return dict(format=UI, archetype_id=archetype_id, tier=tier, hint_mode=hint_mode,
                preview=preview, requires_exact_profile_match=True)


@dataclass(frozen=True)
class PublicAction:
    kind: str

    def __post_init__(self):
        require(self.kind in ('build', 'develop', 'bonus_develop', 'sell', 'network', 'loan', 'scout', 'pass'),
                'Only completed visible actions may enter default indicators')


PUBLIC_TEXT = dict(build='Investing in industry', develop='Preparing higher-level industries',
                   bonus_develop='Preparing higher-level industries', sell='Completing sales',
                   network='Expanding connections', loan='Preparing funds', scout='Revising options')
PUBLIC_TEXT['pass'] = 'Waiting for another opportunity'
HINT_TEXT = dict(sale_chain='Considering a sale if resources remain available',
                 industry_sequence='Preparing an industry investment',
                 resource_supply='Considering resource supply',
                 network_extension='Looking for useful connections',
                 liquidity_recovery='Preparing funds', flexible='Keeping options open')


class IntentDisplay:
    """At most one update per own turn; two-turn hysteresis for changed hints."""
    def __init__(self, seat):
        require(type(seat) is int and 0 <= seat < 4, 'Invalid displayed seat')
        self.seat = seat
        self.last_turn = None
        self.last_text = None
        self.active = None
        self.pending = None
        self.pending_count = 0
        self.mode = None

    def update(self, *, era, turn, description, mode='public', action=None, own_intent=None):
        require(era in ('CanalEra', 'RailEra') and type(turn) is int and turn >= 0
                and mode in HINT_MODES and isinstance(description, str), 'Invalid public turn/display mode')
        require(action is None or isinstance(action, PublicAction), 'Raw observations are not UI input')
        if mode == 'hidden':
            self.last_text = None; self.mode = mode
            return dict(format=UI, text=None, uncertain=False)
        key = (era, turn)
        if self.last_turn and self.last_turn[0] == era:
            require(turn >= self.last_turn[1], 'Intent UI turn moved backwards')
        if self.last_turn != key:
            if self.last_turn is None or self.last_turn[0] != era:
                self.active = self.pending = None; self.pending_count = 0
            text = PUBLIC_TEXT[action.kind] if action else description
            if mode == 'intent-hints' and own_intent is not None:
                require(own_intent['seat'] == self.seat, 'Cannot display another seat’s private intent')
                profile = Profile.parse(own_intent['profile'])
                goal = own_intent['goal']; context(profile.style, goal)
                # Never display exact region, horizon, probabilities, cards or search traces.
                proposed = goal['family'] if goal and goal['confidence'] >= .65 and goal['age'] < goal['horizon'] else None
                if proposed != self.active:
                    self.pending_count = self.pending_count + 1 if proposed == self.pending else 1
                    self.pending = proposed
                    if self.active is None or proposed is None or self.pending_count >= 2:
                        self.active = proposed
                if self.active: text = HINT_TEXT[self.active]
            else:
                self.active = self.pending = None; self.pending_count = 0
            self.last_turn = key; self.last_text = text
        # Switching from hints to public hides private-derived text immediately.
        if mode == 'public' and self.mode != 'public':
            self.last_text = PUBLIC_TEXT[action.kind] if action else description
            self.active = self.pending = None; self.pending_count = 0
        self.mode = mode
        return dict(format=UI, text=self.last_text, uncertain=mode == 'intent-hints' and self.active is not None)
