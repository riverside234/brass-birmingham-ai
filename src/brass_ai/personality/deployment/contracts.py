"""Versioned P4 selection/calibration contracts, not laptop tier certificates."""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import math

from brass_ai.learning.contracts import digest, read_json, require
from brass_ai.personality.search.contracts import Profile, identity as search_identity

VERSION = 'personality-deployment/1.1.0'
ROSTER = 'personality-roster/1.1.0'
ARCHETYPE = 'personality-profile/1.0.0'
MEMBER = 'personality-micro-variant/1.0.0'
PANEL = 'personality-roster-panel/1.0.0'
REPORT = 'personality-roster-check/1.1.0'
SAMPLING = 'uniform-archetype-members/1.0.0'


@lru_cache(maxsize=1)
def identity():
    return dict(version=VERSION, search=search_identity(), source=digest({
        p.name: p.read_text(encoding='utf-8') for p in sorted(Path(__file__).parent.glob('*.py'))}))


@dataclass(frozen=True)
class Gates:
    # Fractions of each representative's nearest other representative distance.
    radius_fraction: float = .40
    separation_margin: float = .10
    variant_signal: float = .50
    min_setups: int = 8
    win_floor: float = .20
    win_tolerance: float = .05
    vp_tolerance: float = 5.
    rank_tolerance: float = .20
    recognition_floor: float = .60
    variety_gain: float = .02
    human_raters: int = 2
    human_examples: int = 8

    def validate(self):
        require(all(math.isfinite(x) for x in asdict(self).values()), 'Nonfinite roster gates')
        require(0 < self.radius_fraction < .5 and 0 < self.separation_margin < .5
                and .1 <= self.variant_signal <= 2., 'Invalid neighborhood/noise gates')
        require(type(self.min_setups) is int and 8 <= self.min_setups <= 128,
                'Use at least eight independent calibration setups')
        require(0 < self.win_floor <= .5 and 0 <= self.win_tolerance <= .1
                and 0 <= self.vp_tolerance <= 10 and 0 <= self.rank_tolerance <= .3,
                'Invalid P4 quality gates')
        require(.5 <= self.recognition_floor <= 1 and 0 < self.variety_gain < .5
                and self.human_raters >= 2 and self.human_examples >= 8, 'Invalid variety/recognition gates')
        return self


def sealed(value):
    return dict(value, id=digest(value))


def read_sealed(path, version):
    value = read_json(path)
    require(value['format'] == version and value['contract'] == identity()
            and value['id'] == digest({k: v for k, v in value.items() if k != 'id'}),
            'P4 manifest/source changed')
    return value


def validate_roster(value):
    """Also run on freshly built records; hash validity alone is not semantics."""
    groups, members = value['archetypes'], value['members']
    counts = value['counts']
    require(2 <= counts['archetypes'] <= 16 and 2 <= counts['per_archetype'] <= 8,
            'Invalid preview/full roster size')
    require(len(groups) == counts['archetypes'] and len(members) == counts['archetypes'] * counts['per_archetype'],
            'Incomplete roster; never pad missing members')
    ids = [m['id'] for m in members]
    require(len(set(ids)) == len(ids) and len({m['candidate_id'] for m in members}) == len(ids),
            'Duplicate roster member/controller')
    require(len({g['id'] for g in groups}) == len(groups), 'Duplicate archetype')
    assigned = [i for g in groups for i in g['members']]
    require(len(assigned) == len(ids) and set(assigned) == set(ids), 'Overlapping/missing archetype membership')
    for g in groups:
        require(g['format'] == ARCHETYPE and len(g['members']) == counts['per_archetype']
                and g['representative'] in g['members'], 'Representative must be one of the eight members')
        require(g['sampling'] == dict(version=SAMPLING, weights=[1 / len(g['members'])] * len(g['members'])),
                'Unreviewed runtime sampling policy')
        require(g['supported_tiers'] == [], 'P4 cannot certify difficulty tiers')
    for m in members:
        Profile.parse(m['profile'])
        require(m['format'] == MEMBER and m['model_id'] == value['shared_model']['model_id']
                and m['archetype_id'] in {g['id'] for g in groups}
                and m['id'] in next(g for g in groups if g['id'] == m['archetype_id'])['members'],
                'Mixed weights or inconsistent member parent')
        require(m['tier_settings'] == {} and m['fallback'] is None,
                'Tier settings/fallback require P5 qualification')
    require(value['release_qualified'] is False, 'P4 does not authorize a laptop release')
    require(type(value.get('fixture', True)) is bool, 'Invalid fixture provenance')
    require(value['full_size'] == (len(groups)==16 and len(members)==128), 'False full-roster claim')
    require(not value['p4_qualified'] or (value['full_size'] and value['decision_mode']=='searched'
            and value.get('fixture', True) is False and 'qualification' in value),
            'Fixture, preview or unreviewed roster cannot qualify P4')
    return value


def resolve(value, member_id):
    """Small online recipe; descriptor coordinates are never network inputs."""
    validate_roster(value)
    rows = [m for m in value['members'] if m['id'] == member_id]
    require(len(rows) == 1, 'Unknown pinned roster member')
    from brass_ai.search.common import SearchConfig
    return Profile.parse(rows[0]['profile']), SearchConfig(**value['search']).validate()
