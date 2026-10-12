"""Independent, immutable P5 recipes; P0--P4 and Step 5 keep their identities."""
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
import math

from brass_ai.learning.contracts import digest, read_json, require
from brass_ai.personality.deployment.contracts import identity as p4_identity
from brass_ai.search.common import SearchConfig

VERSION = 'personality-difficulty/1.1.0'
PROFILES = 'personality-tier-profiles/1.0.0'
PANEL = 'personality-difficulty-panel/1.0.0'
CHUNK = 'personality-difficulty-chunk/1.0.0'
REPORT = 'personality-difficulty-report/1.0.0'
RECORD = 'personality-difficulty-game/1.0.0'
UI = 'personality-public-ui/1.0.0'
REVIEW = 'personality-play-session-review/1.0.0'
TIERS = ('easy', 'medium', 'hard')
HINT_MODES = ('hidden', 'public', 'intent-hints')
STOCK = 'stock-hard-guarded-1.1.0'
RELEASE_CHECKS = frozenset(('every_easy_member', 'every_easy_mixture', 'same_tier_consistency',
    'tier_ordering', 'complete_panel', 'full_roster', 'qualified_p4', 'real_artifacts',
    'fresh_release_panel', 'no_failed_games', 'searched', 'both_eras',
    'every_controller_searched', 'one_measured_environment', 'human_play_review'))


@lru_cache(maxsize=1)
def identity():
    base = Path(__file__).parent
    root = base.parents[3]
    paths = list(base.glob('*.py')) + list((root/'src/BrassPersonalityHost').glob('*.cs'))
    return dict(version=VERSION, p4=p4_identity(), source=digest({
        p.relative_to(root).as_posix(): p.read_text(encoding='utf-8').replace('\r\n', '\n')
        for p in sorted(paths)}))


def sealed(value):
    return dict(value, id=digest(value))


def read_sealed(path, version):
    value = read_json(path)
    require(value['format'] == version and value['contract'] == identity()
            and value['id'] == digest({k: v for k, v in value.items() if k != 'id'}),
            'P5 evidence/source changed; do not relabel old qualification')
    return value


def validate_report_summary(report, roster):
    """Check the portable evidence claim; raw evidence is rechecked by load_report.

    Neither an empty checks dictionary nor a caller-supplied qualified flag is
    sufficient to enable a menu or export a release candidate.
    """
    require(report['format'] == REPORT and report['contract'] == identity()
            and report['id'] == digest({k:v for k,v in report.items() if k != 'id'})
            and report['roster_id'] == roster['id'], 'Difficulty evidence identity differs')
    checks = report['checks']
    require(set(checks) == RELEASE_CHECKS and all(type(v) is bool for v in checks.values()),
            'Missing or invalid difficulty gates')
    validate_profiles(report['profiles']); gates = Gates(**report['gates']).validate()
    require(type(report['p5_qualified']) is bool and report['p5_qualified'] == all(checks.values())
            and report['release_qualified'] is False, 'False difficulty claim')
    require(report['qualified_tiers'] == (list(TIERS) if report['p5_qualified'] else []), 'Unmeasured tiers')
    if report['p5_qualified']:
        require(roster['p4_qualified'] and roster['full_size'] and report['fixture'] is False
                and report['stage'] == 'release' and report['sources'] and report['review_source']
                and not report['missing_jobs'] and not report['failures'] and not report['stock_failures']
                and report['games'] == report['expected_games'] > 0
                and len(set(report['setup_groups'])) >= gates.min_setups, 'Incomplete release evidence')
        require(set(report['members']) == {m['id'] for m in roster['members']}
                and set(report['mixtures']) == {g['id'] for g in roster['archetypes']}, 'Missing member/mixture evidence')
        for item in list(report['members'].values()) + list(report['mixtures'].values()):
            require(item['easy_original_hard']['passed'] is True and set(item['tiers']) == set(TIERS)
                    and all(item['tiers'][t]['passed'] is True for t in TIERS), 'Unqualified member or mixture')
        require(set(report['ordering']) == {'medium>easy', 'hard>medium'}
                and all(v['passed'] is True for v in report['ordering'].values())
                and report['human']['passed'] is True, 'Difficulty ordering/review is pending')
    return report


def qualified_report(report, roster):
    try:
        return bool(report is not None and validate_report_summary(report, roster)['p5_qualified'])
    except (KeyError, TypeError, ValueError):
        return False


def default_profiles():
    return dict(format=PROFILES, hint_mode='public', selection='supported-root-visits',
                near_best_tolerance=0., workers=2, tensor_threads=2, cpu_role='laptop',
                tiers={tier: dict(seconds=seconds) for tier, seconds in zip(TIERS, (2., 5., 12.))})


def validate_profiles(value):
    require(set(value) == set(default_profiles()) and value['format'] == PROFILES,
            'Unknown tier profile contract')
    require(value['hint_mode'] in HINT_MODES and value['selection'] == 'supported-root-visits'
            and value['near_best_tolerance'] == 0., 'Unsupported hint/selection recipe')
    require(type(value['workers']) is int and value['workers'] in (2, 4)
            and value['tensor_threads'] == 2 and value['cpu_role'] in ('laptop', 'lab'),
            'Invalid declared P5 resources')
    require(set(value['tiers']) == set(TIERS), 'All three versioned tiers required')
    for tier, seconds in zip(TIERS, (2., 5., 12.)):
        require(value['tiers'][tier] == dict(seconds=seconds), 'Preserve the 2/5/12 whole-turn caps')
    return value


def resolve_search(roster, profiles, tier):
    validate_profiles(profiles)
    require(tier in TIERS, 'Unknown difficulty tier')
    # Nothing changes policy, rewards, masks, priors, candidate coverage or NN size.
    base = SearchConfig(**roster['search']).validate()
    return replace(base, seconds=profiles['tiers'][tier]['seconds'], workers=profiles['workers'],
                   cpu_role=profiles['cpu_role']).validate()


@dataclass(frozen=True)
class Gates:
    min_setups: int = 32
    family_alpha: float = .05
    easy_gain: float = .02
    equivalence_margin: float = .05
    ordering_gain: float = .01
    human_raters: int = 2
    human_floor: float = 3.

    def validate(self):
        require(all(math.isfinite(x) for x in asdict(self).values()), 'Nonfinite P5 gates')
        require(type(self.min_setups) is int and 32 <= self.min_setups <= 512,
                'At least 32 independent setup groups for P5 release intervals')
        require(0 < self.family_alpha <= .05 and .01 <= self.easy_gain <= .15
                and .01 <= self.equivalence_margin <= .05 and .005 <= self.ordering_gain <= .1,
                'Invalid prespecified strength/equivalence gates')
        require(type(self.human_raters) is int and self.human_raters >= 2
                and 3 <= self.human_floor <= 5, 'Invalid human review gates')
        return self
