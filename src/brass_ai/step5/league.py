"""Active fresh residual self-play trainer and shared reporting utilities."""
from pathlib import Path
from collections import Counter
import json
from brass_ai.learning.contracts import SOURCE_ROOT,digest,project_path,require
from brass_ai.learning.replay import split_for
from .policy import INDUSTRIES
def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def implementation_sources():
    from brass_ai.learning.source_identity import implementation_sources as sources
    return sources(SOURCE_ROOT)


def implementation_id():
    return digest(implementation_sources())


def seeds_for(start, count, partition='train'):
    seeds = []
    while len(seeds) < count:
        require(start < 2**31, 'Setup seed range exhausted')
        group = digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=start))
        if split_for(group) == partition:
            seeds.append(start)
        start += 1
    return seeds, start


def pottery_metrics(games, learner_ids=None):
    counts, opportunities, focused = Counter(), Counter(), Counter()
    contexts, seen = Counter(), set()
    for game in games:
        for row in game['records']:
            if learner_ids and row['behavior']['policy_id'] not in learner_ids:
                continue
            trajectory = (game['game_id'], row['seat'])
            if trajectory not in seen:
                seen.add(trajectory)
                context = next((name for i, name in enumerate(INDUSTRIES) if row['intent'][i] > .5), 'neutral')
                contexts[context] += 1
            available = {c.get('industry') for c in row['choices'] if c['kind'] == 'build'}
            opportunities.update(available)
            selected = row['choices'][row['chosen']]
            if selected['kind'] == 'build':
                counts[selected['industry']] += 1
                if row['intent'][INDUSTRIES.index(selected['industry'])] > .5:
                    focused[selected['industry']] += 1
    return dict(builds={i: counts[i] for i in INDUSTRIES},
                legal_build_opportunities={i: opportunities[i] for i in INDUSTRIES},
                builds_matching_assigned_focus={i: focused[i] for i in INDUSTRIES},
                seat_trajectory_contexts=dict(contexts),
                pottery_warning=bool(opportunities['Pottery'] >= 30 and counts['Pottery'] == 0),
                interpretation='Opportunity and chosen-build counts, not a mandate to force an industry. Pottery Flex identity is held out, never Pottery actions.')



def validate_config(config):
    from brass_ai.recovery.config import validate_config as validate
    return validate(config)

def train(root,run_path,config,**kwargs):
    from brass_ai.recovery.league import train as run
    return run(root,run_path,config,**kwargs)
