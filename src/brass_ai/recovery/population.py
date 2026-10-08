"""Bounded historical diversity without recycling old trajectories into PPO."""
from brass_ai.learning.contracts import digest


def ordered(entries):
    return sorted({e['model_id']:e for e in entries}.values(),key=lambda e:(e['generation'],e['model_id']))


def retain(entries, config):
    """Recent window + initial anchor + deterministic priority reservoir.

    Fixed priorities are essential: repeatedly spacing an already-pruned list
    steadily discards its middle history. Bottom-k priorities support streaming
    retention without requiring deleted checkpoints or an unbounded archive.
    """
    entries=ordered(entries)
    recent=entries[-config['recent']:];older=entries[:-config['recent']]
    if not older:return recent
    anchor=older[:1]
    reservoir=sorted(older[1:],key=lambda e:digest(dict(contract='historical-reservoir/1',
        generation=e['generation'],model_id=e['model_id'])))[:config['anchors']-1]
    return ordered(anchor+reservoir+recent)


def sample_opponents(entries, current_id, anchor, rng, recent_count, count=3):
    """25% initial/IL, 25% recent, 50% older reservoir per opponent seat.

    The explicit anchor protects exposure to the original economy. No noisy
    four-player win counts are treated as two-player PFSP ratings. The newest
    learner is excluded: pure self-play already has its own table quota.
    """
    available=ordered(e for e in list(entries)+[anchor] if e['model_id']!=current_id)
    if not available:available=[anchor]  # First collection has no older NN yet.
    other=[e for e in available if e['model_id']!=anchor['model_id']]
    recent=other[-recent_count:];older=other[:-recent_count]
    anchor_pool=[e for e in available if e['model_id']==anchor['model_id']]
    result=[]
    for _ in range(count):
        ticket=rng.random()
        pool=anchor_pool if ticket<.25 else recent if ticket<.5 else older
        result.append(rng.choice(pool or other or available))
    return result


def rollout_opponents(entries, current_id, anchor, rng, recent_count):
    """Bounded search-opponent ensemble: current, IL, recent, older."""
    available=ordered(e for e in entries if e['model_id'] not in (current_id,anchor['model_id']))
    selected=[anchor] if anchor['model_id']!=current_id else []
    if available:selected.append(available[-1])
    older=available[:-recent_count]
    if older:selected.append(rng.choice(older))
    return ordered(selected)
