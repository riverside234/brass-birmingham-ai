"""Stdlib-only inventory of current native training continuation dependencies.

Replays and preparation caches deliberately do not travel. The exporter still
retains the small immutable probe, runtime evidence and live opponent state.
"""
from pathlib import Path

from brass_ai.learning.paths import resolve_project_path


REPLAY_POLICY = 'fresh-buffers/1.0.0'


def native_inventory(root, run, state, *, add, read, require, runtime):
    def path(name):
        return resolve_project_path(root, name)

    def pinned(item):
        p = path(item['path']); add(p, item['sha256'])
        require('bytes' not in item or p.stat().st_size == item['bytes'], 'Native dependency size differs: '+str(p))
        return p

    def refs(value, *, required=True):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and value.get('sha256'):
                if required or path(value['path']).is_file(): pinned(value)
            for child in value.values(): refs(child, required=required)
        elif isinstance(value, list):
            for child in value: refs(child, required=required)

    probe = state.get('il_kl_limit', {}).get('probe')
    require(probe is not None, 'Native transfer requires the committed fixed IL probe')
    probe_path = pinned(probe); metadata = read(probe_path)
    require(probe_path == run/'il-kl/manifest.json' and metadata['probe_hash'] == probe['probe_hash']
            and len(metadata['rows']) == 2048, 'Transfer needs the original fixed 2048-position IL probe')
    cache = pinned(metadata['cache'])
    require(cache == run/'il-kl/probe.npz', 'Fixed IL probe array is outside its owned directory')

    # These references are live metadata, weights and immutable probe arrays,
    # not the original IL replay or its .prepared feature cache.
    refs(state.get('quality_pool', {}).get('members', []))
    for key in ('policy_anchor', 'champion_mix', 'origin'):
        refs(state.get(key))
    ratings = state.get('ratings', {})
    refs({k:v for k,v in ratings.items() if k != 'agents'})
    refs(ratings.get('agents', []), required=False)
    for transition in state.get('source_transitions', []):
        add(run/transition['backup'])
    for key in ('auxiliary-checkpoint-inventory.json',):
        if (run/key).is_file(): add(run/key)
    if (run/'machine-transfer').is_dir():
        for p in sorted((run/'machine-transfer').glob('*.json')): add(p)

    seen = set()
    def build(item):
        p = pinned(item)
        if p in seen: return
        seen.add(p); value = read(p)
        for key in ('collector', 'binary', 'tests', 'build_log', 'performance_tests', 'reward_tests'):
            if key in value:
                file = pinned(value[key]); runtime.add(file.relative_to(root).as_posix())
        refs(value.get('bootstrap'))
        for file in (value.get('bootstrap', {}).get('binary'), *value.get('bootstrap', {}).get('dependencies', [])):
            if file: runtime.add(path(file['path']).relative_to(root).as_posix())
        # Preserve originals used to bind additive wire/verification adapters.
        for key in ('original_teacher_receipt','original_verification_receipt'):
            if key in value: build(value[key])
        for previous in value.get('previous_runtimes', []): build(previous)
        if value.get('b01'):
            b01 = pinned(value['b01'])
            fixtures = b01.parent/'inputs/fixtures.json'
            if fixtures.is_file(): add(fixtures)

    native = state['native_backend']
    q = read(pinned(native['qualification']))
    build(native['build'])
    if q.get('teacher_build'): build(q['teacher_build'])
    if native.get('shadow', {}).get('teacher_build'): build(native['shadow']['teacher_build'])
    # The rollback manifest remains pinned audit evidence. Its old diagnostic
    # replay payload is intentionally omitted; it is not a destination backup.
    pinned(native['rollback'])
    for name in ('artifacts/rust-backend/collector-performance/build.json',
                 'artifacts/rust-backend/teacher-verification/build.json',
                 'artifacts/rust-backend/teacher-zero-insolvency/build.json'):
        p = root/name
        if p.is_file():
            import hashlib
            with p.open('rb') as stream: expected = hashlib.file_digest(stream, 'sha256').hexdigest()
            build(dict(path=str(p), sha256=expected))
    # Small sealed source input for a destination-side adapter build, if it
    # wasn't prepared on the source. No IL data or compiler/target cache.
    from brass_ai.learning.post_transfer_reward import eligible
    source=root/'distribution/rust-lab-g1250/inputs.zip'
    if eligible(state['signature']['config']):
        require(source.is_file() or (root/'artifacts/rust-backend/teacher-zero-insolvency/build.json').is_file(),
            'Prepare the post-transfer reward teacher before exporting, or restore distribution/rust-lab-g1250/inputs.zip')
        if source.is_file():add(source)
    return dict(contract=REPLAY_POLICY, origin_generation=state['iteration'],
                auxiliary_buffer_iterations=8, il_probe_positions=2048)
