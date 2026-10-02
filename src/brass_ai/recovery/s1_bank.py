"""Restore frozen pre-ISMCTS S1 code for one-second training teachers only."""
from pathlib import Path
import shutil

from brass_ai.bank import ADAPTER_FILES, digest, load_bank, read, require, sha

SOURCE_BANK = 'bank-b880ac1fd5f1dfe8bcc2'
PARENT_BANK = 'bank-af17b1aadbe7f9b21fa7'
STYLES = ('balanced', 'rail-tempo', 'cotton', 'goods-beer', 'supplier', 'pottery-flex')
ADMISSION = 'user-directed-v3-s1/1.0.0'


def prepare(root, *, publish=True):
    """Clone immutable S1 sources/runtime; never rebuild from active ISMCTS code.

    Five existing one-second profiles are byte-preserved. Pottery Flex uses the
    original v3 parent settings plus the same clock and partition defaults.
    This is explicit user-selected admission, not a strength qualification.
    """
    from brass_ai.step5.league import write_json
    root = Path(root).resolve()
    original = load_bank(root, SOURCE_BANK)['payload']
    parents = load_bank(root, PARENT_BANK)['payload']
    snapshot = root / 'training/opponents/bank' / SOURCE_BANK / 'source'
    private = root / 'artifacts/legacy-ai/bank' / SOURCE_BANK / 'runtime'
    profiles = {}
    origins = {path: snapshot / path for path in original['source_files']
               if not path.startswith('configs/legacy-ai/')}
    for style in STYLES:
        name = style + '-v3-lab-1s'
        profile = original['profiles'].get(name)
        if profile is None:
            parent = parents['profiles'][style + '-v3']
            profile = dict(parent, id=parent['id'] + '-lab-1s', milliseconds=1000,
                           partition_count=1, partition_index=0)
        require(profile['mode'] == 'S1' and profile['milliseconds'] == 1000 and
                profile['cash_policy'] == profile['conversion_policy'] == 1,
                'Use the original four-player v3 S1 contract')
        path = root / 'configs/legacy-ai' / (name + '.json')
        if path.exists():
            require(read(path) == profile, 'Existing one-second S1 profile changed')
        else:
            write_json(path, profile)
        profiles[name] = profile
        origins[path.relative_to(root).as_posix()] = path
    # Keep the compiled C# pins from the original bank and snapshot today's
    # training-only adapter separately. No installed-game file is touched.
    adapters = (*ADAPTER_FILES, 'src/brass_ai/recovery/s1_bank.py')
    for path in adapters:
        origins[path] = root / path
    files = {path: sha(source) for path, source in origins.items()}
    payload = dict(original, profiles=profiles, source_files=files,
                   adapter_files=list(adapters), fixed_work=False,
                   training_origin=dict(backend='S1', source_bank=SOURCE_BANK,
                       parent_bank=PARENT_BANK, seconds=1.0,
                       admission=ADMISSION, scope='training/evaluation only'))
    bank = 'bank-' + digest(payload)[:20]
    directory = root / 'training/opponents/bank' / bank
    if directory.exists():
        load_bank(root, bank)
    else:
        # A clean lab clone already carries this derived source bank and its
        # imported runtime. Creating a new bank locally needs the original
        # private runtime, but reusing the tracked bank does not.
        runtime = read(private / 'runtime.json')
        require(runtime['files'] == original['runtime_files'], 'Original S1 runtime identity changed')
        for entry in runtime['files']:
            require(sha(private / entry['name']) == entry['sha256'], 'Original S1 worker/dependency changed')
        for path, source in origins.items():
            destination = directory / 'source' / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        write_json(directory / 'manifest.json', dict(id=bank, payload=payload))
        target = root / 'artifacts/legacy-ai/bank' / bank / 'runtime'
        target.mkdir(parents=True, exist_ok=False)
        for entry in runtime['files']:
            shutil.copyfile(private / entry['name'], target / entry['name'])
        frozen = dict(runtime, sources=[dict(path=(directory / 'source' / p).relative_to(root).as_posix(),
                                           sha256=files[p]) for p in payload['runtime_source_paths']])
        write_json(target / 'runtime.json', frozen)
    manifest_sha = sha(directory / 'manifest.json')
    record = dict(format='v3-s1-training-bank/1.0.0', bank=bank,
        manifest_sha256=manifest_sha, backend='S1', seconds=1.0,
        profiles=list(profiles), source_bank=SOURCE_BANK, parent_bank=PARENT_BANK,
        admission=ADMISSION, strength='unmeasured-one-second-setting')
    if publish:
        old = root / 'training/opponents/archive/pre-s1-training-pool.json'
        if not old.exists():
            write_json(old, read(root / 'training/opponents/pool.json'))
        pool = dict(format='legacy-pool/2.0.0', backend='S1', training={}, evaluation={},
            note='User-selected original v3 S1, one second; training uses 16 tables x 2 workers. Pottery identity held out; learner Pottery unrestricted.')
        for name in profiles:
            purpose = 'evaluation' if name.startswith('pottery-flex') else 'training'
            pool[purpose][bank + '/' + name] = dict(admission=ADMISSION,
                manifest_sha256=manifest_sha, profile=name)
        write_json(root / 'training/opponents/pool.json', pool)
        write_json(root / 'training/opponents/current.json', record)
        write_json(root / 'training/opponents/lab/current.json', record)
    return record
