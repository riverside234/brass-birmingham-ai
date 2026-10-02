"""Git-ready authored models/workers and a separate private rules transfer.

Original Hard is vendor code inside Assembly-CSharp.dll. Its pinned bytes are
transferred privately with the rules engine, never presented as our model/source.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import zipfile

import torch

from brass_ai.bank import FrozenOpponent, load_bank, read, sha
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint
from brass_ai.learning.contracts import digest, project_path, require
from brass_ai.learning.runner import RandomPolicy, collect_games
from .league import write_json


MODEL_INPUTS = {}  # Historical weights stay archived; no starting checkpoint.

CONTROL_BANKS = {'v3-s1-control': 'bank-d82438ccec287dbc8a64',
                 'v3-masked-control': 'bank-981744ddae4aaf07f1cf',
                 'v3-era-control': 'bank-3f7221cadcd472aafdb9',
                 'v3-industry-control': 'bank-340a20d72b23c652c267',
                 'v3-economy-control': 'bank-f5107c644af2fbf4ac99',
                 'v3-advanced-control': 'bank-3dc65253a610191d66a5'}


def control_banks(root):
    # A proposed comparison must not invalidate an already distributed lab
    # runtime. Ship the additional economic/advanced controls only after the
    # new admission receipt selects that bank for training.
    current = read(root / 'training/opponents/current.json')
    if current.get('format') in ('masked-v3-bank/1.3.0', 'masked-v4-bank/1.0.0'):
        return CONTROL_BANKS
    return {k: v for k, v in CONTROL_BANKS.items()
            if k in ('v3-s1-control', 'v3-masked-control', 'v3-era-control')}


def runtime_files(root):
    bank = read(root / 'training/opponents/current.json')['bank']
    manifest = load_bank(root, bank)
    from brass_ai.searchlab.runtime import install
    install(root)
    directories = ['artifacts/r1-host/guarded', 'artifacts/legacy-ai/bank/' + bank + '/runtime','artifacts/r4-host']
    # Preserve frozen comparison opponents without admitting them to the league.
    # Their proprietary dependencies remain in the private transfer only.
    for control in control_banks(root).values():
        if bank!=control and (root/'training/opponents/bank'/control/'manifest.json').exists():
            load_bank(root,control)
            directories.append('artifacts/legacy-ai/bank/'+control+'/runtime')
    files = {}
    for directory in directories:
        runtime = read(root / directory / 'runtime.json')
        for entry in runtime['files']:
            path = directory + '/' + entry['name']
            require(sha(project_path(root, path)) == entry['sha256'], 'Private runtime changed: ' + path)
            files[path] = entry['sha256']
        path = directory + '/runtime.json'; files[path] = sha(root / path)
    return bank, directories, files


def export_assets(root, *, bundle='artifacts/step5-transfer/private-rules-runtime.zip'):
    root = Path(root).resolve(); torch.set_num_threads(2)
    bank, directories, private_files = runtime_files(root)
    exported = []
    for name, source in MODEL_INPUTS.items():
        model, raw = load_checkpoint(project_path(root, source))
        path = root / 'training/models/bootstrap' / (name + '.pt')
        if path.exists():
            _, existing = load_checkpoint(path)
            require(existing['model_id'] == raw['model_id'], 'Refusing to overwrite different bootstrap weights')
        else:
            save_checkpoint(path, model, step=raw['step'], training=dict(raw['training'],
                distribution_status='R2 scaffold/smoke weights; unqualified playing strength', origin_sha256=sha(root / source)))
        exported.append(dict(path=path.relative_to(root).as_posix(), sha256=sha(path),
            model_id=raw['model_id'], parameters=model.parameter_count, original_mode=raw['training'].get('mode')))
    workers=[('r1', directories[0], 'BrassHost.exe'), ('v3', directories[1], 'BrassLegacyAI.exe')]
    workers += [('v3-lab',directories[1],'BrassLegacyAI.exe'),('search',directories[2],'BrassHost.exe')]
    for name,control in control_banks(root).items():
        directory='artifacts/legacy-ai/bank/'+control+'/runtime'
        if directory in directories:workers.append((name,directory,'BrassLegacyAI.exe'))
    for name, directory, exe in workers:
        path = root / 'training/runtime' / name / exe
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / directory / exe, path)
        exported.append(dict(path=path.relative_to(root).as_posix(), sha256=sha(path),
                             kind='project-authored worker; requires private game dependencies'))
    record = dict(format='step5-distribution/2.2.0', bank=bank,initialization='random',backend='masked_ismcts',legacy_backend=read(root/'training/opponents/current.json')['backend'],
                  assets=exported, private_runtime_files=private_files,
                  original_hard='stock-hard-guarded-1.1.0',
                  note='R3 policies are original C# sources and JSON profiles, not neural weights. Original Hard and game rules are privately transferred vendor DLLs.')
    write_json(root / 'training/runtime/distribution.json', record)
    target = project_path(root, bundle); target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in private_files:
            archive.write(root / relative, relative)
    return dict(manifest='training/runtime/distribution.json', bundle=str(target),
                bundle_sha256=sha(target), git_assets=len(exported), private_runtime_files=len(private_files))


def import_runtime(root, bundle):
    root = Path(root).resolve()
    record = read(root / 'training/runtime/distribution.json')
    expected = record['private_runtime_files']
    import hashlib
    with zipfile.ZipFile(project_path(root, bundle)) as archive:
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected),
                'Bundle file list differs; no arbitrary archive paths are accepted')
        # Validate all bytes before changing any existing runtime file.
        for info in archive.infolist():
            require(info.file_size <= 200_000_000 and not info.is_dir(), 'Unexpected bundle member')
            with archive.open(info) as stream:
                require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected[info.filename],
                        'Private bundle fingerprint mismatch: ' + info.filename)
        for relative in expected:
            path = project_path(root, relative)
            if path.exists():
                require(sha(path) == expected[relative], 'Refusing to replace a different existing runtime: ' + relative)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(relative) as source, path.open('wb') as output:
                shutil.copyfileobj(source, output)
    return dict(imported=True, files=len(expected), bank=record['bank'])


def parity(root, *, mono=None):
    policy = RandomPolicy()
    games = collect_games(root / 'artifacts/r1-host/guarded', [851001, 851002], [policy] * 4,
                          workers=2, sampling_seed=9127, mono=mono)
    # Sort by stable setup group; routing UUIDs, runtime timings and host paths
    # are deliberately excluded from cross-platform semantic comparisons.
    records = []
    for game in sorted(games, key=lambda g: g['setup_group']):
        semantic = [dict(view=r['view'], choices=r['choices'], chosen=r['chosen']) for r in game['records']]
        records.append(dict(setup_group=game['setup_group'], decisions=len(semantic),
                            sha256=digest(semantic), result=game['result']))
    return dict(format='step5-runtime-parity/1.0.0', seeds=[851001, 851002], sampling_seed=9127, games=records)


def preflight(root, *, device='cpu', mono=None, record_parity=False):
    root = Path(root).resolve(); torch.set_num_threads(2)
    distribution = read(root / 'training/runtime/distribution.json')
    for asset in distribution['assets']:
        require(sha(project_path(root, asset['path'])) == asset['sha256'], 'Distributed asset changed: ' + asset['path'])
        if asset['path'].endswith('.pt'):
            load_checkpoint(project_path(root, asset['path']))
    bank, _, files = runtime_files(root)
    require(bank == distribution['bank'] and files == distribution['private_runtime_files'], 'Private runtime differs')
    opponent = FrozenOpponent(root, bank + '/' + next(n for n in load_bank(root,bank)['payload']['profiles'] if n.startswith('balanced-v3')), purpose='diagnostic', mono=mono)
    opponent.close()
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA is unavailable; install the pinned CUDA wheel/compatible driver')
    x = torch.randn(64, 64, device=device, requires_grad=True)
    (x @ x.T).square().mean().backward()
    require(torch.isfinite(x.grad).all(), 'Device backward check failed')
    actual = parity(root, mono=mono)
    path = root / 'training/runtime/parity.json'
    if record_parity:
        require(os.name == 'nt' and not path.exists(), 'Record the initial parity evidence once on the qualified Windows runtime')
        write_json(path, actual)
    require(read(path) == actual, 'Cross-platform full-game parity failed; do not start lab self-play')
    result = dict(format='step5-preflight/1.0.0', passed=True, platform=platform.system(),
                  distribution_sha256=sha(root / 'training/runtime/distribution.json'),
                  parity_sha256=sha(path), device=device, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                  runtime=bank, games=len(actual['games']))
    write_json(root / 'artifacts/step5/preflight.json', result)
    return result


def require_preflight(root):
    root = Path(root)
    record = read(root / 'artifacts/step5/preflight.json')
    require(record['passed'] and record['platform'] == platform.system() and
            record['distribution_sha256'] == sha(root / 'training/runtime/distribution.json') and
            record['parity_sha256'] == sha(root / 'training/runtime/parity.json'),
            'Run Step 5 preflight on this host before collection')
