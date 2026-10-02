"""The previous advanced bank, selected explicitly by the user as v4.

This selection supersedes the strength-promotion experiment. It does not edit
the frozen policy or imply a winning screen. Saved profile keys remain aliases.
"""
from pathlib import Path
from brass_ai.bank import load_bank, read, sha, validate_admission
from brass_ai.learning.contracts import require
from brass_ai.step5.league import write_json

SOURCE_BANK = 'bank-3dc65253a610191d66a5'
ADMISSION = 'user-directed-masked-v4/1.0.0'
RECEIPT = 'training/opponents/evaluations/user-directed-v4-1.0.0/selection.json'


def select(root):
    root = Path(root).resolve(); manifest = load_bank(root, SOURCE_BANK)
    payload = manifest['payload']; manifest_sha = sha(root/f'training/opponents/bank/{SOURCE_BANK}/manifest.json')
    source = root/f'training/opponents/bank/{SOURCE_BANK}/source'
    tactics = (source/'src/BrassLegacyAI/Tactics.cs').read_text(encoding='utf-8-sig')
    search = (source/'src/BrassMaskedSearch/MaskedSearch.cs').read_text(encoding='utf-8-sig')
    require('next.progress_key!=previous.progress_key' in tactics and
            'next.age=retained && !progressed?' in tactics and
            'values[i]+Tactics.Bias(s.profile,s.intent,c,s.root)' in search,
            'Selected bank must contain productive-development and candidate-goal fixes')
    require('src/BrassMaskedSearch/NetworkAndHand.cs' not in payload['runtime_source_paths'],
            'Do not select the rejected network/Scout revision')
    receipt = dict(format=ADMISSION, passed=True, bank=SOURCE_BANK, generation='v4',
        manifest_sha256=manifest_sha, runtime_files=payload['runtime_files'], seconds=1., search_workers=4,
        selection_basis='Explicit user selection of the previous advanced revision as v4; stop further strength tests.',
        user_request='stop the test and use the pervious version (call it v4). Install it to the game and used it for training.',
        required_fix='Productive high-tier development resets the intent stall clock; opening goal influences build/setup candidate selection.',
        fixes=dict(productive_progress=True, opening_candidate_bias=True),
        strength_promoted=False, fresh_strength_games=0,
        note='User-directed selection, not an all-metric tournament pass. Frozen policy parameters and executable bytes remain unchanged.')
    path = root/RECEIPT
    if path.exists(): require(read(path)==receipt, 'Preserve the existing v4 selection receipt')
    else: write_json(path, receipt)
    fingerprint = sha(path)
    pool = dict(format='legacy-opponent-pool/1.0.0', backend='masked_ismcts', generation='v4', training={}, evaluation={})
    for name, profile in payload['profiles'].items():
        require(profile['milliseconds']==1000 and profile['mode']=='masked_ismcts', 'Retain one-second Masked teacher')
        purpose = 'evaluation' if name.startswith('pottery-flex') else 'training'
        identity = SOURCE_BANK+'/'+name
        entry = dict(admission=ADMISSION, manifest_sha256=manifest_sha,
                     qualification=RECEIPT, sha256=fingerprint, profile=name)
        validate_admission(root, identity, entry, purpose)
        pool[purpose][identity] = entry
    old = read(root/'training/opponents/pool.json')
    if old != pool:
        archive = root/'training/opponents/archive/pre-v4-pool.json'
        if archive.exists(): require(read(archive)==old, 'Preserve the previous admitted pool')
        else: write_json(archive, old)
    record = dict(format='masked-v4-bank/1.0.0', generation='v4', bank=SOURCE_BANK,
        manifest_sha256=manifest_sha, profiles=list(payload['profiles']), backend='masked_ismcts',
        seconds=1., search_workers=4, previous_bank='bank-340a20d72b23c652c267',
        qualification=RECEIPT, qualification_sha256=fingerprint, selection_basis=ADMISSION,
        strength_promoted=False, productive_progress_fix=True, opening_candidate_bias=True,
        note='Existing v3 profile identifiers are save/transfer aliases. Public generation is v4; no rejected network/Scout additions.')
    for relative, value in [('training/opponents/pool.json',pool), ('training/opponents/current.json',record),
                            ('training/opponents/lab/current.json',record)]: write_json(root/relative, value)
    return record
